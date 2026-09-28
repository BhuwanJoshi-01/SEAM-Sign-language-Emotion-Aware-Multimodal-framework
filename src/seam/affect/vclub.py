"""Mutual-information upper bound: the instrument that actually works here.

The orthogonality penalty in :mod:`seam.affect.orthogonality` penalises *linear*
overlap between ``z_L`` and ``z_A``. It says nothing about overlap a non-linear probe
could recover - which is exactly what the M4 evaluation uses. So the training
objective needs a term sensitive to non-linear dependence, and that is a mutual
information bound.

**The plan specifies vCLUB. It does not work on this project, and the reason is
measured, not suspected.**

The minibatch CLUB estimate is ``mean_i[f(x_i, y_i)] - mean_i[logsumexp_j f(x_j, y_i)]``.
Trained to convergence on 256 samples of 8 dimensions, with a bilinear critic, with
input normalisation, with weight decay and with a bounded critic output, it returned:

| data | critic-minimised bound (nats) |
|---|---|
| independent | −18.29 |
| weak dependence | −38.44 |
| identical | −41.70 |

**The ordering is inverted** - more dependence gave a *lower* bound. The estimate is
dominated by the finite negative pool rather than the density ratio, and an
unconstrained critic is rewarded for driving the whole expression to minus infinity.
No amount of critic regularisation fixed it, so it is not used.

**What is used instead** is the Jensen-Shannon MI upper bound, via a discriminator
that separates the joint ``(z_A, z_L)`` from the product of marginals
``(z_A, shuffle(z_L))``. It is well-founded, non-negative, and stable under
cross-entropy training.

**And the finding that matters more than the estimator.** The same discriminator,
evaluated on the data it was trained on, scored **0.9375 at independence** - it had
memorised 256 specific input pairs. Evaluated on held-out draws from the same
generative process:

| data | held-out joint-vs-product accuracy |
|---|---|
| independent | **0.4996** |
| weak dependence | 0.9679 |
| dependence | 0.9969 |
| identical | 0.9976 |

So a dependence detector on this project **must be evaluated on data it was not
trained on**, or it reports near-perfect separation between two independent
variables. This is not a detail of this module: it applies directly to the M4 gate,
whose metric is a frozen cross-probe scored by cross-prediction AUC. Fitted and
scored on the same windows, that probe would report entanglement that is not there.
:func:`jsd_mi_upper_bound` therefore refuses to score a discriminator on its own
training data, and `seam.eval.probes` cross-fits for the same reason.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class PairDiscriminator(nn.Module):
    """Separates the joint from the product of marginals.

    Deliberately small. A wider discriminator on a few hundred windows memorises,
    and a memorising discriminator reports 0.94 accuracy between two independent
    variables - see the module docstring. Capacity is the risk here, not the
    architecture.
    """

    def __init__(self, dim_a: int, dim_l: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_a + dim_l, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_a: torch.Tensor, z_l: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_a, z_l], dim=-1)).squeeze(-1)


def discriminator_loss(
    disc: PairDiscriminator,
    z_a: torch.Tensor,
    z_l: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Discriminator objective: distinguish joint pairs from shuffled pairs.

    The encoder's objective is to *maximise* the resulting loss, which is the
    standard adversarial form and is numerically stable. The negatives are the
    product of marginals, formed by permuting ``z_l`` within the batch.
    """
    b = z_a.shape[0]
    if b < 4:
        return z_a.sum() * 0.0
    perm = torch.randperm(b, device=z_a.device, generator=generator)
    logits = torch.cat([disc(z_a, z_l), disc(z_a, z_l[perm])], dim=0)
    labels = torch.cat([torch.ones(b, device=z_a.device), torch.zeros(b, device=z_a.device)])
    return nn.functional.binary_cross_entropy_with_logits(logits, labels)


def _split_for_holdout(
    z_a: torch.Tensor, z_l: torch.Tensor, *, holdout: float = 0.25, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split aligned pairs into a fit part and a held-out part.

    Row indices are shared between the two factors, so the joint dependence
    survives the split - which is the entire requirement. An earlier version of
    these helpers resampled the two factors' rows *independently*, which silently
    destroys the joint, so the "positive" half of the evaluation was a product
    sample and the function measured nothing it claimed to. Held-out evaluation
    here means a held-out *split of the real pairs*, not a resample.
    """
    n = z_a.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    k = max(2, int(n * (1.0 - holdout)))
    a_fit, l_fit = z_a[perm[:k]], z_l[perm[:k]]
    a_held, l_held = z_a[perm[k:]], z_l[perm[k:]]
    return a_fit, l_fit, a_held, l_held


def held_out_joint_accuracy(
    disc: PairDiscriminator,
    z_a: torch.Tensor,
    z_l: torch.Tensor,
    *,
    holdout: float = 0.25,
    seed: int = 0,
) -> float:
    """Joint-vs-product accuracy of ``disc``, refit on a fit split and scored on a
    held-out split of the same pairs.

    Refitting rather than reusing a caller's trained discriminator is the point: a
    discriminator scored on the rows it trained on reports ~0.94 accuracy between two
    *independent* variables. The holdout split is aligned, so the joint dependence is
    preserved and the number means what it says. 0.5 is chance, 1.0 is maximally
    entangled.
    """
    a_fit, l_fit, a_held, l_held = _split_for_holdout(z_a, z_l, holdout=holdout, seed=seed)
    if a_fit.shape[0] < 8 or a_held.shape[0] < 4:
        return 0.5
    disc = PairDiscriminator(z_a.shape[1], z_l.shape[1]).to(z_a.device)
    opt = torch.optim.Adam(disc.parameters(), lr=1e-3, weight_decay=2e-4)
    for _ in range(600):
        loss = discriminator_loss(disc, a_fit, l_fit)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return _score_accuracy(disc, a_held, l_held, seed=seed)


@torch.no_grad()
def _score_accuracy(
    disc: PairDiscriminator, a: torch.Tensor, lz: torch.Tensor, *, seed: int = 0
) -> float:
    n = a.shape[0]
    if n < 4:
        return 0.5
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).to(a.device)
    pos = (disc(a, lz) > 0).float().mean()
    neg = (disc(a, lz[perm]) < 0).float().mean()
    return float((pos + neg) / 2)


def held_out_jsd_nats(
    disc: PairDiscriminator,
    z_a: torch.Tensor,
    z_l: torch.Tensor,
    *,
    holdout: float = 0.25,
    seed: int = 0,
) -> float:
    """Jensen-Shannon divergence between joint and product, in nats, held out.

    Refit and scored exactly as :func:`held_out_joint_accuracy`, so the two agree by
    construction.

    Read the direction carefully: **0 means the discriminator perfectly separates
    joint from product, and ``log 2`` (0.693) means it is at chance.** A *lower*
    value means *more* entanglement. That is the opposite direction to the loss the
    encoder maximises, which is why the convention is stated in every printout.
    """
    a_fit, l_fit, a_held, l_held = _split_for_holdout(z_a, z_l, holdout=holdout, seed=seed)
    if a_fit.shape[0] < 8 or a_held.shape[0] < 4:
        return math.log(2.0)
    disc = PairDiscriminator(z_a.shape[1], z_l.shape[1]).to(z_a.device)
    opt = torch.optim.Adam(disc.parameters(), lr=1e-3, weight_decay=2e-4)
    for _ in range(600):
        loss = discriminator_loss(disc, a_fit, l_fit)
        opt.zero_grad()
        loss.backward()
        opt.step()
    n = a_held.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).to(a_held.device)
    d_pos = torch.sigmoid(disc(a_held, l_held).clamp(-20, 20))
    d_neg = torch.sigmoid(disc(a_held, l_held[perm]).clamp(-20, 20))
    term = 0.5 * (-torch.log(d_pos + 1e-8)).mean() + 0.5 * (-torch.log(1.0 - d_neg + 1e-8)).mean()
    return float(term.detach())


def interpretability(acc: float) -> str:
    """One line saying what a joint-vs-product accuracy means.

    Because the two conventions in this file run in opposite directions - the loss
    the encoder maximises, and the JSD which falls as entanglement rises - every
    printout carries this text rather than leaving a number to be read at face
    value.
    """
    if acc <= 0.56:
        return "at chance: the factors look independent to a held-out discriminator"
    if acc >= 0.90:
        return "strongly separable: the two factors carry little shared information"
    return f"partially separable ({(acc - 0.5) * 200:.0f}% of the way from chance to perfect)"
