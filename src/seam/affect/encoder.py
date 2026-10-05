"""The factorized non-manual encoder: ``z_L`` and ``z_A`` as separate factors.

``z_L = Enc_L(NM)`` and ``z_A = Enc_A(NM, P)``, each a small trunk with its own head.
Both branches see the same non-manual signal, which is the point: the information is
genuinely shared in the input and the only way to keep it apart is to make the
representation keep it apart.

The design decisions that are not free choices, and why:

**Two trunks, not a shared trunk with two heads.** A shared trunk cannot be asked to
disentangle, because the shared weights are exactly the channel through which the
factors leak. A single-branch "entangled baseline" is kept alongside for the
ablation, and the comparison between them is the M4 claim.

**The affine affect branch.** ``z_A`` gets the prosody vector ``P`` added through a
learned scale, so the affect factor is *rooted* in kinetic evidence rather than free
to invent affect from brow motion alone. Without it the model can route all affect
through the facial channel and then separation becomes impossible for a reason that
has nothing to do with the confound.

**Parameter budget is asserted, not assumed.** Both trunks are capped at 1M
parameters and a test fails if either exceeds it, because a budget nobody checks is a
budget nobody meets.

**GRL in both directions.** ``z_A`` is reversed against the linguistic head and ``z_L``
against the affect head, so neither factor can carry the other's private
information. One direction leaves the other factor free to smuggle it through.

**The MI term is a discriminator, and the orthogonality term is scale-normalised.**
See :mod:`seam.affect.vclub` and :mod:`seam.affect.orthogonality` for the measured
reasons; the short version is that both instruments had failure modes that produced
plausible models and wrong conclusions, and both are now pinned by tests.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import torch
from torch import nn

from seam.affect.grl import GradientReversal
from seam.affect.orthogonality import orthogonality_penalty

#: Hard cap per trunk, asserted by ``tests/test_affect_encoder.py``.
MAX_PARAMS_PER_TRUNK = 1_000_000


def mlp(sizes: list[int], *, out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    """A plain MLP trunk. Kept small on purpose; capacity here causes memorisation."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers += [nn.LayerNorm(sizes[i + 1]), nn.GELU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(sizes[-1], out_dim))
    return nn.Sequential(*layers)


@dataclass(slots=True)
class FactorizerConfig:
    dim_nm: int
    dim_p: int
    n_linguistic: int
    #: Number of affect **labels**, not classes. Affect is multi-label: an EmoSign
    #: clip can carry three emotions at once, and framing it as 8-way
    #: single-expression classification discarded 1,451 of 1,765 windows (82%) because
    #: only one in seven clips has a single committed expression. The head is therefore
    #: ``n_affect`` independent binary outputs trained with BCE, and every clip
    #: contributes.
    n_affect: int
    #: BCE needs a per-label positive weight because the labels are independently
    #: imbalanced - "anger" and "joy" are common, "disgust" is rare - and an unweighted
    #: multi-label loss collapses to all-negative.
    pos_weight_affect: float = 3.0
    dim_z_l: int = 32
    dim_z_a: int = 32
    hidden: int = 128
    dropout: float = 0.1
    #: Loss weights. Every one of these is ablated in the per-lambda harness, which is
    #: the only way to know which term is doing the work.
    w_linguistic: float = 1.0
    w_affect: float = 1.0
    w_orthogonality: float = 0.1
    w_mi: float = 0.01
    #: GRL strength on the linguistic side (z_A is trained away from z_L's content).
    lambda_grl: float = 0.5
    #: GRL strength on the affect side. A separate field because one direction can
    #: be ablated without the other, and because a single shared value made the
    #: ``no_affect_from_p`` ablation a no-op: the weight it set was never read by
    #: the loss, so that variant reproduced ``full`` byte for byte and looked like a
    #: result.
    lambda_grl_a: float = 0.5

    def describe(self) -> dict[str, float | int]:
        return {
            "dim_z_l": self.dim_z_l,
            "dim_z_a": self.dim_z_a,
            "hidden": self.hidden,
            "w_linguistic": self.w_linguistic,
            "w_affect": self.w_affect,
            "w_orthogonality": self.w_orthogonality,
            "w_mi": self.w_mi,
            "lambda_grl": self.lambda_grl,
            "lambda_grl_a": self.lambda_grl_a,
        }


class FactorizedEncoder(nn.Module):
    """Two factor trunks over one non-manual signal, with three prediction heads.

    Heads, and why each exists:

    * ``head_L`` — predict the linguistic label from ``z_L``. The main linguistic task.
    * ``head_A`` — predict affect from ``z_A``. The main affect task.
    * ``head_VA`` — predict affect from ``z_L`` **through a gradient reversal**. This
      is the term that makes the design falsifiable: if the linguistic factor
      retained everything needed to read affect, this head would succeed, and the
      reversal makes the trunk actively strip it out. Its accuracy is reported and
      should be *low*; a high value means the factors are not separated no matter what
      the MI bound claims.
    """

    def __init__(self, cfg: FactorizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.trunk_l = mlp(
            [cfg.dim_nm, cfg.hidden, cfg.hidden], out_dim=cfg.dim_z_l, dropout=cfg.dropout
        )
        # The affect trunk additionally receives the prosody vector, concatenated
        # rather than summed, so it can use it as evidence rather than as a bias.
        self.trunk_a = mlp(
            [cfg.dim_nm + cfg.dim_p, cfg.hidden, cfg.hidden],
            out_dim=cfg.dim_z_a,
            dropout=cfg.dropout,
        )
        self.head_l = nn.Linear(cfg.dim_z_l, cfg.n_linguistic)
        self.head_a = nn.Linear(cfg.dim_z_a, cfg.n_affect)
        #: Cross heads. ``head_va`` reads affect from z_L through a reversal, so z_L is
        #: trained to shed affect; ``head_la`` reads linguistic from z_A through one,
        #: so z_A is trained to shed linguistic. Both are trained, and both losses are
        #: in the objective.
        self.head_va = nn.Linear(cfg.dim_z_l, cfg.n_affect)
        self.head_la = nn.Linear(cfg.dim_z_a, cfg.n_linguistic)
        self.grl_on_l = GradientReversal(cfg.lambda_grl)
        self.grl_on_a = GradientReversal(cfg.lambda_grl_a)

    def forward(self, nm: torch.Tensor, p: torch.Tensor) -> dict[str, torch.Tensor]:
        if nm.shape[0] != p.shape[0]:
            raise ValueError(f"nm/p batch mismatch: {nm.shape[0]} vs {p.shape[0]}")
        z_l = self.trunk_l(nm)
        z_a = self.trunk_a(torch.cat([nm, p], dim=-1))
        return {
            "z_l": z_l,
            "z_a": z_a,
            "logits_l": self.head_l(z_l),
            "logits_a": self.head_a(z_a),
            # Reversed, so the trunk is trained to remove affect from z_l while the
            # head is trained to read it. The head's own accuracy is the diagnostic.
            "logits_va": self.head_va(self.grl_on_l(z_l)),
            "logits_va_from_a": self.head_va(self.grl_on_a(z_a)),
            "logits_la": self.head_la(self.grl_on_a(z_a)),
        }

    def trunk_param_counts(self) -> dict[str, int]:
        return {
            "z_l": sum(p.numel() for p in self.trunk_l.parameters()),
            "z_a": sum(p.numel() for p in self.trunk_a.parameters()),
        }

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class EntangledBaseline(nn.Module):
    """Single-branch control: one trunk, both heads, no separation pressure.

    Included because "separation helps" is only a claim if there is something to
    separate from. This is that something, and it is trained under the identical data
    pipeline and schedule so the comparison isolates the factorisation.
    """

    def __init__(self, cfg: FactorizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.trunk = mlp(
            [cfg.dim_nm + cfg.dim_p, cfg.hidden, cfg.hidden],
            out_dim=cfg.dim_z_l,
            dropout=cfg.dropout,
        )
        self.head_l = nn.Linear(cfg.dim_z_l, cfg.n_linguistic)
        self.head_a = nn.Linear(cfg.dim_z_l, cfg.n_affect)

    def forward(self, nm: torch.Tensor, p: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.trunk(torch.cat([nm, p], dim=-1))
        return {
            "z_l": z,
            "z_a": z,
            "logits_l": self.head_l(z),
            "logits_a": self.head_a(z),
            "logits_va": self.head_a(z),
            "logits_va_from_a": self.head_a(z),
            "logits_la": self.head_l(z),
        }

    def trunk_param_counts(self) -> dict[str, int]:
        return {"z": sum(p.numel() for p in self.trunk.parameters())}

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _affect_pos_weight(y_a: torch.Tensor, cfg: FactorizerConfig) -> torch.Tensor | None:
    """``pos_weight`` for the multi-label affect BCE, from the batch being trained.

    Computed per batch rather than once from the training set, so it tracks the local
    imbalance instead of a global constant that is wrong for every batch. Floored at 1.0
    so a batch with no positives does not produce a zero or infinite weight.
    """
    if y_a.dim() != 2:
        return None
    pos = y_a.sum(0).clamp_min(1.0)
    neg = (1.0 - y_a).sum(0).clamp_min(1.0)
    return (neg / pos).clamp(1.0, 20.0)


def class_weights(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights from the *training* labels.

    The linguistic task is ~13% positive at the clip level, so an unweighted loss
    spends its capacity on the majority class and the model ends up below a
    majority-class predictor. Weights are computed from whatever labels the caller
    passes, which is always the training split - computing them over the whole set
    would leak the test fold's class balance into the fit.
    """
    counts = torch.bincount(y, minlength=n_classes).float().clamp_min(1.0)
    w = counts.sum() / (n_classes * counts)
    return w / w.mean()


def direct_task_losses(
    out: dict[str, torch.Tensor],
    y_l: torch.Tensor,
    y_a: torch.Tensor,
    cfg: FactorizerConfig,
    *,
    weight_l: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two supervised terms, ``(loss_l, loss_a)``, defined once.

    Three things must agree on these or the comparison between them means nothing:
    the factorized model's training objective, its checkpoint selection, and the
    entangled baseline. They did not. The baseline was trained with unweighted
    cross-entropy and unweighted BCE while the factorized model had class weights and
    ``pos_weight``, so the baseline predicted almost no positive affect label (micro-F1
    0.06 against 0.33) for a reason unrelated to separation; and checkpoints of the
    factorized model were selected on a *softmax* cross-entropy over the eight affect
    logits, a leftover of the single-expression framing that the training loss had
    already abandoned. Everything now calls this.
    """
    loss_l = nn.functional.cross_entropy(out["logits_l"], y_l, weight=weight_l)
    # Affect is multi-label, so this is BCE over independent binary targets, not
    # softmax cross-entropy over 8 competing classes. The distinction is the whole
    # point of the rerun: softmax forces exactly one label to win, which is what made
    # 82% of the windows untrainable.
    loss_a = nn.functional.binary_cross_entropy_with_logits(
        out["logits_a"], y_a, pos_weight=_affect_pos_weight(y_a, cfg)
    )
    return loss_l, loss_a


def selection_loss(
    out: dict[str, torch.Tensor],
    y_l: torch.Tensor,
    y_a: torch.Tensor,
    cfg: FactorizerConfig,
    *,
    weight_l: torch.Tensor | None = None,
) -> float:
    """Validation criterion for choosing a checkpoint: the weighted direct terms only.

    The separation terms are deliberately excluded. Selecting on them would pick the
    checkpoint that looks most separated on the validation signer, which is the
    quantity under test.
    """
    loss_l, loss_a = direct_task_losses(out, y_l, y_a, cfg, weight_l=weight_l)
    return float((cfg.w_linguistic * loss_l + cfg.w_affect * loss_a).detach())


def factorizer_loss(
    out: dict[str, torch.Tensor],
    y_l: torch.Tensor,
    y_a: torch.Tensor,
    cfg: FactorizerConfig,
    *,
    w_va_aux: torch.Tensor | None = None,
    weight_l: torch.Tensor | None = None,
    weight_a: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The encoder-side objective.

    ``w_va_aux`` lets the caller add an MI-discriminator term, which has to be
    differentiated through the encoders but is trained by a separate optimiser - so
    it arrives here as a pre-computed scalar rather than being computed inside.
    """
    ce = nn.functional.cross_entropy
    bce = nn.functional.binary_cross_entropy_with_logits
    loss_l, loss_a = direct_task_losses(out, y_l, y_a, cfg, weight_l=weight_l)
    # **Both adversarial terms belong in the total.** The first version of this
    # function added only the two direct losses and commented that the GRL heads'
    # "own loss is not minimised - its gradient is reversed", excluding them. That is
    # wrong twice over: excluding them leaves the reversal in the graph but not in the
    # objective, so the GRL heads are *untrained*; and the reported GRL-head accuracy
    # then measured an untrained head rather than adversarial defeat. The consequence
    # was a number that read as a striking result - "the adversarial head reads affect
    # at 0.121 while a frozen probe reads the same factor at 0.871" - and was an
    # artefact of a head that had never been fitted.
    #
    # A reversed branch is a real term: the head learns to predict and the trunk
    # beneath it learns to prevent that. Both gradients are wanted.
    loss_a_from_l = bce(out["logits_va"], y_a, pos_weight=_affect_pos_weight(y_a, cfg))
    loss_l_from_a = ce(out["logits_la"], y_l, weight=weight_l)
    orth = orthogonality_penalty(out["z_l"], out["z_a"])

    total = (
        cfg.w_linguistic * (loss_l + loss_l_from_a)
        + cfg.w_affect * (loss_a + loss_a_from_l)
        + cfg.w_orthogonality * orth
    )
    parts = {
        "loss_l": float(loss_l.detach()),
        "loss_a": float(loss_a.detach()),
        "loss_a_from_l": float(loss_a_from_l.detach()),
        "loss_l_from_a": float(loss_l_from_a.detach()),
        "orthogonality": float(orth.detach()),
    }
    if w_va_aux is not None:
        total = total - cfg.w_mi * w_va_aux
        parts["mi_term"] = float(w_va_aux.detach())
    return total, parts


@dataclass(slots=True)
class LambdaGrid:
    """The per-lambda ablation grid.

    Every entry turns exactly one term off, so the contribution of each is
    attributable. Running the full grid rather than the full model is what turns
    "the factorisation works" into "this specific term is what works".
    """

    base: FactorizerConfig
    variants: dict[str, dict[str, float]] = field(default_factory=dict)

    def names(self) -> list[str]:
        return sorted(self.variants)

    def config_for(self, name: str) -> FactorizerConfig:

        if name == "full":
            return self.base
        if name not in self.variants:
            raise KeyError(f"unknown variant {name!r}; have {self.names()}")
        # Validate the field names, then apply by ``setattr`` on a copy. ``replace``
        # with **kwargs is what mypy rejects here, because the config mixes int
        # dimensions with float weights and the unpacked dict cannot be proven to
        # carry the right type per field - so the names are checked against the real
        # fields and the values assigned individually, which is also a better error
        # message when a grid entry names something that does not exist.
        import copy

        cfg = copy.copy(self.base)
        for key, value in self.overrides_for(name).items():
            setattr(cfg, key, value)
        return cfg

    def overrides_for(self, name: str) -> dict[str, float | int]:
        allowed = {f.name for f in dataclasses.fields(self.base)}
        bad = set(self.variants.get(name, {})) - allowed
        if bad:
            raise KeyError(f"variant {name!r} sets unknown fields: {sorted(bad)}")
        return dict(self.variants.get(name, {}))


def default_lambda_grid(base: FactorizerConfig) -> LambdaGrid:
    """Ablations that each remove one mechanism."""
    return LambdaGrid(
        base=base,
        variants={
            "no_orthogonality": {"w_orthogonality": 0.0},
            "no_mi": {"w_mi": 0.0},
            "no_separation": {
                "w_orthogonality": 0.0,
                "w_mi": 0.0,
                "lambda_grl": 0.0,
                "lambda_grl_a": 0.0,
            },
            # ``no_grl_on_a``: the affect-side reversal off. This is the only lever
            # on the second GRL direction.
            "no_grl_on_a": {"lambda_grl_a": 0.0},
        },
    )
