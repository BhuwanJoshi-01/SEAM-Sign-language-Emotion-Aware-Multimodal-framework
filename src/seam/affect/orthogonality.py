"""Penalise the shared subspace of ``z_L`` and ``z_A``.

``||Z_L Z_A^T||_F^2`` is the squared Frobenius norm of the cross-covariance, i.e. the
total variance each factor explains in the other. It is the one term here that acts
directly on the *representation* rather than on any prediction, so it is the term
that can separate two factors even when no downstream probe would notice the
overlap.

Two properties matter for how it is implemented, and both were learned the hard way
in this project:

* **It is scale-sensitive.** The penalty grows with the norm of each factor, so
  normalising the factors makes the loss depend on *overlap* rather than on
  *magnitude*. An unnormalised version can be minimised by shrinking one branch to
  nothing, which is a degenerate solution that looks like perfect separation and
  destroys the prediction heads that share the encoder.
* **It is batch-sensitive.** With a small batch the cross-covariance is biased
  toward zero by construction, so the penalty shrinks as the batch shrinks and the
  loss silently changes meaning between training and evaluation. The batch size is
  therefore recorded alongside it, and the bias correction is not applied because a
  corrected estimate on a small batch is noisier than the uncorrected one.
"""

from __future__ import annotations

import numpy as np
import torch


def orthogonality_penalty(z_l: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
    """``||Z_L Z_A^T||_F^2`` on unit-norm rows, averaged.

    Rows are L2-normalised first so the loss measures directional overlap rather
    than scale. Without that, a branch that simply shrinks wins the penalty while
    contributing nothing downstream.
    """
    if z_l.ndim != 2 or z_a.ndim != 2:
        raise ValueError(f"expected 2D factors, got {tuple(z_l.shape)} and {tuple(z_a.shape)}")
    if z_l.shape[0] != z_a.shape[0]:
        raise ValueError(f"factors must share a batch: {z_l.shape[0]} vs {z_a.shape[0]}")
    a = torch.nn.functional.normalize(z_l, dim=1, eps=1e-8)
    b = torch.nn.functional.normalize(z_a, dim=1, eps=1e-8)
    cross = a @ b.t()
    return (cross**2).sum() / max(a.shape[0], 1)


def shared_subspace_fraction(z_l: torch.Tensor, z_a: torch.Tensor) -> float:
    """Fraction of total variance shared, for reporting rather than optimising.

    The squared Frobenius penalty is a sum over dimensions and grows with the batch,
    so it is not a readable "how much overlap is there" figure. This is: the mean
    squared normalised cross-correlation, which is 0 for orthogonal factors and 1 for
    identical ones, and is reported next to the cross-prediction AUC so the training
    objective and the evaluation metric can be compared.
    """
    with torch.no_grad():
        a = torch.nn.functional.normalize(z_l.detach().float(), dim=1, eps=1e-8)
        b = torch.nn.functional.normalize(z_a.detach().float(), dim=1, eps=1e-8)
        return float(((a @ b.t()) ** 2).mean())


def canonical_angles(z_l: torch.Tensor, z_a: torch.Tensor) -> np.ndarray:
    """Principal angles between the two row spaces, in degrees.

    Included because a single scalar cannot distinguish "the factors overlap in one
    direction" from "they overlap in all directions", and the first is a nuisance
    while the second is the failure the M4 gate is about. Zero degrees means shared
    subspace, 90 degrees means orthogonal.
    """
    with torch.no_grad():
        a = z_l.detach().float().cpu().numpy()
        b = z_a.detach().float().cpu().numpy()
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
            return np.zeros(0)
        qa, _ = np.linalg.qr(a - a.mean(0, keepdims=True))
        qb, _ = np.linalg.qr(b - b.mean(0, keepdims=True))
        s = np.linalg.svd(qa.T @ qb, compute_uv=False)
        return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))
