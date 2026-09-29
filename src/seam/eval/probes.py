"""Cross-prediction AUC: the M4 gate metric, and the positive control that licenses it.

The M4 claim is that ``z_L`` and ``z_A`` are separated. "Separated" has to be measured
by something that could have come out the other way, which means: **freeze the factors,
fit a probe on one, and ask how well it predicts the other.** Cross-prediction AUC of
0.5 is perfect separation; 1.0 is total entanglement.

**A metric that cannot detect entanglement when it exists proves nothing.** A model
that is completely entangled and one that is perfectly separated would both look fine
to an insensitive probe, and the gate would pass for the wrong reason. So
:func:`signer_probe_auc` is a *positive control* on the same machinery: a signer-identity
probe is scored the same way, and the gate refuses to interpret any separation number
unless the control scores high. This is the plan's rule 13, and it is enforced in code
rather than left to whoever reads the table.

**Every probe here is cross-fitted, and that is not optional.** Measured in
:mod:`seam.affect.vclub`: a dependence detector trained and evaluated on the same few
hundred samples reports **0.94 accuracy between two independent variables**, because it
memorises them. The same detector on held-out draws reports 0.4996. So a probe fitted
and scored on the same windows would manufacture entanglement that is not present - it
would inflate the cross-prediction AUC, which is the number the gate depends on, in the
direction that makes a broken model look good. Each fold therefore splits into
fit/inner-test parts, and the reported AUC aggregates out-of-fold predictions only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from seam.logging import get

log = get(__name__)

#: Above this, the control has failed and no separation number may be interpreted.
SIGNER_CONTROL_MIN_AUC = 0.80

#: The gate target from the plan.
SEPARATION_TARGET_AUC = 0.60


def _fit_logistic(
    x: np.ndarray, y: np.ndarray, *, l2: float = 1.0, max_iter: int = 500
) -> np.ndarray:
    """L2-regularised logistic regression. Returns the weight vector, bias appended.

    Uses scikit-learn's convex solver rather than a hand-rolled gradient step. The
    first version of this function normalised its gradient and took a step of
    ``0.5 * sqrt(n)`` for 200 iterations, which diverges: the resulting probe reported
    an **AUC of 5.08**, a value the metric cannot take, and made the gate verdict
    meaningless rather than merely wrong.

    Standardisation uses the **fit split only** (the caller passes it), and the
    regularisation is deliberately strong. Both matter: a probe fitted on a few
    hundred windows is the main way this metric lies, and the M4 vCLUB work measured
    the general case - a discriminator that memorises reports 0.94 accuracy between
    two independent variables.
    """
    from sklearn.linear_model import LogisticRegression

    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    xs = (x - mu) / sd
    # L2 is the default penalty; passing it explicitly is deprecated in sklearn 1.8
    # and floods the log with one warning per probe fit.
    model = LogisticRegression(C=1.0 / max(l2, 1e-6), max_iter=max_iter, solver="lbfgs")
    model.fit(xs, (np.asarray(y) > 0.5).astype(int))
    # Fold the standardisation back into a single weight vector over raw features,
    # plus the bias, so the caller scores raw held-out rows without re-standardising.
    w = model.coef_[0] / sd.ravel()
    b = float(model.intercept_[0] - (w * mu.ravel()).sum())
    return np.concatenate([w, [b]])


def auc(y_true: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    """ROC AUC via the rank identity, tie-aware. 0.5 = uninformative.

    **The label must be binary, and this raises if it is not.** The first version of
    this function accepted any integer label, and on the 8-class affect labels it
    silently produced **AUC values above 1** - 3.81 and 4.43 in the first real M4 run.
    The cause: samples in classes 2..7 belonged to neither the positive nor the
    negative group, so they consumed rank space without being counted, and the
    Mann-Whitney numerator stopped being bounded by the denominator. A metric that can
    return 4.4 does not measure anything, and it produced a gate verdict that looked
    like a result.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    if len(y) != len(s):
        raise ValueError(f"length mismatch: {len(y)} labels, {len(s)} scores")
    present = sorted(set(y.tolist()))
    if present not in ([0, 1], [0], [1]):
        raise ValueError(
            f"auc() needs a binary label, got classes {present[:8]}"
            f"{'...' if len(present) > 8 else ''}. Use "
            "cross_prediction_multiclass() for a multiclass target - silently "
            "coercing it here produced AUCs above 1."
        )
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    idx = np.random.default_rng(seed).permutation(n)
    return [np.sort(f) for f in np.array_split(idx, k)]


@dataclass(slots=True)
class CrossPrediction:
    """One cross-prediction result, with the direction it was measured in."""

    source: str
    target: str
    auc: float
    n: int
    n_positive: int
    #: AUC of the *same* probe with source and target shuffled, which is the null this
    #: number must beat. A cross-prediction of 0.5 exactly is suspicious rather than
    #: good, and this is the check that says so.
    auc_permuted: float = float("nan")
    interpretable: bool = True
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "auc": None if np.isnan(self.auc) else round(self.auc, 4),
            "n": self.n,
            "n_positive": self.n_positive,
            "auc_permuted": None if np.isnan(self.auc_permuted) else round(self.auc_permuted, 4),
            "interpretable": self.interpretable,
            "note": self.note,
        }


def cross_prediction(
    z_source: np.ndarray,
    y_target: Sequence[int] | np.ndarray,
    *,
    label: str = "",
    n_inner: int = 4,
    seed: int = 0,
) -> CrossPrediction:
    """Out-of-fold AUC for predicting ``y_target`` from ``z_source``.

    Cross-fitted: the probe is fitted on 3/4 of the data and scored on the held-out
    quarter, aggregated across all folds. The permuted control repeats the whole
    procedure with ``y_target`` shuffled, which is the same computation on data with
    no relationship at all - the reference the real number has to beat.
    """
    x = np.asarray(z_source, dtype=float)
    y = np.asarray(y_target).astype(int)
    n = len(x)
    src = label or "z"
    if n < 20:
        return CrossPrediction(
            source=src,
            target="y",
            auc=float("nan"),
            n=n,
            n_positive=int((y == 1).sum()) if n else 0,
            interpretable=False,
            note=f"only {n} samples; a probe cannot be cross-fitted on this",
        )
    if len(set(y.tolist())) < 2:
        return CrossPrediction(
            source=src,
            target="y",
            auc=float("nan"),
            n=n,
            n_positive=int((y == 1).sum()),
            interpretable=False,
            note="target label is constant",
        )

    oof = np.zeros(n, dtype=float)
    for hold in _folds(n, min(n_inner, n), seed):
        mask = np.ones(n, dtype=bool)
        mask[hold] = False
        try:
            w = _fit_logistic(x[mask], y[mask])
        except Exception as exc:  # pragma: no cover - degenerate fold
            log.warning("probe fit failed on a fold: %s", exc)
            continue
        xb = np.hstack([x[hold], np.ones((len(hold), 1))])
        oof[hold] = xb @ w
    a = auc(y, oof)

    rng = np.random.default_rng(seed + 1)
    oof_p = np.zeros(n, dtype=float)
    yp = rng.permutation(y)
    for hold in _folds(n, min(n_inner, n), seed):
        mask = np.ones(n, dtype=bool)
        mask[hold] = False
        try:
            w = _fit_logistic(x[mask], yp[mask])
        except Exception:  # pragma: no cover
            continue
        xb = np.hstack([x[hold], np.ones((len(hold), 1))])
        oof_p[hold] = xb @ w
    ap = auc(yp, oof_p)
    return CrossPrediction(
        source=src,
        target="y",
        auc=a,
        n=n,
        n_positive=int((y == 1).sum()),
        auc_permuted=ap,
    )


def cross_prediction_multiclass(
    z_source: np.ndarray,
    y_multi: Sequence[int] | np.ndarray,
    *,
    label: str = "",
    min_support: int = 12,
    n_inner: int = 4,
    seed: int = 0,
) -> dict[str, object]:
    """One-vs-rest cross-prediction for a multiclass target.

    Affect is 8-class, so a single "predict affect from ``z_L``" AUC is not well
    defined. The question the M4 gate actually asks is whether affect is separable
    **at all**, so this returns one AUC per class and the **worst** class is the
    headline. Using the mean would let a model score well on seven classes and carry
    one entangled class into the average; using the worst class is the conservative
    reading and is what the gate is computed on.

    Classes with too few windows are reported as skipped rather than given an
    unstable AUC from a handful of samples.
    """
    y = np.asarray(y_multi).astype(int)
    per_class: list[dict[str, object]] = []
    for cls in sorted(set(y.tolist())):
        mask = (y == cls).astype(int)
        n_pos, n_neg = int(mask.sum()), int((1 - mask).sum())
        if n_pos < min_support or n_neg < min_support:
            per_class.append(
                {"class": cls, "skipped": True, "n_positive": n_pos, "n_negative": n_neg}
            )
            continue
        cp = cross_prediction(
            z_source, mask, label=f"{label}->class{cls}", n_inner=n_inner, seed=seed
        )
        per_class.append({**cp.as_dict(), "class": cls, "skipped": False})
    usable = [
        float(r["auc"])  # type: ignore[arg-type]
        for r in per_class
        if not r["skipped"] and r.get("auc") is not None
    ]
    return {
        "per_class": per_class,
        "n_usable": len(usable),
        "worst_auc": max(usable) if usable else float("nan"),
        "mean_auc": float(np.mean(usable)) if usable else float("nan"),
        "interpretable": bool(usable),
        "note": (
            "worst one-vs-rest AUC is the headline: separation must hold for every "
            "class to mean the factors do not share affect"
        ),
    }


def cross_prediction_multilabel(
    z_source: np.ndarray,
    y_multi: np.ndarray,
    *,
    label: str = "",
    min_support: int = 20,
    n_inner: int = 4,
    seed: int = 0,
) -> dict[str, object]:
    """Per-label cross-prediction for a multi-hot target.

    Affect is multi-label, so there is no single "predict affect" AUC: each label is
    its own binary target and each gets its own probe. The **worst** label is the
    headline, for the same reason as in :func:`cross_prediction_multiclass` — a
    separation claim has to hold for every label, and an average would let seven
    well-separated labels carry one entangled one.

    Labels with fewer than ``min_support`` positives or negatives are reported as
    skipped rather than given an AUC from a handful of windows, which is the
    difference between a measurement and a coincidence.
    """
    y = np.asarray(y_multi)
    if y.ndim != 2:
        raise ValueError(f"expected a multi-hot (n, L) target, got shape {y.shape}")
    per_label: list[dict[str, object]] = []
    for k in range(y.shape[1]):
        col = y[:, k].astype(int)
        n_pos, n_neg = int(col.sum()), int((1 - col).sum())
        if n_pos < min_support or n_neg < min_support:
            per_label.append(
                {"label_index": k, "skipped": True, "n_positive": n_pos, "n_negative": n_neg}
            )
            continue
        cp = cross_prediction(z_source, col, label=f"{label}->L{k}", n_inner=n_inner, seed=seed + k)
        per_label.append({**cp.as_dict(), "label_index": k, "skipped": False})
    usable = [
        float(r["auc"])  # type: ignore[arg-type]
        for r in per_label
        if not r["skipped"] and r.get("auc") is not None
    ]
    n_skipped = int(y.shape[1]) - len(usable)
    return {
        "per_label": per_label,
        "n_usable": len(usable),
        "n_skipped": n_skipped,
        "n_labels": int(y.shape[1]),
        "worst_auc": max(usable) if usable else float("nan"),
        "mean_auc": float(np.mean(usable)) if usable else float("nan"),
        "interpretable": bool(usable),
        # A fold that cannot estimate most of its own labels cannot support a claim
        # about them. The M4 Ben fold has 7 clips and 3 of 8 labels reach support; the
        # AUC it returns for the rest would be computed from a handful of windows and
        # then used to fail a gate, which is the metric making a claim it has no data
        # for. `coverage_ok` is the check, and callers must not read `worst_auc` when
        # it is false.
        "coverage_ok": bool(usable) and n_skipped <= max(1, int(0.25 * y.shape[1])),
        "note": (
            "worst per-label AUC is the headline, but only when coverage_ok: a fold "
            "that skips most of its labels has not measured them. The balanced-accuracy "
            "reference for a per-label classifier is 0.5 by construction."
        ),
    }


def signer_probe_auc(
    z: np.ndarray, signers: Sequence[str], *, n_inner: int = 4, seed: int = 0
) -> float:
    """Positive control: can the metric detect entanglement when it is definitely there?

    Signer identity *is* trivially recoverable from a representation that has memorised
    its signer, so a probe on signer labels must score high. If it does not, the metric
    is insensitive and a low cross-prediction AUC means nothing.

    Reported as the mean over held-out signers, so it cannot be inflated by one signer
    dominating.
    """
    g = np.asarray([str(s) for s in signers])
    scores: list[float] = []
    for signer in sorted(set(g.tolist())):
        mask = (g == signer).astype(int)
        if len(set(mask.tolist())) < 2 or mask.sum() == 0 or mask.sum() == len(mask):
            continue
        cp = cross_prediction(z, mask, label="z", n_inner=n_inner, seed=seed)
        if cp.interpretable:
            scores.append(cp.auc)
    if not scores:
        return float("nan")
    return float(np.mean(scores))


@dataclass(slots=True)
class SeparationReport:
    """The M4 gate verdict, with the control that licenses it."""

    z_l: np.ndarray
    z_a: np.ndarray
    affect_labels: Sequence[int]
    linguistic_labels: Sequence[int]
    signers: Sequence[str]
    results: list[CrossPrediction] = field(default_factory=list)

    def control_auc(self) -> float:
        # Both factors, so a gate cannot pass by having one factor memorise the
        # signer and the other be clean.
        a = signer_probe_auc(self.z_l, self.signers)
        b = signer_probe_auc(self.z_a, self.signers)
        vals = [v for v in (a, b) if not np.isnan(v)]
        return float(max(vals)) if vals else float("nan")

    def verdict(self) -> dict[str, object]:
        ctrl = self.control_auc()
        control_ok = not np.isnan(ctrl) and ctrl >= SIGNER_CONTROL_MIN_AUC
        worst = max(
            (r.auc for r in self.results if r.interpretable and not np.isnan(r.auc)),
            default=float("nan"),
        )
        gate_ok = control_ok and not np.isnan(worst) and worst <= SEPARATION_TARGET_AUC
        return {
            "worst_cross_auc": None if np.isnan(worst) else round(worst, 4),
            "signer_control_auc": None if np.isnan(ctrl) else round(ctrl, 4),
            "signer_control_min": SIGNER_CONTROL_MIN_AUC,
            "control_passed": bool(control_ok),
            "separation_target": SEPARATION_TARGET_AUC,
            "gate_passed": bool(gate_ok),
            "interpretable": bool(control_ok),
            "reason": (
                ""
                if control_ok
                else (
                    f"signer control scored {ctrl:.3f}, below {SIGNER_CONTROL_MIN_AUC}; the "
                    "metric cannot detect entanglement here, so no separation number from "
                    "this run may be interpreted"
                )
            ),
        }
