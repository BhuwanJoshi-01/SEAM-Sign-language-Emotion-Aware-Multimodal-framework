"""Gloss-recognition metrics and a small ridge multinomial-logistic baseline (M5a).

Kept in the package rather than in `scripts/train_recogniser.py` so the metric is
importable and testable. That is not tidiness: `wer` was inverted in the first run of
M5a - identical reference and hypothesis scored 1.0, a completely wrong one scored 0.0
- which inverted every recognition number in the experiment. The inverted values looked
*better* than chance rather than worse, so no gate caught them. A metric that lives in a
CLI script is a metric nobody tests.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence

import numpy as np


def wer(ref: Sequence[str], hyp: Sequence[str]) -> float:
    """Word error rate: Levenshtein edit distance divided by the reference length.

    Anchors: identical sequences give 0.0, no correct gloss gives 1.0. Pinned in
    `tests/test_recogniser.py` because the inverse of this function produces numbers
    that look like success.
    """
    r, h = list(ref), list(hyp)
    if not r:
        return 0.0
    prev = list(range(len(h) + 1))
    for i, ri in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hj in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ri != hj))
        prev = cur
    return prev[-1] / len(r)


def per_class_recall(
    ref: Sequence[str], hyp: Sequence[str], classes: Sequence[str]
) -> dict[str, float]:
    """Recall per class over the tokens whose reference is that class.

    Macro-averaged over classes the model could in principle have predicted, so a fold
    dominated by frequent glosses cannot hide behind a good aggregate WER.
    """
    out: dict[str, float] = {}
    for c in classes:
        idx = [i for i, v in enumerate(ref) if v == c]
        if not idx:
            continue
        out[c] = float(np.mean([hyp[i] == c for i in idx]))
    return out


def oov_floor(train_glosses: collections.Counter[str], ref: Sequence[str]) -> float:
    """Fraction of test tokens whose gloss the training vocabulary does not contain.

    Structural, not a modelling failure: a classifier restricted to the training
    vocabulary must be wrong on every one of them. On the EmoSign 200 this is 24-48% by
    fold and it dominates the open-vocabulary WER, so it is reported beside the score
    rather than folded into it.
    """
    if not ref:
        return 0.0
    return sum(1 for v in ref if v not in train_glosses) / len(ref)


def fit_logreg(X: np.ndarray, y: Sequence[str], classes: Sequence[str], lam: float) -> np.ndarray:
    """Ridge-regularised multinomial logistic regression, full-batch gradient descent.

    Written out rather than taken from a library so the regulariser is the only knob
    and the whole fit is a few lines that can be checked. The corpus is far too small
    for anything more elaborate, and the shuffled-label control only means something if
    both arms go through identical code.
    """
    Xb = np.asarray(X, dtype=np.float64)
    n, d = Xb.shape
    index = {c: i for i, c in enumerate(classes)}
    missing = set(y) - set(index)
    if missing:
        raise ValueError(f"{len(missing)} label(s) absent from the training vocabulary")
    yi = np.array([index[v] for v in y], dtype=int)
    Xb = np.hstack([Xb, np.ones((n, 1))])
    W = np.zeros((d + 1, len(classes)))
    lr, epochs = 0.5, 400
    if lr * lam >= 1.0:
        # The decay step multiplies the weights by (1 - lr * lam). At or past 1 that
        # flips their sign every epoch and past 2 it diverges to inf, after which every
        # logit is NaN and argmax returns class 0 for everything - a constant predictor
        # that looks like a very strong regulariser.
        raise ValueError(f"ridge {lam} is unstable at step size {lr}; need lam < {1 / lr:g}")
    for _ in range(epochs):
        logits = Xb @ W
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        p[np.arange(n), yi] -= 1.0
        p /= n
        W -= lr * (Xb.T @ p)
        W[:-1] -= lr * lam * W[:-1]
    return W


def predict(W: np.ndarray, classes: Sequence[str], X: np.ndarray) -> list[str]:
    """Argmax over the model's logits, not over the features.

    An earlier version returned ``argmax`` of X itself, so the "prediction" depended on
    which feature column happened to be largest and the fitted weights were never read.
    The bias column is appended here, matching `fit_logreg`.
    """
    cls = list(classes)
    Xb = np.asarray(X, dtype=np.float64)
    logits = np.hstack([Xb, np.ones((Xb.shape[0], 1))]) @ W
    return [cls[i] for i in logits.argmax(axis=1)]


#: Ridge strengths tried by `fit_predict`, strongest first. Three values an order of
#: magnitude apart, fixed before any result was looked at; the choice among them is made
#: on training signers only.
LAMBDA_GRID = (1.0, 0.1, 0.01)


def standardise(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Z-score every array with the *training* mean and standard deviation.

    Pose features on this corpus have standard deviations between 0 and 0.5. A ridge of
    1.0 on features that small leaves the weights at zero and the bias doing all the
    work - which is a most-frequent-class predictor, not a weak recogniser.
    """
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return tuple((a - mu) / sd for a in (train, *others))


def select_lambda(
    X: np.ndarray,
    y: Sequence[str],
    groups: Sequence[str],
    grid: Sequence[float] = LAMBDA_GRID,
) -> float:
    """Pick the ridge strength by leave-one-group-out *inside the training set*.

    The criterion is accuracy on inner-held-out tokens whose gloss the inner training
    split has seen. Ties go to the stronger regulariser. The test signer is never
    consulted, so the choice cannot leak, and the shuffled-label control goes through
    the same selection - a control that skipped it would be easier to beat.
    """
    y_arr, g_arr = np.asarray(y), np.asarray(groups)
    inner = sorted(set(g_arr.tolist()))
    if len(inner) < 2 or len(grid) == 1:
        return float(grid[0])
    best_lam, best_hits = float(grid[0]), -1
    for lam in grid:
        hits = 0
        for held in inner:
            tr, te = g_arr != held, g_arr == held
            classes = sorted(set(y_arr[tr].tolist()))
            Xtr, Xte = standardise(X[tr], X[te])
            W = fit_logreg(Xtr, y_arr[tr], classes, lam=float(lam))
            hyp = predict(W, classes, Xte)
            hits += sum(1 for h, r in zip(hyp, y_arr[te], strict=True) if h == r)
        if hits > best_hits:
            best_lam, best_hits = float(lam), hits
    return best_lam


def fit_predict(
    X_train: np.ndarray,
    y_train: Sequence[str],
    groups_train: Sequence[str],
    X_test: np.ndarray,
    grid: Sequence[float] = LAMBDA_GRID,
) -> tuple[list[str], dict[str, object]]:
    """Standardise, select the regulariser on training groups, fit, predict.

    Returns the predictions and a record of what the model did, including how many
    distinct glosses it predicted. **A model that predicts one gloss is not a weak
    result, it is no result**: its error rate equals the most-frequent baseline by
    construction, on real and shuffled labels alike. The first M5a run was exactly that
    - one distinct prediction in every fold - and was reported as "pose carries no
    usable lexical signal". ``constant`` is the flag that makes it visible.
    """
    classes = sorted(set(np.asarray(y_train).tolist()))
    lam = select_lambda(X_train, y_train, groups_train, grid)
    Xtr, Xte = standardise(X_train, X_test)
    W = fit_logreg(Xtr, y_train, classes, lam=lam)
    hyp = predict(W, classes, Xte)
    distinct = len(set(hyp))
    return hyp, {"lambda": lam, "n_distinct_predictions": distinct, "constant": distinct <= 1}
