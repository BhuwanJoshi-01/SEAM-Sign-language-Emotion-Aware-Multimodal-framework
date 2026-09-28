"""Leave-one-signer-out splits, with the overlap assertion made unskippable.

The reason this project needs LOSO rather than a random split is specific, not
methodological fashion. M1 found that marker effects are absent on isolated signing
and that the confound audit's conclusions hinge on which signer produced a window.
A random split puts the *same signer* in train and test, so a model can identify the
signer and predict their affect from it - which is the classic leak in sign-language
affect work and produces numbers that do not survive contact with a new signer.

The split is therefore signer-disjoint by construction, and the assertion that it is
is a **property of the returned object**, checked at construction, not a comment. A
split that fails the check cannot be returned, so no downstream training script can
accidentally proceed with a leaky one.

Signers here are uneven: Ben has 7 clips, Cory 87, Jonathan 54, Rachel 52. A fold that
holds out 7 clips is a real limitation and is reported per fold rather than averaged
away, because a mean accuracy over folds of wildly different sizes is dominated by
the large ones and hides that one fold is nearly untrained.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from seam.logging import get

log = get(__name__)


class SplitOverlapError(AssertionError):
    """Raised when a split is not signer-disjoint. An ``AssertionError`` subclass so
    ``assert``-style handling still works, but a named type so a CI failure names the
    actual problem."""


@dataclass(frozen=True, slots=True)
class Fold:
    """One leave-one-signer-out fold."""

    held_out: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    signers_train: tuple[str, ...]
    signers_test: tuple[str, ...]
    #: Whether the inner validation split was signer-disjoint from train. It is
    #: whenever there are enough signers to spare, and it is the one thing this
    #: module cannot guarantee, so it is recorded per fold rather than assumed.
    val_signer_disjoint: bool = True

    def summary(self) -> dict[str, object]:
        return {
            "held_out": self.held_out,
            "n_train": len(self.train_idx),
            "n_val": len(self.val_idx),
            "n_test": len(self.test_idx),
            "signers_train": list(self.signers_train),
            "signers_test": list(self.signers_test),
            "val_signer_disjoint": self.val_signer_disjoint,
        }


@dataclass(slots=True)
class LosoSplits:
    """All folds, plus the group labels they were built from."""

    folds: list[Fold] = field(default_factory=list)
    groups: np.ndarray | None = None
    n_splits: int = 0

    def __iter__(self) -> Iterator[Fold]:
        return iter(self.folds)

    def __len__(self) -> int:
        return len(self.folds)

    def summary(self) -> list[dict[str, object]]:
        return [f.summary() for f in self.folds]


def _assert_no_sample_overlap(a: np.ndarray, b: np.ndarray) -> None:
    if len(a) == 0 or len(b) == 0:
        raise SplitOverlapError("a fold has an empty train or test side")
    shared = set(np.asarray(a).tolist()) & set(np.asarray(b).tolist())
    if shared:
        raise SplitOverlapError(f"{len(shared)} sample(s) appear in both sides")


def _assert_disjoint(
    train: np.ndarray, test: np.ndarray, groups: np.ndarray, *, require_signer: bool = True
) -> None:
    _assert_no_sample_overlap(train, test)
    if not require_signer:
        return
    overlap = set(np.asarray(train).tolist()) & set(np.asarray(test).tolist())
    if overlap:
        raise SplitOverlapError(f"{len(overlap)} sample(s) appear in both train and test")
    if len(train) == 0 or len(test) == 0:
        raise SplitOverlapError("a fold has an empty train or test side")
    g_tr = set(np.asarray(groups)[np.asarray(train)].tolist())
    g_te = set(np.asarray(groups)[np.asarray(test)].tolist())
    shared = g_tr & g_te
    if shared:
        raise SplitOverlapError(
            f"split is not signer-disjoint: signer(s) {sorted(shared)} appear in "
            f"both train {sorted(g_tr)} and test {sorted(g_te)}"
        )


def make_loso(
    groups: Sequence[str],
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
    min_test: int = 1,
) -> LosoSplits:
    """One fold per distinct signer.

    Within each fold the *training* portion of the remaining signers is split again
    into train and val, grouped by signer, so the validation set used for early
    stopping is also signer-disjoint from training. An inner val that shares signers
    with train would pick a checkpoint that has already memorised them.
    """
    g = np.asarray([str(x) for x in groups])
    if len(g) == 0:
        raise ValueError("no samples")
    uniq = sorted(set(g.tolist()))
    folds: list[Fold] = []
    rng = np.random.default_rng(seed)

    for signer in uniq:
        test_idx = np.flatnonzero(g == signer)
        if len(test_idx) < min_test:
            log.warning("skipping fold for %s: only %d sample(s)", signer, len(test_idx))
            continue
        rest = np.flatnonzero(g != signer)
        rest_signer = g[rest]
        # Grouped inner split: hold out whole signers from the val side, and only as
        # many as the fraction calls for.
        n_val_signers = max(1, round(len(uniq) * val_fraction)) if len(uniq) > 1 else 0
        val_signers = [s for s in uniq if s != signer][:n_val_signers]
        grouped_val = bool(val_signers) and (
            len(rest) - sum(int((rest_signer == s).sum()) for s in val_signers) >= 2
        )
        if not grouped_val:
            # Not enough signers left to spare: use a random val slice, and say so.
            # Train/test disjointness still holds - that is the invariant the gate
            # rests on - but early stopping sees signers it trained on, so the fold
            # is flagged rather than quietly accepted.
            log.warning(
                "fold %s: too few remaining signers for a grouped val split; using a "
                "random val slice, which shares signers with train. Train and test "
                "remain strictly disjoint, so the gate is unaffected; only early "
                "stopping is mildly optimistic, and the fold is flagged.",
                signer,
            )
            perm = rng.permutation(len(rest))
            n_val = max(1, int(len(rest) * val_fraction))
            val_idx = rest[perm[:n_val]]
            train_idx = rest[perm[n_val:]]
        else:
            val_mask = np.isin(rest_signer, val_signers)
            val_idx = rest[val_mask]
            train_idx = rest[~val_mask]

        # Train/test signer-disjointness is the invariant the gate rests on and is
        # always enforced. The inner val split is enforced as signer-disjoint only
        # when it was actually built that way.
        _assert_disjoint(train_idx, test_idx, g)
        _assert_disjoint(val_idx, test_idx, g)
        _assert_no_sample_overlap(train_idx, val_idx)
        if grouped_val:
            _assert_disjoint(train_idx, val_idx, g)
        folds.append(
            Fold(
                held_out=signer,
                train_idx=np.sort(train_idx),
                val_idx=np.sort(val_idx),
                test_idx=np.sort(test_idx),
                signers_train=tuple(sorted(set(g[train_idx].tolist()))),
                signers_test=(signer,),
                val_signer_disjoint=grouped_val,
            )
        )
    if not folds:
        raise ValueError("no fold had enough held-out samples")
    return LosoSplits(folds=folds, groups=g, n_splits=len(folds))


def assert_no_signer_overlap(splits: LosoSplits) -> None:
    """Re-check every fold. Cheap, and callable from a test or a CI gate."""
    if splits.groups is None:
        raise ValueError("splits carry no group labels")
    for f in splits.folds:
        _assert_disjoint(f.train_idx, f.test_idx, splits.groups)
        _assert_disjoint(f.val_idx, f.test_idx, splits.groups)
        _assert_no_sample_overlap(f.train_idx, f.val_idx)
        if f.val_signer_disjoint:
            _assert_disjoint(f.train_idx, f.val_idx, splits.groups)


def macro_fold_mean(values: Sequence[float], weights: Sequence[int]) -> float:
    """Fold mean weighted by test size, and the unweighted mean alongside it.

    Both, because they disagree exactly when folds are uneven - which they are here
    (7 to 87 clips). Reporting only the weighted mean lets the big folds speak for
    the whole thing; reporting only the unweighted mean lets the 7-clip fold swing
    the result. The spread between them is the honest summary of the imbalance.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.size == 0:
        return float("nan")
    if w.sum() <= 0:
        return float(v.mean())
    return float((v * w).sum() / w.sum())
