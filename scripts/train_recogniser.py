#!/usr/bin/env python
"""M5a: does signing-space pose carry any lexical signal, signer-disjoint?

**What this is.** A smoke experiment, not a recogniser. It asks one question on the 200
EmoSign utterances: given a body-relative pose summary for a gloss token, can a model
rank the right gloss above chance when it has never seen the signer? If the answer is
no, the alignment is wrong or the features carry nothing, and no amount of model
capacity will help. If the answer is yes, M5a is worth scaling.

**What it is not.** A state-of-the-art result. The corpus is 1,738 tokens over 547 gloss
types - 3.2 tokens per type, 56% of types seen once - and 30% of one fold's tokens use a
gloss the other three signers never use. The open-vocabulary error floor is therefore
set by the data, not the model, and is reported as a number rather than buried in a WER.

**The three controls that keep this honest.** A most-frequent baseline, because a model
that always says IX looks like progress on a corpus where IX is 9% of tokens. A
shuffled-label control, because with 3.2 tokens per type a model can memorise frame
fingerprints and score well without reading the sign at all. And a no-duration variant,
because clip duration correlates with syntactic class on this corpus and would let a
duration classifier pass as a sign recogniser.

Writes `artifacts/m5a/recogniser.json`, or `recogniser_upper_hands.json` for
`--part upper+hands`.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from seam.data import asllrp
from seam.eval.recogniser import LAMBDA_GRID, fit_predict, oov_floor, wer
from seam.features.signpose import Part, feature_dim, token_feature
from seam.paths import default_data_root
from seam.provenance import KEY as PROVENANCE_KEY
from seam.provenance import stamp
from seam.seed import set_seed

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "m5a" / "recogniser.json"


def load_landmarks(root: Path) -> dict[str, tuple[np.ndarray, float]]:
    """Landmarks and the clip's own frame rate, per utterance.

    The frame rate is not optional: token bounds are on a 30 fps session timeline and
    138 of the 200 clips are 24 fps. A clip whose sidecar does not state its rate is
    skipped rather than assumed to be 30 fps, which is the assumption that misaligned
    every 24 fps clip in the first M5a run.
    """
    lm = root / "emosign" / "landmarks"
    out: dict[str, tuple[np.ndarray, float]] = {}
    for p in sorted(lm.glob("*.npz")):
        uid = p.stem
        side = p.with_suffix(".json")
        if not side.is_file():
            continue
        fps = json.loads(side.read_text()).get("native_fps")
        if not fps:
            continue
        with np.load(p) as d:
            out[uid] = (np.asarray(d["landmarks"], dtype=np.float32), float(fps))
    return out


def build(
    part: Part, stats: tuple[str, ...]
) -> tuple[np.ndarray, list[str], list[str], dict[str, int]]:
    """Assemble (X, y, signer, drop_counts) over every alignable token.

    Tokens whose mapped range overshoots their clip are **dropped and counted**, never
    clamped: clamping would label the wrong frames, and the resulting WER would be
    describing the misalignment rather than the model.
    """
    toks, _ = asllrp.load()
    lms = load_landmarks(default_data_root())
    X: list[np.ndarray] = []
    y: list[str] = []
    sg: list[str] = []
    drops: collections.Counter[str] = collections.Counter()

    for t in toks:
        got = lms.get(t.utterance_id)
        if got is None:
            drops["no_landmarks"] += 1
            continue
        lm, fps = got
        span = asllrp.crop_frame_range(t, clip_fps=fps)
        if span is None:
            drops["before_utterance"] += 1
            continue
        start, end = span
        if end > lm.shape[0]:
            drops["overshoots_crop"] += 1
            continue
        try:
            X.append(token_feature(lm, start, end, part=part, stats=stats))
        except IndexError:
            drops["feature_failed"] += 1
            continue
        y.append(t.gloss)
        sg.append(t.signer)

    return np.asarray(X, dtype=np.float64), y, sg, dict(drops)


def evaluate(X: np.ndarray, y: np.ndarray, sg: np.ndarray, use_duration: bool, seed: int) -> dict:
    """Leave-one-signer-out with the OOV floor and both controls reported separately."""
    signers = sorted(set(sg))
    rows: list[dict] = []
    for held in signers:
        tr, te = sg != held, sg == held
        train_glosses = collections.Counter(y[tr])
        classes = sorted(train_glosses)
        # OOV floor: a token whose gloss is absent from training cannot be predicted
        # correctly by any model restricted to the training vocabulary. This is a
        # property of the split, and is reported so the WER below can be read against it.
        test_ref = list(y[te])
        oov_frac = oov_floor(train_glosses, test_ref)
        oov_count = sum(1 for v in test_ref if v not in train_glosses)
        Xtr, Xte = X[tr], X[te]
        if not use_duration:
            Xtr, Xte = Xtr[:, :-1], Xte[:, :-1]
        set_seed(seed)
        hyp, fit = fit_predict(Xtr, np.asarray(y)[tr], sg[tr], Xte)
        ref = list(np.asarray(y)[te])
        # Most-frequent baseline on the same split.
        top = train_glosses.most_common(1)[0][0]
        base = [top] * len(ref)
        # Closed-vocabulary view: score only the tokens whose gloss training has seen, so
        # the vocabulary size stops being the thing being measured.
        seen = np.array([v in train_glosses for v in ref])
        closed = (
            wer(
                [r for r, s in zip(ref, seen, strict=True) if s],
                [h for h, s in zip(hyp, seen, strict=True) if s],
            )
            if seen.any()
            else float("nan")
        )

        rows.append(
            {
                "held_out": held,
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                "n_classes": len(classes),
                # Count and fraction are stored separately and computed separately. An
                # earlier version put the fraction in `oov_tokens` and then divided it
                # by the token count a second time, which turned a real 29.5% OOV floor
                # into a reported 0.1% and hid the single most important fact about
                # this corpus.
                "oov_tokens": oov_count,
                "oov_fraction": round(oov_frac, 4),
                "wer": round(wer(ref, hyp), 4),
                "wer_closed_vocab": round(closed, 4),
                "top1": round(float(np.mean([a == b for a, b in zip(ref, hyp, strict=True)])), 4),
                "wer_most_frequent_baseline": round(wer(ref, base), 4),
                "lambda": fit["lambda"],
                "n_distinct_predictions": fit["n_distinct_predictions"],
                "constant_predictor": fit["constant"],
            }
        )
    return {
        "folds": rows,
        "signers": signers,
        # A model that says one thing cannot be scored against a most-frequent
        # baseline: it *is* one. Reported so its WER is not read as a measurement.
        "degenerate": all(r["constant_predictor"] for r in rows),
    }


def shuffled_control(X: np.ndarray, y: np.ndarray, sg: np.ndarray, seed: int) -> dict:
    """Same pipeline, labels permuted within the corpus.

    If a permuted-label model scores near the real one, the score is coming from token
    identity or frame fingerprints - which with 3.2 tokens per gloss type is a real
    risk, not a hypothetical one.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    return evaluate(X, np.asarray(y)[perm], sg, use_duration=True, seed=seed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", default="upper", choices=["upper", "upper+hands"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stats = ("mean", "std", "delta")
    X, y, sg, drops = build(args.part, stats)  # type: ignore[arg-type]
    if not len(y):
        print("no alignable tokens; run `make landmarks` first")
        return 1
    y_arr, sg_arr = np.asarray(y), np.asarray(sg)
    vocab = collections.Counter(y)

    print(f"tokens: {len(y)} over {len(vocab)} glosses ({len(y) / len(vocab):.2f} per type)")
    print(f"dropped: {drops}")
    print(f"hapax types: {sum(1 for v in vocab.values() if v == 1)}/{len(vocab)}")

    real = evaluate(X, y_arr, sg_arr, use_duration=True, seed=args.seed)
    no_dur = evaluate(X, y_arr, sg_arr, use_duration=False, seed=args.seed)
    ctrl = shuffled_control(X, y_arr, sg_arr, seed=args.seed)

    out: dict[str, object] = {
        "design": {
            "part": args.part,
            "stats": list(stats),
            "feature_dim": feature_dim(args.part, stats),  # type: ignore[arg-type]
            "n_tokens": len(y),
            "n_glosses": len(vocab),
            "tokens_per_gloss": round(len(y) / len(vocab), 2),
            "hapax_types": sum(1 for v in vocab.values() if v == 1),
            "dropped": drops,
            "seed": args.seed,
            "frame_mapping": asllrp.MAPPING,
            "features": "z-scored with training-split statistics",
            "lambda_grid": list(LAMBDA_GRID),
            "lambda_selection": "leave-one-signer-out inside the training signers",
        },
        "real": real,
        "real_without_duration": no_dur,
        "shuffled_label_control": ctrl,
        "caveat": (
            f"Smoke experiment on {len(y)} tokens over {len(vocab)} gloss types. The "
            "open-vocabulary WER is dominated by the OOV floor reported per fold, not by "
            "the model; read wer_closed_vocab, n_distinct_predictions and the "
            "shuffled-label control before drawing any conclusion about pose features. "
            "Duration is excluded in one arm because clip duration correlates with "
            "syntactic class on this corpus."
        ),
    }
    out[PROVENANCE_KEY] = stamp(__file__)
    # One file per feature set: the hands variant used to overwrite the body-only run.
    dest = OUT if args.part == "upper" else OUT.with_name("recogniser_upper_hands.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n{'fold':<10}{'n':>6}{'oov%':>7}{'WER':>8}{'closed':>8}{'top1':>7}{'base':>8}")
    for f in real["folds"]:  # type: ignore[index]
        print(
            f"{f['held_out']:<10}{f['n_test']:>6}{f['oov_fraction'] * 100:>6.1f}%"
            f"{f['wer']:>8.3f}{f['wer_closed_vocab']:>8.3f}{f['top1']:>7.3f}"
            f"{f['wer_most_frequent_baseline']:>8.3f}"
        )
    mean = lambda k, r=real: float(np.mean([f[k] for f in r["folds"]]))  # noqa: E731
    print(
        f"\nmean WER {mean('wer'):.3f} | closed-vocab {mean('wer_closed_vocab'):.3f} | "
        f"most-frequent baseline {mean('wer_most_frequent_baseline'):.3f}"
    )
    print(f"shuffled-label control WER {mean('wer', ctrl):.3f}")
    print(
        f"closed-vocab WER: real {mean('wer_closed_vocab'):.3f}, "
        f"shuffled {mean('wer_closed_vocab', ctrl):.3f}"
    )
    for name, arm in (("real", real), ("shuffled", ctrl)):
        d = [f["n_distinct_predictions"] for f in arm["folds"]]  # type: ignore[index]
        lams = [f["lambda"] for f in arm["folds"]]  # type: ignore[index]
        print(f"{name}: distinct glosses predicted per fold {d}, lambda per fold {lams}")
    if real["degenerate"]:
        print("DEGENERATE: the model predicts one gloss in every fold; its WER is the baseline's")
    print(f"without duration: {mean('wer', no_dur):.3f}")
    print(f"\nwrote {dest.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
