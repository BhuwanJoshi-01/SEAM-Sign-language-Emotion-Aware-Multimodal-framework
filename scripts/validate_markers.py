#!/usr/bin/env python
"""Can the non-manual markers be read off the face at all? Against human frame labels.

Every marker in this project has so far been a heuristic over blendshapes and head pose,
and none was ever checked against an annotator. M3 found four of six "degenerate" or
"blind" at clip level and stopped there, because there was nothing to check them against.
There is now: the SignStream annotators marked brow raises, brow furrows, head shakes and
head nods with frame bounds on all 200 EmoSign utterances.

Two questions, per marker, at frame level:

1. **How good is the heuristic?** AUC of the detector's own per-frame evidence against
   the human track. No fitting, so no split is needed, but it is reported per held-out
   signer to sit beside (2).
2. **How good could a detector be from the same inputs?** A logistic regression on the
   52 blendshapes and head pose, with short-range dynamics, trained on three signers and
   scored on the fourth.

Design, fixed before the first run
----------------------------------
* **Features per frame:** 52 blendshapes; head roll, pitch, yaw; their rate of change;
  and their rolling standard deviation over a centred 0.5 s window, because a shake or a
  nod is an oscillation and no single frame shows one. Blendshapes and angles are
  centred on the clip's own median, which removes the signer's resting face and the
  camera angle without reading any label.
* **Model:** logistic regression, C=1, balanced class weights, features standardised
  with training-fold statistics.
* **Evaluation:** leave one signer out. Two AUCs per fold: over all test frames, and the
  mean of per-clip AUCs, which cannot be earned by telling clips apart.
* **Gate:** a marker is **validated** when the supervised detector's within-clip AUC is at
  least 0.80 on each of the three large folds (Cory, Jonathan, Rachel), **usable with
  caution** at 0.70, and **not validated** below that. Ben's seven clips are reported and
  do not decide the verdict.

Writes ``artifacts/m3/marker_validation.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from seam.data import emosign as em
from seam.data.signstream import marker_frame_mask, parse_directory
from seam.eval.probes import auc
from seam.features import markers as VM
from seam.paths import artifacts_root, default_data_root
from seam.perception import extract as extract_mod
from seam.provenance import KEY as PROVENANCE_KEY
from seam.provenance import stamp

MARKERS = ("brow_raise", "brow_furrow", "head_shake", "head_nod")
DECIDING_FOLDS = ("Cory", "Jonathan", "Rachel")
VALIDATED, CAUTION = 0.80, 0.70
ROLL_SECONDS = 0.5


def rolling_std(x: np.ndarray, half: int) -> np.ndarray:
    """Centred rolling standard deviation along axis 0, shrinking at the clip edges."""
    out = np.empty_like(x, dtype=np.float64)
    n = len(x)
    for i in range(n):
        out[i] = x[max(i - half, 0) : min(i + half + 1, n)].std(axis=0)
    return out


def frame_features(blendshapes: np.ndarray, rotation: np.ndarray, fps: float) -> np.ndarray:
    """(T, 61) per-frame features: the face, the head, and how the head is moving."""
    bs = np.asarray(blendshapes, dtype=np.float64)
    euler = np.unwrap(VM._euler_from_matrix(np.asarray(rotation, dtype=np.float64)), axis=0)
    rate = np.gradient(euler, axis=0) * fps if len(euler) > 1 else np.zeros_like(euler)
    half = max(round(ROLL_SECONDS * fps / 2), 1)
    wobble = rolling_std(euler, half)
    bs = bs - np.median(bs, axis=0, keepdims=True)
    euler = euler - np.median(euler, axis=0, keepdims=True)
    return np.hstack([bs, euler, rate, wobble])


def load(xml_dir: Path) -> list[dict]:
    root = default_data_root()
    utterances, _ = parse_directory(xml_dir)
    by_id = {u.utterance_id: u for u in utterances}
    shard_root = root / "emosign" / "landmarks"
    clips: list[dict] = []
    for rec in em.load(root):
        u = by_id.get(rec.utterance_id)
        shard = shard_root / f"{rec.utterance_id}.npz"
        if u is None or not shard.is_file():
            continue
        try:
            arrays, meta = extract_mod.load_shard(shard)
        except extract_mod.ExtractionError:
            continue
        fps = float(meta.native_fps)
        bs, rot = arrays["blendshapes"], arrays["head_rotation"]
        n = len(bs)
        sig = VM.signals(bs.astype(np.float64), rot.astype(np.float64), fps=fps)
        evidence = VM.clip_evidence(sig)
        clips.append(
            {
                "id": rec.utterance_id,
                "signer": rec.signer,
                "x": frame_features(bs, rot, fps),
                "y": {m: marker_frame_mask(u, m, n, fps) for m in MARKERS},
                "heuristic_raw": {
                    m: np.asarray(getattr(sig, m), dtype=np.float64) for m in MARKERS
                },
                "heuristic_evidence": {
                    m: np.asarray(evidence[m], dtype=np.float64) for m in MARKERS
                },
            }
        )
    return clips


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(auc(y, s))


def two_aucs(clips: list[dict], marker: str, scores: list[np.ndarray]) -> dict[str, object]:
    """AUC over all frames, and the mean of per-clip AUCs."""
    y_all = np.concatenate([c["y"][marker] for c in clips])
    s_all = np.concatenate(scores)
    per_clip = [safe_auc(c["y"][marker], s) for c, s in zip(clips, scores, strict=True)]
    per_clip = [v for v in per_clip if v == v]
    return {
        "auc_all_frames": round(safe_auc(y_all, s_all), 4),
        "auc_within_clip": round(float(np.mean(per_clip)), 4) if per_clip else float("nan"),
        "n_clips_with_both_classes": len(per_clip),
        "n_frames": len(y_all),
        "positive_fraction": round(float(y_all.mean()), 4),
    }


def supervised_scores(train: list[dict], test: list[dict], marker: str) -> list[np.ndarray] | None:
    from sklearn.linear_model import LogisticRegression

    x_tr = np.vstack([c["x"] for c in train])
    y_tr = np.concatenate([c["y"][marker] for c in train]).astype(int)
    if y_tr.min() == y_tr.max():
        return None
    mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    model.fit((x_tr - mu) / sd, y_tr)
    return [model.decision_function((c["x"] - mu) / sd) for c in test]


def verdict(within_clip: dict[str, float]) -> str:
    vals = [within_clip.get(f, float("nan")) for f in DECIDING_FOLDS]
    if any(v != v for v in vals):
        return "no verdict: a deciding fold has no clip with both classes"
    worst = min(vals)
    if worst >= VALIDATED:
        return "validated"
    if worst >= CAUTION:
        return "usable with caution"
    return "not validated"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--xml", default=str(default_data_root() / "asllrp_signstream_xml" / "raw"))
    ap.add_argument("--out", default=str(artifacts_root() / "m3" / "marker_validation.json"))
    args = ap.parse_args()

    clips = load(Path(args.xml))
    if not clips:
        print("no clips with landmarks and SignStream annotations")
        return 1
    signers = sorted({c["signer"] for c in clips})

    results: dict[str, dict] = {}
    for marker in MARKERS:
        folds: dict[str, dict] = {}
        for held in signers:
            train = [c for c in clips if c["signer"] != held]
            test = [c for c in clips if c["signer"] == held]
            sup = supervised_scores(train, test, marker)
            folds[held] = {
                "n_clips": len(test),
                "heuristic_raw_signal": two_aucs(
                    test, marker, [c["heuristic_raw"][marker] for c in test]
                ),
                "heuristic_clip_evidence": two_aucs(
                    test, marker, [c["heuristic_evidence"][marker] for c in test]
                ),
                "supervised": two_aucs(test, marker, sup) if sup is not None else None,
            }
        within = {
            f: folds[f]["supervised"]["auc_within_clip"]
            for f in signers
            if folds[f]["supervised"] is not None
        }
        heur_within = {f: folds[f]["heuristic_raw_signal"]["auc_within_clip"] for f in signers}
        results[marker] = {
            "folds": folds,
            "supervised_within_clip_auc_by_fold": within,
            "heuristic_within_clip_auc_by_fold": heur_within,
            "verdict_supervised": verdict(within),
            "verdict_heuristic": verdict(heur_within),
        }

    out = {
        "question": "are the four form markers readable from blendshapes and head pose, "
        "frame by frame, against human annotation?",
        "design": {
            "features": "52 blendshapes + head roll/pitch/yaw, their rate, and their rolling "
            f"sd over {ROLL_SECONDS} s; blendshapes and angles centred on the clip median",
            "model": "logistic regression, C=1, balanced class weights, train-fold standardisation",
            "evaluation": "leave one signer out; AUC over all frames and mean per-clip AUC",
            "gate": f"validated when within-clip AUC >= {VALIDATED} on each of "
            f"{', '.join(DECIDING_FOLDS)}; usable with caution at {CAUTION}",
            "labels": "human SignStream annotation, frame-aligned at the clip's fps",
            "fixed_before_first_run": True,
        },
        "n_clips": len(clips),
        "markers": results,
        PROVENANCE_KEY: stamp(__file__),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")

    for marker, r in results.items():
        print(
            f"\n{marker}: supervised {r['verdict_supervised']}; heuristic {r['verdict_heuristic']}"
        )
        print("  within-clip AUC, with the AUC over all frames in brackets")
        print(
            f"  {'fold':<10}{'clips':>6}{'pos%':>7}  {'heuristic':>10}{'evidence':>10}"
            f"{'supervised':>16}"
        )
        for f, d in r["folds"].items():
            h, e, s = d["heuristic_raw_signal"], d["heuristic_clip_evidence"], d["supervised"]
            sup = "n/a" if s is None else f"{s['auc_within_clip']:.3f} [{s['auc_all_frames']:.3f}]"
            print(
                f"  {f:<10}{d['n_clips']:>6}{100 * h['positive_fraction']:>6.1f}%  "
                f"{h['auc_within_clip']:>10.3f}{e['auc_within_clip']:>10.3f}{sup:>16}"
            )
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
