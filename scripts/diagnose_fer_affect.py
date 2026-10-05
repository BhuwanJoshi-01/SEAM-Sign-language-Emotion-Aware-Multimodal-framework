#!/usr/bin/env python
"""Positive control: does the FER read-out respond to affect on sign video?

M1's audit returned per-window shifts of ~0.004 against a negative-mass baseline
of 0.78, and the per-frame probability spread inside a WLASL clip was 0.0002 while
the spread across clips was 0.283 - one output per clip. That is the signature of
a model reading scene, resolution and lighting rather than expression, and such a
model cannot show a marker-dependent shift in a within-clip contrast no matter how
large the marker is. Before reporting a null confound result we have to know which
of those two situations we are in.

The test uses labels we already hold. EmoSign's 200 clips carry a 7-point
sentiment from three Deaf native signers with professional interpreting
experience. If the FER read-out responds to affect on sign video, predicted
negative mass must track true sentiment. If the correlation is at or near zero,
the instrument is not measuring affect and the audit's null is a statement about
the instrument, not about sign language.

This is the positive control the negative control in the audit cannot be: the
audit's uniform null shows the *machinery* is sound, this shows the *instrument*
is live.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seam.data import emosign as em
from seam.eval import fer
from seam.eval import fer_audit as A
from seam.logging import get, setup
from seam.paths import artifacts_root
from seam.perception import extract as extract_mod
from seam.provenance import KEY as PROVENANCE_KEY
from seam.provenance import stamp


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)


def boot_ci(x: np.ndarray, y: np.ndarray, *, n: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) < 3:
            continue
        vals.append(spearman(x[idx], y[idx]))
    return tuple(float(v) for v in np.quantile(vals, [0.025, 0.975]))  # type: ignore[return-value]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fer-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    setup("INFO")
    log = get("fer-sensitivity")
    fer_dir = Path(args.fer_dir) if args.fer_dir else artifacts_root() / "fer"
    out_path = Path(args.out) if args.out else artifacts_root() / "audit" / "fer_sensitivity.json"

    data_root = Path("/mnt/DevProd/seam_data")
    labels = em.load(data_root)
    video_root = data_root / "emosign" / "video"
    shard_root = data_root / "emosign" / "landmarks"

    import torch

    report: dict = {
        "question": "does the FER read-out track true affect on sign video?",
        "dataset": "EmoSign 200, 7-point sentiment, 3 Deaf native annotators",
        "models": {},
    }

    for ckpt in sorted(fer_dir.glob("*.pt")):
        blob = torch.load(ckpt, weights_only=False)
        spec = fer.FerModelSpec(**blob["spec"])
        model = fer.build_model(spec, n_classes=blob["n_classes"])
        model.load_state_dict(blob["state_dict"])
        model.eval()

        rows = []
        for clip in labels:
            vid = video_root / f"{clip.utterance_id}.mp4"
            shard = shard_root / f"{clip.utterance_id}.npz"
            if not vid.is_file() or not shard.is_file():
                continue
            try:
                arrays, _meta = extract_mod.load_shard(shard)
            except extract_mod.ExtractionError:
                continue
            probs = A.score_clip_faces(
                vid,
                arrays["landmarks"].astype(np.float32),
                arrays["presence"].astype(bool),
                model,
            )
            if not len(probs):
                continue
            face = arrays["presence"].astype(bool)[:, 3]
            if face.mean() < 0.5:
                continue
            mean = probs[face[: len(probs)]].mean(axis=0)
            rows.append(
                {
                    "utterance_id": clip.utterance_id,
                    "signer": clip.signer,
                    "sentiment": clip.sentiment,
                    "valence": clip.valence,
                    "probs": [float(v) for v in mean],
                    "neg": float(fer.negative_mass(mean)),
                    "pos": float(fer.positive_mass(mean)),
                }
            )
        if len(rows) < 20:
            log.warning("%s: only %d usable clips", spec.name, len(rows))
            continue

        sent = np.array([r["sentiment"] for r in rows], dtype=float)
        neg = np.array([r["neg"] for r in rows])
        pos = np.array([r["pos"] for r in rows])
        pred = np.argmax(np.array([r["probs"] for r in rows]), axis=1)

        r_sent_neg = spearman(sent, neg)
        ci = boot_ci(sent, neg)
        per_class = {
            cls: spearman(sent, np.array([r["probs"][k] for r in rows]))
            for k, cls in enumerate(fer.EMOSIGN_LABELS)
        }
        report["models"][spec.name] = {
            "n_clips": len(rows),
            "spearman_sentiment_vs_negmass": r_sent_neg,
            "ci95": list(ci),
            "spearman_sentiment_vs_posmass": spearman(sent, pos),
            "spearman_sentiment_per_class": per_class,
            "pred_distribution": {
                cls: int((pred == k).sum()) for k, cls in enumerate(fer.EMOSIGN_LABELS)
            },
            "mean_negmass": float(neg.mean()),
            "per_signer_mean_negmass": {
                s: float(np.mean([r["neg"] for r in rows if r["signer"] == s]))
                for s in sorted({r["signer"] for r in rows})
            },
            "rows": rows,
        }
        log.info(
            "%s: n=%d  spearman(sentiment, negmass) = %+.3f  CI [%.3f, %.3f]",
            spec.name,
            len(rows),
            r_sent_neg,
            ci[0],
            ci[1],
        )

    report[PROVENANCE_KEY] = stamp(__file__)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {out_path}")
    for name, m in report["models"].items():
        lo, hi = m["ci95"]
        print(
            f"  {name}: n={m['n_clips']}  rho(sentiment, negative mass) = "
            f"{m['spearman_sentiment_vs_negmass']:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
        )
        print(f"      predicted label distribution: {m['pred_distribution']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
