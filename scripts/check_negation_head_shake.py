#!/usr/bin/env python
"""Negation and the head shake: is the association real, or another artifact?

`scripts/label_markers.py` has always tested three expected pairings between a syntactic
label read from the glosses and a visual marker read from the video, by permutation, with
a Bonferroni correction over the three. One of them is negation against head shake. That
pairing has a history: its first result (r = 0.554) was produced by a bug in the yaw
decomposition and withdrawn, and after that fix the marker was blind, because it was
reading the wrong axis.

With the marker reading the head's turn, the registered test now reports an association.
A result that has been wrong once gets checked from more than one side before it is
believed. This script does not re-run the registered test; it reads it from
`artifacts/audit/marker_labels.json` and adds three looks the test does not contain:

1. **The annotators.** Linguists marked head shakes on these clips by hand. If negated
   clips carry more of their head-shake marks too, the phenomenon is in the signing and
   not in our detector.
2. **A placebo axis.** The identical statistic on the nod marker. Negation should not
   predict nodding; if it did, the "association" would be general head movement.
3. **Each signer separately.** Negated clips are spread unevenly over the four signers,
   so a pooled effect could be one person's habit.

These are robustness checks decided after seeing the registered result, and are labelled
so. Writes ``artifacts/audit/negation_head_shake_check.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import label_markers as L

from seam.data import emosign as em
from seam.data.signstream import marker_frame_mask, parse_directory
from seam.paths import artifacts_root, default_data_root
from seam.provenance import KEY as PROVENANCE_KEY
from seam.provenance import stamp


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman's rho with average ranks for ties."""

    def ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        for value in np.unique(x):
            tied = x == value
            r[tied] = r[tied].mean()
        return r

    ra, rb = ranks(np.asarray(a, dtype=np.float64)), ranks(np.asarray(b, dtype=np.float64))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def split(mask: np.ndarray, group: np.ndarray) -> list[int]:
    """[how many in the group have the property, how many are in the group]."""
    return [int((mask & group).sum()), int(group.sum())]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--xml", default=str(default_data_root() / "asllrp_signstream_xml" / "raw"))
    ap.add_argument(
        "--out", default=str(artifacts_root() / "audit" / "negation_head_shake_check.json")
    )
    args = ap.parse_args()

    root = default_data_root()
    clips = L.label_all(root / "emosign" / "landmarks")
    if not clips:
        print("no labelled clips; run `make landmarks` first")
        return 1
    utterances, _ = parse_directory(Path(args.xml))
    by_id = {u.utterance_id: u for u in utterances}
    signer_of = {r.utterance_id: r.signer for r in em.load(root)}

    negated = np.array([bool(c.syntactic.get("negation", False)) for c in clips])
    shake = np.array([c.magnitude.get("head_shake", 0.0) for c in clips], dtype=np.float64)
    nod = np.array([c.magnitude.get("head_nod", 0.0) for c in clips], dtype=np.float64)
    log_duration = np.log([c.frames / max(c.fps, 1e-6) for c in clips])
    signer = np.array([signer_of.get(str(c.utterance_id), "unknown") for c in clips])
    fired = shake > 0

    human = np.full(len(clips), np.nan)
    for i, c in enumerate(clips):
        u = by_id.get(str(c.utterance_id))
        if u is not None:
            human[i] = float(marker_frame_mask(u, "head_shake", c.frames, c.fps).mean())
    known = np.isfinite(human)
    marked = known & (human > 0)

    y = negated.astype(np.float64)
    registered = next(
        (
            a
            for a in json.loads(
                (artifacts_root() / "audit" / "marker_labels.json").read_text(encoding="utf-8")
            )["magnitude_associations"]
            if a["syntactic"] == "negation" and a["visual"] == "head_shake"
        ),
        None,
    )

    by_signer = {}
    for name in sorted(set(signer)):
        m = signer == name
        by_signer[name] = {
            "negated_with_detected_shake": split(fired, m & negated),
            "others_with_detected_shake": split(fired, m & ~negated),
            "standardised_difference": (
                round(L._pb(y[m], shake[m]), 3) if (m & negated).sum() >= 2 else None
            ),
        }

    out = {
        "question": "does the head-shake marker separate negated from non-negated clips, and "
        "is that the head shake or something else?",
        "n_clips": len(clips),
        "n_negated": int(negated.sum()),
        "registered_test": {
            "source": "artifacts/audit/marker_labels.json, magnitude_associations",
            "what": "standardised mean difference of clip-level head-shake magnitude, negated "
            "against other clips, as a partial statistic controlling log duration; "
            "permutation test, Bonferroni over the three registered pairings",
            "result": registered,
        },
        "detected_shake": {
            "negated_clips": split(fired, negated),
            "other_clips": split(fired, ~negated),
        },
        "robustness_checks_decided_after_the_result": {
            "human_annotation": {
                "what": "linguists' own head-shake marks on the same clips",
                "negated_clips_with_a_human_marked_shake": split(marked, known & negated),
                "other_clips_with_a_human_marked_shake": split(marked, known & ~negated),
                "detector_fires_on_clips_with_a_human_marked_shake": split(fired, marked),
                "detector_fires_on_clips_without_one": split(fired, known & ~marked),
                "rank_correlation_detector_magnitude_vs_human_marked_fraction": round(
                    rank_correlation(shake[known], human[known]), 3
                ),
            },
            "placebo_axis": {
                "what": "the same statistic for the nod marker; negation should not predict it",
                "head_shake": round(L._pb(y, shake), 3),
                "head_nod": round(L._pb(y, nod), 3),
                "head_shake_duration_controlled": round(L._partial_pb(y, shake, log_duration), 3),
                "head_nod_duration_controlled": round(L._partial_pb(y, nod, log_duration), 3),
            },
            "by_signer": by_signer,
        },
        "caveats": [
            "27 negated clips, and 15 of them are one signer's",
            "negation is read from the glosses by rule, not annotated; it agrees with the "
            "linguists' negation span on most clips but is not the same thing",
            "the head marker's thresholds were set on these clips against the linguists' "
            "head-shake marks (not against negation), so it is tuned to this corpus",
            "the frame-level head-shake read-out is not validated: 0.70 to 0.76 AUC on "
            "held-out signers, below the registered gate",
        ],
        PROVENANCE_KEY: stamp(__file__),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")

    d = out["detected_shake"]
    h = out["robustness_checks_decided_after_the_result"]["human_annotation"]
    p = out["robustness_checks_decided_after_the_result"]["placebo_axis"]
    print(f"detected head shake: negated {d['negated_clips']}  others {d['other_clips']}")
    print(
        f"human-marked head shake: negated {h['negated_clips_with_a_human_marked_shake']}  "
        f"others {h['other_clips_with_a_human_marked_shake']}"
    )
    print(
        f"standardised difference: head_shake {p['head_shake_duration_controlled']}  "
        f"head_nod (placebo) {p['head_nod_duration_controlled']}"
    )
    for name, row in by_signer.items():
        print(
            f"  {name:<9} negated {row['negated_with_detected_shake']}  "
            f"others {row['others_with_detected_shake']}  d = {row['standardised_difference']}"
        )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
