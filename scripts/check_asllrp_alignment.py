#!/usr/bin/env python
"""Persist the ASLLRP token-to-crop-frame alignment report as an artifact.

The 89.5% figure in `EXPERIMENT_LOG.md` was originally produced in an interactive shell
and written into the log by hand, which is exactly the provenance hole
`tests/test_provenance.py` exists to catch. This script makes it reproducible.

Two reference sets, because they answer slightly different questions:

- **local**: frame counts from the EmoSign landmark sidecars already on disk. No network.
  This is the set the 200 extracted clips correspond to.
- **published**: per-utterance frame counts from the Hugging Face ASLLRP mirror listing.
  Larger, and independent of anything extracted locally. Needs network; cached after
  the first fetch.

Writes `artifacts/data/asllrp_alignment.json`.
"""

from __future__ import annotations

import argparse
import collections
import json
import urllib.request
from pathlib import Path

from seam.data import asllrp
from seam.paths import default_data_root

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "data" / "asllrp_alignment.json"
CACHE = Path("/tmp/opencode/asllrp_listing.json")
MIRROR = "https://huggingface.co/api/datasets/FangSen9000/ASLLRP_utterances_results"


def local_frame_counts() -> dict[str, int]:
    """Frame counts from the EmoSign landmark sidecars already extracted."""
    lm = default_data_root() / "emosign" / "landmarks"
    counts: dict[str, int] = {}
    for p in sorted(lm.glob("*.json")):
        try:
            meta = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        uid, fc = meta.get("utterance_id"), meta.get("frame_count")
        if uid is not None and fc is not None:
            counts[str(uid)] = int(fc)
    return counts


def published_frame_counts(timeout: int) -> dict[str, int]:
    """Per-utterance DWPose frame counts from the mirror listing, cached locally."""
    if CACHE.exists():
        try:
            sib = json.loads(CACHE.read_text())["siblings"]
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
            sib = None
    else:
        sib = None
    if sib is None:
        with urllib.request.urlopen(MIRROR, timeout=timeout) as r:
            sib = json.loads(r.read())["siblings"]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps({"siblings": sib}))
    return collections.Counter(
        p.split("/")[1] for p in (s["rfilename"] for s in sib) if "/results_dwpose/npz/" in p
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-published", action="store_true", help="local counts only")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    tokens, parse_report = asllrp.load()
    out: dict[str, object] = {
        "mapping": "crop frame index = session frame - utterance_start + 1",
        "n_tokens_total": len(tokens),
        "parser": parse_report,
        "reference_sets": {},
    }

    refs = {"local_emosign_landmarks": local_frame_counts()}
    if not args.skip_published:
        try:
            refs["mirror_dwpose_listing"] = dict(published_frame_counts(args.timeout))
        except Exception as exc:  # network, 404, malformed listing
            print(f"  published reference unavailable ({type(exc).__name__}: {exc})")

    for name, counts in refs.items():
        if not counts:
            continue
        rep = asllrp.check_alignment(tokens, counts)
        d = rep.as_dict()
        # A worked example, so the mapping is checkable by hand and not just asserted.
        rows = [t for t in tokens if t.utterance_id in counts]
        if rows:
            t = rows[0]
            d["worked_example"] = {
                "utterance_id": t.utterance_id,
                "utterance_start": t.utterance_start,
                "token_start_frame": t.start_frame,
                "token_end_frame": t.end_frame,
                "crop_frames": counts[t.utterance_id],
                "mapped_start": asllrp.crop_frame_index(t, t.start_frame),
                "mapped_end": asllrp.crop_frame_index(t, t.end_frame),
            }
        out["reference_sets"][name] = d  # type: ignore[index]
        frac = d.get("aligned_fraction", 0.0)
        print(
            f"  {name}: {d['n_tokens_aligned']}/{d['n_tokens']} aligned "
            f"({frac:.1%}), {d['n_tokens_overshoot']} overshoot -> aligned={d['aligned']}"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
