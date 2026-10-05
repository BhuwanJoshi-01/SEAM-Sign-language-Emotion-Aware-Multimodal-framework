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


def local_frame_counts() -> tuple[dict[str, int], dict[str, float]]:
    """Frame counts and frame rates from the EmoSign landmark sidecars already extracted.

    The rate comes from the video container, so it is independent of the token table.
    """
    lm = default_data_root() / "emosign" / "landmarks"
    counts: dict[str, int] = {}
    rates: dict[str, float] = {}
    for p in sorted(lm.glob("*.json")):
        try:
            meta = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        uid, fc, fps = meta.get("utterance_id"), meta.get("frame_count"), meta.get("native_fps")
        if uid is not None and fc is not None and fps:
            counts[str(uid)] = int(fc)
            rates[str(uid)] = float(fps)
    return counts, rates


def inferred_rates(tokens: list[asllrp.SignToken], counts: dict[str, int]) -> dict[str, float]:
    """Nearest of 24 or 30 fps to what an utterance's span and frame count imply.

    The mirror listing carries frame counts and no frame rate. Inferring the rate from
    the count makes "in range" true partly by construction for this reference set, so
    what it contributes is the split - how many utterances sit at each ratio - and not
    the aligned fraction.
    """
    out: dict[str, float] = {}
    for t in tokens:
        fc = counts.get(t.utterance_id)
        if not fc or t.utterance_id in out or t.utterance_end <= t.utterance_start:
            continue
        implied = asllrp.SESSION_FPS * fc / (t.utterance_end - t.utterance_start + 1)
        out[t.utterance_id] = min((24.0, 30.0), key=lambda f: abs(f - implied))
    return out


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
        "mapping": asllrp.MAPPING,
        "n_tokens_total": len(tokens),
        "parser": parse_report,
        "reference_sets": {},
    }

    refs: dict[str, tuple[dict[str, int], dict[str, float], str]] = {}
    local_counts, local_rates = local_frame_counts()
    refs["local_emosign_landmarks"] = (local_counts, local_rates, "video container")
    if not args.skip_published:
        try:
            pub = dict(published_frame_counts(args.timeout))
            refs["mirror_dwpose_listing"] = (
                pub,
                inferred_rates(tokens, pub),
                "inferred from span / frame count; in-range is partly by construction",
            )
        except Exception as exc:  # network, 404, malformed listing
            print(f"  published reference unavailable ({type(exc).__name__}: {exc})")

    for name, (counts, rates, rate_source) in refs.items():
        if not counts:
            continue
        rep = asllrp.check_alignment(tokens, counts, rates)
        d = rep.as_dict()
        d["frame_rate_source"] = rate_source
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
                "clip_fps": rates.get(t.utterance_id),
                "mapped_range": asllrp.crop_frame_range(t, clip_fps=rates[t.utterance_id])
                if t.utterance_id in rates
                else None,
            }
        out["reference_sets"][name] = d  # type: ignore[index]
        frac = d.get("aligned_fraction", 0.0)
        print(
            f"  {name}: {d['n_tokens_aligned']}/{d['n_tokens']} in range "
            f"({frac:.1%}), {d['n_tokens_overshoot']} out; frame-for-frame would give "
            f"{d['n_tokens_in_range_one_to_one']}; clips by fps {d['n_utterances_by_fps']}, "
            f"span/frames {d['span_over_frames_ratio_median_by_fps']}"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
