#!/usr/bin/env python
"""Parse a downloaded SignStream XML export and report what it actually contains.

Run this the moment files land in the data directory. It exists because the interesting
question is not "did the parser run" but "how much of the annotation vocabulary does this
project actually cover", and that has to be answered before any of it is used.

Writes `artifacts/m3/signstream_report.json`.

Usage:
    scripts/parse_signstream.py                       # default data dir
    scripts/parse_signstream.py --path /some/dir
    scripts/parse_signstream.py --min-utterances 100  # gate: fail if too little
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seam.data.signstream import parse_directory
from seam.paths import default_data_root

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "m3" / "signstream_report.json"
DEFAULT = default_data_root() / "asllrp_signstream_xml"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=str(DEFAULT))
    ap.add_argument("--pattern", default="*.xml")
    ap.add_argument("--min-utterances", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"no such directory: {root}")
        print(
            "Download the SignStream XML annotations first - see paper/provenance/STEP3_HANDOFF.md"
        )
        return 1

    files = sorted(root.rglob(args.pattern))
    if not files:
        print(f"no {args.pattern} files under {root}")
        print("Check the download went to data/, and that you took the XML annotations")
        print("and not the video-only option. Verify with:")
        print(f"  grep -l NON_MANUALS {root}/**/*.xml | head")
        return 1

    utterances, rep = parse_directory(root, args.pattern)
    out = {
        "source": str(root),
        "pattern": args.pattern,
        "report": rep.as_dict(),
        "per_participant_utterances": dict(rep.participants),
        "per_participant_events": {},
        "sample_utterance": None,
    }
    per: dict[str, dict[str, int]] = {}
    for u in utterances:
        d = per.setdefault(u.participant or "(unknown)", {"utterances": 0, "events": 0})
        d["utterances"] += 1
        d["events"] += len(u.non_manuals)
    out["per_participant_events"] = per
    if utterances:
        s = utterances[0]
        out["sample_utterance"] = {
            "id": s.utterance_id,
            "participant": s.participant,
            "fields": s.fields,
            "non_manuals": [
                {
                    "label": n.label,
                    "value": n.value,
                    "marker": n.marker,
                    "start_frame": n.start_frame,
                    "end_frame": n.end_frame,
                }
                for n in s.non_manuals[:5]
            ],
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    r = rep.as_dict()
    print(f"files        : {r['files']}")
    print(f"utterances   : {r['utterances']}")
    print(f"non-manual events : {r['non_manual_events']}")
    print(f"mapped       : {r['mapped_events']}  (fraction {r['mapped_fraction']})")
    print(f"unmapped label types: {r['n_unmapped_label_types']}")
    print(f"participants : {dict(rep.participants)}")
    if rep.unmapped_labels:
        print("\ntop unmapped labels (these are REAL annotations we cannot use yet):")
        for label, n in rep.unmapped_labels.most_common(12):
            print(f"  {n:5d}  {label}")
    print(f"\nwrote {OUT.relative_to(REPO)}")

    if args.min_utterances and rep.utterances < args.min_utterances:
        print(
            f"\nFAILED: only {rep.utterances} utterances, expected at least "
            f"{args.min_utterances}. The download is probably partial."
        )
        return 1
    if r["mapped_fraction"] is not None and r["mapped_fraction"] < 0.5:
        print(
            f"\nWARNING: only {r['mapped_fraction']:.0%} of events map to a marker. "
            "Before using these labels for M3, either extend MARKER_PATTERNS or report "
            "the low coverage as a limitation."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
