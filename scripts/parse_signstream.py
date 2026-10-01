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
    from seam.data.signstream import gloss_vocabulary

    vocab = gloss_vocabulary(utterances)
    per: dict[str, dict[str, int]] = {}
    for u in utterances:
        d = per.setdefault(u.participant or "(unknown)", {"utterances": 0, "events": 0, "signs": 0})
        d["utterances"] += 1
        d["events"] += len(u.non_manuals)
        d["signs"] += len(u.glosses)
    out = {
        "source": str(root),
        "pattern": args.pattern,
        "report": rep.as_dict(),
        "per_participant": per,
        "gloss_vocabulary": {
            "n_types": len(vocab),
            "n_tokens": sum(vocab.values()),
            "hapax_types": sum(1 for v in vocab.values() if v == 1),
            "top_20": vocab.most_common(20),
        },
        "sample_utterance": None,
    }
    if utterances:
        u = next((x for x in utterances if x.non_manuals), utterances[0])
        out["sample_utterance"] = {
            "id": u.utterance_id,
            "participant": u.participant,
            "collection": u.collection,
            "start_frame": u.start_frame,
            "end_frame": u.end_frame,
            "translation": u.translation[:160],
            "glosses": [g for g, _a, _b in u.glosses[:8]],
            "non_manuals": [
                {
                    "label": n.label,
                    "value": n.value,
                    "markers": n.markers,
                    "start_frame": n.start_frame,
                    "end_frame": n.end_frame,
                }
                for n in u.non_manuals[:6]
            ],
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    r = rep.as_dict()
    print(f"files            : {r['files']}   collections: {r['collections']}")
    print(f"utterances       : {r['utterances']}   signs: {r['signs']}")
    print(
        f"non-manual events: {r['non_manual_events']}"
        f"  (frame-aligned: {r['events_with_frame_alignment']})"
    )
    print(
        f"mapped to a marker: {r['events_mapped_to_a_marker']}  (fraction {r['mapped_fraction']})"
    )
    print(f"unmapped pairs   : {r['n_unmapped_pairs']}")
    print(f"participants     : {dict(rep.participants.most_common())}")
    print(f"gloss vocabulary : {len(vocab)} types / {sum(vocab.values())} tokens")
    if rep.events_by_marker:
        print("\nevents by marker:")
        for m, n in rep.events_by_marker.most_common(14):
            print(f"  {n:6d}  {m}")
    if rep.unmapped:
        print("\nunmapped label :: value  (REAL annotations this vocabulary cannot express):")
        for label, n in rep.unmapped.most_common(12):
            print(f"  {n:6d}  {label}")
    if rep.unparseable:
        print("\nUNPARSEABLE FILES:")
        for f in rep.unparseable:
            print(f"  {f}")
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
