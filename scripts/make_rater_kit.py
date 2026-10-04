#!/usr/bin/env python3
"""Build the M7 rater kit: the trial list, and a rating sheet with real clip names.

Input is the ``public/manifest.json`` written by :mod:`scripts.make_study_stimuli`.
That script already blinded the filenames; this one does not re-derive or second-guess
them. It groups the manifest into trials, decides which side each clip sits on, and
emits the CSV and the filled-in sheet.

Why "which side" is decided here rather than at generation time: the generator emits
one file per (clip, condition), and nothing in it knows about pairing. Side assignment
is a property of the *study*, not of the stimulus, so pairing and side belong in the
trial list. The side is recorded in ``private/sides.json`` because a fixed side would
let a rater answer by position, and a published side would leak the condition through
the answer key.

Deterministic: same manifest + same seed gives the same sheet. A study instrument that
changes between two printings is not an instrument.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

#: A trial needs exactly two arms. Anything else has no defined forced-choice answer.
ARMS_PER_TRIAL = 2

#: Words that must never appear in a rater-visible file. Mirrors the guard in
#: ``scripts/make_study_stimuli.py`` - duplicated deliberately, because this script
#: writes the *other* files a rater sees and must not rely on the generator having run.
FORBIDDEN_IN_PUBLIC = (
    "system",
    "baseline",
    "condition",
    "proxy",
    "avatar_a",
    "avatar_b",
    "groundtruth",
)


def side_for(seed: int, trial_id: str, token: str) -> int:
    """Return 0 for left, 1 for right. Stable for a given seed/trial/token."""
    h = hashlib.sha256(f"{seed}:{trial_id}:{token}".encode())
    return h.digest()[0] & 1


def load_trials(manifest_path: Path) -> dict[str, list[str]]:
    """Group manifest entries into ``{trial_id: [token, ...]}``."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data["stimuli"] if isinstance(data, dict) else data

    trials: dict[str, list[str]] = {}
    for e in entries:
        trial_id = str(e["trial_id"])
        token = Path(str(e["clip"])).stem
        trials.setdefault(trial_id, []).append(token)

    for trial_id, arms in trials.items():
        if len(arms) != ARMS_PER_TRIAL:
            raise ValueError(
                f"trial {trial_id!r} has {len(arms)} arm(s), expected {ARMS_PER_TRIAL}. "
                "A forced-choice trial compares exactly two clips; the manifest lists "
                f"{arms}."
            )
    return dict(sorted(trials.items()))


def build_sheet(template: str, rows: list[dict[str, str]]) -> str:
    """Fill the trial block of the sheet template with real token pairs."""
    out: list[str] = []
    for r in rows:
        out.append(f"| {r['trial_id']} | `{r['left']}` | `{r['right']}` | ☐ Left ☐ Right | |")
    body = "\n".join(out)

    start = template.find("| Trial | Clip A (left)")
    if start == -1:
        raise ValueError("could not find the trial table in RATING_SHEET.md")
    head_end = template.index("\n", template.index("|---|", start)) + 1
    rest = template[head_end:]
    tail_start = rest.find("| 1 |")
    tail = rest[tail_start:]
    tail = tail[tail.index("\n") + 1 :] if "\n" in tail else ""
    tail_end = tail.find("\n\n")
    tail = tail[tail_end:] if tail_end != -1 else tail
    return template[:head_end] + body + tail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="path to public/manifest.json")
    ap.add_argument("--out", required=True, help="rater-kit directory to write into")
    ap.add_argument("--seed", type=int, default=20261001, help="side-assignment seed")
    args = ap.parse_args(argv)

    manifest = Path(args.manifest)
    if not manifest.is_file():
        print(f"manifest not found: {manifest}")
        print("run scripts/make_study_stimuli.py first - it writes public/manifest.json")
        return 1

    try:
        trials = load_trials(manifest)
    except ValueError as exc:
        print(f"cannot build a trial list: {exc}")
        return 2

    if not trials:
        print("manifest contains no trials")
        return 2

    out = Path(args.out)
    (out / "private").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    sides: dict[str, dict[str, str]] = {}
    for trial_id, arms in trials.items():
        ordered = sorted(arms)
        if side_for(args.seed, trial_id, ordered[0]) == 0:
            left, right = ordered[0], ordered[1]
        else:
            left, right = ordered[1], ordered[0]
        rows.append({"trial_id": trial_id, "left": left, "right": right})
        sides[trial_id] = {"left": left, "right": right}

    with (out / "trials.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["trial_id", "left", "right"])
        w.writeheader()
        w.writerows(rows)

    (out / "private" / "sides.json").write_text(
        json.dumps(sides, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    template_path = out / "RATING_SHEET.md"
    if template_path.is_file():
        filled = build_sheet(template_path.read_text(encoding="utf-8"), rows)
        (out / "RATING_SHEET_filled.md").write_text(filled, encoding="utf-8")
        print(f"filled sheet   : {out / 'RATING_SHEET_filled.md'}")
    else:
        print(f"no template at {template_path}; wrote trials.csv only")

    # Do not walk away believing the sheet is blind without checking.
    blob = (out / "trials.csv").read_text(encoding="utf-8").lower()
    leaked = [w for w in FORBIDDEN_IN_PUBLIC if w in blob]
    if leaked:
        print(f"REFUSING: trials.csv leaks the word(s) {leaked}")
        return 3

    print(f"trials         : {len(rows)}")
    print(f"trial list     : {out / 'trials.csv'}")
    print(f"side assignment: {out / 'private' / 'sides.json'}  (never shown to a rater)")
    print("rater-visible files contain no condition word")
    return 0


if __name__ == "__main__":
    sys.exit(main())
