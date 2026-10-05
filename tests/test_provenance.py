"""Fail if the paper cites a number no artifact produced.

Every quantitative claim in `EXPERIMENT_LOG.md` and `CLAIMS_LEDGER.md` has to come from
a file under `artifacts/`. This test makes that a build failure rather than a habit,
because the alternative is already the known failure mode in this project: a number gets
written into the log by hand, the artifact behind it gets regenerated or deleted, and
nothing notices because prose does not execute.

The check is deliberately strict about the *allowlist*. An allowlist entry must name why
the number cannot come from an artifact - "it is a section number" is a reason, "it is
approximately right" is not - and a number that is allowlisted but then becomes
traceable should be removed rather than left to rot.

Deliberate non-goals:

- This does not check that a number is *correct*, only that something emitted it. A
  wrong number faithfully reproduced from an artifact passes. What it catches is the
  more common failure: a number with no provenance at all.
- Numbers in code, tests and comments are not scanned. Only the two documents that make
  claims to a reader are.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"
ARTIFACTS = REPO / "artifacts"
CLAIMED_DOCS = ("EXPERIMENT_LOG.md", "CLAIMS_LEDGER.md")

#: `report.md` quotes measurements back to the people verifying them, so a number in it
#: that no artifact produced is a verification instruction that cannot be carried out.
#: The file is excluded from the *line-level* ratchet because it also quotes external
#: facts - a remote file size, a byte count - which are not measurements of this project.
#: What it must not do is quote a project measurement wrongly, so it gets a targeted
#: check against the specific values it asserts instead.
#:
#: The three M4 values changed on 2026-10-05, when M4 was re-run on yaw-fixed code with the
#: baseline-weighting and checkpoint-selection defects corrected (0.7276 / 0.9729 / 0.5048
#: before). The gate still fails; see that date in EXPERIMENT_LOG.md.
REPORT_QUOTED = {
    "worst_cross_auc": 0.6944,
    "signer_control_max": 0.9812,
    "cross_a_to_l": 0.5891,
    # M5a, re-run 2026-10-05 with the 30 fps -> clip-rate frame mapping and a classifier
    # that can predict more than one gloss (1563 / 499 / 284 / 0.916 / 0.916 / 0.911 before).
    "n_tokens": 1736,
    "n_glosses": 546,
    "hapax_types": 306,
    "wer_mean": 0.92,
    "wer_baseline": 0.92,
    "wer_shuffled": 0.91,
    "tar_bytes": 1169520640,
}

#: Numbers that legitimately have no artifact behind them, each with the reason.
#: A new entry is a claim that the number is not evidence, and should be argued for in
#: review rather than added to make a red test go green.
ALLOWED: dict[str, str] = {
    "0": "Baseline for a one-sided test.",
    "1": "Ordinals ('track 1', '1 frame').",
    "2": "Ordinals and the two-sided multiplier in a p-value correction.",
    "3": "Ordinals; also the k in Bonferroni for three tests.",
    "4": "Number of signers/folds.",
    "5": "Number of folds in k-fold, raters in the study target.",
    "6": "Ordinals.",
    "8.3": "A document section reference (ASLLRP Report 18 section 8.3), not a measurement.",
    "10": "Ordinals, and the FLAME coefficient count.",
    "12": "fps in the M5b design target.",
    "24": "fps target; number of sampled clips in the face gate.",
    "30": "fps of the source recordings.",
    "45": "MANO coefficients per hand.",
    "52": "MediaPipe blendshape count.",
    "68": "DWPose face landmark count.",
    "100": "Percentage.",
    "200": "EmoSign utterance count, and a percentage.",
    "2026": "Year.",
}

# Numbers in these contexts are structural rather than evidentiary.
_STRUCTURAL = re.compile(
    r"^#+\s|^\s*[-*]\s*$|^\s*\|"  # headings, rules, table rows
)

_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])")

#: A number is traced if it matches an artifact value to within this relative tolerance.
#: Loose enough to survive a re-run that differs in the last digit, tight enough that
#: 0.691 does not match 0.728.
_REL_TOL = 0.005


def _artifact_values() -> set[str]:
    """Every numeric leaf value written by any artifact, normalised as a string."""
    out: set[str] = set()
    for path in ARTIFACTS.rglob("*.json"):
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        def walk(node: object) -> None:
            if isinstance(node, dict):
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
            elif isinstance(node, bool):
                pass
            elif isinstance(node, (int, float)):
                # JSON allows NaN and Infinity, and several reports use them for
                # "not applicable" - an undefined CV, a correlation with no variance.
                # They are not measurements and must not be matched against a citation.
                x = float(node)
                if math.isfinite(x):
                    out.add(_norm(x))

        walk(doc)
    return out


def _norm(x: float) -> str:
    """Render a float the way the log does: integers bare, others to 3-4 s.f."""
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.4g}"


def _matches(cited: float, values: set[str]) -> bool:
    for raw in values:
        try:
            actual = float(raw)
        except ValueError:  # pragma: no cover - _norm only emits numbers
            continue
        if actual == 0:
            if cited == 0:
                return True
            continue
        if abs(actual - cited) <= _REL_TOL * max(abs(actual), abs(cited)):
            return True
    return False


def _cited(doc: str) -> dict[str, list[int]]:
    """Numbers cited per line, skipping headings and table scaffolding."""
    found: dict[str, list[int]] = {}
    for i, line in enumerate(doc.splitlines(), 1):
        if _STRUCTURAL.match(line):
            continue
        nums = [m.group(1) for m in _NUM.finditer(line)]
        # A dense line is a data table or a run-id list, not a claim in prose.
        if len(nums) > 25:
            continue
        if nums:
            found[f"line {i}"] = nums
    return found


@pytest.fixture(scope="module")
def values() -> set[str]:
    if not ARTIFACTS.exists():
        pytest.skip("no artifacts/ directory; run `make repro` first")
    v = _artifact_values()
    if not v:
        pytest.skip("artifacts/ holds no readable numbers")
    return v


def _exempt_lines(doc: str) -> dict[int, str]:
    """Line numbers waived by `paper/provenance_exemptions.json`, with their reason.

    Waivers are keyed on the *text* of the waived line, not its number. Line numbers are
    only valid until the next edit above them: when a section is inserted, every waiver
    below it silently moves onto whatever text took its place, so the ratchet stops
    applying where it was written and starts complaining about unrelated lines instead.
    That is a worse failure than a missing waiver, because it looks like the document grew
    an untraced number when it did not.

    An anchor that matches nothing is an error rather than a silent skip - a stale anchor
    means the claim it was covering has been reworded or deleted, and a reviewer needs to
    decide whether the waiver still applies.
    """
    path = PAPER / "provenance_exemptions.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    # Groups are written against one document by default; a group may widen itself with an
    # explicit "doc". Without this, an EXPERIMENT_LOG waiver is checked against
    # CLAIMS_LEDGER.md and every anchor there is reported as stale.
    scope = raw.get("_scope")
    lines = (PAPER / doc).read_text().splitlines()
    out: dict[int, str] = {}
    for group, entry in raw.items():
        if not isinstance(entry, dict) or "anchors" not in entry:
            continue
        if not (group.startswith("_") or scope is None or entry.get("doc", scope) == doc):
            continue
        if not entry.get("reason", "").strip():
            raise AssertionError(f"exemption group {group!r} has no stated reason")
        for anchor in entry["anchors"]:
            hits = [i for i, line in enumerate(lines, 1) if anchor in line]
            if not hits:
                raise AssertionError(
                    f"exemption group {group!r} has an anchor matching no line of {doc}:\n"
                    f"  {anchor!r}\n"
                    "The waived text has been reworded or removed. Either delete the waiver or "
                    "update it with a reason that still applies - do not re-point it at whatever "
                    "line happens to be nearby."
                )
            if len(hits) > 1:
                raise AssertionError(
                    f"exemption group {group!r} anchor is ambiguous in {doc} "
                    f"({len(hits)} lines match):\n  {anchor!r}\n"
                    "Anchors must be unique enough to identify one line."
                )
            out[hits[0]] = group
    return out


@pytest.mark.parametrize("doc", CLAIMED_DOCS)
def test_every_cited_number_traces_to_an_artifact(doc: str, values: set[str]) -> None:
    path = PAPER / doc
    assert path.exists(), f"{doc} is missing"
    text = path.read_text()
    exempt = _exempt_lines(doc)

    untraced: dict[str, list[str]] = {}
    for where, nums in _cited(text).items():
        line_no = int(where.split()[1])
        if line_no in exempt:
            continue
        missing = sorted({n for n in nums if n not in ALLOWED and not _matches(float(n), values)})
        if missing:
            untraced[where] = missing

    if not untraced:
        return

    report = "\n".join(f"  {w}: {', '.join(v)}" for w, v in sorted(untraced.items())[:25])
    pytest.fail(
        f"{len(untraced)} line(s) in {doc} cite numbers no artifact produced:\n"
        f"{report}\n\n"
        "Either the number is a real measurement and the artifact is missing or was "
        "deleted, or the number is structural and belongs in ALLOWED with a stated "
        "reason. Adding it to ALLOWED silences this test without making the claim "
        "true, so it needs a reason a reviewer can check."
    )


def test_allowed_numbers_are_still_needed() -> None:
    """An allowlist entry that is now traceable is dead weight; remove it.

    Without this the allowlist only ever grows, and a test that can be satisfied by
    widening an exemption stops being evidence of anything.
    """
    values = _artifact_values()
    if not values:
        pytest.skip("no artifacts to check against")
    text = "\n".join((PAPER / d).read_text() for d in CLAIMED_DOCS if (PAPER / d).exists())
    dead = sorted(
        k for k in ALLOWED if re.search(rf"(?<![\w.]){re.escape(k)}(?![\w])", text) is None
    )
    assert not dead, (
        f"ALLOWED entries no longer appear in the claim documents: {dead}. "
        "Remove them so the exemption list stays honest."
    )


def test_report_md_quotes_the_current_measurements() -> None:
    """Every project measurement in report.md must match the artifact, exactly.

    This is the one document a second person reads while deciding whether to trust
    someone else's work, so a stale number here is worse than a stale number in the
    paper: it propagates into a verification decision rather than a citation. The
    values are re-derived from the artifacts rather than hard-coded on both sides.
    """
    import json
    import statistics as st

    path = PAPER.parent / "report.md"
    assert path.exists(), "report.md is missing"
    text = path.read_text()

    repo = PAPER.parent
    m4_path = repo / "artifacts" / "m4" / "factorizer_multilabel.json"
    m5_path = repo / "artifacts" / "m5a" / "recogniser.json"
    if not (m4_path.exists() and m5_path.exists()):
        pytest.skip("M4/M5 artifacts not present")

    m4 = json.loads(m4_path.read_text())["runs"]["full"]
    m5 = json.loads(m5_path.read_text())
    live = {
        "worst_cross_auc": round(m4["gate"]["worst_cross_auc"], 4),
        "signer_control_max": round(m4["gate"]["signer_control_max"], 4),
        "cross_a_to_l": round(m4["n_test_weighted"]["cross_a_to_l"], 4),
        "n_tokens": m5["design"]["n_tokens"],
        "n_glosses": m5["design"]["n_glosses"],
        "hapax_types": m5["design"]["hapax_types"],
        "wer_mean": round(st.mean(f["wer"] for f in m5["real"]["folds"]), 3),
        "wer_baseline": round(
            st.mean(f["wer_most_frequent_baseline"] for f in m5["real"]["folds"]), 3
        ),
        "wer_shuffled": round(st.mean(f["wer"] for f in m5["shuffled_label_control"]["folds"]), 3),
    }
    # 1. What the form asserts must equal what the artifacts say.
    stale = {k: (v, live[k]) for k, v in REPORT_QUOTED.items() if k in live and v != live[k]}
    assert not stale, (
        f"report.md quotes measurements that no longer match the artifacts: {stale}. "
        "A verifier reading a stale number is worse off than one reading nothing."
    )

    # 2. And the values it actually printed must be the ones asserted here, so editing
    #    the form to a different number is caught rather than merely discouraged.
    # Thousands separators are stripped first: a report is read by people, so "1,563" is
    # the right way to write the token count and should not be reported as a mismatch.
    flat = text.replace(",", "")
    absent = [
        k
        for k, v in live.items()
        if re.search(rf"(?<![\w.]){re.escape(str(v))}(?![\w])", flat) is None
    ]
    assert not absent, (
        f"report.md no longer contains these measured values: {absent}. If a number was "
        "changed, change REPORT_QUOTED in this test too, and say why in the commit."
    )
