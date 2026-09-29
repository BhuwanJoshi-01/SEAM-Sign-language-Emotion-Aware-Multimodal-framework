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
    """Line numbers waived by `paper/provenance_exemptions.json`, with their reason."""
    path = PAPER / "provenance_exemptions.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out: dict[int, str] = {}
    for group, entry in raw.items():
        if not isinstance(entry, dict) or "lines" not in entry:
            continue
        if not entry.get("reason", "").strip():
            raise AssertionError(f"exemption group {group!r} has no stated reason")
        for ln in entry["lines"]:
            out[int(ln)] = group
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
