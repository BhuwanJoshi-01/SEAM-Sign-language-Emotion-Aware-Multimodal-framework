"""Tests for the M7 rater-kit builder.

These assert the properties that make a *blinded* study stay blinded: that a trial
always has exactly two arms, that the side assignment is reproducible, and that a
condition word in a rater-visible file stops the build rather than reaching a rater.

The last of those is the one worth having. Every other mistake in a small study shows
up as a strange number; an unblinded study shows up as a *clean* number, which is
exactly the failure that looks like success.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.make_rater_kit import (
    FORBIDDEN_IN_PUBLIC,
    load_trials,
    main,
    side_for,
)

TEMPLATE = """# Rating Sheet

| Trial | Clip A (left) | Clip B (right) | Which? | Why? |
|---|---|---|---|---|
| 1 | `__` | `__` | \u2610 Left \u2610 Right | |

## The most useful thing

| Trial | What looked wrong |
|---|---|
| | |
"""


def _write(path: Path, entries: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"stimuli": entries}), encoding="utf-8")
    return path


def _pair(tmp_path: Path, name: str = "manifest.json", tokens=("aaaa1111", "bbbb2222")):
    """A one-trial manifest: both tokens belong to the same trial, one per condition."""
    return _write(
        tmp_path / name,
        [{"trial_id": "clip_00", "clip": f"{t}.mp4"} for t in tokens],
    )


def test_a_trial_must_have_exactly_two_arms(tmp_path: Path) -> None:
    bad = _write(tmp_path / "m.json", [{"trial_id": "clip_01", "clip": "only.mp4"}])
    with pytest.raises(ValueError, match=r"has 1 arm"):
        load_trials(bad)


def test_three_arms_is_also_refused(tmp_path: Path) -> None:
    # A forced choice has no defined answer for three alternatives.
    bad = _write(
        tmp_path / "m.json",
        [{"trial_id": "clip_01", "clip": f"{t}.mp4"} for t in ("a", "b", "c")],
    )
    with pytest.raises(ValueError, match=r"has 3 arm"):
        load_trials(bad)


def test_sides_are_stable_for_a_seed(tmp_path: Path) -> None:
    # Two printings of the instrument must agree, or the study is not the same study.
    a = side_for(20261001, "clip_01", "aaaa1111")
    b = side_for(20261001, "clip_01", "aaaa1111")
    assert a == b
    assert a in (0, 1)


def test_a_different_seed_can_change_the_side(tmp_path: Path) -> None:
    sides = {side_for(seed, "clip_01", "aaaa1111") for seed in range(40)}
    assert sides == {0, 1}, "side assignment never varies with the seed"


def test_the_build_is_deterministic(tmp_path: Path) -> None:
    manifest = _pair(tmp_path)
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    (out1 / "private").mkdir(parents=True)
    (out2 / "private").mkdir(parents=True)
    (out1 / "RATING_SHEET.md").write_text(TEMPLATE, encoding="utf-8")
    (out2 / "RATING_SHEET.md").write_text(TEMPLATE, encoding="utf-8")

    assert main(["--manifest", str(manifest), "--out", str(out1)]) == 0
    assert main(["--manifest", str(manifest), "--out", str(out2)]) == 0
    assert (out1 / "trials.csv").read_text() == (out2 / "trials.csv").read_text()


def test_a_leaky_manifest_is_refused(tmp_path: Path) -> None:
    # The generator has its own guard. This one must not depend on it having run:
    # it writes the *other* files a rater sees.
    leaky = _write(
        tmp_path / "m.json",
        [
            {"trial_id": "c1", "clip": "system_baseline_a.mp4"},
            {"trial_id": "c1", "clip": "zz11.mp4"},
        ],
    )
    out = tmp_path / "out"
    (out / "private").mkdir(parents=True)
    assert main(["--manifest", str(leaky), "--out", str(out)]) == 3
    assert "system" in FORBIDDEN_IN_PUBLIC


def test_public_files_carry_no_condition_word(tmp_path: Path) -> None:
    manifest = _pair(tmp_path)
    out = tmp_path / "out"
    (out / "private").mkdir(parents=True)
    (out / "RATING_SHEET.md").write_text(TEMPLATE, encoding="utf-8")
    assert main(["--manifest", str(manifest), "--out", str(out)]) == 0

    blob = (out / "trials.csv").read_text(encoding="utf-8").lower()
    for word in FORBIDDEN_IN_PUBLIC:
        assert word not in blob


def test_the_side_assignment_is_not_public(tmp_path: Path) -> None:
    # A published side key lets an answer be read off the key instead of the video.
    manifest = _pair(tmp_path)
    out = tmp_path / "out"
    (out / "private").mkdir(parents=True)
    (out / "RATING_SHEET.md").write_text(TEMPLATE, encoding="utf-8")
    assert main(["--manifest", str(manifest), "--out", str(out)]) == 0
    assert (out / "private" / "sides.json").is_file()
    assert not (out / "sides.json").exists()


def test_the_sheet_is_filled_with_real_tokens(tmp_path: Path) -> None:
    manifest = _pair(tmp_path, tokens=("aaaa1111", "bbbb2222"))
    out = tmp_path / "out"
    (out / "private").mkdir(parents=True)
    (out / "RATING_SHEET.md").write_text(TEMPLATE, encoding="utf-8")
    assert main(["--manifest", str(manifest), "--out", str(out)]) == 0

    filled = (out / "RATING_SHEET_filled.md").read_text(encoding="utf-8")
    assert "aaaa1111" in filled and "bbbb2222" in filled
    # The rest of the form must survive the substitution, or raters lose the
    # question they are supposed to answer.
    assert "What looked wrong" in filled
