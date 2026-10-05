"""The continuous-signing confound audit: its controls and its decision rule.

The window builder, matching and bootstrap are tested in `test_audit.py`, including a
planted effect recovered through human-marked frames. What is tested here is the part
that turns statistics into a verdict, because that is where a result can be read more
generously than the design allows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_confound_audit_continuous as C

from seam.data.signstream import NonManual, Utterance


def _stat(shift: float, p_bonf: float, *, pairs: int = 80, clips: int = 40) -> dict:
    return {
        "mean_shift": shift,
        "p_value": p_bonf / 10,
        "p_bonferroni": p_bonf,
        "n_pairs": pairs,
        "n_clips": clips,
        "mde": 0.01,
    }


def _no_interval() -> dict:
    nan = float("nan")
    return {
        "mean_shift": 0.02,
        "p_value": nan,
        "p_bonferroni": nan,
        "n_pairs": 3,
        "n_clips": 3,
        "mde": nan,
    }


def _models(real: list[dict], placebo: list[dict] | None = None) -> dict[str, dict[str, dict]]:
    placebo = placebo or [_stat(0.0, 1.0)] * len(real)
    out: dict[str, dict[str, dict]] = {}
    for i, (r, pl) in enumerate(zip(real, placebo, strict=True)):
        stats = {m: _stat(0.0, 1.0) for m in C.MARKERS}
        stats.update({C.PLACEBO + m: _stat(0.0, 1.0) for m in C.MARKERS})
        stats["brow_furrow"] = r
        stats[C.PLACEBO + "brow_furrow"] = pl
        out[f"model_{i}"] = stats
    return out


def test_support_needs_a_positive_shift_in_two_models() -> None:
    v = C.verdicts(_models([_stat(0.03, 0.01), _stat(0.02, 0.02), _stat(0.0, 1.0)]))
    assert v["brow_furrow"]["verdict"].startswith("SUPPORTS C1")
    assert v["brow_raise"]["verdict"] == "null at the stated MDE"


def test_one_model_is_not_replication() -> None:
    v = C.verdicts(_models([_stat(0.03, 0.001), _stat(0.01, 0.4), _stat(0.0, 1.0)]))
    assert v["brow_furrow"]["verdict"].startswith("not support")
    assert "1 model(s) positive" in v["brow_furrow"]["verdict"]


def test_a_significant_shift_the_wrong_way_is_not_support() -> None:
    """M1's two significant effects ran the wrong way; the rule must not count those."""
    v = C.verdicts(_models([_stat(-0.03, 0.001), _stat(-0.02, 0.01), _stat(-0.02, 0.01)]))
    assert not v["brow_furrow"]["verdict"].startswith("SUPPORTS")
    assert v["brow_furrow"]["models_negative_significant"] == ["model_0", "model_1", "model_2"]


def test_a_significant_placebo_voids_the_marker() -> None:
    """If the rolled track 'works' too, the effect is not about the marker's frames."""
    real = [_stat(0.03, 0.01), _stat(0.02, 0.02), _stat(0.0, 1.0)]
    placebo = [_stat(0.03, 0.01), _stat(0.0, 1.0), _stat(0.0, 1.0)]
    v = C.verdicts(_models(real, placebo))
    assert v["brow_furrow"]["verdict"].startswith("not interpretable")


def test_too_few_clips_is_no_estimate_not_a_null() -> None:
    v = C.verdicts(_models([_no_interval(), _no_interval(), _no_interval()]))
    assert v["brow_furrow"]["verdict"].startswith("no estimate")


def test_the_placebo_keeps_the_duty_cycle_and_moves_the_track() -> None:
    track = np.zeros(100, dtype=bool)
    track[10:25] = True
    rolled = C.placebo_track(track, np.random.default_rng(0))
    assert rolled.sum() == track.sum()
    shifts = [s for s in range(100) if np.array_equal(np.roll(track, s), rolled)]
    assert len(shifts) == 1 and 25 <= shifts[0] <= 75


def test_the_placebo_is_reproducible_per_clip() -> None:
    u = Utterance(
        utterance_id="1",
        participant="Cory",
        collection="c",
        collection_id="0",
        start_frame=1000,
        end_frame=1100,
        glosses=[("A", 1010, 1020), ("B", 1070, 1090)],
        non_manuals=[
            NonManual(
                label="eye brows",
                value="raised",
                start_frame=1030,
                end_frame=1050,
                markers=["brow_raise"],
            )
        ],
    )
    a = C.tracks_for(u, 81, 24.0, seed=7)
    b = C.tracks_for(u, 81, 24.0, seed=7)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert set(a) == set(C.MARKERS) | {C.PLACEBO + m for m in C.MARKERS}
    # Session frames 1030-1050 are clip frames 24-40 at 24 fps.
    assert a["brow_raise"][24] and a["brow_raise"][40] and not a["brow_raise"][41]
    assert not a["negation"].any()


def test_the_signing_span_runs_from_the_first_gloss_to_the_last() -> None:
    u = Utterance(
        utterance_id="1",
        participant="Cory",
        collection="c",
        collection_id="0",
        start_frame=1000,
        end_frame=1100,
        glosses=[("A", 1010, 1020), ("B", 1070, 1090)],
    )
    at30 = C.signing_span(u, 101, 30.0)
    assert not at30[9] and at30[10] and at30[90] and not at30[91]
    # The same utterance as a 24 fps clip: 8.0 to 72.0.
    at24 = C.signing_span(u, 81, 24.0)
    assert not at24[7] and at24[8] and at24[72] and not at24[73]


def test_an_utterance_with_no_glosses_has_no_signing_span() -> None:
    u = Utterance(
        utterance_id="1",
        participant="Cory",
        collection="c",
        collection_id="0",
        start_frame=1000,
        end_frame=1100,
    )
    assert not C.signing_span(u, 81, 24.0).any()


def test_the_design_constants_are_the_registered_ones() -> None:
    """Changing any of these after a run is a different experiment, not a tweak."""
    assert (C.WINDOW, C.STRIDE, C.BEARING_MIN) == (6, 3, 0.5)
    assert (C.ALPHA, C.MODELS_REQUIRED) == (0.05, 2)
    assert C.MARKERS == (
        "brow_raise",
        "brow_furrow",
        "head_shake",
        "head_nod",
        "negation",
        "topic",
        "conditional",
        "question_yn",
        "question_wh",
        "question_rhetorical",
    )


def test_the_artifact_reports_what_the_rule_was() -> None:
    import json

    path = Path(__file__).resolve().parents[1] / "artifacts/audit/confound_audit_continuous.json"
    if not path.is_file():
        pytest.skip("continuous audit not run")
    d = json.loads(path.read_text())
    assert d["design"]["fixed_before_first_run"] is True
    assert set(d["verdicts"]) == set(C.MARKERS)
    assert d["c1_supported_on_continuous_signing"] == any(
        v["verdict"].startswith("SUPPORTS") for v in d["verdicts"].values()
    )
    # A null is only reported with its minimum detectable effect beside it.
    for marker, v in d["verdicts"].items():
        if v["verdict"] == "null at the stated MDE":
            assert all(x is not None for x in v["mde_by_model"].values()), marker
