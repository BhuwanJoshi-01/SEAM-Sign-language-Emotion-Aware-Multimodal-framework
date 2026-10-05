"""The scripts a teammate runs, and the two checks added on 2026-10-05.

None of these needs a dataset: the hand-slot check is run on shards made here, the bundle
is inspected as a file list, and the video tool's event logic is plain arithmetic.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyse_video as AV
import check_hand_slots as HS
import check_negation_head_shake as NH
import make_share_bundle as B

from seam.perception import extract as X

REPO = Path(__file__).resolve().parents[1]


def _shard(root: Path, name: str, *, same_hand: bool, frames: int = 6) -> None:
    rng = np.random.default_rng(len(name))
    lm = rng.uniform(0.1, 0.9, size=(frames, 553, 3)).astype(np.float16)
    if same_hand:
        lm[:, 54:75] = lm[:, 33:54]
    presence = np.ones((frames, 4), dtype=bool)
    presence[0, 2] = False  # one frame with only a left hand
    arrays = {
        "landmarks": lm,
        "blendshapes": rng.uniform(0, 1, size=(frames, 52)).astype(np.float32),
        "presence": presence,
    }
    order, shapes = X.default_order()
    meta = X.ClipMeta(
        source_path=f"{name}.mp4",
        utterance_id=name,
        native_fps=24.0,
        source_width=320,
        source_height=240,
        frame_count=frames,
        detection_rates={p: float(presence[:, i].mean()) for i, p in enumerate(X.PART_NAMES)},
        landmark_order=order,
        blendshape_order=shapes,
    )
    X.save_shard(root, arrays, meta)


def test_one_hand_filed_twice_is_counted_and_two_hands_are_not(tmp_path: Path) -> None:
    """The defect this check exists for, planted and found."""
    bad, good = tmp_path / "bad", tmp_path / "good"
    bad.mkdir()
    good.mkdir()
    _shard(bad, "a", same_hand=True)
    _shard(good, "a", same_hand=False)
    was = HS.slot_report(bad)
    now = HS.slot_report(good)
    assert was["frames_with_both_slots_filled"] == 5, "the frame with one hand is not counted"
    assert was["of_those_the_two_wrists_are_one_point"] == 5
    assert now["frames_with_both_slots_filled"] == 5
    assert now["of_those_the_two_wrists_are_one_point"] == 0


def test_repeatability_compares_like_with_like(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _shard(a, "x", same_hand=False)
    _shard(b, "x", same_hand=False)
    same = HS.face_repeatability(a, b)
    assert same["fraction_of_frames_identical"] == 1.0
    assert same["clips_with_any_such_frame"] == 0
    _shard(b, "x", same_hand=False, frames=7)  # a different length is not compared
    assert HS.face_repeatability(a, b)["clips_compared"] == 0


def test_the_current_landmarks_hold_two_different_hands() -> None:
    path = REPO / "artifacts" / "m3" / "hand_slots.json"
    if not path.is_file():
        pytest.skip("hand_slots.json not built; run scripts/check_hand_slots.py")
    report = json.loads(path.read_text())
    now = report["current"]
    assert now["frames_with_both_slots_filled"] > 1000
    assert now["of_those_the_two_wrists_are_one_point"] == 0
    before = report.get("before_the_fix")
    if before:
        assert (
            before["of_those_the_two_wrists_are_one_point"]
            == before["frames_with_both_slots_filled"]
        ), "the archived extraction is the one-hand one, and should say so"


def test_rank_correlation_handles_ties_and_constants() -> None:
    x = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    assert NH.rank_correlation(x, x * 10 + 1) == pytest.approx(1.0)
    assert NH.rank_correlation(x, -x) == pytest.approx(-1.0)
    assert np.isnan(NH.rank_correlation(x, np.ones(6)))
    assert NH.split(np.array([True, True, False]), np.array([True, False, True])) == [1, 2]


def test_the_negation_result_is_reported_with_its_controls() -> None:
    """The claim and the three looks that have to stay beside it."""
    path = REPO / "artifacts" / "audit" / "negation_head_shake_check.json"
    if not path.is_file():
        pytest.skip("negation_head_shake_check.json not built")
    d = json.loads(path.read_text())
    neg, other = d["detected_shake"]["negated_clips"], d["detected_shake"]["other_clips"]
    assert neg[0] / neg[1] > other[0] / other[1] + 0.2
    checks = d["robustness_checks_decided_after_the_result"]
    human = checks["human_annotation"]
    a, b = (
        human["negated_clips_with_a_human_marked_shake"],
        human["other_clips_with_a_human_marked_shake"],
    )
    assert a[0] / a[1] > b[0] / b[1], "the annotators' marks must show the same direction"
    placebo = checks["placebo_axis"]
    assert placebo["head_shake_duration_controlled"] > 0.5
    assert abs(placebo["head_nod_duration_controlled"]) < 0.25, "the nod axis is the placebo"
    assert d["registered_test"]["result"]["significant_bonferroni"] is True
    assert len(d["caveats"]) >= 3


def test_events_need_a_fifth_of_a_second() -> None:
    fps = 20.0
    mask = np.zeros(60, dtype=bool)
    mask[5:7] = True  # two frames: 0.1 s, a blip
    mask[20:30] = True  # half a second
    mask[55:60] = True  # runs to the end of the clip
    assert AV.events(mask, fps) == [(1.0, 1.5), (2.75, 3.0)]
    assert AV.events(np.zeros(10, dtype=bool), fps) == []


def test_lite_requirements_are_the_pins_in_pyproject() -> None:
    text = B.lite_requirements()
    pyproject = (REPO / "pyproject.toml").read_text()
    for line in text.splitlines():
        if line and not line.startswith("#"):
            assert f'"{line}"' in pyproject, f"{line} is not a pin from pyproject.toml"
    assert "torch" not in text and "mediapipe" not in text, "the website needs neither"


def test_the_bundle_never_carries_datasets_or_the_licensed_model() -> None:
    """Whatever is on this machine, these must not be in a zip built without the flag."""
    path = REPO / "dist" / "SEAM_share.zip"
    if not path.is_file():
        pytest.skip("no bundle built; run scripts/make_share_bundle.py")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        manifest = json.loads(z.read("SEAM/BUNDLE_MANIFEST.json"))
    lowered = [n.lower() for n in names]
    assert not [n for n in lowered if n.endswith((".mp4", ".mov", ".avi", ".npz", ".xml"))]
    assert not [n for n in lowered if "smplx_" in n or "smpler_x" in n or "/frames/" in n]
    if not manifest["avatar_clips_included"]:
        assert not [n for n in lowered if n.endswith(".glb")]
        assert any("glb" in k for k in manifest["left_out"])
    for needed in (
        "SEAM/START_HERE.md",
        "SEAM/setup_linux.sh",
        "SEAM/setup_windows.bat",
        "SEAM/run_demo.sh",
        "SEAM/run_demo.bat",
        "SEAM/requirements-lite.txt",
        "SEAM/docs/index.html",
        "SEAM/scripts/analyse_video.py",
        "SEAM/models/mediapipe/hand_landmarker.task",
        "SEAM/artifacts/fer/fer_cnn_a.pt",
    ):
        assert needed in names, f"{needed} is missing from the bundle"


def test_windows_scripts_keep_crlf_and_shell_scripts_do_not() -> None:
    """cmd.exe misreads a batch file with Unix line endings; bash chokes on the reverse."""
    for name in ("setup_windows.bat", "run_demo.bat"):
        raw = (REPO / name).read_bytes()
        assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b""), name
        assert raw.isascii(), f"{name}: cmd.exe on a default code page needs plain ASCII"
    for name in ("setup_linux.sh", "run_demo.sh"):
        raw = (REPO / name).read_bytes()
        assert b"\r" not in raw and raw.startswith(b"#!/usr/bin/env bash"), name
