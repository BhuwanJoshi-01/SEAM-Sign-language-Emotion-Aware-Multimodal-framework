"""Perception contracts.

The tests that matter here are the ones guarding against *plausible but wrong*
output. A permuted blendshape basis, a shifted landmark ordering, or a
train/serve normalisation mismatch all produce numbers that look fine and mean
nothing, so each is asserted explicitly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seam.perception import extract as extract_mod
from seam.perception import face_gate as gate_mod
from seam.perception import tasks_api as T

# -- the ordering contract ----------------------------------------------


def test_part_slices_are_contiguous_and_cover_every_point() -> None:
    """A gap or overlap in the landmark stream shifts every downstream feature."""
    expected_start = 0
    for name in T.PART_ORDER:
        start, end = T.PART_SLICES[name]
        assert start == expected_start, f"{name} does not start where the previous part ended"
        assert end > start
        expected_start = end
    assert expected_start == T.TOTAL_POINTS


def test_part_slice_counts_match_their_names() -> None:
    assert T.PART_SLICES["pose"][1] - T.PART_SLICES["pose"][0] == T.POSE_POINTS
    for hand in ("left_hand", "right_hand"):
        start, end = T.PART_SLICES[hand]
        assert end - start == T.HAND_POINTS
    start, end = T.PART_SLICES["face"]
    assert end - start == T.FACE_POINTS


def test_blendshape_basis_is_fifty_two_named_and_indexed() -> None:
    assert len(T.BLENDSHAPE_CATEGORIES) == T.FACE_BLENDSHAPES == 52
    assert len(set(T.BLENDSHAPE_CATEGORIES)) == 52
    for name in ("browInnerUp", "jawOpen", "mouthSmileLeft", "_neutral"):
        assert name in T.BLENDSHAPE_INDEX
    # The measured MediaPipe basis includes eyeLookUp and excludes tongueOut.
    # Getting this wrong is what the first draft of this module did.
    assert "eyeLookUpLeft" in T.BLENDSHAPE_INDEX
    assert "tongueOut" not in T.BLENDSHAPE_INDEX


def test_model_bundles_resolve_to_local_files() -> None:
    """No download should be needed: the bundles are cached on this machine."""
    task = T.landmarker()
    for key in ("face", "hand", "pose"):
        resolved = task._resolve(key)
        assert resolved.endswith(".task")
        if resolved.startswith("/"):
            assert Path(resolved).is_file(), f"{key} bundle missing at {resolved}"


# -- timestamps ---------------------------------------------------------


def test_clock_is_strictly_monotonic() -> None:
    clock = T._MonotonicClock()
    assert clock.next(0) == 0
    assert clock.next(33) == 33
    # A reset counter is the realistic failure: decoding two videos back to back.
    assert clock.next(0) == 34
    assert clock.next(33) == 35
    assert clock.next(100) == 100  # a forward jump is honoured
    assert clock.next(50) == 101  # a backward step is corrected


def test_clock_reset_allows_restart() -> None:
    clock = T._MonotonicClock()
    clock.next(1000)
    clock.reset()
    assert clock.next(0) == 0


def _sidecar(utterance_id: str, frame_count: int) -> dict:
    """A complete, valid sidecar. Tests corrupt one field at a time."""
    from seam.perception.extract import ClipMeta, default_order

    order, shapes = default_order()
    return {
        "source_path": f"{utterance_id}.mp4",
        "utterance_id": utterance_id,
        "native_fps": 24.0,
        "source_width": 256,
        "source_height": 256,
        "frame_count": frame_count,
        "detection_rates": {"pose": 1.0, "left_hand": 0.9, "right_hand": 0.9, "face": 1.0},
        "landmark_order": order,
        "blendshape_order": shapes,
        "extractor": "mediapipe-tasks",
        "schema_version": 1,
    }
    del ClipMeta


# -- shard round trip ---------------------------------------------------


def test_save_and_load_shard_round_trips(tmp_path: Path, shard_arrays: dict) -> None:
    meta = extract_mod.ClipMeta(
        source_path="x.mp4",
        utterance_id="123",
        native_fps=24.0,
        source_width=256,
        source_height=256,
        frame_count=shard_arrays["blendshapes"].shape[0],
        detection_rates={"pose": 1.0, "left_hand": 0.9, "right_hand": 0.9, "face": 1.0},
        landmark_order=extract_mod.default_order()[0],
        blendshape_order=extract_mod.default_order()[1],
    )
    path = extract_mod.save_shard(tmp_path, shard_arrays, meta)
    assert path.is_file()

    arrays, loaded_meta = extract_mod.load_shard(path)
    assert arrays["landmarks"].shape == shard_arrays["landmarks"].shape
    assert arrays["blendshapes"].shape == shard_arrays["blendshapes"].shape
    assert loaded_meta.utterance_id == "123"
    assert np.array_equal(arrays["presence"], shard_arrays["presence"])


def test_is_extracted_rejects_a_frame_count_mismatch(tmp_path: Path, shard_arrays: dict) -> None:
    """A shard truncated by a full disk is present and passes a naive existence check."""
    from tests.conftest import write_shard

    meta = _sidecar("7", shard_arrays["blendshapes"].shape[0])
    meta["frame_count"] = 999
    write_shard(tmp_path, "7", shard_arrays, meta)
    assert not extract_mod.is_extracted(tmp_path, "7")


def test_is_extracted_rejects_a_shard_with_no_sidecar(tmp_path: Path, shard_arrays: dict) -> None:
    from tests.conftest import write_shard

    write_shard(tmp_path, "8", shard_arrays, _sidecar("8", shard_arrays["blendshapes"].shape[0]))
    (tmp_path / "8.json").unlink()
    assert not extract_mod.is_extracted(tmp_path, "8")


def test_load_shard_rejects_inconsistent_landmark_width(tmp_path: Path, shard_arrays: dict) -> None:
    from tests.conftest import write_shard

    bad = dict(shard_arrays)
    bad["landmarks"] = bad["landmarks"][:, :-1, :]
    meta = _sidecar("9", bad["blendshapes"].shape[0])
    write_shard(tmp_path, "9", bad, meta)
    with pytest.raises(extract_mod.ExtractionError, match="inconsistent"):
        extract_mod.load_shard(tmp_path / "9.npz")


def test_load_shard_rejects_wrong_blendshape_width(tmp_path: Path, shard_arrays: dict) -> None:
    from tests.conftest import write_shard

    bad = dict(shard_arrays)
    bad["blendshapes"] = bad["blendshapes"][:, :10]
    meta = _sidecar("10", bad["blendshapes"].shape[0])
    write_shard(tmp_path, "10", bad, meta)
    with pytest.raises(extract_mod.ExtractionError, match="blendshapes width"):
        extract_mod.load_shard(tmp_path / "10.npz")


def test_missing_video_raises_extraction_error(tmp_path: Path) -> None:
    with pytest.raises(extract_mod.ExtractionError, match="no such video"):
        extract_mod.extract_clip(tmp_path / "nope.mp4")


def test_corrupt_video_raises_rather_than_returning_empty(tmp_path: Path) -> None:
    p = tmp_path / "broken.mp4"
    p.write_bytes(b"not a video" * 100)
    with pytest.raises(extract_mod.ExtractionError):
        extract_mod.extract_clip(p)


# -- the face gate ------------------------------------------------------


def test_gate_accepts_a_varying_face() -> None:
    # Variation must be across *frames*. Tiling one row over 40 frames is a
    # static face, which is the case the gate exists to catch.
    rng = np.random.default_rng(0)
    arrays = {
        "blendshapes": (rng.random((40, T.FACE_BLENDSHAPES)) * 0.5).astype(np.float32),
        "presence": np.ones((40, 4), dtype=bool),
    }
    v = gate_mod.judge_clip("1", "Cory", arrays)
    assert v.passed, v.detail
    assert v.face_rate == 1.0


def test_gate_rejects_a_static_face() -> None:
    """Detected in every frame but never moving: present, and useless.

    A detection-rate-only gate would pass this, and the affect milestones would
    then be built on a face channel with no signal in it.
    """
    arrays = {
        "blendshapes": np.zeros((40, T.FACE_BLENDSHAPES), dtype=np.float32),
        "presence": np.ones((40, 4), dtype=bool),
    }
    v = gate_mod.judge_clip("2", "Cory", arrays)
    assert not v.passed
    assert "varying coefficients" in v.detail


def test_gate_rejects_a_missing_face() -> None:
    rng = np.random.default_rng(0)
    arrays = {
        "blendshapes": rng.random((40, T.FACE_BLENDSHAPES)).astype(np.float32),
        "presence": np.zeros((40, 4), dtype=bool),
    }
    arrays["presence"][:, 0] = True
    v = gate_mod.judge_clip("3", "Cory", arrays)
    assert not v.passed
    assert "face rate" in v.detail


def test_gate_report_opens_on_a_clear_majority() -> None:
    report = gate_mod.GateReport()
    for i in range(10):
        report.verdicts.append(
            gate_mod.ClipVerdict(str(i), "Cory", 100, 1.0, 1.0, 0.9, 30, 0.3, True)
        )
    assert report.passed
    report.verdicts[0].passed = False
    assert report.passed
    report.verdicts[1].passed = False
    report.verdicts[2].passed = False
    assert not report.passed


def test_gate_report_is_closed_when_empty() -> None:
    assert not gate_mod.GateReport().passed


def test_gate_render_includes_the_verdict() -> None:
    report = gate_mod.GateReport(sample=1)
    report.verdicts.append(gate_mod.ClipVerdict("1", "Cory", 100, 1.0, 1.0, 0.9, 30, 0.3, True))
    text = report.render()
    assert "GATE: OPEN" in text
    assert "utterance" in text


# -- gaze proxy ---------------------------------------------------------


def test_gaze_proxy_is_none_without_a_face() -> None:
    assert T._gaze_proxy(np.zeros((T.TOTAL_POINTS, 3), dtype=np.float32)) is None


def test_gaze_proxy_is_scale_invariant() -> None:
    """Gaze is an offset normalised by eye span, so scaling the head must not move it."""
    start, end = T.PART_SLICES["face"]
    base = np.zeros((T.TOTAL_POINTS, 3), dtype=np.float32)
    base[start + 33, :2] = (0.3, 0.4)
    base[start + 263, :2] = (0.7, 0.4)
    base[start + 468 : start + 473, :2] = (0.45, 0.45)
    base[start + 473 : start + 478, :2] = (0.55, 0.45)

    g1 = T._gaze_proxy(base)
    assert g1 is not None

    scaled = base.copy()
    scaled[start:end, :2] *= 2.5
    g2 = T._gaze_proxy(scaled)
    assert g2 is not None
    np.testing.assert_allclose(g1, g2, atol=1e-5)
