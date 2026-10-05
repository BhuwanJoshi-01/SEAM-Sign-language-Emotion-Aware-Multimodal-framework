"""The audit's own machinery, tested on data where the answer is known.

The point of these tests is not that the numbers look right on WLASL - nobody
knows the right answer there yet. It is that a *planted* bias is recovered, a
*planted* null is reported as null, and a planted artefact is not mistaken for a
finding. An audit that cannot detect a bias you put there on purpose is not
evidence of anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from seam.eval import fer
from seam.eval import fer_audit as A
from seam.features import markers as M


def make_window(
    clip: str,
    index: int,
    *,
    brow_raise: bool = False,
    amplitude: float = 0.3,
    speed: float = 0.4,
    neg: float = 0.1,
    probs: np.ndarray | None = None,
) -> A.Window:
    """A window with a controllable FER distribution.

    ``neg`` is the total probability to place on the four negative classes, split
    evenly, with the remainder on neutral. That makes the planted effect exact
    rather than approximate.
    """
    amp_i = A.P.PROSODY_FIELDS.index("amplitude")
    spd_i = A.P.PROSODY_FIELDS.index("speed")
    prosody = [0.0] * len(A.P.PROSODY_FIELDS)
    prosody[amp_i] = amplitude
    prosody[spd_i] = speed

    if probs is None:
        p = np.zeros(len(fer.EMOSIGN_LABELS), dtype=np.float64)
        per = neg / len(fer.NEG_IDX)
        p[fer.NEG_IDX] = per
        p[fer.NEUTRAL_IDX] = 1.0 - neg
        probs = p

    markers = [False] * len(M.MARKERS)
    markers[M.MARKERS.index("brow_raise")] = brow_raise
    return A.Window(
        clip=clip,
        index=index,
        start=index * A.STRIDE,
        n_frames=A.WINDOW,
        markers=markers,
        prosody=prosody,
        probs=[float(v) for v in probs],
        scored=True,
    )


# ---------------------------------------------------------------------------
# read-out
# ---------------------------------------------------------------------------


def test_negative_mass_is_the_sum_of_the_four_negative_classes() -> None:
    p = np.zeros(len(fer.EMOSIGN_LABELS))
    p[fer.EMOSIGN_LABELS.index("anger")] = 0.4
    p[fer.EMOSIGN_LABELS.index("sadness")] = 0.3
    assert fer.negative_mass(p) == pytest.approx(0.7)


def test_neutral_is_excluded_from_both_sides() -> None:
    """Neutral must not count as positive or negative, or the read-out answers a
    different question than the one the claim asks."""
    p = np.zeros(len(fer.EMOSIGN_LABELS))
    p[fer.NEUTRAL_IDX] = 1.0
    assert fer.negative_mass(p) == pytest.approx(0.0)
    assert fer.positive_mass(p) == pytest.approx(0.0)
    assert fer.neutral_mass(p) == pytest.approx(1.0)


def test_read_outs_are_consistent() -> None:
    p = np.array([0.4, 0.3, 0.1, 0.1, 0.1, 0.0, 0.0])
    r = fer.read_outs(p)
    assert r["negative"] + r["positive"] + r["neutral"] == pytest.approx(1.0)
    assert r["argmax"] == fer.EMOSIGN_LABELS.index("anger")


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------


def test_matching_pairs_within_the_same_clip_only() -> None:
    windows = [
        make_window("a", 0, brow_raise=True),
        make_window("a", 1, brow_raise=False),
        make_window("b", 0, brow_raise=False),
    ]
    pairs = A.match_windows(windows, "brow_raise")
    assert len(pairs) == 1
    marked, clean = pairs[0]
    assert marked.clip == "a"
    assert clean.clip == "a", "a marker in clip a must not be matched against clip b"


def test_matching_drops_windows_with_no_close_partner() -> None:
    """An unpaired window is missing data; forcing a pair biases the shift to zero."""
    windows = [
        make_window("a", 0, brow_raise=True, amplitude=5.0),
        make_window("a", 1, brow_raise=False, amplitude=0.1),
    ]
    assert A.match_windows(windows, "brow_raise") == []


def test_matching_prefers_the_closest_on_both_covariates() -> None:
    windows = [
        make_window("a", 0, brow_raise=True, amplitude=0.30, speed=0.40),
        make_window("a", 1, brow_raise=False, amplitude=0.30, speed=0.40),
        make_window("a", 2, brow_raise=False, amplitude=0.90, speed=1.80),
    ]
    pairs = A.match_windows(windows, "brow_raise")
    assert pairs[0][1].index == 1


def test_unscored_windows_are_excluded() -> None:
    bad = make_window("a", 0, brow_raise=True)
    bad.scored = False
    bad.probs = None
    windows = [bad, make_window("a", 1, brow_raise=False)]
    assert A.match_windows(windows, "brow_raise") == []


# ---------------------------------------------------------------------------
# the planted-bias test: the whole point of the audit machinery
# ---------------------------------------------------------------------------


def test_a_planted_bias_is_recovered_with_a_ci_that_excludes_zero() -> None:
    """Marker-bearing windows really are more negative here, by construction."""
    rng = np.random.default_rng(0)
    windows = []
    for clip in range(40):
        for i in range(6):
            marked = i < 3
            windows.append(
                make_window(
                    f"c{clip}",
                    i,
                    brow_raise=marked,
                    # a little jitter so the matching has something to work with
                    amplitude=0.30 + rng.normal(0, 0.01),
                    speed=0.40 + rng.normal(0, 0.01),
                    neg=0.40 if marked else 0.20,
                )
            )
    pairs = A.match_windows(windows, "brow_raise", tolerance=2.0)
    assert len(pairs) >= 100, f"only {len(pairs)} pairs; the matcher is too strict"

    st = A.bootstrap_shift(pairs, "brow_raise", n_boot=2000, seed=1)
    assert st.mean_shift == pytest.approx(0.20, abs=0.02)
    assert st.ci_low > 0.0, "a planted bias must produce a CI excluding zero"
    assert st.ci_high > st.ci_low
    assert st.p_value < 0.01
    # anger/sadness/disgust/fear all move up; neutral moves down
    # neg 0.40 vs 0.20 spread over four classes is exactly +0.05 each.
    for cls in fer.NEGATIVE_LABELS:
        assert st.per_class_shift[cls] == pytest.approx(0.05, abs=0.01), cls
    assert st.per_class_shift["neutral"] == pytest.approx(-0.20, abs=0.02)


def test_no_planted_bias_yields_an_interval_spanning_zero() -> None:
    rng = np.random.default_rng(2)
    windows = []
    for clip in range(40):
        for i in range(6):
            marked = i < 3
            windows.append(
                make_window(
                    f"c{clip}",
                    i,
                    brow_raise=marked,
                    amplitude=0.30 + rng.normal(0, 0.02),
                    speed=0.40 + rng.normal(0, 0.02),
                    # Identical expectation either way, with per-window noise. A
                    # perfectly constant effect would give a bootstrap interval of
                    # exactly [0, 0], and this test could not then tell "correctly
                    # null" from "degenerate".
                    neg=0.30 + rng.normal(0, 0.01),
                )
            )
    st = A.bootstrap_shift(
        A.match_windows(windows, "brow_raise", tolerance=2.0),
        "brow_raise",
        n_boot=2000,
        seed=3,
    )
    assert abs(st.mean_shift) < 0.02
    assert st.ci_low < 0.0 < st.ci_high, (
        f"a null effect must not produce a CI excluding zero, got "
        f"[{st.ci_low:.5f}, {st.ci_high:.5f}] over {st.n_pairs} pairs"
    )


def test_bootstrap_resamples_clips_not_windows() -> None:
    """The interval must be a cluster bootstrap, and this tests it directly.

    Two datasets with the *same* mean effect: one where the bias appears in
    every clip, one where it appears in 6 of 30. The mean shift is equal by
    construction, so the only thing that can differ is the width - and it must be
    wider for the concentrated case, because the effective sample size is clips.
    If it were a window-level bootstrap the two widths would be nearly equal and
    the concentrated case would look better-powered than it is.
    """
    rng = np.random.default_rng(5)
    spread, concentrated = [], []
    for clip in range(30):
        for i in range(6):
            marked = i < 3
            base = {
                "clip": f"c{clip}",
                "index": i,
                "brow_raise": marked,
                "amplitude": 0.30 + rng.normal(0, 0.02),
                "speed": 0.40 + rng.normal(0, 0.02),
            }
            # Every clip carries +0.10 -> mean effect 0.10.
            spread.append(make_window(neg=0.30 + (0.10 if marked else 0.0), **base))
            # 6 clips carry +0.50, the rest nothing -> also 6/30 * 0.50 = 0.10.
            big = 0.30 + (0.50 if marked else 0.0) if clip < 6 else 0.30
            concentrated.append(make_window(neg=big, **base))

    a = A.bootstrap_shift(
        A.match_windows(spread, "brow_raise", tolerance=2.0), "x", n_boot=1500, seed=4
    )
    b = A.bootstrap_shift(
        A.match_windows(concentrated, "brow_raise", tolerance=2.0), "x", n_boot=1500, seed=4
    )
    assert a.mean_shift == pytest.approx(b.mean_shift, rel=0.15), (
        f"the two designs must carry a comparable mean effect for this test to "
        f"isolate width: {a.mean_shift:.4f} vs {b.mean_shift:.4f}"
    )

    def width(stat):
        return stat.ci_high - stat.ci_low

    assert width(b) > 1.5 * width(a), (
        f"concentrated width {width(b):.4f} should far exceed spread width {width(a):.4f}; "
        f"if not, the bootstrap is not clustering on clips"
    )


def test_empty_pairs_produce_an_inert_record() -> None:
    st = A.bootstrap_shift([], "brow_raise")
    assert st.n_pairs == 0
    assert st.p_value == 1.0
    assert st.mean_shift == 0.0


# ---------------------------------------------------------------------------
# table and reporting
# ---------------------------------------------------------------------------


def test_marker_emotion_table_splits_on_presence() -> None:
    windows = [make_window("a", i, brow_raise=(i < 2), neg=0.4 if i < 2 else 0.1) for i in range(4)]
    table = A.marker_emotion_table(windows)
    row = table["brow_raise"]
    assert row["n_on"] == 2
    assert row["n_off"] == 2
    assert row["p_anger"] > row["p_anger_off"]


def test_table_omits_markers_with_nothing_on_one_side() -> None:
    windows = [make_window("a", i, neg=0.2) for i in range(4)]
    table = A.marker_emotion_table(windows)
    assert "brow_raise" not in table


def test_render_table_has_a_row_per_stat() -> None:
    st = A.bootstrap_shift(
        [(make_window("a", 0, neg=0.4), make_window("a", 1, neg=0.1))], "brow_raise"
    )
    text = A.render_table([st])
    assert "brow_raise" in text
    assert text.count("\n") >= 2


def test_describe_declares_the_claim_bearing_choices() -> None:
    d = fer.describe()
    assert d["neutral_excluded"] is True
    assert d["pretrained_backbone"] is False
    assert d["non_signer_only"] is True
    assert "RAF-DB" in d["training_data"]


def test_crop_is_none_when_no_face() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    landmarks = np.zeros((553, 3), dtype=np.float32)
    assert fer.face_crop(frame, landmarks, present=False) is None
    assert fer.face_bbox(landmarks, present=True) is None  # degenerate box


def test_crop_is_translation_and_scale_invariant() -> None:
    """The FER model must see the same crop for a signer near the lens and one far away."""
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    base = np.zeros((553, 3), dtype=np.float32)
    fs, fe = 75, 543
    for k in range(fs, fe):
        base[k, 0] = 0.45 + 0.001 * (k % 30)
        base[k, 1] = 0.40 + 0.0008 * (k % 25)
    near = base.copy()
    near[fs:fe, :2] = (base[fs:fe, :2] - 0.425) * 1.8 + 0.5
    c1 = fer.face_crop(frame, base, True, size=64)
    c2 = fer.face_crop(frame, near, True, size=64)
    assert c1 is not None and c2 is not None
    assert c1.shape == c2.shape == (64, 64, 3)
    # The two crops frame the same face, so they should agree to within a small
    # fraction of the dynamic range. 15/255 ~ 6%: loose enough to allow the
    # different pixel scales to change which pixels the resampler lands on,
    # tight enough that a crop pointed at the wrong region would fail.
    assert np.abs(c1.astype(float) - c2.astype(float)).mean() < 15.0


def test_preprocess_faces_gives_a_channel_first_tensor() -> None:
    crops = [np.full((30, 30, 3), 200, dtype=np.uint8) for _ in range(4)]
    x = fer.preprocess_faces(crops, size=16)
    assert x.shape == (4, 3, 16, 16)
    assert x.max() <= 1.0
    assert fer.preprocess_faces([], size=16).shape[0] == 0


def test_build_model_is_small_and_classifies() -> None:
    import torch

    from seam.eval.fer import FerModelSpec, build_model

    spec = FerModelSpec("t", seed=0, width=8, blocks=2)
    model = build_model(spec)
    n = sum(p.numel() for p in model.parameters())
    assert n < 60_000, f"{n:,} params is not a compact baseline"
    out = model(torch.zeros(2, 3, fer.CROP_SIZE, fer.CROP_SIZE))
    assert out.shape == (2, len(fer.EMOSIGN_LABELS))


# ---------------------------------------------------------------------------
# the FER front end, which the first audit run showed was the instrument's fault
# ---------------------------------------------------------------------------


def test_normalize_face_survives_realistic_exposure_change() -> None:
    """The property that actually matters, stated at its real strength.

    The front end is grayscale, then histogram equalisation, then a per-image
    z-score. Only the last step is exactly invariant to a linear intensity
    change; ``equalizeHist`` is a rank-based CDF map, so a brightness shift
    changes it non-trivially. What has to hold for the instrument to work is the
    weaker, practical property: a realistic exposure change leaves the normalised
    crop *highly correlated* with the original and changes it by a bounded
    amount, so the network sees the same face.

    The failure this guards against is concrete: with no front end at all, the
    FER models collapsed to one predicted class across all 200 EmoSign clips.
    """
    rng = np.random.default_rng(0)
    base = rng.integers(40, 200, (60, 80, 3), dtype=np.uint8)
    brighter = np.clip(base.astype(float) * 1.12 + 8, 0, 255).astype(np.uint8)
    darker = np.clip(base.astype(float) * 0.88 - 6, 0, 255).astype(np.uint8)

    a = fer.normalize_face(base, size=64)
    for variant, name in ((brighter, "brighter"), (darker, "darker")):
        b = fer.normalize_face(variant, size=64)
        r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        assert r > 0.9, f"{name}: correlation only {r:.3f}"
        assert np.abs(a - b).mean() < 0.25, f"{name}: mean shift {np.abs(a - b).mean():.3f}"


def test_normalize_face_keeps_structure() -> None:
    """Equalising a face must not flatten it into noise."""
    face = np.zeros((112, 112, 3), dtype=np.uint8)
    face[30:60, 40:70] = 200  # a bright band
    face[70:90, 45:65] = 60  # a dark band
    out = fer.normalize_face(face)
    assert out.std() > 0.5, "a structured input must retain contrast"
    assert abs(float(out.mean())) < 0.1


def test_normalize_face_is_grayscale_replicated() -> None:
    """Colour is the nuisance being removed, so the three channels are identical."""
    colour = np.zeros((40, 40, 3), dtype=np.uint8)
    colour[..., 0] = 200
    out = fer.normalize_face(colour, size=40)
    assert out.shape == (3, 40, 40)
    np.testing.assert_array_equal(out[0], out[1])
    np.testing.assert_array_equal(out[1], out[2])


def test_preprocess_faces_uses_the_same_front_end() -> None:
    crops = [
        np.random.default_rng(i).integers(0, 255, (50, 50, 3), dtype=np.uint8) for i in range(3)
    ]
    x = fer.preprocess_faces(crops, size=32)
    assert x.shape == (3, 3, 32, 32)
    np.testing.assert_allclose(x.mean(axis=(1, 2, 3)), 0.0, atol=1e-5)
    np.testing.assert_allclose(x.std(axis=(1, 2, 3)), 1.0, atol=1e-3)


def test_normalize_face_rejects_a_float_crop() -> None:
    """A dtype mismatch between the fitting and inference paths must be loud.

    The first version of the pipeline returned float32 in [0, 1] from `face_crop`
    while the fitting path handed over uint8 from PIL, so `equalizeHist` raised
    only on the inference side. Asserted so the contract cannot drift again.
    """
    with pytest.raises(TypeError, match="uint8 crop"):
        fer.normalize_face(np.zeros((20, 20, 3), dtype=np.float32))


def test_face_crop_returns_uint8() -> None:
    frame = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    landmarks = np.zeros((553, 3), dtype=np.float32)
    start, end = 75, 543
    for k in range(start, end):
        landmarks[k, 0] = 0.45 + 0.001 * (k % 30)
        landmarks[k, 1] = 0.40 + 0.0008 * (k % 25)
    crop = fer.face_crop(frame, landmarks, present=True, size=32)
    assert crop is not None
    assert crop.dtype == np.uint8
    assert crop.shape == (32, 32, 3)
    # The whole pipeline accepts it without a dtype fix-up.
    assert fer.preprocess_faces([crop], size=16).shape == (1, 3, 16, 16)


def test_every_result_reports_its_minimum_detectable_effect() -> None:
    """A null without its power is an absence of measurement, not a finding.

    The MDE is computed from the same paired statistic as the effect, so a
    reader can see immediately whether a null rules out a large bias or a tiny
    one. This asserts it is populated and ordered sensibly relative to the CI.
    """
    rng = np.random.default_rng(7)
    windows = []
    for clip in range(30):
        for i in range(6):
            marked = i < 3
            windows.append(
                make_window(
                    f"c{clip}",
                    i,
                    brow_raise=marked,
                    amplitude=0.30 + rng.normal(0, 0.02),
                    speed=0.40 + rng.normal(0, 0.02),
                    neg=0.30 + rng.normal(0, 0.01),  # no effect
                )
            )
    st = A.bootstrap_shift(
        A.match_windows(windows, "brow_raise", tolerance=2.0), "brow_raise", n_boot=800, seed=1
    )
    assert not np.isnan(st.mde), "every reported effect needs its power"
    assert st.mde > 0
    # More clips must buy a smaller detectable effect.
    small = A.bootstrap_shift(
        A.match_windows(windows[: 6 * 10], "brow_raise", tolerance=2.0), "x", n_boot=400, seed=1
    )
    assert small.mde > st.mde


def test_too_few_clips_reports_no_interval_and_no_mde() -> None:
    windows = [
        make_window("a", 0, brow_raise=True, neg=0.4),
        make_window("a", 1, neg=0.1),
    ]
    st = A.bootstrap_shift(list(zip(windows, windows, strict=True)), "brow_raise", n_boot=100)
    assert np.isnan(st.ci_low) and np.isnan(st.p_value) and np.isnan(st.mde)


# ---------------------------------------------------------------------------
# human frame-level markers through the same window builder
# ---------------------------------------------------------------------------


def _clip(n: int = 120, fps: float = 24.0, seed: int = 0) -> tuple[dict, object]:
    """A synthetic clip the window builder accepts: a fixed pose with moving hands."""
    from seam.perception.extract import ClipMeta
    from seam.perception.tasks_api import FACE_BLENDSHAPES, TOTAL_POINTS

    rng = np.random.default_rng(seed)
    base = rng.uniform(0.3, 0.7, size=(TOTAL_POINTS, 3)).astype(np.float32)
    wobble = 0.02 * rng.normal(size=(n, TOTAL_POINTS, 3)).astype(np.float32)
    drift = 0.05 * np.sin(np.linspace(0, 6 * np.pi, n))[:, None, None].astype(np.float32)
    arrays = {
        "landmarks": base[None] + wobble + drift,
        "blendshapes": rng.uniform(0, 0.2, size=(n, FACE_BLENDSHAPES)),
        "presence": np.ones((n, 4), dtype=bool),
        "head_rotation": np.tile(np.eye(4), (n, 1, 1)),
    }
    meta = ClipMeta(
        source_path="x.mp4",
        utterance_id="1",
        native_fps=fps,
        source_width=256,
        source_height=256,
        frame_count=n,
        detection_rates={},
    )
    return arrays, meta


def _probs(n: int, negative: np.ndarray) -> np.ndarray:
    """Per-frame FER rows whose negative mass is exactly ``negative``."""
    p = np.zeros((n, len(fer.EMOSIGN_LABELS)), dtype=np.float32)
    for k in fer.NEG_IDX:
        p[:, k] = negative / len(fer.NEG_IDX)
    p[:, fer.NEUTRAL_IDX] = 1.0 - negative
    return p


def test_default_windows_still_carry_the_heuristic_markers() -> None:
    arrays, meta = _clip()
    ws = A.build_windows("c", arrays, meta, _probs(120, np.full(120, 0.3)))
    assert ws and all(w.names == M.MARKERS and w.partial == () for w in ws)
    assert all(w.n_frames <= A.WINDOW for w in ws)


def test_human_markers_give_bearing_free_and_partial_windows() -> None:
    """24 fps clip, analysis at 12 fps: native frames 48-71 are analysis frames 24-35."""
    arrays, meta = _clip()
    track = np.zeros(120, dtype=bool)
    track[48:72] = True
    ws = A.build_windows(
        "c",
        arrays,
        meta,
        _probs(120, np.full(120, 0.3)),
        window=6,
        stride=3,
        frame_markers={"brow_raise": track, "negation": np.zeros(120, dtype=bool)},
    )
    assert ws[0].names == ("brow_raise", "negation")
    by_start = {w.start: w for w in ws}
    assert by_start[24].marker("brow_raise") and by_start[27].marker("brow_raise")
    assert by_start[0].free_of("brow_raise") and by_start[42].free_of("brow_raise")
    # Starts at 21: frames 21-26, half covered -> bearing. Starts at 33: frames 33-38,
    # half covered -> bearing. Starts at 36: untouched -> free.
    assert by_start[21].marker("brow_raise")
    assert by_start[36].free_of("brow_raise")
    # A window the event only clips is on neither side.
    track2 = np.zeros(120, dtype=bool)
    track2[48:52] = True  # analysis frames 24-25 only
    ws2 = A.build_windows(
        "c",
        arrays,
        meta,
        _probs(120, np.full(120, 0.3)),
        window=6,
        stride=3,
        frame_markers={"brow_raise": track2},
    )
    w = {x.start: x for x in ws2}[24]
    assert not w.marker("brow_raise") and not w.free_of("brow_raise")
    assert w.partial == ("brow_raise",)
    assert all(x.free_of("negation") for x in ws)


def test_a_marker_track_of_the_wrong_length_is_refused() -> None:
    """A track built at the wrong frame rate has the wrong length; that must not pass."""
    arrays, meta = _clip()
    with pytest.raises(ValueError, match="frames"):
        A.build_windows(
            "c",
            arrays,
            meta,
            _probs(120, np.full(120, 0.3)),
            frame_markers={"brow_raise": np.zeros(150, dtype=bool)},
        )


def test_partial_windows_are_not_used_as_the_marker_free_side() -> None:
    marked = make_window("c", 0, brow_raise=True, neg=0.5)
    partial = make_window("c", 1, neg=0.9)
    partial.partial = ("brow_raise",)
    clean = make_window("c", 2, neg=0.1)
    pairs = A.match_windows([marked, partial, clean], "brow_raise")
    assert [(a.index, b.index) for a, b in pairs] == [(0, 2)]


def _human_design(effect: float, n_clips: int = 24) -> list[A.Window]:
    """Clips with a human-marked span whose frames carry ``effect`` extra negative mass."""
    out: list[A.Window] = []
    for c in range(n_clips):
        arrays, meta = _clip(seed=c)
        rng = np.random.default_rng(100 + c)
        track = np.zeros(120, dtype=bool)
        start = int(rng.integers(10, 70))
        track[start : start + 24] = True
        negative = 0.3 + 0.02 * rng.normal(size=120) + effect * track
        out += A.build_windows(
            f"clip{c}",
            arrays,
            meta,
            _probs(120, np.clip(negative, 0, 1)),
            window=6,
            stride=3,
            frame_markers={"brow_raise": track},
        )
    return out


def test_a_planted_bias_on_human_marked_frames_is_recovered_end_to_end() -> None:
    """The continuous-signing audit's positive control, through the real window builder.

    Frame-level FER rows, a frame-level human track, resampling to the analysis rate,
    windowing, within-clip matching and the cluster bootstrap: a +0.05 shift planted on
    the marked frames must come out with an interval that excludes zero and covers it.
    """
    pairs = A.match_windows(_human_design(0.05), "brow_raise", tolerance=5.0)
    stats = A.bootstrap_shift(pairs, "brow_raise")
    assert stats.n_clips >= 20
    assert stats.ci_low > 0
    assert stats.ci_low < 0.05 < stats.ci_high or stats.mean_shift == pytest.approx(0.05, abs=0.01)


def test_no_planted_bias_on_human_marked_frames_reads_as_null() -> None:
    pairs = A.match_windows(_human_design(0.0), "brow_raise", tolerance=5.0)
    stats = A.bootstrap_shift(pairs, "brow_raise")
    assert stats.ci_low < 0 < stats.ci_high
    assert stats.mde == stats.mde, "a null must carry its minimum detectable effect"
