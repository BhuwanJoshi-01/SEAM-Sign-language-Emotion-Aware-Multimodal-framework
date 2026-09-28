"""M3: syntactic markers must keep their provenance, and the label harness must
refuse to read a degenerate marker as a null result.

The M2 experience set the pattern for these tests: every instrument in this project
has at some point reported a plausible number that meant nothing. A marker that
fires on 90% of clips will agree with anything at roughly the base rate, and the
resulting lift of ~1.0 looks exactly like a careful null result. It is not one. The
distinction is what these tests pin.
"""

from __future__ import annotations

import numpy as np
import pytest

from seam.features import syntactic as SY

# --- the gloss map ---------------------------------------------------------


def test_gloss_map_joins_all_emosign_utterances() -> None:
    """Every annotated clip must have real linguistic context, or the whole
    syntactic track is unfounded. 200/200 was measured; a drop means a data
    regression, not a sampling detail."""
    import re

    import pandas as pd

    from seam.data.emosign import labels_path
    from seam.paths import default_data_root

    path = labels_path(default_data_root())
    if not path.is_file():
        pytest.skip("EmoSign labels not present")
    gm = SY.load_gloss_map()
    ids = [
        m.group(1)
        for m in (re.search(r"(\d+)$", str(v)) for v in pd.read_csv(path)["video_name"])
        if m
    ]
    missing = [i for i in ids if i not in gm]
    assert not missing, f"{len(missing)}/{len(ids)} utterances have no gloss: {missing[:5]}"


def test_missing_utterance_yields_zero_confidence_not_a_guess() -> None:
    out = SY.classify("00000000", {})
    assert out.n_tokens == 0
    assert out.present() == []
    for lab in out.labels.values():
        assert lab.confidence == 0.0, "an unknown utterance must not be labelled confidently"


# --- classification --------------------------------------------------------


def test_interrogative_detects_wh_words() -> None:
    for gloss in ("WHO GO", "ME TELL YOU WHERE", "WHY THAT", '"WHAT" MEAN'):
        out = SY.classify_tokens(gloss.split())
        assert out.labels["interrogative"].present, f"missed {gloss!r}"
        assert out.labels["interrogative"].provenance == SY.PROV_LEXICAL


def test_negation_detected_with_evidence() -> None:
    out = SY.classify_tokens(["I", "GO", "NOT", "THERE"])
    lab = out.labels["negation"]
    assert lab.present
    assert "NOT" in lab.evidence, "a label without its evidence cannot be audited"


def test_declarative_has_no_markers() -> None:
    out = SY.classify_tokens(["YESTERDAY", "I", "GO", "STORE"])
    assert out.labels["interrogative"].present is False
    assert out.labels["negation"].present is False


def test_annotator_asides_cannot_create_a_marker() -> None:
    """Quoted asides are the annotator talking, not the utterance.

    ``"how"`` inside an aside must not make a statement interrogative, or a
    comment like ``5"hesitation"`` would inject a marker.
    """
    plain = SY.classify_tokens(["I", "GO", "STORE"])
    with_aside = SY.classify_tokens(["I", "GO", "STORE", '5"how', 'interesting"'])
    assert with_aside.labels["interrogative"].present == plain.labels["interrogative"].present


def test_topicalization_is_capped_as_a_weak_inference() -> None:
    """Topicalization is inferred, so it must never claim high confidence.

    The gloss is a flat token sequence with no constituent structure, so
    "this was fronted" is not something the annotation states. A confidence near
    1.0 on this label would be a fabricated claim.
    """
    out = SY.classify_tokens(["IX", "BOOK", "IX", "READ"])
    lab = out.labels["topicalization"]
    assert lab.provenance == SY.PROV_PSEUDO
    assert lab.confidence <= 0.5, f"pseudo-label claimed {lab.confidence} confidence"


def test_compound_prefix_is_not_facial_signal() -> None:
    """``fs-`` marks compound signs, not facial behaviour.

    Reading the prefix as a non-manual marker would have added 98 token types to
    the non-manual vocabulary for no reason. This test exists so that reading
    cannot creep back in.
    """
    assert "compound" in SY.describe()["fs_prefix_meaning"]
    assert "NOT facial" in SY.describe()["fs_prefix_meaning"]
    out = SY.classify_tokens(["fs-BEACH", "fs-LATE", "fs-OF"])
    assert not any(lab.present for lab in out.labels.values()), (
        "compound tokens must not trigger any linguistic marker"
    )


def test_every_label_carries_a_provenance_string() -> None:
    out = SY.classify_tokens(["WHO", "NOT", "IX"])
    for name, lab in out.labels.items():
        assert lab.provenance, f"{name} has no provenance"
        assert lab.provenance in (SY.PROV_LEXICAL, SY.PROV_PSEUDO)


def test_negation_lexicon_is_not_just_this_corpus() -> None:
    """A lexicon containing only what this corpus happens to contain will miss
    negation the moment the corpus changes."""
    assert {"NO", "CAN'T", "WITHOUT"} <= SY.NEGATIONS


# --- head-marker calibration ----------------------------------------------


def test_head_oscillation_requires_large_amplitude_reversals() -> None:
    """A shake is several large deflections, not one sign change.

    The old test counted sign changes anywhere in the clip, so a single wobble
    licensed every noisy turning point and head_shake fired on 98% of clips.
    """
    from seam.features import markers as VM

    fps = 25.0
    th = VM.MarkerThresholds()
    t = np.arange(0, 2.0, 1 / fps)

    # Small-amplitude jitter: many sign changes, none reaching head_angle.
    jitter = 0.05 * np.sin(2 * np.pi * 9 * t) + 0.02 * np.random.default_rng(0).normal(
        0, 0.01, t.size
    )
    assert not VM._oscillation(jitter, fps, th).any(), "tracking jitter must not read as a shake"

    # A real shake: repeated large-amplitude direction reversals.
    big = 0.5 * np.sin(2 * np.pi * 2.5 * t)
    assert VM._oscillation(big, fps, th).any(), "a real shake must be detected"


def test_head_angle_is_used_as_documented_not_halved() -> None:
    """The field documents ~20 degrees; the code previously used half of it.

    A threshold of 10 degrees is inside the tracking jitter of a 256x256 face, so
    this pins the documented value rather than the accidental one.
    """
    from seam.features import markers as VM

    fps = 25.0
    t = np.arange(0, 2.0, 1 / fps)
    # Amplitude 0.3 rad: above the halved threshold (0.175), below 0.35.
    mid = 0.3 * np.sin(2 * np.pi * 2.5 * t)
    assert not VM._oscillation(mid, fps, VM.MarkerThresholds()).any()
    assert VM._oscillation(mid, fps, VM.MarkerThresholds(head_angle=0.175)).any()


# --- the duration confound -------------------------------------------------


def test_magnitude_mean_is_duration_free() -> None:
    """A marker's magnitude must not grow just because the clip is longer.

    Every syntactic label tracks clip length on this corpus, because questions and
    negated statements are longer utterances. An integral statistic therefore
    inherits that confound: the first version of this analysis reported
    interrogative/brow_raise at r_pb = +0.414 on the integral and -0.045 on the
    mean - the whole association was clip length.
    """
    from seam.features import markers as VM

    fps = 25.0
    cols = [VM.BLENDSHAPE_INDEX[name] for name in VM.MOUTH_MORPHEME]
    # The same *per-frame rate* in clips of two different lengths: the marker is up
    # for 20 of 60 frames and for 40 of 120. The mean must be identical, and only
    # the integral may differ. The marker stays a minority of each clip, because the
    # evidence normalises by (p95 - median) of the clip itself and a marker
    # occupying over half the clip drags the baseline up with it - documented on
    # clip_evidence.
    short = np.zeros((60, 52))
    short[10:30, cols] = 1.0
    long = np.zeros((120, 52))
    long[10:50, cols] = 1.0

    def sig_for(arr: np.ndarray) -> VM.MarkerSignals:
        return VM.signals(arr, rotation=np.zeros((len(arr), 4, 4)), fps=fps)

    m_short = VM.clip_magnitude(sig_for(short), fps, reduce="mean")["mouth_morpheme"]
    m_long = VM.clip_magnitude(sig_for(long), fps, reduce="mean")["mouth_morpheme"]
    assert m_short > 0.0, "fixture must produce a real signal"
    assert m_short == pytest.approx(m_long, rel=1e-6), (
        "a per-frame mean must not depend on clip length"
    )

    t_short = VM.clip_magnitude(sig_for(short), fps, reduce="total")["mouth_morpheme"]
    t_long = VM.clip_magnitude(sig_for(long), fps, reduce="total")["mouth_morpheme"]
    assert t_long > t_short * 1.5, "the integral is expected to grow with duration"


def test_clip_magnitude_rejects_unknown_reduction() -> None:
    from seam.features import markers as VM

    sig = VM.signals(np.zeros((10, 52)), rotation=np.zeros((10, 4, 4)), fps=25.0)
    with pytest.raises(ValueError, match="reduce"):
        VM.clip_magnitude(sig, 25.0, reduce="median")


def test_clip_presence_requires_duration_and_peak() -> None:
    """A single-frame spike must not make a clip marker-bearing.

    ASL non-manual markers are held; tracker noise is a blip. The clip-level rule
    is a different instrument from the per-frame one for exactly this reason.
    """
    from seam.features import markers as VM

    fps = 25.0
    n = 100
    cols = [VM.BLENDSHAPE_INDEX[name] for name in VM.MOUTH_MORPHEME]
    blip = np.zeros((n, 52))
    blip[50, cols] = 1.0  # one frame only
    held = np.zeros((n, 52))
    held[40:70, cols] = 1.0  # 1.2 s of sustained signal

    def pres(arr: np.ndarray) -> dict[str, object]:
        sig = VM.signals(arr, rotation=np.zeros((len(arr), 4, 4)), fps=fps)
        return VM.clip_presence(sig, fps)["mouth_morpheme"]

    assert not pres(blip)["present"], "a one-frame spike is not a marker"
    assert pres(held)["present"], "a held marker must be detected"


def test_flat_clip_has_no_exaggerated_evidence() -> None:
    """A clip with no dynamic range cannot be expressive on any channel.

    Dividing by a floored spread would manufacture evidence out of numerical
    noise, so a constant signal yields zero evidence instead.
    """
    from seam.features import markers as VM

    flat = np.zeros((60, 52))
    sig = VM.signals(flat, rotation=np.zeros((60, 4, 4)), fps=25.0)
    ev = VM.clip_evidence(sig)["mouth_positive"]
    assert np.all(ev == 0.0)
    assert VM.clip_magnitude(sig, 25.0)["mouth_positive"] == 0.0
