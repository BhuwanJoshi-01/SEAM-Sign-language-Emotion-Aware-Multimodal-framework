"""The cue vocabulary must not decide its own conclusions.

The grounding report's whole value is that it can say "our feature set does not
recover what the annotators named". That claim is only credible if the instrument
cannot manufacture the opposite, so these tests pin the two ways it could:

* a cue vocabulary that silently drops the strings it cannot parse, which reports a
  clean result on the remainder and hides the selection;
* a feature that cannot discriminate, whose non-separation is reported as a null
  rather than as a blind instrument.
"""

from __future__ import annotations

import numpy as np
import pytest

from seam.features import cues as C


def test_coverage_is_reported_not_assumed() -> None:
    """A vocabulary must state how much of the annotators' text it accounts for."""
    pytest.importorskip("pandas")
    cov = C.coverage()
    assert cov["n_strings"] == 600, "the corpus is 3 annotator columns x 200 clips"
    # Motor coverage has to be substantial for the grounding test to mean anything,
    # and the remainder has to be *accounted for* rather than dropped.
    assert cov["n_with_motor_cue"] > 0.6 * cov["n_strings"]
    assert cov["n_with_motor_cue"] + cov["n_affective_only"] <= cov["n_strings"]
    assert cov["note"], "unmatched strings must be explained, not discarded"


def test_affective_interpretations_are_not_motor_cues() -> None:
    """ "Conveys surprise" is the annotator's reading, not a movement.

    Treating it as a brow-raise cue would manufacture agreement, because the FER
    model is trained to predict exactly that reading from a face.
    """
    p = C.parse_cue("his facial expression conveys a sense of surprise and worry")
    assert p.motor == {} or "brow_raise" not in p.motor
    assert p.affective
    assert p.is_affective_only


def test_motor_cues_are_detected_with_their_span() -> None:
    p = C.parse_cue("raised eye brow and a head shake, with a slight smile")
    for name in ("brow_raise", "head_shake", "smile"):
        assert name in p.motor, f"missed {name}"
        assert p.motor[name], "every cue must carry the span that matched"
    assert not p.is_affective_only


def test_every_cue_pattern_is_reachable() -> None:
    """A pattern that can never match is a claim in the report with no evidence."""
    samples = {
        "brow_raise": "raised eye brows",
        "brow_furrow": "his furrowed brow",
        "head_shake": "a head shake",
        "head_nod": "a head nod",
        "head_tilt": "head tilted down",
        "mouth_shape": "bared teeth",
        "smile": "smiles slightly",
        "eye_widen": "eyes popped wide",
        "blink_close": "eyes closed",
        "gaze_shift": "eye gaze shifted",
        "sign_size": "signs are large in this clip",
        "speed_fast": "very fast signing",
        "speed_slow": "slow and deliberate",
        "repetition": "repeated twice",
        "emphasis": "strong emphasis on it",
        "fingerspelling": "fingerspelled the name",
        "pause_hesitation": "a pause before answering",
        "body_posture": "upright posture",
    }
    for spec in C.MOTOR_CUES:
        assert spec.name in samples, f"no sample string for {spec.name}"
        parsed = C.parse_cue(samples[spec.name])
        assert spec.name in parsed.motor, (
            f"{spec.name} pattern cannot match its own sample {samples[spec.name]!r}"
        )


def test_speed_cues_share_a_feature_in_opposite_directions() -> None:
    """fast and slow must map to the same feature, or one of them is a stub."""
    specs = {c.name: c for c in C.MOTOR_CUES}
    assert specs["speed_fast"].expect == specs["speed_slow"].expect


def test_unmapped_cues_are_a_real_list_not_a_placeholder() -> None:
    """Cues annotators used that no feature tests are the deliverable here."""
    unmapped = C.unmapped_cues()
    assert len(unmapped) >= 5, "the corpus clearly names cues with no feature behind them"
    for name in unmapped:
        assert name in C.MOTOR_CUE_NAMES


def test_expectations_declare_rationale() -> None:
    """A mapping with no stated reason cannot be argued with."""
    for name, info in C.expectations().items():
        assert info["rationale"], f"{name} has no rationale"
        assert isinstance(info["testable_now"], bool)


# --- a blind feature must not be reported as a null ------------------------


def test_zero_inflated_feature_is_not_testable() -> None:
    """A feature that is zero on most clips cannot test a cue.

    Reporting its non-separation as a null would be indistinguishable from saying
    the annotators were wrong, when in fact nothing was measured.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from cue_grounding import ClipFeatures, test_cue

    n = 40
    # head_nod is the real zero-inflated case: measured at zero on 83% of EmoSign
    # clips, which is exactly why it is listed BLIND in the report.
    recs = [
        ClipFeatures(
            utterance_id=str(i),
            duration_s=4.0,
            marker_magnitude={"head_nod": 0.0 if i < 36 else 0.5},
            prosody={},
            hand_visible=1.0,
        )
        for i in range(n)
    ]
    present = [i % 2 == 0 for i in range(n)]
    row = test_cue(recs, present, "head_nod")
    assert not row["interpretable"]
    assert "cannot discriminate" in str(row["note"])


def test_constant_feature_is_not_testable() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from cue_grounding import ClipFeatures, test_cue

    n = 40
    recs = [
        ClipFeatures(
            utterance_id=str(i),
            duration_s=4.0,
            marker_magnitude={"brow_raise": 0.3},
            prosody={},
            hand_visible=1.0,
        )
        for i in range(n)
    ]
    row = test_cue(recs, [i % 2 == 0 for i in range(n)], "brow_raise")
    assert not row["interpretable"]


def test_permutation_null_is_calibrated() -> None:
    """Under no association the permutation p must be uniform, not always small.

    This is the instrument's own calibration. A test that reports p=0.001 on data
    with no signal is worse than no test, because it manufactures findings.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from cue_grounding import ClipFeatures, test_cue

    rng = np.random.default_rng(0)
    n = 120
    recs = [
        ClipFeatures(
            utterance_id=str(i),
            duration_s=4.0 + rng.random(),
            marker_magnitude={"brow_raise": float(rng.random())},
            prosody={},
            hand_visible=1.0,
        )
        for i in range(n)
    ]
    ps = []
    for _ in range(20):
        present = [bool(rng.integers(0, 2)) for _ in range(n)]
        row = test_cue(recs, present, "brow_raise", n_perm=2000)
        assert row["interpretable"], row.get("note")
        ps.append(float(row["perm_p"]))
    # Uniform p: the median should sit near 0.5, not near 0.
    assert 0.2 < float(np.median(ps)) < 0.8, f"permutation p not calibrated: {ps[:5]}"
    # And no draw in a calibrated test should look like a discovery.
    assert sum(1 for p in ps if p < 0.05) <= 4, f"too many false positives: {ps}"
