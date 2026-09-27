"""The EmoSign loader and the ASLLRP join.

The join is the project's critical fact - it is what makes 200 labelled affect
clips reachable at all - so it is tested against the *real* file as well as a
fixture, and the real test is marked ``network`` because it needs the fetched
labels rather than a transcribed sample.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from seam.data import emosign as em


def test_parses_utterance_id_from_both_naming_forms(data_root: Path) -> None:
    labels = em.load(data_root)
    ids = {c.video_name: c.utterance_id for c in labels}
    assert ids["Cory_2013-6-25_sc0106_1014088"] == "1014088"
    assert ids["10-Ben-Conclusion_24363254"] == "24363254"
    assert ids["Rachel_2011-05-31_sc12_777"] == "777"


def test_normalises_signers_and_collapses_session_years(data_root: Path) -> None:
    labels = em.load(data_root)
    assert labels.signer_counts == {"Ben": 1, "Cory": 1, "Rachel": 1}
    rachel = labels.by_signer("Rachel")
    assert len(rachel) == 1


def test_parses_sentiment_and_intensities(data_root: Path) -> None:
    labels = em.load(data_root)
    by_name = {c.video_name: c for c in labels}

    cory = by_name["Cory_2013-6-25_sc0106_1014088"]
    assert cory.sentiment == 3
    assert cory.intensities["joy"] == 4
    assert cory.intensities["anger"] == em.INTENSITY_MIN
    assert cory.sentiment3 == "positive"
    assert cory.valence == pytest.approx(1.0)
    assert cory.dominant_emotion == "joy"
    assert cory.n_active_emotions == 1

    ben = by_name["10-Ben-Conclusion_24363254"]
    assert ben.sentiment3 == "negative"
    # worry=3, frustration=2: two emotions above baseline, but a unique maximum,
    # so there IS a dominant emotion and this is not a single-expression clip.
    # The two concepts are different and the subset rule uses both.
    assert ben.n_active_emotions == 2
    assert ben.dominant_emotion == "worry"
    assert ben not in labels.single_expression()

    neutral = by_name["Rachel_2011-05-31_sc12_777"]
    assert neutral.sentiment3 == "neutral"
    # Nothing above baseline: absence of emotion is not a dominant emotion.
    assert neutral.dominant_emotion is None
    assert neutral.n_active_emotions == 0


def test_keeps_the_three_cue_columns(data_root: Path) -> None:
    labels = em.load(data_root)
    cory = next(c for c in labels if c.signer == "Cory")
    assert cory.cues == ("raised brows, head shake", "wide eyes", "smile")
    ben = next(c for c in labels if c.signer == "Ben")
    assert ben.cues[1] == ""


def test_single_expression_selection(data_root: Path) -> None:
    labels = em.load(data_root)
    assert [c.video_name for c in labels.single_expression()] == ["Cory_2013-6-25_sc0106_1014088"]


def test_mangled_name_raises_rather_than_dropping_silently(data_root: Path) -> None:
    """A row that cannot be joined must fail loudly.

    Dropping it would cost one clip and look fine; dropping all 200 would look
    like an empty dataset. Neither should be silent.
    """
    csv_path = data_root / "emosign" / "emosign_dataset.csv"
    text = csv_path.read_text(encoding="utf-8")
    csv_path.write_text(
        text.replace("Cory_2013-6-25_sc0106_1014088", "no-numeric-suffix"), encoding="utf-8"
    )
    with pytest.raises(em.EmoSignError, match="utterance-ID join"):
        em.load(data_root)


def test_out_of_range_intensity_raises(data_root: Path) -> None:
    csv_path = data_root / "emosign" / "emosign_dataset.csv"
    text = csv_path.read_text(encoding="utf-8")
    csv_path.write_text(text.replace('"joy": ', "joy: ") if False else text, encoding="utf-8")
    original = csv_path.read_text(encoding="utf-8")
    csv_path.write_text(
        original.replace(",4,1,1,1,1,1,1,1,1,1,", ",9,1,1,1,1,1,1,1,1,1,"), encoding="utf-8"
    )
    with pytest.raises(em.EmoSignError, match="outside the"):
        em.load(data_root)


def test_missing_file_message_points_at_the_fetcher(tmp_path: Path) -> None:
    with pytest.raises(em.EmoSignError, match="seam data fetch"):
        em.load(tmp_path / "nothing")


@pytest.mark.network
def test_real_labels_join_two_hundred_of_two_hundred(real_data_root: Path) -> None:
    """The load-bearing assertion: 200/200 rows join to ASLLRP utterance IDs.

    Measured 2026-09-26. If this ever fails, the upstream naming changed and the
    video fetcher will be building wrong URLs, so the failure must be loud.
    """
    labels = em.load(real_data_root)
    assert len(labels) == 200
    assert all(c.utterance_id.isdigit() for c in labels)
    assert len({c.utterance_id for c in labels}) == 200


@pytest.mark.network
def test_real_signer_counts_match_the_plan(real_data_root: Path) -> None:
    labels = em.load(real_data_root)
    assert labels.signer_counts == {"Ben": 7, "Cory": 87, "Jonathan": 54, "Rachel": 52}
    assert labels.signers == ("Ben", "Cory", "Jonathan", "Rachel")


@pytest.mark.network
def test_every_real_clip_has_annotator_cues(real_data_root: Path) -> None:
    """M3's grounding move depends on the free-text cues being present."""
    labels = em.load(real_data_root)
    with_cues = sum(1 for c in labels if any(c.cues))
    assert with_cues == 200


@pytest.mark.network
def test_emotion_intensities_are_within_scale(real_data_root: Path) -> None:
    labels = em.load(real_data_root)
    for clip in labels:
        for name, value in clip.intensities.items():
            assert em.INTENSITY_MIN <= value <= em.INTENSITY_MAX, (clip.video_name, name, value)
