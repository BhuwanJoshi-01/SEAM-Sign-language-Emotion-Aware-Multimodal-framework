"""The hosted demo page is one static file, and it has to stand on its own.

`docs/index.html` is what GitHub Pages serves. There is no server behind it there, so it
may not depend on this project's API, and anything it links to has to be in `docs/`.
Whether it actually tracks a face is checked in a real browser by `scripts/site_smoke.py`.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "index.html"


def test_the_page_exists_and_is_a_document() -> None:
    html = PAGE.read_text(encoding="utf-8")
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert "<title>" in html and 'name="viewport"' in html


def test_the_page_needs_no_server() -> None:
    html = PAGE.read_text(encoding="utf-8")
    assert "/api/" not in html, "the hosted page has no server to call"
    assert "/static/" not in html


def test_all_three_landmarkers_are_loaded() -> None:
    html = PAGE.read_text(encoding="utf-8")
    for needle in ("tasks-vision", "face_landmarker", "hand_landmarker", "pose_landmarker"):
        assert needle in html
    assert "outputFaceBlendshapes: true" in html
    assert "outputFacialTransformationMatrixes: true" in html


def test_every_figure_the_page_shows_is_in_the_repository() -> None:
    html = PAGE.read_text(encoding="utf-8")
    figures = set(re.findall(r'src="(figures/[\w.-]+)"', html))
    assert figures, "the results section should show the report figures"
    missing = sorted(f for f in figures if not (PAGE.parent / f).is_file())
    assert not missing, f"figures referenced but not present: {missing}"


def test_every_image_has_alt_text() -> None:
    html = PAGE.read_text(encoding="utf-8")
    for tag in re.findall(r"<img\b[^>]*>", html):
        assert re.search(r'alt="[^"]{20,}"', tag), f"an image has no real alt text: {tag[:80]}"


def test_the_page_does_not_claim_emotion_recognition() -> None:
    """The affect model did not beat chance, so the page shows signals and says so."""
    html = PAGE.read_text(encoding="utf-8")
    assert "withholds an emotion label" in html
    assert "not an emotion verdict" in html


def test_validation_badges_quote_the_measured_numbers() -> None:
    import json

    import pytest

    path = REPO / "artifacts" / "m3" / "marker_validation.json"
    if not path.is_file():
        pytest.skip("marker_validation.json not built")
    by_fold = json.loads(path.read_text())["markers"]["brow_raise"][
        "heuristic_within_clip_auc_by_fold"
    ]
    vals = [by_fold[f] for f in ("Cory", "Jonathan", "Rachel")]
    html = PAGE.read_text(encoding="utf-8")
    assert f"{min(vals):.2f}–{max(vals):.2f}" in html, "the badge must match the artifact"
