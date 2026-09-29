"""The demo server must never report a number it cannot stand behind.

Three properties, each of which has a specific way of failing quietly:

* **A malformed window is rejected, not repaired.** Padding a short array or reshaping
  a wrong one produces a smooth, confident, meaningless magnitude. The 52-coefficient
  width is the specific trap: MediaPipe emits 52 starting with ``_neutral``, and an
  implementation that dropped ``_neutral`` would shift every marker threshold by one
  while looking perfectly healthy.
* **Absent head tracking stays absent.** Zero-filling a missing pose is not a neutral
  default, it is a *plausible* head pose, and it makes "no tracking" indistinguishable
  from "perfectly still" - which silently changes head_shake and head_nod to zero.
* **Features the build cannot support are withheld, not shown empty.** M4's encoder was
  measured at 0.497 balanced accuracy against a 0.5 reference, so returning its affect
  output would put a number on screen that carries no information. An empty panel is
  honest; a confident wrong one is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from seam.serve.app import (
    MIN_FRAMES,
    NM_DIM,
    BadWindow,
    analyse,
    measure,
    parse_window,
)


def _window(n: int = 30, *, pose: bool = True, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    payload: dict = {"blendshapes": (rng.random((n, NM_DIM)) * 0.2).tolist(), "fps": 24}
    if pose:
        payload["head_pose"] = np.tile(np.eye(4).tolist(), (n, 1, 1)).tolist()
    return payload


# --- validation rejects rather than repairs --------------------------------


def test_accepts_a_well_formed_window() -> None:
    w = parse_window(_window())
    assert w.n_frames == 30
    assert w.blendshapes.shape == (30, NM_DIM)
    assert w.head_pose is not None


def test_rejects_the_wrong_coefficient_count() -> None:
    """51 coefficients means _neutral is missing and every threshold is off by one."""
    with pytest.raises(BadWindow, match="52"):
        parse_window({"blendshapes": [[0.1] * (NM_DIM - 1)] * 20, "fps": 24})
    with pytest.raises(BadWindow, match="52"):
        parse_window({"blendshapes": [[0.1] * (NM_DIM + 1)] * 20, "fps": 24})


def test_rejects_too_short_a_window() -> None:
    with pytest.raises(BadWindow, match=f"at least {MIN_FRAMES}"):
        parse_window(_window(n=MIN_FRAMES - 1))


@pytest.mark.parametrize("fps", [0, -1, 300, 1000])
def test_rejects_impossible_frame_rates(fps: float) -> None:
    p = _window()
    p["fps"] = fps
    with pytest.raises(BadWindow, match="fps"):
        parse_window(p)


def test_rejects_non_finite_values() -> None:
    p = _window()
    p["blendshapes"][3][2] = float("nan")
    with pytest.raises(BadWindow, match="non-finite"):
        parse_window(p)


def test_rejects_a_pose_that_does_not_cover_the_same_window() -> None:
    """Mismatched lengths mean the two signals describe different instants."""
    p = _window(n=30)
    p["head_pose"] = np.tile(np.eye(4).tolist(), (10, 1, 1)).tolist()
    with pytest.raises(BadWindow, match="same window"):
        parse_window(p)


def test_absent_pose_stays_absent() -> None:
    """Zero-filling a missing pose is a plausible pose, not a neutral one."""
    w = parse_window(_window(pose=False))
    assert w.head_pose is None, (
        "a missing pose must not be zero-filled; zeros are a valid head pose and would "
        "make 'no tracking' look like 'perfectly still'"
    )


def test_missing_blendshapes_is_rejected() -> None:
    with pytest.raises(BadWindow, match="blendshapes"):
        parse_window({"fps": 24})


# --- the measurement uses the research code --------------------------------


def test_every_marker_is_reported_with_its_evidence() -> None:
    from seam.features import markers as VM

    out = measure(parse_window(_window()))
    assert set(out["markers"]) == set(VM.MARKERS)
    for m in out["markers"].values():
        assert {"magnitude", "peak", "zero_fraction", "iqr", "present_in_window", "unit"} <= set(m)
        assert m["unit"] in ("mads", "rad", "clip_range")
        assert 0.0 <= m["zero_fraction"] <= 1.0


def test_measure_reports_its_own_provenance() -> None:
    """A demo figure and a paper figure must come from one implementation, and the
    response has to say so rather than leaving the UI to assume it."""
    out = measure(parse_window(_window()))
    prov = out["provenance"]
    assert "client-side" in prov["input"]
    assert "never sees video" in prov["input"]
    assert "duration-free" in prov["aggregation"]
    assert out["caveats"], "the degenerate-marker caveat must travel with the numbers"


def test_absent_pose_produces_zero_head_markers_not_a_crash() -> None:
    out = measure(parse_window(_window(pose=False)))
    assert out["markers"]["head_shake"]["magnitude"] == 0.0
    assert out["markers"]["head_nod"]["magnitude"] == 0.0


# --- unsupported things are withheld ---------------------------------------


def test_affect_prediction_is_withheld_with_a_reason() -> None:
    out = analyse(_window())
    assert out["affect"]["supported"] is False
    assert "0.497" in out["affect"]["reason"], "the reason must cite the measurement"
    assert "m4-factorizer-002" in out["affect"]["reason"]


def test_recognition_is_withheld_with_a_reason() -> None:
    out = analyse(_window())
    assert out["recognition"]["supported"] is False
    assert "Sign video filename" in out["recognition"]["reason"]


def test_linguistic_context_needs_an_utterance_id() -> None:
    out = analyse(_window())
    assert out["linguistic"] is None, "no id means no annotation, not a guess"


def test_linguistic_context_uses_real_annotation() -> None:
    pytest.importorskip("pandas")
    out = analyse({**_window(), "utterance_id": "24363254"})
    ling = out["linguistic"]
    if not ling.get("available"):
        pytest.skip("ASLLRP gloss map not fetched")
    assert ling["gloss"], "the gloss is human-authored annotation"
    assert set(ling["provenance"]) >= {"interrogative", "negation"}
    assert "heuristic" in ling["provenance"]["topicalization"]


def test_unknown_utterance_is_reported_not_guessed() -> None:
    out = analyse({**_window(), "utterance_id": "99999999"})
    ling = out["linguistic"]
    assert ling["available"] is False
    assert "not in the ASLLRP gloss map" in ling["reason"]


# --- the HTTP surface ------------------------------------------------------


def test_health_states_the_request_contract() -> None:
    """The contract is what the browser client codes against, so it is asserted here
    rather than left implicit in two languages that can drift apart."""
    from seam.serve.app import build_app

    app = build_app()
    routes = {getattr(r, "path", None) for r in app.routes}
    assert {"/", "/api/health", "/api/analyse", "/api/coverage"} <= routes


def test_page_assets_resolve_from_the_package_not_the_cwd(tmp_path, monkeypatch) -> None:
    """The first implementation pointed at src/ and served a 44-byte 404 page.

    A demo that starts successfully and shows nothing is worse than one that crashes,
    so the asset path is asserted to contain a real HTML file.
    """
    import seam.serve.app as mod

    monkeypatch.chdir(tmp_path)  # starting from another directory must still work
    app = mod.build_app()
    page = mod.Path(mod.__file__).resolve().parents[1] / "web" / "index.html"
    assert page.is_file(), f"demo page missing at {page}"
    assert "<!doctype html>" in page.read_text(encoding="utf-8").lower()
    # The client-side MediaPipe import and the endpoint it posts to are the two halves
    # of the same contract; if either is renamed the demo breaks silently.
    html = page.read_text(encoding="utf-8")
    assert "api/analyse" in html
    assert "tasks-vision" in html
    assert app is not None
