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
    """Nothing the page shows may depend on the API. It may look for one optional extra.

    The single request it makes is a relative probe for the local avatar viewer, which
    fails quietly on GitHub Pages and only turns a link on when it succeeds.
    """
    html = PAGE.read_text(encoding="utf-8")
    assert "/api/" not in html, "the hosted page has no server to call"
    assert "/static/" not in html
    assert "api/analyse" not in html
    assert html.count("fetch(") == 1 and 'fetch("api/demo/manifest"' in html
    probe = html[html.index('fetch("api/demo/manifest"') :]
    assert ".catch(" in probe.split("</script>")[0], "the probe must fail without a visible error"


def test_the_avatar_is_part_of_this_page_and_explained_where_it_cannot_run() -> None:
    """One site, one design: the 3D viewer is a section of this page, not another page.

    The viewer needs the animated clips, which only the local server has (the body model
    may not be redistributed). So the nav link lands on a section that is real either way:
    rendered frames and the reason on the hosted page, the interactive viewer locally.
    """
    html = PAGE.read_text(encoding="utf-8")
    nav = re.search(r'<nav aria-label="Sections">(.*?)</nav>', html, re.S)
    assert nav and 'href="#avatar"' in nav.group(1) and "3D avatar" in nav.group(1)
    assert '<section class="block" id="avatar">' in html
    assert 'id="avStill" src="figures/avatar_strip.png"' in html, "stills for the hosted page"
    assert '<canvas id="av3d" hidden' in html, "the viewer starts hidden"
    assert 'id="avControls" hidden' in html and 'id="avClipRow" hidden' in html
    assert "may not be redistributed" in html
    assert 'href="avatar"' not in html and "avatar.html" not in html, "no link to a second page"


def test_three_js_is_named_beside_the_page_and_only_loaded_on_demand() -> None:
    """The import map must not point at a CDN or at the server's `/static/`.

    The page is one file that is also hosted where there is no server. Its import map
    names `./vendor/three/`, which the local server provides, and three.js is imported
    inside `startAvatar`, which only runs once the manifest request has succeeded.
    """
    html = PAGE.read_text(encoding="utf-8")
    found = re.search(r'<script type="importmap">(.*?)</script>', html, re.S)
    assert found
    import json

    imports = json.loads(found.group(1))["imports"]
    assert imports == {
        "three": "./vendor/three/three.module.js",
        "three/addons/": "./vendor/three/examples/jsm/",
    }
    assert html.index('type="importmap"') < html.index('<script type="module">')
    viewer = html[html.index("async function startAvatar") :]
    assert 'await import("three")' in viewer
    before = html[: html.index("async function startAvatar")]
    assert 'import("three' not in before, "three.js must not load on the hosted page"


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


def _validation() -> dict:
    import json

    import pytest

    path = REPO / "artifacts" / "m3" / "marker_validation.json"
    if not path.is_file():
        pytest.skip("marker_validation.json not built")
    return json.loads(path.read_text())["markers"]


def test_head_badges_quote_the_detector_the_page_actually_runs() -> None:
    """Shake and nod are scored by the revised detector, so its numbers go on the badge."""
    html = PAGE.read_text(encoding="utf-8")
    markers = _validation()
    for marker, badge in (("head_shake", "bShake"), ("head_nod", "bNod")):
        by_fold = markers[marker]["revised_page_within_clip_auc_by_fold"]
        vals = [by_fold[f] for f in ("Cory", "Jonathan", "Rachel")]
        text = re.search(rf'id="{badge}">([^<]+)<', html)
        assert text, badge
        assert f"{min(vals):.2f}–{max(vals):.2f}" in text.group(1), (marker, text.group(1))
        assert "validated" not in text.group(1).replace("not validated", "")
    for marker, cell in (("head_shake", "eShake"), ("head_nod", "eNod")):
        by_fold = markers[marker]["revised_page_within_clip_auc_by_fold"]
        text = re.search(rf'id="{cell}">([^<]+)<', html)
        assert text, cell
        quoted = " / ".join(f"{by_fold[f]:.3f}" for f in ("Cory", "Jonathan", "Rachel"))
        assert quoted in text.group(1), (marker, text.group(1))


def test_the_page_runs_the_constants_the_validation_checked() -> None:
    """A threshold typed into the page and one in the Python module would drift apart.

    The page, `seam.features.markers` and the validation artifact must name one detector.
    """
    from seam.features import markers as VM

    html = PAGE.read_text(encoding="utf-8")
    head = re.search(r"const HEAD = \{([^}]*)\}", html)
    assert head
    page = {k.strip(): float(v) for k, v in (kv.split(":") for kv in head.group(1).split(","))}
    th = VM.MarkerThresholds()
    assert (page["fast"], page["slow"], page["hold"]) == (
        VM.HEAD_FAST_S,
        VM.HEAD_SLOW_S,
        VM.HEAD_HOLD_S,
    )
    assert (page["shakeOnDeg"], page["nodOnDeg"]) == (th.shake_rms_deg, th.nod_rms_deg)
    gate = re.search(r"new Oscillation\((\d+(?:\.\d+)?), 1300\)", html)
    assert gate and float(gate.group(1)) == th.swing_deg
    markers = _validation()
    for name in ("head_shake", "head_nod"):
        shipped = markers[name]["shipped_in_page"]
        assert shipped["contrast"] == 0.0, "the page subtracts nothing from the other axis"
        assert (shipped["slow_s"], shipped["hold_s"]) == (page["slow"], page["hold"])


def test_the_pages_head_detector_is_the_python_one_number_for_number() -> None:
    """The validation scored `head_motion`; the page runs JavaScript. They must agree.

    The page's own `BandEnergy` and `Oscillation` classes are cut out of the HTML and run
    in Node on a head angle that turns once, shakes, and rests, and the result is compared
    with the Python functions the validation used.
    """
    import json
    import shutil
    import subprocess

    import numpy as np
    import pytest

    from seam.features import head_motion as HM

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the JavaScript cannot be run here")
    html = PAGE.read_text(encoding="utf-8")

    def cut(start: str) -> str:
        """The class starting at `start`, up to the brace that closes it."""
        begin = html.index(start)
        depth = 0
        for pos in range(html.index("{", begin), len(html)):
            depth += {"{": 1, "}": -1}.get(html[pos], 0)
            if depth == 0:
                return html[begin : pos + 1]
        raise AssertionError(f"unbalanced braces after {start!r}")

    fps = 30.0
    t = np.arange(240) / fps
    angle = np.where(t < 2.0, 0.0, 0.0) + np.where((t >= 1.0) & (t < 1.3), (t - 1.0) * 60, 0.0)
    angle = np.where(t >= 1.3, 18.0, angle)
    angle = angle + np.where((t >= 3.0) & (t < 6.0), 9.0 * np.sin(2 * np.pi * 2.0 * t), 0.0)
    head = re.search(r"const HEAD = \{[^}]*\};", html)
    assert head
    script = "\n".join(
        [
            "const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));",
            head.group(0),
            cut("class BandEnergy"),
            cut("class Oscillation"),
            f"const xs = {json.dumps([round(float(v), 6) for v in angle])};",
            "const e = new BandEnergy(), o = new Oscillation(2, 1300);",
            f"const out = xs.map((x, i) => [e.push(x, i * 1000 / {fps}), o.push(x, i * 1000 / {fps})]);",
            "console.log(JSON.stringify(out));",
        ]
    )
    run = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr[-600:]
    js = np.array(json.loads(run.stdout))
    xs = np.round(angle, 6)
    py_energy = HM.live_band_rms(xs, fps, fast_s=0.08, slow_s=0.4, hold_s=0.4)
    py_swing = HM.reversal_score(xs, fps, min_deg=2.0, window_s=1.3)
    np.testing.assert_allclose(js[:, 0], py_energy, atol=1e-6)
    np.testing.assert_allclose(js[:, 1], py_swing, atol=1e-9)
    # And it behaves: the single turn has energy but no swing, the shake has both.
    turn, shake = slice(30, 60), slice(110, 175)
    assert js[turn, 0].max() > 2.44 and js[turn, 1].max() < 0.66
    assert js[shake, 0].min() > 2.44 and js[shake, 1].min() >= 0.66
