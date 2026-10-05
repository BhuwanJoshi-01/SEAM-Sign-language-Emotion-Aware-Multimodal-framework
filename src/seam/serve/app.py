"""The live demo server (M7).

Deliberately **does not receive video**. The browser runs MediaPipe Tasks in WASM and
posts the *numbers* — a window of blendshape coefficients and head pose — to this
process. That choice has three consequences worth stating, because it is the plan's
requirement and not an implementation detail:

* **No video leaves the machine.** The server never sees a face, so there is nothing to
  store, log, or leak, and the demo can be run on footage nobody has consented to share.
* **It runs anywhere.** No GPU, no MediaPipe runtime, no model download on the server.
* **But it also means this process cannot verify what the client measured.** The
  landmarks arrive as numbers from a browser tab, so the marker magnitudes here are only
  as trustworthy as that tab. Every response therefore carries the provenance of its
  inputs, and the UI shows it, because a demo that silently trusts its own frontend
  would be reporting numbers it never checked.

**What this endpoint does and does not claim.** It computes real marker magnitudes with
the same code the research uses (`seam.features.markers`), and it attaches the
linguistic context for the utterance when the client supplies an ASLLRP utterance id —
that part is a real annotation, not a model output. It does **not** return affect
predictions, because M4 measured that model's affect head at balanced accuracy 0.481 to
0.507 over three seeds against a 0.5 reference: it does not learn, so shipping its output
in a demo would be presenting a number that carries no information. The reason is
returned in the payload so the UI can say so rather than showing an empty panel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from seam.features import markers as VM
from seam.features import syntactic as SY
from seam.logging import get

log = get("serve")

#: What the client must send. Fixed, so a malformed request is rejected rather than
#: producing a plausible-looking response from misaligned arrays.
NM_DIM = 52
POSE_DIM = 9

#: Minimum window the marker machinery needs. Below this the robust spread is
#: meaningless and the magnitude is noise.
MIN_FRAMES = 8

#: Balanced accuracy of the M4 affect head, lowest and highest of three seeds, from
#: ``artifacts/m4/factorizer_multilabel_ablation.json`` (variant ``full``). Quoted in the
#: refusal below; ``tests/test_serve.py`` fails if it drifts from that artifact. The
#: figure quoted here until 2026-10-05 was 0.497, from a run made before the yaw fix.
M4_AFFECT_BALANCED_ACCURACY = (0.481, 0.507)

#: Mean WER of the M5a gloss recogniser and of its most-frequent-gloss baseline, from
#: ``artifacts/m5a/recogniser.json``; ``tests/test_serve.py`` fails if they drift from it.
M5A_WER_AND_BASELINE = (0.920, 0.920)

#: The honest answer when a caller asks for a prediction this build cannot support.
NOT_SUPPORTED = {
    "supported": False,
    "reason": (
        "M4's factorized encoder was measured at "
        f"{M4_AFFECT_BALANCED_ACCURACY[0]:.3f} to {M4_AFFECT_BALANCED_ACCURACY[1]:.3f} "
        "balanced accuracy on affect over three seeds, against a 0.5 reference, so it "
        "does not learn the task. Its output is withheld rather than displayed. See "
        "plan.md (M4, refuted) and the 2026-10-05 entry of paper/EXPERIMENT_LOG.md."
    ),
    "instead": "marker magnitudes, which are measured signal, and the linguistic annotation",
}


class BadWindow(ValueError):
    """The client sent a window this process cannot measure from."""


@dataclass(slots=True)
class Window:
    """One validated window of client-side measurements."""

    blendshapes: np.ndarray  # (T, 52)
    #: The raw rotation transform as sent, (T, 4, 4) or (T, 3, 3), or ``None`` when the
    #: client sent no head tracking. Kept as a matrix rather than converted to Euler
    #: here, because :func:`seam.features.markers.signals` does that conversion and
    #: doing it twice is how the two paths drift apart.
    head_pose: np.ndarray | None
    fps: float
    utterance_id: str = ""

    @property
    def n_frames(self) -> int:
        return len(self.blendshapes)


def parse_window(payload: dict[str, Any]) -> Window:
    """Validate a client payload into a :class:`Window`.

    Validation is explicit and rejects rather than repairs. A demo that pads a short
    array or reshapes a wrong one will produce a smooth, confident, meaningless number,
    and the whole point of this endpoint is that its numbers mean something.
    """
    if "blendshapes" not in payload:
        raise BadWindow("missing 'blendshapes'")
    try:
        bs = np.asarray(payload["blendshapes"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise BadWindow(f"blendshapes is not a rectangular array: {exc}") from exc
    if bs.ndim != 2 or bs.shape[1] != NM_DIM:
        raise BadWindow(
            f"blendshapes must be (T, {NM_DIM}), got {bs.shape}. MediaPipe emits 52 "
            "coefficients starting with _neutral; if you sent 51 the _neutral column is "
            "missing and every marker threshold is off by one."
        )
    if len(bs) < MIN_FRAMES:
        raise BadWindow(f"need at least {MIN_FRAMES} frames to measure a marker, got {len(bs)}")
    if not np.isfinite(bs).all():
        raise BadWindow("blendshapes contains non-finite values")

    pose = payload.get("head_pose")
    if pose is None:
        # Absent pose is legitimate - a face with no head tracking - and it changes what
        # can be measured, so it stays ``None`` rather than being zero-filled: zeros are
        # a *plausible* head pose and would make "no tracking" indistinguishable from
        # "perfectly still".
        rot = None
    else:
        rot = np.asarray(pose, dtype=np.float64)
        if rot.ndim != 3 or rot.shape[0] != len(bs):
            raise BadWindow(
                f"head_pose must be ({len(bs)}, 3or4, 3or4) to match blendshapes, "
                f"got {rot.shape}; the two signals must cover the same window or they "
                "are not comparable"
            )
        if not np.isfinite(rot).all():
            raise BadWindow("head_pose contains non-finite values")
        rot = rot[:, :4, :4] if rot.shape[1] >= 4 else rot

    try:
        fps = float(payload.get("fps") or 0.0)
    except (TypeError, ValueError) as exc:
        raise BadWindow(f"fps is not a number: {exc}") from exc
    if not 0 < fps <= 240:
        raise BadWindow(f"fps must be in (0, 240], got {fps}")

    return Window(
        blendshapes=bs,
        head_pose=rot,
        fps=fps,
        utterance_id=str(payload.get("utterance_id") or ""),
    )


def measure(window: Window) -> dict[str, Any]:
    """Marker magnitudes and provenance for one window.

    Uses the same :mod:`seam.features.markers` code the experiments used, so a demo
    figure and a paper figure come from one implementation. The degeneracy caveat is
    carried in the response rather than left for the UI to remember: four of the six
    markers fire on most clips, so their magnitudes vary little between windows.
    """
    signals = VM.signals(window.blendshapes, rotation=window.head_pose, fps=window.fps)
    magnitude = VM.clip_magnitude(signals, window.fps, reduce="mean")
    presence = VM.clip_presence(signals, window.fps)
    ev = VM.clip_evidence(signals)

    per_marker: dict[str, dict[str, Any]] = {}
    for name in VM.MARKERS:
        x = ev[name]
        zero_frac = float(np.mean(np.isclose(x, 0.0))) if len(x) else float("nan")
        iqr = float(np.percentile(x, 75) - np.percentile(x, 25)) if len(x) else float("nan")
        per_marker[name] = {
            "magnitude": round(float(magnitude.get(name, 0.0)), 4),
            "peak": round(float(x.max()) if len(x) else 0.0, 4),
            # Measured on *this window*. Never read as the marker being blind - see
            # `corpus` below, which carries that measurement and where it came from.
            "zero_fraction": round(zero_frac, 4),
            "iqr": round(float(iqr), 5),
            "present_in_window": bool(presence[name]["present"]),
            "unit": presence[name]["peak_unit"],
            # What M3 measured about the marker on the EmoSign 200, sent verbatim so the
            # UI reports a measurement instead of re-deriving one from the window in front
            # of it.
            "corpus": dict(VM.MEASURED_DEGENERACY.get(name, {})),
        }

    return {
        "n_frames": window.n_frames,
        "fps": round(window.fps, 2),
        "markers": per_marker,
        "shared_subspace_hint": None,
        "provenance": {
            "markers": VM.describe()["provenance"],
            "input": (
                "client-side MediaPipe Tasks; this process never sees video and cannot "
                "verify the landmarks it was sent"
            ),
            "aggregation": (
                "clip_magnitude with reduce='mean' (per-frame mean, duration-free); an "
                "integral would inherit clip length, which M3 measured at r_pb=+0.68 "
                "against syntactic labels"
            ),
        },
        "caveats": [
            "Three of the six markers fire on 79-95% of EmoSign clips, so their "
            "magnitudes barely separate windows; only brow_furrow is usable (M3, "
            "EXPERIMENT_LOG entry 27)",
            "Both head markers are blind: head_shake is zero on 90% of EmoSign clips and "
            "head_nod on 83%. No conclusion is available from either, and their "
            "non-separation from a label is not evidence about the label",
            "head_shake was re-measured after a rotation-decomposition bug was fixed and "
            "its previous value was wrong; a published correlation attributed to this "
            "channel has been withdrawn",
            "mouth_positive is the control and is degenerate at 0.905 prevalence, which "
            "is itself a finding rather than a pass",
            "A magnitude here is measured signal, not a validated linguistic label, and "
            "this process cannot verify the landmarks the tab sent it",
        ],
    }


def linguistic_context(utterance_id: str) -> dict[str, Any] | None:
    """The linguistic annotation for an utterance, if it is one we have.

    This is real human annotation - the ASLLRP gloss the Deaf annotators worked from -
    not a model output, and it is the one part of the demo that carries genuine
    linguistic information. Requires the client to supply an utterance id, because a
    live signer's utterance is not identifiable from pixels.
    """
    if not utterance_id:
        return None
    try:
        gloss_map = SY.load_gloss_map()
    except FileNotFoundError:
        return {
            "utterance_id": utterance_id,
            "available": False,
            "reason": "ASLLRP gloss map not fetched; run `seam data fetch asllrp_utterance_map`",
        }
    if utterance_id not in gloss_map:
        return {
            "utterance_id": utterance_id,
            "available": False,
            "reason": f"utterance {utterance_id} is not in the ASLLRP gloss map",
        }
    syn = SY.classify(utterance_id, gloss_map)
    return {
        "utterance_id": utterance_id,
        "available": True,
        "gloss": syn.gloss,
        "n_tokens": syn.n_tokens,
        "markers_present": syn.present(),
        "provenance": syn.provenance(),
        "note": (
            "Human-authored ASLLRP annotation via lexical rules. Topicalization is an "
            "inference capped at 0.5 confidence because the gloss is a flat token "
            "sequence with no constituent structure."
        ),
    }


def analyse(payload: dict[str, Any]) -> dict[str, Any]:
    """The single analysis call behind the demo."""
    t0 = time.perf_counter()
    window = parse_window(payload)
    out = measure(window)
    out["linguistic"] = linguistic_context(window.utterance_id)
    out["affect"] = dict(NOT_SUPPORTED)
    out["recognition"] = {
        "supported": False,
        "reason": (
            "M5a's gloss recogniser does not beat always predicting the most frequent "
            f"gloss: WER {M5A_WER_AND_BASELINE[0]:.3f} against a baseline of "
            f"{M5A_WER_AND_BASELINE[1]:.3f}, signer-disjoint, on 200 utterances with about "
            "three tokens per gloss. Its output is withheld rather than displayed. See the "
            "2026-10-05 entries of paper/EXPERIMENT_LOG.md."
        ),
    }
    out["server_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    out["version"] = _version()
    return out


def _version() -> str:
    from seam import __version__

    return __version__


def _serve_page(web: Path, name: str) -> Any:
    """Serve a hand-written page off disk with caching disabled.

    `no-store` is not optional here. The UI is edited by hand and read straight from disk, so
    a stale browser copy is the default outcome without it — and during the M7 study a cached
    page would silently show raters an older instrument than the one being recorded against,
    which is worse than the extra request.
    """
    from fastapi.responses import HTMLResponse

    page = web / name
    if not page.is_file():
        return HTMLResponse(f"<h1>SEAM</h1><p>web/{name} not found</p>", 404)
    return HTMLResponse(
        page.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"},
    )


def build_app() -> Any:
    """The FastAPI application."""
    import json

    from fastapi import FastAPI
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    app = FastAPI(
        title="SEAM live demo",
        version=_version(),
        description=(
            "Sign-language non-manual analysis. Video is processed in the browser; this "
            "process receives landmark numbers only."
        ),
    )
    # __file__ is src/seam/serve/app.py, so the packaged web assets are one level up
    # and then back into the package. Resolved from __file__ rather than the CWD so the
    # demo serves the same page whichever directory it starts from - the earlier
    # parents[2] pointed at src/ and silently served a 44-byte 404 page.
    web = Path(__file__).resolve().parents[1] / "web"
    # Rendered demo artefacts: `make demo-avatar` writes manifest.json + .glb here.
    demo_dir = Path(__file__).resolve().parents[3] / "artifacts" / "m7a" / "demo"

    @app.get("/api/health")
    def health() -> dict[str, Any]:

        return {
            "ok": True,
            "version": _version(),
            "expects": {
                "blendshapes": [None, NM_DIM],
                "min_frames": MIN_FRAMES,
                "fps": "(0, 240]",
            },
        }

    @app.post("/api/analyse")
    def analyse_endpoint(payload: dict[str, Any]) -> Any:
        try:
            return analyse(payload)
        except BadWindow as exc:
            # 400, not 500: the client sent something unmeasurable, and a 500 would
            # suggest the server broke rather than the request being wrong.
            return JSONResponse(
                status_code=400,
                content={"error": str(exc), "hint": "see /api/health for the contract"},
            )

    # The front page is the standalone demo: every model and every number is computed in
    # the browser, so it needs no server at all. It lives in docs/ because that is what
    # GitHub Pages serves, and this route hands out that same file - the hosted page and
    # the local one cannot drift apart, because there is only one.
    site = Path(__file__).resolve().parents[3] / "docs"

    @app.get("/", response_class=HTMLResponse)
    def index() -> Any:
        """Real-time face, hand and body tracking with the marker read-out, client-side."""
        return _serve_page(site, "index.html")

    @app.get("/live")
    def live_page() -> Any:
        """The front page's address before it became the front page; kept so links work."""
        from fastapi.responses import RedirectResponse

        return RedirectResponse("/", status_code=308)

    @app.get("/server", response_class=HTMLResponse)
    def server_demo() -> Any:
        """The earlier demo, which posts landmark numbers to /api/analyse for the read-out."""
        return _serve_page(web, "index.html")

    @app.get("/figures/{name}")
    def figure(name: str) -> Any:
        """The report figures the standalone page shows, generated from the artifacts."""
        root = (site / "figures").resolve()
        target = (root / name).resolve()
        if not str(target).startswith(str(root) + "/"):
            return JSONResponse({"error": "path escapes figures/"}, status_code=400)
        if not target.is_file() or target.suffix != ".png":
            return JSONResponse({"error": f"{name} not found"}, status_code=404)
        return FileResponse(target, media_type="image/png")

    @app.get("/avatar", response_class=HTMLResponse)
    def avatar_page() -> Any:
        """The avatar product page: a real skinned GLB playing a real animation clip."""
        return _serve_page(web, "avatar.html")

    @app.get("/api/demo/manifest")
    def demo_manifest() -> Any:
        """Manifest for the rendered demo clips.

        Served from disk rather than recomputed, so the page shows exactly what
        `make demo-avatar` produced. A missing manifest returns an empty clip list with a
        200 and a `reason`, because the page already handles "run make demo-avatar" and a
        500 here would replace that message with a stack trace.
        """
        mf = demo_dir / "manifest.json"
        if not mf.is_file():
            return {
                "clips": [],
                "render": {},
                "reason": "no manifest; run `make demo-avatar`",
            }
        return json.loads(mf.read_text(encoding="utf-8"))

    @app.get("/routes", response_class=HTMLResponse)
    def routes_page() -> Any:
        """Every route this server answers, probed live, linked from every page.

        The general escape hatch for discoverability. A route that is not reachable from
        anywhere is a route nobody checks, which is how `/avatar` sat broken behind a
        working server without anyone noticing.
        """
        return _serve_page(web, "routes.html")

    @app.get("/static/vendor/{path:path}")
    def vendored(path: str) -> Any:
        """Serve vendored front-end libraries (three.js and its addons).

        Vendored rather than pulled from a CDN on purpose: a page that needs the network to
        render is not a product, and a CDN dependency cannot be audited or pinned by the
        repo. Constrained to `vendor/` for the same reason as the demo route.
        """
        root = (web / "vendor").resolve()
        target = (root / path).resolve()
        if not str(target).startswith(str(root) + "/"):
            return JSONResponse({"error": "path escapes vendor/"}, status_code=400)
        if not target.is_file():
            return JSONResponse({"error": f"{path} not found"}, status_code=404)
        return FileResponse(
            target, media_type="text/javascript", headers={"Cache-Control": "public, max-age=3600"}
        )

    @app.get("/static/{path:path}")
    def web_asset(path: str) -> Any:
        """Serve a page's own shared asset (currently `nav.js`).

        Declared *after* `/static/vendor/{path}` so the more specific pattern wins for
        vendored files. Same containment rule: the path arrives from the URL, so the check
        is on the resolved path rather than on the route shape.
        """
        root = web.resolve()
        target = (root / path).resolve()
        if not str(target).startswith(str(root) + "/"):
            return JSONResponse({"error": "path escapes the web root"}, status_code=400)
        if not target.is_file():
            return JSONResponse({"error": f"{path} not found"}, status_code=404)
        types = {
            ".js": "text/javascript",
            ".mjs": "text/javascript",
            ".css": "text/css",
            ".html": "text/html",
            ".json": "application/json",
            ".svg": "image/svg+xml",
        }
        return FileResponse(
            target,
            media_type=types.get(target.suffix),
            # tokens.css carries both themes in one file, so a stale copy means stale colours
            # rather than stale text. Revalidate instead of serving it for a day.
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/cue_expectations")
    def cue_expectations() -> Any:
        """The cue-to-feature map, as JSON, for the coverage page.

        JSON rather than server-rendered HTML so the page and the code cannot disagree: both
        read `seam.features.cues`, which is the single source of truth.
        """
        from seam.features import cues as C

        return {"expectations": C.expectations(), "unmapped": C.unmapped_cues()}

    @app.get("/api/demo/{name}")
    def demo_asset(name: str) -> Any:
        """Serve one rendered artefact (`.glb`) from the demo directory.

        Resolved and then checked to be inside `demo_dir`, because `name` arrives from the
        URL. `{name:path}` would also allow `../`, so containment is enforced on the
        resolved path rather than trusted from the pattern.
        """
        target = (demo_dir / name).resolve()
        if not str(target).startswith(str(demo_dir.resolve()) + "/"):
            return JSONResponse({"error": "path escapes the demo directory"}, status_code=400)
        if not target.is_file():
            return JSONResponse({"error": f"{name} not found"}, status_code=404)
        return FileResponse(
            target,
            media_type="model/gltf-binary" if target.suffix == ".glb" else None,
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/coverage", response_class=HTMLResponse)
    def coverage() -> Any:
        """What the demo does and does not claim, served from the same text as the UI."""
        return _serve_page(web, "coverage.html")

    log.info("app built")
    return app
