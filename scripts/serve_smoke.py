"""Boot the demo server and check it over HTTP, so `make serve-check` is a gate.

A demo that starts and then serves a 44-byte 404 page is worse than one that
refuses to start - it looks like it works. The first version of this server had
exactly that bug: the web asset path resolved to `src/` instead of the package, the
app built without complaint, and the page was empty.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PORT = 8079
NM_DIM = 52


def wait_up(port: int, timeout: float = 40.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2).read()
            return True
        except Exception:
            time.sleep(0.4)
    return False


def main() -> int:
    proc = subprocess.Popen(
        [sys.executable, "-m", "seam.cli", "serve", "--port", str(PORT), "--log-level", "warning"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(ROOT),
    )
    failures: list[str] = []
    try:
        if not wait_up(PORT):
            print("FAIL: server did not come up")
            return 1

        page = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=15).read()
        if b"<!doctype html>" not in page[:64].lower():
            failures.append("index did not serve an HTML document")
        if b"tasks-vision" not in page:
            failures.append("page does not load client-side MediaPipe")
        if b"api/analyse" not in page:
            failures.append("page does not post to /api/analyse")

        # ── every page, and every page reachable from every page ─────────────
        # `/avatar` shipped unlinked from `/`, absent from the banner, and absent from
        # this smoke test, and stayed broken for its whole life. A route nothing points at
        # is a route nobody checks, so both properties are asserted here.
        PAGES = {
            "/": (b"tasks-vision", b"api/analyse"),
            "/avatar": (b"/api/demo/manifest", b"importmap"),
            "/api/coverage": (b"api/cue_expectations", b"is not claimed"),
            "/routes": (b"/static/nav.js", b"/api/demo/manifest"),
        }
        for route, needles in PAGES.items():
            try:
                body = urllib.request.urlopen(f"http://127.0.0.1:{PORT}{route}", timeout=15).read()
            except urllib.error.HTTPError as e:
                failures.append(f"{route} returned HTTP {e.code}")
                continue
            print(f"GET {route:17.17} {len(body)} bytes")
            if b"<!doctype html>" not in body[:64].lower():
                failures.append(f"{route} did not serve an HTML document")
                continue
            for needle in needles:
                if needle not in body:
                    failures.append(f"{route} does not reference {needle!r}")
            if b"/static/nav.js" not in body:
                failures.append(f"{route} does not load the shared nav module")
            # Theming. A page that skips the token sheet or the no-flash boot snippet is dark
            # only, and it flashes the wrong theme on every load before the module arrives.
            if b"/static/tokens.css" not in body:
                failures.append(f"{route} does not load tokens.css, so it cannot be themed")
            if b"seam.theme" not in body:
                failures.append(f"{route} has no pre-paint theme boot snippet; it will flash")
            if b"data-seam-header" not in body:
                failures.append(f"{route} does not mount the shared header")
            # The incumbent pages shipped a 3px coloured border-left on every callout, which
            # is the default gesture of every alert box ever built. Keep it out.
            if b"border-left:3px" in body or b"border-left: 3px" in body:
                failures.append(f"{route} uses a coloured border-left on a callout")
            # And monospace as a costume: it belongs on code and raw values, not on body text.
            if b"font:14px/1.5 ui-monospace" in body or b"ui-monospace,SFMono" in body:
                failures.append(f"{route} sets monospace on body text as a costume")

        for asset in ("/static/tokens.css", "/static/theme.js", "/static/nav.js"):
            try:
                body = urllib.request.urlopen(f"http://127.0.0.1:{PORT}{asset}", timeout=15).read()
            except urllib.error.HTTPError as e:
                failures.append(f"{asset} -> HTTP {e.code}")
                continue
            print(f"GET {asset:17.17} {len(body)} bytes")
            if asset.endswith("tokens.css"):
                # Both themes must be real blocks, and the system preference must be honoured
                # when the reader has chosen nothing.
                for needle, why in (
                    (b'[data-theme="light"]', "no light theme block"),
                    (b'[data-theme="dark"]', "no dark theme block"),
                    (b"prefers-color-scheme: light", "no system-preference fallback"),
                    (b"::selection", "text selection is left at the browser default"),
                    (b"focus-visible", "no focus ring"),
                    (b"tabular-nums", "table numerals are not tabular"),
                ):
                    if needle not in body:
                        failures.append(f"tokens.css: {why}")
            if asset.endswith("theme.js") and b"seam.theme" not in body:
                failures.append("theme.js does not persist the reader's choice")

        # The nav module is the single source of truth for cross-page links, so if it is
        # missing a page then that page is genuinely unreachable from the UI.
        nav = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/static/nav.js", timeout=15).read()
        for target in PAGES:
            if f"'{target}'" not in nav.decode():
                failures.append(f"nav.js does not link {target}; that page is unreachable")

        # ── /avatar specifics ────────────────────────────────────────────────
        av = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/avatar", timeout=15).read()
        if b"cdn.jsdelivr" in av or b"unpkg.com" in av:
            failures.append(
                "/avatar loads three.js from a CDN; a product page that needs the network "
                "to render is not a product page"
            )

        # Vendored three.js must actually be reachable, or the page is a red screen.
        for asset in (
            "/static/vendor/three/three.module.js",
            "/static/vendor/three/examples/jsm/controls/OrbitControls.js",
            "/static/vendor/three/examples/jsm/loaders/GLTFLoader.js",
            "/static/vendor/three/examples/jsm/utils/BufferGeometryUtils.js",
        ):
            try:
                body = urllib.request.urlopen(f"http://127.0.0.1:{PORT}{asset}", timeout=30).read()
            except urllib.error.HTTPError as e:
                failures.append(f"{asset} -> HTTP {e.code}; /avatar cannot load three.js")
                continue
            if len(body) < 1000:
                failures.append(f"{asset} served {len(body)} bytes; that is not a library")
            else:
                print(f"GET {asset:52.52} {len(body)} bytes")

        # Path containment on every file route. `{path:path}` patterns match `../`, so the
        # guard has to be on the resolved path rather than trusted from the route shape.
        for route in ("/static/vendor/", "/static/", "/api/demo/"):
            url = f"http://127.0.0.1:{PORT}{route}..%2f..%2f..%2fetc%2fpasswd"
            try:
                body = urllib.request.urlopen(url, timeout=10).read()
                if b"root:" in body:
                    failures.append(f"{route} served /etc/passwd")
            except urllib.error.HTTPError as e:
                if e.code not in (400, 404):
                    failures.append(f"{route} traversal gave HTTP {e.code}, expected 400/404")

        # The manifest endpoint must degrade to a 200 with a reason, never a 500: the page
        # renders that reason, and a traceback would replace it.
        mf = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/demo/manifest", timeout=15).read()
        )
        if not isinstance(mf.get("clips"), list):
            failures.append(f"/api/demo/manifest has no clips list: {list(mf)}")
        elif mf["clips"] and not any(c.get("arm_a_smplerx", {}).get("glb") for c in mf["clips"]):
            failures.append("manifest lists clips but none carry a .glb")
        else:
            print(f"GET /api/demo/manifest      {len(mf['clips'])} clip(s)")

        ce = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/api/cue_expectations", timeout=15
            ).read()
        )
        if not ce.get("expectations"):
            failures.append("/api/cue_expectations returned no cues; the coverage page is empty")
        else:
            print(
                f"GET /api/cue_expectations   {len(ce['expectations'])} cues, "
                f"{len(ce.get('unmapped', []))} without a feature"
            )

        health = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/health", timeout=15).read()
        )
        got = health.get("expects", {}).get("blendshapes")
        if got != [None, NM_DIM]:
            failures.append(
                f"health contract mismatch: NM_DIM={NM_DIM}, got {got!r}; body={health!r}"
            )

        rng = np.random.default_rng(0)
        payload = {
            "blendshapes": (rng.random((30, NM_DIM)) * 0.2).tolist(),
            "head_pose": np.tile(np.eye(4).tolist(), (30, 1, 1)).tolist(),
            "fps": 24,
            "utterance_id": "24363254",
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/api/analyse",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
        )
        out = json.loads(urllib.request.urlopen(req, timeout=20).read())
        print(f"POST /api/analyse     {len(out['markers'])} markers, {out['server_ms']} ms")
        if out["affect"]["supported"] or out["recognition"]["supported"]:
            failures.append("demo is advertising predictions this build cannot support")
        if not out["provenance"]["input"]:
            failures.append("response carries no input provenance")

        # A malformed window must be a 400 with an actionable message, not a 500.
        bad = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/api/analyse",
            data=json.dumps({"blendshapes": [[0.1] * (NM_DIM - 1)] * 20, "fps": 24}).encode(),
            headers={"content-type": "application/json"},
        )
        try:
            urllib.request.urlopen(bad, timeout=15)
            failures.append("a 51-coefficient window was accepted")
        except urllib.error.HTTPError as e:
            if e.code != 400:
                failures.append(f"malformed window gave HTTP {e.code}, expected 400")
            else:
                print("malformed window      400 with a message")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("serve smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
