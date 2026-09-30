#!/usr/bin/env python
"""Check whether the ASLLRP DAI 2 portal is reachable, and log it.

Step 3 of `guide.md` is blocked on a third-party server, and the only way to know it has
come back is to look. This does the looking and appends a dated row to
`artifacts/m7a/dai_portal_status.json` so the wait has a record rather than being
remembered.

Why a log rather than a single check: the portal has been down across two sessions now.
When it returns, the useful question is not "is it up" but "was it up on Tuesday", and
that needs history. It also means the eventual `BLOCKED - EXTERNAL` status can be closed
with evidence rather than optimism.

**Diagnostic value.** A bare HTTP code is not enough to help the maintainers. The earlier
finding is that the host answers while every application path times out, which points at
the application rather than the machine - so this distinguishes the two cases explicitly.

Usage:
    scripts/check_dai_portal.py            # one check, prints a table, appends a row
    scripts/check_dai_portal.py --watch    # poll every 60s until a path answers
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "m7a" / "dai_portal_status.json"

APP_PATHS = (
    "https://dai.cs.rutgers.edu/dai/s/dai",
    "https://dai.cs.rutgers.edu/dai/s/continuoussigndownload",
    "https://dai.cs.rutgers.edu/dai/s/runningstats",
)
HOST_PATHS = ("https://dai.cs.rutgers.edu/",)
CONTROL_PATHS = ("https://www.bu.edu/asllrp/",)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Report redirects instead of following them.

    Not cosmetic, and this was a real bug in an earlier version of this script.
    `dai.cs.rutgers.edu/` answers with a 302 into the application that is hanging, so a
    redirect-following client follows it, waits out the timeout, and then reports the
    *host* as dead when the host is alive. That inverts the APP_DOWN / HOST_DOWN verdict,
    which is exactly the distinction the outage report to the maintainers rests on -
    a tool that misreports the diagnosis is worse than no tool.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def probe(url: str, timeout: float) -> dict[str, object]:
    """One request. Never raises: an unreachable host is a result, not an error."""
    started = time.monotonic()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with _OPENER.open(req, timeout=timeout) as r:
            body = r.read(4096)
            return {
                "url": url,
                "status": r.status,
                "reachable": True,
                "bytes": len(body),
                "seconds": round(time.monotonic() - started, 2),
                "server_error": None,
            }
    except urllib.error.HTTPError as e:
        # A 3xx here means the server answered, which is the entire point of the host
        # probe. With redirects disabled, urllib raises on 3xx, so it arrives here.
        return {
            "url": url,
            "status": e.code,
            "reachable": True,
            "bytes": 0,
            "seconds": round(time.monotonic() - started, 2),
            "server_error": None,
            "note": "redirect not followed; the server answered" if 300 <= e.code < 400 else None,
        }
    except Exception as e:
        return {
            "url": url,
            "status": None,
            "reachable": False,
            "bytes": 0,
            "seconds": round(time.monotonic() - started, 2),
            "server_error": f"{type(e).__name__}: {e}"[:300],
        }


def check(timeout: float) -> dict[str, object]:
    results = {
        "app": [probe(u, timeout) for u in APP_PATHS],
        "host": [probe(u, timeout) for u in HOST_PATHS],
        "control": [probe(u, timeout) for u in CONTROL_PATHS],
    }
    app_up = any(r["reachable"] and r["status"] == 200 for r in results["app"])  # type: ignore[index]
    host_up = any(r["reachable"] for r in results["host"])  # type: ignore[index]
    # The distinction that matters for the maintainers: is the machine up and the
    # application down, or is the whole host gone?
    if app_up:
        verdict, detail = (
            "PORTAL_UP",
            "a /dai/s/ path answered 200; proceed with guide.md step 3 part A",
        )
    elif host_up:
        verdict, detail = (
            "APP_DOWN",
            "the host answered but every /dai/s/ path failed - the application behind "
            "Apache is down, not the machine. Keep waiting; report it if not already known.",
        )
    else:
        verdict, detail = "HOST_DOWN", "no response from the host at all"
    return {
        "checked_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "verdict": verdict,
        "detail": detail,
        **results,
    }


def record(row: dict[str, object]) -> None:
    log: list[dict[str, object]] = []
    if OUT.exists():
        try:
            log = json.loads(OUT.read_text()).get("checks", [])
        except (json.JSONDecodeError, OSError):
            log = []
    log.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "note": (
                    "Dated reachability history for the ASLLRP DAI 2 portal, which gates "
                    "guide.md step 3. Appended by scripts/check_dai_portal.py."
                ),
                "checks": log[-200:],
            },
            indent=2,
        )
        + "\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--interval", type=int, default=60, help="--watch poll seconds")
    ap.add_argument("--watch", action="store_true", help="poll until the portal answers")
    ap.add_argument("--no-log", action="store_true", help="do not append to the history")
    args = ap.parse_args()

    while True:
        row = check(args.timeout)
        if not args.no_log:
            record(row)
        print(f"\n{row['checked_at_utc']}  {row['verdict']}")
        print(f"  {row['detail']}")
        for group in ("host", "app", "control"):
            for r in row[group]:  # type: ignore[index]
                st = r["status"] if r["status"] is not None else "no response"
                print(f"    [{group:7}] {r['url']!s:58} {st:>14}  {r['seconds']:>6.2f}s")
                if r["server_error"]:
                    print(f"              -> {str(r['server_error'])[:120]}")
        print(f"\nlogged to {OUT.relative_to(REPO)}")

        if not args.watch or row["verdict"] == "PORTAL_UP":
            return 0 if row["verdict"] == "PORTAL_UP" else 1
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nstopped")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
