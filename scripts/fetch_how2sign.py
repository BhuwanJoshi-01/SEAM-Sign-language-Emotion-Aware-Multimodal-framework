#!/usr/bin/env python
"""Fetch the How2Sign MediaPipe pose keypoints (31 parquet shards, ~14 GB).

Why a script rather than `seam data fetch`
------------------------------------------
The registry declares only ``README.md`` for this resource: listing 31 shards in
`sources.py` would make the licence/role metadata unreadable, and the shard count is a
property of the upstream repo rather than of this project. So the registry stays declarative
and the bulk transfer lives here, where resume and retry can be handled properly.

Resumable on purpose. At the ~2.7 MB/s this link sustains, a full pass is ~85 minutes, and a
single dropped connection must not throw that away - `.part` files are kept and resumed with
a Range request, and only renamed into place once the byte count matches the server's.

Ungated (`gated=False`), which is what makes M5 reachable at all: ASLLRP English is not in
the mirror, and How2Sign's published captions are the only translation substrate left.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "Kavitha/how2sign_user3_mediapipe_pose"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/data/"
UA = "seam/0.1 (research; contact via repo owner)"
DEFAULT_OUT = Path("/mnt/DevProd/seam_data/how2sign_pose")


def shard_names(n: int = 31) -> list[str]:
    return [f"train-{i:05d}-of-{n:05d}.parquet" for i in range(n)]


def remote_size(name: str) -> int | None:
    """Content-Length for one shard, or None if the server will not say."""
    req = urllib.request.Request(BASE + name, method="HEAD", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def fetch_one(name: str, out: Path, *, attempts: int = 5) -> tuple[str, int]:
    """Download one shard with resume. Returns ``(status, bytes)``."""
    dest = out / name
    part = out / (name + ".part")
    if dest.is_file():
        return "skip", dest.stat().st_size

    want = remote_size(name)
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.is_file() else 0
        if want is not None and have == want:
            part.rename(dest)
            return "ok", have
        if want is not None and have > want:
            # A stale .part from a different revision; start over rather than resume into it.
            part.unlink()
            have = 0
        headers = {"User-Agent": UA}
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(BASE + name, headers=headers)
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=180) as r:
                # A server that ignores Range replies 200 with the whole body; appending
                # then would produce a file with a duplicated prefix that still "verifies".
                mode = "ab" if (have and r.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                with part.open(mode) as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            got = part.stat().st_size
            mb = got / 1e6
            rate = (got - have) / 1e6 / max(time.time() - t0, 0.01)
            print(f"  {name}: {mb:.0f} MB at {rate:.1f} MB/s", flush=True)
            if want is None or got == want:
                part.rename(dest)
                return "ok", got
            print(f"  short read: {got} of {want}", flush=True)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  attempt {attempt}/{attempts} for {name}: {exc}", flush=True)
            time.sleep(min(10 * attempt, 60))
    return "failed", part.stat().st_size if part.is_file() else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help="only the first N shards")
    ap.add_argument("--manifest-only", action="store_true", help="print sizes and exit")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    names = shard_names()
    total = 0
    for name in names:
        size = remote_size(name)
        if size is None:
            print(f"  {name}: not on the remote any more", flush=True)
            continue
        total += size
        if not args.manifest_only:
            fetch_one(name, args.out)

    summary = {
        "repo": REPO,
        "shards": len(names),
        "bytes": total,
        "gigabytes": round(total / 1e9, 2),
        "out": str(args.out),
        "license": "ungated (gated=False, private=False) as checked 2026-10-04",
    }
    (args.out / "fetch_manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
