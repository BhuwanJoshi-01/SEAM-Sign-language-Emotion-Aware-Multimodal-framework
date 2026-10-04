#!/usr/bin/env python
"""Fetch the How2Sign MediaPipe landmark cache (4.8 GB, 991 NPZ shards + metadata).

Why this dataset and not the one `plan.md` names
-------------------------------------------------
`plan.md` §1 lists `Kavitha/how2sign_user3_mediapipe_pose` as the ungated M5b substrate,
described as "media-pipe keypoints published, plus official English". Measured on 2026-10-04
after fetching 7.2 GB of it, that description is wrong on both counts:

* **There are no keypoints.** The schema is `image` (JPEG bytes), `conditioning_image`
  (JPEG bytes), `text` (string). 31 shards, ~91,500 rows, 14.12 GB.
* **There are no English captions.** `text` is the constant string `"signer signing"` on
  every row sampled. With no landmarks and no target text it cannot train or evaluate a
  translation model at all.

`martinctl/how2sign-asl-landmarks` (CC-BY-NC-4.0, ungated, 4.82 GB) has what the plan
assumed, and it was checked rather than assumed:

* `metadata.parquet`, **35,176 rows**, one per sentence, with a real `sentence` field
  ("My name is Dr. Art Bowler.") and official train/validation/test `split`
* `shards/shard_NNNNN.npz` holding `landmarks_image`, `landmarks_world`,
  `features_geometric`, `valid_mask`, `timestamps_ms`, `handedness_scores`
* 51 ARKit face landmark indices in `preprocessing_config.json` — the same face channel
  this project already extracts, so M5b features are comparable with M3/M4
* failures recorded in `failures.parquet` rather than silently dropped

Resumable on purpose: `.part` files are kept and continued with a Range request, and only
renamed into place once the byte count matches the server's.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "martinctl/how2sign-asl-landmarks"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
UA = "seam/0.1 (research; contact via repo owner)"
DEFAULT_OUT = Path("/mnt/DevProd/seam_data/how2sign_landmarks")

#: Fetched before the bulk shards, because they are what makes the corpus interpretable and
#: because they are small enough to fail fast on a bad repo id.
CONTROL_FILES = (
    "README.md",
    "metadata.parquet",
    "feature_names.json",
    "preprocessing_config.json",
    "audit_report.json",
    "failures.parquet",
)


def _get(url: str, timeout: int = 60) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": UA, "timeout": str(timeout)})


def remote_size(path: str) -> int | None:
    try:
        with urllib.request.urlopen(_get(BASE + path), timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def fetch_one(path: str, out: Path, *, attempts: int = 5) -> tuple[str, int]:
    """Download one file with resume. Returns ``(status, bytes)``."""
    dest = out / path
    part = out / (path + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return "skip", dest.stat().st_size

    want = remote_size(path)
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.is_file() else 0
        if want is not None and have == want:
            part.rename(dest)
            return "ok", have
        if want is not None and have > want:
            part.unlink()  # stale .part from another revision; resuming into it would corrupt
            have = 0
        headers = {"User-Agent": UA}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            t0 = time.time()
            with urllib.request.urlopen(_get(BASE + path), timeout=180) as r:
                # A server that ignores Range replies 200 with the whole body; appending then
                # would produce a file with a duplicated prefix that still "verifies".
                mode = "ab" if (have and r.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                with part.open(mode) as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            got = part.stat().st_size
            rate = (got - have) / 1e6 / max(time.time() - t0, 0.01)
            print(f"  {path}: {got / 1e6:.1f} MB at {rate:.1f} MB/s", flush=True)
            if want is None or got == want:
                part.rename(dest)
                return "ok", got
            print(f"  short read: {got} of {want}", flush=True)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  attempt {attempt}/{attempts} for {path}: {exc}", flush=True)
            time.sleep(min(5 * attempt, 45))
    return "failed", part.stat().st_size if part.is_file() else 0


def shard_paths(n: int) -> list[str]:
    return [f"shards/shard_{i:05d}.npz" for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--shards", type=int, default=991)
    ap.add_argument("--control-only", action="store_true")
    ap.add_argument("--plan-only", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    shards = shard_paths(args.shards)
    total = 0
    for path in (*CONTROL_FILES, *shards):
        size = remote_size(path)
        if size is None:
            print(f"  {path}: not on the remote", flush=True)
            continue
        total += size
        if args.plan_only:
            print(f"  {path:34} {size / 1e6:9.1f} MB")
            continue
        if path in CONTROL_FILES or not args.control_only:
            fetch_one(path, args.out)

    if args.plan_only:
        print(f"total {total / 1e9:.2f} GB over {len(CONTROL_FILES) + len(shards)} files")
        return 0

    summary = {
        "repo": REPO,
        "license": "cc-by-nc-4.0",
        "gated": False,
        "shards": args.shards,
        "bytes": total,
        "gigabytes": round(total / 1e9, 2),
        "out": str(args.out),
        "supersedes": (
            "Kavitha/how2sign_user3_mediapipe_pose, which plan.md named but which contains "
            "JPEG images and the constant caption 'signer signing' - no landmarks, no English"
        ),
    }
    (args.out / "fetch_manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
