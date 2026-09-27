#!/usr/bin/env python
"""Fetch the remaining RAF-DB shards, retrying transient DNS failures."""

from __future__ import annotations

import pathlib
import time
import urllib.request

BASE = "https://huggingface.co/datasets/Pelmeshek/raf-db-7emotions-mediapipe-768/resolve/main/data/"
WANT = {
    "train-00000-of-00004.parquet": "train_0-of-4.parquet",
    "test-00000-of-00001.parquet": "test.parquet",
}


def main() -> int:
    out = pathlib.Path("/mnt/DevProd/seam_data/rafdb")
    for remote, local in WANT.items():
        dest = out / local
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"skip {local}", flush=True)
            continue
        for attempt in range(4):
            try:
                t0 = time.time()
                req = urllib.request.Request(BASE + remote, headers={"User-Agent": "seam/0.1"})
                with urllib.request.urlopen(req, timeout=900) as r, dest.open("wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                mb = dest.stat().st_size / 1e6
                print(f"{local}: {mb:.0f} MB in {time.time() - t0:.0f}s", flush=True)
                break
            except Exception as exc:
                print(f"attempt {attempt + 1} for {remote} failed: {exc}", flush=True)
                time.sleep(15)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
