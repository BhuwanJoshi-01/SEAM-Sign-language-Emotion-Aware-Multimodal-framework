#!/usr/bin/env python
"""Fetch RAF-DB shards. The val split is enough for the audit; train shards are
for fitting the non-signer FER baselines the audit indicts."""

from __future__ import annotations

import pathlib
import sys
import time
import urllib.request

BASE = "https://huggingface.co/datasets/Pelmeshek/raf-db-7emotions-mediapipe-768/resolve/main/data/"
FILES = [
    "train-00000-of-00004.parquet",
    "train-00001-of-00004.parquet",
    "train-00002-of-00004.parquet",
    "train-00003-of-00004.parquet",
]


def main() -> int:
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/DevProd/seam_data/rafdb")
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = out / name.replace("-0000", "-").replace("train-", "train_").replace(
            ".parquet", ".parquet"
        )
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"skip {dest.name}", flush=True)
            continue
        t0 = time.time()
        try:
            req = urllib.request.Request(BASE + name, headers={"User-Agent": "seam/0.1"})
            with urllib.request.urlopen(req, timeout=600) as r, dest.open("wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            print(
                f"{dest.name}: {dest.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f}s",
                flush=True,
            )
        except Exception as exc:
            print(f"FAILED {name}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
