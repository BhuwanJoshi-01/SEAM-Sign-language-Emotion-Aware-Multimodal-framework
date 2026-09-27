"""Batch extraction driver, resumable at the clip level, for any manifest.

Parallelism is process-based, and the worker count is deliberately conservative.
The perception stage already runs three landmarker graphs on three threads
inside each worker, so N workers occupy ~3N threads plus a graph-construction
burst. On a 16-vCPU machine that is also running a desktop, two browsers and an
editor, 2 workers is the point where throughput still improves and the
interactive session stays usable. Measured: the marginal return flattens past
that, and the memory cost is the binding constraint - this machine has about
3 GB of free RAM and MediaPipe graphs are not free.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from seam.logging import get
from seam.perception import extract as extract_mod

log = get(__name__)

#: Frames kept per clip. The recognition window is T=64, so anything past that is
#: discarded downstream anyway; capping here bounds the extraction cost without
#: changing any number that reaches the paper.
DEFAULT_MAX_FRAMES = 96


@dataclass(slots=True)
class Job:
    video: str
    key: str


@dataclass(slots=True)
class Result:
    key: str
    frames: int
    status: str


def _one(args: tuple[str, str, str, int]) -> Result:
    video, key, out_root, max_frames = args
    try:
        arrays, meta = extract_mod.extract_clip(
            Path(video), utterance_id=key, max_frames=max_frames
        )
        extract_mod.save_shard(Path(out_root), arrays, meta)
        return Result(key, meta.frame_count, "ok")
    except extract_mod.ExtractionError as exc:
        return Result(key, 0, f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # a worker must never take the batch down
        return Result(key, 0, f"{type(exc).__name__}: {exc}")


def run(
    jobs: Iterable[Job],
    out_root: Path,
    *,
    workers: int = 1,
    max_frames: int | None = DEFAULT_MAX_FRAMES,
    force: bool = False,
    label: str = "extract",
) -> int:
    """Extract every job that does not already have a valid shard.

    Returns 0 when every requested clip produced a shard.
    """
    out_root.mkdir(parents=True, exist_ok=True)

    if force:
        for job in jobs:
            extract_mod.shard_path(out_root, job.key).unlink(missing_ok=True)

    todo = [j for j in jobs if force or not extract_mod.is_extracted(out_root, j.key)]
    total = len(list(jobs))
    if not todo:
        log.info("%s: all %d clips already extracted", label, total)
        print(f"{label}: all {total} clips already extracted under {out_root}")
        return 0

    if not force:
        log.info("%s: resuming, %d of %d remaining", label, len(todo), total)

    payload = [(j.video, j.key, str(out_root), max_frames or 0) for j in todo]
    started = time.perf_counter()
    ok = 0
    failures: list[tuple[str, str]] = []

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for n, res in enumerate(pool.map(_one, payload), start=1):
                ok, failures = _accumulate(res, n, len(payload), started, ok, failures)
    else:
        for n, args in enumerate(payload, start=1):
            ok, failures = _accumulate(_one(args), n, len(payload), started, ok, failures)

    elapsed = time.perf_counter() - started
    print(
        f"\n{label}: extracted {ok}/{len(payload)} clips in {elapsed / 60:.1f} min "
        f"({len(payload) / max(elapsed, 1e-6):.2f} clips/s) -> {out_root}"
    )
    for key, status in failures[:10]:
        print(f"  FAILED {key}: {status}")
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more")
    return 0 if not failures else 1


def _accumulate(
    res: Result,
    n: int,
    total: int,
    started: float,
    ok: int,
    failures: list[tuple[str, str]],
) -> tuple[int, list[tuple[str, str]]]:
    if res.status == "ok":
        ok += 1
    else:
        failures.append((res.key, res.status))
    if n % 25 and n != total:
        return ok, failures
    elapsed = time.perf_counter() - started
    rate = n / max(elapsed, 1e-6)
    print(
        f"\r  {n}/{total}  ok={ok} fail={len(failures)}  "
        f"{rate:.2f} clips/s  eta {(total - n) / max(rate, 1e-9) / 60:.1f}m",
        end="",
        file=sys.stderr,
        flush=True,
    )
    if n == total:
        print(file=sys.stderr)
    return ok, failures
