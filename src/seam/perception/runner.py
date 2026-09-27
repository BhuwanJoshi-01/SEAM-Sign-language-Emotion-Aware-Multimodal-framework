"""Batch extraction driver, resumable at the clip level.

Parallelism is process-based, not thread-based, and the reason is measured: the
perception stage already uses three threads internally to run the three graphs
concurrently. Adding worker processes on top oversubscribes a 16-vCPU machine
that is already running a desktop, two browsers and an editor, and the
per-clip latency measurement in M2 depends on this being stable and explainable.

The default is therefore ``--workers 1`` and a larger value is opt-in.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from seam.data import emosign as emosign_mod
from seam.logging import get
from seam.perception import extract as extract_mod

log = get(__name__)


def shard_root(data_root: Path) -> Path:
    return data_root / "emosign" / "landmarks"


def _one(args: tuple[str, str, int]) -> tuple[str, int, str]:
    video, utterance_id, every_n = args
    try:
        arrays, meta = extract_mod.extract_clip(
            Path(video), utterance_id=utterance_id, every_n=every_n
        )
        extract_mod.save_shard(Path(video).parent.parent / "landmarks", arrays, meta)
        return utterance_id, meta.frame_count, "ok"
    except extract_mod.ExtractionError as exc:
        return utterance_id, 0, f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # a worker must never take the batch down
        return utterance_id, 0, f"{type(exc).__name__}: {exc}"


def extract_emosign(
    data_root: Path,
    *,
    limit: int | None = None,
    force: bool = False,
    workers: int = 1,
    every_n: int = 1,
) -> int:
    """Extract landmarks and blendshapes for the EmoSign clips.

    Returns a process exit code: 0 if every requested clip produced a shard.
    """
    labels = emosign_mod.load(data_root)
    video_root = data_root / "emosign" / "video"
    out_root = shard_root(data_root)
    out_root.mkdir(parents=True, exist_ok=True)

    targets = [c for c in labels if (video_root / f"{c.utterance_id}.mp4").is_file()]
    if not targets:
        print(f"no EmoSign videos under {video_root}; run `seam data fetch --all` first")
        return 1
    if limit is not None:
        targets = targets[:limit]

    if force:
        for clip in targets:
            extract_mod.shard_path(out_root, clip.utterance_id).unlink(missing_ok=True)

    todo = [c for c in targets if force or not extract_mod.is_extracted(out_root, c.utterance_id)]
    skipped = len(targets) - len(todo)
    if skipped:
        log.info("resuming: %d already extracted, %d to do", skipped, len(todo))
    if not todo:
        print(f"all {len(targets)} clips already extracted under {out_root}")
        return 0

    jobs = [(str(video_root / f"{c.utterance_id}.mp4"), c.utterance_id, every_n) for c in todo]
    started = time.perf_counter()
    ok = 0
    failures: list[tuple[str, str]] = []

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for n, (uid, _frames, status) in enumerate(pool.map(_one, jobs), start=1):
                if status == "ok":
                    ok += 1
                else:
                    failures.append((uid, status))
                _progress(n, len(jobs), started, ok, failures)
    else:
        for n, job in enumerate(jobs, start=1):
            uid, _frames, status = _one(job)
            if status == "ok":
                ok += 1
            else:
                failures.append((uid, status))
            _progress(n, len(jobs), started, ok, failures)

    elapsed = time.perf_counter() - started
    print(
        f"\nextracted {ok}/{len(jobs)} clips in {elapsed:.0f}s "
        f"({len(jobs) / max(elapsed, 1e-6):.2f} clips/s) -> {out_root}"
    )
    for uid, status in failures[:10]:
        print(f"  FAILED {uid}: {status}")
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more")
    return 0 if not failures else 1


def _progress(n: int, total: int, started: float, ok: int, failures: list[tuple[str, str]]) -> None:
    if n % 10 and n != total:
        return
    elapsed = time.perf_counter() - started
    rate = n / max(elapsed, 1e-6)
    eta = (total - n) / max(rate, 1e-9)
    msg = f"\r  {n}/{total} clips  ok={ok} fail={len(failures)}  {rate:.2f}/s  eta {eta / 60:.1f}m"
    print(msg, end="", file=sys.stderr, flush=True)
    if n == total:
        print(file=sys.stderr)
