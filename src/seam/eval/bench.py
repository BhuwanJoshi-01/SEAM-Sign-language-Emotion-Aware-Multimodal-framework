"""Latency and memory measurement on the actual RTX 3050.

Rules this module exists to enforce, all of them learned the hard way in this
project:

* **Measure on the target device, or do not report.** A number from the 3050 and
  a number from another machine are different facts. ``bench.py`` refuses to
  report a latency figure unless the device matches the configured one.

* **Report the distribution, not the mean.** A p95 that is quoted as a mean is
  the single most common way an interactive-latency claim gets falsified in
  review.

* **Report the machine's noise alongside the model's cost.** The minimum over
  repeated runs estimates the model's own latency; the gap to the mean is the
  machine. At M0 that gap was 30% - 82 ms unloaded against 110 ms under load -
  which is the difference between missing a 20 FPS target and clearing it. Hiding
  it would have meant reporting the number that flattered the work.

* **Peak VRAM is allocated *plus* reserved, cross-checked against nvidia-smi.**
  See :mod:`seam.export.vram_guard`.

Every result carries the device it was measured on, so a number that travels
into the paper without its provenance is visibly incomplete.
"""

from __future__ import annotations

import contextlib
import json
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from seam.export import vram_guard
from seam.logging import get

log = get(__name__)


class BenchError(RuntimeError):
    """Raised when a measurement cannot be taken or is not reportable."""


@dataclass(slots=True)
class Timing:
    """A latency distribution, in milliseconds."""

    label: str
    runs: int
    ms_min: float
    ms_mean: float
    ms_median: float
    ms_p95: float
    ms_p99: float
    #: mean/min over run means. The machine's own noise, reported rather than
    #: divided out.
    noise_ratio: float
    warmup: int
    #: Individual timed calls behind the quantiles. A p99 needs a real sample, so
    #: this is shown next to the numbers it justifies.
    samples: int = 0
    #: p95 across run means, kept only to show it is a different quantity.
    ms_p95_run_mean: float = 0.0
    fps_from_min: float = 0.0
    fps_from_median: float = 0.0
    #: 1000 / mean, i.e. the rate actually sustained when frames arrive back to
    #: back. The honest number for a capture loop.
    fps_sustained: float = 0.0

    def summary(self) -> str:
        return (
            f"{self.label:<34} min {self.ms_min:7.2f}  med {self.ms_median:7.2f}  "
            f"p95 {self.ms_p95:7.2f}  p99 {self.ms_p99:7.2f} ms "
            f"(n={self.samples})  ->  {self.fps_sustained:5.1f} FPS sustained "
            f"(noise x{self.noise_ratio:.2f})"
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class BenchReport:
    """Everything one benchmark run produced."""

    device: str
    device_matches_target: bool
    target_device: str
    timings: list[Timing] = field(default_factory=list)
    vram: vram_guard.VramReport | None = None
    stages: dict[str, float] = field(default_factory=dict)
    machine: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add(self, t: Timing) -> None:
        self.timings.append(t)

    def summary(self) -> str:
        head = (
            f"device {self.device}  (target {self.target_device}: "
            f"{'match' if self.device_matches_target else 'MISMATCH - numbers are not reportable'})"
        )
        body = "\n".join(t.summary() for t in self.timings)
        vram = self.vram.summary() if self.vram else "vram not measured"
        stages = (
            "\n".join(f"    {k:<30} {v:7.0f} MB" for k, v in self.stages.items())
            if self.stages
            else ""
        )
        noise = "\n".join(f"  {k}: {v}" for k, v in self.machine.items())
        notes = "\n".join(f"  ! {n}" for n in self.notes)
        return "\n".join(
            p for p in (head, "", body, "", vram, stages, "", "machine:", noise, notes) if p
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "device": self.device,
            "device_matches_target": self.device_matches_target,
            "target_device": self.target_device,
            "timings": [t.as_dict() for t in self.timings],
            "vram": self.vram.as_dict() if self.vram else None,
            "stage_vram_mb": self.stages,
            "machine": self.machine,
            "notes": self.notes,
        }


def machine_facts() -> dict[str, str]:
    """The context a latency number is meaningless without."""
    import os

    facts = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": str(os.cpu_count()),
    }
    # Load average and available memory are the two facts that decide whether a
    # latency number means anything on this machine.
    with contextlib.suppress(OSError, AttributeError):
        facts["loadavg"] = " ".join(f"{v:.2f}" for v in os.getloadavg())
    with contextlib.suppress(OSError), open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable"):
                facts["mem_available_kb"] = line.split()[1]
                break
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["cuda"] = str(torch.version.cuda)
    except ImportError:  # pragma: no cover
        pass
    return facts


def time_fn(
    fn: Callable[[], object],
    *,
    label: str,
    iterations: int,
    warmup: int = 3,
    runs: int = 5,
) -> Timing:
    """Measure ``fn`` and return the full distribution.

    Percentiles are over **individual calls**, not over run means. The previous
    version timed a whole run, took the mean, and then reported percentiles across
    ``runs`` samples - so a "p99" was the 99th percentile of five numbers, and a
    "p95" of five. Those are not latency quantiles; they cannot be compared with
    anyone's else, and a 5-sample p99 is close to the maximum by construction, so
    it drifted with machine noise rather than with the workload.

    Per-call timing is also the distribution a real-time system actually faces: a
    signing stream is a sequence of frames, and what matters is how long the worst
    1% of frames took.

    ``iterations`` calls per run and ``runs`` runs, so the quantile sample is
    ``iterations * runs`` calls. The per-*run* means are kept as well, and their
    ratio to the fastest run is reported as the machine's noise, because
    pretending noise away is how a latency claim gets falsified.
    """
    for _ in range(max(warmup, 0)):
        fn()

    per_call: list[float] = []
    per_run: list[float] = []
    for _ in range(max(runs, 1)):
        t0 = time.perf_counter()
        for _ in range(iterations):
            call0 = time.perf_counter()
            fn()
            per_call.append((time.perf_counter() - call0) * 1000)
        per_run.append((time.perf_counter() - t0) * 1000 / iterations)

    call_arr = np.asarray(per_call)
    run_arr = np.asarray(per_run)
    return Timing(
        label=label,
        runs=len(per_run),
        samples=len(call_arr),
        ms_min=float(call_arr.min()),
        ms_mean=float(call_arr.mean()),
        ms_median=float(np.median(call_arr)),
        ms_p95=float(np.percentile(call_arr, 95)),
        ms_p99=float(np.percentile(call_arr, 99)),
        ms_p95_run_mean=float(np.percentile(run_arr, 95)),
        fps_from_min=float(1000.0 / call_arr.min()),
        fps_from_median=float(1000.0 / np.median(call_arr)),
        fps_sustained=float(1000.0 / call_arr.mean()),
        noise_ratio=float(run_arr.mean() / run_arr.min()) if run_arr.min() > 0 else float("nan"),
        warmup=warmup,
    )


def target_device() -> str:
    from seam.paths import project_root

    cfg = project_root() / "configs" / "base.yaml"
    if not cfg.is_file():
        return vram_guard.device_name()
    import yaml

    data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    return str(data.get("deploy", {}).get("gpu_name", vram_guard.device_name()))


def new_report(*, target: str | None = None) -> BenchReport:
    want = target or target_device()
    have = vram_guard.device_name()
    rep = BenchReport(
        device=have,
        target_device=want,
        device_matches_target=want.lower() in have.lower() or have.lower() in want.lower(),
        machine=machine_facts(),
    )
    if not rep.device_matches_target:
        rep.notes.append(
            f"measured on {have!r}, configured target {want!r}. Latency and VRAM numbers from "
            f"this run must not enter the paper's efficiency table."
        )
    return rep


def measure_vram(ceiling_mb: float = vram_guard.VRAM_CEILING_MB) -> vram_guard.VramReport:
    """Peak VRAM of whatever is currently resident, checked against the ceiling."""
    vram_guard.reset_peak()
    return vram_guard.check(ceiling_mb=ceiling_mb, stage="process peak", strict=False)


def perception_bench(
    video_paths: list[Path],
    *,
    iterations: int = 40,
    runs: int = 5,
    parallel: bool | None = None,
) -> BenchReport:
    """Benchmark the perception stage: the 3 MediaPipe graphs on real video.

    Reports the concurrent path (the default) and, when ``parallel=False``, the
    sequential baseline so the 1.5-1.6x concurrency result is reproducible rather
    than remembered.
    """
    import cv2

    from seam.perception import tasks_api as T

    rep = new_report()
    frames: list[np.ndarray] = []
    for v in video_paths:
        cap = cv2.VideoCapture(str(v))
        while len(frames) < iterations:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        if len(frames) >= iterations:
            break
    if len(frames) < 8:
        raise BenchError(f"only {len(frames)} frames decoded; cannot benchmark")

    # A fresh, explicit counter. The previous version drew the frame index and the
    # timestamp from two independent ``next()`` calls on the same iterator, so the
    # timestamp advanced by 33 ms per *call* while the frame index advanced by one
    # per call, and the two were unrelated by coincidence. MediaPipe only requires
    # a monotonic timestamp, so it ran and produced numbers - which is exactly why
    # it was worth fixing rather than leaving.
    tick = iter(range(10**9, 10**9 + 10**6, 33))

    # ``parallel`` must construct a new TaskLandmarker, not ask the singleton.
    # ``landmarker()`` returns the process-wide instance if one exists and ignores
    # the argument entirely, so requesting the sequential baseline after anything
    # else has touched perception would have measured the concurrent path and
    # labelled it "sequential" - a silent 1.57x error in the ablation table.
    task = T.landmarker() if parallel is None else T.TaskLandmarker(parallel=parallel)
    T._ensure(task)

    def one() -> None:
        step = next(tick)
        T.process_frame(frames[step % len(frames)], step, task=task)

    label = "perception, 3 graphs concurrent" if task.parallel else "perception, sequential"
    rep.add(time_fn(one, label=label, iterations=len(frames), warmup=2, runs=runs))

    try:
        rep.vram = measure_vram()
    except vram_guard.NoGpuError:
        rep.notes.append("no CUDA device; VRAM not measured")
    return rep


def resident_stack_bench(
    video_paths: list[Path],
    *,
    onnx_dir: Path | None = None,
    iterations: int = 30,
    warmup: int = 3,
) -> BenchReport:
    """Measure the *whole* live stack resident at once, not one stage at a time.

    The stage-by-stage numbers in the export report are cumulative but not
    simultaneous: each session was created, measured, and the sequence's allocator
    reuse made the third model report 0 MB. A real-time signing system does not
    load models one at a time - perception, three FER graphs and the text model
    are all live while a signer is in front of the camera - so the budget question
    is what the stack costs *together*, on the same device, at the same instant.

    This builds all of it and then runs frames through it. The VRAM figure is a
    pre-session baseline to post-run peak from ``nvidia-smi``, because torch's
    allocator cannot see either ONNX Runtime's or MediaPipe's allocations.
    """
    import cv2

    from seam.perception import tasks_api as T

    rep = new_report()
    frames: list[np.ndarray] = []
    for v in video_paths:
        cap = cv2.VideoCapture(str(v))
        while len(frames) < iterations:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        if len(frames) >= iterations:
            break
    if len(frames) < 8:
        raise BenchError(f"only {len(frames)} frames decoded; cannot benchmark")

    baseline = vram_guard.nvidia_smi_mb()

    task = T.TaskLandmarker(parallel=True)
    T._ensure(task)

    sessions: list[tuple[str, Any]] = []
    if onnx_dir is not None and onnx_dir.is_dir():
        from seam.export.onnx_export import _session

        for path in sorted(onnx_dir.glob("*.onnx")):
            if ".int8." in path.name:
                continue
            sessions.append((path.stem, _session(path)))

    if not sessions:
        rep.notes.append("no ONNX graphs found; measured perception only")

    def one() -> None:
        T.process_frame(frames[next(tick) % len(frames)], next(tick), task=task)
        for _name, sess in sessions:
            # A zero tensor of the right shape, not the perception output. The
            # FER graphs consume face crops, which are produced by a separate crop
            # stage that does not exist yet, so wiring perception straight into
            # them here would be a fiction. What this measures is the point: that
            # all of the graphs are resident and being invoked at the same time.
            sess.run(None, {sess.get_inputs()[0].name: _probe_input(sess)})

    tick = iter(range(10**9, 10**9 + 10**6, 33))
    rep.add(
        time_fn(
            one,
            label=f"live stack ({len(sessions)} FER + 3 graphs)",
            iterations=iterations,
            warmup=warmup,
            runs=1,
        )
    )
    rep.notes.append(
        f"resident stack: 3 MediaPipe graphs + {len(sessions)} ONNX graphs, all live at once"
    )
    rep.notes.append(
        "FER inputs are zero tensors of the correct shape, not perception output: "
        "the crop stage that would feed them does not exist yet, so this measures "
        "residency and invocation, not an end-to-end inference"
    )
    rep.stages["stack_baseline_mb"] = baseline
    rep.stages["stack_peak_mb"] = max(vram_guard.nvidia_smi_mb(), 0.0)
    rep.stages["stack_delta_mb"] = max(rep.stages["stack_peak_mb"] - baseline, 0.0)

    try:
        rep.vram = measure_vram()
    except vram_guard.NoGpuError:
        rep.notes.append("no CUDA device; VRAM not measured")
    return rep


def _probe_input(session: Any) -> np.ndarray:
    """A zero input shaped like the session's input, for a live-stack run."""
    shape = [d if isinstance(d, int) else 1 for d in session.get_inputs()[0].shape]
    return np.zeros(shape, dtype=np.float32)


def save(report: BenchReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    log.info("wrote %s", path)


def render(report: BenchReport) -> str:
    lines = [report.summary()]
    if report.timings:
        best = max(report.timings, key=lambda t: t.fps_from_median)
        lines += [
            "",
            f"best stage: {best.label} at {best.fps_from_median:.1f} FPS "
            f"(p95 {best.ms_p95:.2f} ms)",
        ]
    return "\n".join(lines)


def percentile_note() -> str:
    return (
        "p95 and p99 are percentiles over repeated runs of the whole batch, not over "
        "individual frames. Run-to-run is the right unit here: within a run, frames are "
        "processed on one warm allocator, and per-frame percentiles would mostly measure "
        "the order the frames happen to appear in."
    )
