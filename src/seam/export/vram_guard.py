"""The VRAM ceiling, enforced rather than hoped for.

Two numbers, and confusing them is the easiest way to ship a benchmark that
passes for the wrong reason. ``nvidia-smi`` reports **4096 MiB** for this board;
``torch.cuda.get_device_properties`` reports **3770 MB** usable, because roughly
326 MB is reserved by the display/EGL path before any process starts. A guard
written against the board figure fails for a reason that has nothing to do with
the model, and a guard written against the usable figure with the wrong ceiling
fails on a laptop that is merely running a browser.

The ceiling itself is 2500 MB - **66% of usable**, chosen so the pipeline still
fits alongside a desktop session rather than only on an otherwise-idle machine.

The guard's job is to *fail loudly*. A degraded fallback - quietly running on CPU,
or skipping a model - would turn a hard constraint into a soft one, and the
efficiency claim would become unfalsifiable. Every allowance the guard makes is
printed.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

#: Board memory as reported by nvidia-smi, and what torch reports usable.
#: Measured 2026-09-26; both are in configs/base.yaml.
BOARD_MB = 4096
USABLE_MB = 3770

#: The project's ceiling. 66% of usable.
VRAM_CEILING_MB = 2500


class VramBudgetExceeded(RuntimeError):
    """Raised when peak VRAM exceeds the ceiling.

    Deliberately an exception rather than a warning. The plan's rule is that the
    constraint is asserted, not degraded around.
    """


class NoGpuError(RuntimeError):
    """Raised when a measurement is requested and there is no CUDA device."""


@dataclass(slots=True)
class VramReport:
    """A peak-VRAM measurement and its verdict."""

    peak_mb: float
    ceiling_mb: float
    device: str
    ok: bool
    detail: str = ""
    #: Per-stage peaks, when the caller supplies them.
    stages: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        verdict = "PASS" if self.ok else "FAIL"
        return (
            f"[{verdict}] peak {self.peak_mb:.0f} MB / ceiling {self.ceiling_mb:.0f} MB "
            f"on {self.device} ({self.peak_mb / VRAM_CEILING_MB * 100:.0f}% of budget)"
            + (f" - {self.detail}" if self.detail else "")
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "peak_mb": round(self.peak_mb, 1),
            "ceiling_mb": self.ceiling_mb,
            "device": self.device,
            "ok": self.ok,
            "utilisation_pct": round(self.peak_mb / self.ceiling_mb * 100, 1),
            "detail": self.detail,
            "stages": {k: round(v, 1) for k, v in self.stages.items()},
        }


def device_name(index: int = 0) -> str:
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_properties(index).name


def reset_peak() -> None:
    """Reset the peak counter so a measurement covers only what follows."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_mb(index: int = 0) -> float:
    """Peak allocated + reserved VRAM in MB.

    *Reserved* rather than merely *allocated*: reserved is what the caching
    allocator has actually taken from the driver, and it is the number that
    decides whether a second model can load. Measuring only allocations reports a
    figure the system never actually used, which flatters the result.
    """
    import torch

    if not torch.cuda.is_available():
        raise NoGpuError("no CUDA device; VRAM cannot be measured")
    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated(index)
    reserved = torch.cuda.max_memory_reserved(index)
    return (allocated + reserved) / 1024**2


def nvidia_smi_mb(index: int = 0) -> float:
    """Peak used VRAM as the driver sees it, via nvidia-smi.

    Independent of torch's accounting, and therefore a real cross-check: a
    torch-only number can be wrong if something outside torch allocates on the
    device, which is exactly the kind of error that makes a budget assertion
    unfalsifiable.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                f"--id={index}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return float(out.stdout.strip().splitlines()[0])
    except (subprocess.SubprocessError, ValueError, IndexError) as exc:
        log.warning("nvidia-smi unavailable: %s", exc)
        return float("nan")


def check(
    peak: float | None = None,
    *,
    ceiling_mb: float = VRAM_CEILING_MB,
    stage: str = "",
    strict: bool = True,
) -> VramReport:
    """Compare a peak measurement against the ceiling.

    ``strict`` raises on breach. A non-strict call still returns a report with
    ``ok=False``, which is what a benchmark sweep wants; a test or a CI gate
    wants ``strict=True`` so a breach fails the build.
    """
    device = device_name()
    detail = stage
    smi = nvidia_smi_mb()
    if peak is None:
        # Enforce on the larger of torch's figure and the driver's, and say when
        # they disagree. Two situations make torch the wrong instrument, both
        # measured on this machine: ONNX Runtime allocates outside torch's caching
        # allocator and reports 0.0 MB through it, and MediaPipe Tasks under the
        # CPU delegate never touches torch at all. Both times torch said 0 MB while
        # nvidia-smi showed a live allocation. A gate that takes the smaller number
        # would have passed a stage that was using the device.
        measured = peak_mb()
        if not _isnan(smi) and smi > measured:
            detail = (
                f"{detail}; nvidia-smi reports {smi:.0f} MB, torch {measured:.0f} MB - "
                f"enforcing the larger".strip("; ")
            )
            measured = smi
    else:
        measured = peak
    ok = measured <= ceiling_mb
    report = VramReport(measured, ceiling_mb, device, ok, detail)

    if not ok and strict:
        raise VramBudgetExceeded(
            f"peak VRAM {measured:.0f} MB exceeds the {ceiling_mb:.0f} MB ceiling on {device}. "
            f"The efficiency claim is not a soft target; refusing to continue. "
            f"Remedies, in the order the plan gives them: drop to 12 fps, shrink the "
            f"text model, reduce the keypoint subset."
        )
    return report


def _isnan(x: float) -> bool:
    return x != x


def free_mb(index: int = 0) -> float:
    import torch

    if not torch.cuda.is_available():
        raise NoGpuError("no CUDA device")
    free, _total = torch.cuda.mem_get_info(index)
    return free / 1024**2


def budget_note() -> str:
    """The two-number explanation, for the paper's implementation details."""
    return (
        f"The RTX 3050 reports {BOARD_MB} MB of board memory but {USABLE_MB} MB usable to "
        f"torch; ~{BOARD_MB - USABLE_MB} MB is reserved by the display path before any "
        f"process starts. The {VRAM_CEILING_MB} MB ceiling is "
        f"{VRAM_CEILING_MB / USABLE_MB * 100:.0f}% of the usable figure, chosen so the "
        f"pipeline fits alongside a desktop session rather than only on an idle machine. "
        f"VRAM is measured as torch's peak allocated *plus reserved*, and cross-checked "
        f"against nvidia-smi, because a torch-only number can miss allocation done outside "
        f"torch and an assertion that cannot be falsified is not an assertion."
    )
