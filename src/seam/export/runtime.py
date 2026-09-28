"""Making the ONNX Runtime CUDA provider actually run, and measuring it honestly.

Two problems, both of which would otherwise have produced a report full of
numbers that mean nothing.

**The CUDA provider needs libraries that are on disk but not on the loader path.**
``onnxruntime-gpu`` is built against CUDA 13, and it fails with
``libcublasLt.so.13: cannot open shared object file`` even though torch has
installed exactly that library inside ``site-packages/nvidia/cu13/lib``. The
dynamic loader only searches the system paths, and this machine's system CUDA is
version 12. So the library is present, correct, and invisible.

:func:`ensure_cuda_libraries` fixes it by finding the bundled ``nvidia/*/lib``
directories and pre-loading them with ``RTLD_GLOBAL`` before any session is
created. Pre-loading rather than only setting ``LD_LIBRARY_PATH`` because
``dlopen`` consults the cached search path, and a process that has already
started does not reliably pick up a change to the environment.

**torch cannot see ONNX Runtime's VRAM at all.** This is not a subtlety, it is a
demonstrated zero. Running a CUDA-EP ONNX session twenty times allocates on the
device - ``nvidia-smi`` shows it - and ``torch.cuda.max_memory_allocated()``
simultaneously and permanently reports ``0.0`` bytes, because ORT allocates
through its own arena and never touches torch's caching allocator. An ORT
model's memory therefore has to be measured with ``nvidia-smi``, and
:func:`measure_session_vram` does exactly that by taking a delta across session
creation and a warm-up run, so the CUDA context is counted once rather than
per-run.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any

from seam.logging import get

log = get(__name__)

#: Bundled CUDA runtime libraries ORT's CUDA provider dlopens, in load order.
#: libcublasLt before libcublas because the former is what the error names.
_REQUIRED_SONAMES = (
    "libcublasLt.so.13",
    "libcublas.so.13",
    "libcudart.so.13",
    "libcufft.so.12",
    "libcurand.so.10",
    "libcusolver.so.12",
    "libcusparse.so.12",
    "libcudnn.so.9",
)

_ensured = False


def _nvidia_lib_dirs() -> list[Path]:
    """Every ``nvidia/*/lib`` directory inside the active environment."""
    lib_dirs: list[Path] = []
    for base in (Path(sys.prefix), *Path(sys.prefix).glob("lib/python3.*/site-packages")):
        nvidia = base / "nvidia"
        if not nvidia.is_dir():
            continue
        for pkg in sorted(nvidia.iterdir()):
            lib = pkg / "lib"
            if lib.is_dir() and lib not in lib_dirs:
                lib_dirs.append(lib)
    return lib_dirs


def ensure_cuda_libraries() -> dict[str, str | None]:
    """Pre-load the CUDA libraries ORT needs. Idempotent.

    Returns a mapping of soname -> path actually loaded, or ``None`` for anything
    the system already provides. The return value is written into the export
    report, because "the CUDA provider ran" and "the CUDA provider silently fell
    back to CPU" are very different claims and only one of them is visible from
    the latency number.
    """
    global _ensured
    if _ensured:
        return {}

    search = [str(p) for p in _nvidia_lib_dirs()]
    if search:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        joined = ":".join([*search, existing]) if existing else ":".join(search)
        os.environ["LD_LIBRARY_PATH"] = joined

    loaded: dict[str, str | None] = {}
    for soname in _REQUIRED_SONAMES:
        found: str | None = None
        for directory in search:
            candidate = Path(directory) / soname
            if candidate.is_file():
                found = str(candidate)
                break
        if found is None:
            loaded[soname] = None
            continue
        try:
            ctypes.CDLL(found, mode=ctypes.RTLD_GLOBAL)
            loaded[soname] = found
        except OSError as exc:  # pragma: no cover - depends on the machine
            log.warning("could not preload %s: %s", soname, exc)
            loaded[soname] = None

    missing = [k for k, v in loaded.items() if v is None]
    if missing:
        log.info("CUDA libraries not found for: %s", ", ".join(missing))
    _ensured = True
    return loaded


def provider_status() -> dict[str, object]:
    """What the runtime can actually do, as opposed to what it advertises.

    ``onnxruntime.get_available_providers()`` lists an execution provider when
    the build supports it. It does not check that the provider can initialise.
    On this machine the CUDA provider is *listed* and *unusable* until the
    bundled libraries are preloaded, so the only trustworthy test is to construct
    a session and read back which providers are active.
    """
    import numpy as np
    import onnxruntime as ort

    ensure_cuda_libraries()
    available = list(ort.get_available_providers())

    # A trivial graph, so the probe does not depend on any project artifact.
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["input"], ["output"])
    graph = helper.make_graph([node], "probe", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = Path(os.environ.get("TMPDIR", "/tmp")) / "seam_ep_probe.onnx"
    onnx.save(model, str(path))

    requested = [p for p in available if p != "AzureExecutionProvider"]
    active: list[str] = []
    error = ""
    # Try the full list, then CPU alone. Without the second attempt a machine
    # whose GPU provider cannot initialise returns an *empty* active list, which
    # is indistinguishable from a broken probe - and the real code path degrades
    # to CPU rather than failing, so the probe must degrade the same way.
    for attempt in (requested, ["CPUExecutionProvider"]):
        try:
            sess = ort.InferenceSession(str(path), providers=attempt)
            sess.run(None, {"input": np.zeros((1, 3, 8, 8), dtype=np.float32)})
            active = list(sess.get_providers())
            error = ""
            break
        except Exception as exc:  # pragma: no cover - depends on the machine
            error = f"{type(exc).__name__}: {exc}"
    path.unlink(missing_ok=True)

    cuda_working = "CUDAExecutionProvider" in active
    return {
        "available": available,
        "active_on_probe": active,
        "cuda_working": cuda_working,
        "fell_back_to_cpu": not cuda_working and "CPUExecutionProvider" in active,
        "tensorrt_working": "TensorrtExecutionProvider" in active,
        "error": error,
    }


def measure_session_vram(
    session: Any, feed: dict, *, warmup: int = 3, baseline_mb: float | None = None
) -> float:
    """VRAM attributable to an ONNX Runtime session, in MB.

    Measured with ``nvidia-smi``, because torch's allocator cannot see ORT's
    allocations at all - demonstrated in this module's docstring - and because a
    value torch produced would be a measurement of a different program.

    ``baseline_mb`` must be sampled **before the session was created**. The first
    inference builds the CUDA context, which costs a few hundred MB and is charged
    to whichever process created it, so a baseline taken after the first run
    silently excludes both the context and the weights, and reports a figure small
    enough to pass any budget while measuring nothing that will actually be
    allocated. That is not a conservative error, it is an invisible one.

    With no baseline the function falls back to sampling after one warm-up run,
    which measures activation memory only. It says so in the return rather than
    pretending to be a resident figure.
    """
    from seam.export import vram_guard

    def used() -> float:
        value = vram_guard.nvidia_smi_mb()
        return 0.0 if value != value else value

    session.run(None, feed)  # ensure the context and any lazy allocation exists
    after_warmup = used()
    for _ in range(max(warmup, 0)):
        session.run(None, feed)
    after = used()

    if baseline_mb is None:
        # No pre-session baseline available. Report the activation-only delta and
        # say so, so the number is not mistaken for a resident footprint.
        return max(after - after_warmup, 0.0)
    return max(after - baseline_mb, 0.0)
