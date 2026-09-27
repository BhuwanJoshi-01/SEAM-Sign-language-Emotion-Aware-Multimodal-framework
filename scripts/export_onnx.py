#!/usr/bin/env python
"""Export the FER baselines to ONNX, check parity, and measure the budget.

M2's evidence gate. Three things must hold and all three fail loudly:

1. the exported FP32 graph matches PyTorch on real inputs;
2. INT8 dynamic quantisation stays inside tolerance and preserves the argmax;
3. the resident model fits the 2500 MB VRAM ceiling.

Every number is written to ``artifacts/export/`` with the device it was measured
on, because a latency or memory figure without its device is not a measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from seam.eval import bench, fer
from seam.eval.fer import FerModelSpec
from seam.export import onnx_export as X
from seam.export import runtime as RT
from seam.export import vram_guard as V
from seam.logging import get, setup

log = get(__name__)
from seam.paths import artifacts_root


def load_fer(ckpt: Path):
    blob = torch.load(ckpt, weights_only=False)
    spec = FerModelSpec(**blob["spec"])
    model = fer.build_model(spec, n_classes=blob["n_classes"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return spec, model


def parity_inputs(count: int, size: int) -> tuple[list[np.ndarray], str]:
    """Parity inputs, and a note describing where they came from.

    Real held-out RAF-DB faces, through the same decoder the models were fitted
    with. The first version of this function synthesised random noise, which was a
    mistake worth recording: noise images drive a FER model to near-uniform
    probabilities, so the argmax is decided by a vanishing margin and flips on
    numerical noise alone. That produced a 95.8% argmax agreement for the INT8
    ``fer_cnn_a``, a number that describes noise and not the quantized model.
    Parity is a question about real inputs.
    """
    try:
        from seam.data import rafdb

        return rafdb.parity_batches(size=size, per_batch=4, count=count), "rafdb held-out faces"
    except Exception as exc:
        log.warning("falling back to synthetic parity inputs: %s", exc)
        rng = np.random.default_rng(0)
        out = []
        for i in range(count):
            b = (1, 2, 4)[i % 3]
            crops = [
                fer.normalize_face(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8), size)
                for _ in range(b)
            ]
            out.append(np.stack(crops))
        return out, "synthetic noise (no held-out data available)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fer-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--inputs", type=int, default=24)
    ap.add_argument("--ceiling-mb", type=float, default=V.VRAM_CEILING_MB)
    args = ap.parse_args()

    setup("INFO")
    get("export")
    RT.ensure_cuda_libraries()
    fer_dir = Path(args.fer_dir) if args.fer_dir else artifacts_root() / "fer"
    out_dir = Path(args.out) if args.out else artifacts_root() / "export"
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs, input_source = parity_inputs(args.inputs, fer.CROP_SIZE)
    print(f"parity inputs: {len(inputs)} batches from {input_source}")
    print()
    results: list[X.ParityResult] = []
    status = RT.provider_status()
    extra: dict = {
        "machine": bench.machine_facts(),
        "execution_providers": status,
        "bench_note": bench.percentile_note(),
        "parity_input_source": input_source,
        "vram_instrument": (
            "nvidia-smi delta, not torch: torch's caching allocator cannot see ONNX "
            "Runtime allocations and reports 0.0 MB for a live CUDA session"
        ),
    }
    print(f"execution providers: available={status['available']}")
    print(f"                     active   ={status['active_on_probe']}")
    if status["fell_back_to_cpu"]:
        print("  ! fell back to the CPU provider; GPU latency is not being measured")
    print()

    for ckpt in sorted(fer_dir.glob("*.pt")):
        spec, model = load_fer(ckpt)
        fp32_path = out_dir / f"{spec.name}.onnx"
        int8_path = out_dir / f"{spec.name}.int8.onnx"

        # Sampled before the session exists, so the CUDA context and the model
        # weights are inside the measured delta rather than outside it.
        baseline = V.nvidia_smi_mb()

        X.export_fp32(model, inputs[0], fp32_path)
        s_fp32 = X._session(fp32_path)
        r32 = X.compare(
            model,
            s_fp32,
            inputs,
            name=spec.name,
            precision="fp32",
            tolerance=X.FP32_TOLERANCE,
        )

        X.quantize_int8(fp32_path, int8_path)
        s_int8 = X._session(int8_path)
        r8 = X.compare(
            model,
            s_int8,
            inputs,
            name=spec.name,
            precision="int8",
            tolerance=X.INT8_P95_TOLERANCE,
        )

        # Resident footprint of each, with the budget enforced. Measured with
        # nvidia-smi, because torch cannot see ORT's allocations.
        for sess, res, path in ((s_fp32, r32, fp32_path), (s_int8, r8, int8_path)):
            feed = {sess.get_inputs()[0].name: inputs[0]}
            res.vram_mb = RT.measure_session_vram(sess, feed, warmup=5, baseline_mb=baseline)
            X.enforce_budget(res, ceiling_mb=args.ceiling_mb)
            res.detail = (res.detail + f"; {path.stat().st_size / 1024:.0f} KB").strip("; ")

        reduction = X.size_reduction(fp32_path, int8_path)
        extra.setdefault("size", {})[spec.name] = {
            "fp32_kb": round(fp32_path.stat().st_size / 1024, 1),
            "int8_kb": round(int8_path.stat().st_size / 1024, 1),
            "reduction_pct": round(100 * reduction, 1),
        }

        results += [r32, r8]
        print(r32.summary())
        print(r8.summary())
        print(
            f"  size {fp32_path.stat().st_size / 1024:.0f} KB -> "
            f"{int8_path.stat().st_size / 1024:.0f} KB ({100 * reduction:.0f}% smaller)"
        )

    X.save_report(results, out_dir / "parity.json", extra=extra)

    failed = [r for r in results if not r.passed]
    print()
    print(V.budget_note())
    if failed:
        print(f"\n{len(failed)} of {len(results)} parity checks FAILED")
        for r in failed:
            print(f"  {r.name} [{r.precision}]: {r.summary()}")
        return 1
    print(f"\nall {len(results)} parity checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
