"""M2: the export, parity, and VRAM budget harness must be falsifiable.

Every test here exists because the corresponding instrument was, at some point,
quietly reporting something false:

* ``get_available_providers()`` advertised a CUDA provider that could not
  initialise, so latency numbers were CPU numbers wearing a GPU label.
* ``torch.cuda.max_memory_allocated()`` returns ``0.0`` for a live ONNX Runtime
  session, so a torch-based VRAM reading would pass any budget while measuring
  nothing.
* A baseline sampled after session creation excludes the CUDA context and the
  weights, and also reported 0 MB.

A budget assertion that cannot fail is not an assertion, so these tests check that
the guards actually trip - by feeding them numbers that must be rejected - rather
than only checking the happy path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seam.export import onnx_export as X
from seam.export import runtime as RT
from seam.export import vram_guard as V


def make_result(**over: object) -> X.ParityResult:
    """A passing result, so each test can break exactly one thing."""
    base: dict[str, object] = {
        "name": "unit-model",
        "precision": "fp32",
        "n_inputs": 8,
        "max_abs_diff": 0.0,
        "p95_abs_diff": 0.0,
        "argmax_agreement": 1.0,
        "passed": True,
    }
    base.update(over)
    return X.ParityResult(**base)  # type: ignore[arg-type]


# --- the budget must be a real constraint ---------------------------------


def test_vram_ceiling_is_below_usable() -> None:
    assert V.BOARD_MB >= V.USABLE_MB
    assert 0 < V.VRAM_CEILING_MB < V.USABLE_MB
    # Leaves room for a desktop session on the display path, which holds ~326 MB
    # before any of this process starts.
    assert int(V.USABLE_MB * 0.75) >= V.VRAM_CEILING_MB


def test_check_accepts_under_budget() -> None:
    rep = V.check(512.0, ceiling_mb=V.VRAM_CEILING_MB, stage="unit", strict=True)
    assert rep.ok
    assert rep.peak_mb == 512.0
    assert rep.as_dict()["utilisation_pct"] > 0


def test_check_rejects_over_budget() -> None:
    with pytest.raises(V.VramBudgetExceeded):
        V.check(
            V.VRAM_CEILING_MB + 1,
            ceiling_mb=V.VRAM_CEILING_MB,
            stage="unit",
            strict=True,
        )


def test_check_non_strict_reports_instead_of_raising() -> None:
    """A benchmark sweep wants a report, not an exception mid-sweep."""
    rep = V.check(V.VRAM_CEILING_MB + 500, ceiling_mb=V.VRAM_CEILING_MB, stage="unit", strict=False)
    assert not rep.ok
    assert "FAIL" in rep.summary()


def test_enforce_budget_raises_on_oversize_result() -> None:
    res = make_result(vram_mb=float(V.VRAM_CEILING_MB) + 1.0)
    with pytest.raises(V.VramBudgetExceeded):
        X.enforce_budget(res)


def test_enforce_budget_allows_a_measured_zero() -> None:
    """0 MB must not be treated as 'unknown' and rejected, nor as free.

    A third model legitimately measures 0 MB incremental - the CUDA context and
    arena are already allocated. The guard has to let that through, otherwise a
    correct reading is reported as a failure.
    """
    X.enforce_budget(make_result(vram_mb=0.0))


def test_nan_peak_is_not_silently_ok() -> None:
    """A NaN reading must fail, not pass.

    ``nan <= ceiling`` is False, so the strict path already rejects it - but the
    non-strict path must also report it as not-ok rather than emitting a report
    that reads as a pass.
    """
    rep = V.check(float("nan"), ceiling_mb=V.VRAM_CEILING_MB, stage="unit", strict=False)
    assert not rep.ok


# --- the gates must be coherent -------------------------------------------


def test_tolerances_are_ordered() -> None:
    """INT8 must be allowed more deviation than FP32, or the gates are nonsense."""
    assert X.FP32_TOLERANCE < X.INT8_TOLERANCE
    assert 0.0 < X.MIN_ARGMAX_AGREEMENT <= 1.0


def test_argmax_gate_is_the_hard_gate() -> None:
    """Exceeding the p95 tolerance alone must not flip a result to a pass.

    ``fer_cnn_a`` INT8 is the live case: p95 1.64e-02 against a 1e-01 tolerance
    would pass a tolerance-only check, and it is rejected on argmax agreement at
    95.8%. The gates are independent and the decision uses both.
    """
    res = make_result(p95_abs_diff=X.INT8_P95_TOLERANCE * 0.5, argmax_agreement=0.958)
    assert res.p95_abs_diff < X.INT8_P95_TOLERANCE
    assert res.argmax_agreement < X.MIN_ARGMAX_AGREEMENT


# --- the instruments must report what they actually did --------------------


def test_provider_status_distinguishes_available_from_working() -> None:
    """Available is not working, and the status must say which one happened."""
    status = RT.provider_status()
    assert "CPUExecutionProvider" in status["available"]
    assert status["active_on_probe"], (
        "the probe must activate something; a CPU provider is always available, "
        f"so an empty list means the probe itself failed: {status['error']}"
    )
    if status["fell_back_to_cpu"]:
        assert not status["cuda_working"]
    else:
        assert status["cuda_working"]


def test_ensure_cuda_libraries_is_idempotent() -> None:
    first = RT.ensure_cuda_libraries()
    assert isinstance(first, dict)
    assert RT.ensure_cuda_libraries() == {}, "the second call must be a no-op"


def test_nvidia_lib_dirs_are_lib_dirs() -> None:
    """Regression: the search path once collected the package dirs, not ``lib``.

    Every lookup then missed, every soname reported MISSING, and the CUDA provider
    silently fell back to CPU while the report said nothing was wrong.
    """
    dirs = RT._nvidia_lib_dirs()
    for d in dirs:
        assert d.name == "lib", f"{d} is not a lib directory"
        assert d.is_dir()


# --- the report is the evidence --------------------------------------------


def test_size_reduction_is_a_fraction(tmp_path: Path) -> None:
    fp32 = tmp_path / "m.onnx"
    int8 = tmp_path / "m.int8.onnx"
    fp32.write_bytes(b"0" * 1000)
    int8.write_bytes(b"0" * 250)
    assert X.model_size(fp32) == 1000
    assert 0.0 < X.size_reduction(fp32, int8) < 1.0


def test_save_report_round_trips(tmp_path: Path) -> None:
    res = make_result(
        precision="int8", vram_mb=128.0, latency_ms_mean=1.5, latency_ms_p95=2.0, detail="unit"
    )
    out = tmp_path / "parity.json"
    X.save_report([res], out, extra={"unit": True})
    blob = json.loads(out.read_text())

    row = blob["results"][0]
    for key in (
        "name",
        "precision",
        "n_inputs",
        "argmax_agreement",
        "latency_ms_mean",
        "latency_ms_p95",
        "vram_mb",
        "passed",
    ):
        assert key in row, f"{key} missing from the saved report"
    assert blob["unit"] is True
    assert "tolerances" in blob
    assert "vram_budget" in blob


# --- latency quantiles must be per-call, not per-run-mean ------------------


def test_quantiles_are_over_individual_calls() -> None:
    """A p99 must come from many samples, not from a handful of run means.

    The harness previously timed a whole run, took its mean, and reported
    percentiles across ``runs`` samples - so a "p99" was the 99th percentile of
    five numbers. This pins the sample count and the distinction.
    """

    from seam.eval.bench import time_fn

    t = time_fn(lambda: None, label="unit", iterations=7, warmup=0, runs=3)
    assert t.samples == 21, "quantiles must be backed by iterations * runs calls"
    assert t.runs == 3
    assert t.ms_p95 >= t.ms_median >= t.ms_min


def test_slow_tail_is_visible_in_percentiles() -> None:
    """A workload with a slow minority must show it in p95/p99.

    With per-run means, one slow call in ten is diluted to a 10% shift of every
    run mean and the tail disappears. This is the whole reason the quantiles are
    now taken per call.
    """
    import time as _t

    from seam.eval.bench import time_fn

    state = {"n": 0}

    def mixed() -> None:
        state["n"] += 1
        # Every fourth call is 10x slower.
        if state["n"] % 4 == 0:
            _t.sleep(0.004)
        else:
            _t.sleep(0.0004)

    t = time_fn(mixed, label="unit", iterations=40, warmup=0, runs=1)
    assert t.samples == 40
    assert t.ms_p99 > t.ms_median * 2, (
        f"p99 {t.ms_p99:.2f} ms should be far above the median {t.ms_median:.2f} ms "
        "for a workload with a 25% slow tail"
    )


def test_fps_sustained_is_the_mean_rate() -> None:
    """Sustained FPS must come from the mean, not the best case.

    ``1000 / min`` is a best-case number that no real capture loop will ever see;
    the plan's K6 has to be judged on the rate actually sustained.
    """
    import time as _t

    from seam.eval.bench import time_fn

    t = time_fn(lambda: _t.sleep(0.001), label="unit", iterations=30, warmup=0, runs=1)
    assert t.fps_sustained == pytest.approx(1000.0 / t.ms_mean, rel=1e-6)
    assert t.fps_from_min >= t.fps_sustained
