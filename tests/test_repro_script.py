"""`scripts/repro_all.sh` must run what it says it runs.

Until 2026-10-05 its `run` helper took a label and returned success without executing
anything when the label was not a requested stage. Six commands had labels of their own -
the face gate, both FER diagnostics, the sequential benchmark, the ablation and the
provenance guard - so `make repro` printed their stage headers, skipped them, and ended
with "done". The guard that fails the build on an untraced number had never run from the
script that exists to run it.

These tests drive the script with a stand-in interpreter that records its arguments, so
they exercise the control flow and nothing else.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "repro_all.sh"


def _run(tmp_path: Path, stages: str, exit_code: int = 0) -> tuple[int, list[str], str]:
    """Run the script with `PY` replaced by a recorder. Returns (rc, calls, output)."""
    log = tmp_path / "calls.log"
    stub = tmp_path / "py"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'printf \'%s\\n\' "$*" >> "{log}"\n'
        # The version probe at the top of the script must pass whatever is being simulated.
        'if [[ "$1" == "-c" ]]; then exit 0; fi\n'
        f"exit {exit_code}\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        env={**os.environ, "PY": str(stub), "SEAM_REPRO_STAGES": stages},
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return proc.returncode, calls, proc.stdout + proc.stderr


def test_every_command_of_a_requested_stage_runs(tmp_path: Path) -> None:
    rc, calls, out = _run(tmp_path, "audit,provenance")
    assert rc == 0, out
    joined = "\n".join(calls)
    for expected in (
        "scripts/run_confound_audit.py",
        "scripts/diagnose_fer_affect.py",
        "scripts/diagnose_fer_sensitivity.py",
        "tests/test_provenance.py",
        "tests/test_artifact_staleness.py",
    ):
        assert expected in joined, f"{expected} was skipped:\n{joined}"


def test_a_stage_that_was_not_requested_does_not_run(tmp_path: Path) -> None:
    rc, calls, out = _run(tmp_path, "grounding")
    assert rc == 0, out
    assert any("scripts/cue_grounding.py" in c for c in calls)
    assert not any("train_factorizer" in c or "run_confound_audit" in c for c in calls)


def test_the_m4_stage_runs_all_three_arms_into_separate_files(tmp_path: Path) -> None:
    """The ablation used to be skipped, and when run by hand overwrote the seed-0 file."""
    rc, calls, out = _run(tmp_path, "m4")
    assert rc == 0, out
    m4 = [c for c in calls if "train_factorizer.py" in c]
    assert len(m4) == 3, m4
    assert sum("--labels human" in c for c in m4) == 1
    assert sum("--ablate" in c and "--tag ablation" in c for c in m4) == 1


def test_an_unmet_m4_gate_is_a_result_but_a_crash_stops_the_run(tmp_path: Path) -> None:
    rc, _, out = _run(tmp_path, "m4,m5a", exit_code=1)
    # Exit 1 from train_factorizer means "gate not met". It is recorded, and the run goes
    # on - to the next stage, where the same exit code from another script is a failure.
    assert out.count("gate not met") == 3
    assert rc != 0 and "FAILED" in out and "train_recogniser.py" in out

    rc, _, out = _run(tmp_path, "m4", exit_code=2)
    assert rc != 0 and "FAILED" in out


def test_a_failing_command_fails_the_run(tmp_path: Path) -> None:
    rc, _, out = _run(tmp_path, "provenance", exit_code=1)
    assert rc != 0
    assert "FAILED" in out


def test_an_unknown_stage_name_is_an_error_not_a_silent_skip(tmp_path: Path) -> None:
    rc, calls, out = _run(tmp_path, "audit,provenence")
    assert rc != 0
    assert "unknown stage 'provenence'" in out
    assert not any("run_confound_audit" in c for c in calls), "nothing should run on a typo"


@pytest.mark.parametrize("stage", ["signstream", "landmarks"])
def test_multi_command_stages_run_both_commands(tmp_path: Path, stage: str) -> None:
    rc, calls, out = _run(tmp_path, stage)
    if stage == "signstream" and rc != 0 and "SignStream XML not found" in out:
        pytest.skip("SignStream XML is licence-gated and not on this machine")
    assert rc == 0, out
    scripted = [c for c in calls if not c.startswith("-c")]
    assert len(scripted) == 2, scripted
