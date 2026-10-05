"""Fail when a cited result was computed by code that has since changed.

`tests/test_provenance.py` asks whether a number in the log came from *some* artifact. It
cannot ask whether that artifact still describes the code in the tree, and that is the
question that mattered on 2026-10-04: the yaw decomposition in `seam.features.markers`
was fixed, and the M4 factorizer runs, the M1 confound audit and the M3 marker labels -
all computed through the broken one - stayed on disk and stayed cited. 484 tests were
green throughout, because none of them reads a JSON file's history.

The canonical path under `artifacts/{audit,m3,m4,m5a}` always holds the current run. A
result that is overtaken is moved to `artifacts/superseded/<why>/` rather than deleted -
the append-only log cites its numbers - following the precedent of
`artifacts/fer_v1_broken_preproc/`.

Three rules, enforced against `paper/artifact_status.json`:

1. A stamped artifact at a canonical path must match the code it names.
2. A superseded artifact must live under `artifacts/superseded/`, give a reason, and
   name a canonical successor that exists and is stamped.
3. A canonical result with no stamp must be listed with a reason. The list is a ratchet -
   it may shrink as scripts gain stamps, and a new unstamped result fails the build.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seam import provenance as P

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
STATUS = REPO / "paper" / "artifact_status.json"

#: Directories whose JSON files are experimental results. Benchmarks, exports and data
#: manifests have their own gates and run-stamped names.
RESULT_DIRS = ("audit", "m3", "m4", "m5a")


# ---------------------------------------------------------------- the fingerprint itself


def test_a_comment_or_docstring_does_not_change_the_fingerprint() -> None:
    a = 'def f(x):\n    """Doc."""\n    return x + 1  # one\n'
    b = 'def f(x):\n    """A rewritten docstring."""\n\n    # moved\n    return x + 1\n'
    assert P.fingerprint_source(a) == P.fingerprint_source(b)


def test_a_changed_expression_changes_the_fingerprint() -> None:
    """The yaw bug was two swapped index pairs; that is the size of change to catch."""
    before = "def yaw(r):\n    return atan2(r[2][0], r[1][0])\n"
    after = "def yaw(r):\n    return atan2(r[1][0], r[0][0])\n"
    assert P.fingerprint_source(before) != P.fingerprint_source(after)


def test_a_changed_constant_changes_the_fingerprint() -> None:
    assert P.fingerprint_source("K = 1.5\n") != P.fingerprint_source("K = 4.5\n")


def test_a_current_stamp_is_not_stale_and_a_tampered_one_is() -> None:
    import seam.features.markers  # noqa: F401 - loaded so the stamp has to include it

    record = P.stamp(REPO / "scripts" / "label_markers.py")
    assert P.stale_reasons(record) == []
    assert "seam.features.markers" in record["code"]  # type: ignore[operator]

    record["code"]["seam.features.markers"] = "0" * 16  # type: ignore[index]
    reasons = P.stale_reasons(record)
    assert reasons == ["seam.features.markers changed since the artifact was written"]


def test_the_dependency_list_is_read_from_the_interpreter_not_typed() -> None:
    """A hand-written module list is a claim nobody checks.

    The first version of `stamp` took one, and the M4 script's list omitted the SignStream
    parser its human-label run reads. Whatever is imported when the artifact is written is
    what the artifact depended on.
    """
    import seam.data.signstream  # noqa: F401

    record = P.stamp(REPO / "scripts" / "train_factorizer.py")
    code = record["code"]
    assert "seam.data.signstream" in code  # type: ignore[operator]
    assert "seam.data.asllrp" in code, "and what that module imports"  # type: ignore[operator]
    assert "seam.provenance" not in code  # type: ignore[operator]
    assert isinstance(record["git_dirty"], bool)


def test_a_stamp_naming_no_modules_is_not_treated_as_current() -> None:
    """An empty dependency list would otherwise pass forever, which is no stamp at all."""
    assert P.stale_reasons({"git_sha": "abc", "code": {}})


# ---------------------------------------------------------------- the artifacts on disk


def _results() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sub in RESULT_DIRS:
        for path in sorted((ARTIFACTS / sub).rglob("*.json")):
            try:
                doc = json.loads(path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(doc, dict):
                out[str(path.relative_to(REPO))] = doc
    return out


@pytest.fixture(scope="module")
def results() -> dict[str, dict]:
    found = _results()
    if not found:
        pytest.skip("no result artifacts on disk; run `make repro` first")
    return found


@pytest.fixture(scope="module")
def status() -> dict:
    assert STATUS.is_file(), f"{STATUS.relative_to(REPO)} is missing"
    return json.loads(STATUS.read_text())


def test_every_stamped_artifact_matches_the_code_in_the_tree(results: dict[str, dict]) -> None:
    # A canonical path is never excused: an overtaken result is superseded by moving it.
    stale = {name: P.stale_reasons(doc[P.KEY]) for name, doc in results.items() if P.KEY in doc}
    stale = {k: v for k, v in stale.items() if v}
    assert not stale, (
        "these artifacts were written by code that has since changed:\n"
        + "\n".join(f"  {k}: {'; '.join(v)}" for k, v in sorted(stale.items()))
        + "\n\nRe-run the script that writes each one. If the old numbers are cited in "
        "EXPERIMENT_LOG.md, first move the old file under artifacts/superseded/<why>/, "
        "record it in paper/artifact_status.json, and append a superseding log entry."
    )


def test_a_superseded_artifact_names_a_current_successor(
    results: dict[str, dict], status: dict
) -> None:
    problems: list[str] = []
    for old, entry in sorted(status["superseded"].items()):
        if not old.startswith("artifacts/superseded/"):
            problems.append(f"{old}: a superseded artifact must be moved off the canonical path")
        if not str(entry.get("reason", "")).strip():
            problems.append(f"{old}: no reason given")
        new = entry.get("by", "")
        doc = results.get(new)
        if doc is None:
            problems.append(f"{old}: successor {new!r} is not a result on disk")
        elif P.KEY not in doc:
            problems.append(f"{old}: successor {new} carries no stamp")
    assert not problems, "\n".join(problems)


def test_every_unstamped_result_is_accounted_for(results: dict[str, dict], status: dict) -> None:
    known = set(status["unstamped"])
    unstamped = sorted(name for name, doc in results.items() if P.KEY not in doc)
    unknown = [name for name in unstamped if name not in known]
    assert not unknown, (
        f"result artifacts with no provenance stamp and no entry in {STATUS.name}: {unknown}. "
        "Stamp them in the script that writes them (seam.provenance.stamp)."
    )

    for name, reason in status["unstamped"].items():
        assert str(reason).strip(), f"{name}: an unstamped artifact needs a stated reason"

    # The ratchet: an entry whose artifact has since gained a stamp is dead weight.
    dead = sorted(n for n in status["unstamped"] if n in results and P.KEY in results[n])
    assert not dead, f"now stamped, remove from `unstamped` in {STATUS.name}: {dead}"
