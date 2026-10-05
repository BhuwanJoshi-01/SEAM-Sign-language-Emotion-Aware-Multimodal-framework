"""Stamp an artifact with the code that produced it, so staleness is detectable.

Why this exists. On 2026-10-04 the yaw decomposition in ``seam.features.markers`` was
fixed. Three artifacts computed with the broken one - the M4 factorizer runs, the M1
confound audit and the M3 marker labels - stayed on disk, cited, for a day, and one of
them was the basis of a "refuted" verdict. Nothing noticed, because an artifact carried
no record of which code wrote it and a green test suite says nothing about a JSON file
written last week.

A stamp is the git commit plus one fingerprint per ``seam`` module loaded when the result
was written. ``tests/test_artifact_staleness.py`` recomputes the fingerprints and fails
when a stamped artifact no longer matches the code in the tree.

The fingerprint is over the module's AST with docstrings removed, not over its bytes:
a comment, a docstring or a reformat must not invalidate a week of GPU time, while any
change to an expression, a constant or a signature must.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from seam.paths import project_root

#: The key a stamp is stored under in an artifact's top-level JSON object.
KEY = "_provenance"


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            # An empty body is a syntax error on re-parse but harmless for ast.dump.
            node.body = body[1:]


def fingerprint_source(source: str) -> str:
    """Hash of what `source` computes: its AST, minus comments and docstrings."""
    tree = ast.parse(source)
    _strip_docstrings(tree)
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()[:16]


def module_path(name: str) -> Path:
    """Source file of an importable module, without importing it."""
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(f"cannot locate source for module {name!r}")
    return Path(spec.origin)


def fingerprint_module(name: str) -> str:
    return fingerprint_source(module_path(name).read_text(encoding="utf-8"))


def fingerprint_file(path: Path) -> str:
    return fingerprint_source(Path(path).read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=project_root(),
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


#: Loaded ``seam`` modules that are deliberately left out of a stamp: the stamper itself
#: and the log formatter. Neither can change a number. Everything else that was imported
#: by the time the artifact is written is in.
NOT_PART_OF_A_RESULT = frozenset({"seam.provenance", "seam.logging"})


def loaded_modules() -> list[str]:
    """Every ``seam`` module imported so far in this process, minus the two above.

    The dependency list is read off the interpreter rather than typed in by hand. The
    first version of `stamp` took a hand-written list, and within an hour of being
    written one script's list had already left out a module its result depended on -
    the same "true by inspection" failure this file exists to end.
    """
    out = []
    for name, mod in sys.modules.items():
        if name != "seam" and not name.startswith("seam."):
            continue
        origin = getattr(mod, "__file__", None)
        if name in NOT_PART_OF_A_RESULT or origin is None:
            continue
        if Path(origin).name == "__init__.py" and not Path(origin).read_text().strip():
            continue
        out.append(name)
    return sorted(out)


def stamp(script: str | Path, *, also: Iterable[str] = ()) -> dict[str, object]:
    """Provenance record for an artifact written by `script`.

    Call it at the point of writing, after everything the result needed has been
    imported. It records the commit, whether the tree had uncommitted changes (a commit
    hash alone would claim more than is true), a fingerprint of every loaded ``seam``
    module, and a fingerprint of the script itself - a window size or a label rule
    living in the script is as much part of the result as anything in the library.
    ``also`` adds modules that are not imported in this process but shaped the inputs.
    """
    names = sorted(set(loaded_modules()) | set(also))
    record: dict[str, object] = {
        "git_sha": _git("rev-parse", "--short", "HEAD") or "unknown",
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "code": {name: fingerprint_module(name) for name in names},
    }
    path = Path(script).resolve()
    try:
        rel = str(path.relative_to(project_root()))
    except ValueError:
        rel = str(path)
    record["script"] = {"path": rel, "fingerprint": fingerprint_file(path)}
    return record


def stale_reasons(record: dict[str, object]) -> list[str]:
    """Why a stamped artifact no longer matches the tree; empty when it is current."""
    reasons: list[str] = []
    code = record.get("code")
    if not isinstance(code, dict) or not code:
        return ["stamp names no modules"]
    for name, recorded in sorted(code.items()):
        try:
            now = fingerprint_module(str(name))
        except ModuleNotFoundError:
            reasons.append(f"{name} no longer exists")
            continue
        if now != recorded:
            reasons.append(f"{name} changed since the artifact was written")
    script = record.get("script")
    if isinstance(script, dict):
        path = project_root() / str(script.get("path", ""))
        if not path.is_file():
            reasons.append(f"{script.get('path')} no longer exists")
        elif fingerprint_file(path) != script.get("fingerprint"):
            reasons.append(f"{script.get('path')} changed since the artifact was written")
    return reasons
