"""Project-root and data-root resolution.

Every path in the project is derived from one root, never hardcoded. The
resolution order is deliberate:

1. ``SEAM_PROJECT_ROOT`` - explicit override, used by tests and by the bench
   harness when it needs to point at a scratch copy of the data.
2. An ancestor of this file containing ``pyproject.toml`` - the normal case.
3. The current working directory, as a last resort for a source checkout that
   has been moved without its metadata.

Data and artifacts are deliberately *not* under the project root by default. The
repository lives on a 36 GB volume while ``/mnt/DevProd`` has 110 GB free, and
the bulk artifacts (landmark shards, checkpoints, the 113 MB of EmoSign video)
are the things that fill a disk. ``configs/base.yaml`` overrides both.
"""

from __future__ import annotations

import os
from pathlib import Path

_MARKER = "pyproject.toml"


def project_root() -> Path:
    """Return the repository root."""
    env = os.environ.get("SEAM_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / _MARKER).is_file():
            return candidate

    cwd = Path.cwd().resolve()
    if (cwd / _MARKER).is_file():
        return cwd

    # Last resort: the package lives at <root>/src/seam/paths.py, so the root is
    # three levels up even if pyproject.toml is missing.
    return here.parents[2]


def default_data_root() -> Path:
    """Return the root for downloaded/derived data.

    Prefers the high-capacity volume when it exists, because the plan's storage
    rule is "video is transient, keypoints are the asset" and keypoint shards for
    tens of thousands of clips are still measured in gigabytes.
    """
    env = os.environ.get("SEAM_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    for candidate in ("/mnt/DevProd", "/mnt/Volume2"):
        path = Path(candidate)
        if path.is_dir() and os.access(path, os.W_OK):
            return path / "seam_data"
    return project_root() / "artifacts" / "data"


def artifacts_root() -> Path:
    """Return the root for checkpoints, reports, ONNX exports."""
    env = os.environ.get("SEAM_ARTIFACTS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return project_root() / "artifacts"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` and its parents if needed, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
