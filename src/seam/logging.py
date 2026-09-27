"""Console + file logging.

File logging is opt-in via ``SEAM_LOG_FILE`` because a benchmark that writes a log
line per frame will itself perturb the latency it is trying to measure.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup(level: int | str = logging.INFO, *, quiet: bool = False) -> None:
    """Install the root handlers. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(_FORMAT))
    if not quiet:
        root.addHandler(console)

    log_file = os.environ.get("SEAM_LOG_FILE")
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(fh)

    _CONFIGURED = True


def get(name: str) -> logging.Logger:
    """Return a named logger, configuring the root on first use."""
    if not _CONFIGURED:
        setup()
    return logging.getLogger(name)
