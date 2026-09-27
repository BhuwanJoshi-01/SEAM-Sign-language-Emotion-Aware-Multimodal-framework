"""Deterministic seeding.

Every number that reaches the paper is produced under one of these seeds, and the
seed is recorded alongside the run ID. Without that, a result that cannot be
reproduced is indistinguishable from a result that was never measured.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` (if importable).

    ``deterministic_torch`` additionally forces cuDNN into its deterministic
    algorithms. It costs some throughput, so the latency benchmark disables it -
    a reproducibility flag must not silently inflate the number it measures.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for DataLoader.

    This machine has ~3 GB of free RAM, so the project runs with
    ``num_workers: 0``. The hook exists anyway so that raising the worker count
    later cannot silently change the augmentation stream.
    """
    try:
        import torch

        base_seed = int(torch.initial_seed() % 2**32)
    except ImportError:  # pragma: no cover
        base_seed = 0
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
