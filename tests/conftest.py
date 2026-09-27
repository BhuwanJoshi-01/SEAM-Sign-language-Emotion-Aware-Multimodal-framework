"""Shared fixtures.

Tests that need the real datasets are marked ``network`` or ``slow`` and skipped
by default, because the default suite must stay runnable on a machine with no
network and 3 GB of free RAM. What runs by default is the part that encodes
contracts - the ones whose violation produces plausible-but-wrong numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from seam.perception.tasks_api import FACE_BLENDSHAPES, PART_SLICES, TOTAL_POINTS

#: A two-row slice of the real EmoSign CSV, transcribed so the join, the signer
#: normalisation and the intensity parsing are tested without the network. Both
#: naming forms appear, because 7 of the 200 real rows use the ``<n>-<Signer>-``
#: form and a fixture that only has the common one would not have caught the bug
#: that did.
EMOSIGN_ROWS = [
    {
        "video_name": "Cory_2013-6-25_sc0106_1014088",
        "Sentiment": "3",
        "joy": "4",
        "excited": "1",
        "surprise_pos": "1",
        "surprise_neg": "1",
        "worry": "1",
        "sadness": "1",
        "fear": "1",
        "disgust": "1",
        "frustration": "1",
        "anger": "1",
        "Reasoning_1": "raised brows, head shake",
        "Reasoning_2": "wide eyes",
        "Reasoning_3": "smile",
    },
    {
        "video_name": "10-Ben-Conclusion_24363254",
        "Sentiment": "-1",
        "joy": "1",
        "excited": "1",
        "surprise_pos": "1",
        "surprise_neg": "1",
        "worry": "3",
        "sadness": "1",
        "fear": "1",
        "disgust": "1",
        "frustration": "2",
        "anger": "1",
        "Reasoning_1": "furrowed brows",
        "Reasoning_2": "",
        "Reasoning_3": "pursed lips",
    },
    {
        "video_name": "Rachel_2011-05-31_sc12_777",
        "Sentiment": "0",
        "joy": "1",
        "excited": "1",
        "surprise_pos": "1",
        "surprise_neg": "1",
        "worry": "1",
        "sadness": "1",
        "fear": "1",
        "disgust": "1",
        "frustration": "1",
        "anger": "1",
        "Reasoning_1": "flat expression",
        "Reasoning_2": "",
        "Reasoning_3": "",
    },
]


def _write_emosign_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def emosign_csv(tmp_path: Path) -> Path:
    """A minimal EmoSign label file at the path the loader expects."""
    return _write_emosign_csv(tmp_path / "emosign" / "emosign_dataset.csv", EMOSIGN_ROWS)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "seam_data"
    _write_emosign_csv(root / "emosign" / "emosign_dataset.csv", EMOSIGN_ROWS)
    return root


@pytest.fixture
def real_data_root() -> Path:
    """The real data root, skipped when the datasets have not been fetched."""
    from seam.paths import default_data_root

    root = default_data_root()
    if not (root / "emosign" / "emosign_dataset.csv").is_file():
        pytest.skip("EmoSign labels not fetched; run `seam data fetch --all`")
    return root


def fake_arrays(n_frames: int = 30, *, face_rate: float = 1.0, seed: int = 0) -> dict:
    """A synthetic shard payload with the real array contract."""
    rng = np.random.default_rng(seed)
    presence = np.ones((n_frames, 4), dtype=bool)
    presence[:, 3] = rng.random(n_frames) < face_rate
    shapes = np.zeros((n_frames, FACE_BLENDSHAPES), dtype=np.float32)
    shapes[:, 1:12] = rng.random((n_frames, 11)).astype(np.float32) * 0.4
    return {
        "landmarks": rng.standard_normal((n_frames, TOTAL_POINTS, 3)).astype(np.float16),
        "blendshapes": shapes,
        "presence": presence,
    }


@pytest.fixture
def shard_arrays() -> dict:
    return fake_arrays()


def write_shard(root: Path, utterance_id: str, arrays: dict, meta: dict) -> Path:
    """Write a shard + sidecar the way ``extract.save_shard`` does."""
    import numpy as np

    root.mkdir(parents=True, exist_ok=True)
    npz = root / f"{utterance_id}.npz"
    with npz.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    (root / f"{utterance_id}.json").write_text(json.dumps(meta), encoding="utf-8")
    return npz


@pytest.fixture
def part_slices_total() -> int:
    """Sanity constant so tests can assert the ordering contract is contiguous."""
    ends = [end for _, end in PART_SLICES.values()]
    assert ends[-1] == TOTAL_POINTS
    return TOTAL_POINTS
