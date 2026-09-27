"""Config, paths, and the CLI surface."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from seam import config as config_mod
from seam import paths as paths_mod

# -- config -------------------------------------------------------------


def test_load_and_dot_access(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"train": {"batch": 32, "lr": 0.001}}))
    cfg = config_mod.Config.load(p)
    assert cfg.get("train.batch") == 32
    assert cfg.get("train.lr") == pytest.approx(0.001)
    assert cfg.get("train.epochs", 100) == 100


def test_missing_required_key_raises(tmp_path: Path) -> None:
    """A silently-absent key is how a run trains on the wrong split."""
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"train": {}}))
    cfg = config_mod.Config.load(p)
    with pytest.raises(config_mod.ConfigError, match="missing required config key"):
        cfg.get("train.batch")
    assert cfg.get("train.batch", None) is None


def test_dot_access_through_a_scalar_raises_for_a_required_key(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"train": 5}))
    cfg = config_mod.Config.load(p)
    with pytest.raises(config_mod.ConfigError):
        cfg.get("train.batch")
    assert cfg.get("train.batch", "d") == "d"


def test_section_returns_a_config(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"train": {"batch": 8}, "epochs": 100}))
    cfg = config_mod.Config.load(p)
    section = cfg.section("train")
    assert section.get("batch") == 8
    # A key that exists but is a scalar is the case that must be rejected.
    with pytest.raises(config_mod.ConfigError, match="not a section"):
        cfg.section("epochs")
    with pytest.raises(config_mod.ConfigError, match="missing required config key"):
        cfg.section("absent")


def test_defaults_file_is_merged_under_the_specific_file(tmp_path: Path) -> None:
    (tmp_path / "defaults.yaml").write_text(
        yaml.safe_dump({"train": {"batch": 32, "lr": 0.01}, "seed": 42})
    )
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({"train": {"batch": 8}}))
    cfg = config_mod.Config.load(tmp_path / "c.yaml")
    assert cfg.get("train.batch") == 8  # overridden
    assert cfg.get("train.lr") == 0.01  # inherited
    assert cfg.get("seed") == 42


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(config_mod.ConfigError, match="config not found"):
        config_mod.Config.load(tmp_path / "nope.yaml")


def test_non_mapping_top_level_raises(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(config_mod.ConfigError, match="mapping at the top level"):
        config_mod.Config.load(p)


# -- paths --------------------------------------------------------------


def test_project_root_finds_pyproject() -> None:
    root = paths_mod.project_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "seam").is_dir()


def test_env_override_is_honoured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    monkeypatch.setenv("SEAM_PROJECT_ROOT", str(tmp_path))
    assert paths_mod.project_root() == tmp_path.resolve()


def test_data_root_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SEAM_DATA_ROOT", str(tmp_path))
    assert paths_mod.default_data_root() == tmp_path.resolve()


def test_default_data_root_avoids_the_small_volume() -> None:
    """The repo volume has 36 GB; bulk artifacts must not land there by default."""
    root = paths_mod.default_data_root()
    assert "seam_data" in str(root)


def test_ensure_dir_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b"
    assert paths_mod.ensure_dir(target).is_dir()
    assert paths_mod.ensure_dir(target).is_dir()


# -- CLI ----------------------------------------------------------------


def test_parser_builds_and_requires_a_command() -> None:
    from seam.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    assert parser.parse_args(["doctor"]).command == "doctor"


def test_verify_accepts_the_gate_flag() -> None:
    from seam.cli import build_parser

    args = build_parser().parse_args(["--data-root", "/tmp", "data", "verify", "--gate", "M1"])
    assert args.gate == "M1"
    assert args.data_root == "/tmp"


def test_face_gate_sample_flag() -> None:
    from seam.cli import build_parser

    args = build_parser().parse_args(["landmarks", "face-gate", "--sample", "8"])
    assert args.sample == 8


def test_landmarks_extract_limit_flag() -> None:
    from seam.cli import build_parser

    args = build_parser().parse_args(["landmarks", "extract", "--limit", "3"])
    assert args.limit == 3
