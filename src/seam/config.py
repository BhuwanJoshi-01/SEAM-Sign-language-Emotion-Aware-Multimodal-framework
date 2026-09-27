"""YAML configuration loading with deep merge and dot access.

Config is the single source of truth for every path and hyperparameter. The rule
from the plan is "config-driven, seeded, reproducible", which is unenforceable if
a module can read an environment variable directly - so nothing else in the
package calls ``os.environ`` for anything except the three root overrides in
``paths.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or lacks a required key."""


class _Required:
    """Sentinel distinguishing "no default given" from "default is None"."""

    def __repr__(self) -> str:
        return "<required>"


REQUIRED = _Required()


class Config:
    """An immutable-ish nested dict with dot access and deep merge."""

    def __init__(self, data: dict[str, Any], *, source: Path | None = None) -> None:
        self._data = data
        self.source = source

    @classmethod
    def load(cls, path: str | Path) -> Config:
        """Load a YAML file, merging it over every ``defaults.yaml`` beside it.

        The merge chain is ``<dir>/defaults.yaml`` then the file itself, so a
        family of configs (all of ``configs/data/*.yaml``) can share one base
        without each repeating it.
        """
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config not found: {p}")

        merged: dict[str, Any] = {}
        base = p.parent / "defaults.yaml"
        if base.is_file():
            merged = _deep_merge(merged, _read_yaml(base))
        merged = _deep_merge(merged, _read_yaml(p))
        return cls(merged, source=p)

    def get(self, dotted: str, default: Any = REQUIRED) -> Any:
        """Return the value at ``dotted``, or ``default`` if it is absent.

        Raises rather than returning ``None`` for a missing required key: a
        silently-absent config key is how a run ends up training on the wrong
        dataset split.
        """
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is REQUIRED:
                    raise ConfigError(
                        f"missing required config key {dotted!r}"
                        + (f" in {self.source}" if self.source else "")
                    )
                return default
            node = node[part]
        return node

    def section(self, dotted: str) -> Config:
        """Return a sub-tree as its own Config, for passing to a component."""
        value = self.get(dotted)
        if not isinstance(value, dict):
            raise ConfigError(f"config key {dotted!r} is not a section")
        return Config(value, source=self.source)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def __contains__(self, dotted: str) -> bool:
        return self.get(dotted, None) is not None

    def __repr__(self) -> str:
        return f"Config(source={self.source}, keys={sorted(self._data)})"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path} must contain a mapping at the top level, got {type(data).__name__}"
        )
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
