"""Centralized configuration loader.

Every script in this project loads its parameters through `load_config()`
rather than hard-coding paths, hyperparameters, or audio settings. See
configs/default.yaml for the single source of truth.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict:
    """Load the YAML config, optionally applying a dict of dotted-key overrides.

    Example:
        cfg = load_config(overrides={"training.batch_size": 32})
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        cfg = copy.deepcopy(cfg)
        for dotted_key, value in overrides.items():
            _set_dotted(cfg, dotted_key, value)

    return cfg


def _set_dotted(d: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def resolve_path(cfg: dict, relative_path: str) -> Path:
    """Resolve a config-relative path (e.g. cfg['dataset']['raw_dir']) against
    the project root, so scripts work regardless of the caller's cwd."""
    return PROJECT_ROOT / relative_path
