"""
Configuration management.

Loads config.yaml once and exposes it as a nested namespace.
Every module imports `cfg` from here instead of hardcoding paths
or thresholds, which keeps the project reproducible and auditable.
"""

from pathlib import Path
from types import SimpleNamespace

import yaml


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to SimpleNamespace for dot-access."""
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = _dict_to_namespace(value)
    return SimpleNamespace(**d)


def load_config(config_path: str | Path | None = None) -> SimpleNamespace:
    """Load YAML config file and return as a dot-accessible namespace.

    Parameters
    ----------
    config_path : str or Path, optional
        Explicit path to config.yaml. If None, walks up from this file
        to find the project root (the directory containing 'configs/').

    Returns
    -------
    SimpleNamespace
        Nested namespace with all configuration values.
    """
    if config_path is None:
        # Walk up from src/utils/config.py -> src/utils -> src -> project root
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "configs" / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    return _dict_to_namespace(raw)


# Module-level singleton so every import gets the same object
cfg = load_config()