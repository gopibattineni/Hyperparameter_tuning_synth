"""Shared YAML / path helpers for the HPO framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required for HPO configs. "
        "Install with: pip install pyyaml"
    ) from exc

# hyper parameter tuning/ package root
PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = PACKAGE_ROOT / "config"
# Benchmark data / generators / vendor live under SYNTH_BENCHMARK/SYNTH
REPO_ROOT = Path("/home/gopi.battineni/SYNTH_BENCHMARK/SYNTH")


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def load_config(name: str) -> dict[str, Any]:
    """Load a top-level config by stem name, e.g. ``\"objective\"`` → objective.yaml."""
    path = CONFIG_ROOT / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_yaml(path)


def load_generator_config(generator_name: str) -> dict[str, Any]:
    """Load ``config/generators/{generator_name}.yaml``."""
    path = CONFIG_ROOT / "generators" / f"{generator_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_yaml(path)


def list_generator_configs() -> list[str]:
    """Return stems of all generator YAML files."""
    return sorted(p.stem for p in (CONFIG_ROOT / "generators").glob("*.yaml"))
