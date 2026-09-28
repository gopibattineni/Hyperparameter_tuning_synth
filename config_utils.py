"""Shared YAML / path helpers for the HPO framework."""

from __future__ import annotations

import os
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


def _looks_like_synth_root(path: Path) -> bool:
    return (
        (path / "_vendor" / "tab-ddpm" / "CTAB-GAN-Plus" / "model" / "ctabgan.py").is_file()
        or (path / "Generators" / "Other GANS" / "CTAB-GAN-Plus" / "model" / "ctabgan.py").is_file()
        or (path / "Datasets").is_dir()
    )


def _resolve_repo_root() -> Path:
    """Locate SYNTH_BENCHMARK (vendor generators + local CSVs).

    Prefers ``SYNTH_REPO_ROOT``, then this machine, then the original
    ``gopi.battineni`` layout so scripts work after a home-directory move.
    """
    env = os.environ.get("SYNTH_REPO_ROOT")
    if env:
        env_path = Path(env).expanduser()
        if env_path.is_dir():
            return env_path

    candidates = [
        Path("/home/gopi_b/SYNTH_BENCHMARK"),
        Path("/home/gopi.battineni/SYNTH_BENCHMARK/SYNTH"),
        Path("/home/gopi.battineni/SYNTH_BENCHMARK"),
        PACKAGE_ROOT.parent / "SYNTH_BENCHMARK",
        PACKAGE_ROOT.parent / "SYNTH",
    ]
    for candidate in candidates:
        if candidate.is_dir() and _looks_like_synth_root(candidate):
            return candidate
    return candidates[0]


# Benchmark data / generators / vendor live under SYNTH_BENCHMARK
REPO_ROOT = _resolve_repo_root()


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
