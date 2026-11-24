"""Canonical dataset catalogue for Multivariate TPE (must match Standard TPE).

Classification: 9 · Regression: 6 · Total: 15
"""

from __future__ import annotations

from datasets.loader import load_dataset_config

# Explicit keys (order matches config/datasets.yaml / Standard TPE).
CLASSIFICATION_DATASETS: list[str] = [
    "cancer",
    "alzheimers",
    "adult",
    "forest_cover",
    "bank_marketing",
    "wine_quality",
    "cdc_diabetes",
    "mushroom",
    "magic_gamma",
]

REGRESSION_DATASETS: list[str] = [
    "metro_interstate",
    "online_shopping",
    "air_quality",
    "concrete",
    "energy_efficiency",
    "real_estate",
]

ALL_DATASETS: list[str] = CLASSIFICATION_DATASETS + REGRESSION_DATASETS

# Generators used for the full sweep (ForestDiffusion excluded).
GENERATORS_NO_FOREST: list[str] = [
    "gaussian_copula",
    "copulagan",
    "ctgan",
    "ctabgan",
    "tvae",
    "wgan_gp",
    "tabddpm",
]


def assert_catalog() -> None:
    """Fail fast if YAML config drifts from the expected 9 + 6 layout."""
    cfg = load_dataset_config()
    cls = [k for k, v in cfg.items() if v.get("task") == "classification"]
    reg = [k for k, v in cfg.items() if v.get("task") == "regression"]
    if len(cls) != 9 or set(cls) != set(CLASSIFICATION_DATASETS):
        raise SystemExit(
            f"Expected 9 classification datasets {CLASSIFICATION_DATASETS}, got {cls}"
        )
    if len(reg) != 6 or set(reg) != set(REGRESSION_DATASETS):
        raise SystemExit(
            f"Expected 6 regression datasets {REGRESSION_DATASETS}, got {reg}"
        )
    if list(cfg.keys()) != ALL_DATASETS:
        # Order may differ; only require same set.
        if set(cfg.keys()) != set(ALL_DATASETS):
            raise SystemExit(
                f"Dataset key mismatch: config={list(cfg.keys())} expected={ALL_DATASETS}"
            )


def task_of(dataset: str) -> str:
    if dataset in CLASSIFICATION_DATASETS:
        return "classification"
    if dataset in REGRESSION_DATASETS:
        return "regression"
    raise KeyError(dataset)
