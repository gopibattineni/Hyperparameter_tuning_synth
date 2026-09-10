"""Map YAML ``search_space`` entries to Optuna ``trial.suggest_*`` calls.

Each generator YAML documents paper-aligned ranges with::

    param:
      type: int | float | categorical
      suggest: int | float | categorical
      low / high / step / log / choices
      default: ...
      explanation: ...
"""

from __future__ import annotations

from typing import Any, Mapping

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore


def suggest_from_yaml(
    trial: "optuna.Trial",
    search_space: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Sample one concrete hyperparameter dict for the current trial.

    Parameters
    ----------
    trial:
        Optuna trial.
    search_space:
        The ``search_space`` mapping from a generator YAML.
    """
    if optuna is None:  # pragma: no cover
        raise ImportError("optuna is required for suggest_from_yaml")

    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        suggest = str(spec.get("suggest") or spec.get("type") or "categorical").lower()
        if suggest in {"categorical", "cat"}:
            choices = list(spec["choices"])
            # Optuna cannot hash lists directly — stringify nested dims, restore later
            if choices and isinstance(choices[0], (list, tuple)):
                labels = [str(list(c)) for c in choices]
                picked = trial.suggest_categorical(name, labels)
                params[name] = eval(picked, {"__builtins__": {}})  # noqa: S307 — controlled labels
            else:
                params[name] = trial.suggest_categorical(name, choices)
        elif suggest in {"int", "integer"}:
            low = int(spec["low"])
            high = int(spec["high"])
            step = spec.get("step")
            if step is None:
                params[name] = trial.suggest_int(name, low, high)
            else:
                params[name] = trial.suggest_int(name, low, high, step=int(step))
        elif suggest in {"float", "uniform", "loguniform"}:
            low = float(spec["low"])
            high = float(spec["high"])
            log = bool(spec.get("log", False))
            params[name] = trial.suggest_float(name, low, high, log=log)
        else:
            raise ValueError(f"Unsupported suggest type {suggest!r} for parameter {name!r}")
    return params


def restore_optuna_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Restore values Optuna stored as strings (list dims, YAML bools).

    Nested list categoricals are suggested as ``'[256, 256]'`` labels so
    Optuna can hash them. ``study.best_params`` therefore returns those
    strings, not the lists used during trials.
    """
    restored: dict[str, Any] = {}
    for name, value in params.items():
        if isinstance(value, str):
            stripped = value.strip()
            if stripped in {"true", "True"}:
                restored[name] = True
                continue
            if stripped in {"false", "False"}:
                restored[name] = False
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                restored[name] = eval(stripped, {"__builtins__": {}})  # noqa: S307
                continue
        restored[name] = value
    return restored


def defaults_from_yaml(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return default hyperparameters from a generator config.

    Prefer explicit ``defaults:`` block; otherwise fall back to each
    search-space entry's ``default`` field.
    """
    if "defaults" in cfg and isinstance(cfg["defaults"], dict):
        return dict(cfg["defaults"])
    space = cfg.get("search_space") or {}
    return {k: v["default"] for k, v in space.items() if "default" in v}


def describe_search_space(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten search-space metadata for documentation / CSV export."""
    rows = []
    for name, spec in (cfg.get("search_space") or {}).items():
        rows.append(
            {
                "parameter": name,
                "suggest": spec.get("suggest") or spec.get("type"),
                "low": spec.get("low"),
                "high": spec.get("high"),
                "choices": spec.get("choices"),
                "default": spec.get("default"),
                "log": spec.get("log", False),
                "explanation": (spec.get("explanation") or "").strip(),
            }
        )
    return rows
