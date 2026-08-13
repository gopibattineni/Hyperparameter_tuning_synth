"""Generator registry: string name → BaseGenerator subclass."""

from __future__ import annotations

from typing import Dict, Type

from .base import BaseGenerator

# Populated as wrappers are implemented (Steps 3, 6, 7).
_REGISTRY: Dict[str, Type[BaseGenerator]] = {}


def register(cls: Type[BaseGenerator]) -> Type[BaseGenerator]:
    """Class decorator that registers ``cls.name`` in the global registry."""
    key = getattr(cls, "name", None)
    if not key or key == "base":
        raise ValueError(f"{cls.__name__} must define a non-empty class attribute `name`.")
    if key in _REGISTRY and _REGISTRY[key] is not cls:
        raise KeyError(f"Generator name already registered: {key!r}")
    _REGISTRY[key] = cls
    return cls


def get_generator_class(name: str) -> Type[BaseGenerator]:
    """Resolve a registry key to its generator class."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered yet)"
        raise KeyError(
            f"Unknown generator {name!r}. Known: {known}. "
            "Wrappers are registered in later implementation steps."
        ) from exc


def list_generators() -> list[str]:
    """Return sorted registered generator names."""
    return sorted(_REGISTRY)


def create_generator(name: str, params: dict | None = None) -> BaseGenerator:
    """Instantiate a registered generator from name + params."""
    cls = get_generator_class(name)
    return cls.from_params(params or {})
