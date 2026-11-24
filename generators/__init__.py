"""Public exports for generator wrappers."""

from .base import BaseGenerator
from .bootstrap import BootstrapNoiseGenerator
from .registry import create_generator, get_generator_class, list_generators, register

# Side-effect imports register concrete generators.
from . import bootstrap as _bootstrap  # noqa: F401
from . import sdv_wrappers as _sdv  # noqa: F401
from . import advanced_wrappers as _adv  # noqa: F401

from .sdv_wrappers import (
    CopulaGANGenerator,
    CTGANGenerator,
    GaussianCopulaGenerator,
    TVAEGenerator,
)
from .advanced_wrappers import (
    CTABGANGenerator,
    ForestDiffusionGenerator,
    TabDDPMGenerator,
    WGANGPGenerator,
)

__all__ = [
    "BaseGenerator",
    "BootstrapNoiseGenerator",
    "GaussianCopulaGenerator",
    "CopulaGANGenerator",
    "CTGANGenerator",
    "TVAEGenerator",
    "CTABGANGenerator",
    "WGANGPGenerator",
    "ForestDiffusionGenerator",
    "TabDDPMGenerator",
    "create_generator",
    "get_generator_class",
    "list_generators",
    "register",
]
