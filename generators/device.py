"""GPU-only device helpers for neural generators.

Neural HPO must not silently fall back to host CPU. Set
``SYNTH_REQUIRE_CUDA=0`` only for deliberate CPU debugging.
"""

from __future__ import annotations

import os


# Pure-CPU synthesizers (no torch training loop).
CPU_ONLY_GENERATORS = frozenset({"gaussian_copula", "bootstrap_noise"})


def require_cuda_enabled() -> bool:
    return os.environ.get("SYNTH_REQUIRE_CUDA", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def assert_cuda_available(*, context: str = "") -> None:
    """Raise if CUDA is required but not visible to this process."""
    if not require_cuda_enabled():
        return
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None or str(cvd).strip() == "":
        raise RuntimeError(
            f"CUDA required{(' for ' + context) if context else ''} but "
            f"CUDA_VISIBLE_DEVICES is unset/empty. Refusing CPU fallback."
        )
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            f"CUDA required{(' for ' + context) if context else ''} but torch is not installed"
        ) from e
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA required{(' for ' + context) if context else ''} but "
            f"torch.cuda.is_available() is False "
            f"(CUDA_VISIBLE_DEVICES={cvd!r}). Refusing CPU fallback."
        )
    # Fail closed if torch somehow resolved to CPU-only tensors.
    if torch.device("cuda").type != "cuda":
        raise RuntimeError(
            f"CUDA required{(' for ' + context) if context else ''} but "
            "torch.device('cuda') did not resolve to a CUDA device."
        )


def cuda_device_str(*, context: str = "") -> str:
    assert_cuda_available(context=context)
    return "cuda"


def cuda_torch_device(*, context: str = ""):
    import torch

    assert_cuda_available(context=context)
    return torch.device("cuda")
