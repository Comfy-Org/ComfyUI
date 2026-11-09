"""The numeric settings a pack's process takes from ComfyUI's, so pack code computes as it would in ComfyUI.

PyTorch's global switches (TF32, deterministic algorithms, matmul precision...) are per process, and so is the grad
mode a call runs in. A pack's process started with PyTorch's defaults computed differently from the same code in
ComfyUI: FDGuidance's guidance function ran with TF32 convolutions and gradients on, where ComfyUI has TF32 off and
inference mode on, and its output differed. ``snapshot`` is taken in ComfyUI and ``apply`` in the pack's process at
the handshake; ``call_mode`` / ``in_mode`` carry the grad / inference mode of each call into the pack.
"""
from __future__ import annotations

import contextlib
from typing import Any


def snapshot() -> dict:
    """This process's PyTorch numeric settings, as plain data."""
    try:
        import torch
    except ImportError:
        return {}
    settings: dict[str, Any] = {}

    def read(key: str, getter) -> None:
        try:
            settings[key] = getter()
        except Exception:
            pass

    read("cudnn_allow_tf32", lambda: bool(torch.backends.cudnn.allow_tf32))
    read("cudnn_benchmark", lambda: bool(torch.backends.cudnn.benchmark))
    read("cudnn_deterministic", lambda: bool(torch.backends.cudnn.deterministic))
    read("matmul_allow_tf32", lambda: bool(torch.backends.cuda.matmul.allow_tf32))
    read("matmul_fp16_reduced", lambda: bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction))
    read("matmul_bf16_reduced", lambda: bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction))
    read("float32_matmul_precision", lambda: str(torch.get_float32_matmul_precision()))
    read("deterministic", lambda: bool(torch.are_deterministic_algorithms_enabled()))
    read("deterministic_warn_only", lambda: bool(torch.is_deterministic_algorithms_warn_only_enabled()))
    return settings


def apply(settings: dict | None) -> list[str]:
    """Take ComfyUI's settings in this process. Returns the ones that couldn't be set."""
    if not settings:
        return []
    try:
        import torch
    except ImportError:
        return list(settings)
    failed = []

    def write(key: str, setter) -> None:
        if key in settings:
            try:
                setter(settings[key])
            except Exception:
                failed.append(key)

    write("cudnn_allow_tf32", lambda v: setattr(torch.backends.cudnn, "allow_tf32", bool(v)))
    write("cudnn_benchmark", lambda v: setattr(torch.backends.cudnn, "benchmark", bool(v)))
    write("cudnn_deterministic", lambda v: setattr(torch.backends.cudnn, "deterministic", bool(v)))
    write("matmul_allow_tf32", lambda v: setattr(torch.backends.cuda.matmul, "allow_tf32", bool(v)))
    write("matmul_fp16_reduced",
          lambda v: setattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", bool(v)))
    write("matmul_bf16_reduced",
          lambda v: setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", bool(v)))
    write("float32_matmul_precision", lambda v: torch.set_float32_matmul_precision(str(v)))
    if "deterministic" in settings:
        try:
            torch.use_deterministic_algorithms(bool(settings["deterministic"]),
                                               warn_only=bool(settings.get("deterministic_warn_only", False)))
        except Exception:
            failed.append("deterministic")
    return failed


def call_mode() -> dict:
    """The grad / inference mode the calling thread is in, for a call into a pack's process."""
    try:
        import torch
    except ImportError:
        return {}
    return {"grad": bool(torch.is_grad_enabled()), "inference": bool(torch.is_inference_mode_enabled())}


def in_mode(mode: dict | None):
    """A context running a call in the caller's grad / inference mode (as ComfyUI ran it)."""
    if not mode:
        return contextlib.nullcontext()
    try:
        import torch
    except ImportError:
        return contextlib.nullcontext()
    if mode.get("inference"):
        return torch.inference_mode(True)
    return torch.set_grad_enabled(bool(mode.get("grad", True)))
