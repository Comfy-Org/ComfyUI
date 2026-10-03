import pytest  # noqa: F401
import torch

from comfy.cli_args import args


def _has_accelerator():
    """Whether torch can reach any device comfy.model_management would pick.

    Mirrors the detection in comfy/model_management.py, guarded the same way,
    so a machine with MPS or XPU is not pushed onto the CPU path.
    """
    if torch.cuda.is_available():
        return True
    try:
        if torch.xpu.is_available():
            return True
    except Exception:
        pass
    try:
        if torch.backends.mps.is_available():
            return True
    except Exception:
        pass
    return False


# Several test modules import comfy.model_management, directly or through a
# node module, and it picks a torch device at import time. Without this, those
# modules fail at collection with "Torch not compiled with CUDA enabled" on a
# CPU only install, and take their whole directory with them.
if not _has_accelerator():
    args.cpu = True
