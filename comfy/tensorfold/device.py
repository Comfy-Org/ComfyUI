import os

import torch


def require_cuda(device):
    if device.type != "cuda" or torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("TensorFold media requires an NVIDIA CUDA device on Linux.")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if torch.cuda.get_device_capability(index) not in ((12, 0), (12, 1)):
        raise RuntimeError("This TensorFold media build supports CUDA SM 120 and 121 (RTX 5090 and GB10).")
    return index


def subprocess_environment(device):
    index = require_cuda(device)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        selected = str(index)
    else:
        devices = [item.strip() for item in visible.split(",")]
        if index >= len(devices) or not devices[index] or devices[index] == "-1":
            raise RuntimeError("ComfyUI's CUDA device is outside CUDA_VISIBLE_DEVICES.")
        selected = devices[index]
    return {"CUDA_VISIBLE_DEVICES": selected}
