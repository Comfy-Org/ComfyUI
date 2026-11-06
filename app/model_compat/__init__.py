"""Model compatibility: will the installed models run on this machine.

A weight file's size, read beside the GPU's VRAM and the machine's RAM, says
whether ComfyUI can hold it on the GPU, has to offload part of it to RAM, or
cannot load it at all. Everything is read locally; nothing goes over the network.
"""

from __future__ import annotations

import os
import re

import psutil
import torch
from aiohttp import web

import comfy.model_management
import folder_paths

GIB = 1024 ** 3
# Room the GPU needs beside the weights: CUDA context and activations at ~1 megapixel.
HEADROOM = 1.5 * GIB
# Share of RAM ComfyUI can reasonably fill with offloaded weights.
OFFLOAD_RAM_SHARE = 0.5
LOCAL_FOLDERS = ["checkpoints", "diffusion_models", "text_encoders", "vae", "loras", "controlnet", "clip_vision", "upscale_models"]

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def hardware() -> dict:
    device = comfy.model_management.get_torch_device()
    capability = torch.cuda.get_device_capability(device) if device.type == "cuda" else None
    return {
        # "cuda:0 NVIDIA GeForce RTX 5070 Ti : cudaMallocAsync" -> the card's own name
        "name": re.sub(r"^\w+:\d+\s+|\s+:\s+\w+$", "", comfy.model_management.get_torch_device_name(device)),
        "type": device.type,
        "vram_total": comfy.model_management.get_total_memory(device),
        "ram_total": psutil.virtual_memory().total,
        "capability": list(capability) if capability else None,
        "supports": {
            "bf16": bool(capability and capability >= (8, 0)),
            "fp8": bool(capability and capability >= (8, 9)),
            "fp4": bool(capability and capability >= (10, 0)),
        },
    }


def precision(name: str) -> str | None:
    lower = name.lower()
    if lower.endswith(".gguf"):
        flat = re.sub(r"[-.]", "_", lower)
        quant = re.search(r"(?:^|_)(i?q\d(?:_k)?(?:_(?:xxs|xs|nl|s|m|l|0|1))?|bf16|f16|f32)(?=_)", flat)
        return f"GGUF {quant.group(1).upper()}" if quant else "GGUF"
    for pattern, label in (
        (r"nvfp4|fp4|mxfp4", "FP4"),
        (r"fp8|e4m3|e5m2", "FP8"),
        (r"int8", "INT8"),
        (r"bf16", "BF16"),
        (r"fp16|f16|half", "FP16"),
        (r"fp32|f32", "FP32"),
    ):
        if re.search(pattern, lower):
            return label
    return None


def judge(size: int, label: str | None, hw: dict) -> dict:
    """fits: whole on the GPU; offload: runs with part of it in RAM, slower; no: too big."""
    vram, ram = hw["vram_total"], hw["ram_total"]
    if hw["type"] != "cuda":
        level = "offload" if size <= ram * OFFLOAD_RAM_SHARE else "no"
    elif size + HEADROOM <= vram:
        level = "fits"
    elif size <= vram - HEADROOM + ram * OFFLOAD_RAM_SHARE:
        level = "offload"
    else:
        level = "no"

    notes = []
    supports = hw["supports"]
    if label == "FP8" and not supports["fp8"]:
        notes.append("This GPU has no native fp8; ComfyUI upcasts it, with no speedup.")
    if label == "FP4" and not supports["fp4"]:
        notes.append("Fast FP4 needs an RTX 50 series (Blackwell) GPU; it runs slowly here.")
    if label == "BF16" and not supports["bf16"]:
        notes.append("This GPU has no bf16; the weights are cast to fp16/fp32.")
    if level == "offload":
        notes.append("Part of the weights stays in RAM: it runs, but slower.")
    return {"level": level, "notes": notes}


def describe(folder: str, name: str, size: int, hw: dict) -> dict:
    label = precision(os.path.basename(name))
    return {"folder": folder, "path": f"{folder}/{name}", "name": os.path.basename(name), "size": size,
            "precision": label, **judge(size, label, hw)}


class ModelCompatManager:
    WEB_DIR = WEB_DIR

    def add_routes(self, routes) -> None:
        @routes.get("/model_compat/hardware")
        async def get_hardware(request):
            return web.json_response(hardware())

        @routes.get("/model_compat/local")
        async def get_local_models(request):
            hw = hardware()
            files = []
            for folder in LOCAL_FOLDERS:
                for name in folder_paths.get_filename_list(folder):
                    full = folder_paths.get_full_path(folder, name)
                    if full and os.path.isfile(full):
                        files.append(describe(folder, name, os.path.getsize(full), hw))
            return web.json_response({"hardware": hw, "files": sorted(files, key=lambda f: (f["folder"], -f["size"]))})
