"""Run on a disposable CUDA host with the matching Aimdo build.

Checks that a model patch on a plain (non-comfy_cast_weights) module is baked
into the weight by ModelPatcherDynamic.load, like the cast-module path does.
"""
import os
from pathlib import Path
import sys

os.environ.setdefault("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import comfy_aimdo.control as aimdo

assert aimdo.init("cuda") and aimdo.init_device(0)
from comfy import lora as comfy_lora, memory_management, model_patcher, ops

memory_management.aimdo_enabled = True
device = torch.device("cuda", 0)


class PlainNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.weight


model = torch.nn.Module()
model.norm = PlainNorm(8)
model.linear = ops.manual_cast.Linear(8, 8, dtype=torch.float32)
model.linear.weight = torch.nn.Parameter(torch.zeros(8, 8))

diff = torch.full((8,), 0.25)
patches = {}
patches.update(comfy_lora.load_lora({"norm.weight.diff": diff}, {"norm.weight": "norm.weight"}))
patches.update(comfy_lora.load_lora({"linear.weight.diff": torch.full((8, 8), 0.25)}, {"linear.weight": "linear.weight"}))

patcher = model_patcher.ModelPatcherDynamic(model, device, torch.device("cpu"))
assert set(patcher.add_patches(patches)) == {"norm.weight", "linear.weight"}
patcher.load(device)

params = dict(model.named_parameters())
assert torch.allclose(params["norm.weight"].detach().float().cpu(), torch.full((8,), 1.25)), \
    "the patch on the plain module was dropped"
assert torch.allclose(params["linear.weight"].detach().float().cpu(), torch.full((8, 8), 0.25)), \
    "the cast-module control patch was dropped"
