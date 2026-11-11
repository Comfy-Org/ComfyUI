import json

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.utils
from comfy import ops
from comfy.model_patcher import ModelPatcher
from comfy.quant_ops import QuantizedTensor
from comfy_api.torch_helpers import set_torch_compile_wrapper
from comfy_api.torch_helpers.torch_compile import TORCH_COMPILE_KWARGS


class TinyDiffusion(torch.nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.proj = operations.Linear(8, 8, device="cpu", dtype=torch.bfloat16)

    def forward(self, x):
        return self.proj(x)


class TinyModel(torch.nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.diffusion_model = TinyDiffusion(operations)


def load_quantized_model(quant_format, make_weight):
    """Build a tiny model whose projection weight is a QuantizedTensor, like an INT8 checkpoint load."""
    layer_quant_config = {"diffusion_model.proj": {"format": quant_format, "params": {}}}
    state_dict = {
        "diffusion_model.proj.weight": make_weight(),
        "diffusion_model.proj.bias": torch.randn(8, dtype=torch.bfloat16),
        "diffusion_model.proj.weight_scale": torch.tensor(0.5, dtype=torch.float32),
    }
    state_dict, _ = comfy.utils.convert_old_quants(
        state_dict, metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})}
    )
    model = TinyModel(operations=ops.mixed_precision_ops({}))
    model.load_state_dict(state_dict, strict=False)
    return model


def make_patcher(model, load_device):
    return ModelPatcher(model, load_device=load_device, offload_device=torch.device("cpu"))


def test_rejects_int8_quantized_weights_on_cuda():
    model = load_quantized_model(
        "int8_tensorwise", lambda: torch.randint(-127, 127, (8, 8), dtype=torch.int8)
    )
    assert isinstance(model.diffusion_model.proj.weight, QuantizedTensor)
    patcher = make_patcher(model, torch.device("cuda"))
    with pytest.raises(ValueError, match=r"diffusion_model\.proj\.weight"):
        set_torch_compile_wrapper(model=patcher, backend="inductor")


def test_rejects_fp8_quantized_weights_on_cuda():
    model = load_quantized_model(
        "float8_e4m3fn", lambda: torch.randn(8, 8).to(torch.float8_e4m3fn)
    )
    assert isinstance(model.diffusion_model.proj.weight, QuantizedTensor)
    patcher = make_patcher(model, torch.device("cuda"))
    with pytest.raises(ValueError, match="torch.compile cannot be applied"):
        set_torch_compile_wrapper(model=patcher, backend="inductor")


def test_allows_quantized_weights_on_cpu():
    # On non-CUDA devices the quantized kernels are pure torch ops that Dynamo can trace.
    model = load_quantized_model(
        "int8_tensorwise", lambda: torch.randint(-127, 127, (8, 8), dtype=torch.int8)
    )
    patcher = make_patcher(model, torch.device("cpu"))
    set_torch_compile_wrapper(model=patcher, backend="inductor")
    assert TORCH_COMPILE_KWARGS in patcher.model_options


def test_allows_plain_weights_on_cuda():
    model = TinyModel(operations=ops.mixed_precision_ops({}))
    patcher = make_patcher(model, torch.device("cuda"))
    set_torch_compile_wrapper(model=patcher, backend="inductor")
    assert TORCH_COMPILE_KWARGS in patcher.model_options
