from dataclasses import dataclass
import math

import torch
from comfy_kitchen.tensor import BaseLayoutParams, QuantizedLayout, QuantizedTensor, register_layout_class


def backend():
    try:
        import bitsandbytes.functional as functional
    except ImportError as error:
        raise RuntimeError("NF4 loading requires bitsandbytes. Install it in the ComfyUI Python environment with: python -m pip install bitsandbytes") from error
    return functional


class NF4Layout(QuantizedLayout):
    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        code: torch.Tensor
        blocksize: int = 64
        quant_dtype: torch.dtype = torch.bfloat16

        def _tensor_fields(self):
            return ["scale", "code"]

    @classmethod
    def quantize(cls, tensor, **kwargs):
        raise NotImplementedError(
            "Marigold NF4 requires a prequantized checkpoint and separate LoRA computation; "
            "runtime quantization is unsupported")

    @classmethod
    def quant_state(cls, params):
        return backend().QuantState(absmax=params.scale, shape=params.orig_shape,
                                    code=params.code, blocksize=params.blocksize,
                                    quant_type="nf4", dtype=params.quant_dtype)

    @classmethod
    def dequantize(cls, qdata, params):
        return backend().dequantize_4bit(qdata, cls.quant_state(params)).to(params.orig_dtype)

    @classmethod
    def linear(cls, input, weight, bias):
        import bitsandbytes
        # Preserve bitsandbytes' single-vector kernel used by timestep embeddings.
        return bitsandbytes.matmul_4bit(input, weight._qdata.t(), bias=bias,
                                       quant_state=cls.quant_state(weight.params))

    @classmethod
    def get_plain_tensors(cls, tensor):
        return tensor._qdata, tensor.params.scale, tensor.params.code

    @classmethod
    def state_dict_tensors(cls, qdata, params):
        return {"": qdata, "_scale": params.scale, "_code": params.code}


register_layout_class("NF4Layout", NF4Layout)


def restore_weights(state_dict, layers):
    for name, config in layers.items():
        shape = config.get("shape", [])
        if (config.get("format") != "nf4" or config.get("blocksize") != 64
                or config.get("dtype") != "bfloat16" or config.get("double_quant") is not False
                or len(shape) != 2 or any(not isinstance(n, int) or n <= 0 for n in shape)):
            raise ValueError(f"Invalid Marigold NF4 metadata: {name}")
        keys = [f"{name}.weight", f"{name}.weight_scale", f"{name}.weight_code"]
        if any(key not in state_dict for key in keys):
            raise ValueError(f"Missing Marigold NF4 weights or metadata: {name}")
        packed, scale, code = (state_dict[key] for key in keys)
        size = math.prod(shape)
        if (packed.dtype != torch.uint8 or packed.numel() != (size + 1) // 2
                or scale.dtype != torch.float32 or scale.numel() != (size + 63) // 64
                or code.dtype != torch.float32 or code.shape != (16,)):
            raise ValueError(f"Incompatible Marigold NF4 tensors: {name}")
        params = NF4Layout.Params(scale, torch.bfloat16, tuple(shape), code)
        state_dict[keys[0]] = QuantizedTensor(packed.reshape(-1, 1), "NF4Layout", params)
        del state_dict[keys[1]], state_dict[keys[2]]
