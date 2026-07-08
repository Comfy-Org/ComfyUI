import torch
import logging

from comfy.cli_args import args

try:
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import (
        QuantizedTensor,
        QuantizedLayout,
        TensorCoreFP8Layout as _CKFp8Layout,
        TensorCoreNVFP4Layout as _CKNvfp4Layout,
        TensorWiseINT8Layout as _CKTensorWiseINT8Layout,
        register_layout_op,
        register_layout_class,
        get_layout_class,
    )
    _CK_AVAILABLE = True
    if torch.version.cuda is None:
        ck.registry.disable("cuda")
    else:
        cuda_version = tuple(map(int, str(torch.version.cuda).split('.')))
        if cuda_version < (13,):
            ck.registry.disable("cuda")
            logging.warning("WARNING: You need pytorch with cu130 or higher to use optimized CUDA operations.")

    if args.enable_triton_backend:
        try:
            import triton
            logging.info("Found triton %s. Enabling comfy-kitchen triton backend.", triton.__version__)
        except ImportError as e:
            logging.error(f"Failed to import triton, Error: {e}, the comfy-kitchen triton backend will not be available.")
            ck.registry.disable("triton")
    else:
        ck.registry.disable("triton")
    for k, v in ck.list_backends().items():
        logging.info(f"Found comfy_kitchen backend {k}: {v}")
except ImportError as e:
    logging.error(f"Failed to import comfy_kitchen, Error: {e}, fp8 and fp4 support will not be available.")
    _CK_AVAILABLE = False

    class QuantizedTensor:
        pass

    class _CKFp8Layout:
        pass

    class _CKNvfp4Layout:
        pass

    class _CKTensorWiseINT8Layout:
        pass

    def register_layout_class(name, cls):
        pass

    def get_layout_class(name):
        return None

_CK_MXFP8_AVAILABLE = False
if _CK_AVAILABLE:
    try:
        from comfy_kitchen.tensor import TensorCoreMXFP8Layout as _CKMxfp8Layout
        _CK_MXFP8_AVAILABLE = True
    except ImportError:
        logging.warning("comfy_kitchen does not support MXFP8, please update comfy_kitchen.")

if not _CK_MXFP8_AVAILABLE:
    class _CKMxfp8Layout:
        pass

import comfy.float

# ==============================================================================
# FP8 Layouts with Comfy-Specific Extensions
# ==============================================================================

class _TensorCoreFP8LayoutBase(_CKFp8Layout):
    FP8_DTYPE = None  # Must be overridden in subclass

    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        if cls.FP8_DTYPE is None:
            raise NotImplementedError(f"{cls.__name__} must define FP8_DTYPE")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if isinstance(scale, str) and scale == "recalculate":
            scale = torch.amax(tensor.abs()).to(dtype=torch.float32) / torch.finfo(cls.FP8_DTYPE).max
            if tensor.dtype not in [torch.float32, torch.bfloat16]:  # Prevent scale from being too small
                tensor_info = torch.finfo(tensor.dtype)
                scale = (1.0 / torch.clamp((1.0 / scale), min=tensor_info.min, max=tensor_info.max))

        if scale is None:
            scale = torch.ones((), device=tensor.device, dtype=torch.float32)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, device=tensor.device, dtype=torch.float32)

        if stochastic_rounding > 0:
            if inplace_ops:
                tensor *= (1.0 / scale).to(tensor.dtype)
            else:
                tensor = tensor * (1.0 / scale).to(tensor.dtype)
            qdata = comfy.float.stochastic_rounding(tensor, dtype=cls.FP8_DTYPE, seed=stochastic_rounding)
        else:
            qdata = ck.quantize_per_tensor_fp8(tensor, scale, cls.FP8_DTYPE)

        params = cls.Params(scale=scale.float(), orig_dtype=orig_dtype, orig_shape=orig_shape)
        return qdata, params


class TensorCoreMXFP8Layout(_CKMxfp8Layout):
    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        if tensor.dim() != 2:
            raise ValueError(f"MXFP8 requires 2D tensor, got {tensor.dim()}D")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        padded_shape = cls.get_padded_shape(orig_shape)
        needs_padding = padded_shape != orig_shape

        if stochastic_rounding > 0:
            qdata, block_scale = comfy.float.stochastic_round_quantize_mxfp8_by_block(tensor, pad_32x=needs_padding, seed=stochastic_rounding)
        else:
            qdata, block_scale = ck.quantize_mxfp8(tensor, pad_32x=needs_padding)

        params = cls.Params(
            scale=block_scale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
        )
        return qdata, params


class TensorCoreNVFP4Layout(_CKNvfp4Layout):
    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        if tensor.dim() != 2:
            raise ValueError(f"NVFP4 requires 2D tensor, got {tensor.dim()}D")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if scale is None or (isinstance(scale, str) and scale == "recalculate"):
            scale = torch.amax(tensor.abs()) / (ck.float_utils.F8_E4M3_MAX * ck.float_utils.F4_E2M1_MAX)

        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
        scale = scale.to(device=tensor.device, dtype=torch.float32)

        padded_shape = cls.get_padded_shape(orig_shape)
        needs_padding = padded_shape != orig_shape

        if stochastic_rounding > 0:
            qdata, block_scale = comfy.float.stochastic_round_quantize_nvfp4_by_block(tensor, scale, pad_16x=needs_padding, seed=stochastic_rounding)
        else:
            qdata, block_scale = ck.quantize_nvfp4(tensor, scale, pad_16x=needs_padding)

        params = cls.Params(
            scale=scale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
            block_scale=block_scale,
        )
        return qdata, params


class TensorCoreFP8E4M3Layout(_TensorCoreFP8LayoutBase):
    FP8_DTYPE = torch.float8_e4m3fn


class TensorCoreFP8E5M2Layout(_TensorCoreFP8LayoutBase):
    FP8_DTYPE = torch.float8_e5m2


# Backward compatibility alias - default to E4M3
TensorCoreFP8Layout = TensorCoreFP8E4M3Layout
TensorWiseINT8Layout = _CKTensorWiseINT8Layout


# ==============================================================================
# Registry
# ==============================================================================

register_layout_class("TensorCoreFP8Layout", TensorCoreFP8Layout)
register_layout_class("TensorCoreFP8E4M3Layout", TensorCoreFP8E4M3Layout)
register_layout_class("TensorCoreFP8E5M2Layout", TensorCoreFP8E5M2Layout)
register_layout_class("TensorCoreNVFP4Layout", TensorCoreNVFP4Layout)
register_layout_class("TensorWiseINT8Layout", _CKTensorWiseINT8Layout)
if _CK_MXFP8_AVAILABLE:
    register_layout_class("TensorCoreMXFP8Layout", TensorCoreMXFP8Layout)

QUANT_ALGOS = {
    "float8_e4m3fn": {
        "storage_t": torch.float8_e4m3fn,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreFP8E4M3Layout",
    },
    "float8_e5m2": {
        "storage_t": torch.float8_e5m2,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreFP8E5M2Layout",
    },
    "nvfp4": {
        "storage_t": torch.uint8,
        "parameters": {"weight_scale", "weight_scale_2", "input_scale"},
        "comfy_tensor_layout": "TensorCoreNVFP4Layout",
        "group_size": 16,
    },
}

if _CK_MXFP8_AVAILABLE:
    QUANT_ALGOS["mxfp8"] = {
        "storage_t": torch.float8_e4m3fn,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreMXFP8Layout",
        "group_size": 32,
    }

QUANT_ALGOS["int8_tensorwise"] = {
    "storage_t": torch.int8,
    "parameters": {"weight_scale"},
    "comfy_tensor_layout": "TensorWiseINT8Layout",
    "quantize_input": False,
}


# ==============================================================================
# Re-exports for backward compatibility
# ==============================================================================

__all__ = [
    "QuantizedTensor",
    "QuantizedLayout",
    "TensorCoreFP8Layout",
    "TensorCoreFP8E4M3Layout",
    "TensorCoreFP8E5M2Layout",
    "TensorCoreNVFP4Layout",
    "TensorWiseINT8Layout",
    "QUANT_ALGOS",
    "register_layout_op",
]
