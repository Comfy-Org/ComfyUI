"""Opt-in acceleration for Krea 2 Turbo on Windows SM120."""

from contextlib import contextmanager
from contextvars import ContextVar
import ctypes
from functools import lru_cache
import hashlib
from importlib.metadata import PackageNotFoundError, distribution
import logging
import os
from pathlib import Path

import torch
from comfy_kitchen.float_utils import to_blocked

from comfy import model_management
from comfy.quant_ops import QuantizedTensor


active_sample = ContextVar("krea2_sm120_sample", default=None)
TUNED_FP8_ALGORITHMS = {1536: 2, 6144: 0, 16384: 2}
TUNED_ROWS = 16862
WIDTH = 6144
WORKSPACE_SIZE = 32 << 20
CUBLASLT_SHA256 = "cad63434448e7141629e240ea093ad596a7ef6a0f67b468ba9bc1df6e1eeee33"

Pointer = ctypes.c_void_p
Size = ctypes.c_uint64


def current_sample():
    if torch.compiler.is_compiling():
        return None
    return active_sample.get()


def bind(library, name, arguments, returns=ctypes.c_int):
    function = getattr(library, name)
    function.argtypes = arguments
    function.restype = returns
    return function


class FP8Library:
    def __init__(self, path, cublas=None):
        dll = ctypes.CDLL(str(path))
        if cublas is not None:
            configure = bind(dll, "configure", [ctypes.c_wchar_p])
            status = configure(str(cublas))
            if status:
                raise RuntimeError(f"cuBLASLt configuration failed ({status})")
            self.candidates = bind(dll, "mx_candidates", [Pointer])
            self.select = bind(dll, "mx_select", [Pointer, ctypes.c_int], None)
        self.prepare = bind(dll, "prepare_mx", [ctypes.POINTER(Pointer), Size, Size, Size, ctypes.c_int,
                                               Pointer, Pointer, Pointer, ctypes.c_size_t, ctypes.c_int])
        self.run = bind(dll, "run_mx", [Pointer] * 8 + [ctypes.c_size_t, Pointer])
        self.destroy = bind(dll, "destroy_mx", [Pointer], None)


def unit_scales(device, rows, columns):
    # E8M0 exponent 127 encodes 1. The original per-tensor scales remain in alpha.
    scales = torch.full((rows, (columns + 31) // 32), 127, dtype=torch.uint8, device=device)
    return to_blocked(scales, flatten=False)


class FP8Plan:
    def __init__(self, library, x, weight, bias, workspace, algorithm):
        rows, columns = x.shape
        outputs = weight.shape[0]
        self.library = library
        self.handle = Pointer()
        self.input_scales = unit_scales(x.device, rows, columns)
        self.weight_scales = unit_scales(x.device, outputs, columns)
        self.zero = torch.zeros((), device=x.device, dtype=torch.float32)
        status = library.prepare(
            ctypes.byref(self.handle), rows, outputs, columns, 0,
            self.input_scales.data_ptr(), self.weight_scales.data_ptr(),
            None if bias is None else bias.data_ptr(), workspace.numel(), 1,
        )
        if status in (-1, 7, 15):
            return
        if status:
            raise RuntimeError(f"cuBLASLt FP8 plan failed ({status})")
        if algorithm is not None:
            if algorithm >= library.candidates(self.handle):
                self.close()
            else:
                library.select(self.handle, algorithm)

    def linear(self, x, weight, scale, bias, workspace, stream):
        x = x.contiguous()
        result = torch.empty((x.shape[0], weight.shape[0]), dtype=torch.float16, device=x.device)
        status = self.library.run(
            self.handle, x.data_ptr(), weight.data_ptr(), scale.data_ptr(), self.zero.data_ptr(),
            None if bias is None else bias.data_ptr(), result.data_ptr(),
            workspace.data_ptr(), workspace.numel(), stream,
        )
        if status:
            raise RuntimeError(f"cuBLASLt FP8 linear failed ({status})")
        return result

    def close(self):
        if self.handle:
            self.library.destroy(self.handle)
            self.handle = Pointer()


class Backend:
    def __init__(self, folder, cublas):
        checksum = hashlib.sha256()
        with cublas.open("rb") as binary:
            for chunk in iter(lambda: binary.read(1 << 20), b""):
                checksum.update(chunk)
        if checksum.hexdigest() != CUBLASLT_SHA256:
            raise RuntimeError("the tuned plans require the verified cuBLASLt 13.8.0.4 DLL")
        with os.add_dll_directory(str(Path(torch.__file__).parent / "lib")):
            self.default_fp8 = FP8Library(folder / "mxlt.dll")
            self.tuned_fp8 = FP8Library(folder / "mx.dll", cublas)
            half = ctypes.CDLL(str(folder / "half.dll"))
            rms = ctypes.CDLL(str(folder / "rms.dll"))
        status = bind(half, "configure", [ctypes.c_wchar_p])(str(cublas))
        if status:
            raise RuntimeError(f"cuBLASLt FP16 configuration failed ({status})")
        self.prepare_half = bind(half, "prepare", [ctypes.POINTER(Pointer), ctypes.c_int, ctypes.c_int,
                                                  ctypes.c_int, ctypes.c_size_t])
        self.run_half = bind(half, "run", [Pointer, ctypes.c_int, Pointer, Pointer, Pointer, Pointer,
                                          ctypes.c_size_t, Pointer])
        self.destroy_half = bind(half, "destroy_half", [Pointer], None)
        self.launch_rms = bind(rms, "launch_rms", [Pointer] * 3 + [ctypes.c_int] * 5 + [ctypes.c_float, Pointer])
        self.launch_affine = None
        if hasattr(rms, "launch_affine"):
            self.launch_affine = bind(rms, "launch_affine", [Pointer] * 4 + [Size, ctypes.c_int, Pointer])


@lru_cache(maxsize=1)
def load_backend(folder):
    try:
        if folder is True:
            folder = Path(distribution("comfyui-krea2-sm120").locate_file("krea2_sm120")).resolve()
            cublas = Path(distribution("nvidia-cublas").locate_file("nvidia/cu13/bin/x86_64/cublasLt64_13.dll")).resolve()
        else:
            folder = folder.resolve()
            cublas = folder / "cublasLt64_13_8.dll"
        return Backend(folder, cublas)
    except PackageNotFoundError:
        logging.warning("Krea 2 SM120 unavailable. Install requirements-krea2-sm120.txt to enable the profile.")
    except (OSError, ImportError, AttributeError, RuntimeError) as error:
        logging.warning("Krea 2 SM120 unavailable: %s", error)
    return None


class Sample:
    def __init__(self, backend):
        self.backend = backend
        self.workspaces = {}
        self.fp8_plans = {}
        self.half_plans = {}
        self.tuned_fp8_calls = self.other_fp8_calls = self.fp16_calls = 0
        self.rms_calls = self.affine_calls = self.text_preparations = 0

    def workspace(self, device, stream):
        key = device, stream
        if key not in self.workspaces:
            self.workspaces[key] = torch.empty(WORKSPACE_SIZE, device=device, dtype=torch.uint8)
        return self.workspaces[key]

    def fp8_linear(self, x, weight, scale_x, scale_weight, bias):
        rows, columns = x.shape
        outputs = weight.shape[0]
        if (
            not x.is_cuda or x.dtype != torch.float8_e4m3fn or weight.dtype != x.dtype
            or any(t.device != x.device for t in (weight, scale_x, scale_weight))
            or any(t.numel() != 1 or t.dtype != torch.float32 for t in (scale_x, scale_weight))
            or columns % 32 or outputs % 16 or rows < 32 or not weight.is_contiguous()
            or weight.shape[1] != columns or (bias is not None and (
                bias.dtype != torch.float16 or bias.device != x.device
                or bias.shape != (outputs,) or not bias.is_contiguous()))
        ):
            return None

        stream = torch.cuda.current_stream(x.device).cuda_stream
        workspace = self.workspace(x.device, stream)
        key = x.device, stream, rows, outputs, columns, bias is not None
        algorithm = TUNED_FP8_ALGORITHMS.get(outputs) if rows == TUNED_ROWS and columns == WIDTH and bias is None else None
        if key not in self.fp8_plans:
            library = self.backend.tuned_fp8 if algorithm is not None else self.backend.default_fp8
            self.fp8_plans[key] = FP8Plan(library, x, weight, bias, workspace, algorithm)
        plan = self.fp8_plans[key]
        if not plan.handle:
            return None
        result = plan.linear(x, weight, scale_x * scale_weight, bias, workspace, stream)
        if algorithm is None:
            self.other_fp8_calls += 1
        else:
            self.tuned_fp8_calls += 1
        return result

    def fp16_linear(self, x, weight, bias):
        if (
            bias is not None or type(x) is not torch.Tensor or type(weight) not in (torch.Tensor, torch.nn.Parameter)
            or not x.is_cuda or x.dtype != torch.float16 or weight.device != x.device or weight.dtype != x.dtype
            or x.ndim not in (2, 3) or weight.ndim != 2 or not x.is_contiguous() or not weight.is_contiguous()
            or x.shape[-2] != TUNED_ROWS or x.numel() // x.shape[-1] != TUNED_ROWS
            or weight.shape[0] != WIDTH or weight.shape[1] not in (WIDTH, 16384) or x.shape[-1] != weight.shape[1]
        ):
            return None
        stream = torch.cuda.current_stream(x.device).cuda_stream
        workspace = self.workspace(x.device, stream)
        key = x.device, stream, weight.shape[1]
        if key not in self.half_plans:
            handle = Pointer()
            self.backend.prepare_half(ctypes.byref(handle), TUNED_ROWS, WIDTH, weight.shape[1], workspace.numel())
            self.half_plans[key] = handle
        handle = self.half_plans[key]
        if not handle:
            return None
        output = torch.empty((*x.shape[:-1], WIDTH), device=x.device, dtype=x.dtype)
        status = self.backend.run_half(
            handle, 0, x.data_ptr(), weight.data_ptr(), output.data_ptr(),
            workspace.data_ptr(), workspace.numel(), stream,
        )
        if status:
            raise RuntimeError(f"cuBLASLt FP16 linear failed ({status})")
        self.fp16_calls += 1
        return output

    def close(self):
        for device in {key[0] for key in self.workspaces}:
            torch.cuda.synchronize(device)
        for plan in self.fp8_plans.values():
            plan.close()
        for handle in self.half_plans.values():
            if handle:
                self.backend.destroy_half(handle)


def linear(sample, x, weight, bias):
    if isinstance(x, QuantizedTensor) and isinstance(weight, QuantizedTensor):
        if x.ndim != 2 or weight.ndim != 2 or x._params.orig_dtype != torch.float16:
            return None
        if x._qdata.dtype != torch.float8_e4m3fn or weight._qdata.dtype != x._qdata.dtype:
            return None
        return sample.fp8_linear(x._qdata, weight._qdata, x._params.scale, weight._params.scale, bias)
    return sample.fp16_linear(x, weight, bias)


def rms_norm(sample, x, weight, eps):
    if (
        not x.is_cuda or x.dtype != torch.float16 or x.ndim not in (3, 4) or x.shape[0] != 1
        or x.shape[-1] not in (128, WIDTH) or x.stride(-1) != 1 or x.data_ptr() % 8
        or any(stride % 4 for stride in x.stride()[-3:-1])
        or not 0 < x.numel() < 2**31
        or sum((n - 1) * stride for n, stride in zip(x.shape, x.stride())) >= 2**31
    ):
        return None
    heads, length, width = x.shape[-3:]
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    status = sample.backend.launch_rms(
        x.data_ptr(), weight.data_ptr(), output.data_ptr(), heads, length, width,
        x.stride(-3), x.stride(-2), eps, stream,
    )
    if status:
        raise RuntimeError(f"Krea 2 RMSNorm launch failed ({status})")
    sample.rms_calls += 1
    return output


def affine(sample, x, multiplier, offset, residual=False):
    channels = x.shape[-1]
    if (
        sample.backend.launch_affine is None or channels != WIDTH
        or x.ndim != 3 or x.shape[0] != 1 or not x.is_cuda or x.dtype != torch.float16
        or any(t.dtype != x.dtype or t.device != x.device or not t.is_contiguous()
               for t in (x, multiplier, offset))
        or multiplier.shape != (1, 1, channels)
        or offset.shape != (x.shape if residual else multiplier.shape)
    ):
        return x + multiplier * offset if residual else x * multiplier + offset
    output = torch.empty_like(x)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    status = sample.backend.launch_affine(
        x.data_ptr(), multiplier.data_ptr(), offset.data_ptr(), output.data_ptr(), x.numel(), residual, stream,
    )
    if status:
        raise RuntimeError(f"Krea 2 modulation launch failed ({status})")
    sample.affine_calls += 1
    return output


def eligible(sampler, noise, positive, negative, cfg, sigmas, denoise_mask):
    # This is a sampling entry point. model_base imports the Krea model and this module.
    from comfy.model_base import Krea2

    patcher = sampler.model
    model = patcher.model
    device = torch.device(sampler.device)
    if (
        os.name != "nt" or not isinstance(model, Krea2) or not torch.is_inference_mode_enabled()
        or model.get_dtype_inference() != torch.float16 or device.type != "cuda"
        or torch.cuda.get_device_capability(device) != (12, 0)
        or torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 5090"
        or not torch.backends.cuda.matmul.allow_fp16_accumulation
        or sampler.sampler not in ("euler", "heun", "dpmpp_2m") or sampler.scheduler != "simple"
        or sampler.denoise != 1.0 or cfg != 1 or denoise_mask is not None
        or sampler.steps < 1 or sigmas.numel() != sampler.steps + 1
        or noise.shape[:-2] not in ((1, 16), (1, 16, 1)) or 0 in noise.shape[-2:]
        or not patcher.loaded_size() or model.current_weight_patches_uuid != patcher.patches_uuid
        or patcher.hook_patches or patcher.object_patches or patcher.injections
        or any(patcher.wrappers.values()) or any(patcher.callbacks.values())
    ):
        return False
    if model_management.comfy_kitchen_attention_enabled():
        return False
    if not model_management.sage_attention_enabled() and (
        model_management.flash_attention_enabled()
        or model_management.xformers_enabled()
        or not model_management.pytorch_attention_enabled()
    ):
        return False
    projection = model.diffusion_model.blocks[0].attn.wq
    quant_format = getattr(projection, "quant_format", None)
    if quant_format == "int8_tensorwise":
        weight = projection.weight
        if not isinstance(weight, QuantizedTensor) or not weight._params.convrot:
            return False
    elif quant_format != "float8_e4m3fn":
        return False
    model_options = patcher.model_options
    if any(model_options.get(key) for key in (
        "disable_cfg1_optimization", "model_function_wrapper", "sampler_cfg_function",
        "sampler_pre_cfg_function", "sampler_post_cfg_function", "sampler_calc_cond_batch_function",
    )):
        return False
    if positive is None or len(positive) != 1:
        return False
    context, _ = positive[0]
    if (
        not torch.is_tensor(context) or context.ndim != 3
        or context.shape[0] != 1 or context.shape[1] == 0 or context.shape[2] != 30720
    ):
        return False
    # Hooks from the negative branch may apply to both branches, including at CFG 1.
    for _, options in (*positive, *(negative or ())):
        if any(options.get(key) is not None for key in ("hooks", "mask", "area", "control", "reference_latents")):
            return False
    transformer_options = model_options.get("transformer_options", {})
    return not any(transformer_options.get(key) for key in (
        "patches", "patches_replace", "wrappers", "callbacks", "optimized_attention_override",
    ))


@contextmanager
def sample_scope(folder, sampler, noise, positive, negative, cfg, sigmas, denoise_mask):
    sample = None
    if eligible(sampler, noise, positive, negative, cfg, sigmas, denoise_mask):
        backend = load_backend(folder)
        if backend is not None:
            sample = Sample(backend)
    # A fallback sample can run inside another sample's callback.
    token = active_sample.set(sample)
    try:
        yield
    finally:
        active_sample.reset(token)
        if sample is not None:
            sample.close()
            logging.info(
                "Krea 2 SM120: %d tuned MXFP8, %d other MXFP8, %d tuned FP16, %d RMSNorm, %d affine, %d text preparation",
                sample.tuned_fp8_calls, sample.other_fp8_calls, sample.fp16_calls, sample.rms_calls,
                sample.affine_calls, sample.text_preparations,
            )
