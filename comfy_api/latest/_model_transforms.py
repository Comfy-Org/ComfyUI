"""Core-owned model transforms available through ``ModelRef.patch``.

The host accepts only names in ``TRANSFORMS`` and validates every parameter.
Transform implementations run in the trusted process and cannot be registered
by a guest. Each transform returns a cloned patcher so requests compose without
mutating their input model.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

_MISSING = object()


class TransformError(Exception):
    """A transform request the host refuses. Always says what was wrong."""


# --------------------------------------------------------------------------- #
# Parameter specs
#
# Declarative on purpose. Validation, the guest-facing error message, and the
# generated documentation all read the SAME object, so a parameter cannot be
# accepted with a range the docs do not state.
# --------------------------------------------------------------------------- #

class Param:
    type_name = "param"

    def __init__(self, *, doc: str = "", default: Any = _MISSING) -> None:
        self.doc = doc
        self.default = default

    @property
    def required(self) -> bool:
        return self.default is _MISSING

    def check(self, name: str, value: Any) -> Any:
        raise NotImplementedError

    def describe(self) -> dict:
        d: dict[str, Any] = {"type": self.type_name, "doc": self.doc}
        if not self.required:
            d["default"] = self.default
        return d


class OneOf(Param):
    type_name = "enum"

    def __init__(self, choices: Iterable[str], **kw: Any) -> None:
        super().__init__(**kw)
        self.choices = tuple(choices)

    def check(self, name: str, value: Any) -> Any:
        if value not in self.choices:
            raise TransformError(
                f"{name}={value!r} is not one of {list(self.choices)}")
        return value

    def describe(self) -> dict:
        return {**super().describe(), "choices": list(self.choices)}


class Bool(Param):
    type_name = "bool"

    def check(self, name: str, value: Any) -> Any:
        if not isinstance(value, bool):
            raise TransformError(f"{name} must be a bool, got {type(value).__name__}")
        return value


class NullableBool(Bool):
    type_name = "bool-or-null"

    def check(self, name: str, value: Any) -> Any:
        if value is None:
            return None
        return super().check(name, value)


class _Numeric(Param):
    py_type: type = float

    def __init__(self, minimum: float, maximum: float, **kw: Any) -> None:
        super().__init__(**kw)
        self.minimum = minimum
        self.maximum = maximum

    def check(self, name: str, value: Any) -> Any:
        # bool is an int in Python and would silently pass an Int check as 0/1.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TransformError(
                f"{name} must be a {self.type_name}, got {type(value).__name__}")
        if self.py_type is int and not isinstance(value, int):
            raise TransformError(f"{name} must be a whole number, got {value!r}")
        if not (self.minimum <= value <= self.maximum):
            raise TransformError(
                f"{name}={value} is outside [{self.minimum}, {self.maximum}]")
        return self.py_type(value)

    def describe(self) -> dict:
        return {**super().describe(), "min": self.minimum, "max": self.maximum}


class Int(_Numeric):
    type_name = "int"
    py_type = int


class Float(_Numeric):
    type_name = "float"
    py_type = float


class IntList(Param):
    type_name = "int-list"

    def __init__(self, max_items: int, **kw: Any) -> None:
        super().__init__(**kw)
        self.max_items = max_items

    def check(self, name: str, value: Any) -> list[int]:
        if not isinstance(value, (list, tuple)):
            raise TransformError(
                f"{name} must be a list of integers, got "
                f"{type(value).__name__}")
        if len(value) > self.max_items:
            raise TransformError(
                f"{name} has {len(value)} entries; maximum is {self.max_items}")
        if any(isinstance(item, bool) or not isinstance(item, int)
               for item in value):
            raise TransformError(f"{name} must contain only integers")
        return list(value)

    def describe(self) -> dict:
        return {**super().describe(), "max_items": self.max_items}


class TokenWeights(Param):
    type_name = "token-weight-list"

    def __init__(self, max_items: int, **kw: Any) -> None:
        super().__init__(**kw)
        self.max_items = max_items

    def check(self, name: str, value: Any) -> list[tuple[int, float, float]]:
        import math

        if not isinstance(value, (list, tuple)):
            raise TransformError(f"{name} must be a list of token weights")
        if len(value) > self.max_items:
            raise TransformError(
                f"{name} has {len(value)} entries; maximum is {self.max_items}")
        checked = []
        for index, item in enumerate(value):
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise TransformError(
                    f"{name}[{index}] must be (position, value factor, key bias)")
            position, value_factor, key_bias = item
            if (
                isinstance(position, bool)
                or not isinstance(position, int)
                or not 0 <= position <= 1000000
            ):
                raise TransformError(
                    f"{name}[{index}] has an invalid token position")
            if any(
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
                or abs(float(number)) > 1000000
                for number in (value_factor, key_bias)
            ):
                raise TransformError(
                    f"{name}[{index}] has a non-finite or unbounded weight")
            checked.append((position, float(value_factor), float(key_bias)))
        return checked

    def describe(self) -> dict:
        return {**super().describe(), "max_items": self.max_items}


class FloatList(Param):
    type_name = "float-list"

    def __init__(self, minimum: float, maximum: float, max_items: int,
                 **kw: Any) -> None:
        super().__init__(**kw)
        self.minimum = minimum
        self.maximum = maximum
        self.max_items = max_items

    def check(self, name: str, value: Any) -> list[float]:
        import math

        if not isinstance(value, (list, tuple)):
            raise TransformError(f"{name} must be a list of numbers")
        if not value or len(value) > self.max_items:
            raise TransformError(
                f"{name} must contain 1 to {self.max_items} entries")
        out = []
        for index, item in enumerate(value):
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or not self.minimum <= float(item) <= self.maximum
            ):
                raise TransformError(
                    f"{name}[{index}] must be within "
                    f"[{self.minimum}, {self.maximum}]")
            out.append(float(item))
        return out

    def describe(self) -> dict:
        return {
            **super().describe(),
            "min": self.minimum,
            "max": self.maximum,
            "max_items": self.max_items,
        }


class RefOf(Param):
    """A parameter that is another ref — a mask, a conditioning.

    Several transforms are parameterised by a tensor rather than a scalar
    (a NAG conditioning, an attention mask). A ref token cannot be smuggled
    inside a value payload, because values go out-of-band over the buffer
    channel while tokens go over the JSON wire. So the option payload accepts
    tokens natively and the host resolves them, checking the kind, before the
    implementation ever sees an object.
    """
    type_name = "ref"

    def __init__(self, kind: str, **kw: Any) -> None:
        super().__init__(**kw)
        self.kind = kind

    def check(self, name: str, value: Any) -> Any:
        from ._sdk import Ref

        if value is None and not self.required:
            return None
        if not isinstance(value, Ref) or value.kind != self.kind:
            raise TransformError(
                f"{name} must be a {self.kind} ref, got {type(value).__name__}")
        return value

    def describe(self) -> dict:
        return {**super().describe(), "ref_kind": self.kind}


class SafeTensorName(Param):
    """A confined logical filename in a fixed host model catalogue."""

    type_name = "safetensors-name"

    def check(self, name: str, value: Any) -> str:
        if not isinstance(value, str):
            raise TransformError(f"{name} must be a string")
        if (
            not value
            or len(value) > 1024
            or "\\" in value
            or "\x00" in value
            or value.startswith("/")
            or "://" in value
            or any(part in ("", ".", "..") for part in value.split("/"))
            or not value.lower().endswith((".safetensors", ".sft"))
        ):
            raise TransformError(
                f"{name} must be a confined SafeTensors catalogue name")
        return value


class WeightArchiveName(Param):
    """A confined logical filename for a restricted tensor archive."""

    type_name = "weight-archive-name"

    def check(self, name: str, value: Any) -> str:
        if not isinstance(value, str):
            raise TransformError(f"{name} must be a string")
        if (
            not value
            or len(value) > 1024
            or "\\" in value
            or "\x00" in value
            or value.startswith("/")
            or "://" in value
            or any(part in ("", ".", "..") for part in value.split("/"))
            or not value.lower().endswith(
                (".bin", ".ckpt", ".patch", ".pt", ".pth")
            )
        ):
            raise TransformError(
                f"{name} must be a confined PyTorch weight-archive name")
        return value


# --------------------------------------------------------------------------- #
# The transform table
# --------------------------------------------------------------------------- #

class Transform:
    def __init__(self, name: str, summary: str, params: dict[str, Param],
                 apply: Callable[..., Any], *, experimental: bool = False) -> None:
        self.name = name
        self.summary = summary
        self.params = params
        self.apply = apply
        self.experimental = experimental

    def describe(self) -> dict:
        return {
            "name": self.name,
            "summary": self.summary,
            "experimental": self.experimental,
            "params": {k: v.describe() for k, v in self.params.items()},
        }


def validate(name: str, params: dict) -> dict:
    """Resolve a request against the table. Refuses everything unrecognised.

    Unknown parameter names are an error rather than being ignored. A silently
    dropped parameter is the worst outcome available here: the node appears to
    work and the setting the user asked for is simply absent.
    """
    t = TRANSFORMS.get(name)
    if t is None:
        raise TransformError(
            f"unknown model transform {name!r}. Available: "
            f"{sorted(TRANSFORMS)}")

    unknown = set(params) - set(t.params)
    if unknown:
        raise TransformError(
            f"{name}: unknown parameter(s) {sorted(unknown)}; "
            f"accepts {sorted(t.params)}")

    out = {}
    for pname, spec in t.params.items():
        if pname in params:
            out[pname] = spec.check(f"{name}.{pname}", params[pname])
        elif spec.required:
            raise TransformError(f"{name}: missing required parameter {pname!r}")
        else:
            out[pname] = spec.default
    return out


# --------------------------------------------------------------------------- #
# Implementations. Core's own, calling core's own functions.
# --------------------------------------------------------------------------- #

_ATTENTION_IMPLS = ("pytorch", "basic", "split", "sub_quad", "xformers",
                    "sage", "flash")


def _attention_impl(patcher, mode: str, allow_compile: bool):
    """Choose which of core's attention implementations this model uses.

    Core already ships every one of these and already reads
    ``transformer_options["optimized_attention_override"]`` (attention.py:176).
    What was missing was a way to pick one PER MODEL rather than per process —
    which is exactly what the pack nodes were hand-rolling.

    An implementation whose backing library is absent is refused HERE, by name,
    rather than at the first attention call thirty seconds into a sample.
    """
    from comfy.ldm.modules import attention as _attn

    fn = getattr(_attn, f"attention_{mode}", None)
    if fn is None:  # pragma: no cover - guarded by the OneOf spec
        raise TransformError(f"attention implementation {mode!r} is not built in")
    if mode == "sage" and not getattr(_attn, "SAGE_ATTENTION_IS_AVAILABLE", False):
        raise TransformError(
            "sage attention is not installed on this host "
            "(pip install sageattention)")
    if mode == "flash" and not getattr(_attn, "FLASH_ATTENTION_IS_AVAILABLE", False):
        raise TransformError(
            "flash attention is not installed on this host "
            "(pip install flash-attn)")
    if mode == "xformers":
        from comfy import model_management

        if not model_management.xformers_enabled():
            raise TransformError("xformers attention is not enabled on this host")

    if not allow_compile:
        import torch

        fn = torch.compiler.disable()(fn)

    def override(func, *args, **kwargs):
        return fn(*args, **kwargs)

    m = patcher.clone()
    m.model_options.setdefault("transformer_options", {})
    m.model_options["transformer_options"]["optimized_attention_override"] = override
    return m


_SAGE_VARIANTS = (
    "disabled",
    "auto",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda++",
    "sageattn3",
    "sageattn3_per_block_mean",
)


def _sage_attention_variant(patcher, mode: str, allow_compile: bool):
    if mode == "disabled":
        return patcher

    import torch
    from comfy.ldm.modules.attention import attention_pytorch, wrap_attn

    try:
        if mode == "auto":
            from sageattention import sageattn as implementation

            def sage_func(q, k, v, is_causal=False, attn_mask=None,
                          tensor_layout="NHD"):
                return implementation(
                    q, k, v, is_causal=is_causal, attn_mask=attn_mask,
                    tensor_layout=tensor_layout)
        elif mode == "sageattn_qk_int8_pv_fp16_cuda":
            from sageattention import (
                sageattn_qk_int8_pv_fp16_cuda as implementation,
            )

            def sage_func(q, k, v, is_causal=False, attn_mask=None,
                          tensor_layout="NHD"):
                return implementation(
                    q, k, v, is_causal=is_causal, attn_mask=attn_mask,
                    pv_accum_dtype="fp32", tensor_layout=tensor_layout)
        elif mode == "sageattn_qk_int8_pv_fp16_triton":
            from sageattention import (
                sageattn_qk_int8_pv_fp16_triton as implementation,
            )

            def sage_func(q, k, v, is_causal=False, attn_mask=None,
                          tensor_layout="NHD"):
                return implementation(
                    q, k, v, is_causal=is_causal, attn_mask=attn_mask,
                    tensor_layout=tensor_layout)
        elif mode in (
            "sageattn_qk_int8_pv_fp8_cuda",
            "sageattn_qk_int8_pv_fp8_cuda++",
        ):
            from sageattention import (
                sageattn_qk_int8_pv_fp8_cuda as implementation,
            )
            accumulation = (
                "fp32+fp16"
                if mode.endswith("++")
                else "fp32+fp32"
            )

            def sage_func(q, k, v, is_causal=False, attn_mask=None,
                          tensor_layout="NHD"):
                return implementation(
                    q, k, v, is_causal=is_causal, attn_mask=attn_mask,
                    pv_accum_dtype=accumulation, tensor_layout=tensor_layout)
        else:
            from sageattn3 import sageattn3_blackwell as implementation
            per_block_mean = mode == "sageattn3_per_block_mean"

            def sage_func(q, k, v, is_causal=False, attn_mask=None,
                          tensor_layout="NHD"):
                if tensor_layout == "NHD":
                    q, k, v = (value.transpose(1, 2)
                               for value in (q, k, v))
                out = implementation(
                    q, k, v, is_causal=is_causal, attn_mask=attn_mask,
                    per_block_mean=per_block_mean)
                return out.transpose(1, 2) if tensor_layout == "NHD" else out
    except ImportError as error:
        package = "sageattn3" if mode.startswith("sageattn3") else "sageattention"
        raise TransformError(
            f"{mode} requires the {package} package on this host") from error

    if not allow_compile:
        sage_func = torch.compiler.disable()(sage_func)

    @wrap_attn
    def attention_sage(q, k, v, heads, mask=None, attn_precision=None,
                       skip_reshape=False, skip_output_reshape=False, **kwargs):
        if kwargs.get("low_precision_attention", True) is False:
            return attention_pytorch(
                q, k, v, heads, mask=mask, skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape, **kwargs)
        in_dtype = v.dtype
        if any(value.dtype == torch.float32 for value in (q, k, v)):
            q, k, v = (value.to(torch.float16) for value in (q, k, v))
        if skip_reshape:
            batch, _, _, dim_head = q.shape
            tensor_layout = "HND"
        else:
            batch, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = (
                value.view(batch, -1, heads, dim_head)
                for value in (q, k, v)
            )
            tensor_layout = "NHD"
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        sequence_dim = 2 if tensor_layout == "HND" else 1
        if any(
            (value.shape[sequence_dim] - 1) * value.stride(sequence_dim)
            >= 2**31
            for value in (q, k, v)
        ):
            q, k, v = (value.contiguous() for value in (q, k, v))
        out = sage_func(
            q, k, v, attn_mask=mask, is_causal=False,
            tensor_layout=tensor_layout).to(in_dtype)
        if tensor_layout == "HND":
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(
                    batch, -1, heads * dim_head)
        elif skip_output_reshape:
            out = out.transpose(1, 2)
        else:
            out = out.reshape(batch, -1, heads * dim_head)
        return out

    model = patcher.clone()
    options = model.model_options.setdefault("transformer_options", {})

    def override(func, *args, **kwargs):
        return attention_sage.__wrapped__(*args, **kwargs)

    options["optimized_attention_override"] = override
    return model


def _strict_flash_attention(patcher, allow_compile: bool):
    import torch
    from comfy.ldm.modules.attention import wrap_attn

    is_fa3 = False
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        try:
            from flash_attn_interface import flash_attn_func
            is_fa3 = True
        except ImportError as error:
            raise TransformError(
                "strict_flash_attention requires flash_attn (FA2) or "
                "flash_attn_interface (FA3) on this host") from error

    inference_dtype = (
        patcher.model.get_dtype_inference()
        if hasattr(patcher.model, "get_dtype_inference")
        else torch.float16
    )
    cast_dtype = (
        inference_dtype
        if inference_dtype in (torch.float16, torch.bfloat16)
        else torch.float16
    )

    def flash_func(q, k, v):
        if is_fa3:
            out = flash_attn_func(q, k, v, causal=False)
        else:
            out = flash_attn_func(
                q, k, v, dropout_p=0.0, causal=False)
        return out[0] if isinstance(out, tuple) else out

    if not allow_compile:
        flash_func = torch.compiler.disable()(flash_func)
    if torch.cuda.is_available():
        probe = torch.zeros(
            1, 8, 2, 64, dtype=cast_dtype, device="cuda")
        flash_func(probe, probe, probe)

    @wrap_attn
    def attention_flash(q, k, v, heads, mask=None, attn_precision=None,
                        skip_reshape=False, skip_output_reshape=False, **kwargs):
        if mask is not None:
            raise RuntimeError("Flash attention does not support attention masks")
        in_dtype = v.dtype
        if any(value.dtype == torch.float32 for value in (q, k, v)):
            q, k, v = (value.to(cast_dtype) for value in (q, k, v))
        if skip_reshape:
            batch, _, _, dim_head = q.shape
            q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        else:
            batch, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = (
                value.view(batch, -1, heads, dim_head)
                for value in (q, k, v)
            )
        out = flash_func(q, k, v).to(in_dtype)
        if skip_output_reshape:
            out = out.transpose(1, 2)
        else:
            out = out.reshape(batch, -1, heads * dim_head)
        return out

    model = patcher.clone()
    options = model.model_options.setdefault("transformer_options", {})

    def override(func, *args, **kwargs):
        return attention_flash.__wrapped__(*args, **kwargs)

    options["optimized_attention_override"] = override
    return model


def _kohya_deep_shrink(
    patcher, block_number: int, downscale_factor: float,
    start_percent: float, end_percent: float,
    downscale_after_skip: bool, downscale_method: str,
    upscale_method: str,
):
    """Expose core's canonical PatchModelAddDownscale implementation."""
    if start_percent > end_percent:
        raise TransformError(
            "kohya_deep_shrink.start_percent must not exceed end_percent")
    from comfy_extras.nodes_model_downscale import PatchModelAddDownscale

    output = PatchModelAddDownscale.execute(
        patcher,
        block_number,
        downscale_factor,
        start_percent,
        end_percent,
        downscale_after_skip,
        downscale_method,
        upscale_method,
    )
    return output.result[0]


def _nabla_sparse_attention(
    patcher, latent, window_time: int, window_width: int,
    window_height: int, sparsity: float, compile_attention: bool,
):
    import math
    import torch
    from comfy import model_management
    from comfy.ldm.modules.attention import optimized_attention

    try:
        from torch.nn.attention.flex_attention import BlockMask, flex_attention
    except ImportError as error:
        raise TransformError(
            "nabla_sparse_attention requires torch flex_attention") from error

    samples = latent.get("samples") if isinstance(latent, dict) else None
    if not isinstance(samples, torch.Tensor) or samples.ndim != 5:
        raise TransformError(
            "nabla_sparse_attention.latent must contain a 5D samples tensor")

    _, _, frames, height, width = samples.shape
    frames, height, width = frames, height // 16, width // 16
    max_axis = torch.tensor(
        [frames, height, width], dtype=torch.float32).amax()
    axis = torch.arange(
        0, max_axis, 1, dtype=torch.int16,
        device=model_management.get_torch_device())
    distances = (axis.unsqueeze(1) - axis.unsqueeze(0)).abs()
    temporal = distances[:frames, :frames].flatten() <= window_time // 2
    vertical = distances[:height, :height].flatten() <= window_height // 2
    horizontal = distances[:width, :width].flatten() <= window_width // 2
    spatial = (
        vertical.unsqueeze(1) * horizontal.unsqueeze(0)
    ).reshape(height, height, width, width).transpose(1, 2).flatten()
    static_mask = (
        temporal.unsqueeze(1) * spatial.unsqueeze(0)
    ).reshape(
        frames, frames, height * width, height * width
    ).transpose(1, 2).reshape(
        frames * height * width, frames * height * width
    ).unsqueeze_(0).unsqueeze_(0)

    class NablaAttention:
        def __call__(self, q, k, v, heads, **kwargs):
            if q.shape[-2] < 3000 or k.shape[-2] < 3000:
                return optimized_attention(q, k, v, heads, **kwargs)
            block_mask = self._block_mask(q, k)
            return flex_attention(
                q, k, v, block_mask=block_mask,
            ).transpose(1, 2).contiguous().flatten(-2, -1)

        @staticmethod
        def _block_mask(q, k):
            block_size = 64
            batch, heads, sequence, dim = q.shape
            blocks = sequence // block_size
            q_average = q.reshape(
                batch, heads, blocks, block_size, dim).mean(-2)
            k_average = k.reshape(
                batch, heads, blocks, block_size, dim,
            ).mean(-2).transpose(-2, -1)
            attention_map = torch.softmax(
                (q_average @ k_average) / math.sqrt(dim), dim=-1)
            values, indices = attention_map.sort(-1)
            cumulative = values.cumsum_(-1)
            mask = (cumulative >= 1 - sparsity).int()
            mask = mask.gather(-1, indices.argsort(-1))
            mask = torch.logical_or(mask, static_mask)
            kv_blocks = mask.sum(-1).to(torch.int32)
            kv_indices = mask.argsort(
                dim=-1, descending=True).to(torch.int32)
            return BlockMask.from_kv_blocks(
                torch.zeros_like(kv_blocks), kv_indices,
                kv_blocks, kv_indices, BLOCK_SIZE=block_size,
                mask_mod=None)

    attention = NablaAttention()

    def override(func, *args, **kwargs):
        return attention(*args, **kwargs)

    if compile_attention:
        override = torch.compile(
            override, mode="max-autotune-no-cudagraphs", dynamic=True)

    model = patcher.clone()
    options = model.model_options.setdefault("transformer_options", {})
    options["optimized_attention_override"] = override
    return model


def _feta_score(query, key, head_dim: int, num_frames: int, weight: float):
    import torch

    attention = (query * (head_dim ** -0.5)) @ key.transpose(-2, -1)
    attention = attention.to(torch.float32).softmax(dim=-1)
    attention = attention.reshape(-1, num_frames, num_frames)
    diagonal = torch.eye(
        num_frames, device=attention.device, dtype=torch.bool,
    ).unsqueeze(0).expand(attention.shape[0], -1, -1)
    off_diagonal = attention.masked_fill(diagonal, 0)
    count = num_frames * num_frames - num_frames
    mean_scores = off_diagonal.sum(dim=(1, 2)) / count
    return (mean_scores.mean() * (num_frames + weight)).clamp(min=1)


def _feta_scores(query, key, num_frames: int, weight: float,
                 num_heads: int = 12):
    from einops import rearrange

    if query.ndim == 4:
        batch, sequence, num_heads, head_dim = query.shape
    elif query.ndim == 3:
        batch, sequence, hidden_dim = query.shape
        head_dim = hidden_dim // num_heads
        query = query.view(batch, sequence, num_heads, head_dim)
        key = key.view(batch, sequence, num_heads, head_dim)
    else:
        raise TransformError(
            f"enhance_a_video attention must be 3D or 4D, got {query.ndim}D")
    spatial = sequence // num_frames
    query_image = rearrange(
        query, "B (T S) N C -> (B S) N T C",
        T=num_frames, S=spatial, N=num_heads, C=head_dim)
    key_image = rearrange(
        key, "B (T S) N C -> (B S) N T C",
        T=num_frames, S=spatial, N=num_heads, C=head_dim)
    return _feta_score(query_image, key_image, head_dim, num_frames, weight)


def _wan_enhance_forward(module, x, freqs, num_frames: int, weight: float,
                         transformer_options=None):
    from comfy.ldm.flux.math import apply_rope
    from comfy.ldm.modules.attention import optimized_attention

    transformer_options = transformer_options or {}
    batch, sequence = x.shape[:2]
    heads, head_dim = module.num_heads, module.head_dim
    q = module.norm_q(module.q(x)).view(batch, sequence, heads, head_dim)
    k = module.norm_k(module.k(x)).view(batch, sequence, heads, head_dim)
    v = module.v(x).view(batch, sequence, heads * head_dim)
    q, k = apply_rope(q, k, freqs)
    score = _feta_scores(q, k, num_frames, weight)
    out = optimized_attention(
        q.view(batch, sequence, heads * head_dim),
        k.view(batch, sequence, heads * head_dim),
        v, heads=heads, transformer_options=transformer_options)
    return module.o(out) * score


def _ltx_enhance_forward(
    module, x, num_frames: int, weight: float, context=None, mask=None,
    pe=None, k_pe=None, transformer_options=None,
):
    from comfy.ldm.modules.attention import (
        optimized_attention, optimized_attention_masked,
    )

    transformer_options = transformer_options or {}
    q = module.to_q(x)
    context = x if context is None else context
    k = module.to_k(context)
    v = module.to_v(context)
    q = module.q_norm(q)
    k = module.k_norm(k)
    if pe is not None:
        try:
            from comfy.ldm.lightricks.model import apply_rotary_emb
        except ImportError as error:
            raise TransformError(
                "LTX Enhance-A-Video needs core Lightricks rotary support") from error
        q = apply_rotary_emb(q, pe)
        k = apply_rotary_emb(k, pe if k_pe is None else k_pe)
    score = _feta_scores(q, k, num_frames, weight, module.heads)
    try:
        from comfy.ldm.lightricks.model import (
            GuideAttentionMask, _attention_with_guide_mask,
        )
    except ImportError:
        GuideAttentionMask = None
        _attention_with_guide_mask = None
    if mask is None:
        out = optimized_attention(
            q, k, v, module.heads, attn_precision=module.attn_precision,
            transformer_options=transformer_options)
    elif GuideAttentionMask is not None and isinstance(mask, GuideAttentionMask):
        out = _attention_with_guide_mask(
            q, k, v, module.heads, mask,
            attn_precision=module.attn_precision,
            transformer_options=transformer_options)
    else:
        out = optimized_attention_masked(
            q, k, v, module.heads, mask,
            attn_precision=module.attn_precision,
            transformer_options=transformer_options)
    if module.to_gate_logits is not None:
        import torch

        gate_logits = module.to_gate_logits(x)
        batch, sequence, _ = out.shape
        out = out.view(batch, sequence, module.heads, module.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)
        out = (out * gates.unsqueeze(-1)).view(
            batch, sequence, module.heads * module.dim_head)
    return module.to_out(out) * score


def _enhance_a_video(patcher, latent, architecture: str, weight: float):
    import types
    import torch

    if weight == 0:
        return patcher
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if not isinstance(samples, torch.Tensor) or samples.ndim != 5:
        raise TransformError(
            "enhance_a_video.latent must contain a 5D samples tensor")
    num_frames = samples.shape[2]
    model = patcher.clone()
    options = model.model_options.setdefault("transformer_options", {})
    options["enhance_weight"] = weight
    diffusion_model = model.get_model_object("diffusion_model")

    if architecture == "wan":
        blocks = getattr(diffusion_model, "blocks", None)
        if blocks is None:
            raise TransformError(
                "Wan Enhance-A-Video needs diffusion_model.blocks")
        compile_settings = getattr(patcher.model, "compile_settings", None)
        for index, block in enumerate(blocks):
            attention = block.self_attn

            def forward(self_module, x, freqs, transformer_options=None,
                        _frames=num_frames, _weight=weight):
                return _wan_enhance_forward(
                    self_module, x, freqs, _frames, _weight,
                    transformer_options)

            patched = types.MethodType(forward, attention)
            if compile_settings is not None:
                patched = torch.compile(
                    patched, mode=compile_settings["mode"],
                    dynamic=compile_settings["dynamic"],
                    fullgraph=compile_settings["fullgraph"],
                    backend=compile_settings["backend"])
            model.add_object_patch(
                f"diffusion_model.blocks.{index}.self_attn.forward", patched)
        return model

    if architecture != "ltx":
        raise TransformError(
            f"unknown Enhance-A-Video architecture {architecture!r}")
    blocks = getattr(diffusion_model, "transformer_blocks", None)
    if blocks is None:
        raise TransformError(
            "LTX Enhance-A-Video needs diffusion_model.transformer_blocks")
    for index, block in enumerate(blocks):
        attention = block.attn1

        def forward(self_module, x, context=None, mask=None, pe=None, k_pe=None,
                    transformer_options=None, _frames=num_frames,
                    _weight=weight):
            return _ltx_enhance_forward(
                self_module, x, _frames, _weight, context, mask, pe, k_pe,
                transformer_options)

        model.add_object_patch(
            f"diffusion_model.transformer_blocks.{index}.attn1.forward",
            types.MethodType(forward, attention))
    return model


def _wan_nag_compute(module, query, context, transformer_options):
    from comfy.ldm.modules.attention import optimized_attention

    key = module.norm_k(module.k(context))
    value = module.v(context)
    return optimized_attention(
        query, key, value, heads=module.num_heads,
        transformer_options=transformer_options).flatten(2)


def _wan_nag_guidance(positive, negative, scale: float, alpha: float,
                      tau: float, inplace: bool):
    import torch

    if inplace:
        guidance = negative.mul_(scale - 1).neg_().add_(positive, alpha=scale)
    else:
        guidance = negative * (scale - 1)
        guidance = (positive * scale).sub_(guidance)
    norm_positive = torch.norm(positive, p=1, dim=-1, keepdim=True)
    norm_guidance = torch.norm(guidance, p=1, dim=-1, keepdim=True)
    ratio = norm_guidance / norm_positive
    torch.nan_to_num_(ratio, nan=10.0)
    mask = ratio > tau
    adjustment = (norm_positive * tau) / (norm_guidance + 1e-7)
    guidance.mul_(torch.where(mask, adjustment, 1.0))
    if inplace:
        return guidance.sub_(positive).mul_(alpha).add_(positive)
    guidance.mul_(alpha)
    return guidance.add_(positive * (1 - alpha))


def _wan_nag_forward(
    module, x, context, nag_context, nag_scale: float, nag_alpha: float,
    nag_tau: float, input_type: str, inplace: bool,
    transformer_options=None,
):
    import torch
    from comfy.ldm.modules.attention import optimized_attention

    transformer_options = transformer_options or {}
    if input_type == "default":
        if context.shape[0] == 1:
            x_positive, context_positive = x, context
            x_negative = context_negative = None
        else:
            x_positive, x_negative = torch.chunk(x, 2, dim=0)
            context_positive, context_negative = torch.chunk(context, 2, dim=0)
    else:
        x_positive, context_positive = x, context
        x_negative = context_negative = None
    query_positive = module.norm_q(module.q(x_positive))
    if input_type == "batch":
        nag_context = nag_context.repeat(x_positive.shape[0], 1, 1)
    positive = _wan_nag_compute(
        module, query_positive, context_positive, transformer_options)
    negative = _wan_nag_compute(
        module, query_positive, nag_context, transformer_options)
    positive_output = _wan_nag_guidance(
        positive, negative, nag_scale, nag_alpha, nag_tau, inplace)
    if x_negative is not None and context_negative is not None:
        query_negative = module.norm_q(module.q(x_negative))
        key_negative = module.norm_k(module.k(context_negative))
        value_negative = module.v(context_negative)
        negative_output = optimized_attention(
            query_negative, key_negative, value_negative,
            heads=module.num_heads,
            transformer_options=transformer_options)
        out = torch.cat([positive_output, negative_output], dim=0)
    else:
        out = positive_output
    return module.o(out)


def _wan_i2v_nag_forward(
    module, x, context, context_img_len: int, nag_context,
    nag_scale: float, nag_alpha: float, nag_tau: float, inplace: bool,
    transformer_options=None,
):
    import torch
    from comfy.ldm.modules.attention import optimized_attention

    transformer_options = transformer_options or {}
    context_image = context[:, :context_img_len]
    context = context[:, context_img_len:]
    query_image = module.norm_q(module.q(x))
    key_image = module.norm_k_img(module.k_img(context_image))
    value_image = module.v_img(context_image)
    image_output = optimized_attention(
        query_image, key_image, value_image, heads=module.num_heads,
        transformer_options=transformer_options)
    if context.shape[0] == 2:
        x, x_real_negative = torch.chunk(x, 2, dim=0)
        context_positive, context_negative = torch.chunk(context, 2, dim=0)
    else:
        context_positive = context
        context_negative = None
    query = module.norm_q(module.q(x))
    positive = _wan_nag_compute(
        module, query, context_positive, transformer_options)
    negative = _wan_nag_compute(
        module, query, nag_context, transformer_options)
    out = _wan_nag_guidance(
        positive, negative, nag_scale, nag_alpha, nag_tau, inplace)
    if context_negative is not None:
        query_negative = module.norm_q(module.q(x_real_negative))
        key_negative = module.norm_k(module.k(context_negative))
        value_negative = module.v(context_negative)
        x_real_negative = optimized_attention(
            query_negative, key_negative, value_negative,
            heads=module.num_heads,
            transformer_options=transformer_options)
        out = torch.cat([out, x_real_negative], dim=0)
    return module.o(out + image_output)


def _wan_video_nag(
    patcher, conditioning, nag_scale: float, nag_alpha: float,
    nag_tau: float, input_type: str, inplace: bool,
):
    import torch
    import types
    from comfy import model_management

    if nag_scale == 0:
        return patcher
    if (
        not isinstance(conditioning, (list, tuple))
        or not conditioning
        or not isinstance(conditioning[0], (list, tuple))
        or not conditioning[0]
        or not isinstance(conditioning[0][0], torch.Tensor)
    ):
        raise TransformError(
            "wan_video_nag.conditioning must contain an embedding tensor")
    model = patcher.clone()
    diffusion_model = model.get_model_object("diffusion_model")
    text_embedding = getattr(diffusion_model, "text_embedding", None)
    blocks = getattr(diffusion_model, "blocks", None)
    if text_embedding is None or blocks is None:
        raise TransformError(
            "wan_video_nag needs Wan text_embedding and blocks")
    device = model_management.get_torch_device()
    dtype = model_management.unet_dtype()
    text_embedding.to(device)
    nag_context = text_embedding(conditioning[0][0].to(device, dtype))
    for index, block in enumerate(blocks):
        attention = block.cross_attn
        is_i2v = hasattr(attention, "k_img")
        if is_i2v:
            def forward(
                self_module, x, context, context_img_len,
                transformer_options=None, _context=nag_context,
                _scale=nag_scale, _alpha=nag_alpha, _tau=nag_tau,
                _inplace=inplace,
            ):
                return _wan_i2v_nag_forward(
                    self_module, x, context, context_img_len, _context,
                    _scale, _alpha, _tau, _inplace, transformer_options)
        else:
            def forward(
                self_module, x, context, transformer_options=None,
                _context=nag_context, _scale=nag_scale, _alpha=nag_alpha,
                _tau=nag_tau, _input_type=input_type, _inplace=inplace,
            ):
                return _wan_nag_forward(
                    self_module, x, context, _context, _scale, _alpha, _tau,
                    _input_type, _inplace, transformer_options)
        model.add_object_patch(
            f"diffusion_model.blocks.{index}.cross_attn.forward",
            types.MethodType(forward, attention))
    return model


def _krea2_attention_forward(
    module, x, freqs=None, mask=None, transformer_options=None,
):
    import torch
    from einops import rearrange
    from comfy.ldm.flux.math import apply_rope
    from comfy.ldm.modules.attention import (
        attention_pytorch, optimized_attention,
    )

    transformer_options = transformer_options or {}
    weights = transformer_options.get("krea2_token_weights")
    q, k, v, gate = module.wq(x), module.wk(x), module.wv(x), module.gate(x)
    q = rearrange(q, "B L (H D) -> B H L D", H=module.heads)
    k = rearrange(k, "B L (H D) -> B H L D", H=module.kvheads)
    v = rearrange(v, "B L (H D) -> B H L D", H=module.kvheads)
    if weights:
        v = v.clone()
        for position, value_factor, _ in weights:
            if value_factor != 1.0 and position < v.shape[2]:
                v[:, :, position] = v[:, :, position] * value_factor
    q, k = module.qknorm(q, k)
    if freqs is not None:
        q, k = apply_rope(q, k, freqs)
    if module.kvheads != module.heads:
        repeat = module.heads // module.kvheads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    bias = None
    if weights and any(key_bias != 0.0 for _, _, key_bias in weights):
        bias = q.new_zeros(1, k.shape[2])
        for position, _, key_bias in weights:
            if key_bias != 0.0 and position < bias.shape[1]:
                bias[:, position] = key_bias
    if bias is not None:
        out = attention_pytorch(
            q, k, v, module.heads, mask=bias, skip_reshape=True)
    else:
        out = optimized_attention(
            q, k, v, module.heads, mask=mask, skip_reshape=True,
            transformer_options=transformer_options)
    return module.wo(out * torch.sigmoid(gate))


def _krea2_token_weights(patcher, weights):
    import types

    if not weights:
        return patcher
    model = patcher.clone()
    diffusion_model = model.get_model_object("diffusion_model")
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None or any(not hasattr(block, "attn") for block in blocks):
        raise TransformError(
            "krea2_token_weights needs Krea2 diffusion blocks with attention")
    options = dict(model.model_options.get("transformer_options", {}))
    options["krea2_token_weights"] = list(weights)
    model.model_options["transformer_options"] = options
    for index, block in enumerate(blocks):
        attention = block.attn

        def forward(
            self_module, x, freqs=None, mask=None, transformer_options=None,
        ):
            return _krea2_attention_forward(
                self_module, x, freqs, mask, transformer_options)

        model.add_object_patch(
            f"diffusion_model.blocks.{index}.attn.forward",
            types.MethodType(forward, attention))
    return model


def _ltx2_audio_normalization(patcher, factors):
    from comfy.patcher_extension import WrappersMP

    def wrapper(
        executor, noise, latent_image, sampler, sigmas, denoise_mask,
        callback, disable_pbar, seed, latent_shapes,
    ):
        import latent_preview
        import comfy.utils

        guider = executor.class_obj
        ltxav = guider.model_patcher.model.diffusion_model
        x0_output = {}
        total_steps = sigmas.shape[-1] - 1
        progress = comfy.utils.ProgressBar(total_steps)
        completed = 0
        previewer = latent_preview.get_previewer(
            guider.model_patcher.load_device,
            guider.model_patcher.model.latent_format)

        def sampling_callback(step, x0, x, callback_total_steps):
            nonlocal completed
            x0_output["x0"] = x0
            preview_bytes = None
            if previewer:
                preview_bytes = previewer.decode_latent_to_preview_image(
                    "JPEG", x0)
            completed += 1
            progress.update_absolute(
                completed, total_steps, preview_bytes)

        active_factors = list(factors)
        if len(active_factors) < total_steps:
            active_factors.extend(
                [active_factors[-1]] * (total_steps - len(active_factors)))
        split_indices = [
            index + 1 for index, factor in enumerate(active_factors)
            if factor != 1.0
        ]
        if split_indices:
            chunks = []
            previous = 0
            for index in sorted(set(split_indices)):
                if previous < index:
                    chunks.append(sigmas[previous:index + 1])
                previous = index
            if previous < len(sigmas):
                chunks.append(sigmas[previous:])
        else:
            chunks = [sigmas]

        completed_steps = 0
        for sigma_chunk in chunks:
            completed_steps += len(sigma_chunk) - 1
            latent_image = executor(
                noise, latent_image, sampler, sigma_chunk, denoise_mask,
                sampling_callback, disable_pbar, seed,
                latent_shapes=latent_shapes)
            if "x0" in x0_output:
                latent_image = guider.model_patcher.model.process_latent_out(
                    x0_output["x0"])
            if completed_steps - 1 < len(active_factors):
                unpacked = comfy.utils.unpack_latents(
                    latent_image, latent_shapes)
                video, audio = ltxav.separate_audio_and_video_latents(
                    unpacked, None)
                if denoise_mask is not None:
                    unpacked_mask = comfy.utils.unpack_latents(
                        denoise_mask, latent_shapes)
                    audio_mask = ltxav.separate_audio_and_video_latents(
                        unpacked_mask, None)[1]
                    audio = (
                        audio * audio_mask * active_factors[completed_steps - 1]
                        + audio * (1 - audio_mask)
                    )
                else:
                    audio = audio * active_factors[completed_steps - 1]
                latent_image = comfy.utils.pack_latents(
                    ltxav.recombine_audio_and_video_latents(video, audio))[0]
        return latent_image

    model = patcher.clone()
    model.add_wrapper_with_key(
        WrappersMP.OUTER_SAMPLE,
        "ltx2_audio_normalization",
        wrapper,
    )
    return model


def _ltx_nag_compute(
    module, query, context, transformer_options, mask=None,
):
    from comfy.ldm.modules.attention import (
        attention_pytorch, optimized_attention,
    )

    key = module.k_norm(module.to_k(context)).to(query.dtype)
    value = module.to_v(context).to(query.dtype)
    if mask is None:
        out = optimized_attention(
            query, key, value, heads=module.heads,
            attn_precision=module.attn_precision,
            transformer_options=transformer_options)
    else:
        out = attention_pytorch(
            query, key, value, heads=module.heads, mask=mask,
            attn_precision=module.attn_precision,
            _inside_attn_wrapper=True,
            transformer_options=transformer_options)
    return out.flatten(2)


def _ltx_nag_forward(
    module, x, context, nag_context, nag_scale: float, nag_alpha: float,
    nag_tau: float, inplace: bool, mask=None, transformer_options=None,
):
    import torch
    from comfy.ldm.modules.attention import optimized_attention

    transformer_options = transformer_options or {}
    if mask is None:
        mask_provider = transformer_options.get("promptrelay_mask_fn")
        if mask_provider is not None:
            mask = mask_provider(
                x.shape[1], context.shape[1], x.dtype, x.device,
                transformer_options)
    if context.shape[0] == 1:
        x_positive, context_positive = x, context
        x_negative = context_negative = None
    else:
        x_positive, x_negative = torch.chunk(x, 2, dim=0)
        context_positive, context_negative = torch.chunk(context, 2, dim=0)
    query_positive = module.q_norm(module.to_q(x_positive))
    positive = _ltx_nag_compute(
        module, query_positive, context_positive,
        transformer_options, mask)
    negative = _ltx_nag_compute(
        module, query_positive, nag_context, transformer_options)
    positive_output = _wan_nag_guidance(
        positive, negative, nag_scale, nag_alpha, nag_tau, inplace)
    if x_negative is not None and context_negative is not None:
        query_negative = module.q_norm(module.to_q(x_negative))
        key_negative = module.k_norm(module.to_k(context_negative))
        value_negative = module.to_v(context_negative)
        negative_output = optimized_attention(
            query_negative, key_negative, value_negative,
            heads=module.heads, attn_precision=module.attn_precision,
            transformer_options=transformer_options)
        out = torch.cat([positive_output, negative_output], dim=0)
    else:
        out = positive_output
    if module.to_gate_logits is not None:
        gate_logits = module.to_gate_logits(x)
        batch, sequence, _ = out.shape
        out = out.view(batch, sequence, module.heads, module.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)
        out = (out * gates.unsqueeze(-1)).view(
            batch, sequence, module.heads * module.dim_head)
    return module.to_out(out)


def _ltx2_nag(
    patcher, nag_scale: float, nag_alpha: float, nag_tau: float,
    video_conditioning, audio_conditioning, inplace: bool,
):
    import torch
    import types
    from comfy import model_management

    if nag_scale == 0:
        return patcher
    for name, conditioning in (
        ("video_conditioning", video_conditioning),
        ("audio_conditioning", audio_conditioning),
    ):
        if conditioning is not None and (
            not isinstance(conditioning, (list, tuple))
            or not conditioning
            or not isinstance(conditioning[0], (list, tuple))
            or not conditioning[0]
            or not isinstance(conditioning[0][0], torch.Tensor)
        ):
            raise TransformError(
                f"ltx2_nag.{name} must contain an embedding tensor")

    model = patcher.clone()
    diffusion_model = model.get_model_object("diffusion_model")
    blocks = getattr(diffusion_model, "transformer_blocks", None)
    if blocks is None:
        raise TransformError("ltx2_nag needs LTX2 transformer blocks")
    dtype = getattr(patcher.model, "manual_cast_dtype", None)
    if dtype is None:
        dtype = diffusion_model.dtype
    device = model_management.get_torch_device()
    offload_device = model_management.unet_offload_device()

    def project(conditioning, *, audio: bool):
        context = conditioning[0][0].to(device, dtype)
        video_width = getattr(diffusion_model, "cross_attention_dim", None)
        audio_width = diffusion_model.audio_cross_attention_dim
        if video_width is not None and context.shape[-1] == video_width + audio_width:
            context = context[:, :, video_width:] if audio else context[:, :, :video_width]
        if (
            diffusion_model.caption_proj_before_connector
            and diffusion_model.caption_projection_first_linear
        ):
            projection = (
                diffusion_model.audio_caption_projection
                if audio else diffusion_model.caption_projection
            )
            projection.to(device)
            try:
                context = projection(context)
            finally:
                projection.to(offload_device)
        connector_name = (
            "audio_embeddings_connector" if audio
            else "video_embeddings_connector"
        )
        if hasattr(diffusion_model, connector_name):
            connector = getattr(diffusion_model, connector_name)
            connector.to(device)
            try:
                context = connector(context)[0]
            finally:
                connector.to(offload_device)
        width = (
            diffusion_model.audio_inner_dim if audio
            else diffusion_model.inner_dim
        )
        return context.view(1, -1, width)

    contexts = (
        ("attn2", project(video_conditioning, audio=False))
        if video_conditioning is not None else None,
        ("audio_attn2", project(audio_conditioning, audio=True))
        if (
            audio_conditioning is not None
            and diffusion_model.audio_caption_projection is not None
        ) else None,
    )
    for target in contexts:
        if target is None:
            continue
        attribute, nag_context = target
        for index, block in enumerate(blocks):
            attention = getattr(block, attribute)

            def forward(
                self_module, x, context, mask=None, transformer_options=None,
                _nag_context=nag_context, _scale=nag_scale,
                _alpha=nag_alpha, _tau=nag_tau, _inplace=inplace,
            ):
                return _ltx_nag_forward(
                    self_module, x, context, _nag_context, _scale, _alpha,
                    _tau, _inplace, mask, transformer_options)

            model.add_object_patch(
                f"diffusion_model.transformer_blocks.{index}."
                f"{attribute}.forward",
                types.MethodType(forward, attention))
    return model


def _ideogram4_optimizations(
    patcher, chunk_ffn: bool, ffn_chunks: int,
    ffn_seq_threshold: int, bf16_rope: bool,
):
    import types

    import torch
    from comfy.ldm.lumina.model import FeedForward
    from comfy.ldm.modules.attention import optimized_attention_masked

    if not chunk_ffn and not bf16_rope:
        return patcher
    model = patcher.clone()
    diffusion_model = model.get_model_object("diffusion_model")
    layers = getattr(diffusion_model, "layers", None)
    if (
        not layers
        or not hasattr(layers[0], "feed_forward")
        or not hasattr(layers[0], "attention")
    ):
        return patcher

    def ffn_forward(module, value, *, chunks, threshold):
        if value.shape[1] > threshold and chunks > 1:
            outputs = [
                FeedForward.forward(module, part)
                for part in torch.chunk(value, chunks, dim=1)
            ]
            return torch.cat(outputs, dim=1)
        return FeedForward.forward(module, value)

    def apply_rope(query, key, frequencies):
        cosine = frequencies[0].to(query.dtype)
        sine = frequencies[1].to(query.dtype)
        negative_sine = frequencies[2].to(query.dtype)
        query_output = query * cosine
        query_split = query_output.shape[-1] // 2
        query_output[..., :query_split].addcmul_(
            query[..., query_split:], negative_sine)
        query_output[..., query_split:].addcmul_(
            query[..., :query_split], sine)
        key_output = key * cosine
        key_split = key_output.shape[-1] // 2
        key_output[..., :key_split].addcmul_(
            key[..., key_split:], negative_sine)
        key_output[..., key_split:].addcmul_(
            key[..., :key_split], sine)
        return query_output, key_output

    def attention_forward(
        module, value, attention_mask, frequencies,
        transformer_options=None,
    ):
        batch, sequence, _ = value.shape
        query, key, result = module.qkv(value).view(
            batch, sequence, 3, module.num_heads,
            module.head_dim).unbind(dim=2)
        query = module.norm_q(query).transpose(1, 2)
        key = module.norm_k(key).transpose(1, 2)
        result = result.transpose(1, 2)
        query, key = apply_rope(query, key, frequencies)
        result = optimized_attention_masked(
            query, key, result, module.num_heads, attention_mask,
            skip_reshape=True,
            transformer_options=transformer_options or {})
        return module.o(result)

    for index, block in enumerate(layers):
        if chunk_ffn and ffn_chunks > 1:
            def patched_ffn(
                self_module, value, _chunks=ffn_chunks,
                _threshold=ffn_seq_threshold,
            ):
                return ffn_forward(
                    self_module, value, chunks=_chunks,
                    threshold=_threshold)

            model.add_object_patch(
                f"diffusion_model.layers.{index}.feed_forward.forward",
                types.MethodType(patched_ffn, block.feed_forward))
        if bf16_rope:
            model.add_object_patch(
                f"diffusion_model.layers.{index}.attention.forward",
                types.MethodType(attention_forward, block.attention))
    return model


def _ltx2_attention_tuner(
    patcher, blocks: list[int], video_scale: float, audio_scale: float,
    audio_to_video_scale: float, video_to_audio_scale: float,
    triton_kernels: bool,
):
    model = patcher.clone()
    diffusion_model = model.get_model_object("diffusion_model")
    transformer_blocks = getattr(
        diffusion_model, "transformer_blocks", None)
    if transformer_blocks is None:
        raise TransformError(
            "ltx2_attention_tuner needs LTX2 transformer blocks")
    if blocks:
        selected = set(blocks)
        if any(index < 0 or index >= len(transformer_blocks)
               for index in selected):
            raise TransformError(
                "ltx2_attention_tuner block index is out of range")
    else:
        selected = set(range(len(transformer_blocks)))
    for index in range(len(transformer_blocks)):
        scales = (
            video_scale,
            audio_scale,
            audio_to_video_scale,
            video_to_audio_scale,
        ) if index in selected else (1.0, 1.0, 1.0, 1.0)
        prefix = f"diffusion_model.transformer_blocks.{index}"
        for name, value in zip((
            "video_scale",
            "audio_scale",
            "audio_to_video_scale",
            "video_to_audio_scale",
        ), scales):
            model.add_object_patch(f"{prefix}.{name}", value)
        model.add_object_patch(
            f"{prefix}.use_triton_kernels", triton_kernels)
    return model


def _memory_efficient_sage(
    patcher, architecture: str, triton_kernels: bool,
):
    from . import _memory_attention

    if architecture == "ltx2":
        return _memory_attention.apply_ltx2(patcher, triton_kernels)
    if architecture == "minimax":
        return _memory_attention.apply_minimax(patcher)
    if architecture == "wan":
        return _memory_attention.apply_wan(patcher)
    raise TransformError(
        f"unsupported memory-efficient Sage architecture {architecture!r}")


def _minimax_chunk_feed_forward(
    patcher, chunks: int, seq_threshold: int,
):
    import types

    import torch
    import comfy.ops

    if chunks == 1:
        return patcher

    model = patcher.clone()
    diffusion_model = model.get_model_object("diffusion_model")
    blocks = getattr(diffusion_model, "blocks", None)
    if (
        not blocks
        or not hasattr(blocks[0], "mlp")
        or not hasattr(blocks[0].mlp, "fc1")
    ):
        return patcher

    linear_input_act = getattr(comfy.ops, "linear_input_act", None)
    if not callable(linear_input_act):
        raise TransformError(
            "MiniMax feed-forward chunking requires core "
            "comfy.ops.linear_input_act; update ComfyUI for MiniMax H3 support")

    for index, block in enumerate(blocks):
        def chunked(
            self_module, value, _chunks=chunks,
            _threshold=seq_threshold,
        ):
            self_module.kj_num_chunks = _chunks
            self_module.kj_seq_threshold = _threshold
            if value.shape[0] > _threshold and _chunks > 1:
                output = torch.empty_like(value)
                offset = 0
                for part in torch.chunk(value, _chunks, dim=0):
                    end = offset + part.shape[0]
                    output[offset:end] = linear_input_act(
                        self_module.fc2, self_module.fc1(part), "swiglu")
                    offset = end
                return output
            return linear_input_act(
                self_module.fc2, self_module.fc1(value), "swiglu")

        model.add_object_patch(
            f"diffusion_model.blocks.{index}.mlp.forward",
            types.MethodType(chunked, block.mlp))
    return model


def _minimax_low_vram_attention(patcher, head_chunks: int):
    import types

    import torch
    import comfy.model_management as model_management
    import comfy.quant_ops
    from comfy.ldm.modules.attention import optimized_attention

    model = patcher.clone()
    diffusion_model = model.get_model_object("diffusion_model")
    blocks = getattr(diffusion_model, "blocks", None)
    if (
        not blocks
        or not hasattr(blocks[0], "attn")
        or not hasattr(blocks[0].attn, "qkv_proj")
    ):
        return patcher

    try:
        from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift
    except ImportError as error:
        raise TransformError(
            "MiniMax low-VRAM attention requires core MiniMax H3 support; "
            "update ComfyUI") from error

    def attention_forward(
        self_module, value, rope_freqs=None, transformer_options={},
    ):
        if isinstance(value, list):
            value = value.pop()
        sequence = value.shape[0]
        device = value.device
        dtype = value.dtype
        qkv = self_module.qkv_proj(value)
        del value
        query, key, result = qkv.split(
            self_module.heads * self_module.head_dim, dim=-1)
        result = result.view(
            sequence, self_module.heads, self_module.head_dim)
        if rope_freqs is not None:
            query = query.view(
                1, sequence, self_module.heads, self_module.head_dim)
            key = key.view(
                1, sequence, self_module.heads, self_module.head_dim)
            query_weight = model_management.cast_to(
                self_module.q_norm.weight, device=device)
            key_weight = model_management.cast_to(
                self_module.k_norm.weight, device=device)
            rotation = rope_freqs.shape[-3] * 2
            if model_management.in_training:
                query, key = comfy.quant_ops.ck.rms_rope_split_half(
                    query, key, rope_freqs, query_weight, key_weight,
                    epsilon=self_module.q_norm.eps, rot_dim=rotation)
            else:
                comfy.quant_ops.ck.rms_rope_split_half_(
                    query, key, rope_freqs, query_weight, key_weight,
                    epsilon=self_module.q_norm.eps, rot_dim=rotation)
            query = query[0]
            key = key[0]
        else:
            query = self_module.q_norm(query.view(
                sequence, self_module.heads, self_module.head_dim))
            key = self_module.k_norm(key.view(
                sequence, self_module.heads, self_module.head_dim))
        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        result = result.transpose(0, 1).unsqueeze(0)
        groups = (
            min(transformer_options.get("minimax_head_chunks", 1),
                self_module.heads)
            if isinstance(transformer_options, dict) else 1
        )
        if groups <= 1:
            output = optimized_attention(
                query, key, result, self_module.heads, mask=None,
                skip_reshape=True,
                transformer_options=transformer_options).squeeze(0)
        else:
            output = torch.empty(
                (sequence, self_module.heads * self_module.head_dim),
                dtype=dtype, device=device)
            start = 0
            sizes = [
                self_module.heads // groups
                + (1 if index < self_module.heads % groups else 0)
                for index in range(groups)
            ]
            for size in sizes:
                stop = start + size
                current = optimized_attention(
                    query[:, start:stop], key[:, start:stop],
                    result[:, start:stop], size, mask=None,
                    skip_reshape=True,
                    transformer_options=transformer_options)
                output[:, start * self_module.head_dim:
                       stop * self_module.head_dim] = current.squeeze(0)
                start = stop
        del query, key, result, qkv
        return self_module.out_proj(output)

    attention_forward._uses_optimized_attention = True

    def block_forward(
        self_module, value, timestep, mod_segments, rope_freqs,
        transformer_options={},
    ):
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
         gate_mlp) = self_module.adaln_proj(timestep)
        hidden = [_mod_scale_shift(
            self_module.norm1(value), shift_msa, scale_msa, mod_segments)]
        value = _mod_gate(
            value, gate_msa,
            self_module.attn(
                hidden, rope_freqs=rope_freqs,
                transformer_options=transformer_options),
            mod_segments)
        hidden = _mod_scale_shift(
            self_module.norm2(value), shift_mlp, scale_mlp, mod_segments)
        return _mod_gate(
            value, gate_mlp, self_module.mlp(hidden), mod_segments)

    options = model.model_options.setdefault("transformer_options", {})
    if head_chunks > 1:
        options["minimax_head_chunks"] = head_chunks
    options["sol_take_forward"] = attention_forward
    for index, block in enumerate(blocks):
        prefix = f"diffusion_model.blocks.{index}"
        model.add_object_patch(
            f"{prefix}.forward", types.MethodType(block_forward, block))
        attention_key = f"{prefix}.attn.forward"
        if attention_key not in getattr(model, "object_patches", {}):
            model.add_object_patch(
                attention_key,
                types.MethodType(attention_forward, block.attn))
    return model


def _matmul_fp16_accumulation(patcher, enabled: bool):
    """Select fp16 matmul accumulation for each application of this model."""
    import torch

    if not hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
        if enabled:
            raise TransformError(
                "this torch build has no matmul.allow_fp16_accumulation "
                "(requires torch 2.7.0 or newer)")
        return patcher.clone()

    m = patcher.clone()
    options = m.model_options.setdefault("transformer_options", {})
    flags = dict(options.get("model_backend_flags", {}))
    flags["allow_fp16_accumulation"] = enabled
    options["model_backend_flags"] = flags
    return m


def _memory_usage_factor(patcher, factor: float):
    """Override the model's memory-usage estimate during sampling.

    The estimate drives how much core keeps resident. It is a hint, so a node
    correcting it for an architecture core estimates badly is legitimate — and
    it is a float, which is why this needs no new machinery at all.
    """
    from comfy.patcher_extension import WrappersMP

    def wrapper(executor, model, *args, **kwargs):
        adjusted = model.clone()
        original = adjusted.model.memory_usage_factor
        adjusted.model.memory_usage_factor = factor
        try:
            return executor(adjusted, *args, **kwargs)
        finally:
            adjusted.model.memory_usage_factor = original

    m = patcher.clone()
    m.add_wrapper_with_key(WrappersMP.PREPARE_SAMPLING,
                           "comfy.model_transform.memory_usage_factor", wrapper)
    return m


_FFN_TARGETS = ("blocks_ffn", "ltx_transformer_ff")


def _ffn_chunking(
    patcher, chunks: int, dim_threshold: int, target: str = "blocks_ffn",
):
    """Trade time for peak VRAM in the feed-forward blocks.

    Chunking activations along the token dimension is arithmetically identical
    to not chunking — the same function applied to slices of a sum-free
    elementwise path — so this changes memory, not results, up to float
    non-associativity.

    The target is a closed core-owned model layout, not an object path supplied
    by the guest. A model shaped differently is refused rather than silently
    left unpatched.
    """
    import torch

    if chunks == 1:
        return patcher.clone()

    m = patcher.clone()
    diffusion_model = m.get_model_object("diffusion_model")
    if target == "ltx_transformer_ff":
        import types

        blocks = getattr(diffusion_model, "transformer_blocks", None)
        if blocks is None:
            raise TransformError(
                "ffn_chunking target ltx_transformer_ff needs "
                "diffusion_model.transformer_blocks")
        feed_forwards = [getattr(block, "ff", None) for block in blocks]
        if not feed_forwards or any(
            feed_forward is None or not hasattr(feed_forward, "net")
            for feed_forward in feed_forwards
        ):
            raise TransformError(
                "ffn_chunking target ltx_transformer_ff needs every "
                "transformer block to expose ff.net")
        for index, feed_forward in enumerate(feed_forwards):
            def chunked(self_module, x, _chunks=chunks,
                        _threshold=dim_threshold):
                self_module.num_chunks = _chunks
                self_module.dim_threshold = _threshold
                if x.shape[1] > self_module.dim_threshold:
                    chunk_size = x.shape[1] // self_module.num_chunks
                    for chunk_index in range(self_module.num_chunks):
                        start = chunk_index * chunk_size
                        end = (
                            (chunk_index + 1) * chunk_size
                            if chunk_index < self_module.num_chunks - 1
                            else x.shape[1]
                        )
                        x[:, start:end] = self_module.net(x[:, start:end])
                    return x
                return self_module.net(x)

            m.add_object_patch(
                f"diffusion_model.transformer_blocks.{index}.ff.forward",
                types.MethodType(chunked, feed_forward),
            )
        return m

    if target != "blocks_ffn":
        raise TransformError(f"unknown ffn_chunking target {target!r}")

    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None:
        raise TransformError(
            "ffn_chunking needs a model with a `blocks` transformer stack; "
            f"{type(diffusion_model).__name__} has none")

    patched = 0
    for idx, block in enumerate(blocks):
        ffn = getattr(block, "ffn", None)
        if ffn is None:
            continue

        def chunked(*args, _ffn=ffn, **kwargs):
            x = args[0]
            if x.shape[1] <= dim_threshold:
                return _ffn.__class__.forward(_ffn, x, *args[1:], **kwargs)
            parts = [_ffn.__class__.forward(_ffn, c, *args[1:], **kwargs)
                     for c in torch.chunk(x, chunks, dim=1)]
            return torch.cat(parts, dim=1)

        m.add_object_patch(f"diffusion_model.blocks.{idx}.ffn.forward", chunked)
        patched += 1

    if patched == 0:
        raise TransformError(
            "ffn_chunking found no `ffn` submodule in any block of "
            f"{type(diffusion_model).__name__}")
    return m


_COMPILE_SCOPES = (
    "whole",
    "known_transformer_blocks",
    "flux_blocks",
    "wan_blocks",
)


def _compile_guard_filter(guard_entries):
    return [("transformer_options" not in entry.name)
            for entry in guard_entries]


_compile_aimdo_patched = False
_COMPILE_DYNAMO_POLICY_KEY = "comfy.model_transform.compile.dynamo_policy"


def _patch_aimdo_for_compile():
    global _compile_aimdo_patched
    if _compile_aimdo_patched:
        return
    _compile_aimdo_patched = True

    import torch
    import comfy.ops

    names = (
        "cast_bias_weight",
        "uncast_bias_weight",
        "cast_modules_with_vbar",
        "resolve_cast_module_with_vbar",
    )
    for name in names:
        fn = getattr(comfy.ops, name, None)
        if fn is not None:
            setattr(comfy.ops, name, torch._dynamo.disable(fn))
    try:
        import comfy_aimdo.torch as aimdo_torch
    except ImportError:
        return
    aimdo_torch.get_tensor_from_raw_ptr = torch._dynamo.disable(
        aimdo_torch.get_tensor_from_raw_ptr)


def _compile_keys(diffusion_model, scope: str, double_blocks: bool,
                  single_blocks: bool) -> list[str]:
    if scope == "whole":
        return ["diffusion_model"]

    keys = []
    if scope == "known_transformer_blocks":
        layer_types = (
            "double_blocks",
            "single_blocks",
            "layers",
            "transformer_blocks",
            "blocks",
            "visual_transformer_blocks",
            "text_transformer_blocks",
            "patch_blocks",
            "pixel_blocks",
        )
        for layer_name in layer_types:
            blocks = getattr(diffusion_model, layer_name, None)
            if blocks is not None:
                keys.extend(
                    f"diffusion_model.{layer_name}.{i}"
                    for i in range(len(blocks)))
    elif scope == "flux_blocks":
        if double_blocks:
            keys.extend(
                f"diffusion_model.double_blocks.{i}"
                for i in range(len(diffusion_model.double_blocks)))
        if single_blocks:
            keys.extend(
                f"diffusion_model.single_blocks.{i}"
                for i in range(len(diffusion_model.single_blocks)))
    elif scope == "wan_blocks":
        keys.extend(
            f"diffusion_model.blocks.{i}"
            for i in range(len(diffusion_model.blocks)))

    if not keys:
        if scope == "known_transformer_blocks":
            logging.warning(
                "No known transformer blocks found to compile, compiling "
                "entire diffusion model instead")
        return ["diffusion_model"]
    return keys


def _install_compile_dynamo_policy(model, settings: dict[str, Any]) -> None:
    from comfy.patcher_extension import WrappersMP

    model.remove_wrappers_with_key(
        WrappersMP.APPLY_MODEL, _COMPILE_DYNAMO_POLICY_KEY)
    if not settings:
        return

    def apply_with_dynamo_policy(executor, *args, **kwargs):
        import torch

        with torch._dynamo.config.patch(**settings):
            return executor(*args, **kwargs)

    model.add_wrapper_with_key(
        WrappersMP.APPLY_MODEL,
        _COMPILE_DYNAMO_POLICY_KEY,
        apply_with_dynamo_policy,
    )


def _compile(patcher, backend: str, mode: str, fullgraph: bool,
             dynamic: bool | None, scope: str = "whole",
             double_blocks: bool = True, single_blocks: bool = True,
             dynamo_cache_size_limit: int | None = None,
             force_parameter_static_shapes: bool | None = None,
             dynamic_vram: str = "disable", guard_filter: bool = False,
             debug_compile_keys: bool = False,
             default_mode: str = "omit"):
    """torch.compile the diffusion model.

    Core's compile wrapper swaps compiled modules only while BaseModel applies
    the model, preserving ModelPatcher's offload and low-VRAM lifecycle.
    """
    from comfy_api.torch_helpers import set_torch_compile_wrapper

    if dynamic_vram == "disable":
        m = patcher.clone(disable_dynamic=True)
    else:
        m = patcher.clone()

    diffusion_model = m.get_model_object("diffusion_model")
    keys = _compile_keys(diffusion_model, scope, double_blocks, single_blocks)
    if debug_compile_keys and keys != ["diffusion_model"]:
        logging.info("TorchCompileModelAdvanced: Compile key list:")
        for key in keys:
            logging.info(" - %s", key)

    if dynamic_vram == "stabilize" and m.is_dynamic():
        _patch_aimdo_for_compile()

    compile_kwargs = {
        "backend": backend,
        "mode": (None if mode == "default" and default_mode == "omit"
                 else mode),
        "fullgraph": fullgraph,
        "dynamic": dynamic,
        "keys": keys,
    }
    if guard_filter and mode == "default":
        compile_kwargs["options"] = {
            "guard_filter_fn": _compile_guard_filter,
        }

    dynamo_settings = {}
    if dynamo_cache_size_limit is not None:
        dynamo_settings["cache_size_limit"] = dynamo_cache_size_limit
    if force_parameter_static_shapes is not None:
        dynamo_settings["force_parameter_static_shapes"] = (
            force_parameter_static_shapes)

    if dynamo_settings:
        import torch

        with torch._dynamo.config.patch(**dynamo_settings):
            set_torch_compile_wrapper(m, **compile_kwargs)
    else:
        set_torch_compile_wrapper(m, **compile_kwargs)
    _install_compile_dynamo_policy(m, dynamo_settings)
    return m


_CONTEXT_SCHEDULES = (
    "standard_static",
    "standard_static_balanced",
    "standard_uniform",
    "looped_uniform",
    "batched",
    "batched_shifted",
)
_CONTEXT_FUSE_METHODS = (
    "pyramid",
    "relative",
    "flat",
    "overlap-linear",
    "hann",
    "gaussian",
)


def _context_windows_static_balanced(
    num_frames, handler, model_options=None,
):
    import math

    frame_count = int(num_frames)
    length = int(handler.context_length)
    overlap = int(handler.context_overlap)
    if frame_count <= length:
        return [list(range(frame_count))]
    stride = max(1, length - overlap)
    count = -(-(frame_count - length) // stride) + 1
    return [
        list(range(
            math.floor(index * (frame_count - length) / (count - 1) + 0.5),
            math.floor(index * (frame_count - length) / (count - 1) + 0.5)
            + length,
        ))
        for index in range(count)
    ]


def _context_windows_batched_shifted(
    num_frames, handler, model_options=None,
):
    from comfy.context_windows import ordered_halving

    frame_count = int(num_frames)
    length = int(handler.context_length)
    if frame_count <= length:
        return [list(range(frame_count))]
    offset = round(frame_count * ordered_halving(int(handler._step))) % length
    windows = []
    start = 0
    if offset > 0:
        windows.append(list(range(offset)))
        start = offset
    while start < frame_count:
        end = min(start + length, frame_count)
        windows.append(list(range(start, end)))
        start = end
    return windows


def _context_weights_hann(
    length, full_length=None, idxs=None, handler=None, **kwargs,
):
    import math

    weights = [1.0] * length
    overlap = min(
        max(int(getattr(handler, "context_overlap", 0) or 0), 0), length)
    if overlap >= 1 and idxs:
        denominator = max(overlap - 1, 1)
        if min(idxs) > 0:
            for index in range(overlap):
                weights[index] = max(
                    0.5 * (1 - math.cos(math.pi * index / denominator)),
                    1e-37)
        if full_length is not None and max(idxs) < full_length - 1:
            for index in range(overlap):
                weights[length - overlap + index] = max(
                    0.5 * (1 - math.cos(
                        math.pi * (denominator - index) / denominator)),
                    1e-37)
    return weights


def _context_weights_gaussian(length, **kwargs):
    import math

    if length <= 1:
        return [1.0] * length
    center = (length - 1) / 2.0
    standard_deviation = max(length / 4.0, 1e-6)
    return [
        math.exp(-0.5 * ((index - center) / standard_deviation) ** 2)
        for index in range(length)
    ]


def _context_windows(
    patcher, context_schedule: str, fuse_method: str,
    context_length: int, context_overlap: int, context_stride: int,
    closed_loop: bool, dim: int, freenoise: bool,
    causal_window_fix: bool, cond_retain_indices: list[int],
):
    import comfy.context_windows as context_windows

    local_schedules = {
        "standard_static_balanced": _context_windows_static_balanced,
        "batched_shifted": _context_windows_batched_shifted,
    }
    local_fuse_methods = {
        "hann": _context_weights_hann,
        "gaussian": _context_weights_gaussian,
    }
    if context_schedule in local_schedules:
        schedule = context_windows.ContextSchedule(
            context_schedule, local_schedules[context_schedule])
    else:
        schedule = context_windows.get_matching_context_schedule(
            context_schedule)
    if fuse_method in local_fuse_methods:
        fuse = context_windows.ContextFuseMethod(
            fuse_method, local_fuse_methods[fuse_method])
    else:
        fuse = context_windows.get_matching_fuse_method(fuse_method)

    model = patcher.clone()
    handler = context_windows.IndexListContextHandler(
        context_schedule=schedule,
        fuse_method=fuse,
        context_length=context_length,
        context_overlap=context_overlap,
        context_stride=context_stride,
        closed_loop=closed_loop,
        dim=dim,
        freenoise=freenoise,
        causal_window_fix=causal_window_fix,
        cond_retain_index_list=",".join(
            str(index) for index in cond_retain_indices),
    )
    model.model_options["context_handler"] = handler
    context_windows.create_prepare_sampling_wrapper(model)
    if freenoise:
        context_windows.create_sampler_sample_wrapper(model)
    return model


def _cfg_zero_star(patcher, use_zero_init: bool, zero_init_steps: int):
    import torch

    def cfg_zero_star(args):
        cond = args["cond"]
        timestep = args["timestep"]
        sigmas = args["model_options"]["transformer_options"]["sample_sigmas"]
        matched = (sigmas == timestep[0]).nonzero()
        if len(matched) > 0:
            current_step = matched.item()
        else:
            current_step = 0
            for index in range(len(sigmas) - 1):
                if ((sigmas[index] - timestep[0])
                        * (sigmas[index + 1] - timestep[0])) <= 0:
                    current_step = index
                    break

        if use_zero_init and current_step <= zero_init_steps:
            return cond * 0

        uncond = args["uncond"]
        cond_scale = args["cond_scale"]
        batch_size = cond.shape[0]
        positive_flat = cond.view(batch_size, -1)
        negative_flat = uncond.view(batch_size, -1)
        dot_product = torch.sum(
            positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm = torch.sum(
            negative_flat ** 2, dim=1, keepdim=True) + 1e-8
        alpha = (dot_product / squared_norm).view(
            batch_size, *([1] * (len(cond.shape) - 1)))
        return uncond * alpha + cond_scale * (cond - uncond * alpha)

    model = patcher.clone()
    model.set_model_sampler_cfg_function(cfg_zero_star)
    return model


_PID_BIAS_COEF_FLUX2 = (
    (-0.130306, +0.127184, +0.014058),
    (-0.053279, -0.408929, +0.004243),
    (-0.009386, +0.109546, -0.134091),
    (-0.033373, -0.011615, -0.026129),
    (+0.180052, +0.062021, +0.071317),
    (-0.067958, -0.058595, -0.098645),
    (-0.248116, -0.240633, -0.105600),
    (+0.304035, +0.322566, +0.093224),
    (-0.157648, -0.227127, -0.112368),
    (-0.062814, +0.030765, +0.062735),
)


def _pid_color_bias(patcher, strength: float, backbone: str):
    import torch

    if strength == 0.0:
        return patcher

    def correct(args):
        denoised = args["denoised"]
        try:
            sigmas = args["model_options"]["transformer_options"][
                "sample_sigmas"]
            sigma = args.get("sigma", args.get("timestep"))
            if sigma is None or not torch.isclose(
                    sigma.max(), sigmas[0]).item():
                return denoised
        except (KeyError, AttributeError):
            sigma = args.get("sigma")
            if sigma is None or sigma.max().item() < 0.95:
                return denoised

        coef = torch.tensor(
            _PID_BIAS_COEF_FLUX2,
            device=denoised.device,
            dtype=denoised.dtype,
        )
        rgb_mean = denoised.mean(dim=(0, 2, 3))
        rgb_std = denoised.std(dim=(0, 2, 3))
        features = torch.stack((
            rgb_mean[0], rgb_mean[1], rgb_mean[2],
            rgb_std[0], rgb_std[1], rgb_std[2],
            rgb_mean[0] * rgb_mean[1],
            rgb_mean[0] * rgb_mean[2],
            rgb_mean[1] * rgb_mean[2],
            denoised.new_tensor(1.0),
        ))
        bias = features @ coef
        return denoised - strength * bias.view(1, 3, 1, 1)

    model = patcher.clone()
    model.set_model_sampler_post_cfg_function(correct)
    return model


_DYNAMIC_THRESHOLD_MODES = (
    "Constant",
    "Linear Down",
    "Cosine Down",
    "Half Cosine Down",
    "Linear Up",
    "Cosine Up",
    "Half Cosine Up",
    "Power Up",
    "Power Down",
    "Linear Repeating",
    "Cosine Repeating",
    "Sawtooth",
)


def _dynamic_threshold_scale(
    value: float, minimum: float, mode: str, fraction: float,
    schedule_value: float,
) -> float:
    """Evaluate one bounded Dynamic Thresholding schedule."""
    import math

    amplitude = value - minimum
    if mode == "Linear Down":
        amplitude *= 1.0 - fraction
    elif mode == "Half Cosine Down":
        amplitude *= math.cos(fraction)
    elif mode == "Cosine Down":
        amplitude *= math.cos(fraction * 1.5707)
    elif mode == "Linear Up":
        amplitude *= fraction
    elif mode == "Half Cosine Up":
        amplitude *= 1.0 - math.cos(fraction)
    elif mode == "Cosine Up":
        amplitude *= 1.0 - math.cos(fraction * 1.5707)
    elif mode == "Power Up":
        amplitude *= math.pow(fraction, schedule_value)
    elif mode == "Power Down":
        amplitude *= 1.0 - math.pow(fraction, schedule_value)
    elif mode == "Linear Repeating":
        portion = (fraction * schedule_value) % 1.0
        amplitude *= (
            (0.5 - portion) * 2.0
            if portion < 0.5
            else (portion - 0.5) * 2.0
        )
    elif mode == "Cosine Repeating":
        amplitude *= (
            math.cos(fraction * 6.28318 * schedule_value) * 0.5
            + 0.5
        )
    elif mode == "Sawtooth":
        amplitude *= (fraction * schedule_value) % 1.0
    return amplitude + minimum


def _dynamic_threshold_result(
    cond, uncond, cfg_scale,
    *,
    fraction: float,
    mimic_scale: float,
    threshold_percentile: float,
    mimic_mode: str,
    mimic_scale_min: float,
    cfg_mode: str,
    cfg_scale_min: float,
    schedule_value: float,
    separate_feature_channels: bool,
    scaling_startpoint: str,
    variability_measure: str,
    interpolate_phi: float,
):
    """Return canonical Dynamic Thresholding guidance for one sample step."""
    import torch

    if cond.ndim < 3 or uncond.ndim != cond.ndim:
        raise ValueError(
            "dynamic_thresholding needs matching batched latent predictions")
    if uncond.shape[0] < 1 or cond.shape[0] % uncond.shape[0] != 0:
        raise ValueError(
            "dynamic_thresholding needs a constant number of conditions "
            "per batch item")
    if tuple(cond.shape[1:]) != tuple(uncond.shape[1:]):
        raise ValueError(
            "dynamic_thresholding conditional and unconditional shapes differ")

    mimic = _dynamic_threshold_scale(
        mimic_scale, mimic_scale_min, mimic_mode, fraction, schedule_value)
    cfg = _dynamic_threshold_scale(
        float(cfg_scale), cfg_scale_min, cfg_mode, fraction, schedule_value)
    conditions_per_batch = cond.shape[0] // uncond.shape[0]
    cond_stacked = cond.reshape(
        (-1, conditions_per_batch) + tuple(uncond.shape[1:]))
    relative = (cond_stacked - uncond.unsqueeze(1)).sum(1)
    mimic_target = uncond + relative * mimic
    cfg_target = uncond + relative * cfg

    mimic_flat = mimic_target.flatten(2)
    cfg_flat = cfg_target.flatten(2)
    mimic_mean = mimic_flat.mean(dim=2, keepdim=True)
    cfg_mean = cfg_flat.mean(dim=2, keepdim=True)
    mimic_centered = mimic_flat - mimic_mean
    cfg_centered = cfg_flat - cfg_mean

    if separate_feature_channels:
        if variability_measure == "STD":
            mimic_reference = mimic_centered.std(
                dim=2, keepdim=True)
            cfg_reference = cfg_centered.std(
                dim=2, keepdim=True)
        else:
            mimic_reference = mimic_centered.abs().amax(
                dim=2, keepdim=True)
            cfg_reference = torch.quantile(
                cfg_centered.abs(), threshold_percentile,
                dim=2, keepdim=True)
    elif variability_measure == "STD":
        mimic_reference = mimic_centered.std()
        cfg_reference = cfg_centered.std()
    else:
        mimic_reference = mimic_centered.abs().amax()
        cfg_reference = torch.quantile(
            cfg_centered.abs(), threshold_percentile)

    if scaling_startpoint == "ZERO":
        result = cfg_flat * (mimic_reference / cfg_reference)
    elif variability_measure == "STD":
        result = (
            cfg_centered / cfg_reference * mimic_reference
            + cfg_mean
        )
    else:
        maximum = torch.maximum(mimic_reference, cfg_reference)
        result = (
            cfg_centered.clamp(-maximum, maximum)
            / maximum
            * mimic_reference
            + cfg_mean
        )

    result = result.reshape_as(mimic_target)
    if interpolate_phi != 1.0:
        result = (
            result * interpolate_phi
            + cfg_target * (1.0 - interpolate_phi)
        )
    return result


def _dynamic_thresholding(
    patcher,
    mimic_scale: float,
    threshold_percentile: float,
    mimic_mode: str,
    mimic_scale_min: float,
    cfg_mode: str,
    cfg_scale_min: float,
    schedule_value: float,
    separate_feature_channels: bool,
    scaling_startpoint: str,
    variability_measure: str,
    interpolate_phi: float,
):
    """Install core's closed, data-configured Dynamic Thresholding behavior."""
    import torch

    model_sampling = patcher.get_model_object("model_sampling")

    def dynamic_cfg(args):
        input_value = args["input"]
        cond = input_value - args["cond"]
        uncond = input_value - args["uncond"]
        sigma = args.get("sigma")
        if not isinstance(sigma, torch.Tensor) or sigma.numel() == 0:
            raise ValueError(
                "dynamic_thresholding needs the current diffusion sigma")
        timestep = model_sampling.timestep(sigma)
        if not isinstance(timestep, torch.Tensor) or timestep.numel() == 0:
            raise ValueError(
                "dynamic_thresholding model did not produce a diffusion "
                "timestep")
        # The pinned implementation intentionally uses max_steps=999 and then
        # divides by max_steps-1.  Its final t=0 point is therefore slightly
        # above one; preserve that endpoint instead of silently clamping it.
        fraction = (
            999.0 - float(timestep.reshape(-1)[0].item())
        ) / 998.0
        if float(args["cond_scale"]) == mimic_scale:
            return input_value - (
                uncond + (cond - uncond) * float(args["cond_scale"])
            )
        guided = _dynamic_threshold_result(
            cond,
            uncond,
            args["cond_scale"],
            fraction=fraction,
            mimic_scale=mimic_scale,
            threshold_percentile=threshold_percentile,
            mimic_mode=mimic_mode,
            mimic_scale_min=mimic_scale_min,
            cfg_mode=cfg_mode,
            cfg_scale_min=cfg_scale_min,
            schedule_value=schedule_value,
            separate_feature_channels=separate_feature_channels,
            scaling_startpoint=scaling_startpoint,
            variability_measure=variability_measure,
            interpolate_phi=interpolate_phi,
        )
        return input_value - guided

    model = patcher.clone()
    model.set_model_sampler_cfg_function(dynamic_cfg)
    return model


def _style_aligned_expand_reference(value, scale: float = 1.0):
    """Expand the first item in each CFG half across that half's batch."""
    import torch

    batch = int(value.shape[0])
    if batch < 2 or batch % 2:
        return value
    half = batch // 2
    references = torch.stack((value[0], value[half]), dim=0).unsqueeze(1)
    references = references.expand(
        (2, half) + tuple(value.shape[1:]))
    if scale != 1.0 and half > 1:
        references = references.clone()
        references[:, 1:] *= scale
    return references.reshape_as(value)


def _style_aligned_adain(value):
    mean = value.mean(dim=-2, keepdim=True)
    std = value.var(dim=-2, keepdim=True, correction=0).add(1e-5).sqrt()
    reference_mean = _style_aligned_expand_reference(mean)
    reference_std = _style_aligned_expand_reference(std)
    return (value - mean) / std * reference_std + reference_mean


def _style_aligned_batch(
    patcher, share_norm: str, share_attention: str, scale: float,
):
    """Share reference style statistics within a classic UNet image batch."""
    import types

    import torch
    import torch.nn as nn

    diffusion = getattr(getattr(patcher, "model", None), "diffusion_model", None)
    if (
        diffusion is None
        or not hasattr(diffusion, "input_blocks")
        or not callable(getattr(diffusion, "named_modules", None))
    ):
        raise TransformError(
            "style_aligned_batch needs a classic latent-diffusion UNet")

    model = patcher.clone()
    if hasattr(model, "disable_model_cfg1_optimization"):
        model.disable_model_cfg1_optimization()

    share_group = share_norm in {"group", "both"}
    share_layer = share_norm in {"layer", "both"}
    for name, module in diffusion.named_modules():
        if not name or not (
            (share_group and isinstance(module, nn.GroupNorm))
            or (share_layer and isinstance(module, nn.LayerNorm))
        ):
            continue
        key = f"diffusion_model.{name}.forward"
        original = patcher.get_model_object(key)

        def shared_norm(
            self_module, hidden_states, *args,
            _original=original, **kwargs,
        ):
            if hidden_states.ndim < 3 or hidden_states.shape[0] % 2:
                return _original(hidden_states, *args, **kwargs)
            tokens = hidden_states.shape[-2]
            shared = _style_aligned_expand_reference(
                hidden_states, scale=1.0)
            shared = torch.cat((hidden_states, shared), dim=-2)
            normalized = _original(shared, *args, **kwargs)
            return normalized[..., :tokens, :]

        model.add_object_patch(key, types.MethodType(shared_norm, module))

    if share_attention != "disabled":
        adain_queries = "q" in share_attention
        adain_keys = "k" in share_attention
        adain_values = "v" in share_attention

        def shared_attention(q, k, v, _extra_options=None):
            if q.shape[0] % 2:
                return q, k, v
            if adain_queries:
                q = _style_aligned_adain(q)
            if adain_keys:
                k = _style_aligned_adain(k)
            if adain_values:
                v = _style_aligned_adain(v)
            k = torch.cat(
                (k, _style_aligned_expand_reference(k, scale=scale)),
                dim=-2,
            )
            v = torch.cat(
                (v, _style_aligned_expand_reference(v)), dim=-2)
            return q, k, v

        model.set_model_attn1_patch(shared_attention)
    return model


def _controlnet_lllite(
    patcher, adapter: str, image, strength: float, steps: int,
    start_percent: float, end_percent: float,
):
    """Apply the canonical legacy SD ControlNet-LLLite attention adapter."""
    import sys

    import folder_paths
    import nodes
    import torch

    if (
        not isinstance(image, torch.Tensor)
        or image.ndim != 4
        or not 1 <= image.shape[0] <= 4096
        or image.shape[-1] < 3
    ):
        raise TransformError(
            "controlnet_lllite needs a bounded BHWC RGB image batch")
    height, width = map(int, image.shape[1:3])
    if (
        height <= 0
        or width <= 0
        or height * width * int(image.shape[0]) > 268_435_456
    ):
        raise TransformError("controlnet_lllite image dimensions are invalid")
    effective_end = 100.0 if end_percent == 0.0 else end_percent
    if start_percent > effective_end:
        raise TransformError(
            "controlnet_lllite start_percent must not exceed end_percent")

    path = folder_paths.get_full_path_or_raise("controlnet", adapter)
    node_class = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get("LLLiteLoader")
    module = (
        None if node_class is None
        else sys.modules.get(getattr(node_class, "__module__", ""))
    )
    load_patch = getattr(module, "load_control_net_lllite_patch", None)
    if not callable(load_patch):
        raise TransformError(
            "controlnet_lllite requires the host-installed canonical "
            "kohya-ss ControlNet-LLLite-ComfyUI extension")
    patch = load_patch(
        path,
        image[..., :3],
        strength,
        steps,
        start_percent,
        end_percent,
    )
    if not callable(patch):
        raise TransformError("ControlNet-LLLite returned no attention patch")
    model = patcher.clone()
    model.set_model_attn1_patch(patch)
    model.set_model_attn2_patch(patch)
    return model


def _differential_diffusion(patcher, strength: float):
    from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion

    return DifferentialDiffusion.execute(patcher, strength).result[0]


def _fooocus_inpaint(patcher, latent, head: str, patch: str):
    """Apply Fooocus' canonical SDXL inpaint head and quantized delta.

    This is a closed model transform because the only irreducible part of the
    behavior is installing a host-side UNet callback. Packs still choose and
    declare the weights and own their inpaint-pipeline orchestration.
    """
    import torch
    import torch.nn.functional as F
    import comfy.lora
    import comfy.utils
    import folder_paths

    if not isinstance(latent, dict):
        raise TransformError("fooocus_inpaint.latent is not a latent payload")
    samples = latent.get("samples")
    noise_mask = latent.get("noise_mask")
    if (
        not isinstance(samples, torch.Tensor)
        or samples.ndim != 4
        or samples.shape[1] != 4
        or not isinstance(noise_mask, torch.Tensor)
        or noise_mask.ndim not in (3, 4)
    ):
        raise TransformError(
            "fooocus_inpaint needs four-channel samples and a noise mask")
    if noise_mask.ndim == 3:
        noise_mask = noise_mask.unsqueeze(1)
    if noise_mask.shape[1] != 1:
        raise TransformError("fooocus_inpaint noise mask must have one channel")
    if noise_mask.shape[0] not in (1, samples.shape[0]):
        raise TransformError(
            "fooocus_inpaint sample and noise-mask batches do not match")
    head_path = folder_paths.get_full_path_or_raise("inpaint", head)
    patch_path = folder_paths.get_full_path_or_raise("inpaint", patch)
    head_state = comfy.utils.load_torch_file(head_path, safe_load=True)
    head_weight = head_state.get("head") if isinstance(head_state, dict) else None
    if (
        not isinstance(head_weight, torch.Tensor)
        or tuple(head_weight.shape) != (320, 5, 3, 3)
        or len(head_state) != 1
    ):
        raise TransformError("Fooocus inpaint head has an invalid state dict")

    base_model = patcher.model
    latent_pixels = base_model.process_latent_in(samples)
    if tuple(noise_mask.shape[-2:]) == tuple(latent_pixels.shape[-2:]):
        latent_mask = noise_mask.round().to(latent_pixels)
    else:
        latent_mask = F.max_pool2d(
            noise_mask.round(), (8, 8)).round().to(latent_pixels)
    if latent_mask.shape[0] == 1 and latent_pixels.shape[0] > 1:
        latent_mask = latent_mask.repeat(latent_pixels.shape[0], 1, 1, 1)
    if tuple(latent_mask.shape[-2:]) != tuple(latent_pixels.shape[-2:]):
        raise TransformError(
            "Fooocus mask pooling does not match the model latent dimensions")
    feed = torch.cat([latent_mask, latent_pixels], dim=1)
    feature = F.conv2d(
        F.pad(feed, (1, 1, 1, 1), "replicate"),
        head_weight.to(device=feed.device, dtype=feed.dtype),
    )

    patch_state = comfy.utils.load_torch_file(patch_path, safe_load=True)
    if not isinstance(patch_state, dict):
        raise TransformError("Fooocus inpaint patch is not a state dict")
    model_keys = comfy.lora.model_lora_keys_unet(base_model, {})
    model_keys.update({key: key for key in base_model.state_dict().keys()})
    loaded = {}
    for key in model_keys.values():
        value = patch_state.get(key)
        if value is None:
            continue
        if (
            not isinstance(value, (tuple, list))
            or len(value) != 3
            or not all(isinstance(item, torch.Tensor) for item in value)
        ):
            raise TransformError(
                f"Fooocus patch entry {key!r} is not a quantized weight")
        loaded[key] = ("fooocus", tuple(value))
    if not loaded:
        raise TransformError(
            "Fooocus inpaint patch has no weights for this model")

    def input_block_patch(hidden, transformer_options):
        block = transformer_options.get("block")
        if isinstance(block, (tuple, list)) and len(block) > 1 and block[1] == 0:
            if hidden.shape[1:] != feature.shape[1:]:
                raise RuntimeError(
                    "Fooocus inpaint feature does not match the UNet input block")
            if hidden.shape[0] % feature.shape[0] != 0:
                raise RuntimeError(
                    "Fooocus inpaint feature batch does not match the UNet batch")
            repeated = feature.to(hidden).repeat(
                hidden.shape[0] // feature.shape[0], 1, 1, 1)
            hidden = hidden + repeated
        return hidden

    model = patcher.clone()
    model.set_model_input_block_patch(input_block_patch)
    patched = set(model.add_patches(loaded, 1.0))
    missing = set(loaded) - patched
    if missing:
        logging.warning(
            "Fooocus inpaint could not attach %d model weights", len(missing))
    model.model_options.setdefault("transformer_options", {})["fooocus"] = True
    return model


def _diffusion_weight_delta(
    patcher, model_patch: str, strength: float,
    pad_input_channels: bool,
):
    """Apply a shape-checked SafeTensors diffusion-model delta.

    This is deliberately architecture-neutral. The file may only change keys
    already present in the model. The sole permitted shape change is widening
    the first convolution's input channels when explicitly requested.
    """
    import torch
    import comfy.utils
    import folder_paths

    path = folder_paths.get_full_path_or_raise("model_patches", model_patch)
    state = comfy.utils.load_torch_file(path, safe_load=True)
    if not isinstance(state, dict) or not state or len(state) > 100_000:
        raise TransformError(
            "diffusion_weight_delta requires a non-empty tensor state dict")

    prefixes = ("model.diffusion_model.", "diffusion_model.")
    normalized = {}
    for source_key, value in state.items():
        if not isinstance(source_key, str) or not isinstance(value, torch.Tensor):
            raise TransformError(
                "diffusion_weight_delta accepts tensor-only SafeTensors weights")
        key = source_key
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        if key in normalized:
            raise TransformError(
                f"diffusion_weight_delta has duplicate key {key!r}")
        normalized[key] = value

    diffusion = getattr(getattr(patcher, "model", None), "diffusion_model", None)
    if diffusion is None or not callable(getattr(diffusion, "state_dict", None)):
        raise TransformError(
            "diffusion_weight_delta needs a model with diffusion weights")
    target = diffusion.state_dict()
    input_key = "input_blocks.0.0.weight"
    patches = {}
    for key, value in normalized.items():
        existing = target.get(key)
        if existing is None:
            raise TransformError(
                f"diffusion_weight_delta key {key!r} is absent from the model")
        can_pad = bool(pad_input_channels and key == input_key)
        if tuple(value.shape) != tuple(existing.shape):
            valid_padding = (
                can_pad
                and value.ndim == existing.ndim == 4
                and value.shape[0] == existing.shape[0]
                and value.shape[1] >= existing.shape[1]
                and tuple(value.shape[2:]) == tuple(existing.shape[2:])
            )
            if not valid_padding:
                raise TransformError(
                    f"diffusion_weight_delta shape mismatch for {key!r}: "
                    f"{tuple(value.shape)} != {tuple(existing.shape)}")
        patches["diffusion_model." + key] = (
            "diff", (value, {"pad_weight": can_pad}))

    result = patcher.clone()
    loaded = result.add_patches(patches, float(strength))
    if loaded is not None and len(loaded) != len(patches):
        raise TransformError(
            "diffusion_weight_delta did not match every model weight")
    return result


def _serialized_model_patch(
    patcher, model_patch: str, strength: float, pad_diff_weights: bool,
):
    """Apply a tensor-only serialized Comfy model patch.

    Some model authors publish Comfy patch tuples in SafeTensors by encoding
    ``model-key::patch-type::slot`` in each tensor name.  Parsing that small
    interchange format is a reusable engine capability; the model-specific
    orchestration that chooses a file and conditioning remains pack-side.
    Only diffusion-model ``diff`` and ``lora`` patches are accepted here.
    """
    import torch
    import comfy.utils
    import folder_paths

    path = folder_paths.get_full_path_or_raise("model_patches", model_patch)
    state = comfy.utils.load_torch_file(path, safe_load=True)
    if not isinstance(state, dict) or not state or len(state) > 100_000:
        raise TransformError(
            "serialized_model_patch requires a non-empty tensor state dict")

    grouped: dict[str, tuple[str, list[Any]]] = {}
    for encoded_key, tensor in state.items():
        if not isinstance(encoded_key, str) or not isinstance(tensor, torch.Tensor):
            raise TransformError(
                "serialized_model_patch accepts tensor-only SafeTensors weights")
        parts = encoded_key.rsplit("::", 2)
        if len(parts) != 3:
            raise TransformError(
                "serialized_model_patch tensor names must use "
                "model-key::patch-type::slot")
        model_key, patch_type, slot_text = parts
        if (
            not model_key.startswith("diffusion_model.")
            or len(model_key) > 1024
            or patch_type not in {"diff", "lora"}
        ):
            raise TransformError(
                "serialized_model_patch contains an unsupported model key or "
                "patch type")
        try:
            slot = int(slot_text)
        except ValueError as exc:
            raise TransformError(
                "serialized_model_patch slots must be integers") from exc
        if not 0 <= slot < 16 or str(slot) != slot_text:
            raise TransformError(
                "serialized_model_patch slots must be canonical integers in "
                "[0, 15]")
        current = grouped.get(model_key)
        if current is None:
            current = (patch_type, [None] * 16)
            grouped[model_key] = current
        elif current[0] != patch_type:
            raise TransformError(
                f"serialized_model_patch mixes patch types for {model_key!r}")
        if current[1][slot] is not None:
            raise TransformError(
                f"serialized_model_patch repeats slot {slot} for {model_key!r}")
        current[1][slot] = tensor

    patches = {}
    for model_key, (patch_type, values) in grouped.items():
        required_slots = (0,) if patch_type == "diff" else (0, 1)
        if any(values[index] is None for index in required_slots):
            raise TransformError(
                f"serialized_model_patch has an incomplete {patch_type} patch "
                f"for {model_key!r}")
        if patch_type == "diff" and pad_diff_weights:
            patches[model_key] = (
                "diff", [values[0], {"pad_weight": True}])
        else:
            patches[model_key] = (patch_type, values)

    result = patcher.clone()
    loaded = result.add_patches(patches, float(strength))
    if loaded is not None and len(loaded) != len(patches):
        raise TransformError(
            "serialized_model_patch did not match every model weight")
    return result


def _layer_diffusion_attention_sharing(
    patcher, model_patch: str, frames: int, control_image=None,
    first_conditioning=None, second_conditioning=None,
    third_conditioning=None,
):
    """Bridge the canonical SD1 Layer Diffusion attention-sharing patcher.

    The attention implementation remains owned by ComfyUI-layerdiffuse.  Core
    supplies only a confined SafeTensor state dict and bounded typed inputs;
    it does not carry a copy of the extension's model algorithm.
    """
    import sys
    import torch
    import comfy.supported_models
    import comfy.utils
    import folder_paths
    import nodes

    config = getattr(getattr(patcher, "model", None), "model_config", None)
    if not isinstance(
        config, (comfy.supported_models.SD15, comfy.supported_models.SD20),
    ):
        raise TransformError(
            "layer_diffusion_attention_sharing requires an SD1.x model")

    path = folder_paths.get_full_path_or_raise("model_patches", model_patch)
    state = comfy.utils.load_torch_file(path, safe_load=True)
    if (
        not isinstance(state, dict)
        or not state
        or len(state) > 100_000
        or any(
            not isinstance(key, str) or not isinstance(value, torch.Tensor)
            for key, value in state.items()
        )
    ):
        raise TransformError(
            "Layer Diffusion attention weights must be a tensor-only "
            "SafeTensors state dict")

    node_class = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get(
        "LayeredDiffusionApply")
    module = (
        None if node_class is None
        else sys.modules.get(getattr(node_class, "__module__", ""))
    )
    attention_patcher = getattr(module, "AttentionSharingPatcher", None)
    if not callable(attention_patcher):
        raise TransformError(
            "layer_diffusion_attention_sharing requires the host-installed "
            "canonical ComfyUI-layerdiffuse extension")

    control = None
    if control_image is not None:
        if (
            not isinstance(control_image, torch.Tensor)
            or control_image.ndim != 4
            or not 1 <= control_image.shape[0] <= 64
            or control_image.shape[-1] < 3
            or control_image.numel() > 268_435_456
        ):
            raise TransformError(
                "Layer Diffusion control_image must be a bounded BHWC image")
        control = control_image[..., :3].movedim(-1, 1)

    result = patcher.clone()
    adapter = attention_patcher(
        result, int(frames), use_control=control is not None)
    adapter.load_state_dict(state, strict=True)
    if control is not None:
        adapter.set_control(control)

    conditionings = (
        first_conditioning, second_conditioning, third_conditioning)
    active = conditionings[:int(frames)]
    if int(frames) > 1 or any(value is not None for value in active):
        overwritten = []
        for index, conditioning in enumerate(active):
            if conditioning is None:
                overwritten.append(None)
                continue
            if (
                not isinstance(conditioning, (list, tuple))
                or not conditioning
                or not isinstance(conditioning[0], (list, tuple))
                or not conditioning[0]
                or not isinstance(conditioning[0][0], torch.Tensor)
            ):
                raise TransformError(
                    "Layer Diffusion conditioning must contain a host "
                    f"embedding row at position {index}")
            overwritten.append(conditioning[0][0])
        result.model_options.setdefault("transformer_options", {})[
            "cond_overwrite"] = overwritten
    return result


def _concat_latent_input(patcher, latent):
    """Inject an encoded latent as bounded model ``c_concat`` input."""
    import torch

    samples = latent.get("samples") if isinstance(latent, dict) else None
    if (
        not isinstance(samples, torch.Tensor)
        or samples.ndim != 4
        or not 1 <= samples.shape[0] <= 64
        or not 1 <= samples.shape[1] <= 64
        or samples.shape[2] <= 0
        or samples.shape[3] <= 0
        or samples.numel() > 268_435_456
    ):
        raise TransformError(
            "concat_latent_input needs a bounded BCHW latent tensor")
    latent_format = getattr(
        getattr(getattr(patcher, "model", None), "model_config", None),
        "latent_format", None)
    scale_factor = getattr(latent_format, "scale_factor", None)
    if not isinstance(scale_factor, (int, float)):
        raise TransformError(
            "concat_latent_input needs a model latent scale factor")
    concat = torch.cat(
        [item.detach().unsqueeze(0) for item in samples], dim=1,
    ) * float(scale_factor)

    result = patcher.clone()
    existing_wrapper = result.model_options.get("model_function_wrapper")

    def wrapper(apply_model, args):
        updated = dict(args)
        conditioning = dict(args["c"])
        sample = args["input"]
        conditioning["c_concat"] = concat.to(sample).repeat(
            sample.shape[0], 1, 1, 1)
        updated["c"] = conditioning
        if existing_wrapper is not None:
            return existing_wrapper(apply_model, updated)
        return apply_model(
            x=updated["input"], t=updated["timestep"], **conditioning)

    result.set_model_unet_function_wrapper(wrapper)
    return result


def _sampling_memory_report(patcher):
    import torch
    from comfy import model_management
    from comfy.patcher_extension import CallbacksMP

    device = model_management.get_torch_device()

    def reset(_model):
        torch.cuda.reset_peak_memory_stats(device)

    def report(_model):
        allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
        logging.info(
            "Sampling max allocated memory: max_memory=%.3f GB", allocated)
        logging.info(
            "Sampling max reserved memory: max_reserved=%.3f GB", reserved)

    model = patcher.clone()
    model.add_callback(CallbacksMP.ON_PRE_RUN, reset)
    model.add_callback(CallbacksMP.ON_CLEANUP, report)
    return model


def _riflex_rope(
    patcher, architecture: str, num_frames: int, intrinsic_frequency: int,
):
    import torch
    from comfy import model_management

    diffusion_model = patcher.model.diffusion_model
    if architecture == "wan":
        dim = diffusion_model.dim // diffusion_model.num_heads
        theta = 10000.0
        axes_dim = [
            dim - 4 * (dim // 6),
            2 * (dim // 6),
            2 * (dim // 6),
        ]
        patch_key = "diffusion_model.rope_embedder"
    else:
        params = diffusion_model.params
        dim = params.hidden_size // params.num_heads
        theta = params.theta
        axes_dim = list(params.axes_dim)
        patch_key = "diffusion_model.pe_embedder"

    frequency_count = axes_dim[0] // 2
    if intrinsic_frequency > frequency_count:
        raise TransformError(
            f"riflex_rope.intrinsic_frequency={intrinsic_frequency} exceeds "
            f"the temporal axis's {frequency_count} frequencies")

    class EmbedNDRifleX(torch.nn.Module):
        def forward(self, ids):
            embeddings = []
            for axis in range(ids.shape[-1]):
                pos = ids[..., axis]
                if (model_management.is_device_mps(pos.device)
                        or model_management.is_intel_xpu()
                        or model_management.is_directml_enabled()):
                    device = torch.device("cpu")
                else:
                    device = pos.device
                axis_dim = axes_dim[axis]
                scale = torch.linspace(
                    0, (axis_dim - 2) / axis_dim,
                    steps=axis_dim // 2,
                    dtype=torch.float64,
                    device=device,
                )
                omega = 1.0 / (theta ** scale)
                if axis == 0:
                    omega[intrinsic_frequency - 1] = (
                        0.9 * 2 * torch.pi / num_frames)
                out = torch.einsum(
                    "...n,d->...nd",
                    pos.to(dtype=torch.float32, device=device),
                    omega,
                )
                out = torch.stack(
                    (torch.cos(out), -torch.sin(out),
                     torch.sin(out), torch.cos(out)),
                    dim=-1,
                )
                embeddings.append(out.reshape(*out.shape[:-1], 2, 2).to(
                    dtype=torch.float32, device=pos.device))
            return torch.cat(embeddings, dim=-3).unsqueeze(1)

    model = patcher.clone()
    model.add_object_patch(patch_key, EmbedNDRifleX())
    return model


def _wan_skip_layer_guidance(
    patcher, blocks: list[int], start_percent: float, end_percent: float,
):
    import torch

    def skip(args, extra_args):
        transformer_options = extra_args.get("transformer_options", {})
        original_block = extra_args["original_block"]
        if not transformer_options:
            raise ValueError(
                "transformer_options is required for Wan skip-layer guidance")
        current = transformer_options["current_percent"]
        if not start_percent <= current <= end_percent:
            return original_block(args)
        if args["img"].shape[0] == 2:
            previous_uncond = args["img"][0].unsqueeze(0)
            conditional = {
                "img": args["img"][1].unsqueeze(0),
                "txt": args["txt"][1].unsqueeze(0),
                "vec": args["vec"][1].unsqueeze(0),
                "pe": args["pe"][1].unsqueeze(0),
            }
            block_out = original_block(conditional)
            return {
                "img": torch.cat((previous_uncond, block_out["img"]), dim=0),
                "txt": args["txt"],
                "vec": args["vec"],
                "pe": args["pe"],
            }
        if transformer_options.get("cond_or_uncond") == [0]:
            return original_block(args)
        return args

    model = patcher.clone()
    for block_index in blocks:
        transformer_options = model.model_options["transformer_options"].copy()
        patches_replace = transformer_options.get("patches_replace", {}).copy()
        dit = patches_replace.get("dit", {}).copy()
        dit[("double_block", block_index)] = skip
        patches_replace["dit"] = dit
        transformer_options["patches_replace"] = patches_replace
        model.model_options["transformer_options"] = transformer_options
    return model


def _perturb_weights(
    patcher, joint_blocks: float, final_layer: float,
    rest_of_the_blocks: float, seed: int,
):
    import copy
    import torch
    from comfy import model_management
    from comfy.utils import ProgressBar

    device = model_management.get_torch_device()
    model = copy.deepcopy(patcher)
    model.model.to(device)
    state_dict = model.model.diffusion_model.state_dict()
    generator = torch.Generator(device=device).manual_seed(seed)
    progress = ProgressBar(len(state_dict))
    for key, value in state_dict.items():
        if key.startswith("joint_blocks"):
            multiplier = joint_blocks
        elif key.startswith("final_layer"):
            multiplier = final_layer
        else:
            multiplier = rest_of_the_blocks
        mean = torch.zeros_like(value) * value.mean()
        std = torch.ones_like(value) * value.std() * multiplier
        value.add_(torch.normal(mean, std, generator=generator).to(device))
        progress.update(1)
    model.model.diffusion_model.load_state_dict(state_dict)
    return model


def _hunyuan_concat_image(patcher):
    model = patcher.clone()
    model.add_object_patch("concat_keys", ("concat_image",))
    return model


def _flux_block_scales(patcher, double_blocks, single_blocks):
    """Scale complete Flux double/single blocks by bounded layer tables."""
    import re

    double = list(double_blocks) + [1.0] * (19 - len(double_blocks))
    single = list(single_blocks) + [1.0] * (38 - len(single_blocks))
    model = patcher.clone()
    for key in patcher.model_state_dict():
        match = re.search(
            r"double_blocks\.(\d+)\.(img|txt)_(mod|attn|mlp)\."
            r"(lin|qkv|proj|0|2)\.(weight|bias)",
            key,
        )
        scale = None
        if match is not None:
            index = int(match.group(1))
            if index < len(double):
                scale = double[index]
        else:
            match = re.search(
                r"single_blocks\.(\d+)\."
                r"(linear[12]|modulation\.lin)\.(weight|bias)",
                key,
            )
            if match is not None:
                index = int(match.group(1))
                if index < len(single):
                    scale = single[index]
        if scale is not None and scale != 1.0:
            model.add_patches({key: (None,)}, 0.0, scale)
    return model


def _guidance_timestepping(patcher, value, start_at, end_at):
    sigma_start = patcher.get_model_object(
        "model_sampling").percent_to_sigma(start_at)
    sigma_end = patcher.get_model_object(
        "model_sampling").percent_to_sigma(end_at)

    def guidance(args):
        cond = args["cond"]
        uncond = args["uncond"]
        cond_scale = args["cond_scale"]
        sigma = args["sigma"].detach().cpu()[0].item()
        if sigma <= sigma_start and sigma > sigma_end:
            cond_scale = value
        return uncond + (cond - uncond) * cond_scale

    model = patcher.clone()
    model.set_model_sampler_cfg_function(guidance)
    return model


def _sd3_advanced_sampling(patcher, shift, cut_off, shift_multiplier):
    import torch
    import comfy.model_sampling

    class ModelSamplingDiscreteFlowCustom(torch.nn.Module):
        def __init__(self, model_config=None):
            super().__init__()
            settings = (
                model_config.sampling_settings
                if model_config is not None else {}
            )
            self.set_parameters(
                shift=settings.get("shift", 1.0),
                multiplier=settings.get("multiplier", 1000),
            )

        def set_parameters(
            self, shift=1.0, timesteps=1000, multiplier=1000,
            cut_off=1.0, shift_multiplier=0,
        ):
            self.shift = shift
            self.multiplier = multiplier
            self.cut_off = cut_off
            self.shift_multiplier = shift_multiplier
            timesteps_tensor = self.sigma(
                (torch.arange(1, timesteps + 1, 1) / timesteps) * multiplier
            )
            self.register_buffer("sigmas", timesteps_tensor)

        @property
        def sigma_min(self):
            return self.sigmas[0]

        @property
        def sigma_max(self):
            return self.sigmas[-1]

        def timestep(self, sigma):
            return sigma * self.multiplier

        def sigma(self, timestep):
            current_shift = self.shift
            if timestep.dim() == 0:
                normalized = timestep.cpu().item() / self.multiplier
                if normalized <= self.cut_off:
                    current_shift *= self.shift_multiplier
            return comfy.model_sampling.time_snr_shift(
                current_shift, timestep / self.multiplier
            )

        def percent_to_sigma(self, percent):
            if percent <= 0.0:
                return 1.0
            if percent >= 1.0:
                return 0.0
            return 1.0 - percent

    class ModelSamplingAdvanced(
        ModelSamplingDiscreteFlowCustom, comfy.model_sampling.CONST
    ):
        pass

    model = patcher.clone()
    model_sampling = ModelSamplingAdvanced(patcher.model.model_config)
    model_sampling.set_parameters(
        shift=shift,
        multiplier=1000,
        cut_off=cut_off,
        shift_multiplier=shift_multiplier,
    )
    model.add_object_patch("model_sampling", model_sampling)
    return model


def _flux_sampler_sampling(patcher, max_shift, base_shift, width, height):
    import comfy.model_base
    import comfy.model_sampling

    if patcher.model.model_type == comfy.model_base.ModelType.FLOW:
        sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
        shift = base_shift
        multiplier = 1.0
    else:
        x1 = 256
        x2 = 4096
        slope = (max_shift - base_shift) / (x2 - x1)
        intercept = base_shift - slope * x1
        shift = (width * height / (8 * 8 * 2 * 2)) * slope + intercept
        sampling_base = comfy.model_sampling.ModelSamplingFlux
        multiplier = None

    class ModelSamplingAdvanced(sampling_base, comfy.model_sampling.CONST):
        pass

    model = patcher.clone()
    model_sampling = ModelSamplingAdvanced(patcher.model.model_config)
    if multiplier is None:
        model_sampling.set_parameters(shift=shift)
    else:
        model_sampling.set_parameters(shift=shift, multiplier=multiplier)
        original = model.get_model_object("model_sampling")
        if hasattr(original, "noise_scale"):
            model_sampling.set_noise_scale(original.noise_scale)
    model.add_object_patch("model_sampling", model_sampling)
    return model


def _latent_inpaint_ttm(patcher, steps: int, mask=None):
    import torch
    from comfy.patcher_extension import WrappersMP
    from comfy.sampler_helpers import prepare_mask

    class ApplyModelWrapper:
        def __init__(
            self, reference_samples, noise, motion_mask, scale_latent_inpaint,
        ):
            self.reference_samples = reference_samples
            self.noise = noise
            self.motion_mask = motion_mask
            self.scale_latent_inpaint = scale_latent_inpaint

        def __call__(
            self, executor, x, timestep, c_concat, c_crossattn, control,
            transformer_options, **kwargs,
        ):
            sigmas = transformer_options["sample_sigmas"]
            matched = (sigmas == timestep).nonzero(as_tuple=True)[0]
            if matched.numel() > 0:
                current_step = matched.item()
            else:
                crossing = (
                    (sigmas[:-1] - timestep)
                    * (sigmas[1:] - timestep) <= 0
                ).nonzero(as_tuple=True)[0]
                current_step = crossing.item() if crossing.numel() > 0 else 0
            next_sigma = sigmas[
                current_step + 1
                if current_step < len(sigmas) - 1 else current_step]
            if current_step != 0 and current_step < steps:
                noisy_latent = self.scale_latent_inpaint(
                    x=x,
                    sigma=torch.tensor([next_sigma]),
                    noise=self.noise.to(x),
                    latent_image=self.reference_samples.to(x),
                )
                if self.motion_mask is None:
                    x = noisy_latent
                else:
                    motion_mask = self.motion_mask.to(x)
                    x = x * (1 - motion_mask) + noisy_latent * motion_mask
            return executor(
                x, timestep, c_concat, c_crossattn, control,
                transformer_options, **kwargs)

    class OuterSampleWrapper:
        def __call__(
            self, executor, noise, latent_image, sampler, sigmas,
            denoise_mask, callback, disable_pbar, seed, latent_shapes,
        ):
            guider = executor.class_obj
            wrappers = guider.model_options["transformer_options"][
                "wrappers"]
            apply_wrappers = wrappers.setdefault(WrappersMP.APPLY_MODEL, {})
            motion_mask = None
            if mask is not None:
                motion_mask = mask.reshape(
                    (-1, 1, mask.shape[-2], mask.shape[-1]))
                motion_mask = prepare_mask(
                    motion_mask, latent_shapes[0], noise.device)
            scale_latent_inpaint = (
                guider.model_patcher.model.scale_latent_inpaint)
            apply_wrappers["TTM_ApplyModel_Wrapper"] = [ApplyModelWrapper(
                latent_image, noise, motion_mask, scale_latent_inpaint)]
            return executor(
                noise, latent_image, sampler, sigmas, denoise_mask, callback,
                disable_pbar, seed, latent_shapes=latent_shapes)

    model = patcher.clone()
    model.add_wrapper_with_key(
        WrappersMP.OUTER_SAMPLE,
        "TTM_OuterSampleWrapper",
        OuterSampleWrapper(),
    )
    return model


def _leapfusion_hunyuan_i2v(
    patcher, latent, index: int, strength: float,
    start_percent: float, end_percent: float,
):
    import torch

    samples = latent.get("samples") if isinstance(latent, dict) else None
    if not isinstance(samples, torch.Tensor) or samples.ndim != 5:
        raise TransformError(
            "leapfusion_hunyuan_i2v.latent must contain a 5D samples tensor")
    if not -samples.shape[2] <= index < samples.shape[2]:
        raise TransformError(
            f"leapfusion_hunyuan_i2v.index={index} is outside "
            f"the latent's {samples.shape[2]} frames")
    replacement = samples * 0.476986 * strength

    def unet_wrapper(apply_model, args):
        sigmas = args["c"]["transformer_options"]["sample_sigmas"]
        image = args["input"]
        timestep = args["timestep"]
        conditioning = args["c"]
        matched = (sigmas == timestep).nonzero()
        if len(matched) > 0:
            current_step = matched.item()
        else:
            current_step = 0
            for step in range(len(sigmas) - 1):
                if ((sigmas[step] - timestep[0])
                        * (sigmas[step + 1] - timestep[0])) <= 0:
                    current_step = step
                    break
        current_percent = current_step / (len(sigmas) - 1)
        if start_percent <= current_percent <= end_percent:
            image[:, :, [index], :, :] = replacement[:, :, [0], :, :].to(
                image)
        else:
            image[:, :, [index], :, :] = torch.zeros(1)
        return apply_model(image, timestep, **conditioning)

    model = patcher.clone()
    model.set_model_unet_function_wrapper(unet_wrapper)
    return model


def _spatial_tiled_evaluation(
    patcher, rows: int, columns: int, overlap: float,
    overlap_x: int, overlap_y: int, blend: str,
    preserve_existing: bool,
):
    """Evaluate one denoise prediction over a bounded overlapping grid.

    This is the common MultiDiffusion primitive only: callers still own image
    encoding, grid options, sampling, decoding, and output policy.  Spatially
    aware model patches receive a closed tile descriptor through transformer
    options so vendor integrations can crop their own control inputs.
    """
    if rows * columns > 256:
        raise TransformError(
            "spatial_tiled_evaluation supports at most 256 tiles")
    if blend != "linear":  # guarded by OneOf; defensive for direct callers
        raise TransformError("unsupported spatial tile blend policy")

    old_wrapper = patcher.model_options.get("model_function_wrapper")

    def spans(total: int, count: int, extension: int):
        if count > total:
            raise TransformError(
                "spatial tile grid exceeds the latent dimensions")
        base = max(1, total // count)
        return [
            (
                max(0, 0 if index == 0 else index * base - extension),
                min(total, total if index == count - 1
                    else (index + 1) * base + extension),
            )
            for index in range(count)
        ]

    def weight_1d(length, taper_left, taper_right, device, dtype):
        import torch

        weight = torch.ones(length, device=device, dtype=dtype)
        left = min(int(taper_left), length // 2)
        right = min(int(taper_right), length // 2)
        if left:
            weight[:left] = torch.linspace(
                0, 1, left + 2, device=device, dtype=dtype)[1:-1]
        if right:
            weight[length - right:] = torch.linspace(
                1, 0, right + 2, device=device, dtype=dtype)[1:-1]
        return weight

    cache: dict[tuple, list[dict[str, Any]]] = {}

    def wrapper(apply_model, args):
        import torch

        x_in = args.get("input")
        timestep = args.get("timestep")
        conditioning = args.get("c")
        if (not isinstance(x_in, torch.Tensor) or x_in.ndim < 4
                or not isinstance(conditioning, dict)):
            raise TransformError(
                "spatial_tiled_evaluation requires a spatial model input")
        height, width = int(x_in.shape[-2]), int(x_in.shape[-1])
        if rows > height or columns > width:
            raise TransformError(
                "spatial tile grid exceeds the latent dimensions")
        tile_height = max(1, height // rows)
        tile_width = max(1, width // columns)
        extend_y = (0 if rows == 1 else min(
            int(tile_height * overlap) + overlap_y, tile_height // 2))
        extend_x = (0 if columns == 1 else min(
            int(tile_width * overlap) + overlap_x, tile_width // 2))
        key = (
            x_in.device, x_in.dtype, height, width,
            rows, columns, extend_y, extend_x,
        )
        tiles = cache.get(key)
        if tiles is None:
            ys = spans(height, rows, extend_y)
            xs = spans(width, columns, extend_x)
            tiles = []
            for row, (top, bottom) in enumerate(ys):
                taper_top = ys[row - 1][1] - top if row else 0
                taper_bottom = (
                    bottom - ys[row + 1][0] if row < rows - 1 else 0)
                wy = weight_1d(
                    bottom - top, taper_top, taper_bottom,
                    x_in.device, x_in.dtype)
                for column, (left, right) in enumerate(xs):
                    taper_left = xs[column - 1][1] - left if column else 0
                    taper_right = (
                        right - xs[column + 1][0]
                        if column < columns - 1 else 0)
                    wx = weight_1d(
                        right - left, taper_left, taper_right,
                        x_in.device, x_in.dtype)
                    tiles.append({
                        "top": top, "bottom": bottom,
                        "left": left, "right": right,
                        "weight": wy[:, None] * wx[None, :],
                    })
            cache[key] = tiles

        accumulated = torch.zeros_like(x_in)
        leading = (1,) * (x_in.ndim - 2)
        weight_sum = torch.zeros(
            leading + (height, width),
            device=x_in.device, dtype=x_in.dtype)
        for tile in tiles:
            top, bottom = tile["top"], tile["bottom"]
            left, right = tile["left"], tile["right"]
            tile_input = x_in[..., top:bottom, left:right]
            tile_conditioning = dict(conditioning)
            transformer_options = dict(
                conditioning.get("transformer_options") or {})
            transformer_options["spatial_tile"] = {
                "top": top, "bottom": bottom,
                "left": left, "right": right,
                "source_height": height, "source_width": width,
            }
            tile_conditioning["transformer_options"] = transformer_options
            if preserve_existing and old_wrapper is not None:
                prediction = old_wrapper(apply_model, {
                    "input": tile_input,
                    "timestep": timestep,
                    "c": tile_conditioning,
                })
            else:
                prediction = apply_model(
                    tile_input, timestep, **tile_conditioning)
            if (not isinstance(prediction, torch.Tensor)
                    or prediction.shape != tile_input.shape):
                raise TransformError(
                    "spatial tile model prediction has an invalid shape")
            tile_weight = tile["weight"].view(
                leading + tile["weight"].shape)
            accumulated[..., top:bottom, left:right] += (
                prediction * tile_weight)
            weight_sum[..., top:bottom, left:right] += tile_weight
        return accumulated / weight_sum.clamp(min=1e-6)

    model = patcher.clone()
    model.set_model_unet_function_wrapper(wrapper)
    return model


TRANSFORMS: dict[str, Transform] = {
    t.name: t for t in (
        Transform(
            "attention_impl",
            "Select which attention implementation this model uses.",
            {
                "mode": OneOf(_ATTENTION_IMPLS,
                              doc="Which of core's attention implementations to use."),
                "allow_compile": Bool(default=False,
                                      doc="Let torch.compile trace into the attention "
                                          "function. Off by default, matching core."),
            },
            _attention_impl,
            experimental=True,
        ),
        Transform(
            "sage_attention_variant",
            "Select an exact KJ SageAttention kernel for this model.",
            {
                "mode": OneOf(
                    _SAGE_VARIANTS,
                    doc="Exact SageAttention kernel variant, or disabled."),
                "allow_compile": Bool(
                    default=False,
                    doc="Let torch.compile trace into the selected kernel."),
            },
            _sage_attention_variant,
            experimental=True,
        ),
        Transform(
            "strict_flash_attention",
            "Use FlashAttention 2 or 3 without an SDPA fallback.",
            {
                "allow_compile": Bool(
                    default=False,
                    doc="Let torch.compile trace into the FlashAttention call."),
            },
            _strict_flash_attention,
            experimental=True,
        ),
        Transform(
            "kohya_deep_shrink",
            "Apply core's Kohya Deep Shrink UNet patch.",
            {
                "block_number": Int(1, 32, doc="UNet input block index."),
                "downscale_factor": Float(
                    0.1, 9.0, doc="Temporary feature downscale factor."),
                "start_percent": Float(
                    0.0, 1.0, doc="First sampling percentage."),
                "end_percent": Float(
                    0.0, 1.0, doc="Last sampling percentage."),
                "downscale_after_skip": Bool(
                    doc="Patch after the skip connection."),
                "downscale_method": OneOf(
                    ("bicubic", "nearest-exact", "bilinear", "area", "bislerp"),
                    doc="Core feature downscale method."),
                "upscale_method": OneOf(
                    ("bicubic", "nearest-exact", "bilinear", "area", "bislerp"),
                    doc="Core feature restore method."),
            },
            _kohya_deep_shrink,
        ),
        Transform(
            "nabla_sparse_attention",
            "Apply KJ's NABLA spatiotemporal sparse-attention policy.",
            {
                "latent": RefOf(
                    "LATENT",
                    doc="Latent whose video dimensions define the sparse mask."),
                "window_time": Int(
                    1, 100000, doc="Temporal local-attention window."),
                "window_width": Int(
                    1, 100000, doc="Horizontal local-attention window."),
                "window_height": Int(
                    1, 100000, doc="Vertical local-attention window."),
                "sparsity": Float(
                    0.0, 1.0, doc="Attention mass retained per sparse block."),
                "compile_attention": Bool(
                    default=True,
                    doc="Compile the host-owned NABLA attention override."),
            },
            _nabla_sparse_attention,
            experimental=True,
        ),
        Transform(
            "enhance_a_video",
            "Apply KJ's temporal Enhance-A-Video attention policy.",
            {
                "latent": RefOf(
                    "LATENT", doc="Latent defining the video frame count."),
                "architecture": OneOf(
                    ("wan", "ltx"), doc="Core-owned attention layout."),
                "weight": Float(
                    0.0, 100.0, doc="Temporal-attention enhancement strength."),
            },
            _enhance_a_video,
            experimental=True,
        ),
        Transform(
            "wan_video_nag",
            "Apply KJ's normalized-attention guidance to Wan cross-attention.",
            {
                "conditioning": RefOf(
                    "CONDITIONING",
                    doc="Negative-guidance conditioning embedding."),
                "nag_scale": Float(
                    0.0, 100.0, doc="Negative-guidance strength."),
                "nag_alpha": Float(
                    0.0, 1.0, doc="Mix with the positive attention output."),
                "nag_tau": Float(
                    0.0, 10.0, doc="L1 guidance clipping threshold."),
                "input_type": OneOf(
                    ("default", "batch"), default="default",
                    doc="Wan conditioning batch interpretation."),
                "inplace": Bool(
                    default=False,
                    doc="Use KJ's lower-memory in-place arithmetic."),
            },
            _wan_video_nag,
            experimental=True,
        ),
        Transform(
            "krea2_token_weights",
            "Apply Krea2 per-token attention value and key weighting.",
            {
                "weights": TokenWeights(
                    4096,
                    doc="Conditioning positions with value factor and key bias."),
            },
            _krea2_token_weights,
            experimental=True,
        ),
        Transform(
            "ltx2_audio_normalization",
            "Normalize LTX2 audio latents between configured sample steps.",
            {
                "factors": FloatList(
                    -1000.0, 1000.0, 10000,
                    doc="Per-step audio-latent multipliers."),
            },
            _ltx2_audio_normalization,
            experimental=True,
        ),
        Transform(
            "ltx2_nag",
            "Apply normalized-attention guidance to LTX2 video or audio.",
            {
                "nag_scale": Float(
                    0.0, 100.0, doc="Negative-guidance strength."),
                "nag_alpha": Float(
                    0.0, 1.0, doc="Mix with positive attention."),
                "nag_tau": Float(
                    0.0, 10.0, doc="L1 guidance clipping threshold."),
                "video_conditioning": RefOf(
                    "CONDITIONING", default=None,
                    doc="Optional LTX2 video negative conditioning."),
                "audio_conditioning": RefOf(
                    "CONDITIONING", default=None,
                    doc="Optional LTX2 audio negative conditioning."),
                "inplace": Bool(
                    default=True,
                    doc="Use KJ's lower-memory in-place arithmetic."),
            },
            _ltx2_nag,
            experimental=True,
        ),
        Transform(
            "ideogram4_optimizations",
            "Bound Ideogram4 feed-forward and RoPE activation memory.",
            {
                "chunk_ffn": Bool(
                    default=True,
                    doc="Chunk long feed-forward token sequences."),
                "ffn_chunks": Int(
                    1, 64, default=2,
                    doc="Number of token-sequence chunks."),
                "ffn_seq_threshold": Int(
                    256, 65536, default=1024,
                    doc="Minimum sequence length to chunk."),
                "bf16_rope": Bool(
                    default=True,
                    doc="Keep rotary activations in the input dtype."),
            },
            _ideogram4_optimizations,
            experimental=True,
        ),
        Transform(
            "ltx2_attention_tuner",
            "Scale LTX2 video and audio attention paths per block.",
            {
                "blocks": IntList(
                    4096,
                    doc="Selected transformer blocks; empty selects all."),
                "video_scale": Float(
                    0.0, 100.0, doc="Video self and text-attention scale."),
                "audio_scale": Float(
                    0.0, 100.0, doc="Audio self and text-attention scale."),
                "audio_to_video_scale": Float(
                    0.0, 100.0, doc="Audio-to-video attention scale."),
                "video_to_audio_scale": Float(
                    0.0, 100.0, doc="Video-to-audio attention scale."),
                "triton_kernels": Bool(
                    default=True,
                    doc="Use fused kernels when the host supports them."),
            },
            _ltx2_attention_tuner,
            experimental=True,
        ),
        Transform(
            "memory_efficient_sage",
            "Use architecture-specific SageAttention with bounded buffers.",
            {
                "architecture": OneOf(
                    ("ltx2", "minimax", "wan"),
                    doc="Core-owned model architecture policy."),
                "triton_kernels": Bool(
                    default=True,
                    doc="Use fused rotary kernels when supported."),
            },
            _memory_efficient_sage,
            experimental=True,
        ),
        Transform(
            "minimax_chunk_feed_forward",
            "Chunk MiniMax H3 packed-token feed-forward activations.",
            {
                "chunks": Int(
                    1, 64, doc="Number of packed-token chunks."),
                "seq_threshold": Int(
                    256, 262144,
                    doc="Only chunk longer packed-token sequences."),
            },
            _minimax_chunk_feed_forward,
            experimental=True,
        ),
        Transform(
            "minimax_low_vram_attention",
            "Release MiniMax H3 attention buffers early and group heads.",
            {
                "head_chunks": Int(
                    1, 56, doc="Number of independent head groups."),
            },
            _minimax_low_vram_attention,
            experimental=True,
        ),
        Transform(
            "matmul_fp16_accumulation",
            "Set torch's fp16 matmul accumulation for the duration of the run.",
            {"enabled": Bool(doc="Whether to allow fp16 accumulation.")},
            _matmul_fp16_accumulation,
            experimental=True,
        ),
        Transform(
            "memory_usage_factor",
            "Override the model's memory-usage estimate during sampling.",
            {"factor": Float(0.0, 100.0,
                             doc="Multiplier replacing the model's own estimate.")},
            _memory_usage_factor,
            experimental=True,
        ),
        Transform(
            "ffn_chunking",
            "Chunk feed-forward activations to cut peak VRAM.",
            {
                "chunks": Int(1, 100, doc="Number of chunks. 1 is a no-op."),
                "dim_threshold": Int(0, 16384, default=4096,
                                     doc="Only chunk sequences longer than this."),
                "target": OneOf(
                    _FFN_TARGETS, default="blocks_ffn",
                    doc="Core-owned feed-forward model layout."),
            },
            _ffn_chunking,
            experimental=True,
        ),
        Transform(
            "compile",
            "torch.compile the diffusion model.",
            {
                "backend": OneOf(("inductor", "cudagraphs", "eager", "aot_eager"),
                                 default="inductor", doc="torch.compile backend."),
                "mode": OneOf(("default", "reduce-overhead", "max-autotune",
                               "max-autotune-no-cudagraphs"),
                              default="default", doc="torch.compile mode."),
                "fullgraph": Bool(default=False,
                                  doc="Require a single graph with no breaks."),
                "dynamic": NullableBool(
                    default=False,
                    doc="Allow dynamic shapes, or null for torch's automatic mode."),
                "scope": OneOf(
                    _COMPILE_SCOPES,
                    default="whole",
                    doc="Core-owned model-module selection policy."),
                "double_blocks": Bool(
                    default=True,
                    doc="Include Flux double blocks when scope is flux_blocks."),
                "single_blocks": Bool(
                    default=True,
                    doc="Include Flux single blocks when scope is flux_blocks."),
                "dynamo_cache_size_limit": Int(
                    0, 1024, default=None,
                    doc="Dynamo cache limit applied while compiling, or null to preserve it."),
                "force_parameter_static_shapes": NullableBool(
                    default=None,
                    doc="Dynamo parameter-shape policy applied while compiling."),
                "dynamic_vram": OneOf(
                    ("disable", "preserve", "stabilize"),
                    default="disable",
                    doc="Clone policy for ComfyUI dynamic VRAM models."),
                "guard_filter": Bool(
                    default=False,
                    doc="Ignore transformer_options guards in default compile mode."),
                "debug_compile_keys": Bool(
                    default=False,
                    doc="Log the core-selected module keys before compiling."),
                "default_mode": OneOf(
                    ("omit", "explicit"),
                    default="omit",
                    doc="Whether to pass torch.compile's default mode explicitly."),
            },
            _compile,
            experimental=True,
        ),
        Transform(
            "context_windows",
            "Apply a closed context-window schedule and fuse policy.",
            {
                "context_schedule": OneOf(
                    _CONTEXT_SCHEDULES,
                    doc="Core-owned context-window placement policy."),
                "fuse_method": OneOf(
                    _CONTEXT_FUSE_METHODS,
                    doc="Core-owned overlap blending policy."),
                "context_length": Int(
                    1, 100000, doc="Latent frames in each window."),
                "context_overlap": Int(
                    0, 100000, doc="Latent-frame overlap between windows."),
                "context_stride": Int(
                    1, 32, doc="Maximum stride power for uniform schedules."),
                "closed_loop": Bool(
                    doc="Allow looped schedules to wrap to frame zero."),
                "dim": Int(
                    0, 5, doc="Temporal dimension in the model latent."),
                "freenoise": Bool(
                    doc="Install core's FreeNoise sampler wrapper."),
                "causal_window_fix": Bool(
                    doc="Prepend and then strip the prior causal frame."),
                "cond_retain_indices": IntList(
                    4096, default=(),
                    doc="Window-relative conditioning indices to retain."),
            },
            _context_windows,
            experimental=True,
        ),
        Transform(
            "cfg_zero_star",
            "Apply CFG-Zero* guidance and optional initial-step zeroing.",
            {
                "use_zero_init": Bool(
                    default=True, doc="Zero the initial guided predictions."),
                "zero_init_steps": Int(
                    0, 100000, default=0,
                    doc="Last zero-based step whose prediction is zeroed."),
            },
            _cfg_zero_star,
            experimental=True,
        ),
        Transform(
            "pid_color_bias",
            "Apply the calibrated PiD Flux2 first-step color correction.",
            {
                "strength": Float(
                    -20.0, 20.0, default=1.0,
                    doc="Multiplier applied to the calibrated RGB bias."),
                "backbone": OneOf(
                    ("flux2",), default="flux2",
                    doc="Calibrated diffusion backbone."),
            },
            _pid_color_bias,
            experimental=True,
        ),
        Transform(
            "dynamic_thresholding",
            "Apply canonical Dynamic Thresholding to classifier-free guidance.",
            {
                "mimic_scale": Float(
                    0.0, 100.0, default=7.0,
                    doc="Target CFG scale whose variability is mimicked."),
                "threshold_percentile": Float(
                    0.0, 1.0, default=1.0,
                    doc="Absolute-deviation quantile used for clipping."),
                "mimic_mode": OneOf(
                    _DYNAMIC_THRESHOLD_MODES, default="Constant",
                    doc="Schedule applied to mimic_scale."),
                "mimic_scale_min": Float(
                    0.0, 100.0, default=0.0,
                    doc="Minimum scheduled mimic scale."),
                "cfg_mode": OneOf(
                    _DYNAMIC_THRESHOLD_MODES, default="Constant",
                    doc="Schedule applied to the sampler CFG scale."),
                "cfg_scale_min": Float(
                    0.0, 100.0, default=0.0,
                    doc="Minimum scheduled CFG scale."),
                "schedule_value": Float(
                    0.0, 100.0, default=1.0,
                    doc="Power or repetition value for scheduled modes."),
                "separate_feature_channels": Bool(
                    default=True,
                    doc="Measure variability independently per channel."),
                "scaling_startpoint": OneOf(
                    ("MEAN", "ZERO"), default="MEAN",
                    doc="Center scaling on each channel mean or zero."),
                "variability_measure": OneOf(
                    ("AD", "STD"), default="AD",
                    doc="Use absolute deviation or standard deviation."),
                "interpolate_phi": Float(
                    0.0, 1.0, default=1.0,
                    doc="Blend thresholded guidance with ordinary CFG."),
            },
            _dynamic_thresholding,
        ),
        Transform(
            "style_aligned_batch",
            "Share reference style statistics across a classic UNet batch.",
            {
                "share_norm": OneOf(
                    ("both", "group", "layer", "disabled"),
                    default="both",
                    doc="Normalization families that share reference stats."),
                "share_attention": OneOf(
                    ("q+k", "q+k+v", "disabled"),
                    default="q+k",
                    doc="Attention tensors receiving reference AdaIN."),
                "scale": Float(
                    0.0, 1.0, default=1.0,
                    doc="Reference-key scale for non-reference images."),
            },
            _style_aligned_batch,
        ),
        Transform(
            "controlnet_lllite",
            "Apply a canonical legacy SD ControlNet-LLLite adapter.",
            {
                "adapter": SafeTensorName(
                    doc="Logical adapter name in the controlnet catalogue."),
                "image": RefOf(
                    "IMAGE", doc="RGB control image batch."),
                "strength": Float(
                    0.0, 10.0, default=1.0,
                    doc="LLLite residual multiplier."),
                "steps": Int(
                    0, 200, default=0,
                    doc="Sampler step count, or zero for no step window."),
                "start_percent": Float(
                    0.0, 100.0, default=0.0,
                    doc="First active percentage of the sampling run."),
                "end_percent": Float(
                    0.0, 100.0, default=0.0,
                    doc="Last active percentage; zero means the run end."),
            },
            _controlnet_lllite,
        ),
        Transform(
            "differential_diffusion",
            "Apply core's canonical differential-diffusion mask behavior.",
            {"strength": Float(
                0.0, 1.0, default=1.0,
                doc="Blend strength for the binary differential mask.")},
            _differential_diffusion,
            experimental=True,
        ),
        Transform(
            "fooocus_inpaint",
            "Apply Fooocus' canonical SDXL inpaint head and quantized delta.",
            {
                "latent": RefOf(
                    "LATENT", doc="Inpaint latent with a noise mask."),
                "head": WeightArchiveName(
                    doc="Logical Fooocus head name in the inpaint catalogue."),
                "patch": WeightArchiveName(
                    doc="Logical Fooocus patch name in the inpaint catalogue."),
            },
            _fooocus_inpaint,
        ),
        Transform(
            "diffusion_weight_delta",
            "Apply a shape-checked SafeTensors diffusion-model delta.",
            {
                "model_patch": SafeTensorName(
                    doc="Logical name in the model_patches catalogue."),
                "strength": Float(
                    -10.0, 10.0, default=1.0,
                    doc="Delta multiplier."),
                "pad_input_channels": Bool(
                    default=False,
                    doc="Allow only the first convolution input to widen."),
            },
            _diffusion_weight_delta,
            experimental=True,
        ),
        Transform(
            "serialized_model_patch",
            "Apply a tensor-only serialized Comfy diffusion-model patch.",
            {
                "model_patch": SafeTensorName(
                    doc="Logical name in the model_patches catalogue."),
                "strength": Float(
                    -10.0, 10.0, default=1.0,
                    doc="Patch multiplier."),
                "pad_diff_weights": Bool(
                    default=False,
                    doc="Allow serialized diff patches to widen inputs."),
            },
            _serialized_model_patch,
            experimental=True,
        ),
        Transform(
            "layer_diffusion_attention_sharing",
            "Apply canonical SD1 Layer Diffusion attention sharing.",
            {
                "model_patch": SafeTensorName(
                    doc="Logical SafeTensor name in model_patches."),
                "frames": Int(
                    1, 3, default=1,
                    doc="Interleaved Layer Diffusion frame count."),
                "control_image": RefOf(
                    "IMAGE", default=None,
                    doc="Optional bounded RGB control image."),
                "first_conditioning": RefOf(
                    "CONDITIONING", default=None,
                    doc="Optional first per-frame conditioning."),
                "second_conditioning": RefOf(
                    "CONDITIONING", default=None,
                    doc="Optional second per-frame conditioning."),
                "third_conditioning": RefOf(
                    "CONDITIONING", default=None,
                    doc="Optional third per-frame conditioning."),
            },
            _layer_diffusion_attention_sharing,
            experimental=True,
        ),
        Transform(
            "concat_latent_input",
            "Inject one encoded latent as model c_concat input.",
            {
                "latent": RefOf(
                    "LATENT", doc="Encoded latent to inject."),
            },
            _concat_latent_input,
            experimental=True,
        ),
        Transform(
            "sampling_memory_report",
            "Log peak allocated and reserved accelerator memory after sampling.",
            {},
            _sampling_memory_report,
            experimental=True,
        ),
        Transform(
            "riflex_rope",
            "Install a RIFLEx temporal rotary-position embedder.",
            {
                "architecture": OneOf(
                    ("wan", "hunyuan"),
                    doc="Core-owned rotary embedder layout."),
                "num_frames": Int(
                    1, 100000, doc="Target latent frame count."),
                "intrinsic_frequency": Int(
                    1, 100, doc="One-based temporal frequency index."),
            },
            _riflex_rope,
            experimental=True,
        ),
        Transform(
            "wan_skip_layer_guidance",
            "Skip selected Wan unconditional double blocks in a step range.",
            {
                "blocks": IntList(
                    4096, doc="Wan double-block indices to skip."),
                "start_percent": Float(
                    0.0, 1.0, doc="First sampling fraction to apply."),
                "end_percent": Float(
                    0.0, 1.0, doc="Last sampling fraction to apply."),
            },
            _wan_skip_layer_guidance,
            experimental=True,
        ),
        Transform(
            "perturb_weights",
            "Add deterministic Gaussian perturbations to diffusion weights.",
            {
                "joint_blocks": Float(
                    0.001, 10.0,
                    doc="Noise standard-deviation multiplier for joint blocks."),
                "final_layer": Float(
                    0.001, 10.0,
                    doc="Noise standard-deviation multiplier for final layers."),
                "rest_of_the_blocks": Float(
                    0.001, 10.0,
                    doc="Noise multiplier for all other diffusion weights."),
                "seed": Int(
                    0, 0xffffffffffffffff,
                    doc="Execution-local random seed."),
            },
            _perturb_weights,
            experimental=True,
        ),
        Transform(
            "hunyuan_concat_image",
            "Configure Hunyuan Video conditioning to use concat_image.",
            {},
            _hunyuan_concat_image,
            experimental=True,
        ),
        Transform(
            "flux_block_scales",
            "Scale complete Flux double and single transformer blocks.",
            {
                "double_blocks": FloatList(
                    0.0, 5.0, 19,
                    doc="Scale table for Flux double blocks 0 through 18."),
                "single_blocks": FloatList(
                    0.0, 5.0, 38,
                    doc="Scale table for Flux single blocks 0 through 37."),
            },
            _flux_block_scales,
            experimental=True,
        ),
        Transform(
            "guidance_timestepping",
            "Override CFG within a model-sigma percentage range.",
            {
                "value": Float(0.0, 100.0, doc="CFG value in the range."),
                "start_at": Float(
                    0.0, 1.0, doc="First sampling percentage."),
                "end_at": Float(
                    0.0, 1.0, doc="Last sampling percentage."),
            },
            _guidance_timestepping,
            experimental=True,
        ),
        Transform(
            "sd3_advanced_sampling",
            "Install Essentials' cut-off SD3 flow schedule.",
            {
                "shift": Float(0.0, 100.0, doc="Base flow shift."),
                "cut_off": Float(
                    0.0, 1.0, doc="Normalized timestep cut-off."),
                "shift_multiplier": Float(
                    0.0, 10.0, doc="Shift multiplier below the cut-off."),
            },
            _sd3_advanced_sampling,
            experimental=True,
        ),
        Transform(
            "flux_sampler_sampling",
            "Install the Flux or AuraFlow resolution-aware schedule.",
            {
                "max_shift": Float(0.0, 100.0, doc="Maximum Flux shift."),
                "base_shift": Float(0.0, 100.0, doc="Base Flux shift."),
                "width": Int(1, 16384, doc="Target image width."),
                "height": Int(1, 16384, doc="Target image height."),
            },
            _flux_sampler_sampling,
            experimental=True,
        ),
        Transform(
            "latent_inpaint_ttm",
            "Apply Time-To-Move latent inpainting during early sample steps.",
            {
                "steps": Int(
                    0, 888, doc="Number of early sample steps to modify."),
                "mask": RefOf(
                    "MASK", default=None,
                    doc="Optional motion-region mask."),
            },
            _latent_inpaint_ttm,
            experimental=True,
        ),
        Transform(
            "leapfusion_hunyuan_i2v",
            "Replace one Hunyuan latent frame during a sampling step range.",
            {
                "latent": RefOf(
                    "LATENT", doc="Source latent supplying the replacement frame."),
                "index": Int(
                    -1, 1000, doc="Target latent frame index."),
                "strength": Float(
                    -10.0, 10.0, doc="Replacement latent multiplier."),
                "start_percent": Float(
                    0.0, 1.0, doc="First sampling fraction to replace."),
                "end_percent": Float(
                    0.0, 1.0, doc="Last sampling fraction to replace."),
            },
            _leapfusion_hunyuan_i2v,
            experimental=True,
        ),
        Transform(
            "spatial_tiled_evaluation",
            "Evaluate one denoise prediction over bounded overlapping tiles.",
            {
                "rows": Int(1, 256, doc="Latent tile rows."),
                "columns": Int(1, 256, doc="Latent tile columns."),
                "overlap": Float(
                    0.0, 0.5, default=0.0,
                    doc="Fractional overlap added to each tile."),
                "overlap_x": Int(
                    0, 8192, default=0,
                    doc="Additional horizontal overlap in latent cells."),
                "overlap_y": Int(
                    0, 8192, default=0,
                    doc="Additional vertical overlap in latent cells."),
                "blend": OneOf(
                    ("linear",), default="linear",
                    doc="Canonical overlap blend policy."),
                "preserve_existing": Bool(
                    default=True,
                    doc="Delegate through an existing model wrapper."),
            },
            _spatial_tiled_evaluation,
        ),
    )
}


def describe_all() -> list[dict]:
    """The vocabulary, as data — for documentation and for the guest's error
    messages, so both are generated from the table rather than restated."""
    return [t.describe() for t in TRANSFORMS.values()]
