from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F

ATTENTION_MEASURE_KEY = "attention_measure_v1"
ATTENTION_MEASURE_CAPABILITIES_KEY = "attention_measure_capabilities_v1"
ATTENTION_MEASURE_API = 1
KEY_LOG_MEASURE_OPERATOR = "key_log_measure"
H3_SOURCE_CARRIER_NORMALIZATION = "h3_native_source_carrier_v1"
MIXED_GRID_TOPOLOGY = "mixed_grid_low_suffix"
H3_COORDINATE_POLICY = "minimax_h3_native_frame_grid_v1"
SUPPORTED_PROFILES = frozenset({"dense_exact_v1", "weighted_exact_blocks_v1"})


@dataclass(frozen=True, slots=True)
class MeasureExecutionContext:
    provider_identity: str
    block_index: int
    owner: object
    owner_generation: str
    layout: object
    q_rows: int
    kv_rows: int
    dtype: torch.dtype
    device: torch.device
    head_dim: int
    mask_class: str
    preprocess_digest: str
    numerical_route: str
    existing_sink: tuple[int, int] = (0, 0)
    external_sequence: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class BoundMeasurePlan:
    semantic_digest: str
    implementation_profile: str
    provider_identity: str
    owner_generation: str
    numerical_route: str
    preprocess_digest: str
    q_rows: int
    kv_rows: int
    exact_k_block_range: tuple[int, int]
    exact_range_digest: str
    key_log_measure: torch.Tensor
    completion_hook: Callable[[Mapping[str, Any]], None] | None = None


def _integer(name: str, value: Any, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"attention measure {name} must be an integer")
    if value < minimum:
        raise ValueError(f"attention measure {name} must be >= {minimum}")
    return value


def _grid(name: str, value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"attention measure {name} must be a two-element grid")
    return _integer(f"{name}[0]", value[0], 1), _integer(f"{name}[1]", value[1], 1)


def _ratio(name: str, num: Any, den: Any) -> Fraction:
    num = _integer(f"{name}.mass_num", num, 1)
    den = _integer(f"{name}.mass_den", den, 1)
    ratio = Fraction(num, den)
    if not math.isfinite(math.log(ratio.numerator) - math.log(ratio.denominator)):
        raise ValueError(f"attention measure {name} derives a non-finite key-log-measure")
    return ratio


def _canonical_segments(raw_segments, *, kv_rows: int):
    if not isinstance(raw_segments, (list, tuple)) or not raw_segments:
        raise TypeError("attention measure segments must be a nonempty list")
    segments: list[dict[str, int]] = []
    cursor = 0
    for i, raw in enumerate(raw_segments):
        if not isinstance(raw, Mapping):
            raise TypeError(f"attention measure segments[{i}] must be a mapping")
        start = _integer(f"segments[{i}].start", raw.get("start"))
        stop = _integer(f"segments[{i}].stop", raw.get("stop"))
        if start != cursor or stop <= start or stop > kv_rows:
            raise ValueError("attention measure segments must be sorted, contiguous, nonempty and in range")
        ratio = _ratio(f"segments[{i}]", raw.get("mass_num"), raw.get("mass_den"))
        if segments and (segments[-1]["mass_num"], segments[-1]["mass_den"]) == (ratio.numerator, ratio.denominator):
            segments[-1]["stop"] = stop
        else:
            segments.append({
                "start": start,
                "stop": stop,
                "mass_num": ratio.numerator,
                "mass_den": ratio.denominator,
            })
        cursor = stop
    if cursor != kv_rows:
        raise ValueError("attention measure segments must cover every key row exactly once")
    return segments


def normalize(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise TypeError("attention_measure_v1 must be a mapping")
    literals = {
        "api": ATTENTION_MEASURE_API,
        "operator": KEY_LOG_MEASURE_OPERATOR,
        "normalization": H3_SOURCE_CARRIER_NORMALIZATION,
        "topology": MIXED_GRID_TOPOLOGY,
        "coordinate_policy": H3_COORDINATE_POLICY,
    }
    for key, expected in literals.items():
        if request.get(key) != expected:
            raise ValueError(f"attention measure {key} must be {expected!r}")

    q_rows = _integer("q_rows", request.get("q_rows"), 1)
    kv_rows = _integer("kv_rows", request.get("kv_rows"), 1)
    video_start = _integer("video_start", request.get("video_start"))
    temporal = _integer("temporal", request.get("temporal"), 2)
    prefix_t = _integer("prefix_t", request.get("prefix_t"), 1)
    source_grid = _grid("source_grid", request.get("source_grid"))
    prefix_grid = _grid("prefix_grid", request.get("prefix_grid"))
    if q_rows != kv_rows:
        raise ValueError("attention measure requires the all-row self-attention domain")
    if video_start >= kv_rows or prefix_t >= temporal:
        raise ValueError("attention measure video geometry is invalid")
    source_rows = math.prod(source_grid)
    prefix_rows = math.prod(prefix_grid)
    if source_rows > prefix_rows:
        raise ValueError("mixed-grid source measure cannot exceed the protected-prefix grid")
    if q_rows != video_start + prefix_t * prefix_rows + (temporal - prefix_t) * source_rows:
        raise ValueError("attention measure row count disagrees with mixed-grid geometry")

    segments = _canonical_segments(request.get("segments"), kv_rows=kv_rows)
    weighted_start = video_start
    weighted_stop = video_start + prefix_t * prefix_rows
    wanted = Fraction(source_rows, prefix_rows)
    for segment in segments:
        start, stop = segment["start"], segment["stop"]
        overlap = max(start, weighted_start) < min(stop, weighted_stop)
        ratio = Fraction(segment["mass_num"], segment["mass_den"])
        if overlap and ratio != wanted:
            raise ValueError("protected-prefix keys use the wrong spatial measure")
        if not overlap and ratio != 1:
            raise ValueError("non-prefix keys must retain unit measure")
        if wanted != 1 and (start < weighted_start < stop or start < weighted_stop < stop):
            raise ValueError("attention measure boundaries must align with the protected-prefix interval")

    return {
        **literals,
        "q_rows": q_rows,
        "kv_rows": kv_rows,
        "video_start": video_start,
        "temporal": temporal,
        "prefix_t": prefix_t,
        "source_grid": list(source_grid),
        "prefix_grid": list(prefix_grid),
        "segments": segments,
    }


def semantic_digest(request: Mapping[str, Any]) -> str:
    encoded = json.dumps(normalize(request), sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def is_unit_measure(request: Mapping[str, Any]) -> bool:
    return all((s["mass_num"], s["mass_den"]) == (1, 1) for s in normalize(request)["segments"])


def validate_h3(request, *, layout, q_rows, kv_rows, external_sequence=None):
    normalized = normalize(request)
    if type(q_rows) is not int or type(kv_rows) is not int:
        raise TypeError("attention measure runtime row counts must be integers")
    if normalized["q_rows"] != q_rows or normalized["kv_rows"] != kv_rows:
        raise ValueError("attention measure row counts are stale")
    if layout is None or getattr(layout, "seq_len", None) != q_rows:
        raise ValueError("attention measure requires the actual H3 mixed layout")
    video = next(((a, b) for a, b, kind in getattr(layout, "segments", ()) if kind == "video"), None)
    if video != (normalized["video_start"], q_rows):
        raise ValueError("attention measure video interval disagrees with the H3 layout")
    if external_sequence is not None:
        if not isinstance(external_sequence, Mapping):
            raise TypeError("attention measure VDN external-sequence contract must be a mapping")
        integer_fields = (
            "api",
            "native_sequence_rows",
            "sequence_rows",
            "video_start",
            "temporal",
            "prefix_t",
            "source_rows_per_frame",
            "prefix_rows_per_frame",
        )
        if any(type(external_sequence.get(key)) is not int for key in integer_fields):
            raise TypeError("attention measure VDN external-sequence row counts must be integers")
        source_rows = math.prod(normalized["source_grid"])
        prefix_rows = math.prod(normalized["prefix_grid"])
        native_rows = normalized["video_start"] + normalized["temporal"] * source_rows
        checks = {
            "api": 2,
            "mode": "dense_gate_no_linear",
            "topology": MIXED_GRID_TOPOLOGY,
            "native_sequence_rows": native_rows,
            "sequence_rows": q_rows,
            "video_start": normalized["video_start"],
            "temporal": normalized["temporal"],
            "prefix_t": normalized["prefix_t"],
            "source_rows_per_frame": source_rows,
            "prefix_rows_per_frame": prefix_rows,
        }
        if any(external_sequence.get(k) != v for k, v in checks.items()):
            raise ValueError("attention measure disagrees with the VDN external-sequence contract")
    return normalized


def materialize(request, *, device, dtype=torch.float32):
    normalized = normalize(request)
    if not dtype.is_floating_point:
        raise TypeError("attention measure bias requires floating point")
    out = torch.empty(normalized["kv_rows"], device=device, dtype=dtype)
    for segment in normalized["segments"]:
        value = math.log(segment["mass_num"]) - math.log(segment["mass_den"])
        out[segment["start"]:segment["stop"]] = value
    return out.contiguous()


def merge_exact_k_blocks(request, block_size: int, existing=(0, 0)) -> tuple[int, int]:
    normalized = normalize(request)
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if (
        not isinstance(existing, (tuple, list))
        or len(existing) != 2
        or any(type(x) is not int for x in existing)
        or existing[0] < 0
        or existing[1] < existing[0]
    ):
        raise ValueError("existing sink range is invalid")
    max_blocks = (normalized["kv_rows"] + block_size - 1) // block_size
    if existing[1] > max_blocks:
        raise ValueError("existing sink range exceeds the K/V domain")
    weighted = [s for s in normalized["segments"] if (s["mass_num"], s["mass_den"]) != (1, 1)]
    if not weighted:
        return int(existing[0]), int(existing[1])
    first = min(s["start"] for s in weighted) // block_size
    last = (max(s["stop"] for s in weighted) + block_size - 1) // block_size
    if existing[0] == existing[1]:
        return first, last
    return min(existing[0], first), max(existing[1], last)


def bind(
    request,
    *,
    context: MeasureExecutionContext,
    block_size: int,
    implementation_profile: str,
    completion_hook: Callable[[Mapping[str, Any]], None] | None = None,
):
    normalized = validate_h3(
        request,
        layout=context.layout,
        q_rows=context.q_rows,
        kv_rows=context.kv_rows,
        external_sequence=context.external_sequence,
    )
    if implementation_profile not in SUPPORTED_PROFILES:
        raise ValueError("unsupported attention-measure implementation profile")
    if not context.provider_identity or not context.owner_generation or not context.preprocess_digest:
        raise ValueError("attention measure execution ownership identity is incomplete")
    blocks = merge_exact_k_blocks(normalized, block_size, context.existing_sink)
    range_digest = hashlib.sha256(f"{blocks[0]}:{blocks[1]}".encode("ascii")).hexdigest()
    return BoundMeasurePlan(
        semantic_digest(normalized),
        implementation_profile,
        context.provider_identity,
        context.owner_generation,
        context.numerical_route,
        context.preprocess_digest,
        context.q_rows,
        context.kv_rows,
        blocks,
        range_digest,
        materialize(normalized, device=context.device),
        completion_hook,
    )


def register_capability(transformer_options, provider_identity, capability):
    if not isinstance(provider_identity, str) or not provider_identity:
        raise ValueError("attention-measure provider identity must be a nonempty string")
    if not callable(getattr(capability, "prepare", None)):
        raise TypeError("attention-measure capability must provide prepare(request, execution_context)")
    current = transformer_options.get(ATTENTION_MEASURE_CAPABILITIES_KEY)
    registry = dict(current) if isinstance(current, Mapping) else {}
    existing = registry.get(provider_identity)
    if existing is not None and existing is not capability:
        raise RuntimeError(f"attention-measure provider {provider_identity!r} already has another owner")
    registry[provider_identity] = capability
    transformer_options[ATTENTION_MEASURE_CAPABILITIES_KEY] = registry


def prepare_capability(transformer_options, request, context: MeasureExecutionContext) -> BoundMeasurePlan:
    registry = transformer_options.get(ATTENTION_MEASURE_CAPABILITIES_KEY)
    if not isinstance(registry, Mapping):
        raise RuntimeError("attention measure requested but no provider capabilities are registered")
    capability = registry.get(context.provider_identity)
    if capability is None:
        raise RuntimeError(f"attention measure provider {context.provider_identity!r} is not capable")
    plan = capability.prepare(request, context)
    if not isinstance(plan, BoundMeasurePlan):
        raise TypeError("attention-measure capability returned an invalid bound plan")
    if (
        plan.provider_identity != context.provider_identity
        or plan.owner_generation != context.owner_generation
        or plan.preprocess_digest != context.preprocess_digest
        or plan.q_rows != context.q_rows
        or plan.kv_rows != context.kv_rows
        or plan.semantic_digest != semantic_digest(request)
    ):
        raise RuntimeError("attention-measure capability returned a stale or wrong-owner bound plan")
    return plan


def _mask(mask, *, batch, q_rows, kv_rows, dtype, device):
    if not torch.is_tensor(mask) or mask.device != device:
        raise ValueError("attention measure mask must be a tensor on the attention device")
    if mask.ndim == 2 and tuple(mask.shape) == (q_rows, kv_rows):
        mask = mask[None, None]
    elif mask.ndim == 2 and tuple(mask.shape) == (batch, kv_rows):
        mask = mask[:, None, None, :]
    elif mask.ndim == 3 and tuple(mask.shape) == (batch, q_rows, kv_rows):
        mask = mask[:, None]
    elif mask.ndim == 4 and mask.shape[-2:] == (q_rows, kv_rows) and mask.shape[0] in (1, batch):
        pass
    else:
        raise ValueError("unsupported attention mask shape for key-measure composition")
    if mask.dtype == torch.bool:
        return torch.zeros(mask.shape, dtype=dtype, device=device).masked_fill(~mask, float("-inf"))
    if not mask.dtype.is_floating_point:
        raise TypeError("attention mask must be boolean or floating point")
    if torch.isnan(mask).any() or torch.isposinf(mask).any():
        raise ValueError("additive attention mask cannot contain NaN or +inf")
    return mask.to(dtype=dtype)


def _heads(q, k, v, heads, skip_reshape):
    if skip_reshape:
        if any(x.ndim != 4 for x in (q, k, v)) or q.shape[1] != heads or k.shape[1] != heads or v.shape[1] != heads:
            raise ValueError("weighted dense skip_reshape expects matching BHND Q/K/V heads")
        return q, k, v, q.shape[0], q.shape[2], q.shape[-1]
    if any(x.ndim != 3 for x in (q, k, v)):
        raise ValueError("weighted dense expects BND Q/K/V when skip_reshape is false")
    batch, q_rows, inner = q.shape
    if inner % heads or k.shape[-1] != inner or v.shape[-1] != inner:
        raise ValueError("weighted dense head geometry is invalid")
    dim_head = inner // heads
    return (
        q.view(batch, q_rows, heads, dim_head).transpose(1, 2),
        k.view(batch, k.shape[1], heads, dim_head).transpose(1, 2),
        v.view(batch, v.shape[1], heads, dim_head).transpose(1, 2),
        batch,
        q_rows,
        dim_head,
    )


def weighted_dense(
    q,
    k,
    v,
    heads,
    *,
    key_bias,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    skip_output_reshape=False,
    scale=None,
):
    """Exact all-row key-measure attention through PyTorch SDPA.

    ``key_bias`` is natural-log measure and is composed as additive score bias;
    it is never expanded to a materialized BxHxTq x Tkv tensor by this helper.
    """
    qh, kh, vh, batch, q_rows, dim_head = _heads(q, k, v, heads, skip_reshape)
    kv_rows = kh.shape[2]
    if kh.shape != vh.shape or qh.shape[-1] != kh.shape[-1]:
        raise ValueError("weighted dense Q/K/V geometry is inconsistent")
    if not torch.is_tensor(key_bias) or key_bias.ndim != 1 or key_bias.shape[0] != kv_rows or key_bias.device != q.device:
        raise ValueError("key bias must be a device-matched vector with one value per K/V row")
    if not key_bias.dtype.is_floating_point:
        raise TypeError("key bias must be floating point")
    dtype = torch.float32 if attn_precision == torch.float32 else qh.dtype
    bias = key_bias.to(dtype=dtype).view(1, 1, 1, kv_rows)
    if mask is not None:
        bias = _mask(mask, batch=batch, q_rows=q_rows, kv_rows=kv_rows, dtype=dtype, device=q.device) + bias
    out = F.scaled_dot_product_attention(
        qh.to(dtype), kh.to(dtype), vh.to(dtype), attn_mask=bias, dropout_p=0.0, is_causal=False, scale=scale
    ).to(v.dtype)
    if skip_output_reshape:
        return out
    return out.transpose(1, 2).reshape(batch, q_rows, heads * dim_head)


def weighted_dense_streaming(
    q,
    k,
    v,
    heads,
    *,
    key_bias,
    mask=None,
    skip_reshape=False,
    skip_output_reshape=False,
    scale=None,
    key_chunk_size=2048,
):
    """Bounded-memory exact fallback with online max/numerator/denominator accumulation."""
    qh, kh, vh, batch, q_rows, dim_head = _heads(q, k, v, heads, skip_reshape)
    kv_rows = kh.shape[2]
    if kh.shape != vh.shape or qh.shape[-1] != kh.shape[-1]:
        raise ValueError("weighted dense Q/K/V geometry is inconsistent")
    if not torch.is_tensor(key_bias) or key_bias.ndim != 1 or key_bias.shape[0] != kv_rows or key_bias.device != q.device:
        raise ValueError("key bias must be a device-matched vector with one value per K/V row")
    if type(key_chunk_size) is not int or key_chunk_size <= 0:
        raise ValueError("key_chunk_size must be a positive integer")
    if scale is None:
        scale = dim_head ** -0.5
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("attention scale must be finite and positive")

    work = torch.float32
    qf, kf, vf = qh.to(work), kh.to(work), vh.to(work)
    bias = key_bias.to(work)
    composed_mask = None if mask is None else _mask(
        mask,
        batch=batch,
        q_rows=q_rows,
        kv_rows=kv_rows,
        dtype=work,
        device=q.device,
    )
    row_max = torch.full((batch, heads, q_rows, 1), float("-inf"), device=q.device, dtype=work)
    row_sum = torch.zeros_like(row_max)
    numerator = torch.zeros((batch, heads, q_rows, dim_head), device=q.device, dtype=work)
    for start in range(0, kv_rows, key_chunk_size):
        stop = min(start + key_chunk_size, kv_rows)
        scores = torch.matmul(qf, kf[:, :, start:stop].transpose(-1, -2)) * scale
        scores.add_(bias[start:stop])
        if composed_mask is not None:
            scores.add_(composed_mask[..., start:stop])
        chunk_max = scores.amax(dim=-1, keepdim=True)
        new_max = torch.maximum(row_max, chunk_max)
        safe_old = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        safe_new = torch.where(torch.isfinite(new_max), new_max, torch.zeros_like(new_max))
        old_scale = torch.where(torch.isfinite(row_max), torch.exp(safe_old - safe_new), torch.zeros_like(row_sum))
        prob = torch.exp(scores - safe_new)
        prob = torch.where(torch.isfinite(scores), prob, torch.zeros_like(prob))
        numerator = numerator * old_scale + torch.matmul(prob, vf[:, :, start:stop])
        row_sum = row_sum * old_scale + prob.sum(dim=-1, keepdim=True)
        row_max = new_max
    out = (numerator / row_sum.clamp_min(torch.finfo(work).tiny)).to(v.dtype)
    if skip_output_reshape:
        return out
    return out.transpose(1, 2).reshape(batch, q_rows, heads * dim_head)
