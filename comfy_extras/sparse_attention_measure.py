from __future__ import annotations

import inspect
from collections.abc import Mapping

import torch

from comfy.attention_measure import (
    ATTENTION_MEASURE_CAPABILITIES_KEY,
    ATTENTION_MEASURE_KEY,
    BoundMeasurePlan,
    MeasureExecutionContext,
    bind,
    prepare_capability,
    register_capability,
    semantic_digest,
    validate_h3,
    weighted_dense,
)

PROVIDER_IDENTITY = "comfy.core.block_sparse_attention"
VDN_EPILOGUE_KEY = "vdn_h3_external_softmax_epilogue_v1"
VDN_EXTERNAL_SEQUENCE_KEY = "vdn_h3_external_sequence_v1"
VDN_FORWARD_MARKER = "_vdn_forward"
VDN_EXTERNAL_SEQUENCE_API_ATTR = "_vdn_external_sequence_api"


def supports_key_bias(provider) -> bool:
    try:
        parameters = inspect.signature(provider).parameters
    except (TypeError, ValueError):
        return False
    return "key_bias" in parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


# Keep the currently staged BlockSparseAttention call site valid while this
# companion remains split across files. This alias is temporary compatibility,
# not a second capability surface.
_supports_key_bias = supports_key_bias


def pool_key(block_index, rows, uuids, plan: BoundMeasurePlan | None):
    """Return the BSA calibration key without perturbing the established path.

    Unweighted H3 calls intentionally retain the historical three-tuple used by
    Spectrum's reviewed cold/primed ownership proof. Weighted calls add the
    numerical measure identity so their kmean/vscale calibration cannot be
    mistaken for an unweighted or differently weighted attention domain.
    """

    base = (int(block_index), int(rows), tuple(uuids))
    if plan is None:
        return base
    return (
        *base,
        (
            ATTENTION_MEASURE_KEY,
            plan.semantic_digest,
            plan.provider_identity,
            plan.owner_generation,
            plan.numerical_route,
            plan.preprocess_digest,
            plan.implementation_profile,
        ),
    )


def prepare_vdn_epilogue(attn, x, rope_freqs, transformer_options, block_index):
    """Resolve the optional VDN-owned external Mixed-Grid gate/projection epilogue.

    Flow's API-2 external-sequence contract describes mixed geometry and can be
    present without VDN. Only a concrete VDN attention owner must provide the
    owner-bound epilogue; an installed VDN owner without that capability remains
    fail-closed because falling back to ``attn.out_proj`` would drop its learned
    gate.
    """

    external = transformer_options.get(VDN_EXTERNAL_SEQUENCE_KEY)
    if not isinstance(external, Mapping) or external.get("api") != 2:
        return None
    if external.get("mode") != "dense_gate_no_linear" or external.get("topology") != "mixed_grid_low_suffix":
        raise RuntimeError("Mixed-Grid attention measure received an unsupported VDN external-sequence contract")
    forward = getattr(attn, "forward", None)
    if getattr(forward, VDN_FORWARD_MARKER, False) is not True:
        return None
    if getattr(forward, VDN_EXTERNAL_SEQUENCE_API_ATTR, None) != 2:
        raise RuntimeError("VDN Mixed-Grid attention owner does not advertise external-sequence API 2")
    capability = getattr(forward, VDN_EPILOGUE_KEY, None)
    prepare_epilogue = getattr(capability, "prepare", None)
    if not callable(prepare_epilogue):
        raise RuntimeError(
            "VDN API-2 Mixed-Grid attention requires the owner-bound external softmax epilogue capability"
        )
    bound = prepare_epilogue(x, rope_freqs, transformer_options, block_index)
    if not callable(getattr(bound, "apply", None)) or not callable(getattr(bound, "receipt_fields", None)):
        raise RuntimeError("VDN external softmax epilogue capability returned an invalid bound owner")
    return bound


def _cache(patch):
    cache = getattr(patch, "measure_plans", None)
    if cache is None:
        cache = {}
        patch.measure_plans = cache
    return cache


def prepare(
    patch,
    transformer_options,
    *,
    block_index,
    q_rows,
    kv_rows,
    device,
    dtype,
    head_dim,
    mask_class,
    existing_sink,
    provider,
    profile,
    numerical_route,
    preprocess_digest,
) -> BoundMeasurePlan | None:
    request = transformer_options.get(ATTENTION_MEASURE_KEY)
    if request is None:
        return None
    if patch.vsa:
        raise RuntimeError("Mixed-Grid attention measure is incompatible with VSA tile planning")
    if profile == "weighted_exact_blocks_v1" and not supports_key_bias(provider):
        raise RuntimeError(
            "Mixed-Grid attention measure selected a sparse provider without key_bias support; "
            "install a comfy-kitchen build with weighted sparse attention support"
        )
    context = MeasureExecutionContext(
        provider_identity=PROVIDER_IDENTITY,
        block_index=int(block_index),
        owner=patch,
        owner_generation=patch.measure_owner_generation,
        layout=transformer_options.get("minimax_h3_layout"),
        q_rows=int(q_rows),
        kv_rows=int(kv_rows),
        dtype=dtype,
        device=torch.device(device),
        head_dim=int(head_dim),
        mask_class=str(mask_class),
        preprocess_digest=str(preprocess_digest),
        numerical_route=str(numerical_route),
        existing_sink=tuple(existing_sink),
        external_sequence=transformer_options.get(VDN_EXTERNAL_SEQUENCE_KEY),
    )

    # A cached plan owns only the immutable O(T) measure materialization. Runtime
    # layout/external geometry and the concrete capability registry are live
    # execution facts and must be re-proven on every use; otherwise a wrapper can
    # replace them after the first bind while the cache still returns stale proof.
    normalized = validate_h3(
        request,
        layout=context.layout,
        q_rows=context.q_rows,
        kv_rows=context.kv_rows,
        external_sequence=context.external_sequence,
    )
    digest = semantic_digest(normalized)
    registry = transformer_options.get(ATTENTION_MEASURE_CAPABILITIES_KEY)
    capability = registry.get(PROVIDER_IDENTITY) if isinstance(registry, Mapping) else None
    if (
        capability is not getattr(patch, "measure_capability", None)
        or getattr(capability, "patch", None) is not patch
    ):
        raise RuntimeError("core sparse attention measure capability owner changed after binding")

    key = (
        digest,
        context.block_index,
        context.q_rows,
        context.kv_rows,
        str(context.device),
        str(context.dtype),
        context.head_dim,
        context.mask_class,
        context.existing_sink,
        profile,
        context.numerical_route,
        context.preprocess_digest,
        context.owner_generation,
    )
    cache = _cache(patch)
    result = cache.get(key)
    if result is None:
        result = prepare_capability(transformer_options, normalized, context)
        if result.implementation_profile != profile:
            raise RuntimeError(
                "core sparse attention measure capability selected an unexpected implementation profile"
            )
        cache[key] = result
    return result


def dense_call(
    patch,
    transformer_options,
    q,
    k,
    v,
    heads,
    *,
    mask,
    attn_precision,
    skip_reshape,
    skip_output_reshape,
    scale,
    block_index,
):
    q_rows = q.shape[2] if skip_reshape else q.shape[1]
    kv_rows = k.shape[2] if skip_reshape else k.shape[1]
    sink, _ = patch.sinks(transformer_options, q_rows)
    plan = prepare(
        patch,
        transformer_options,
        block_index=block_index,
        q_rows=q_rows,
        kv_rows=kv_rows,
        device=q.device,
        dtype=q.dtype,
        head_dim=q.shape[-1] if skip_reshape else q.shape[-1] // heads,
        mask_class="none" if mask is None else ("boolean" if mask.dtype == torch.bool else "additive"),
        existing_sink=sink,
        provider=weighted_dense,
        profile="dense_exact_v1",
        numerical_route="core_dense_sdpa",
        preprocess_digest="caller_attention_domain_v1",
    )
    if plan is None:
        return None
    return weighted_dense(
        q,
        k,
        v,
        heads,
        key_bias=plan.key_log_measure,
        mask=mask,
        attn_precision=attn_precision,
        skip_reshape=skip_reshape,
        skip_output_reshape=skip_output_reshape,
        scale=scale,
    )


class Capability:
    api = 1
    operator = "key_log_measure"
    provider_identity = PROVIDER_IDENTITY

    def __init__(self, patch):
        self.patch = patch

    def prepare(self, request, execution_context):
        if not isinstance(execution_context, MeasureExecutionContext):
            raise TypeError("core sparse attention requires MeasureExecutionContext")
        if execution_context.owner is not self.patch:
            raise RuntimeError("core sparse attention measure capability is bound to another patch owner")
        profile = (
            "weighted_exact_blocks_v1"
            if execution_context.numerical_route.startswith("core_bsa")
            else "dense_exact_v1"
        )
        return bind(
            request,
            context=execution_context,
            block_size=64,
            implementation_profile=profile,
        )


def register(patch, transformer_options):
    if getattr(patch, "measure_capability", None) is None:
        patch.measure_capability = Capability(patch)
    register_capability(transformer_options, PROVIDER_IDENTITY, patch.measure_capability)
