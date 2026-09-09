"""Numerical-backend history contract for core block-sparse attention.

The contract is intentionally duck-typed: forecast providers such as Spectrum can
consume it without ComfyUI importing them. A policy describes the numerical route
that the next MiniMax-H3 transformer evaluation would take; successful actual
calls append receipts that confirm the route that really executed.
"""
from __future__ import annotations

from dataclasses import dataclass


POLICIES_KEY = "attention_backend_history_v1"
RECEIPTS_KEY = "attention_backend_receipts_v1"
PROVIDER_KEY = "comfy_core_block_sparse_attention"
CONTRACT_VERSION = 1
H3_BLOCK_SCOPE = "_block_sparse_attention_h3_block_v1"

_ALLOWED_ROUTES = {
    "h3_dense",
    "h3_override_sparse",
    "h3_chunked_sparse_cold",
    "h3_chunked_sparse_primed",
}


def _freeze(value):
    """Turn small runtime/config values into a stable hashable identity."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    return repr(value)


def _callable_identity(function):
    if function is None:
        return ("comfy.default",)
    function = getattr(function, "__func__", function)
    return (
        str(getattr(function, "__module__", "<unknown>")),
        str(getattr(function, "__qualname__", type(function).__name__)),
        id(function),
    )


def _attention_provider_identity(provider):
    """Describe an inherited dense provider plus explicit preprocessing chain."""
    transforms = []
    seen = set()
    while hasattr(provider, "attention_preprocess_v1"):
        if id(provider) in seen:
            return ("cyclic_preprocess", id(provider))
        seen.add(id(provider))
        contract = provider.attention_preprocess_v1
        if not isinstance(contract, tuple) or len(contract) != 2:
            return ("invalid_preprocess", _callable_identity(provider))
        transform, provider = contract
        transforms.append(_callable_identity(transform))
    return (tuple(transforms), _callable_identity(provider))


def mark_attention_override(override, patch, previous):
    """Publish enough provenance for preflight to prove which BSA owns the hook."""
    override._block_sparse_attention_patch = patch
    override._block_sparse_attention_previous = previous
    return override


def mark_h3_block_patch(block_patch, patch, block_index, previous):
    block_patch._block_sparse_attention_patch = patch
    block_patch._block_sparse_attention_index = int(block_index)
    block_patch._block_sparse_attention_previous = previous
    return block_patch


def append_history_receipt(options, patch, block_index, route, tokens, sink=(0, 0), sink_q=(0, 0)):
    """Append a receipt only when a forecast provider requested receipt tracking."""
    receipts = options.get(RECEIPTS_KEY)
    if receipts is None:
        return
    receipts.append((
        PROVIDER_KEY,
        CONTRACT_VERSION,
        id(patch),
        int(block_index),
        str(route),
        int(tokens),
        tuple(int(value) for value in sink),
        tuple(int(value) for value in sink_q),
    ))


def append_scoped_h3_receipt(options, patch, route, tokens, sink=(0, 0), sink_q=(0, 0)):
    """Record generic-override execution only while a main H3 block owns the call."""
    scope = options.get(H3_BLOCK_SCOPE)
    if not isinstance(scope, tuple) or len(scope) != 2 or scope[0] is not patch:
        return
    append_history_receipt(options, patch, scope[1], route, tokens, sink, sink_q)


def _patch_mode(patch):
    if patch.vsa:
        return "vsa"
    return "sol-attn" if float(patch.topk_ratio) == 0.0 else "sla"


def _sigma(options):
    sigmas = options.get("sigmas")
    if sigmas is None:
        return None
    try:
        if len(sigmas) == 0:
            return None
    except TypeError:
        return None
    try:
        return float(sigmas[0])
    except (TypeError, ValueError, RuntimeError):
        return None


def _replacement_identity(options, block_count):
    replacements = options.get("patches_replace", {}).get("dit", {})
    identities = []
    for index in range(block_count):
        replacement = replacements.get(("double_block", index))
        if replacement is None:
            return None
        identities.append(_callable_identity(replacement))
    return tuple(identities)


def _forward_identity(model):
    identities = []
    for block in model.blocks:
        attn = getattr(block, "attn", None)
        forward = getattr(attn, "forward", None)
        if forward is None:
            return None
        identities.append(_callable_identity(forward))
    return tuple(identities)


@dataclass(frozen=True)
class SparseAttentionHistoryPolicy:
    """Preflight identity and receipt validator for MiniMax-H3 core BSA.

    H3's chunked Sol-Attn producer reuses per-block pooled K/V statistics from
    the previous actual evaluation. The policy therefore distinguishes a cold
    geometry/conditioning branch from a primed one. A newly primed key changes
    the policy identity and forces Spectrum to establish a fresh actual anchor
    before forecasting through that regime.
    """

    patch: object

    def __call__(self, *, layout, options, model):
        patch = self.patch
        block_count = getattr(patch, "h3_block_count", None)
        if patch.vsa or type(block_count) is not int or block_count <= 0:
            return None
        blocks = getattr(model, "blocks", None)
        if blocks is None or len(blocks) != block_count:
            return None

        seq_len = getattr(layout, "seq_len", None)
        segments = getattr(layout, "segments", None)
        if type(seq_len) is not int or seq_len <= 0 or not segments:
            return None

        current = options.get("optimized_attention_override")
        if getattr(current, "_block_sparse_attention_patch", None) is not patch:
            return None
        previous = getattr(current, "_block_sparse_attention_previous", None)
        # Two BSA instances stacked on one attention hook are not an audited
        # numerical route. They may still execute, but forecasts stay fail-closed.
        if getattr(previous, "_block_sparse_attention_patch", None) is not None:
            return None

        sigma = _sigma(options)
        if sigma is None:
            return None
        replacement_identity = _replacement_identity(options, block_count)
        forward_identity = _forward_identity(model)
        if replacement_identity is None or forward_identity is None:
            return None

        inside_window = patch.sigma_end <= sigma <= patch.sigma_start
        sparse_phase = inside_window and seq_len >= patch.min_tokens
        phase = "sparse" if sparse_phase else "dense"

        uuids = tuple(options.get("uuids", ()))
        primed = tuple(sorted(
            int(key[0])
            for key in patch.pooled
            if isinstance(key, tuple)
            and len(key) == 3
            and key[1] == seq_len
            and key[2] == uuids
        ))

        signature = _freeze(getattr(layout, "signature", None))
        return (
            PROVIDER_KEY,
            CONTRACT_VERSION,
            id(patch),
            _patch_mode(patch),
            float(patch.tau),
            float(patch.topk_ratio),
            int(patch.extra_tokens),
            float(patch.sigma_start),
            float(patch.sigma_end),
            phase,
            int(patch.min_tokens),
            tuple(sorted(int(index) for index in patch.dense_blocks)),
            str(patch.sink_conditioning),
            seq_len,
            signature,
            _freeze(tuple(segments)),
            _freeze(uuids),
            primed,
            str(getattr(model, "dtype", None)),
            _attention_provider_identity(previous),
            replacement_identity,
            forward_identity,
        )

    def accept_receipts(self, receipts):
        block_count = getattr(self.patch, "h3_block_count", None)
        if type(block_count) is not int or block_count <= 0 or not receipts:
            return False
        seen = set()
        token_counts = set()
        for item in receipts:
            if not isinstance(item, tuple) or len(item) != 8:
                return False
            provider, version, patch_id, block, route, tokens, sink, sink_q = item
            if provider != PROVIDER_KEY or version != CONTRACT_VERSION or patch_id != id(self.patch):
                return False
            if type(block) is not int or block < 0 or block >= block_count or block in seen:
                return False
            if route not in _ALLOWED_ROUTES or type(tokens) is not int or tokens <= 0:
                return False
            if not (
                isinstance(sink, tuple) and len(sink) == 2
                and isinstance(sink_q, tuple) and len(sink_q) == 2
            ):
                return False
            seen.add(block)
            token_counts.add(tokens)
        return seen == set(range(block_count)) and len(token_counts) == 1


def install_history_policy(patch, transformer_options):
    """Publish BSA's provider without disturbing unrelated history providers."""
    policy = getattr(patch, "history_policy", None)
    if policy is None:
        policy = SparseAttentionHistoryPolicy(patch)
        patch.history_policy = policy
    current = transformer_options.get(POLICIES_KEY)
    if current is None:
        providers = {}
    elif isinstance(current, dict):
        providers = dict(current)
    else:
        # An unknown owner already uses the key with a non-contract shape. Do not
        # overwrite it; Spectrum will continue to fail closed for core BSA.
        return False
    providers[PROVIDER_KEY] = policy
    transformer_options[POLICIES_KEY] = providers
    return True
