"""Proof of concept: reuse a hybrid (attention + DeltaNet) LLM's prompt state across generate calls.

One global slot holds the full-attention KV of the last call plus DeltaNet state checkpoints at the
end of the system message, the last few message boundaries, the end of the prompt and the end of
generation. A recurrent state can't be truncated, so a new prompt resumes from the latest
checkpoint inside its common prefix with the cached tokens and prefills only the rest; the KV is
truncated to match through the decode bias.

COMFY_LLM_PREFIX_CACHE: off (default) | on (kept in VRAM, moved to RAM when another model needs the
memory) | ram (moved to RAM after every call). It is dropped when there is no RAM for it.
"""
import logging
import os
import time

import torch

MODE = os.environ.get("COMFY_LLM_PREFIX_CACHE", "off")
HEADROOM = int(os.environ.get("COMFY_LLM_PREFIX_CACHE_HEADROOM", "4096"))  # spare KV slots so the next, longer prompt fits in place
CKPT_RECENT = int(os.environ.get("COMFY_LLM_PREFIX_CACHE_CKPTS", "6"))  # message-boundary checkpoints kept
MSG_END = (248046, 198)  # "<|im_end|>\n" in the Qwen3.5 vocab: a checkpoint may sit after it
MIN_SUFFIX = 7  # suffixes of 6 or fewer tokens would take the decode/verify paths

_slot = None
stats = {}


def enabled():
    return MODE in ("on", "ram")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _host(t):
    # pinned when the driver allows it; ComfyUI's own pinned weights can exhaust that before RAM runs out
    try:
        h = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    except RuntimeError:
        h = torch.empty(t.shape, dtype=t.dtype)
    h.copy_(t, non_blocking=h.is_pinned())
    return h


class _Slot:
    def __init__(self, key, pkv, ids, checkpoints):
        from .llama import FixedKVBias
        self.key = key
        self.attn = {i: kv for i, kv in enumerate(pkv) if isinstance(kv, FixedKVBias)}
        self.capacity = next(iter(self.attn.values())).key.shape[2]
        self.ids = list(ids)  # the tokens the KV holds
        self.checkpoints = checkpoints  # {position: {layer: (conv, recurrent)}}
        self.parked = None  # {layer: (key, value)} in host RAM

    def nbytes(self):
        if self.parked is not None:
            n = sum(k.nbytes + v.nbytes for k, v in self.parked.values())
        else:
            n = sum(kv.key[:, :, :len(self.ids)].nbytes + kv.value[:, :, :len(self.ids)].nbytes for kv in self.attn.values())
        return n + sum(c.nbytes + r.nbytes for st in self.checkpoints.values() for c, r in st.values())

    def park(self):
        # copy the used KV and the checkpoints to host memory and free the device copies
        if self.parked is not None:
            return
        t = time.perf_counter()
        n = len(self.ids)
        self.parked = {i: (_host(kv.key[:, :, :n]), _host(kv.value[:, :, :n])) for i, kv in self.attn.items()}
        self.checkpoints = {pos: {i: (_host(c), _host(r)) for i, (c, r) in st.items()} for pos, st in self.checkpoints.items()}
        device = next(iter(self.attn.values())).key.device
        _sync(device)
        self.attn = None
        logging.info("llm prefix cache: moved %.2f GB to RAM in %.2f s", self.nbytes() / 1e9, time.perf_counter() - t)


def _match(ids):
    """(position, states) of the latest checkpoint inside the common prefix, or None."""
    lcp = 0
    for a, b in zip(ids, _slot.ids):
        if a != b:
            break
        lcp += 1
    limit = min(lcp, len(ids) - MIN_SUFFIX)
    usable = [pos for pos in _slot.checkpoints if pos <= limit]
    if not usable:
        return None
    pos = max(usable)
    return pos, _slot.checkpoints[pos]


def _set_prefix(pkv, n):
    # every layer at index n; decode bias open below n and closed from n on
    from .llama import FixedKVBias
    for kv in pkv:
        kv.index = n
    for kv in pkv:
        if isinstance(kv, FixedKVBias):
            kv.bias[..., :n] = 0
            kv.bias[..., n:] = torch.finfo(kv.bias.dtype).min
            kv.tracker["step"] = -1
            break


def _restore(model, key, ids, need, device, dtype):
    """Returns (pkv, reused prefix length, checkpoints still valid). Allocates a fresh cache on a miss."""
    global _slot
    from .qwen35 import LinearKV
    hit = _match(ids) if _slot is not None and _slot.key == key else None
    if hit is None:
        _slot = None  # drop the old KV before allocating the new one
        return model.init_kv_cache(1, need + HEADROOM, device, dtype), 0, {}
    n, st = hit
    carried = {pos: cst for pos, cst in _slot.checkpoints.items() if pos <= n}
    if _slot.parked is None and _slot.capacity >= need:
        pkv = [None] * len(model.model.config.layer_types)
        for i, kv in _slot.attn.items():
            pkv[i] = kv
        stats["restore"] = "in-place"
    else:
        # parked, or too small for this prompt + max_length: copy the prefix into a new allocation
        pkv = model.init_kv_cache(1, max(need + HEADROOM, _slot.capacity), device, dtype)
        src = _slot.parked if _slot.parked is not None else {i: (kv.key, kv.value) for i, kv in _slot.attn.items()}
        for i, (k, v) in src.items():
            pkv[i].key[:, :, :n].copy_(k[:, :, :n], non_blocking=True)
            pkv[i].value[:, :, :n].copy_(v[:, :, :n], non_blocking=True)
        stats["restore"] = "from-ram" if _slot.parked is not None else "grow"
    for i, (c, r) in st.items():
        pkv[i] = LinearKV(c.to(device, copy=True, non_blocking=True), r.to(device, copy=True, non_blocking=True), n, None, None)
    _set_prefix(pkv, n)
    _slot = None  # rebuilt from this call's cache in prefill()
    return pkv, n, carried


def _checkpoint_positions(ids, start):
    # after the system message (the static prefix) and after the last few messages
    ends = [j for j in range(2, len(ids) + 1) if ids[j - 2] == MSG_END[0] and ids[j - 1] == MSG_END[1]]
    want = sorted(set(ends[:1] + ends[-CKPT_RECENT:])) if CKPT_RECENT > 0 else []
    out, prev = [], start
    for pos in want:
        if pos - prev >= MIN_SUFFIX and len(ids) - pos >= MIN_SUFFIX:
            out.append(pos)
            prev = pos
    return out


def _linear_states(pkv, clone):
    from .qwen35 import LinearKV
    out = {}
    for i, kv in enumerate(pkv):
        if isinstance(kv, LinearKV):
            out[i] = (kv.conv_state.clone(), kv.recurrent_state.clone()) if clone else (kv.conv_state, kv.recurrent_state)
            if not clone:
                kv.snapshots = kv.snap_backing = kv.conv_snap_backing = None
    return out


def _key(model, device, dtype):
    # which weights produced the cache: model type, device, dtype and a fingerprint of a few small
    # unquantized tensors, so another checkpoint or fine-tune of the same type can't reuse it.
    # Weight patches applied at load time (LoRA) are not covered.
    t = model.model
    norms = [t.layers[0].input_layernorm, t.layers[len(t.layers) // 2].post_attention_layernorm, t.norm]
    fingerprint = tuple(round(float(m.weight.detach().float().sum()), 4) for m in norms if m is not None)
    return getattr(model, "model_type", type(model).__name__), str(device), str(dtype), fingerprint


def prefill(model, embeds, ids, capacity, forward):
    """Prefill `embeds` (the whole prompt) reusing a cached prefix of `ids`. Returns (pkv, x of the last position)."""
    global _slot
    stats.clear()
    device, dtype = embeds.device, embeds.dtype
    key = _key(model, device, dtype)
    _sync(device)
    stats["t0"] = time.perf_counter()
    pkv, n, checkpoints = _restore(model, key, ids, capacity, device, dtype)
    _sync(device)
    t1 = time.perf_counter()
    stats.update(prompt=len(ids), reused=n, restore_s=round(t1 - stats["t0"], 3))
    # prefill in segments that end on checkpoint positions; the DeltaNet states are cloned at each
    # because the next segment and decode update them in place
    start = n
    for pos in _checkpoint_positions(ids, n) + [len(ids)]:
        x = forward(embeds[:, start:pos], pkv)
        checkpoints[pos] = _linear_states(pkv, clone=True)
        start = pos
    _sync(device)
    stats["prefill_s"] = round(time.perf_counter() - t1, 3)
    if len(checkpoints) > CKPT_RECENT + 3:
        keep = sorted(checkpoints)
        keep = keep[:1] + keep[-(CKPT_RECENT + 2):]
        checkpoints = {pos: checkpoints[pos] for pos in keep}
    _slot = _Slot(key, pkv, ids, checkpoints)
    return pkv, x


def note_token(i):
    if "t0" in stats and f"token{i}_s" not in stats:
        stats[f"token{i}_s"] = round(time.perf_counter() - stats["t0"], 3)


def finish(pkv, ids, generated):
    """Checkpoint the end of generation (prompt + the generated tokens the model consumed), then park if asked."""
    from .qwen35 import LinearKV
    if _slot is None:
        return
    full = list(ids) + list(generated)
    idx = next(kv.index for kv in pkv if isinstance(kv, LinearKV))
    # MTP may have consumed accepted drafts past a stop token: then idx > len(full) and the state is unusable
    if _slot.parked is None and len(ids) < idx <= len(full):
        _slot.checkpoints[idx] = _linear_states(pkv, clone=False)
        _slot.ids = full[:idx]
    stats["end_s"] = round(time.perf_counter() - stats["t0"], 3)
    stats["cache_gb"] = round(_slot.nbytes() / 1e9, 2)
    if MODE == "ram":
        _park_or_drop()
    logging.info("llm prefix cache: %s", {k: v for k, v in stats.items() if k != "t0"})


def drop():
    global _slot
    _slot = None


def _park_or_drop():
    # parking must never fail the caller: without the RAM for it, the cache is dropped
    try:
        _slot.park()
    except RuntimeError as e:
        logging.warning("llm prefix cache: dropped, could not move it to RAM: %s", e)
        drop()


def on_memory_pressure(memory_required, device):
    # model_management.free_memory hook: park instead of holding VRAM another model needs
    if _slot is None or _slot.parked is not None or device is None or device.type != "cuda":
        return
    import comfy.model_management
    if comfy.model_management.get_free_memory(device) < memory_required:
        _park_or_drop()
