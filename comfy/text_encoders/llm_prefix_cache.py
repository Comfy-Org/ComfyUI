"""Spike: reuse a hybrid (attention + DeltaNet) LLM's prompt state across generate calls.

One global slot. It keeps the full-attention KV of the last call and the DeltaNet states at up to
two checkpoints: the end of the prompt prefill and the end of generation. A recurrent state can't
be truncated, so a new prompt reuses a checkpoint only when it starts with exactly that
checkpoint's token ids; the KV is truncated to match by the decode bias.

COMFY_LLM_PREFIX_CACHE: off (default) | gpu (resident, parked in pinned RAM when another model
needs the VRAM) | cpu (parked in pinned RAM after every call).
COMFY_LLM_PREFILL_CHUNK: prefill in chunks of this many tokens (0 = one pass).
"""
import json
import logging
import os
import time

import torch

MODE = os.environ.get("COMFY_LLM_PREFIX_CACHE", "off")
CHUNK = int(os.environ.get("COMFY_LLM_PREFILL_CHUNK", "0"))
HEADROOM = int(os.environ.get("COMFY_LLM_PREFIX_CACHE_HEADROOM", "4096"))
MIN_SUFFIX = 7  # suffixes of 6 or fewer tokens would take the decode/verify paths

_slot = None
stats = {}


def enabled():
    return MODE in ("gpu", "cpu")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _Slot:
    def __init__(self, key, pkv):
        from .llama import FixedKVBias
        self.key = key
        self.attn = {i: kv for i, kv in enumerate(pkv) if isinstance(kv, FixedKVBias)}
        self.shared = None
        for kv in self.attn.values():
            self.shared = (kv.position, kv.bias, kv.tracker)
            break
        self.capacity = next(iter(self.attn.values())).key.shape[2]
        self.kv_len = 0
        self.checkpoints = []  # [(ids, {layer: (conv, recurrent)})]
        self.cpu = None  # {layer: (key, value)} when parked

    def nbytes(self):
        n = 0
        if self.cpu is not None:
            n += sum(k.nbytes + v.nbytes for k, v in self.cpu.values())
        else:
            n += sum(kv.key[:, :, :self.kv_len].nbytes + kv.value[:, :, :self.kv_len].nbytes for kv in self.attn.values())
        n += sum(c.nbytes + r.nbytes for _, st in self.checkpoints for c, r in st.values())
        return n

    def park(self):
        # copy the used KV and the states to pinned host memory and free the device copies
        if self.cpu is not None:
            return
        t = time.perf_counter()
        cpu = {}
        for i, kv in self.attn.items():
            k = torch.empty(kv.key[:, :, :self.kv_len].shape, dtype=kv.key.dtype, pin_memory=True)
            v = torch.empty_like(k, pin_memory=True)
            k.copy_(kv.key[:, :, :self.kv_len], non_blocking=True)
            v.copy_(kv.value[:, :, :self.kv_len], non_blocking=True)
            cpu[i] = (k, v)
        cps = []
        for ids, st in self.checkpoints:
            pst = {}
            for i, (c, r) in st.items():
                pc = torch.empty(c.shape, dtype=c.dtype, pin_memory=True)
                pr = torch.empty(r.shape, dtype=r.dtype, pin_memory=True)
                pc.copy_(c, non_blocking=True)
                pr.copy_(r, non_blocking=True)
                pst[i] = (pc, pr)
            cps.append((ids, pst))
        torch.cuda.synchronize()
        self.cpu = cpu
        self.checkpoints = cps
        self.attn = None
        self.shared = None
        stats["park_s"] = round(time.perf_counter() - t, 3)
        logging.info("llm prefix cache: parked %.2f GB in pinned RAM in %.3f s", self.nbytes() / 1e9, stats["park_s"])


def _match(ids):
    best = None
    if _slot is None:
        return None
    for cid, st in _slot.checkpoints:
        n = len(cid)
        if n <= len(ids) - MIN_SUFFIX and (best is None or n > len(best[0])) and ids[:n] == cid:
            best = (cid, st)
    return best


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
    """Returns (pkv, prefix_len). Allocates a fresh cache on a miss."""
    global _slot
    from .qwen35 import LinearKV
    hit = _match(ids) if _slot is not None and _slot.key == key else None
    if hit is None:
        if _slot is not None:
            stats["miss_reason"] = "key" if _slot.key != key else "prefix"
        _slot = None  # drop the old KV before allocating the new one
        return model.init_kv_cache(1, need + HEADROOM, device, dtype), 0
    cid, st = hit
    n = len(cid)
    if _slot.cpu is None and _slot.capacity >= need:
        pkv = [None] * len(model.model.config.layer_types)
        for i, kv in _slot.attn.items():
            pkv[i] = kv
        stats["restore"] = "in-place"
    else:
        # parked, or too small for this prompt + max_length: copy the prefix into a new allocation
        pkv = model.init_kv_cache(1, max(need + HEADROOM, _slot.capacity), device, dtype)
        src = _slot.cpu if _slot.cpu is not None else {i: (kv.key, kv.value) for i, kv in _slot.attn.items()}
        for i, (k, v) in src.items():
            pkv[i].key[:, :, :n].copy_(k[:, :, :n], non_blocking=True)
            pkv[i].value[:, :, :n].copy_(v[:, :, :n], non_blocking=True)
        stats["restore"] = "from-pinned" if _slot.cpu is not None else "grow"
    for i, (c, r) in st.items():
        pkv[i] = LinearKV(c.to(device, copy=True, non_blocking=True), r.to(device, copy=True, non_blocking=True), n, None, None)
    _set_prefix(pkv, n)
    # the slot is rebuilt from this call's cache in commit()
    _slot = None
    return pkv, n


def prefill(model, embeds, ids, capacity, forward):
    """Prefill `embeds` (whole prompt) reusing a cached prefix of `ids`. Returns (pkv, x of the last position)."""
    global _slot
    stats.clear()
    device, dtype = embeds.device, embeds.dtype
    key = (getattr(model, "model_type", type(model).__name__), str(dtype))
    _sync(device)
    t0 = time.perf_counter()
    stats["t0"] = t0
    pkv, n = _restore(model, key, ids, capacity, device, dtype)
    _sync(device)
    t1 = time.perf_counter()
    stats.update(prompt_len=len(ids), prefix_len=n, hit=n > 0, restore_s=round(t1 - t0, 3))
    x = _run_prefill(embeds[:, n:], pkv, forward)
    _sync(device)
    stats["prefill_s"] = round(time.perf_counter() - t1, 3)
    # checkpoint A: the prompt end; the DeltaNet states are cloned because decode updates them in place
    _slot = _Slot(key, pkv)
    _slot.kv_len = len(ids)
    _slot.checkpoints = [(list(ids), _linear_states(pkv, clone=True))]
    return pkv, x


def _run_prefill(embeds, pkv, forward):
    total = embeds.shape[1]
    if CHUNK <= 0 or total <= CHUNK + MIN_SUFFIX:
        return forward(embeds, pkv)
    x = None
    start = 0
    while start < total:
        end = start + CHUNK
        if total - end < MIN_SUFFIX:
            end = total
        x = forward(embeds[:, start:end], pkv)[:, -1:]
        start = end
    return x


def _linear_states(pkv, clone):
    from .qwen35 import LinearKV
    out = {}
    for i, kv in enumerate(pkv):
        if isinstance(kv, LinearKV):
            out[i] = (kv.conv_state.clone(), kv.recurrent_state.clone()) if clone else (kv.conv_state, kv.recurrent_state)
            if not clone:
                kv.snapshots = kv.snap_backing = kv.conv_snap_backing = None
    return out


def note_token(i):
    if "t0" in stats and f"t_tok{i}" not in stats:
        stats[f"t_tok{i}"] = round(time.perf_counter() - stats["t0"], 3)


def finish(pkv, ids, generated):
    """Checkpoint B (prompt + the generated tokens the model has consumed), then park if asked."""
    from .qwen35 import LinearKV
    global _slot
    if _slot is None:
        return
    stats["t_end"] = round(time.perf_counter() - stats["t0"], 3)
    stats["gen_tokens"] = len(generated)
    full = list(ids) + list(generated)
    idx = next(kv.index for kv in pkv if isinstance(kv, LinearKV))
    if len(ids) < idx <= len(full):
        # MTP may have consumed accepted drafts past a stop token: then idx > len(full) and the state is unusable
        _slot.checkpoints.append((full[:idx], _linear_states(pkv, clone=False)))
        _slot.kv_len = idx
    stats["cache_gb"] = round(_slot.nbytes() / 1e9, 3)
    if MODE == "cpu":
        _slot.park()
    out = {k: v for k, v in stats.items() if k != "t0"}
    logging.info("LLM_PREFIX_CACHE %s", json.dumps(out))


def drop():
    global _slot
    _slot = None


def on_memory_pressure(memory_required, device):
    # model_management.free_memory hook: park instead of holding VRAM another model needs
    if _slot is None or _slot.cpu is not None or device is None or device.type != "cuda":
        return
    import comfy.model_management
    if comfy.model_management.get_free_memory(device) < memory_required:
        _slot.park()
