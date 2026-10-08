"""Spike: reuse a hybrid (attention + DeltaNet) LLM's prompt state across generate calls.

One global slot. It keeps the full-attention KV of the last call and DeltaNet state checkpoints at
the end of the system message, the last few message boundaries, the end of the prompt and the end
of generation. A recurrent state can't be truncated, so a new prompt resumes from the latest
checkpoint inside its common prefix with the cached tokens; the KV is truncated to match by the
decode bias.

COMFY_LLM_PREFIX_CACHE: off (default) | gpu (resident, parked in pinned RAM when another model
needs the VRAM) | vram (resident, never parked) | cpu (parked in pinned RAM after every call) |
disk (parked to COMFY_LLM_PREFIX_CACHE_DIR after every call, mmapped back) | stats (same timing log, never reuses).
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
MARGINS = os.environ.get("COMFY_LLM_PREFIX_CACHE_MARGINS", "0") == "1"  # log top-2 logit gaps (plain decode)
DISK_DIR = os.environ.get("COMFY_LLM_PREFIX_CACHE_DIR", "/tmp/llm_prefix_cache")
CKPT_RECENT = int(os.environ.get("COMFY_LLM_PREFIX_CACHE_CKPTS", "6"))  # message-boundary checkpoints kept
MSG_END = (248046, 198)  # "<|im_end|>\n" in the Qwen3.5 vocab: a checkpoint may sit after it
MIN_SUFFIX = 7  # suffixes of 6 or fewer tokens would take the decode/verify paths

_slot = None
stats = {}
_disk_gen = [0]


class _Host:
    # parks one tensor at a time: pinned RAM, or a file on disk that is mmapped back
    def __init__(self, disk):
        self.disk = disk
        if disk:
            _disk_gen[0] += 1
            os.makedirs(DISK_DIR, exist_ok=True)
            for f in os.listdir(DISK_DIR):
                if not f.startswith(f"g{_disk_gen[0]}-"):
                    try:
                        os.remove(os.path.join(DISK_DIR, f))
                    except OSError:
                        pass
        self.n = 0

    def __call__(self, t):
        if not self.disk:
            # pinned when the driver allows it; ComfyUI's own pinned weights can exhaust that before RAM runs out
            try:
                h = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
            except RuntimeError:
                h = torch.empty(t.shape, dtype=t.dtype)
                stats["park_pageable"] = True
            h.copy_(t, non_blocking=h.is_pinned())
            return h
        self.n += 1
        path = os.path.join(DISK_DIR, f"g{_disk_gen[0]}-{self.n}.pt")
        torch.save(t.cpu(), path)
        return torch.load(path, mmap=True)


def enabled():
    return MODE in ("gpu", "vram", "cpu", "disk", "stats")


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
        self.ids = []  # the tokens the KV holds, up to kv_len
        self.checkpoints = {}  # {position: {layer: (conv, recurrent)}}
        self.cpu = None  # {layer: (key, value)} when parked

    def nbytes(self):
        n = 0
        if self.cpu is not None:
            n += sum(k.nbytes + v.nbytes for k, v in self.cpu.values())
        else:
            n += sum(kv.key[:, :, :self.kv_len].nbytes + kv.value[:, :, :self.kv_len].nbytes for kv in self.attn.values())
        n += sum(c.nbytes + r.nbytes for st in self.checkpoints.values() for c, r in st.values())
        return n

    def park(self):
        # copy the used KV and the states to pinned host memory and free the device copies
        if self.cpu is not None:
            return
        t = time.perf_counter()
        host = _Host(MODE == "disk")
        cpu = {}
        for i, kv in self.attn.items():
            cpu[i] = (host(kv.key[:, :, :self.kv_len]), host(kv.value[:, :, :self.kv_len]))
        cps = {pos: {i: (host(c), host(r)) for i, (c, r) in st.items()} for pos, st in self.checkpoints.items()}
        torch.cuda.synchronize()
        self.cpu = cpu
        self.where = "disk" if host.disk else "pinned"
        self.checkpoints = cps
        self.attn = None
        self.shared = None
        stats["park_s"] = round(time.perf_counter() - t, 3)
        logging.info("llm prefix cache: parked %.2f GB (%s) in %.3f s", self.nbytes() / 1e9, MODE, stats["park_s"])


def _match(ids):
    """(position, states) of the latest checkpoint inside the common prefix, or None."""
    if _slot is None:
        return None
    lcp = 0
    for a, b in zip(ids, _slot.ids):
        if a != b:
            break
        lcp += 1
    stats["common_prefix"] = lcp
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
    """Returns (pkv, prefix_len, carried checkpoints). Allocates a fresh cache on a miss."""
    global _slot
    from .qwen35 import LinearKV
    hit = _match(ids) if _slot is not None and _slot.key == key and MODE != "stats" else None
    if hit is None:
        if _slot is not None:
            stats["miss_reason"] = "key" if _slot.key != key else "prefix"
        _slot = None  # drop the old KV before allocating the new one
        # stats mode sizes the cache as core does without it
        return model.init_kv_cache(1, need + (0 if MODE == "stats" else HEADROOM), device, dtype), 0, {}
    n, st = hit
    carried = {pos: cst for pos, cst in _slot.checkpoints.items() if pos <= n}
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
        stats["restore"] = ("from-" + _slot.where) if _slot.cpu is not None else "grow"
    for i, (c, r) in st.items():
        pkv[i] = LinearKV(c.to(device, copy=True, non_blocking=True), r.to(device, copy=True, non_blocking=True), n, None, None)
    _set_prefix(pkv, n)
    # the slot is rebuilt from this call's cache in prefill()
    _slot = None
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


def prefill(model, embeds, ids, capacity, forward):
    """Prefill `embeds` (whole prompt) reusing a cached prefix of `ids`. Returns (pkv, x of the last position)."""
    global _slot
    stats.clear()
    device, dtype = embeds.device, embeds.dtype
    key = (getattr(model, "model_type", type(model).__name__), str(dtype))
    _sync(device)
    t0 = time.perf_counter()
    stats["t0"] = t0
    pkv, n, checkpoints = _restore(model, key, ids, capacity, device, dtype)
    _sync(device)
    t1 = time.perf_counter()
    stats.update(prompt_len=len(ids), prefix_len=n, hit=n > 0, restore_s=round(t1 - t0, 3))
    # prefill in segments that end on checkpoint positions; the DeltaNet states are cloned at each
    # because the next segment and decode update them in place
    start = n
    # stats mode prefills in one pass, as core does without the cache
    for pos in ([] if MODE == "stats" else _checkpoint_positions(ids, n)) + [len(ids)]:
        x = _run_prefill(embeds[:, start:pos], pkv, forward)
        if MODE != "stats":
            checkpoints[pos] = _linear_states(pkv, clone=True)
        start = pos
    _sync(device)
    stats["prefill_s"] = round(time.perf_counter() - t1, 3)
    if len(checkpoints) > CKPT_RECENT + 3:
        keep = sorted(checkpoints)
        keep = keep[:1] + keep[-(CKPT_RECENT + 2):]
        checkpoints = {pos: checkpoints[pos] for pos in keep}
    stats["checkpoints"] = sorted(checkpoints)
    _slot = _Slot(key, pkv)
    _slot.kv_len = len(ids)
    _slot.ids = list(ids)
    _slot.checkpoints = checkpoints
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
    stats["gen_ids"] = list(generated)
    full = list(ids) + list(generated)
    idx = next(kv.index for kv in pkv if isinstance(kv, LinearKV))
    if _slot.cpu is None and len(ids) < idx <= len(full):
        # MTP may have consumed accepted drafts past a stop token: then idx > len(full) and the state is unusable
        _slot.checkpoints[idx] = _linear_states(pkv, clone=False)
        _slot.ids = full[:idx]
        _slot.kv_len = idx
    stats["cache_gb"] = round(_slot.nbytes() / 1e9, 3)
    stats["mode"] = MODE
    if MODE == "stats":
        _slot = None
    elif MODE in ("cpu", "disk"):
        _slot.park()
    out = {k: v for k, v in stats.items() if k != "t0"}
    logging.info("LLM_PREFIX_CACHE %s", json.dumps(out))


def drop():
    global _slot
    _slot = None


def on_memory_pressure(memory_required, device):
    # model_management.free_memory hook: park instead of holding VRAM another model needs
    if MODE == "vram" or _slot is None or _slot.cpu is not None or device is None or device.type != "cuda":
        return
    import comfy.model_management
    if comfy.model_management.get_free_memory(device) < memory_required:
        _slot.park()
