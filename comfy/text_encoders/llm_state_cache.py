"""Reuse a hybrid (attention + DeltaNet) LLM's prompt state across TextGenerate calls, kept in model
states (comfy.model_state) so the KV lives in dynamic VRAM, spills to RAM when another model needs the card, and
streams back per layer.

The cache keeps a few resident states, one per conversation. A new prompt takes the state with the longest common
token prefix, resumes from the latest DeltaNet checkpoint inside it (a recurrent state can't be truncated; the
attention KV is truncated by moving its index back), and prefills the rest in batched segments that end on message
boundaries, where it takes the next checkpoints. Image positions get ids from a hash of the image, so image prompts
match too.

Memory: states stay loaded between calls (nothing is written per call). Past COMFY_LLM_STATE_CACHE_GB of states, or
COMFY_LLM_STATE_CACHE_ENTRIES of them, the least recently used is saved to a file in COMFY_LLM_STATE_CACHE_DIR and
dropped; its next match loads the file. The shared start (the prompt up to the end of the system message) is saved
once to a file named by the hash of its tokens, if no such file exists, and a new conversation with no resident match
starts from it.

COMFY_LLM_PREFIX_CACHE: off (default) | on.
With the int8 KV cache, a state resumes only where its bf16 window still holds the two pages before the resume point.
"""
import contextlib
import hashlib
import json
import logging
import os
import time

import safetensors
import safetensors.torch
import torch

MODE = os.environ.get("COMFY_LLM_PREFIX_CACHE", "off")
CACHE_DIR = os.environ.get("COMFY_LLM_STATE_CACHE_DIR", "/tmp/llm_state_cache")
BUDGET = int(float(os.environ.get("COMFY_LLM_STATE_CACHE_GB", "16")) * 2**30)
MAX_ENTRIES = int(os.environ.get("COMFY_LLM_STATE_CACHE_ENTRIES", "4"))
HEADROOM = int(os.environ.get("COMFY_LLM_PREFIX_CACHE_HEADROOM", "8192"))  # capacity beyond this call, for later turns
CKPT_RECENT = int(os.environ.get("COMFY_LLM_PREFIX_CACHE_CKPTS", "6"))  # message-boundary checkpoints kept
MSG_END = (248046, 198)  # "<|im_end|>\n" in the Qwen3.5 vocab: a checkpoint may sit after it
MIN_SUFFIX = 9  # suffixes of llama.SPEC_ROWS (8) or fewer tokens would take the decode/verify paths
ROUND = 4096

stats = {}
_entries = []  # resident and spilled states, most recently used last
_tick = [0]


_call = [None]  # a per-call mode set by the TextGenerate node, else MODE


def enabled():
    return (_call[0] or MODE) == "on"


@contextlib.contextmanager
def call_mode(mode):
    _call[0] = mode
    try:
        yield
    finally:
        _call[0] = None


class _Entry:
    """One conversation's state: a loaded ModelState (or a file it was spilled to), the tokens its KV holds, and DeltaNet
    checkpoints {position: {layer: (conv, recurrent)}} on the CPU."""

    def __init__(self, key, state, capacity):
        self.key = key
        self.state = state  # comfy.model_state.ModelState, loaded; None once spilled
        self.capacity = capacity
        self.ids = []
        self.checkpoints = {}
        self.file = None
        self.used = 0
        self.window = _int8_window(state.patcher.model) if state is not None else None  # (rows, page) for an int8 cache
        self.written = 0  # rows written past the committed ones (draft rows included) bound what the window still holds

    def nbytes(self):
        n = sum(c.nbytes + r.nbytes for st in self.checkpoints.values() for c, r in st.values())
        if self.state is not None and self.state.patcher is not None:
            n += sum(t.nbytes for t in self.state.patcher.model.state_dict().values())
        return n


def _model_key(model, dtype, device):
    from .qwen35 import GatedAttention
    attn = next(m for m in model.modules() if isinstance(m, GatedAttention))
    return f"{getattr(model, 'model_type', type(model).__name__)}|{dtype}|{attn.kv_cache_type(device, dtype).__name__}"


def _int8_window(kvm):
    from .llama import Int8FixedKVCache
    kv = next((kv for kv in kvm.caches if isinstance(kv, Int8FixedKVCache)), None)
    return None if kv is None else (kv.key.shape[2], kv.packed.key.shape[3])


def _resumable(entry, pos):
    """An int8 cache refreshes its pages from the bf16 window: the two pages before pos must still be in it."""
    if entry.window is None:
        return True
    rows, page = entry.window
    return entry.written - rows <= max(0, (pos - 1) // page * page - page)


def _lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _best(key, ids):
    """(entry, checkpoint position) with the most reusable tokens, or (None, 0)."""
    best, best_pos = None, 0
    limit_all = len(ids) - MIN_SUFFIX
    for e in _entries:
        if e.key != key:
            continue
        lcp = _lcp(ids, e.ids)
        usable = [p for p in e.checkpoints if p <= min(lcp, limit_all) and _resumable(e, p)]
        if usable and max(usable) > best_pos:
            best, best_pos = e, max(usable)
    return best, best_pos


@contextlib.contextmanager
def _cast(model, kvs, device):
    """Device views of these caches' tensors outside a forward (their storage streams through a prefetch queue)."""
    import comfy.model_prefetch
    kvs = [kv for kv in kvs if kv.storage is not None]
    if not kvs:
        yield
        return
    storages = tuple(kv.storage for kv in kvs)
    queue = comfy.model_prefetch.make_prefetch_queue([storages], device, {"prefetch_dynamic_vbars": model.model.prefetch_dynamic_vbars})
    with contextlib.ExitStack() as casts:
        comfy.model_prefetch.prefetch_queue_pop(queue, device, storages, core=lambda: [casts.enter_context(kv.cast(device)) for kv in kvs])
        yield
    comfy.model_prefetch.prefetch_queue_pop(queue, device, None)


def _linear(cache):
    from .qwen35 import LinearKV
    return {i: kv for i, kv in enumerate(cache.caches) if isinstance(kv, LinearKV)}


def _snapshot(model, cache, device):
    lin = _linear(cache)
    with _cast(model, lin.values(), device):
        out = {i: (kv.conv_state.to("cpu", copy=True), kv.recurrent_state.to("cpu", copy=True)) for i, kv in lin.items()}
    return out


def _restore_linear(model, cache, states, device):
    lin = _linear(cache)
    with _cast(model, lin.values(), device):
        for i, (c, r) in states.items():
            lin[i].conv_state.copy_(c)
            lin[i].recurrent_state.copy_(r)


def _set_index(cache, n):
    """Every layer at index n with no step in flight: slots from n on are masked by the next prepare()."""
    from .llama import FixedKVState
    from .qwen35 import LinearKV
    for kv in cache.caches:
        kv.index = n
        if isinstance(kv, FixedKVState):
            kv.tracker["step"] = -1
            kv.tracker["tree"] = None
        elif isinstance(kv, LinearKV):
            kv.uncommitted = 0
            kv.ctl_tracker.update(step=None, path=None, tree=None)
    for kv in cache.extra_caches:  # the MTP layer's cache starts empty on every call, as on a fresh generate
        kv.index = 0
        if isinstance(kv, FixedKVState):
            kv.tracker["step"] = -1
            kv.tracker["tree"] = None


def _capacity(need):
    return (need + HEADROOM + ROUND - 1) // ROUND * ROUND


def _padded(tensors, model_sd):
    """CPU tensors shaped like model_sd: equal shapes pass through, attention caches (capacity on dim 2, int8 pages on
    dim 1) are copied into zeros of the new capacity, cut to it if longer."""
    out = {}
    for name, like in model_sd.items():
        t = tensors.get(name)
        if t is None:
            out[name] = torch.zeros(like.shape, dtype=like.dtype)
        elif tuple(t.shape) == tuple(like.shape):
            out[name] = t
        else:
            z = torch.zeros(like.shape, dtype=like.dtype)
            common = tuple(slice(0, min(a, b)) for a, b in zip(t.shape, like.shape))
            z[common] = t[common]
            out[name] = z
    return out


def _in_use(patcher, flag):
    # only a dynamic-VRAM patcher pins for the current prompt (none on a CPU device)
    if patcher.is_dynamic():
        patcher.set_in_use_by_current_prompt(flag)


def _load(model, capacity, device, dtype, tensors=None):
    """A loaded ModelState holding a fresh KV cache model of this capacity, filled from tensors (CPU, any capacity)."""
    import comfy.model_state
    kvm = model.init_kv_cache(1, capacity, device, dtype, mtp=True)
    state = comfy.model_state.ModelState()
    if tensors:
        state.tensors = _padded(tensors, kvm.state_dict())
    state.load(kvm, device)
    _in_use(state.patcher, True)
    return state, kvm


def _activate(entry, device):
    import comfy.model_management
    comfy.model_management.load_models_gpu([entry.state.patcher])
    _in_use(entry.state.patcher, True)
    return entry.state.patcher.model


def _attention_layout(kvm):
    """{state-dict name: (dim, rows per slot)} of the attention caches, the MTP layer's included: capacity on dim 2 (an
    int8 cache's bf16 window too: row r sits in slot r % window), an int8 cache's pages on dim 1."""
    from .qwen35 import LinearKV
    out = {}
    for i, kv in enumerate(kvm.caches + kvm.extra_caches):
        if isinstance(kv, LinearKV):
            continue
        for key in kv.state_tensors():
            out[f"layers.{i}.{key}.weight"] = (1, kv.packed.key.shape[3]) if key.startswith("packed_") else (2, 1)
    return out


def _trimmed(t, n, layout):
    dim, unit = layout
    return t.narrow(dim, 0, min(t.shape[dim], -(-n // unit))).contiguous()


def _trimmed_tensors(state_tensors, n, attention):
    """A state's CPU tensors with the attention caches cut to the n tokens used (the rest is masked anyway)."""
    return {name: _trimmed(t, n, attention[name]) if name in attention else t for name, t in state_tensors.items()}


def _file_tensors(entry_tensors, checkpoints):
    t = dict(entry_tensors)
    for pos, st in checkpoints.items():
        for i, (c, r) in st.items():
            t[f"ckpt.{pos}.{i}.conv"] = c.contiguous()
            t[f"ckpt.{pos}.{i}.rec"] = r.contiguous()
    return t


def _write(path, tensors, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    safetensors.torch.save_file(tensors, tmp, {"llm_state_cache": json.dumps(meta)})
    os.replace(tmp, path)


def _read(path):
    with safetensors.safe_open(path, "pt") as f:
        meta = json.loads(f.metadata()["llm_state_cache"])
        tensors, cps = {}, {}
        for name in f.keys():
            if name.startswith("ckpt."):
                _, pos, i, kind = name.split(".")
                cps.setdefault(int(pos), {}).setdefault(int(i), [None, None])[0 if kind == "conv" else 1] = f.get_tensor(name)
            else:
                tensors[name] = f.get_tensor(name)
    return tensors, {p: {i: tuple(v) for i, v in st.items()} for p, st in cps.items()}, meta


def _ids_json(ids):
    return [list(t) if isinstance(t, tuple) else t for t in ids]


def _prefix_path(key, ids):
    h = hashlib.blake2b(json.dumps([key, _ids_json(ids)]).encode(), digest_size=16).hexdigest()
    return os.path.join(CACHE_DIR, f"prefix-{h}.safetensors")


def _spill(entry):
    """Save a resident entry to a file and free its state; its next match loads the file."""
    t = time.perf_counter()
    state = entry.state
    attention = _attention_layout(state.patcher.model)
    state.flush()
    n = len(entry.ids)
    path = os.path.join(CACHE_DIR, f"spill-{os.getpid()}-{id(entry)}.safetensors")
    _write(path, _file_tensors(_trimmed_tensors(state.tensors, n, attention), entry.checkpoints),
           {"key": entry.key, "ids": _ids_json(entry.ids)})
    entry.state = None
    entry.file = path
    entry.checkpoints = {p: {} for p in entry.checkpoints}  # positions kept for matching; the states are in the file
    logging.info("llm state cache: spilled %d tokens to disk in %.2f s", n, time.perf_counter() - t)


def _drop(entry):
    if entry.state is not None:
        entry.state.release()
    if entry.file is not None:
        try:
            os.remove(entry.file)
        except OSError:
            pass
    _entries.remove(entry)


def _enforce_budget(keep):
    resident = [e for e in _entries if e.state is not None]
    resident.sort(key=lambda e: e.used)
    total = sum(e.nbytes() for e in resident)
    for e in resident:
        if e is keep:
            continue
        if total <= BUDGET and len(resident) <= MAX_ENTRIES:
            break
        total -= e.nbytes()
        _spill(e)
        resident = [r for r in resident if r is not e]
    spilled = [e for e in _entries if e.state is None]
    spilled.sort(key=lambda e: e.used)
    while len(spilled) > MAX_ENTRIES:  # spilled files are bounded too
        _drop(spilled.pop(0))


def _take(model, key, ids, need, device, dtype):
    """(entry, cache model, prefix length) for this prompt, allocating or loading as needed."""
    entry, n = _best(key, ids)
    start = _shared_start(ids)
    path = _prefix_path(key, ids[:start]) if start else None
    if entry is not None and n <= start < len(entry.ids) and path is not None and os.path.exists(path):
        entry, n = None, 0  # another conversation: fork from the shared-start file, keep its state whole
        stats["fork"] = True
    if entry is not None:
        _entries.remove(entry)
        if entry.state is None:
            tensors, cps, _ = _read(entry.file)
            os.remove(entry.file)
            entry.file = None
            cap = max(_capacity(need), entry.capacity)
            entry.state, kvm = _load(model, cap, device, dtype, tensors)
            entry.capacity = cap
            entry.checkpoints = {p: cps[p] for p in cps if p in entry.checkpoints}
            stats["restore"] = "from-disk"
        elif entry.capacity < need:
            attention = _attention_layout(entry.state.patcher.model)
            entry.state.flush()
            cap = _capacity(need)
            tensors = _trimmed_tensors(entry.state.tensors, n, attention)
            entry.state, kvm = _load(model, cap, device, dtype, tensors)
            entry.capacity = cap
            stats["restore"] = "grow"
        else:
            kvm = _activate(entry, device)
            stats["restore"] = "resident"
        _restore_linear(model, kvm, entry.checkpoints[n], device)
        _set_index(kvm, n)
        entry.checkpoints = {p: st for p, st in entry.checkpoints.items() if p <= n}
        entry.ids = list(ids[:n])
        return entry, kvm, n
    cap = _capacity(need)
    if path is not None and os.path.exists(path):
        tensors, cps, _ = _read(path)
        state, kvm = _load(model, cap, device, dtype, tensors)
        entry = _Entry(key, state, cap)
        entry.written = start
        entry.checkpoints = {start: cps[start]}
        _restore_linear(model, kvm, cps[start], device)
        _set_index(kvm, start)
        entry.ids = list(ids[:start])
        stats["restore"] = "shared-start-file"
        return entry, kvm, start
    state, kvm = _load(model, cap, device, dtype)
    return _Entry(key, state, cap), kvm, 0


def _boundaries(ids):
    return [j for j in range(2, len(ids) + 1) if ids[j - 2] == MSG_END[0] and ids[j - 1] == MSG_END[1]]


def _shared_start(ids):
    """The end of the system message (the first message boundary), where a conversation's shared start ends."""
    ends = _boundaries(ids)
    if not ends or ends[0] < MIN_SUFFIX or len(ids) - ends[0] < MIN_SUFFIX:
        return 0
    return ends[0]


def _checkpoint_positions(ids, start):
    ends = _boundaries(ids)
    want = sorted(set(ends[:1] + ends[-CKPT_RECENT:])) if CKPT_RECENT > 0 else []
    out, prev = [], start
    for pos in want:
        if pos - prev >= MIN_SUFFIX and len(ids) - pos >= MIN_SUFFIX:
            out.append(pos)
            prev = pos
    return out


def prompt_ids(tokens, embeds_info, seq_len):
    """Cache ids for one prompt, one per embed position: the token ids, and for each image span a key per position
    made from a hash of the image. None when the prompt holds anything else (video, audio, raw embeddings) or the
    spans don't line up."""
    spans = iter(embeds_info)
    out = []
    for t in tokens:
        if isinstance(t, int):
            out.append(t)
            continue
        if not isinstance(t, dict) or t.get("type") != "image" or not torch.is_tensor(t.get("data")):
            return None
        info = next(spans, None)
        if info is None or info.get("type") != "image" or info.get("index") != len(out):
            return None
        digest = _image_digest(t["data"])
        out.extend((digest, k) for k in range(info["size"]))
    if next(spans, None) is not None or len(out) != seq_len:
        return None
    return out


def _image_digest(image):
    data = image.detach().to("cpu").contiguous()
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{tuple(data.shape)}|{data.dtype}".encode())
    h.update(data.view(torch.uint8).numpy().tobytes() if data.numel() else b"")
    return "img:" + h.hexdigest()


_current = {}


def prefill(model, embeds, ids, need, forward, position_ids=None):
    """Prefill the whole prompt `embeds`, reusing a cached prefix of `ids`. Returns (cache model, x of the last position, capacity).
    `forward(embeds, caches, position_ids)` runs one segment."""
    if _current:
        abort()  # the previous generate failed after its prefill
    stats.clear()
    device, dtype = embeds.device, embeds.dtype
    key = _model_key(model, dtype, device)
    entry, kvm, n = _take(model, key, ids, need, device, dtype)
    stats.update(prompt_len=len(ids), prefix_len=n)
    start, x = n, None
    shared = _shared_start(ids)
    for pos in _checkpoint_positions(ids, n) + [len(ids)]:
        x = forward(embeds[:, start:pos], kvm.caches, None if position_ids is None else position_ids[..., start:pos])
        entry.checkpoints[pos] = _snapshot(model, kvm, device)
        if pos == shared and n < shared:
            _save_shared_start(model, kvm, key, ids, shared, entry.checkpoints[pos], device)
        start = pos
    if len(entry.checkpoints) > CKPT_RECENT + 3:
        keep = sorted(entry.checkpoints)
        keep = keep[:1] + keep[-(CKPT_RECENT + 2):]
        entry.checkpoints = {p: entry.checkpoints[p] for p in keep}
    entry.ids = list(ids)
    entry.written = len(ids)
    _current.update(entry=entry, kvm=kvm, device=device, model=model)
    return kvm, x, entry.capacity


def _save_shared_start(model, kvm, key, ids, pos, linear_states, device):
    """Write the shared start (KV up to pos + the DeltaNet states at pos) once, if no file has it."""
    path = _prefix_path(key, ids[:pos])
    if os.path.exists(path):
        return
    linear = _linear(kvm)
    attn = {i: kv for i, kv in enumerate(kvm.caches) if i not in linear}
    layout = _attention_layout(kvm)
    tensors = {}
    with _cast(model, attn.values(), device):
        for i, kv in attn.items():
            for key, (owner, field) in kv.state_tensors().items():
                name = f"layers.{i}.{key}.weight"
                tensors[name] = _trimmed(getattr(owner, field), pos, layout[name]).to("cpu", copy=True)
    _write(path, _file_tensors(tensors, {pos: linear_states}), {"key": key, "ids": _ids_json(ids[:pos])})
    stats["shared_start_saved"] = True


def finish(ids, generated):
    """Checkpoint the prompt plus the generated tokens the DeltaNet state has committed, keep the state resident, and
    spill others past the budget."""
    from . import llama
    entry, kvm, device, model = (_current.get(k) for k in ("entry", "kvm", "device", "model"))
    _current.clear()
    if entry is None:
        return
    stats["gen_tokens"] = len(generated)
    full = list(ids) + list(generated)
    lin = next(iter(_linear(kvm).values()))
    # deferred commit: a decode step leaves the DeltaNet state untouched until the next step, so the state lags the
    # index by the last step's still-uncommitted tokens
    idx = lin.index - lin.uncommitted
    if len(ids) < idx <= len(full):
        entry.checkpoints[idx] = _snapshot(model, kvm, device)
        entry.ids = full[:idx]
    entry.written = max(kv.index for kv in kvm.caches) + llama.SPEC_ROWS  # a verify step writes up to SPEC_ROWS rows ahead
    _in_use(entry.state.patcher, False)
    _tick[0] += 1
    entry.used = _tick[0]
    _entries.append(entry)
    _enforce_budget(entry)
    stats["entries"] = len(_entries)
    stats["resident"] = sum(1 for e in _entries if e.state is not None)
    stats["cache_gb"] = round(sum(e.nbytes() for e in _entries if e.state is not None) / 1e9, 3)
    logging.info("LLM_PREFIX_CACHE %s", json.dumps(stats))


def abort():
    """A generate failed after prefill: drop the half-written state."""
    entry = _current.get("entry")
    _current.clear()
    if entry is not None and entry.state is not None:
        entry.state.release()
