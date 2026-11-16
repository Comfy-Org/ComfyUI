"""Two-device sequence parallelism with replicas owned by model management."""

from contextlib import contextmanager
import ctypes
from functools import partial
import logging
import sys

import torch

import comfy.model_management as mm
import comfy.model_patcher
from comfy.ldm.minimax.model import MiniMaxH3Model, _mod_gate, _mod_scale_shift


@contextmanager
def sampling_memory_policy(source, target):
    if (sys.platform != "linux" or torch.version.hip is None
            or any(torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] != "gfx1201" for device in (source, target))):
        yield
        return
    try:
        numa = ctypes.CDLL("libnuma.so.1", use_errno=True)
        numa.get_mempolicy.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong]
        numa.get_mempolicy.restype = ctypes.c_int
        numa.set_mempolicy.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong]
        numa.set_mempolicy.restype = ctypes.c_int
        mode = ctypes.c_int()
        if numa.get_mempolicy(ctypes.byref(mode), None, 0, None, 0):
            raise OSError(ctypes.get_errno(), "get_mempolicy failed")
    except OSError as error:
        logging.warning("MiniMax H3 NUMA binding unavailable (%s); using the existing memory policy", error)
        yield
        return
    if mode.value != 0:
        yield
        return
    try:
        with open("/proc/self/status") as status:
            allowed = next(line.split(":", 1)[1].strip() for line in status if line.startswith("Mems_allowed_list:"))
        nodes = []
        for entry in allowed.split(","):
            limits = [int(value) for value in entry.split("-")]
            nodes.extend(range(limits[0], limits[-1] + 1))
        bits = ctypes.sizeof(ctypes.c_ulong) * 8
        words = (max(nodes) + bits) // bits
        mask = (ctypes.c_ulong * words)()
        for node in nodes:
            mask[node // bits] |= 1 << (node % bits)
        # Binding without MPOL_F_NUMA_BALANCING avoids NUMA-driven SVM queue evictions.
        if numa.set_mempolicy(2, mask, words * bits):
            raise OSError(ctypes.get_errno(), "set_mempolicy failed")
    except OSError as error:
        logging.warning("MiniMax H3 NUMA binding unavailable (%s); using the existing memory policy", error)
        yield
        return
    try:
        yield
    finally:
        if numa.set_mempolicy(0, None, 0):
            raise OSError(ctypes.get_errno(), "Could not restore sampling memory policy")


def partition_segments(segments, start, end, device):
    result = []
    for a, b, row in segments:
        lo, hi = max(a, start), min(b, end)
        if lo >= hi:
            continue
        if isinstance(row, torch.Tensor):
            if row.ndim:
                row = row[lo - a:hi - a]
            row = row.to(device)
        result.append((lo - start, hi - start, row))
    return result


def adjacent_kv_rows(k, v):
    k, v = k.transpose(1, 2), v.transpose(1, 2)
    batch, rows, heads, width = k.shape
    if (k.dtype != v.dtype or k.stride() != v.stride() or k.stride(-1) != 1
            or k.stride(-2) != width
            or k.untyped_storage().data_ptr() != v.untyped_storage().data_ptr()
            or v.storage_offset() - k.storage_offset() != heads * width):
        return None
    # The inference projection retains adjacent K/V columns in its QKV buffer.
    return k.as_strided((batch, rows, 2, heads, width),
                        (k.stride(0), k.stride(1), heads * width, width, 1))


def gather_full_kv(qkvs, devices):
    packed = [adjacent_kv_rows(qkv[1], qkv[2]) for qkv in qkvs]
    full_kv = []
    for device in devices:
        with torch.cuda.device(device):
            if all(value is not None for value in packed):
                full = torch.cat([value.to(device, non_blocking=True) for value in packed], dim=1)
                full_kv.append(tuple(value.transpose(1, 2) for value in full.unbind(dim=2)))
            else:
                full_kv.append(tuple(torch.cat([qkv[j].transpose(1, 2).to(device, non_blocking=True) for qkv in qkvs], dim=1).transpose(1, 2) for j in (1, 2)))
    return full_kv


def parallel_block(block, replica, target, x, t_emb, segments, rope, options):
    source = x.device
    # Round the midpoint to a tile boundary without creating an empty partition.
    cut = min(max(((x.shape[0] + 127) // 256) * 128, 1), x.shape[0] - 1)
    with torch.cuda.device(target):
        xs = [x[:cut], x[cut:].to(target, non_blocking=True)]
        ts = [t_emb, t_emb.to(target, non_blocking=True)]
        ropes = [rope[:, :cut], rope[:, cut:].to(target, non_blocking=True)] if rope is not None else [None, None]
    local_segments = [partition_segments(segments, 0, cut, source),
                      partition_segments(segments, cut, x.shape[0], target)]
    blocks, devices = (block, replica), (source, target)
    mods, qkvs = [], []
    for i, device in enumerate(devices):
        with torch.cuda.device(device):
            shift, scale, gate, shift2, scale2, gate2 = blocks[i].adaln_proj(ts[i])
            mods.append((gate, shift2, scale2, gate2))
            h = _mod_scale_shift(blocks[i].norm1(xs[i]), shift, scale, local_segments[i])
            qkvs.append(blocks[i].attn.project_qkv(h, ropes[i]))
    full_kv = gather_full_kv(qkvs, devices)
    for i, device in enumerate(devices):
        with torch.cuda.device(device):
            b = blocks[i]
            gate, shift, scale, gate2 = mods[i]
            projected = b.attn.attend(qkvs[i][0], *full_kv[i], transformer_options=options)
            xs[i] = _mod_gate(xs[i], gate, projected, local_segments[i])
            h = _mod_scale_shift(b.norm2(xs[i]), shift, scale, local_segments[i])
            xs[i] = _mod_gate(xs[i], gate2, b.mlp(h), local_segments[i])
    with torch.cuda.device(source):
        x[cut:].copy_(xs[1], non_blocking=True)
    return x


def run_block(block, replica, target, patcher, args, extra):
    if patcher.hook_patches:
        raise ValueError("MiniMax H3 sequence parallelism does not support scheduled weight hooks")
    x = args["img"]
    if x.shape[0] < 2:
        return extra["original_block"](args)
    out = parallel_block(block, replica, target, x, args["t_emb"], args["mod_segments"], args["rope_freqs"], args["transformer_options"])
    return {"img": out}


class SequenceParallelSampling:
    def __init__(self, device_index):
        self.device_index = device_index

    def __call__(self, executor, *args, **kwargs):
        guider = executor.class_obj
        patcher = guider.model_patcher
        model = patcher.get_model_object("diffusion_model")
        if not isinstance(model, MiniMaxH3Model):
            raise ValueError("Sequence parallelism requires a MiniMax H3 model")
        if self.device_index < 0:
            raise ValueError("Select a second available GPU for MiniMax H3 sequence parallelism")
        source, target = patcher.load_device, torch.device("cuda", self.device_index)
        if mm.in_training or source.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("MiniMax H3 sequence parallelism requires CUDA/ROCm inference")
        if target.index >= torch.cuda.device_count() or target.index < 0 or source == target:
            raise ValueError("Select a second available GPU for MiniMax H3 sequence parallelism")
        if not torch.cuda.can_device_access_peer(source.index, target.index) or not torch.cuda.can_device_access_peer(target.index, source.index):
            raise ValueError("MiniMax H3 sequence parallelism requires peer access in both directions")
        options = guider.model_options
        transformer = options.get("transformer_options", {})
        if patcher.hook_patches or patcher.get_additional_models_with_key("multigpu") or transformer.get("patches") or transformer.get("patches_replace"):
            raise ValueError("MiniMax H3 sequence parallelism cannot be combined with scheduled hooks, model parallel clones or transformer patches")
        stored = patcher.get_additional_models_with_key("minimax_sequence_parallel")[0]
        secondary = patcher.clone(disable_dynamic=True, model_override=stored.get_clone_model_override())
        secondary.remove_additional_models("minimax_sequence_parallel")
        secondary.load_device = target
        patcher.set_additional_models("minimax_sequence_parallel", [secondary])
        replica_model = secondary.get_model_object("diffusion_model")
        guider.model_options = comfy.model_patcher.create_model_options_clone(options)
        replacements = guider.model_options["transformer_options"].setdefault("patches_replace", {}).setdefault("dit", {})
        for i, block in enumerate(model.blocks):
            replacements[("double_block", i)] = partial(run_block, block, replica_model.blocks[i], target, patcher)
        try:
            with sampling_memory_policy(source, target):
                return executor(*args, **kwargs)
        finally:
            guider.model_options = options
