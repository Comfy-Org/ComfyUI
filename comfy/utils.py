"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Comfy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import torch
import math
import struct
import ctypes
import os
import comfy.memory_management
import safetensors.torch
import numpy as np
from PIL import Image
import logging
import itertools
from torch.nn.functional import interpolate
from tqdm.auto import trange
from einops import rearrange
from comfy.cli_args import args
import json
import time
import threading
import warnings

MMAP_TORCH_FILES = args.mmap_torch_files
DISABLE_MMAP = args.disable_mmap


if True:  # ckpt/pt file whitelist for safe loading of old sd files
    class ModelCheckpoint:
        pass
    ModelCheckpoint.__module__ = "pytorch_lightning.callbacks.model_checkpoint"

    def scalar(*args, **kwargs):
        return None
    scalar.__module__ = "numpy.core.multiarray"

    from numpy import dtype
    from numpy.dtypes import Float64DType

    def encode(*args, **kwargs):  # no longer necessary on newer torch
        return None
    encode.__module__ = "_codecs"

    torch.serialization.add_safe_globals([ModelCheckpoint, scalar, dtype, Float64DType, encode])
    logging.info("Checkpoint files will always be loaded safely.")


# Current as of safetensors 0.7.0
_TYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "C64": torch.complex64,

    "U64": torch.uint64,
    "U32": torch.uint32,
    "U16": torch.uint16,
}

def load_safetensors(ckpt):
    import comfy_aimdo.model_mmap

    file_lock = threading.Lock()
    model_mmap = comfy_aimdo.model_mmap.ModelMMAP(ckpt)
    f = model_mmap.get_file_handle()
    file_size = os.path.getsize(ckpt)
    mv = memoryview((ctypes.c_uint8 * file_size).from_address(model_mmap.get()))

    header_size = struct.unpack("<Q", mv[:8])[0]
    header = json.loads(mv[8:8 + header_size].tobytes().decode("utf-8"))

    mv = mv[(data_base_offset := 8 + header_size):]

    sd = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue

        start, end = info["data_offsets"]
        if start == end:
            sd[name] = torch.empty(info["shape"], dtype =_TYPES[info["dtype"]])
        else:
            with warnings.catch_warnings():
                #We are working with read-only RAM by design
                warnings.filterwarnings("ignore", message="The given buffer is not writable")
                tensor = torch.frombuffer(mv[start:end], dtype=_TYPES[info["dtype"]]).view(info["shape"])
                storage = tensor.untyped_storage()
                setattr(storage,
                        "_comfy_tensor_file_slice",
                        comfy.memory_management.TensorFileSlice(f, file_lock, data_base_offset + start, end - start))
                setattr(storage, "_comfy_tensor_mmap_refs", (model_mmap, mv))
                sd[name] = tensor

    return sd, header.get("__metadata__", {}),


def load_torch_file(ckpt, safe_load=False, device=None, return_metadata=False):
    if device is None:
        device = torch.device("cpu")
    metadata = None
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        try:
            if comfy.memory_management.aimdo_enabled:
                sd, metadata = load_safetensors(ckpt)
                if not return_metadata:
                    metadata = None
            else:
                with safetensors.safe_open(ckpt, framework="pt", device=device.type) as f:
                    sd = {}
                    for k in f.keys():
                        tensor = f.get_tensor(k)
                        if DISABLE_MMAP:  # TODO: Not sure if this is the best way to bypass the mmap issues
                            tensor = tensor.to(device=device, copy=True)
                        sd[k] = tensor
                    if return_metadata:
                        metadata = f.metadata()
        except Exception as e:
            if len(e.args) > 0:
                message = e.args[0]
                if "HeaderTooLarge" in message:
                    raise ValueError("{}\n\nFile path: {}\n\nThe safetensors file is corrupt or invalid. Make sure this is actually a safetensors file and not a ckpt or pt or other filetype.".format(message, ckpt))
                if "MetadataIncompleteBuffer" in message:
                    raise ValueError("{}\n\nFile path: {}\n\nThe safetensors file is corrupt/incomplete. Check the file size and make sure you have copied/downloaded it correctly.".format(message, ckpt))
            raise e
    else:
        torch_args = {}
        if MMAP_TORCH_FILES:
            torch_args["mmap"] = True

        pl_sd = torch.load(ckpt, map_location=device, weights_only=True, **torch_args)

        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            if len(pl_sd) == 1:
                key = list(pl_sd.keys())[0]
                sd = pl_sd[key]
                if not isinstance(sd, dict):
                    sd = pl_sd
            else:
                sd = pl_sd
    return (sd, metadata) if return_metadata else sd

def save_torch_file(sd, ckpt, metadata=None):
    if metadata is not None:
        safetensors.torch.save_file(sd, ckpt, metadata=metadata)
    else:
        safetensors.torch.save_file(sd, ckpt)

def calculate_parameters(sd, prefix=""):
    params = 0
    for k in sd.keys():
        if k.startswith(prefix):
            w = sd[k]
            params += w.nelement()
    return params

def weight_dtype(sd, prefix=""):
    dtypes = {}
    for k in sd.keys():
        if k.startswith(prefix):
            w = sd[k]
            dtypes[w.dtype] = dtypes.get(w.dtype, 0) + w.numel()

    if len(dtypes) == 0:
        return None

    return max(dtypes, key=dtypes.get)

def state_dict_key_replace(state_dict, keys_to_replace):
    for x in keys_to_replace:
        if x in state_dict:
            state_dict[keys_to_replace[x]] = state_dict.pop(x)
    return state_dict

def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp):])), filter(lambda a: a.startswith(rp), state_dict.keys())))
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


def transformers_convert(sd, prefix_from, prefix_to, number):
    keys_to_replace = {
        "{}positional_embedding": "{}embeddings.position_embedding.weight",
        "{}token_embedding.weight": "{}embeddings.token_embedding.weight",
        "{}ln_final.weight": "{}final_layer_norm.weight",
        "{}ln_final.bias": "{}final_layer_norm.bias",
    }

    for k in keys_to_replace:
        x = k.format(prefix_from)
        if x in sd:
            sd[keys_to_replace[k].format(prefix_to)] = sd.pop(x)

    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    for resblock in range(number):
        for x in resblock_to_replace:
            for y in ["weight", "bias"]:
                k = "{}transformer.resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                if k in sd:
                    sd[k_to] = sd.pop(k)

        for y in ["weight", "bias"]:
            k_from = "{}transformer.resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
            if k_from in sd:
                weights = sd.pop(k_from)
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                    sd[k_to] = weights[shape_from*x:shape_from*(x + 1)]

    return sd

def clip_text_transformers_convert(sd, prefix_from, prefix_to):
    sd = transformers_convert(sd, prefix_from, "{}text_model.".format(prefix_to), 32)

    tp = "{}text_projection.weight".format(prefix_from)
    if tp in sd:
        sd["{}text_projection.weight".format(prefix_to)] = sd.pop(tp)

    tp = "{}text_projection".format(prefix_from)
    if tp in sd:
        sd["{}text_projection.weight".format(prefix_to)] = sd.pop(tp).transpose(0, 1).contiguous()
    return sd


UNET_MAP_ATTENTIONS = {
    "proj_in.weight",
    "proj_in.bias",
    "proj_out.weight",
    "proj_out.bias",
    "norm.weight",
    "norm.bias",
}

TRANSFORMER_BLOCKS = {
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "norm3.weight",
    "norm3.bias",
    "attn1.to_q.weight",
    "attn1.to_k.weight",
    "attn1.to_v.weight",
    "attn1.to_out.0.weight",
    "attn1.to_out.0.bias",
    "attn2.to_q.weight",
    "attn2.to_k.weight",
    "attn2.to_v.weight",
    "attn2.to_out.0.weight",
    "attn2.to_out.0.bias",
    "ff.net.0.proj.weight",
    "ff.net.0.proj.bias",
    "ff.net.2.weight",
    "ff.net.2.bias",
}

UNET_MAP_RESNET = {
    "in_layers.2.weight": "conv1.weight",
    "in_layers.2.bias": "conv1.bias",
    "emb_layers.1.weight": "time_emb_proj.weight",
    "emb_layers.1.bias": "time_emb_proj.bias",
    "out_layers.3.weight": "conv2.weight",
    "out_layers.3.bias": "conv2.bias",
    "skip_connection.weight": "conv_shortcut.weight",
    "skip_connection.bias": "conv_shortcut.bias",
    "in_layers.0.weight": "norm1.weight",
    "in_layers.0.bias": "norm1.bias",
    "out_layers.0.weight": "norm2.weight",
    "out_layers.0.bias": "norm2.bias",
}

UNET_MAP_BASIC = {
    ("label_emb.0.0.weight", "class_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "class_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "class_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "class_embedding.linear_2.bias"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias")
}

def unet_to_diffusers(unet_config):
    if "num_res_blocks" not in unet_config:
        return {}
    num_res_blocks = unet_config["num_res_blocks"]
    channel_mult = unet_config["channel_mult"]
    transformer_depth = unet_config["transformer_depth"][:]
    transformer_depth_output = unet_config["transformer_depth_output"][:]
    num_blocks = len(channel_mult)

    transformers_mid = unet_config.get("transformer_depth_middle", None)

    diffusers_unet_map = {}
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["down_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "input_blocks.{}.0.{}".format(n, b)
            num_transformers = transformer_depth.pop(0)
            if num_transformers > 0:
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["down_blocks.{}.attentions.{}.{}".format(x, i, b)] = "input_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["down_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "input_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            n += 1
        for k in ["weight", "bias"]:
            diffusers_unet_map["down_blocks.{}.downsamplers.0.conv.{}".format(x, k)] = "input_blocks.{}.0.op.{}".format(n, k)

    i = 0
    for b in UNET_MAP_ATTENTIONS:
        diffusers_unet_map["mid_block.attentions.{}.{}".format(i, b)] = "middle_block.1.{}".format(b)
    for t in range(transformers_mid):
        for b in TRANSFORMER_BLOCKS:
            diffusers_unet_map["mid_block.attentions.{}.transformer_blocks.{}.{}".format(i, t, b)] = "middle_block.1.transformer_blocks.{}.{}".format(t, b)

    for i, n in enumerate([0, 2]):
        for b in UNET_MAP_RESNET:
            diffusers_unet_map["mid_block.resnets.{}.{}".format(i, UNET_MAP_RESNET[b])] = "middle_block.{}.{}".format(n, b)

    num_res_blocks = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks[x] + 1) * x
        l = num_res_blocks[x] + 1
        for i in range(l):
            c = 0
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "output_blocks.{}.0.{}".format(n, b)
            c += 1
            num_transformers = transformer_depth_output.pop()
            if num_transformers > 0:
                c += 1
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["up_blocks.{}.attentions.{}.{}".format(x, i, b)] = "output_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["up_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "output_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            if i == l - 1:
                for k in ["weight", "bias"]:
                    diffusers_unet_map["up_blocks.{}.upsamplers.0.conv.{}".format(x, k)] = "output_blocks.{}.{}.conv.{}".format(n, c, k)
            n += 1

    for k in UNET_MAP_BASIC:
        diffusers_unet_map[k[1]] = k[0]

    return diffusers_unet_map

def swap_scale_shift(weight):
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight

MMDIT_MAP_BASIC = {
    ("context_embedder.bias", "context_embedder.bias"),
    ("context_embedder.weight", "context_embedder.weight"),
    ("t_embedder.mlp.0.bias", "time_text_embed.timestep_embedder.linear_1.bias"),
    ("t_embedder.mlp.0.weight", "time_text_embed.timestep_embedder.linear_1.weight"),
    ("t_embedder.mlp.2.bias", "time_text_embed.timestep_embedder.linear_2.bias"),
    ("t_embedder.mlp.2.weight", "time_text_embed.timestep_embedder.linear_2.weight"),
    ("x_embedder.proj.bias", "pos_embed.proj.bias"),
    ("x_embedder.proj.weight", "pos_embed.proj.weight"),
    ("y_embedder.mlp.0.bias", "time_text_embed.text_embedder.linear_1.bias"),
    ("y_embedder.mlp.0.weight", "time_text_embed.text_embedder.linear_1.weight"),
    ("y_embedder.mlp.2.bias", "time_text_embed.text_embedder.linear_2.bias"),
    ("y_embedder.mlp.2.weight", "time_text_embed.text_embedder.linear_2.weight"),
    ("pos_embed", "pos_embed.pos_embed"),
    ("final_layer.adaLN_modulation.1.bias", "norm_out.linear.bias", swap_scale_shift),
    ("final_layer.adaLN_modulation.1.weight", "norm_out.linear.weight", swap_scale_shift),
    ("final_layer.linear.bias", "proj_out.bias"),
    ("final_layer.linear.weight", "proj_out.weight"),
}

MMDIT_MAP_BLOCK = {
    ("context_block.adaLN_modulation.1.bias", "norm1_context.linear.bias"),
    ("context_block.adaLN_modulation.1.weight", "norm1_context.linear.weight"),
    ("context_block.attn.proj.bias", "attn.to_add_out.bias"),
    ("context_block.attn.proj.weight", "attn.to_add_out.weight"),
    ("context_block.mlp.fc1.bias", "ff_context.net.0.proj.bias"),
    ("context_block.mlp.fc1.weight", "ff_context.net.0.proj.weight"),
    ("context_block.mlp.fc2.bias", "ff_context.net.2.bias"),
    ("context_block.mlp.fc2.weight", "ff_context.net.2.weight"),
    ("context_block.attn.ln_q.weight", "attn.norm_added_q.weight"),
    ("context_block.attn.ln_k.weight", "attn.norm_added_k.weight"),
    ("x_block.adaLN_modulation.1.bias", "norm1.linear.bias"),
    ("x_block.adaLN_modulation.1.weight", "norm1.linear.weight"),
    ("x_block.attn.proj.bias", "attn.to_out.0.bias"),
    ("x_block.attn.proj.weight", "attn.to_out.0.weight"),
    ("x_block.attn.ln_q.weight", "attn.norm_q.weight"),
    ("x_block.attn.ln_k.weight", "attn.norm_k.weight"),
    ("x_block.attn2.proj.bias", "attn2.to_out.0.bias"),
    ("x_block.attn2.proj.weight", "attn2.to_out.0.weight"),
    ("x_block.attn2.ln_q.weight", "attn2.norm_q.weight"),
    ("x_block.attn2.ln_k.weight", "attn2.norm_k.weight"),
    ("x_block.mlp.fc1.bias", "ff.net.0.proj.bias"),
    ("x_block.mlp.fc1.weight", "ff.net.0.proj.weight"),
    ("x_block.mlp.fc2.bias", "ff.net.2.bias"),
    ("x_block.mlp.fc2.weight", "ff.net.2.weight"),
}

def mmdit_to_diffusers(mmdit_config, output_prefix=""):
    key_map = {}

    depth = mmdit_config.get("depth", 0)
    num_blocks = mmdit_config.get("num_blocks", depth)
    for i in range(num_blocks):
        block_from = "transformer_blocks.{}".format(i)
        block_to = "{}joint_blocks.{}".format(output_prefix, i)

        offset = depth * 64

        for end in ("weight", "bias"):
            k = "{}.attn.".format(block_from)
            qkv = "{}.x_block.attn.qkv.{}".format(block_to, end)
            key_map["{}to_q.{}".format(k, end)] = (qkv, (0, 0, offset))
            key_map["{}to_k.{}".format(k, end)] = (qkv, (0, offset, offset))
            key_map["{}to_v.{}".format(k, end)] = (qkv, (0, offset * 2, offset))

            qkv = "{}.context_block.attn.qkv.{}".format(block_to, end)
            key_map["{}add_q_proj.{}".format(k, end)] = (qkv, (0, 0, offset))
            key_map["{}add_k_proj.{}".format(k, end)] = (qkv, (0, offset, offset))
            key_map["{}add_v_proj.{}".format(k, end)] = (qkv, (0, offset * 2, offset))

            k = "{}.attn2.".format(block_from)
            qkv = "{}.x_block.attn2.qkv.{}".format(block_to, end)
            key_map["{}to_q.{}".format(k, end)] = (qkv, (0, 0, offset))
            key_map["{}to_k.{}".format(k, end)] = (qkv, (0, offset, offset))
            key_map["{}to_v.{}".format(k, end)] = (qkv, (0, offset * 2, offset))

        for k in MMDIT_MAP_BLOCK:
            key_map["{}.{}".format(block_from, k[1])] = "{}.{}".format(block_to, k[0])

    map_basic = MMDIT_MAP_BASIC.copy()
    map_basic.add(("joint_blocks.{}.context_block.adaLN_modulation.1.bias".format(depth - 1), "transformer_blocks.{}.norm1_context.linear.bias".format(depth - 1), swap_scale_shift))
    map_basic.add(("joint_blocks.{}.context_block.adaLN_modulation.1.weight".format(depth - 1), "transformer_blocks.{}.norm1_context.linear.weight".format(depth - 1), swap_scale_shift))

    for k in map_basic:
        if len(k) > 2:
            key_map[k[1]] = ("{}{}".format(output_prefix, k[0]), None, k[2])
        else:
            key_map[k[1]] = "{}{}".format(output_prefix, k[0])

    return key_map

PIXART_MAP_BASIC = {
    ("csize_embedder.mlp.0.weight", "adaln_single.emb.resolution_embedder.linear_1.weight"),
    ("csize_embedder.mlp.0.bias", "adaln_single.emb.resolution_embedder.linear_1.bias"),
    ("csize_embedder.mlp.2.weight", "adaln_single.emb.resolution_embedder.linear_2.weight"),
    ("csize_embedder.mlp.2.bias", "adaln_single.emb.resolution_embedder.linear_2.bias"),
    ("ar_embedder.mlp.0.weight", "adaln_single.emb.aspect_ratio_embedder.linear_1.weight"),
    ("ar_embedder.mlp.0.bias", "adaln_single.emb.aspect_ratio_embedder.linear_1.bias"),
    ("ar_embedder.mlp.2.weight", "adaln_single.emb.aspect_ratio_embedder.linear_2.weight"),
    ("ar_embedder.mlp.2.bias", "adaln_single.emb.aspect_ratio_embedder.linear_2.bias"),
    ("x_embedder.proj.weight", "pos_embed.proj.weight"),
    ("x_embedder.proj.bias", "pos_embed.proj.bias"),
    ("y_embedder.y_embedding", "caption_projection.y_embedding"),
    ("y_embedder.y_proj.fc1.weight", "caption_projection.linear_1.weight"),
    ("y_embedder.y_proj.fc1.bias", "caption_projection.linear_1.bias"),
    ("y_embedder.y_proj.fc2.weight", "caption_projection.linear_2.weight"),
    ("y_embedder.y_proj.fc2.bias", "caption_projection.linear_2.bias"),
    ("t_embedder.mlp.0.weight", "adaln_single.emb.timestep_embedder.linear_1.weight"),
    ("t_embedder.mlp.0.bias", "adaln_single.emb.timestep_embedder.linear_1.bias"),
    ("t_embedder.mlp.2.weight", "adaln_single.emb.timestep_embedder.linear_2.weight"),
    ("t_embedder.mlp.2.bias", "adaln_single.emb.timestep_embedder.linear_2.bias"),
    ("t_block.1.weight", "adaln_single.linear.weight"),
    ("t_block.1.bias", "adaln_single.linear.bias"),
    ("final_layer.linear.weight", "proj_out.weight"),
    ("final_layer.linear.bias", "proj_out.bias"),
    ("final_layer.scale_shift_table", "scale_shift_table"),
}

PIXART_MAP_BLOCK = {
    ("scale_shift_table", "scale_shift_table"),
    ("attn.proj.weight", "attn1.to_out.0.weight"),
    ("attn.proj.bias", "attn1.to_out.0.bias"),
    ("mlp.fc1.weight", "ff.net.0.proj.weight"),
    ("mlp.fc1.bias", "ff.net.0.proj.bias"),
    ("mlp.fc2.weight", "ff.net.2.weight"),
    ("mlp.fc2.bias", "ff.net.2.bias"),
    ("cross_attn.proj.weight" ,"attn2.to_out.0.weight"),
    ("cross_attn.proj.bias"   ,"attn2.to_out.0.bias"),
}

def pixart_to_diffusers(mmdit_config, output_prefix=""):
    key_map = {}

    depth = mmdit_config.get("depth", 0)
    offset = mmdit_config.get("hidden_size", 1152)

    for i in range(depth):
        block_from = "transformer_blocks.{}".format(i)
        block_to = "{}blocks.{}".format(output_prefix, i)

        for end in ("weight", "bias"):
            s = "{}.attn1.".format(block_from)
            qkv = "{}.attn.qkv.{}".format(block_to, end)
            key_map["{}to_q.{}".format(s, end)] = (qkv, (0, 0, offset))
            key_map["{}to_k.{}".format(s, end)] = (qkv, (0, offset, offset))
            key_map["{}to_v.{}".format(s, end)] = (qkv, (0, offset * 2, offset))

            s = "{}.attn2.".format(block_from)
            q = "{}.cross_attn.q_linear.{}".format(block_to, end)
            kv = "{}.cross_attn.kv_linear.{}".format(block_to, end)

            key_map["{}to_q.{}".format(s, end)] = q
            key_map["{}to_k.{}".format(s, end)] = (kv, (0, 0, offset))
            key_map["{}to_v.{}".format(s, end)] = (kv, (0, offset, offset))

        for k in PIXART_MAP_BLOCK:
            key_map["{}.{}".format(block_from, k[1])] = "{}.{}".format(block_to, k[0])

    for k in PIXART_MAP_BASIC:
        key_map[k[1]] = "{}{}".format(output_prefix, k[0])

    return key_map

def auraflow_to_diffusers(mmdit_config, output_prefix=""):
    n_double_layers = mmdit_config.get("n_double_layers", 0)
    n_layers = mmdit_config.get("n_layers", 0)

    key_map = {}
    for i in range(n_layers):
        if i < n_double_layers:
            index = i
            prefix_from = "joint_transformer_blocks"
            prefix_to = "{}double_layers".format(output_prefix)
            block_map = {
                            "attn.to_q.weight": "attn.w2q.weight",
                            "attn.to_k.weight": "attn.w2k.weight",
                            "attn.to_v.weight": "attn.w2v.weight",
                            "attn.to_out.0.weight": "attn.w2o.weight",
                            "attn.add_q_proj.weight": "attn.w1q.weight",
                            "attn.add_k_proj.weight": "attn.w1k.weight",
                            "attn.add_v_proj.weight": "attn.w1v.weight",
                            "attn.to_add_out.weight": "attn.w1o.weight",
                            "ff.linear_1.weight": "mlpX.c_fc1.weight",
                            "ff.linear_2.weight": "mlpX.c_fc2.weight",
                            "ff.out_projection.weight": "mlpX.c_proj.weight",
                            "ff_context.linear_1.weight": "mlpC.c_fc1.weight",
                            "ff_context.linear_2.weight": "mlpC.c_fc2.weight",
                            "ff_context.out_projection.weight": "mlpC.c_proj.weight",
                            "norm1.linear.weight": "modX.1.weight",
                            "norm1_context.linear.weight": "modC.1.weight",
                        }
        else:
            index = i - n_double_layers
            prefix_from = "single_transformer_blocks"
            prefix_to = "{}single_layers".format(output_prefix)

            block_map = {
                            "attn.to_q.weight": "attn.w1q.weight",
                            "attn.to_k.weight": "attn.w1k.weight",
                            "attn.to_v.weight": "attn.w1v.weight",
                            "attn.to_out.0.weight": "attn.w1o.weight",
                            "norm1.linear.weight": "modCX.1.weight",
                            "ff.linear_1.weight": "mlp.c_fc1.weight",
                            "ff.linear_2.weight": "mlp.c_fc2.weight",
                            "ff.out_projection.weight": "mlp.c_proj.weight"
                        }

        for k in block_map:
            key_map["{}.{}.{}".format(prefix_from, index, k)] = "{}.{}.{}".format(prefix_to, index, block_map[k])

    MAP_BASIC = {
        ("positional_encoding", "pos_embed.pos_embed"),
        ("register_tokens", "register_tokens"),
        ("t_embedder.mlp.0.weight", "time_step_proj.linear_1.weight"),
        ("t_embedder.mlp.0.bias", "time_step_proj.linear_1.bias"),
        ("t_embedder.mlp.2.weight", "time_step_proj.linear_2.weight"),
        ("t_embedder.mlp.2.bias", "time_step_proj.linear_2.bias"),
        ("cond_seq_linear.weight", "context_embedder.weight"),
        ("init_x_linear.weight", "pos_embed.proj.weight"),
        ("init_x_linear.bias", "pos_embed.proj.bias"),
        ("final_linear.weight", "proj_out.weight"),
        ("modF.1.weight", "norm_out.linear.weight", swap_scale_shift),
    }

    for k in MAP_BASIC:
        if len(k) > 2:
            key_map[k[1]] = ("{}{}".format(output_prefix, k[0]), None, k[2])
        else:
            key_map[k[1]] = "{}{}".format(output_prefix, k[0])

    return key_map

def flux_to_diffusers(mmdit_config, output_prefix=""):
    n_double_layers = mmdit_config.get("depth", 0)
    n_single_layers = mmdit_config.get("depth_single_blocks", 0)
    hidden_size = mmdit_config.get("hidden_size", 0)

    key_map = {}
    for index in range(n_double_layers):
        prefix_from = "transformer_blocks.{}".format(index)
        prefix_to = "{}double_blocks.{}".format(output_prefix, index)

        for end in ("weight", "bias"):
            k = "{}.attn.".format(prefix_from)
            qkv = "{}.img_attn.qkv.{}".format(prefix_to, end)
            key_map["{}to_q.{}".format(k, end)] = (qkv, (0, 0, hidden_size))
            key_map["{}to_k.{}".format(k, end)] = (qkv, (0, hidden_size, hidden_size))
            key_map["{}to_v.{}".format(k, end)] = (qkv, (0, hidden_size * 2, hidden_size))

            k = "{}.attn.".format(prefix_from)
            qkv = "{}.txt_attn.qkv.{}".format(prefix_to, end)
            key_map["{}add_q_proj.{}".format(k, end)] = (qkv, (0, 0, hidden_size))
            key_map["{}add_k_proj.{}".format(k, end)] = (qkv, (0, hidden_size, hidden_size))
            key_map["{}add_v_proj.{}".format(k, end)] = (qkv, (0, hidden_size * 2, hidden_size))

        block_map = {
                        "attn.to_out.0.weight": "img_attn.proj.weight",
                        "attn.to_out.0.bias": "img_attn.proj.bias",
                        "norm1.linear.weight": "img_mod.lin.weight",
                        "norm1.linear.bias": "img_mod.lin.bias",
                        "norm1_context.linear.weight": "txt_mod.lin.weight",
                        "norm1_context.linear.bias": "txt_mod.lin.bias",
                        "attn.to_add_out.weight": "txt_attn.proj.weight",
                        "attn.to_add_out.bias": "txt_attn.proj.bias",
                        "ff.net.0.proj.weight": "img_mlp.0.weight",
                        "ff.net.0.proj.bias": "img_mlp.0.bias",
                        "ff.net.2.weight": "img_mlp.2.weight",
                        "ff.net.2.bias": "img_mlp.2.bias",
                        "ff_context.net.0.proj.weight": "txt_mlp.0.weight",
                        "ff_context.net.0.proj.bias": "txt_mlp.0.bias",
                        "ff_context.net.2.weight": "txt_mlp.2.weight",
                        "ff_context.net.2.bias": "txt_mlp.2.bias",
                        "ff.linear_in.weight": "img_mlp.0.weight",  # LyCoris LoKr
                        "ff.linear_in.bias": "img_mlp.0.bias",
                        "ff.linear_out.weight": "img_mlp.2.weight",
                        "ff.linear_out.bias": "img_mlp.2.bias",
                        "ff_context.linear_in.weight": "txt_mlp.0.weight",
                        "ff_context.linear_in.bias": "txt_mlp.0.bias",
                        "ff_context.linear_out.weight": "txt_mlp.2.weight",
                        "ff_context.linear_out.bias": "txt_mlp.2.bias",
                        "attn.norm_q.weight": "img_attn.norm.query_norm.weight",
                        "attn.norm_k.weight": "img_attn.norm.key_norm.weight",
                        "attn.norm_added_q.weight": "txt_attn.norm.query_norm.weight",
                        "attn.norm_added_k.weight": "txt_attn.norm.key_norm.weight",
                    }

        for k in block_map:
            key_map["{}.{}".format(prefix_from, k)] = "{}.{}".format(prefix_to, block_map[k])

    for index in range(n_single_layers):
        prefix_from = "single_transformer_blocks.{}".format(index)
        prefix_to = "{}single_blocks.{}".format(output_prefix, index)

        for end in ("weight", "bias"):
            k = "{}.attn.".format(prefix_from)
            qkv = "{}.linear1.{}".format(prefix_to, end)
            key_map["{}to_q.{}".format(k, end)] = (qkv, (0, 0, hidden_size))
            key_map["{}to_k.{}".format(k, end)] = (qkv, (0, hidden_size, hidden_size))
            key_map["{}to_v.{}".format(k, end)] = (qkv, (0, hidden_size * 2, hidden_size))
            key_map["{}.proj_mlp.{}".format(prefix_from, end)] = (qkv, (0, hidden_size * 3, hidden_size * 4))

        block_map = {
                        "norm.linear.weight": "modulation.lin.weight",
                        "norm.linear.bias": "modulation.lin.bias",
                        "proj_out.weight": "linear2.weight",
                        "proj_out.bias": "linear2.bias",
                        "attn.norm_q.weight": "norm.query_norm.weight",
                        "attn.norm_k.weight": "norm.key_norm.weight",
                        "attn.to_qkv_mlp_proj.weight": "linear1.weight", # Flux 2
                        "attn.to_out.weight": "linear2.weight", # Flux 2
                    }

        for k in block_map:
            key_map["{}.{}".format(prefix_from, k)] = "{}.{}".format(prefix_to, block_map[k])

    MAP_BASIC = {
        ("final_layer.linear.bias", "proj_out.bias"),
        ("final_layer.linear.weight", "proj_out.weight"),
        ("img_in.bias", "x_embedder.bias"),
        ("img_in.weight", "x_embedder.weight"),
        ("time_in.in_layer.bias", "time_text_embed.timestep_embedder.linear_1.bias"),
        ("time_in.in_layer.weight", "time_text_embed.timestep_embedder.linear_1.weight"),
        ("time_in.out_layer.bias", "time_text_embed.timestep_embedder.linear_2.bias"),
        ("time_in.out_layer.weight", "time_text_embed.timestep_embedder.linear_2.weight"),
        ("txt_in.bias", "context_embedder.bias"),
        ("txt_in.weight", "context_embedder.weight"),
        ("vector_in.in_layer.bias", "time_text_embed.text_embedder.linear_1.bias"),
        ("vector_in.in_layer.weight", "time_text_embed.text_embedder.linear_1.weight"),
        ("vector_in.out_layer.bias", "time_text_embed.text_embedder.linear_2.bias"),
        ("vector_in.out_layer.weight", "time_text_embed.text_embedder.linear_2.weight"),
        ("guidance_in.in_layer.bias", "time_text_embed.guidance_embedder.linear_1.bias"),
        ("guidance_in.in_layer.weight", "time_text_embed.guidance_embedder.linear_1.weight"),
        ("guidance_in.out_layer.bias", "time_text_embed.guidance_embedder.linear_2.bias"),
        ("guidance_in.out_layer.weight", "time_text_embed.guidance_embedder.linear_2.weight"),
        ("final_layer.adaLN_modulation.1.bias", "norm_out.linear.bias", swap_scale_shift),
        ("final_layer.adaLN_modulation.1.weight", "norm_out.linear.weight", swap_scale_shift),
        ("pos_embed_input.bias", "controlnet_x_embedder.bias"),
        ("pos_embed_input.weight", "controlnet_x_embedder.weight"),
    }

    for k in MAP_BASIC:
        if len(k) > 2:
            key_map[k[1]] = ("{}{}".format(output_prefix, k[0]), None, k[2])
        else:
            key_map[k[1]] = "{}{}".format(output_prefix, k[0])

    return key_map

def z_image_to_diffusers(mmdit_config, output_prefix=""):
    n_layers = mmdit_config.get("n_layers", 0)
    hidden_size = mmdit_config.get("dim", 0)
    n_context_refiner = mmdit_config.get("n_refiner_layers", 2)
    n_noise_refiner = mmdit_config.get("n_refiner_layers", 2)
    key_map = {}

    def add_block_keys(prefix_from, prefix_to, has_adaln=True):
        for end in ("weight", "bias"):
            k = "{}.attention.".format(prefix_from)
            qkv = "{}.attention.qkv.{}".format(prefix_to, end)
            key_map["{}to_q.{}".format(k, end)] = (qkv, (0, 0, hidden_size))
            key_map["{}to_k.{}".format(k, end)] = (qkv, (0, hidden_size, hidden_size))
            key_map["{}to_v.{}".format(k, end)] = (qkv, (0, hidden_size * 2, hidden_size))

        block_map = {
            "attention.norm_q.weight": "attention.q_norm.weight",
            "attention.norm_k.weight": "attention.k_norm.weight",
            "attention.to_out.0.weight": "attention.out.weight",
            "attention.to_out.0.bias": "attention.out.bias",
            "attention_norm1.weight": "attention_norm1.weight",
            "attention_norm2.weight": "attention_norm2.weight",
            "feed_forward.w1.weight": "feed_forward.w1.weight",
            "feed_forward.w2.weight": "feed_forward.w2.weight",
            "feed_forward.w3.weight": "feed_forward.w3.weight",
            "ffn_norm1.weight": "ffn_norm1.weight",
            "ffn_norm2.weight": "ffn_norm2.weight",
        }
        if has_adaln:
            block_map["adaLN_modulation.0.weight"] = "adaLN_modulation.0.weight"
            block_map["adaLN_modulation.0.bias"] = "adaLN_modulation.0.bias"
        for k, v in block_map.items():
            key_map["{}.{}".format(prefix_from, k)] = "{}.{}".format(prefix_to, v)

    for i in range(n_layers):
        add_block_keys("layers.{}".format(i), "{}layers.{}".format(output_prefix, i))

    for i in range(n_context_refiner):
        add_block_keys("context_refiner.{}".format(i), "{}context_refiner.{}".format(output_prefix, i))

    for i in range(n_noise_refiner):
        add_block_keys("noise_refiner.{}".format(i), "{}noise_refiner.{}".format(output_prefix, i))

    MAP_BASIC = [
        ("final_layer.linear.weight", "all_final_layer.2-1.linear.weight"),
        ("final_layer.linear.bias", "all_final_layer.2-1.linear.bias"),
        ("final_layer.adaLN_modulation.1.weight", "all_final_layer.2-1.adaLN_modulation.1.weight"),
        ("final_layer.adaLN_modulation.1.bias", "all_final_layer.2-1.adaLN_modulation.1.bias"),
        ("x_embedder.weight", "all_x_embedder.2-1.weight"),
        ("x_embedder.bias", "all_x_embedder.2-1.bias"),
        ("x_pad_token", "x_pad_token"),
        ("cap_embedder.0.weight", "cap_embedder.0.weight"),
        ("cap_embedder.1.weight", "cap_embedder.1.weight"),
        ("cap_embedder.1.bias", "cap_embedder.1.bias"),
        ("cap_pad_token", "cap_pad_token"),
        ("t_embedder.mlp.0.weight", "t_embedder.mlp.0.weight"),
        ("t_embedder.mlp.0.bias", "t_embedder.mlp.0.bias"),
        ("t_embedder.mlp.2.weight", "t_embedder.mlp.2.weight"),
        ("t_embedder.mlp.2.bias", "t_embedder.mlp.2.bias"),
    ]

    for c, diffusers in MAP_BASIC:
        key_map[diffusers] = "{}{}".format(output_prefix, c)

    return key_map

def krea2_to_diffusers(mmdit_config, output_prefix=""):
    n_layers = mmdit_config.get("layers", 0)
    n_txt_layerwise = 2  # TextFusionTransformer hardcodes 2 layerwise + 2 refiner blocks
    n_txt_refiner = 2
    key_map = {}

    def add_block(prefix_to, prefix_from):
        block_map = {
            "attn.to_q": "attn.wq", "attn.to_k": "attn.wk", "attn.to_v": "attn.wv",
            "attn.to_gate": "attn.gate", "attn.to_out.0": "attn.wo",
            "attn.to_out": "attn.wo",  # some tools drop the ".0" on to_out
            "ff.gate": "mlp.gate", "ff.up": "mlp.up", "ff.down": "mlp.down",
        }
        for d, c in block_map.items():
            key_map["{}.{}.weight".format(prefix_to, d)] = "{}{}.{}.weight".format(output_prefix, prefix_from, c)

    for i in range(n_layers):
        add_block("transformer_blocks.{}".format(i), "blocks.{}".format(i))
    for i in range(n_txt_layerwise):
        add_block("text_fusion.layerwise_blocks.{}".format(i), "txtfusion.layerwise_blocks.{}".format(i))
    for i in range(n_txt_refiner):
        add_block("text_fusion.refiner_blocks.{}".format(i), "txtfusion.refiner_blocks.{}".format(i))

    MAP_BASIC = [
        ("img_in", "first"),
        ("time_embed.linear_1", "tmlp.0"),
        ("time_embed.linear_2", "tmlp.2"),
        ("time_mod_proj", "tproj.1"),
        ("txt_in.linear_1", "txtmlp.1"),
        ("txt_in.linear_2", "txtmlp.3"),
        ("text_fusion.projector", "txtfusion.projector"),
        ("final_layer.linear", "last.linear"),
    ]
    for d, c in MAP_BASIC:
        key_map["{}.weight".format(d)] = "{}{}.weight".format(output_prefix, c)

    return key_map

def repeat_to_batch_size(tensor, batch_size, dim=0):
    if tensor.shape[dim] > batch_size:
        return tensor.narrow(dim, 0, batch_size)
    elif tensor.shape[dim] < batch_size:
        return tensor.repeat(dim * [1] + [math.ceil(batch_size / tensor.shape[dim])] + [1] * (len(tensor.shape) - 1 - dim)).narrow(dim, 0, batch_size)
    return tensor

def resize_to_batch_size(tensor, batch_size):
    in_batch_size = tensor.shape[0]
    if in_batch_size == batch_size:
        return tensor

    if batch_size <= 1:
        return tensor[:batch_size]

    output = torch.empty([batch_size] + list(tensor.shape)[1:], dtype=tensor.dtype, device=tensor.device)
    if batch_size < in_batch_size:
        scale = (in_batch_size - 1) / (batch_size - 1)
        for i in range(batch_size):
            output[i] = tensor[min(round(i * scale), in_batch_size - 1)]
    else:
        scale = in_batch_size / batch_size
        for i in range(batch_size):
            output[i] = tensor[min(math.floor((i + 0.5) * scale), in_batch_size - 1)]

    return output

def resize_list_to_batch_size(l, batch_size):
    in_batch_size = len(l)
    if in_batch_size == batch_size or in_batch_size == 0:
        return l

    if batch_size <= 1:
        return l[:batch_size]

    output = []
    if batch_size < in_batch_size:
        scale = (in_batch_size - 1) / (batch_size - 1)
        for i in range(batch_size):
            output.append(l[min(round(i * scale), in_batch_size - 1)])
    else:
        scale = in_batch_size / batch_size
        for i in range(batch_size):
           output.append(l[min(math.floor((i + 0.5) * scale), in_batch_size - 1)])

    return output

def convert_sd_to(state_dict, dtype):
    keys = list(state_dict.keys())
    for k in keys:
        state_dict[k] = state_dict[k].to(dtype)
    return state_dict

def safetensors_header(safetensors_path, max_size=100*1024*1024):
    with open(safetensors_path, "rb") as f:
        header = f.read(8)
        length_of_header = struct.unpack('<Q', header)[0]
        if length_of_header > max_size:
            return None
        return f.read(length_of_header)

ATTR_UNSET={}

def resolve_attr(obj, attr):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    return obj, attrs[-1]

def set_attr(obj, attr, value):
    obj, name = resolve_attr(obj, attr)
    prev = getattr(obj, name, ATTR_UNSET)
    if value is ATTR_UNSET:
        delattr(obj, name)
    else:
        setattr(obj, name, value)
    return prev

def set_attr_param(obj, attr, value):
    # Clone inference tensors (created under torch.inference_mode) since
    # their version counter is frozen and nn.Parameter() cannot wrap them.
    if (not torch.is_inference_mode_enabled()) and value.is_inference():
        value = value.clone()
    return set_attr(obj, attr, torch.nn.Parameter(value, requires_grad=False))

def set_attr_buffer(obj, attr, value):
    obj, name = resolve_attr(obj, attr)
    prev = getattr(obj, name, ATTR_UNSET)
    persistent = name not in getattr(obj, "_non_persistent_buffers_set", set())
    obj.register_buffer(name, value, persistent=persistent)
    return prev

def copy_to_param(obj, attr, value):
    # inplace update tensor instead of replacing it
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    prev.data.copy_(value)

def get_attr(obj, attr: str):
    """Retrieves a nested attribute from an object using dot notation.

    Args:
        obj: The object to get the attribute from
        attr (str): The attribute path using dot notation (e.g. "model.layer.weight")

    Returns:
        The value of the requested attribute

    Example:
        model = MyModel()
        weight = get_attr(model, "layer1.conv.weight")
        # Equivalent to: model.layer1.conv.weight

    Important:
        Always prefer `comfy.model_patcher.ModelPatcher.get_model_object` when
        accessing nested model objects under `ModelPatcher.model`.
    """
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj

def bislerp(samples, width, height):
    def slerp(b1, b2, r):
        '''slerps batches b1, b2 according to ratio r, batches should be flat e.g. NxC'''

        c = b1.shape[-1]

        #norms
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        #normalize
        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms

        #zero when norms are zero
        b1_normalized[b1_norms.expand(-1,c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1,c) == 0.0] = 0.0

        #slerp
        dot = (b1_normalized*b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        #technically not mathematically correct, but more pleasing?
        res = (torch.sin((1.0-r.squeeze(1))*omega)/so).unsqueeze(1)*b1_normalized + (torch.sin(r.squeeze(1)*omega)/so).unsqueeze(1) * b2_normalized
        res *= (b1_norms * (1.0-r) + b2_norms * r).expand(-1,c)

        #edge cases for same or polar opposites
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5]
        res[dot < 1e-5 - 1] = (b1 * (1.0-r) + b2 * r)[dot < 1e-5 - 1]
        return res

    def generate_bilinear_data(length_old, length_new, device):
        coords_1 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1,1,1,-1))
        coords_1 = torch.nn.functional.interpolate(coords_1, size=(1, length_new), mode="bilinear")
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)

        coords_2 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1,1,1,-1)) + 1
        coords_2[:,:,:,-1] -= 1
        coords_2 = torch.nn.functional.interpolate(coords_2, size=(1, length_new), mode="bilinear")
        coords_2 = coords_2.to(torch.int64)
        return ratios, coords_1, coords_2

    orig_dtype = samples.dtype
    samples = samples.float()
    n,c,h,w = samples.shape
    h_new, w_new = (height, width)

    #linear w
    ratios, coords_1, coords_2 = generate_bilinear_data(w, w_new, samples.device)
    coords_1 = coords_1.expand((n, c, h, -1))
    coords_2 = coords_2.expand((n, c, h, -1))
    ratios = ratios.expand((n, 1, h, -1))

    pass_1 = samples.gather(-1,coords_1).movedim(1, -1).reshape((-1,c))
    pass_2 = samples.gather(-1,coords_2).movedim(1, -1).reshape((-1,c))
    ratios = ratios.movedim(1, -1).reshape((-1,1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h, w_new, c).movedim(-1, 1)

    #linear h
    ratios, coords_1, coords_2 = generate_bilinear_data(h, h_new, samples.device)
    coords_1 = coords_1.reshape((1,1,-1,1)).expand((n, c, -1, w_new))
    coords_2 = coords_2.reshape((1,1,-1,1)).expand((n, c, -1, w_new))
    ratios = ratios.reshape((1,1,-1,1)).expand((n, 1, -1, w_new))

    pass_1 = result.gather(-2,coords_1).movedim(1, -1).reshape((-1,c))
    pass_2 = result.gather(-2,coords_2).movedim(1, -1).reshape((-1,c))
    ratios = ratios.movedim(1, -1).reshape((-1,1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
    return result.to(orig_dtype)

def lanczos(samples, width, height):
    #the below API is strict and expects grayscale to be squeezed
    if samples.ndim == 4:
        samples = samples.squeeze(1) if samples.shape[1] == 1 else samples.movedim(1, -1)
    images = [Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(t).movedim(-1, 0) if (t := np.array(image).astype(np.float32) / 255.0).ndim == 3 else torch.from_numpy(t) for image in images]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)

def common_upscale(samples, width, height, upscale_method, crop):
        orig_shape = tuple(samples.shape)
        if len(orig_shape) > 4:
            samples = samples.reshape(samples.shape[0], samples.shape[1], -1, samples.shape[-2], samples.shape[-1])
            samples = samples.movedim(2, 1)
            samples = samples.reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])
        if crop == "center":
            old_width = samples.shape[-1]
            old_height = samples.shape[-2]
            old_aspect = old_width / old_height
            new_aspect = width / height
            x = 0
            y = 0
            if old_aspect > new_aspect:
                x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
            s = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
        else:
            s = samples

        if upscale_method == "bislerp":
            out = bislerp(s, width, height)
        elif upscale_method == "lanczos":
            out = lanczos(s, width, height)
        else:
            out = torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

        if len(orig_shape) == 4:
            return out

        out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
        return out.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))

def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    rows = 1 if height <= tile_y else math.ceil((height - overlap) / (tile_y - overlap))
    cols = 1 if width <= tile_x else math.ceil((width - overlap) / (tile_x - overlap))
    return rows * cols

@torch.inference_mode()
def tiled_scale_multidim(samples, function, tile=(64, 64), overlap=8, upscale_amount=4, out_channels=3, output_device="cpu", downscale=False, index_formulas=None, pbar=None):
    dims = len(tile)

    if not (isinstance(upscale_amount, (tuple, list))):
        upscale_amount = [upscale_amount] * dims

    if not (isinstance(overlap, (tuple, list))):
        overlap = [overlap] * dims

    if index_formulas is None:
        index_formulas = upscale_amount

    if not (isinstance(index_formulas, (tuple, list))):
        index_formulas = [index_formulas] * dims

    def get_upscale(dim, val):
        up = upscale_amount[dim]
        if callable(up):
            return up(val)
        else:
            return up * val

    def get_downscale(dim, val):
        up = upscale_amount[dim]
        if callable(up):
            return up(val)
        else:
            return val / up

    def get_upscale_pos(dim, val):
        up = index_formulas[dim]
        if callable(up):
            return up(val)
        else:
            return up * val

    def get_downscale_pos(dim, val):
        up = index_formulas[dim]
        if callable(up):
            return up(val)
        else:
            return val / up

    if downscale:
        get_scale = get_downscale
        get_pos = get_downscale_pos
    else:
        get_scale = get_upscale
        get_pos = get_upscale_pos

    def mult_list_upscale(a):
        out = []
        for i in range(len(a)):
            out.append(round(get_scale(i, a[i])))
        return out

    output = torch.empty([samples.shape[0], out_channels] + mult_list_upscale(samples.shape[2:]), device=output_device)

    for b in range(samples.shape[0]):
        s = samples[b:b+1]

        # handle entire input fitting in a single tile
        if all(s.shape[d+2] <= tile[d] for d in range(dims)):
            output[b:b+1] = function(s).to(output_device)
            if pbar is not None:
                pbar.update(1)
            continue

        out = output[b:b+1].zero_()
        out_div = torch.zeros([s.shape[0], 1] + mult_list_upscale(s.shape[2:]), device=output_device)

        positions = [range(0, s.shape[d+2] - overlap[d], tile[d] - overlap[d]) if s.shape[d+2] > tile[d] else [0] for d in range(dims)]

        for it in itertools.product(*positions):
            s_in = s
            upscaled = []

            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap[d], it[d]))
                l = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(get_pos(d, pos)))

            ps = function(s_in).to(output_device)
            mask = torch.ones([1, 1] + list(ps.shape[2:]), device=output_device)

            for d in range(2, dims + 2):
                feather = round(get_scale(d - 2, overlap[d - 2]))
                if feather >= mask.shape[d]:
                    continue
                for t in range(feather):
                    a = (t + 1) / feather
                    mask.narrow(d, t, 1).mul_(a)
                    mask.narrow(d, mask.shape[d] - 1 - t, 1).mul_(a)

            o = out
            o_d = out_div
            ps_view = ps
            mask_view = mask
            for d in range(dims):
                l = min(ps_view.shape[d + 2], o.shape[d + 2] - upscaled[d])
                o = o.narrow(d + 2, upscaled[d], l)
                o_d = o_d.narrow(d + 2, upscaled[d], l)
                if l < ps_view.shape[d + 2]:
                    ps_view = ps_view.narrow(d + 2, 0, l)
                    mask_view = mask_view.narrow(d + 2, 0, l)

            o.add_(ps_view * mask_view)
            o_d.add_(mask_view)

            if pbar is not None:
                pbar.update(1)

        out.div_(out_div)
    return output

def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap = 8, upscale_amount = 4, out_channels = 3, output_device="cpu", pbar = None):
    return tiled_scale_multidim(samples, function, (tile_y, tile_x), overlap=overlap, upscale_amount=upscale_amount, out_channels=out_channels, output_device=output_device, pbar=pbar)

def model_trange(*args, **kwargs):
    if not comfy.memory_management.aimdo_enabled:
        return trange(*args, **kwargs)

    pbar = trange(*args, **kwargs, smoothing=1.0)
    pbar._i = 0
    pbar.set_postfix_str("  Model Initializing ...  ")

    _update = pbar.update

    def warmup_update(n=1):
        pbar._i += 1
        if pbar._i == 1:
            pbar.i1_time = time.time()
            pbar.set_postfix_str(" Model Initialization complete!  ")
        elif pbar._i == 2:
            #bring forward the effective start time based the diff between first and second iteration
            #to attempt to remove load overhead from the final step rate estimate.
            pbar.start_t = pbar.i1_time - (time.time() - pbar.i1_time)
            pbar.set_postfix_str("")

        _update(n)

    pbar.update = warmup_update
    return pbar

PROGRESS_BAR_ENABLED = True
def set_progress_bar_enabled(enabled):
    global PROGRESS_BAR_ENABLED
    PROGRESS_BAR_ENABLED = enabled

PROGRESS_BAR_HOOK = None
def set_progress_bar_global_hook(function):
    global PROGRESS_BAR_HOOK
    PROGRESS_BAR_HOOK = function

# Throttle settings for progress bar updates to reduce WebSocket flooding
PROGRESS_THROTTLE_MIN_INTERVAL = 0.1  # 100ms minimum between updates
PROGRESS_THROTTLE_MIN_PERCENT = 0.5   # 0.5% minimum progress change

class ProgressBar:
    def __init__(self, total, node_id=None):
        global PROGRESS_BAR_HOOK
        self.total = total
        self.current = 0
        self.hook = PROGRESS_BAR_HOOK
        self.node_id = node_id
        self._last_update_time = 0.0
        self._last_sent_value = -1

    def update_absolute(self, value, total=None, preview=None):
        if total is not None:
            self.total = total
        if value > self.total:
            value = self.total
        self.current = value
        if self.hook is not None:
            current_time = time.perf_counter()
            is_first = (self._last_sent_value < 0)
            is_final = (value >= self.total)
            has_preview = (preview is not None)

            # Always send immediately for previews, first update, or final update
            if has_preview or is_first or is_final:
                self.hook(self.current, self.total, preview, node_id=self.node_id)
                self._last_update_time = current_time
                self._last_sent_value = value
                return

            # Apply throttling for regular progress updates
            if self.total > 0:
                percent_changed = ((value - max(0, self._last_sent_value)) / self.total) * 100
            else:
                percent_changed = 100
            time_elapsed = current_time - self._last_update_time

            if time_elapsed >= PROGRESS_THROTTLE_MIN_INTERVAL and percent_changed >= PROGRESS_THROTTLE_MIN_PERCENT:
                self.hook(self.current, self.total, preview, node_id=self.node_id)
                self._last_update_time = current_time
                self._last_sent_value = value

    def update(self, value):
        self.update_absolute(self.current + value)

def reshape_mask(input_mask, output_shape):
    dims = len(output_shape) - 2

    if dims == 1:
        scale_mode = "linear"

    if dims == 2:
        input_mask = input_mask.reshape((-1, 1, input_mask.shape[-2], input_mask.shape[-1]))
        scale_mode = "bilinear"

    if dims == 3:
        if len(input_mask.shape) < 5:
            input_mask = input_mask.reshape((1, 1, -1, input_mask.shape[-2], input_mask.shape[-1]))
        scale_mode = "trilinear"

    mask = torch.nn.functional.interpolate(input_mask, size=output_shape[2:], mode=scale_mode)
    if mask.shape[1] < output_shape[1]:
        mask = mask.repeat((1, output_shape[1]) + (1,) * dims)[:,:output_shape[1]]
    mask = repeat_to_batch_size(mask, output_shape[0])
    return mask

def upscale_dit_mask(mask: torch.Tensor, img_size_in, img_size_out):
        hi, wi = img_size_in
        ho, wo = img_size_out
        # if it's already the correct size, no need to do anything
        if (hi, wi) == (ho, wo):
            return mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3:
            raise ValueError(f"Got a mask of shape {list(mask.shape)}, expected [b, q, k] or [q, k]")
        txt_tokens = mask.shape[1] - (hi * wi)
        # quadrants of the mask
        txt_to_txt = mask[:, :txt_tokens, :txt_tokens]
        txt_to_img = mask[:, :txt_tokens, txt_tokens:]
        img_to_img = mask[:, txt_tokens:, txt_tokens:]
        img_to_txt = mask[:, txt_tokens:, :txt_tokens]

        # convert to 1d x 2d, interpolate, then back to 1d x 1d
        txt_to_img = rearrange  (txt_to_img, "b t (h w) -> b t h w", h=hi, w=wi)
        txt_to_img = interpolate(txt_to_img, size=img_size_out, mode="bilinear")
        txt_to_img = rearrange  (txt_to_img, "b t h w -> b t (h w)")
        # this one is hard because we have to do it twice
        # convert to 1d x 2d, interpolate, then to 2d x 1d, interpolate, then 1d x 1d
        img_to_img = rearrange  (img_to_img, "b hw (h w) -> b hw h w", h=hi, w=wi)
        img_to_img = interpolate(img_to_img, size=img_size_out, mode="bilinear")
        img_to_img = rearrange  (img_to_img, "b (hk wk) hq wq -> b (hq wq) hk wk", hk=hi, wk=wi)
        img_to_img = interpolate(img_to_img, size=img_size_out, mode="bilinear")
        img_to_img = rearrange  (img_to_img, "b (hq wq) hk wk -> b (hk wk) (hq wq)", hq=ho, wq=wo)
        # convert to 2d x 1d, interpolate, then back to 1d x 1d
        img_to_txt = rearrange  (img_to_txt, "b (h w) t -> b t h w", h=hi, w=wi)
        img_to_txt = interpolate(img_to_txt, size=img_size_out, mode="bilinear")
        img_to_txt = rearrange  (img_to_txt, "b t h w -> b (h w) t")

        # reassemble the mask from blocks
        out = torch.cat([
            torch.cat([txt_to_txt, txt_to_img], dim=2),
            torch.cat([img_to_txt, img_to_img], dim=2)],
            dim=1
        )
        return out

def pack_latents(latents):
    latent_shapes = []
    tensors = []
    for tensor in latents:
        latent_shapes.append(tensor.shape)
        tensors.append(tensor.reshape(tensor.shape[0], 1, -1))

    latent = torch.cat(tensors, dim=-1)
    return latent, latent_shapes

def unpack_latents(combined_latent, latent_shapes):
    if len(latent_shapes) > 1:
        output_tensors = []
        for shape in latent_shapes:
            cut = math.prod(shape[1:])
            tens = combined_latent[:, :, :cut]
            combined_latent = combined_latent[:, :, cut:]
            output_tensors.append(tens.reshape([tens.shape[0]] + list(shape)[1:]))
    else:
        output_tensors = [combined_latent]
    return output_tensors

def detect_layer_quantization(state_dict, prefix):
    for k in state_dict:
        if k.startswith(prefix) and k.endswith(".comfy_quant"):
            logging.info("Found quantization metadata version 1")
            return {"mixed_ops": True}
    return None

def convert_old_quants(state_dict, model_prefix="", metadata={}):
    if metadata is None:
        metadata = {}

    quant_metadata = None
    if "_quantization_metadata" not in metadata:
        scaled_fp8_key = "{}scaled_fp8".format(model_prefix)

        if scaled_fp8_key in state_dict:
            scaled_fp8_weight = state_dict[scaled_fp8_key]
            scaled_fp8_dtype = scaled_fp8_weight.dtype
            if scaled_fp8_dtype == torch.float32:
                scaled_fp8_dtype = torch.float8_e4m3fn

            if scaled_fp8_weight.nelement() == 2:
                full_precision_matrix_mult = True
            else:
                full_precision_matrix_mult = False

            out_sd = {}
            layers = {}
            for k in list(state_dict.keys()):
                if k == scaled_fp8_key:
                    continue
                if not k.startswith(model_prefix):
                    out_sd[k] = state_dict[k]
                    continue
                k_out = k
                w = state_dict.pop(k)
                layer = None
                if k_out.endswith(".scale_weight"):
                    layer = k_out[:-len(".scale_weight")]
                    k_out = "{}.weight_scale".format(layer)

                if layer is not None:
                    layer_conf = {"format": "float8_e4m3fn"}
                    if full_precision_matrix_mult:
                        layer_conf["full_precision_matrix_mult"] = full_precision_matrix_mult
                    layers[layer] = layer_conf

                if k_out.endswith(".scale_input"):
                    layer = k_out[:-len(".scale_input")]
                    k_out = "{}.input_scale".format(layer)
                    if w.item() == 1.0:
                        continue

                out_sd[k_out] = w

            state_dict = out_sd
            quant_metadata = {"layers": layers}
    else:
        quant_metadata = json.loads(metadata["_quantization_metadata"])

    if quant_metadata is not None:
        layers = quant_metadata["layers"]
        for k, v in layers.items():
            state_dict["{}.comfy_quant".format(k)] = torch.tensor(list(json.dumps(v).encode('utf-8')), dtype=torch.uint8)

    return state_dict, metadata

def string_to_seed(data):
    crc = 0xFFFFFFFF
    for byte in data:
        if isinstance(byte, str):
            byte = ord(byte)
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF

def deepcopy_list_dict(obj, memo=None):
    if memo is None:
        memo = {}

    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]

    if isinstance(obj, dict):
        res = {deepcopy_list_dict(k, memo): deepcopy_list_dict(v, memo) for k, v in obj.items()}
    elif isinstance(obj, list):
        res = [deepcopy_list_dict(i, memo) for i in obj]
    else:
        res = obj

    memo[obj_id] = res
    return res

def bit_reverse_range(index, bits):
    result = 0
    for _ in range(bits):
        result = (result << 1) | (index & 1)
        index >>= 1
    return result
