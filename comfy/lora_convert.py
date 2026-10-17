import logging
import re

import torch
import comfy.utils


def convert_lora_bfl_control(sd): #BFL loras for Flux
    sd_out = {}
    for k in sd:
        k_to = "diffusion_model.{}".format(k.replace(".lora_B.bias", ".diff_b").replace("_norm.scale", "_norm.set_weight"))
        sd_out[k_to] = sd[k]

    sd_out["diffusion_model.img_in.reshape_weight"] = torch.tensor([sd["img_in.lora_B.weight"].shape[0], sd["img_in.lora_A.weight"].shape[1]])
    return sd_out


def convert_lora_wan_fun(sd): #Wan Fun loras
    return comfy.utils.state_dict_prefix_replace(sd, {"lora_unet__": "lora_unet_"})

def convert_uso_lora(sd):
    sd_out = {}
    for k in sd:
        tensor = sd[k]
        k_to = "diffusion_model.{}".format(k.replace(".down.weight", ".lora_down.weight")
                                           .replace(".up.weight", ".lora_up.weight")
                                           .replace(".qkv_lora2.", ".txt_attn.qkv.")
                                           .replace(".qkv_lora1.", ".img_attn.qkv.")
                                           .replace(".proj_lora1.", ".img_attn.proj.")
                                           .replace(".proj_lora2.", ".txt_attn.proj.")
                                           .replace(".qkv_lora.", ".linear1_qkv.")
                                           .replace(".proj_lora.", ".linear2.")
                                           .replace(".processor.", ".")
                                           )
        sd_out[k_to] = tensor
    return sd_out


# q/k/v markers of a weight or lora key name.
QKV_MARKERS = ((".wq", "_to_q", "_wq"), (".wk", "_to_k", "_wk"), (".wv", "_to_v", "_wv"))


def qkv_projection(name):
    """q, k or v when a weight or lora key name is one projection, None for a fused name."""
    if "qkv" in name:
        return None
    for projection, markers in zip("qkv", QKV_MARKERS):
        if any(marker in name for marker in markers):
            return projection
    return None


def qkv_projection_key(x, target, lora):
    """The key a fused qkv adapter has to be loaded under, x when it is not a split fused key.

    A fused qkv adapter covers the q, k and v projections of one module, so a caller that maps
    the fused key itself can only be loaded for the projection its target weight names.
    """
    if not x.endswith("qkv"):
        return x
    projection = qkv_projection(target)
    if projection is None:
        return x
    key = x[:-len("qkv")] + projection
    # a split adapter has its fused factors replaced by the projection ones
    split = any(key + suffix in lora for suffix in QKV_FACTORS)
    if split and not any(x + suffix in lora for suffix in QKV_FACTORS):
        return key
    return x


FUSED_QKV = re.compile(r"(.*attn[._]to_)qkv(?=\.|$)")
QKV_PROJECTION = re.compile(r"(.*attn[._]to_)([qkv])(?=\.|$)")

# Parameters of a fused qkv adapter, grouped by how they are turned into per-projection
# parameters. Shared parameters repeat for to_q/to_k/to_v, output parameters are split into rows
# and the LoKr outer factor is split too when the factorization allows it.
QKV_SHARED = (
    ".lokr_w2", ".lokr_w2_a", ".lokr_w2_b", ".lokr_t2", ".lokr_w1_b",
    ".alpha", ".lora_A.weight", ".lora_down.weight", ".lora_mid.weight",
)
QKV_OUTER = (".lokr_w1", ".lokr_w1_a")
QKV_OUTPUT = (".dora_scale", ".lora_B.weight", ".lora_up.weight")
QKV_FACTORS = QKV_SHARED + QKV_OUTER + QKV_OUTPUT


def krea2_fused_qkv_lora(sd):
    """Whether a lora keeps Krea2 attention in one fused to_qkv adapter.

    Krea2 names its attention after transformer_blocks or text_fusion, other models with a fused
    to_qkv module name it differently.
    """
    return any(FUSED_QKV.search(k) is not None and ("text_fusion" in k or "transformer_blocks" in k) for k in sd)


def _qkv_shapes(tensors):
    """(output size, input size, LoKr inner factor) of a fused qkv adapter."""
    if ".lokr_w2" in tensors and (".lokr_w1" in tensors or ".lokr_w1_a" in tensors):
        out_inner = tensors[".lokr_w2"].shape[0]
        in_inner = tensors[".lokr_w2"].shape[1]
    elif ".lokr_w2_a" in tensors and ".lokr_w2_b" in tensors and ".lokr_w1_a" in tensors:
        out_inner = tensors[".lokr_w2_a"].shape[0]
        in_inner = tensors[".lokr_w2_b"].shape[1]
    else:
        out_inner = in_inner = None

    if ".lokr_w1" in tensors:
        if out_inner is None:
            return None
        out_dim = tensors[".lokr_w1"].shape[0] * out_inner
        in_dim = tensors[".lokr_w1"].shape[1] * in_inner
    elif ".lokr_w1_a" in tensors and ".lokr_w1_b" in tensors:
        if out_inner is None:
            return None
        out_dim = tensors[".lokr_w1_a"].shape[0] * out_inner
        in_dim = tensors[".lokr_w1_b"].shape[1] * in_inner
    elif ".lora_B.weight" in tensors and ".lora_A.weight" in tensors:
        out_dim = tensors[".lora_B.weight"].shape[0]
        in_dim = tensors[".lora_A.weight"].shape[1]
    elif ".lora_up.weight" in tensors and ".lora_down.weight" in tensors:
        out_dim = tensors[".lora_up.weight"].shape[0]
        in_dim = tensors[".lora_down.weight"].shape[1]
    else:
        return None
    if ".dora_scale" in tensors and tensors[".dora_scale"].shape[0] != out_dim:
        return None
    return out_dim, in_dim, out_inner


def _qkv_has_delta(tensors):
    """Whether a per-projection qkv entry holds a trained delta instead of being a placeholder."""
    w1 = [tensors[name].any() for name in (".lokr_w1", ".lokr_w1_a", ".lokr_w1_b") if name in tensors]
    w2 = [tensors[name].any() for name in (".lokr_w2", ".lokr_w2_a", ".lokr_w2_b") if name in tensors]
    if w1 and w2:
        return all(w1 + w2)
    up = [tensors[name].any() for name in (".lora_up.weight", ".lora_B.weight") if name in tensors]
    down = [tensors[name].any() for name in (".lora_down.weight", ".lora_A.weight") if name in tensors]
    if up and down:
        return all(up + down)
    return False


def convert_fused_qkv_lora(sd):
    # SimpleTuner/LyCORIS save attention as one fused to_qkv adapter while ComfyUI's model keeps
    # wq/wk/wv separate. Split the fused adapter (and its DoRA scale) into the three projections
    # and drop the unused per-projection entries written next to it.
    fused = {}
    for k in sd:
        m = FUSED_QKV.search(k)
        if m is not None:
            fused.setdefault(m.group(1), {})[k[m.end():]] = k

    sd_out = {}
    converted_bases = set()
    converted_keys = set()
    for base, params in fused.items():
        tensors = {suffix: sd[key] for suffix, key in params.items()}
        shapes = _qkv_shapes(tensors)
        if shapes is None:
            continue
        out_dim, in_dim, out_inner = shapes

        # The query keeps the width of the input, the remaining outputs are split evenly between
        # key and value.
        q_out = in_dim
        kv_out = (out_dim - q_out) // 2
        if kv_out <= 0 or q_out + 2 * kv_out != out_dim:
            continue
        output_sizes = (q_out, kv_out, kv_out)

        # The LoKr outer factor can be split only when its inner factor lines up with the q/k/v
        # boundaries. Otherwise the fused factor is kept and each projection takes its own rows
        # of the Kronecker product.
        outer_split = out_inner is not None and q_out % out_inner == 0 and kv_out % out_inner == 0
        outer_sizes = (q_out // out_inner, kv_out // out_inner, kv_out // out_inner) if outer_split else None
        row_offsets = not outer_split and any(suffix in tensors for suffix in QKV_OUTER)

        converted_bases.add(base)
        converted_keys.update(params.values())
        for suffix, key in params.items():
            tensor = tensors[suffix]
            if suffix in QKV_SHARED:
                parts = (tensor, tensor, tensor)
            elif suffix in QKV_OUTPUT:
                parts = tensor.split(output_sizes, dim=0)
            elif suffix in QKV_OUTER:
                parts = tensor.split(outer_sizes, dim=0) if outer_split else (tensor, tensor, tensor)
            else:
                sd_out[key] = tensor
                continue
            for proj, part in zip("qkv", parts):
                sd_out["{}{}{}".format(base, proj, suffix)] = part
        if row_offsets:
            offset = 0
            for proj, size in zip("qkv", output_sizes):
                sd_out["{}{}.row_offset".format(base, proj)] = torch.tensor(offset)
                offset += size

    # The per-projection entries next to a fused adapter belong to the unused unfused modules and
    # their delta is empty. The split emits the trained delta and DoRA scale of every projection
    # and only one adapter can be applied per weight, so dropping them costs nothing. A trained
    # entry would be lost and is reported.
    duplicates = {}
    for k in sd:
        if k in converted_keys:
            continue
        m = QKV_PROJECTION.search(k)
        if m is not None and m.group(1) in converted_bases:
            duplicates.setdefault(m.group(1) + m.group(2), {})[k[m.end():]] = k
            continue
        sd_out[k] = sd[k]

    trained = [base for base, params in duplicates.items()
               if _qkv_has_delta({suffix: sd[key] for suffix, key in params.items()})]
    if trained:
        logging.warning(
            "The trained per-projection adapters {} are not applied, the fused to_qkv "
            "adapter of the same module already covers these weights".format(", ".join(sorted(trained)[:2]))
        )
    return sd_out


def convert_lora(sd):
    if "img_in.lora_A.weight" in sd and "single_blocks.0.norm.key_norm.scale" in sd:
        return convert_lora_bfl_control(sd)
    if "lora_unet__blocks_0_cross_attn_k.lora_down.weight" in sd:
        return convert_lora_wan_fun(sd)
    if "single_blocks.37.processor.qkv_lora.up.weight" in sd and "double_blocks.18.processor.qkv_lora2.up.weight" in sd:
        return convert_uso_lora(sd)
    if krea2_fused_qkv_lora(sd):
        return convert_fused_qkv_lora(sd)
    return sd
