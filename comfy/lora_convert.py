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


def convert_krea2_lora(sd):
    sd_out = {}
    for k in sd:
        if ".attn.to_qkv." not in k:
            sd_out[k] = sd[k]
            continue

        base, suffix = k.split(".attn.to_qkv.", 1)

        if suffix in ("lora_B.weight", "lora.up.weight", "lora_up.weight"):
            tensor = sd[k]
            for down_sfx in ("lora_A.weight", "lora.down.weight", "lora_down.weight"):
                down_key = "{}.attn.to_qkv.{}".format(base, down_sfx)
                if down_key in sd:
                    input_dim = sd[down_key].shape[1]
                    break
            else:
                sd_out[k] = sd[k]
                continue

            output_dim = tensor.shape[0]
            q_dim = input_dim
            kv_dim = (output_dim - q_dim) // 2
            q_up, k_up, v_up = tensor.split([q_dim, kv_dim, output_dim - q_dim - kv_dim], dim=0)
            sd_out["{}.attn.to_q.{}".format(base, suffix)] = q_up
            sd_out["{}.attn.to_k.{}".format(base, suffix)] = k_up
            sd_out["{}.attn.to_v.{}".format(base, suffix)] = v_up
        else:
            for name in ("to_q", "to_k", "to_v"):
                sd_out["{}.attn.{}.{}".format(base, name, suffix)] = sd[k]

    return sd_out


def convert_lora(sd):
    if "img_in.lora_A.weight" in sd and "single_blocks.0.norm.key_norm.scale" in sd:
        return convert_lora_bfl_control(sd)
    if "lora_unet__blocks_0_cross_attn_k.lora_down.weight" in sd:
        return convert_lora_wan_fun(sd)
    if "single_blocks.37.processor.qkv_lora.up.weight" in sd and "double_blocks.18.processor.qkv_lora2.up.weight" in sd:
        return convert_uso_lora(sd)
    if any(".text_fusion." in k for k in sd) and any(".attn.to_qkv." in k for k in sd):
        return convert_krea2_lora(sd)
    return sd
