import torch


def convert_sensenova(state_dict, prefix):
    out = {}
    for key, value in state_dict.items():
        name = key.removeprefix(prefix)
        if name.startswith("vision_model.") or (name.startswith("language_model.") and "_mot_gen" not in name):
            target = "text_encoders."
            name = name.replace("vision_model.embeddings.", "vision_model.", 1)
        else:
            target = "model.diffusion_model."
            name = name.removeprefix("language_model.model.").removeprefix("fm_modules.")
            name = name.replace("vision_model_mot_gen.embeddings.", "patch_embed.").replace("fm_head.", "head.").replace("_mot_gen", "")
        out[target + name] = value
    out["vae.pixel_space_vae"] = torch.tensor(1.0)
    return out


def convert_checkpoint(state_dict):
    sensenova_keys = (
        "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight",
        "fm_modules.fm_head.conv1.weight",
        "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight",
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight",
    )
    for prefix in ("", "model.diffusion_model."):
        if all(prefix + key in state_dict for key in sensenova_keys):
            return convert_sensenova(state_dict, prefix)
    return state_dict
