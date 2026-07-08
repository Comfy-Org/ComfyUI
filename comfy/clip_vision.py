from .utils import load_torch_file, transformers_convert, state_dict_prefix_replace
import os
import json
import logging

import comfy.ops
import comfy.model_patcher
import comfy.model_management
import comfy.utils
import comfy.clip_model
import comfy.image_encoders.dino2
import comfy.image_encoders.dino3

class Output:
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, item):
        setattr(self, key, item)

clip_preprocess = comfy.clip_model.clip_preprocess  # Prevent some stuff from breaking, TODO: remove eventually

IMAGE_ENCODERS = {
    "clip_vision_model": comfy.clip_model.CLIPVisionModelProjection,
    "siglip_vision_model": comfy.clip_model.CLIPVisionModelProjection,
    "siglip2_vision_model": comfy.clip_model.CLIPVisionModelProjection,
    "dinov2": comfy.image_encoders.dino2.Dinov2Model,
    "dinov3": comfy.image_encoders.dino3.DINOv3ViTModel,
}

class ClipVisionModel():
    def __init__(self, json_config):
        if isinstance(json_config, dict):
            config = json_config
        else:
            with open(json_config) as f:
                config = json.load(f)

        self.image_size = config.get("image_size", 224)
        self.image_mean = config.get("image_mean", [0.48145466, 0.4578275, 0.40821073])
        self.image_std = config.get("image_std", [0.26862954, 0.26130258, 0.27577711])
        self.model_type = config.get("model_type", "clip_vision_model")
        self.config = config.copy()
        model_class = IMAGE_ENCODERS.get(self.model_type)
        if self.model_type == "siglip_vision_model":
            self.return_all_hidden_states = True
        else:
            self.return_all_hidden_states = False

        self.load_device = comfy.model_management.text_encoder_device()
        offload_device = comfy.model_management.text_encoder_offload_device()
        self.dtype = comfy.model_management.text_encoder_dtype(self.load_device)
        self.model = model_class(config, self.dtype, offload_device, comfy.ops.manual_cast)
        self.model.eval()

        self.patcher = comfy.model_patcher.CoreModelPatcher(self.model, load_device=self.load_device, offload_device=offload_device)

    def load_sd(self, sd):
        return self.model.load_state_dict(sd, strict=False, assign=self.patcher.is_dynamic())

    def get_sd(self):
        return self.model.state_dict()

    def encode_image(self, image, crop=True):
        comfy.model_management.load_model_gpu(self.patcher)
        if self.model_type == "siglip2_vision_model":
            pixel_values = comfy.clip_model.siglip2_preprocess(image.to(self.load_device), size=self.image_size, patch_size=self.config.get("patch_size", 16), num_patches=self.config.get("num_patches", 256), mean=self.image_mean, std=self.image_std, crop=crop).float()
        else:
            pixel_values = comfy.clip_model.clip_preprocess(image.to(self.load_device), size=self.image_size, mean=self.image_mean, std=self.image_std, crop=crop).float()
        out = self.model(pixel_values=pixel_values, intermediate_output='all' if self.return_all_hidden_states else -2)

        outputs = Output()
        outputs["last_hidden_state"] = out[0].to(comfy.model_management.intermediate_device())
        outputs["image_embeds"] = out[2].to(comfy.model_management.intermediate_device())
        outputs["image_sizes"] = [pixel_values.shape[1:]] * pixel_values.shape[0]
        if self.return_all_hidden_states:
            all_hs = out[1].to(comfy.model_management.intermediate_device())
            outputs["penultimate_hidden_states"] = all_hs[:, -2]
            outputs["all_hidden_states"] = all_hs
        else:
            outputs["penultimate_hidden_states"] = out[1].to(comfy.model_management.intermediate_device())

        outputs["mm_projected"] = out[3]
        return outputs

def convert_to_transformers(sd, prefix):
    sd_k = sd.keys()
    if "{}transformer.resblocks.0.attn.in_proj_weight".format(prefix) in sd_k:
        keys_to_replace = {
            "{}class_embedding".format(prefix): "vision_model.embeddings.class_embedding",
            "{}conv1.weight".format(prefix): "vision_model.embeddings.patch_embedding.weight",
            "{}positional_embedding".format(prefix): "vision_model.embeddings.position_embedding.weight",
            "{}ln_post.bias".format(prefix): "vision_model.post_layernorm.bias",
            "{}ln_post.weight".format(prefix): "vision_model.post_layernorm.weight",
            "{}ln_pre.bias".format(prefix): "vision_model.pre_layrnorm.bias",
            "{}ln_pre.weight".format(prefix): "vision_model.pre_layrnorm.weight",
        }

        for x in keys_to_replace:
            if x in sd_k:
                sd[keys_to_replace[x]] = sd.pop(x)

        if "{}proj".format(prefix) in sd_k:
            sd['visual_projection.weight'] = sd.pop("{}proj".format(prefix)).transpose(0, 1)

        sd = transformers_convert(sd, prefix, "vision_model.", 48)
    else:
        replace_prefix = {prefix: ""}
        sd = state_dict_prefix_replace(sd, replace_prefix)
    return sd

def load_clipvision_from_sd(sd, prefix="", convert_keys=False):
    if convert_keys:
        sd = convert_to_transformers(sd, prefix)
    if "vision_model.encoder.layers.47.layer_norm1.weight" in sd:
        json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_g.json")
    elif "vision_model.encoder.layers.30.layer_norm1.weight" in sd:
        json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_h.json")
    elif "vision_model.encoder.layers.22.layer_norm1.weight" in sd:
        embed_shape = sd["vision_model.embeddings.position_embedding.weight"].shape[0]
        if sd["vision_model.encoder.layers.0.layer_norm1.weight"].shape[0] == 1152:
            patch_embedding_shape = sd["vision_model.embeddings.patch_embedding.weight"].shape
            if len(patch_embedding_shape) == 2:
                json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_siglip2_base_naflex.json")
            else:
                if embed_shape == 729:
                    json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_siglip_384.json")
                elif embed_shape == 1024:
                    json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_siglip_512.json")
        elif embed_shape == 577:
            if "multi_modal_projector.linear_1.bias" in sd:
                json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_vitl_336_llava.json")
            else:
                json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_vitl_336.json")
        else:
            json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_vitl.json")

    # Dinov2
    elif 'encoder.layer.39.layer_scale2.lambda1' in sd:
        json_config = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "image_encoders"), "dino2_giant.json")
    elif 'encoder.layer.23.layer_scale2.lambda1' in sd:
        json_config = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "image_encoders"), "dino2_large.json")
    elif 'layer.0.mlp.gate_proj.weight' in sd and 'layer.31.norm1.weight' in sd: # Dinov3 ViT-H/16+ (SwiGLU gated MLP, 32 layers)
        json_config = comfy.image_encoders.dino3.DINOV3_VITH_CONFIG
    else:
        return None

    clip = ClipVisionModel(json_config)
    m, u = clip.load_sd(sd)
    if len(m) > 0:
        logging.warning("missing clip vision: {}".format(m))
    u = set(u)
    keys = list(sd.keys())
    for k in keys:
        if k not in u:
            sd.pop(k)
    return clip

def load(ckpt_path):
    sd = load_torch_file(ckpt_path)
    if "visual.transformer.resblocks.0.attn.in_proj_weight" in sd:
        return load_clipvision_from_sd(sd, prefix="visual.", convert_keys=True)
    else:
        return load_clipvision_from_sd(sd)
