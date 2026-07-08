from .utils import load_torch_file
import os
import json
import torch
import logging

import comfy.ops
import comfy.model_patcher
import comfy.model_management
import comfy.clip_model
import comfy.background_removal.birefnet

BG_REMOVAL_MODELS = {
    "birefnet": comfy.background_removal.birefnet.BiRefNet
}

class BackgroundRemovalModel():
    def __init__(self, json_config):
        with open(json_config) as f:
            config = json.load(f)

        self.image_size = config.get("image_size", 1024)
        self.image_mean = config.get("image_mean", [0.0, 0.0, 0.0])
        self.image_std = config.get("image_std", [1.0, 1.0, 1.0])
        self.model_type = config.get("model_type", "birefnet")
        self.config = config.copy()
        model_class = BG_REMOVAL_MODELS.get(self.model_type)

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

    def encode_image(self, image):
        comfy.model_management.load_model_gpu(self.patcher)
        H, W = image.shape[1], image.shape[2]
        pixel_values = comfy.clip_model.clip_preprocess(image.to(self.load_device), size=self.image_size, mean=self.image_mean, std=self.image_std, crop=False)

        if pixel_values.shape[0] > 1:
            out = torch.cat([
                self.model(pixel_values=pixel_values[i:i+1])
                for i in range(pixel_values.shape[0])
            ], dim=0)
        else:
            out = self.model(pixel_values=pixel_values)
        out = torch.nn.functional.interpolate(out, size=(H, W), mode="bicubic", antialias=False)

        mask = out.sigmoid().to(device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
        return mask.squeeze(1)  # (B, 1, H, W) -> (B, H, W)


def load_background_removal_model(sd):
    if "bb.layers.1.blocks.0.attn.relative_position_index" in sd:
        json_config = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "background_removal"), "birefnet.json")
    else:
        return None

    bg_model = BackgroundRemovalModel(json_config)
    m, u = bg_model.load_sd(sd)
    if len(m) > 0:
        logging.warning("missing background removal: {}".format(m))
    u = set(u)
    keys = list(sd.keys())
    for k in keys:
        if k not in u:
            sd.pop(k)
    return bg_model

def load(ckpt_path):
    sd = load_torch_file(ckpt_path)
    return load_background_removal_model(sd)
