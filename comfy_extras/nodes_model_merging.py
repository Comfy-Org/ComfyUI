import comfy.sd
import comfy.utils
import comfy.model_base
import comfy.model_management
import comfy.model_sampling

import torch
import folder_paths
import json
import os

from comfy.cli_args import args

class ModelMergeSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model1": ("MODEL",),
                              "model2": ("MODEL",),
                              "ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"

    CATEGORY = "model/merging"

    def merge(self, model1, model2, ratio):
        m = model1.clone()
        kp = model2.get_key_patches("diffusion_model.")
        for k in kp:
            m.add_patches({k: kp[k]}, 1.0 - ratio, ratio)
        return (m, )

class ModelSubtract:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model1": ("MODEL",),
                              "model2": ("MODEL",),
                              "multiplier": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"

    CATEGORY = "model/merging"

    def merge(self, model1, model2, multiplier):
        m = model1.clone()
        kp = model2.get_key_patches("diffusion_model.")
        for k in kp:
            m.add_patches({k: kp[k]}, - multiplier, multiplier)
        return (m, )

class ModelAdd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model1": ("MODEL",),
                              "model2": ("MODEL",),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"

    CATEGORY = "model/merging"

    def merge(self, model1, model2):
        m = model1.clone()
        kp = model2.get_key_patches("diffusion_model.")
        for k in kp:
            m.add_patches({k: kp[k]}, 1.0, 1.0)
        return (m, )


class CLIPMergeSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip1": ("CLIP",),
                              "clip2": ("CLIP",),
                              "ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "merge"

    CATEGORY = "model/merging"

    def merge(self, clip1, clip2, ratio):
        m = clip1.clone()
        kp = clip2.get_key_patches()
        for k in kp:
            if k.endswith(".position_ids") or k.endswith(".logit_scale"):
                continue
            m.add_patches({k: kp[k]}, 1.0 - ratio, ratio)
        return (m, )


class CLIPSubtract:
    SEARCH_ALIASES = ["clip difference", "text encoder subtract"]
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip1": ("CLIP",),
                              "clip2": ("CLIP",),
                              "multiplier": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "merge"

    CATEGORY = "model/merging"

    def merge(self, clip1, clip2, multiplier):
        m = clip1.clone()
        kp = clip2.get_key_patches()
        for k in kp:
            if k.endswith(".position_ids") or k.endswith(".logit_scale"):
                continue
            m.add_patches({k: kp[k]}, - multiplier, multiplier)
        return (m, )


class CLIPAdd:
    SEARCH_ALIASES = ["combine clip"]
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip1": ("CLIP",),
                              "clip2": ("CLIP",),
                              }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "merge"

    CATEGORY = "model/merging"

    def merge(self, clip1, clip2):
        m = clip1.clone()
        kp = clip2.get_key_patches()
        for k in kp:
            if k.endswith(".position_ids") or k.endswith(".logit_scale"):
                continue
            m.add_patches({k: kp[k]}, 1.0, 1.0)
        return (m, )


class ModelMergeBlocks:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model1": ("MODEL",),
                              "model2": ("MODEL",),
                              "input": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "middle": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "out": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"

    CATEGORY = "model/merging"

    def merge(self, model1, model2, **kwargs):
        m = model1.clone()
        kp = model2.get_key_patches("diffusion_model.")
        default_ratio = next(iter(kwargs.values()))

        for k in kp:
            ratio = default_ratio
            k_unet = k[len("diffusion_model."):]

            last_arg_size = 0
            for arg in kwargs:
                if k_unet.startswith(arg) and last_arg_size < len(arg):
                    ratio = kwargs[arg]
                    last_arg_size = len(arg)

            m.add_patches({k: kp[k]}, 1.0 - ratio, ratio)
        return (m, )

def save_checkpoint(model, clip=None, vae=None, clip_vision=None, filename_prefix=None, output_dir=None, prompt=None, extra_pnginfo=None):
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir)
    prompt_info = ""
    if prompt is not None:
        prompt_info = json.dumps(prompt)

    metadata = {}

    enable_modelspec = True
    if isinstance(model.model, comfy.model_base.SDXL):
        if isinstance(model.model, comfy.model_base.SDXL_instructpix2pix):
            metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-edit"
        else:
            metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-base"
    elif isinstance(model.model, comfy.model_base.SDXLRefiner):
        metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-refiner"
    elif isinstance(model.model, comfy.model_base.SVD_img2vid):
        metadata["modelspec.architecture"] = "stable-video-diffusion-img2vid-v1"
    elif isinstance(model.model, comfy.model_base.SD3):
        metadata["modelspec.architecture"] = "stable-diffusion-v3-medium" #TODO: other SD3 variants
    else:
        enable_modelspec = False

    if enable_modelspec:
        metadata["modelspec.sai_model_spec"] = "1.0.0"
        metadata["modelspec.implementation"] = "sgm"
        metadata["modelspec.title"] = "{} {}".format(filename, counter)

    #TODO:
    # "stable-diffusion-v1", "stable-diffusion-v1-inpainting", "stable-diffusion-v2-512",
    # "stable-diffusion-v2-768-v", "stable-diffusion-v2-unclip-l", "stable-diffusion-v2-unclip-h",
    # "v2-inpainting"

    extra_keys = {}
    model_sampling = model.get_model_object("model_sampling")
    if isinstance(model_sampling, comfy.model_sampling.ModelSamplingContinuousEDM):
        if isinstance(model_sampling, comfy.model_sampling.V_PREDICTION):
            extra_keys["edm_vpred.sigma_max"] = torch.tensor(model_sampling.sigma_max).float()
            extra_keys["edm_vpred.sigma_min"] = torch.tensor(model_sampling.sigma_min).float()

    if model.model.model_type == comfy.model_base.ModelType.EPS:
        metadata["modelspec.predict_key"] = "epsilon"
    elif model.model.model_type == comfy.model_base.ModelType.V_PREDICTION:
        metadata["modelspec.predict_key"] = "v"
        extra_keys["v_pred"] = torch.tensor([])
        if getattr(model_sampling, "zsnr", False):
            extra_keys["ztsnr"] = torch.tensor([])

    if not args.disable_metadata:
        metadata["prompt"] = prompt_info
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata[x] = json.dumps(extra_pnginfo[x])

    output_checkpoint = f"{filename}_{counter:05}_.safetensors"
    output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

    comfy.sd.save_checkpoint(output_checkpoint, model, clip, vae, clip_vision, metadata=metadata, extra_keys=extra_keys)

class CheckpointSave:
    SEARCH_ALIASES = ["save model", "export checkpoint", "merge save"]
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP",),
                              "vae": ("VAE",),
                              "filename_prefix": ("STRING", {"default": "checkpoints/ComfyUI"}),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "model/merging"

    def save(self, model, clip, vae, filename_prefix, prompt=None, extra_pnginfo=None):
        save_checkpoint(model, clip=clip, vae=vae, filename_prefix=filename_prefix, output_dir=self.output_dir, prompt=prompt, extra_pnginfo=extra_pnginfo)
        return {}

class CLIPSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip": ("CLIP",),
                              "filename_prefix": ("STRING", {"default": "clip/ComfyUI"}),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "model/merging"

    def save(self, clip, filename_prefix, prompt=None, extra_pnginfo=None):
        prompt_info = ""
        if prompt is not None:
            prompt_info = json.dumps(prompt)

        metadata = {}
        if not args.disable_metadata:
            metadata["format"] = "pt"
            metadata["prompt"] = prompt_info
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        clip.load_model()
        clip_sd = clip.state_dict_for_saving()

        for prefix in ["clip_l.", "clip_g.", "clip_h.", "t5xxl.", "pile_t5xl.", "mt5xl.", "umt5xxl.", "t5base.", "gemma2_2b.", "llama.", "hydit_clip.", ""]:
            k = list(filter(lambda a: a.startswith(prefix), clip_sd.keys()))
            current_clip_sd = {}
            for x in k:
                current_clip_sd[x] = clip_sd.pop(x)
            if len(current_clip_sd) == 0:
                continue

            p = prefix[:-1]
            replace_prefix = {}
            filename_prefix_ = filename_prefix
            if len(p) > 0:
                filename_prefix_ = "{}_{}".format(filename_prefix_, p)
                replace_prefix[prefix] = ""
            replace_prefix["transformer."] = ""

            full_output_folder, filename, counter, subfolder, filename_prefix_ = folder_paths.get_save_image_path(filename_prefix_, self.output_dir)

            output_checkpoint = f"{filename}_{counter:05}_.safetensors"
            output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

            current_clip_sd = comfy.utils.state_dict_prefix_replace(current_clip_sd, replace_prefix)

            comfy.utils.save_torch_file(current_clip_sd, output_checkpoint, metadata=metadata)
        return {}

class VAESave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "vae": ("VAE",),
                              "filename_prefix": ("STRING", {"default": "vae/ComfyUI_vae"}),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "model/merging"

    def save(self, vae, filename_prefix, prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        prompt_info = ""
        if prompt is not None:
            prompt_info = json.dumps(prompt)

        metadata = {}
        if not args.disable_metadata:
            metadata["prompt"] = prompt_info
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

        comfy.utils.save_torch_file(vae.get_sd(), output_checkpoint, metadata=metadata)
        return {}

class ModelSave:
    SEARCH_ALIASES = ["export model", "checkpoint save"]
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "filename_prefix": ("STRING", {"default": "diffusion_models/ComfyUI"}),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "model/merging"

    def save(self, model, filename_prefix, prompt=None, extra_pnginfo=None):
        save_checkpoint(model, filename_prefix=filename_prefix, output_dir=self.output_dir, prompt=prompt, extra_pnginfo=extra_pnginfo)
        return {}

NODE_CLASS_MAPPINGS = {
    "ModelMergeSimple": ModelMergeSimple,
    "ModelMergeBlocks": ModelMergeBlocks,
    "ModelMergeSubtract": ModelSubtract,
    "ModelMergeAdd": ModelAdd,
    "CheckpointSave": CheckpointSave,
    "CLIPMergeSimple": CLIPMergeSimple,
    "CLIPMergeSubtract": CLIPSubtract,
    "CLIPMergeAdd": CLIPAdd,
    "CLIPSave": CLIPSave,
    "VAESave": VAESave,
    "ModelSave": ModelSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointSave": "Save Checkpoint",
}
