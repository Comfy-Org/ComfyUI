import nodes
import torch
import comfy.utils
import comfy.sd
import folder_paths
import comfy_extras.nodes_model_merging
import node_helpers


class ImageOnlyCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             }}
    RETURN_TYPES = ("MODEL", "CLIP_VISION", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "model/loaders"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=False, output_clipvision=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return (out[0], out[3], out[2])


class SVD_img2vid_Conditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_vision": ("CLIP_VISION",),
                              "init_image": ("IMAGE",),
                              "vae": ("VAE",),
                              "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 576, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                              "video_frames": ("INT", {"default": 14, "min": 1, "max": 4096}),
                              "motion_bucket_id": ("INT", {"default": 127, "min": 1, "max": 1023, "advanced": True}),
                              "fps": ("INT", {"default": 6, "min": 1, "max": 1024}),
                              "augmentation_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01, "advanced": True})
                             }}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    FUNCTION = "encode"

    CATEGORY = "model/conditioning/stable video"

    def encode(self, clip_vision, init_image, vae, width, height, video_frames, motion_bucket_id, fps, augmentation_level):
        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = comfy.utils.common_upscale(init_image.movedim(-1,1), width, height, "bilinear", "center").movedim(1,-1)
        encode_pixels = pixels[:,:,:,:3]
        if augmentation_level > 0:
            encode_pixels += torch.randn_like(pixels) * augmentation_level
        t = vae.encode(encode_pixels)
        positive = [[pooled, {"motion_bucket_id": motion_bucket_id, "fps": fps, "augmentation_level": augmentation_level, "concat_latent_image": t}]]
        negative = [[torch.zeros_like(pooled), {"motion_bucket_id": motion_bucket_id, "fps": fps, "augmentation_level": augmentation_level, "concat_latent_image": torch.zeros_like(t)}]]
        latent = torch.zeros([video_frames, 4, height // 8, width // 8])
        return (positive, negative, {"samples":latent})

class VideoLinearCFGGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "min_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01, "advanced": True}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model/sampling/guiders"

    def patch(self, model, min_cfg):
        def linear_cfg(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]

            scale = torch.linspace(min_cfg, cond_scale, cond.shape[0], device=cond.device).reshape((cond.shape[0], 1, 1, 1))
            return uncond + scale * (cond - uncond)

        m = model.clone()
        m.set_model_sampler_cfg_function(linear_cfg)
        return (m, )

class VideoTriangleCFGGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "min_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01, "advanced": True}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model/sampling/guiders"

    def patch(self, model, min_cfg):
        def linear_cfg(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            period = 1.0
            values = torch.linspace(0, 1, cond.shape[0], device=cond.device)
            values = 2 * (values / period - torch.floor(values / period + 0.5)).abs()
            scale = (values * (cond_scale - min_cfg) + min_cfg).reshape((cond.shape[0], 1, 1, 1))

            return uncond + scale * (cond - uncond)

        m = model.clone()
        m.set_model_sampler_cfg_function(linear_cfg)
        return (m, )

class ImageOnlyCheckpointSave(comfy_extras.nodes_model_merging.CheckpointSave):
    CATEGORY = "model/merging"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip_vision": ("CLIP_VISION",),
                              "vae": ("VAE",),
                              "filename_prefix": ("STRING", {"default": "checkpoints/ComfyUI"}),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}

    def save(self, model, clip_vision, vae, filename_prefix, prompt=None, extra_pnginfo=None):
        comfy_extras.nodes_model_merging.save_checkpoint(model, clip_vision=clip_vision, vae=vae, filename_prefix=filename_prefix, output_dir=self.output_dir, prompt=prompt, extra_pnginfo=extra_pnginfo)
        return {}


class ConditioningSetAreaPercentageVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "width": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.01}),
                             "height": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.01}),
                             "temporal": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.01}),
                             "x": ("FLOAT", {"default": 0, "min": 0, "max": 1.0, "step": 0.01}),
                             "y": ("FLOAT", {"default": 0, "min": 0, "max": 1.0, "step": 0.01}),
                             "z": ("FLOAT", {"default": 0, "min": 0, "max": 1.0, "step": 0.01}),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "model/conditioning/transform"

    def append(self, conditioning, width, height, temporal, x, y, z, strength):
        c = node_helpers.conditioning_set_values(conditioning, {"area": ("percentage", temporal, height, width, z, y, x),
                                                                "strength": strength,
                                                                "set_area_to_bounds": False})
        return (c, )


NODE_CLASS_MAPPINGS = {
    "ImageOnlyCheckpointLoader": ImageOnlyCheckpointLoader,
    "SVD_img2vid_Conditioning": SVD_img2vid_Conditioning,
    "VideoLinearCFGGuidance": VideoLinearCFGGuidance,
    "VideoTriangleCFGGuidance": VideoTriangleCFGGuidance,
    "ImageOnlyCheckpointSave": ImageOnlyCheckpointSave,
    "ConditioningSetAreaPercentageVideo": ConditioningSetAreaPercentageVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageOnlyCheckpointLoader": "Load Checkpoint Image Only (img2vid model)",
    "VideoLinearCFGGuidance": "Video Linear CFG Guidance",
    "VideoTriangleCFGGuidance": "Video Triangle CFG Guidance",
    "ConditioningSetAreaPercentageVideo": "Conditioning (Set Area with Percentage for Video)",
}
