import folder_paths
import comfy.utils
import comfy.sd


class LoraLoaderBypass:
    """
    Apply LoRA in bypass mode without modifying base model weights.

    Bypass mode computes: output = base_forward(x) + lora_path(x)
    This is useful for training and when model weights are offloaded.
    """

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"

    CATEGORY = "model/loaders"
    DESCRIPTION = "Apply LoRA in bypass mode. Unlike regular LoRA, this doesn't modify model weights - instead it injects the LoRA computation during forward pass. Useful for training scenarios."
    EXPERIMENTAL = True

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_bypass_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)


class LoraLoaderBypassModelOnly(LoraLoaderBypass):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "lora_name": (folder_paths.get_filename_list("loras"), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora_model_only"

    def load_lora_model_only(self, model, lora_name, strength_model):
        return (self.load_lora(model, None, lora_name, strength_model, 0)[0],)


NODE_CLASS_MAPPINGS = {
    "LoraLoaderBypass": LoraLoaderBypass,
    "LoraLoaderBypassModelOnly": LoraLoaderBypassModelOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraLoaderBypass": "Load LoRA (Bypass) (For debugging)",
    "LoraLoaderBypassModelOnly": "Load LoRA (Bypass, Model Only) (for debugging)",
}
