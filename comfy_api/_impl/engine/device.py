"""Core engine extension: the devices and dtypes ComfyUI computes with."""


class EngineOps:
    def info(self):
        import comfy.model_management as mm

        return {
            "torch_device": str(mm.get_torch_device()),
            "intermediate_device": str(mm.intermediate_device()),
            "intermediate_dtype": str(mm.intermediate_dtype()).removeprefix("torch."),
            "unet_dtype": str(mm.unet_dtype()).removeprefix("torch."),
            "offload_device": str(mm.unet_offload_device()),
            "vae_device": str(mm.vae_device()),
            "vae_dtype": str(mm.vae_dtype()).removeprefix("torch."),
            "text_encoder_device": str(mm.text_encoder_device()),
            "vae_offload_device": str(mm.vae_offload_device()),
            "text_encoder_offload_device": str(mm.text_encoder_offload_device()),
        }
