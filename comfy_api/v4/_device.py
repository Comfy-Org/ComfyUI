from typing import TypedDict

from comfy_api.internal.singleton import ProxiedSingleton


class DeviceInfo(TypedDict):
    torch_device: str
    intermediate_device: str
    intermediate_dtype: str
    unet_dtype: str
    offload_device: str
    vae_device: str
    vae_dtype: str
    text_encoder_device: str
    vae_offload_device: str
    text_encoder_offload_device: str


class Device(ProxiedSingleton):
    async def info(self) -> DeviceInfo:
        """Return ComfyUI's device choices and dtype names without the `torch.` prefix.

        These are placement defaults, not a reservation of device memory. `unet_dtype` is the
        choice for a model that runs in float16, bfloat16 or float32, and `vae_dtype` is float32,
        unless a launch flag such as `--fp8_e4m3fn-unet` sets them; a loaded model or VAE may use
        another dtype.
        """
        from comfy import model_management

        return {
            "torch_device": str(model_management.get_torch_device()),
            "intermediate_device": str(model_management.intermediate_device()),
            "intermediate_dtype": str(
                model_management.intermediate_dtype()
            ).removeprefix("torch."),
            "unet_dtype": str(model_management.unet_dtype()).removeprefix("torch."),
            "offload_device": str(model_management.unet_offload_device()),
            "vae_device": str(model_management.vae_device()),
            "vae_dtype": str(model_management.vae_dtype()).removeprefix("torch."),
            "text_encoder_device": str(model_management.text_encoder_device()),
            "vae_offload_device": str(model_management.vae_offload_device()),
            "text_encoder_offload_device": str(
                model_management.text_encoder_offload_device()
            ),
        }
