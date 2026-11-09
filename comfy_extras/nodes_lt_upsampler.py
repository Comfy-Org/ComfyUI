from comfy import model_management
from comfy_api.latest import ComfyExtension, IO
from typing_extensions import override
import math


class LTXVLatentUpsampler(IO.ComfyNode):
    """
    Upsamples a video latent by a factor of 2.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LTXVLatentUpsampler",
            category="model/latent/ltxv",
            is_experimental=True,
            inputs=[
                IO.Latent.Input("samples"),
                IO.LatentUpscaleModel.Input("upscale_model"),
                IO.Vae.Input("vae"),
            ],
            outputs=[
                IO.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples, upscale_model, vae) -> IO.NodeOutput:
        """
        Upsample the input latent using the provided model.

        Args:
            samples (dict): Input latent samples
            upscale_model (LatentUpsampler): Loaded upscale model
            vae: VAE model for normalization

        Returns:
            tuple: Tuple containing the upsampled latent
        """
        device = upscale_model.load_device
        model = upscale_model.model
        model_dtype = upscale_model.model_dtype()
        latents = samples["samples"]
        input_dtype = latents.dtype

        memory_required = math.prod(latents.shape) * 3000.0  # TODO: more accurate
        model_management.load_models_gpu([upscale_model], memory_required=memory_required)

        latents = latents.to(dtype=model_dtype, device=device)

        """Upsample latents without tiling."""
        latents = vae.first_stage_model.per_channel_statistics.un_normalize(latents)
        upsampled_latents = model(latents)

        upsampled_latents = vae.first_stage_model.per_channel_statistics.normalize(
            upsampled_latents
        )
        upsampled_latents = upsampled_latents.to(dtype=input_dtype, device=model_management.intermediate_device())
        return_dict = samples.copy()
        return_dict["samples"] = upsampled_latents
        return_dict.pop("noise_mask", None)
        return IO.NodeOutput(return_dict)

    upsample_latent = execute  # TODO: remove


class LTXVLatentUpsamplerExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [LTXVLatentUpsampler]


async def comfy_entrypoint() -> LTXVLatentUpsamplerExtension:
    return LTXVLatentUpsamplerExtension()
