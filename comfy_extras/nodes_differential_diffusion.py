# code adapted from https://github.com/exx8/differential-diffusion

from typing_extensions import override

import torch
from comfy_api.latest import ComfyExtension, io


class DifferentialDiffusion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DifferentialDiffusion",
            search_aliases=["inpaint gradient", "variable denoise strength"],
            display_name="Differential Diffusion",
            category="experimental",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                ),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, strength=1.0) -> io.NodeOutput:
        model = model.clone()
        model.set_model_denoise_mask_function(lambda *args, **kwargs: cls.forward(*args, **kwargs, strength=strength))
        return io.NodeOutput(model)

    @classmethod
    def forward(cls, sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict, strength: float):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = model.inner_model.model_sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]

        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])

        threshold = (current_ts - ts_to) / (ts_from - ts_to)

        # Generate the binary mask based on the threshold
        binary_mask = (denoise_mask >= threshold).to(denoise_mask.dtype)

        # Blend binary mask with the original denoise_mask using strength
        if strength and strength < 1:
            blended_mask = strength * binary_mask + (1 - strength) * denoise_mask
            return blended_mask
        else:
            return binary_mask


class DifferentialDiffusionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DifferentialDiffusion,
        ]


async def comfy_entrypoint() -> DifferentialDiffusionExtension:
    return DifferentialDiffusionExtension()
