from typing_extensions import override
from typing import Callable

import torch

import comfy.model_management
from comfy_api.latest import ComfyExtension, io

import nodes

class EmptyChromaRadianceLatentImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EmptyChromaRadianceLatentImage",
            category="model/latent/chroma radiance",
            inputs=[
                io.Int.Input(id="width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input(id="height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input(id="batch_size", default=1, min=1, max=4096),
            ],
            outputs=[io.Latent().Output()],
        )

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int=1) -> io.NodeOutput:
        latent = torch.zeros((batch_size, 3, height, width), device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples":latent})


class ChromaRadianceOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ChromaRadianceOptions",
            category="model/patch/chroma radiance",
            description="Allows setting advanced options for the Chroma Radiance model.",
            inputs=[
                io.Model.Input(id="model"),
                io.Boolean.Input(
                    id="preserve_wrapper",
                    default=True,
                    tooltip="When enabled, will delegate to an existing model function wrapper if it exists. Generally should be left enabled.",
                ),
                io.Float.Input(
                    id="start_sigma",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    tooltip="First sigma that these options will be in effect.",
                    advanced=True,
                ),
                io.Float.Input(
                    id="end_sigma",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    tooltip="Last sigma that these options will be in effect.",
                    advanced=True,
                ),
                io.Int.Input(
                    id="nerf_tile_size",
                    default=-1,
                    min=-1,
                    tooltip="Allows overriding the default NeRF tile size. -1 means use the default (32). 0 means use non-tiling mode (may require a lot of VRAM).",
                    advanced=True,
                ),
                io.Boolean.Input(
                    id="force_sequential_txt_ids",
                    default=False,
                    tooltip="Force usage of sequential text token IDs instead of zeroes. Should be used for checkpoints from 2026-05-22 to 2026-06-01 that are trained in this way but do not contain the __sequential__ key in the state dict.",
                    advanced=True,
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        *,
        model: io.Model.Type,
        preserve_wrapper: bool,
        start_sigma: float,
        end_sigma: float,
        nerf_tile_size: int,
        force_sequential_txt_ids: bool,
    ) -> io.NodeOutput:
        radiance_options = {}
        if nerf_tile_size >= 0:
            radiance_options["nerf_tile_size"] = nerf_tile_size

        if force_sequential_txt_ids:
            radiance_options["use_sequential_txt_ids"] = True

        if not radiance_options:
            return io.NodeOutput(model)

        old_wrapper = model.model_options.get("model_function_wrapper")

        def model_function_wrapper(apply_model: Callable, args: dict) -> torch.Tensor:
            c = args["c"].copy()
            sigma = args["timestep"].max().detach().cpu().item()
            if end_sigma <= sigma <= start_sigma:
                transformer_options = c.get("transformer_options", {}).copy()
                transformer_options["chroma_radiance_options"] = radiance_options.copy()
                c["transformer_options"] = transformer_options
            if not (preserve_wrapper and old_wrapper):
                return apply_model(args["input"], args["timestep"], **c)
            return old_wrapper(apply_model, args | {"c": c})

        model = model.clone()
        model.set_model_unet_function_wrapper(model_function_wrapper)
        return io.NodeOutput(model)


class ChromaRadianceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            EmptyChromaRadianceLatentImage,
            ChromaRadianceOptions,
        ]


async def comfy_entrypoint() -> ChromaRadianceExtension:
    return ChromaRadianceExtension()
