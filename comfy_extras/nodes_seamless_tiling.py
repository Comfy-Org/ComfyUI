from collections.abc import Callable

import torch
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


TILING_MODES = ["x", "y", "x and y"]


def _padding(mode: str, context_size: int) -> tuple[int, int]:
    return (context_size if mode != "x" else 0, context_size if mode != "y" else 0)


def _periodic_pad_2d(x: torch.Tensor, pad_y: int, pad_x: int) -> torch.Tensor:
    if x.ndim not in (4, 5):
        raise ValueError("Seamless tiling currently supports image latents only.")

    if pad_y:
        y = torch.arange(-pad_y, x.shape[-2] + pad_y, device=x.device) % x.shape[-2]
        x = x.index_select(-2, y)
    if pad_x:
        x_indices = torch.arange(-pad_x, x.shape[-1] + pad_x, device=x.device) % x.shape[-1]
        x = x.index_select(-1, x_indices)
    return x


def _seam_blend(length: int, width: int, x: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(length, device=x.device, dtype=x.dtype)
    return 1.0 - positions.minimum(length - positions - 1).div(width).clamp(max=1)


class ModelPatchSeamlessTiling(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ModelPatchSeamlessTiling",
            display_name="Seamless Tiling",
            category="model/patch",
            description="Blends cyclically shifted denoiser predictions for seamless image tiling. Runs the model twice per step for one axis or four times for both axes.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("tiling", options=TILING_MODES, default="x and y"),
                io.Int.Input(
                    "context_size",
                    default=8,
                    min=1,
                    max=256,
                    step=1,
                    tooltip="Width of the seam blend on each enabled side, in latent pixels.",
                ),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model: io.Model.Type, tiling: str, context_size: int) -> io.NodeOutput:
        model = model.clone()
        old_wrapper = model.model_options.get("model_function_wrapper")
        blend_y_width, blend_x_width = _padding(tiling, context_size)

        def model_function_wrapper(apply_model: Callable, args: dict) -> torch.Tensor:
            if args["c"].get("control") is not None:
                raise ValueError("Seamless Tiling cannot be used with ControlNet because its feature maps cannot be shifted with the latent.")

            x = args["input"]
            height, width = x.shape[-2:]
            y_shifts = (0, height // 2) if blend_y_width else (0,)
            x_shifts = (0, width // 2) if blend_x_width else (0,)
            blend_y = _seam_blend(height, blend_y_width, x) if blend_y_width else torch.zeros(height, device=x.device, dtype=x.dtype)
            blend_x = _seam_blend(width, blend_x_width, x) if blend_x_width else torch.zeros(width, device=x.device, dtype=x.dtype)

            result = None
            for shift_y in y_shifts:
                weight_y = blend_y if shift_y else 1.0 - blend_y
                for shift_x in x_shifts:
                    weight_x = blend_x if shift_x else 1.0 - blend_x
                    shifts = (shift_y, shift_x)
                    c = args["c"].copy()
                    c_concat = c.get("c_concat")
                    if c_concat is not None:
                        c["c_concat"] = torch.roll(c_concat, shifts, dims=(-2, -1))

                    shifted = torch.roll(x, shifts, dims=(-2, -1))
                    wrapped_args = args | {"input": shifted, "c": c}
                    if old_wrapper is not None:
                        output = old_wrapper(apply_model, wrapped_args)
                    else:
                        output = apply_model(shifted, args["timestep"], **c)
                    output = torch.roll(output, (-shift_y, -shift_x), dims=(-2, -1))

                    weight = (weight_y[:, None] * weight_x[None, :]).reshape([1] * (output.ndim - 2) + [height, width])
                    if result is None:
                        result = output * weight
                    else:
                        result.addcmul_(output, weight)
            return result

        model.set_model_unet_function_wrapper(model_function_wrapper)
        return io.NodeOutput(model)


class VAEDecodeSeamlessTiling(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VAEDecodeSeamlessTiling",
            display_name="Seamless Tiling VAE Decode",
            category="model/latent",
            description="Decodes an image latent with periodic context and crops the result back to its original size.",
            inputs=[
                io.Latent.Input("samples"),
                io.Vae.Input("vae"),
                io.Combo.Input("tiling", options=TILING_MODES, default="x and y"),
                io.Int.Input(
                    "context_size",
                    default=8,
                    min=1,
                    max=256,
                    step=1,
                    tooltip="Periodic context on each enabled side, in latent pixels. Larger values use more VRAM.",
                ),
            ],
            outputs=[io.Image.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, samples: io.Latent.Type, vae: io.Vae.Type, tiling: str, context_size: int) -> io.NodeOutput:
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]

        height, width = latent.shape[-2:]
        pad_y, pad_x = _padding(tiling, context_size)
        padded = _periodic_pad_2d(latent, pad_y, pad_x)
        images = vae.decode(padded)

        scale_y = images.shape[-3] / padded.shape[-2]
        scale_x = images.shape[-2] / padded.shape[-1]
        top = round(pad_y * scale_y)
        left = round(pad_x * scale_x)
        images = images[..., top:top + round(height * scale_y), left:left + round(width * scale_x), :]
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return io.NodeOutput(images)


class SeamlessTilingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ModelPatchSeamlessTiling, VAEDecodeSeamlessTiling]


async def comfy_entrypoint() -> SeamlessTilingExtension:
    return SeamlessTilingExtension()
