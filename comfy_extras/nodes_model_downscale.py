from typing_extensions import override
import comfy.utils
from comfy_api.latest import ComfyExtension, io


class PatchModelAddDownscale(io.ComfyNode):
    UPSCALE_METHODS = ["bicubic", "nearest-exact", "bilinear", "area", "bislerp"]
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PatchModelAddDownscale",
            display_name="PatchModelAddDownscale (Kohya Deep Shrink)",
            category="model/patch/unet",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("block_number", default=3, min=1, max=32, step=1, advanced=True),
                io.Float.Input("downscale_factor", default=2.0, min=0.1, max=9.0, step=0.001),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Float.Input("end_percent", default=0.35, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Boolean.Input("downscale_after_skip", default=True, advanced=True),
                io.Combo.Input("downscale_method", options=cls.UPSCALE_METHODS),
                io.Combo.Input("upscale_method", options=cls.UPSCALE_METHODS),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, block_number, downscale_factor, start_percent, end_percent, downscale_after_skip, downscale_method, upscale_method) -> io.NodeOutput:
        model_sampling = model.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_percent)
        sigma_end = model_sampling.percent_to_sigma(end_percent)

        def input_block_patch(h, transformer_options):
            if transformer_options["block"][1] == block_number:
                sigma = transformer_options["sigmas"][0].item()
                if sigma <= sigma_start and sigma >= sigma_end:
                    h = comfy.utils.common_upscale(h, round(h.shape[-1] * (1.0 / downscale_factor)), round(h.shape[-2] * (1.0 / downscale_factor)), downscale_method, "disabled")
            return h

        def output_block_patch(h, hsp, transformer_options):
            if h.shape[2] != hsp.shape[2]:
                h = comfy.utils.common_upscale(h, hsp.shape[-1], hsp.shape[-2], upscale_method, "disabled")
            return h, hsp

        m = model.clone()
        if downscale_after_skip:
            m.set_model_input_block_patch_after_skip(input_block_patch)
        else:
            m.set_model_input_block_patch(input_block_patch)
        m.set_model_output_block_patch(output_block_patch)
        return io.NodeOutput(m)


class ModelDownscaleExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            PatchModelAddDownscale,
        ]


async def comfy_entrypoint() -> ModelDownscaleExtension:
    return ModelDownscaleExtension()
