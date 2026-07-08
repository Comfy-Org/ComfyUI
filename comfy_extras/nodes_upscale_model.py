import logging
from spandrel import ModelLoader, ImageModelDescriptor
from comfy import model_management
import torch
import comfy.utils
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import comfy.model_management

try:
    from spandrel_extra_arches import EXTRA_REGISTRY
    from spandrel import MAIN_REGISTRY
    MAIN_REGISTRY.add(*EXTRA_REGISTRY)
    logging.info("Successfully imported spandrel_extra_arches: support for non commercial upscale models.")
except:
    pass

class UpscaleModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UpscaleModelLoader",
            display_name="Load Upscale Model",
            category="model/loaders",
            inputs=[
                io.Combo.Input("model_name", options=folder_paths.get_filename_list("upscale_models")),
            ],
            outputs=[
                io.UpscaleModel.Output(),
            ],
        )

    @classmethod
    def execute(cls, model_name) -> io.NodeOutput:
        model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
        out = ModelLoader().load_from_state_dict(sd).eval()

        if not isinstance(out, ImageModelDescriptor):
            raise Exception("Upscale model must be a single-image model.")

        return io.NodeOutput(out)

    load_model = execute  # TODO: remove


class ImageUpscaleWithModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageUpscaleWithModel",
            display_name="Upscale Image (using Model)",
            category="image/upscaling",
            search_aliases=["upscale", "upscaler", "upsc", "enlarge image", "super resolution", "hires", "superres", "increase resolution"],
            inputs=[
                io.UpscaleModel.Input("upscale_model"),
                io.Image.Input("image"),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, upscale_model, image) -> io.NodeOutput:
        device = model_management.get_torch_device()

        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0 #The 384.0 is an estimate of how much some of these models take, TODO: make it more accurate
        memory_required += image.nelement() * image.element_size()
        model_management.free_memory(memory_required, device)

        upscale_model.to(device)
        in_img = image.movedim(-1,-3).to(device)

        tile = 512
        overlap = 32

        output_device = comfy.model_management.intermediate_device()

        oom = True
        try:
            while oom:
                try:
                    steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                    pbar = comfy.utils.ProgressBar(steps)
                    s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a.float()), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar, output_device=output_device)
                    oom = False
                except Exception as e:
                    model_management.raise_non_oom(e)
                    tile //= 2
                    if tile < 128:
                        raise e
        finally:
            upscale_model.to("cpu")

        s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0).to(comfy.model_management.intermediate_dtype())
        return io.NodeOutput(s)

    upscale = execute  # TODO: remove


class UpscaleModelExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            UpscaleModelLoader,
            ImageUpscaleWithModel,
        ]


async def comfy_entrypoint() -> UpscaleModelExtension:
    return UpscaleModelExtension()
