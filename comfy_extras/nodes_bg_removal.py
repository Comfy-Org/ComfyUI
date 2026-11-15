import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO
from comfy.bg_removal_model import load


class LoadBackgroundRemovalModel(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        files = folder_paths.get_filename_list("background_removal")
        return IO.Schema(
            node_id="LoadBackgroundRemovalModel",
            display_name="Load Background Removal Model",
            category="model/loaders",
            inputs=[
                IO.Combo.Input("bg_removal_name", options=sorted(files), tooltip="The model used to remove backgrounds from images"),
            ],
            outputs=[
                IO.BackgroundRemoval.Output("bg_model")
            ]
        )
    @classmethod
    def execute(cls, bg_removal_name):
        path = folder_paths.get_full_path_or_raise("background_removal", bg_removal_name)
        bg = load(path)
        if bg is None:
            raise RuntimeError("ERROR: background model file is invalid and does not contain a valid background removal model.")
        return IO.NodeOutput(bg)

class RemoveBackground(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RemoveBackground",
            display_name="Remove Background",
            category="image/background removal",
            description="Generates a foreground mask to remove the background from an image using a background removal model.",
            inputs=[
                IO.BackgroundRemoval.Input("bg_removal_model", tooltip="Background removal model used to generate the mask"),
                IO.Image.Input("image", tooltip="Input image to remove the background from")
            ],
            outputs=[
                IO.Mask.Output("mask", tooltip="Generated foreground mask")
            ]
        )
    @classmethod
    def execute(cls, bg_removal_model, image):
        mask = bg_removal_model.encode_image(image)
        return IO.NodeOutput(mask)

class BackgroundRemovalExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            LoadBackgroundRemovalModel,
            RemoveBackground
        ]


async def comfy_entrypoint() -> BackgroundRemovalExtension:
    return BackgroundRemovalExtension()
