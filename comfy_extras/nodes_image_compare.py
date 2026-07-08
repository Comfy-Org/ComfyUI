import nodes

from typing_extensions import override
from comfy_api.latest import IO, ComfyExtension


class ImageCompare(IO.ComfyNode):
    """Compares two images with a slider interface."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageCompare",
            display_name="Compare Images",
            description="Compares two images side by side with a slider.",
            category="image",
            essentials_category="Image Tools",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                IO.Image.Input("image_a", optional=True),
                IO.Image.Input("image_b", optional=True),
                IO.ImageCompare.Input("compare_view"),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, image_a=None, image_b=None, compare_view=None) -> IO.NodeOutput:
        result = {"a_images": [], "b_images": []}

        preview_node = nodes.PreviewImage()

        if image_a is not None and len(image_a) > 0:
            saved = preview_node.save_images(image_a, "comfy.compare.a")
            result["a_images"] = saved["ui"]["images"]

        if image_b is not None and len(image_b) > 0:
            saved = preview_node.save_images(image_b, "comfy.compare.b")
            result["b_images"] = saved["ui"]["images"]

        return IO.NodeOutput(ui=result)


class ImageCompareExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ImageCompare,
        ]


async def comfy_entrypoint() -> ImageCompareExtension:
    return ImageCompareExtension()
