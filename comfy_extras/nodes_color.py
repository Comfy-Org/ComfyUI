from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_extras.color_util import hex_to_rgb


class ColorToRGBInt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ColorToRGBInt",
            display_name="Color Picker",
            category="utilities",
            description="Return a color RGB integer value and hexadecimal representation.",
            inputs=[
                io.Color.Input("color"),
            ],
            outputs=[
                io.Int.Output(display_name="rgb_int"),
                io.Color.Output(display_name="hex")
            ],
        )

    @classmethod
    def execute(cls, color: str) -> io.NodeOutput:
        # expect format #RRGGBB
        if len(color) != 7 or color[0] != "#":
            raise ValueError("Color must be in format #RRGGBB")
        try:
            int(color[1:], 16)
        except ValueError:
            raise ValueError("Color must be in format #RRGGBB") from None
        r, g, b = hex_to_rgb(color)

        rgb_int = r * 256 * 256 + g * 256 + b
        return io.NodeOutput(rgb_int, color)


class ColorExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ColorToRGBInt]


async def comfy_entrypoint() -> ColorExtension:
    return ColorExtension()
