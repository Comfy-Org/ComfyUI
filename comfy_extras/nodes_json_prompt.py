from typing_extensions import override

from comfy_api.latest import ComfyExtension, io
from comfy_extras.color_util import normalize_palette


class BuildJsonPromptIdeogram(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        color_palette = io.Colors.Input(
            "color_palette",
            socketless=False,
            tooltip="Hex color codes that steer the image's dominant colors. Up to 16 entries.",
        )
        return io.Schema(
            node_id="BuildJsonPromptIdeogram",
            display_name="Build JSON Prompt (Ideogram)",
            category="text",
            description="Build a JSON prompt for the Ideogram 4 model.",
            inputs=[
                io.Array.Input("element", tooltip="Prompt elements from the node Create Bounding Boxes."),
                io.String.Input("high_level_description", multiline=True, default="",
                                tooltip="Optional description of the image in one or two sentences. Strongly recommended."),
                io.String.Input("background", multiline=True, default="",
                                tooltip="Mandatory description of the image background or environment."),
                io.DynamicCombo.Input("style", options=[
                    io.DynamicCombo.Option("none", []),
                    io.DynamicCombo.Option("photo", [io.String.Input("photo", default="", tooltip="Camera or lens details for photographic outputs (e.g. 35mm, f/1.4, bokeh).")]),
                    io.DynamicCombo.Option("art_style", [io.String.Input("art_style", default="", tooltip="Art style description (e.g. flat vector illustration, bold outlines).")]),
                ]),
                io.String.Input("aesthetics", default="", tooltip="Mandatory aesthetic keywords (e.g. moody, cinematic, desaturated)."),
                io.String.Input("lighting", default="", tooltip="Mandatory lighting description (e.g. golden hour, rim light, dramatic shadows)."),
                io.String.Input("medium", default="", tooltip="Mandatory medium type (e.g. photograph, illustration, 3d_render, painting, graphic_design). When style = photo, set to photograph."),
                color_palette,
            ],
            outputs=[io.Dict.Output(display_name="prompt")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, element, style, high_level_description="", background="",
                aesthetics="", lighting="", medium="", color_palette=None) -> io.NodeOutput:
        elements = element if isinstance(element, list) else []
        kind = style.get("style", "none") if isinstance(style, dict) else "none"
        photo = style.get("photo", "") if isinstance(style, dict) else ""
        art_style = style.get("art_style", "") if isinstance(style, dict) else ""
        palette = normalize_palette(color_palette or [])

        caption: dict = {}
        if high_level_description.strip():
            caption["high_level_description"] = high_level_description
        if kind != "none":
            style_desc: dict = {"aesthetics": aesthetics, "lighting": lighting}
            if kind == "photo":
                style_desc["photo"] = photo
                style_desc["medium"] = medium
            else:
                style_desc["medium"] = medium
                style_desc["art_style"] = art_style
            if palette:
                style_desc["color_palette"] = palette
            caption["style_description"] = style_desc
        caption["compositional_deconstruction"] = {
            "background": background,
            "elements": elements,
        }
        return io.NodeOutput(caption)


class JsonPromptExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [BuildJsonPromptIdeogram]


async def comfy_entrypoint() -> JsonPromptExtension:
    return JsonPromptExtension()
