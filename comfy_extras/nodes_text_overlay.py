import numpy as np
import torch
from PIL import Image as PILImage, ImageColor, ImageDraw, ImageFont
from typing_extensions import override

from comfy_api.latest import ComfyExtension, IO


class TextOverlay(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TextOverlay",
            display_name="Draw Text Overlay",
            category="text",
            description="Draw text overlay on an image or batch of images.",
            search_aliases=["text", "label", "caption", "subtitle", "watermark", "title", "addlabel", "overlay"],
            inputs=[
                IO.Image.Input("images"),
                IO.String.Input("text", multiline=True, default=""),
                IO.Float.Input("font_size", default=5.0, min=0.5, max=50.0, step=0.5, tooltip="Font size as a percentage of the image height."),
                IO.Color.Input("color", default="#ffffff", tooltip="Color of the text."),
                IO.Combo.Input("position", options=["top", "bottom"], default="top"),
                IO.Combo.Input("align", options=["left", "center", "right"], default="left"),
                IO.Boolean.Input("outline", default=True, tooltip="Draw a black outline around the text."),
            ],
            outputs=[IO.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, images, text, font_size, color, position, align, outline) -> IO.NodeOutput:
        if text.strip() == "":
            return IO.NodeOutput(images)

        text = text.replace("\\n", "\n").replace("\\t", "\t")

        text_rgba = cls.parse_color_to_rgba(color)
        outline_rgba = (0, 0, 0, 255) if outline else (0, 0, 0, 0)

        # Render the overlay once and composite it across all frames in the batch
        height = images.shape[1]
        width = images.shape[2]
        overlay_rgb, overlay_alpha = cls.render_overlay_text(width, height, text, position, align, font_size, text_rgba, outline_rgba)
        overlay_rgb = overlay_rgb.to(device=images.device, dtype=images.dtype)
        overlay_alpha = overlay_alpha.to(device=images.device, dtype=images.dtype)

        if images.shape[-1] == 4:
            # Source over, premultiplied, as in porter_duff_composite's SRC_OVER.
            rgb, dst_alpha = images[..., :3], images[..., 3:]
            out_alpha = overlay_alpha + (1.0 - overlay_alpha) * dst_alpha
            out_rgb = overlay_rgb * overlay_alpha + (1.0 - overlay_alpha) * rgb * dst_alpha
            out_rgb = torch.where(out_alpha > 1e-5, out_rgb / out_alpha, torch.zeros_like(out_rgb))
            return IO.NodeOutput(torch.cat((torch.clamp(out_rgb, 0, 1), out_alpha), dim=-1))

        result = images * (1.0 - overlay_alpha) + overlay_rgb * overlay_alpha
        return IO.NodeOutput(result)

    @staticmethod
    def parse_color_to_rgba(color_string):
        parsed = ImageColor.getrgb(color_string)

        if len(parsed) == 3:
            return (*parsed, 255)

        return parsed

    @classmethod
    def render_overlay_text(cls, width, height, text, position, align, font_size, text_rgba, outline_rgba):
        line_spacing = 1.2
        margin_percent = 1.0
        min_font_percent = 2.0
        min_font_pixels = 10
        outline_thickness_factor = 0.04

        # Draw onto a transparent layer so the result can be alpha-composited over any frame.
        layer = PILImage.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        margin = int(round(margin_percent / 100.0 * min(width, height)))
        max_width = max(1, width - 2 * margin)
        max_height = max(1, height - 2 * margin)

        # Font scales with resolution, then shrinks to fit the height.
        size = max(1, int(round(font_size / 100.0 * height)))
        floor = min(size, max(min_font_pixels, int(round(min_font_percent / 100.0 * height))))

        while True:
            font = ImageFont.load_default(size=size)
            stroke = max(1, int(round(size * outline_thickness_factor))) if outline_rgba[3] > 0 else 0
            block = "\n".join(cls.wrap_text(text, font, max_width))
            # convert line spacing to pixel spacing
            single = draw.textbbox((0, 0), "Ay", font=font, stroke_width=stroke)
            double = draw.multiline_textbbox((0, 0), "Ay\nAy", font=font, spacing=0, stroke_width=stroke)
            natural_advance = (double[3] - double[1]) - (single[3] - single[1])
            pixel_spacing = int(round(size * line_spacing - natural_advance))
            box = draw.multiline_textbbox((0, 0), block, font=font, spacing=pixel_spacing, stroke_width=stroke)
            block_height = box[3] - box[1]

            if block_height <= max_height or size <= floor:
                break

            size = max(floor, int(size * 0.9))

        anchor_h, x = {"left": ("l", margin), "center": ("m", width / 2), "right": ("r", width - margin)}[align]

        # Offset y so the rendered text sits flush against the margin
        if position == "bottom":
            y = height - margin - box[3]
        else:
            y = margin - box[1]

        draw.multiline_text((x, y), block, font=font, fill=text_rgba, anchor=anchor_h + "a",
                            align=align, spacing=pixel_spacing, stroke_width=stroke, stroke_fill=outline_rgba)

        overlay = np.array(layer).astype(np.float32) / 255.0
        overlay_rgb = torch.from_numpy(overlay[:, :, :3])
        overlay_alpha = torch.from_numpy(overlay[:, :, 3:4])
        return overlay_rgb, overlay_alpha

    @staticmethod
    def wrap_text(text, font, max_width):
        lines = []
        for raw_line in text.split("\n"):
            words = raw_line.split()
            if not words:
                lines.append("")
                continue
            current = ""
            # Break the line into words and split words that are too long
            for word in words:
                while font.getlength(word) > max_width and len(word) > 1:
                    cut = 1
                    while cut < len(word) and font.getlength(word[:cut + 1]) <= max_width:
                        cut += 1
                    if current:
                        lines.append(current)
                        current = ""
                    lines.append(word[:cut])
                    word = word[cut:]
                candidate = word if not current else current + " " + word
                if not current or font.getlength(candidate) <= max_width:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            if current:
                lines.append(current)
        return lines


class TextOverlayExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [TextOverlay]


async def comfy_entrypoint() -> TextOverlayExtension:
    return TextOverlayExtension()
