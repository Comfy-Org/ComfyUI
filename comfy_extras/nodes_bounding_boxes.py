import json

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io
from comfy_extras.color_util import hex_to_rgb, normalize_palette, readable_color

_PREVIEW_LONG_EDGE = 1024
_PREVIEW_DIM = 0.25


def pixels_to_fractions(box: dict, width: int, height: int) -> dict:
    w = width or 1
    h = height or 1
    return {
        "x": box.get("x", 0) / w,
        "y": box.get("y", 0) / h,
        "w": box.get("width", 0) / w,
        "h": box.get("height", 0) / h,
    }


def fractions_to_pixels(box: dict, width: int, height: int) -> dict:
    x, y = box.get("x", 0.0), box.get("y", 0.0)
    w, h = box.get("w", 0.0), box.get("h", 0.0)
    if w < 0:
        x, w = x + w, -w
    if h < 0:
        y, h = y + h, -h
    return {
        "x": round(x * width),
        "y": round(y * height),
        "width": round(w * width),
        "height": round(h * height),
    }


def fractions_to_bbox_frame(boxes: list, width: int, height: int) -> list:
    pixels = [
        fractions_to_pixels(box, width, height)
        for box in boxes
        if isinstance(box, dict)
    ]
    return [pixels] if pixels else []


def _font(size: int):
    try:
        return ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default()


def _wrap(draw, text: str, font, max_w: float) -> list[str]:
    lines = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            test = word if not line else line + " " + word
            if line and draw.textlength(test, font=font) > max_w:
                lines.append(line)
                line = word
            else:
                line = test
        lines.append(line)
    return lines


def _bg_from_image(image) -> Image.Image | None:
    if image is None:
        return None
    try:
        arr = (image[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    except Exception:
        return None


def render_preview(regions, width, height, bg=None):
    if bg is not None:
        iw, ih = bg.size
        long_edge = max(iw, ih) or 1
        scale = min(1.0, _PREVIEW_LONG_EDGE / long_edge)
        rw, rh = max(1, round(iw * scale)), max(1, round(ih * scale))
        base = bg.convert("RGB").resize((rw, rh), Image.LANCZOS)
        base = ImageEnhance.Brightness(base).enhance(_PREVIEW_DIM)
        img = base.convert("RGBA")
    else:
        long_edge = max(width, height) or 1
        scale = min(1.0, _PREVIEW_LONG_EDGE / long_edge)
        rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
        grey = round(_PREVIEW_DIM * 128)
        img = Image.new("RGBA", (rw, rh), (grey, grey, grey, 255))

    overlay = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fs = max(10, round(rh / 64))
    font = _font(fs)
    tag_font = _font(max(9, fs - 2))
    line_h = fs + 2

    for i, region in enumerate(regions):
        if not isinstance(region, dict):
            continue
        palette = [c for c in (region.get("palette") or []) if c]
        r, g, b = hex_to_rgb(palette[0]) if palette else (140, 140, 140)
        x1 = max(0, min(rw, round(region.get("x", 0) * rw)))
        y1 = max(0, min(rh, round(region.get("y", 0) * rh)))
        x2 = max(0, min(rw, round((region.get("x", 0) + region.get("w", 0)) * rw)))
        y2 = max(0, min(rh, round((region.get("y", 0) + region.get("h", 0)) * rh)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 255), width=2)

        swatches = palette[:5]
        if swatches and (x2 - x1) > 2:
            sh = max(5, fs // 2)
            seg = (x2 - x1) / len(swatches)
            for p, hexc in enumerate(swatches):
                sx = x1 + round(p * seg)
                draw.rectangle([sx, y1, x1 + round((p + 1) * seg), y1 + sh], fill=hex_to_rgb(hexc))

        etype = "text" if region.get("type") == "text" else "obj"
        tag = str(i + 1).zfill(2)
        tw = draw.textlength(tag, font=tag_font)
        draw.rectangle([x1, y1, x1 + tw + 6, y1 + fs + 2], fill=(r, g, b, 255))
        tag_fill = (0, 0, 0, 255) if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else (255, 255, 255, 255)
        draw.text((x1 + 3, y1 + 1), tag, fill=tag_fill, font=tag_font)

        body = region.get("desc", "") or ""
        if etype == "text" and region.get("text"):
            body = '"%s"%s' % (region["text"], " — " + body if body else "")
        if body and (x2 - x1) > 8:
            ty = y1 + fs + 5
            for line in _wrap(draw, body, font, x2 - x1 - 8):
                if ty > y2:
                    break
                draw.text((x1 + 4, ty), line, fill=readable_color((r, g, b)) + (255,), font=font)
                ty += line_h

    composed = Image.alpha_composite(img, overlay).convert("RGB")
    arr = np.asarray(composed, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def boxes_to_regions(boxes, width: int, height: int) -> list:
    regions: list = []
    if not isinstance(boxes, list):
        return regions
    for box in boxes:
        if not isinstance(box, dict):
            continue
        meta = box.get("metadata")
        meta = meta if isinstance(meta, dict) else {}
        regions.append({
            **pixels_to_fractions(box, width, height),
            "type": meta.get("type", "obj"),
            "text": meta.get("text", ""),
            "desc": meta.get("desc", ""),
            "palette": meta.get("palette", []),
        })
    return regions


def normalize_incoming_boxes(bboxes) -> list:
    if isinstance(bboxes, dict):
        frame = [bboxes]
    elif not isinstance(bboxes, list) or not bboxes:
        frame = []
    elif isinstance(bboxes[0], dict):
        frame = bboxes
    else:
        frame = bboxes[0] if isinstance(bboxes[0], list) else []
    boxes = []
    for box in frame:
        if not isinstance(box, dict):
            continue
        norm = {
            "x": box.get("x", 0),
            "y": box.get("y", 0),
            "width": box.get("width", 0),
            "height": box.get("height", 0),
        }
        meta = box.get("metadata")
        if isinstance(meta, dict):
            norm["metadata"] = meta
        boxes.append(norm)
    return boxes


def _looks_like_element(box: dict) -> bool:
    bbox = box.get("bbox")
    return isinstance(bbox, (list, tuple)) and len(bbox) == 4


def _looks_like_bbox(box: dict) -> bool:
    return all(key in box for key in ("x", "y", "width", "height"))


def elements_to_boxes(elements: list, width: int, height: int) -> list:
    boxes = []
    for element in elements:
        if not isinstance(element, dict):
            continue
        bbox = element.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            raise ValueError("bboxes element is missing a valid 'bbox' [ymin, xmin, ymax, xmax]")
        try:
            ymin, xmin, ymax, xmax = (float(v) / 1000.0 for v in bbox)
        except (TypeError, ValueError):
            raise ValueError("bboxes element 'bbox' must contain four numbers")
        etype = "text" if element.get("type") == "text" else "obj"
        boxes.append({
            "x": round(min(xmin, xmax) * width),
            "y": round(min(ymin, ymax) * height),
            "width": round(abs(xmax - xmin) * width),
            "height": round(abs(ymax - ymin) * height),
            "metadata": {
                "type": etype,
                "text": element.get("text", "") if etype == "text" else "",
                "desc": element.get("desc", ""),
                "palette": element.get("color_palette", []) or [],
            },
        })
    return boxes


def boxes_from_input(data, width: int, height: int) -> list:
    if data is None:
        return []
    if isinstance(data, str):
        text = data.strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"bboxes string input is not valid JSON: {exc}") from exc
    if isinstance(data, dict):
        if _looks_like_element(data):
            return elements_to_boxes([data], width, height)
        if _looks_like_bbox(data):
            return normalize_incoming_boxes(data)
        raise ValueError(
            "bboxes dict must be a bounding box (x, y, width, height) or an element (with a 'bbox')"
        )
    if not isinstance(data, list):
        raise ValueError(
            "bboxes input must be bounding boxes, elements, or a JSON string, "
            f"got {type(data).__name__}"
        )
    if not data:
        return []
    first = data[0]
    if isinstance(first, list):
        return normalize_incoming_boxes(data)
    if isinstance(first, dict):
        if _looks_like_element(first):
            return elements_to_boxes(data, width, height)
        if _looks_like_bbox(first):
            return normalize_incoming_boxes(data)
        raise ValueError(
            "bboxes items must be bounding boxes (x, y, width, height) or elements (with a 'bbox')"
        )
    raise ValueError(
        f"bboxes list must contain bounding boxes or elements, got {type(first).__name__}"
    )


def _norm_bbox(region: dict) -> list[int]:
    def grid(value: float) -> int:
        return max(0, min(1000, round(value * 1000)))

    x, y = region.get("x", 0.0), region.get("y", 0.0)
    w, h = region.get("w", 0.0), region.get("h", 0.0)
    ymin, xmin, ymax, xmax = grid(y), grid(x), grid(y + h), grid(x + w)
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    return [ymin, xmin, ymax, xmax]


def build_elements(regions: list) -> list:
    elements = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        etype = "text" if region.get("type") == "text" else "obj"
        element = {"type": etype}
        element["bbox"] = _norm_bbox(region)
        if etype == "text":
            element["text"] = region.get("text", "")
        element["desc"] = region.get("desc", "")
        palette = normalize_palette(region.get("palette", []))
        if palette:
            element["color_palette"] = palette[:5]
        elements.append(element)
    return elements


class CreateBoundingBoxes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        editor_state = io.BoundingBoxes.Input(
            "editor_state",
            socketless=False,
            tooltip="Draw bounding boxes and set each box type, text, description, color palette. Start with background element first and foreground last.",
        )
        return io.Schema(
            node_id="CreateBoundingBoxes",
            display_name="Create Bounding Boxes",
            category="utilities",
            description="Draw bounding boxes in a canvas. Outputs Ideogram prompt elements, pixel-space bounding boxes, and a preview image.",
            inputs=[
                io.Image.Input(
                    "background",
                    optional=True,
                    tooltip="Optional image used as background in the canvas and preview.",
                ),
                io.MultiType.Input(
                    "bboxes",
                    [io.BoundingBox, io.Array, io.String],
                    optional=True,
                    tooltip="Bounding boxes, elements, or a JSON string to initialize the canvas. A new upstream value initializes the canvas; edits made on the canvas take priority and are kept until the upstream value changes again.",
                ),
                io.Int.Input("width", default=1024, min=64, max=16384, step=16,
                             tooltip="Width of the canvas and the pixel grid for the bounding boxes."),
                io.Int.Input("height", default=1024, min=64, max=16384, step=16,
                             tooltip="Height of the canvas and the pixel grid for the bounding boxes."),
                editor_state,
                io.BoundingBoxes.Input(
                    "last_incoming",
                    optional=True,
                    tooltip="Internal state managed by the canvas: the upstream bboxes value that last initialized it. Leave empty to re-initialize the canvas from the bboxes input on the next run.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="preview"),
                io.BoundingBox.Output(display_name="bboxes"),
                io.Array.Output(display_name="elements"),
            ],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, width, height, editor_state=None, last_incoming=None, background=None, bboxes=None) -> io.NodeOutput:
        incoming = boxes_from_input(bboxes, width, height)
        applied = last_incoming if isinstance(last_incoming, list) else []
        upstream_changed = bool(incoming) and incoming != applied
        source = incoming if upstream_changed else (editor_state or [])
        regions = boxes_to_regions(source, width, height)
        preview = render_preview(regions, width, height, _bg_from_image(background))
        ui = {"dims": [width, height]}
        if incoming:
            ui["input_bboxes"] = incoming
        return io.NodeOutput(
            preview,
            fractions_to_bbox_frame(regions, width, height),
            build_elements(regions),
            ui=ui,
        )


class BoundingBoxesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CreateBoundingBoxes]


async def comfy_entrypoint() -> BoundingBoxesExtension:
    return BoundingBoxesExtension()
