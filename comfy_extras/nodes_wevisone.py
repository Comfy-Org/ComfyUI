import json
import re

import torch
import torch.nn.functional as F
from typing_extensions import override

import comfy.model_management
import comfy.ops
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_rtdetr import bboxes_per_frame

TASKS = ["segmentation", "detection", "referring", "point", "count_box", "count_point", "ocr", "ocr_grounding", "visual_prompt", "freeform"]

ITEM_RE = re.compile(r"<(\d{1,4})>|<OBJ>")
SHAPE_RE = re.compile(r"<ref>(.*?)</ref>\s*<(box|point|polygon)>(.*?)(?:</(?:box|point|polygon)>|(?=<ref>)|$)", re.DOTALL)

# the instructions the checkpoint was trained with; detection and point take one label per line
PROMPTS = {
    "segmentation": "Locate the instances that match the following description and provide the alpha mattes: {phrase}",
    "detection": "Locate all the instances that match the following description: {labels}.",
    "referring": "Locate the instances that match the following description: {phrase}",
    "point": "point to: {labels}",
    "count_box": "How many {subject} in the image? Please count them with boxes.",
    "count_point": "How many {subject} in the image?",
    "ocr": "detect all the word in the box format",
    "ocr_grounding": "Locate the following text in the image: {text}",
}


def build_prompt(task, query, boxes, width, height):
    text = query.strip()
    if task == "visual_prompt":
        coords = ",".join("".join("<{}>".format(max(0, min(1000, int(v * 1000)))) for v in (b["x"] / width, b["y"] / height, (b["x"] + b["width"]) / width, (b["y"] + b["height"]) / height)) for b in boxes)
        return "Locate all similar objects in the image based on the reference boxes: <ref>object 1</ref><box>" + coords + "</box>"
    if task == "freeform":
        return text
    labels = ". ".join(line.strip() for line in query.splitlines() if line.strip())
    phrase = text if not text or text[-1] in ".?!" else text + "."
    return PROMPTS[task].format(text=text, labels=labels, phrase=phrase, subject=text or "objects")


def parse_shapes(text, width, height):
    # boxes and points in pixels; objects maps a box index to (ordinal of the <OBJ> written right after it in the answer, its normalized xyxy box)
    boxes, points, objects = [], [], {}
    obj_positions = [m.start() for m in re.finditer("<OBJ>", text)]
    for m in SHAPE_RE.finditer(text):
        label = m.group(1).strip() or "object"
        shape = m.group(2)
        for segment in re.finditer(r"[^,]+", m.group(3)):
            values, paired = [], {}
            for item in ITEM_RE.finditer(segment.group()):
                if item.group(1) is None:
                    paired[len(values) // 4 - 1] = obj_positions.index(m.start(3) + segment.start() + item.start())
                else:
                    values.append(int(item.group(1)))
            if shape == "point":
                for k in range(0, len(values) - 1, 2):
                    points.append({"x": round(values[k] / 1000 * width), "y": round(values[k + 1] / 1000 * height), "label": label})
            elif shape == "box":
                for k in range(0, len(values) - 3, 4):
                    x1, y1, x2, y2 = values[k] / 1000 * width, values[k + 1] / 1000 * height, values[k + 2] / 1000 * width, values[k + 3] / 1000 * height
                    if x2 > x1 and y2 > y1:
                        if k // 4 in paired:
                            objects[len(boxes)] = (paired[k // 4], [v / 1000 for v in values[k:k + 4]])
                        boxes.append({"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "label": label})
    return boxes, points, objects


class WeVisOneDetect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WeVisOneDetect",
            display_name="WeVisOne-VL Detect",
            category="image/detection",
            search_aliases=["wevisone", "segment", "matting", "alpha matte", "grounding", "referring", "pointing", "counting", "ocr"],
            inputs=[
                io.Clip.Input("clip", tooltip="A WeVisOne-VL text encoder."),
                io.Image.Input("image"),
                io.Combo.Input("task", options=TASKS, default="segmentation", tooltip="segmentation: boxes and alpha mattes. detection and point take one label per line."),
                io.String.Input("query", multiline=True, default="", tooltip="What to find. freeform sends it as the whole prompt."),
                io.BoundingBox.Input("bboxes", optional=True, force_input=True, tooltip="Reference boxes for visual_prompt."),
                io.Boolean.Input("individual_masks", default=False, tooltip="Output one mask per box, in bboxes order with the frames in sequence, instead of one union per frame."),
                io.Int.Input("max_length", default=4096, min=1, max=32768, advanced=True),
            ],
            outputs=[
                io.Mask.Output("masks"),
                io.BoundingBox.Output("bboxes"),
                io.String.Output("points", tooltip="Points as JSON [{\"x\", \"y\", \"label\", \"frame\"}] in pixels, for the point and count_point tasks."),
                io.String.Output("text", tooltip="The raw answer."),
            ],
        )

    @classmethod
    def execute(cls, clip, image, task, query, individual_masks, bboxes=None, max_length=4096) -> io.NodeOutput:
        B, H, W, _ = image.shape
        bboxes = bboxes_per_frame(bboxes, B)

        idev = comfy.model_management.intermediate_device()
        texts, all_masks, all_boxes, all_points = [], [], [], []
        for b in range(B):
            frame = image[b:b + 1]
            prompt = build_prompt(task, query, bboxes[b], W, H)
            tokens = clip.tokenize(prompt, image=frame)
            # top_k=1 sampling is greedy decoding with the reference's repetition penalty
            ids = clip.generate(tokens, do_sample=True, max_length=max_length, top_k=1, repetition_penalty=1.05, seed=0)
            text = clip.decode(ids).strip()
            texts.append(text)
            boxes, points, objects = parse_shapes(text, W, H)
            all_boxes.append(boxes)
            all_points.extend({**p, "frame": b} for p in points)

            masks = torch.zeros((len(boxes) if individual_masks or len(objects) > 0 else 1, H, W), device=idev)
            if len(objects) > 0:
                device = clip.patcher.load_device
                with comfy.model_management.cuda_device_context(device), comfy.ops.use_quantized_matmul(clip.cond_stage_model, device):
                    logits = clip.cond_stage_model.decode_masks(tokens, ids, frame, device, list(objects.values()))
                masks[list(objects)] = F.interpolate(logits.sigmoid()[None], size=(H, W), mode="bilinear", align_corners=False)[0].to(idev)
            if not individual_masks:
                masks = masks.amax(dim=0, keepdim=True)
            all_masks.append(masks)

        masks = torch.cat(all_masks)
        if len(masks) == 0:
            masks = torch.zeros((1, H, W), device=idev)
        return io.NodeOutput(masks, all_boxes, json.dumps(all_points), "\n".join(texts))


class WeVisOneExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [WeVisOneDetect]


async def comfy_entrypoint() -> WeVisOneExtension:
    return WeVisOneExtension()
