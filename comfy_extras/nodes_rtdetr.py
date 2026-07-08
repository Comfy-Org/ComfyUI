from typing_extensions import override

import torch
from comfy.ldm.rt_detr.rtdetr_v4 import COCO_CLASSES
import comfy.model_management
import comfy.utils
from comfy_api.latest import ComfyExtension, io
from torchvision.transforms import ToPILImage, ToTensor
from PIL import ImageDraw, ImageFont


class RTDETR_detect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RTDETR_detect",
            display_name="Run Real-Time Detection (RT-DETR)",
            category="image/detection",
            search_aliases=["bbox", "bounding box", "object detection", "coco"],
            inputs=[
                io.Model.Input("model", display_name="model"),
                io.Image.Input("image", display_name="image"),
                io.Float.Input("threshold", display_name="threshold", default=0.5),
                io.Combo.Input("class_name", options=["all"] + COCO_CLASSES, default="all", tooltip="Filter detections by class. Set to 'all' to disable filtering."),
                io.Int.Input("max_detections", display_name="max_detections", default=100, tooltip="Maximum number of detections to return per image. In order of descending confidence score."),
            ],
            outputs=[
                io.BoundingBox.Output("bboxes")],
        )

    @classmethod
    def execute(cls, model, image, threshold, class_name, max_detections) -> io.NodeOutput:
        B, H, W, C = image.shape

        comfy.model_management.load_model_gpu(model)
        results = []
        for i in range(0, B, 32):
            batch = image[i:i + 32]
            image_in = comfy.utils.common_upscale(batch.movedim(-1, 1), 640, 640, "bilinear", crop="disabled")
            results.extend(model.model.diffusion_model(image_in, (W, H)))

        all_bbox_dicts = []

        for det in results:
            keep   = det['scores'] > threshold
            boxes  = det['boxes'][keep].cpu()
            labels = det['labels'][keep].cpu()
            scores = det['scores'][keep].cpu()

            bbox_dicts = [
                {
                    "x": float(box[0]),
                    "y": float(box[1]),
                    "width": float(box[2] - box[0]),
                    "height": float(box[3] - box[1]),
                    "label": COCO_CLASSES[int(label)],
                    "score": float(score)
                }
                for box, label, score in zip(boxes, labels, scores)
                if class_name == "all" or COCO_CLASSES[int(label)] == class_name
            ]
            bbox_dicts.sort(key=lambda d: d["score"], reverse=True)
            all_bbox_dicts.append(bbox_dicts[:max_detections])

        return io.NodeOutput(all_bbox_dicts)


class DrawBBoxes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DrawBBoxes",
            display_name="Draw BBoxes",
            category="image/detection",
            search_aliases=["bbox", "bounding box", "object detection", "rt_detr", "visualize detections", "coco"],
            inputs=[
                io.Image.Input("image", optional=True),
                io.BoundingBox.Input("bboxes", force_input=True),
            ],
            outputs=[
                io.Image.Output("out_image"),
            ],
        )

    @classmethod
    def execute(cls, bboxes, image=None) -> io.NodeOutput:
        # Normalise to list[list[dict]], then fit to batch size B.
        B = image.shape[0] if image is not None else 1
        if isinstance(bboxes, dict):
            bboxes = [[bboxes]]
        elif not isinstance(bboxes, list) or not bboxes:
            bboxes = [[]]
        elif isinstance(bboxes[0], dict):
            bboxes = [bboxes]  # flat list → same detections for every image

        if len(bboxes) == 1:
            bboxes = bboxes * B
        bboxes = (bboxes + [[]] * B)[:B]

        if image is None:
            B = len(bboxes)
            max_w = max((int(d["x"] + d["width"])  for frame in bboxes for d in frame), default=640)
            max_h = max((int(d["y"] + d["height"]) for frame in bboxes for d in frame), default=640)
            image = torch.zeros((B, max_h, max_w, 3), dtype=torch.float32)

        all_out_images = []
        for i in range(B):
            detections = bboxes[i]
            if detections:
                boxes  = torch.tensor([[d["x"], d["y"], d["x"] + d["width"], d["y"] + d["height"]] for d in detections])
                labels = [d.get("label") if d.get("label") in COCO_CLASSES else None for d in detections]
                scores = torch.tensor([d.get("score", 1.0) for d in detections])
            else:
                boxes  = torch.zeros((0, 4))
                labels = []
                scores = torch.zeros((0,))

            pil_image = image[i].movedim(-1, 0)
            img = ToPILImage()(pil_image)
            if detections:
                img = cls.draw_detections(img, boxes, labels, scores)
            all_out_images.append(ToTensor()(img).unsqueeze(0).movedim(1, -1))

        out_images = torch.cat(all_out_images, dim=0).to(comfy.model_management.intermediate_device())
        return io.NodeOutput(out_images)

    @classmethod
    def draw_detections(cls, img, boxes, labels, scores):
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype('arial.ttf', 16)
        except Exception:
            font = ImageFont.load_default()
        colors = [(255,0,0),(0,200,0),(0,0,255),(255,165,0),(128,0,128),
                (0,255,255),(255,20,147),(100,149,237)]
        for box, label, score in sorted(zip(boxes, labels, scores), key=lambda x: x[2].item()):
            x1, y1, x2, y2 = box.tolist()
            color_idx = COCO_CLASSES.index(label) if label is not None else 0
            c = colors[color_idx % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=c, width=3)
            if label is not None:
                draw.text((x1 + 2, y1 + 2), f'{label} {score:.2f}', fill=c, font=font)
        return img


class RTDETRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            RTDETR_detect,
            DrawBBoxes,
        ]


async def comfy_entrypoint() -> RTDETRExtension:
    return RTDETRExtension()
