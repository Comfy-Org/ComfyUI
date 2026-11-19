"""
SAM3 (Segment Anything 3) nodes for detection, segmentation, and video tracking.
"""

from typing_extensions import override

import json
import os
import torch
import torch.nn.functional as F
import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io, ui
import av
from fractions import Fraction


def _extract_text_prompts(conditioning, device, dtype):
    """Extract list of (text_embeddings, text_mask) from conditioning."""
    cond_meta = conditioning[0][1]
    multi = cond_meta.get("sam3_multi_cond")
    prompts = []
    if multi is not None:
        for entry in multi:
            emb = entry["cond"].to(device=device, dtype=dtype)
            mask = entry["attention_mask"].to(device) if entry["attention_mask"] is not None else None
            if mask is None:
                mask = torch.ones(emb.shape[0], emb.shape[1], dtype=torch.int64, device=device)
            prompts.append((emb, mask, entry.get("max_detections", 1)))
    else:
        emb = conditioning[0][0].to(device=device, dtype=dtype)
        mask = cond_meta.get("attention_mask")
        if mask is not None:
            mask = mask.to(device)
        else:
            mask = torch.ones(emb.shape[0], emb.shape[1], dtype=torch.int64, device=device)
        prompts.append((emb, mask, 1))
    return prompts


def _refine_mask(sam3_model, orig_image_hwc, coarse_mask, box_xyxy, H, W, device, dtype, iterations):
    """Refine a coarse detector mask via SAM decoder, cropping to the detection box.

    Returns: [1, H, W] binary mask
    """
    def _coarse_fallback():
        return (F.interpolate(coarse_mask.unsqueeze(0).unsqueeze(0), size=(H, W),
                              mode="bilinear", align_corners=False)[0] > 0).float()

    if iterations <= 0:
        return _coarse_fallback()

    pad_frac = 0.1
    x1, y1, x2, y2 = box_xyxy.tolist()
    bw, bh = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - bw * pad_frac))
    cy1 = max(0, int(y1 - bh * pad_frac))
    cx2 = min(W, int(x2 + bw * pad_frac))
    cy2 = min(H, int(y2 + bh * pad_frac))
    if cx2 <= cx1 or cy2 <= cy1:
        return _coarse_fallback()

    crop = orig_image_hwc[cy1:cy2, cx1:cx2, :3]
    crop_1008 = comfy.utils.common_upscale(crop.unsqueeze(0).movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")
    crop_frame = crop_1008.to(device=device, dtype=dtype)
    crop_h, crop_w = cy2 - cy1, cx2 - cx1

    # Crop coarse mask and refine via SAM on the cropped image
    mask_h, mask_w = coarse_mask.shape[-2:]
    mx1, my1 = int(cx1 / W * mask_w), int(cy1 / H * mask_h)
    mx2, my2 = int(cx2 / W * mask_w), int(cy2 / H * mask_h)
    if mx2 <= mx1 or my2 <= my1:
        return _coarse_fallback()
    mask_logit = coarse_mask[..., my1:my2, mx1:mx2].unsqueeze(0).unsqueeze(0)
    for _ in range(iterations):
        coarse_input = F.interpolate(mask_logit, size=(1008, 1008), mode="bilinear", align_corners=False)
        mask_logit = sam3_model.forward_segment(crop_frame, mask_inputs=coarse_input)

    refined_crop = F.interpolate(mask_logit, size=(crop_h, crop_w), mode="bilinear", align_corners=False)
    full_mask = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
    full_mask[:, :, cy1:cy2, cx1:cx2] = refined_crop
    coarse_full = F.interpolate(coarse_mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)
    return ((full_mask[0] > 0) | (coarse_full[0] > 0)).float()



class SAM3_Detect(io.ComfyNode):
    """Open-vocabulary detection and segmentation using text, box, or point prompts."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_Detect",
            display_name="SAM3 Detect",
            category="image/detection",
            search_aliases=["sam3", "segment anything", "open vocabulary", "text detection", "segment"],
            inputs=[
                io.Model.Input("model", display_name="model"),
                io.Image.Input("image", display_name="image"),
                io.Conditioning.Input("conditioning", display_name="conditioning", optional=True, tooltip="Text conditioning from CLIPTextEncode"),
                io.BoundingBox.Input("bboxes", display_name="bboxes", force_input=True, optional=True, tooltip="Bounding boxes to segment within"),
                io.String.Input("positive_coords", display_name="positive_coords", force_input=True, optional=True, tooltip="Positive point prompts as JSON [{\"x\": int, \"y\": int}, ...] (pixel coords)"),
                io.String.Input("negative_coords", display_name="negative_coords", force_input=True, optional=True, tooltip="Negative point prompts as JSON [{\"x\": int, \"y\": int}, ...] (pixel coords)"),
                io.Float.Input("threshold", display_name="threshold", default=0.5, min=0.0, max=1.0, step=0.01),
                io.Int.Input("refine_iterations", display_name="refine_iterations", default=2, min=0, max=5, tooltip="SAM decoder refinement passes (0=use raw detector masks)"),
                io.Boolean.Input("individual_masks", display_name="individual_masks", default=False, tooltip="Output per-object masks instead of union"),
            ],
            outputs=[
                io.Mask.Output("masks"),
                io.BoundingBox.Output("bboxes"),
            ],
        )

    @classmethod
    def execute(cls, model, image, conditioning=None, bboxes=None, positive_coords=None, negative_coords=None, threshold=0.5, refine_iterations=2, individual_masks=False) -> io.NodeOutput:
        B, H, W, C = image.shape
        image_in = comfy.utils.common_upscale(image[..., :3].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")

        # Convert bboxes to normalized cxcywh format, per-frame list of [1, N, 4] tensors.
        # Supports: single dict (all frames), list[dict] (all frames), list[list[dict]] (per-frame).
        def _boxes_to_tensor(box_list):
            coords = []
            for d in box_list:
                cx = (d["x"] + d["width"] / 2) / W
                cy = (d["y"] + d["height"] / 2) / H
                coords.append([cx, cy, d["width"] / W, d["height"] / H])
            return torch.tensor([coords], dtype=torch.float32)  # [1, N, 4]

        per_frame_boxes = None
        if bboxes is not None:
            if isinstance(bboxes, dict):
                # Single box → same for all frames
                shared = _boxes_to_tensor([bboxes])
                per_frame_boxes = [shared] * B
            elif isinstance(bboxes, list) and len(bboxes) > 0 and isinstance(bboxes[0], list):
                # list[list[dict]] → per-frame boxes
                per_frame_boxes = [_boxes_to_tensor(frame_boxes) if frame_boxes else None for frame_boxes in bboxes]
                # Pad to B if fewer frames provided
                while len(per_frame_boxes) < B:
                    per_frame_boxes.append(per_frame_boxes[-1] if per_frame_boxes else None)
            elif isinstance(bboxes, list) and len(bboxes) > 0:
                # list[dict] → same boxes for all frames
                shared = _boxes_to_tensor(bboxes)
                per_frame_boxes = [shared] * B

        # Parse point prompts from JSON (KJNodes PointsEditor format: [{"x": int, "y": int}, ...])
        pos_pts = json.loads(positive_coords) if positive_coords else []
        neg_pts = json.loads(negative_coords) if negative_coords else []
        has_points = len(pos_pts) > 0 or len(neg_pts) > 0

        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        sam3_model = model.model.diffusion_model

        # Build point inputs for tracker SAM decoder path
        point_inputs = None
        if has_points:
            all_coords = [[p["x"] / W * 1008, p["y"] / H * 1008] for p in pos_pts] + \
                         [[p["x"] / W * 1008, p["y"] / H * 1008] for p in neg_pts]
            all_labels = [1] * len(pos_pts) + [0] * len(neg_pts)
            point_inputs = {
                "point_coords": torch.tensor([all_coords], dtype=dtype, device=device),
                "point_labels": torch.tensor([all_labels], dtype=torch.int32, device=device),
            }

        cond_list = _extract_text_prompts(conditioning, device, dtype) if conditioning is not None and len(conditioning) > 0 else []
        has_text = len(cond_list) > 0

        # Run per-image through detector (text/boxes) and/or tracker (points)
        all_bbox_dicts = []
        all_masks = []
        pbar = comfy.utils.ProgressBar(B)

        for b in range(B):
            frame = image_in[b:b+1].to(device=device, dtype=dtype)
            b_boxes = None
            if per_frame_boxes is not None and per_frame_boxes[b] is not None:
                b_boxes = per_frame_boxes[b].to(device=device, dtype=dtype)

            frame_bbox_dicts = []
            frame_masks = []

            # Point prompts: tracker SAM decoder path with iterative refinement
            if point_inputs is not None:
                mask_logit = sam3_model.forward_segment(frame, point_inputs=point_inputs)
                for _ in range(max(0, refine_iterations - 1)):
                    mask_logit = sam3_model.forward_segment(frame, mask_inputs=mask_logit)
                mask = F.interpolate(mask_logit, size=(H, W), mode="bilinear", align_corners=False)
                frame_masks.append((mask[0] > 0).float())

            # Box prompts: SAM decoder path (segment inside each box)
            if b_boxes is not None and not has_text:
                for box_cxcywh in b_boxes[0]:
                    cx, cy, bw, bh = box_cxcywh.tolist()
                    # Convert cxcywh normalized → xyxy in 1008 space → [1, 2, 2] corners
                    sam_box = torch.tensor([[[(cx - bw/2) * 1008, (cy - bh/2) * 1008],
                                             [(cx + bw/2) * 1008, (cy + bh/2) * 1008]]],
                                           device=device, dtype=dtype)
                    mask_logit = sam3_model.forward_segment(frame, box_inputs=sam_box)
                    for _ in range(max(0, refine_iterations - 1)):
                        mask_logit = sam3_model.forward_segment(frame, mask_inputs=mask_logit)
                    mask = F.interpolate(mask_logit, size=(H, W), mode="bilinear", align_corners=False)
                    frame_masks.append((mask[0] > 0).float())

            # Text prompts: run detector per text prompt (each detects one category)
            for text_embeddings, text_mask, max_det in cond_list:
                results = sam3_model(
                    frame, text_embeddings=text_embeddings, text_mask=text_mask,
                    boxes=b_boxes, threshold=threshold, orig_size=(H, W))

                pred_boxes = results["boxes"][0]
                scores = results["scores"][0]
                masks = results["masks"][0]

                probs = scores.sigmoid()
                keep = probs > threshold
                kept_boxes = pred_boxes[keep].cpu()
                kept_scores = probs[keep].cpu()
                kept_masks = masks[keep]

                order = kept_scores.argsort(descending=True)[:max_det]
                kept_boxes = kept_boxes[order]
                kept_scores = kept_scores[order]
                kept_masks = kept_masks[order]

                for box, score in zip(kept_boxes, kept_scores):
                    frame_bbox_dicts.append({
                        "x": float(box[0]), "y": float(box[1]),
                        "width": float(box[2] - box[0]), "height": float(box[3] - box[1]),
                        "score": float(score),
                    })
                for m, box in zip(kept_masks, kept_boxes):
                    frame_masks.append(_refine_mask(
                        sam3_model, image[b], m, box, H, W, device, dtype, refine_iterations))

            all_bbox_dicts.append(frame_bbox_dicts)
            if len(frame_masks) > 0:
                combined = torch.cat(frame_masks, dim=0)  # [N_obj, H, W]
                if individual_masks:
                    all_masks.append(combined)
                else:
                    all_masks.append((combined > 0).any(dim=0).float())
            else:
                if individual_masks:
                    all_masks.append(torch.zeros(0, H, W, device=comfy.model_management.intermediate_device()))
                else:
                    all_masks.append(torch.zeros(H, W, device=comfy.model_management.intermediate_device()))
            pbar.update(1)

        idev = comfy.model_management.intermediate_device()
        all_masks = [m.to(idev) for m in all_masks]
        mask_out = torch.cat(all_masks, dim=0) if individual_masks else torch.stack(all_masks)
        return io.NodeOutput(mask_out, all_bbox_dicts)


SAM3TrackData = io.Custom("SAM3_TRACK_DATA")

class SAM3_VideoTrack(io.ComfyNode):
    """Track objects across video frames using SAM3's memory-based tracker."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_VideoTrack",
            display_name="Run SAM3 Video Track",
            category="image/detection",
            search_aliases=["sam3", "video", "track", "propagate"],
            inputs=[
                io.Image.Input("images", display_name="images", tooltip="Video frames as batched images"),
                io.Model.Input("model", display_name="model"),
                io.Mask.Input("initial_mask", display_name="initial_mask", optional=True, tooltip="Mask(s) for the first frame to track (one per object)"),
                io.Conditioning.Input("conditioning", display_name="conditioning", optional=True, tooltip="Text conditioning for detecting new objects during tracking"),
                io.Float.Input("detection_threshold", display_name="detection_threshold", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="Score threshold for text-prompted detection."),
                io.Int.Input("max_objects", display_name="max_objects", default=4, min=0, max=64, tooltip="Max tracked objects. Initial masks count toward this limit. 0 uses the internal cap of 64."),
                io.Int.Input("detect_interval", display_name="detect_interval", default=1, min=1, tooltip="Run detection every N frames (1=every frame). Higher values save compute."),
            ],
            outputs=[
                SAM3TrackData.Output("track_data", display_name="track_data"),
            ],
        )

    @classmethod
    def execute(cls, images, model, initial_mask=None, conditioning=None, detection_threshold=0.5, max_objects=0, detect_interval=1) -> io.NodeOutput:
        N, H, W, C = images.shape

        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        sam3_model = model.model.diffusion_model

        frames_in = images[..., :3].movedim(-1, 1)

        init_masks = None
        if initial_mask is not None:
            init_masks = initial_mask.unsqueeze(1).to(device=device, dtype=dtype)

        pbar = comfy.utils.ProgressBar(N)

        text_prompts = None
        if conditioning is not None and len(conditioning) > 0:
            text_prompts = [(emb, mask) for emb, mask, _ in _extract_text_prompts(conditioning, device, dtype)]
        elif initial_mask is None:
            raise ValueError("Either initial_mask or conditioning must be provided")

        result = sam3_model.forward_video(
            images=frames_in, initial_masks=init_masks, pbar=pbar, text_prompts=text_prompts,
            new_det_thresh=detection_threshold, max_objects=max_objects,
            detect_interval=detect_interval, target_device=device, target_dtype=dtype)
        result["orig_size"] = (H, W)
        return io.NodeOutput(result)


class SAM3_TrackPreview(io.ComfyNode):
    """Visualize tracked objects with distinct colors as a video preview. No tensor output — saves to temp video."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_TrackPreview",
            display_name="SAM3 Track Preview",
            category="image/detection",
            inputs=[
                SAM3TrackData.Input("track_data", display_name="track_data"),
                io.Image.Input("images", display_name="images", optional=True),
                io.Float.Input("opacity", display_name="opacity", default=0.5, min=0.0, max=1.0, step=0.05),
                io.Float.Input("fps", display_name="fps", default=24.0, min=1.0, max=120.0, step=1.0),
            ],
            is_output_node=True,
        )

    COLORS = [
        (0.12, 0.47, 0.71), (1.0, 0.5, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16),
        (0.58, 0.4, 0.74), (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.5, 0.5, 0.5),
        (0.74, 0.74, 0.13), (0.09, 0.75, 0.81), (0.94, 0.76, 0.06), (0.42, 0.68, 0.84),
    ]

    # 5x3 bitmap font atlas for digits 0-9 [10, 5, 3]
    _glyph_cache = {}  # (device, scale) -> (glyphs, outlines, gh, gw, oh, ow)

    @staticmethod
    def _get_glyphs(device, scale=3):
        key = (device, scale)
        if key in SAM3_TrackPreview._glyph_cache:
            return SAM3_TrackPreview._glyph_cache[key]
        atlas = torch.tensor([
            [[1,1,1],[1,0,1],[1,0,1],[1,0,1],[1,1,1]],
            [[0,1,0],[1,1,0],[0,1,0],[0,1,0],[1,1,1]],
            [[1,1,1],[0,0,1],[1,1,1],[1,0,0],[1,1,1]],
            [[1,1,1],[0,0,1],[1,1,1],[0,0,1],[1,1,1]],
            [[1,0,1],[1,0,1],[1,1,1],[0,0,1],[0,0,1]],
            [[1,1,1],[1,0,0],[1,1,1],[0,0,1],[1,1,1]],
            [[1,1,1],[1,0,0],[1,1,1],[1,0,1],[1,1,1]],
            [[1,1,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],
            [[1,1,1],[1,0,1],[1,1,1],[1,0,1],[1,1,1]],
            [[1,1,1],[1,0,1],[1,1,1],[0,0,1],[1,1,1]],
        ], dtype=torch.bool)
        glyphs, outlines = [], []
        for d in range(10):
            g = atlas[d].repeat_interleave(scale, 0).repeat_interleave(scale, 1)
            padded = F.pad(g.float().unsqueeze(0).unsqueeze(0), (1,1,1,1))
            o = (F.max_pool2d(padded, 3, stride=1, padding=1)[0, 0] > 0)
            glyphs.append(g.to(device))
            outlines.append(o.to(device))
        gh, gw = glyphs[0].shape
        oh, ow = outlines[0].shape
        SAM3_TrackPreview._glyph_cache[key] = (glyphs, outlines, gh, gw, oh, ow)
        return SAM3_TrackPreview._glyph_cache[key]

    @staticmethod
    def _draw_number_gpu(frame, number, cx, cy, color, scale=3):
        """Draw a number on a GPU tensor [H, W, 3] float 0-1 at (cx, cy) with outline."""
        H, W = frame.shape[:2]
        device = frame.device
        glyphs, outlines, gh, gw, oh, ow = SAM3_TrackPreview._get_glyphs(device, scale)
        color_t = torch.tensor(color, device=device, dtype=frame.dtype)
        digs = [int(d) for d in str(number)]
        total_w = len(digs) * (gw + scale) - scale
        x0 = cx - total_w // 2
        y0 = cy - gh // 2
        for i, d in enumerate(digs):
            dx = x0 + i * (gw + scale)
            # Black outline
            oy0, ox0 = y0 - 1, dx - 1
            osy1, osx1 = max(0, -oy0), max(0, -ox0)
            osy2, osx2 = min(oh, H - oy0), min(ow, W - ox0)
            if osy2 > osy1 and osx2 > osx1:
                fy1, fx1 = oy0 + osy1, ox0 + osx1
                frame[fy1:fy1+(osy2-osy1), fx1:fx1+(osx2-osx1)][outlines[d][osy1:osy2, osx1:osx2]] = 0
            # Colored fill
            sy1, sx1 = max(0, -y0), max(0, -dx)
            sy2, sx2 = min(gh, H - y0), min(gw, W - dx)
            if sy2 > sy1 and sx2 > sx1:
                fy1, fx1 = y0 + sy1, dx + sx1
                frame[fy1:fy1+(sy2-sy1), fx1:fx1+(sx2-sx1)][glyphs[d][sy1:sy2, sx1:sx2]] = color_t

    @classmethod
    def execute(cls, track_data, images=None, opacity=0.5, fps=24.0) -> io.NodeOutput:

        from comfy.ldm.sam3.tracker import unpack_masks
        packed = track_data["packed_masks"]
        H, W = track_data["orig_size"]
        if images is not None:
            H, W = images.shape[1], images.shape[2]
        if packed is None:
            N, N_obj = track_data["n_frames"], 0
        else:
            N, N_obj = packed.shape[0], packed.shape[1]

        import uuid
        gpu = comfy.model_management.get_torch_device()
        temp_dir = folder_paths.get_temp_directory()
        filename = f"sam3_track_preview_{uuid.uuid4().hex[:8]}.mp4"
        filepath = os.path.join(temp_dir, filename)
        with av.open(filepath, mode='w') as output:
            stream = output.add_stream('h264', rate=Fraction(round(fps * 1000), 1000))
            stream.width = W
            stream.height = H
            stream.pix_fmt = 'yuv420p'

            frame_cpu = torch.empty(H, W, 3, dtype=torch.uint8)
            frame_np = frame_cpu.numpy()
            if N_obj > 0:
                colors_t = torch.tensor([cls.COLORS[i % len(cls.COLORS)] for i in range(N_obj)],
                                       device=gpu, dtype=torch.float32)
                grid_y = torch.arange(H, device=gpu).view(1, H, 1)
                grid_x = torch.arange(W, device=gpu).view(1, 1, W)
            for t in range(N):
                if images is not None and t < images.shape[0]:
                    frame = images[t].clone()
                else:
                    frame = torch.zeros(H, W, 3)

                if N_obj > 0:
                    frame_binary = unpack_masks(packed[t:t+1].to(gpu))  # [1, N_obj, H, W] bool
                    frame_masks = F.interpolate(frame_binary.float(), size=(H, W), mode="nearest")[0]
                    frame_gpu = frame.to(gpu)
                    bool_masks = frame_masks > 0.5
                    any_mask = bool_masks.any(dim=0)
                    if any_mask.any():
                        obj_idx_map = bool_masks.to(torch.uint8).argmax(dim=0)
                        color_overlay = colors_t[obj_idx_map]
                        mask_3d = any_mask.unsqueeze(-1)
                        frame_gpu = torch.where(mask_3d, frame_gpu * (1 - opacity) + color_overlay * opacity, frame_gpu)
                    area = bool_masks.sum(dim=(-1, -2)).clamp_(min=1)
                    cy = (bool_masks * grid_y).sum(dim=(-1, -2)) // area
                    cx = (bool_masks * grid_x).sum(dim=(-1, -2)) // area
                    has = area > 1
                    scores = track_data.get("scores", [])
                    label_scale = max(3, H // 240) # Scale font with resolutio
                    size_caps = (area.float().sqrt() / 15).clamp_(min=1).long().tolist() #cap per-object so the number doesn't dwarf small masks
                    for obj_idx in range(N_obj):
                        if has[obj_idx]:
                            _cx, _cy = int(cx[obj_idx]), int(cy[obj_idx])
                            color = cls.COLORS[obj_idx % len(cls.COLORS)]
                            obj_scale = min(label_scale, size_caps[obj_idx])
                            score_scale = max(1, obj_scale * 2 // 3)
                            SAM3_TrackPreview._draw_number_gpu(frame_gpu, obj_idx, _cx, _cy, color, scale=obj_scale)
                            if obj_idx < len(scores) and scores[obj_idx] < 1.0:
                                SAM3_TrackPreview._draw_number_gpu(frame_gpu, int(scores[obj_idx] * 100),
                                                                   _cx, _cy + 5 * obj_scale + 3, color, scale=score_scale)
                    frame_cpu.copy_(frame_gpu.clamp_(0, 1).mul_(255).byte())
                else:
                    frame_cpu.copy_(frame.clamp_(0, 1).mul_(255).byte())

                vframe = av.VideoFrame.from_ndarray(frame_np, format='rgb24')
                output.mux(stream.encode(vframe.reformat(format='yuv420p')))
            output.mux(stream.encode(None))
        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(filename, "", io.FolderType.temp)]))


class SAM3_TrackToMask(io.ComfyNode):
    """Select tracked objects by index and output as mask."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_TrackToMask",
            display_name="SAM3 Track to Mask",
            category="image/detection",
            inputs=[
                SAM3TrackData.Input("track_data", display_name="track_data"),
                io.String.Input("object_indices", display_name="object_indices", default="",
                                tooltip="Comma-separated object indices to include (e.g. '0,2,3'). Empty = all objects."),
            ],
            outputs=[
                io.Mask.Output("masks", display_name="masks"),
            ],
        )

    @classmethod
    def execute(cls, track_data, object_indices="") -> io.NodeOutput:
        from comfy.ldm.sam3.tracker import unpack_masks
        packed = track_data["packed_masks"]
        H, W = track_data["orig_size"]

        if packed is None:
            N = track_data["n_frames"]
            return io.NodeOutput(torch.zeros(N, H, W, device=comfy.model_management.intermediate_device()))

        N, N_obj = packed.shape[0], packed.shape[1]

        if object_indices.strip():
            indices = [int(i.strip()) for i in object_indices.split(",") if i.strip().isdigit()]
            indices = [i for i in indices if 0 <= i < N_obj]
        else:
            indices = list(range(N_obj))

        if not indices:
            return io.NodeOutput(torch.zeros(N, H, W, device=comfy.model_management.intermediate_device()))

        union_packed = packed[:, indices[0]].clone()
        for i in indices[1:]:
            union_packed |= packed[:, i]
        union = unpack_masks(union_packed).unsqueeze(1).float()  # [N, 1, Hm, Wm]
        mask_out = F.interpolate(union, size=(H, W), mode="bilinear", align_corners=False)[:, 0]
        return io.NodeOutput(mask_out)


class SAM3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SAM3_Detect,
            SAM3_VideoTrack,
            SAM3_TrackPreview,
            SAM3_TrackToMask,
        ]


async def comfy_entrypoint() -> SAM3Extension:
    return SAM3Extension()
