import torch
import comfy.utils
import comfy.model_management
import numpy as np
import math
import colorsys
from tqdm import tqdm
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_lotus import LotusConditioning


def _preprocess_keypoints(kp_raw, sc_raw):
    """Insert neck keypoint and remap from MMPose to OpenPose ordering.

    Returns (kp, sc) where kp has shape (134, 2) and sc has shape (134,).
    Layout:
      0-17   body  (18 kp, OpenPose order)
      18-23  feet  (6 kp)
      24-91  face  (68 kp)
      92-112 right hand (21 kp)
      113-133 left hand (21 kp)
    """
    kp = np.array(kp_raw, dtype=np.float32)
    sc = np.array(sc_raw, dtype=np.float32)
    if len(kp) >= 17:
        neck = (kp[5] + kp[6]) / 2
        neck_score = min(sc[5], sc[6]) if sc[5] > 0.3 and sc[6] > 0.3 else 0
        kp = np.insert(kp, 17, neck, axis=0)
        sc = np.insert(sc, 17, neck_score)
        mmpose_idx   = np.array([17, 6,  8, 10,  7,  9, 12, 14, 16, 13, 15, 2, 1, 4, 3])
        openpose_idx = np.array([ 1, 2,  3,  4,  6,  7,  8,  9, 10, 12, 13, 14, 15, 16, 17])
        tmp_kp, tmp_sc = kp.copy(), sc.copy()
        tmp_kp[openpose_idx] = kp[mmpose_idx]
        tmp_sc[openpose_idx] = sc[mmpose_idx]
        kp, sc = tmp_kp, tmp_sc
    return kp, sc


def _to_openpose_frames(all_keypoints, all_scores, height, width):
    """Convert raw keypoint lists to a list of OpenPose-style frame dicts.

    Each frame dict contains:
      canvas_width, canvas_height, people: list of person dicts with keys:
        pose_keypoints_2d       - 18 body kp  as flat [x,y,score,...] (absolute pixels)
        foot_keypoints_2d       -  6 foot kp  as flat [x,y,score,...] (absolute pixels)
        face_keypoints_2d       - 70 face kp  as flat [x,y,score,...] (absolute pixels)
                                   indices 0-67: 68 face landmarks
                                   index  68:    right eye (body[14])
                                   index  69:    left  eye (body[15])
        hand_right_keypoints_2d - 21 right-hand kp (absolute pixels)
        hand_left_keypoints_2d  - 21 left-hand  kp (absolute pixels)
    """
    def _flatten(kp_slice, sc_slice):
        return np.stack([kp_slice[:, 0], kp_slice[:, 1], sc_slice], axis=1).flatten().tolist()

    frames = []
    for img_idx in range(len(all_keypoints)):
        people = []
        for kp_raw, sc_raw in zip(all_keypoints[img_idx], all_scores[img_idx]):
            kp, sc = _preprocess_keypoints(kp_raw, sc_raw)
            # 70 face kp = 68 face landmarks + REye (body[14]) + LEye (body[15])
            face_kp = np.concatenate([kp[24:92], kp[[14, 15]]], axis=0)
            face_sc = np.concatenate([sc[24:92], sc[[14, 15]]], axis=0)
            people.append({
                "pose_keypoints_2d":       _flatten(kp[0:18],   sc[0:18]),
                "foot_keypoints_2d":       _flatten(kp[18:24],  sc[18:24]),
                "face_keypoints_2d":       _flatten(face_kp,    face_sc),
                "hand_right_keypoints_2d": _flatten(kp[92:113], sc[92:113]),
                "hand_left_keypoints_2d":  _flatten(kp[113:134], sc[113:134]),
            })
        frames.append({"canvas_width": width, "canvas_height": height, "people": people})
    return frames


class KeypointDraw:
    """
    Pose keypoint drawing class that supports both numpy and cv2 backends.
    """
    def __init__(self):
        try:
            import cv2
            self.draw = cv2
        except ImportError:
            self.draw = self

        # Hand connections (same for both hands)
        self.hand_edges = [
            [0, 1], [1, 2], [2, 3], [3, 4],      # thumb
            [0, 5], [5, 6], [6, 7], [7, 8],      # index
            [0, 9], [9, 10], [10, 11], [11, 12], # middle
            [0, 13], [13, 14], [14, 15], [15, 16], # ring
            [0, 17], [17, 18], [18, 19], [19, 20], # pinky
        ]

        # Body connections - matching DWPose limbSeq (1-indexed, converted to 0-indexed)
        self.body_limbSeq = [
            [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
            [10, 11], [2, 12], [12, 13], [13, 14]
        ]

        # Head connections (1-indexed, converted to 0-indexed)
        self.head_edges = [
            [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]
        ]

        # Colors matching DWPose
        self.colors = [
            [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
            [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
            [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
        ]

    @staticmethod
    def circle(canvas_np, center, radius, color, **kwargs):
        """Draw a filled circle using NumPy vectorized operations."""
        cx, cy = center
        h, w = canvas_np.shape[:2]

        radius_int = int(np.ceil(radius))

        y_min, y_max = max(0, cy - radius_int), min(h, cy + radius_int + 1)
        x_min, x_max = max(0, cx - radius_int), min(w, cx + radius_int + 1)

        if y_max <= y_min or x_max <= x_min:
            return

        y, x = np.ogrid[y_min:y_max, x_min:x_max]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        canvas_np[y_min:y_max, x_min:x_max][mask] = color

    @staticmethod
    def line(canvas_np, pt1, pt2, color, thickness=1, **kwargs):
        """Draw line using Bresenham's algorithm with NumPy operations."""
        x0, y0, x1, y1 = *pt1, *pt2
        h, w = canvas_np.shape[:2]
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err, x, y, line_points = dx - dy, x0, y0, []

        while True:
            line_points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err, x = err - dy, x + sx
            if e2 < dx:
                err, y = err + dx, y + sy

        if thickness > 1:
            radius, radius_int = (thickness / 2.0) + 0.5, int(np.ceil((thickness / 2.0) + 0.5))
            for px, py in line_points:
                y_min, y_max, x_min, x_max = max(0, py - radius_int), min(h, py + radius_int + 1), max(0, px - radius_int), min(w, px + radius_int + 1)
                if y_max > y_min and x_max > x_min:
                    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
                    canvas_np[y_min:y_max, x_min:x_max][(xx - px)**2 + (yy - py)**2 <= radius**2] = color
        else:
            line_points = np.array(line_points)
            valid = (line_points[:, 1] >= 0) & (line_points[:, 1] < h) & (line_points[:, 0] >= 0) & (line_points[:, 0] < w)
            if (valid_points := line_points[valid]).size:
                canvas_np[valid_points[:, 1], valid_points[:, 0]] = color

    @staticmethod
    def fillConvexPoly(canvas_np, pts, color, **kwargs):
        """Fill polygon using vectorized scanline algorithm."""
        if len(pts) < 3:
            return
        pts = np.array(pts, dtype=np.int32)
        h, w = canvas_np.shape[:2]
        y_min, y_max, x_min, x_max = max(0, pts[:, 1].min()), min(h, pts[:, 1].max() + 1), max(0, pts[:, 0].min()), min(w, pts[:, 0].max() + 1)
        if y_max <= y_min or x_max <= x_min:
            return
        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        mask = np.zeros((y_max - y_min, x_max - x_min), dtype=bool)

        for i in range(len(pts)):
            p1, p2 = pts[i], pts[(i + 1) % len(pts)]
            y1, y2 = p1[1], p2[1]
            if y1 == y2:
                continue
            if y1 > y2:
                p1, p2, y1, y2 = p2, p1, p2[1], p1[1]
            if not (edge_mask := (yy >= y1) & (yy < y2)).any():
                continue
            mask ^= edge_mask & (xx >= p1[0] + (yy - y1) * (p2[0] - p1[0]) / (y2 - y1))

        canvas_np[y_min:y_max, x_min:x_max][mask] = color

    @staticmethod
    def ellipse2Poly(center, axes, angle, arc_start, arc_end, delta=1, **kwargs):
        """Python implementation of cv2.ellipse2Poly."""
        axes = (axes[0] + 0.5, axes[1] + 0.5) # to better match cv2 output
        angle = angle % 360
        if arc_start > arc_end:
            arc_start, arc_end = arc_end, arc_start
        while arc_start < 0:
            arc_start, arc_end = arc_start + 360, arc_end + 360
        while arc_end > 360:
            arc_end, arc_start = arc_end - 360, arc_start - 360
        if arc_end - arc_start > 360:
            arc_start, arc_end = 0, 360

        angle_rad = math.radians(angle)
        alpha, beta = math.cos(angle_rad), math.sin(angle_rad)
        pts = []
        for i in range(arc_start, arc_end + delta, delta):
            theta_rad = math.radians(min(i, arc_end))
            x, y = axes[0] * math.cos(theta_rad), axes[1] * math.sin(theta_rad)
            pts.append([int(round(center[0] + x * alpha - y * beta)), int(round(center[1] + x * beta + y * alpha))])

        unique_pts, prev_pt = [], (float('inf'), float('inf'))
        for pt in pts:
            if (pt_tuple := tuple(pt)) != prev_pt:
                unique_pts.append(pt)
                prev_pt = pt_tuple

        return unique_pts if len(unique_pts) > 1 else [[center[0], center[1]], [center[0], center[1]]]

    def draw_wholebody_keypoints(self, canvas, keypoints, scores=None, threshold=0.3,
                                 draw_body=True, draw_head=True, draw_feet=True, draw_face=True, draw_hands=True, stick_width=4, face_point_size=3):
        """
        Draw wholebody keypoints (134 keypoints after processing) in DWPose style.

        Expected keypoint format (after neck insertion and remapping):
        - Body: 0-17 (18 keypoints in OpenPose format, neck at index 1)
        - Foot: 18-23 (6 keypoints)
        - Face: 24-91 (68 landmarks)
        - Right hand: 92-112 (21 keypoints)
        - Left hand: 113-133 (21 keypoints)

        Args:
            canvas: The canvas to draw on (numpy array)
            keypoints: Array of keypoint coordinates
            scores: Optional confidence scores for each keypoint
            threshold: Minimum confidence threshold for drawing keypoints

        Returns:
            canvas: The canvas with keypoints drawn
        """
        H, W, C = canvas.shape

        # Draw body limbs & head connections
        if (draw_body or draw_head) and len(keypoints) >= 18:
            colorIndexOffset = 0
            edges = []
            if draw_body:
                edges += self.body_limbSeq
            else:
                colorIndexOffset += len(self.body_limbSeq)
            if draw_head:
                edges += self.head_edges
            for i, limb in enumerate(edges):
                # Convert from 1-indexed to 0-indexed
                idx1, idx2 = limb[0] - 1, limb[1] - 1

                if idx1 >= 18 or idx2 >= 18:
                    continue

                if scores is not None:
                    if scores[idx1] < threshold or scores[idx2] < threshold:
                        continue

                Y = [keypoints[idx1][0], keypoints[idx2][0]]
                X = [keypoints[idx1][1], keypoints[idx2][1]]
                mX, mY = (X[0] + X[1]) / 2, (Y[0] + Y[1]) / 2
                length = math.sqrt((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2)

                if length < 1:
                    continue

                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

                polygon = self.draw.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stick_width), int(angle), 0, 360, 1)

                self.draw.fillConvexPoly(canvas, polygon, self.colors[(i + colorIndexOffset) % len(self.colors)])

        # Draw body & head keypoints
        if (draw_body or draw_head) and len(keypoints) >= 18:
            head_keypoints = {0, 14, 15, 16, 17} # nose, eyes, ears
            neck_point = 1
            for i in range(18):
                if not draw_head and i in head_keypoints:
                    continue
                if not draw_body and i not in head_keypoints and i != neck_point:
                    continue
                if scores is not None and scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if 0 <= x < W and 0 <= y < H:
                    self.draw.circle(canvas, (x, y), 4, self.colors[i % len(self.colors)], thickness=-1)

        # Draw foot keypoints (18-23, 6 keypoints)
        if draw_feet and len(keypoints) >= 24:
            for i in range(18, 24):
                if scores is not None and scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if 0 <= x < W and 0 <= y < H:
                    self.draw.circle(canvas, (x, y), 4, self.colors[i % len(self.colors)], thickness=-1)

        # Draw right hand (92-112)
        if draw_hands and len(keypoints) >= 113:
            eps = 0.01
            for ie, edge in enumerate(self.hand_edges):
                idx1, idx2 = 92 + edge[0], 92 + edge[1]
                if scores is not None:
                    if scores[idx1] < threshold or scores[idx2] < threshold:
                        continue

                x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1])
                x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])

                if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                    if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                        # HSV to RGB conversion for rainbow colors
                        r, g, b = colorsys.hsv_to_rgb(ie / float(len(self.hand_edges)), 1.0, 1.0)
                        color = (int(r * 255), int(g * 255), int(b * 255))
                        self.draw.line(canvas, (x1, y1), (x2, y2), color, thickness=2)

            # Draw right hand keypoints
            for i in range(92, 113):
                if scores is not None and scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                    self.draw.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)

        # Draw left hand (113-133)
        if draw_hands and len(keypoints) >= 134:
            eps = 0.01
            for ie, edge in enumerate(self.hand_edges):
                idx1, idx2 = 113 + edge[0], 113 + edge[1]
                if scores is not None:
                    if scores[idx1] < threshold or scores[idx2] < threshold:
                        continue

                x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1])
                x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])

                if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                    if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                        # HSV to RGB conversion for rainbow colors
                        r, g, b = colorsys.hsv_to_rgb(ie / float(len(self.hand_edges)), 1.0, 1.0)
                        color = (int(r * 255), int(g * 255), int(b * 255))
                        self.draw.line(canvas, (x1, y1), (x2, y2), color, thickness=2)

            # Draw left hand keypoints
            for i in range(113, 134):
                if scores is not None and i < len(scores) and scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                    self.draw.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)

        # Draw face keypoints (24-91) - white dots only, no lines
        if draw_face and len(keypoints) >= 92:
            eps = 0.01
            for i in range(24, 92):
                if scores is not None and scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                    self.draw.circle(canvas, (x, y), face_point_size, (255, 255, 255), thickness=-1)

        return canvas

class SDPoseDrawKeypoints(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SDPoseDrawKeypoints",
            display_name="SDPose Draw Keypoints",
            category="image/detection",
            search_aliases=["openpose", "pose detection", "preprocessor", "keypoints", "pose"],
            inputs=[
                io.Custom("POSE_KEYPOINT").Input("keypoints"),
                io.Boolean.Input("draw_body", default=True),
                io.Boolean.Input("draw_hands", default=True),
                io.Boolean.Input("draw_face", default=True),
                io.Boolean.Input("draw_feet", default=False),
                io.Int.Input("stick_width", default=4, min=1, max=10, step=1),
                io.Int.Input("face_point_size", default=3, min=1, max=10, step=1),
                io.Float.Input("score_threshold", default=0.3, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("draw_head", default=True),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, keypoints, draw_body, draw_hands, draw_face, draw_feet, stick_width, face_point_size, score_threshold, draw_head) -> io.NodeOutput:
        if not keypoints:
            return io.NodeOutput(torch.zeros((1, 64, 64, 3), dtype=torch.float32))
        height = keypoints[0]["canvas_height"]
        width  = keypoints[0]["canvas_width"]

        def _parse(flat, n):
            arr = np.array(flat, dtype=np.float32).reshape(n, 3)
            return arr[:, :2], arr[:, 2]

        def _zeros(n):
            return np.zeros((n, 2), dtype=np.float32), np.zeros(n, dtype=np.float32)

        pose_outputs = []
        drawer = KeypointDraw()

        for frame in tqdm(keypoints, desc="Drawing keypoints on frames"):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            for person in frame["people"]:
                body_kp,  body_sc  = _parse(person["pose_keypoints_2d"],       18)
                foot_raw = person.get("foot_keypoints_2d")
                foot_kp,  foot_sc  = _parse(foot_raw, 6) if foot_raw else _zeros(6)
                face_kp,  face_sc  = _parse(person["face_keypoints_2d"],       70)
                face_kp,  face_sc  = face_kp[:68], face_sc[:68]  # drop appended eye kp; body already draws them
                rhand_kp, rhand_sc = _parse(person["hand_right_keypoints_2d"], 21)
                lhand_kp, lhand_sc = _parse(person["hand_left_keypoints_2d"],  21)

                kp = np.concatenate([body_kp, foot_kp, face_kp, rhand_kp, lhand_kp], axis=0)
                sc = np.concatenate([body_sc, foot_sc, face_sc, rhand_sc, lhand_sc], axis=0)

                canvas = drawer.draw_wholebody_keypoints(
                    canvas, kp, sc,
                    threshold=score_threshold,
                    draw_body=draw_body, draw_head=draw_head, draw_feet=draw_feet,
                    draw_face=draw_face, draw_hands=draw_hands,
                    stick_width=stick_width, face_point_size=face_point_size,
                )
            pose_outputs.append(canvas)

        pose_outputs_np = np.stack(pose_outputs) if len(pose_outputs) > 1 else np.expand_dims(pose_outputs[0], 0)
        final_pose_output = torch.from_numpy(pose_outputs_np).to(
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype()) / 255.0
        return io.NodeOutput(final_pose_output)

class SDPoseKeypointExtractor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SDPoseKeypointExtractor",
            display_name="SDPose Keypoint Extractor",
            category="image/detection",
            search_aliases=["openpose", "pose detection", "preprocessor", "keypoints", "sdpose"],
            description="Extract pose keypoints from images using the SDPose model: https://huggingface.co/Comfy-Org/SDPose/tree/main/checkpoints",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.Image.Input("image"),
                io.Int.Input("batch_size", default=16, min=1, max=10000, step=1),
                io.BoundingBox.Input("bboxes", optional=True, force_input=True, tooltip="Optional bounding boxes for more accurate detections. Required for multi-person detection."),
            ],
            outputs=[
                io.Custom("POSE_KEYPOINT").Output("keypoints", tooltip="Keypoints in OpenPose frame format (canvas_width, canvas_height, people)"),
            ],
        )

    @classmethod
    def execute(cls, model, vae, image, batch_size, bboxes=None) -> io.NodeOutput:

        height, width = image.shape[-3], image.shape[-2]
        context = LotusConditioning().execute().result[0]

        # Use output_block_patch to capture the last 640-channel feature
        def output_patch(h, hsp, transformer_options):
            nonlocal captured_feat
            if h.shape[1] == 640:  # Capture the features for wholebody
                captured_feat = h.clone()
            return h, hsp

        model_clone = model.clone()
        model_clone.model_options["transformer_options"] = {"patches": {"output_block_patch": [output_patch]}}

        if not hasattr(model.model.diffusion_model, 'heatmap_head'):
            raise ValueError("The provided model does not have a heatmap_head. Please use SDPose model from here https://huggingface.co/Comfy-Org/SDPose/tree/main/checkpoints.")

        head = model.model.diffusion_model.heatmap_head
        total_images = image.shape[0]
        captured_feat = None

        model_w = int(head.heatmap_size[0]) * 4   # 192 * 4 = 768
        model_h = int(head.heatmap_size[1]) * 4   # 256 * 4 = 1024

        def _resize_to_model(imgs):
            """Stretch BHWC images to (model_h, model_w), model expects no aspect preservation."""
            h, w = imgs.shape[-3], imgs.shape[-2]
            method = "area" if (model_h <= h and model_w <= w) else "bilinear"
            chw = imgs.permute(0, 3, 1, 2).float()
            scaled = comfy.utils.common_upscale(chw, model_w, model_h, upscale_method=method, crop="disabled")
            return scaled.permute(0, 2, 3, 1), model_w / w, model_h / h

        def _remap_keypoints(kp, scale_x, scale_y, offset_x=0, offset_y=0):
            """Remap keypoints from model space back to original image space."""
            kp = kp.copy() if isinstance(kp, np.ndarray) else np.array(kp, dtype=np.float32)
            invalid = kp[..., 0] < 0
            kp[..., 0] = kp[..., 0] / scale_x + offset_x
            kp[..., 1] = kp[..., 1] / scale_y + offset_y
            kp[invalid] = -1
            return kp

        def _run_on_latent(latent_batch):
            """Run one forward pass and return (keypoints_list, scores_list) for the batch."""
            nonlocal captured_feat
            captured_feat = None
            _ = comfy.sample.sample(
                model_clone,
                noise=torch.zeros_like(latent_batch),
                steps=1, cfg=1.0,
                sampler_name="euler", scheduler="simple",
                positive=context, negative=context,
                latent_image=latent_batch, disable_noise=True, disable_pbar=True,
            )
            return head(captured_feat)  # keypoints_batch, scores_batch

        # all_keypoints / all_scores are lists-of-lists:
        #   outer index = input image index
        #   inner index = detected person (one per bbox, or one for full-image)
        all_keypoints = []  # shape: [n_images][n_persons]
        all_scores = []     # shape: [n_images][n_persons]
        pbar = comfy.utils.ProgressBar(total_images)

        if bboxes is not None:
            if not isinstance(bboxes, list):
                bboxes = [[bboxes]]
            elif len(bboxes) == 0:
                bboxes = [None] * total_images
            # --- bbox-crop mode: one forward pass per crop -------------------------
            for img_idx in tqdm(range(total_images), desc="Extracting keypoints from crops"):
                img = image[img_idx:img_idx + 1]  # (1, H, W, C)
                # Broadcasting: if fewer bbox lists than images, repeat the last one.
                img_bboxes = bboxes[min(img_idx, len(bboxes) - 1)] if bboxes else None

                img_keypoints = []
                img_scores = []

                if img_bboxes:
                    for bbox in img_bboxes:
                        x1 = max(0, int(bbox["x"]))
                        y1 = max(0, int(bbox["y"]))
                        x2 = min(width,  int(bbox["x"] + bbox["width"]))
                        y2 = min(height, int(bbox["y"] + bbox["height"]))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        crop = img[:, y1:y2, x1:x2, :]  # (1, crop_h, crop_w, C)
                        crop_resized, sx, sy = _resize_to_model(crop)

                        latent_crop = vae.encode(crop_resized)
                        kp_batch, sc_batch = _run_on_latent(latent_crop)
                        kp = _remap_keypoints(kp_batch[0], sx, sy, x1, y1)
                        img_keypoints.append(kp)
                        img_scores.append(sc_batch[0])
                else:
                    img_resized, sx, sy = _resize_to_model(img)
                    latent_img = vae.encode(img_resized)
                    kp_batch, sc_batch = _run_on_latent(latent_img)
                    img_keypoints.append(_remap_keypoints(kp_batch[0], sx, sy))
                    img_scores.append(sc_batch[0])

                all_keypoints.append(img_keypoints)
                all_scores.append(img_scores)
                pbar.update(1)

        else: # full-image mode, batched
            for batch_start in tqdm(range(0, total_images, batch_size), desc="Extracting keypoints"):
                batch_resized, sx, sy = _resize_to_model(image[batch_start:batch_start + batch_size])
                latent_batch = vae.encode(batch_resized)
                kp_batch, sc_batch = _run_on_latent(latent_batch)

                for kp, sc in zip(kp_batch, sc_batch):
                    all_keypoints.append([_remap_keypoints(kp, sx, sy)])
                    all_scores.append([sc])

                pbar.update(len(kp_batch))

        openpose_frames = _to_openpose_frames(all_keypoints, all_scores, height, width)
        return io.NodeOutput(openpose_frames)


def get_face_bboxes(kp2ds, scale, image_shape):
    h, w = image_shape
    kp2ds_face = kp2ds.copy()[1:] * (w, h)

    min_x, min_y = np.min(kp2ds_face, axis=0)
    max_x, max_y = np.max(kp2ds_face, axis=0)

    initial_width = max_x - min_x
    initial_height = max_y - min_y

    if initial_width <= 0 or initial_height <= 0:
        return [0, 0, 0, 0]

    initial_area = initial_width * initial_height

    expanded_area = initial_area * scale

    new_width = np.sqrt(expanded_area * (initial_width / initial_height))
    new_height = np.sqrt(expanded_area * (initial_height / initial_width))

    delta_width = (new_width - initial_width) / 2
    delta_height = (new_height - initial_height) / 4

    expanded_min_x = max(min_x - delta_width, 0)
    expanded_max_x = min(max_x + delta_width, w)
    expanded_min_y = max(min_y - 3 * delta_height, 0)
    expanded_max_y = min(max_y + delta_height, h)

    return [int(expanded_min_x), int(expanded_max_x), int(expanded_min_y), int(expanded_max_y)]

class SDPoseFaceBBoxes(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SDPoseFaceBBoxes",
            display_name="SDPose Face Bounding Boxes",
            category="image/detection",
            search_aliases=["face bbox", "face bounding box", "pose", "keypoints"],
            inputs=[
                io.Custom("POSE_KEYPOINT").Input("keypoints"),
                io.Float.Input("scale", default=1.5, min=1.0, max=10.0, step=0.1, tooltip="Multiplier for the bounding box area around each detected face."),
                io.Boolean.Input("force_square", default=True, tooltip="Expand the shorter bbox axis so the crop region is always square."),
            ],
            outputs=[
                io.BoundingBox.Output("bboxes", tooltip="Face bounding boxes per frame, compatible with SDPoseKeypointExtractor bboxes input."),
            ],
        )

    @classmethod
    def execute(cls, keypoints, scale, force_square) -> io.NodeOutput:
        all_bboxes = []
        for frame in keypoints:
            h = frame["canvas_height"]
            w = frame["canvas_width"]
            frame_bboxes = []
            for person in frame["people"]:
                face_flat = person.get("face_keypoints_2d", [])
                if not face_flat:
                    continue
                # Parse absolute-pixel face keypoints (70 kp: 68 landmarks + REye + LEye)
                face_arr = np.array(face_flat, dtype=np.float32).reshape(-1, 3)
                face_xy  = face_arr[:, :2]  # (70, 2) in absolute pixels

                kp_norm = face_xy / np.array([w, h], dtype=np.float32)
                kp_padded = np.vstack([np.zeros((1, 2), dtype=np.float32), kp_norm])  # (71, 2)

                x1, x2, y1, y2 = get_face_bboxes(kp_padded, scale, (h, w))
                if x2 > x1 and y2 > y1:
                    if force_square:
                        bw, bh = x2 - x1, y2 - y1
                        if bw != bh:
                            side = max(bw, bh)
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            half = side // 2
                            x1 = max(0, cx - half)
                            y1 = max(0, cy - half)
                            x2 = min(w, x1 + side)
                            y2 = min(h, y1 + side)
                            # Re-anchor if clamped
                            x1 = max(0, x2 - side)
                            y1 = max(0, y2 - side)
                    frame_bboxes.append({"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1})

            all_bboxes.append(frame_bboxes)

        return io.NodeOutput(all_bboxes)


class CropByBBoxes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CropByBBoxes",
            display_name="Crop By Bounding Boxes",
            category="image/transform",
            search_aliases=["crop", "face crop", "bbox crop", "pose", "bounding box"],
            description="Crop and resize regions from the input image batch based on provided bounding boxes.",
            inputs=[
                io.Image.Input("image"),
                io.BoundingBox.Input("bboxes", force_input=True),
                io.Int.Input("output_width",  default=512, min=64, max=4096, step=8, tooltip="Width each crop is resized to."),
                io.Int.Input("output_height", default=512, min=64, max=4096, step=8, tooltip="Height each crop is resized to."),
                io.Int.Input("padding", default=0, min=0, max=1024, step=1, tooltip="Extra padding in pixels added on each side of the bbox before cropping."),
                io.Combo.Input("keep_aspect", options=["stretch", "pad"], default="stretch", tooltip="Whether to stretch the crop to fit the output size, or pad with black pixels to preserve aspect ratio."),
            ],
            outputs=[
                io.Image.Output(tooltip="All crops stacked into a single image batch."),
            ],
        )

    @classmethod
    def execute(cls, image, bboxes, output_width, output_height, padding, keep_aspect="stretch") -> io.NodeOutput:
        total_frames = image.shape[0]
        img_h = image.shape[1]
        img_w = image.shape[2]
        num_ch = image.shape[3]

        if not isinstance(bboxes, list):
            bboxes = [[bboxes]]
        elif len(bboxes) == 0:
            return io.NodeOutput(image)

        crops = []

        for frame_idx in range(total_frames):
            frame_bboxes = bboxes[min(frame_idx, len(bboxes) - 1)]
            if not frame_bboxes:
                continue

            frame_chw = image[frame_idx].permute(2, 0, 1).unsqueeze(0)  # BHWC → BCHW (1, C, H, W)

            # Union all bboxes for this frame into a single crop region
            x1 = min(int(b["x"]) for b in frame_bboxes)
            y1 = min(int(b["y"]) for b in frame_bboxes)
            x2 = max(int(b["x"] + b["width"])  for b in frame_bboxes)
            y2 = max(int(b["y"] + b["height"]) for b in frame_bboxes)

            if padding > 0:
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(img_w, x2 + padding)
                y2 = min(img_h, y2 + padding)

            x1, x2 = max(0, x1), min(img_w, x2)
            y1, y2 = max(0, y1), min(img_h, y2)

            # Fallback for empty/degenerate crops
            if x2 <= x1 or y2 <= y1:
                fallback_size = int(min(img_h, img_w) * 0.3)
                fb_x1 = max(0, (img_w - fallback_size) // 2)
                fb_y1 = max(0, int(img_h * 0.1))
                fb_x2 = min(img_w, fb_x1 + fallback_size)
                fb_y2 = min(img_h, fb_y1 + fallback_size)
                if fb_x2 <= fb_x1 or fb_y2 <= fb_y1:
                    crops.append(torch.zeros(1, num_ch, output_height, output_width, dtype=image.dtype, device=image.device))
                    continue
                x1, y1, x2, y2 = fb_x1, fb_y1, fb_x2, fb_y2

            crop_chw = frame_chw[:, :, y1:y2, x1:x2]  # (1, C, crop_h, crop_w)

            if keep_aspect == "pad":
                crop_h, crop_w = y2 - y1, x2 - x1
                scale = min(output_width / crop_w, output_height / crop_h)
                scaled_w = int(round(crop_w * scale))
                scaled_h = int(round(crop_h * scale))
                scaled = comfy.utils.common_upscale(crop_chw, scaled_w, scaled_h, upscale_method="area", crop="disabled")
                pad_left = (output_width  - scaled_w) // 2
                pad_top  = (output_height - scaled_h) // 2
                resized = torch.zeros(1, num_ch, output_height, output_width, dtype=image.dtype, device=image.device)
                resized[:, :, pad_top:pad_top + scaled_h, pad_left:pad_left + scaled_w] = scaled
            else:  # "stretch"
                resized = comfy.utils.common_upscale(crop_chw, output_width, output_height, upscale_method="area", crop="disabled")
            crops.append(resized)

        if not crops:
            return io.NodeOutput(image)

        out_images = torch.cat(crops, dim=0).permute(0, 2, 3, 1)  # (N, H, W, C)
        return io.NodeOutput(out_images)


class SDPoseExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SDPoseKeypointExtractor,
            SDPoseDrawKeypoints,
            SDPoseFaceBBoxes,
            CropByBBoxes,
        ]

async def comfy_entrypoint() -> SDPoseExtension:
    return SDPoseExtension()
