"""Pose keypoint drawing primitives shared across pose nodes.

`KeypointDraw` exposes native drawing primitives through the same API used by
the pose renderers:

  kd = KeypointDraw()
  kd.draw.circle(canvas, (x, y), radius, color, thickness=-1)
  kd.draw.line(canvas, p1, p2, color, thickness=4)
  kd.draw.fillConvexPoly(canvas, polygon, color)
  kd.draw.ellipse2Poly(center, axes, angle, 0, 360, 1)

It also carries DWPose's body/hand topology + color tables, used by:
  - comfy_extras.nodes_sdpose              (SDPose pose drawing)
  - comfy_extras.pose.export.openpose_2d   (SAM 3D Body 2D pose viz)
  - comfy_extras.pose.export.glb_shared    (SAM 3D Body GLB tables)
"""

import colorsys
import math

import numpy as np
from PIL import Image, ImageDraw

_FULL_ELLIPSE_RADIANS = np.deg2rad(np.arange(361))
_FULL_ELLIPSE_COS = np.cos(_FULL_ELLIPSE_RADIANS)
_FULL_ELLIPSE_SIN = np.sin(_FULL_ELLIPSE_RADIANS)


class KeypointDraw:
    """
    Native pose keypoint drawing primitives and topology data.
    """
    def __init__(self):
        self.draw = self

        # Hand connections (same for both hands)
        self.hand_edges = [
            [0, 1], [1, 2], [2, 3], [3, 4],      # thumb
            [0, 5], [5, 6], [6, 7], [7, 8],      # index
            [0, 9], [9, 10], [10, 11], [11, 12], # middle
            [0, 13], [13, 14], [14, 15], [15, 16], # ring
            [0, 17], [17, 18], [18, 19], [19, 20], # pinky
        ]

        # Head connections (1-indexed, converted to 0-indexed): nose-neck, eyes, ears
        self.head_edges = [
            [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]
        ]

        # Body connections - matching DWPose limbSeq (1-indexed, converted to 0-indexed).
        # body_limbSeq is the full 18-point skeleton (body + head_edges last); the head
        # edges are kept as the trailing entries so callers can toggle them via draw_head.
        self.body_limbSeq = [
            [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
            [10, 11], [2, 12], [12, 13], [13, 14],
        ] + self.head_edges

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

    def circles(self, canvas_np, centers, radius, colors):
        if not centers:
            return
        color_array = np.asarray(colors)
        uniform_color = color_array.ndim == 1
        centers = np.asarray(centers, dtype=np.int32)
        h, w = canvas_np.shape[:2]
        offset_x, offset_y = _disk_offsets(radius)
        if uniform_color:
            _draw_disk_points(canvas_np, centers, offset_x, offset_y, colors, h, w)
        else:
            chunk_size = max(1, 1_000_000 // len(offset_x))
            for start in range(0, len(centers), chunk_size):
                points = centers[start:start + chunk_size]
                xx = points[:, 0, None] + offset_x
                yy = points[:, 1, None] + offset_y
                valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
                color_values = np.broadcast_to(color_array[start:start + chunk_size, None, :], (*xx.shape, 3))
                canvas_np[yy[valid], xx[valid]] = color_values[valid]

    @staticmethod
    def line(canvas_np, pt1, pt2, color, thickness=1, **kwargs):
        """Draw line using Bresenham's algorithm with NumPy operations."""
        h, w = canvas_np.shape[:2]
        line_points = _line_points(pt1, pt2)
        if thickness > 1:
            offset_x, offset_y = _disk_offsets((thickness / 2.0) + 0.5)
            _draw_disk_points(canvas_np, line_points, offset_x, offset_y, color, h, w)
        else:
            valid = (line_points[:, 1] >= 0) & (line_points[:, 1] < h) & (line_points[:, 0] >= 0) & (line_points[:, 0] < w)
            if (valid_points := line_points[valid]).size:
                canvas_np[valid_points[:, 1], valid_points[:, 0]] = color

    def lines(self, canvas_np, starts, ends, colors, thickness=1):
        if not starts:
            return
        h, w = canvas_np.shape[:2]
        if thickness > 1:
            offset_x, offset_y = _disk_offsets((thickness / 2.0) + 0.5)
            for pt1, pt2, color in zip(starts, ends, colors):
                _draw_disk_points(canvas_np, _line_points(pt1, pt2), offset_x, offset_y, color, h, w)
        else:
            for pt1, pt2, color in zip(starts, ends, colors):
                points = _line_points(pt1, pt2)
                valid = (points[:, 1] >= 0) & (points[:, 1] < h) & (points[:, 0] >= 0) & (points[:, 0] < w)
                if (valid_points := points[valid]).size:
                    canvas_np[valid_points[:, 1], valid_points[:, 0]] = color

    @staticmethod
    def fillConvexPoly(canvas_np, pts, color, **kwargs):
        """Fill polygon using vectorized scanline algorithm."""
        region = _convex_poly_mask(pts, canvas_np.shape[0], canvas_np.shape[1])
        if region is None:
            return
        y_min, y_max, x_min, x_max, mask = region
        canvas_np[y_min:y_max, x_min:x_max][mask] = color

    @staticmethod
    def ellipse2Poly(center, axes, angle, arc_start, arc_end, delta=1, **kwargs):
        """Build integer points along an ellipse arc."""
        axes = (axes[0] + 0.5, axes[1] + 0.5)
        angle = angle % 360
        if arc_start > arc_end:
            arc_start, arc_end = arc_end, arc_start
        while arc_start < 0:
            arc_start, arc_end = arc_start + 360, arc_end + 360
        while arc_end > 360:
            arc_end, arc_start = arc_end - 360, arc_start - 360
        if arc_end - arc_start > 360:
            arc_start, arc_end = 0, 360

        if arc_start == 0 and arc_end == 360 and delta == 1:
            x = axes[0] * _FULL_ELLIPSE_COS
            y = axes[1] * _FULL_ELLIPSE_SIN
        else:
            theta = np.deg2rad(np.minimum(np.arange(arc_start, arc_end + delta, delta), arc_end))
            x = axes[0] * np.cos(theta)
            y = axes[1] * np.sin(theta)
        angle_rad = math.radians(angle)
        alpha, beta = math.cos(angle_rad), math.sin(angle_rad)
        pts = np.rint(np.column_stack((
            center[0] + x * alpha - y * beta,
            center[1] + x * beta + y * alpha,
        ))).astype(np.int32)
        keep = np.ones(pts.shape[0], dtype=bool)
        keep[1:] = np.any(pts[1:] != pts[:-1], axis=1)
        pts = pts[keep]

        return pts.tolist() if len(pts) > 1 else [[center[0], center[1]], [center[0], center[1]]]

    def draw_wholebody_keypoints(self, canvas, keypoints, scores=None, threshold=0.3,
                                 draw_body=True, draw_head=True, draw_feet=True, draw_face=True, draw_hands=True,
                                 stick_width=4, face_point_size=3,
                                 marker_radius=4, hand_stick_width=2, hand_marker_radius=4,
                                 limb_alpha=1.0, hand_dot_color=(0, 0, 255)):
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
            draw_head: Toggle head edges/keypoints (nose, eyes, ears) independently of draw_body.
            stick_width: Body limb half-width (passed to ellipse2Poly).
            face_point_size: Radius of the white face dots.
            marker_radius: Radius of body/foot dots. Defaults to 4 (DWPose).
            hand_stick_width: Thickness of hand limb lines. Defaults to 2.
            hand_marker_radius: Radius of hand dots. Defaults to 4.
            limb_alpha: Body-limb alpha blend (0..1). 1.0 = opaque fill (default),
                <1.0 enables per-limb bbox-clipped alpha overlay (DWPose semantics
                where overlapping limbs darken).
            hand_dot_color: Either an (R, G, B) tuple/list of ints for solid-color
                hand dots (default (0, 0, 255), DWPose blue), or a (21, 3) array
                for per-keypoint hand-dot colors (OpenPose-style rainbow palette).

        Returns:
            canvas: The canvas with keypoints drawn
        """
        H, W, C = canvas.shape

        # Normalize hand_dot_color to a (21, 3) int array.
        hdc_arr = np.asarray(hand_dot_color, dtype=int)
        if hdc_arr.ndim == 1:
            hdc_arr = np.tile(hdc_arr.reshape(1, 3), (21, 1))
        hand_dot_tuples = [tuple(int(c) for c in hdc_arr[i]) for i in range(21)]

        do_alpha = float(limb_alpha) < 1.0
        backend = _PillowDraw(canvas, self)
        ellipse2poly = backend.ellipse2Poly
        fill_poly_alpha = backend.fillConvexPolyAlpha
        fill_poly = backend.fillConvexPoly
        draw_circles = backend.circles
        draw_lines = backend.lines

        # Draw body limbs & head connections. body_limbSeq holds the full skeleton
        # with head edges trailing; draw_body / draw_head toggle each group while the
        # color index stays aligned to the full sequence.
        if (draw_body or draw_head) and len(keypoints) >= 18:
            body_core = self.body_limbSeq[:len(self.body_limbSeq) - len(self.head_edges)]
            edges, color_offset = [], 0
            if draw_body:
                edges += body_core
            else:
                color_offset += len(body_core)
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

                polygon = ellipse2poly((int(mY), int(mX)), (int(length / 2), stick_width), int(angle), 0, 360, 1)

                color = self.colors[(i + color_offset) % len(self.colors)]
                if do_alpha:
                    fill_poly_alpha(canvas, polygon, color, limb_alpha)
                else:
                    fill_poly(canvas, polygon, color)

        # Draw body & head keypoints
        if (draw_body or draw_head) and len(keypoints) >= 18:
            head_keypoints = {0, 14, 15, 16, 17} # nose, eyes, ears
            neck_point = 1
            centers, point_colors = [], []
            for i in range(18):
                if not draw_head and i in head_keypoints:
                    continue
                if not draw_body and i not in head_keypoints and i != neck_point:
                    continue
                if scores is not None and scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if 0 <= x < W and 0 <= y < H:
                    centers.append((x, y))
                    point_colors.append(self.colors[i % len(self.colors)])
            draw_circles(canvas, centers, marker_radius, point_colors)

        # Draw foot keypoints (18-23, 6 keypoints)
        if draw_feet and len(keypoints) >= 24:
            centers, point_colors = [], []
            for i in range(18, 24):
                if scores is not None and scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if 0 <= x < W and 0 <= y < H:
                    centers.append((x, y))
                    point_colors.append(self.colors[i % len(self.colors)])
            draw_circles(canvas, centers, marker_radius, point_colors)

        # Draw right hand (92-112)
        if draw_hands and len(keypoints) >= 113:
            eps = 0.01
            starts, ends, line_colors = [], [], []
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
                        starts.append((x1, y1))
                        ends.append((x2, y2))
                        line_colors.append(color)
            draw_lines(canvas, starts, ends, line_colors, thickness=hand_stick_width)

            # Draw right hand keypoints
            centers, point_colors = [], []
            for i in range(92, 113):
                if scores is not None and scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                    centers.append((x, y))
                    point_colors.append(hand_dot_tuples[i - 92])
            draw_circles(canvas, centers, hand_marker_radius, point_colors)

        # Draw left hand (113-133)
        if draw_hands and len(keypoints) >= 134:
            eps = 0.01
            starts, ends, line_colors = [], [], []
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
                        starts.append((x1, y1))
                        ends.append((x2, y2))
                        line_colors.append(color)
            draw_lines(canvas, starts, ends, line_colors, thickness=hand_stick_width)

            # Draw left hand keypoints
            centers, point_colors = [], []
            for i in range(113, 134):
                if scores is not None and i < len(scores) and scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                    centers.append((x, y))
                    point_colors.append(hand_dot_tuples[i - 113])
            draw_circles(canvas, centers, hand_marker_radius, point_colors)

        # Draw face keypoints (24-91) - white dots only, no lines
        if draw_face and len(keypoints) >= 92:
            eps = 0.01
            centers = []
            for i in range(24, 92):
                if scores is not None and scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                    centers.append((x, y))
            draw_circles(canvas, centers, face_point_size, (255, 255, 255))

        backend.finish(canvas)

        return canvas


class _PillowDraw:
    def __init__(self, canvas, native_draw):
        self.canvas = canvas
        self.native_draw = native_draw
        self.polygons = []

    def _flush(self):
        if not self.polygons:
            return
        points = np.concatenate([polygon for polygon, _, _ in self.polygons])
        h, w = self.canvas.shape[:2]
        y_min, y_max = max(0, int(points[:, 1].min())), min(h, int(points[:, 1].max()) + 1)
        x_min, x_max = max(0, int(points[:, 0].min())), min(w, int(points[:, 0].max()) + 1)
        if y_max > y_min and x_max > x_min:
            roi = self.canvas[y_min:y_max, x_min:x_max]
            image = Image.fromarray(roi)
            draw = ImageDraw.Draw(image, "RGBA")
            offset = np.array([x_min, y_min], dtype=np.int32)
            for polygon, color, alpha in self.polygons:
                fill = tuple(color) if alpha is None else (*color, int(round(alpha * 255.0)))
                draw.polygon((polygon - offset).reshape(-1).tolist(), fill=fill)
            roi[:] = np.asarray(image)
        self.polygons.clear()

    @staticmethod
    def ellipse2Poly(center, axes, angle, arc_start, arc_end, delta=1, **kwargs):
        return KeypointDraw.ellipse2Poly(center, axes, angle, arc_start, arc_end, max(delta, 4), **kwargs)

    def fillConvexPolyAlpha(self, canvas, polygon, color, alpha):
        self.polygons.append((np.asarray(polygon, dtype=np.int32), tuple(color), float(alpha)))

    def fillConvexPoly(self, canvas, polygon, color):
        self.polygons.append((np.asarray(polygon, dtype=np.int32), tuple(color), None))

    def circles(self, canvas, centers, radius, colors):
        self._flush()
        self.native_draw.circles(canvas, centers, radius, colors)

    def lines(self, canvas, starts, ends, colors, thickness=1):
        self._flush()
        self.native_draw.lines(canvas, starts, ends, colors, thickness)

    def finish(self, canvas):
        self._flush()


def _disk_offsets(radius):
    radius_int = int(np.ceil(radius))
    offset_y, offset_x = np.mgrid[-radius_int:radius_int + 1, -radius_int:radius_int + 1]
    disk = offset_x * offset_x + offset_y * offset_y <= radius * radius
    return offset_x[disk], offset_y[disk]


def _draw_disk_points(canvas, points, offset_x, offset_y, color, h, w):
    chunk_size = max(1, 1_000_000 // len(offset_x))
    for start in range(0, len(points), chunk_size):
        chunk = points[start:start + chunk_size]
        xx = chunk[:, 0, None] + offset_x
        yy = chunk[:, 1, None] + offset_y
        valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        canvas[yy[valid], xx[valid]] = color


def _line_points(pt1, pt2):
    x0, y0, x1, y1 = *pt1, *pt2
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err, x, y, points = dx - dy, x0, y0, []
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err, x = err - dy, x + sx
        if e2 < dx:
            err, y = err + dx, y + sy
    return np.asarray(points, dtype=np.int32)


def _convex_poly_mask(pts, h, w):
    if len(pts) < 3:
        return None
    pts = np.asarray(pts, dtype=np.int32)
    y_min, y_max = max(0, pts[:, 1].min()), min(h, pts[:, 1].max() + 1)
    x_min, x_max = max(0, pts[:, 0].min()), min(w, pts[:, 0].max() + 1)
    if y_max <= y_min or x_max <= x_min:
        return None

    p1 = pts
    p2 = np.roll(pts, -1, axis=0)
    nonhorizontal = p1[:, 1] != p2[:, 1]
    if not nonhorizontal.any():
        return None
    p1 = p1[nonhorizontal]
    p2 = p2[nonhorizontal]
    swap = p1[:, 1] > p2[:, 1]
    lower = np.where(swap[:, None], p2, p1)
    upper = np.where(swap[:, None], p1, p2)

    starts = np.maximum(lower[:, 1], y_min)
    ends = np.minimum(upper[:, 1], y_max)
    counts = np.maximum(ends - starts, 0)
    keep = counts > 0
    lower = lower[keep]
    upper = upper[keep]
    starts = starts[keep]
    counts = counts[keep]
    edge_idx = np.repeat(np.arange(len(counts)), counts)
    block_starts = np.repeat(np.cumsum(counts) - counts, counts)
    yy = np.repeat(starts, counts) + np.arange(counts.sum()) - block_starts
    intersections = lower[edge_idx, 0] + (yy - lower[edge_idx, 1]) * (upper[edge_idx, 0] - lower[edge_idx, 0]) / (upper[edge_idx, 1] - lower[edge_idx, 1])
    rows = yy - y_min
    left = np.full(y_max - y_min, np.inf)
    right = np.full(y_max - y_min, -np.inf)
    np.minimum.at(left, rows, intersections)
    np.maximum.at(right, rows, intersections)
    xx = np.arange(x_min, x_max, dtype=np.int32)[None, :]
    return y_min, y_max, x_min, x_max, (xx >= left[:, None]) & (xx < right[:, None])
