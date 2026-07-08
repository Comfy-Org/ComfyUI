"""ComfyUI nodes for the pure-PyTorch MediaPipe Face Landmarker port.

Custom IO types:
  FACE_LANDMARKER  — FaceLandmarkerModel wrapper (ModelPatcher inside)
  FACE_LANDMARKS   — {"frames": List[List[face_dict]], "image_size": (H, W),
                      "connection_sets": dict[str, frozenset[(int, int)]]}
                     face_dict: bbox_xyxy, blendshapes, landmarks_xy,
                                landmarks_3d, presence, score, transformation_matrix

MediaPipeFaceLandmarker also emits the core BOUNDING_BOX type — pair with DrawBBoxes.
"""


import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw
from tqdm.auto import tqdm
from typing_extensions import override

import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io

from comfy_extras.mediapipe.face_landmarker import FaceLandmarker
from comfy_extras.mediapipe.face_geometry import transformation_matrix_from_detection


FaceDetectionType = io.Custom("FACE_DETECTION_MODEL")
FaceLandmarksType = io.Custom("FACE_LANDMARKS")

_CANONICAL_KEYS = ("canonical_vertices", "procrustes_indices", "procrustes_weights")
_CONTOUR_PARTS = ("face_oval", "left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "lips")


class FaceLandmarkerModel:
    """Loaded FaceLandmarker variants + ModelPatcher per variant.

    Safetensors layout: `detector_short.*` / `detector_full.*` plus shared
    `mesh.*`, `blendshapes.*`, `canonical_*`, and `topology.*`.
    PReLU forces plain-nn / fp32 (manual_cast strands buffers across devices).
    """

    def __init__(self, state_dict: dict):
        self.load_device = comfy.model_management.text_encoder_device()
        offload_device = comfy.model_management.text_encoder_offload_device()
        self.dtype = torch.float32

        # FACEMESH_* connection sets, embedded as int32 (N, 2) under topology.*.
        base: dict[str, frozenset] = {}
        for k in [k for k in state_dict if k.startswith("topology.")]:
            base[k[len("topology."):]] = frozenset(map(tuple, state_dict.pop(k).tolist()))
        base["contours"] = frozenset().union(*(base[p] for p in _CONTOUR_PARTS))
        base["all"] = base["contours"] | base["irises"] | base["nose"]

        self.connection_sets: dict[str, frozenset] = base
        self.canonical_data: dict[str, np.ndarray] = {k: state_dict.pop(k).numpy() for k in _CANONICAL_KEYS}

        shared = {k: v for k, v in state_dict.items() if k.startswith(("mesh.", "blendshapes."))}

        self.models: dict[str, FaceLandmarker] = {}
        self.patchers: dict[str, comfy.model_patcher.ModelPatcher] = {}
        for variant in ("short", "full"):
            prefix = f"detector_{variant}."
            sub = dict(shared)
            sub.update({f"detector.{k[len(prefix):]}": v for k, v in state_dict.items() if k.startswith(prefix)})
            fl = FaceLandmarker(device=offload_device, dtype=self.dtype, operations=None, detector_variant=variant).eval()
            fl.load_state_dict(sub, strict=False)

            self.models[variant] = fl
            self.patchers[variant] = comfy.model_patcher.CoreModelPatcher(
                fl, load_device=self.load_device, offload_device=offload_device,
                size=comfy.model_management.module_size(fl),
            )

    def detect_batch(self, images, num_faces: int, score_thresh: float, variant: str):
        comfy.model_management.load_model_gpu(self.patchers[variant])
        return self.models[variant].detect_batch(images, num_faces=num_faces, score_thresh=score_thresh)


def _image_to_uint8(image: torch.Tensor) -> np.ndarray:
    return image[..., :3].mul(255.0).add_(0.5).clamp_(0, 255).to(torch.uint8).cpu().numpy()


def _parse_color(color: str) -> tuple[int, int, int]:
    try:
        return ImageColor.getrgb(color)[:3]
    except ValueError:
        return (0, 255, 0)


def _copy_face(face: dict) -> dict:
    """Shallow copy of a face_dict with array-fields cloned so callers can mutate."""
    return {
        "bbox_xyxy":    face["bbox_xyxy"].copy(),
        "blendshapes":  dict(face["blendshapes"]),
        "landmarks_xy": face["landmarks_xy"].copy(),
        "landmarks_3d": face["landmarks_3d"].copy(),
        "presence":     face["presence"],
        "score":        face["score"],
    }


def _lerp_face(a: dict, b: dict, t: float) -> dict:
    return {
        "bbox_xyxy":    (1 - t) * a["bbox_xyxy"]    + t * b["bbox_xyxy"],
        "blendshapes":  {k: (1 - t) * a["blendshapes"][k] + t * b["blendshapes"][k] for k in a["blendshapes"]},
        "landmarks_xy": (1 - t) * a["landmarks_xy"] + t * b["landmarks_xy"],
        "landmarks_3d": (1 - t) * a["landmarks_3d"] + t * b["landmarks_3d"],
        "presence":     (1 - t) * a["presence"] + t * b["presence"],
        "score":        (1 - t) * a["score"]    + t * b["score"],
    }


def _match_faces(a: list[dict], b: list[dict]) -> list[tuple[int, int]]:
    """Greedy nearest-neighbour pairing of faces between two frames by bbox
    centre distance. Unmatched (when counts differ) are dropped."""
    if not a or not b:
        return []
    centers_a = np.array([(0.5 * (f["bbox_xyxy"][0] + f["bbox_xyxy"][2]),
                           0.5 * (f["bbox_xyxy"][1] + f["bbox_xyxy"][3])) for f in a])
    centers_b = np.array([(0.5 * (f["bbox_xyxy"][0] + f["bbox_xyxy"][2]),
                           0.5 * (f["bbox_xyxy"][1] + f["bbox_xyxy"][3])) for f in b])
    dists = np.linalg.norm(centers_a[:, None] - centers_b[None], axis=-1)
    pairs: list[tuple[int, int]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    candidates = sorted((dists[ia, ib], ia, ib) for ia in range(len(a)) for ib in range(len(b)))
    for _, ia, ib in candidates:
        if ia in used_a or ib in used_b:
            continue
        pairs.append((ia, ib))
        used_a.add(ia)
        used_b.add(ib)
    return pairs


def _fill_missing_frames(frames: list[list[dict]], mode: str) -> None:
    """In-place fill empty frame slots from neighbouring detections. Multi-face
    aware: pairs faces across bracketing frames by greedy bbox-centre NN.
    When counts differ, unmatched faces are dropped from the synthesised frame."""
    if mode == "empty":
        return
    valid = [i for i, fr in enumerate(frames) if fr]
    if not valid:
        return  # nothing to fill from
    if mode == "previous":
        last: list[dict] = []
        for i, fr in enumerate(frames):
            if fr:
                last = fr
            elif last:
                frames[i] = [_copy_face(f) for f in last]
        return
    # interpolate: lerp between bracketing valid frames; clamp at ends.
    for i in range(len(frames)):
        if frames[i]:
            continue
        prev_i = max((v for v in valid if v < i), default=None)
        next_i = min((v for v in valid if v > i), default=None)
        if prev_i is None:
            frames[i] = [_copy_face(f) for f in frames[next_i]]
        elif next_i is None:
            frames[i] = [_copy_face(f) for f in frames[prev_i]]
        else:
            t = (i - prev_i) / (next_i - prev_i)
            pairs = _match_faces(frames[prev_i], frames[next_i])
            frames[i] = [_lerp_face(frames[prev_i][a], frames[next_i][b], t) for a, b in pairs]


def _ordered_rings(edges: frozenset[tuple[int, int]]) -> list[list[int]]:
    """Walk an unordered edge set into one or more closed-loop vertex rings
    (handles multi-loop sets like FACEMESH_LIPS: outer + inner)."""
    adj: dict[int, set[int]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    visited: set[int] = set()
    rings: list[list[int]] = []
    for start in adj:
        if start in visited:
            continue
        ring = [start]
        visited.add(start)
        prev, cur = -1, start
        while True:
            nxt = next((v for v in adj[cur] if v != prev), None)
            if nxt is None or nxt == start:
                break
            ring.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
        rings.append(ring)
    return rings


class LoadMediaPipeFaceLandmarker(io.ComfyNode):
    """Load MediaPipe Face Landmarker v2 weights. Contains both detector variants
    (short / full), shared mesh, blendshapes, and canonical geometry."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadMediaPipeFaceLandmarker",
            search_aliases=["face", "facial", "mediapipe", "face landmark", "face mesh", "blazeface", "face detection"],
            display_name="Load Face Detection Model (MediaPipe)",
            category="model/loaders",
            inputs=[
                io.Combo.Input("model_name", options=folder_paths.get_filename_list("detection"),
                               tooltip="Face detection model from models/detection/."),
            ],
            outputs=[FaceDetectionType.Output()],
        )

    @classmethod
    def execute(cls, model_name) -> io.NodeOutput:
        sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("detection", model_name), safe_load=True)
        wrapper = FaceLandmarkerModel(sd)
        return io.NodeOutput(wrapper)


# Per-frame fallback modes for detection failures in a batch.
_FALLBACK_MODES = ("empty", "previous", "interpolate")


class MediaPipeFaceLandmarker(io.ComfyNode):
    """BlazeFace → FaceMesh v2 → ARKit-52 blendshapes, batched across the
    input. Also emits a BOUNDING_BOX list (landmark-extent bbox per face) —
    pair with DrawBBoxes for detector-only viz or MediaPipeFaceMeshVisualize
    for the mesh overlay."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MediaPipeFaceLandmarker",
            search_aliases=["face", "facial", "mediapipe", "face landmark", "face mesh", "blazeface", "face detection"],
            display_name="Detect Face Landmarks (MediaPipe)",
            category="image/detection",
            description="Detects facial landmarks using MediaPipe model.",
            inputs=[
                FaceDetectionType.Input("face_detection_model"),
                io.Image.Input("image"),
                io.Combo.Input("detector_variant", options=["short", "full", "both"], default="short",
                               tooltip="Face detector range. 'short' is tuned for close-up faces "
                                       "(within ~2 m of the camera); 'full' covers farther / smaller "
                                       "faces (up to ~5 m) but is slower. 'both' runs both detectors and "
                                       "keeps whichever found more faces per frame (~2× detection cost)."),
                io.Int.Input("num_faces", default=1, min=0, max=16, step=1,
                             tooltip="Maximum faces to return per frame. 0 = no cap (return all detected)."),
                io.Float.Input("min_confidence", default=0.5, min=0.0, max=1.0, step=0.01, advanced=True,
                               tooltip="BlazeFace score threshold. Lower to catch small/occluded faces."),
                io.Combo.Input("missing_frame_fallback", options=list(_FALLBACK_MODES), default="empty", advanced=True,
                               tooltip="Per-frame behaviour when detection fails in a batch. "
                                       "'empty' leaves the frame faceless. 'previous' copies the most recent successful "
                                       "detection. 'interpolate' lerps landmarks/bbox/blendshapes between bracketing "
                                       "successful frames. Multi-face: pairs faces across frames by greedy bbox-centre NN."),
            ],
            outputs=[
                FaceLandmarksType.Output(display_name="face_landmarks"),
                io.BoundingBox.Output("bboxes"),
            ],
        )

    @classmethod
    def execute(cls, face_detection_model, image, detector_variant, num_faces, min_confidence,
                missing_frame_fallback) -> io.NodeOutput:
        canonical = face_detection_model.canonical_data
        img_np = _image_to_uint8(image)
        B, H, W = img_np.shape[:3]
        chunk = 16
        is_both = detector_variant == "both"
        total_work = 2 * B if is_both else B
        pbar = comfy.utils.ProgressBar(total_work)

        def _run(variant: str) -> list[list[dict]]:
            res: list[list[dict]] = []
            with tqdm(total=B, desc=f"MediaPipe Face Landmarker ({variant})") as tq:
                for i in range(0, B, chunk):
                    end = min(i + chunk, B)
                    res.extend(face_detection_model.detect_batch(
                        [img_np[bi] for bi in range(i, end)],
                        num_faces=int(num_faces),
                        score_thresh=float(min_confidence),
                        variant=variant,
                    ))
                    pbar.update_absolute(min(pbar.current + (end - i), total_work))
                    tq.update(end - i)
            return res

        if is_both:
            short_res = _run("short")
            full_res = _run("full")
            # Per-frame keep whichever found more faces (tie → short).
            frames: list[list[dict]] = [
                short_res[bi] if len(short_res[bi]) >= len(full_res[bi]) else full_res[bi]
                for bi in range(B)
            ]
        else:
            frames = _run(detector_variant)
        _fill_missing_frames(frames, missing_frame_fallback)
        bboxes = []
        for per_frame in frames:
            per_bb = []
            for f in per_frame:
                f["transformation_matrix"] = transformation_matrix_from_detection(f, W, H, canonical)
                x1, y1, x2, y2 = (float(v) for v in f["bbox_xyxy"])
                per_bb.append({"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "label": "face", "score": float(f["score"])})
            bboxes.append(per_bb)
        return io.NodeOutput({"frames": frames, "image_size": (H, W),
                              "connection_sets": face_detection_model.connection_sets}, bboxes)


# Topology keys unioned by the 'all' connections preset (contour parts + irises + nose).
_ALL_CONNECTION_PARTS: tuple[str, ...] = (*_CONTOUR_PARTS, "irises", "nose")
_CUSTOM_FEATURES: tuple[tuple[str, bool], ...] = (
    ("face_oval",     True),
    ("lips",          True),
    ("left_eye",      True),
    ("right_eye",     True),
    ("left_eyebrow",  True),
    ("right_eyebrow", True),
    ("irises",        True),
    ("nose",          True),
    ("tesselation",   False),
)


class MediaPipeFaceMeshVisualize(io.ComfyNode):
    """Draw a FACEMESH_* subset over an image. Topology travels with the
    FACE_LANDMARKS payload (set at detection time)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MediaPipeFaceMeshVisualize",
            search_aliases=["face", "facial", "mediapipe", "face landmark", "face mesh", "blazeface", "face detection", "visualize"],
            display_name="Visualize Face Landmarks (MediaPipe)",
            category="image/detection",
            description="Draws face landmarks mesh on the input image.",
            inputs=[
                FaceLandmarksType.Input("face_landmarks"),
                io.Image.Input("image", optional=True, tooltip="If not connected, a black canvas will be used."),
                io.DynamicCombo.Input(
                    "connections",
                    tooltip="'all' = oval+eyes+brows+lips+irises+nose. 'fill' = solid face_oval polygon (silhouette mask). 'custom' = toggle each feature individually (including 'tesselation', the full 2547-edge wireframe).",
                    options=[
                        io.DynamicCombo.Option("all", []),
                        io.DynamicCombo.Option("fill", []),
                        io.DynamicCombo.Option("custom", [
                            io.Boolean.Input(feat, default=default,
                                             tooltip=f"Draw the '{feat}' connection set.")
                            for feat, default in _CUSTOM_FEATURES
                        ]),
                    ],
                ),
                io.Color.Input("color", default="#00ff00"),
                io.Int.Input("thickness", default=1, min=0, max=8, step=1,
                             tooltip="Edge line thickness in pixels. 0 disables edge drawing."),
                io.Int.Input("point_size", default=2, min=0, max=16, step=1,
                             tooltip="Landmark dot radius in pixels. 0 disables point drawing."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, face_landmarks, connections, color, thickness, point_size, image=None) -> io.NodeOutput:
        sets = face_landmarks["connection_sets"]
        sel = connections["connections"]
        fill_rings: list[list[int]] | None = None
        if sel == "fill":
            fill_rings = _ordered_rings(sets["face_oval"])
            edges = frozenset()
        elif sel == "custom":
            parts = [feat for feat, _ in _CUSTOM_FEATURES if connections.get(feat, False)]
            edges = frozenset().union(*(sets[p] for p in parts))
        else:  # "all"
            edges = frozenset().union(*(sets[p] for p in _ALL_CONNECTION_PARTS))
        rgb, thick, psize = _parse_color(color), int(thickness), int(point_size)
        frames = face_landmarks["frames"]
        if image is None:
            H, W = face_landmarks["image_size"]
            img_np = np.zeros((len(frames), H, W, 3), dtype=np.uint8)
        else:
            img_np = _image_to_uint8(image)
        B = img_np.shape[0]
        n_frames = len(frames)
        pbar = comfy.utils.ProgressBar(B)
        out = np.empty_like(img_np)
        for bi in range(B):
            faces = frames[bi] if bi < n_frames else []
            out[bi] = _draw_mesh(img_np[bi], faces, edges, rgb, thick, psize, fill_rings)
            pbar.update_absolute(bi + 1)
        return io.NodeOutput(torch.from_numpy(out).to(
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        ).div_(255.0))


def _draw_mesh(image_rgb: np.ndarray, faces: list, edges,
               rgb: tuple[int, int, int], thickness: int,
               point_size: int, fill_rings: list[list[int]] | None = None) -> np.ndarray:
    draw_edges = thickness > 0 and edges
    if not faces or (fill_rings is None and not draw_edges and point_size <= 0):
        return image_rgb.copy()
    pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil)
    r = point_size * 0.5
    if fill_rings is not None:
        for f in faces:
            lmks = f["landmarks_xy"]
            for ring in fill_rings:
                draw.polygon([(float(lmks[i, 0]), float(lmks[i, 1])) for i in ring], fill=rgb)
        return np.asarray(pil)
    for f in faces:
        lmks = f["landmarks_xy"]
        n = lmks.shape[0]
        if draw_edges:
            for a, b in edges:
                if a < n and b < n:
                    draw.line([(float(lmks[a, 0]), float(lmks[a, 1])),
                               (float(lmks[b, 0]), float(lmks[b, 1]))], fill=rgb, width=thickness)
        if point_size == 1:
            draw.point(lmks.flatten().tolist(), fill=rgb)
        elif point_size > 1:
            for x, y in lmks:
                draw.ellipse((float(x) - r, float(y) - r, float(x) + r, float(y) + r), fill=rgb)
    return np.asarray(pil)


# Mask region presets — closed-loop topologies only.
_MASK_REGIONS: tuple[str, ...] = ("face_oval", "lips", "left_eye", "right_eye", "irises")
_MASK_CUSTOM_FEATURES: tuple[tuple[str, bool], ...] = (
    ("face_oval",  True),
    ("lips",       False),
    ("left_eye",   False),
    ("right_eye",  False),
    ("irises",     False),
)


class MediaPipeFaceMask(io.ComfyNode):
    """Binary mask from face landmarks, filled polygon per face. One mask per
    frame in the batch; faces in the same frame composite (union)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MediaPipeFaceMask",
            search_aliases=["face", "facial", "mediapipe", "face mask", "blazeface", "face detection", "visualize"],
            display_name="Draw Face Mask (MediaPipe)",
            category="image/detection",
            description="Draws a mask from face landmarks.",
            inputs=[
                FaceLandmarksType.Input("face_landmarks"),
                io.DynamicCombo.Input(
                    "regions",
                    tooltip="'all' = union of face_oval+lips+eyes+irises (which collapses to face_oval since it encloses the rest). 'custom' = toggle each region individually for combos like lips+eyes.",
                    options=[
                        io.DynamicCombo.Option("all", []),
                        io.DynamicCombo.Option("custom", [
                            io.Boolean.Input(reg, default=default,
                                             tooltip=f"Include the '{reg}' region in the mask.")
                            for reg, default in _MASK_CUSTOM_FEATURES
                        ]),
                    ],
                ),
            ],
            outputs=[io.Mask.Output()],
        )

    @classmethod
    def execute(cls, face_landmarks, regions) -> io.NodeOutput:
        sets = face_landmarks["connection_sets"]
        sel = regions["regions"]
        if sel == "custom":
            picked = [reg for reg, _ in _MASK_CUSTOM_FEATURES if regions.get(reg, False)]
        else:
            picked = list(_MASK_REGIONS)
        rings = [r for reg in picked for r in _ordered_rings(sets[reg])]
        frames = face_landmarks["frames"]
        H, W = face_landmarks["image_size"]
        masks = np.zeros((len(frames), H, W), dtype=np.uint8)
        pbar = comfy.utils.ProgressBar(len(frames))
        for bi, per_frame in enumerate(frames):
            if per_frame:
                pil = Image.new("L", (W, H), 0)
                draw = ImageDraw.Draw(pil)
                for f in per_frame:
                    lmks = f["landmarks_xy"]
                    for ring in rings:
                        draw.polygon([(float(lmks[i, 0]), float(lmks[i, 1])) for i in ring], fill=255)
                masks[bi] = np.asarray(pil)
            pbar.update_absolute(bi + 1)
        return io.NodeOutput(torch.from_numpy(masks).to(
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        ).div_(255.0))


class MediaPipeFaceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LoadMediaPipeFaceLandmarker, MediaPipeFaceLandmarker, MediaPipeFaceMeshVisualize, MediaPipeFaceMask]


async def comfy_entrypoint() -> MediaPipeFaceExtension:
    return MediaPipeFaceExtension()
