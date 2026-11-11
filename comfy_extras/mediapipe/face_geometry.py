"""Pure-numpy port of MediaPipe's face_geometry (FACE_LANDMARK_PIPELINE mode)
+ weighted Procrustes solver. Computes the 4x4 facial transformation matrix.
"""


import math
import numpy as np


def _solve_weighted_orthogonal_problem(src: np.ndarray, tgt: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted orthogonal Procrustes (similarity). Returns 4x4 M with
    `target ≈ M @ homogeneous(source)` in the weighted LS sense. fp64 for
    SVD stability. Port of procrustes_solver.cc."""
    sqrt_w = np.sqrt(weights.astype(np.float64))
    w_total = float((sqrt_w ** 2).sum())
    ws = src.astype(np.float64) * sqrt_w
    wt = tgt.astype(np.float64) * sqrt_w

    c_w = (ws @ sqrt_w) / w_total
    centered = ws - np.outer(c_w, sqrt_w)
    U, _S, Vt = np.linalg.svd(wt @ centered.T, full_matrices=True)
    # Disallow reflection: flip the least-significant axis when det(U)·det(V)<0.
    post, pre = U.copy(), Vt.T.copy()
    if np.linalg.det(post) * np.linalg.det(pre) < 0:
        post[:, 2] *= -1.0
    R = post @ pre.T

    denom = float((centered * ws).sum())
    if denom < 1e-12:
        raise ValueError("Procrustes denominator collapsed (degenerate source).")
    scale = float((R @ centered * wt).sum()) / denom
    translation = ((wt - scale * (R @ ws)) @ sqrt_w) / w_total

    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = scale * R
    M[:3, 3] = translation
    return M


def _estimate_scale(canonical: np.ndarray, runtime: np.ndarray, weights: np.ndarray) -> float:
    """scale = ‖first column of M[:3]‖ per geometry_pipeline.cc::EstimateScale."""
    return float(np.linalg.norm(_solve_weighted_orthogonal_problem(canonical, runtime, weights)[:3, 0]))


def solve_facial_transformation_matrix(
    landmarks_normalized: np.ndarray,
    canonical_vertices: np.ndarray,
    procrustes_indices: np.ndarray,
    procrustes_weights: np.ndarray,
    image_width: int,
    image_height: int,
    # face_geometry_calculator_options.pbtxt defaults
    vertical_fov_degrees: float = 63.0,
    near: float = 1.0,
) -> np.ndarray:
    """4x4 facial transformation matrix via two-pass scale recovery
    `landmarks_normalized` is (N, 3) in MediaPipe normalized convention: x, y
    in [0,1] with TOP-LEFT origin, z in width-scaled units.
    """

    h_near = 2.0 * near * math.tan(0.5 * math.radians(vertical_fov_degrees))
    w_near = image_width * h_near / image_height

    sub = procrustes_indices.astype(np.int64)
    screen = landmarks_normalized[sub].T.astype(np.float64).copy()
    canon = canonical_vertices[sub].T.astype(np.float64).copy()
    weights = procrustes_weights.astype(np.float64)

    # ProjectXY (TOP_LEFT y-flip, then scale all 3 axes; z uses x-scale).
    screen[1] = 1.0 - screen[1]
    screen[0] = screen[0] * w_near - 0.5 * w_near
    screen[1] = screen[1] * h_near - 0.5 * h_near
    screen[2] = screen[2] * w_near
    depth_offset = float(screen[2].mean())

    def _unproject(s: np.ndarray, scale: float) -> np.ndarray:
        s = s.copy()
        s[2] = (s[2] - depth_offset + near) / scale
        s[0] *= s[2] / near
        s[1] *= s[2] / near
        s[2] *= -1.0
        return s

    first = screen.copy()
    first[2] *= -1.0
    s1 = _estimate_scale(canon, first, weights) # 1st pass: Procrustes on projected XY
    s2 = _estimate_scale(canon, _unproject(screen, s1), weights) # 2nd pass: rescale z by s1, un-project XY
    return _solve_weighted_orthogonal_problem(canon, _unproject(screen, s1 * s2), weights).astype(np.float32)


def transformation_matrix_from_detection(face_dict: dict, image_width: int, image_height: int, canonical_data: dict) -> np.ndarray:
    """Adapt a FaceLandmarker face dict to MP's normalized convention and solve.
    FaceMesh emits (x, y, z) in 192-canonical units; MP's geometry expects
    z_norm = z_canonical * scale_x / image_width"""

    lmks_xy, lmks_3d = face_dict["landmarks_xy"], face_dict["landmarks_3d"]
    aug = np.concatenate([lmks_3d[:, :2].astype(np.float64), np.ones((lmks_xy.shape[0], 1))], axis=1)
    M, *_ = np.linalg.lstsq(aug, lmks_xy.astype(np.float64), rcond=None)
    scale_x = float(np.linalg.norm(M[0]))
    z_scale = scale_x / image_width if scale_x > 1e-6 else 1.0 / image_width

    normalized = np.empty((lmks_xy.shape[0], 3), dtype=np.float32)
    normalized[:, 0] = lmks_xy[:, 0] / image_width
    normalized[:, 1] = lmks_xy[:, 1] / image_height
    normalized[:, 2] = lmks_3d[:, 2] * z_scale
    return solve_facial_transformation_matrix(
        normalized, canonical_data["canonical_vertices"],
        canonical_data["procrustes_indices"], canonical_data["procrustes_weights"],
        image_width=image_width, image_height=image_height,
    )
