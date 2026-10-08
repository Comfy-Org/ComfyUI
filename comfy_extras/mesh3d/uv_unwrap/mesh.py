"""Mesh container, edge/face adjacency, manifold cleanup."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch import Tensor


# ---- Per-face / per-vertex geometry ----

def face_normals(vertices: Tensor, faces: Tensor) -> Tensor:
    """[F,3] unit face normals (degenerate faces -> zero)."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    n = torch.linalg.cross(v1 - v0, v2 - v0)
    return n / n.norm(dim=1, keepdim=True).clamp_min(1e-20)


def face_areas(vertices: Tensor, faces: Tensor) -> Tensor:
    """[F] triangle areas."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return 0.5 * torch.linalg.cross(v1 - v0, v2 - v0).norm(dim=1)


def face_centroids(vertices: Tensor, faces: Tensor) -> Tensor:
    """[F,3] triangle centroids."""
    return vertices[faces].mean(dim=1)


def face_edge_lengths(vertices: Tensor, faces: Tensor) -> Tensor:
    """[F,3] edge lengths; column e = |v[faces[:,e]] - v[faces[:,(e+1)%3]]|."""
    va = vertices[faces]
    vb = vertices[faces.roll(shifts=-1, dims=1)]
    return (vb - va).norm(dim=-1).to(torch.float32)


def chart_3d_areas(face_area: Tensor, face_chart: Tensor, n_charts: int) -> Tensor:
    """[n_charts] sum of face areas per chart."""
    out = torch.zeros(n_charts, dtype=face_area.dtype, device=face_area.device)
    out.scatter_add_(0, face_chart, face_area)
    return out


@dataclass
class MeshData:
    """Cleaned mesh with adjacency; face_face[f, i] = face sharing edge (faces[f,i], faces[f,(i+1)%3]) or -1 if boundary."""

    vertices: Tensor                 # [V, 3] float
    faces: Tensor                    # [F, 3] long
    face_face: Tensor                # [F, 3] long, neighbor face id or -1
    face_normal: Tensor              # [F, 3] float
    face_area: Tensor                # [F] float
    face_centroid: Tensor            # [F, 3] float
    component: Tensor                # [F] long, connected-component id
    n_components: int


def build_mesh(vertices: Tensor, faces: Tensor) -> MeshData:
    """Build adjacency; non-manifold edges (>2 incident faces) get no neighbor and act as boundary."""
    if vertices.dtype != torch.float32:
        vertices = vertices.to(torch.float32)
    if faces.dtype != torch.long:
        faces = faces.to(torch.long)

    device = faces.device
    V = vertices.shape[0]
    F = faces.shape[0]

    # Per directed face-edge; flat layout p = f*3+i.
    a = faces.flatten()
    b = faces.roll(shifts=-1, dims=1).flatten()
    lo = torch.minimum(a, b)
    hi = torch.maximum(a, b)
    edge_key = lo * (V + 1) + hi

    # Pair manifold (count==2) face-edges; others get no neighbor.
    _, inverse, counts = torch.unique(edge_key, return_inverse=True, return_counts=True)
    edge_count = counts[inverse]
    manifold_mask = edge_count == 2

    sort_idx = torch.argsort(edge_key, stable=True)
    sorted_manifold = manifold_mask[sort_idx]
    pair_positions = sort_idx[sorted_manifold]
    pair_a = pair_positions[0::2]
    pair_b = pair_positions[1::2]

    face_id_flat = torch.arange(F, device=device).repeat_interleave(3)
    face_face_flat = torch.full((3 * F,), -1, dtype=torch.long, device=device)
    face_face_flat[pair_a] = face_id_flat[pair_b]
    face_face_flat[pair_b] = face_id_flat[pair_a]
    face_face = face_face_flat.view(F, 3)

    face_face_np = face_face.cpu().numpy()
    rows_mask = face_face_np >= 0
    if rows_mask.any():
        rows = np.broadcast_to(np.arange(F)[:, None], (F, 3))[rows_mask]
        cols = face_face_np[rows_mask]
        adj = csr_matrix(
            (np.ones(rows.size, dtype=np.int8), (rows, cols)),
            shape=(F, F),
        )
    else:
        adj = csr_matrix((F, F), dtype=np.int8)
    n_components, labels = connected_components(adj, directed=False)

    face_normal = face_normals(vertices, faces)
    face_area = face_areas(vertices, faces)
    face_centroid = face_centroids(vertices, faces)

    return MeshData(
        vertices=vertices,
        faces=faces,
        face_face=face_face,
        face_normal=face_normal,
        face_area=face_area,
        face_centroid=face_centroid,
        component=torch.from_numpy(labels.astype(np.int64)).to(device),
        n_components=int(n_components),
    )


def chart_boundary_loops(
    faces_subset: Tensor, face_face_subset: Tensor
) -> List[List[int]]:
    """Return ordered boundary vertex loops for a chart submesh (face_face_subset[f,i]==-1 marks a boundary edge)."""
    F = faces_subset.shape[0]
    faces_np = faces_subset.cpu().numpy()
    ff = face_face_subset.cpu().numpy()

    next_v: Dict[int, int] = {}
    for f in range(F):
        for i in range(3):
            if ff[f, i] == -1:
                a = int(faces_np[f, i])
                b = int(faces_np[f, (i + 1) % 3])
                next_v[a] = b

    loops: List[List[int]] = []
    visited = set()
    for start in list(next_v.keys()):
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        cur = next_v.get(start)
        while cur is not None and cur != start:
            if cur in visited:
                break
            loop.append(cur)
            visited.add(cur)
            cur = next_v.get(cur)
        if len(loop) >= 3:
            loops.append(loop)
    return loops
