import shutil
from io import BytesIO
from pathlib import Path
from typing import IO

import torch


class VOXEL:
    def __init__(self, data: torch.Tensor):
        self.data = data


class SPLAT:
    """A batch of 3D Gaussian splats in render-ready (activated, world-space) form.

    Tensors are (B, N, ...) and zero-padded to a common N across the batch; `counts` (B,) holds the
    real per-item lengths (None when rows are uniform and no slicing is needed). SH coefficients are
    stored as (B, N, K, 3) with K = (sh_degree + 1)**2; the DC (diffuse) term is sh[..., 0, :].
    """

    def __init__(self, positions: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor,
                 opacities: torch.Tensor, sh: torch.Tensor, counts: torch.Tensor | None = None):
        self.positions = positions    # (B, N, 3) world-space centers
        self.scales = scales          # (B, N, 3) linear (positive) per-axis std
        self.rotations = rotations    # (B, N, 4) quaternion wxyz (normalized)
        self.opacities = opacities    # (B, N, 1) in [0, 1]
        self.sh = sh                  # (B, N, K, 3) spherical-harmonic color coefficients
        self.counts = counts          # (B,) real lengths, or None


class MESH:
    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor,
                 uvs: torch.Tensor | None = None,
                 vertex_colors: torch.Tensor | None = None,
                 texture: torch.Tensor | None = None,
                 vertex_counts: torch.Tensor | None = None,
                 face_counts: torch.Tensor | None = None,
                 unlit: bool = False):

        assert (vertex_counts is None) == (face_counts is None), \
            "vertex_counts and face_counts must be provided together (both or neither)"
        self.vertices = vertices            # vertices: (B, N, 3)
        self.faces = faces                  # faces: (B, M, 3)
        self.uvs = uvs                      # uvs: (B, N, 2)
        self.vertex_colors = vertex_colors  # vertex_colors: (B, N, 3 or 4)
        self.texture = texture              # texture: (B, H, W, 3)
        # When vertices/faces are zero-padded to a common N/M across the batch (variable-size mesh batch),
        # these hold the real per-item lengths (B,). None means rows are uniform and no slicing is needed.
        self.vertex_counts = vertex_counts
        self.face_counts = face_counts
        # Render flat / emissive (no scene lighting) when saved, e.g. for gaussian-splat-derived meshes.
        self.unlit = unlit


class File3D:
    """Class representing a 3D file from a file path or binary stream.

    Supports both disk-backed (file path) and memory-backed (BytesIO) storage.
    """

    def __init__(self, source: str | IO[bytes], file_format: str = ""):
        self._source = source
        self._format = file_format or self._infer_format()

    def _infer_format(self) -> str:
        if isinstance(self._source, str):
            return Path(self._source).suffix.lstrip(".").lower()
        return ""

    @property
    def format(self) -> str:
        return self._format

    @format.setter
    def format(self, value: str) -> None:
        self._format = value.lstrip(".").lower() if value else ""

    @property
    def is_disk_backed(self) -> bool:
        return isinstance(self._source, str)

    def get_source(self) -> str | IO[bytes]:
        if isinstance(self._source, str):
            return self._source
        if hasattr(self._source, "seek"):
            self._source.seek(0)
        return self._source

    def get_data(self) -> BytesIO:
        if isinstance(self._source, str):
            with open(self._source, "rb") as f:
                result = BytesIO(f.read())
            return result
        if hasattr(self._source, "seek"):
            self._source.seek(0)
        if isinstance(self._source, BytesIO):
            return self._source
        return BytesIO(self._source.read())

    def save_to(self, path: str) -> str:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(self._source, str):
            if Path(self._source).resolve() != dest.resolve():
                shutil.copy2(self._source, dest)
        else:
            if hasattr(self._source, "seek"):
                self._source.seek(0)
            with open(dest, "wb") as f:
                f.write(self._source.read())
        return str(dest)

    def get_bytes(self) -> bytes:
        if isinstance(self._source, str):
            return Path(self._source).read_bytes()
        if hasattr(self._source, "seek"):
            self._source.seek(0)
        return self._source.read()

    def __repr__(self) -> str:
        if isinstance(self._source, str):
            return f"File3D(source={self._source!r}, format={self._format!r})"
        return f"File3D(<stream>, format={self._format!r})"
