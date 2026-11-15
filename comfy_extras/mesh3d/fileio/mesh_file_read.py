import logging
import re
import struct

import numpy as np


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def load_obj(data: bytes) -> dict:
    text = data.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\\\n", "")
    positions: list[tuple] = []
    pos_colors: list[tuple] = []
    uvs: list[tuple] = []
    normals: list[tuple] = []

    corner_map: dict[tuple, int] = {}
    out_pos: list[tuple] = []
    out_col: list[tuple] = []
    out_uv: list[tuple] = []
    out_nrm: list[tuple] = []
    faces: list[tuple] = []
    has_color = False
    any_uv = False
    any_normal = False
    missing_normal = False
    warned_mtl = False

    def resolve(index_str: str, count: int) -> int:
        i = int(index_str)
        resolved = i - 1 if i > 0 else count + i
        if i == 0 or not 0 <= resolved < count:
            raise ValueError(f"OBJ index {i} out of range for {count} entries")
        return resolved

    def corner(spec: str) -> int:
        nonlocal any_uv, any_normal, missing_normal
        parts = spec.split("/")
        vi = resolve(parts[0], len(positions))
        ti = resolve(parts[1], len(uvs)) if len(parts) > 1 and parts[1] else None
        ni = resolve(parts[2], len(normals)) if len(parts) > 2 and parts[2] else None
        key = (vi, ti, ni)
        cached = corner_map.get(key)
        if cached is not None:
            return cached
        index = len(out_pos)
        out_pos.append(positions[vi])
        out_col.append(pos_colors[vi])
        if ti is not None:
            any_uv = True
            u, v = uvs[ti]
            out_uv.append((u, 1.0 - v))
        else:
            out_uv.append((0.0, 0.0))
        if ni is not None:
            any_normal = True
            out_nrm.append(normals[ni])
        else:
            missing_normal = True
            out_nrm.append((0.0, 0.0, 0.0))
        corner_map[key] = index
        return index

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tag = parts[0]
        if tag == "v":
            positions.append(tuple(float(x) for x in parts[1:4]))
            if len(parts) >= 7:
                pos_colors.append(tuple(float(x) for x in parts[4:7]))
                has_color = True
            else:
                pos_colors.append((1.0, 1.0, 1.0))
        elif tag == "vt":
            uvs.append((float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0))
        elif tag == "vn":
            normals.append(tuple(float(x) for x in parts[1:4]))
        elif tag == "f":
            specs = parts[1:]
            if len(specs) < 3:
                continue
            indices = [corner(s) for s in specs]
            for i in range(1, len(indices) - 1):
                faces.append((indices[0], indices[i], indices[i + 1]))
        elif tag in ("mtllib", "usemtl") and not warned_mtl:
            warned_mtl = True
            logging.warning("Get3DComponents: OBJ materials (.mtl) are not loaded; geometry only")

    if not faces:
        raise ValueError("OBJ contains no faces")

    prim = {
        "positions": np.array(out_pos, np.float32),
        "faces": np.array(faces, np.int64),
        "uvs": np.array(out_uv, np.float32) if any_uv else None,
        "colors": _srgb_to_linear(np.clip(np.array(out_col, np.float32), 0.0, 1.0)) if has_color else None,
        "normals": None,
        "tangents": None,
        "material": None,
    }
    if any_normal and missing_normal:
        logging.warning("Get3DComponents: OBJ has faces without vn indices; normals dropped")
    elif any_normal:
        nrm = np.array(out_nrm, np.float32)
        lengths = np.linalg.norm(nrm, axis=1, keepdims=True)
        if float(lengths.max()) > 1e-6:
            prim["normals"] = nrm / np.maximum(lengths, 1e-12)
    return prim


_STL_RECORD = np.dtype([("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)), ("attr", "<u2")])

_STL_ASCII_FACET = re.compile(
    rb"facet\s+normal\s+(\S+)\s+(\S+)\s+(\S+).*?"
    rb"vertex\s+(\S+)\s+(\S+)\s+(\S+).*?"
    rb"vertex\s+(\S+)\s+(\S+)\s+(\S+).*?"
    rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)",
    re.DOTALL,
)


def load_stl(data: bytes) -> dict:
    n_faces = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else 0
    if n_faces > 0 and 84 + n_faces * _STL_RECORD.itemsize == len(data):
        return _load_stl_binary(data, n_faces)
    if b"solid" in data[:9]:
        facets = _STL_ASCII_FACET.findall(data)
        if not facets:
            raise ValueError("ASCII STL contains no facets")
        values = np.array(facets).astype(np.float32)
        positions = values[:, 3:12].reshape(-1, 3)
        file_normals = np.repeat(values[:, 0:3], 3, axis=0)
        return _stl_prim(positions, file_normals)
    available = (len(data) - 84) // _STL_RECORD.itemsize if len(data) >= 84 else 0
    n_faces = min(n_faces, available)
    if n_faces <= 0:
        raise ValueError("not a valid STL file (neither binary layout nor ASCII 'solid')")
    return _load_stl_binary(data, n_faces)


def _load_stl_binary(data: bytes, n_faces: int) -> dict:
    records = np.frombuffer(data, _STL_RECORD, count=n_faces, offset=84)
    positions = records["verts"].reshape(-1, 3).astype(np.float32)
    file_normals = np.repeat(records["normal"], 3, axis=0).astype(np.float32)
    return _stl_prim(positions, file_normals, _stl_binary_colors(data[:80], records["attr"]))


def _stl_binary_colors(header: bytes, attr: np.ndarray):
    pos = header.find(b"COLOR=")
    if pos < 0 or pos + 10 > len(header):
        return None
    default = np.frombuffer(header, np.uint8, 3, pos + 6).astype(np.float32) / 255.0
    a = attr.astype(np.uint16)
    per_face = np.stack([a & 0x1F, (a >> 5) & 0x1F, (a >> 10) & 0x1F], axis=1).astype(np.float32) / 31.0
    face_colors = np.where(((a & 0x8000) != 0)[:, None], default[None, :], per_face)
    return _srgb_to_linear(np.repeat(face_colors, 3, axis=0))


def _stl_prim(positions: np.ndarray, normals: np.ndarray, colors=None) -> dict:
    if positions.shape[0] == 0:
        raise ValueError("STL contains no facets")
    faces = np.arange(positions.shape[0], dtype=np.int64).reshape(-1, 3)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-12) if float(lengths.max()) > 1e-6 else None
    return {"positions": np.ascontiguousarray(positions), "faces": faces, "uvs": None,
            "colors": colors, "normals": normals, "tangents": None, "material": None}
