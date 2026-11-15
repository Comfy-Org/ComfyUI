import base64
import json
import os
import struct
import urllib.parse
from io import BytesIO

import numpy as np
from PIL import Image

_COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}

_SUPPORTED_REQUIRED = {
    "EXT_mesh_gpu_instancing",
    "EXT_texture_webp",
    "KHR_materials_emissive_strength",
    "KHR_materials_unlit",
}

_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942


def parse_container(data: bytes):
    if data[:4] == b"glTF":
        if len(data) < 12:
            raise ValueError("GLB file truncated (missing 12-byte header)")
        _, version, _ = struct.unpack_from("<4sII", data, 0)
        if version != 2:
            raise ValueError(f"unsupported GLB container version {version}")
        json_chunk = None
        bin_chunk = None
        offset = 12
        while offset + 8 <= len(data):
            chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
            offset += 8
            chunk = data[offset:offset + chunk_len]
            offset += chunk_len
            if chunk_type == _JSON_CHUNK and json_chunk is None:
                json_chunk = chunk
            elif chunk_type == _BIN_CHUNK and bin_chunk is None:
                bin_chunk = chunk
        if json_chunk is None:
            raise ValueError("GLB file has no JSON chunk")
        return json.loads(json_chunk), bin_chunk
    return json.loads(data), None


def _resolve_uri(uri: str, base_dir: str | None) -> bytes:
    if uri.startswith("data:"):
        header, _, payload = uri.partition(",")
        if ";base64" in header:
            return base64.b64decode(payload)
        return urllib.parse.unquote_to_bytes(payload)
    scheme = urllib.parse.urlparse(uri).scheme
    if scheme:
        raise ValueError(f"glTF references URI scheme {scheme!r}; only relative file paths are allowed")
    if base_dir is None:
        raise ValueError(
            f"glTF references external file {uri!r} but the source is an in-memory stream; "
            "use .glb (self-contained) or a disk-backed file"
        )
    relative = urllib.parse.unquote(uri)
    root = os.path.realpath(base_dir)
    path = os.path.realpath(os.path.join(root, relative))
    try:
        contained = not os.path.isabs(relative) and os.path.commonpath([root, path]) == root
    except ValueError:
        contained = False
    if not contained:
        raise ValueError(f"glTF references file {uri!r} outside the model directory")
    with open(path, "rb") as f:
        return f.read()


def load_buffers(gltf: dict, bin_chunk: bytes | None, base_dir: str | None) -> list[bytes]:
    buffers = []
    for buf in gltf.get("buffers", []):
        if "uri" in buf:
            buffers.append(_resolve_uri(buf["uri"], base_dir))
        else:
            if bin_chunk is None:
                raise ValueError("glTF buffer has no uri and there is no GLB BIN chunk")
            buffers.append(bin_chunk)
    return buffers


def _view_data(gltf: dict, buffers: list[bytes], view_index: int) -> bytes:
    view = gltf["bufferViews"][view_index]
    buf = buffers[view.get("buffer", 0)]
    start = view.get("byteOffset", 0)
    return buf[start:start + view["byteLength"]]


def read_accessor(gltf: dict, buffers: list[bytes], index: int):
    acc = gltf["accessors"][index]
    count = acc["count"]
    ncomp = _TYPE_SIZES[acc["type"]]
    dtype = np.dtype(_COMPONENT_DTYPES[acc["componentType"]])
    elem = dtype.itemsize * ncomp

    if "bufferView" in acc:
        view = gltf["bufferViews"][acc["bufferView"]]
        buf = buffers[view.get("buffer", 0)]
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = view.get("byteStride") or elem
        if stride == elem:
            arr = np.frombuffer(buf, dtype, count * ncomp, start).reshape(count, ncomp).copy()
        else:
            raw = np.frombuffer(buf, np.uint8, stride * (count - 1) + elem, start)
            rows = np.lib.stride_tricks.as_strided(raw, (count, elem), (stride, 1))
            arr = np.ascontiguousarray(rows).view(dtype).reshape(count, ncomp)
    else:
        arr = np.zeros((count, ncomp), dtype)

    sparse = acc.get("sparse")
    if sparse:
        n = sparse["count"]
        idx_def = sparse["indices"]
        val_def = sparse["values"]
        idx_dtype = np.dtype(_COMPONENT_DTYPES[idx_def["componentType"]])
        iview = gltf["bufferViews"][idx_def["bufferView"]]
        ibuf = buffers[iview.get("buffer", 0)]
        sidx = np.frombuffer(ibuf, idx_dtype, n,
                             iview.get("byteOffset", 0) + idx_def.get("byteOffset", 0)).astype(np.int64)
        vview = gltf["bufferViews"][val_def["bufferView"]]
        vbuf = buffers[vview.get("buffer", 0)]
        svals = np.frombuffer(vbuf, dtype, n * ncomp,
                              vview.get("byteOffset", 0) + val_def.get("byteOffset", 0)).reshape(n, ncomp)
        arr[sidx] = svals
    return arr, bool(acc.get("normalized"))


def to_float(arr: np.ndarray, normalized: bool) -> np.ndarray:
    if arr.dtype == np.float32:
        return arr
    out = arr.astype(np.float32)
    if not normalized:
        return out
    if arr.dtype == np.uint8:
        return out / 255.0
    if arr.dtype == np.uint16:
        return out / 65535.0
    if arr.dtype == np.int8:
        return np.maximum(out / 127.0, -1.0)
    if arr.dtype == np.int16:
        return np.maximum(out / 32767.0, -1.0)
    return out


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def _node_local_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], np.float32).reshape(4, 4).T  # glTF stores column-major
    m = np.eye(4, dtype=np.float32)
    rot = _quat_to_matrix(*node.get("rotation", (0.0, 0.0, 0.0, 1.0)))
    scale = np.array(node.get("scale", (1.0, 1.0, 1.0)), np.float32)
    m[:3, :3] = rot * scale
    m[:3, 3] = node.get("translation", (0.0, 0.0, 0.0))
    return m


def _instance_matrices(gltf: dict, buffers: list[bytes], node: dict, warn):
    attrs = node.get("extensions", {}).get("EXT_mesh_gpu_instancing", {}).get("attributes", {})
    if not attrs:
        return None
    counts = {gltf["accessors"][a]["count"] for a in attrs.values()}
    if len(counts) > 1:
        raise ValueError(f"EXT_mesh_gpu_instancing attribute accessors disagree on instance count: {sorted(counts)}")
    if not any(k in attrs for k in ("TRANSLATION", "ROTATION", "SCALE")):
        warn("instancing-custom", "EXT_mesh_gpu_instancing has only custom attributes; importing a single copy")
        return None
    translation = to_float(*read_accessor(gltf, buffers, attrs["TRANSLATION"])) if "TRANSLATION" in attrs else None
    rotation = to_float(*read_accessor(gltf, buffers, attrs["ROTATION"])) if "ROTATION" in attrs else None
    scale = to_float(*read_accessor(gltf, buffers, attrs["SCALE"])) if "SCALE" in attrs else None
    count = counts.pop()
    matrices = []
    for i in range(count):
        m = np.eye(4, dtype=np.float32)
        rot = _quat_to_matrix(*rotation[i]) if rotation is not None else np.eye(3, dtype=np.float32)
        m[:3, :3] = rot * (scale[i] if scale is not None else 1.0)
        if translation is not None:
            m[:3, 3] = translation[i]
        matrices.append(m)
    return matrices


def _iter_mesh_nodes(gltf: dict, buffers: list[bytes], warn):
    nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes")
    if scenes:
        roots = scenes[gltf.get("scene", 0)].get("nodes", [])
    elif nodes:
        children = {c for n in nodes for c in n.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]
    else:
        for i in range(len(gltf.get("meshes", []))):
            yield {"mesh": i}, np.eye(4, dtype=np.float32)
        return
    seen = set()
    stack = [(i, np.eye(4, dtype=np.float32)) for i in roots]
    while stack:
        index, parent = stack.pop()
        if index in seen:
            continue
        seen.add(index)
        node = nodes[index]
        world = parent @ _node_local_matrix(node)
        if "mesh" in node:
            instances = _instance_matrices(gltf, buffers, node, warn)
            if instances is None:
                yield node, world
            else:
                warn("instancing", f"EXT_mesh_gpu_instancing: expanding {len(instances)} instances into merged geometry")
                for matrix in instances:
                    yield node, world @ matrix
        for child in node.get("children", []):
            stack.append((child, world))


def _to_triangles(indices: np.ndarray, mode: int) -> np.ndarray:
    if mode == 4:
        if len(indices) % 3:
            raise ValueError("TRIANGLES primitive index count must be divisible by 3")
        return indices.reshape(-1, 3)
    if len(indices) < 3:
        return np.zeros((0, 3), np.int64)
    if mode == 6:
        first = np.full(len(indices) - 2, indices[0], dtype=np.int64)
        return np.stack([first, indices[1:-1], indices[2:]], axis=1)
    tris = np.stack([indices[:-2], indices[1:-1], indices[2:]], axis=1)
    tris[1::2] = tris[1::2, ::-1]
    return tris


def _vertex_attr(gltf, buffers, accessor_index, n_verts, warn, label):
    arr = to_float(*read_accessor(gltf, buffers, accessor_index))
    if arr.shape[0] < n_verts:
        warn(f"count:{label}", f"{label} has {arr.shape[0]} entries for {n_verts} vertices; attribute dropped")
        return None
    return arr[:n_verts]


def load_scene_geometry(gltf: dict, buffers: list[bytes], warn) -> list[dict]:
    required = set(gltf.get("extensionsRequired", []))
    unsupported = required - _SUPPORTED_REQUIRED
    if unsupported:
        raise ValueError(f"glTF requires extensions this loader does not support: {sorted(unsupported)}")

    prims = []
    for node, world in _iter_mesh_nodes(gltf, buffers, warn):
        if "skin" in node:
            warn("skin", "skinned mesh: joints/weights ignored, geometry imported in bind pose")
        linear = world[:3, :3]
        det = float(np.linalg.det(linear))
        normal_mat = np.linalg.inv(linear).T if abs(det) > 1e-12 else linear
        flip_winding = det < 0.0

        for prim in gltf["meshes"][node["mesh"]].get("primitives", []):
            mode = prim.get("mode", 4)
            if mode not in (4, 5, 6):
                warn(f"mode{mode}", f"skipping non-triangle primitive (mode {mode})")
                continue
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs:
                continue
            pos = to_float(*read_accessor(gltf, buffers, attrs["POSITION"]))[:, :3]
            n_verts = pos.shape[0]
            if n_verts == 0:
                continue
            pos = pos @ linear.T + world[:3, 3]

            if "indices" in prim:
                indices = read_accessor(gltf, buffers, prim["indices"])[0].reshape(-1).astype(np.int64)
            else:
                indices = np.arange(n_verts, dtype=np.int64)
            faces = _to_triangles(indices, mode)
            if faces.shape[0] == 0:
                continue
            if faces.min() < 0 or faces.max() >= n_verts:
                raise ValueError("primitive contains a face index outside its POSITION accessor")
            if flip_winding:
                faces = np.ascontiguousarray(faces[:, ::-1])
            if prim.get("targets"):
                warn("morph", "morph targets ignored; base geometry imported")

            out = {"positions": np.ascontiguousarray(pos, np.float32), "faces": faces,
                   "uvs": None, "colors": None, "normals": None, "tangents": None,
                   "material": prim.get("material")}
            if "TEXCOORD_0" in attrs:
                uv = _vertex_attr(gltf, buffers, attrs["TEXCOORD_0"], n_verts, warn, "TEXCOORD_0")
                out["uvs"] = uv[:, :2] if uv is not None else None
            if "COLOR_0" in attrs:
                arr, normalized = read_accessor(gltf, buffers, attrs["COLOR_0"])
                arr = to_float(arr, normalized or arr.dtype != np.float32)
                if arr.shape[0] >= n_verts:
                    out["colors"] = np.clip(arr[:n_verts], 0.0, 1.0)
            if "NORMAL" in attrs:
                nrm = _vertex_attr(gltf, buffers, attrs["NORMAL"], n_verts, warn, "NORMAL")
                if nrm is not None:
                    nrm = nrm[:, :3] @ normal_mat.T
                    out["normals"] = np.ascontiguousarray(
                        nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12), np.float32)
            if "TANGENT" in attrs:
                tan = _vertex_attr(gltf, buffers, attrs["TANGENT"], n_verts, warn, "TANGENT")
                if tan is not None and tan.shape[1] == 4:
                    txyz = tan[:, :3] @ linear.T
                    txyz /= np.maximum(np.linalg.norm(txyz, axis=1, keepdims=True), 1e-12)
                    tw = tan[:, 3:4] * (-1.0 if flip_winding else 1.0)
                    out["tangents"] = np.ascontiguousarray(np.concatenate([txyz, tw], axis=1), np.float32)
            prims.append(out)
    return prims


def _decode_texture(gltf, buffers, base_dir, tex_info, warn, label):
    if tex_info is None:
        return None
    if tex_info.get("texCoord", 0) != 0:
        warn(f"texcoord:{label}", f"{label} uses TEXCOORD_{tex_info['texCoord']}; MESH only carries TEXCOORD_0")
    if "KHR_texture_transform" in tex_info.get("extensions", {}):
        warn("textransform", "KHR_texture_transform ignored; UVs used as-is")
    tex_def = gltf["textures"][tex_info["index"]]
    source = tex_def.get("source")
    if source is None:
        webp = tex_def.get("extensions", {}).get("EXT_texture_webp")
        source = webp.get("source") if webp is not None else None
    if source is None:
        warn(f"compressed:{label}", f"{label}: compressed texture (basisu/ktx2) without fallback; skipped")
        return None
    image_def = gltf["images"][source]
    if "bufferView" in image_def:
        raw = _view_data(gltf, buffers, image_def["bufferView"])
    elif "uri" in image_def:
        raw = _resolve_uri(image_def["uri"], base_dir)
    else:
        return None
    img = Image.open(BytesIO(bytes(raw)))
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def extract_material(gltf: dict, buffers: list[bytes], base_dir: str | None, material_index, warn) -> dict:
    result = {"texture": None, "metallic_roughness": None, "normal_map": None, "emissive": None,
              "occlusion_in_mr": False, "unlit": False, "material": None}
    if material_index is None:
        return result
    mat = gltf["materials"][material_index]
    extensions = mat.get("extensions", {})
    result["unlit"] = "KHR_materials_unlit" in extensions

    pbr = mat.get("pbrMetallicRoughness", {})
    result["texture"] = _decode_texture(gltf, buffers, base_dir, pbr.get("baseColorTexture"), warn, "baseColorTexture")
    mr_info = pbr.get("metallicRoughnessTexture")
    result["metallic_roughness"] = _decode_texture(gltf, buffers, base_dir, mr_info, warn, "metallicRoughnessTexture")
    normal_info = mat.get("normalTexture")
    result["normal_map"] = _decode_texture(gltf, buffers, base_dir, normal_info, warn, "normalTexture")
    result["emissive"] = _decode_texture(gltf, buffers, base_dir, mat.get("emissiveTexture"), warn, "emissiveTexture")

    occlusion = mat.get("occlusionTexture")
    if occlusion is not None:
        if mr_info is not None and occlusion["index"] == mr_info["index"]:
            result["occlusion_in_mr"] = True
        else:
            warn("occlusion", "standalone occlusionTexture not representable in MESH (ORM packing only); skipped")

    overrides = {}
    base_color = pbr.get("baseColorFactor")
    if base_color is not None and list(base_color) != [1.0, 1.0, 1.0, 1.0]:
        overrides["base_color_factor"] = [float(c) for c in base_color]
    overrides["metallic_factor"] = float(pbr.get("metallicFactor", 1.0))
    overrides["roughness_factor"] = float(pbr.get("roughnessFactor", 1.0))
    overrides["double_sided"] = bool(mat.get("doubleSided", False))
    if normal_info is not None and normal_info.get("scale", 1.0) != 1.0:
        overrides["normal_scale"] = float(normal_info["scale"])
    if occlusion is not None and occlusion.get("strength", 1.0) != 1.0:
        overrides["occlusion_strength"] = float(occlusion["strength"])
    emissive_factor = mat.get("emissiveFactor", (0.0, 0.0, 0.0))
    if any(c > 0.0 for c in emissive_factor):
        overrides["emissive_factor"] = [float(c) for c in emissive_factor]
    strength = extensions.get("KHR_materials_emissive_strength", {}).get("emissiveStrength")
    if strength is not None:
        overrides["emissive_strength"] = float(strength)
    result["material"] = overrides
    return result


def load_gltf(data: bytes, base_dir: str | None, warn):
    gltf, bin_chunk = parse_container(data)
    version = str(gltf.get("asset", {}).get("version", ""))
    if version.partition(".")[0] != "2":
        raise ValueError(f"unsupported glTF asset version {version or 'unknown'}; only glTF 2.x is supported")
    buffers = load_buffers(gltf, bin_chunk, base_dir)
    return gltf, buffers, load_scene_geometry(gltf, buffers, warn)


__all__ = ["load_gltf", "extract_material", "parse_container", "read_accessor", "to_float"]
