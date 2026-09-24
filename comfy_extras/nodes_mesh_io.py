import logging
import os

import numpy as np
import torch
from typing_extensions import override

from comfy_api.latest import ComfyExtension, IO, Types
from comfy_extras.mesh3d.fileio import gltf_read, mesh_file_read


def _sniff_format(data: bytes) -> str:
    if data[:4] == b"glTF":
        return "glb"
    head = data[:512].lstrip()
    if head[:1] == b"{":
        return "gltf"
    if head[:5].lower() == b"solid":
        return "stl"
    return ""


def _merge_primitives(prims: list[dict]) -> dict:
    any_uv = any(p["uvs"] is not None for p in prims)
    any_color = any(p["colors"] is not None for p in prims)
    all_normals = all(p["normals"] is not None for p in prims)
    all_tangents = all_normals and all(p["tangents"] is not None for p in prims)
    color_channels = max((p["colors"].shape[1] for p in prims if p["colors"] is not None), default=3)
    if not all_normals and any(p["normals"] is not None for p in prims):
        logging.warning("Get3DComponents: some primitives lack normals; normals dropped "
                        "(MeshSmoothNormals can regenerate them)")

    verts, faces, uvs, colors, normals, tangents = [], [], [], [], [], []
    offset = 0
    for p in prims:
        v = p["positions"]
        n = v.shape[0]
        verts.append(v)
        faces.append(p["faces"] + offset)
        offset += n
        if any_uv:
            uvs.append(p["uvs"] if p["uvs"] is not None else np.zeros((n, 2), np.float32))
        if any_color:
            c = p["colors"] if p["colors"] is not None else np.ones((n, color_channels), np.float32)
            if c.shape[1] < color_channels:
                c = np.concatenate([c, np.ones((n, color_channels - c.shape[1]), np.float32)], axis=1)
            colors.append(c)
        if all_normals:
            normals.append(p["normals"])
        if all_tangents:
            tangents.append(p["tangents"])

    return {
        "vertices": np.concatenate(verts, axis=0),
        "faces": np.concatenate(faces, axis=0),
        "uvs": np.concatenate(uvs, axis=0) if any_uv else None,
        "colors": np.concatenate(colors, axis=0) if any_color else None,
        "normals": np.concatenate(normals, axis=0) if all_normals else None,
        "tangents": np.concatenate(tangents, axis=0) if all_tangents else None,
    }


def _batch(arr):
    return torch.from_numpy(arr)[None] if arr is not None else None


class Get3DComponents(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Get3DComponents",
            display_name="Get 3D Components",
            category="3d",
            description=(
                "Parse a 3D model file (GLB, GLTF, OBJ, STL) into an editable MESH for the "
                "mesh-processing nodes (decimate, remesh, UV unwrap, bake, ...). All scene "
                "nodes/primitives are merged into one mesh with their transforms applied; "
                "textures and material factors come from the first material. "
                "Counterpart of MeshToFile3D."
            ),
            search_aliases=["file 3d to mesh", "extract mesh", "convert 3d", "parse glb", "import mesh",
                            "load mesh from file", "file to mesh"],
            is_experimental=True,
            inputs=[
                IO.MultiType.Input(
                    "model_3d",
                    types=[IO.File3DGLB, IO.File3DGLTF, IO.File3DOBJ, IO.File3DSTL, IO.File3DAny],
                    tooltip="3D model file from Load 3D or another 3D node. "
                            "FBX/USDZ are not supported - convert to GLB first.",
                ),
            ],
            outputs=[IO.Mesh.Output(display_name="mesh")],
        )

    @classmethod
    def execute(cls, model_3d: Types.File3D) -> IO.NodeOutput:
        data = model_3d.get_bytes()
        fmt = (model_3d.format or _sniff_format(data)).lower()
        if fmt in ("fbx", "usdz"):
            raise ValueError(f"Get3DComponents: .{fmt} parsing is not supported; convert the model to GLB/GLTF first")

        warned = set()

        def warn_once(key, message):
            if key not in warned:
                warned.add(key)
                logging.warning("Get3DComponents: %s", message)

        material_info = None
        if fmt in ("glb", "gltf"):
            base_dir = os.path.dirname(model_3d.get_source()) if model_3d.is_disk_backed else None
            gltf, buffers, prims = gltf_read.load_gltf(data, base_dir, warn_once)
            if not prims:
                raise ValueError("Get3DComponents: no triangle geometry found in the glTF scene")
            material_indices = [p["material"] for p in prims if p["material"] is not None]
            if len(set(material_indices)) > 1:
                warn_once("multimat", f"{len(set(material_indices))} materials found; "
                                      "keeping textures/factors of the first only")
            first_material = material_indices[0] if material_indices else None
            material_info = gltf_read.extract_material(gltf, buffers, base_dir, first_material, warn_once)
        elif fmt == "obj":
            prims = [mesh_file_read.load_obj(data)]
        elif fmt == "stl":
            prims = [mesh_file_read.load_stl(data)]
        else:
            raise ValueError(f"Get3DComponents: unsupported or unrecognized format {fmt!r} "
                             "(supported: glb, gltf, obj, stl)")

        merged = _merge_primitives(prims)
        n_verts = merged["vertices"].shape[0]
        max_face = int(merged["faces"].max())
        if max_face >= n_verts:
            raise ValueError(f"Get3DComponents: face index {max_face} out of range for {n_verts} vertices (corrupt file?)")

        material_info = material_info or {}
        mesh = Types.MESH(
            vertices=_batch(merged["vertices"]),
            faces=_batch(merged["faces"]),
            uvs=_batch(merged["uvs"]),
            vertex_colors=_batch(merged["colors"]),
            normals=_batch(merged["normals"]),
            tangents=_batch(merged["tangents"]),
            texture=_batch(material_info.get("texture")),
            metallic_roughness=_batch(material_info.get("metallic_roughness")),
            normal_map=_batch(material_info.get("normal_map")),
            emissive=_batch(material_info.get("emissive")),
            unlit=bool(material_info.get("unlit", False)),
            occlusion_in_mr=bool(material_info.get("occlusion_in_mr", False)),
            material=material_info.get("material") or None,
        )
        logging.info("Get3DComponents: %s -> %d vertices, %d faces (%d primitives)",
                     fmt, n_verts, merged["faces"].shape[0], len(prims))
        return IO.NodeOutput(mesh)


class MeshIOExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [Get3DComponents]


async def comfy_entrypoint() -> MeshIOExtension:
    return MeshIOExtension()
