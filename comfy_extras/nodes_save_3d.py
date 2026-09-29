"""Save-side 3D nodes: mesh packing/slicing helpers + GLB writer + SaveGLB node."""

import copy
import json
import logging
import math
import os
import struct
from io import BytesIO
from typing import TypedDict

import numpy as np
from PIL import Image
import torch
from typing_extensions import override

import folder_paths
from comfy.cli_args import args
from comfy_api.latest import ComfyExtension, IO, Types, UI
from server import PromptServer


def pack_variable_mesh_batch(vertices, faces, colors=None, uvs=None, texture=None, unlit=False,
                             normals=None, metallic_roughness=None, tangents=None, normal_map=None,
                             occlusion_in_mr=False, material=None, emissive=None):
    # Pack per-item tensors into padded batches, stashing per-item lengths as runtime attrs.
    # colors/uvs/normals/tangents are 1:1 with vertices (padded to max_vertices); texture/
    # metallic_roughness/normal_map are (B,H,W,*) image stacks passed through unchanged.
    batch_size = len(vertices)
    max_vertices = max(v.shape[0] for v in vertices)
    max_faces = max(f.shape[0] for f in faces)

    packed_vertices = vertices[0].new_zeros((batch_size, max_vertices, vertices[0].shape[1]))
    packed_faces = faces[0].new_zeros((batch_size, max_faces, faces[0].shape[1]))
    vertex_counts = torch.tensor([v.shape[0] for v in vertices], device=vertices[0].device, dtype=torch.int64)
    face_counts = torch.tensor([f.shape[0] for f in faces], device=faces[0].device, dtype=torch.int64)

    for i, (v, f) in enumerate(zip(vertices, faces)):
        packed_vertices[i, :v.shape[0]] = v
        packed_faces[i, :f.shape[0]] = f

    packed_colors = None
    if colors is not None:
        packed_colors = colors[0].new_zeros((batch_size, max_vertices, colors[0].shape[1]))
        for i, c in enumerate(colors):
            assert c.shape[0] == vertices[i].shape[0], (
                f"vertex_colors[{i}] has {c.shape[0]} entries, expected {vertices[i].shape[0]} (1:1 with vertices)"
            )
            packed_colors[i, :c.shape[0]] = c

    packed_uvs = None
    if uvs is not None:
        packed_uvs = uvs[0].new_zeros((batch_size, max_vertices, uvs[0].shape[1]))
        for i, u in enumerate(uvs):
            assert u.shape[0] == vertices[i].shape[0], (
                f"uvs[{i}] has {u.shape[0]} entries, expected {vertices[i].shape[0]} (1:1 with vertices)"
            )
            packed_uvs[i, :u.shape[0]] = u

    packed_normals = None
    if normals is not None:
        packed_normals = normals[0].new_zeros((batch_size, max_vertices, normals[0].shape[1]))
        for i, nrm in enumerate(normals):
            assert nrm.shape[0] == vertices[i].shape[0], (
                f"normals[{i}] has {nrm.shape[0]} entries, expected {vertices[i].shape[0]} (1:1 with vertices)"
            )
            packed_normals[i, :nrm.shape[0]] = nrm

    packed_tangents = None
    if tangents is not None:
        packed_tangents = tangents[0].new_zeros((batch_size, max_vertices, tangents[0].shape[1]))
        for i, tn in enumerate(tangents):
            assert tn.shape[0] == vertices[i].shape[0], (
                f"tangents[{i}] has {tn.shape[0]} entries, expected {vertices[i].shape[0]} (1:1 with vertices)"
            )
            packed_tangents[i, :tn.shape[0]] = tn

    return Types.MESH(packed_vertices, packed_faces,
                      uvs=packed_uvs, vertex_colors=packed_colors, texture=texture,
                      metallic_roughness=metallic_roughness,
                      vertex_counts=vertex_counts, face_counts=face_counts, unlit=unlit,
                      normals=packed_normals, tangents=packed_tangents,
                      normal_map=normal_map, occlusion_in_mr=occlusion_in_mr,
                      material=material, emissive=emissive)


def get_mesh_batch_item(mesh, index):
    # Returns (vertices, faces, colors, uvs) for batch index, slicing to real lengths
    # if the mesh carries per-item counts (variable-size batch).
    v_colors = mesh.vertex_colors
    v_uvs = mesh.uvs
    v_normals = mesh.normals
    if mesh.vertex_counts is not None:
        vertex_count = int(mesh.vertex_counts[index].item())
        face_count = int(mesh.face_counts[index].item())
        vertices = mesh.vertices[index, :vertex_count]
        faces = mesh.faces[index, :face_count]
        colors = v_colors[index, :vertex_count] if v_colors is not None else None
        uvs = v_uvs[index, :vertex_count] if v_uvs is not None else None
        normals = v_normals[index, :vertex_count] if v_normals is not None else None
        return vertices, faces, colors, uvs, normals

    colors = v_colors[index] if v_colors is not None else None
    uvs = v_uvs[index] if v_uvs is not None else None
    normals = v_normals[index] if v_normals is not None else None
    return mesh.vertices[index], mesh.faces[index], colors, uvs, normals


def save_glb(vertices, faces, filepath=None, metadata=None,
             uvs=None, vertex_colors=None, texture_image=None,
             metallic_roughness_image=None, unlit=False,
             normals=None, normal_map_image=None, tangents=None, occlusion_in_mr=False,
             material=None, emissive_image=None):
    """
    Save PyTorch tensor vertices and faces as a GLB file without external dependencies.

    Parameters:
    vertices: torch.Tensor of shape (N, 3) - The vertex coordinates
    faces: torch.Tensor of shape (M, 3) - The face indices (triangle faces)
    filepath: str - Output filepath (should end with .glb). None returns the GLB bytes instead of writing.
    metadata: dict - Optional asset.extras metadata
    uvs: torch.Tensor of shape (N, 2) - Optional per-vertex texture coordinates
    vertex_colors: torch.Tensor of shape (N, 3) or (N, 4) - Optional per-vertex colors in [0, 1]
    texture_image: PIL.Image - Optional baseColor texture, embedded as PNG
    metallic_roughness_image: PIL.Image - Optional glTF metallicRoughness texture
        (R unused, G=roughness, B=metallic), embedded as PNG
    normals: torch.Tensor of shape (N, 3) - Optional per-vertex normals, written as the
        glTF NORMAL attribute. When omitted, NO normals are written and viewers fall back
        to flat (per-face) shading — use the MeshSmoothNormals node to generate them.
    normal_map_image: PIL.Image - Optional tangent-space normal map (glTF/OpenGL +Y),
        written as the material normalTexture. Needs TEXCOORD_0.
    tangents: torch.Tensor of shape (N, 4) - Optional per-vertex tangents (xyz + handedness w),
        written as the glTF TANGENT attribute. Without it viewers derive tangents in-shader.
    occlusion_in_mr: bool - When True, R of metallic_roughness_image holds AO (ORM packing) and
        occlusionTexture is pointed at that same image.
    material: dict - Optional scalar overrides from SetMeshMaterial (base_color_factor,
        metallic/roughness_factor with <0 = auto, emissive_factor/strength, normal_scale,
        occlusion_strength, double_sided).
    emissive_image: PIL.Image - Optional emissive (glow) texture, written as emissiveTexture.
    """

    # Convert tensors to numpy arrays
    vertices_np = vertices.cpu().numpy().astype(np.float32)
    faces_signed = faces.cpu().numpy().astype(np.int64)
    uvs_np = uvs.cpu().numpy().astype(np.float32) if uvs is not None else None
    colors_np = vertex_colors.cpu().numpy().astype(np.float32) if vertex_colors is not None else None
    if colors_np is not None:
        colors_np = np.clip(colors_np, 0.0, 1.0)

    n_verts = vertices_np.shape[0]
    if n_verts == 0:
        raise ValueError("save_glb: vertices is empty")
    if faces_signed.size > 0:
        fmin = int(faces_signed.min())
        fmax = int(faces_signed.max())
        if fmin < 0 or fmax >= n_verts:
            raise ValueError(
                f"save_glb: face index out of range [0, {n_verts}): min={fmin}, max={fmax}"
            )
    if uvs_np is not None and uvs_np.shape[0] != n_verts:
        raise ValueError(
            f"save_glb: uvs has {uvs_np.shape[0]} entries but vertex count is {n_verts}"
        )
    if colors_np is not None and colors_np.shape[0] != n_verts:
        raise ValueError(
            f"save_glb: vertex_colors has {colors_np.shape[0]} entries but vertex count is {n_verts}"
        )

    normals_np = normals.cpu().numpy().astype(np.float32) if normals is not None else None
    if normals_np is not None and normals_np.shape[0] != n_verts:
        raise ValueError(
            f"save_glb: normals has {normals_np.shape[0]} entries but vertex count is {n_verts}"
        )
    tangents_np = tangents.cpu().numpy().astype(np.float32) if tangents is not None else None
    if tangents_np is not None and tangents_np.shape != (n_verts, 4):
        raise ValueError(
            f"save_glb: tangents must be (N, 4) with N={n_verts}, got {tuple(tangents_np.shape)}"
        )
    faces_np = faces_signed.astype(np.uint32)
    texture_png_bytes = None
    if texture_image is not None:
        buf = BytesIO()
        texture_image.save(buf, format="PNG")
        texture_png_bytes = buf.getvalue()
    mr_png_bytes = None
    if metallic_roughness_image is not None:
        buf = BytesIO()
        metallic_roughness_image.save(buf, format="PNG")
        mr_png_bytes = buf.getvalue()
    nm_png_bytes = None
    if normal_map_image is not None:
        buf = BytesIO()
        normal_map_image.save(buf, format="PNG")
        nm_png_bytes = buf.getvalue()
    em_png_bytes = None
    if emissive_image is not None:
        buf = BytesIO()
        emissive_image.save(buf, format="PNG")
        em_png_bytes = buf.getvalue()

    vertices_buffer = vertices_np.tobytes()
    indices_buffer = faces_np.tobytes()
    uvs_buffer = uvs_np.tobytes() if uvs_np is not None else b""
    colors_buffer = colors_np.tobytes() if colors_np is not None else b""
    normals_buffer = normals_np.tobytes() if normals_np is not None else b""
    tangents_buffer = tangents_np.tobytes() if tangents_np is not None else b""
    texture_buffer = texture_png_bytes if texture_png_bytes is not None else b""
    mr_buffer = mr_png_bytes if mr_png_bytes is not None else b""
    nm_buffer = nm_png_bytes if nm_png_bytes is not None else b""
    em_buffer = em_png_bytes if em_png_bytes is not None else b""

    def pad_to_4_bytes(buffer):
        padding_length = (4 - (len(buffer) % 4)) % 4
        return buffer + b'\x00' * padding_length

    # Blob order in one place; offsets accumulated in a pass so adding a buffer is one entry.
    _blobs = [
        ("vertices", vertices_buffer), ("indices", indices_buffer), ("uvs", uvs_buffer),
        ("colors", colors_buffer), ("normals", normals_buffer), ("tangents", tangents_buffer),
        ("texture", texture_buffer), ("mr", mr_buffer), ("nm", nm_buffer), ("em", em_buffer),
    ]
    byte_offset = {}
    acc = 0
    parts = []
    for name, b in _blobs:
        padded = pad_to_4_bytes(b)
        byte_offset[name] = acc
        acc += len(padded)
        parts.append(padded)
    buffer_data = b"".join(parts)

    vertices_byte_length = len(vertices_buffer)
    indices_byte_length = len(indices_buffer)
    vertices_byte_offset = byte_offset["vertices"]
    indices_byte_offset = byte_offset["indices"]
    uvs_byte_offset = byte_offset["uvs"]
    colors_byte_offset = byte_offset["colors"]
    normals_byte_offset = byte_offset["normals"]
    tangents_byte_offset = byte_offset["tangents"]
    texture_byte_offset = byte_offset["texture"]
    mr_byte_offset = byte_offset["mr"]
    nm_byte_offset = byte_offset["nm"]
    em_byte_offset = byte_offset["em"]

    buffer_views = [
        {
            "buffer": 0,
            "byteOffset": vertices_byte_offset,
            "byteLength": vertices_byte_length,
            "target": 34962  # ARRAY_BUFFER
        },
        {
            "buffer": 0,
            "byteOffset": indices_byte_offset,
            "byteLength": indices_byte_length,
            "target": 34963  # ELEMENT_ARRAY_BUFFER
        }
    ]
    accessors = [
        {
            "bufferView": 0,
            "byteOffset": 0,
            "componentType": 5126,  # FLOAT
            "count": len(vertices_np),
            "type": "VEC3",
            "max": vertices_np.max(axis=0).tolist(),
            "min": vertices_np.min(axis=0).tolist()
        },
        {
            "bufferView": 1,
            "byteOffset": 0,
            "componentType": 5125,  # UNSIGNED_INT
            "count": faces_np.size,
            "type": "SCALAR"
        }
    ]
    primitive_attributes = {"POSITION": 0}

    if uvs_np is not None and len(uvs_np) > 0:
        buffer_views.append({
            "buffer": 0,
            "byteOffset": uvs_byte_offset,
            "byteLength": len(uvs_buffer),
            "target": 34962
        })
        accessor_idx = len(accessors)
        accessors.append({
            "bufferView": len(buffer_views) - 1,
            "byteOffset": 0,
            "componentType": 5126,
            "count": len(uvs_np),
            "type": "VEC2",
        })
        primitive_attributes["TEXCOORD_0"] = accessor_idx

    if colors_np is not None and len(colors_np) > 0:
        buffer_views.append({
            "buffer": 0,
            "byteOffset": colors_byte_offset,
            "byteLength": len(colors_buffer),
            "target": 34962
        })
        accessor_idx = len(accessors)
        accessors.append({
            "bufferView": len(buffer_views) - 1,
            "byteOffset": 0,
            "componentType": 5126,
            "count": len(colors_np),
            "type": "VEC3" if colors_np.shape[1] == 3 else "VEC4",
        })
        primitive_attributes["COLOR_0"] = accessor_idx

    if normals_np is not None and len(normals_np) > 0:
        buffer_views.append({
            "buffer": 0,
            "byteOffset": normals_byte_offset,
            "byteLength": len(normals_buffer),
            "target": 34962
        })
        accessor_idx = len(accessors)
        accessors.append({
            "bufferView": len(buffer_views) - 1,
            "byteOffset": 0,
            "componentType": 5126,  # FLOAT
            "count": len(normals_np),
            "type": "VEC3",
        })
        primitive_attributes["NORMAL"] = accessor_idx

    if tangents_np is not None and len(tangents_np) > 0:
        buffer_views.append({
            "buffer": 0,
            "byteOffset": tangents_byte_offset,
            "byteLength": len(tangents_buffer),
            "target": 34962
        })
        accessor_idx = len(accessors)
        accessors.append({
            "bufferView": len(buffer_views) - 1,
            "byteOffset": 0,
            "componentType": 5126,  # FLOAT
            "count": len(tangents_np),
            "type": "VEC4",  # xyz tangent + w handedness (glTF TANGENT)
        })
        primitive_attributes["TANGENT"] = accessor_idx

    primitive = {
        "attributes": primitive_attributes,
        "indices": 1,
        "mode": 4  # TRIANGLES
    }

    images = []
    textures = []
    samplers = []
    materials = []
    extensions_used = []

    def add_image_texture(png_byte_offset, png_byte_length):
        """Append an embedded PNG image + a texture referencing it; return the texture index."""
        buffer_views.append({"buffer": 0, "byteOffset": png_byte_offset, "byteLength": png_byte_length})
        images.append({"bufferView": len(buffer_views) - 1, "mimeType": "image/png"})
        if not samplers:
            samplers.append({"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071})
        textures.append({"source": len(images) - 1, "sampler": 0})
        return len(textures) - 1

    has_uv = "TEXCOORD_0" in primitive_attributes
    if unlit and texture_png_bytes is None:
        # Flat, light-independent shading (KHR_materials_unlit): COLOR_0 is shown as-is, matching how a
        # gaussian splat renders (emissive). Without this the viewer lights the mesh and washes the colours.
        if nm_png_bytes is not None or em_png_bytes is not None or occlusion_in_mr or material is not None:
            logging.warning(
                "save_glb: unlit material ignores normal/occlusion/emissive maps and SetMeshMaterial "
                "overrides — those are PBR-lit features. Disable unlit to export them.")
        materials.append({
            "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0},
            "extensions": {"KHR_materials_unlit": {}},
            "doubleSided": True,
        })
        extensions_used.append("KHR_materials_unlit")
        primitive["material"] = 0
    else:
        pbr = {
            "metallicFactor": 0.0,
            "roughnessFactor": 0.5,
            "baseColorFactor": [0.22, 0.22, 0.22, 1.0],   # neutral-gray fallback for bare geometry only
        }
        if texture_png_bytes is not None and has_uv:
            pbr["baseColorTexture"] = {"index": add_image_texture(texture_byte_offset, len(texture_buffer)), "texCoord": 0}

        if (texture_png_bytes is not None and has_uv) or "COLOR_0" in primitive_attributes:
            pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
            pbr["roughnessFactor"] = 1.0

        if mr_png_bytes is not None and has_uv:
            mr_texture_index = add_image_texture(mr_byte_offset, len(mr_buffer))
            pbr["metallicRoughnessTexture"] = {"index": mr_texture_index, "texCoord": 0}
            # When a metallicRoughness texture is present, the factors scale it; use 1.0
            # so the texture values pass through unchanged (glTF convention).
            pbr["metallicFactor"] = 1.0
            pbr["roughnessFactor"] = 1.0

        mat = material if isinstance(material, dict) else {}
        # Scalar overrides from SetMeshMaterial (factor < 0 means "leave auto").
        if mat.get("base_color_factor") is not None:
            pbr["baseColorFactor"] = [float(x) for x in mat["base_color_factor"]]
        if mat.get("metallic_factor", -1.0) >= 0.0:
            pbr["metallicFactor"] = float(mat["metallic_factor"])
        if mat.get("roughness_factor", -1.0) >= 0.0:
            pbr["roughnessFactor"] = float(mat["roughness_factor"])

        material = {
            "pbrMetallicRoughness": pbr,
            "doubleSided": bool(mat.get("double_sided", True)),
        }
        if occlusion_in_mr and mr_png_bytes is not None and has_uv:
            # ORM packing: occlusionTexture reuses the MR image (glTF reads its R channel).
            material["occlusionTexture"] = {"index": mr_texture_index, "texCoord": 0,
                                            "strength": float(mat.get("occlusion_strength", 1.0))}
        if nm_png_bytes is not None and has_uv:
            material["normalTexture"] = {"index": add_image_texture(nm_byte_offset, len(nm_buffer)),
                                         "texCoord": 0, "scale": float(mat.get("normal_scale", 1.0))}

        emissive_factor = [float(x) for x in mat.get("emissive_factor", [0.0, 0.0, 0.0])]
        emissive_strength = float(mat.get("emissive_strength", 1.0))
        has_em_tex = em_png_bytes is not None and has_uv
        if any(c > 0.0 for c in emissive_factor) or has_em_tex:
            # glTF multiplies emissiveFactor × texture, so a texture with no color would go black;
            # default the factor to white in that case.
            if has_em_tex and not any(c > 0.0 for c in emissive_factor):
                emissive_factor = [1.0, 1.0, 1.0]
            material["emissiveFactor"] = [min(1.0, c) for c in emissive_factor]
            if has_em_tex:
                material["emissiveTexture"] = {"index": add_image_texture(em_byte_offset, len(em_buffer)),
                                               "texCoord": 0}
            if emissive_strength != 1.0:
                material.setdefault("extensions", {})["KHR_materials_emissive_strength"] = {
                    "emissiveStrength": emissive_strength}
                if "KHR_materials_emissive_strength" not in extensions_used:
                    extensions_used.append("KHR_materials_emissive_strength")

        materials.append(material)
        primitive["material"] = 0

    gltf = {
        "asset": {"version": "2.0", "generator": "ComfyUI"},
        "buffers": [{"byteLength": len(buffer_data)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "meshes": [{"primitives": [primitive]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    if images:
        gltf["images"] = images
    if samplers:
        gltf["samplers"] = samplers
    if textures:
        gltf["textures"] = textures
    if materials:
        gltf["materials"] = materials
    if extensions_used:
        gltf["extensionsUsed"] = extensions_used

    if metadata:
        gltf["asset"]["extras"] = metadata

    # Convert the JSON to bytes
    gltf_json = json.dumps(gltf).encode('utf8')

    def pad_json_to_4_bytes(buffer):
        padding_length = (4 - (len(buffer) % 4)) % 4
        return buffer + b' ' * padding_length

    gltf_json_padded = pad_json_to_4_bytes(gltf_json)

    # Create the GLB header (a 4-byte ASCII magic identifier glTF)
    glb_header = struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(gltf_json_padded) + 8 + len(buffer_data))

    # Create JSON chunk header (chunk type 0)
    json_chunk_header = struct.pack('<II', len(gltf_json_padded), 0x4E4F534A)  # "JSON" in little endian

    # Create BIN chunk header (chunk type 1)
    bin_chunk_header = struct.pack('<II', len(buffer_data), 0x004E4942)  # "BIN\0" in little endian

    glb = b"".join([glb_header, json_chunk_header, gltf_json_padded, bin_chunk_header, buffer_data])
    if filepath is None:
        return glb                       # in-memory GLB bytes (e.g. for a File3D object)
    with open(filepath, 'wb') as f:
        f.write(glb)
    return filepath


def mesh_item_to_glb_bytes(mesh, index, metadata=None):
    """Serialize one batch item of a MESH to in-memory GLB bytes, carrying every PBR attribute
    (uvs, colors, normals, texture, ORM/occlusion, normal map + tangents, emissive, material).
    Returns None for an empty item. Shared by SaveGLB (per item) and MeshToFile3D."""
    vertices_i, faces_i, v_colors, uvs_i, normals_i = get_mesh_batch_item(mesh, index)
    if vertices_i.shape[0] == 0 or faces_i.shape[0] == 0:
        return None

    def _img(attr):
        t = getattr(mesh, attr, None)
        if t is None:
            return None
        a = (t[index].clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
        assert a.ndim == 3 and a.shape[-1] == 3, f"{attr} must be (B, H, W, 3), got {tuple(t.shape)}"
        return Image.fromarray(a, mode="RGB")

    tangents_b = mesh.tangents
    tangents_i = tangents_b[index, :vertices_i.shape[0]] if tangents_b is not None else None
    return save_glb(
        vertices_i, faces_i, None, metadata,
        uvs=uvs_i,
        vertex_colors=v_colors,
        texture_image=_img("texture"),
        metallic_roughness_image=_img("metallic_roughness"),
        unlit=mesh.unlit,
        normals=normals_i,
        normal_map_image=_img("normal_map"),
        tangents=tangents_i,
        occlusion_in_mr=mesh.occlusion_in_mr,
        material=mesh.material,
        emissive_image=_img("emissive"),
    )


class SaveGLB(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveGLB",
            display_name="Save 3D Model",
            search_aliases=["export 3d model", "save mesh"],
            category="3d",
            essentials_category="Basics",
            is_output_node=True,
            inputs=[
                IO.MultiType.Input(
                    IO.Mesh.Input("mesh"),
                    types=[
                        IO.File3DGLB,
                        IO.File3DGLTF,
                        IO.File3DOBJ,
                        IO.File3DFBX,
                        IO.File3DSTL,
                        IO.File3DUSDZ,
                        IO.File3DPLY,
                        IO.File3DSPLAT,
                        IO.File3DSPZ,
                        IO.File3DKSPLAT,
                        IO.File3DSplatAny,
                        IO.File3DPointCloudAny,
                        IO.File3DAny,
                    ],
                    tooltip="Mesh or 3D file to save",
                ),
                IO.String.Input("filename_prefix", default="3d/ComfyUI"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo]
        )

    @classmethod
    def execute(cls, mesh: Types.MESH | Types.File3D, filename_prefix: str) -> IO.NodeOutput:
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        results = []

        metadata = {}
        if not args.disable_metadata:
            if cls.hidden.prompt is not None:
                metadata["prompt"] = json.dumps(cls.hidden.prompt)
            if cls.hidden.extra_pnginfo is not None:
                for x in cls.hidden.extra_pnginfo:
                    metadata[x] = json.dumps(cls.hidden.extra_pnginfo[x])

        if isinstance(mesh, Types.File3D):
            # Handle File3D input - save BytesIO data to output folder
            ext = mesh.format or "glb"
            f = f"{filename}_{counter:05}_.{ext}"
            mesh.save_to(os.path.join(full_output_folder, f))
            results.append({
                "filename": f,
                "subfolder": subfolder,
                "type": "output"
            })
            counter += 1
        else:
            # Handle Mesh input - save vertices and faces as GLB; carry optional UVs / colors / texture.
            for i in range(mesh.vertices.shape[0]):
                glb = mesh_item_to_glb_bytes(mesh, i, metadata)
                if glb is None:
                    logging.warning(f"SaveGLB: skipping empty mesh at batch index {i}")
                    continue
                f = f"{filename}_{counter:05}_.glb"
                with open(os.path.join(full_output_folder, f), "wb") as fh:
                    fh.write(glb)
                results.append({
                    "filename": f,
                    "subfolder": subfolder,
                    "type": "output"
                })
                counter += 1
        return IO.NodeOutput(ui={"3d": results})


class MeshToFile3D(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MeshToFile3D",
            display_name="Create 3D File (from Mesh)",
            search_aliases=["mesh to glb", "mesh to file", "export mesh"],
            category="3d",
            description="Serialize a mesh to a GLB File3D object for Save / Preview 3D nodes, "
                        "carrying its UVs, colors, normals, texture, normal/occlusion/emissive "
                        "maps and material. Supports one item per batch only.",
            inputs=[IO.Mesh.Input("mesh")],
            outputs=[IO.File3DGLB.Output(display_name="model_3d")],
        )

    @classmethod
    def execute(cls, mesh) -> IO.NodeOutput:
        if mesh.vertices.shape[0] > 1:
            logging.warning("MeshToFile3D supports one item per batch only. Got %d; using first.",
                            mesh.vertices.shape[0])
        glb = mesh_item_to_glb_bytes(mesh, 0)
        if glb is None:
            raise ValueError("MeshToFile3D: mesh is empty (no vertices/faces).")
        return IO.NodeOutput(Types.File3D(BytesIO(glb), file_format="glb"))


class RotateMesh(IO.ComfyNode):
    class ModeValues(TypedDict, total=False):
        mode: str
        angle_x: float
        angle_y: float
        angle_z: float
        qw: float
        qx: float
        qy: float
        qz: float

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RotateMesh",
            display_name="Rotate Mesh",
            category="3d/mesh",
            description=(
                "Rotate a mesh. Euler XYZ applies X then Y then Z about the world axes (degrees). "
                "Quaternion is (w, x, y, z), auto-normalized."
            ),
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.DynamicCombo.Input(
                    "mode",
                    options=[
                        IO.DynamicCombo.Option("euler_xyz", [
                            IO.Float.Input("angle_x", default=0.0, min=-360.0, max=360.0, step=0.1,
                                           tooltip="Rotation around the X axis in degrees."),
                            IO.Float.Input("angle_y", default=0.0, min=-360.0, max=360.0, step=0.1,
                                           tooltip="Rotation around the Y axis in degrees."),
                            IO.Float.Input("angle_z", default=0.0, min=-360.0, max=360.0, step=0.1,
                                           tooltip="Rotation around the Z axis in degrees."),
                        ]),
                        IO.DynamicCombo.Option("quaternion", [
                            IO.Float.Input("qw", default=1.0, min=-1.0, max=1.0, step=0.001),
                            IO.Float.Input("qx", default=0.0, min=-1.0, max=1.0, step=0.001),
                            IO.Float.Input("qy", default=0.0, min=-1.0, max=1.0, step=0.001),
                            IO.Float.Input("qz", default=0.0, min=-1.0, max=1.0, step=0.001),
                        ]),
                    ],
                ),
            ],
            outputs=[IO.Mesh.Output("mesh")],
        )

    @classmethod
    def execute(cls, mesh: Types.MESH, mode: ModeValues) -> IO.NodeOutput:
        mode_name = mode["mode"]
        if mode_name == "euler_xyz":
            ax = math.radians(mode["angle_x"])
            ay = math.radians(mode["angle_y"])
            az = math.radians(mode["angle_z"])
            if ax == 0.0 and ay == 0.0 and az == 0.0:
                return IO.NodeOutput(mesh)
            cx, sx = math.cos(ax), math.sin(ax)
            cy, sy = math.cos(ay), math.sin(ay)
            cz, sz = math.cos(az), math.sin(az)
            R_rows = [
                [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
                [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
                [-sy,     sx * cy,                cx * cy],
            ]
        elif mode_name == "quaternion":
            qw, qx, qy, qz = mode["qw"], mode["qx"], mode["qy"], mode["qz"]
            n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            if n < 1e-8:
                raise ValueError("RotateMesh: quaternion has zero magnitude")
            qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
            if qw == 1.0 and qx == 0.0 and qy == 0.0 and qz == 0.0:
                return IO.NodeOutput(mesh)
            R_rows = [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
            ]
        else:
            raise ValueError(f"RotateMesh: unknown mode {mode_name!r}")

        def rotate(v: torch.Tensor) -> torch.Tensor:
            R = torch.tensor(R_rows, device=v.device, dtype=v.dtype)
            return v @ R.T

        out = copy.copy(mesh)
        if isinstance(mesh.vertices, list):
            out.vertices = [rotate(v) for v in mesh.vertices]
        else:
            out.vertices = rotate(mesh.vertices)
        # Normals are directions, rotate them too (R is orthogonal).
        nrm = mesh.normals
        if nrm is not None:
            out.normals = [rotate(n) for n in nrm] if isinstance(nrm, list) else rotate(nrm)
        # Tangents (xyz + handedness w) are directions too: rotate xyz, keep w
        tng = mesh.tangents
        if tng is not None:
            def rotate_tangent(t: torch.Tensor) -> torch.Tensor:
                rt = t.clone()
                rt[..., :3] = rotate(t[..., :3])
                return rt
            out.tangents = ([rotate_tangent(t) for t in tng] if isinstance(tng, list)
                            else rotate_tangent(tng))
        return IO.NodeOutput(out)


class MergeMeshes(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = IO.Autogrow.TemplatePrefix(
            IO.Mesh.Input("mesh"), prefix="mesh", min=2, max=50,
        )
        return IO.Schema(
            node_id="MergeMeshes",
            display_name="Merge Meshes",
            category="3d/mesh",
            description=(
                "Concatenate N meshes into one by offsetting face indices and stacking verts, "
                "faces, uvs, and colors."
            ),
            inputs=[
                IO.Autogrow.Input("meshes", template=autogrow_template),
            ],
            outputs=[IO.Mesh.Output("mesh")],
        )

    @classmethod
    def execute(cls, meshes: IO.Autogrow.Type) -> IO.NodeOutput:
        # Concatenate the input meshes into one (B=1) mesh: cumulative face-index offset,
        # missing uvs/colors padded (zeros/white), texture from the first input that has one
        # (later dropped — a single-primitive glb can't carry multiple atlases).
        meshes = list(meshes.values())
        if not meshes:
            raise ValueError("MergeMeshes: need at least one mesh")

        any_uvs = any(m.uvs is not None for m in meshes)
        any_colors = any(m.vertex_colors is not None for m in meshes)
        color_channels = max((int(m.vertex_colors.shape[-1]) for m in meshes if m.vertex_colors is not None), default=3)

        verts_list, faces_list, uvs_list, colors_list = [], [], [], []
        texture = None
        offset = 0
        for m in meshes:
            # Coerce to CPU so CUDA-side (MoGe) meshes merge cleanly with our outputs.
            v, f, mc, mu, _ = get_mesh_batch_item(m, 0)
            v = v.cpu()
            f = f.cpu()
            verts_list.append(v)
            faces_list.append(f + offset)
            offset += v.shape[0]
            if any_uvs:
                uvs_list.append(mu.cpu() if mu is not None else v.new_zeros((v.shape[0], 2)))
            if any_colors:
                c = mc.cpu() if mc is not None else v.new_ones((v.shape[0], color_channels))
                if c.shape[-1] < color_channels:
                    c = torch.cat((c, c.new_ones((c.shape[0], color_channels - c.shape[-1]))), dim=-1)
                colors_list.append(c)
            mt = m.texture
            if mt is not None:
                if texture is None:
                    texture = mt.cpu()
                else:
                    logging.warning("MergeMeshes: dropping extra texture from input; only one texture is kept.")

        merged_verts = torch.cat(verts_list, dim=0).unsqueeze(0)
        merged_faces = torch.cat(faces_list, dim=0).unsqueeze(0)
        merged_uvs = torch.cat(uvs_list, dim=0).unsqueeze(0) if any_uvs else None
        merged_colors = torch.cat(colors_list, dim=0).unsqueeze(0) if any_colors else None

        return IO.NodeOutput(Types.MESH(
            vertices=merged_verts,
            faces=merged_faces,
            uvs=merged_uvs,
            vertex_colors=merged_colors,
            texture=texture,
        ))


class GetMeshInfo(IO.ComfyNode):
    """Report vertex / face counts and attributes for a MESH, displayed on the
    node (and as a string output). Counts are comma-formatted since meshes can
    run into the millions of faces. Passes the mesh through unchanged."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GetMeshInfo",
            display_name="Get Mesh Info",
            category="3d/mesh",
            inputs=[IO.Mesh.Input("mesh")],
            outputs=[
                IO.Mesh.Output(display_name="mesh"),
                IO.String.Output(display_name="info"),
            ],
            hidden=[IO.Hidden.unique_id],
        )

    @staticmethod
    def _fmt(n: int) -> str:
        # e.g. 1234567 -> "1,234,567 (1.23M)"; small numbers stay plain.
        s = f"{n:,}"
        if n >= 1_000_000:
            s += f" ({n / 1_000_000:.2f}M)"
        elif n >= 10_000:
            s += f" ({n / 1_000:.1f}K)"
        return s

    @classmethod
    def execute(cls, mesh):
        B = mesh.vertices.shape[0]
        # Honour per-item counts when the batch is zero-padded; else use the row sizes.
        if mesh.vertex_counts is not None:
            v_counts = [int(x) for x in mesh.vertex_counts.tolist()]
            f_counts = [int(x) for x in mesh.face_counts.tolist()]
        else:
            v_counts = [int(mesh.vertices.shape[1])] * B
            f_counts = [int(mesh.faces.shape[1])] * B

        attrs = []
        for name in ("uvs", "vertex_colors", "normals", "tangents", "texture", "metallic_roughness", "normal_map"):
            t = getattr(mesh, name, None)
            if t is not None:
                if name in ("texture", "metallic_roughness", "normal_map"):
                    attrs.append(f"{name} {int(t.shape[-3])}×{int(t.shape[-2])}")  # H×W
                else:
                    attrs.append(name)

        lines = []
        if B > 1:
            lines.append(f"Batch:      {B} meshes")
            lines.append(f"Vertices:   {cls._fmt(sum(v_counts))} total")
            lines.append(f"Faces:      {cls._fmt(sum(f_counts))} total")
            for i in range(B):
                lines.append(f"  [{i}]  {v_counts[i]:>10,} verts  ·  {f_counts[i]:>10,} faces")
        else:
            lines.append(f"Vertices:   {cls._fmt(v_counts[0])}")
            lines.append(f"Faces:      {cls._fmt(f_counts[0])}")
        lines.append(f"Attributes: {', '.join(attrs) if attrs else 'none'}")

        info = "\n".join(lines)
        logging.info("[GetMeshInfo]\n%s", info)

        if cls.hidden.unique_id:
            PromptServer.instance.send_progress_text(info, cls.hidden.unique_id)
        return IO.NodeOutput(mesh, info, ui=UI.PreviewText(info))


def _save_file3d_to_output(model_3d: Types.File3D, filename_prefix: str) -> UI.SavedResult:
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix, folder_paths.get_output_directory()
    )
    ext = model_3d.format or "glb"
    saved_filename = f"{filename}_{counter:05}.{ext}"
    model_3d.save_to(os.path.join(full_output_folder, saved_filename))
    return UI.SavedResult(saved_filename, subfolder, IO.FolderType.output)


def execute_save_3d_advanced(model_3d, viewport_state, width, height, filename_prefix, kwargs) -> IO.NodeOutput:
    saved = _save_file3d_to_output(model_3d, filename_prefix)
    model_file = f"{saved.subfolder}/{saved.filename}" if saved.subfolder else saved.filename
    viewport_state = viewport_state if isinstance(viewport_state, dict) else {}
    camera_info_input = kwargs.get("camera_info", None)
    camera_info = camera_info_input if camera_info_input is not None else viewport_state.get('camera_info')
    model_3d_info_input = kwargs.get("model_3d_info", None)
    model_3d_info = model_3d_info_input if model_3d_info_input is not None else viewport_state.get('model_3d_info', [])
    return IO.NodeOutput(
        model_3d,
        model_3d_info,
        camera_info,
        width,
        height,
        ui=UI.PreviewUI3DAdvanced(model_file, camera_info, model_3d_info, saved_result=saved),
    )


class Save3DAdvanced(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Save3DAdvanced",
            display_name="Save 3D (Advanced)",
            search_aliases=["save 3d", "export 3d model", "save mesh advanced"],
            category="3d",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                IO.MultiType.Input(
                    "model_3d",
                    types=[
                        IO.File3DGLB,
                        IO.File3DGLTF,
                        IO.File3DFBX,
                        IO.File3DOBJ,
                        IO.File3DSTL,
                        IO.File3DUSDZ,
                        IO.File3DAny,
                    ],
                    tooltip="3D model file from an upstream 3D node.",
                ),
                IO.String.Input("filename_prefix", default="3d/ComfyUI"),
                IO.Load3D.Input("viewport_state"),
                IO.Load3DModelInfo.Input("model_3d_info", optional=True, advanced=True, tooltip="Placement of each model in the scene: position, rotation, and scale (Y-up world space)."),
                IO.Load3DCamera.Input("camera_info", optional=True, advanced=True, tooltip="Viewport camera information: position, look-at target, zoom, and type."),
                IO.Int.Input("width", default=1024, min=1, max=4096, step=1, tooltip="Render width of the viewport in pixels."),
                IO.Int.Input("height", default=1024, min=1, max=4096, step=1, tooltip="Render height of the viewport in pixels."),
            ],
            outputs=[
                IO.File3DAny.Output(display_name="model_3d", tooltip="3D model file (glb/obj/stl/etc.) from an upstream 3D node."),
                IO.Load3DModelInfo.Output(display_name="model_3d_info", tooltip="Placement of each model in the scene: position, rotation, and scale (Y-up world space)."),
                IO.Load3DCamera.Output(display_name="camera_info", tooltip="Viewport camera information: position, look-at target, zoom, and type."),
                IO.Int.Output(display_name="width", tooltip="Render width of the viewport in pixels."),
                IO.Int.Output(display_name="height", tooltip="Render height of the viewport in pixels."),
            ],
        )

    @classmethod
    def execute(cls, model_3d: Types.File3D, viewport_state, width: int, height: int, filename_prefix: str, **kwargs) -> IO.NodeOutput:
        return execute_save_3d_advanced(model_3d, viewport_state, width, height, filename_prefix, kwargs)


class SaveGaussianSplat(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveGaussianSplat",
            display_name="Save Splat",
            search_aliases=["save splat", "save gaussian splat", "export gaussian", "export splat"],
            category="3d",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                IO.MultiType.Input(
                    "model_3d",
                    types=[
                        IO.File3DSplatAny,
                        IO.File3DPLY,
                        IO.File3DSPLAT,
                        IO.File3DSPZ,
                        IO.File3DKSPLAT,
                    ],
                    tooltip="A gaussian splat 3D file.",
                ),
                IO.String.Input("filename_prefix", default="3d/ComfyUI"),
                IO.Load3D.Input("viewport_state"),
                IO.Load3DModelInfo.Input("model_3d_info", optional=True, advanced=True),
                IO.Load3DCamera.Input("camera_info", optional=True, advanced=True),
                IO.Int.Input("width", default=1024, min=1, max=4096, step=1),
                IO.Int.Input("height", default=1024, min=1, max=4096, step=1),
            ],
            outputs=[
                IO.File3DSplatAny.Output(display_name="model_3d"),
                IO.Load3DModelInfo.Output(display_name="model_3d_info"),
                IO.Load3DCamera.Output(display_name="camera_info"),
                IO.Int.Output(display_name="width"),
                IO.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, model_3d: Types.File3D, viewport_state, width: int, height: int, filename_prefix: str, **kwargs) -> IO.NodeOutput:
        return execute_save_3d_advanced(model_3d, viewport_state, width, height, filename_prefix, kwargs)


class SavePointCloud(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SavePointCloud",
            display_name="Save Point Cloud",
            search_aliases=["save point cloud", "save pointcloud", "export point cloud"],
            category="3d",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                IO.MultiType.Input(
                    "model_3d",
                    types=[
                        IO.File3DPointCloudAny,
                        IO.File3DPLY,
                    ],
                    tooltip="Point cloud file (.ply)",
                ),
                IO.String.Input("filename_prefix", default="3d/ComfyUI"),
                IO.Load3D.Input("viewport_state"),
                IO.Load3DModelInfo.Input("model_3d_info", optional=True, advanced=True),
                IO.Load3DCamera.Input("camera_info", optional=True, advanced=True),
                IO.Int.Input("width", default=1024, min=1, max=4096, step=1),
                IO.Int.Input("height", default=1024, min=1, max=4096, step=1),
            ],
            outputs=[
                IO.File3DPointCloudAny.Output(display_name="model_3d"),
                IO.Load3DModelInfo.Output(display_name="model_3d_info"),
                IO.Load3DCamera.Output(display_name="camera_info"),
                IO.Int.Output(display_name="width"),
                IO.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, model_3d: Types.File3D, viewport_state, width: int, height: int, filename_prefix: str, **kwargs) -> IO.NodeOutput:
        return execute_save_3d_advanced(model_3d, viewport_state, width, height, filename_prefix, kwargs)


class Save3DExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [SaveGLB, MeshToFile3D, RotateMesh, MergeMeshes, GetMeshInfo, Save3DAdvanced, SaveGaussianSplat, SavePointCloud]


async def comfy_entrypoint() -> Save3DExtension:
    return Save3DExtension()
