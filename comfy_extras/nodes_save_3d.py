"""Save-side 3D nodes: mesh packing/slicing helpers + GLB writer + SaveGLB node."""

import json
import logging
import os
import struct
from io import BytesIO

import numpy as np
from PIL import Image
import torch
from typing_extensions import override

import folder_paths
from comfy.cli_args import args
from comfy_api.latest import ComfyExtension, IO, Types


def pack_variable_mesh_batch(vertices, faces, colors=None, uvs=None, texture=None, unlit=False):
    # Pack lists of (Nᵢ, *) vertex/face/color/uv tensors into padded batched tensors,
    # stashing per-item lengths as runtime attrs so consumers can recover the real slice.
    # colors and uvs are 1:1 with vertices, so they're padded to max_vertices and read with vertex_counts.
    # texture is (B, H, W, 3) — passed through unchanged
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

    return Types.MESH(packed_vertices, packed_faces,
                      uvs=packed_uvs, vertex_colors=packed_colors, texture=texture,
                      vertex_counts=vertex_counts, face_counts=face_counts, unlit=unlit)


def get_mesh_batch_item(mesh, index):
    # Returns (vertices, faces, colors, uvs) for batch index, slicing to real lengths
    # if the mesh carries per-item counts (variable-size batch).
    v_colors = getattr(mesh, "vertex_colors", None)
    v_uvs = getattr(mesh, "uvs", None)
    if getattr(mesh, "vertex_counts", None) is not None:
        vertex_count = int(mesh.vertex_counts[index].item())
        face_count = int(mesh.face_counts[index].item())
        vertices = mesh.vertices[index, :vertex_count]
        faces = mesh.faces[index, :face_count]
        colors = v_colors[index, :vertex_count] if v_colors is not None else None
        uvs = v_uvs[index, :vertex_count] if v_uvs is not None else None
        return vertices, faces, colors, uvs

    colors = v_colors[index] if v_colors is not None else None
    uvs = v_uvs[index] if v_uvs is not None else None
    return mesh.vertices[index], mesh.faces[index], colors, uvs


def save_glb(vertices, faces, filepath, metadata=None,
             uvs=None, vertex_colors=None, texture_image=None, unlit=False):
    """
    Save PyTorch tensor vertices and faces as a GLB file without external dependencies.

    Parameters:
    vertices: torch.Tensor of shape (N, 3) - The vertex coordinates
    faces: torch.Tensor of shape (M, 3) - The face indices (triangle faces)
    filepath: str - Output filepath (should end with .glb)
    metadata: dict - Optional asset.extras metadata
    uvs: torch.Tensor of shape (N, 2) - Optional per-vertex texture coordinates
    vertex_colors: torch.Tensor of shape (N, 3) or (N, 4) - Optional per-vertex colors in [0, 1]
    texture_image: PIL.Image - Optional baseColor texture, embedded as PNG
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
    faces_np = faces_signed.astype(np.uint32)
    texture_png_bytes = None
    if texture_image is not None:
        buf = BytesIO()
        texture_image.save(buf, format="PNG")
        texture_png_bytes = buf.getvalue()

    vertices_buffer = vertices_np.tobytes()
    indices_buffer = faces_np.tobytes()
    uvs_buffer = uvs_np.tobytes() if uvs_np is not None else b""
    colors_buffer = colors_np.tobytes() if colors_np is not None else b""
    texture_buffer = texture_png_bytes if texture_png_bytes is not None else b""

    def pad_to_4_bytes(buffer):
        padding_length = (4 - (len(buffer) % 4)) % 4
        return buffer + b'\x00' * padding_length

    vertices_buffer_padded = pad_to_4_bytes(vertices_buffer)
    indices_buffer_padded = pad_to_4_bytes(indices_buffer)
    uvs_buffer_padded = pad_to_4_bytes(uvs_buffer)
    colors_buffer_padded = pad_to_4_bytes(colors_buffer)
    texture_buffer_padded = pad_to_4_bytes(texture_buffer)

    buffer_data = b"".join([
        vertices_buffer_padded,
        indices_buffer_padded,
        uvs_buffer_padded,
        colors_buffer_padded,
        texture_buffer_padded,
    ])

    vertices_byte_length = len(vertices_buffer)
    vertices_byte_offset = 0
    indices_byte_length = len(indices_buffer)
    indices_byte_offset = len(vertices_buffer_padded)
    uvs_byte_offset = indices_byte_offset + len(indices_buffer_padded)
    colors_byte_offset = uvs_byte_offset + len(uvs_buffer_padded)
    texture_byte_offset = colors_byte_offset + len(colors_buffer_padded)

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
    if unlit and texture_png_bytes is None:
        # Flat, light-independent shading (KHR_materials_unlit): COLOR_0 is shown as-is, matching how a
        # gaussian splat renders (emissive). Without this the viewer lights the mesh and washes the colours.
        materials.append({
            "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0},
            "extensions": {"KHR_materials_unlit": {}},
            "doubleSided": True,
        })
        extensions_used.append("KHR_materials_unlit")
        primitive["material"] = 0
    if texture_png_bytes is not None and "TEXCOORD_0" in primitive_attributes:
        buffer_views.append({
            "buffer": 0,
            "byteOffset": texture_byte_offset,
            "byteLength": len(texture_buffer),
        })
        images.append({"bufferView": len(buffer_views) - 1, "mimeType": "image/png"})
        samplers.append({"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071})
        textures.append({"source": 0, "sampler": 0})
        materials.append({
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0, "texCoord": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "doubleSided": True,
        })
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

    # Write the GLB file
    with open(filepath, 'wb') as f:
        f.write(glb_header)
        f.write(json_chunk_header)
        f.write(gltf_json_padded)
        f.write(bin_chunk_header)
        f.write(buffer_data)

    return filepath


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
            texture_b = getattr(mesh, "texture", None)
            texture_np = None
            if texture_b is not None:
                texture_np = (texture_b.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
                assert texture_np.ndim == 4 and texture_np.shape[-1] == 3, (
                    f"texture must be (B, H, W, 3) RGB, got shape {tuple(texture_np.shape)}"
                )
            for i in range(mesh.vertices.shape[0]):
                vertices_i, faces_i, v_colors, uvs_i = get_mesh_batch_item(mesh, i)
                if vertices_i.shape[0] == 0 or faces_i.shape[0] == 0:
                    logging.warning(f"SaveGLB: skipping empty mesh at batch index {i}")
                    continue
                tex_img = Image.fromarray(texture_np[i], mode="RGB") if texture_np is not None else None
                f = f"{filename}_{counter:05}_.glb"
                save_glb(vertices_i, faces_i, os.path.join(full_output_folder, f), metadata,
                         uvs=uvs_i,
                         vertex_colors=v_colors,
                         texture_image=tex_img,
                         unlit=getattr(mesh, "unlit", False))
                results.append({
                    "filename": f,
                    "subfolder": subfolder,
                    "type": "output"
                })
                counter += 1
        return IO.NodeOutput(ui={"3d": results})


class Save3DExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [SaveGLB]


async def comfy_entrypoint() -> Save3DExtension:
    return Save3DExtension()
