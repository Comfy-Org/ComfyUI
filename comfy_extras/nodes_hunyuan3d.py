import torch
from comfy.ldm.modules.diffusionmodules.mmdit import get_1d_sincos_pos_embed_from_grid_torch
import comfy.model_management
from comfy_extras.nodes_save_3d import pack_variable_mesh_batch
from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO, Types
from comfy_api.latest._util import MESH, VOXEL  # only for backward compatibility if someone import it from this file (will be removed later) # noqa


class EmptyLatentHunyuan3Dv2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyLatentHunyuan3Dv2",
            category="model/latent/hunyuan 3d",
            inputs=[
                IO.Int.Input("resolution", default=3072, min=1, max=8192),
                IO.Int.Input("batch_size", default=1, min=1, max=4096, tooltip="The number of latent images in the batch."),
            ],
            outputs=[
                IO.Latent.Output(),
            ]
        )

    @classmethod
    def execute(cls, resolution, batch_size) -> IO.NodeOutput:
        latent = torch.zeros([batch_size, 64, resolution], device=comfy.model_management.intermediate_device())
        return IO.NodeOutput({"samples": latent, "type": "hunyuan3dv2"})

    generate = execute  # TODO: remove


class Hunyuan3Dv2Conditioning(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Hunyuan3Dv2Conditioning",
            category="model/conditioning/hunyuan 3d",
            inputs=[
                IO.ClipVisionOutput.Input("clip_vision_output"),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
            ]
        )

    @classmethod
    def execute(cls, clip_vision_output) -> IO.NodeOutput:
        embeds = clip_vision_output.last_hidden_state
        positive = [[embeds, {}]]
        negative = [[torch.zeros_like(embeds), {}]]
        return IO.NodeOutput(positive, negative)

    encode = execute  # TODO: remove


class Hunyuan3Dv2ConditioningMultiView(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Hunyuan3Dv2ConditioningMultiView",
            category="model/conditioning/hunyuan 3d",
            inputs=[
                IO.ClipVisionOutput.Input("front", optional=True),
                IO.ClipVisionOutput.Input("left", optional=True),
                IO.ClipVisionOutput.Input("back", optional=True),
                IO.ClipVisionOutput.Input("right", optional=True),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
            ]
        )

    @classmethod
    def execute(cls, front=None, left=None, back=None, right=None) -> IO.NodeOutput:
        all_embeds = [front, left, back, right]
        out = []
        pos_embeds = None
        for i, e in enumerate(all_embeds):
            if e is not None:
                if pos_embeds is None:
                    pos_embeds = get_1d_sincos_pos_embed_from_grid_torch(e.last_hidden_state.shape[-1], torch.arange(4))
                out.append(e.last_hidden_state + pos_embeds[i].reshape(1, 1, -1))

        embeds = torch.cat(out, dim=1)
        positive = [[embeds, {}]]
        negative = [[torch.zeros_like(embeds), {}]]
        return IO.NodeOutput(positive, negative)

    encode = execute  # TODO: remove


class VAEDecodeHunyuan3D(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VAEDecodeHunyuan3D",
            category="model/latent/hunyuan 3d",
            inputs=[
                IO.Latent.Input("samples"),
                IO.Vae.Input("vae"),
                IO.Int.Input("num_chunks", default=8000, min=1000, max=500000, advanced=True),
                IO.Int.Input("octree_resolution", default=256, min=16, max=512, advanced=True),
            ],
            outputs=[
                IO.Voxel.Output(),
            ]
        )

    @classmethod
    def execute(cls, vae, samples, num_chunks, octree_resolution) -> IO.NodeOutput:
        voxels = Types.VOXEL(vae.decode(samples["samples"], vae_options={"num_chunks": num_chunks, "octree_resolution": octree_resolution}))
        return IO.NodeOutput(voxels)

    decode = execute  # TODO: remove


def voxel_to_mesh(voxels, threshold=0.5, device=None):
    if device is None:
        device = torch.device("cpu")
    voxels = voxels.to(device)

    binary = (voxels > threshold).float()
    padded = torch.nn.functional.pad(binary, (1, 1, 1, 1, 1, 1), 'constant', 0)

    D, H, W = binary.shape

    neighbors = torch.tensor([
        [0, 0, 1],
        [0, 0, -1],
        [0, 1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [-1, 0, 0]
    ], device=device)

    z, y, x = torch.meshgrid(
        torch.arange(D, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    voxel_indices = torch.stack([z.flatten(), y.flatten(), x.flatten()], dim=1)

    solid_mask = binary.flatten() > 0
    solid_indices = voxel_indices[solid_mask]

    corner_offsets = [
        torch.tensor([
            [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]
        ], device=device),
        torch.tensor([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], device=device),
        torch.tensor([
            [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]
        ], device=device),
        torch.tensor([
            [0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]
        ], device=device),
        torch.tensor([
            [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]
        ], device=device),
        torch.tensor([
            [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0]
        ], device=device)
    ]

    all_vertices = []
    all_indices = []

    vertex_count = 0

    for face_idx, offset in enumerate(neighbors):
        neighbor_indices = solid_indices + offset

        padded_indices = neighbor_indices + 1

        is_exposed = padded[
            padded_indices[:, 0],
            padded_indices[:, 1],
            padded_indices[:, 2]
        ] == 0

        if not is_exposed.any():
            continue

        exposed_indices = solid_indices[is_exposed]

        corners = corner_offsets[face_idx].unsqueeze(0)

        face_vertices = exposed_indices.unsqueeze(1) + corners

        all_vertices.append(face_vertices.reshape(-1, 3))

        num_faces = exposed_indices.shape[0]
        face_indices = torch.arange(
            vertex_count,
            vertex_count + 4 * num_faces,
            device=device
        ).reshape(-1, 4)

        all_indices.append(torch.stack([face_indices[:, 0], face_indices[:, 1], face_indices[:, 2]], dim=1))
        all_indices.append(torch.stack([face_indices[:, 0], face_indices[:, 2], face_indices[:, 3]], dim=1))

        vertex_count += 4 * num_faces

    if len(all_vertices) > 0:
        vertices = torch.cat(all_vertices, dim=0)
        faces = torch.cat(all_indices, dim=0)
    else:
        vertices = torch.zeros((1, 3))
        faces = torch.zeros((1, 3))

    v_min = 0
    v_max = max(voxels.shape)

    vertices = vertices - (v_min + v_max) / 2

    scale = (v_max - v_min) / 2
    if scale > 0:
        vertices = vertices / scale

    vertices = torch.fliplr(vertices)
    return vertices, faces

def voxel_to_mesh_surfnet(voxels, threshold=0.5, device=None):
    if device is None:
        device = torch.device("cpu")
    voxels = voxels.to(device)

    D, H, W = voxels.shape

    padded = torch.nn.functional.pad(voxels, (1, 1, 1, 1, 1, 1), 'constant', 0)
    z, y, x = torch.meshgrid(
        torch.arange(D, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    cell_positions = torch.stack([z.flatten(), y.flatten(), x.flatten()], dim=1)

    corner_offsets = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], device=device)

    pos = cell_positions.unsqueeze(1) + corner_offsets.unsqueeze(0)
    z_idx, y_idx, x_idx = pos.unbind(-1)
    corner_values = padded[z_idx, y_idx, x_idx]

    corner_signs = corner_values > threshold
    has_inside = torch.any(corner_signs, dim=1)
    has_outside = torch.any(~corner_signs, dim=1)
    contains_surface = has_inside & has_outside

    active_cells = cell_positions[contains_surface]
    active_signs = corner_signs[contains_surface]
    active_values = corner_values[contains_surface]

    if active_cells.shape[0] == 0:
        return torch.zeros((0, 3), device=device), torch.zeros((0, 3), dtype=torch.long, device=device)

    edges = torch.tensor([
        [0, 1], [0, 2], [0, 4], [1, 3],
        [1, 5], [2, 3], [2, 6], [3, 7],
        [4, 5], [4, 6], [5, 7], [6, 7]
    ], device=device)

    cell_vertices = {}
    progress = comfy.utils.ProgressBar(100)

    for edge_idx, (e1, e2) in enumerate(edges):
        progress.update(1)
        crossing = active_signs[:, e1] != active_signs[:, e2]
        if not crossing.any():
            continue

        cell_indices = torch.nonzero(crossing, as_tuple=True)[0]

        v1 = active_values[cell_indices, e1]
        v2 = active_values[cell_indices, e2]

        t = torch.zeros_like(v1, device=device)
        denom = v2 - v1
        valid = denom != 0
        t[valid] = (threshold - v1[valid]) / denom[valid]
        t[~valid] = 0.5

        p1 = corner_offsets[e1].float()
        p2 = corner_offsets[e2].float()

        intersection = p1.unsqueeze(0) + t.unsqueeze(1) * (p2.unsqueeze(0) - p1.unsqueeze(0))

        for i, point in zip(cell_indices.tolist(), intersection):
            if i not in cell_vertices:
                cell_vertices[i] = []
            cell_vertices[i].append(point)

    # Calculate the final vertices as the average of intersection points for each cell
    vertices = []
    vertex_lookup = {}

    vert_progress_mod = round(len(cell_vertices)/50)

    for i, points in cell_vertices.items():
        if not i % vert_progress_mod:
            progress.update(1)

        if points:
            vertex = torch.stack(points).mean(dim=0)
            vertex = vertex + active_cells[i].float()
            vertex_lookup[tuple(active_cells[i].tolist())] = len(vertices)
            vertices.append(vertex)

    if not vertices:
        return torch.zeros((0, 3), device=device), torch.zeros((0, 3), dtype=torch.long, device=device)

    final_vertices = torch.stack(vertices)

    inside_corners_mask = active_signs
    outside_corners_mask = ~active_signs

    inside_counts = inside_corners_mask.sum(dim=1, keepdim=True).float()
    outside_counts = outside_corners_mask.sum(dim=1, keepdim=True).float()

    inside_pos = torch.zeros((active_cells.shape[0], 3), device=device)
    outside_pos = torch.zeros((active_cells.shape[0], 3), device=device)

    for i in range(8):
        mask_inside = inside_corners_mask[:, i].unsqueeze(1)
        mask_outside = outside_corners_mask[:, i].unsqueeze(1)
        inside_pos += corner_offsets[i].float().unsqueeze(0) * mask_inside
        outside_pos += corner_offsets[i].float().unsqueeze(0) * mask_outside

    inside_pos /= inside_counts
    outside_pos /= outside_counts
    gradients = inside_pos - outside_pos

    pos_dirs = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], device=device)

    cross_products = [
        torch.linalg.cross(pos_dirs[i].float(), pos_dirs[j].float())
        for i in range(3) for j in range(i+1, 3)
    ]

    faces = []
    all_keys = set(vertex_lookup.keys())

    face_progress_mod = round(len(active_cells)/38*3)

    for pair_idx, (i, j) in enumerate([(0,1), (0,2), (1,2)]):
        dir_i = pos_dirs[i]
        dir_j = pos_dirs[j]
        cross_product = cross_products[pair_idx]

        ni_positions = active_cells + dir_i
        nj_positions = active_cells + dir_j
        diag_positions = active_cells + dir_i + dir_j

        alignments = torch.matmul(gradients, cross_product)

        valid_quads = []
        quad_indices = []

        for idx, active_cell in enumerate(active_cells):
            if not idx % face_progress_mod:
                progress.update(1)
            cell_key = tuple(active_cell.tolist())
            ni_key = tuple(ni_positions[idx].tolist())
            nj_key = tuple(nj_positions[idx].tolist())
            diag_key = tuple(diag_positions[idx].tolist())

            if cell_key in all_keys and ni_key in all_keys and nj_key in all_keys and diag_key in all_keys:
                v0 = vertex_lookup[cell_key]
                v1 = vertex_lookup[ni_key]
                v2 = vertex_lookup[nj_key]
                v3 = vertex_lookup[diag_key]

                valid_quads.append((v0, v1, v2, v3))
                quad_indices.append(idx)

        for q_idx, (v0, v1, v2, v3) in enumerate(valid_quads):
            cell_idx = quad_indices[q_idx]
            if alignments[cell_idx] > 0:
                faces.append(torch.tensor([v0, v1, v3], device=device, dtype=torch.long))
                faces.append(torch.tensor([v0, v3, v2], device=device, dtype=torch.long))
            else:
                faces.append(torch.tensor([v0, v3, v1], device=device, dtype=torch.long))
                faces.append(torch.tensor([v0, v2, v3], device=device, dtype=torch.long))

    if faces:
        faces = torch.stack(faces)
    else:
        faces = torch.zeros((0, 3), dtype=torch.long, device=device)

    v_min = 0
    v_max = max(D, H, W)

    final_vertices = final_vertices - (v_min + v_max) / 2

    scale = (v_max - v_min) / 2
    if scale > 0:
        final_vertices = final_vertices / scale

    final_vertices = torch.fliplr(final_vertices)

    return final_vertices, faces


class VoxelToMeshBasic(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VoxelToMeshBasic",
            display_name="Voxel to Mesh (Basic) (DEPRECATED)",
            category="3d",
            description="Converts a voxel grid to a mesh.",
            is_deprecated=True, # This node is superseded by the Voxel To Mesh node
            inputs=[
                IO.Voxel.Input("voxel"),
                IO.Float.Input("threshold", default=0.6, min=-1.0, max=1.0, step=0.01),
            ],
            outputs=[
                IO.Mesh.Output(),
            ],
        )

    @classmethod
    def execute(cls, voxel, threshold) -> IO.NodeOutput:
        vertices = []
        faces = []
        for x in voxel.data:
            v, f = voxel_to_mesh(x, threshold=threshold, device=None)
            vertices.append(v)
            faces.append(f)

        if vertices and all(v.shape == vertices[0].shape for v in vertices) and all(f.shape == faces[0].shape for f in faces):
            return IO.NodeOutput(Types.MESH(torch.stack(vertices), torch.stack(faces)))
        return IO.NodeOutput(pack_variable_mesh_batch(vertices, faces))

    decode = execute  # TODO: remove


class VoxelToMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VoxelToMesh",
            display_name="Voxel to Mesh",
            category="3d",
            description="Converts a voxel grid to a mesh.",
            inputs=[
                IO.Voxel.Input("voxel"),
                IO.Combo.Input("algorithm", options=["surface net", "basic"]),
                IO.Float.Input("threshold", default=0.6, min=-1.0, max=1.0, step=0.01),
            ],
            outputs=[
                IO.Mesh.Output(),
            ]
        )

    @classmethod
    def execute(cls, voxel, algorithm, threshold) -> IO.NodeOutput:
        vertices = []
        faces = []

        if algorithm == "basic":
            mesh_function = voxel_to_mesh
        elif algorithm == "surface net":
            mesh_function = voxel_to_mesh_surfnet

        for x in voxel.data:
            v, f = mesh_function(x, threshold=threshold, device=None)
            vertices.append(v)
            faces.append(f)

        if vertices and all(v.shape == vertices[0].shape for v in vertices) and all(f.shape == faces[0].shape for f in faces):
            return IO.NodeOutput(Types.MESH(torch.stack(vertices), torch.stack(faces)))
        return IO.NodeOutput(pack_variable_mesh_batch(vertices, faces))

    decode = execute  # TODO: remove


class Hunyuan3dExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            EmptyLatentHunyuan3Dv2,
            Hunyuan3Dv2Conditioning,
            Hunyuan3Dv2ConditioningMultiView,
            VAEDecodeHunyuan3D,
            VoxelToMeshBasic,
            VoxelToMesh,
        ]


async def comfy_entrypoint() -> Hunyuan3dExtension:
    return Hunyuan3dExtension()
