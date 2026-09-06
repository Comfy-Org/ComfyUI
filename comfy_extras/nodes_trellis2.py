from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO, Types, io
from comfy.ldm.trellis2.vae import SparseTensor
from comfy.ldm.trellis2.model import build_proj_transform_matrix, compute_stage_proj_feats

from comfy_extras.nodes_mesh_postprocess import pack_variable_mesh_batch
import comfy.latent_formats
import comfy.model_management
import comfy.utils
import math
import torch

ShapeSubdivides = io.Custom("SHAPE_SUBDIVIDES")


shape_slat_format = comfy.latent_formats.Trellis2ShapeSLAT()
tex_slat_format = comfy.latent_formats.Trellis2TexSLAT()

def shape_norm(shape_latent, coords):
    feats = shape_slat_format.process_out(shape_latent)
    return SparseTensor(feats=feats, coords=coords)


def _move_sparse_tensor_uncached(tensor, device):
    return SparseTensor(
        feats=tensor.feats.to(device),
        coords=tensor.coords.to(device),
        shape=tensor.shape,
        scale=tensor._scale,
    )


def _sparse_vae_decode_memory(point_count, dtype):
    # Last 128-channel stage: feature intermediates plus 27 int32 neighbor indices,
    # plus sparse-convolution workspace.
    bytes_per_point = 896 * comfy.model_management.dtype_size(dtype) + 27 * 4
    return 2 * 1024 ** 3 + int(point_count) * bytes_per_point


def infer_batched_coord_layout(coords):
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"Expected Trellis2 coords with shape [N, 4], got {tuple(coords.shape)}")

    if coords.shape[0] == 0:
        raise ValueError("Trellis2 coords can't be empty")

    batch_ids = coords[:, 0].to(torch.int64)
    if (batch_ids < 0).any():
        raise ValueError(f"Trellis2 batch ids must be non-negative, got {batch_ids.unique(sorted=True).tolist()}")
    batch_size = int(batch_ids.max().item()) + 1
    counts = torch.bincount(batch_ids, minlength=batch_size)

    if (counts == 0).any():
        raise ValueError(f"Non-contiguous Trellis2 batch ids in coords: {batch_ids.unique(sorted=True).tolist()}")

    max_tokens = int(counts.max().item())
    return batch_size, counts, max_tokens


def split_batched_coords(coords, coord_counts):
    if coord_counts.ndim != 1:
        raise ValueError(f"Trellis2 coord_counts must be 1D, got shape {tuple(coord_counts.shape)}")
    counts = [int(count) for count in coord_counts.tolist()]
    if any(count < 0 for count in counts):
        raise ValueError(f"Trellis2 coord_counts must be non-negative, got {counts}")
    if sum(counts) != coords.shape[0]:
        raise ValueError(
            f"Trellis2 coord_counts total {sum(counts)} does not match coords rows {coords.shape[0]}"
        )

    batch_ids = coords[:, 0].to(torch.int64)
    order = torch.argsort(batch_ids, stable=True)
    sorted_coords = coords.index_select(0, order)
    sorted_batch_ids = batch_ids.index_select(0, order)

    items = []
    start = 0
    for i, count in enumerate(counts):
        coords_i = sorted_coords[start:start + count]
        ids_i = sorted_batch_ids[start:start + count]
        if coords_i.shape[0] != count or not torch.all(ids_i == i):
            raise ValueError(f"Trellis2 coords rows for batch {i} expected {count}, got {coords_i.shape[0]}")
        items.append(coords_i)
        start += count
    return items

def flatten_batched_sparse_latent(samples, coords, coord_counts):
    samples = samples.squeeze(-1).transpose(1, 2)
    if coord_counts is None:
        return samples.reshape(-1, samples.shape[-1]), coords

    coords_items = split_batched_coords(coords, coord_counts)
    feat_list = []
    coord_list = []
    for i, coords_i in enumerate(coords_items):
        count = coords_i.shape[0]
        feat_list.append(samples[i, :count])
        coord_list.append(coords_i)

    return torch.cat(feat_list, dim=0), torch.cat(coord_list, dim=0)


def split_batched_sparse_latent(samples, coords, coord_counts):
    samples = samples.squeeze(-1).transpose(1, 2)
    if coord_counts is None:
        return [(samples.reshape(-1, samples.shape[-1]), coords)]

    coords_items = split_batched_coords(coords, coord_counts)
    items = []
    for i, coords_i in enumerate(coords_items):
        count = coords_i.shape[0]
        items.append((samples[i, :count], coords_i))
    return items

class VaeDecodeShapeTrellis(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VaeDecodeShapeTrellis",
            category="model/latent/trellis",
            inputs=[
                IO.Latent.Input("samples"),
                IO.Vae.Input("vae"),
            ],
            outputs=[
                IO.Mesh.Output("mesh"),
                ShapeSubdivides.Output(display_name = "shape_subdivides"),
            ]
        )

    @classmethod
    def execute(cls, samples, vae):
        # Mesh grid_size must match the actual coord resolution the upstream
        # stage was run at (1024 cascade -> 64, 1536 cascade -> 96). The VAE's
        # built-in `.resolution` buffer defaults to 1024 and is otherwise stale;
        # take coord_resolution from the latent dict if the stage node set it.
        coord_resolution = samples.get("coord_resolution")
        if coord_resolution is not None:
            resolution = int(coord_resolution) * 16
        else:
            resolution = int(vae.first_stage_model.resolution.item())
        model_frame = samples.get("model_frame", "y_up")
        sample_tensor = samples["samples"]
        device = comfy.model_management.get_torch_device()
        coords = samples["coords"]
        surface_point_estimate = resolution * resolution * 5 // 4
        vae.prepare_decode(
            sample_tensor.shape,
            memory_required=_sparse_vae_decode_memory(surface_point_estimate, vae.vae_dtype),
        )
        trellis_vae = vae.first_stage_model
        coord_counts = samples.get("coord_counts")

        samples = samples["samples"]
        if coord_counts is None:
            samples, coords = flatten_batched_sparse_latent(samples, coords, coord_counts)
            samples = shape_norm(samples.to(device), coords.to(device))
            mesh, subs = trellis_vae.decode_shape_slat(samples.to(vae.vae_dtype), resolution)
        else:
            split_items = split_batched_sparse_latent(samples, coords, coord_counts)
            mesh = []
            subs_per_sample = []
            for feats_i, coords_i in split_items:
                coords_i = coords_i.to(device).clone()
                coords_i[:, 0] = 0
                sample_i = shape_norm(feats_i.to(device), coords_i)
                mesh_i, subs_i = trellis_vae.decode_shape_slat(sample_i.to(vae.vae_dtype), resolution)
                mesh.append(mesh_i[0])
                subs_per_sample.append(subs_i)

            subs = []
            for stage_index in range(len(subs_per_sample[0])):
                stage_tensors = [sample_subs[stage_index] for sample_subs in subs_per_sample]
                feats_list = [stage_tensor.feats for stage_tensor in stage_tensors]
                coords_list = [stage_tensor.coords for stage_tensor in stage_tensors]
                subs.append(SparseTensor.from_tensor_list(feats_list, coords_list))

        # Rotate Z-up (Trellis2 training frame) vertices to glTF Y-up. Pixal3D outputs are already Y-up.
        if model_frame == "z_up":
            vert_list = [torch.stack([v[..., 0], v[..., 2], -v[..., 1]], dim=-1).float().cpu()
                         for v, _ in mesh]
        else:
            vert_list = [v.float().cpu() for v, _ in mesh]
        face_list = [f.int().cpu() for _, f in mesh]
        if all(v.shape == vert_list[0].shape for v in vert_list) and all(f.shape == face_list[0].shape for f in face_list):
            mesh = Types.MESH(vertices=torch.stack(vert_list), faces=torch.stack(face_list))
        else:
            mesh = pack_variable_mesh_batch(vert_list, face_list)
        output_device = comfy.model_management.intermediate_device()
        subs = [_move_sparse_tensor_uncached(sub, output_device) for sub in subs]
        return IO.NodeOutput(mesh, subs)

class VaeDecodeTextureTrellis(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VaeDecodeTextureTrellis",
            category="model/latent/trellis",
            inputs=[
                IO.Latent.Input("samples"),
                IO.Vae.Input("vae"),
                ShapeSubdivides.Input("shape_subdivides",
                                 tooltip=(
                                    "Shape information used to guide higher-detail reconstruction during decoding. "
                                    "Helps preserve structure consistency at higher resolutions."
                                 )),
            ],
            outputs=[
                IO.Voxel.Output("voxel_colors"),
            ]
        )

    @classmethod
    def execute(cls, samples, vae, shape_subdivides):
        sample_tensor = samples["samples"]
        device = comfy.model_management.get_torch_device()
        coords = samples["coords"]
        vae.prepare_decode(
            sample_tensor.shape,
            memory_required=_sparse_vae_decode_memory(shape_subdivides[-1].feats.shape[0], vae.vae_dtype),
        )
        trellis_vae = vae.first_stage_model
        coord_counts = samples.get("coord_counts")
        model_frame = samples.get("model_frame", "y_up")
        coord_resolution = samples.get("coord_resolution")

        samples = samples["samples"]
        samples, coords = flatten_batched_sparse_latent(samples, coords, coord_counts)
        samples = samples.to(device)
        feats = tex_slat_format.process_out(samples)
        samples = SparseTensor(feats=feats, coords=coords.to(device))
        shape_subdivides = [_move_sparse_tensor_uncached(sub, device) for sub in shape_subdivides]

        voxel = trellis_vae.decode_tex_slat(samples.to(vae.vae_dtype), shape_subdivides)
        # Keep all decoded channels. The texture VAE emits 6: base_color (0:3),
        # metallic (3), roughness (4), alpha (5) — all in [0, 1]. Vertex-color
        # consumers (PaintMesh) slice [:3]
        color_feats = voxel.feats
        voxel_coords = voxel.coords

        if coord_resolution is not None:
            tex_resolution = int(coord_resolution) * 16
        elif voxel_coords.numel() > 0 and voxel_coords.shape[-1] >= 3:
            spatial = voxel_coords[:, -3:] if voxel_coords.shape[-1] == 4 else voxel_coords
            max_idx = int(spatial.max().item()) + 1
            tex_resolution = next((r for r in (256, 512, 1024, 1536, 2048) if r >= max_idx), max_idx)
        else:
            tex_resolution = 1024

        # Remap Z-up voxel coords to Y-up: (x, y, z) -> (x, z, R-1-y), matching the
        # R_x(-90°) applied to mesh vertices in VaeDecodeShapeTrellis. Keeps PaintMesh's
        # NN lookup correctly aligned without it needing to know the source frame.
        if model_frame == "z_up" and voxel_coords.numel() > 0 and voxel_coords.shape[-1] >= 3:
            R = tex_resolution
            if voxel_coords.shape[-1] == 4:
                batch_col = voxel_coords[:, :1]
                spatial = voxel_coords[:, 1:]
                spatial_yup = torch.stack(
                    [spatial[:, 0], spatial[:, 2], (R - 1) - spatial[:, 1]], dim=-1
                )
                voxel_coords = torch.cat([batch_col, spatial_yup], dim=-1)
            else:
                voxel_coords = torch.stack(
                    [voxel_coords[:, 0], voxel_coords[:, 2], (R - 1) - voxel_coords[:, 1]],
                    dim=-1,
                )

        output_device = comfy.model_management.intermediate_device()
        voxel_coords = voxel_coords.to(output_device)
        color_feats = color_feats.to(output_device)
        voxel = Types.VOXEL(voxel_coords, color_feats, tex_resolution)
        return IO.NodeOutput(voxel)

class VaeDecodeStructureTrellis2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VaeDecodeStructureTrellis2",
            category="model/latent/trellis",
            inputs=[
                IO.Latent.Input("samples"),
                IO.Vae.Input("vae"),
                IO.Combo.Input("resolution", options=["32", "64"], default="32"),
            ],
            outputs=[
                IO.Voxel.Output("voxel"),
            ]
        )

    @classmethod
    def execute(cls, samples, vae, resolution):
        resolution = int(resolution)
        sample_tensor = samples["samples"]
        sample_tensor = sample_tensor[:, :8]
        batch_number = vae.prepare_decode(sample_tensor.shape)
        shape_vae = vae.first_stage_model
        load_device = comfy.model_management.get_torch_device()
        decoded_batches = []
        for start in range(0, sample_tensor.shape[0], batch_number):
            sample_chunk = sample_tensor[start:start + batch_number].to(load_device)
            decoded_batches.append(shape_vae.decode_structure(sample_chunk.to(vae.vae_dtype)) > 0)
        decoded = torch.cat(decoded_batches, dim=0)
        current_res = decoded.shape[2]

        if current_res != resolution:
            ratio = current_res // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        voxel_data = decoded.squeeze(1).float()
        return IO.NodeOutput(Types.VOXEL(voxel_data))

class Trellis2UpsampleStage(IO.ComfyNode):
    """Cascade-upsamples a 512-resolution shape latent into high-resolution
    sparse coords and sets up the second shape-stage sampling pass at the
    target resolution, attaching per-stage metadata to the conditioning for
    the model to consume via extra_conds."""
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2UpsampleStage",
            category="model/conditioning/trellis2",
            display_name="Trellis2 Upsample Stage",
            inputs=[
                IO.Conditioning.Input("positive"),
                IO.Conditioning.Input("negative"),
                IO.Latent.Input("shape_latent", tooltip="The 512-resolution shape latent output from the first shape-stage KSampler."),
                IO.Vae.Input("vae"),
                IO.Int.Input("target_resolution", default=1024, min=1024, max=2048, step=128,
                             tooltip="Voxel resolution of the upsampled shape. Higher = more detail, more VRAM."),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
                IO.Latent.Output(),
            ]
        )

    @staticmethod
    def _quantize_unique(hr_coords: torch.Tensor, lr_resolution: int, hr_resolution: int, pixal3d_mode: bool = False) -> torch.Tensor:
        # Trellis2 uses `floor((c+0.5) * grid_res / lr_res)
        # Pixal3D uses `round((c+0.5) * (grid_res-1) / lr_res)`
        # this is a half-cell spatial shift. Branch so each upstream is matched bit-for-bit.
        grid_res = hr_resolution // 16
        spatial = hr_coords[:, 1:].float()
        if pixal3d_mode:
            spatial.add_(0.5).mul_((grid_res - 1) / lr_resolution).round_()
        else:
            spatial.add_(0.5).mul_(grid_res / lr_resolution)
        quant = torch.cat([hr_coords[:, :1], spatial.int()], dim=1)
        return quant.unique(dim=0)

    @classmethod
    def execute(cls, positive, negative, shape_latent, vae, target_resolution):
        device = comfy.model_management.get_torch_device()
        vae.prepare_decode(shape_latent["samples"].shape)

        coord_counts = shape_latent.get("coord_counts")
        shape_vae = vae.first_stage_model
        lr_resolution = 512
        proj_pack = _proj_pack_from_conditioning(positive)
        pixal3d_mode = proj_pack is not None

        # Decode each sample's HR coords, then search for the largest hr_resolution
        # that fits under max_tokens across all samples.
        if coord_counts is None:
            feats, coords_512 = flatten_batched_sparse_latent(
                shape_latent["samples"], shape_latent["coords"], coord_counts,
            )
            slat = shape_norm(feats.to(device), coords_512.to(device))
            sample_hr_coords = [shape_vae.upsample_shape(slat.to(vae.vae_dtype), upsample_times=4)]
        else:
            items = split_batched_sparse_latent(
                shape_latent["samples"], shape_latent["coords"], coord_counts,
            )
            sample_hr_coords = []
            for feats_i, coords_i in items:
                coords_i = coords_i.to(device).clone()
                coords_i[:, 0] = 0
                slat_i = shape_norm(feats_i.to(device), coords_i)
                sample_hr_coords.append(shape_vae.upsample_shape(slat_i.to(vae.vae_dtype), upsample_times=4))

        hr_resolution = target_resolution
        quant_unique_list = [
            cls._quantize_unique(hr_coords_i, lr_resolution, hr_resolution, pixal3d_mode)
            for hr_coords_i in sample_hr_coords
        ]

        # Rewrite batch column to match per-sample offset and concat.
        per_sample_counts = []
        for sample_offset, qu in enumerate(quant_unique_list):
            qu[:, 0] = sample_offset
            per_sample_counts.append(int(qu.shape[0]))
        coords = torch.cat(quant_unique_list, dim=0)
        counts = torch.tensor(per_sample_counts, dtype=torch.int64)
        coord_resolution = hr_resolution // 16

        batch_size, _, max_tokens_out = infer_batched_coord_layout(coords)
        latent = torch.zeros(batch_size, 32, max_tokens_out, 1)

        extras = {
            "trellis2_generation_mode": "shape_generation",
            "trellis2_coords": coords,
            "trellis2_coord_counts": counts,
        }
        if proj_pack is not None:
            extras["trellis2_proj_feats"] = compute_stage_proj_feats(
                proj_pack, "shape_1024", coords=coords, coord_resolution=coord_resolution,
            )
        positive_out = _conditioning_set_extras(positive, extras)
        negative_out = _conditioning_set_extras(negative, extras)
        out_latent = {"samples": latent, "coords": coords, "coord_counts": counts,
                      "coord_resolution": coord_resolution, "type": "trellis2",
                      "model_frame": shape_latent.get("model_frame",
                                                       "y_up" if proj_pack is not None else "z_up")}
        return IO.NodeOutput(positive_out, negative_out, out_latent)

def _dinov3_encode(model, image_bchw, image_size, want_patches=False):
    """Run DINOv3 once at the requested resolution.

    image_bchw: [B, 3, H, W] float in [0, 1] (any source resolution; resized here).
    Returns the full sequence tensor (Trellis2 path) or a dict with the global
    tokens split out + a 2D patch grid (Pixal3D path) when `want_patches=True`.
    """
    model_internal = model.model
    device = comfy.model_management.get_torch_device()
    img_t = comfy.utils.common_upscale(image_bchw, image_size, image_size, "lanczos", "disabled").to(device)
    mean = torch.tensor(model.image_mean or [0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor(model.image_std or [0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img_t = (img_t - mean) / std
    tokens = model_internal(img_t, skip_norm_elementwise=True)[0]
    if not want_patches:
        return tokens
    h_p = w_p = image_size // 16
    n_reg = tokens.shape[1] - 1 - h_p * w_p
    return {"tokens": tokens[:, :1 + n_reg], "patches_2d": _dinov3_patches_to_2d(tokens, image_size)}


class Trellis2Conditioning(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2Conditioning",
            category="model/conditioning/trellis2",
            inputs=[
                IO.ClipVision.Input("clip_vision_model"),
                IO.Image.Input("image", tooltip="Preprocessed image from ImageCropToMask (pad_factor=1.0 for TRELLIS.2)."),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
            ]
        )

    @classmethod
    def execute(cls, clip_vision_model, image) -> IO.NodeOutput:
        out_device = comfy.model_management.intermediate_device()
        cond = _dino_encode_batch(clip_vision_model, image, out_device)
        cond_512_batched, cond_1024_batched = cond["global_512"], cond["global_1024"]
        neg_cond_batched = torch.zeros_like(cond_512_batched)
        neg_embeds_batched = torch.zeros_like(cond_1024_batched)

        positive = [[cond_512_batched, {"embeds": cond_1024_batched}]]
        negative = [[neg_cond_batched, {"embeds": neg_embeds_batched}]]
        return IO.NodeOutput(positive, negative)

def _proj_pack_from_conditioning(conditioning):
    """Return the proj_feat_pack dict embedded in a Pixal3D conditioning (or None
    for vanilla Trellis2 / no conditioning connected). Pixal3DConditioning ships
    the pack in cond[0][1]["proj_feat_pack"]; Trellis2Conditioning doesn't set it."""
    if not conditioning:
        return None
    entry = conditioning[0]
    if not isinstance(entry, (list, tuple)) or len(entry) < 2 or not isinstance(entry[1], dict):
        return None
    return entry[1].get("proj_feat_pack")


def _conditioning_set_extras(conditioning, extras: dict):
    """Return a copy of `conditioning` with `extras` merged into each entry's
    dict — same shallow-copy pattern ControlNetApplyAdvanced uses. The dicts
    are copied so we don't mutate upstream conditioning."""
    out = []
    for entry in conditioning:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
            new_dict = entry[1].copy()
            new_dict.update(extras)
            out.append([entry[0], new_dict])
        else:
            out.append(entry)
    return out


class Trellis2ShapeStage(IO.ComfyNode):
    """Sets up the first shape-stage sampling pass: extracts sparse coords from
    the dense structure voxel produced by VaeDecodeStructureTrellis2, builds an
    empty sparse latent, and attaches per-stage metadata to the conditioning so
    the model reads it via extra_conds at sample time. For the second shape pass
    (post-upsample), use Trellis2UpsampleStage instead — it combines the cascade
    and the second-pass stage setup."""
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2ShapeStage",
            category="model/conditioning/trellis2",
            inputs=[
                IO.Conditioning.Input("positive"),
                IO.Conditioning.Input("negative"),
                IO.Voxel.Input(
                    "voxel",
                    tooltip="Dense structure voxel from VaeDecodeStructureTrellis2.",
                ),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
                IO.Latent.Output(),
            ]
        )

    @classmethod
    def execute(cls, positive, negative, voxel):
        decoded = voxel.data.unsqueeze(1)
        coords = torch.argwhere(decoded.bool())[:, [0, 2, 3, 4]].int()
        coord_resolution = int(decoded.shape[-1])

        # Dispatch based on the upstream voxel resolution, mirroring upstream's
        # pipeline_type → ss_res table:
        #   coord_res == 32 → first cascade shape pass OR pure-512 pipeline
        #                     (img2shape_512 + shape_512 proj stage, 512 DINO).
        #   coord_res >  32 → pure-1024 non-cascade pipeline
        #                     (img2shape + shape_1024 proj stage, 1024 DINO).
        if coord_resolution <= 32:
            mode = "shape_generation_512"
            stage = "shape_512"
        else:
            mode = "shape_generation"
            stage = "shape_1024"

        batch_size, counts, max_tokens = infer_batched_coord_layout(coords)
        latent = torch.zeros(batch_size, 32, max_tokens, 1)

        extras = {
            "trellis2_generation_mode": mode,
            "trellis2_coords": coords,
            "trellis2_coord_counts": counts,
        }
        proj_pack = _proj_pack_from_conditioning(positive)
        if proj_pack is not None:
            extras["trellis2_proj_feats"] = compute_stage_proj_feats(
                proj_pack, stage, coords=coords, coord_resolution=coord_resolution,
            )
        positive_out = _conditioning_set_extras(positive, extras)
        negative_out = _conditioning_set_extras(negative, extras)
        out_latent = {"samples": latent, "coords": coords, "coord_counts": counts,
                      "coord_resolution": coord_resolution, "type": "trellis2",
                      "model_frame": "y_up" if proj_pack is not None else "z_up"}
        return IO.NodeOutput(positive_out, negative_out, out_latent)


class Trellis2TextureStage(IO.ComfyNode):
    """Sets up the texture-stage sampling pass. Reads coords / coord_counts /
    coord_resolution and the shape_slat (the per-voxel shape latent) from the
    incoming shape_latent dict — set there by Trellis2ShapeStage or
    Trellis2UpsampleStage. Builds an empty sparse latent at the same coord
    layout and attaches per-stage metadata to the conditioning."""
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2TextureStage",
            category="model/conditioning/trellis2",
            inputs=[
                IO.Conditioning.Input("positive"),
                IO.Conditioning.Input("negative"),
                IO.Latent.Input("shape_latent"),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
                IO.Latent.Output(),
            ]
        )

    @classmethod
    def execute(cls, positive, negative, shape_latent):
        channels = 32
        coords = shape_latent["coords"]
        coord_resolution = shape_latent.get("coord_resolution")

        batch_size, counts, max_tokens = infer_batched_coord_layout(coords)

        shape_slat = shape_latent["samples"]
        if shape_slat.ndim == 4:
            shape_slat, _ = flatten_batched_sparse_latent(shape_slat, coords, counts)

        latent = torch.zeros(batch_size, channels, max_tokens, 1)
        proj_pack = _proj_pack_from_conditioning(positive)
        model_frame = shape_latent.get("model_frame",
                                       "y_up" if proj_pack is not None else "z_up")
        extras = {
            "trellis2_generation_mode": "texture_generation",
            "trellis2_coords": coords,
            "trellis2_coord_counts": counts,
            "trellis2_shape_slat": shape_slat,
            "trellis2_model_frame": model_frame,
        }
        if proj_pack is not None and coord_resolution is not None:
            extras["trellis2_proj_feats"] = compute_stage_proj_feats(
                proj_pack, "tex_1024", coords=coords, coord_resolution=coord_resolution,
            )

        positive_out = _conditioning_set_extras(positive, extras)
        negative_out = _conditioning_set_extras(negative, extras)
        out_latent = {"samples": latent, "type": "trellis2", "coords": coords, "coord_counts": counts,
                      "model_frame": shape_latent.get("model_frame",
                                                       "y_up" if proj_pack is not None else "z_up")}
        if coord_resolution is not None:
            out_latent["coord_resolution"] = coord_resolution
        return IO.NodeOutput(positive_out, negative_out, out_latent)


class EmptyTrellis2LatentStructure(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyTrellis2LatentStructure",
            category="model/latent/trellis",
            inputs=[
                IO.Int.Input("batch_size", default=1, min=1, max=4096, tooltip="The number of latent images in the batch."),
            ],
            outputs=[
                IO.Latent.Output(),
            ]
        )
    @classmethod
    def execute(cls, batch_size):
        in_channels = 32
        resolution = 16
        latent = torch.zeros(batch_size, in_channels, resolution, resolution, resolution)
        return IO.NodeOutput({"samples": latent, "type": "trellis2"})

def _dinov3_patches_to_2d(tokens, image_size, patch_size=16):
    h_p = w_p = image_size // patch_size
    n_patches = h_p * w_p
    n_reg = tokens.shape[1] - 1 - n_patches
    if n_reg < 0 or tokens.shape[1] != 1 + n_reg + n_patches:
        raise ValueError(
            f"_dinov3_patches_to_2d: got {tokens.shape[1]} tokens, expected "
            f"1 (CLS) + N_reg + {h_p}*{w_p}={n_patches} patches at image_size={image_size}, "
            f"patch_size={patch_size}. Inferred N_reg={n_reg} which is invalid."
        )
    start = 1 + n_reg
    patches = tokens[:, start:start + n_patches]
    return patches.transpose(1, 2).reshape(tokens.shape[0], -1, h_p, w_p).contiguous()


def _dino_encode_batch(clip_vision_model, image, out_device, *, want_patches=False):
    """Encode an already-preprocessed image through DINOv3 at 512 and 1024.

    Expects `image` to be a comfy IMAGE tensor [B, H, W, 3] of squared composites
    (from ImageCropToMask). Returns batched global tokens; with want_patches also
    the 2D patch grids and the per-item BCHW composites that the Pixal3D NAF path needs."""
    image = image[..., :3]
    batch_size = image.shape[0]
    comfy.model_management.load_model_gpu(clip_vision_model.patcher)

    cond_512_list, cond_1024_list = [], []
    patches_512_list, patches_1024_list = [], []
    composite_list = []
    for b in range(batch_size):
        item = image[b].movedim(-1, -3).unsqueeze(0).contiguous().float().clamp(0, 1)
        c512 = _dinov3_encode(clip_vision_model, item, 512, want_patches=want_patches)
        c1024 = _dinov3_encode(clip_vision_model, item, 1024, want_patches=want_patches)
        if want_patches:
            cond_512_list.append(c512["tokens"].to(out_device))
            cond_1024_list.append(c1024["tokens"].to(out_device))
            patches_512_list.append(c512["patches_2d"].to(out_device))
            patches_1024_list.append(c1024["patches_2d"].to(out_device))
            composite_list.append(item)
        else:
            cond_512_list.append(c512.to(out_device))
            cond_1024_list.append(c1024.to(out_device))

    out = {
        "batch_size": batch_size,
        "global_512": torch.cat(cond_512_list, dim=0),
        "global_1024": torch.cat(cond_1024_list, dim=0),
    }
    if want_patches:
        out["patches_512"] = torch.cat(patches_512_list, dim=0)
        out["patches_1024"] = torch.cat(patches_1024_list, dim=0)
        out["composites"] = composite_list
    return out

class Pixal3DConditioning(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Pixal3DConditioning",
            category="model/conditioning/trellis2",
            inputs=[
                IO.ClipVision.Input("clip_vision_model", tooltip="DINOv3 ViT-L/16 ClipVision."),
                IO.Image.Input("image", tooltip="Preprocessed image from ImageCropToMask (pad_factor=1.1 for Pixal3D)."),
                IO.Float.Input(
                    "camera_angle_x", display_name="fov",
                    default=49.13, min=1.0, max=170.0, step=0.01,
                    tooltip="Horizontal FOV in degrees. Wire a MoGeGeometryToFOV "
                            "(axis='horizontal', unit='degrees') for a per-image FoV (matches upstream default).",
                ),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, clip_vision_model, image, camera_angle_x) -> IO.NodeOutput:
        naf_model = clip_vision_model.naf
        out_device = comfy.model_management.intermediate_device()
        compute_device = comfy.model_management.get_torch_device()

        cond = _dino_encode_batch(clip_vision_model, image, out_device, want_patches=True)
        batch_size = cond["batch_size"]
        global_512, global_1024 = cond["global_512"], cond["global_1024"]
        fm_512_dino, fm_1024_dino = cond["patches_512"], cond["patches_1024"]
        composite_list = cond["composites"]

        # The LR DINO grid AND the NAF HR grid are sampled separately
        # NAF targets per stage: shape_512=512, shape_1024=512, tex_1024=1024.
        def _naf_hr(lr_feat, composites, image_size, naf_target):
            if naf_model is None or naf_target is None:
                return None
            comfy.model_management.load_model_gpu(naf_model)
            inner = naf_model.model
            model_dtype = next(inner.parameters()).dtype  # set at load time (see clip_vision NAF)
            hrs = []
            for i, c in enumerate(composites):
                img_i = comfy.utils.common_upscale(c, image_size, image_size, "lanczos", "disabled")\
                    .to(compute_device).to(model_dtype)
                lr_i = lr_feat[i:i + 1].to(compute_device).to(model_dtype)
                output = torch.empty((1, lr_i.shape[1], *naf_target), device=out_device, dtype=model_dtype)
                hr_i = inner(img_i, lr_i, naf_target, output=output)
                hrs.append(hr_i)
            return torch.cat(hrs, dim=0)

        hr_shape_512  = _naf_hr(fm_512_dino,  composite_list, 512,  (512, 512))
        hr_shape_1024 = _naf_hr(fm_1024_dino, composite_list, 1024, (512, 512))
        hr_tex_1024   = _naf_hr(fm_1024_dino, composite_list, 1024, (1024, 1024))

        # distance_from_fov: grid_point (-1, 0, 0) projects to pixel (0, image_resolution-1).
        # FOV widget is in degrees for UX; trig + downstream projection expect radians.
        camera_angle_x = math.radians(float(camera_angle_x))
        distance = 0.5 / math.tan(camera_angle_x / 2.0)
        cam_angle_t = torch.tensor([camera_angle_x] * batch_size, device=out_device, dtype=torch.float32)
        dist_t = torch.tensor([distance] * batch_size, device=out_device, dtype=torch.float32)
        scale_t = torch.ones(batch_size, device=out_device, dtype=torch.float32)
        T = build_proj_transform_matrix(dist_t, batch_size, device=out_device, dtype=torch.float32)

        proj_pack = {
            "stages": {
                "ss":         {"feature_map": fm_512_dino,    "feature_map_hr": None,         "image_resolution": 512},
                "shape_512":  {"feature_map": fm_512_dino,    "feature_map_hr": hr_shape_512, "image_resolution": 512},
                "shape_1024": {"feature_map": fm_1024_dino,   "feature_map_hr": hr_shape_1024,"image_resolution": 1024},
                "tex_1024":   {"feature_map": fm_1024_dino,   "feature_map_hr": hr_tex_1024,  "image_resolution": 1024},
            },
            "transform_matrix": T,
            "camera_angle_x": cam_angle_t,
            "mesh_scale": scale_t,
            "distance": dist_t,
            "patch_size": 16,
        }

        # global_512 → SS/shape_512 cross-attn; global_1024 → shape_1024/tex_1024.
        ss_proj_feats = compute_stage_proj_feats(
            proj_pack, "ss", dense_grid_resolution=16, batch_size=batch_size,
            device=compute_device,
        )
        neg_global = torch.zeros_like(global_512)
        neg_embeds = torch.zeros_like(global_1024)
        base_extras = {
            "embeds": global_1024, "proj_feat_pack": proj_pack,
            "trellis2_proj_feats": ss_proj_feats,
        }
        neg_extras = {
            "embeds": neg_embeds, "proj_feat_pack": proj_pack,
            "trellis2_proj_feats": ss_proj_feats,
        }
        positive = [[global_512, base_extras]]
        negative = [[neg_global, neg_extras]]
        return IO.NodeOutput(positive, negative)


class Trellis2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            Trellis2Conditioning,
            Pixal3DConditioning,
            Trellis2ShapeStage,
            EmptyTrellis2LatentStructure,
            Trellis2TextureStage,
            VaeDecodeTextureTrellis,
            VaeDecodeShapeTrellis,
            VaeDecodeStructureTrellis2,
            Trellis2UpsampleStage,
        ]


async def comfy_entrypoint() -> Trellis2Extension:
    return Trellis2Extension()
