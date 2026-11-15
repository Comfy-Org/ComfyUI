"""SCAIL / SCAIL-2 nodes: the WanSCAILToVideo conditioning node and the SAM3
preprocessing that turns video tracks into the bundle the SCAIL-2 model consumes."""

from typing_extensions import override

import torch
import torch.nn.functional as F

import nodes
import node_helpers
import comfy.model_management
import comfy.utils
from comfy_api.latest import ComfyExtension, io
from comfy.ldm.sam3.tracker import unpack_masks

SAM3TrackData = io.Custom("SAM3_TRACK_DATA")


# Model was trained on these exact colors; deviating degrades multi-identity quality.
DEFAULT_PALETTE = [
    (0.0, 0.0, 1.0),  # Blue
    (1.0, 0.0, 0.0),  # Red
    (0.0, 1.0, 0.0),  # Green
    (1.0, 0.0, 1.0),  # Magenta
    (0.0, 1.0, 1.0),  # Cyan
    (1.0, 1.0, 0.0),  # Yellow
]


def _unpack(track_data):
    packed = track_data["packed_masks"]
    if packed is None or packed.shape[1] == 0:
        return None
    return unpack_masks(packed)


def _first_appearance_cx_area(masks_bool):
    """Per object: first frame it appears in, plus centroid-x and area in that frame."""
    m = masks_bool.float()
    T, H, W = m.shape[0], m.shape[-2], m.shape[-1]
    grid_x = torch.arange(W, device=m.device, dtype=m.dtype).view(1, 1, 1, W)
    area_t = m.sum(dim=(-1, -2))
    cx_t = (m * grid_x).sum(dim=(-1, -2)) / area_t.clamp(min=1)
    present = area_t > 0
    frame_idx = torch.arange(T, device=m.device).unsqueeze(1)
    first_t = torch.where(present, frame_idx, T).amin(dim=0)
    sel = first_t.clamp(max=T - 1).unsqueeze(0)
    cx = cx_t.gather(0, sel).squeeze(0)
    area = area_t.gather(0, sel).squeeze(0)
    return first_t.tolist(), (cx / W).tolist(), (area / (H * W)).tolist()


def _subset_track_data(track_data, obj_indices):
    out = dict(track_data)
    packed = track_data["packed_masks"]
    if packed is None or not obj_indices:
        out["packed_masks"] = None
        if "scores" in out:
            out["scores"] = []
        return out
    out["packed_masks"] = packed[:, obj_indices].contiguous()
    scores = track_data.get("scores")
    if scores is not None:
        out["scores"] = [scores[i] for i in obj_indices if i < len(scores)]
    return out


def _render_colored_masks(track_data, background="black"):
    packed = track_data["packed_masks"]
    H, W = track_data["orig_size"]
    device = comfy.model_management.intermediate_device()
    dtype = comfy.model_management.intermediate_dtype()
    bg_rgb = (1.0, 1.0, 1.0) if background.startswith("white") else (0.0, 0.0, 0.0)
    if packed is None or packed.shape[1] == 0:
        T = track_data.get("n_frames", 1) if packed is None else packed.shape[0]
        out = torch.empty(T, H, W, 3, device=device, dtype=dtype)
        out[..., 0], out[..., 1], out[..., 2] = bg_rgb[0], bg_rgb[1], bg_rgb[2]
        return out
    T, N_obj = packed.shape[0], packed.shape[1]
    colors = torch.tensor(
        [DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)] for i in range(N_obj)],
        device=device, dtype=dtype,
    )
    masks_full = unpack_masks(packed.to(device)).float()
    Hm, Wm = masks_full.shape[-2], masks_full.shape[-1]
    masks_full = F.interpolate(
        masks_full.view(T * N_obj, 1, Hm, Wm), size=(H, W), mode="nearest"
    ).view(T, N_obj, H, W) > 0.5
    any_mask = masks_full.any(dim=1)
    color_overlay = colors[masks_full.to(torch.uint8).argmax(dim=1)]
    bg_tensor = torch.tensor(bg_rgb, device=device, dtype=color_overlay.dtype).view(1, 1, 1, 3)
    return torch.where(any_mask.unsqueeze(-1), color_overlay, bg_tensor.expand_as(color_overlay))


def _render_mask_as_identity(mask, background="black"):
    """Plain comfy MASK (B,H,W) or (H,W) -> (B,H,W,3) rendered as a single identity (palette[0])
    on the given background. A batch is treated as multiple views of that one subject."""
    device = comfy.model_management.intermediate_device()
    dtype = comfy.model_management.intermediate_dtype()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    mask = mask.to(device=device, dtype=dtype)
    B, H, W = mask.shape
    bg_rgb = (1.0, 1.0, 1.0) if background.startswith("white") else (0.0, 0.0, 0.0)
    color = torch.tensor(DEFAULT_PALETTE[0], device=device, dtype=dtype).view(1, 1, 1, 3)
    bg = torch.tensor(bg_rgb, device=device, dtype=dtype).view(1, 1, 1, 3)
    return torch.where((mask > 0.5).unsqueeze(-1), color.expand(B, H, W, 3), bg.expand(B, H, W, 3))


def _extract_mask_to_28ch(rgb_video):
    """Colored RGB mask (T, H, W, 3) in [0, 1] -> SCAIL-2 28-channel binary latent
    (1, T_lat, 28, H_lat, W_lat). 7 per-color binary channels (white/r/g/b/y/m/c)
    threshold-extracted at 225/255, 8x spatial downsample, 4-frame temporal stacking."""
    T, H, W, _ = rgb_video.shape
    _ON_THRESH = 225.0 / 255.0
    mask = rgb_video.movedim(-1, 1).float()
    R = (mask[:, 0:1] > _ON_THRESH).float()
    G = (mask[:, 1:2] > _ON_THRESH).float()
    B = (mask[:, 2:3] > _ON_THRESH).float()
    nR, nG, nB = 1 - R, 1 - G, 1 - B
    binary_7ch = torch.cat([
        R * G * B,    # white
        R * nG * nB,  # red
        nR * G * nB,  # green
        nR * nG * B,  # blue
        R * G * nB,   # yellow
        R * nG * B,   # magenta
        nR * G * B,   # cyan
    ], dim=1)
    H_lat, W_lat = H, W
    for _ in range(3):
        H_lat = (H_lat + 1) // 2
        W_lat = (W_lat + 1) // 2
    binary_7ch = torch.nn.functional.interpolate(binary_7ch, size=(H_lat, W_lat), mode='area')
    T_latent = (T - 1) // 4 + 1
    padded = torch.cat([binary_7ch[:1].repeat(4, 1, 1, 1), binary_7ch[1:]], dim=0)
    out = padded.view(T_latent, 28, H_lat, W_lat)
    return out.unsqueeze(0)


class WanSCAILToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanSCAILToVideo",
            category="model/conditioning/wan/scail",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=512, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=896, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("pose_video", optional=True, tooltip="Video used for pose conditioning. Will be downscaled to half the resolution of the main video."),
                io.Image.Input("pose_video_mask", optional=True, tooltip="SCAIL-2 only. Colored per-identity SAM3 mask video at the same resolution as pose_video."),
                io.Boolean.Input("replacement_mode", default=False, optional=True, tooltip="SCAIL-2 only. False = Animation Mode (pose_video_mask should have black background). True = Replacement Mode (pose_video_mask should have white background)."),
                io.Float.Input("pose_strength", default=1.0, min=0.0, max=10.0, step=0.01, tooltip="Strength of the pose latent."),
                io.Float.Input("pose_start", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Start step of the pose conditioning."),
                io.Float.Input("pose_end", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="End step of the pose conditioning."),
                io.Image.Input("reference_image", optional=True, tooltip="Reference image. The first image is the primary reference (composite all identities onto it). SCAIL-2: extra batch images are used as additional views (back view, close-up, occluded background), each needing a matching reference_image_mask in that identity's color."),
                io.Image.Input("reference_image_mask", optional=True, tooltip="SCAIL-2 only. Colored reference mask, batch matching reference_image (first = primary reference mask, rest = identity masks for the additional reference_image)."),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True, tooltip="CLIP vision features for conditioning. Model is trained with stretch resize to aspect ratio."),
                io.Int.Input("video_frame_offset", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Cumulative output frame this chunk begins at. Wire from the previous chunk's video_frame_offset output."),
                io.Int.Input("previous_frame_count", default=5, min=1, max=nodes.MAX_RESOLUTION, step=4, tooltip="Tail frames of previous_frames to anchor. SCAIL-2 trained at 5 (81-frame chunks, 76-frame step)."),
                io.Image.Input("previous_frames", optional=True, tooltip="SCAIL-2 only. Full decoded output of the previous chunk. Only the last previous_frame_count are used as the extension anchor."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent", tooltip="Empty latent of the generation size."),
                io.Int.Output(display_name="video_frame_offset", tooltip="Adjusted offset + length. Wire into the next chunk."),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, pose_strength, pose_start, pose_end,
                video_frame_offset, previous_frame_count, replacement_mode=False, reference_image=None, clip_vision_output=None, pose_video=None,
                pose_video_mask=None, reference_image_mask=None, previous_frames=None) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        noise_mask = None

        ref_mask_flag = not replacement_mode
        positive = node_helpers.conditioning_set_values(positive, {"ref_mask_flag": ref_mask_flag})
        negative = node_helpers.conditioning_set_values(negative, {"ref_mask_flag": ref_mask_flag})

        prev_trimmed = None
        if previous_frames is not None and previous_frames.shape[0] > 0:
            prev_trimmed = previous_frames[-previous_frame_count:]
            video_frame_offset -= prev_trimmed.shape[0]
            video_frame_offset = max(0, video_frame_offset)

        if reference_image is not None:
            ref_imgs = comfy.utils.common_upscale(reference_image.movedim(-1, 1), width, height, "bicubic", "center").movedim(1, -1)
            n_ref = ref_imgs.shape[0]
            # SCAIL-2 multi-reference: the first image is the primary ref, the rest are additional references.

            # Replacement Mode: composite each ref on black bg using its mask as alpha matte
            if replacement_mode and reference_image_mask is not None:
                rm = comfy.utils.common_upscale(reference_image_mask.movedim(-1, 1), width, height, "nearest-exact", "center").movedim(1, -1)
                rm = rm[[min(i, rm.shape[0] - 1) for i in range(n_ref)]]
                is_char = (rm[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(ref_imgs.dtype)
                ref_imgs = ref_imgs * is_char
            # encode each ref individually so each stays a single latent frame (a batched encode would be treated as a video)
            ref_latents = [vae.encode(ref_imgs[i:i + 1, :, :, :3]) for i in range(n_ref)]
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        if pose_video is not None:
            if pose_video.shape[0] <= video_frame_offset:
                pose_video = None
            else:
                pose_video = pose_video[video_frame_offset:]
        if pose_video_mask is not None:
            if pose_video_mask.shape[0] <= video_frame_offset:
                pose_video_mask = None
            else:
                pose_video_mask = pose_video_mask[video_frame_offset:]

        # Truncate pose+mask jointly to the shorter of the two, capped at length.
        ts = [v.shape[0] for v in (pose_video, pose_video_mask) if v is not None]
        if ts:
            T_kept = ((min(min(ts), length) - 1) // 4) * 4 + 1
            if pose_video is not None:
                pose_video = pose_video[:T_kept]
            if pose_video_mask is not None:
                pose_video_mask = pose_video_mask[:T_kept]

        if pose_video is not None:
            pose_video = comfy.utils.common_upscale(pose_video[:length].movedim(-1, 1), width // 2, height // 2, "area", "center").movedim(1, -1)
            pose_video_latent = vae.encode(pose_video[:, :, :, :3]) * pose_strength
            positive = node_helpers.conditioning_set_values_with_timestep_range(positive, {"pose_video_latent": pose_video_latent}, pose_start, pose_end)
            negative = node_helpers.conditioning_set_values_with_timestep_range(negative, {"pose_video_latent": pose_video_latent}, pose_start, pose_end)

        if pose_video_mask is not None:
            mask_video_hw = comfy.utils.common_upscale(pose_video_mask[:length].movedim(-1, 1), width // 2, height // 2, "area", "center").movedim(1, -1)
            driving_mask_28ch = _extract_mask_to_28ch(mask_video_hw)
            positive = node_helpers.conditioning_set_values(positive, {"driving_mask_28ch": driving_mask_28ch})
            negative = node_helpers.conditioning_set_values(negative, {"driving_mask_28ch": driving_mask_28ch})

        # The ref mask binds reference frames to identities, so it only applies when there's a reference image.
        if reference_image_mask is not None and reference_image is not None:
            ref_mask_hw = comfy.utils.common_upscale(reference_image_mask.movedim(-1, 1), width, height, "nearest-exact", "center").movedim(1, -1)
            n_masks = ref_mask_hw.shape[0]
            n_ref = reference_image.shape[0]

            add_masks = [_extract_mask_to_28ch(ref_mask_hw[min(i, n_masks - 1)][None]) for i in range(1, n_ref)]
            ref_mask_1f = _extract_mask_to_28ch(ref_mask_hw[:1])
            zeros = torch.zeros((1, latent.shape[2], 28, ref_mask_1f.shape[-2], ref_mask_1f.shape[-1]), device=ref_mask_1f.device, dtype=ref_mask_1f.dtype)
            ref_mask_28ch = torch.cat(add_masks + [ref_mask_1f, zeros], dim=1)
            positive = node_helpers.conditioning_set_values(positive, {"ref_mask_28ch": ref_mask_28ch})
            negative = node_helpers.conditioning_set_values(negative, {"ref_mask_28ch": ref_mask_28ch})

        if prev_trimmed is not None:
            pf = comfy.utils.common_upscale(prev_trimmed.movedim(-1, 1), width, height, "bicubic", "center").movedim(1, -1)
            prev_latent = vae.encode(pf[:, :, :, :3])
            prev_latent_frames  = min(prev_latent.shape[2], latent.shape[2])
            latent[:, :, :prev_latent_frames] = prev_latent[:, :, :prev_latent_frames].to(latent.dtype)
            noise_mask = torch.ones((1, 1, latent.shape[2], latent.shape[-2], latent.shape[-1]), device=latent.device, dtype=latent.dtype)
            noise_mask[:, :, :prev_latent_frames] = 0.0

        out_latent = {"samples": latent}
        if noise_mask is not None:
            out_latent["noise_mask"] = noise_mask
        return io.NodeOutput(positive, negative, out_latent, video_frame_offset + length)


class SCAIL2ColoredMask(io.ComfyNode):
    """Render SAM3 tracks for the driving pose video and reference image(s) into the
    colored masks WanSCAILToVideo consumes. Shared `sort_by` keeps each identity on the
    same color across both outputs.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SCAIL2ColoredMask",
            display_name="Create SCAIL-2 Colored Mask",
            category="model/conditioning/wan/scail",
            inputs=[
                SAM3TrackData.Input("driving_track_data", tooltip="SAM3 track of the driving pose video. Will be rendered into the pose_video_mask output."),
                io.MultiType.Input("ref_track_data", [SAM3TrackData, io.Mask], optional=True, display_name="reference_masks",
                                   tooltip="SAM3 track of the reference image(s) (one identity per object, colored in batch order), or a plain MASK of the reference subject (rendered as a single identity)."),
                io.String.Input("object_indices", default="",
                                tooltip="Comma-separated list of person indices to include (e.g. '0,2,3'). Applied to both reference and pose video masks. Empty = all."),
                io.Combo.Input("sort_by", options=["none", "left_to_right", "area"], default="left_to_right",
                               tooltip="Order in which palette colors are assigned to the tracked objects (applied to both reference and pose video so each identity keeps the same color). Objects that appear in earlier frames always come first; within a frame, left_to_right = leftmost object (by centroid at first appearance) gets the first color, area = biggest object (by mask area at first appearance) gets the first color; none = keep SAM3's order."),
                io.Boolean.Input("replacement_mode", default=False,
                    tooltip="False = Animation Mode (pose_video_mask has black background, reference_image_mask has white background). "
                    "True = Replacement Mode (pose_video_mask has white background, reference_image_mask has black background)."),
            ],
            outputs=[
                io.Image.Output("pose_video_mask"),
                io.Image.Output("reference_image_mask"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, driving_track_data, object_indices, sort_by, replacement_mode, ref_track_data=None):
        def _prep(td):
            masks_bool = _unpack(td)
            if sort_by != "none" and masks_bool is not None:
                first_t, cx, area = _first_appearance_cx_area(masks_bool)
                if sort_by == "left_to_right":
                    order = sorted(range(len(cx)), key=lambda i: (first_t[i], cx[i]))
                else:  # "area"
                    order = sorted(range(len(area)), key=lambda i: (first_t[i], -area[i]))
                td = _subset_track_data(td, order)
            if object_indices.strip():
                indices = [int(i.strip()) for i in object_indices.split(",") if i.strip().isdigit()]
                packed = td.get("packed_masks")
                n_obj = packed.shape[1] if packed is not None else 0
                indices = [i for i in indices if 0 <= i < n_obj]
                td = _subset_track_data(td, indices)
            return td

        drv = _prep(driving_track_data)
        # Animation: driving=black, ref=white. Replacement: driving=white, ref=black.
        mask_video = _render_colored_masks(drv, "white" if replacement_mode else "black")
        ref_bg = "black" if replacement_mode else "white"

        if ref_track_data is not None:
            if isinstance(ref_track_data, torch.Tensor):  # plain comfy MASK
                reference_image_mask = _render_mask_as_identity(ref_track_data, ref_bg)
            else:
                reference_image_mask = _render_colored_masks(_prep(ref_track_data), ref_bg)
        else:
            H, W = drv["orig_size"]
            fill_value = 1.0 if ref_bg == "white" else 0.0
            reference_image_mask = torch.full((1, H, W, 3), fill_value, device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())

        return io.NodeOutput(mask_video, reference_image_mask)


class SCAILExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            WanSCAILToVideo,
            SCAIL2ColoredMask,
        ]


async def comfy_entrypoint() -> SCAILExtension:
    return SCAILExtension()
