import nodes
import node_helpers
import torch
import torchvision.transforms.functional as TF
import comfy.model_management
import comfy.utils
import numpy as np
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_wan import parse_json_tracks

# https://github.com/ali-vilab/Wan-Move/blob/main/wan/modules/trajectory.py
from PIL import Image, ImageDraw

SKIP_ZERO = False

def get_pos_emb(
    pos_k: torch.Tensor, # A 1D tensor containing positions for which to generate embeddings.
    pos_emb_dim: int,
    theta_func: callable = lambda i, d: torch.pow(10000, torch.mul(2, torch.div(i.to(torch.float32), d))), #Function to compute thetas based on position and embedding dimensions.
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor: # The position embeddings (batch_size, pos_emb_dim)

    assert pos_emb_dim % 2 == 0, "The dimension of position embeddings must be even."
    pos_k = pos_k.to(device, dtype)
    if SKIP_ZERO:
        pos_k = pos_k + 1
    batch_size = pos_k.size(0)

    denominator = torch.arange(0, pos_emb_dim // 2, device=device, dtype=dtype)
    # Expand denominator to match the shape needed for broadcasting
    denominator_expanded = denominator.view(1, -1).expand(batch_size, -1)

    thetas = theta_func(denominator_expanded, pos_emb_dim)

    # Ensure pos_k is in the correct shape for broadcasting
    pos_k_expanded = pos_k.view(-1, 1).to(dtype)
    sin_thetas = torch.sin(torch.div(pos_k_expanded, thetas))
    cos_thetas = torch.cos(torch.div(pos_k_expanded, thetas))

    # Concatenate sine and cosine embeddings along the last dimension
    pos_emb = torch.cat([sin_thetas, cos_thetas], dim=-1)

    return pos_emb

def create_pos_embeddings(
    pred_tracks: torch.Tensor, # the predicted tracks, [T, N, 2]
    pred_visibility: torch.Tensor, # the predicted visibility [T, N]
    downsample_ratios: list[int], # the ratios for downsampling time, height, and width
    height: int, # the height of the feature map
    width: int, # the width of the feature map
    track_num: int = -1, # the number of tracks to use
    t_down_strategy: str = "sample", # the strategy for downsampling time dimension
):
    assert t_down_strategy in ["sample", "average"], "Invalid strategy for downsampling time dimension."

    t, n, _ = pred_tracks.shape
    t_down, h_down, w_down = downsample_ratios
    track_pos = - torch.ones(n, (t-1) // t_down + 1, 2, dtype=torch.long)

    if track_num == -1:
        track_num = n

    tracks_idx = torch.randperm(n)[:track_num]
    tracks = pred_tracks[:, tracks_idx]
    visibility = pred_visibility[:, tracks_idx]

    for t_idx in range(0, t, t_down):
        if t_down_strategy == "sample" or t_idx == 0:
            cur_tracks = tracks[t_idx] # [N, 2]
            cur_visibility = visibility[t_idx] # [N]
        else:
            cur_tracks = tracks[t_idx:t_idx+t_down].mean(dim=0)
            cur_visibility = torch.any(visibility[t_idx:t_idx+t_down], dim=0)

        for i in range(track_num):
            if not cur_visibility[i] or cur_tracks[i][0] < 0 or cur_tracks[i][1] < 0 or cur_tracks[i][0] >= width or cur_tracks[i][1] >= height:
                continue
            x, y = cur_tracks[i]
            x, y = int(x // w_down), int(y // h_down)
            track_pos[i, t_idx // t_down, 0], track_pos[i, t_idx // t_down, 1] = y, x

    return track_pos # the position embeddings, [N, T', 2], 2 = height, width

def replace_feature(
    vae_feature: torch.Tensor,  # [B, C', T', H', W']
    track_pos: torch.Tensor,    # [B, N, T', 2]
    strength: float = 1.0
) -> torch.Tensor:
    b, _, t, h, w = vae_feature.shape
    assert b == track_pos.shape[0], "Batch size mismatch."
    n = track_pos.shape[1]

    # Shuffle the trajectory order
    track_pos = track_pos[:, torch.randperm(n), :, :]

    # Extract coordinates at time steps ≥ 1 and generate a valid mask
    current_pos = track_pos[:, :, 1:, :]  # [B, N, T-1, 2]
    mask = (current_pos[..., 0] >= 0) & (current_pos[..., 1] >= 0)  # [B, N, T-1]

    # Get all valid indices
    valid_indices = mask.nonzero(as_tuple=False)  # [num_valid, 3]
    num_valid = valid_indices.shape[0]

    if num_valid == 0:
        return vae_feature

    # Decompose valid indices into each dimension
    batch_idx = valid_indices[:, 0]
    track_idx = valid_indices[:, 1]
    t_rel = valid_indices[:, 2]
    t_target = t_rel + 1  # Convert to original time step indices

    # Extract target position coordinates
    h_target = current_pos[batch_idx, track_idx, t_rel, 0].long()  # Ensure integer indices
    w_target = current_pos[batch_idx, track_idx, t_rel, 1].long()

    # Extract source position coordinates (t=0)
    h_source = track_pos[batch_idx, track_idx, 0, 0].long()
    w_source = track_pos[batch_idx, track_idx, 0, 1].long()

    # Get source features and assign to target positions
    src_features = vae_feature[batch_idx, :, 0, h_source, w_source]
    dst_features = vae_feature[batch_idx, :, t_target, h_target, w_target]

    vae_feature[batch_idx, :, t_target, h_target, w_target] = dst_features + (src_features - dst_features) * strength


    return vae_feature

# Visualize functions

def _draw_gradient_polyline_on_overlay(overlay, line_width, points, start_color, opacity=1.0):
    draw = ImageDraw.Draw(overlay, 'RGBA')
    points = points[::-1]

    # Compute total length
    total_length = 0
    segment_lengths = []
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        length = (dx * dx + dy * dy) ** 0.5
        segment_lengths.append(length)
        total_length += length

    if total_length == 0:
        return

    accumulated_length = 0

    # Draw the gradient polyline
    for idx, (start_point, end_point) in enumerate(zip(points[:-1], points[1:])):
        segment_length = segment_lengths[idx]
        steps = max(int(segment_length), 1)

        for i in range(steps):
            current_length = accumulated_length + (i / steps) * segment_length
            ratio = current_length / total_length

            alpha = int(255 * (1 - ratio) * opacity)
            color = (*start_color, alpha)

            x = int(start_point[0] + (end_point[0] - start_point[0]) * i / steps)
            y = int(start_point[1] + (end_point[1] - start_point[1]) * i / steps)

            dynamic_line_width = max(int(line_width * (1 - ratio)), 1)
            draw.line([(x, y), (x + 1, y)], fill=color, width=dynamic_line_width)

        accumulated_length += segment_length


def add_weighted(rgb, track):
    rgb = np.array(rgb) # [H, W, C] "RGB"
    track = np.array(track) # [H, W, C] "RGBA"

    alpha = track[:, :, 3] / 255.0
    alpha = np.stack([alpha] * 3, axis=-1)
    blend_img = track[:, :, :3] * alpha + rgb * (1 - alpha)

    return Image.fromarray(blend_img.astype(np.uint8))

def draw_tracks_on_video(video, tracks, visibility=None, track_frame=24, circle_size=12, opacity=0.5, line_width=16):
    color_map = [(102, 153, 255), (0, 255, 255), (255, 255, 0), (255, 102, 204), (0, 255, 0)]

    video = video.byte().cpu().numpy()  # (81, 480, 832, 3)
    tracks = tracks[0].long().detach().cpu().numpy()
    if visibility is not None:
        visibility = visibility[0].detach().cpu().numpy()

    num_frames, height, width = video.shape[:3]
    num_tracks = tracks.shape[1]
    alpha_opacity = int(255 * opacity)

    output_frames = []
    for t in range(num_frames):
        frame_rgb = video[t].astype(np.float32)

        # Create a single RGBA overlay for all tracks in this frame
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)

        polyline_data = []

        # Draw all circles on a single overlay
        for n in range(num_tracks):
            if visibility is not None and visibility[t, n] == 0:
                continue

            track_coord = tracks[t, n]
            color = color_map[n % len(color_map)]
            circle_color = color + (alpha_opacity,)

            draw_overlay.ellipse((track_coord[0] - circle_size, track_coord[1] - circle_size, track_coord[0] + circle_size, track_coord[1] + circle_size),
                fill=circle_color
            )

            # Store polyline data for batch processing
            tracks_coord = tracks[max(t - track_frame, 0):t + 1, n]
            if len(tracks_coord) > 1:
                polyline_data.append((tracks_coord, color))

        # Blend circles overlay once
        overlay_np = np.array(overlay)
        alpha = overlay_np[:, :, 3:4] / 255.0
        frame_rgb = overlay_np[:, :, :3] * alpha + frame_rgb * (1 - alpha)

        # Draw all polylines on a single overlay
        if polyline_data:
            polyline_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            for tracks_coord, color in polyline_data:
                _draw_gradient_polyline_on_overlay(polyline_overlay, line_width, tracks_coord, color, opacity)

            # Blend polylines overlay once
            polyline_np = np.array(polyline_overlay)
            alpha = polyline_np[:, :, 3:4] / 255.0
            frame_rgb = polyline_np[:, :, :3] * alpha + frame_rgb * (1 - alpha)

        output_frames.append(Image.fromarray(frame_rgb.astype(np.uint8)))

    return output_frames


class WanMoveVisualizeTracks(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanMoveVisualizeTracks",
            category="model/conditioning/wan/move",
            inputs=[
                io.Image.Input("images"),
                io.Tracks.Input("tracks", optional=True),
                io.Int.Input("line_resolution", default=24, min=1, max=1024),
                io.Int.Input("circle_size", default=12, min=1, max=128, advanced=True),
                io.Float.Input("opacity", default=0.75, min=0.0, max=1.0, step=0.01),
                io.Int.Input("line_width", default=16, min=1, max=128, advanced=True),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, images, line_resolution, circle_size, opacity, line_width, tracks=None) -> io.NodeOutput:
        if tracks is None:
            return io.NodeOutput(images)

        track_path = tracks["track_path"].unsqueeze(0)
        track_visibility = tracks["track_visibility"].unsqueeze(0)
        images_in = images * 255.0
        if images_in.shape[0] != track_path.shape[1]:
            repeat_count = track_path.shape[1] // images.shape[0]
            images_in = images_in.repeat(repeat_count, 1, 1, 1)
        track_video = draw_tracks_on_video(images_in, track_path, track_visibility, track_frame=line_resolution, circle_size=circle_size, opacity=opacity, line_width=line_width)
        track_video = torch.stack([TF.to_tensor(frame) for frame in track_video], dim=0).movedim(1, -1).float()

        return io.NodeOutput(track_video.to(comfy.model_management.intermediate_device()))


class WanMoveTracksFromCoords(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanMoveTracksFromCoords",
            category="model/conditioning/wan/move",
            inputs=[
                io.String.Input("track_coords", force_input=True, default="[]", optional=True),
                io.Mask.Input("track_mask", optional=True),
            ],
            outputs=[
                io.Tracks.Output(),
                io.Int.Output(display_name="track_length"),
            ],
        )

    @classmethod
    def execute(cls, track_coords, track_mask=None) -> io.NodeOutput:
        device=comfy.model_management.intermediate_device()

        tracks_data = parse_json_tracks(track_coords)
        track_length = len(tracks_data[0])

        track_list = [
                [[track[frame]['x'], track[frame]['y']] for track in tracks_data]
                for frame in range(len(tracks_data[0]))
            ]
        tracks = torch.tensor(track_list, dtype=torch.float32, device=device)  # [frames, num_tracks, 2]

        num_tracks = tracks.shape[-2]
        if track_mask is None:
            track_visibility = torch.ones((track_length, num_tracks), dtype=torch.bool, device=device)
        else:
            track_visibility = (track_mask > 0).any(dim=(1, 2)).unsqueeze(-1)

        out_track_info = {}
        out_track_info["track_path"] = tracks
        out_track_info["track_visibility"] = track_visibility
        return io.NodeOutput(out_track_info, track_length)


class GenerateTracks(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GenerateTracks",
            search_aliases=["motion paths", "camera movement", "trajectory"],
            display_name="Generate Video Tracks",
            category="model/conditioning/wan/move",
            inputs=[
                io.Int.Input("width", default=832, min=16, max=4096, step=16),
                io.Int.Input("height", default=480, min=16, max=4096, step=16),
                io.Float.Input("start_x", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Normalized X coordinate (0-1) for start position."),
                io.Float.Input("start_y", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Normalized Y coordinate (0-1) for start position."),
                io.Float.Input("end_x", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Normalized X coordinate (0-1) for end position."),
                io.Float.Input("end_y", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Normalized Y coordinate (0-1) for end position."),
                io.Int.Input("num_frames", default=81, min=1, max=1024),
                io.Int.Input("num_tracks", default=5, min=1, max=100),
                io.Float.Input("track_spread", default=0.025, min=0.0, max=1.0, step=0.001, tooltip="Normalized distance between tracks. Tracks are spread perpendicular to the motion direction."),
                io.Boolean.Input("bezier", default=False, tooltip="Enable Bezier curve path using the mid point as control point."),
                io.Float.Input("mid_x", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="Normalized X control point for Bezier curve. Only used when 'bezier' is enabled."),
                io.Float.Input("mid_y", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="Normalized Y control point for Bezier curve. Only used when 'bezier' is enabled."),
                io.Combo.Input(
                    "interpolation",
                    options=["linear", "ease_in", "ease_out", "ease_in_out", "constant"],
                    tooltip="Controls the timing/speed of movement along the path.",
                ),
                io.Mask.Input("track_mask", optional=True, tooltip="Optional mask to indicate visible frames."),
            ],
            outputs=[
                io.Tracks.Output(),
                io.Int.Output(display_name="track_length"),
            ],
        )

    @classmethod
    def execute(cls, width, height, start_x, start_y, mid_x, mid_y, end_x, end_y, num_frames, num_tracks,
                track_spread, bezier=False, interpolation="linear", track_mask=None) -> io.NodeOutput:
        device = comfy.model_management.intermediate_device()
        track_length = num_frames

        # normalized coordinates to pixel coordinates
        start_x_px = start_x * width
        start_y_px = start_y * height
        mid_x_px = mid_x * width
        mid_y_px = mid_y * height
        end_x_px = end_x * width
        end_y_px = end_y * height

        track_spread_px = track_spread * (width + height) / 2 # Use average of width/height for spread to keep it proportional

        t = torch.linspace(0, 1, num_frames, device=device)
        if interpolation == "constant": # All points stay at start position
            interp_values = torch.zeros_like(t)
        elif interpolation == "linear":
            interp_values = t
        elif interpolation == "ease_in":
            interp_values = t ** 2
        elif interpolation == "ease_out":
            interp_values = 1 - (1 - t) ** 2
        elif interpolation == "ease_in_out":
            interp_values = t * t * (3 - 2 * t)

        if bezier: # apply interpolation to t for timing control along the bezier path
            t_interp = interp_values
            one_minus_t = 1 - t_interp
            x_positions = one_minus_t ** 2 * start_x_px + 2 * one_minus_t * t_interp * mid_x_px + t_interp ** 2 * end_x_px
            y_positions = one_minus_t ** 2 * start_y_px + 2 * one_minus_t * t_interp * mid_y_px + t_interp ** 2 * end_y_px
            tangent_x = 2 * one_minus_t * (mid_x_px - start_x_px) + 2 * t_interp * (end_x_px - mid_x_px)
            tangent_y = 2 * one_minus_t * (mid_y_px - start_y_px) + 2 * t_interp * (end_y_px - mid_y_px)
        else: # calculate base x and y positions for each frame (center track)
            x_positions = start_x_px + (end_x_px - start_x_px) * interp_values
            y_positions = start_y_px + (end_y_px - start_y_px) * interp_values
            # For non-bezier, tangent is constant (direction from start to end)
            tangent_x = torch.full_like(t, end_x_px - start_x_px)
            tangent_y = torch.full_like(t, end_y_px - start_y_px)

        track_list = []
        for frame_idx in range(num_frames):
            # Calculate perpendicular direction at this frame
            tx = tangent_x[frame_idx].item()
            ty = tangent_y[frame_idx].item()
            length = (tx ** 2 + ty ** 2) ** 0.5

            if length > 0: # Perpendicular unit vector (rotate 90 degrees)
                perp_x = -ty / length
                perp_y = tx / length
            else: # If tangent is zero, spread horizontally
                perp_x = 1.0
                perp_y = 0.0

            frame_tracks = []
            for track_idx in range(num_tracks): # center tracks around the main path offset ranges from -(num_tracks-1)/2 to +(num_tracks-1)/2
                offset = (track_idx - (num_tracks - 1) / 2) * track_spread_px
                track_x = x_positions[frame_idx].item() + perp_x * offset
                track_y = y_positions[frame_idx].item() + perp_y * offset
                frame_tracks.append([track_x, track_y])
            track_list.append(frame_tracks)

        tracks = torch.tensor(track_list, dtype=torch.float32, device=device)  # [frames, num_tracks, 2]

        if track_mask is None:
            track_visibility = torch.ones((track_length, num_tracks), dtype=torch.bool, device=device)
        else:
            track_visibility = (track_mask > 0).any(dim=(1, 2)).unsqueeze(-1)

        out_track_info = {}
        out_track_info["track_path"] = tracks
        out_track_info["track_visibility"] = track_visibility
        return io.NodeOutput(out_track_info, track_length)


class WanMoveConcatTrack(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanMoveConcatTrack",
            category="model/conditioning/wan/move",
            inputs=[
                io.Tracks.Input("tracks_1"),
                io.Tracks.Input("tracks_2", optional=True),
            ],
            outputs=[
                io.Tracks.Output(),
            ],
        )

    @classmethod
    def execute(cls, tracks_1=None, tracks_2=None) -> io.NodeOutput:
        if tracks_2 is None:
            return io.NodeOutput(tracks_1)

        tracks_out = torch.cat([tracks_1["track_path"], tracks_2["track_path"]], dim=1)  # Concatenate along the track dimension
        mask_out = torch.cat([tracks_1["track_visibility"], tracks_2["track_visibility"]], dim=-1)

        out_track_info = {}
        out_track_info["track_path"] = tracks_out
        out_track_info["track_visibility"] = mask_out
        return io.NodeOutput(out_track_info)


class WanMoveTrackToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanMoveTrackToVideo",
            category="model/conditioning/wan/move",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Tracks.Input("tracks", optional=True),
                io.Float.Input("strength", default=1.0, min=0.0, max=100.0, step=0.01, tooltip="Strength of the track conditioning."),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("start_image"),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, strength, tracks=None, start_image=None, clip_vision_output=None) -> io.NodeOutput:
        device=comfy.model_management.intermediate_device()
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=device)
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            image = torch.ones((length, height, width, start_image.shape[-1]), device=start_image.device, dtype=start_image.dtype) * 0.5
            image[:start_image.shape[0]] = start_image

            concat_latent_image = vae.encode(image[:, :, :, :3])
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
            mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

            if tracks is not None and strength > 0.0:
                tracks_path = tracks["track_path"][:length]  # [T, N, 2]
                num_tracks = tracks_path.shape[-2]

                track_visibility = tracks.get("track_visibility", torch.ones((length, num_tracks), dtype=torch.bool, device=device))

                track_pos = create_pos_embeddings(tracks_path, track_visibility, [4, 8, 8], height, width, track_num=num_tracks)
                track_pos = comfy.utils.resize_to_batch_size(track_pos.unsqueeze(0), batch_size)
                concat_latent_image_pos = replace_feature(concat_latent_image, track_pos, strength)
            else:
                concat_latent_image_pos = concat_latent_image

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image_pos, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent)


class WanMoveExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            WanMoveTrackToVideo,
            WanMoveTracksFromCoords,
            WanMoveConcatTrack,
            WanMoveVisualizeTracks,
            GenerateTracks,
        ]

async def comfy_entrypoint() -> WanMoveExtension:
    return WanMoveExtension()
