from typing_extensions import override

import torch

import comfy.model_management
import comfy.patcher_extension
import node_helpers
from comfy_api.latest import ComfyExtension, io


class EmptyHiDreamO1LatentImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EmptyHiDreamO1LatentImage",
            display_name="Empty HiDream-O1 Latent Image",
            category="model/latent/hidream",
            description=(
                "Empty pixel-space latent for HiDream-O1-Image. The model was "
                "trained at ~4 megapixels; lower resolutions go off-distribution "
                "and quality regresses noticeably. Trained resolutions: "
                "2048x2048, 2304x1728, 1728x2304, 2560x1440, 1440x2560, "
                "2496x1664, 1664x2496, 3104x1312, 1312x3104, 2304x1792, 1792x2304."
            ),
            inputs=[
                io.Int.Input(id="width", default=2048, min=64, max=4096, step=32),
                io.Int.Input(id="height", default=2048, min=64, max=4096, step=32),
                io.Int.Input(id="batch_size", default=1, min=1, max=64),
            ],
            outputs=[io.Latent().Output()],
        )

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int = 1) -> io.NodeOutput:
        latent = torch.zeros(
            (batch_size, 3, height, width),
            device=comfy.model_management.intermediate_device(),
        )
        return io.NodeOutput({"samples": latent})


class HiDreamO1ReferenceImages(io.ComfyNode):
    """Attach reference images to both positive and negative conditioning."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HiDreamO1ReferenceImages",
            display_name="HiDream-O1 Reference Images",
            category="model/conditioning/hidream",
            description=(
                "Attach 1-10 reference images to conditioning, one for edit instruction"
                "or multiple for subject-driven personalization."
            ),
            inputs=[
                io.Conditioning.Input(id="positive"),
                io.Conditioning.Input(id="negative"),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, 11)],
                        min=1,
                    ),
                    tooltip=("Reference images. 1 image = instruction edit; 2-10 images = multi reference."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, *, positive, negative, images: io.Autogrow.Type) -> io.NodeOutput:
        refs = [images[f"image_{i}"] for i in range(1, 11) if f"image_{i}" in images]
        positive = node_helpers.conditioning_set_values(positive, {"reference_latents": refs}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"reference_latents": refs}, append=True)
        return io.NodeOutput(positive, negative)


class HiDreamO1PatchSeamSmoothing(io.ComfyNode):
    PATCH_SIZE = 32
    EDGE_FEATHER = 4

    # Shift presets per (pattern, N). 8-pass = 4-quadrant + 4 quarter-patch offsets.
    SHIFTS_BY_PATTERN = {
        ("single_shift", 2): [(0, 0), (16, 16)],
        ("single_shift", 4): [(0, 0), (16, 0), (0, 16), (16, 16)],
        ("single_shift", 8): [(0, 0), (16, 0), (0, 16), (16, 16),
                              (8, 8), (24, 8), (8, 24), (24, 24)],
        ("symmetric", 2):    [(-8, -8), (8, 8)],
        ("symmetric", 4):    [(-8, -8), (8, -8), (-8, 8), (8, 8)],
        ("symmetric", 8):    [(-12, -12), (4, -12), (-12, 4), (4, 4),
                              (-4, -4), (12, -4), (-4, 12), (12, 12)],
    }
    RAMP_LEVELS = {
        "2":          [2],
        "4":          [4],
        "ramp_2_4":   [2, 4],
        "ramp_2_4_8": [2, 4, 8],
    }

    @staticmethod
    def _hann_tile(cy: int, cx: int, size: int = 32) -> torch.Tensor:
        """size x size Hann tile peaking at (cy, cx) within a patch."""
        half = size // 2
        yy = torch.arange(size).view(size, 1)
        xx = torch.arange(size).view(1, size)
        dy = ((yy - cy + half) % size) - half
        dx = ((xx - cx + half) % size) - half
        return 0.25 * (1 + torch.cos(torch.pi * dy / half)) * (1 + torch.cos(torch.pi * dx / half))

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HiDreamO1PatchSeamSmoothing",
            display_name="HiDream-O1 Patch Seam Smoothing",
            category="model/patch/hidream",
            is_experimental=True,
            description=(
                "Average the model output across multiple shifted patch-grid "
                "positions during the late portion of sampling. Cancels seams."
            ),
            inputs=[
                io.Model.Input(id="model"),
                io.Float.Input(id="start_percent", default=0.8, min=0.0, max=1.0, step=0.01,
                    tooltip="Sampling progress (0=start, 1=end) at which the blend turns ON.",
                ),
                io.Float.Input(id="end_percent", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Sampling progress at which the blend turns OFF.",
                ),
                io.Combo.Input(
                    id="pattern",
                    options=["single_shift", "symmetric"],
                    default="single_shift",
                    tooltip="Shift layout. single_shift: one pass at the natural patch grid + others offset. symmetric: all passes off-grid, shifts split around origin.",
                ),
                io.Combo.Input(
                    id="passes",
                    options=["2", "4", "ramp_2_4", "ramp_2_4_8"],
                    default="2",
                    tooltip="Number of passes per gated step. 2/4 = fixed. ramp_*: pass count increases as sampling approaches end (more smoothing where seams are most visible).",
                ),
                io.Combo.Input(
                    id="blend",
                    options=["average", "window", "median"],
                    default="average",
                    tooltip="average: equal-weight mean. window: Hann-windowed weighting favoring each pass away from its patch boundaries. median: per-pixel median, rejects wraparound-outlier passes.",
                ),
                io.Float.Input(id="strength", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Interpolation between the natural-grid pred (0) and the averaged result (1).",
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, *, model, start_percent: float, end_percent: float, pattern: str, passes: str, blend: str, strength: float) -> io.NodeOutput:
        if strength <= 0.0 or end_percent <= start_percent:
            return io.NodeOutput(model)

        P = cls.PATCH_SIZE
        half = P // 2
        shift_levels = [cls.SHIFTS_BY_PATTERN[(pattern, n)] for n in cls.RAMP_LEVELS[passes]]

        if blend == "window":
            window_tile_levels = [
                torch.stack([cls._hann_tile((half - sy) % P, (half - sx) % P, P) for sy, sx in lst], dim=0)
                for lst in shift_levels
            ]
        else:
            window_tile_levels = [None] * len(shift_levels)

        m = model.clone()
        model_sampling = m.get_model_object("model_sampling")
        multiplier = float(model_sampling.multiplier)
        start_t = float(model_sampling.percent_to_sigma(start_percent)) * multiplier
        end_t = float(model_sampling.percent_to_sigma(end_percent)) * multiplier

        edge_ramp_cache: dict = {}

        def get_edge_ramp(H: int, W: int, device, dtype) -> torch.Tensor:
            key = (H, W, device, dtype)
            cached = edge_ramp_cache.get(key)
            if cached is not None:
                return cached
            feather = cls.EDGE_FEATHER
            ys = torch.minimum(torch.arange(H, device=device, dtype=torch.float32),
                               (H - 1) - torch.arange(H, device=device, dtype=torch.float32))
            xs = torch.minimum(torch.arange(W, device=device, dtype=torch.float32),
                               (W - 1) - torch.arange(W, device=device, dtype=torch.float32))
            y_mask = ((ys - P) / feather).clamp(0, 1)
            x_mask = ((xs - P) / feather).clamp(0, 1)
            ramp = (y_mask[:, None] * x_mask[None, :]).to(dtype)
            edge_ramp_cache[key] = ramp
            return ramp

        def smoothing_wrapper(executor, *args, **kwargs):
            x = args[0]
            t = float(args[1][0])
            pred = executor(*args, **kwargs)
            if not (end_t <= t <= start_t):
                return pred
            # Pick shift-level by sigma phase across the gated range.
            if len(shift_levels) == 1:
                level_idx = 0
            else:
                phase = (start_t - t) / max(start_t - end_t, 1e-8)
                level_idx = min(int(phase * len(shift_levels)), len(shift_levels) - 1)
            shifts = shift_levels[level_idx]
            window_tiles = window_tile_levels[level_idx]

            preds = []
            for sy, sx in shifts:
                if sy == 0 and sx == 0:
                    preds.append(pred)
                    continue
                x_rolled = torch.roll(x, shifts=(sy, sx), dims=(-2, -1))
                pred_rolled = executor(x_rolled, *args[1:], **kwargs)
                preds.append(torch.roll(pred_rolled, shifts=(-sy, -sx), dims=(-2, -1)))
            stacked = torch.stack(preds, dim=0)  # (N, B, C, H, W)
            _, _, _, H, W = stacked.shape
            if blend == "window":
                N = stacked.shape[0]
                tiles = window_tiles.to(device=stacked.device, dtype=stacked.dtype)
                w = tiles.repeat(1, H // P, W // P)[:, :H, :W]
                sum_w = w.sum(dim=0, keepdim=True)
                w = torch.where(sum_w < 1e-3, torch.full_like(w, 1.0 / N), w / sum_w.clamp(min=1e-8))
                avg = (stacked * w[:, None, None, :, :]).sum(dim=0)
            elif blend == "median":
                avg = torch.median(stacked, dim=0).values
            else:
                avg = stacked.mean(dim=0)

            # Mask out the P-px wraparound contamination strip at each edge.
            mask = get_edge_ramp(H, W, pred.device, pred.dtype)
            return pred * (1.0 - mask * strength) + avg * (mask * strength)

        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "hidream_o1_patch_seam_smoothing", smoothing_wrapper)
        return io.NodeOutput(m)


class HiDreamO1Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            EmptyHiDreamO1LatentImage,
            HiDreamO1ReferenceImages,
            HiDreamO1PatchSeamSmoothing,
        ]


async def comfy_entrypoint() -> HiDreamO1Extension:
    return HiDreamO1Extension()
