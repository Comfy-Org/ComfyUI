import math
import re

import node_helpers
import torch
from comfy.ldm.lightricks.symmetric_patchifier import SymmetricPatchifier, latent_to_pixel_coords
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_lt import (
    LTXVAddGuide,
    _append_guide_attention_entry,
    conditioning_get_any_value,
    get_keyframe_idxs,
    get_noise_mask,
)
from typing_extensions import override

DEFAULT_TEMPORAL_SCALE = 8
_OCCUPIED_MASK_MAX = 1.0 - 1e-4


def get_generated_keyframes(cond):
    return conditioning_get_any_value(cond, "generated_keyframes", None)


def _parse_frame_index_list(value, field, expected_count, first, last, expected_desc, empty_hint):
    """Parse a manual frame index override and validate count, uniqueness and range.

    Order is not significant — only the set of positions matters — so the list
    is returned as written rather than sorted. ``expected_count`` may be None
    when the list itself defines how many keyframes to add.
    """
    parts = [part for part in re.split(r"[,\s]+", value.strip()) if part]
    if not parts:
        raise ValueError(
            f"{field} is empty. Provide at least one index, or leave {field} empty {empty_hint}."
        )
    try:
        indices = [int(part) for part in parts]
    except ValueError:
        bad = ", ".join(repr(part) for part in parts if not re.fullmatch(r"-?\d+", part))
        raise ValueError(
            f"{field} must be a comma-separated list of integers, but could not parse {bad}."
        ) from None

    if expected_count is not None and len(indices) != expected_count:
        raise ValueError(
            f"{field} lists {len(indices)} index/indices but {expected_desc}. Provide exactly "
            f"{expected_count}, or leave {field} empty {empty_hint}."
        )

    duplicates = sorted({index for index in indices if indices.count(index) > 1})
    if duplicates:
        raise ValueError(
            f"{field} must not place two keyframes on the same pixel frame, but "
            f"{', '.join(str(index) for index in duplicates)} appear(s) more than once."
        )

    out_of_range = [index for index in indices if not first <= index <= last]
    if out_of_range:
        raise ValueError(
            f"{field} must lie between {first} and {last}, but got "
            f"{', '.join(str(index) for index in out_of_range)}."
        )
    return indices


def _grow_guide_attention_entry(positive, negative, index, extra_pre_filter_count, extra_frames):
    """Grow an existing guide_attention_entry when extending generated keyframes."""
    results = []
    for cond in (positive, negative):
        existing = []
        for t in cond:
            found = t[1].get("guide_attention_entries", None)
            if found is not None:
                existing = found
                break
        if index >= len(existing):
            raise ValueError(
                f"The generated keyframes recorded guide entry {index} but the conditioning only has "
                f"{len(existing)}. The conditioning was rebuilt after they were added."
            )
        entries = list(existing)
        grown = dict(entries[index])
        grown["pre_filter_count"] = grown["pre_filter_count"] + extra_pre_filter_count
        shape = list(grown["latent_shape"])
        shape[0] = shape[0] + extra_frames
        grown["latent_shape"] = shape
        entries[index] = grown
        results.append(node_helpers.conditioning_set_values(cond, {"guide_attention_entries": entries}))
    return results[0], results[1]


def _spaced_positions_keep_last(num_keyframes: int, num_frames: int) -> list[int]:
    return (
        torch.linspace(0, num_frames - 1, num_keyframes + 1)
        .round()
        .to(torch.int64)
        .tolist()[1:]
    )


def detailing_positions(num_frames: int, interval_frames: float) -> list[int]:
    """About one detailing keyframe every ``interval_frames`` pixel frames.

    Skips frame 0 (already a standalone token) and keeps the last frame.
    ``interval_frames`` 24 is about 1/s at 24 fps.
    """
    if interval_frames <= 0:
        raise ValueError(f"interval_frames must be > 0, got {interval_frames}")
    if num_frames <= 1:
        raise ValueError(
            f"A {num_frames}-frame target has no pixel frames to place keyframes on."
        )
    count = max(1, round((num_frames - 1) / interval_frames))
    positions = [index for index in _spaced_positions_keep_last(count, num_frames) if index != 0]
    if not positions:
        raise ValueError(
            f"A {num_frames}-frame target has no pixel frames to place keyframes on."
        )
    return positions


def free_detailing_slots(num_frames: int, interval_frames: float, occupied: set[int]) -> list[int]:
    """Density candidates that are not already I2V / guides / detailing KFs."""
    taken = set(occupied)
    positions = [
        index
        for index in detailing_positions(num_frames, interval_frames)
        if index not in taken
    ]
    if not positions:
        raise ValueError(
            "Every candidate detailing-keyframe pixel already has an image keyframe "
            "or a guide. Leave at least one unoccupied frame, or pass frame_indices."
        )
    return positions


def scale_frame_indices(indices: list[int], old_num_frames: int, new_num_frames: int) -> list[int]:
    """Map pixel indices from one canvas length onto another (e.g. temporal x2)."""
    if old_num_frames <= 1:
        raise ValueError(f"Cannot scale keyframe indices from a {old_num_frames}-frame canvas.")
    if new_num_frames <= 1:
        raise ValueError(f"Cannot scale keyframe indices onto a {new_num_frames}-frame canvas.")
    scale = (new_num_frames - 1) / (old_num_frames - 1)
    remapped = [int(round(index * scale)) for index in indices]
    duplicates = sorted({index for index in remapped if remapped.count(index) > 1})
    if duplicates:
        raise ValueError(
            "Scaling keyframe indices onto the new canvas collapsed "
            f"{', '.join(str(index) for index in duplicates)} onto the same pixel frame."
        )
    return remapped


def _as_int_set(value) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (int, float)):
        return {int(round(value))}
    if isinstance(value, (set, frozenset)):
        return {int(round(item)) for item in value}
    if isinstance(value, (list, tuple)):
        out = set()
        for item in value:
            out |= _as_int_set(item)
        return out
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return _as_int_set(value.tolist())
    return {int(round(float(value)))}


def pixel_frames_from_keyframe_idxs(keyframe_idxs) -> set[int]:
    """Unique RoPE *start* pixel times of extra guide / keyframe tokens.

    ``keyframe_idxs`` is ``(B, 3, tokens, 2)`` (t/h/w × start/end). A keyframe
    at pixel 24 spans the half-open interval ``[24, 25)``, so only 24 is
    occupied — the exclusive end is not a slot.
    """
    if keyframe_idxs is None:
        return set()
    if keyframe_idxs.ndim >= 4:
        starts = keyframe_idxs[:, 0, :, 0]
    else:
        starts = keyframe_idxs[:, 0]
    return _as_int_set(starts)


def occupied_pixel_frames(latent, temporal_scale: int, num_frames: int, video_latent_frames=None) -> set[int]:
    """Pixel frames that already hold an in-place image keyframe.

    Prefers ``noise_mask`` (0 means frozen / guided). Falls back to
    non-zero latent frames when no mask is present. Each occupied latent
    index maps to its representative pixel: 0, ``t * scale``, or last frame.

    ``video_latent_frames`` limits the scan to the video portion of T so
    appended guide / detailing-keyframe tokens are not treated as video frames.
    Extra-token pixel times come from ``pixel_frames_from_keyframe_idxs``.
    """
    samples = latent["samples"]
    latent_frames = samples.shape[2]
    if video_latent_frames is None:
        scan_frames = latent_frames
    else:
        scan_frames = min(max(int(video_latent_frames), 0), latent_frames)
    taken = set()

    def add_latent_index(index: int) -> None:
        if index <= 0:
            pixel = 0
        elif index >= scan_frames - 1:
            pixel = num_frames - 1
        else:
            pixel = index * temporal_scale
        taken.add(min(max(pixel, 0), num_frames - 1))

    mask = latent.get("noise_mask")
    if mask is not None and getattr(mask, "ndim", 0) >= 3:
        for index in range(min(mask.shape[2], scan_frames)):
            if torch.any(mask[:, :, index : index + 1] < _OCCUPIED_MASK_MAX):
                add_latent_index(index)
        return taken

    for index in range(scan_frames):
        if torch.any(samples[:, :, index : index + 1] != 0):
            add_latent_index(index)
    return taken


def nearest_latent_index(pixel_frame: int, temporal_scale: int, num_latent_frames: int) -> int:
    return min(max(round(pixel_frame / temporal_scale), 0), num_latent_frames - 1)


def should_copy_nearest_video_frames(keyframe_t, requested_count, has_recorded_indices, batched_singles):
    """True when ``keyframes`` is a longer video to sample, not stacked keyframes."""
    return (
        not has_recorded_indices
        and requested_count is not None
        and not batched_singles
        and keyframe_t > requested_count
    )


def keyframes_from_video(samples, indices, temporal_scale: int):
    """Stack the nearest video latent frame at each pixel index."""
    if not torch.is_tensor(samples) or samples.ndim != 5:
        raise ValueError(
            "Initializing keyframes from a video needs a plain 5D video latent. "
            "Split audio with Separate AV Latent first, and peel generated "
            "keyframes before copying from the video."
        )
    temporal_scale = int(temporal_scale)
    if temporal_scale < 1:
        raise ValueError(f"temporal_scale must be >= 1, got {temporal_scale}")
    num_latent_frames = samples.shape[2]
    if num_latent_frames < 1:
        raise ValueError("The video latent has no frames to copy from.")
    frames = []
    for pixel_frame in indices:
        idx = nearest_latent_index(pixel_frame, temporal_scale, num_latent_frames)
        frames.append(samples[:, :, idx : idx + 1])
    return torch.cat(frames, dim=2)


def _fit_keyframe_samples(keyframe_samples, samples, num_slots):
    """Pad stacked keyframe tokens with zeros up to ``num_slots``; reject extras."""
    expected = (
        samples.shape[0],
        samples.shape[1],
        num_slots,
        samples.shape[3],
        samples.shape[4],
    )
    have = keyframe_samples.shape[2]
    if (
        keyframe_samples.shape[0] != expected[0]
        or keyframe_samples.shape[1] != expected[1]
        or keyframe_samples.shape[3:] != expected[3:]
    ):
        raise ValueError(
            "The keyframes latent must hold whole latent frames at this latent's shape, expected "
            f"{list(expected)} but got {list(keyframe_samples.shape)}. Resize it to this stage's "
            "resolution first."
        )
    if have > num_slots:
        raise ValueError(
            f"The keyframes latent holds {have} frame(s) but only {num_slots} free slot(s) "
            "are available on this canvas. Pass fewer keyframes, or set frame_indices."
        )
    if have < num_slots:
        pad = torch.zeros(
            (expected[0], expected[1], num_slots - have, expected[3], expected[4]),
            dtype=keyframe_samples.dtype,
            device=keyframe_samples.device,
        )
        keyframe_samples = torch.cat([keyframe_samples, pad], dim=2)
    return keyframe_samples


class LTXVAddGeneratedKeyframes(io.ComfyNode):
    PATCHIFIER = SymmetricPatchifier(1, start_end=True)

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVAddGeneratedKeyframes",
            display_name="LTXV Add Generated Keyframes",
            category="model/conditioning/ltxv",
            search_aliases=["detailing", "dfr", "generated keyframes"],
            description=(
                "Append detailing keyframes to a video latent. Each keyframe is one latent "
                "frame of tokens whose RoPE position spans a single pixel frame; they are "
                "denoised with the video and are not part of the decoded output. Placement "
                "is one slot every interval_frames pixels, skipping I2V frames, existing "
                "guides, and detailing keyframes already on the cond. Connected keyframes "
                "are content only and are re-placed on this canvas unless frame_indices is "
                "set. Pull them back out with LTXV Separate Generated Keyframes. Requires a "
                "checkpoint trained for generated keyframes (one carrying "
                "keyframes_abs_pos_embedding)."
            ),
            inputs=[
                io.Conditioning.Input(
                    "positive",
                    tooltip="Positive conditioning the keyframes are attached to.",
                ),
                io.Conditioning.Input(
                    "negative",
                    tooltip="Negative conditioning the keyframes are attached to.",
                ),
                io.Vae.Input(
                    "vae", tooltip="Only used to read the latent scale factors."
                ),
                io.Latent.Input(
                    "latent",
                    tooltip=(
                        "Plain 5D video latent to generate keyframes alongside. Add them "
                        "before Concat AV Latent."
                    ),
                ),
                io.Int.Input(
                    "interval_frames",
                    optional=True,
                    default=24,
                    min=1,
                    max=1024,
                    tooltip=(
                        "Pixel-frame stride for auto placement. Default 24 is about one "
                        "keyframe per second at 24 fps. Occupied pixels are skipped. "
                        "Ignored when frame_indices is set."
                    ),
                ),
                io.Latent.Input(
                    "keyframes",
                    optional=True,
                    tooltip=(
                        "Optional content to initialize the new keyframes with. Connect "
                        "keyframes from an earlier Separate (same spatial size), or a "
                        "plain video latent to copy the nearest frame at each new slot "
                        "(e.g. after temporal upscale). These are still denoised, not "
                        "pinned as guides. Recorded indices on a keyframes latent are "
                        "ignored unless frame_indices is set. Only has an effect when "
                        "sampling starts below sigma 1."
                    ),
                ),
                io.String.Input(
                    "frame_indices",
                    optional=True,
                    default="",
                    tooltip=(
                        "Optional pixel-frame indices. Leave empty to place from "
                        "interval_frames on the current canvas. When set, this list is "
                        "the placement (connected keyframes are matched in order). The "
                        "last frame is allowed; frame 0 is not (it is already a "
                        "standalone token)."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    display_name="positive",
                    tooltip="Positive conditioning with generated-keyframe attention attached.",
                ),
                io.Conditioning.Output(
                    display_name="negative",
                    tooltip="Negative conditioning with generated-keyframe attention attached.",
                ),
                io.Latent.Output(
                    display_name="latent",
                    tooltip="Video latent with generated keyframes appended on T.",
                ),
            ],
        )

    @classmethod
    def parse_frame_indices(cls, frame_indices, num_pixel_frames, expected_count=None):
        """Pixel frame positions from a manual override.

        Frame 0 is excluded (already a standalone token). The terminal frame is
        allowed so a DFR segment grid that includes N-1 is legal.
        """
        first, last = 1, num_pixel_frames - 1
        if last < first:
            raise ValueError(
                f"A {num_pixel_frames}-frame target has no pixel frames to place keyframes on."
            )
        return _parse_frame_index_list(
            frame_indices,
            "frame_indices",
            expected_count,
            first,
            last,
            expected_desc=(
                f"{expected_count} keyframe(s)"
                if expected_count is not None
                else "the list sets the count"
            ),
            empty_hint="to place them from interval_frames",
        )

    @classmethod
    def keyframe_coords(cls, latent, frame_index, scale_factors):
        """Pixel coordinates of one keyframe: the full spatial grid over [t, t + 1)."""
        _, latent_coords = cls.PATCHIFIER.patchify(latent[:, :, :1])
        pixel_coords = latent_to_pixel_coords(latent_coords, scale_factors, causal_fix=True)
        pixel_coords[:, 0] += frame_index
        return pixel_coords

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        latent,
        interval_frames=24,
        keyframes=None,
        frame_indices="",
    ) -> io.NodeOutput:
        samples = latent["samples"]
        if not torch.is_tensor(samples) or samples.ndim != 5:
            raise ValueError(
                "Generated keyframes must be added to a plain video latent. Add them before "
                "merging the video and audio latents with Concat AV Latent."
            )

        existing_record = get_generated_keyframes(positive)
        if existing_record is not None:
            prev_tokens_per_frame = existing_record["tokens_per_frame"]
            if prev_tokens_per_frame != samples.shape[3] * samples.shape[4]:
                raise ValueError(
                    f"The existing generated keyframes were added at {prev_tokens_per_frame} tokens per latent "
                    f"frame but this latent has {samples.shape[3] * samples.shape[4]}. The latent was rescaled "
                    "after they were added, so more keyframes cannot be appended to them."
                )
            prev_first = existing_record["first_latent_frame"]
            prev_count = existing_record["num_keyframes"]
            if prev_first + prev_count != samples.shape[2]:
                raise ValueError(
                    f"The existing generated keyframes end at latent frame {prev_first + prev_count} but the "
                    f"latent has {samples.shape[2]}. Something was appended after them, so more keyframes would "
                    "not be contiguous with the existing block. Add all the keyframes before any guides."
                )

        scale_factors = vae.downscale_index_formula
        time_scale_factor = scale_factors[0]
        keyframe_idxs, num_guide_frames = get_keyframe_idxs(positive, samples.shape)
        num_target_frames = samples.shape[2] - num_guide_frames
        num_pixel_frames = (num_target_frames - 1) * time_scale_factor + 1
        occupied = occupied_pixel_frames(
            latent,
            time_scale_factor,
            num_pixel_frames,
            video_latent_frames=num_target_frames,
        )
        occupied |= pixel_frames_from_keyframe_idxs(keyframe_idxs)
        prev_indices = list(existing_record["frame_indices"]) if existing_record is not None else []
        occupied |= set(prev_indices)

        if frame_indices and str(frame_indices).strip():
            indices = cls.parse_frame_indices(frame_indices, num_pixel_frames)
        else:
            indices = free_detailing_slots(num_pixel_frames, float(interval_frames), occupied)

        clashes = sorted(set(indices) & occupied)
        if clashes:
            raise ValueError(
                f"frame_indices reuses pixel frame(s) {', '.join(str(i) for i in clashes)}, which already hold "
                "an image keyframe, a guide, or a generated keyframe. Each keyframe needs its own pixel frame."
            )

        num_keyframes = len(indices)
        if keyframes is not None:
            keyframe_samples = keyframes["samples"]
            if keyframe_samples.ndim != 5:
                raise ValueError(
                    f"The keyframes latent must be 5 dimensional, got {list(keyframe_samples.shape)}."
                )
            recorded_positions = keyframes.get("generated_keyframe_indices")
            batched_singles = (
                keyframe_samples.shape[2] == 1
                and keyframe_samples.shape[0] != samples.shape[0]
            )
            if batched_singles:
                if keyframe_samples.shape[0] % samples.shape[0] != 0:
                    raise ValueError(
                        f"The keyframes latent batch ({keyframe_samples.shape[0]}) is not a "
                        f"multiple of the video latent batch ({samples.shape[0]}), so it "
                        "cannot be reshaped into per-video keyframes."
                    )
                stacked = keyframe_samples.shape[0] // samples.shape[0]
                keyframe_samples = keyframe_samples.reshape(
                    samples.shape[0],
                    stacked,
                    keyframe_samples.shape[1],
                    *keyframe_samples.shape[3:],
                ).movedim(1, 2)
                batched_singles = False
            if should_copy_nearest_video_frames(
                keyframe_samples.shape[2],
                num_keyframes,
                recorded_positions is not None,
                batched_singles,
            ):
                keyframe_samples = keyframes_from_video(
                    keyframe_samples,
                    indices,
                    time_scale_factor,
                )
            else:
                keyframe_samples = _fit_keyframe_samples(
                    keyframe_samples.to(samples), samples, num_keyframes
                )
            keyframe_samples = keyframe_samples.to(samples)
        else:
            keyframe_samples = torch.zeros(
                (
                    samples.shape[0],
                    samples.shape[1],
                    num_keyframes,
                    samples.shape[3],
                    samples.shape[4],
                ),
                dtype=samples.dtype,
                device=samples.device,
            )

        generated_coords = torch.cat(
            [cls.keyframe_coords(samples, index, scale_factors) for index in indices],
            dim=2,
        )
        if keyframe_idxs is not None:
            generated_coords = torch.cat([keyframe_idxs, generated_coords.to(keyframe_idxs)], dim=2)

        existing_entries = conditioning_get_any_value(positive, "guide_attention_entries", None) or []
        generated_keyframes = {
            "first_latent_frame": (
                existing_record["first_latent_frame"] if existing_record else samples.shape[2]
            ),
            "num_keyframes": (
                existing_record["num_keyframes"] + num_keyframes if existing_record else num_keyframes
            ),
            "frame_indices": prev_indices + list(indices),
            "num_pixel_frames": num_pixel_frames,
            "guide_entry_index": (
                existing_record["guide_entry_index"] if existing_record else len(existing_entries)
            ),
            "tokens_per_frame": samples.shape[3] * samples.shape[4],
        }

        values = {
            "keyframe_idxs": generated_coords,
            "generated_keyframes": generated_keyframes,
        }
        positive = node_helpers.conditioning_set_values(positive, values)
        negative = node_helpers.conditioning_set_values(negative, values)

        if existing_record is not None:
            positive, negative = _grow_guide_attention_entry(
                positive,
                negative,
                existing_record["guide_entry_index"],
                extra_pre_filter_count=num_keyframes * samples.shape[3] * samples.shape[4],
                extra_frames=num_keyframes,
            )
        else:
            positive, negative = _append_guide_attention_entry(
                positive,
                negative,
                pre_filter_count=num_keyframes * samples.shape[3] * samples.shape[4],
                latent_shape=[num_keyframes, samples.shape[3], samples.shape[4]],
                strength=1.0,
            )

        noise_mask = get_noise_mask(latent)
        keyframe_noise_mask = torch.ones(
            (
                noise_mask.shape[0],
                1,
                num_keyframes,
                noise_mask.shape[3],
                noise_mask.shape[4],
            ),
            dtype=noise_mask.dtype,
            device=noise_mask.device,
        )

        out = latent.copy()
        out["samples"] = torch.cat([samples, keyframe_samples], dim=2)
        out["noise_mask"] = torch.cat([noise_mask, keyframe_noise_mask], dim=2)
        return io.NodeOutput(positive, negative, out)


class LTXVSeparateGeneratedKeyframes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVSeparateGeneratedKeyframes",
            display_name="LTXV Separate Generated Keyframes",
            category="model/conditioning/ltxv",
            search_aliases=["detailing", "dfr", "peel keyframes"],
            description=(
                "Split the generated keyframes added by LTXV Add Generated Keyframes "
                "back out of a sampled latent, and remove them from the conditioning. "
                "Separate them before spatially upscaling the video latent. Do not run "
                "LTXV Crop Guides first — it treats generated keyframes as disposable "
                "guides and drops them."
            ),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent"),
                io.Boolean.Input(
                    "keyframes_to_batch",
                    default=False,
                    tooltip=(
                        "Return the keyframes as a batch of single-frame latents. Leave off "
                        "to get them as one multi-frame latent, which is what the latent "
                        "upsampler and a later Add Generated Keyframes expect."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    display_name="positive",
                    tooltip="Positive conditioning with generated-keyframe metadata removed.",
                ),
                io.Conditioning.Output(
                    display_name="negative",
                    tooltip="Negative conditioning with generated-keyframe metadata removed.",
                ),
                io.Latent.Output(
                    display_name="latent",
                    tooltip="Video latent with the generated keyframes stripped.",
                ),
                io.Latent.Output(
                    display_name="keyframes",
                    tooltip=(
                        "The peeled keyframes, labeled with generated_keyframe_indices and "
                        "generated_keyframe_num_frames. Feed these to a later Add Generated "
                        "Keyframes to initialize new slots, or to Generated Keyframes To "
                        "Guides to pin them as frozen image guides (indices are remapped "
                        "if the canvas length changed)."
                    ),
                ),
            ],
        )

    @classmethod
    def strip_keyframe_idxs(cls, cond, first_token, num_tokens):
        keyframe_idxs = conditioning_get_any_value(cond, "keyframe_idxs", None)
        if keyframe_idxs is None:
            return None
        remaining = torch.cat(
            [
                keyframe_idxs[:, :, :first_token],
                keyframe_idxs[:, :, first_token + num_tokens :],
            ],
            dim=2,
        )
        return remaining if remaining.shape[2] > 0 else None

    @classmethod
    def strip_guide_entry(cls, cond, entry_index):
        entries = conditioning_get_any_value(cond, "guide_attention_entries", None)
        if not entries:
            return None
        if entry_index >= len(entries):
            raise ValueError(
                f"The generated keyframes recorded guide entry {entry_index} but the conditioning only has "
                f"{len(entries)}. The conditioning was rebuilt after they were added."
            )
        remaining = entries[:entry_index] + entries[entry_index + 1 :]
        return remaining or None

    @classmethod
    def execute(cls, positive, negative, latent, keyframes_to_batch=False) -> io.NodeOutput:
        generated_keyframes = get_generated_keyframes(positive)
        if generated_keyframes is None:
            raise ValueError(
                "This latent has no generated keyframes. Add them with LTXV Add Generated Keyframes first."
            )

        samples = latent["samples"]
        if not torch.is_tensor(samples) or samples.ndim != 5:
            raise ValueError(
                "Generated keyframes must be separated from a plain video latent. Split the video and "
                "audio latents with Separate AV Latent first."
            )

        tokens_per_frame = samples.shape[3] * samples.shape[4]
        if generated_keyframes["tokens_per_frame"] != tokens_per_frame:
            raise ValueError(
                f"The generated keyframes were added at {generated_keyframes['tokens_per_frame']} tokens per "
                f"latent frame but this latent has {tokens_per_frame}. The latent was rescaled after they were "
                "added, so the keyframes no longer line up. Separate them before upscaling the latent."
            )

        first_frame = generated_keyframes["first_latent_frame"]
        num_keyframes = generated_keyframes["num_keyframes"]
        end_frame = first_frame + num_keyframes
        if end_frame > samples.shape[2]:
            raise ValueError(
                f"The generated keyframes span latent frames [{first_frame}, {end_frame}) but the latent "
                f"only has {samples.shape[2]}. It was recorded against a different latent."
            )

        keyframe_samples = samples[:, :, first_frame:end_frame].clone()
        if keyframes_to_batch:
            batch, channels, _, height, width = keyframe_samples.shape
            keyframe_samples = keyframe_samples.movedim(2, 1).reshape(
                batch * num_keyframes, channels, 1, height, width
            )

        video_samples = torch.cat(
            [samples[:, :, :first_frame], samples[:, :, end_frame:]], dim=2
        )
        noise_mask = get_noise_mask(latent)
        video_noise_mask = torch.cat(
            [noise_mask[:, :, :first_frame], noise_mask[:, :, end_frame:]], dim=2
        )

        _, num_guide_frames = get_keyframe_idxs(positive, samples.shape)
        first_token = (first_frame - (samples.shape[2] - num_guide_frames)) * tokens_per_frame
        entry_index = generated_keyframes["guide_entry_index"]

        outputs = []
        for cond in (positive, negative):
            outputs.append(
                node_helpers.conditioning_set_values(
                    cond,
                    {
                        "keyframe_idxs": cls.strip_keyframe_idxs(
                            cond, first_token, num_keyframes * tokens_per_frame
                        ),
                        "guide_attention_entries": cls.strip_guide_entry(cond, entry_index),
                        "generated_keyframes": None,
                    },
                )
            )

        out = latent.copy()
        out["samples"] = video_samples
        out["noise_mask"] = video_noise_mask
        num_pixel_frames = generated_keyframes.get("num_pixel_frames")
        if num_pixel_frames is None:
            num_pixel_frames = (first_frame - 1) * DEFAULT_TEMPORAL_SCALE + 1
        keyframes_out = {
            "samples": keyframe_samples,
            "generated_keyframe_indices": list(generated_keyframes["frame_indices"]),
            "generated_keyframe_num_frames": int(num_pixel_frames),
        }
        return io.NodeOutput(outputs[0], outputs[1], out, keyframes_out)


class LTXVGeneratedKeyframesToGuides(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVGeneratedKeyframesToGuides",
            display_name="LTXV Generated Keyframes to Guides",
            category="model/conditioning/ltxv",
            search_aliases=["detailing", "dfr", "keyframe guides"],
            description=(
                "Pin generated keyframes from an earlier stage as frozen image guides on "
                "a later canvas. They are decoded as standalone frames, resized if needed, "
                "and written with noise_mask=0 so they are not denoised again. After a "
                "temporal upscale, recorded indices are scaled from the canvas they were "
                "generated on onto this one (same moments). Use override_frame_indices to "
                "set positions explicitly. To keep generating (and denoising) keyframes at "
                "new positions, use Add Generated Keyframes instead."
            ),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Latent.Input(
                    "latent",
                    tooltip="The target video latent to add the guides to, e.g. the temporally upscaled one.",
                ),
                io.Latent.Input(
                    "keyframes",
                    tooltip=(
                        "The keyframes output of LTXV Separate Generated Keyframes, which "
                        "carries the pixel frame index each keyframe was generated at."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="Guide strength. 1.0 is a hard pin; lower values relax it.",
                ),
                io.String.Input(
                    "override_frame_indices",
                    optional=True,
                    default="",
                    tooltip=(
                        "Optional — pin at these pixel frames instead of the recorded "
                        "(or auto-scaled) positions. Provide one index per keyframe. "
                        "Leave empty to reuse recorded positions, or to scale them when "
                        "the target canvas is a different length (e.g. after temporal x2)."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    display_name="positive",
                    tooltip="Positive conditioning with the keyframes pinned as image guides.",
                ),
                io.Conditioning.Output(
                    display_name="negative",
                    tooltip="Negative conditioning with the keyframes pinned as image guides.",
                ),
                io.Latent.Output(
                    display_name="latent",
                    tooltip="Target video latent with the keyframes added as frozen guides.",
                ),
            ],
        )

    @classmethod
    def decode_single_frames(cls, vae, keyframe_samples):
        """Decode each keyframe latent on its own, never as one clip."""
        if keyframe_samples.shape[2] != 1:
            batch, channels, num_keyframes, height, width = keyframe_samples.shape
            keyframe_samples = keyframe_samples.movedim(2, 1).reshape(
                batch * num_keyframes, channels, 1, height, width
            )
        images = vae.decode(keyframe_samples)
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        return images

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        latent,
        keyframes,
        strength,
        override_frame_indices="",
    ) -> io.NodeOutput:
        indices = keyframes.get("generated_keyframe_indices", None)
        if indices is None:
            raise ValueError(
                "This latent does not carry generated keyframe positions. Connect the keyframes output "
                "of LTXV Separate Generated Keyframes."
            )
        if get_generated_keyframes(positive) is not None:
            raise ValueError(
                "This conditioning still carries generated keyframes. Connect the positive and "
                "negative outputs of LTXV Separate Generated Keyframes."
            )

        samples = latent["samples"]
        if not torch.is_tensor(samples) or samples.ndim != 5:
            raise ValueError(
                "Generated keyframe guides must be added to a plain video latent. Add them before "
                "merging the video and audio latents with Concat AV Latent."
            )
        if samples.shape[0] != 1:
            raise ValueError(
                f"Only a batch size of 1 is supported, got {samples.shape[0]}. Each guide is encoded from "
                "one image, so it cannot differ across batch elements."
            )

        kf_samples = keyframes["samples"]
        if kf_samples.shape[2] != 1:
            batch, channels, num_keyframes, height, width = kf_samples.shape
            kf_samples = kf_samples.movedim(2, 1).reshape(
                batch * num_keyframes, channels, 1, height, width
            )

        resize_needed = kf_samples.shape[3:] != samples.shape[3:]
        guides = cls.decode_single_frames(vae, kf_samples) if resize_needed else kf_samples
        if guides.shape[0] != len(indices):
            raise ValueError(
                f"Got {guides.shape[0]} keyframes for {len(indices)} recorded positions."
            )

        _, num_guide_frames = get_keyframe_idxs(positive, samples.shape)
        time_scale_factor = vae.downscale_index_formula[0]
        num_pixel_frames = (samples.shape[2] - num_guide_frames - 1) * time_scale_factor + 1
        if override_frame_indices and str(override_frame_indices).strip():
            indices = _parse_frame_index_list(
                override_frame_indices,
                "override_frame_indices",
                len(indices),
                1,
                num_pixel_frames - 1,
                expected_desc=f"the keyframes latent carries {len(indices)}",
                empty_hint="to reuse the recorded positions",
            )
        else:
            old_len = keyframes.get("generated_keyframe_num_frames")
            if old_len is not None and int(old_len) != num_pixel_frames:
                indices = scale_frame_indices(list(indices), int(old_len), num_pixel_frames)
        if indices and max(indices) >= num_pixel_frames:
            raise ValueError(
                f"Keyframe position {max(indices)} is outside this latent's {num_pixel_frames} frames. The "
                "target was resized temporally after the keyframes were generated."
            )

        for index, frame_idx in enumerate(indices):
            if resize_needed:
                added = LTXVAddGuide.execute(
                    positive,
                    negative,
                    vae,
                    latent,
                    guides[index].unsqueeze(0),
                    int(frame_idx),
                    strength,
                )
                positive, negative, latent = added[0], added[1], added[2]
            else:
                positive, negative, latent = cls.append_latent_keyframe(
                    positive,
                    negative,
                    vae,
                    latent,
                    guides[index : index + 1],
                    int(frame_idx),
                    strength,
                )

        return io.NodeOutput(positive, negative, latent)

    @classmethod
    def append_latent_keyframe(cls, positive, negative, vae, latent, guiding_latent, frame_idx, strength):
        """Append an already encoded keyframe latent the same way LTXVAddGuide would after encoding."""
        noise_mask = get_noise_mask(latent)
        positive, negative, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
            positive,
            negative,
            frame_idx,
            latent["samples"],
            noise_mask,
            guiding_latent,
            strength,
            vae.downscale_index_formula,
        )
        positive, negative = _append_guide_attention_entry(
            positive,
            negative,
            pre_filter_count=math.prod(guiding_latent.shape[2:]),
            latent_shape=list(guiding_latent.shape[2:]),
            strength=strength,
        )
        out = latent.copy()
        out["samples"] = latent_image
        out["noise_mask"] = noise_mask
        return positive, negative, out


class LTXVFreezeLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVFreezeLatent",
            display_name="LTXV Freeze Latent",
            category="model/latent/ltxv",
            search_aliases=["noise mask", "freeze audio", "freeze video"],
            description=(
                "Set noise_mask to 0 so this latent is kept clean during sampling. "
                "Works on video or audio. Typical uses: freeze audio before Concat AV "
                "so it only provides cross-attention, or freeze any latent that should "
                "not be denoised."
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip="Video or audio latent to freeze. Audio is 4D; video is 5D.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, latent) -> io.NodeOutput:
        samples = latent["samples"]
        if not torch.is_tensor(samples):
            raise ValueError(
                "Freeze Latent expects a plain tensor, not a concatenated AV latent. "
                "Split with Separate AV Latent first."
            )
        out = latent.copy()
        if samples.ndim == 5:
            batch, _, frames, _, _ = samples.shape
            out["noise_mask"] = torch.zeros(
                (batch, 1, frames, 1, 1),
                dtype=torch.float32,
                device=samples.device,
            )
        elif samples.ndim == 4:
            batch, _, frames, _ = samples.shape
            out["noise_mask"] = torch.zeros(
                (batch, 1, frames, 1),
                dtype=torch.float32,
                device=samples.device,
            )
        else:
            raise ValueError(
                f"Expected a 4D audio or 5D video latent, got shape {list(samples.shape)}."
            )
        return io.NodeOutput(out)


class LTXVKeyframesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LTXVAddGeneratedKeyframes,
            LTXVSeparateGeneratedKeyframes,
            LTXVGeneratedKeyframesToGuides,
            LTXVFreezeLatent,
        ]


async def comfy_entrypoint() -> LTXVKeyframesExtension:
    return LTXVKeyframesExtension()
