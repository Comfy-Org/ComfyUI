from __future__ import annotations
from typing import TYPE_CHECKING, Callable
import torch
import numpy as np
import collections
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import comfy.model_management
import comfy.patcher_extension
import comfy.utils
import comfy.conds
if TYPE_CHECKING:
    from comfy.model_base import BaseModel
    from comfy.model_patcher import ModelPatcher
    from comfy.controlnet import ControlBase


class ContextWindowABC(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def get_tensor(self, full: torch.Tensor) -> torch.Tensor:
        """
        Get torch.Tensor applicable to current window.
        """
        raise NotImplementedError("Not implemented.")

    @abstractmethod
    def add_window(self, full: torch.Tensor, to_add: torch.Tensor) -> torch.Tensor:
        """
        Apply torch.Tensor of window to the full tensor, in place. Returns reference to updated full tensor, not a copy.
        """
        raise NotImplementedError("Not implemented.")

class ContextHandlerABC(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def should_use_context(self, model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep: torch.Tensor, model_options: dict[str]) -> bool:
        raise NotImplementedError("Not implemented.")

    @abstractmethod
    def get_resized_cond(self, cond_in: list[dict], x_in: torch.Tensor, window: ContextWindowABC, device=None) -> list:
        raise NotImplementedError("Not implemented.")

    @abstractmethod
    def execute(self, calc_cond_batch: Callable, model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep: torch.Tensor, model_options: dict[str]):
        raise NotImplementedError("Not implemented.")



class IndexListContextWindow(ContextWindowABC):
    def __init__(self, index_list: list[int], dim: int=0, total_frames: int=0, modality_windows: dict=None, context_overlap: int=0):
        self.index_list = index_list
        self.context_length = len(index_list)
        self.context_overlap = context_overlap
        self.dim = dim
        self.total_frames = total_frames
        self.center_ratio = (min(index_list) + max(index_list)) / (2 * total_frames)
        self.modality_windows = modality_windows  # dict of {mod_idx: IndexListContextWindow}
        self.guide_frames_indices: list[int] = []
        self.guide_overlap_info: list[tuple[int, int]] = []
        self.guide_kf_local_positions: list[int] = []
        self.guide_downscale_factors: list[int] = []

    def get_tensor(self, full: torch.Tensor, device=None, dim=None, retain_index_list=[]) -> torch.Tensor:
        if dim is None:
            dim = self.dim
        if dim == 0 and full.shape[dim] == 1:
            return full
        indices = self.index_list
        anchor_idx = getattr(self, 'causal_anchor_index', None)
        if anchor_idx is not None and anchor_idx >= 0:
            indices = [anchor_idx] + list(indices)
        idx = tuple([slice(None)] * dim + [indices])
        window = full[idx]
        if retain_index_list:
            idx = tuple([slice(None)] * dim + [retain_index_list])
            window[idx] = full[idx]
        return window.to(device)

    def add_window(self, full: torch.Tensor, to_add: torch.Tensor, dim=None) -> torch.Tensor:
        if dim is None:
            dim = self.dim
        idx = tuple([slice(None)] * dim + [self.index_list])
        full[idx] += to_add
        return full

    def get_region_index(self, num_regions: int) -> int:
        region_idx = int(self.center_ratio * num_regions)
        return min(max(region_idx, 0), num_regions - 1)

    def get_window_for_modality(self, modality_idx: int) -> 'IndexListContextWindow':
        if modality_idx == 0:
            return self
        return self.modality_windows[modality_idx]


class IndexListCallbacks:
    EVALUATE_CONTEXT_WINDOWS = "evaluate_context_windows"
    COMBINE_CONTEXT_WINDOW_RESULTS = "combine_context_window_results"
    EXECUTE_START = "execute_start"
    EXECUTE_CLEANUP = "execute_cleanup"
    RESIZE_COND_ITEM = "resize_cond_item"

    def init_callbacks(self):
        return {}


def slice_cond(cond_value, window: IndexListContextWindow, x_in: torch.Tensor, device, temporal_dim: int, temporal_scale: int=1, temporal_offset: int=0, retain_index_list: list[int]=[]):
    if not (hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor)):
        return None
    cond_tensor = cond_value.cond
    if temporal_dim >= cond_tensor.ndim:
        return None

    cond_size = cond_tensor.size(temporal_dim)

    if temporal_scale == 1:
        expected_size = x_in.size(window.dim) - temporal_offset
        if cond_size != expected_size:
            return None

    if temporal_offset == 0 and temporal_scale == 1:
        sliced = window.get_tensor(cond_tensor, device, dim=temporal_dim, retain_index_list=retain_index_list)
        return cond_value._copy_with(sliced)

    # skip leading latent positions that have no corresponding conditioning (e.g. reference frames)
    if temporal_offset > 0:
        anchor_idx = getattr(window, 'causal_anchor_index', None)
        if anchor_idx is not None and anchor_idx >= 0:
            # anchor occupies one of the no-cond positions, so skip one fewer from window.index_list
            skip_count = temporal_offset - 1
        else:
            skip_count = temporal_offset

        indices = [i - temporal_offset for i in window.index_list[skip_count:]]
        indices = [i for i in indices if 0 <= i]
    else:
        indices = list(window.index_list)

    if not indices:
        return None

    if temporal_scale > 1:
        scaled = []
        for i in indices:
            for k in range(temporal_scale):
                si = i * temporal_scale + k
                if si < cond_size:
                    scaled.append(si)
        indices = scaled
        if not indices:
            return None

    idx = tuple([slice(None)] * temporal_dim + [indices])
    sliced = cond_tensor[idx].to(device)
    return cond_value._copy_with(sliced)


def compute_guide_overlap(guide_entries: list[dict], keyframe_idxs: torch.Tensor, temporal_downscale_ratio: int, window_index_list: list[int]):
    """Compute which concatenated guide frames overlap with a context window.

    Each guide's latent-space start is derived from its first token's pixel-t-start
    in keyframe_idxs (shape (B, [t,h,w], num_tokens, [start, end])), divided by the
    model's temporal_downscale_ratio.

    Args:
        guide_entries: list of guide_attention_entry dicts
        keyframe_idxs: per-token pixel coords cond tensor for the modality
        temporal_downscale_ratio: model's pixel-to-latent temporal compression ratio
        window_index_list: the window's frame indices into the video portion

    Returns:
        suffix_indices: indices into the guide_frames tensor for frame selection
        overlap_info: list of (entry_idx, overlap_count) for guide_attention_entries adjustment
        kf_local_positions: window-local frame positions for keyframe_idxs regeneration
        total_overlap: total number of overlapping guide frames
    """
    window_set = set(window_index_list)
    window_list = list(window_index_list)
    suffix_indices = []
    overlap_info = []
    kf_local_positions = []
    suffix_base = 0
    token_offset = 0

    for entry_idx, entry in enumerate(guide_entries):
        first_t_pixel = int(keyframe_idxs[0, 0, token_offset, 0].item())
        latent_start = (first_t_pixel + temporal_downscale_ratio - 1) // temporal_downscale_ratio
        guide_len = entry["latent_shape"][0]
        entry_overlap = 0

        for local_offset in range(guide_len):
            video_pos = latent_start + local_offset
            if video_pos in window_set:
                suffix_indices.append(suffix_base + local_offset)
                kf_local_positions.append(window_list.index(video_pos))
                entry_overlap += 1

        if entry_overlap > 0:
            overlap_info.append((entry_idx, entry_overlap))
        suffix_base += guide_len
        token_offset += entry["pre_filter_count"]

    return suffix_indices, overlap_info, kf_local_positions, len(suffix_indices)


@dataclass
class WindowingState:
    """Per-modality context windowing state for each step,
    built using IndexListContextHandler._build_window_state().
    For non-multimodal models the lists are length 1
    """
    latents: list[torch.Tensor]                  # per-modality working latents (guide frames stripped)
    guide_latents: list[torch.Tensor | None]     # per-modality guide frames stripped from latents
    guide_entries: list[list[dict] | None]       # per-modality guide_attention_entry metadata
    keyframe_idxs: list[torch.Tensor | None]     # per-modality keyframe_idxs tensor for guide latent_start derivation
    latent_shapes: list | None                   # original packed shapes for unpack/pack (None if not multimodal)
    dim: int = 0                                 # primary modality temporal dim for context windowing
    is_multimodal: bool = False
    temporal_downscale_ratio: int = 1            # model's pixel-to-latent temporal compression ratio

    def prepare_window(self, window: IndexListContextWindow, model) -> IndexListContextWindow:
        """Reformat window for multimodal contexts by deriving per-modality index lists.
        Non-multimodal contexts return the input window unchanged.
        """
        if not self.is_multimodal:
            return window

        x = self.latents[0]
        primary_total = self.latent_shapes[0][self.dim]
        primary_overlap = window.context_overlap
        map_shapes = self.latent_shapes
        if x.size(self.dim) != primary_total:
            map_shapes = list(self.latent_shapes)
            video_shape = list(self.latent_shapes[0])
            video_shape[self.dim] = x.size(self.dim)
            map_shapes[0] = torch.Size(video_shape)
        try:
            per_modality_indices = model.map_context_window_to_modalities(
                window.index_list, map_shapes, self.dim)
        except AttributeError:
            raise NotImplementedError(
                f"{type(model).__name__} must implement map_context_window_to_modalities for multimodal context windows.")
        modality_windows = {}
        for mod_idx in range(1, len(self.latents)):
            modality_total_frames = self.latents[mod_idx].shape[self.dim]
            ratio = modality_total_frames / primary_total if primary_total > 0 else 1
            modality_overlap = max(round(primary_overlap * ratio), 0)
            modality_windows[mod_idx] = IndexListContextWindow(
                per_modality_indices[mod_idx], dim=self.dim,
                total_frames=modality_total_frames,
                context_overlap=modality_overlap)
        return IndexListContextWindow(
            window.index_list, dim=self.dim, total_frames=x.shape[self.dim],
            modality_windows=modality_windows, context_overlap=primary_overlap)

    def slice_for_window(self, window: IndexListContextWindow, retain_index_list: list[int], device=None) -> tuple[list[torch.Tensor], list[int]]:
        """Slice latents for a context window, injecting guide frames where applicable.
        For multimodal contexts, uses the modality-specific windows derived in prepare_window().
        """
        sliced = []
        guide_frame_counts = []
        for idx in range(len(self.latents)):
            modality_window = window.get_window_for_modality(idx)
            retain = retain_index_list if idx == 0 else []
            s = modality_window.get_tensor(self.latents[idx], device, retain_index_list=retain)
            if self.guide_entries[idx] is not None:
                s, ng = self._inject_guide_frames(s, modality_window, modality_idx=idx)
            else:
                ng = 0
            sliced.append(s)
            guide_frame_counts.append(ng)
        return sliced, guide_frame_counts

    def strip_guide_frames(self, out_per_modality: list[list[torch.Tensor]], guide_frame_counts: list[int], window: IndexListContextWindow):
        """Strip injected guide frames from per-cond, per-modality outputs in place."""
        for idx in range(len(self.latents)):
            if guide_frame_counts[idx] > 0:
                window_len = len(window.get_window_for_modality(idx).index_list)
                for ci in range(len(out_per_modality)):
                    out_per_modality[ci][idx] = out_per_modality[ci][idx].narrow(self.dim, 0, window_len)

    def _inject_guide_frames(self, latent_slice: torch.Tensor, window: IndexListContextWindow, modality_idx: int = 0) -> tuple[torch.Tensor, int]:
        guide_entries = self.guide_entries[modality_idx]
        guide_frames = self.guide_latents[modality_idx]
        keyframe_idxs = self.keyframe_idxs[modality_idx]
        suffix_idx, overlap_info, kf_local_pos, guide_frame_count = compute_guide_overlap(
            guide_entries, keyframe_idxs, self.temporal_downscale_ratio, window.index_list)
        # Shift keyframe positions to account for causal_window_fix anchor occupying sub-pos 0.
        anchor_idx = getattr(window, 'causal_anchor_index', None)
        if anchor_idx is not None and anchor_idx >= 0:
            kf_local_pos = [p + 1 for p in kf_local_pos]
        window.guide_frames_indices = suffix_idx
        window.guide_overlap_info = overlap_info
        window.guide_kf_local_positions = kf_local_pos

        # Derive per-overlap-entry latent_downscale_factor from guide entry latent_shape vs guide frame spatial dims.
        # guide_frames has full (post-dilation) spatial dims; entry["latent_shape"] has pre-dilation dims.
        guide_downscale_factors = []
        if guide_frame_count > 0:
            full_H = guide_frames.shape[3]
            for entry_idx, _ in overlap_info:
                entry_H = guide_entries[entry_idx]["latent_shape"][1]
                guide_downscale_factors.append(full_H // entry_H)
        window.guide_downscale_factors = guide_downscale_factors

        if guide_frame_count > 0:
            idx = tuple([slice(None)] * self.dim + [suffix_idx])
            return torch.cat([latent_slice, guide_frames[idx]], dim=self.dim), guide_frame_count
        return latent_slice, 0

    def patch_latent_shapes(self, sub_conds, new_shapes):
        if not self.is_multimodal:
            return

        for cond_list in sub_conds:
            if cond_list is None:
                continue
            for cond_dict in cond_list:
                model_conds = cond_dict.get('model_conds', {})
                if 'latent_shapes' in model_conds:
                    model_conds['latent_shapes'] = comfy.conds.CONDConstant(new_shapes)


@dataclass
class ContextSchedule:
    name: str
    func: Callable

@dataclass
class ContextFuseMethod:
    name: str
    func: Callable

ContextResults = collections.namedtuple("ContextResults", ['window_idx', 'sub_conds_out', 'sub_conds', 'window'])
class IndexListContextHandler(ContextHandlerABC):
    def __init__(self, context_schedule: ContextSchedule, fuse_method: ContextFuseMethod, context_length: int=1, context_overlap: int=0, context_stride: int=1,
                 closed_loop: bool=False, dim:int=0, freenoise: bool=False, cond_retain_index_list: list[int]=[], split_conds_to_windows: bool=False,
                 latent_retain_index_list: list[int]=[], causal_window_fix: bool=True):
        self.context_schedule = context_schedule
        self.fuse_method = fuse_method
        self.context_length = context_length
        self.context_overlap = context_overlap
        self.context_stride = context_stride
        self.closed_loop = closed_loop
        self.dim = dim
        self._step = 0
        self.freenoise = freenoise
        self.cond_retain_index_list = [int(x.strip()) for x in cond_retain_index_list.split(",")] if cond_retain_index_list else []
        self.split_conds_to_windows = split_conds_to_windows
        self.latent_retain_index_list = [int(x.strip()) for x in latent_retain_index_list.split(",")] if latent_retain_index_list else []
        self.causal_window_fix = causal_window_fix

        self.callbacks = {}

    @staticmethod
    def _get_latent_shapes(conds):
        for cond_list in conds:
            if cond_list is None:
                continue
            for cond_dict in cond_list:
                model_conds = cond_dict.get('model_conds', {})
                if 'latent_shapes' in model_conds:
                    return model_conds['latent_shapes'].cond
        return None

    @staticmethod
    def _get_guide_entries(conds):
        for cond_list in conds:
            if cond_list is None:
                continue
            for cond_dict in cond_list:
                model_conds = cond_dict.get('model_conds', {})
                entries = model_conds.get('guide_attention_entries')
                if entries is not None and hasattr(entries, 'cond') and entries.cond:
                    return entries.cond
        return None

    @staticmethod
    def _get_keyframe_idxs(conds):
        for cond_list in conds:
            if cond_list is None:
                continue
            for cond_dict in cond_list:
                model_conds = cond_dict.get('model_conds', {})
                kf = model_conds.get('keyframe_idxs')
                if kf is not None and hasattr(kf, 'cond') and kf.cond is not None:
                    return kf.cond
        return None

    def _apply_freenoise(self, noise: torch.Tensor, conds: list[list[dict]], seed: int) -> torch.Tensor:
        """Apply FreeNoise shuffling, scaling context length/overlap per-modality by frame ratio.
        If guide frames are present on the primary modality, only the video portion is shuffled.
        """
        guide_entries = self._get_guide_entries(conds)
        guide_count = sum(e["latent_shape"][0] for e in guide_entries) if guide_entries else 0

        latent_shapes = self._get_latent_shapes(conds)
        if latent_shapes is not None and len(latent_shapes) > 1:
            modalities = comfy.utils.unpack_latents(noise, latent_shapes)
            primary_total = latent_shapes[0][self.dim]
            primary_video_count = modalities[0].size(self.dim) - guide_count
            apply_freenoise(modalities[0].narrow(self.dim, 0, primary_video_count), self.dim, self.context_length, self.context_overlap, seed)
            for i in range(1, len(modalities)):
                mod_total = latent_shapes[i][self.dim]
                ratio = mod_total / primary_total if primary_total > 0 else 1
                mod_ctx_len = max(round(self.context_length * ratio), 1)
                mod_ctx_overlap = max(round(self.context_overlap * ratio), 0)
                modalities[i] = apply_freenoise(modalities[i], self.dim, mod_ctx_len, mod_ctx_overlap, seed)
            noise, _ = comfy.utils.pack_latents(modalities)
            return noise
        video_count = noise.size(self.dim) - guide_count
        apply_freenoise(noise.narrow(self.dim, 0, video_count), self.dim, self.context_length, self.context_overlap, seed)
        return noise

    def _build_window_state(self, x_in: torch.Tensor, conds: list[list[dict]], model: BaseModel) -> WindowingState:
        """Build windowing state for the current step, including unpacking latents and extracting guide frame info from conds."""
        latent_shapes = self._get_latent_shapes(conds)
        is_multimodal = latent_shapes is not None and len(latent_shapes) > 1
        unpacked_latents = comfy.utils.unpack_latents(x_in, latent_shapes) if is_multimodal else [x_in]

        unpacked_latents_list = list(unpacked_latents)
        guide_latents_list = [None] * len(unpacked_latents)
        guide_entries_list = [None] * len(unpacked_latents)
        keyframe_idxs_list = [None] * len(unpacked_latents)

        extracted_guide_entries = self._get_guide_entries(conds)
        extracted_keyframe_idxs = self._get_keyframe_idxs(conds)

        # Strip guide frames (only from first modality for now)
        if extracted_guide_entries is not None:
            guide_count = sum(e["latent_shape"][0] for e in extracted_guide_entries)
            if guide_count > 0:
                x = unpacked_latents[0]
                latent_count = x.size(self.dim) - guide_count
                unpacked_latents_list[0] = x.narrow(self.dim, 0, latent_count)
                guide_latents_list[0] = x.narrow(self.dim, latent_count, guide_count)
                guide_entries_list[0] = extracted_guide_entries
                keyframe_idxs_list[0] = extracted_keyframe_idxs


        return WindowingState(
            latents=unpacked_latents_list,
            guide_latents=guide_latents_list,
            guide_entries=guide_entries_list,
            keyframe_idxs=keyframe_idxs_list,
            latent_shapes=latent_shapes,
            dim=self.dim,
            is_multimodal=is_multimodal,
            temporal_downscale_ratio=model.latent_format.temporal_downscale_ratio)

    def should_use_context(self, model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep: torch.Tensor, model_options: dict[str]) -> bool:
        window_state = self._build_window_state(x_in, conds, model) # build window_state to check frame counts, will be built again in execute
        total_frame_count = window_state.latents[0].size(self.dim)
        if total_frame_count > self.context_length:
            logging.info(f"\nUsing context windows: Context length {self.context_length} with overlap {self.context_overlap} for {total_frame_count} frames.")
            if self.cond_retain_index_list:
                logging.info(f"Retaining original cond for indexes: {self.cond_retain_index_list}")
            if self.latent_retain_index_list:
                logging.info(f"Retaining original latent for indexes: {self.latent_retain_index_list}")
            return True
        logging.info(f"\nNot using context windows since context length ({self.context_length}) exceeds input frames ({total_frame_count}).")
        return False

    def prepare_control_objects(self, control: ControlBase, device=None) -> ControlBase:
        if control.previous_controlnet is not None:
            self.prepare_control_objects(control.previous_controlnet, device)
        return control

    def get_resized_cond(self, cond_in: list[dict], x_in: torch.Tensor, window: IndexListContextWindow, device=None) -> list:
        if cond_in is None:
            return None
        # reuse or resize cond items to match context requirements
        resized_cond = []
        # if multiple conds, split based on primary region
        if self.split_conds_to_windows and len(cond_in) > 1:
            region = window.get_region_index(len(cond_in))
            logging.info(f"Splitting conds to windows; using region {region} for window {window.index_list[0]}-{window.index_list[-1]} with center ratio {window.center_ratio:.3f}")
            cond_in = [cond_in[region]]
        # cond object is a list containing a dict - outer list is irrelevant, so just loop through it
        for actual_cond in cond_in:
            resized_actual_cond = actual_cond.copy()
            # now we are in the inner dict - "pooled_output" is a tensor, "control" is a ControlBase object, "model_conds" is dictionary
            for key in actual_cond:
                try:
                    cond_item = actual_cond[key]
                    if isinstance(cond_item, torch.Tensor):
                        # check that tensor is the expected length - x.size(0)
                        if self.dim < cond_item.ndim and cond_item.size(self.dim) == x_in.size(self.dim):
                            # if so, it's subsetting time - tell controls the expected indeces so they can handle them
                            actual_cond_item = window.get_tensor(cond_item)
                            resized_actual_cond[key] = actual_cond_item.to(device)
                        else:
                            resized_actual_cond[key] = cond_item.to(device)
                    # look for control
                    elif key == "control":
                        resized_actual_cond[key] = self.prepare_control_objects(cond_item, device)
                    elif isinstance(cond_item, dict):
                        new_cond_item = cond_item.copy()
                        # when in dictionary, look for tensors and CONDCrossAttn [comfy/conds.py] (has cond attr that is a tensor)
                        for cond_key, cond_value in new_cond_item.items():
                            # Allow callbacks to handle custom conditioning items
                            handled = False
                            for callback in comfy.patcher_extension.get_all_callbacks(
                                IndexListCallbacks.RESIZE_COND_ITEM, self.callbacks
                            ):
                                result = callback(cond_key, cond_value, window, x_in, device, new_cond_item)
                                if result is not None:
                                    new_cond_item[cond_key] = result
                                    handled = True
                                    break
                            if not handled and self._model is not None:
                                result = self._model.resize_cond_for_context_window(
                                    cond_key, cond_value, window, x_in, device,
                                    retain_index_list=self.cond_retain_index_list)
                                if result is not None:
                                    new_cond_item[cond_key] = result
                                    handled = True
                            if handled:
                                continue
                            if isinstance(cond_value, torch.Tensor):
                                if (self.dim < cond_value.ndim and cond_value.size(self.dim) == x_in.size(self.dim)) or \
                                   (cond_value.ndim < self.dim and cond_value.size(0) == x_in.size(self.dim)):
                                    new_cond_item[cond_key] = window.get_tensor(cond_value, device)
                            # Handle audio_embed (temporal dim is 1)
                            elif cond_key == "audio_embed" and hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
                                audio_cond = cond_value.cond
                                if audio_cond.ndim > 1 and audio_cond.size(1) == x_in.size(self.dim):
                                    new_cond_item[cond_key] = cond_value._copy_with(window.get_tensor(audio_cond, device, dim=1))
                            # Handle vace_context (temporal dim is 3)
                            elif cond_key == "vace_context" and hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
                                vace_cond = cond_value.cond
                                if vace_cond.ndim >= 4 and vace_cond.size(3) == x_in.size(self.dim):
                                    sliced_vace = window.get_tensor(vace_cond, device, dim=3, retain_index_list=self.cond_retain_index_list)
                                    new_cond_item[cond_key] = cond_value._copy_with(sliced_vace)
                            # if has cond that is a Tensor, check if needs to be subset
                            elif hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
                                if  (self.dim < cond_value.cond.ndim and cond_value.cond.size(self.dim) == x_in.size(self.dim)) or \
                                    (cond_value.cond.ndim < self.dim and cond_value.cond.size(0) == x_in.size(self.dim)):
                                    new_cond_item[cond_key] = cond_value._copy_with(window.get_tensor(cond_value.cond, device, retain_index_list=self.cond_retain_index_list))
                            elif cond_key == "num_video_frames": # for SVD
                                new_cond_item[cond_key] = cond_value._copy_with(cond_value.cond)
                                new_cond_item[cond_key].cond = window.context_length
                        resized_actual_cond[key] = new_cond_item
                    else:
                        resized_actual_cond[key] = cond_item
                finally:
                    del cond_item  # just in case to prevent VRAM issues
            resized_cond.append(resized_actual_cond)
        return resized_cond

    def set_step(self, timestep: torch.Tensor, model_options: dict[str]):
        sample_sigmas = model_options["transformer_options"]["sample_sigmas"]
        current_timestep = timestep[0].to(sample_sigmas.dtype)
        mask = torch.isclose(sample_sigmas, current_timestep, rtol=0.0001)
        matches = torch.nonzero(mask)
        if torch.numel(matches) == 0:
            return  # substep from multi-step sampler: keep self._step from the last full step
        self._step = int(matches[0].item())

    def get_context_windows(self, model: BaseModel, x_in: torch.Tensor, model_options: dict[str]) -> list[IndexListContextWindow]:
        full_length = x_in.size(self.dim) # TODO: choose dim based on model
        context_windows = self.context_schedule.func(full_length, self, model_options)
        context_windows = [IndexListContextWindow(window, dim=self.dim, total_frames=full_length, context_overlap=self.context_overlap) for window in context_windows]
        return context_windows

    def execute(self, calc_cond_batch: Callable, model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep: torch.Tensor, model_options: dict[str]):
        self._model = model
        self.set_step(timestep, model_options)

        window_state = self._build_window_state(x_in, conds, model)
        num_modalities = len(window_state.latents)

        context_windows = self.get_context_windows(model, window_state.latents[0], model_options)
        enumerated_context_windows = list(enumerate(context_windows))
        total_windows = len(enumerated_context_windows)

        # Initialize per-modality accumulators (length 1 for single-modality)
        accum = [[torch.zeros_like(m) for _ in conds] for m in window_state.latents]
        if self.fuse_method.name == ContextFuseMethods.RELATIVE:
            counts = [[torch.ones(get_shape_for_dim(m, self.dim), device=m.device) for _ in conds] for m in window_state.latents]
        else:
            counts = [[torch.zeros(get_shape_for_dim(m, self.dim), device=m.device) for _ in conds] for m in window_state.latents]
        biases = [[([0.0] * m.shape[self.dim]) for _ in conds] for m in window_state.latents]

        for callback in comfy.patcher_extension.get_all_callbacks(IndexListCallbacks.EXECUTE_START, self.callbacks):
            callback(self, model, x_in, conds, timestep, model_options)

        # accumulate results from each context window
        for enum_window in enumerated_context_windows:
            results = self.evaluate_context_windows(
                calc_cond_batch, model, x_in, conds, timestep, [enum_window],
                model_options, window_state=window_state, total_windows=total_windows)
            for result in results:
                # result.sub_conds_out is per-cond, per-modality: list[list[Tensor]]
                for mod_idx in range(num_modalities):
                    mod_out = [result.sub_conds_out[ci][mod_idx] for ci in range(len(conds))]
                    modality_window = result.window.get_window_for_modality(mod_idx)
                    self.combine_context_window_results(
                        window_state.latents[mod_idx], mod_out, result.sub_conds, modality_window,
                        result.window_idx, total_windows, timestep,
                        accum[mod_idx], counts[mod_idx], biases[mod_idx])

        # fuse accumulated results into final conds
        try:
            result_out = []
            for ci in range(len(conds)):
                finalized = []
                for mod_idx in range(num_modalities):
                    if self.fuse_method.name != ContextFuseMethods.RELATIVE:
                        accum[mod_idx][ci] /= counts[mod_idx][ci]
                    f = accum[mod_idx][ci]

                    # if guide frames were injected, append them to the end of the fused latents for the next step
                    if window_state.guide_latents[mod_idx] is not None:
                        f = torch.cat([f, window_state.guide_latents[mod_idx]], dim=self.dim)
                    finalized.append(f)

                # pack modalities together if needed
                if window_state.is_multimodal and len(finalized) > 1:
                    packed, _ = comfy.utils.pack_latents(finalized)
                else:
                    packed = finalized[0]

                result_out.append(packed)
            return result_out
        finally:
            for callback in comfy.patcher_extension.get_all_callbacks(IndexListCallbacks.EXECUTE_CLEANUP, self.callbacks):
                callback(self, model, x_in, conds, timestep, model_options)

    def evaluate_context_windows(self, calc_cond_batch: Callable, model: BaseModel, x_in: torch.Tensor, conds,
                                timestep: torch.Tensor, enumerated_context_windows: list[tuple[int, IndexListContextWindow]],
                                model_options, window_state: WindowingState, total_windows: int = None,
                                device=None, first_device=None):
        """Evaluate context windows and return per-cond, per-modality outputs in ContextResults.sub_conds_out

        For each window:
        1. Builds windows (for each modality if multimodal)
        2. Slices window for each modality
        3. Injects concatenated latent guide frames where present
        4. Packs together if needed and calls model
        5. Unpacks and strips any guides from outputs
        """
        x = window_state.latents[0]

        results: list[ContextResults] = []
        for window_idx, window in enumerated_context_windows:
            # allow processing to end between context window executions for faster Cancel
            comfy.model_management.throw_exception_if_processing_interrupted()

            # prepare the window accounting for multimodal windows
            window = window_state.prepare_window(window, model)

            # causal_window_fix: prepend a pre-window frame that will be stripped post-forward.
            # Set anchor before slice_for_window so the latent slice and downstream cond slices both pick it up.
            anchor_applied = False
            if self.causal_window_fix:
                anchor_idx = window.index_list[0] - 1
                if 0 <= anchor_idx < x_in.size(self.dim):
                    window.causal_anchor_index = anchor_idx
                    anchor_applied = True

            # slice the window for each modality, injecting guide frames where applicable
            sliced, guide_frame_counts_per_modality = window_state.slice_for_window(window, self.latent_retain_index_list, device)

            for callback in comfy.patcher_extension.get_all_callbacks(IndexListCallbacks.EVALUATE_CONTEXT_WINDOWS, self.callbacks):
                callback(self, model, x_in, conds, timestep, model_options, window_idx, window, model_options, device, first_device)

            logging.info(f"Context window {window_idx + 1}/{total_windows or len(enumerated_context_windows)}: frames {window.index_list[0]}-{window.index_list[-1]} of {x.shape[self.dim]}"
                         + (f" (+{guide_frame_counts_per_modality[0]} guide frames)" if guide_frame_counts_per_modality[0] > 0 else "")
                         )

            # if multimodal, pack modalities together
            if window_state.is_multimodal and len(sliced) > 1:
                sub_x, sub_shapes = comfy.utils.pack_latents(sliced)
            else:
                sub_x, sub_shapes = sliced[0], [sliced[0].shape]

            # get resized conds for window
            model_options["transformer_options"]["context_window"] = window
            sub_timestep = window.get_tensor(timestep, dim=0)
            sub_conds = [self.get_resized_cond(cond, x, window) for cond in conds]

            # if multimodal, patch latent_shapes in conds for correct unpacking in model
            window_state.patch_latent_shapes(sub_conds, sub_shapes)

            # call model on window
            sub_conds_out = calc_cond_batch(model, sub_conds, sub_x, sub_timestep, model_options)

            # unpack outputs
            out_per_modality = [comfy.utils.unpack_latents(sub_conds_out[i], sub_shapes) for i in range(len(sub_conds_out))]

            # strip causal_window_fix anchor from primary modality before guide strip so window_len math stays correct
            if anchor_applied:
                for ci in range(len(out_per_modality)):
                    t = out_per_modality[ci][0]
                    out_per_modality[ci][0] = t.narrow(self.dim, 1, t.shape[self.dim] - 1)

            # strip injected guide frames
            window_state.strip_guide_frames(out_per_modality, guide_frame_counts_per_modality, window)

            results.append(ContextResults(window_idx, out_per_modality, sub_conds, window))
        return results


    def combine_context_window_results(self, x_in: torch.Tensor, sub_conds_out, sub_conds, window: IndexListContextWindow, window_idx: int, total_windows: int, timestep: torch.Tensor,
                                    conds_final: list[torch.Tensor], counts_final: list[torch.Tensor], biases_final: list[torch.Tensor]):
        if self.fuse_method.name == ContextFuseMethods.RELATIVE:
            for pos, idx in enumerate(window.index_list):
                # bias is the influence of a specific index in relation to the whole context window
                bias = 1 - abs(idx - (window.index_list[0] + window.index_list[-1]) / 2) / ((window.index_list[-1] - window.index_list[0] + 1e-2) / 2)
                bias = max(1e-2, bias)
                # take weighted average relative to total bias of current idx
                for i in range(len(sub_conds_out)):
                    bias_total = biases_final[i][idx]
                    prev_weight = (bias_total / (bias_total + bias))
                    new_weight = (bias / (bias_total + bias))
                    # account for dims of tensors
                    idx_window = tuple([slice(None)] * self.dim + [idx])
                    pos_window = tuple([slice(None)] * self.dim + [pos])
                    # apply new values
                    conds_final[i][idx_window] = conds_final[i][idx_window] * prev_weight + sub_conds_out[i][pos_window] * new_weight
                    biases_final[i][idx] = bias_total + bias
        else:
            # add conds and counts based on weights of fuse method
            weights = get_context_weights(window.context_length, x_in.shape[self.dim], window.index_list, self, sigma=timestep, context_overlap=window.context_overlap)
            weights_tensor = match_weights_to_dim(weights, x_in, self.dim, device=x_in.device)
            for i in range(len(sub_conds_out)):
                window.add_window(conds_final[i], sub_conds_out[i] * weights_tensor)
                window.add_window(counts_final[i], weights_tensor)

        for callback in comfy.patcher_extension.get_all_callbacks(IndexListCallbacks.COMBINE_CONTEXT_WINDOW_RESULTS, self.callbacks):
            callback(self, x_in, sub_conds_out, sub_conds, window, window_idx, total_windows, timestep, conds_final, counts_final, biases_final)


def _prepare_sampling_wrapper(executor, model, noise_shape: torch.Tensor, conds, *args, **kwargs):
    # Scale noise_shape to a single context window so VRAM estimation budgets per-window.
    model_options = kwargs.get("model_options", None)
    if model_options is None:
        raise Exception("model_options not found in prepare_sampling_wrapper; this should never happen, something went wrong.")
    handler: IndexListContextHandler = model_options.get("context_handler", None)
    if handler is not None:
        noise_shape = list(noise_shape)
        is_packed = len(noise_shape) == 3 and noise_shape[1] == 1
        if is_packed:
            # TODO: latent_shapes cond isn't attached yet at this point, so we can't compute a
            # per-window flat latent here. Skipping the clamp over-estimates but prevents immediate OOM.
            pass
        elif handler.dim < len(noise_shape) and noise_shape[handler.dim] > handler.context_length:
            noise_shape[handler.dim] = min(noise_shape[handler.dim], handler.context_length)
    return executor(model, noise_shape, conds, *args, **kwargs)


def create_prepare_sampling_wrapper(model: ModelPatcher):
    model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
        "ContextWindows_prepare_sampling",
        _prepare_sampling_wrapper
    )


def _sampler_sample_wrapper(executor, guider, sigmas, extra_args, callback, noise, *args, **kwargs):
    model_options = extra_args.get("model_options", None)
    if model_options is None:
        raise Exception("model_options not found in sampler_sample_wrapper; this should never happen, something went wrong.")
    handler: IndexListContextHandler = model_options.get("context_handler", None)
    if handler is None:
        raise Exception("context_handler not found in sampler_sample_wrapper; this should never happen, something went wrong.")
    if not handler.freenoise:
        return executor(guider, sigmas, extra_args, callback, noise, *args, **kwargs)

    conds = [guider.conds.get('positive', guider.conds.get('negative', []))]
    noise = handler._apply_freenoise(noise, conds, extra_args["seed"])

    return executor(guider, sigmas, extra_args, callback, noise, *args, **kwargs)

def create_sampler_sample_wrapper(model: ModelPatcher):
    model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
        "ContextWindows_sampler_sample",
        _sampler_sample_wrapper
    )

def match_weights_to_dim(weights: list[float], x_in: torch.Tensor, dim: int, device=None) -> torch.Tensor:
    total_dims = len(x_in.shape)
    weights_tensor = torch.Tensor(weights).to(device=device)
    for _ in range(dim):
        weights_tensor = weights_tensor.unsqueeze(0)
    for _ in range(total_dims - dim - 1):
        weights_tensor = weights_tensor.unsqueeze(-1)
    return weights_tensor

def get_shape_for_dim(x_in: torch.Tensor, dim: int) -> list[int]:
    total_dims = len(x_in.shape)
    shape = []
    for _ in range(dim):
        shape.append(1)
    shape.append(x_in.shape[dim])
    for _ in range(total_dims - dim - 1):
        shape.append(1)
    return shape

class ContextSchedules:
    UNIFORM_LOOPED = "looped_uniform"
    UNIFORM_STANDARD = "standard_uniform"
    STATIC_STANDARD = "standard_static"
    BATCHED = "batched"


# from https://github.com/neggles/animatediff-cli/blob/main/src/animatediff/pipelines/context.py
def create_windows_uniform_looped(num_frames: int, handler: IndexListContextHandler, model_options: dict[str]):
    windows = []
    if num_frames < handler.context_length:
        windows.append(list(range(num_frames)))
        return windows

    context_stride = min(handler.context_stride, int(np.ceil(np.log2(num_frames / handler.context_length))) + 1)
    # obtain uniform windows as normal, looping and all
    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(handler._step)))
        for j in range(
            int(ordered_halving(handler._step) * context_step) + pad,
            num_frames + pad + (0 if handler.closed_loop else -handler.context_overlap),
            (handler.context_length * context_step - handler.context_overlap),
        ):
            windows.append([e % num_frames for e in range(j, j + handler.context_length * context_step, context_step)])

    return windows

def create_windows_uniform_standard(num_frames: int, handler: IndexListContextHandler, model_options: dict[str]):
    # unlike looped, uniform_straight does NOT allow windows that loop back to the beginning;
    # instead, they get shifted to the corresponding end of the frames.
    # in the case that a window (shifted or not) is identical to the previous one, it gets skipped.
    windows = []
    if num_frames <= handler.context_length:
        windows.append(list(range(num_frames)))
        return windows

    context_stride = min(handler.context_stride, int(np.ceil(np.log2(num_frames / handler.context_length))) + 1)
    # first, obtain uniform windows as normal, looping and all
    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(handler._step)))
        for j in range(
            int(ordered_halving(handler._step) * context_step) + pad,
            num_frames + pad + (-handler.context_overlap),
            (handler.context_length * context_step - handler.context_overlap),
        ):
            windows.append([e % num_frames for e in range(j, j + handler.context_length * context_step, context_step)])

    # now that windows are created, shift any windows that loop, and delete duplicate windows
    delete_idxs = []
    win_i = 0
    while win_i < len(windows):
        # if window is rolls over itself, need to shift it
        is_roll, roll_idx = does_window_roll_over(windows[win_i], num_frames)
        if is_roll:
            roll_val = windows[win_i][roll_idx]  # roll_val might not be 0 for windows of higher strides
            shift_window_to_end(windows[win_i], num_frames=num_frames)
            # check if next window (cyclical) is missing roll_val
            if roll_val not in windows[(win_i+1) % len(windows)]:
                # need to insert new window here - just insert window starting at roll_val
                windows.insert(win_i+1, list(range(roll_val, roll_val + handler.context_length)))
        # delete window if it's not unique
        for pre_i in range(0, win_i):
            if windows[win_i] == windows[pre_i]:
                delete_idxs.append(win_i)
                break
        win_i += 1

    # reverse delete_idxs so that they will be deleted in an order that doesn't break idx correlation
    delete_idxs.reverse()
    for i in delete_idxs:
        windows.pop(i)

    return windows


def create_windows_static_standard(num_frames: int, handler: IndexListContextHandler, model_options: dict[str]):
    windows = []
    if num_frames <= handler.context_length:
        windows.append(list(range(num_frames)))
        return windows
    # always return the same set of windows
    delta = handler.context_length - handler.context_overlap
    for start_idx in range(0, num_frames, delta):
        # if past the end of frames, move start_idx back to allow same context_length
        ending = start_idx + handler.context_length
        if ending >= num_frames:
            final_delta = ending - num_frames
            final_start_idx = start_idx - final_delta
            windows.append(list(range(final_start_idx, final_start_idx + handler.context_length)))
            break
        windows.append(list(range(start_idx, start_idx + handler.context_length)))
    return windows


def create_windows_batched(num_frames: int, handler: IndexListContextHandler, model_options: dict[str]):
    windows = []
    if num_frames <= handler.context_length:
        windows.append(list(range(num_frames)))
        return windows
    # always return the same set of windows;
    # no overlap, just cut up based on context_length;
    # last window size will be different if num_frames % opts.context_length != 0
    for start_idx in range(0, num_frames, handler.context_length):
        windows.append(list(range(start_idx, min(start_idx + handler.context_length, num_frames))))
    return windows


def create_windows_default(num_frames: int, handler: IndexListContextHandler):
    return [list(range(num_frames))]


CONTEXT_MAPPING = {
    ContextSchedules.UNIFORM_LOOPED: create_windows_uniform_looped,
    ContextSchedules.UNIFORM_STANDARD: create_windows_uniform_standard,
    ContextSchedules.STATIC_STANDARD: create_windows_static_standard,
    ContextSchedules.BATCHED: create_windows_batched,
}


def get_matching_context_schedule(context_schedule: str) -> ContextSchedule:
    func = CONTEXT_MAPPING.get(context_schedule, None)
    if func is None:
        raise ValueError(f"Unknown context_schedule '{context_schedule}'.")
    return ContextSchedule(context_schedule, func)


def get_context_weights(length: int, full_length: int, idxs: list[int], handler: IndexListContextHandler, sigma: torch.Tensor=None, context_overlap: int=None):
    context_overlap = handler.context_overlap if context_overlap is None else context_overlap
    return handler.fuse_method.func(length, sigma=sigma, handler=handler, full_length=full_length, idxs=idxs, context_overlap=context_overlap)


def create_weights_flat(length: int, **kwargs) -> list[float]:
    # weight is the same for all
    return [1.0] * length

def create_weights_pyramid(length: int, **kwargs) -> list[float]:
    # weight is based on the distance away from the edge of the context window;
    # based on weighted average concept in FreeNoise paper
    if length % 2 == 0:
        max_weight = length // 2
        weight_sequence = list(range(1, max_weight + 1, 1)) + list(range(max_weight, 0, -1))
    else:
        max_weight = (length + 1) // 2
        weight_sequence = list(range(1, max_weight, 1)) + [max_weight] + list(range(max_weight - 1, 0, -1))
    return weight_sequence

def create_weights_overlap_linear(length: int, full_length: int, idxs: list[int], context_overlap: int, **kwargs):
    # based on code in Kijai's WanVideoWrapper: https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/dbb2523b37e4ccdf45127e5ae33e31362f755c8e/nodes.py#L1302
    # only expected overlap is given different weights
    weights_torch = torch.ones((length))
    # blend left-side on all except first window
    if min(idxs) > 0:
        ramp_up = torch.linspace(1e-37, 1, context_overlap)
        weights_torch[:context_overlap] = ramp_up
    # blend right-side on all except last window
    if max(idxs) < full_length-1:
        ramp_down = torch.linspace(1, 1e-37, context_overlap)
        weights_torch[-context_overlap:] = ramp_down
    return weights_torch

class ContextFuseMethods:
    FLAT = "flat"
    PYRAMID = "pyramid"
    RELATIVE = "relative"
    OVERLAP_LINEAR = "overlap-linear"

    LIST = [PYRAMID, FLAT, OVERLAP_LINEAR]
    LIST_STATIC = [PYRAMID, RELATIVE, FLAT, OVERLAP_LINEAR]


FUSE_MAPPING = {
    ContextFuseMethods.FLAT: create_weights_flat,
    ContextFuseMethods.PYRAMID: create_weights_pyramid,
    ContextFuseMethods.RELATIVE: create_weights_pyramid,
    ContextFuseMethods.OVERLAP_LINEAR: create_weights_overlap_linear,
}

def get_matching_fuse_method(fuse_method: str) -> ContextFuseMethod:
    func = FUSE_MAPPING.get(fuse_method, None)
    if func is None:
        raise ValueError(f"Unknown fuse_method '{fuse_method}'.")
    return ContextFuseMethod(fuse_method, func)

# Returns fraction that has denominator that is a power of 2
def ordered_halving(val):
    # get binary value, padded with 0s for 64 bits
    bin_str = f"{val:064b}"
    # flip binary value, padding included
    bin_flip = bin_str[::-1]
    # convert binary to int
    as_int = int(bin_flip, 2)
    # divide by 1 << 64, equivalent to 2**64, or 18446744073709551616,
    # or b10000000000000000000000000000000000000000000000000000000000000000 (1 with 64 zero's)
    return as_int / (1 << 64)


def get_missing_indexes(windows: list[list[int]], num_frames: int) -> list[int]:
    all_indexes = list(range(num_frames))
    for w in windows:
        for val in w:
            try:
                all_indexes.remove(val)
            except ValueError:
                pass
    return all_indexes


def does_window_roll_over(window: list[int], num_frames: int) -> tuple[bool, int]:
    prev_val = -1
    for i, val in enumerate(window):
        val = val % num_frames
        if val < prev_val:
            return True, i
        prev_val = val
    return False, -1


def shift_window_to_start(window: list[int], num_frames: int):
    start_val = window[0]
    for i in range(len(window)):
        # 1) subtract each element by start_val to move vals relative to the start of all frames
        # 2) add num_frames and take modulus to get adjusted vals
        window[i] = ((window[i] - start_val) + num_frames) % num_frames


def shift_window_to_end(window: list[int], num_frames: int):
    # 1) shift window to start
    shift_window_to_start(window, num_frames)
    end_val = window[-1]
    end_delta = num_frames - end_val - 1
    for i in range(len(window)):
        # 2) add end_delta to each val to slide windows to end
        window[i] = window[i] + end_delta


# https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/blob/90fb1331201a4b29488089e4fbffc0d82cc6d0a9/animatediff/sample_settings.py#L465
def apply_freenoise(noise: torch.Tensor, dim: int, context_length: int, context_overlap: int, seed: int):
    logging.info("Context windows: Applying FreeNoise")
    generator = torch.Generator(device='cpu').manual_seed(seed)
    latent_video_length = noise.shape[dim]
    delta = context_length - context_overlap

    for start_idx in range(0, latent_video_length - context_length, delta):
        place_idx = start_idx + context_length

        actual_delta = min(delta, latent_video_length - place_idx)
        if actual_delta <= 0:
            break

        list_idx = torch.randperm(actual_delta, generator=generator, device='cpu') + start_idx

        source_slice = [slice(None)] * noise.ndim
        source_slice[dim] = list_idx
        target_slice = [slice(None)] * noise.ndim
        target_slice[dim] = slice(place_idx, place_idx + actual_delta)

        noise[tuple(target_slice)] = noise[tuple(source_slice)]

    return noise
