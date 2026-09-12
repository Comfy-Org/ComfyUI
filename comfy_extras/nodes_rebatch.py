from typing_extensions import override
import torch

import comfy.utils
from comfy_api.latest import ComfyExtension, io


MERGE_MODES = ["truncate start", "truncate end", "fade smooth", "fade linear"]


def _first(value):
    return value[0] if isinstance(value, list) else value


def _slice_tensor(tensor, dimension, start=None, end=None):
    index = (slice(None),) * dimension + (slice(start, end),)
    return tensor[index]


def _compatible(first, second, dimension):
    return first.ndim == second.ndim and all(
        first.shape[index] == second.shape[index] for index in range(first.ndim) if index != dimension
    )


def _merge_tensors(tensors, dimension, overlap, merge_mode):
    if len(tensors) == 1:
        return tensors[0]

    shape = list(tensors[0].shape)
    shape[dimension] = sum(tensor.shape[dimension] for tensor in tensors) - overlap * (len(tensors) - 1)
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    merged = tensors[0].new_empty(shape, dtype=dtype)
    end = tensors[0].shape[dimension]
    _slice_tensor(merged, dimension, end=end).copy_(tensors[0])

    weight = None
    if overlap and merge_mode in ("fade smooth", "fade linear"):
        weight = torch.linspace(0, 1, overlap, device=merged.device, dtype=merged.dtype)
        if merge_mode == "fade smooth":
            weight = weight * weight * (3 - 2 * weight)
        weight = weight.reshape([1] * dimension + [overlap] + [1] * (merged.ndim - dimension - 1))

    for tensor in tensors[1:]:
        if overlap:
            destination = _slice_tensor(merged, dimension, end - overlap, end)
            if merge_mode == "truncate end":
                destination.copy_(_slice_tensor(tensor, dimension, end=overlap))
            elif weight is not None:
                current = _slice_tensor(tensor, dimension, end=overlap).to(destination)
                destination.copy_(destination * (1 - weight) + current * weight)
        next_end = end + tensor.shape[dimension] - overlap
        _slice_tensor(merged, dimension, end, next_end).copy_(_slice_tensor(tensor, dimension, start=overlap))
        end = next_end
    return merged


def _rebatch(values, dimension, batch_size, overlap, merge_mode, get_tensor, merge, slice_value):
    if not values:
        return []
    if overlap >= batch_size:
        raise ValueError("Rebatch overlap must be smaller than the batch size")
    first = get_tensor(values[0])
    if dimension >= first.ndim:
        raise ValueError(f"Cannot rebatch dimension {dimension} of a {first.ndim}-dimensional tensor")

    outputs = []
    values = list(values)
    while values:
        count = 1
        first = get_tensor(values[0])
        length = first.shape[dimension]
        while length < batch_size and count < len(values):
            tensor = get_tensor(values[count])
            if not _compatible(first, tensor, dimension):
                break
            if overlap > min(length, tensor.shape[dimension]):
                raise ValueError("Rebatch overlap cannot exceed an input batch")
            length += tensor.shape[dimension] - overlap
            count += 1

        merged = merge(values[:count], dimension, overlap, merge_mode)
        del values[:count]
        outputs.append(slice_value(merged, dimension, end=batch_size))
        if length > batch_size:
            values.insert(0, slice_value(merged, dimension, start=batch_size - overlap))
    return outputs


def _slice_latent(latent, dimension, start=None, end=None):
    result = latent.copy()
    result["samples"] = _slice_tensor(latent["samples"], dimension, start, end)
    if "noise_mask" in latent:
        result["noise_mask"] = _slice_tensor(latent["noise_mask"], dimension, start, end)
    if dimension == 0:
        result["batch_index"] = latent["batch_index"][slice(start, end)]
    return result


def _merge_latents(latents, dimension, overlap, merge_mode):
    result = latents[0].copy()
    result["samples"] = _merge_tensors([latent["samples"] for latent in latents], dimension, overlap, merge_mode)

    masks = [latent.get("noise_mask") for latent in latents]
    if any(mask is not None for mask in masks):
        template = next(mask for mask in masks if mask is not None)
        for index, mask in enumerate(masks):
            if mask is None:
                shape = list(template.shape)
                shape[dimension] = latents[index]["samples"].shape[dimension]
                masks[index] = torch.ones(shape, dtype=template.dtype, device=template.device)
        result["noise_mask"] = _merge_tensors(masks, dimension, overlap, merge_mode)
    else:
        result.pop("noise_mask", None)

    if dimension == 0:
        indices = list(latents[0]["batch_index"])
        for latent in latents[1:]:
            current = latent["batch_index"]
            if overlap and merge_mode == "truncate end":
                indices[-overlap:] = current[:overlap]
            indices.extend(current[overlap:])
        result["batch_index"] = indices
    return result


def _rebatch_latents(latents, dimension, batch_size, overlap, merge_mode):
    prepared = []
    processed = 0
    for latent in latents:
        latent = latent.copy()
        samples = latent["samples"]
        mask = latent.get("noise_mask")
        if torch.is_tensor(mask):
            latent["noise_mask"] = comfy.utils.reshape_mask(mask, samples.shape, expand=False)
        else:
            latent.pop("noise_mask", None)
        if dimension == 0 and "batch_index" not in latent:
            latent["batch_index"] = list(range(processed, processed + samples.shape[0]))
        processed += samples.shape[0]
        prepared.append(latent)

    outputs = _rebatch(prepared, dimension, batch_size, overlap, merge_mode,
                       lambda latent: latent["samples"], _merge_latents, _slice_latent)
    for latent in outputs:
        if "noise_mask" in latent and torch.all(latent["noise_mask"] == 1):
            del latent["noise_mask"]
    return outputs


class LatentRebatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RebatchLatents",
            display_name="Rebatch Latents",
            category="model/latent/batch",
            is_input_list=True,
            inputs=[
                io.Latent.Input("latents"),
                io.Int.Input("batch_size", default=1, min=1, max=4096, tooltip="Output size along the selected dimension."),
                io.Combo.Input(
                    "dimension",
                    options=["b", "t (video only)", "t (audio only)"],
                    default="b",
                    optional=True,
                    tooltip="Rebatch batch items, video latent time, or audio latent time.",
                ),
                io.Int.Input("overlap", default=0, min=0, max=4096, optional=True, advanced=True, tooltip="Samples duplicated between output splits and reconciled between input batches."),
                io.Combo.Input(
                    "merge_overlap_mode",
                    options=MERGE_MODES,
                    default="truncate start",
                    optional=True,
                    advanced=True,
                    tooltip="How overlapping samples from consecutive input batches are merged.",
                ),
            ],
            outputs=[
                io.Latent.Output(is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, latents, batch_size, dimension="b", overlap=0, merge_overlap_mode="truncate start"):
        batch_size = _first(batch_size)
        dimension = _first(dimension)
        overlap = _first(overlap)
        merge_overlap_mode = _first(merge_overlap_mode)
        if not latents:
            return io.NodeOutput([])
        if any(not torch.is_tensor(latent["samples"]) for latent in latents):
            raise ValueError("Rebatch Latents does not support nested latents; separate their streams first")
        if dimension == "t (video only)" and any(latent["samples"].ndim == 4 for latent in latents):
            raise ValueError("Rebatch Latents cannot use the video time dimension with image latents")
        dimension = {"b": 0, "t (video only)": 2, "t (audio only)": latents[0]["samples"].ndim - 1}[dimension]
        return io.NodeOutput(_rebatch_latents(latents, dimension, batch_size, overlap, merge_overlap_mode))

class ImageRebatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RebatchImages",
            display_name="Rebatch Images",
            category="image/batch",
            is_input_list=True,
            inputs=[
                io.Image.Input("images"),
                io.Int.Input("batch_size", default=1, min=1, max=4096, tooltip="Number of images or frames in each output batch."),
                io.Int.Input("overlap", default=0, min=0, max=4096, optional=True, advanced=True, tooltip="Frames duplicated between output splits and reconciled between input batches."),
                io.Combo.Input(
                    "merge_overlap_mode",
                    options=MERGE_MODES,
                    default="truncate start",
                    optional=True,
                    advanced=True,
                    tooltip="How overlapping frames from consecutive input batches are merged.",
                ),
            ],
            outputs=[
                io.Image.Output(is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, images, batch_size, overlap=0, merge_overlap_mode="truncate start"):
        return io.NodeOutput(
            _rebatch(
                images,
                0,
                _first(batch_size),
                _first(overlap),
                _first(merge_overlap_mode),
                lambda tensor: tensor,
                _merge_tensors,
                _slice_tensor,
            )
        )


class RebatchExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LatentRebatch,
            ImageRebatch,
        ]


async def comfy_entrypoint() -> RebatchExtension:
    return RebatchExtension()
