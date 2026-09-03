"""Operations that belong to specific third-party packs.

These are dispatched through the same registry as the core primitives, but they
are not part of the engine vocabulary: each one exists to serve one pack's
model family. They live here so the core dispatcher stays a closed set, and so
they can move out to the packs that own them without touching it.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Optional

from . import _sdk

if TYPE_CHECKING:
    from ._sdk import (
        ClipSegRef,
        ControlNetWeightsRef,
        ImagePreprocessorRef,
        ImageRef,
        InpaintModelRef,
        InterpolationStatesRef,
        LatentRef,
        MaskRef,
        ModelRef,
        Ref,
        SamModelRef,
        TimestepKeyframeRef,
    )


async def interpolation_states_skip_mask(states: "InterpolationStatesRef", pair_count: int,
) -> list[bool]:
    """Project a foreign interpolation policy without invoking it.

    ComfyUI-Frame-Interpolation represents this value as a small object
    with two instance-data fields.  Treat those fields as data only: an
    arbitrary method, property, iterator, or string conversion on the
    foreign object must never execute in the trusted process.
    """
    if isinstance(pair_count, bool) or not isinstance(pair_count, int):
        raise TypeError("interpolation pair_count must be an integer")
    if not 1 <= pair_count <= 100_000:
        raise ValueError("interpolation pair_count must be in [1, 100000]")

    value = await _sdk.current_runtime().refs.resolve(states)
    try:
        fields = object.__getattribute__(value, "__dict__")
    except (AttributeError, TypeError) as error:
        raise TypeError(
            "INTERPOLATION_STATES must expose fixed instance data"
        ) from error
    if type(fields) is not dict:
        raise TypeError(
            "INTERPOLATION_STATES must expose fixed instance data")
    if set(fields) != {"frame_indices", "is_skip_list"}:
        raise TypeError(
            "INTERPOLATION_STATES has an unsupported field layout")
    indices = fields["frame_indices"]
    skip_list = fields["is_skip_list"]
    if type(indices) is not list:
        raise TypeError("interpolation frame_indices must be a list")
    if len(indices) > 100_000:
        raise ValueError(
            "interpolation frame_indices exceeds the 100000 item limit")
    if type(skip_list) is not bool:
        raise TypeError("interpolation is_skip_list must be a boolean")

    selected: set[int] = set()
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(
                "interpolation frame indices must be integers")
        if index < 0:
            raise ValueError(
                "interpolation frame indices must be non-negative")
        if index < pair_count:
            selected.add(index)
    if skip_list:
        return [index in selected for index in range(pair_count)]
    return [index not in selected for index in range(pair_count)]

async def advanced_control_weights_from_list(_subject: Optional["Ref"], weights: list,
    uncond_multiplier: float = 1.0, extras: Any = None,
) -> tuple["ControlNetWeightsRef", "TimestepKeyframeRef"]:
    import math

    if not isinstance(weights, (list, tuple)):
        raise TypeError("ControlNet weights must be a list")
    if len(weights) > 4096:
        raise ValueError("ControlNet weights are limited to 4096 values")
    checked_weights = []
    for value in weights:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("ControlNet weights must contain only numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("ControlNet weights must be finite")
        checked_weights.append(number)
    if (isinstance(uncond_multiplier, bool)
            or not isinstance(uncond_multiplier, (int, float))):
        raise TypeError("uncond_multiplier must be a number")
    multiplier = float(uncond_multiplier)
    if not math.isfinite(multiplier) or not 0.0 <= multiplier <= 1.0:
        raise ValueError("uncond_multiplier must be finite and in [0, 1]")

    rt = _sdk.current_runtime()
    checked_extras = extras
    if isinstance(checked_extras, _sdk.Ref):
        if checked_extras.kind != "VALUE":
            raise TypeError("ControlNet extras ref must contain VALUE data")
        checked_extras = await rt.refs.resolve(checked_extras)
    if checked_extras is None:
        checked_extras = {}
    if not isinstance(checked_extras, dict):
        raise TypeError("ControlNet extras must be a mapping")

    def validate_extra(value: Any, depth: int = 0) -> None:
        if depth > 32:
            raise ValueError("ControlNet extras nesting exceeds 32 levels")
        if _sdk._looks_like_tensor(value):
            return
        if value is None or isinstance(value, (str, bool, int)):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("ControlNet extras must contain finite numbers")
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                validate_extra(item, depth + 1)
            return
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("ControlNet extras keys must be strings")
            for item in value.values():
                validate_extra(item, depth + 1)
            return
        raise TypeError(
            f"ControlNet extras cannot contain {type(value).__name__}")

    validate_extra(checked_extras)

    utils = _sdk._advanced_control_module("utils")
    control_weights = utils.ControlWeights.controlnet(
        weights_input=checked_weights,
        uncond_multiplier=multiplier,
        extras=checked_extras,
    )
    keyframe = utils.TimestepKeyframe(
        control_weights=control_weights)
    shortcut = utils.TimestepKeyframeGroup.default(keyframe)
    weights_ref = _sdk.ControlNetWeightsRef._wrap(await rt.refs.create(
        "CONTROL_NET_WEIGHTS", control_weights))
    shortcut_ref = _sdk.TimestepKeyframeRef._wrap(await rt.refs.create(
        "TIMESTEP_KEYFRAME", shortcut))
    return weights_ref, shortcut_ref

async def advanced_control_scaled_soft_weights(_subject: Optional["Ref"], base_multiplier: float = 0.825,
    uncond_multiplier: float = 1.0,
) -> tuple["ControlNetWeightsRef", "TimestepKeyframeRef"]:
    import math

    def multiplier(value: Any, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field} must be a number")
        result = float(value)
        if not math.isfinite(result) or not 0.0 <= result <= 1.0:
            raise ValueError(f"{field} must be finite and in [0, 1]")
        return result

    base = multiplier(base_multiplier, "base_multiplier")
    uncond = multiplier(uncond_multiplier, "uncond_multiplier")
    utils = _sdk._advanced_control_module("utils")
    control_weights = utils.ControlWeights.universal(
        base_multiplier=base,
        uncond_multiplier=uncond,
        extras={},
    )
    shortcut = utils.TimestepKeyframeGroup.default(
        utils.TimestepKeyframe(control_weights=control_weights))
    rt = _sdk.current_runtime()
    return (
        _sdk.ControlNetWeightsRef._wrap(await rt.refs.create(
            "CONTROL_NET_WEIGHTS", control_weights)),
        _sdk.TimestepKeyframeRef._wrap(await rt.refs.create(
            "TIMESTEP_KEYFRAME", shortcut)),
    )  # type: ignore[return-value]

async def inpaint_model_inpaint(inpaint_model: "InpaintModelRef",
    image: "ImageRef", mask: "MaskRef",
) -> "ImageRef":
    import torch
    import comfy.model_management

    rt = _sdk.current_runtime()
    bundle = await rt.refs.resolve(inpaint_model)
    pixels = await rt.refs.resolve(image)
    mask_value = await rt.refs.resolve(mask)
    if bundle.get("secure_kind") != "image_inpaint.big-lama":
        raise ValueError("unknown image inpaint model")
    if pixels.ndim != 4 or pixels.shape[-1] < 3 or not 1 <= len(pixels) <= 4096:
        raise ValueError("inpaint images must be a non-empty BHWC batch")
    if mask_value.ndim == 4 and mask_value.shape[1] == 1:
        mask_value = mask_value[:, 0]
    elif mask_value.ndim == 4 and mask_value.shape[-1] == 1:
        mask_value = mask_value[..., 0]
    if mask_value.ndim != 3:
        raise ValueError("inpaint masks must be a BHW batch")
    height, width = map(int, pixels.shape[1:3])
    if tuple(mask_value.shape[-2:]) != (height, width):
        raise ValueError("inpaint image and mask dimensions must match")
    if len(mask_value) not in (1, len(pixels)):
        raise ValueError("inpaint image and mask batches must match")
    if min(height, width) < 16 or height % 8 or width % 8:
        raise ValueError(
            "Big-LaMa image dimensions must be multiples of 8 and at least 16")
    if len(pixels) * height * width > 67_108_864:
        raise ValueError("inpaint batch exceeds 67108864 pixels")

    model = bundle["model"]
    model_lock = bundle["lock"]
    device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    source = pixels[..., :3].movedim(-1, 1).to(
        device=device, dtype=torch.float32)
    holes = mask_value.unsqueeze(1).to(
        device=device, dtype=torch.float32)
    if len(holes) == 1 and len(source) > 1:
        holes = holes.expand(len(source), -1, -1, -1)
    source = source.clamp(0.0, 1.0)
    holes = holes.clamp(0.0, 1.0)
    with model_lock:
        model.to(device=device, dtype=torch.float32)
        try:
            result = model(source, holes)
            result = result.movedim(1, -1).clamp(0.0, 1.0)
            result = result.detach().to(device="cpu", dtype=torch.float32)
        finally:
            model.to(offload_device)
    comfy.model_management.soft_empty_cache()
    return _sdk.ImageRef._wrap(await rt.refs.create(
        "IMAGE", result))  # type: ignore[return-value]

async def clipseg_segment(clipseg: "ClipSegRef", images: "ImageRef", text: str,
    threshold: float = 0.5, binary_mask: bool = True,
    combine_mask: bool = False, use_accelerator: bool = True,
    blur_sigma: float = 0.0, previous_mask: Optional["MaskRef"] = None,
    invert: bool = False, image_background_level: float = 0.5,
) -> tuple["MaskRef", "ImageRef"]:
    from contextlib import nullcontext
    import numpy as np
    import torch
    import torch.nn.functional as functional
    import torchvision.transforms as transforms
    from PIL import Image
    import comfy.model_management

    threshold = float(threshold)
    blur_sigma = float(blur_sigma)
    background = float(image_background_level)
    if not 0.0 <= threshold <= 10.0:
        raise ValueError("CLIPSeg threshold must be in [0, 10]")
    if not 0.0 <= blur_sigma <= 100.0:
        raise ValueError("CLIPSeg blur_sigma must be in [0, 100]")
    if not 0.0 <= background <= 1.0:
        raise ValueError("CLIPSeg image background level must be in [0, 1]")

    rt = _sdk.current_runtime()
    bundle = await rt.refs.resolve(clipseg)
    pixels = await rt.refs.resolve(images)
    previous = (None if previous_mask is None
                else await rt.refs.resolve(previous_mask))
    model = bundle["model"]
    processor = bundle["processor"]
    model_lock = bundle.get("lock")
    offload_device = comfy.model_management.unet_offload_device()
    device = (comfy.model_management.get_torch_device()
              if use_accelerator else torch.device("cpu"))
    dtype = comfy.model_management.unet_dtype()
    with model_lock if model_lock is not None else nullcontext():
        model.to(dtype).to(device)
        try:
            height, width = pixels.shape[1:3]
            source = pixels.to(device)
            autocast = (
                dtype != torch.float32
                and not comfy.model_management.is_device_mps(device))
            scope = (torch.autocast(
                comfy.model_management.get_autocast_device(device), dtype=dtype)
                if autocast else nullcontext())
            with scope, torch.inference_mode():
                pil_images = [Image.fromarray(np.clip(
                    255.0 * image.cpu().numpy().squeeze(), 0, 255
                ).astype(np.uint8)) for image in source]
                inputs = processor(
                    text=[str(text)] * len(source), images=pil_images,
                    return_tensors="pt", padding=True, truncation=True,
                    max_length=77)
                inputs = {
                    key: value.to(device) for key, value in inputs.items()
                }
                outputs = model(**inputs)
            mask = torch.sigmoid(outputs.logits)
            minimum, maximum = mask.amin(), mask.amax()
            scale = (maximum - minimum).clamp_min(
                torch.finfo(mask.dtype).eps)
            mask = (mask - minimum) / scale
            mask = torch.where(
                mask > threshold, mask,
                torch.tensor(0, dtype=torch.float, device=mask.device))
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = functional.interpolate(
                mask.unsqueeze(1), size=(height, width), mode="nearest"
            ).squeeze(1)
        finally:
            model.to(offload_device)

    if binary_mask:
        mask = (mask > 0).float()
    if blur_sigma > 0:
        kernel_size = 6 * int(blur_sigma) + 1
        mask = transforms.GaussianBlur(
            kernel_size=(kernel_size, kernel_size),
            sigma=(blur_sigma, blur_sigma))(mask)
    if combine_mask:
        mask = torch.max(mask, dim=0)[0].unsqueeze(0).repeat(
            len(source), 1, 1)
    comfy.model_management.soft_empty_cache()
    if previous is not None:
        if previous.shape != mask.shape:
            previous = functional.interpolate(
                previous.unsqueeze(1), size=(height, width), mode="nearest"
            ).squeeze(1)
        mask = mask + previous.to(device)
        mask = torch.clamp(mask, min=0.0, max=1.0)
    if invert:
        mask = 1 - mask
    result_image = torch.clamp(
        source * mask.unsqueeze(-1)
        + (1 - mask.unsqueeze(-1)) * background,
        min=0.0, max=1.0).cpu().float()
    result_mask = mask.cpu().float()
    mask_ref = _sdk.MaskRef._wrap(await rt.refs.create("MASK", result_mask))
    image_ref = _sdk.ImageRef._wrap(await rt.refs.create("IMAGE", result_image))
    return mask_ref, image_ref  # type: ignore[return-value]

async def clipseg_predict_mask(clipseg: "ClipSegRef", images: "ImageRef", text: str,
    use_accelerator: bool = True,
) -> "MaskRef":
    """Run CLIPSeg while leaving thresholding/post-processing to the node."""
    from contextlib import nullcontext
    import numpy as np
    import torch
    import comfy.model_management

    text = str(text)
    if len(text) > 32768:
        raise ValueError("CLIPSeg text exceeds 32768 characters")
    rt = _sdk.current_runtime()
    bundle = await rt.refs.resolve(clipseg)
    pixels = await rt.refs.resolve(images)
    if pixels.ndim != 4 or pixels.shape[-1] < 3:
        raise ValueError("CLIPSeg images must be a non-empty BHWC batch")
    if not 1 <= len(pixels) <= 4096:
        raise ValueError("CLIPSeg batch size must be in [1, 4096]")

    model = bundle["model"]
    processor = bundle["processor"]
    model_lock = bundle.get("lock")
    offload_device = comfy.model_management.unet_offload_device()
    if use_accelerator:
        device = comfy.model_management.get_torch_device()
        dtype = comfy.model_management.unet_dtype()
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    with model_lock if model_lock is not None else nullcontext():
        model.to(dtype).to(device)
        try:
            autocast = (
                dtype != torch.float32
                and not comfy.model_management.is_device_mps(device)
            )
            outputs = []
            for image in pixels:
                array = np.clip(
                    image.detach().cpu().numpy() * 255.0, 0, 255
                ).astype(np.uint8)
                inputs = processor(
                    text=text, images=[array], return_tensors="pt",
                    padding=True, truncation=True, max_length=77)
                inputs = {
                    key: value.to(device) for key, value in inputs.items()
                }
                scope = (
                    torch.autocast(
                        comfy.model_management.get_autocast_device(device),
                        dtype=dtype,
                    )
                    if autocast else nullcontext()
                )
                with scope, torch.inference_mode():
                    prediction = model(**inputs).logits.unsqueeze(1)
                outputs.append(torch.sigmoid(prediction[0][0]))
            result = torch.stack(outputs, dim=0).cpu().float()
        finally:
            model.to(offload_device)
    comfy.model_management.soft_empty_cache()
    return _sdk.MaskRef._wrap(
        await rt.refs.create("MASK", result)
    )  # type: ignore[return-value]

async def ipadapter_apply(pipeline: "Ref",
    model: "ModelRef",
    image: "ImageRef",
    negative_image: Optional["ImageRef"] = None,
    attn_mask: Optional["MaskRef"] = None,
    style_image: Optional["ImageRef"] = None,
    composition_image: Optional["ImageRef"] = None,
    weight: float = 0.7,
    weight_type: str = "channel penalty",
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    combine_embeds: str = "concat",
    weight_faceidv2: float = 1.0,
    embeds_scaling: str = "V only",
    unfold_batch: bool = False,
    layer_weights: Optional[str] = None,
    weight_style: float = 1.0,
    weight_composition: float = 1.0,
    expand_style: bool = False,
) -> "ModelRef":
    """Apply one fixed IP-Adapter operation on the trusted plane.

    This is intentionally an integration boundary, not an Impact detailer
    implementation.  It accepts an opaque pipeline produced by a trusted
    host node and invokes the canonical IPAdapterAdvanced operation.  The
    guest remains responsible for SEGS traversal, crop choice, and chains.
    """
    import math
    import nodes
    import torch

    weight = float(weight)
    start_percent = float(start_percent)
    end_percent = float(end_percent)
    weight_faceidv2 = float(weight_faceidv2)
    weight_style = float(weight_style)
    weight_composition = float(weight_composition)
    if not all(math.isfinite(value) for value in (
        weight, start_percent, end_percent, weight_faceidv2,
        weight_style, weight_composition,
    )):
        raise ValueError("IP-Adapter numeric parameters must be finite")
    if not -1.0 <= weight <= 3.0:
        raise ValueError("IP-Adapter weight must be in [-1, 3]")
    if not 0.0 <= start_percent <= end_percent <= 1.0:
        raise ValueError(
            "IP-Adapter percentages must satisfy 0 <= start <= end <= 1")
    if not -1.0 <= weight_faceidv2 <= 5.0:
        raise ValueError("IP-Adapter FaceID v2 weight must be in [-1, 5]")
    if not -1.0 <= weight_style <= 5.0:
        raise ValueError("IP-Adapter style weight must be in [-1, 5]")
    if not -1.0 <= weight_composition <= 5.0:
        raise ValueError("IP-Adapter composition weight must be in [-1, 5]")
    if weight_type not in {
        "original", "linear", "channel penalty", "ease in", "ease out",
        "ease in-out", "reverse in-out", "weak input", "weak output",
        "weak middle", "strong middle", "style transfer", "composition",
        "strong style transfer", "style and composition",
        "style transfer precise", "composition precise",
    }:
        raise ValueError("unsupported IP-Adapter weight type")
    if combine_embeds not in {
        "concat", "add", "subtract", "average", "norm average",
    }:
        raise ValueError("unsupported IP-Adapter embedding combination")
    if embeds_scaling not in {
        "V only", "K+V", "K+V w/ C penalty",
        "K+mean(V) w/ C penalty",
    }:
        raise ValueError("unsupported IP-Adapter embedding scaling")
    if type(unfold_batch) is not bool:
        raise TypeError("IP-Adapter unfold_batch must be a bool")
    if type(expand_style) is not bool:
        raise TypeError("IP-Adapter expand_style must be a bool")
    if (layer_weights is not None
            and (not isinstance(layer_weights, str)
                 or len(layer_weights) > 16_384)):
        raise ValueError("IP-Adapter layer weights are invalid")

    rt = _sdk.current_runtime()
    pipe_value = await rt.refs.resolve(pipeline)
    if not _sdk._is_ipadapter_pipe(pipe_value):
        raise TypeError(
            "IPADAPTER_PIPE is not a host-created IP-Adapter pipeline")
    model_value = await rt.refs.resolve(model)
    pixels = await rt.refs.resolve(image)
    negative_pixels = (
        None if negative_image is None
        else await rt.refs.resolve(negative_image)
    )
    style_pixels = (
        None if style_image is None
        else await rt.refs.resolve(style_image)
    )
    composition_pixels = (
        None if composition_image is None
        else await rt.refs.resolve(composition_image)
    )
    mask_value = (
        None if attn_mask is None
        else await rt.refs.resolve(attn_mask)
    )
    for name, value in (
        ("image", pixels),
        ("negative_image", negative_pixels),
        ("style_image", style_pixels),
        ("composition_image", composition_pixels),
    ):
        if value is None:
            continue
        if (not isinstance(value, torch.Tensor) or value.ndim != 4
                or value.shape[-1] < 3 or value.shape[0] < 1
                or value.shape[0] > 4096):
            raise ValueError(
                f"IP-Adapter {name} must be a bounded BHWC image batch")
        height, width = map(int, value.shape[1:3])
        if (height <= 0 or width <= 0
                or height * width * int(value.shape[0]) > 268_435_456):
            raise ValueError(f"IP-Adapter {name} dimensions are invalid")

    if mask_value is not None:
        if (not isinstance(mask_value, torch.Tensor)
                or mask_value.ndim not in (2, 3)
                or mask_value.numel() <= 0
                or mask_value.numel() > 268_435_456):
            raise ValueError("IP-Adapter attention mask must be bounded HW/BHW")

    if isinstance(pipe_value, dict):
        ipadapter = pipe_value["ipadapter"]
        clip_vision = pipe_value["clip_vision"]
        insightface = None
        patched_model = model_value
    else:
        ipadapter, _unused, clip_vision, insightface, lora_loader = pipe_value
        if not callable(lora_loader):
            raise TypeError(
                "IPADAPTER_PIPE does not contain a host LoRA loader")
        patched_model = lora_loader(model_value)
    node_name = "IPAdapterBatch" if unfold_batch else "IPAdapterAdvanced"
    node_class = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get(node_name)
    if node_class is None:
        if getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get(
                "IPAdapterApply") is not None:
            raise RuntimeError(
                "ComfyUI IPAdapter Plus is installed but outdated; "
                "IPAdapterAdvanced is required")
        raise RuntimeError(
            "IP-Adapter application requires the host-installed "
            "ComfyUI IPAdapter Plus extension")

    result = node_class().apply_ipadapter(
        model=patched_model,
        ipadapter=ipadapter,
        weight=weight,
        weight_type=weight_type,
        start_at=start_percent,
        end_at=end_percent,
        combine_embeds=combine_embeds,
        clip_vision=clip_vision,
        image=pixels,
        image_negative=negative_pixels,
        attn_mask=mask_value,
        insightface=insightface,
        weight_faceidv2=weight_faceidv2,
        embeds_scaling=embeds_scaling,
        layer_weights=layer_weights,
        image_style=style_pixels,
        image_composition=composition_pixels,
        weight_style=weight_style,
        weight_composition=weight_composition,
        expand_style=expand_style,
    )
    if not isinstance(result, (tuple, list)) or not result:
        raise RuntimeError("IPAdapterAdvanced returned no model")
    return _sdk.ModelRef._wrap(await rt.refs.create(
        "MODEL", result[0]))  # type: ignore[return-value]

async def ipadapter_apply_tiled(pipeline: "Ref",
    model: "ModelRef",
    image: "ImageRef",
    negative_image: Optional["ImageRef"] = None,
    attn_mask: Optional["MaskRef"] = None,
    weight: float = 0.7,
    weight_type: str = "linear",
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    combine_embeds: str = "concat",
    embeds_scaling: str = "V only",
    sharpening: float = 0.0,
    unfold_batch: bool = False,
) -> tuple["ModelRef", "ImageRef", "MaskRef"]:
    """Invoke the host extension's canonical tiled operation."""
    import math
    import nodes
    import torch

    weight = float(weight)
    start_percent = float(start_percent)
    end_percent = float(end_percent)
    sharpening = float(sharpening)
    if not all(math.isfinite(value) for value in (
        weight, start_percent, end_percent, sharpening,
    )):
        raise ValueError("IP-Adapter tiled parameters must be finite")
    if not -1.0 <= weight <= 3.0:
        raise ValueError("IP-Adapter weight must be in [-1, 3]")
    if not 0.0 <= start_percent <= end_percent <= 1.0:
        raise ValueError(
            "IP-Adapter percentages must satisfy 0 <= start <= end <= 1")
    if not 0.0 <= sharpening <= 1.0:
        raise ValueError("IP-Adapter sharpening must be in [0, 1]")
    if weight_type not in {
        "linear", "ease in", "ease out", "ease in-out",
        "reverse in-out", "weak input", "weak output", "weak middle",
        "strong middle", "style transfer", "composition",
        "strong style transfer", "style and composition",
        "style transfer precise", "composition precise",
    }:
        raise ValueError("unsupported IP-Adapter weight type")
    if combine_embeds not in {
        "concat", "add", "subtract", "average", "norm average",
    }:
        raise ValueError("unsupported IP-Adapter embedding combination")
    if embeds_scaling not in {
        "V only", "K+V", "K+V w/ C penalty",
        "K+mean(V) w/ C penalty",
    }:
        raise ValueError("unsupported IP-Adapter embedding scaling")
    if type(unfold_batch) is not bool:
        raise TypeError("IP-Adapter unfold_batch must be a bool")

    rt = _sdk.current_runtime()
    pipe_value = await rt.refs.resolve(pipeline)
    if not _sdk._is_ipadapter_pipe(pipe_value):
        raise TypeError(
            "IPADAPTER_PIPE is not a host-created IP-Adapter pipeline")
    model_value = await rt.refs.resolve(model)
    pixels = await rt.refs.resolve(image)
    negative_pixels = (
        None if negative_image is None
        else await rt.refs.resolve(negative_image)
    )
    mask_value = (
        None if attn_mask is None
        else await rt.refs.resolve(attn_mask)
    )
    for name, value in (("image", pixels),
                        ("negative_image", negative_pixels)):
        if value is None:
            continue
        if (not isinstance(value, torch.Tensor) or value.ndim != 4
                or value.shape[-1] < 3 or value.shape[0] < 1
                or value.shape[0] > 4096):
            raise ValueError(
                f"IP-Adapter {name} must be a bounded BHWC image batch")
        height, width = map(int, value.shape[1:3])
        if (height <= 0 or width <= 0
                or height * width * int(value.shape[0]) > 268_435_456):
            raise ValueError(f"IP-Adapter {name} dimensions are invalid")
    if mask_value is not None and (
        not isinstance(mask_value, torch.Tensor)
        or mask_value.ndim not in (2, 3)
        or mask_value.numel() <= 0
        or mask_value.numel() > 268_435_456
    ):
        raise ValueError("IP-Adapter attention mask must be bounded HW/BHW")

    if isinstance(pipe_value, dict):
        ipadapter = pipe_value["ipadapter"]
        clip_vision = pipe_value["clip_vision"]
        patched_model = model_value
    else:
        ipadapter, _unused, clip_vision, _insightface, lora_loader = pipe_value
        if not callable(lora_loader):
            raise TypeError(
                "IPADAPTER_PIPE does not contain a host LoRA loader")
        patched_model = lora_loader(model_value)

    node_name = (
        "IPAdapterTiledBatch" if unfold_batch else "IPAdapterTiled")
    node_class = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get(node_name)
    if node_class is None:
        raise RuntimeError(
            "tiled IP-Adapter application requires the host-installed "
            "ComfyUI IPAdapter Plus extension")
    result = node_class().apply_tiled(
        model=patched_model,
        ipadapter=ipadapter,
        image=pixels,
        weight=weight,
        weight_type=weight_type,
        start_at=start_percent,
        end_at=end_percent,
        sharpening=sharpening,
        combine_embeds=combine_embeds,
        image_negative=negative_pixels,
        attn_mask=mask_value,
        clip_vision=clip_vision,
        embeds_scaling=embeds_scaling,
    )
    if (not isinstance(result, (tuple, list)) or len(result) < 3
            or not isinstance(result[1], torch.Tensor)
            or not isinstance(result[2], torch.Tensor)):
        raise RuntimeError("IPAdapterTiled returned invalid outputs")
    return (
        _sdk.ModelRef._wrap(await rt.refs.create("MODEL", result[0])),
        _sdk.ImageRef._wrap(await rt.refs.create("IMAGE", result[1])),
        _sdk.MaskRef._wrap(await rt.refs.create("MASK", result[2])),
    )

async def ipadapter_encode(pipeline: "Ref",
    image: "ImageRef",
    weight: float = 1.0,
    mask: Optional["MaskRef"] = None,
) -> tuple["Ref", "Ref"]:
    import math
    import nodes
    import torch

    weight = float(weight)
    if not math.isfinite(weight) or not -1.0 <= weight <= 3.0:
        raise ValueError("IP-Adapter embedding weight must be in [-1, 3]")
    rt = _sdk.current_runtime()
    pipe_value = await rt.refs.resolve(pipeline)
    if not _sdk._is_ipadapter_pipe(pipe_value):
        raise TypeError(
            "IPADAPTER_PIPE is not a host-created IP-Adapter pipeline")
    pixels = await rt.refs.resolve(image)
    mask_value = None if mask is None else await rt.refs.resolve(mask)
    if (not isinstance(pixels, torch.Tensor) or pixels.ndim != 4
            or pixels.shape[-1] < 3 or not 1 <= pixels.shape[0] <= 4096
            or pixels.shape[1] <= 0 or pixels.shape[2] <= 0
            or pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            > 268_435_456):
        raise ValueError(
            "IP-Adapter encoding requires a bounded BHWC image batch")
    if mask_value is not None and (
        not isinstance(mask_value, torch.Tensor)
        or mask_value.ndim not in (2, 3)
        or not 0 < mask_value.numel() <= 268_435_456
    ):
        raise ValueError("IP-Adapter mask must be bounded HW/BHW")
    if isinstance(pipe_value, dict):
        ipadapter = pipe_value["ipadapter"]
        clip_vision = pipe_value["clip_vision"]
    else:
        ipadapter, _unused, clip_vision, _insightface, _loader = pipe_value
    node_class = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get(
        "IPAdapterEncoder")
    if node_class is None:
        raise RuntimeError(
            "IP-Adapter encoding requires the host-installed "
            "ComfyUI IPAdapter Plus extension")
    result = node_class().encode(
        ipadapter=ipadapter,
        image=pixels,
        weight=weight,
        mask=mask_value,
        clip_vision=clip_vision,
    )
    if (not isinstance(result, (tuple, list)) or len(result) < 2
            or any(not isinstance(value, torch.Tensor) for value in result[:2])
            or any(not 0 < value.numel() <= 268_435_456
                   for value in result[:2])):
        raise RuntimeError("IPAdapterEncoder returned invalid embeddings")
    return (
        await rt.refs.create("IPADAPTER_EMBEDS", result[0]),
        await rt.refs.create("IPADAPTER_EMBEDS", result[1]),
    )

async def ipadapter_embeds_combine(first: "Ref",
    others: list["Ref"],
    method: str = "concat",
) -> "Ref":
    import nodes
    import torch

    if method not in {
        "concat", "add", "subtract", "average", "norm average",
        "max", "min",
    }:
        raise ValueError("unsupported IP-Adapter embedding combination")
    if not isinstance(others, list) or len(others) > 4:
        raise ValueError("at most five IP-Adapter embeddings may be combined")
    refs = [first, *others]
    if any(getattr(ref, "kind", None) != "IPADAPTER_EMBEDS" for ref in refs):
        raise TypeError("IP-Adapter embedding combination needs typed refs")
    rt = _sdk.current_runtime()
    values = [await rt.refs.resolve(ref) for ref in refs]
    if any(
        not isinstance(value, torch.Tensor)
        or value.ndim < 2
        or not 0 < value.numel() <= 268_435_456
        for value in values
    ):
        raise ValueError("IP-Adapter embeddings are invalid")
    node_class = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get(
        "IPAdapterCombineEmbeds")
    if node_class is None:
        raise RuntimeError(
            "IP-Adapter embedding combination requires the "
            "host-installed ComfyUI IPAdapter Plus extension")
    padded = values + [None] * (5 - len(values))
    result = node_class().batch(
        embed1=padded[0],
        embed2=padded[1],
        embed3=padded[2],
        embed4=padded[3],
        embed5=padded[4],
        method=method,
    )
    if (not isinstance(result, (tuple, list)) or not result
            or not isinstance(result[0], torch.Tensor)
            or not 0 < result[0].numel() <= 268_435_456):
        raise RuntimeError(
            "IPAdapterCombineEmbeds returned invalid embeddings")
    return await rt.refs.create("IPADAPTER_EMBEDS", result[0])

async def ipadapter_apply_embeds(pipeline: "Ref",
    model: "ModelRef",
    positive: "Ref",
    negative: Optional["Ref"] = None,
    attn_mask: Optional["MaskRef"] = None,
    weight: float = 1.0,
    weight_type: str = "linear",
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    embeds_scaling: str = "V only",
) -> "ModelRef":
    import math
    import nodes
    import torch

    weight = float(weight)
    start_percent = float(start_percent)
    end_percent = float(end_percent)
    if not all(math.isfinite(value) for value in (
        weight, start_percent, end_percent,
    )):
        raise ValueError("IP-Adapter embedding parameters must be finite")
    if not -1.0 <= weight <= 3.0:
        raise ValueError("IP-Adapter weight must be in [-1, 3]")
    if not 0.0 <= start_percent <= end_percent <= 1.0:
        raise ValueError(
            "IP-Adapter percentages must satisfy 0 <= start <= end <= 1")
    if weight_type not in {
        "linear", "ease in", "ease out", "ease in-out",
        "reverse in-out", "weak input", "weak output", "weak middle",
        "strong middle", "style transfer", "composition",
        "strong style transfer", "style and composition",
        "style transfer precise", "composition precise",
    }:
        raise ValueError("unsupported IP-Adapter weight type")
    if embeds_scaling not in {
        "V only", "K+V", "K+V w/ C penalty",
        "K+mean(V) w/ C penalty",
    }:
        raise ValueError("unsupported IP-Adapter embedding scaling")
    if getattr(positive, "kind", None) != "IPADAPTER_EMBEDS" or (
        negative is not None
        and getattr(negative, "kind", None) != "IPADAPTER_EMBEDS"
    ):
        raise TypeError("IP-Adapter application needs typed embedding refs")

    rt = _sdk.current_runtime()
    pipe_value = await rt.refs.resolve(pipeline)
    if not _sdk._is_ipadapter_pipe(pipe_value):
        raise TypeError(
            "IPADAPTER_PIPE is not a host-created IP-Adapter pipeline")
    model_value = await rt.refs.resolve(model)
    pos_value = await rt.refs.resolve(positive)
    neg_value = None if negative is None else await rt.refs.resolve(negative)
    mask_value = (
        None if attn_mask is None else await rt.refs.resolve(attn_mask))
    for value in (pos_value, neg_value):
        if value is not None and (
            not isinstance(value, torch.Tensor)
            or value.ndim < 2
            or not 0 < value.numel() <= 268_435_456
        ):
            raise ValueError("IP-Adapter embeddings are invalid")
    if mask_value is not None and (
        not isinstance(mask_value, torch.Tensor)
        or mask_value.ndim not in (2, 3)
        or not 0 < mask_value.numel() <= 268_435_456
    ):
        raise ValueError("IP-Adapter attention mask must be bounded HW/BHW")
    if isinstance(pipe_value, dict):
        ipadapter = pipe_value["ipadapter"]
        clip_vision = pipe_value["clip_vision"]
        patched_model = model_value
    else:
        ipadapter, _unused, clip_vision, _insightface, loader = pipe_value
        if not callable(loader):
            raise TypeError(
                "IPADAPTER_PIPE does not contain a host LoRA loader")
        patched_model = loader(model_value)
    node_class = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get(
        "IPAdapterEmbeds")
    if node_class is None:
        raise RuntimeError(
            "IP-Adapter embedding application requires the "
            "host-installed ComfyUI IPAdapter Plus extension")
    result = node_class().apply_ipadapter(
        model=patched_model,
        ipadapter=ipadapter,
        pos_embed=pos_value,
        neg_embed=neg_value,
        attn_mask=mask_value,
        clip_vision=clip_vision,
        weight=weight,
        weight_type=weight_type,
        start_at=start_percent,
        end_at=end_percent,
        embeds_scaling=embeds_scaling,
    )
    if not isinstance(result, (tuple, list)) or not result:
        raise RuntimeError("IPAdapterEmbeds returned no model")
    return _sdk.ModelRef._wrap(await rt.refs.create(
        "MODEL", result[0]))  # type: ignore[return-value]

async def image_preprocessor_apply(preprocessor: "ImagePreprocessorRef",
    image: "ImageRef",
    mask: Optional["MaskRef"] = None,
) -> "ImageRef":
    """Invoke one trusted provider object's image-to-image operation."""
    import torch

    rt = _sdk.current_runtime()
    provider = await rt.refs.resolve(preprocessor)
    apply = getattr(provider, "apply", None)
    if not callable(apply) or not _sdk._is_image_preprocessor(provider):
        raise TypeError(
            "IMAGE_PREPROCESSOR is not a recognized host provider")
    pixels = await rt.refs.resolve(image)
    mask_value = None if mask is None else await rt.refs.resolve(mask)
    if (not isinstance(pixels, torch.Tensor) or pixels.ndim != 4
            or pixels.shape[0] < 1 or pixels.shape[0] > 4096
            or pixels.shape[-1] < 3):
        raise ValueError(
            "image preprocessing requires a bounded BHWC image batch")
    height, width = map(int, pixels.shape[1:3])
    if (height <= 0 or width <= 0
            or height * width * int(pixels.shape[0]) > 268_435_456):
        raise ValueError("image preprocessor input dimensions are invalid")
    if mask_value is not None:
        if (not isinstance(mask_value, torch.Tensor)
                or mask_value.ndim not in {2, 3, 4}):
            raise ValueError("image preprocessor mask has an invalid shape")
    result = apply(pixels, mask_value)
    if (not isinstance(result, torch.Tensor) or result.ndim != 4
            or result.shape[0] < 1 or result.shape[0] > 4096
            or result.shape[-1] < 3):
        raise RuntimeError(
            "image preprocessor returned an invalid image batch")
    out_height, out_width = map(int, result.shape[1:3])
    if (out_height <= 0 or out_width <= 0
            or out_height * out_width * int(result.shape[0])
            > 268_435_456):
        raise RuntimeError(
            "image preprocessor output dimensions are invalid")
    return _sdk.ImageRef._wrap(await rt.refs.create(
        "IMAGE", result))  # type: ignore[return-value]

async def sam_segment(sam: "SamModelRef", image: "ImageRef",
    boxes: list[Optional[list[float]]],
    point_coords: Optional[list[list[list[float]]]] = None,
    point_labels: Optional[list[list[int]]] = None,
    multimask_output: bool = True,
) -> tuple["MaskRef", list[list[float]]]:
    from contextlib import nullcontext
    import math
    import numpy as np
    import torch
    import comfy.model_management
    from segment_anything import SamPredictor

    rt = _sdk.current_runtime()
    bundle = await rt.refs.resolve(sam)
    if (not isinstance(bundle, dict)
            or bundle.get("secure_kind") != "sam.v1"):
        raise TypeError("SAM_MODEL is not a trusted SAM v1 bundle")
    pixels = await rt.refs.resolve(image)
    if (not isinstance(pixels, torch.Tensor) or pixels.ndim != 4
            or pixels.shape[0] != 1 or pixels.shape[-1] < 3):
        raise ValueError("SAM segmentation requires one BHWC RGB image")
    height, width = int(pixels.shape[1]), int(pixels.shape[2])
    if (height <= 0 or width <= 0 or height * width > 268_435_456):
        raise ValueError("SAM image dimensions are invalid or too large")
    if not isinstance(boxes, (list, tuple)) or not 1 <= len(boxes) <= 1024:
        raise ValueError("SAM boxes must contain 1 to 1024 queries")
    if type(multimask_output) is not bool:
        raise TypeError("SAM multimask_output must be a bool")

    query_count = len(boxes)
    if point_coords is None:
        point_coords = [[] for _ in range(query_count)]
    if point_labels is None:
        point_labels = [[] for _ in range(query_count)]
    if (not isinstance(point_coords, (list, tuple))
            or not isinstance(point_labels, (list, tuple))
            or len(point_coords) != query_count
            or len(point_labels) != query_count):
        raise ValueError("SAM point hints must match the box query count")

    def finite_number(value: Any) -> bool:
        return (type(value) in (int, float)
                and math.isfinite(float(value)))

    normalized_boxes = []
    normalized_points = []
    normalized_labels = []
    total_points = 0
    for box, points, labels in zip(boxes, point_coords, point_labels):
        if box is None:
            normalized_box = None
        else:
            if (not isinstance(box, (list, tuple)) or len(box) != 4
                    or not all(finite_number(value) for value in box)):
                raise ValueError("each SAM box must be x1,y1,x2,y2 or null")
            x1, y1, x2, y2 = (float(value) for value in box)
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("SAM boxes must be inside the image")
            normalized_box = [x1, y1, x2, y2]
        if (not isinstance(points, (list, tuple))
                or not isinstance(labels, (list, tuple))
                or len(points) != len(labels)):
            raise ValueError("SAM points and labels must have equal lengths")
        if len(points) > 4096:
            raise ValueError("a SAM query may contain at most 4096 points")
        query_points = []
        query_labels = []
        for point, label in zip(points, labels):
            if (not isinstance(point, (list, tuple)) or len(point) != 2
                    or not all(finite_number(value) for value in point)):
                raise ValueError("each SAM point must be an x,y pair")
            x, y = float(point[0]), float(point[1])
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError("SAM points must be inside the image")
            if type(label) is not int or label not in (0, 1):
                raise ValueError("SAM point labels must be 0 or 1")
            query_points.append([x, y])
            query_labels.append(label)
        if normalized_box is None and not query_points:
            raise ValueError("each SAM query needs a box or point hint")
        total_points += len(query_points)
        normalized_boxes.append(normalized_box)
        normalized_points.append(query_points)
        normalized_labels.append(query_labels)
    if total_points > 65_536:
        raise ValueError("SAM request contains too many point hints")

    model = bundle["model"]
    model_lock = bundle.get("lock")
    device_mode = bundle["device_mode"]
    device = (
        torch.device("cpu") if device_mode == "CPU"
        else comfy.model_management.get_torch_device()
    )
    source = np.clip(
        pixels[0, ..., :3].detach().cpu().numpy() * 255.0,
        0, 255,
    ).astype(np.uint8)
    masks_by_query = []
    scores_by_query = []
    with model_lock if model_lock is not None else nullcontext():
        model.to(device)
        try:
            predictor = SamPredictor(model)
            with torch.inference_mode():
                predictor.set_image(source, image_format="RGB")
                for box, points, labels in zip(
                    normalized_boxes, normalized_points, normalized_labels,
                ):
                    masks, scores, _logits = predictor.predict(
                        point_coords=(
                            None if not points
                            else np.asarray(points, dtype=np.float32)),
                        point_labels=(
                            None if not labels
                            else np.asarray(labels, dtype=np.int64)),
                        box=(
                            None if box is None
                            else np.asarray(box, dtype=np.float32)),
                        multimask_output=multimask_output,
                        return_logits=False,
                    )
                    masks_by_query.append(torch.from_numpy(masks).float())
                    scores_by_query.append([
                        float(score) for score in np.asarray(scores).tolist()
                    ])
        finally:
            if device_mode != "Prefer GPU":
                model.to("cpu")
    if device_mode == "AUTO":
        comfy.model_management.soft_empty_cache()
    output = torch.stack(masks_by_query, dim=0).cpu()
    return (
        _sdk.MaskRef._wrap(await rt.refs.create("MASK", output)),
        scores_by_query,
    )  # type: ignore[return-value]

async def sam_segment_video(sam: "SamModelRef", frames: "ImageRef",
    boxes: list[list[float]],
) -> "MaskRef":
    from contextlib import nullcontext
    import math
    import torch
    import torch.nn.functional as functional
    import comfy.model_management

    rt = _sdk.current_runtime()
    bundle = await rt.refs.resolve(sam)
    if (not isinstance(bundle, dict)
            or bundle.get("secure_kind") != "sam.v2"):
        raise TypeError("SAM_MODEL is not a trusted SAM2 bundle")
    pixels = await rt.refs.resolve(frames)
    if (not isinstance(pixels, torch.Tensor) or pixels.ndim != 4
            or pixels.shape[-1] < 3 or pixels.shape[0] < 1):
        raise ValueError("SAM2 requires a non-empty BHWC RGB video")
    frame_count, height, width = map(int, pixels.shape[:3])
    if (frame_count > 1024
            or height <= 0 or width <= 0
            or frame_count * height * width > 268_435_456):
        raise ValueError("SAM2 video dimensions are invalid or too large")
    if not isinstance(boxes, (list, tuple)) or not 1 <= len(boxes) <= 128:
        raise ValueError("SAM2 boxes must contain 1 to 128 queries")
    normalized_boxes = []
    for box in boxes:
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or not all(type(value) in (int, float)
                           and math.isfinite(float(value))
                           for value in box)):
            raise ValueError("each SAM2 box must be x1,y1,x2,y2")
        x1, y1, x2, y2 = (float(value) for value in box)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("SAM2 boxes must be inside the video frame")
        normalized_boxes.append([x1, y1, x2, y2])

    predictor = bundle["model"]
    model_lock = bundle.get("lock")
    device_mode = bundle["device_mode"]
    device = (
        torch.device("cpu") if device_mode == "CPU"
        else comfy.model_management.get_torch_device()
    )
    image_size = int(predictor.image_size)
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, min(image_size, int(width * scale)))
    resized_height = max(1, min(image_size, int(height * scale)))
    pad_left = (image_size - resized_width) // 2
    pad_right = image_size - resized_width - pad_left
    pad_top = (image_size - resized_height) // 2
    pad_bottom = image_size - resized_height - pad_top
    source = functional.interpolate(
        pixels[..., :3].movedim(-1, 1).float(),
        size=(resized_height, resized_width),
        mode="bilinear", align_corners=False,
    )
    source = functional.pad(
        source, (pad_left, pad_right, pad_top, pad_bottom))
    mean = source.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = source.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    source = (source - mean) / std
    adjusted_boxes = [[
        box[0] * resized_width / width + pad_left,
        box[1] * resized_height / height + pad_top,
        box[2] * resized_width / width + pad_left,
        box[3] * resized_height / height + pad_top,
    ] for box in normalized_boxes]

    inference_state = None
    with model_lock if model_lock is not None else nullcontext():
        predictor.to(device)
        try:
            inference_state = {
                "images": source,
                "num_frames": frame_count,
                "video_height": image_size,
                "video_width": image_size,
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": device_mode == "CPU",
                "device": predictor.device,
                "storage_device": (
                    torch.device("cpu") if device_mode == "CPU"
                    else predictor.device),
                "point_inputs_per_obj": {},
                "mask_inputs_per_obj": {},
                "cached_features": {},
                "constants": {},
                "obj_id_to_idx": OrderedDict(),
                "obj_idx_to_id": OrderedDict(),
                "obj_ids": [],
                "output_dict_per_obj": {},
                "temp_output_dict_per_obj": {},
                "frames_tracked_per_obj": {},
            }
            predictor._get_image_feature(
                inference_state, frame_idx=0, batch_size=1)
            for object_id, box in enumerate(adjusted_boxes):
                center = [[
                    (box[0] + box[2]) / 2.0,
                    (box[1] + box[3]) / 2.0,
                ]]
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=object_id,
                    points=center,
                    labels=[1],
                    box=box,
                )
            logits: list[list[Optional[torch.Tensor]]] = [
                [None] * frame_count for _ in adjusted_boxes
            ]
            for frame_index, object_ids, masks in (
                predictor.propagate_in_video(inference_state)
            ):
                for mask_index, object_id in enumerate(object_ids):
                    logits[int(object_id)][int(frame_index)] = (
                        masks[mask_index, 0].detach().cpu())
            if any(mask is None for item in logits for mask in item):
                raise RuntimeError("SAM2 did not return every object frame")
            output = torch.stack([
                torch.stack(item) for item in logits
            ])
            output = output[
                :, :, pad_top:image_size - pad_bottom,
                pad_left:image_size - pad_right,
            ]
            output = functional.interpolate(
                output.flatten(0, 1).unsqueeze(1),
                size=(height, width), mode="bilinear",
                align_corners=False,
            ).squeeze(1).unflatten(0, (len(adjusted_boxes), frame_count))
        finally:
            if inference_state is not None:
                predictor.reset_state(inference_state)
            if device_mode != "Prefer GPU":
                predictor.to("cpu")
    if device_mode == "AUTO":
        comfy.model_management.soft_empty_cache()
    return _sdk.MaskRef._wrap(
        await rt.refs.create("MASK", output.cpu().float())
    )  # type: ignore[return-value]
