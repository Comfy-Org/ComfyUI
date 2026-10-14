"""Fixed canonical concat phase; no pack algorithm or host object is exported."""
from __future__ import annotations

import asyncio
import inspect


def canonical_model(patcher):
    from comfy.model_base import BaseModel, WAN21
    from comfy.model_patcher import ModelPatcher
    from comfy.latent_formats import Wan21

    if not isinstance(patcher, ModelPatcher) or type(patcher.model) is not WAN21:
        raise TypeError("concat conditioning requires a canonical WAN21 MODEL")
    model = patcher.model
    if (type(model.latent_format) is not Wan21
            or getattr(model.process_latent_in, "__func__", None) is not BaseModel.process_latent_in
            or getattr(model.latent_format.process_in, "__func__", None) is not Wan21.process_in):
        raise TypeError("concat conditioning requires canonical latent normalization")
    return model


def tensor(value, label):
    import torch
    from ._node_closures import MAX_CAPTURE_BYTES

    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.device.type == "meta" or not torch.is_floating_point(value)
            or value.ndim != 5 or any(size < 1 for size in value.shape)):
        raise TypeError(f"concat {label} must be a dense floating BCTHW tensor")
    if value.numel() * value.element_size() > MAX_CAPTURE_BYTES:
        raise ValueError("concat tensor exceeds the closure phase byte limit")
    return value


class Invocation:
    """One trusted model/concat call; also the local implementation of the broker."""

    def __init__(self, model, kwargs):
        self.model = model
        self.noise = tensor(kwargs.get("noise"), "noise")
        self.extra = int(model.diffusion_model.patch_embedding.weight.shape[1]) - self.noise.shape[1]
        self.kwargs = dict(kwargs)
        self.active = True
        self.image_read = False
        self.mask_read = False
        self.normalizations = 0
        self.image = None
        self.mask = None
        self.result_issued = False
        self.input_bytes = self.noise.numel() * self.noise.element_size()

    def current(self):
        if not self.active:
            raise RuntimeError("concat invocation is closed")

    def metadata(self):
        from ._sdk import ModelConcatMetadata

        self.current()
        # Native zero-extra return is handled before this required lookup.
        device = self.kwargs["device"]
        return ModelConcatMetadata(
            patch_input_channels=int(self.model.diffusion_model.patch_embedding.weight.shape[1]),
            image_to_video=bool(self.model.image_to_video), device=str(device))

    def get_image(self):
        self.current()
        if self.image_read:
            raise RuntimeError("concat image can be read only once")
        self.image_read = True
        self.image = self.kwargs.get("concat_latent_image")
        if self.image is not None:
            tensor(self.image, "image")
            self.charge(self.image)
        return self.image

    def image_shape(self):
        if not self.image_read:
            raise RuntimeError("concat image has not been selected")
        if self.image is None:
            return tuple(self.noise.shape[:1]) + (self.extra,) + tuple(self.noise.shape[2:])
        return (self.noise.shape[0], self.image.shape[1], self.image.shape[2], *self.noise.shape[-2:])

    def needs_mask(self):
        shape = self.image_shape()
        return self.model.image_to_video and self.extra != shape[1]

    def get_mask(self):
        self.current()
        if self.mask_read or not self.needs_mask():
            raise RuntimeError("concat mask is not available at this phase")
        if self.normalizations != self.normalization_count():
            raise RuntimeError("concat normalization must finish before mask access")
        self.mask_read = True
        self.mask = self.kwargs.get("concat_mask", self.kwargs.get("denoise_mask"))
        if self.mask is not None:
            tensor(self.mask, "mask")
            self.charge(self.mask)
        return self.mask

    def charge(self, value):
        from ._node_closures import MAX_CAPTURE_BYTES
        self.input_bytes += value.numel() * value.element_size()
        if self.input_bytes > MAX_CAPTURE_BYTES:
            self.active = False
            raise ValueError("concat selected inputs exceed the closure phase byte limit")

    def normalization_count(self):
        return 0 if self.image is None else (self.image.shape[1] + 15) // 16

    def normalize(self, value):
        self.current()
        if not self.image_read or self.image is None or self.mask_read:
            raise RuntimeError("concat normalization is not available at this phase")
        count = self.normalization_count()
        if self.normalizations >= count:
            raise RuntimeError("concat exceeded its native normalization call count")
        value = tensor(value, "normalization input")
        from ._node_closures import MAX_CAPTURE_BYTES
        maximum_result_bytes = value.numel() * value.element_size() * (16 if value.shape[1] == 1 else 1)
        if self.input_bytes + value.numel() * value.element_size() + maximum_result_bytes > MAX_CAPTURE_BYTES:
            raise ValueError("concat normalization working set exceeds the closure phase byte limit")
        start = self.normalizations * 16
        channels = min(16, self.image.shape[1] - start)
        expected = (self.image.shape[0], channels, self.image.shape[2], *self.noise.shape[-2:])
        if (tuple(value.shape) != expected or value.dtype != self.image.dtype
                or str(value.device) != str(self.kwargs["device"])):
            raise ValueError("concat normalization input changed its source geometry, dtype or device")
        self.normalizations += 1
        # Do not impose a full-16 slice or mask the genuine native error.
        result = self.model.process_latent_in(value)
        tensor(result, "normalization result")
        return result

    def result_signature(self):
        import torch

        self.current()
        if not self.image_read or self.normalizations != self.normalization_count():
            raise RuntimeError("concat result precedes native image preparation")
        shape = self.image_shape()
        dtype = self.noise.dtype if self.image is None else self.image.dtype
        device = self.noise.device if self.image is None else torch.device(self.kwargs["device"])
        if self.needs_mask():
            if not self.mask_read:
                raise RuntimeError("concat result precedes native mask selection")
            stop = self.extra - 4
            channels = len(range(shape[1])[:stop]) if shape[1] > stop else shape[1]
            mask_channels = min(4, self.noise.shape[1]) if self.mask is None else 4
            shape = (1, channels + mask_channels, *shape[2:])
            dtype = torch.promote_types(dtype, self.noise.dtype if self.mask is None else self.mask.dtype)
        return shape, dtype, device

    def validate_result(self, value):
        value = tensor(value, "result")
        shape, dtype, device = self.result_signature()
        if tuple(value.shape) != shape or value.dtype not in self.result_dtypes() or value.device != device:
            raise ValueError("concat result changed its native branch shape, dtype or device")
        return value

    def result_dtypes(self):
        import torch
        _, dtype, _ = self.result_signature()
        # Tensor-only motion math may naturally promote low precision through
        # floating coordinate/weight calculations. Early image returns do not.
        allowed = {dtype}
        if self.needs_mask() and dtype in {torch.float16, torch.bfloat16}:
            allowed.add(torch.float32)
        return allowed

    async def concat_latent_image(self):
        try:
            return self.get_image()
        except Exception:
            self.active = False
            raise

    async def concat_mask(self):
        try:
            return self.get_mask()
        except Exception:
            self.active = False
            raise

    async def process_latent_in(self, value):
        try:
            return self.normalize(value)
        except Exception:
            self.active = False
            raise

    def close(self):
        self.active = False
        self.kwargs.clear()
        self.image = self.mask = self.model = self.noise = None


def attach(patcher, fn):
    model = canonical_model(patcher)
    clone = patcher.clone()

    def concat_cond(**kwargs):
        invocation = Invocation(model, kwargs)
        try:
            if invocation.extra == 0:
                return None
            metadata = invocation.metadata()
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                raise RuntimeError("concat conditioning must run on its owning preparation worker")
            result = fn(invocation, invocation.noise, metadata)
            if inspect.isawaitable(result):
                result = asyncio.run(result)
            return invocation.validate_result(result)
        finally:
            invocation.close()

    clone.add_object_patch("concat_cond", concat_cond)
    return clone
