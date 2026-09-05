from __future__ import annotations

import base64
import inspect
import io
import logging
import queue
import threading
import time


logger = logging.getLogger(__name__)


def _bounded_int(name, value, minimum, maximum):
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _settings(
    max_resolution, jpeg_quality, suppress_default_preview,
    preview_frames, preview_fps, tiny_vae,
):
    if type(suppress_default_preview) is not bool:
        raise TypeError("suppress_default_preview must be a boolean")
    if type(tiny_vae) is not str or not tiny_vae or len(tiny_vae) > 1024:
        raise ValueError("tiny_vae must be a logical catalogue name")
    if tiny_vae != "none":
        import folder_paths

        if ("\x00" in tiny_vae or tiny_vae.startswith(("/", "\\"))
                or ".." in tiny_vae.replace("\\", "/").split("/")
                or tiny_vae not in folder_paths.get_filename_list(
                    "vae_approx")):
            raise ValueError(
                f"tiny VAE {tiny_vae!r} is not in the vae_approx catalogue")
    return {
        "max_resolution": _bounded_int(
            "max_resolution", max_resolution, 0, 8192),
        "jpeg_quality": _bounded_int(
            "jpeg_quality", jpeg_quality, 30, 100),
        "suppress_default_preview": suppress_default_preview,
        "preview_frames": _bounded_int(
            "preview_frames", preview_frames, 1, 1024),
        "preview_fps": _bounded_int("preview_fps", preview_fps, 1, 60),
        "tiny_vae": tiny_vae,
    }


class _AsyncEncoder:
    _STOP = object()

    def __init__(self):
        self._queue = queue.Queue(maxsize=2)
        self._thread = threading.Thread(
            target=self._run, name="comfy_preview_override", daemon=True)
        self._thread.start()

    def submit(self, fn):
        try:
            self._queue.put_nowait(fn)
            return True
        except queue.Full:
            return False

    def _run(self):
        while True:
            item = self._queue.get()
            if item is self._STOP:
                return
            try:
                item()
            except Exception:
                logger.exception("preview override encoder failed")

    def close(self):
        try:
            self._queue.put(self._STOP, timeout=5.0)
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)


def _fit_rgb_frames(frames, max_resolution):
    from PIL import Image, ImageOps

    result = []
    for frame in frames:
        frame = frame if frame.mode == "RGB" else frame.convert("RGB")
        if (max_resolution > 0 and
                (frame.width > max_resolution
                 or frame.height > max_resolution)):
            frame = ImageOps.contain(
                frame, (max_resolution, max_resolution), Image.Resampling.LANCZOS)
        result.append(frame)
    return result


def _encode_mp4(frames, fps, max_resolution):
    try:
        import av

        av.Codec("h264_nvenc", "w")
    except Exception:
        return None, 0, 0
    frames = _fit_rgb_frames(frames, max_resolution)
    width, height = frames[0].width & ~1, frames[0].height & ~1
    if width < 145 or height < 49:
        return None, 0, 0
    if (width, height) != frames[0].size:
        from PIL import Image

        frames = [frame.resize(
            (width, height), Image.Resampling.LANCZOS) for frame in frames]
    for options in (
        {"preset": "p1", "rc": "vbr", "cq": "23"},
        {"preset": "p1"},
    ):
        buffer = io.BytesIO()
        try:
            container = av.open(
                buffer, mode="w", format="mp4",
                options={
                    "movflags":
                        "frag_keyframe+empty_moov+default_base_moof",
                })
            stream = container.add_stream("h264_nvenc", rate=max(1, fps))
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.options = options
            for frame in frames:
                for packet in stream.encode(av.VideoFrame.from_image(frame)):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            return (
                base64.b64encode(buffer.getvalue()).decode("ascii"),
                width,
                height,
            )
        except Exception:
            continue
    return None, 0, 0


def _encode_webp(frames, fps, quality, max_resolution):
    frames = _fit_rgb_frames(frames, max_resolution)
    buffer = io.BytesIO()
    frames[0].save(
        buffer, format="WEBP", save_all=True,
        append_images=frames[1:], duration=max(1, round(1000 / fps)),
        loop=0, quality=quality, method=4)
    return (
        base64.b64encode(buffer.getvalue()).decode("ascii"),
        frames[0].width,
        frames[0].height,
    )


def _encode_jpeg(frame, quality, max_resolution):
    frame = _fit_rgb_frames([frame], max_resolution)[0]
    buffer = io.BytesIO()
    frame.save(buffer, format="JPEG", quality=quality)
    return (
        base64.b64encode(buffer.getvalue()).decode("ascii"),
        frame.width,
        frame.height,
    )


def _normalize_packed(value, latent_shapes, keyframes):
    if latent_shapes:
        target = latent_shapes[0]
        if value.ndim == 3 and len(target) >= 3:
            length = 1
            for dimension in target[1:]:
                length *= int(dimension)
            value = value[:, :, :length].reshape(
                [value.shape[0]] + list(target)[1:])
    if keyframes > 0 and value.ndim == 5:
        value = value[:, :, :-keyframes]
    return value


def _keyframe_count(guider):
    try:
        import torch

        positive = guider.conds.get("positive")
        keyframes = positive[0].get("keyframe_idxs") if positive else None
        return (int(torch.unique(keyframes[0, 0, :, 0]).numel())
                if keyframes is not None else 0)
    except Exception:
        return 0


def _preview_rate(value):
    import math

    if type(value) not in {int, float}:
        raise TypeError("preview_rate must be a number")
    value = float(value)
    if not math.isfinite(value) or not 1.0 <= value <= 60.0:
        raise ValueError("preview_rate must be in [1, 60]")
    return value


def _decode_video_l2rgb(value, latent_format, max_frames):
    import numpy
    import torch
    from PIL import Image

    if value.ndim != 5:
        return []
    factors = getattr(latent_format, "latent_rgb_factors", None)
    if factors is None:
        return []
    reshape = getattr(latent_format, "latent_rgb_factors_reshape", None)
    value = reshape(value) if reshape is not None else value
    factors = torch.tensor(
        factors, device=value.device, dtype=value.dtype).transpose(0, 1)
    bias = getattr(latent_format, "latent_rgb_factors_bias", None)
    bias = (torch.tensor(bias, device=value.device, dtype=value.dtype)
            if bias is not None else None)
    value = value[0]
    if 0 < max_frames < value.shape[1]:
        indices = numpy.linspace(
            0, value.shape[1] - 1, max_frames).round().astype(int).tolist()
        value = value[:, indices]
    rgb = torch.nn.functional.linear(
        value.movedim(0, -1), factors, bias=bias)
    rgb.add_(1.0).mul_(127.5).clamp_(0, 255)
    arrays = rgb.to(torch.uint8).cpu().numpy()
    return [Image.fromarray(array) for array in arrays]


class _LtxPreviewer:
    def __init__(self, factors, bias, vae=None):
        import torch

        self._factors = torch.tensor(factors, device="cpu").transpose(0, 1)
        self._bias = (torch.tensor(bias, device="cpu")
                      if bias is not None else None)
        self._vae = vae

    def decode(self, value):
        import torch

        if self._vae is not None:
            device = next(self._vae.first_stage_model.parameters()).device
            dtype = self._vae.first_stage_model.decoder[1].weight.dtype
            decoded = self._vae.first_stage_model.decode(
                value.unsqueeze(0).to(device=device, dtype=dtype))
            return decoded[0].permute(1, 2, 3, 0)
        factors = self._factors.to(device=value.device, dtype=value.dtype)
        bias = (self._bias.to(device=value.device, dtype=value.dtype)
                if self._bias is not None else None)
        return torch.sigmoid(torch.nn.functional.linear(
            value.movedim(1, -1), factors, bias=bias))


class _Ltx2SamplingPreviewer:
    def __init__(self, factors, bias, rate, vae=None):
        self._decoder = _LtxPreviewer(factors, bias, vae)
        self._rate = rate
        self._first = True
        self._last_time = 0.0
        self._index = 0
        self._taehv = vae is not None

    def decode_latent_to_preview_image(self, _preview_format, value):
        if value.ndim == 5:
            value = value.movedim(2, 1)
            value = value.reshape((-1,) + value.shape[-3:])
        count = value.size(0)
        now = time.time()
        preview_count = int((now - self._last_time) * self._rate)
        self._last_time += preview_count / self._rate
        preview_count = min(preview_count, count)
        if preview_count <= 0:
            return None
        if self._first:
            self._first = False
            from server import PromptServer

            instance = PromptServer.instance
            instance.send_sync("VHS_latentpreview", {
                "length": count,
                "rate": self._rate,
                "id": instance.last_node_id,
            })
            self._last_time = now + 1 / self._rate
        if self._index + preview_count > count:
            selected = value.roll(-self._index, 0)[:preview_count]
        else:
            selected = value[self._index:self._index + preview_count]
        self._send(selected, self._index, count)
        self._index = (self._index + preview_count) % count
        return None

    def _send(self, value, index, total):
        import struct
        from io import BytesIO

        import torch
        import torch.nn.functional as functional
        from PIL import Image
        from server import BinaryEventTypes, PromptServer

        decoded = self._decoder.decode(value)
        if decoded.size(1) < 256 or decoded.size(2) < 256:
            decoded = functional.interpolate(
                decoded.movedim(-1, 0), scale_factor=4,
                mode="nearest").movedim(0, -1)
        if decoded.size(1) > 512 or decoded.size(2) > 512:
            channels_first = decoded.movedim(-1, 0)
            if channels_first.size(2) < channels_first.size(3):
                height = (
                    512 * channels_first.size(2)
                    // channels_first.size(3))
                channels_first = functional.interpolate(
                    channels_first, (height, 512), mode="nearest")
            else:
                width = (
                    512 * channels_first.size(3)
                    // channels_first.size(2))
                channels_first = functional.interpolate(
                    channels_first, (512, width), mode="nearest")
            decoded = channels_first.movedim(0, -1)
        previews = decoded.clamp(0, 1).mul(255).to(
            device="cpu", dtype=torch.uint8)
        instance = PromptServer.instance
        for preview in previews:
            message = BytesIO()
            message.write((1).to_bytes(length=4, byteorder="big") * 2)
            message.write(index.to_bytes(length=4, byteorder="big"))
            message.write(struct.pack(
                "16p", (instance.last_node_id or "").encode("ascii")))
            Image.fromarray(preview.numpy()).save(
                message, format="JPEG", quality=95, compress_level=1)
            instance.send_sync(
                BinaryEventTypes.PREVIEW_IMAGE,
                message.getvalue(), instance.client_id)
            modulus = ((total - 1) * 8 + 1) if self._taehv else total
            index = (index + 1) % modulus


def _unwrap_latent_upscaler(value):
    return getattr(value, "model", value)


class Ltx2SamplingPreviewWrapper:
    def __init__(
        self, *, latent_upscale_model=None, vae=None,
        preview_rate=8.0, taehv=False,
    ):
        self.latent_upscale_model = latent_upscale_model
        self.vae = vae
        self.preview_rate = preview_rate
        self.taeltx = taehv
        self.x0_output = {}

    def __call__(
        self, executor, noise, latent_image, sampler, sigmas,
        denoise_mask, callback, disable_pbar, seed, latent_shapes,
    ):
        import comfy.model_management as model_management
        import comfy.utils
        from ._preview_ltx_factors import (
            LTX2_RGB_BIAS, LTX2_RGB_FACTORS,
            LTX_RGB_BIAS, LTX_RGB_FACTORS,
        )

        guider = executor.class_obj
        diffusion = guider.model_patcher.model.diffusion_model
        is_ltx2 = not diffusion.caption_projection_first_linear
        factors = LTX2_RGB_FACTORS if is_ltx2 else LTX_RGB_FACTORS
        bias = LTX2_RGB_BIAS if is_ltx2 else LTX_RGB_BIAS
        upscaler = (
            _unwrap_latent_upscaler(self.latent_upscale_model)
            if self.latent_upscale_model is not None else None)
        target_device = model_management.get_torch_device()
        vae_device = None
        if upscaler is not None:
            upscaler.to(target_device)
        if self.vae is not None and self.taeltx:
            try:
                vae_device = next(
                    self.vae.first_stage_model.parameters()).device
            except StopIteration:
                vae_device = None
            self.vae.first_stage_model.to(target_device)

        previewer = _Ltx2SamplingPreviewer(
            factors, bias, self.preview_rate,
            self.vae if self.taeltx else None)
        progress = comfy.utils.ProgressBar(len(sigmas) - 1)
        keyframes = _keyframe_count(guider)
        shape = (
            latent_shapes[0]
            if latent_shapes is not None and len(latent_shapes) > 1
            else None)
        upscaler_dtype = (
            next(upscaler.parameters()).dtype
            if upscaler is not None else None)

        def preview_callback(step, x0, value, total_steps):
            original_x0 = x0
            if x0 is not None and shape is not None:
                length = 1
                for dimension in shape[1:]:
                    length *= int(dimension)
                x0 = x0[:, :, :length].reshape(
                    [x0.shape[0]] + list(shape)[1:])
            if keyframes > 0:
                x0 = x0[:, :, :-keyframes]
            if upscaler is not None:
                statistics = self.vae.first_stage_model.per_channel_statistics
                x0 = statistics.un_normalize(x0)
                x0 = upscaler(x0.to(upscaler_dtype))
                x0 = statistics.normalize(x0)
            previewer.decode_latent_to_preview_image("JPEG", x0)
            progress.update_absolute(step + 1, total_steps, None)
            if callback is not None:
                callback(step, original_x0, value, total_steps)

        try:
            return executor(
                noise, latent_image, sampler, sigmas, denoise_mask,
                preview_callback, disable_pbar, seed,
                latent_shapes=latent_shapes)
        finally:
            if upscaler is not None:
                upscaler.to(model_management.unet_offload_device())
            if vae_device is not None:
                self.vae.first_stage_model.to(vae_device)


def _decode_ltx(previewer, value, max_frames):
    import numpy
    import torch
    from PIL import Image

    if previewer is None or value.ndim != 5:
        return []
    moved = value.movedim(2, 1)
    decoded = previewer.decode(
        moved.reshape((-1,) + moved.shape[-3:]))
    if decoded is None:
        return []
    if decoded.ndim == 3:
        decoded = decoded.unsqueeze(0)
    if decoded.ndim != 4:
        return []
    if 0 < max_frames < decoded.shape[0]:
        indices = numpy.linspace(
            0, decoded.shape[0] - 1,
            max_frames).round().astype(int).tolist()
        decoded = decoded[indices]
    arrays = decoded.mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return [Image.fromarray(array) for array in arrays]


def _decode_vae(vae, value, max_frames):
    import numpy
    import torch
    from PIL import Image

    if vae is None or value.ndim != 5:
        return []
    decoded = vae.decode(value)
    if decoded.ndim == 5:
        decoded = decoded[0]
    if decoded.ndim != 4:
        return []
    if 0 < max_frames < decoded.shape[0]:
        indices = numpy.linspace(
            0, decoded.shape[0] - 1,
            max_frames).round().astype(int).tolist()
        decoded = decoded[indices]
    arrays = decoded.float().mul(255).clamp(
        0, 255).to(torch.uint8).cpu().numpy()
    return [Image.fromarray(array) for array in arrays]


def _decode_tiny(decoder, value, max_frames):
    import numpy
    import torch
    from PIL import Image

    if value.ndim == 4:
        decoded = decoder.decode(value[:1])[0].movedim(
            0, -1).unsqueeze(0).contiguous()
    elif value.ndim == 5:
        indices = list(range(value.shape[2]))
        if 0 < max_frames < len(indices):
            picks = numpy.linspace(
                0, len(indices) - 1,
                max_frames).round().astype(int).tolist()
            indices = [indices[index] for index in picks]
        decoded = decoder.decode_video(value[:1], frame_indices=indices)
    else:
        return []
    arrays = decoded.clamp(0, 1).mul(
        255).to(torch.uint8).cpu().numpy()
    return [Image.fromarray(array) for array in arrays]


def _core_previewer(device, latent_format):
    import latent_preview

    function = latent_preview.get_previewer
    seen = set()
    while hasattr(function, "__wrapped__") and id(function) not in seen:
        seen.add(id(function))
        function = function.__wrapped__
    return function(device, latent_format)


def _call_original(callback, suppress, step, x0, value, total):
    if callback is None:
        return
    if not suppress:
        callback(step, x0, value, total)
        return
    try:
        previewer = inspect.getclosurevars(callback).nonlocals.get("previewer")
    except (TypeError, ValueError):
        previewer = None
    if previewer is None or not hasattr(
            previewer, "decode_latent_to_preview_image"):
        callback(step, x0, value, total)
        return
    name = "decode_latent_to_preview_image"
    marker = object()
    previous = vars(previewer).get(name, marker)
    setattr(previewer, name, lambda *_args, **_kwargs: None)
    try:
        callback(step, x0, value, total)
    finally:
        if previous is marker:
            delattr(previewer, name)
        else:
            setattr(previewer, name, previous)


def _interpolate(value, xs, ys):
    if value <= xs[0]:
        return ys[0]
    if value >= xs[-1]:
        return ys[-1]
    for index in range(len(xs) - 1):
        if xs[index] <= value <= xs[index + 1]:
            span = xs[index + 1] - xs[index]
            return (ys[index] if span <= 0 else
                    ys[index] + (value - xs[index]) / span
                    * (ys[index + 1] - ys[index]))
    return 0.0


def _detail_boost_curve(sampler, model, sigmas):
    try:
        options = getattr(sampler, "extra_options", None) or {}
        xs = options.get("db_curve_xs")
        ys = options.get("db_curve_ys")
        if ("db_wrapped_sampler" not in options or not xs or not ys
                or len(xs) != len(ys) or len(xs) < 2):
            return None
        sampling = model.get_model_object("model_sampling")
        start = float(sampling.percent_to_sigma(
            options.get("db_start_percent", 0.0)))
        end = float(sampling.percent_to_sigma(
            options.get("db_end_percent", 1.0)))
        result = []
        for sigma in sigmas:
            sigma = float(sigma)
            if (sigma <= 0 or start <= end or sigma >= start or sigma <= end):
                result.append(None)
            else:
                result.append(_interpolate(
                    (start - sigma) / (start - end), xs, ys))
        return result
    except Exception:
        logger.exception("preview override detail-boost inspection failed")
        return None


def _send(node_id, payload):
    if node_id is None:
        return
    try:
        from server import PromptServer

        instance = getattr(PromptServer, "instance", None)
        if instance is not None:
            instance.send_sync(
                "kj_preview_override", payload, instance.client_id)
    except Exception:
        logger.exception("preview override event publication failed")


class PreviewOverrideWrapper:
    def __init__(
        self, *, node_id, max_resolution, jpeg_quality,
        suppress_default_preview, preview_frames, preview_fps,
        vae=None, tiny_vae="none",
    ):
        self.node_id = None if node_id is None else str(node_id)
        self.max_resolution = max_resolution
        self.jpeg_quality = jpeg_quality
        self.suppress_default = suppress_default_preview
        self.preview_frames = preview_frames
        self.preview_fps = preview_fps
        self.vae = vae
        self.tiny_vae = tiny_vae
        self.frames = []

    def __call__(
        self, executor, noise, latent_image, sampler, sigmas, denoise_mask,
        callback, disable_pbar, seed, latent_shapes,
    ):
        import latent_preview
        from ._preview_ltx_factors import (
            LTX2_RGB_BIAS, LTX2_RGB_FACTORS,
            LTX_RGB_BIAS, LTX_RGB_FACTORS,
        )

        guider = executor.class_obj
        model = guider.model_patcher
        latent_format = model.model.latent_format
        is_ltx = "LTX" in type(latent_format).__name__
        try:
            diffusion = model.model.diffusion_model
            is_ltx2 = is_ltx and not getattr(
                diffusion, "caption_projection_first_linear", True)
        except Exception:
            is_ltx2 = False
        keyframes = _keyframe_count(guider) if is_ltx else 0

        tiny = None
        if self.tiny_vae != "none":
            try:
                from ._preview_tiny_vae import load

                tiny = load(self.tiny_vae)
                if (latent_shapes and len(latent_shapes[0]) >= 2
                        and int(latent_shapes[0][1]) != tiny.latent_channels):
                    tiny = None
            except Exception:
                logger.exception("preview override tiny VAE setup failed")

        ltx_previewer = None
        full_vae = None
        restore_device = None
        if is_ltx:
            factors = LTX2_RGB_FACTORS if is_ltx2 else LTX_RGB_FACTORS
            bias = LTX2_RGB_BIAS if is_ltx2 else LTX_RGB_BIAS
            taehv = None
            if self.vae is not None:
                if type(self.vae.first_stage_model).__name__ == "TAEHV":
                    try:
                        import comfy.model_management

                        restore_device = next(
                            self.vae.first_stage_model.parameters()).device
                        self.vae.first_stage_model.to(
                            comfy.model_management.get_torch_device())
                        taehv = self.vae
                    except Exception:
                        logger.exception("preview override TAEHV setup failed")
                else:
                    full_vae = self.vae
            ltx_previewer = _LtxPreviewer(factors, bias, taehv)

        previewer = _core_previewer(model.load_device, latent_format)
        fallback = None
        factors = getattr(latent_format, "latent_rgb_factors", None)
        if factors is not None:
            fallback = latent_preview.Latent2RGBPreviewer(
                factors,
                getattr(latent_format, "latent_rgb_factors_bias", None),
                getattr(latent_format, "latent_rgb_factors_reshape", None))

        sigma_values = (sigmas.detach().cpu().tolist()
                        if sigmas is not None else [])
        initial = None
        try:
            if sigma_values:
                sigma = sigmas[0].to(noise.device)
                initial = _normalize_packed(
                    noise * sigma, latent_shapes, keyframes
                ).detach().float().cpu()
        except Exception:
            logger.exception("preview override initial delta failed")
        state = {"previous": initial, "time": None, "window": []}
        total_steps = max(0, len(sigma_values) - 1)
        self.frames = []

        if self.node_id is not None:
            payload = {
                "node_id": self.node_id,
                "step": 0,
                "total": total_steps,
                "sigma": sigma_values[0] if sigma_values else None,
                "sigmas": sigma_values,
            }
            curve = _detail_boost_curve(sampler, model, sigma_values)
            if curve is not None:
                payload["db_curve"] = curve
            try:
                initial_value = (
                    noise * sigmas[0].to(noise.device)
                    if sigma_values else noise)
                initial_value = _normalize_packed(
                    initial_value, latent_shapes, keyframes)
                frames = (
                    _decode_tiny(tiny, initial_value, 1) if tiny is not None
                    else _decode_ltx(ltx_previewer, initial_value, 1)
                    if ltx_previewer is not None and initial_value.ndim == 5
                    else [])
                if not frames and factors is not None:
                    decoded = fallback.decode_latent_to_preview(initial_value)
                    frames = [decoded] if hasattr(decoded, "save") else []
                if frames:
                    encoded, width, height = _encode_jpeg(
                        frames[0], self.jpeg_quality, self.max_resolution)
                    payload.update({
                        "image": encoded, "w": width, "h": height,
                    })
            except Exception:
                logger.exception("preview override initial image failed")
            _send(self.node_id, payload)

        encoder = _AsyncEncoder()

        def preview_callback(step, x0, value, callback_total):
            nonlocal tiny
            try:
                view = _normalize_packed(x0, latent_shapes, keyframes)
                count = self.preview_frames if self.preview_frames > 1 else 1
                frames = []
                if tiny is not None:
                    try:
                        frames = _decode_tiny(tiny, view, count)
                    except Exception:
                        logger.exception("preview override tiny VAE decode failed")
                        tiny = None
                if not frames and full_vae is not None and view.ndim == 5:
                    try:
                        frames = _decode_vae(full_vae, view, count)
                    except Exception:
                        logger.exception("preview override VAE decode failed")
                if not frames and ltx_previewer is not None and view.ndim == 5:
                    frames = _decode_ltx(ltx_previewer, view, count)
                if (not frames and self.preview_frames > 1
                        and view.ndim == 5):
                    frames = _decode_video_l2rgb(view, latent_format, count)
                if not frames:
                    for candidate in (previewer, fallback):
                        if candidate is None:
                            continue
                        try:
                            decoded = candidate.decode_latent_to_preview(view)
                        except Exception:
                            continue
                        if hasattr(decoded, "save"):
                            frames = [decoded]
                            break
                if frames:
                    first = (frames[0] if frames[0].mode == "RGB"
                             else frames[0].convert("RGB"))
                    frames[0] = first
                    self.frames.append(first)
                    if self.node_id is not None:
                        current = view.detach().float().cpu()
                        previous = state["previous"]
                        state["previous"] = current
                        now = time.perf_counter()
                        step_ms = (None if state["time"] is None else
                                   (now - state["time"]) * 1000)
                        state["time"] = now
                        if step_ms is not None:
                            state["window"].append(step_ms)
                            del state["window"][:-8]
                        average = (sum(state["window"]) / len(state["window"])
                                   if state["window"] else None)
                        sigma = (sigma_values[step]
                                 if 0 <= step < len(sigma_values) else None)

                        def encode_and_send(
                            frames=frames,
                            current=current,
                            previous=previous,
                            step=step,
                            callback_total=callback_total,
                            sigma=sigma,
                            step_ms=step_ms,
                            average=average,
                        ):
                            if len(frames) > 1:
                                encoded, width, height = _encode_mp4(
                                    frames, self.preview_fps,
                                    self.max_resolution)
                                mime = "video/mp4"
                                if not encoded:
                                    encoded, width, height = _encode_webp(
                                        frames, self.preview_fps,
                                        self.jpeg_quality,
                                        self.max_resolution)
                                    mime = "image/webp"
                            else:
                                encoded, width, height = _encode_jpeg(
                                    frames[0], self.jpeg_quality,
                                    self.max_resolution)
                                mime = "image/jpeg"
                            delta = None
                            if (previous is not None
                                    and previous.shape == current.shape):
                                difference = current - previous
                                delta = (difference.norm()
                                         / max(1, difference.numel()) ** 0.5
                                         ).item()
                            _send(self.node_id, {
                                "node_id": self.node_id,
                                "image": encoded,
                                "mime": mime,
                                "w": width,
                                "h": height,
                                "step": step + 1,
                                "total": callback_total,
                                "sigma": sigma,
                                "sigmas": None,
                                "delta": delta,
                                "step_ms": step_ms,
                                "avg_step_ms": average,
                                "fps": (self.preview_fps if mime in {
                                    "video/mp4", "image/webp"} else None),
                            })

                        encoder.submit(encode_and_send)
            except Exception:
                logger.exception("preview override callback failed")
            _call_original(
                callback, self.suppress_default, step, x0, value,
                callback_total)

        try:
            state["time"] = time.perf_counter()
            return executor(
                noise, latent_image, sampler, sigmas, denoise_mask,
                preview_callback, disable_pbar, seed,
                latent_shapes=latent_shapes)
        finally:
            encoder.close()
            if restore_device is not None and self.vae is not None:
                try:
                    self.vae.first_stage_model.to(restore_device)
                except Exception:
                    logger.exception("preview override VAE restore failed")


class InProcessPreviewOverride:
    def __init__(self, node_id):
        self._node_id = node_id

    async def attach(
        self, model, *, max_resolution=1024, jpeg_quality=80,
        suppress_default_preview=True, preview_frames=1, preview_fps=12,
        vae=None, tiny_vae="none",
    ):
        from comfy.patcher_extension import WrappersMP
        from ._sdk import ModelRef, Ref, current_runtime

        checked = _settings(
            max_resolution, jpeg_quality, suppress_default_preview,
            preview_frames, preview_fps, tiny_vae)
        runtime = current_runtime()
        if not isinstance(model, Ref) or model.kind != "MODEL":
            raise TypeError("preview override requires a MODEL ref")
        value = await runtime.refs.resolve(model)
        vae_value = None
        if vae is not None:
            if not isinstance(vae, Ref) or vae.kind != "VAE":
                raise TypeError("preview override vae must be a VAE ref")
            vae_value = await runtime.refs.resolve(vae)
        result = value.clone()
        result.add_wrapper_with_key(
            WrappersMP.OUTER_SAMPLE,
            "kj_preview_override",
            PreviewOverrideWrapper(
                node_id=self._node_id, vae=vae_value, **checked))
        return ModelRef._wrap(await runtime.refs.create("MODEL", result))

    async def attach_ltx2(
        self, model, *, preview_rate=8.0,
        latent_upscale_model=None, vae=None,
    ):
        from comfy.patcher_extension import WrappersMP
        from ._sdk import ModelRef, Ref, current_runtime

        rate = _preview_rate(preview_rate)
        runtime = current_runtime()
        if not isinstance(model, Ref) or model.kind != "MODEL":
            raise TypeError("LTX2 sampling preview requires a MODEL ref")
        value = await runtime.refs.resolve(model)
        vae_value = None
        taehv = False
        if vae is not None:
            if not isinstance(vae, Ref) or vae.kind != "VAE":
                raise TypeError("LTX2 sampling preview vae must be a VAE ref")
            vae_value = await runtime.refs.resolve(vae)
            taehv = (
                type(vae_value.first_stage_model).__name__ == "TAEHV")
        upscale_value = None
        if latent_upscale_model is not None and not taehv:
            if (not isinstance(latent_upscale_model, Ref)
                    or latent_upscale_model.kind not in {
                        "LATENT_UPSCALE_MODEL", "MODEL", "OPAQUE",
                    }):
                raise TypeError(
                    "latent_upscale_model must be a host-issued latent "
                    "upscale model ref")
            upscale_value = await runtime.refs.resolve(latent_upscale_model)
            unwrapped = _unwrap_latent_upscaler(upscale_value)
            if (not callable(unwrapped)
                    or not callable(getattr(unwrapped, "parameters", None))
                    or not callable(getattr(unwrapped, "to", None))):
                raise TypeError(
                    "latent_upscale_model does not provide the closed "
                    "upscaler interface")
        result = value.clone()
        result.add_wrapper_with_key(
            WrappersMP.OUTER_SAMPLE,
            "sampling_preview",
            Ltx2SamplingPreviewWrapper(
                latent_upscale_model=upscale_value,
                vae=vae_value,
                preview_rate=rate,
                taehv=taehv))
        return ModelRef._wrap(await runtime.refs.create("MODEL", result))

    async def frames(self, model, after_sample):
        import numpy
        import torch
        from comfy.patcher_extension import WrappersMP
        from ._sdk import ImageRef, Ref, current_runtime

        runtime = current_runtime()
        if not isinstance(model, Ref) or model.kind != "MODEL":
            raise TypeError("preview frames requires a MODEL ref")
        if (not isinstance(after_sample, Ref)
                or after_sample.kind not in {"LATENT", "IMAGE"}):
            raise TypeError("after_sample must be a LATENT or IMAGE ref")
        value = await runtime.refs.resolve(model)
        await runtime.refs.resolve(after_sample)
        wrappers = value.get_wrappers(
            WrappersMP.OUTER_SAMPLE, "kj_preview_override")
        if not wrappers:
            raise RuntimeError(
                "Get Preview Override Frames: no Model Preview Override "
                "wrapper found on this model.")
        frames = wrappers[-1].frames
        if not frames:
            raise RuntimeError(
                "Get Preview Override Frames: no frames captured. Ensure the "
                "sampler ran with this model.")
        tensors = [
            torch.from_numpy(numpy.asarray(
                frame, dtype=numpy.float32) / 255.0)
            for frame in frames
        ]
        return ImageRef._wrap(await runtime.refs.create(
            "IMAGE", torch.stack(tensors, dim=0)))
