"""Grow a silent LTX text-to-video shot from saved, fixed-size latent windows."""
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

import av
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import folder_paths
from comfy.model_management import throw_exception_if_processing_interrupted
from comfy_api.latest import InputImpl
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, SamplerCustomAdvanced
from nodes import VAEDecodeTiled

# LTX decodes T latent positions into 1 + 8*(T-1) frames.
BASE_FRAMES = 9
NEW_FRAMES = 16
CONTEXT_POSITIONS = 2


def shot_key(prompt, unique_id, width, height, fps, seed):
    """Only upstream generation inputs define the shot. Ignore export length and UI layout."""
    def upstream(value):
        if not (isinstance(value, list) and len(value) == 2 and
                isinstance(value[0], str) and value[0] in prompt and isinstance(value[1], int)):
            return value
        node = prompt[value[0]]
        inputs = {k: upstream(v) for k, v in node["inputs"].items()}
        for field, category in (("unet_name", "diffusion_models"), ("clip_name", "text_encoders"), ("vae_name", "vae")):
            if isinstance(inputs.get(field), str):
                path = folder_paths.get_full_path(category, inputs[field])
                if path:
                    stat = Path(path).stat()
                    inputs[field] = [inputs[field], stat.st_size, stat.st_mtime_ns]
        return [node["class_type"], value[1], inputs]

    linked = prompt[str(unique_id)]["inputs"]
    key = ["ltx-text-windows-v1", width, height, fps, seed,
           {k: upstream(linked[k]) for k in ("noise", "guider", "sampler", "sigmas", "vae")}]
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def sample_window(context, width, height, noise, guider, sampler, sigmas, vae):
    if context is None:
        latent = {"samples": torch.zeros(1, 128, 2, height // 32, width // 32)}
        trim = 0
    else:
        new_positions = NEW_FRAMES // 8
        extension = context.new_zeros(1, 128, new_positions, height // 32, width // 32)
        samples = torch.cat((context, extension), dim=2)
        mask = samples.new_ones(1, 1, samples.size(2), 1, 1)
        mask[:, :, :CONTEXT_POSITIONS] = 0
        latent = {"samples": samples, "noise_mask": mask}
        trim = 1 + 8 * (CONTEXT_POSITIONS - 1)

    sampled = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)[0]
    samples = sampled["samples"].detach().cpu()
    if context is not None:
        samples = samples.clone()
        samples[:, :, :CONTEXT_POSITIONS] = context
    images, = VAEDecodeTiled().decode(vae, {"samples": samples}, 512, 64, 64, 8)
    images = images[trim:].detach().cpu()
    expected = BASE_FRAMES if context is None else NEW_FRAMES
    if tuple(images.shape) != (expected, height, width, 3) or not torch.isfinite(images).all():
        raise ValueError("VideoVAE returned invalid extension frames")
    return {"rgb": images.mul(255).round().clamp_(0, 255).to(torch.uint8).contiguous(),
            "context": samples[:, :, -CONTEXT_POSITIONS:].clone().contiguous()}


def publish_window(path, tensors):
    # Write once. Concurrent requests reuse the winning window, including its context.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial") as tmp:
        save_file(tensors, tmp.name)
        try:
            os.link(tmp.name, path)
        except FileExistsError:
            pass


def export_prefix(paths, frame_count, fps, width, height, target):
    # Fixed-QP intra frames keep the decoded prefix independent of export length.
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".mp4", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        with av.open(str(temp_path), "w") as writer:
            stream = writer.add_stream("libx264", rate=fps, options={"qp": "18", "g": "1", "bf": "0", "threads": "1"})
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            written = 0
            for path in paths:
                throw_exception_if_processing_interrupted()
                with safe_open(path, framework="pt", device="cpu") as f:
                    frames = f.get_tensor("rgb")
                    for frame in frames[:frame_count - written]:
                        for packet in stream.encode(av.VideoFrame.from_ndarray(frame.numpy(), format="rgb24")):
                            writer.mux(packet)
                        written += 1
            if written != frame_count:
                raise ValueError(f"Missing output frames: got {written}, expected {frame_count}")
            for packet in stream.encode():
                writer.mux(packet)
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)


class LTXTextContinuation:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "noise": ("NOISE",), "guider": ("GUIDER",), "sampler": ("SAMPLER",),
            "sigmas": ("SIGMAS",), "vae": ("VAE",),
            "width": ("INT", {"default": 480, "min": 64, "max": 4096, "step": 32}),
            "height": ("INT", {"default": 288, "min": 64, "max": 4096, "step": 32}),
            "total_length": ("INT", {"default": 9, "min": 1, "max": 16384}),
            "unit": (["frames", "seconds"],),
            "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
        }, "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}}

    CATEGORY = "sampling/video/LTX"
    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "render"

    def render(self, noise, guider, sampler, sigmas, vae, width, height, total_length, unit, fps, prompt, unique_id):
        if width % 32 or height % 32:
            raise ValueError("LTX dimensions must be multiples of 32")
        count = total_length * fps if unit == "seconds" else total_length
        key = shot_key(prompt, unique_id, width, height, fps, noise.seed)
        directory = Path(folder_paths.get_output_directory()) / "ltx-text-shots" / key
        directory.mkdir(parents=True, exist_ok=True)
        n = 1 + (max(0, count - BASE_FRAMES) + NEW_FRAMES - 1) // NEW_FRAMES
        paths = [directory / f"window-{i:05}.safetensors" for i in range(n)]
        for i, path in enumerate(paths):
            throw_exception_if_processing_interrupted()
            if path.exists():
                logging.info("LTX T2V %s: reusing window %d/%d", key[:10], i+1, n)
                continue
            logging.info("LTX T2V %s: sampling window %d/%d", key[:10], i+1, n)
            context = None
            if i:
                with safe_open(paths[i-1], framework="pt", device="cpu") as f:
                    context = f.get_tensor("context")
            # The workflow supplies LTXDurationNoise, so the first window reproduces
            # the old nine-frame run. Give each continuation a stable distinct seed.
            window_noise = Noise_RandomNoise((noise.seed + i) % (1 << 64)) if i else noise
            publish_window(path, sample_window(context, width, height, window_noise, guider, sampler, sigmas, vae))
        temp = Path(folder_paths.get_temp_directory())
        temp.mkdir(parents=True, exist_ok=True)
        movie = temp / f"ltx-text-{key}-{count}.mp4"
        export_prefix(paths, count, fps, width, height, movie)
        return (InputImpl.VideoFromFile(str(movie)),)


NODE_CLASS_MAPPINGS = {"LTXTextContinuation": LTXTextContinuation}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXTextContinuation": "LTX Text to Video · Extend Same Shot"}
