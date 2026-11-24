"""ComfyUI NOISE for silent LTX videos with a duration-independent prefix."""
import torch

from comfy.sample import prepare_noise_inner


class LTXDurationNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        samples = input_latent["samples"]
        if samples.is_nested or samples.ndim != 5 or samples.size(2) < 1:
            raise ValueError("LTX duration noise requires a video latent [B,C,T,H,W] with T >= 1")

        # Two latent positions are nine LTX frames. Keep this draw shape fixed
        # even for the last, partial block so all shared coordinates agree.
        block_size = 2
        block_shape = samples[:, :, :1].expand(-1, -1, block_size, -1, -1)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        noise = torch.empty(samples.shape, device="cpu", dtype=samples.dtype)
        for start in range(0, samples.size(2), block_size):
            block = prepare_noise_inner(block_shape, generator, input_latent.get("batch_index"))
            target = noise[:, :, start:start + block_size]
            target.copy_(block[:, :, :target.size(2)])
        return noise


class LTXDurationNoiseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "noise_seed": ("INT", {"default": 10000001, "min": 0,
                                    "max": 0xffffffffffffffff, "control_after_generate": True}),
        }}

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = "sampling/custom_sampling/noise"

    def get_noise(self, noise_seed):
        return (LTXDurationNoise(noise_seed),)


NODE_CLASS_MAPPINGS = {"LTXDurationNoise": LTXDurationNoiseNode}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXDurationNoise": "LTX Duration-Stable Noise"}
