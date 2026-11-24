"""Run from the fork with its ComfyUI Python: python extra/iris-xe/duration-noise/check_noise.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import comfy.options
comfy.options.enable_args_parsing()

import torch
from comfy.sample import prepare_noise
from ltx_duration_noise import LTXDurationNoise


def check(height, width, batch, indices):
    seed = 10000001
    previous = None
    for t in (1, 2, 3, 4, 13):
        samples = torch.empty((batch, 128, t, height, width))
        latent = {"samples": samples, "batch_index": indices}
        noise = LTXDurationNoise(seed).generate_noise(latent)
        assert noise.shape == samples.shape
        if previous is not None:
            assert torch.equal(noise[:, :, :previous.size(2)], previous)
        if t == 2:
            assert torch.equal(noise, prepare_noise(samples, seed, indices))
        assert not torch.equal(noise, LTXDurationNoise(seed + 1).generate_noise(latent))
        previous = noise


if __name__ == "__main__":
    for height, width in ((9, 15), (17, 30)):
        check(height, width, 1, None)
        check(height, width, 3, [1, 0, 1])
    print("PASS: stock nine-frame noise and exact prefixes at both resolutions")
