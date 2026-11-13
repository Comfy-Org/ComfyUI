import numpy as np
import pytest
import torch
from PIL import Image

import comfy.utils


def reference_lanczos(samples, width, height):
    # comfy.utils.lanczos before frames were resized in parallel
    if samples.ndim == 4:
        samples = samples.squeeze(1) if samples.shape[1] == 1 else samples.movedim(1, -1)
    images = [Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(t).movedim(-1, 0) if (t := np.array(image).astype(np.float32) / 255.0).ndim == 3 else torch.from_numpy(t) for image in images]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)


@pytest.mark.parametrize("shape,size", [
    ((9, 3, 37, 53), (101, 80)),    # a batch of frames, upscaled, odd sizes
    ((6, 3, 120, 90), (41, 59)),    # downscaled
    ((3, 4, 30, 20), (60, 45)),     # RGBA
    ((4, 1, 30, 20), (31, 17)),     # grayscale (squeezed)
    ((1, 3, 16, 16), (32, 32)),     # a single image
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_lanczos_matches_the_sequential_version(shape, size, dtype):
    torch.manual_seed(0)
    samples = (torch.rand(shape) * 1.2 - 0.1).to(dtype)    # values outside 0..1 are clipped the same way
    expected = reference_lanczos(samples, *size)
    result = comfy.utils.lanczos(samples, *size)
    assert result.shape == expected.shape and result.dtype == expected.dtype
    assert torch.equal(result, expected)


def test_lanczos_does_not_depend_on_the_thread_count():
    torch.manual_seed(1)
    samples = torch.rand(12, 3, 48, 64)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        single = comfy.utils.lanczos(samples, 96, 70)
        torch.set_num_threads(4)
        parallel = comfy.utils.lanczos(samples, 96, 70)
    finally:
        torch.set_num_threads(threads)
    assert torch.equal(single, parallel)


@pytest.mark.parametrize("threads", [1, 4])
def test_lanczos_in_inference_mode(threads):
    # nodes run inside torch.inference_mode(), which is thread local: the parallel path must work there too
    torch.manual_seed(3)
    samples = torch.rand(8, 3, 40, 30)
    expected = reference_lanczos(samples, 64, 80)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        with torch.inference_mode():
            result = comfy.utils.lanczos(samples, 64, 80)
    finally:
        torch.set_num_threads(previous)
    assert torch.equal(result, expected)


def test_lanczos_does_not_keep_a_float_copy_of_the_batch(monkeypatch):
    # frames are moved to the CPU one at a time, as before, not the whole batch at once
    torch.manual_seed(4)
    samples = torch.rand(5, 3, 20, 20)
    calls = []
    original = torch.Tensor.cpu

    def cpu(self, *args, **kwargs):
        calls.append(tuple(self.shape))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    comfy.utils.lanczos(samples, 10, 10)
    assert calls and all(len(shape) == 3 for shape in calls)


def test_common_upscale_lanczos_video_batch():
    # ImageScale on a video: (frames, channels, h, w) through common_upscale
    torch.manual_seed(2)
    samples = torch.rand(10, 3, 40, 30)
    result = comfy.utils.common_upscale(samples, 64, 80, "lanczos", "disabled")
    assert torch.equal(result, reference_lanczos(samples, 64, 80))
