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
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
@pytest.mark.parametrize("threads", [1, 4])
def test_lanczos_matches_the_sequential_version(shape, size, dtype, threads):
    torch.manual_seed(0)
    samples = (torch.rand(shape) * 1.2 - 0.1).to(dtype)    # values outside 0..1 are clipped the same way
    expected = reference_lanczos(samples, *size)
    with num_threads(threads):
        result = comfy.utils.lanczos(samples, *size)
    assert result.shape == expected.shape and result.dtype == expected.dtype
    assert result.stride() == expected.stride()
    assert torch.equal(result, expected)


@pytest.mark.parametrize("samples", [
    torch.rand(5, 30, 20),                                # 3-D: a batch of grayscale images
    torch.rand(5, 20, 30, 3).movedim(-1, 1),              # not contiguous
    torch.rand(20, 3, 16, 12),                            # more frames than LANCZOS_MAX_THREADS
], ids=["3d", "non-contiguous", "many-frames"])
def test_lanczos_matches_for_other_layouts(samples):
    expected = reference_lanczos(samples, 25, 19)
    with num_threads(16):
        result = comfy.utils.lanczos(samples, 25, 19)
    assert result.stride() == expected.stride() and torch.equal(result, expected)


class num_threads:
    def __init__(self, n):
        self.n = n

    def __enter__(self):
        self.previous = torch.get_num_threads()
        torch.set_num_threads(self.n)

    def __exit__(self, *exc):
        torch.set_num_threads(self.previous)


def test_lanczos_does_not_depend_on_the_thread_count():
    torch.manual_seed(1)
    samples = torch.rand(12, 3, 48, 64)
    expected = reference_lanczos(samples, 96, 70)
    for threads in (1, 2, 4, 16):
        with num_threads(threads):
            assert torch.equal(comfy.utils.lanczos(samples, 96, 70), expected)


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("inference_input", [False, True])
def test_lanczos_in_inference_mode(threads, inference_input):
    # nodes run inside torch.inference_mode(), which is thread local: the parallel path must work there too
    torch.manual_seed(3)
    samples = torch.rand(8, 3, 40, 30)
    expected = reference_lanczos(samples, 64, 80)
    with num_threads(threads), torch.inference_mode():
        if inference_input:
            samples = samples.clone()                     # an inference tensor
        result = comfy.utils.lanczos(samples, 64, 80)
        assert result.is_inference()
    assert torch.equal(result, expected)


@pytest.mark.parametrize("threads", [1, 4])
def test_lanczos_under_no_grad_with_a_tensor_that_requires_grad(threads):
    # grad mode is thread local too: under no_grad the frames of such a tensor are read as before
    samples = torch.rand(6, 3, 10, 10, requires_grad=True)
    with torch.no_grad():
        expected = reference_lanczos(samples, 20, 20)
        with num_threads(threads):
            result = comfy.utils.lanczos(samples, 20, 20)
    assert torch.equal(result, expected) and not result.requires_grad
    with pytest.raises(RuntimeError), num_threads(threads):  # outside no_grad it fails, as before
        comfy.utils.lanczos(samples, 20, 20)


def test_lanczos_empty_batch_raises_as_before():
    with pytest.raises(RuntimeError):
        reference_lanczos(torch.rand(0, 3, 10, 10), 20, 20)
    with pytest.raises(RuntimeError):
        comfy.utils.lanczos(torch.rand(0, 3, 10, 10), 20, 20)


def test_lanczos_error_in_a_frame_stops_the_rest(monkeypatch):
    done = []
    real = comfy.utils._lanczos_image

    def failing(image, width, height):
        done.append(1)
        if len(done) == 3:
            raise ValueError("bad frame")
        return real(image, width, height)

    monkeypatch.setattr(comfy.utils, "_lanczos_image", failing)
    with pytest.raises(ValueError, match="bad frame"), num_threads(2):
        comfy.utils.lanczos(torch.rand(200, 3, 64, 64), 128, 128)
    assert len(done) < 200


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
