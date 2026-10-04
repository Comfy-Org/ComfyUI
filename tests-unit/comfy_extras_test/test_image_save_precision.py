import io
from pathlib import Path

import av
import numpy as np
import pytest
import torch

from comfy.cli_args import args

_original_cpu = args.cpu
if not torch.cuda.is_available():
    args.cpu = True
try:
    import folder_paths
    from nodes import EmptyImage
    from comfy_extras.nodes_images import SaveImageAdvanced, _avif_frame, _encode_image
finally:
    args.cpu = _original_cpu


def decode_image(encoded):
    with av.open(io.BytesIO(encoded)) as container:
        return next(container.decode(video=0))


def plane_values(frame):
    dtype = np.dtype(np.uint16 if max(component.bits for component in frame.format.components) > 8 else np.uint8)
    return [
        np.frombuffer(plane, dtype=dtype).reshape(plane.height, plane.line_size // dtype.itemsize)[:, :plane.width]
        for plane in frame.planes
    ]


@pytest.mark.parametrize(
    "dtype,channels",
    [(torch.float16, 3), (torch.bfloat16, 3), (torch.float32, 3), (torch.float64, 3),
     (torch.float16, 4), (torch.bfloat16, 4),
     (torch.float16, 1), (torch.bfloat16, 1), (torch.float32, 1)],
)
def test_png16_preserves_integer_pixels(dtype, channels):
    values = [-0.25, 0, 1 / 256, 0.25, 0.5, 0.75, 1, 1.25]
    codes = [0, 0, 255, 16383, 32767, 49151, 65535, 65535]
    bands = np.stack([np.roll(values, channel) for channel in range(channels)], axis=-1)
    image = torch.tensor(bands, dtype=dtype).unsqueeze(0).repeat(16, 1, 1).transpose(0, 1)
    if channels == 1 and dtype != torch.float32:
        image = image[..., 0]
    original = image.clone()
    assert not image.is_contiguous()

    frame = decode_image(_encode_image(image, "png", "16-bit", "sRGB"))
    assert max(component.bits for component in frame.format.components) == 16
    pixels = frame.to_ndarray(format={1: "gray16le", 3: "rgb48le", 4: "rgba64le"}[channels])
    expected_bands = np.stack([np.roll(codes, channel) for channel in range(channels)], axis=-1)
    expected = np.repeat(expected_bands[:, None, :], 16, axis=1).astype(np.uint16)
    if channels == 1:
        expected = expected[..., 0]
    np.testing.assert_array_equal(pixels, expected)
    torch.testing.assert_close(image, original, rtol=0, atol=0)


def test_png16_keeps_float64_quantization_boundary():
    image = torch.tensor([(12345 - 0.0001) / 65535, (32768 - 0.0001) / 65535, 1], dtype=torch.float64)
    image = image.reshape(1, 3, 1).repeat(8, 1, 3)
    frame = decode_image(_encode_image(image, "png", "16-bit", "sRGB"))
    expected = np.broadcast_to(np.array([12344, 32767, 65535], dtype=np.uint16)[None, :, None], (8, 3, 3))
    np.testing.assert_array_equal(frame.to_ndarray(format="rgb48le"), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_png8_keeps_existing_quantization(dtype):
    image = torch.tensor([0, 1 / 255, 0.25, 0.5, 0.75, 1], dtype=dtype).reshape(1, 6, 1).repeat(8, 1, 3)
    frame = decode_image(_encode_image(image, "png", "8-bit", "sRGB"))
    expected = np.broadcast_to(np.array([0, 1, 63, 127, 191, 255], dtype=np.uint8)[None, :, None], (8, 6, 3))
    np.testing.assert_array_equal(frame.to_ndarray(format="rgb24"), expected)


@pytest.mark.parametrize(
    "dtype,channels",
    [(torch.float16, 1), (torch.bfloat16, 1), (torch.float16, 3), (torch.bfloat16, 3),
     (torch.float32, 3), (torch.float64, 3)],
)
def test_avif10_preserves_quantized_frame_planes(dtype, channels):
    values = [-0.25, 0, 1 / 256, 0.25, 0.5, 0.75, 1, 1.25]
    codes = [0, 0, 255, 16383, 32767, 49151, 65535, 65535]
    bands = np.stack([np.roll(values, channel) for channel in range(channels)], axis=-1)
    image = torch.tensor(bands, dtype=dtype).unsqueeze(0).repeat(16, 1, 1).transpose(0, 1)
    encoded_values = np.stack([np.roll(codes, channel) for channel in range(channels)], axis=-1)
    encoded_values = np.repeat(encoded_values[:, None, :], 16, axis=1).astype(np.uint16)
    if channels == 1:
        image = image[..., 0]
        encoded_values = encoded_values[..., 0]
    original = image.clone()
    expected = av.VideoFrame.from_ndarray(encoded_values, format="gray16le" if channels == 1 else "rgb48le")
    expected = expected.reformat(format="yuv420p10le", dst_colorspace=1)

    actual = _avif_frame(image, "10-bit YUV420", "sRGB", "yuv420p10le")
    assert actual.format.name == "yuv420p10le"
    for actual_plane, expected_plane in zip(plane_values(actual), plane_values(expected)):
        np.testing.assert_array_equal(actual_plane, expected_plane)
    torch.testing.assert_close(image, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_avif8_keeps_existing_quantization(dtype):
    image = torch.tensor([0, 1 / 255, 0.25, 0.5, 0.75, 1], dtype=dtype).reshape(1, 6, 1).repeat(8, 1, 3)
    pixels = np.broadcast_to(np.array([0, 1, 63, 127, 191, 255], dtype=np.uint8)[None, :, None], (8, 6, 3)).copy()
    expected = av.VideoFrame.from_ndarray(pixels, format="rgb24").reformat(format="yuv420p", dst_colorspace=1)
    actual = _avif_frame(image, "8-bit YUV420", "sRGB", "yuv420p")
    for actual_plane, expected_plane in zip(plane_values(actual), plane_values(expected)):
        np.testing.assert_array_equal(actual_plane, expected_plane)


@pytest.mark.parametrize("dtype,depth", [(torch.float16, "16-bit float"), (torch.float32, "32-bit float")])
def test_exr_keeps_floating_point_range(dtype, depth):
    image = torch.tensor([-0.25, 0, 0.25, 0.5, 1, 2, 4], dtype=dtype).reshape(1, 7, 1).repeat(8, 1, 3)
    frame = decode_image(_encode_image(image, "exr", depth, "linear"))
    if depth == "16-bit float":
        assert frame.format.name == "gbrpf16le"
        for plane, channel in zip(frame.planes, (1, 2, 0)):
            pixels = np.frombuffer(plane, dtype=np.float16).reshape(plane.height, plane.line_size // 2)[:, :plane.width]
            np.testing.assert_array_equal(pixels, image[..., channel].numpy())
    else:
        np.testing.assert_array_equal(frame.to_ndarray(format="gbrpf32le"), image.float().numpy())


@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize(
    "file_format,depth,pixel_format,bits",
    [("png", "8-bit", "rgb24", 8), ("png", "16-bit", "rgb48le", 16),
     ("avif", "8-bit YUV420", "rgb24", 8), ("avif", "10-bit YUV420", "rgb48le", 10)],
)
def test_empty_image_saves_with_intermediate_dtype(tmp_path, monkeypatch, fp16, file_format, depth, pixel_format, bits):
    monkeypatch.setattr(args, "fp16_intermediates", fp16)
    monkeypatch.setattr(args, "gpu_only", False)
    monkeypatch.setattr(args, "disable_metadata", True)
    monkeypatch.setattr(folder_paths, "output_directory", str(tmp_path))
    images = EmptyImage().generate(64, 64, color=0xFF8040)[0]
    assert images.dtype == (torch.float16 if fp16 else torch.float32)
    original = images.clone()
    settings = {"format": file_format, "bit_depth": depth, "input_color_space": "sRGB"}
    if file_format == "avif":
        settings.update({"crf": 18, "save_mode": {"save_mode": "still images"}})
    saver = SaveImageAdvanced.PREPARE_CLASS_CLONE(None)
    result = saver.execute(images, "precision", settings)
    entry = result.ui["images"][0]
    output = Path(tmp_path, entry["subfolder"], entry["filename"])
    frame = decode_image(output.read_bytes())
    assert max(component.bits for component in frame.format.components) == bits
    pixels = frame.to_ndarray(format=pixel_format)
    if bits == 8:
        expected = [255, 128, 64]
    else:
        expected = [65535, 32895, 16447] if fp16 else [65535, 32896, 16448]
    expected = np.broadcast_to(np.array(expected, dtype=pixels.dtype), (64, 64, 3))
    if file_format == "png":
        np.testing.assert_array_equal(pixels, expected)
    else:
        np.testing.assert_allclose(pixels, expected, rtol=0, atol=4 if bits == 8 else 512)
    assert result.result[0] is images
    torch.testing.assert_close(images, original, rtol=0, atol=0)
