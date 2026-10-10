import math

import pytest
import torch
from torch import nn

from comfy.cli_args import args

_original_cpu = args.cpu
if not torch.cuda.is_available():
    args.cpu = True
try:
    import comfy.sd
    import comfy.taesd.taehv as taehv
    from comfy_extras.nodes_lt import EmptyLTXVLatentVideo, LTXVImgToVideoInplace
    from nodes import VAEDecodeTiled, VAEEncodeTiled
finally:
    args.cpu = _original_cpu


class GeometryCodec(nn.Module):
    """Replace CNN computation, retaining geometry from the real TAEHV architecture."""

    def __init__(self, architecture):
        super().__init__()
        self.latent_channels = architecture.latent_channels
        self.temporal_encode = architecture.t_downscale
        self.temporal_decode = architecture.t_upscale
        self.frames_to_trim = architecture.frames_to_trim
        self.spatial_encode = architecture.patch_size * math.prod(
            layer.stride[0] for layer in architecture.encoder if isinstance(layer, nn.Conv2d)
        )
        self.spatial_decode = architecture.patch_size * int(math.prod(
            layer.scale_factor for layer in architecture.decoder if isinstance(layer, nn.Upsample)
        ))
        self.encode_calls = []
        self.decode_calls = []

    def decode(self, samples):
        self.decode_calls.append(tuple(samples.shape))
        pixels = samples[:, :3].repeat_interleave(self.temporal_decode, dim=2)
        pixels = pixels[:, :, self.frames_to_trim:]
        return pixels.repeat_interleave(self.spatial_decode, dim=3).repeat_interleave(self.spatial_decode, dim=4)

    def encode(self, pixels):
        self.encode_calls.append(tuple(pixels.shape))
        samples = pixels[:, :1, ::self.temporal_encode, ::self.spatial_encode, ::self.spatial_encode]
        return samples.repeat(1, self.latent_channels, 1, 1, 1)


def make_vae(monkeypatch, channels):
    monkeypatch.setattr(args, "gpu_only", False)
    monkeypatch.setattr(args, "fp16_intermediates", False)
    with torch.device("meta"):
        architecture = taehv.TAEHV(latent_channels=channels, latent_format=None)
    codec = GeometryCodec(architecture)
    signature = {
        key: torch.zeros(architecture.state_dict()[key].shape)
        for key in ("decoder.1.weight", "decoder.22.bias")
    }

    def create_codec(latent_channels, latent_format=None):
        assert latent_channels == channels
        assert latent_format is None
        return codec

    monkeypatch.setattr(taehv, "TAEHV", create_codec)
    monkeypatch.setattr(comfy.sd.model_management, "load_models_gpu", lambda *args, **kwargs: None)
    vae = comfy.sd.VAE(sd=signature, device=torch.device("cpu"), dtype=torch.float32)
    return vae, codec


def coordinates(batch, channels, frames, height, width):
    b = torch.arange(batch, dtype=torch.float32).reshape(batch, 1, 1, 1, 1) / 8
    c = torch.arange(channels, dtype=torch.float32).reshape(1, channels, 1, 1, 1) / 128
    t = torch.arange(frames, dtype=torch.float32).reshape(1, 1, frames, 1, 1) / (frames * 16)
    y = torch.arange(height, dtype=torch.float32).reshape(1, 1, 1, height, 1) / (height * 8)
    x = torch.arange(width, dtype=torch.float32).reshape(1, 1, 1, 1, width) / (width * 4)
    return 0.125 + b + c + t + y + x


@pytest.mark.parametrize(
    "channels,batch,frames,height,width,tile_size,temporal_size,multiple_tiles",
    [
        pytest.param(128, 1, 1, 2, 3, 512, 64, False, id="ltx-single-frame"),
        pytest.param(128, 2, 2, 3, 2, 512, 64, False, id="ltx-batch-single-tile"),
        pytest.param(128, 1, 10, 2, 3, 512, 64, True, id="ltx-temporal-tiles"),
        pytest.param(128, 2, 2, 5, 7, 128, 64, True, id="ltx-spatial-tiles"),
        pytest.param(48, 1, 2, 4, 6, 512, 64, False, id="wan-single-tile"),
        pytest.param(48, 2, 10, 2, 3, 512, 32, True, id="wan-temporal-tiles"),
    ],
)
def test_tiny_vae_tiled_decode_places_all_pixels(
    monkeypatch, channels, batch, frames, height, width, tile_size, temporal_size, multiple_tiles,
):
    vae, codec = make_vae(monkeypatch, channels)
    samples = coordinates(batch, channels, frames, height, width)
    if batch == 2:
        samples = samples.transpose(-1, -2).contiguous().transpose(-1, -2)
        assert not samples.is_contiguous()
    original = samples.clone()
    expected = codec.decode(samples).movedim(1, -1).flatten(0, 1)
    codec.decode_calls.clear()

    actual = VAEDecodeTiled().decode(
        vae, {"samples": samples}, tile_size=tile_size, overlap=32 if tile_size == 128 else 64,
        temporal_size=temporal_size, temporal_overlap=8,
    )[0]

    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(samples, original, rtol=0, atol=0)
    assert (len(codec.decode_calls) > batch) == multiple_tiles


@pytest.mark.parametrize(
    "channels,frames,height,width,tile_size,temporal_size,multiple_tiles",
    [
        pytest.param(128, 1, 64, 96, 512, 64, False, id="ltx-single-frame"),
        pytest.param(128, 9, 96, 64, 512, 64, False, id="ltx-single-tile"),
        pytest.param(128, 73, 64, 96, 512, 64, True, id="ltx-temporal-tiles"),
        pytest.param(128, 9, 160, 224, 128, 64, True, id="ltx-spatial-tiles"),
        pytest.param(128, 16, 80, 112, 512, 64, False, id="ltx-crop-and-existing-temporal-truncation"),
        pytest.param(48, 5, 64, 96, 512, 64, False, id="wan-single-tile"),
        pytest.param(48, 37, 32, 48, 512, 32, True, id="wan-temporal-tiles"),
    ],
)
def test_tiny_vae_tiled_encode_places_all_latents(
    monkeypatch, channels, frames, height, width, tile_size, temporal_size, multiple_tiles,
):
    vae, codec = make_vae(monkeypatch, channels)
    video = coordinates(1, 3, frames, height, width)
    pixels = video[0].movedim(0, -1)
    original = pixels.clone()
    cropped_height = height // codec.spatial_encode * codec.spatial_encode
    cropped_width = width // codec.spatial_encode * codec.spatial_encode
    y, x = (height - cropped_height) // 2, (width - cropped_width) // 2
    # Preserve the generic tiled encoder's existing restriction to k*stride+1 frames.
    kept_frames = (frames - 1) // codec.temporal_encode * codec.temporal_encode + 1
    cropped = video[:, :, :kept_frames, y:y + cropped_height, x:x + cropped_width]
    expected = codec.encode(cropped)
    codec.encode_calls.clear()

    actual = VAEEncodeTiled().encode(
        vae, pixels, tile_size=tile_size, overlap=32 if tile_size == 128 else 64,
        temporal_size=temporal_size, temporal_overlap=8,
    )[0]["samples"]

    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(pixels, original, rtol=0, atol=0)
    assert (len(codec.encode_calls) > 1) == multiple_tiles
    if not multiple_tiles:
        assert codec.encode_calls == [tuple(cropped.shape)]


def test_tiny_ltx_reference_image_matches_official_latent_grid(monkeypatch):
    vae, codec = make_vae(monkeypatch, 128)
    latent = EmptyLTXVLatentVideo.execute(width=128, height=128, length=9, batch_size=1).result[0]
    original = latent["samples"].clone()
    reference = coordinates(1, 3, 1, 128, 128)
    image = reference[0].movedim(0, -1)
    expected_reference = codec.encode(reference)
    expected = torch.cat((expected_reference, torch.zeros_like(original[:, :, 1:])), dim=2)

    actual = LTXVImgToVideoInplace.execute(vae, image, latent, strength=1.0).result[0]

    torch.testing.assert_close(actual["samples"], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual["noise_mask"], torch.tensor([0.0, 1.0]).reshape(1, 1, 2, 1, 1), rtol=0, atol=0)
    torch.testing.assert_close(latent["samples"], original, rtol=0, atol=0)
    assert latent["downscale_ratio_spacial"] == 32
