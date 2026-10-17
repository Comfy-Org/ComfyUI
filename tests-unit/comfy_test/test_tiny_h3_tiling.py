from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args

_original_cpu = args.cpu
if not torch.cuda.is_available():
    args.cpu = True
try:
    import comfy.sd as sd
    import comfy.taesd.taehv as taehv
    from nodes import VAEDecodeTiled, VAEEncodeTiled
finally:
    args.cpu = _original_cpu


def coordinates(batch, channels, frames, height, width):
    b = torch.arange(batch).reshape(batch, 1, 1, 1, 1) / 16
    c = torch.arange(channels).reshape(1, channels, 1, 1, 1) / 256
    t = torch.arange(frames).reshape(1, 1, frames, 1, 1) / (frames * 16)
    y = torch.arange(height).reshape(1, 1, 1, height, 1) / (height * 16)
    x = torch.arange(width).reshape(1, 1, 1, 1, width) / (width * 16)
    return 0.125 + b + c + t + y + x


def encoded_grid(pixels, channels=24, spatial=16, h3=True):
    frames = pixels.shape[2]
    length = (1 if frames == 1 else 5 * ((frames + 16) // 17) - 3) if h3 else (frames + 3) // 4
    t = torch.arange(length)
    positions = t // 5 * 17 + t % 5 * 4 if h3 else t * 4
    return pixels[:, torch.arange(channels) % 3][:, :, positions, ::spatial, ::spatial]


def decoded_grid(samples, spatial=16, h3=True):
    length = samples.shape[2]
    frames = (1 if length == 1 else (length - 2) // 5 * 17 + 5) if h3 else length * 4 - 3
    t = torch.arange(frames)
    positions = t // 17 * 5 + (t % 17 + 3) // 4 if h3 else (t + 3) // 4
    return samples[:, :3, positions].repeat_interleave(spatial, dim=3).repeat_interleave(spatial, dim=4)


def assert_grid(actual, expected):
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def make_vae(monkeypatch, channels=24):
    """Keep VAE detection and TAEHV's real padding/grouping/trimming; replace only CNN work."""
    monkeypatch.setattr(args, "gpu_only", False)
    monkeypatch.setattr(args, "fp16_intermediates", False)
    monkeypatch.setattr(sd.model_management, "intermediate_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(sd.model_management, "load_models_gpu", lambda *a, **k: None)
    with torch.device("meta"):
        architecture = taehv.TAEHV(latent_channels=channels, latent_format=None)
    assert architecture.t_downscale == architecture.t_upscale == 4
    assert architecture.patch_size == 2
    signature = {
        key: torch.zeros(architecture.state_dict()[key].shape)
        for key in ("decoder.1.weight", "decoder.22.bias")
    }

    def neural_geometry(model, x, parallel, show_progress_bar, output_device=None, patch_size=1, decode=False):
        if decode:
            return x[:, :, :3].repeat_interleave(4, dim=1).repeat_interleave(16, dim=3).repeat_interleave(16, dim=4)
        # H3's real caller prepends three frames to each 17-frame group.
        start = 3 if channels == 24 else 0
        x = x[:, start::4, :, ::16, ::16]
        return x[:, :, torch.arange(channels) % 3]

    monkeypatch.setattr(taehv, "apply_model_with_memblocks", neural_geometry)
    with torch.random.fork_rng(devices=[]):
        vae = sd.VAE(sd=signature, device=torch.device("cpu"), dtype=torch.float32)
    calls = SimpleNamespace(encode=[], decode=[], estimates=[])
    for name in ("encode", "decode"):
        original = getattr(vae.first_stage_model, name)
        recorded = getattr(calls, name)

        def track(samples, *a, _original=original, _recorded=recorded, **k):
            _recorded.append(tuple(samples.shape))
            return _original(samples, *a, **k)

        monkeypatch.setattr(vae.first_stage_model, name, track)
    memory = vae.memory_used_decode

    def estimate(shape, dtype):
        calls.estimates.append(tuple(shape))
        return memory(shape, dtype)

    monkeypatch.setattr(vae, "memory_used_decode", estimate)
    return vae, calls


@pytest.mark.parametrize("direction", ["encode", "decode"])
@pytest.mark.parametrize(
    "frames,temporal_size,temporal_overlap",
    [
        pytest.param(1, 8, 4, id="single-image"),
        pytest.param(5, 8, 4, id="short-video"),
        pytest.param(22, 64, 8, id="short-22-frames"),
        pytest.param(39, 64, 8, id="short-default"),
        pytest.param(73, 64, 8, id="default-multiple-tiles"),
        pytest.param(73, 8, 4, id="minimum-ui-budget"),
        pytest.param(73, 32, 12, id="intermediate-budget"),
        pytest.param(175, 128, 32, id="larger-budget"),
    ],
)
def test_public_h3_tiling_preserves_coordinates(monkeypatch, direction, frames, temporal_size, temporal_overlap):
    vae, calls = make_vae(monkeypatch)
    if direction == "encode":
        source = coordinates(1, 3, frames, 32, 48)
        expected = encoded_grid(source)
        pixels = source[0].movedim(0, -1)
        actual = VAEEncodeTiled().encode(
            vae, pixels, tile_size=512, overlap=64,
            temporal_size=temporal_size, temporal_overlap=temporal_overlap,
        )[0]["samples"]
    else:
        length = 1 if frames == 1 else (frames - 5) // 17 * 5 + 2
        source = coordinates(1, 24, length, 2, 3)
        expected = decoded_grid(source).movedim(1, -1).flatten(0, 1)
        actual = VAEDecodeTiled().decode(
            vae, {"samples": source}, tile_size=512, overlap=64,
            temporal_size=temporal_size, temporal_overlap=temporal_overlap,
        )[0]
        assert calls.estimates[0][2] == max(shape[2] for shape in calls.decode)
    assert_grid(actual, expected)
    lengths = [shape[2] for shape in getattr(calls, direction)]
    assert len(lengths) > 1 if frames >= 73 else len(lengths) == 1
    if frames == 73 and temporal_size == 64:
        assert lengths == ([56, 22] if direction == "encode" else [17, 7])
    if frames == 73 and temporal_size == 8:
        assert max(lengths) == (22 if direction == "encode" else 7)


@pytest.mark.parametrize("direction", ["encode", "decode"])
@pytest.mark.parametrize("time_parameters", ["small-direct", "none-public"])
def test_h3_time_and_space_tiles_preserve_noncontiguous_batches(monkeypatch, direction, time_parameters):
    vae, calls = make_vae(monkeypatch)
    if direction == "encode":
        source = coordinates(2, 3, 73, 80, 112)
        expected = encoded_grid(source)
        options = dict(tile_x=64, tile_y=64, overlap=(19, 16, 16), tile_t=20)
    else:
        source = coordinates(2, 24, 22, 5, 7)
        expected = decoded_grid(source)
        options = dict(tile_x=4, tile_y=4, overlap=(4, 1, 1), tile_t=6)
    source = source.transpose(-1, -2).contiguous().transpose(-1, -2)
    original = source.clone()
    assert not source.is_contiguous()
    if time_parameters == "small-direct":
        actual = getattr(vae, direction + "_tiled_3d")(source, **options)
    elif direction == "encode":
        # Public IMAGE inputs are T,H,W,C; direct 3-D calls above cover joint B2 input.
        actual = torch.cat([
            vae.encode_tiled(video.movedim(0, -1), tile_x=64, tile_y=64, overlap=16)
            for video in source
        ])
    else:
        actual = vae.decode_tiled(source, tile_x=4, tile_y=4, overlap=1).movedim(-1, 1)
        assert calls.estimates[0][2] == source.shape[2]
    assert_grid(actual, expected)
    torch.testing.assert_close(source, original, rtol=0, atol=0)
    assert len(getattr(calls, direction)) > source.shape[0]
    assert all(shape[0] == 1 for shape in getattr(calls, direction))


@pytest.mark.parametrize("direction", ["encode", "decode"])
def test_h3_regular_oom_reaches_real_tiler(monkeypatch, direction):
    vae, calls = make_vae(monkeypatch)
    original = getattr(vae.first_stage_model, direction)
    attempts = []

    def first_call_oom(samples, **kwargs):
        attempts.append(tuple(samples.shape))
        if len(attempts) == 1:
            raise sd.model_management.OOM_EXCEPTION("forced regular codec allocation failure")
        return original(samples, **kwargs)

    monkeypatch.setattr(vae.first_stage_model, direction, first_call_oom)
    monkeypatch.setattr(vae.patcher, "get_free_memory", lambda *a: 1_000_000)
    monkeypatch.setattr(sd.model_management, "soft_empty_cache", lambda: None)
    if direction == "decode":
        source = coordinates(1, 24, 22, 2, 3)
        expected = decoded_grid(source).movedim(1, -1)
        monkeypatch.setattr(sd.model_management, "get_total_memory", lambda *a: 100)

        def estimate(shape, dtype):
            calls.estimates.append(tuple(shape))
            # Even the minimum legal block exceeds this artificial budget.
            # Bound observations so an accidental 7 -> 4 -> 7 loop fails promptly.
            assert len(calls.estimates) <= 5
            return shape[2] * 100

        monkeypatch.setattr(vae, "memory_used_decode", estimate)
        actual = vae.decode(source)
        assert [shape[2] for shape in calls.estimates] == [22, 22, 7, 7, 7]
        assert max(shape[2] for shape in calls.decode) == 7
        assert len(calls.decode) > 1
    else:
        source = coordinates(1, 3, 73, 32, 48)
        expected = encoded_grid(source)
        actual = vae.encode(source[0].movedim(0, -1))
    assert len(attempts) > 1
    assert_grid(actual, expected)


def causal_vae():
    # Existing test fixtures construct VAE without __init__.
    vae = sd.VAE.__new__(sd.VAE)
    vae.device = vae.output_device = torch.device("cpu")
    vae.vae_dtype = torch.float32
    vae.output_channels = 3
    vae.latent_channels = 4
    vae.extra_1d_channel = None
    vae.process_input = vae.process_output = lambda x: x
    vae.vae_output_dtype = lambda: torch.float32
    vae.upscale_ratio = (lambda length: max(0, length * 4 - 3), 8, 8)
    vae.downscale_ratio = (lambda frames: (frames + 3) // 4, 8, 8)
    vae.upscale_index_formula = vae.downscale_index_formula = (4, 8, 8)
    vae.first_stage_model = SimpleNamespace(
        encode=lambda x: encoded_grid(x, channels=4, spatial=8, h3=False),
        decode=lambda x: decoded_grid(x, spatial=8, h3=False),
    )
    return vae


@pytest.mark.parametrize("kind", ["tiny-wan", "causal-without-init"])
@pytest.mark.parametrize("direction", ["encode", "decode"])
def test_other_vaes_keep_scalar_overlap_and_unaligned_time_budget(monkeypatch, kind, direction):
    if kind == "tiny-wan":
        vae, _ = make_vae(monkeypatch, channels=48)
        channels, spatial = 48, 16
    else:
        vae = causal_vae()
        channels, spatial = 4, 8
    observed = []
    tiler = sd.comfy.utils.tiled_scale_multidim

    def record(*a, **kwargs):
        observed.append((kwargs["tile"], kwargs["overlap"]))
        return tiler(*a, **kwargs)

    monkeypatch.setattr(sd.comfy.utils, "tiled_scale_multidim", record)
    if direction == "decode":
        source = coordinates(2, channels, 10, 2, 3)
        expected = decoded_grid(source, spatial=spatial, h3=False)
        options = dict(tile_t=8, tile_x=3, tile_y=3, overlap=1)
    else:
        source = coordinates(2, 3, 37, 2 * spatial, 3 * spatial)
        expected = encoded_grid(source, channels=channels, spatial=spatial, h3=False)
        options = dict(tile_t=29, tile_x=3 * spatial, tile_y=3 * spatial, overlap=1)
    actual = getattr(vae, direction + "_tiled_3d")(source, **options)
    assert_grid(actual, expected)
    assert observed == [((options["tile_t"], options["tile_x"], options["tile_y"]), 1)]
    assert vae._tile_bounded_shape((2, channels, 10, 2, 3), 3, 2, 3) == (2, channels, 3, 2, 3)
