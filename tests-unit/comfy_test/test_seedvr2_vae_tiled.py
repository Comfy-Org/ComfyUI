from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.seedvr.vae as vae_mod  # noqa: E402
import comfy.ldm.seedvr.vae as seedvr_vae_mod  # noqa: E402
import comfy.sd as sd_mod  # noqa: E402
from comfy.ldm.seedvr.vae import MemoryState, tiled_vae  # noqa: E402


_LATENT_CHANNELS = seedvr_vae_mod.SEEDVR2_LATENT_CHANNELS


def test_runtime_decode_zero_temporal_size_preserves_model_slicing():
    class StubVAEModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.slicing_latent_min_size = 2
            self.spatial_downsample_factor = 8
            self.temporal_downsample_factor = 4
            self.device = torch.device("cpu")
            self.use_slicing = True
            self._dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.decode_min_sizes = []
            self.memory_states = []

        def decode_(self, t_chunk):
            self.decode_min_sizes.append(self.slicing_latent_min_size)
            return vae_mod.VideoAutoencoderKL.slicing_decode(self, t_chunk)

        def _decode(self, z, memory_state=MemoryState.DISABLED, memory_cache=None):
            self.memory_states.append(memory_state)
            b, c, d, h, w = z.shape
            return torch.zeros((b, 3, d, h * 8, w * 8), dtype=z.dtype)

    vae = StubVAEModel()
    z = torch.zeros((1, _LATENT_CHANNELS, 5, 8, 8), dtype=torch.float32)

    tiled_vae(
        z,
        vae,
        tile_size=(64, 64),
        tile_overlap=(0, 0),
        temporal_size=0,
        temporal_overlap=0,
        encode=False,
    )

    assert vae.decode_min_sizes == [2]
    assert vae.memory_states == [MemoryState.INITIALIZING, MemoryState.ACTIVE]
    assert vae.slicing_latent_min_size == 2


def test_zero_temporal_size_preserves_min_size_when_encode_raises():
    class RaisingVAEModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.slicing_sample_min_size = 4
            self.spatial_downsample_factor = 8
            self.temporal_downsample_factor = 4
            self.device = torch.device("cpu")
            self._dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def encode(self, t_chunk):
            raise RuntimeError("simulated encode failure")

    vae = RaisingVAEModel()
    x = torch.zeros((1, 3, 12, 64, 64), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="simulated encode failure"):
        tiled_vae(
            x,
            vae,
            tile_size=(64, 64),
            tile_overlap=(0, 0),
            temporal_size=0,
            temporal_overlap=0,
            encode=True,
        )

    assert vae.slicing_sample_min_size == 4


def test_tiled_vae_encode_uses_tensor_return_without_indexing():
    class TensorEncodeVAEModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.slicing_sample_min_size = 4
            self.spatial_downsample_factor = 8
            self.temporal_downsample_factor = 4
            self.device = torch.device("cpu")
            self._dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.calls = []

        def encode(self, t_chunk):
            self.calls.append(tuple(t_chunk.shape))
            b, _, _, h, w = t_chunk.shape
            return torch.ones((b, _LATENT_CHANNELS, 1, h // 8, w // 8), dtype=t_chunk.dtype)

    vae = TensorEncodeVAEModel()
    x = torch.zeros((2, 3, 1, 64, 64), dtype=torch.float32)

    out = tiled_vae(
        x,
        vae,
        tile_size=(64, 64),
        tile_overlap=(0, 0),
        temporal_size=0,
        temporal_overlap=0,
        encode=True,
    )

    assert vae.calls == [(2, 3, 1, 64, 64)]
    assert tuple(out.shape) == (2, _LATENT_CHANNELS, 1, 8, 8)


def test_tiled_vae_preserves_compute_dtype_with_different_parameter_dtype():
    class DummyVAE(nn.Module):
        spatial_downsample_factor = 8
        temporal_downsample_factor = 4
        slicing_sample_min_size = 8

        def __init__(self):
            super().__init__()
            self.device = torch.device("cpu")
            self._dummy = nn.Parameter(torch.zeros(1, dtype=torch.float16))
            self.input_dtype = None

        def encode(self, t_chunk):
            self.input_dtype = t_chunk.dtype
            b, _, _, h, w = t_chunk.shape
            return torch.ones((b, _LATENT_CHANNELS, 1, h // 8, w // 8), dtype=t_chunk.dtype)

    vae = DummyVAE()
    x = torch.zeros((1, 3, 1, 64, 64), dtype=torch.float32)

    tiled_vae(x, vae, tile_size=(64, 64), tile_overlap=(16, 16), encode=True)

    assert vae.input_dtype == torch.float32


def test_tiled_vae_preserves_input_dtype_on_single_tile():
    class FloatOutputVAEModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.slicing_sample_min_size = 4
            self.spatial_downsample_factor = 8
            self.temporal_downsample_factor = 4
            self.device = torch.device("cpu")
            self._dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def encode(self, t_chunk):
            b, _, _, h, w = t_chunk.shape
            return torch.ones((b, _LATENT_CHANNELS, 1, h // 8, w // 8), dtype=torch.float32)

    out = tiled_vae(
        torch.zeros((1, 3, 1, 64, 64), dtype=torch.float16),
        FloatOutputVAEModel(),
        tile_size=(64, 64),
        tile_overlap=(0, 0),
        temporal_size=0,
        temporal_overlap=0,
        encode=True,
    )

    assert out.dtype == torch.float16


class _SlicingDecodeVAE(nn.Module):
    def __init__(self, slicing_latent_min_size):
        super().__init__()
        self.slicing_latent_min_size = slicing_latent_min_size
        self.spatial_downsample_factor = 8
        self.temporal_downsample_factor = 4
        self.device = torch.device("cpu")
        self.use_slicing = True
        self._dummy = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.decode_min_sizes = []
        self.memory_states = []

    def decode_(self, z):
        self.decode_min_sizes.append(self.slicing_latent_min_size)
        return vae_mod.VideoAutoencoderKL.slicing_decode(self, z)

    def _decode(self, z, memory_state=MemoryState.DISABLED, memory_cache=None):
        self.memory_states.append(memory_state)
        x = z[:, :1].repeat(
            1,
            3,
            1,
            self.spatial_downsample_factor,
            self.spatial_downsample_factor,
        )
        return x


def test_decode_tiled_vae_maps_temporal_args_to_latent_slicing_min_size():
    vae = _SlicingDecodeVAE(slicing_latent_min_size=2)
    z = torch.arange(
        _LATENT_CHANNELS * 5 * 8 * 8,
        dtype=torch.float32,
    ).reshape(1, _LATENT_CHANNELS, 5, 8, 8)

    tiled_vae(
        z,
        vae,
        tile_size=(64, 64),
        tile_overlap=(0, 0),
        temporal_size=12,
        temporal_overlap=4,
        encode=False,
    )

    assert vae.decode_min_sizes == [2]
    assert vae.memory_states == [MemoryState.INITIALIZING, MemoryState.ACTIVE]
    assert vae.slicing_latent_min_size == 2

    wrapper = vae_mod.VideoAutoencoderKLWrapper.__new__(
        vae_mod.VideoAutoencoderKLWrapper
    )
    nn.Module.__init__(wrapper)
    seedvr2_tiling = {
        "enable_tiling": True,
        "tile_size": (64, 64),
        "tile_overlap": (0, 0),
        "temporal_size": 8,
        "temporal_overlap": 7,
    }

    captured = {}

    def _fake_tiled_vae(latent, model, **kwargs):
        captured.update(kwargs)
        return torch.zeros(1, 3, 1, 16, 16)

    with patch.object(vae_mod, "tiled_vae", side_effect=_fake_tiled_vae):
        wrapper.decode(torch.zeros(1, _LATENT_CHANNELS, 2, 2), seedvr2_tiling=seedvr2_tiling)

    assert captured["temporal_overlap"] == 7


def _force_oom(*a, **k):
    raise torch.cuda.OutOfMemoryError("forced OOM for dispatcher test")


def _make_vae(first_stage_model, latent_channels, latent_dim):
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    vae.first_stage_model = first_stage_model
    vae.patcher = MagicMock()
    vae.patcher.get_free_memory = MagicMock(return_value=8 * 1024 * 1024 * 1024)
    vae.device = vae.output_device = torch.device("cpu")
    vae.vae_dtype = torch.float32
    vae.disable_offload = True
    vae.extra_1d_channel = None
    vae.upscale_ratio = vae.downscale_ratio = 8
    vae.upscale_index_formula = vae.downscale_index_formula = None
    vae.output_channels = 3
    vae.latent_channels = latent_channels
    vae.latent_dim = latent_dim
    vae.vae_output_dtype = lambda: torch.float32
    vae.spacial_compression_decode = lambda: 8
    vae.handles_tiling = isinstance(first_stage_model, seedvr_vae_mod.VideoAutoencoderKLWrapper)
    vae.format_encoded = None
    vae.process_input = lambda x: x
    vae.process_output = lambda x: x
    vae.throw_exception_if_invalid = lambda: None
    vae.memory_used_decode = lambda *a, **k: 1
    return vae


def _dispatch(vae, samples, seedvr2_call, generic_call, patch_wrapper_decode):
    mm = sd_mod.model_management
    with ExitStack() as stack:
        stack.enter_context(patch.object(mm, "raise_non_oom", lambda e: None))
        stack.enter_context(patch.object(mm, "load_models_gpu", lambda *a, **k: None))
        stack.enter_context(patch.object(mm, "soft_empty_cache", lambda: None))
        stack.enter_context(patch.object(sd_mod.VAE, "_decode_tiled_owned", seedvr2_call))
        stack.enter_context(patch.object(sd_mod.VAE, "decode_tiled_", generic_call))
        if patch_wrapper_decode:
            stack.enter_context(patch.object(
                seedvr_vae_mod.VideoAutoencoderKLWrapper, "decode",
                side_effect=_force_oom))
        vae.decode(samples)


def test_4d_seedvr2_latent_routes_to_owned_decode_tiled():
    wrapper = seedvr_vae_mod.VideoAutoencoderKLWrapper.__new__(
        seedvr_vae_mod.VideoAutoencoderKLWrapper)
    vae = _make_vae(wrapper, latent_channels=_LATENT_CHANNELS, latent_dim=3)
    seedvr2_call = MagicMock(return_value=torch.zeros(1, 3, 9, 64, 64))
    generic_call = MagicMock(return_value=torch.zeros(1, 3, 64, 64))
    _dispatch(vae, torch.zeros(1, _LATENT_CHANNELS * 3, 8, 8), seedvr2_call, generic_call, True)
    assert seedvr2_call.call_count == 1
    assert generic_call.call_count == 0


def test_4d_non_seedvr2_latent_still_routes_to_generic_decode_tiled():
    first_stage = MagicMock()
    first_stage.decode = MagicMock(side_effect=_force_oom)
    vae = _make_vae(first_stage, latent_channels=4, latent_dim=2)
    seedvr2_call = MagicMock(return_value=torch.zeros(1, 3, 9, 64, 64))
    generic_call = MagicMock(return_value=torch.zeros(1, 3, 64, 64))
    _dispatch(vae, torch.zeros(1, 4, 8, 8), seedvr2_call, generic_call, False)
    assert generic_call.call_count == 1
    assert seedvr2_call.call_count == 0


def _populate_common_vae_attrs_fallback(vae):
    vae.patcher = MagicMock()
    vae.patcher.get_free_memory = MagicMock(return_value=8 * 1024 * 1024 * 1024)
    vae.device = torch.device("cpu")
    vae.output_device = torch.device("cpu")
    vae.vae_dtype = torch.float32
    vae.disable_offload = True
    vae.extra_1d_channel = None
    vae.upscale_ratio = 8
    vae.upscale_index_formula = None
    vae.output_channels = 3
    vae.latent_channels = _LATENT_CHANNELS
    vae.latent_dim = 3
    vae.downscale_ratio = 8
    vae.downscale_index_formula = None
    vae.not_video = False
    vae.crop_input = False
    vae.pad_channel_value = None
    vae.handles_tiling = isinstance(vae.first_stage_model, seedvr_vae_mod.VideoAutoencoderKLWrapper)
    vae.format_encoded = None

    vae.vae_output_dtype = lambda: torch.float32
    vae.spacial_compression_encode = lambda: 8
    vae.process_input = lambda x: x
    vae.process_output = lambda x: x
    vae.throw_exception_if_invalid = lambda: None
    vae.memory_used_encode = lambda *a, **k: 1


def _make_seedvr2_vae_fallback():
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    wrapper = seedvr_vae_mod.VideoAutoencoderKLWrapper.__new__(
        seedvr_vae_mod.VideoAutoencoderKLWrapper
    )
    vae.first_stage_model = wrapper
    _populate_common_vae_attrs_fallback(vae)
    return vae


def _make_non_seedvr2_vae_fallback():
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    vae.first_stage_model = MagicMock()
    _populate_common_vae_attrs_fallback(vae)
    return vae


def _force_regular_encode_oom(*args, **kwargs):
    raise torch.cuda.OutOfMemoryError("forced OOM for dispatcher test")


def test_seedvr2_3d_routes_to_owned_encode_tiled_on_oom():
    vae = _make_seedvr2_vae_fallback()
    pixel_samples = torch.zeros((1, 8, 64, 64, 3))

    seedvr2_call = MagicMock(return_value=torch.zeros(1, _LATENT_CHANNELS, 2, 8, 8))
    generic_call = MagicMock(return_value=torch.zeros(1, _LATENT_CHANNELS, 2, 8, 8))

    with patch.object(sd_mod.model_management, "raise_non_oom",
                      lambda e: None), \
         patch.object(sd_mod.model_management, "load_models_gpu",
                      lambda *a, **k: None), \
         patch.object(sd_mod.model_management, "soft_empty_cache",
                      lambda: None), \
         patch.object(seedvr_vae_mod.VideoAutoencoderKLWrapper, "encode",
                      side_effect=_force_regular_encode_oom), \
         patch.object(sd_mod.VAE, "_encode_tiled_owned", seedvr2_call), \
         patch.object(sd_mod.VAE, "encode_tiled_3d", generic_call):
        vae.encode(pixel_samples)

    assert seedvr2_call.call_count == 1, (
        f"Expected _encode_tiled_owned to be called once for a SeedVR2 3D "
        f"input under OOM fallback; got {seedvr2_call.call_count} calls."
    )
    assert generic_call.call_count == 0, (
        f"encode_tiled_3d must NOT be called for a SeedVR2 input; got "
        f"{generic_call.call_count} calls."
    )


def test_non_seedvr2_encode_tiled_3d_default_overlap_is_concrete():
    vae = _make_non_seedvr2_vae_fallback()
    vae.downscale_ratio = (lambda a: max(1, a // 4), 8, 8)
    vae.upscale_ratio = (lambda a: a * 4, 8, 8)
    generic_call = MagicMock(return_value=torch.zeros(1, _LATENT_CHANNELS, 2, 8, 8))
    pixel_samples = torch.zeros((1, 8, 64, 64, 3))

    with patch.object(sd_mod.model_management, "load_models_gpu",
                      lambda *a, **k: None), \
         patch.object(sd_mod.VAE, "encode_tiled_3d", generic_call):
        vae.encode_tiled(pixel_samples)

    assert generic_call.call_args.kwargs["overlap"] == (1, 64, 64)
