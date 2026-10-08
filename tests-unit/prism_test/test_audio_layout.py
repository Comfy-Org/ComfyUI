"""Audio layout regression tests through the real VAE decode entry points."""
from types import SimpleNamespace

import pytest
import torch

from comfy import model_management
from comfy.ldm.audio.prism_dac import PrismDAC
from comfy.nested_tensor import NestedTensor
from comfy.sd import VAE
from comfy_extras.nodes_audio import vae_decode_audio


class HopDecoder(torch.nn.Module):
    """Small deterministic codec substitute; preserve DAC channel-first output."""

    def forward(self, latent):
        """Expand latent steps by the DAC hop without loading a large checkpoint."""
        return latent.repeat_interleave(960, dim=-1)


@pytest.fixture
def audio_vae(monkeypatch):
    """Keep real PrismDAC/VAE decode methods; stub weights and device loading only."""
    codec = PrismDAC.__new__(PrismDAC)
    torch.nn.Module.__init__(codec)
    codec.post_quant_conv = torch.nn.Identity()
    codec.decoder = HopDecoder()
    vae = VAE.__new__(VAE)
    vae.first_stage_model = codec
    vae.device = vae.output_device = torch.device('cpu')
    vae.vae_dtype = torch.float32
    vae.latent_dim = 1
    vae.extra_1d_channel = None
    vae.disable_offload = False
    vae.handles_tiling = False
    vae.output_channels = 1
    vae.upscale_ratio = 960
    vae.audio_sample_rate = 48000
    vae.process_output = lambda audio: audio
    vae.memory_used_decode = lambda shape, dtype: 1
    vae.patcher = SimpleNamespace(get_free_memory=lambda device: 1024)
    monkeypatch.setattr(model_management, 'load_models_gpu', lambda *args, **kwargs: None)
    monkeypatch.setattr(model_management, 'intermediate_dtype', lambda: torch.float32)
    # Match the graph executor: tiled_scale_multidim returns inference tensors.
    with torch.inference_mode():
        yield vae


@pytest.mark.parametrize('tiled', [False, True])
@pytest.mark.parametrize('channels', [1, 2])
@pytest.mark.parametrize('num_samples', [None, 162000, 162240])
@pytest.mark.parametrize('nested', [False, True])
def test_audio_sample_axis_and_trim(audio_vae, tiled, channels, num_samples, nested):
    """VAE wrappers and AUDIO conversion preserve channels and trim only time."""
    # 81 frames at 24 fps: 162000 samples, padded to 169 codec hops.
    latent = torch.linspace(-0.1, 0.1, 2 * channels * 169).reshape(2, channels, 169)
    audio_vae.output_channels = channels
    raw = audio_vae.first_stage_model.decode(latent)
    assert raw.shape == (2, channels, 162240)
    decoded = (audio_vae.decode_tiled(latent, tile_x=64, tile_y=64, overlap=8)
               if tiled else audio_vae.decode(latent))
    assert decoded.shape == (2, 162240, channels)
    torch.testing.assert_close(decoded.movedim(-1, 1), raw)
    samples = {'samples': NestedTensor((torch.zeros(2, 16, 1, 2, 2), latent)) if nested else latent}
    if num_samples is not None:
        samples['num_samples'] = num_samples
    actual = vae_decode_audio(audio_vae, samples, tile=64 if tiled else None, overlap=8 if tiled else None)
    expected = raw if num_samples is None else raw[..., :num_samples]
    assert actual['waveform'].shape == expected.shape
    torch.testing.assert_close(actual['waveform'], expected)
    assert actual['sample_rate'] == 48000


@pytest.mark.parametrize('num_samples', [0, -1, 1921, 1500.5, '1500'])
@pytest.mark.parametrize('tiled', [False, True])
def test_audio_invalid_original_length(audio_vae, num_samples, tiled):
    """Invalid counts are checked against decoded samples, not channel count."""
    samples = {'samples': torch.zeros(1, 1, 2), 'num_samples': num_samples}
    with pytest.raises(ValueError, match='within the decoded length'):
        vae_decode_audio(audio_vae, samples, tile=64 if tiled else None, overlap=8 if tiled else None)


def test_audio_sample_rate_override(audio_vae):
    """Keep an explicit latent sample rate when trimming codec-hop padding."""
    audio = vae_decode_audio(audio_vae, {'samples': torch.zeros(1, 1, 2), 'num_samples': 1500, 'sample_rate': 24000})
    assert audio['waveform'].shape == (1, 1, 1500)
    assert audio['sample_rate'] == 24000
