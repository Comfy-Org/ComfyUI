from fractions import Fraction
from types import SimpleNamespace

import av
import pytest
import torch

import comfy.model_management as model_management
import folder_paths
from comfy.ldm.wan.model_vsr import RelayFlash, relay_keyframes, relay_latent_size
from comfy_extras.nodes_wan import VSRDecodeAndSave


@pytest.fixture
def relay_video(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(model_management, "load_models_gpu", lambda *args, **kwargs: None)

    def make(scale, count):
        decoder = RelayFlash(scale, latent_channels=4, dim=64, depth=0, num_heads=1, kv_groups=1, head_dim=64, device="cpu")
        with torch.no_grad():
            for param in decoder.parameters():
                param.zero_()
        # Solid colors make the frame order easy to check even after lossy encoding.
        images = torch.linspace(0.1, 0.9, count).view(count, 1, 1, 1).expand(count, 9, 11, 3)
        height, width = relay_latent_size(9, 11, decoder.latent_stride)
        samples = torch.zeros(1, 4, len(relay_keyframes(count)), height, width)
        vae = SimpleNamespace(
            first_stage_model=decoder,
            device=torch.device("cpu"),
            vae_dtype=torch.float32,
            vae_output_dtype=lambda: torch.float32,
            patcher=None,
            disable_offload=False,
            memory_used_decode=lambda shape, dtype: 0,
            throw_exception_if_invalid=lambda: None,
        )
        return vae, samples, images

    return make


@pytest.mark.parametrize("scale,count,with_audio", [(2, 1, False), (4, 17, True)])
@torch.inference_mode()
def test_relayvsr_streamed_video(relay_video, tmp_path, monkeypatch, scale, count, with_audio):
    vae, samples, images = relay_video(scale, count)
    decoder = vae.first_stage_model
    dense = decoder.decode(samples, lq_images=images)
    assert dense.shape == (1, 3, count, 9 * scale, 11 * scale)
    torch.testing.assert_close(dense[0, 0, :, 0, 0].cpu(), images[:, 0, 0, 0])

    def forbid_dense_decode(*args, **kwargs):
        pytest.fail("Streaming must not call the dense decoder")

    monkeypatch.setattr(decoder, "decode", forbid_dense_decode)
    audio = {"waveform": torch.zeros(1, 2, 48000), "sample_rate": 48000} if with_audio else None
    result = VSRDecodeAndSave.execute(vae, {"samples": samples}, images, 29.97, "video/test", audio=audio)
    entry, = result.ui.as_dict()["images"]
    path = tmp_path / entry["subfolder"] / entry["filename"]
    assert entry["type"] == "output"
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        assert (stream.width, stream.height) == (11 * scale, 9 * scale)
        assert stream.average_rate == Fraction(2997, 100)
        frames = list(video.decode(video=0))
        assert len(frames) == count
        for frame, expected in zip(frames, images[:, 0, 0, 0]):
            assert frame.to_ndarray(format="rgb24").mean() == pytest.approx(float(expected) * 255, abs=3)
    with av.open(str(path)) as video:
        assert bool(video.streams.audio) == with_audio
        if with_audio:
            stream = video.streams.audio[0]
            assert stream.layout.name == "stereo"
            assert stream.sample_rate == 48000
            assert float(stream.duration * stream.time_base) == pytest.approx(count / 29.97, abs=1024 / 48000)


@torch.inference_mode()
def test_relayvsr_cancel_removes_partial_file(relay_video, tmp_path, monkeypatch):
    vae, samples, images = relay_video(2, 4)
    checks = iter([False, False, True])

    def interrupt():
        if next(checks):
            raise model_management.InterruptProcessingException()

    monkeypatch.setattr(model_management, "throw_exception_if_processing_interrupted", interrupt)
    with pytest.raises(model_management.InterruptProcessingException):
        VSRDecodeAndSave.execute(vae, {"samples": samples}, images, 24, "test")
    assert not list(tmp_path.iterdir())
