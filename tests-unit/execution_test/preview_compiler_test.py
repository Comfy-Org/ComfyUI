from unittest.mock import MagicMock, Mock

import pytest
import torch

import latent_preview
from comfy import latent_formats


def test_minimax_h3_enables_preview_compiler():
    assert latent_formats.MiniMaxH3Video.compile_preview
    assert latent_formats.MiniMaxH3AV.compile_preview
    assert not latent_formats.HunyuanVideo.compile_preview


def test_video_preview_compiles_decode(monkeypatch):
    taesd = Mock()
    taesd.device = "cuda:0"
    taesd.decode.return_value = [[Mock()]]
    latent_format = Mock()
    latent_format.process_out = Mock(side_effect=lambda x: x)
    previewer = latent_preview.TAEHVPreviewerImpl(taesd, latent_format, compile_preview=True)
    x0 = MagicMock()
    samples = Mock(shape=(1, 24, 1, 30, 52))
    x0.__getitem__.return_value = samples

    monkeypatch.setattr(latent_preview, "preview_to_image", Mock())
    monkeypatch.setattr(latent_preview.comfy.model_prefetch, "malloc_graph_enabled", Mock(return_value=True))
    calls = []
    taesd.decode.side_effect = lambda value: calls.append(("decode", value)) or [[Mock()]]
    begin = Mock(side_effect=lambda device: calls.append(("begin", device)))
    end = Mock(side_effect=lambda: calls.append(("end",)))
    monkeypatch.setattr(latent_preview.comfy.model_prefetch, "malloc_graph_begin", begin)
    monkeypatch.setattr(latent_preview.comfy.model_prefetch, "malloc_graph_end", end)

    previewer.decode_latent_to_preview(x0)
    assert calls == [
        ("begin", "cuda:0"),
        ("decode", samples),
        ("end",),
    ]


def test_video_preview_leaves_failed_compiler_scope_for_execution_cleanup(monkeypatch):
    taesd = Mock()
    taesd.device = "cuda:0"
    taesd.decode.side_effect = RuntimeError("decode failed")
    latent_format = Mock()
    latent_format.process_out = Mock(side_effect=lambda x: x)
    previewer = latent_preview.TAEHVPreviewerImpl(taesd, latent_format, compile_preview=True)
    x0 = MagicMock()

    monkeypatch.setattr(latent_preview, "preview_to_image", Mock())
    monkeypatch.setattr(latent_preview.comfy.model_prefetch, "malloc_graph_enabled", Mock(return_value=True))
    monkeypatch.setattr(latent_preview.comfy.model_prefetch, "malloc_graph_begin", Mock())
    end = Mock()
    monkeypatch.setattr(latent_preview.comfy.model_prefetch, "malloc_graph_end", end)

    with pytest.raises(RuntimeError, match="decode failed"):
        previewer.decode_latent_to_preview(x0)

    end.assert_not_called()


def test_video_preview_unnormalizes_wan21_latent_before_decode(monkeypatch):
    # lighttaew2_1 is trained on raw Wan 2.1 latents (comfy.taesd.taehv.TAEHV
    # is built with latent_format=None for it), but the sampler's x0 is in
    # Wan21's mean/std-normalized model space, so it must be un-normalized
    # with the model's own latent_format before decoding.
    taesd = Mock()
    taesd.device = "cpu"
    taesd.decode.return_value = [[Mock()]]
    monkeypatch.setattr(latent_preview, "preview_to_image", Mock())

    latent_format = latent_formats.Wan21()
    previewer = latent_preview.TAEHVPreviewerImpl(taesd, latent_format, compile_preview=False)

    x0 = torch.randn(1, latent_format.latent_channels, 3, 4, 4)
    previewer.decode_latent_to_preview(x0)

    decoded_input = taesd.decode.call_args[0][0]
    normalized = x0[:1, :, :1]
    expected = latent_format.process_out(normalized)
    assert torch.allclose(decoded_input, expected)
    assert not torch.allclose(decoded_input, normalized)
