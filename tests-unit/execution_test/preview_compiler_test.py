from unittest.mock import MagicMock, Mock

import pytest

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
    previewer = latent_preview.TAEHVPreviewerImpl(taesd, compile_preview=True)
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
    previewer = latent_preview.TAEHVPreviewerImpl(taesd, compile_preview=True)
    x0 = MagicMock()

    monkeypatch.setattr(latent_preview, "preview_to_image", Mock())
    monkeypatch.setattr(latent_preview.comfy.model_prefetch, "malloc_graph_enabled", Mock(return_value=True))
    monkeypatch.setattr(latent_preview.comfy.model_prefetch, "malloc_graph_begin", Mock())
    end = Mock()
    monkeypatch.setattr(latent_preview.comfy.model_prefetch, "malloc_graph_end", end)

    with pytest.raises(RuntimeError, match="decode failed"):
        previewer.decode_latent_to_preview(x0)

    end.assert_not_called()
