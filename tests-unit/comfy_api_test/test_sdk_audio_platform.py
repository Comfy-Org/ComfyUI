import os

import pytest

from comfy_api.latest._deferred_audio import AudioWindow, DeferredAudioPool


def test_capture_platform_refusal_precedes_source_open(tmp_path, monkeypatch):
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    with pytest.raises(NotImplementedError, match="no-follow"):
        DeferredAudioPool().capture(str(tmp_path / "absent.wav"), AudioWindow(0, None, 1024, 2), 4096)
    assert not list(tmp_path.iterdir())
