import os

import pytest

from comfy_api.latest._encoded_media import EncodedMediaGroup


def test_group_platform_refusal_has_no_filesystem_side_effect(tmp_path, monkeypatch):
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    with pytest.raises(NotImplementedError, match="descriptor-relative"):
        EncodedMediaGroup(str(tmp_path), "example", "output", max_bytes=4096,
                          max_entries=4, timeout_seconds=2, failed_entry_policy="retain")
    assert not list(tmp_path.iterdir())
