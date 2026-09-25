"""Tests for the model downloader's truncation protection.

An interrupted download used to leave a silently truncated model file (the
safetensors loader then crashes with 'shape [...] is invalid for input of
size ...'). The resolver now verifies byte sizes against the server metadata
before and after copying to the models dir.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import huggingface_hub

from comfy import model_downloader

URL = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors"


def _make_file(path, size):
    with open(path, "wb") as f:
        f.write(b"x" * size)
    return path


class _Meta:
    def __init__(self, size):
        self.size = size


class TestResolveUrlDownloadVerification(unittest.TestCase):
    def test_truncated_cache_blob_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_file = _make_file(os.path.join(tmp, "blob"), 100)  # truncated cache
            dest = os.path.join(tmp, "model.safetensors")
            with mock.patch.object(huggingface_hub, "hf_hub_download", return_value=cache_file), \
                 mock.patch.object(huggingface_hub, "get_hf_file_metadata", return_value=_Meta(1000)):
                result = model_downloader._download_huggingface_resolve_url(URL, dest)
            self.assertFalse(result)
            self.assertFalse(os.path.exists(dest), "truncated file must not be copied to the models dir")

    def test_truncated_copy_is_rejected_and_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_file = _make_file(os.path.join(tmp, "blob"), 1000)
            dest = os.path.join(tmp, "model.safetensors")

            def truncated_copy(src, dst):
                _make_file(dst, 100)  # simulate a disk-full copy

            with mock.patch.object(huggingface_hub, "hf_hub_download", return_value=cache_file), \
                 mock.patch.object(huggingface_hub, "get_hf_file_metadata", return_value=_Meta(1000)), \
                 mock.patch.object(model_downloader, "_copy_to_dest", side_effect=truncated_copy):
                result = model_downloader._download_huggingface_resolve_url(URL, dest)
            self.assertFalse(result)
            self.assertFalse(os.path.exists(dest), "truncated copy must be removed")

    def test_full_download_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_file = _make_file(os.path.join(tmp, "blob"), 1000)
            dest = os.path.join(tmp, "model.safetensors")
            with mock.patch.object(huggingface_hub, "hf_hub_download", return_value=cache_file), \
                 mock.patch.object(huggingface_hub, "get_hf_file_metadata", return_value=_Meta(1000)):
                result = model_downloader._download_huggingface_resolve_url(URL, dest)
            self.assertTrue(result)
            self.assertEqual(os.path.getsize(dest), 1000)


class TestAtFormatDownloadVerification(unittest.TestCase):
    def test_truncated_copy_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_file = _make_file(os.path.join(tmp, "blob"), 500)
            dest = os.path.join(tmp, "model.safetensors")

            def truncated_copy(src, dst):
                _make_file(dst, 10)

            with mock.patch.object(huggingface_hub, "hf_hub_download", return_value=cache_file), \
                 mock.patch.object(model_downloader, "_copy_to_dest", side_effect=truncated_copy):
                result = model_downloader._download_huggingface_file(
                    "Comfy-Org/MiniMax-H3@text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors", dest)
            self.assertFalse(result)
            self.assertFalse(os.path.exists(dest))


def _fake_local_dir_download(size):
    """hf_hub_download stand-in: with local_dir it writes local_dir/<filename>
    (what huggingface_hub does); direct mode must always pass local_dir."""
    calls = []

    def fake(repo_id, filename, local_dir=None, **kwargs):
        calls.append({"repo_id": repo_id, "filename": filename, "local_dir": local_dir})
        assert local_dir is not None, "direct mode must download with local_dir"
        path = os.path.join(local_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return _make_file(path, size)

    return fake, calls


class TestDirectDownloadMode(unittest.TestCase):
    """COMFY_HF_DOWNLOAD_MODE=direct keeps one copy of each file (no HF cache)."""

    def setUp(self):
        self._env = mock.patch.dict(os.environ, {"COMFY_HF_DOWNLOAD_MODE": "direct"})
        self._env.start()

    def tearDown(self):
        self._env.stop()

    def _models(self, tmp):
        dest = os.path.join(tmp, "models", "text_encoders", "model.safetensors")
        return os.path.join(tmp, "models"), dest

    def test_resolve_url_lands_one_copy_and_cleans_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            models, dest = self._models(tmp)
            fake, calls = _fake_local_dir_download(1000)
            with mock.patch.object(huggingface_hub, "hf_hub_download", side_effect=fake), \
                 mock.patch.object(huggingface_hub, "get_hf_file_metadata", return_value=_Meta(1000)), \
                 mock.patch.object(model_downloader.shutil, "copy2", side_effect=AssertionError("must not copy")):
                result = model_downloader._download_huggingface_resolve_url(URL, dest)
            self.assertTrue(result)
            self.assertEqual(os.path.getsize(dest), 1000)
            self.assertEqual(calls[0]["filename"], "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors")
            # Staged beside the folders ComfyUI scans, then removed.
            self.assertTrue(calls[0]["local_dir"].startswith(os.path.join(models, ".hf-staging")))
            self.assertEqual(os.listdir(os.path.join(models, ".hf-staging")), [])
            self.assertEqual(os.listdir(os.path.dirname(dest)), ["model.safetensors"])

    def test_truncated_direct_download_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, dest = self._models(tmp)
            fake, _ = _fake_local_dir_download(100)
            with mock.patch.object(huggingface_hub, "hf_hub_download", side_effect=fake), \
                 mock.patch.object(huggingface_hub, "get_hf_file_metadata", return_value=_Meta(1000)):
                result = model_downloader._download_huggingface_resolve_url(URL, dest)
            self.assertFalse(result)
            self.assertFalse(os.path.exists(dest), "a truncated file must never be left in the models dir")

    def test_at_format_lands_one_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, dest = self._models(tmp)
            fake, _ = _fake_local_dir_download(500)
            with mock.patch.object(huggingface_hub, "hf_hub_download", side_effect=fake), \
                 mock.patch.object(model_downloader.shutil, "copy2", side_effect=AssertionError("must not copy")):
                result = model_downloader._download_huggingface_file(
                    "Comfy-Org/MiniMax-H3@text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors", dest)
            self.assertTrue(result)
            self.assertEqual(os.path.getsize(dest), 500)

    def test_cache_mode_is_still_the_default(self):
        with mock.patch.dict(os.environ, {"COMFY_HF_DOWNLOAD_MODE": ""}):
            self.assertFalse(model_downloader._hf_direct_mode())
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(model_downloader._hf_direct_mode())
        self.assertTrue(model_downloader._hf_direct_mode())


if __name__ == "__main__":
    unittest.main()
