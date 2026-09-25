"""What the volume prefetch downloads: the files the wrapper will load."""

import importlib.util
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from api_wrapper import workflows as wf  # noqa: E402


def _prefetch():
    spec = importlib.util.spec_from_file_location("prefetch_models", os.path.join(ROOT, "docker", "prefetch_models.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPrefetchSpecs(unittest.TestCase):
    def test_the_default_set_is_the_rigs_int8_unets(self):
        """A shot must look the same whichever GPU renders it, and the rig runs int8."""
        self.assertEqual(_prefetch().DEFAULT_SPECS, ["minimaxh3:int8", "minimaxh3-ref:int8", "flux2klein9b"])

    def test_an_int8_unet_comes_with_the_encoder_the_picker_loads(self):
        files = {m["filename"] for m in _prefetch().models_for("minimaxh3-ref:int8")}
        self.assertIn(wf.MINIMAX_H3_REF2VA_UNET_INT8, files)
        self.assertIn(wf.MINIMAX_H3_CLIP_NVFP4, files)          # 16 GB, what runs
        self.assertNotIn(wf.MINIMAX_H3_CLIP_INT8, files)        # 34 GB, never loaded
        picked = wf.minimax_h3_model_pick("int8", True, lambda folder, name: name in files)
        self.assertEqual(picked[:2], (wf.MINIMAX_H3_REF2VA_UNET_INT8, wf.MINIMAX_H3_CLIP_NVFP4))

    def test_nvfp4_still_means_the_fp8_unet(self):
        files = {m["filename"] for m in _prefetch().models_for("minimaxh3:nvfp4")}
        self.assertIn(wf.MINIMAX_H3_UNET_FP8, files)
        self.assertIn(wf.MINIMAX_H3_CLIP_NVFP4, files)


if __name__ == "__main__":
    unittest.main()
