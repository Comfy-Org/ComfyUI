"""Run with unittest; no model weights or CUDA allocations are needed."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import comfy.sd
from comfy.text_encoders import krea2, qwen3vl


class Krea2MemoryTests(unittest.TestCase):
    def setUp(self):
        self.config = qwen3vl.Qwen3VL_4BConfig()
        self.model = SimpleNamespace(
            clip_name="qwen3vl_4b", clip="qwen3vl_4b",
            qwen3vl_4b=SimpleNamespace(transformer=SimpleNamespace(model=SimpleNamespace(config=self.config))),
        )
        self.intermediate = patch.object(krea2.model_management, "intermediate_device", return_value=torch.device("cpu"))
        self.intermediate.start()
        self.addCleanup(self.intermediate.stop)

    def tokens(self, length, batch=1, weight=1.0):
        return {"qwen3vl_4b": [[(100, weight)] * length for _ in range(batch)]}

    def estimate(self, tokens, device="cuda:0"):
        return krea2.Krea2TEModel.memory_estimation_function(self.model, tokens, device=device)

    def test_empty_tokens(self):
        for tokens in ({}, {"qwen3vl_4b": []}, {"qwen3vl_4b": [[]]}):
            with self.subTest(tokens=tokens):
                self.assertEqual(self.estimate(tokens), 0)

    def test_sequence_growth_includes_quadratic_workspace(self):
        small = self.estimate(self.tokens(100))
        medium = self.estimate(self.tokens(200))
        large = self.estimate(self.tokens(300))
        self.assertGreater(small, 0)
        self.assertGreater(medium, small)
        self.assertGreater(large - medium, medium - small)

    def test_batch_and_weighted_empty_section(self):
        single = self.estimate(self.tokens(100))
        double = self.estimate(self.tokens(100, batch=2))
        weighted = self.estimate(self.tokens(100, weight=1.2))
        self.assertGreater(double, single)
        self.assertEqual(weighted, double)

    def test_fp8_weights_do_not_reduce_activation_estimate(self):
        transformer = self.model.qwen3vl_4b.transformer
        transformer.dtype = torch.float16
        expected = self.estimate(self.tokens(100))
        transformer.dtype = torch.float8_e4m3fn
        self.assertEqual(self.estimate(self.tokens(100)), expected)
        self.assertGreaterEqual(expected, 2 * 12 * 100 * self.config.hidden_size * 4)

    def test_config_dimensions_not_packed_weight_shape(self):
        original = self.estimate(self.tokens(100))
        self.config.intermediate_size *= 2
        self.assertGreater(self.estimate(self.tokens(100)), original)

    def test_tensor_and_dictionary_embedding_expansion(self):
        # Meta tensors carry shapes but no storage; the estimator must not read their data.
        embedding = torch.empty((1, 100, self.config.hidden_size), device="meta")
        expected = self.estimate(self.tokens(100))
        for token in (embedding, {"type": "embedding", "data": embedding}):
            self.assertEqual(self.estimate({"qwen3vl_4b": [[(token, 1.0)]]}), expected)

    def test_image_language_sequence_expansion(self):
        # This verifies language-model length, not the separate vision encoder's peak.
        for height, width, expected in ((512, 512, 256), (32, 32, 4), (4096, 4096, 12544)):
            image = torch.empty((1, height, width, 3), device="meta")
            tokens = {"qwen3vl_4b": [[({"type": "image", "data": image}, 1.0)]]}
            with self.subTest(height=height, width=width):
                self.assertEqual(self.estimate(tokens), self.estimate(self.tokens(expected)))

    def test_gpu_intermediates_include_conditioning_copy(self):
        tokens = self.tokens(100)
        cpu_intermediate = self.estimate(tokens)
        with patch.object(krea2.model_management, "intermediate_device", return_value=torch.device("cuda:0")):
            self.assertGreater(self.estimate(tokens), cpu_intermediate)

    def test_device_contract_and_no_tensor_allocations(self):
        tokens = self.tokens(100)
        original = copy.deepcopy(tokens)
        with patch.object(torch, "empty", side_effect=AssertionError("allocation")), patch.object(torch, "tensor", side_effect=AssertionError("allocation")), patch.object(torch.cuda, "current_device", side_effect=AssertionError("CUDA access")):
            for device in (None, "cpu", "cuda:0", "mps", "xpu:0", "privateuseone:0"):
                self.assertIsInstance(self.estimate(tokens, device=device), int)
        self.assertEqual(tokens, original)

    def test_clip_load_model_passes_estimate_to_manager(self):
        tokens = self.tokens(100)
        expected = self.estimate(tokens)
        encoder = SimpleNamespace(memory_estimation_function=lambda values, device: self.estimate(values, device))
        clip = SimpleNamespace(cond_stage_model=encoder, patcher=SimpleNamespace(load_device=torch.device("cuda:0"), is_dynamic=lambda: False))
        with patch.object(comfy.sd.model_management, "load_models_gpu") as load:
            self.assertIs(comfy.sd.CLIP.load_model(clip, tokens), clip.patcher)
        load.assert_called_once_with([clip.patcher], memory_required=expected)

    def test_legacy_encoder_without_estimator_keeps_default(self):
        clip = SimpleNamespace(cond_stage_model=object(), patcher=SimpleNamespace(load_device=torch.device("cpu"), is_dynamic=lambda: False))
        with patch.object(comfy.sd.model_management, "load_models_gpu") as load:
            comfy.sd.CLIP.load_model(clip, {})
        load.assert_called_once_with([clip.patcher], memory_required=0)

    def test_non_dynamic_reservation_can_reclaim_dynamic_diffusion_weights(self):
        management = comfy.sd.model_management
        device = torch.device("cuda:0")
        for for_dynamic in (False, True):
            loaded = Mock(device=device)
            loaded.is_dead.return_value = False
            loaded.model.is_dynamic.return_value = True
            loaded.model.loaded_size.return_value = 1024
            loaded.model_offloaded_memory.return_value = 0
            loaded.model_memory.return_value = 1024
            loaded.model_unload.return_value = True
            with self.subTest(for_dynamic=for_dynamic), patch.object(management, "current_loaded_models", [loaded]), patch.object(management, "cleanup_models_gc"), patch.object(management, "DISABLE_SMART_MEMORY", False), patch.object(management, "get_free_memory", side_effect=lambda device, torch_free_too=False: (0, 0) if torch_free_too else 0), patch.object(management, "soft_empty_cache"):
                management.free_memory(512, device, for_dynamic=for_dynamic)
            if for_dynamic:
                loaded.model_unload.assert_not_called()
            else:
                loaded.model_unload.assert_called_once_with(512)

    def test_only_krea2_selects_non_dynamic_text_encoder(self):
        for clip_type, expected in ((comfy.sd.CLIPType.KREA2, True), (comfy.sd.CLIPType.MAGE, False), (comfy.sd.CLIPType.STABLE_DIFFUSION, False)):
            with self.subTest(clip_type=clip_type), patch.object(comfy.sd, "detect_te_model", return_value=comfy.sd.TEModel.QWEN3VL_4B), patch.object(comfy.sd, "llama_detect", return_value={}), patch.object(comfy.sd, "CLIP") as clip:
                comfy.sd.load_text_encoder_state_dicts([{}], clip_type=clip_type)
                self.assertEqual(clip.call_args.kwargs["disable_dynamic"], expected)


if __name__ == "__main__":
    unittest.main()
