"""Qwen Image workspace and patcher tests without model weights or CUDA allocations.

Run through test_memory.py to keep CPU setup in a separate process.
"""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import comfy.sd
import comfy.supported_models
from comfy import model_patcher, sd1_clip
from comfy.text_encoders import hunyuan_image, kandinsky5, llama, qwen3vl, qwen_image, qwen_image21


class QwenImageMemoryTests(unittest.TestCase):
    def setUp(self):
        self.variants = (
            ("qwen25_7b", llama.Qwen25_7BVLI_Config(), qwen_image.te()),
            ("qwen3vl_8b", qwen3vl.Qwen3VL_8BConfig(), qwen_image21.te()),
        )
        intermediate = patch.object(qwen_image.model_management, "intermediate_device", return_value=torch.device("cpu"))
        intermediate.start()
        self.addCleanup(intermediate.stop)

    def model(self, name, config):
        embedding = SimpleNamespace(weight=SimpleNamespace(dtype=torch.bfloat16))
        language_model = SimpleNamespace(config=config, embed_tokens=embedding)
        return SimpleNamespace(clip_name=name, clip=name, **{
            name: SimpleNamespace(transformer=SimpleNamespace(model=language_model)),
        })

    def tokens(self, name, length, batch=1, weight=1.0):
        return {name: [[(100, weight)] * length for _ in range(batch)]}

    def estimate(self, variant, tokens, device="cuda:0"):
        name, config, cls = variant
        return cls.memory_estimation_function(self.model(name, config), tokens, device=device)

    def test_empty_tokens(self):
        for variant in self.variants:
            name = variant[0]
            for tokens in ({}, {name: []}, {name: [[]]}):
                with self.subTest(name=name, tokens=tokens):
                    self.assertEqual(self.estimate(variant, tokens), 0)

    def test_sequence_growth_includes_quadratic_workspace(self):
        for variant in self.variants:
            values = [self.estimate(variant, self.tokens(variant[0], length)) for length in (100, 200, 300)]
            self.assertGreater(values[0], 0)
            self.assertGreater(values[1], values[0])
            self.assertGreater(values[2] - values[1], values[1] - values[0])

    def test_batch_and_weighted_empty_section(self):
        for variant in self.variants:
            name = variant[0]
            single = self.estimate(variant, self.tokens(name, 100))
            double = self.estimate(variant, self.tokens(name, 100, batch=2))
            self.assertGreater(double, single)
            self.assertEqual(self.estimate(variant, self.tokens(name, 100, weight=1.2)), double)

    def test_fp8_weights_do_not_reduce_fp32_activation_estimate(self):
        for name, config, cls in self.variants:
            model = self.model(name, config)
            transformer = getattr(model, name).transformer
            transformer.dtype = torch.bfloat16
            expected = cls.memory_estimation_function(model, self.tokens(name, 100), device="cuda:0")
            transformer.dtype = torch.float8_e4m3fn
            self.assertEqual(cls.memory_estimation_function(model, self.tokens(name, 100), device="cuda:0"), expected)

    def test_raw_fp8_embedding_reserves_full_vocabulary_cast(self):
        for name, config, cls in self.variants:
            model = self.model(name, config)
            embedding = getattr(model, name).transformer.model.embed_tokens
            tokens = self.tokens(name, 100)
            normal = cls.memory_estimation_function(model, tokens, device="cuda:0")
            for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                embedding.weight.dtype = dtype
                with patch.object(torch, "empty", side_effect=AssertionError("allocation")):
                    estimated = cls.memory_estimation_function(model, tokens, device="cuda:0")
                self.assertGreater(estimated, config.vocab_size * config.hidden_size * 4)
                self.assertGreater(estimated, normal)

    def test_config_dimensions_not_packed_weight_shape(self):
        for variant in self.variants:
            tokens = self.tokens(variant[0], 100)
            expected = self.estimate(variant, tokens)
            variant[1].intermediate_size *= 2
            self.assertGreater(self.estimate(variant, tokens), expected)

    def test_tensor_and_dictionary_embeddings(self):
        for variant in self.variants:
            name, config, _ = variant
            embedding = torch.empty((2, 50, config.hidden_size), device="meta")
            expected = self.estimate(variant, self.tokens(name, 100))
            for token in (embedding, {"type": "embedding", "data": embedding}):
                self.assertEqual(self.estimate(variant, {name: [[(token, 1.0)]]}), expected)

    def test_language_image_expansion_matches_preprocessor(self):
        # Disable visual workspace/features here to isolate the expanded language length.
        for variant, factor, examples in (
            (self.variants[0], 28, ((512, 512, 324), (32, 32, 4), (4096, 4096, 16384), (70, 98, 8))),
            (self.variants[1], 32, ((512, 512, 256), (32, 32, 4), (4096, 4096, 12544), (80, 112, 8))),
        ):
            name, config, _ = variant
            vision_config = dict(patch_size=factor // 2, spatial_merge_size=2, hidden_size=0, intermediate_size=0, num_heads=0)
            for height, width, length in examples:
                image = torch.empty((1, height, width, 3), device="meta")
                sections = [[({"type": "image", "data": image}, 1.0)]]
                expected = qwen_image._estimate_qwen_image_memory([[(100, 1.0)] * length], config, vision_config)
                # The helper retains one image-language embedding alongside LM buffers.
                expected += length * config.hidden_size * 4
                self.assertEqual(qwen_image._estimate_qwen_image_memory(sections, config, vision_config), expected)

    def test_image_batch_placeholder_uses_only_first_item(self):
        for variant in self.variants:
            name = variant[0]
            values = []
            for batch in (1, 3):
                image = torch.empty((batch, 512, 512, 3), device="meta")
                values.append(self.estimate(variant, {name: [[({"type": "image", "data": image}, 1.0)]]}))
            self.assertEqual(values[0], values[1])

    def test_vision_attention_uses_premerge_patch_count(self):
        for variant, patches in ((self.variants[0], 1296), (self.variants[1], 1024)):
            name = variant[0]
            image = torch.empty((1, 512, 512, 3), device="meta")
            estimated = self.estimate(variant, {name: [[({"type": "image", "data": image}, 1.0)]]})
            self.assertGreater(estimated, 8 * 16 * patches * patches)

    def test_multiple_images_do_not_sum_visual_attention_peaks(self):
        name, config, _ = self.variants[1]
        config = copy.copy(config)
        # Isolate score workspace with negligible language/vision features.
        config.hidden_size = config.intermediate_size = 1
        config.num_attention_heads = config.num_key_value_heads = config.head_dim = 1
        visual = dict(patch_size=16, spatial_merge_size=2, hidden_size=0, intermediate_size=0, num_heads=16)
        image = torch.empty((1, 512, 512, 3), device="meta")
        token = ({"type": "image", "data": image}, 1.0)
        one = qwen_image._estimate_qwen_image_memory([[token]], config, visual)
        two = qwen_image._estimate_qwen_image_memory([[token, token]], config, visual)
        self.assertGreater(two, one)
        self.assertLess(two, 1.1 * one)

    def test_deepstack_features_are_retained(self):
        _, config, _ = self.variants[1]
        visual = {**qwen3vl.QWEN3VL_VISION_COMMON, **qwen3vl.QWEN3VL_VISION["qwen3vl_8b"]}
        image = torch.empty((1, 512, 512, 3), device="meta")
        sections = [[({"type": "image", "data": image}, 1.0)]]
        stacked = qwen_image._estimate_qwen_image_memory(sections, config, visual)
        visual["deepstack_visual_indexes"] = []
        plain = qwen_image._estimate_qwen_image_memory(sections, config, visual)
        self.assertEqual(stacked - plain, 2 * 3 * 256 * config.hidden_size * 4)

    def test_keep_vision_does_not_reduce_encode_workspace(self):
        variant = self.variants[1]
        image = torch.empty((1, 512, 512, 3), device="meta")
        tokens = {variant[0]: [[({"type": "image", "data": image}, 1.0)]]}
        expected = self.estimate(variant, tokens)
        for keep in (False, True):
            self.assertEqual(self.estimate(variant, {**tokens, "keep_vision": keep}), expected)

    def test_gpu_intermediate_conditioning_copies(self):
        for variant in self.variants:
            tokens = self.tokens(variant[0], 100)
            expected = self.estimate(variant, tokens)
            with patch.object(qwen_image.model_management, "intermediate_device", return_value=torch.device("cuda:0")):
                self.assertEqual(self.estimate(variant, tokens) - expected, 2 * 100 * variant[1].hidden_size * 4)

    def test_no_allocations_cuda_access_or_input_mutation(self):
        for variant in self.variants:
            tokens = self.tokens(variant[0], 100)
            original = copy.deepcopy(tokens)
            image = torch.empty((1, 512, 512, 3), device="meta")
            images = {variant[0]: [[({"type": "image", "data": image}, 1.0)]]}
            with patch.object(torch, "empty", side_effect=AssertionError("allocation")), patch.object(torch, "tensor", side_effect=AssertionError("allocation")), patch.object(torch.cuda, "current_device", side_effect=AssertionError("CUDA access")):
                for device in (None, "cpu", "cuda:0", "mps", "xpu:0", "privateuseone:0"):
                    self.assertIsInstance(self.estimate(variant, tokens, device), int)
                    self.assertIsInstance(self.estimate(variant, images, device), int)
            self.assertEqual(tokens, original)
            self.assertIs(images[variant[0]][0][0][0]["data"], image)

    def test_clip_load_model_forwards_estimate(self):
        for variant in self.variants:
            tokens = self.tokens(variant[0], 100)
            expected = self.estimate(variant, tokens)
            encoder = SimpleNamespace(memory_estimation_function=lambda values, device: self.estimate(variant, values, device))
            clip = SimpleNamespace(cond_stage_model=encoder, patcher=SimpleNamespace(load_device=torch.device("cuda:0"), is_dynamic=lambda: False))
            with patch.object(comfy.sd.model_management, "load_models_gpu") as load:
                comfy.sd.CLIP.load_model(clip, tokens)
            load.assert_called_once_with([clip.patcher], memory_required=expected)

    def test_shared_base_does_not_change_other_families(self):
        for cls in (qwen_image.QwenImageTEModel, hunyuan_image.HunyuanImageTEModel, kandinsky5.Kandinsky5TEModel, qwen3vl.Qwen3VLTEModel):
            self.assertFalse(hasattr(cls, "memory_estimation_function"))

    def assert_non_dynamic_clip(self, target):
        def init(model, **kwargs):
            torch.nn.Module.__init__(model)
            model.dtypes = set()

        target.tokenizer = Mock()
        with patch.object(sd1_clip.SD1ClipModel, "__init__", init), patch.object(model_patcher, "ModelPatcher") as legacy, patch.object(model_patcher, "CoreModelPatcher") as dynamic, patch.object(comfy.sd.model_management, "archive_model_dtypes"), patch.object(comfy.sd.model_management, "load_models_gpu"):
            clip = comfy.sd.CLIP(target, model_options={"load_device": torch.device("cpu"), "offload_device": torch.device("cpu"), "initial_device": torch.device("cpu"), "dtype": torch.float32})
        legacy.assert_called_once()
        dynamic.assert_not_called()
        self.assertTrue(clip.cond_stage_model.disable_offload)
        clone = clip.clone()
        self.assertIs(clone.cond_stage_model, clip.cond_stage_model)
        clip.patcher.clone.assert_called_once_with(disable_dynamic=False)

    def test_standalone_loader_selects_non_dynamic_only_for_qwen_image(self):
        for te_model, clip_type, expected in (
            (comfy.sd.TEModel.QWEN25_7B, comfy.sd.CLIPType.QWEN_IMAGE, True),
            (comfy.sd.TEModel.QWEN3VL_8B, comfy.sd.CLIPType.QWEN_IMAGE, True),
            (comfy.sd.TEModel.QWEN25_7B, comfy.sd.CLIPType.HUNYUAN_IMAGE, False),
            (comfy.sd.TEModel.QWEN25_7B, comfy.sd.CLIPType.LONGCAT_IMAGE, False),
            (comfy.sd.TEModel.QWEN3VL_8B, comfy.sd.CLIPType.JOYIMAGE, False),
            (comfy.sd.TEModel.QWEN3VL_8B, comfy.sd.CLIPType.STABLE_DIFFUSION, False),
        ):
            with self.subTest(te_model=te_model, clip_type=clip_type), patch.object(comfy.sd, "detect_te_model", return_value=te_model), patch.object(comfy.sd, "llama_detect", return_value={}), patch.object(comfy.sd, "CLIP") as clip:
                comfy.sd.load_text_encoder_state_dicts([{}], clip_type=clip_type)
                target = clip.call_args.args[0]
            self.assertEqual(hasattr(target.clip, "memory_estimation_function"), expected)
            if expected:
                self.assert_non_dynamic_clip(target)

    def test_checkpoint_targets_select_non_dynamic_patcher(self):
        for cls in (comfy.supported_models.QwenImage, comfy.supported_models.QwenImage21):
            config = cls.__new__(cls)
            with patch.object(comfy.sd.comfy.text_encoders.hunyuan_video, "llama_detect", return_value={}):
                target = config.clip_target({})
            self.assert_non_dynamic_clip(target)


if __name__ == "__main__":
    unittest.main()
