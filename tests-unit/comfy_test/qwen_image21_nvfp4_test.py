import json
import unittest
from unittest.mock import patch

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy import ops, sd1_clip
from comfy.text_encoders.qwen_image21 import QwenImage21Qwen3VLClipModel


class TestQwenImage21NVFP4(unittest.TestCase):
    def setUp(self):
        self.model = QwenImage21Qwen3VLClipModel.__new__(QwenImage21Qwen3VLClipModel)
        torch.nn.Module.__init__(self.model)
        self.model.nvfp4_conditioning = True
        self.model.execution_device = torch.device("cuda")
        operations = ops.mixed_precision_ops({}, torch.bfloat16, full_precision_mm=True)
        self.model.transformer = torch.nn.Module()
        self.model.transformer.model = torch.nn.Sequential(operations.Linear(32, 32), operations.Linear(32, 32))
        self.model.transformer.visual = operations.Linear(32, 32)
        for layer in [*self.model.transformer.model, self.model.transformer.visual]:
            layer.quant_format = "nvfp4"
            layer._full_precision_mm_config = False
        self.model.transformer.model[1]._full_precision_mm_config = True

    def test_detection_only_enables_nvfp4_language_layers(self):
        for key, quant_format, expected in [
            ("model.layers.0.self_attn.q_proj.comfy_quant", "nvfp4", True),
            ("model.layers.0.self_attn.q_proj.comfy_quant", "int8_tensorwise", False),
            ("model.layers.0.self_attn.q_proj.comfy_quant", "float8_e4m3fn", False),
            ("visual.blocks.0.attn.qkv.comfy_quant", "nvfp4", False),
        ]:
            with self.subTest(key=key, quant_format=quant_format):
                metadata = torch.tensor(list(json.dumps({"format": quant_format}).encode()), dtype=torch.uint8)
                state_dict = {key: metadata}
                with patch.object(sd1_clip.SDClipModel, "load_sd", return_value="loaded") as load:
                    self.assertEqual(self.model.load_sd(state_dict), "loaded")
                    load.assert_called_once_with(state_dict)
                self.assertEqual(self.model.nvfp4_conditioning, expected)
        with patch.object(sd1_clip.SDClipModel, "load_sd"):
            metadata = torch.tensor(list(json.dumps({"format": "nvfp4", "full_precision_matrix_mult": True}).encode()), dtype=torch.uint8)
            self.model.load_sd({"model.layers.0.self_attn.q_proj.comfy_quant": metadata})
            self.assertFalse(self.model.nvfp4_conditioning)
            self.model.load_sd({})
        self.assertFalse(self.model.nvfp4_conditioning)

    def test_embeddings_preserve_masks_and_image_spans(self):
        embeds = torch.randn(1, 4, 32)
        mask = torch.ones(1, 4, dtype=torch.long)
        counts = [4]
        info = [{"type": "image", "index": 1, "size": 2}]
        for nvfp4, device, supported, expected_dtype in [
            (True, "cuda", True, torch.bfloat16),
            (True, "cuda", False, torch.float32),
            (True, "cpu", True, torch.float32),
            (False, "cuda", True, torch.float32),
        ]:
            with self.subTest(nvfp4=nvfp4, device=device, supported=supported):
                self.model.nvfp4_conditioning = nvfp4
                with patch.object(sd1_clip.SDClipModel, "process_tokens", return_value=(embeds, mask, counts, info)), patch("comfy.model_management.supports_nvfp4_compute", return_value=supported) as supports:
                    result = self.model.process_tokens([], torch.device(device))
                self.assertEqual(result[0].dtype, expected_dtype)
                torch.testing.assert_close(result[0], embeds.to(expected_dtype))
                self.assertIs(result[1], mask)
                self.assertIs(result[2], counts)
                self.assertIs(result[3], info)
                self.assertEqual(self.model.image_spans, [(1, 2)])
                if device == "cpu":
                    supports.assert_not_called()

    def test_context_excludes_vision_and_restores_on_error(self):
        for fail in [False, True]:
            with self.subTest(fail=fail):
                def forward(tokens):
                    self.assertFalse(self.model.transformer.model[0]._full_precision_mm)
                    self.assertTrue(self.model.transformer.model[1]._full_precision_mm)
                    self.assertTrue(self.model.transformer.visual._full_precision_mm)
                    if fail:
                        raise RuntimeError("encoding failed")
                    return tokens

                with patch("comfy.model_management.supports_nvfp4_compute", return_value=True), patch("comfy.ops.get_disabled_quant_formats", return_value=set()), patch.object(sd1_clip.SDClipModel, "forward", side_effect=forward):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "encoding failed"):
                            self.model.forward([])
                    else:
                        tokens = [[1, 2]]
                        self.assertIs(self.model.forward(tokens), tokens)
                self.assertTrue(self.model.transformer.model[0]._full_precision_mm)

    def test_unsupported_and_unquantized_forward_use_original_path(self):
        for nvfp4, device, supported in [(True, "cpu", True), (True, "cuda", False), (False, "cuda", True)]:
            with self.subTest(nvfp4=nvfp4, device=device, supported=supported):
                self.model.nvfp4_conditioning = nvfp4
                self.model.execution_device = torch.device(device)
                with patch("comfy.model_management.supports_nvfp4_compute", return_value=supported) as supports, patch("comfy.ops.use_quantized_matmul") as context, patch.object(sd1_clip.SDClipModel, "forward", return_value="original"):
                    self.assertEqual(self.model.forward([]), "original")
                context.assert_not_called()
                if device == "cpu":
                    supports.assert_not_called()
