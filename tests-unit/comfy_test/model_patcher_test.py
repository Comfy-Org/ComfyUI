import tempfile
import unittest
from unittest import mock

import safetensors.torch
import torch

import comfy.sd
from comfy.model_patcher import (
    LazyCastingParam,
    LazyCastingParamPiece,
    LazyCastingQuantizedParam,
)


class FakeModelPatcher:
    def __init__(self, weight):
        self.load_device = torch.device("cpu")
        self.weight = weight
        self.calls = 0

    def patch_weight_to_device(self, key, device_to=None, return_weight=False):
        self.calls += 1
        return self.weight


class FakeQuantizedWeight:
    def __init__(self, state_dict):
        self.state_dict_values = state_dict

    def state_dict(self, key):
        return self.state_dict_values


class FakeSaveModel:
    def __init__(self, state_dict):
        self.state_dict = state_dict

    def state_dict_for_saving(self, clip_state_dict, vae_state_dict, clip_vision_state_dict):
        return self.state_dict


class TestLazyCastingParam(unittest.TestCase):
    def test_safetensors_ignores_shared_noncontiguous_backing_storage(self):
        storage = torch.arange(16, dtype=torch.float32)
        first = storage[:12].reshape(3, 4).T
        second = storage[4:].reshape(3, 4).T
        first_patcher = FakeModelPatcher(torch.full(first.shape, 1.0))
        second_patcher = FakeModelPatcher(torch.full(second.shape, 2.0))

        serialized = safetensors.torch.save(
            {
                "first": LazyCastingParam(first_patcher, "first", first),
                "second": LazyCastingParam(second_patcher, "second", second),
            }
        )
        actual = safetensors.torch.load(serialized)

        torch.testing.assert_close(actual["first"], first_patcher.weight)
        torch.testing.assert_close(actual["second"], second_patcher.weight)
        self.assertEqual(first_patcher.calls, 1)
        self.assertEqual(second_patcher.calls, 1)

    def test_quantized_pieces_share_one_materialization(self):
        original_data = torch.arange(12, dtype=torch.int8).reshape(3, 4).T
        original_scale = torch.tensor(1.0)
        patched_data = torch.full(original_data.shape, 7, dtype=torch.int8)
        patched_scale = torch.tensor(0.25)
        patched_weight = FakeQuantizedWeight(
            {
                "linear.weight": patched_data,
                "linear.weight_scale": patched_scale,
            }
        )
        patcher = FakeModelPatcher(patched_weight)
        caster = LazyCastingQuantizedParam(patcher, "linear.weight")
        lazy_data = LazyCastingParamPiece(caster, "linear.weight", original_data)
        lazy_scale = LazyCastingParamPiece(
            caster, "linear.weight_scale", original_scale
        )

        serialized = safetensors.torch.save(
            {
                "linear.weight": lazy_data,
                "linear.weight_scale": lazy_scale,
            }
        )
        actual = safetensors.torch.load(serialized)

        self.assertFalse(original_data.is_contiguous())
        self.assertFalse(lazy_data.requires_grad)
        self.assertEqual(lazy_data.device, torch.device("meta"))
        torch.testing.assert_close(actual["linear.weight"], patched_data)
        torch.testing.assert_close(actual["linear.weight_scale"], patched_scale)
        self.assertEqual(patcher.calls, 1)

        converted_data = torch.full((3, 4), 9, dtype=torch.int8).T
        conversion_weight = FakeQuantizedWeight({"linear.weight": converted_data})
        conversion_patcher = FakeModelPatcher(conversion_weight)
        conversion_caster = LazyCastingQuantizedParam(
            conversion_patcher, "linear.weight"
        )
        lazy_conversion = LazyCastingParamPiece(
            conversion_caster, "linear.weight", original_data
        )

        converted = lazy_conversion.to(dtype=torch.float32, copy=True)

        self.assertFalse(converted_data.is_contiguous())
        self.assertEqual(converted.dtype, torch.float32)
        self.assertTrue(converted.is_contiguous())
        torch.testing.assert_close(converted, converted_data.to(torch.float32))
        self.assertEqual(conversion_patcher.calls, 1)

    def test_to_applies_requested_dtype_after_materialization(self):
        original = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
        patched = torch.full(original.shape, 42.0)
        patcher = FakeModelPatcher(patched)
        lazy_weight = LazyCastingParam(patcher, "linear.weight", original)

        actual = lazy_weight.to(dtype=torch.float64, copy=True)

        torch.testing.assert_close(actual, patched.to(torch.float64))
        self.assertEqual(actual.dtype, torch.float64)
        self.assertEqual(patcher.calls, 1)

    def test_rejects_materialized_metadata_mismatch(self):
        original = torch.zeros(2, 3)
        mismatches = (torch.ones(3, 2), torch.ones(2, 3, dtype=torch.float64))

        for patched in mismatches:
            with self.subTest(shape=patched.shape, dtype=patched.dtype):
                patcher = FakeModelPatcher(patched)
                lazy_weight = LazyCastingParam(patcher, "linear.weight", original)

                with self.assertRaisesRegex(RuntimeError, "metadata does not match"):
                    lazy_weight.to("cpu")

    def test_save_checkpoint_materializes_noncontiguous_weight(self):
        original = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
        patched = torch.full(original.shape, 42.0)
        patcher = FakeModelPatcher(patched)
        lazy_weight = LazyCastingParam(patcher, "linear.weight", original)
        model = FakeSaveModel({"linear.weight": lazy_weight})

        self.assertFalse(original.is_contiguous())
        self.assertEqual(lazy_weight.device, torch.device("meta"))
        self.assertEqual(lazy_weight.untyped_storage().data_ptr(), 0)
        self.assertTrue(lazy_weight.is_contiguous())
        self.assertEqual(patcher.calls, 0)

        with tempfile.TemporaryDirectory() as directory:
            output = f"{directory}/model.safetensors"
            with mock.patch.object(comfy.sd.model_management, "load_models_gpu"):
                comfy.sd.save_checkpoint(output, model)
            with open(output, "rb") as checkpoint:
                serialized = checkpoint.read()

        actual = safetensors.torch.load(serialized)["linear.weight"]

        torch.testing.assert_close(actual, patched)
        self.assertEqual(patcher.calls, 1)


if __name__ == "__main__":
    unittest.main()
