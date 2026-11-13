import unittest
import torch
import sys
import os
import json

# Add comfy to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

def has_gpu():
    return torch.cuda.is_available()

from comfy.cli_args import args
if not has_gpu():
    args.cpu = True

from comfy import ops
from comfy.quant_ops import QuantizedTensor
import comfy.utils


def marker_json(state_dict, key):
    """Decode a `<key>.comfy_quant` marker tensor back into its layer_conf dict."""
    return json.loads(state_dict[key].numpy().tobytes())


class SimpleModel(torch.nn.Module):
    """Mirrors tests-unit/comfy_quant/test_mixed_precision.py::SimpleModel."""

    def __init__(self, operations=ops.disable_weight_init):
        super().__init__()
        self.layer1 = operations.Linear(10, 20, device="cpu", dtype=torch.bfloat16)

    def forward(self, x):
        return self.layer1(x)


class TestConvertOldQuantsPrefixAware(unittest.TestCase):
    """Regression tests for GitHub #11864 / #13328: convert_old_quants()'s
    new-format (_quantization_metadata) branch must match the layer key
    convention actually used by the state_dict it is given, regardless of
    whether that convention is prefixed or already stripped, and regardless
    of whether model_prefix is stripped/added/empty at the call site.
    """

    # ---- scenario 1: metadata key and sd key both carry the prefix (aligned) ----
    def test_scenario1_prefixed_metadata_matches_prefixed_sd(self):
        layer_quant_config = {"model.diffusion_model.proj_in": {"format": "float8_e4m3fn"}}
        state_dict = {
            "model.diffusion_model.proj_in.weight": torch.randn(4, 4, dtype=torch.float32).to(torch.float8_e4m3fn),
            "model.diffusion_model.proj_in.weight_scale": torch.tensor(1.0),
            "model.diffusion_model.other.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        }
        out_sd, _ = comfy.utils.convert_old_quants(
            state_dict,
            model_prefix="model.diffusion_model.",
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )
        self.assertIn("model.diffusion_model.proj_in.comfy_quant", out_sd)
        self.assertNotIn("proj_in.comfy_quant", out_sd)
        self.assertEqual(marker_json(out_sd, "model.diffusion_model.proj_in.comfy_quant")["format"], "float8_e4m3fn")

    # ---- scenario 2: sd already stripped of prefix, metadata key still prefixed (the bug) ----
    def test_scenario2_prefixed_metadata_matches_stripped_sd_after_fix(self):
        layer_quant_config = {"model.diffusion_model.proj_in": {"format": "float8_e4m3fn"}}
        state_dict = {
            "proj_in.weight": torch.randn(4, 4, dtype=torch.float32).to(torch.float8_e4m3fn),
            "proj_in.weight_scale": torch.tensor(1.0),
            "other.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        }
        out_sd, _ = comfy.utils.convert_old_quants(
            state_dict,
            model_prefix="model.diffusion_model.",
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )
        # Before the fix this would have blindly written
        # "model.diffusion_model.proj_in.comfy_quant", which never matches
        # "proj_in.weight" -> detect_layer_quantization()/MixedPrecisionOps
        # would find no marker and load the layer as a plain dtype tensor.
        self.assertIn("proj_in.comfy_quant", out_sd)
        self.assertNotIn("model.diffusion_model.proj_in.comfy_quant", out_sd)
        self.assertEqual(marker_json(out_sd, "proj_in.comfy_quant")["format"], "float8_e4m3fn")

    # ---- scenario 3: metadata key already stripped, sd already stripped (today's working path) ----
    def test_scenario3_stripped_metadata_matches_stripped_sd_unchanged(self):
        layer_quant_config = {"proj_in": {"format": "float8_e4m3fn"}}
        state_dict = {
            "proj_in.weight": torch.randn(4, 4, dtype=torch.float32).to(torch.float8_e4m3fn),
            "proj_in.weight_scale": torch.tensor(1.0),
        }
        out_sd, _ = comfy.utils.convert_old_quants(
            state_dict,
            model_prefix="model.diffusion_model.",
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )
        # Zero behavior change vs. today: direct match succeeds immediately,
        # key is written exactly as it always was.
        self.assertIn("proj_in.comfy_quant", out_sd)
        self.assertEqual(marker_json(out_sd, "proj_in.comfy_quant")["format"], "float8_e4m3fn")

    # Same as scenario 3 but with model_prefix="" (the literal value passed
    # by comfy/sd.py::load_diffusion_model_state_dict at both call sites).
    def test_scenario3b_stripped_metadata_empty_model_prefix_unchanged(self):
        layer_quant_config = {"proj_in": {"format": "nvfp4"}}
        state_dict = {
            "proj_in.weight": torch.randint(0, 255, (4, 2), dtype=torch.uint8),
            "proj_in.weight_scale": torch.tensor(1.0),
            "proj_in.weight_scale_2": torch.tensor(1.0),
        }
        out_sd, _ = comfy.utils.convert_old_quants(
            state_dict,
            model_prefix="",
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )
        self.assertIn("proj_in.comfy_quant", out_sd)

    # ---- scenario 4: legacy scaled_fp8 branch must be completely unaffected ----
    def test_scenario4_legacy_scaled_fp8_branch_unaffected(self):
        state_dict = {
            "model.diffusion_model.scaled_fp8": torch.tensor([0.0], dtype=torch.float32),
            "model.diffusion_model.proj_in.weight": torch.randn(4, 4, dtype=torch.float32).to(torch.float8_e4m3fn),
            "model.diffusion_model.proj_in.scale_weight": torch.tensor(2.0),
            "model.diffusion_model.other.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        }
        out_sd, metadata = comfy.utils.convert_old_quants(
            state_dict,
            model_prefix="model.diffusion_model.",
            metadata={},
        )
        # Old-format branch derives layer keys straight from state_dict's own
        # (already correctly prefixed) keys, so resolution is a same-key
        # direct match every time -- this path must be byte-for-byte identical
        # to pre-fix behavior.
        self.assertNotIn("model.diffusion_model.scaled_fp8", out_sd)
        self.assertIn("model.diffusion_model.proj_in.weight_scale", out_sd)
        self.assertIn("model.diffusion_model.proj_in.comfy_quant", out_sd)
        self.assertEqual(
            marker_json(out_sd, "model.diffusion_model.proj_in.comfy_quant")["format"],
            "float8_e4m3fn",
        )
        self.assertNotIn("proj_in.comfy_quant", out_sd)  # not stripped/mismatched

    # ---- extra: idempotency across the exact two-call pattern comfy/sd.py uses ----
    def test_two_call_pattern_mirrors_load_diffusion_model_state_dict(self):
        """Simulates comfy/sd.py::load_diffusion_model_state_dict() verbatim:
        convert_old_quants(sd, "", metadata=metadata) is called once before
        the diffusion_model_prefix strip and once after, both with an empty
        model_prefix. This must work for BOTH metadata conventions without
        any change to the call site (see PR #13328, closed for reordering
        the calls and breaking the other convention instead)."""
        for convention, layer_key in (
            ("prefixed", "model.diffusion_model.proj_in"),
            ("stripped", "proj_in"),
        ):
            with self.subTest(convention=convention):
                metadata = {"_quantization_metadata": json.dumps(
                    {"layers": {layer_key: {"format": "float8_e4m3fn"}}}
                )}
                sd = {
                    "model.diffusion_model.proj_in.weight": torch.randn(4, 4, dtype=torch.float32).to(torch.float8_e4m3fn),
                    "model.diffusion_model.proj_in.weight_scale": torch.tensor(1.0),
                    "unrelated.top_level.weight": torch.randn(2, 2, dtype=torch.bfloat16),
                }

                # call 1: before stripping, model_prefix="" (as sd.py does)
                sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)

                # simulate state_dict_prefix_replace(sd, {prefix: ""}, filter_keys=True)
                prefix = "model.diffusion_model."
                temp_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
                self.assertGreater(len(temp_sd), 0)
                sd = temp_sd

                # call 2: after stripping, model_prefix="" again (as sd.py does)
                sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)

                self.assertIn("proj_in.comfy_quant", sd,
                               f"{convention} metadata convention did not resolve after the two-call dance")
                self.assertNotIn("model.diffusion_model.proj_in.comfy_quant", sd)
                self.assertEqual(marker_json(sd, "proj_in.comfy_quant")["format"], "float8_e4m3fn")

    # ---- extra: repeated calls with identical inputs don't duplicate/clobber ----
    def test_marker_write_is_idempotent(self):
        layer_quant_config = {"proj_in": {"format": "float8_e4m3fn"}}
        metadata = {"_quantization_metadata": json.dumps({"layers": layer_quant_config})}
        state_dict = {
            "proj_in.weight": torch.randn(4, 4, dtype=torch.float32).to(torch.float8_e4m3fn),
            "proj_in.weight_scale": torch.tensor(1.0),
        }
        out_sd1, _ = comfy.utils.convert_old_quants(dict(state_dict), model_prefix="", metadata=dict(metadata))
        keys_before = set(out_sd1.keys())
        original_marker = out_sd1["proj_in.comfy_quant"]
        marker_before = original_marker.clone()
        out_sd2, _ = comfy.utils.convert_old_quants(dict(out_sd1), model_prefix="", metadata=dict(metadata))
        self.assertEqual(keys_before, set(out_sd2.keys()))
        self.assertTrue(torch.equal(marker_before, out_sd2["proj_in.comfy_quant"]))
        self.assertIs(original_marker, out_sd2["proj_in.comfy_quant"])

    def test_conflicting_marker_is_replaced_with_current_metadata(self):
        layer_quant_config = {"proj_in": {"format": "float8_e4m3fn"}}
        old_marker = torch.tensor(list(json.dumps({"format": "nvfp4"}).encode("utf-8")), dtype=torch.uint8)
        state_dict = {
            "proj_in.weight": torch.randn(4, 4, dtype=torch.float32).to(torch.float8_e4m3fn),
            "proj_in.weight_scale": torch.tensor(1.0),
            "proj_in.comfy_quant": old_marker,
        }
        out_sd, _ = comfy.utils.convert_old_quants(
            state_dict,
            model_prefix="",
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )
        self.assertEqual(marker_json(out_sd, "proj_in.comfy_quant"), {"format": "float8_e4m3fn"})
        self.assertIsNot(old_marker, out_sd["proj_in.comfy_quant"])

    # ---- extra: functional end-to-end, proving the fixed layer actually loads as QuantizedTensor ----
    def test_functional_load_after_prefix_mismatch_fix(self):
        layer_quant_config = {"model.diffusion_model.layer1": {"format": "float8_e4m3fn"}}
        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        # sd already stripped of "model.diffusion_model." (as it is by the
        # time comfy/sd.py's second convert_old_quants call runs), while
        # metadata still carries the full prefix.
        state_dict = {
            "layer1.weight": fp8_weight,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(2.0, dtype=torch.float32),
        }
        state_dict, _ = comfy.utils.convert_old_quants(
            state_dict,
            model_prefix="model.diffusion_model.",
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )
        model = SimpleModel(operations=ops.mixed_precision_ops({}))
        model.load_state_dict(state_dict, strict=False)

        self.assertIsInstance(model.layer1.weight, QuantizedTensor)
        self.assertEqual(model.layer1.weight._layout_cls, "TensorCoreFP8E4M3Layout")
        self.assertEqual(model.layer1.weight._params.scale.item(), 2.0)


if __name__ == "__main__":
    unittest.main()
