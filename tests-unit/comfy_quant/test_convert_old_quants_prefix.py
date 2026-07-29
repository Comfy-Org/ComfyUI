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
import comfy.model_detection
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


class TestResidualPrefixPoisoning(unittest.TestCase):
    def test_real_world_aligned_prefixes_remain_unchanged(self):
        for prefix in ("", "model.", "model.diffusion_model."):
            with self.subTest(prefix=prefix):
                layer = f"{prefix}block"
                sd = {f"{layer}.weight": torch.empty(1)}
                metadata = {
                    "_quantization_metadata": json.dumps({
                        "layers": {layer: {"format": "nvfp4"}}
                    })
                }

                sd, _ = comfy.utils.convert_old_quants(sd, "", metadata=metadata)

                self.assertIn(f"{layer}.comfy_quant", sd)

    def test_prefixed_metadata_with_unprefixed_weights_survives_loader_flow(self):
        for metadata_prefix in ("model.diffusion_model.", "model.model.", "model.", "net."):
            for num_layers in (1, 5, 6, 10):
                with self.subTest(metadata_prefix=metadata_prefix, num_layers=num_layers):
                    sd = {}
                    layers = {}
                    for i in range(num_layers):
                        sd[f"block{i}.weight"] = torch.randn(4, 4, dtype=torch.float32).to(torch.float8_e4m3fn)
                        sd[f"block{i}.weight_scale"] = torch.tensor(1.0)
                        layers[f"{metadata_prefix}block{i}"] = {"format": "float8_e4m3fn"}
                    metadata = {"_quantization_metadata": json.dumps({"layers": layers})}

                    sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)
                    prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
                    temp_sd = comfy.utils.state_dict_prefix_replace(sd, {prefix: ""}, filter_keys=True)
                    if temp_sd:
                        sd = temp_sd
                        sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)

                    self.assertEqual(sum(k.endswith(".weight") for k in sd), num_layers)
                    for i in range(num_layers):
                        self.assertIn(f"block{i}.comfy_quant", sd)
                        self.assertNotIn(f"{metadata_prefix}block{i}.comfy_quant", sd)

    def test_unique_weight_suffix_resolves_without_model_prefix(self):
        sd = {"block.weight": torch.empty(1)}
        metadata = {
            "_quantization_metadata": json.dumps({
                "layers": {"model.diffusion_model.block": {"format": "nvfp4"}}
            })
        }

        sd, _ = comfy.utils.convert_old_quants(sd, "", metadata=metadata)

        self.assertIn("block.comfy_quant", sd)
        self.assertNotIn("model.diffusion_model.block.comfy_quant", sd)

    def test_known_wrapper_does_not_overwrite_shorter_layer(self):
        stale_marker = torch.tensor(
            list(json.dumps({"format": "nvfp4"}).encode("utf-8")),
            dtype=torch.uint8,
        )
        sd = {
            "foo.bar.weight": torch.empty(1),
            "bar.weight": torch.empty(1),
            "bar.comfy_quant": stale_marker,
        }
        metadata = {
            "_quantization_metadata": json.dumps({
                "layers": {"model.foo.bar": {"format": "float8_e4m3fn"}}
            })
        }

        sd, _ = comfy.utils.convert_old_quants(sd, "", metadata=metadata)

        self.assertIs(sd["bar.comfy_quant"], stale_marker)
        self.assertEqual(marker_json(sd, "foo.bar.comfy_quant"), {"format": "float8_e4m3fn"})
        self.assertNotIn("model.foo.bar.comfy_quant", sd)

    def test_multiple_metadata_layers_cannot_collapse_to_one_weight(self):
        sd = {"block.weight": torch.empty(1)}
        metadata = {
            "_quantization_metadata": json.dumps({
                "layers": {
                    "model.a.block": {"format": "nvfp4"},
                    "model.b.block": {"format": "float8_e4m3fn"},
                }
            })
        }

        sd, _ = comfy.utils.convert_old_quants(sd, "", metadata=metadata)

        self.assertFalse(any(k.endswith(".comfy_quant") for k in sd))

    def test_direct_and_known_wrapper_layers_can_coexist(self):
        sd = {
            "direct.weight": torch.empty(1),
            "wrapped.weight": torch.empty(1),
        }
        metadata = {
            "_quantization_metadata": json.dumps({
                "layers": {
                    "direct": {"format": "nvfp4"},
                    "model.diffusion_model.wrapped": {"format": "float8_e4m3fn"},
                }
            })
        }

        sd, _ = comfy.utils.convert_old_quants(sd, "", metadata=metadata)

        self.assertIn("direct.comfy_quant", sd)
        self.assertIn("wrapped.comfy_quant", sd)

    def test_direct_and_wrapper_collision_is_rejected(self):
        sd = {"block.weight": torch.empty(1)}
        metadata = {
            "_quantization_metadata": json.dumps({
                "layers": {
                    "block": {"format": "nvfp4"},
                    "model.diffusion_model.block": {"format": "float8_e4m3fn"},
                }
            })
        }

        sd, _ = comfy.utils.convert_old_quants(sd, "", metadata=metadata)

        self.assertNotIn("block.comfy_quant", sd)

    def test_direct_and_model_prefix_targets_are_not_guessed(self):
        sd = {
            "block.weight": torch.empty(1),
            "model.diffusion_model.block.weight": torch.empty(1),
        }
        metadata = {
            "_quantization_metadata": json.dumps({
                "layers": {"block": {"format": "nvfp4"}}
            })
        }

        sd, _ = comfy.utils.convert_old_quants(
            sd,
            "model.diffusion_model.",
            metadata=metadata,
        )

        self.assertFalse(any(k.endswith(".comfy_quant") for k in sd))

    def test_equivalent_full_and_stripped_metadata_aliases_are_coalesced(self):
        prefix = "model.diffusion_model."
        layer_config = {"format": "nvfp4"}
        sd = {f"{prefix}block.weight": torch.empty(1)}
        metadata = {
            "_quantization_metadata": json.dumps({
                "layers": {
                    "block": layer_config,
                    f"{prefix}block": layer_config,
                }
            })
        }

        sd, _ = comfy.utils.convert_old_quants(sd, prefix, metadata=metadata)

        self.assertEqual(marker_json(sd, f"{prefix}block.comfy_quant"), layer_config)

    def test_unmatched_metadata_does_not_create_orphan_marker(self):
        sd = {"present.weight": torch.empty(1)}
        metadata = {
            "_quantization_metadata": json.dumps({
                "layers": {"model.diffusion_model.missing": {"format": "nvfp4"}}
            })
        }

        sd, _ = comfy.utils.convert_old_quants(sd, "", metadata=metadata)

        self.assertEqual(set(sd), {"present.weight"})

    def test_partial_payload_does_not_accept_one_accidental_metadata_match(self):
        sd = {
            "vocoder.weight": torch.empty(1),
            "block7.weight": torch.empty(1),
        }
        layers = {
            f"model.diffusion_model.block{i}": {"format": "nvfp4"}
            for i in range(8)
        }
        metadata = {"_quantization_metadata": json.dumps({"layers": layers})}

        sd, _ = comfy.utils.convert_old_quants(sd, "", metadata=metadata)

        self.assertFalse(any(k.endswith(".comfy_quant") for k in sd))

    def test_existing_marker_without_file_metadata_is_unchanged(self):
        marker = torch.tensor(
            list(json.dumps({"format": "nvfp4"}).encode("utf-8")),
            dtype=torch.uint8,
        )
        sd = {
            "block.weight": torch.empty(1),
            "block.comfy_quant": marker,
        }

        sd, _ = comfy.utils.convert_old_quants(sd, "", metadata={})

        self.assertIs(sd["block.comfy_quant"], marker)

    def test_unprefixed_legacy_scaled_fp8_is_unchanged(self):
        sd = {
            "scaled_fp8": torch.tensor([0.0], dtype=torch.float32),
            "block.weight": torch.randn(4, 4, dtype=torch.float32).to(torch.float8_e4m3fn),
            "block.scale_weight": torch.tensor(2.0),
        }

        sd, _ = comfy.utils.convert_old_quants(sd, "", metadata={})

        self.assertNotIn("scaled_fp8", sd)
        self.assertIn("block.weight_scale", sd)
        self.assertEqual(marker_json(sd, "block.comfy_quant"), {"format": "float8_e4m3fn"})


if __name__ == "__main__":
    unittest.main()
