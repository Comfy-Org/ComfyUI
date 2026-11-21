"""FP8 quantize/dequantize on MPS.

The MPS backend has no float8 kernels at all: any cast touching torch.float8_e4m3fn
or torch.float8_e5m2 raises "RuntimeError: Undefined type Float8_e4m3fn". The comfy
layouts have to work around that, so check the workaround stays bit exact against the
CPU implementation.
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from comfy.cli_args import args
if not torch.cuda.is_available():
    args.cpu = True

import comfy.model_patcher  # noqa: E402
from comfy import ops
from comfy.quant_ops import (  # noqa: E402
    QuantizedTensor,
    TensorCoreFP8E4M3Layout,
    TensorCoreFP8E5M2Layout,
)

MPS_AVAILABLE = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
# layout class, registered layout name, fp8 dtype, mantissa bits
FP8_LAYOUTS = ((TensorCoreFP8E4M3Layout, "TensorCoreFP8E4M3Layout", torch.float8_e4m3fn, 3),
               (TensorCoreFP8E5M2Layout, "TensorCoreFP8E5M2Layout", torch.float8_e5m2, 2))


@unittest.skipUnless(MPS_AVAILABLE, "requires MPS")
class TestFP8OnMPS(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # CastWeightBiasOp.weight_function / bias_function are class level mutable
        # defaults, so any test that appends to a layer without its own list leaks
        # that function into every layer in the process. Start from a clean slate.
        ops.CastWeightBiasOp.weight_function = []
        ops.CastWeightBiasOp.bias_function = []

    def test_dequantize_matches_cpu(self):
        for layout, _name, fp8_dtype, _mbits in FP8_LAYOUTS:
            for out_dtype in (torch.float16, torch.bfloat16, torch.float32):
                with self.subTest(dtype=fp8_dtype, out=out_dtype):
                    qdata = torch.arange(256, dtype=torch.uint8).view(fp8_dtype).reshape(16, 16)
                    params = layout.Params(scale=torch.tensor(0.25), orig_dtype=out_dtype,
                                           orig_shape=tuple(qdata.shape))
                    expected = layout.dequantize(qdata, params)

                    mps_params = layout.Params(scale=params.scale.to("mps"), orig_dtype=out_dtype,
                                               orig_shape=tuple(qdata.shape))
                    got = layout.dequantize(qdata.to("mps"), mps_params)

                    self.assertEqual(got.device.type, "mps")
                    self.assertEqual(got.dtype, expected.dtype)
                    # NaN encodings must stay NaN, everything else bit identical.
                    self.assertTrue(torch.equal(got.cpu().isnan(), expected.isnan()))
                    finite = ~expected.isnan()
                    self.assertTrue(torch.equal(got.cpu()[finite], expected[finite]))

    def test_quantized_tensor_dequantize_roundtrip(self):
        x = torch.randn(64, 128, dtype=torch.float16) * 3.0
        for _layout, name, fp8_dtype, _mbits in FP8_LAYOUTS:
            with self.subTest(dtype=fp8_dtype):
                qt = QuantizedTensor.from_float(x, name, scale="recalculate")
                expected = qt.dequantize()

                qt_mps = QuantizedTensor.from_float(x.to("mps"), name, scale="recalculate")
                got = qt_mps.dequantize()

                self.assertEqual(got.device.type, "mps")
                self.assertEqual(got.shape, expected.shape)
                torch.testing.assert_close(got.cpu().float(), expected.float(), rtol=0, atol=0)

    def test_quantize_matches_cpu(self):
        x = torch.randn(32, 64, dtype=torch.float16) * 3.0
        for layout, _name, fp8_dtype, _mbits in FP8_LAYOUTS:
            with self.subTest(dtype=fp8_dtype):
                expected_qdata, expected_params = layout.quantize(x.clone(), scale="recalculate")
                got_qdata, got_params = layout.quantize(x.to("mps"), scale="recalculate")

                self.assertEqual(got_qdata.device.type, "mps")
                self.assertEqual(got_qdata.dtype, fp8_dtype)
                self.assertEqual(got_params.scale.device.type, "mps")
                self.assertTrue(torch.equal(got_qdata.cpu().view(torch.uint8),
                                            expected_qdata.view(torch.uint8)))
                torch.testing.assert_close(got_params.scale.cpu(), expected_params.scale)

    def test_stochastic_quantize_matches_cpu(self):
        """LoRA patching requantizes with stochastic rounding; same seed must give same bytes."""
        x = torch.randn(32, 64, dtype=torch.float16)
        for layout, _name, fp8_dtype, mbits in FP8_LAYOUTS:
            with self.subTest(dtype=fp8_dtype):
                expected_qdata, _ = layout.quantize(x.clone(), scale="recalculate", stochastic_rounding=1234)
                qdata, params = layout.quantize(x.to("mps"), scale="recalculate", stochastic_rounding=1234)

                self.assertEqual(qdata.device.type, "mps")
                self.assertEqual(qdata.dtype, fp8_dtype)
                self.assertTrue(torch.equal(qdata.cpu().view(torch.uint8), expected_qdata.view(torch.uint8)))

                dq = layout.dequantize(qdata, params)
                self.assertFalse(dq.isnan().any())
                # Stochastic rounding lands on one of the two neighbouring grid points,
                # so the error is bounded by one mantissa step.
                torch.testing.assert_close(dq.cpu().float(), x.float(),
                                           rtol=2.0 ** -mbits, atol=float(params.scale) * 2.0)

    def test_linear_forward(self):
        """Full mixed precision Linear on MPS: weights stay fp8, forward dequantizes."""
        quant_config = {"layer": {"format": "float8_e4m3fn"}}
        operations = ops.mixed_precision_ops(
            quant_config, torch.float16, disabled={"float8_e4m3fn", "float8_e5m2"}
        )
        layer = operations.Linear(64, 32, bias=False, device="mps", dtype=torch.float16)

        weight = torch.randn(32, 64, dtype=torch.float16)
        qdata, params = TensorCoreFP8E4M3Layout.quantize(weight.clone(), scale="recalculate")
        sd = {
            "weight": qdata,
            "weight_scale": params.scale,
            "comfy_quant": torch.tensor(
                list(b'{"format": "float8_e4m3fn"}'), dtype=torch.uint8),
        }
        layer.load_state_dict(sd, strict=False, assign=True)

        x = torch.randn(4, 64, dtype=torch.float16, device="mps")
        out = layer(x)
        self.assertEqual(out.shape, (4, 32))
        self.assertFalse(out.isnan().any())

        # Reference matmul on the same device, so the MPS/CPU fp16 accumulation
        # difference stays out of it: the forward must be exactly dequantize + linear.
        mps_params = TensorCoreFP8E4M3Layout.Params(scale=params.scale.to("mps"),
                                                    orig_dtype=params.orig_dtype,
                                                    orig_shape=params.orig_shape)
        weight_dq = TensorCoreFP8E4M3Layout.dequantize(qdata.to("mps"), mps_params)
        expected = torch.nn.functional.linear(x, weight_dq)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


    def test_lora_patch_on_mps(self):
        """The reported crash: patching a LoRA onto fp8 weights dequantizes them on device."""
        model = _fp8_linear_model()
        patcher = comfy.model_patcher.ModelPatcher(model, load_device=torch.device("mps"),
                                                   offload_device=torch.device("cpu"))
        diff = torch.full((32, 64), 0.01, dtype=torch.float16)
        patcher.add_patches({"layer.weight": (diff,)}, strength_patch=1.0)

        before = model.layer.weight.dequantize().float()
        patcher.patch_weight_to_device("layer.weight", device_to=torch.device("mps"))
        after = model.layer.weight.dequantize().float()

        self.assertIsInstance(model.layer.weight.data, QuantizedTensor)
        self.assertEqual(model.layer.weight._qdata.dtype, torch.float8_e4m3fn)
        self.assertEqual(model.layer.weight.device.type, "mps")
        # Requantization is lossy, so just check the patch moved the weights the right way.
        self.assertGreater((after - before.to(after.device)).mean().item(), 0.0)


def _fp8_linear_model():
    quant_config = {"layer": {"format": "float8_e4m3fn"}}
    operations = ops.mixed_precision_ops(
        quant_config, torch.float16, disabled={"float8_e4m3fn", "float8_e5m2"}
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = operations.Linear(64, 32, bias=False, device="mps", dtype=torch.float16)

    model = Model()
    weight = torch.randn(32, 64, dtype=torch.float16)
    qdata, params = TensorCoreFP8E4M3Layout.quantize(weight, scale="recalculate")
    model.load_state_dict({
        "layer.weight": qdata,
        "layer.weight_scale": params.scale,
        "layer.comfy_quant": torch.tensor(list(b'{"format": "float8_e4m3fn"}'), dtype=torch.uint8),
    }, strict=False, assign=True)
    return model


if __name__ == "__main__":
    unittest.main()
