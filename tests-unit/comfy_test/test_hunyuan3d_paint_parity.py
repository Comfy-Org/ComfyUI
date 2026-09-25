"""Tiny-goldens parity test for the Hunyuan3D 2.1 paint UNet port.

Rebuilds a seeded 2-block micro-config UNet2p5DConditionModel and asserts its
forward pass reproduces the committed golden bundle (inputs + expected noise
prediction, <100 KB safetensors). Guards the numeric behaviour of every
attention mechanism in the port against silent drift. See paint_parity/README.md
for the full harness (including the author-side reference capture).
"""

from __future__ import annotations

import json
import os
import sys

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paint_parity import bundle_format, harness  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "paint_parity", "goldens", "tiny_golden.safetensors")


def test_tiny_golden_bundle_is_small_and_described():
    assert os.path.exists(GOLDEN), "run paint_parity/make_goldens.py"
    assert os.path.getsize(GOLDEN) < 100_000  # committed artifact stays tiny
    _tensors, metadata = bundle_format.load_bundle(GOLDEN)
    assert metadata["source"] == "native-tiny"
    assert json.loads(metadata["config"]) == {
        k: list(v) if isinstance(v, tuple) else v for k, v in harness.TINY_CONFIG.items()}
    assert json.loads(metadata["input_args"]) == harness.TINY_INPUT_ARGS


def test_tiny_model_build_is_deterministic():
    a = harness.build_tiny_model()
    b = harness.build_tiny_model()
    sd_a, sd_b = a.state_dict(), b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), k


def test_tiny_golden_forward_parity():
    tensors, _metadata = bundle_format.load_bundle(GOLDEN)
    model = harness.build_tiny_model()
    out, _ = harness.run_model(model, tensors)
    expected = tensors["output/noise_pred"]
    assert torch.isfinite(out).all()
    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected, atol=harness.TINY_ATOL, rtol=harness.TINY_RTOL)


def test_block_capture_covers_every_boundary():
    tensors, _metadata = bundle_format.load_bundle(GOLDEN)
    model = harness.build_tiny_model()
    _out, acts = harness.run_model(model, tensors, capture_blocks=True)
    expected_names = bundle_format.block_names(2, 2)
    assert sorted(acts.keys()) == sorted(f"act/{n}" for n in expected_names)
    for v in acts.values():
        assert torch.isfinite(v).all()
