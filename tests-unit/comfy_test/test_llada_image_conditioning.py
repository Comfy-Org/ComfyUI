import hashlib
import importlib.util
import os
from pathlib import Path

import torch
import pytest

from comfy.cli_args import args

args.cpu = True

import comfy.ops
from comfy.ldm.llada_image.conditioning import QueryFormer, SigVQ, TextProjection


UPSTREAM_MODEL = (
    Path(__file__).resolve().parents[3]
    / "LLaDA-Image"
    / "src"
    / "models"
    / "transformer_llada_image.py"
)
UPSTREAM_MODEL_SHA256 = (
    "1460e875568f80c3c153ff07888a1b855bd1f5c290db3d16bad5288d29fcbbf2"
)
pytestmark = pytest.mark.skipif(
    not UPSTREAM_MODEL.is_file(),
    reason="optional parity test requires a sibling LLaDA-Image checkout",
)


def load_reference_module():
    actual_sha256 = hashlib.sha256(UPSTREAM_MODEL.read_bytes()).hexdigest()
    if actual_sha256 != UPSTREAM_MODEL_SHA256:
        raise ValueError(
            "upstream conditioning fixture hash mismatch: expected "
            f"{UPSTREAM_MODEL_SHA256}, got {actual_sha256}"
        )
    spec = importlib.util.spec_from_file_location(
        "llada_image_conditioning_reference", UPSTREAM_MODEL
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parity_device():
    device = torch.device(os.environ.get("LLADA_IMAGE_PARITY_DEVICE", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("LLADA_IMAGE_PARITY_DEVICE=cuda requires a CUDA PyTorch build")
    return device


def assert_bfloat16_parity(actual, expected):
    absolute_error = (actual.float() - expected.float()).abs()
    assert float(absolute_error.max()) <= 1.0 / 64.0
    assert float(absolute_error.mean()) <= 1.0 / 256.0


def test_queryformer_matches_reference():
    reference_class = load_reference_module().LLaDAImageQueryFormerModel
    config = {
        "num_queries": 5,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 24,
        "dropout": 0.0,
        "norm_eps": 1e-6,
    }
    torch.manual_seed(2)
    reference = reference_class(**config).eval()
    model = QueryFormer(
        **{key: value for key, value in config.items() if key != "dropout"},
        operations=comfy.ops.disable_weight_init,
    ).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    inputs = torch.randn(2, 7, 16)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool
    )

    expected = reference(inputs, mask).query_embeds
    actual = model(inputs, mask)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_text_projection_matches_reference():
    reference_class = load_reference_module().LLaDAImageTextProjectionModel
    config = {
        "hidden_size": 16,
        "intermediate_size": 28,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "projection_dim": 20,
        "attention_dropout": 0.0,
        "norm_eps": 1e-6,
    }
    torch.manual_seed(3)
    reference = reference_class(**config).eval()
    model = TextProjection(
        **{key: value for key, value in config.items() if key != "attention_dropout"},
        operations=comfy.ops.disable_weight_init,
    ).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    hidden_states = torch.randn(2, 9, 16)

    expected = reference(hidden_states).hidden_states
    actual = model(hidden_states)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_sigvq_image_and_token_paths_match_reference():
    reference_class = load_reference_module().LLaDAImageSigVQModel
    config = {
        "image_size": 16,
        "patch_size": 4,
        "in_channels": 3,
        "hidden_size": 16,
        "intermediate_size": 28,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "attention_bias": True,
        "attention_dropout": 0.0,
        "norm_eps": 1e-6,
        "codebook_size": 8,
        "codebook_embed_dim": 4,
        "semantic_embed_dim": 10,
    }
    torch.manual_seed(4)
    reference = reference_class(**config).eval()
    model = SigVQ(
        **{key: value for key, value in config.items() if key != "attention_dropout"},
        operations=comfy.ops.disable_weight_init,
    ).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    pixels = torch.randn(2, 3, 8, 12)

    expected_image = reference(pixel_values=pixels)
    actual_semantic, actual_tokens = model(pixel_values=pixels)
    torch.testing.assert_close(
        actual_semantic, expected_image.semantic_features, atol=2e-5, rtol=2e-5
    )
    assert torch.equal(actual_tokens, expected_image.token_ids)

    token_ids = torch.tensor([[0, 2, 4], [1, 3, 5]])
    expected_tokens = reference(token_ids=token_ids)
    actual_semantic, actual_tokens = model(token_ids=token_ids)
    torch.testing.assert_close(
        actual_semantic, expected_tokens.semantic_features, atol=2e-5, rtol=2e-5
    )
    assert torch.equal(actual_tokens, token_ids)


def test_queryformer_bfloat16_matches_reference():
    device = parity_device()
    reference_class = load_reference_module().LLaDAImageQueryFormerModel
    config = {
        "num_queries": 5,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 24,
        "dropout": 0.0,
        "norm_eps": 1e-6,
    }
    torch.manual_seed(102)
    reference = reference_class(**config).to(device, torch.bfloat16).eval()
    model = QueryFormer(
        **{key: value for key, value in config.items() if key != "dropout"},
        dtype=torch.bfloat16,
        device=device,
        operations=comfy.ops.disable_weight_init,
    ).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    inputs = torch.randn(2, 7, 16, dtype=torch.bfloat16, device=device)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]],
        dtype=torch.bool,
        device=device,
    )

    with torch.inference_mode():
        expected = reference(inputs, mask).query_embeds
        actual = model(inputs, mask)

    assert_bfloat16_parity(actual, expected)


def test_text_projection_bfloat16_matches_reference():
    device = parity_device()
    reference_class = load_reference_module().LLaDAImageTextProjectionModel
    config = {
        "hidden_size": 16,
        "intermediate_size": 28,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "projection_dim": 20,
        "attention_dropout": 0.0,
        "norm_eps": 1e-6,
    }
    torch.manual_seed(103)
    reference = reference_class(**config).to(device, torch.bfloat16).eval()
    model = TextProjection(
        **{
            key: value
            for key, value in config.items()
            if key != "attention_dropout"
        },
        dtype=torch.bfloat16,
        device=device,
        operations=comfy.ops.disable_weight_init,
    ).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    hidden_states = torch.randn(2, 9, 16, dtype=torch.bfloat16, device=device)

    with torch.inference_mode():
        expected = reference(hidden_states).hidden_states
        actual = model(hidden_states)

    assert_bfloat16_parity(actual, expected)


def test_sigvq_bfloat16_image_and_token_paths_match_reference():
    device = parity_device()
    reference_class = load_reference_module().LLaDAImageSigVQModel
    config = {
        "image_size": 16,
        "patch_size": 4,
        "in_channels": 3,
        "hidden_size": 16,
        "intermediate_size": 28,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "attention_bias": True,
        "attention_dropout": 0.0,
        "norm_eps": 1e-6,
        "codebook_size": 8,
        "codebook_embed_dim": 4,
        "semantic_embed_dim": 10,
    }
    torch.manual_seed(104)
    reference = reference_class(**config).to(device, torch.bfloat16).eval()
    model = SigVQ(
        **{
            key: value
            for key, value in config.items()
            if key != "attention_dropout"
        },
        dtype=torch.bfloat16,
        device=device,
        operations=comfy.ops.disable_weight_init,
    ).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    pixels = torch.randn(2, 3, 8, 12, dtype=torch.bfloat16, device=device)

    with torch.inference_mode():
        expected_image = reference(pixel_values=pixels)
        actual_semantic, actual_tokens = model(pixel_values=pixels)
    torch.testing.assert_close(
        actual_semantic, expected_image.semantic_features, rtol=0, atol=0
    )
    assert torch.equal(actual_tokens, expected_image.token_ids)

    token_ids = torch.tensor([[0, 2, 4], [1, 3, 5]], device=device)
    with torch.inference_mode():
        expected_tokens = reference(token_ids=token_ids)
        actual_semantic, actual_tokens = model(token_ids=token_ids)
    torch.testing.assert_close(
        actual_semantic, expected_tokens.semantic_features, rtol=0, atol=0
    )
    assert torch.equal(actual_tokens, token_ids)


@pytest.mark.parametrize("component", (QueryFormer, TextProjection, SigVQ))
def test_conditioning_components_reject_nondivisible_attention_heads(component):
    with pytest.raises(ValueError, match="must be divisible"):
        component(
            hidden_size=10,
            num_attention_heads=3,
            dtype=torch.float32,
            device=torch.device("cpu"),
            operations=comfy.ops.disable_weight_init,
        )


def test_sigvq_matches_official_input_validation():
    model = SigVQ(
        image_size=16,
        patch_size=4,
        hidden_size=16,
        intermediate_size=28,
        num_hidden_layers=0,
        num_attention_heads=2,
        codebook_size=8,
        codebook_embed_dim=4,
        semantic_embed_dim=10,
        dtype=torch.float32,
        device=torch.device("cpu"),
        operations=comfy.ops.disable_weight_init,
    )

    with pytest.raises(ValueError, match="4 dimensions"):
        model(pixel_values=torch.zeros(3, 8, 8))
    with pytest.raises(ValueError, match="divisible by 4"):
        model(pixel_values=torch.zeros(1, 3, 8, 10))
    with pytest.raises(ValueError, match="2 dimensions"):
        model(token_ids=torch.zeros(4, dtype=torch.long))
