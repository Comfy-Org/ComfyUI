import hashlib
import importlib.util
import os
from pathlib import Path

import torch
import pytest

from comfy.cli_args import args

args.cpu = True

import comfy.model_management
import comfy.ops
from comfy.ldm.llada_image.model import LLaDAImage


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
requires_upstream = pytest.mark.skipif(
    not UPSTREAM_MODEL.is_file(),
    reason="optional parity test requires a sibling LLaDA-Image checkout",
)


def load_reference_class():
    actual_sha256 = hashlib.sha256(UPSTREAM_MODEL.read_bytes()).hexdigest()
    if actual_sha256 != UPSTREAM_MODEL_SHA256:
        raise ValueError(
            "upstream transformer fixture hash mismatch: expected "
            f"{UPSTREAM_MODEL_SHA256}, got {actual_sha256}"
        )
    spec = importlib.util.spec_from_file_location(
        "llada_image_reference", UPSTREAM_MODEL
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LLaDAImageTransformer2DModel


def make_models(dtype=torch.float32, device=torch.device("cpu")):
    config = model_config()
    torch.manual_seed(1)
    reference = (
        load_reference_class()(**config, axes_lens=(512, 32, 32))
        .to(device=device, dtype=dtype)
        .eval()
    )
    model = make_native_model(dtype, device)
    incompatible = model.load_state_dict(reference.state_dict(), strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    return reference, model


def model_config():
    return {
        "all_patch_size": (1,),
        "all_f_patch_size": (1,),
        "in_channels": 4,
        "dim": 32,
        "n_layers": 2,
        "n_refiner_layers": 1,
        "n_heads": 2,
        "norm_eps": 1e-5,
        "qk_norm": True,
        "cap_feat_dim": 8,
        "semantic_feat_dim": 10,
        "rope_theta": 256.0,
        "t_scale": 1000.0,
        "axes_dims": (4, 6, 6),
    }


def make_native_model(dtype=torch.float32, device=torch.device("cpu")):
    return LLaDAImage(
        **model_config(),
        dtype=dtype,
        device=device,
        operations=comfy.ops.disable_weight_init,
    ).eval()


def pad_features(features):
    length = max(len(value) for value in features)
    output = features[0].new_zeros((len(features), length, features[0].shape[-1]))
    mask = torch.zeros(
        (len(features), length), dtype=torch.bool, device=features[0].device
    )
    for index, value in enumerate(features):
        output[index, : len(value)] = value
        mask[index, : len(value)] = True
    return output, mask


@requires_upstream
def test_text_to_image_matches_reference():
    reference, model = make_models()
    x = torch.randn(2, 4, 4, 4)
    timestep = torch.tensor([0.8, 0.35])
    captions = [torch.randn(3, 8), torch.randn(5, 8)]
    semantics = [torch.randn(3, 10), torch.randn(5, 10)]
    context, attention_mask = pad_features(captions)
    semantic_features, semantic_mask = pad_features(semantics)

    expected = reference(
        x=[value.unsqueeze(1) for value in x],
        t=timestep,
        cap_feats=captions,
        glm_cap_feats=semantics,
    ).sample
    expected = -torch.stack([value.squeeze(1) for value in expected])
    actual = model(
        x,
        timestep,
        context=context,
        attention_mask=attention_mask,
        semantic_features=semantic_features,
        semantic_mask=semantic_mask,
    )

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@requires_upstream
def test_editing_matches_reference():
    reference, model = make_models()
    x = torch.randn(2, 4, 4, 4)
    source = torch.randn(2, 4, 4, 4)
    timestep = torch.tensor([0.7, 0.2])
    captions = [torch.randn(3, 8), torch.randn(5, 8)]
    semantics = [torch.randn(2, 10), torch.randn(4, 10)]
    context, attention_mask = pad_features(captions)
    semantic_features, semantic_mask = pad_features(semantics)

    expected = reference(
        x=[value.unsqueeze(1) for value in x],
        t=timestep,
        cap_feats=captions,
        glm_cap_feats=semantics,
        source_latents=[value.unsqueeze(1) for value in source],
    ).sample
    expected = -torch.stack([value.squeeze(1) for value in expected])
    actual = model(
        x,
        timestep,
        context=context,
        attention_mask=attention_mask,
        semantic_features=semantic_features,
        semantic_mask=semantic_mask,
        source_latents=source,
    )

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("editing", (False, True))
@requires_upstream
def test_bfloat16_matches_reference(editing):
    device = torch.device(os.environ.get("LLADA_IMAGE_PARITY_DEVICE", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("LLADA_IMAGE_PARITY_DEVICE=cuda requires a CUDA PyTorch build")
    reference, model = make_models(torch.bfloat16, device)
    torch.manual_seed(101)
    x = torch.randn(2, 4, 4, 4, dtype=torch.bfloat16, device=device)
    timestep = torch.tensor([0.7, 0.2], dtype=torch.bfloat16, device=device)
    captions = [
        torch.randn(3, 8, dtype=torch.bfloat16, device=device),
        torch.randn(5, 8, dtype=torch.bfloat16, device=device),
    ]
    context, attention_mask = pad_features(captions)
    reference_kwargs = {}
    native_kwargs = {}
    if editing:
        semantics = [
            torch.randn(2, 10, dtype=torch.bfloat16, device=device),
            torch.randn(4, 10, dtype=torch.bfloat16, device=device),
        ]
        semantic_features, semantic_mask = pad_features(semantics)
        source = torch.randn_like(x)
        reference_kwargs = {
            "glm_cap_feats": semantics,
            "source_latents": [value.unsqueeze(1) for value in source],
        }
        native_kwargs = {
            "semantic_features": semantic_features,
            "semantic_mask": semantic_mask,
            "source_latents": source,
        }

    with torch.inference_mode():
        expected = reference(
            x=[value.unsqueeze(1) for value in x],
            t=timestep,
            cap_feats=captions,
            **reference_kwargs,
        ).sample
        expected = -torch.stack([value.squeeze(1) for value in expected])
        actual = model(
            x,
            timestep,
            context=context,
            attention_mask=attention_mask,
            **native_kwargs,
        )

    absolute_error = (actual.float() - expected.float()).abs()
    # Diffusers and Core dispatch different BF16 attention kernels. Keep the
    # tolerance bounded to two BF16 quanta around unit-scale activations and also
    # constrain aggregate drift; this was measured on both CPU and RTX 5090 CUDA.
    assert float(absolute_error.max()) <= 1.0 / 64.0
    assert float(absolute_error.mean()) <= 1.0 / 256.0


def test_transformer_matches_official_config_validation():
    with pytest.raises(ValueError, match="same length"):
        LLaDAImage(
            all_patch_size=(1, 2),
            all_f_patch_size=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
            operations=comfy.ops.disable_weight_init,
        )
    with pytest.raises(ValueError, match="must be divisible"):
        LLaDAImage(
            dim=31,
            n_heads=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
            operations=comfy.ops.disable_weight_init,
        )
    with pytest.raises(ValueError, match="sum of axes_dims"):
        LLaDAImage(
            dim=32,
            n_heads=2,
            axes_dims=(4, 4, 4),
            dtype=torch.float32,
            device=torch.device("cpu"),
            operations=comfy.ops.disable_weight_init,
        )


def test_transformer_matches_official_conditioning_validation():
    model = make_native_model()
    latent = torch.randn(1, 4, 2, 2)
    timestep = torch.tensor([0.5])

    with pytest.raises(ValueError, match="requires context or semantic_features"):
        model(latent, timestep)
    with pytest.raises(ValueError, match="editing requires"):
        model(
            latent,
            timestep,
            context=torch.randn(1, 2, 8),
            source_latents=latent,
        )
