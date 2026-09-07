import pytest
import torch
from types import SimpleNamespace

from comfy.cli_args import args

args.cpu = True

import comfy.ops
from comfy.model_base import LLaDAImage as LLaDAImageBase, LLaDAImageSampling
from comfy.ldm.llada_image.conditioning import QueryFormer
from comfy.ldm.llada_image.model import LLaDAImage


def make_model(dtype=torch.float32, device=torch.device("cpu"), layers=1, refiners=1):
    model = LLaDAImage(
        all_patch_size=(1,),
        all_f_patch_size=(1,),
        in_channels=4,
        dim=32,
        n_layers=layers,
        n_refiner_layers=refiners,
        n_heads=2,
        cap_feat_dim=8,
        semantic_feat_dim=10,
        axes_dims=(4, 6, 6),
        dtype=dtype,
        device=device,
        operations=comfy.ops.disable_weight_init,
    ).eval()
    if device.type != "meta":
        torch.manual_seed(41)
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, std=0.02)
    return model


@pytest.mark.parametrize("inference_dtype", (torch.float32, torch.bfloat16))
def test_initial_noise_matches_official_dtype_roundtrip(inference_dtype):
    class ModelConfig:
        sampling_settings = {"shift": 1.0, "multiplier": 1.0}

    sampling = LLaDAImageSampling(ModelConfig(), inference_dtype)
    noise = torch.tensor([0.1234567, -0.9876543]).reshape(1, 1, 1, 2)
    latent = torch.tensor([0.25, -0.5]).reshape(1, 1, 1, 2)
    expected = noise.to(inference_dtype).float() + latent

    first = sampling.noise_scaling(torch.tensor(0.9999), noise, latent)
    second = sampling.noise_scaling(torch.tensor(0.5), noise, latent)

    assert torch.equal(first, expected)
    assert torch.equal(second, expected)


@pytest.mark.parametrize("inference_dtype", (torch.float32, torch.bfloat16))
def test_timestep_rounds_before_transformer_scaling(inference_dtype):
    model = SimpleNamespace(get_dtype_inference=lambda: inference_dtype)
    sigma = torch.tensor([0.9998922348, 0.35123456])
    original = sigma.clone()

    timestep = LLaDAImageBase.process_timestep(model, sigma, x=torch.empty(1))

    assert timestep.dtype == inference_dtype
    assert torch.equal(timestep * 1000.0, sigma.to(inference_dtype) * 1000.0)
    assert torch.equal(sigma, original)
    if inference_dtype == torch.bfloat16:
        assert not torch.equal(timestep.float() * 1000.0, sigma * 1000.0)


def test_patchify_pads_and_unpatchify_crops_nondivisible_latents():
    model = make_model(layers=0, refiners=0)
    image = torch.arange(4 * 3 * 5 * 7, dtype=torch.float32).reshape(4, 3, 5, 7)

    patches, image_size, token_grid_size = model.patchify_image(image, 2, 2)
    padding = (-len(patches)) % 32
    hidden_states = torch.cat(
        (patches, patches.new_zeros(padding, patches.shape[-1])), dim=0
    ).unsqueeze(0)
    restored = model.unpatchify(hidden_states, [image_size], 2, 2)[0]

    assert token_grid_size == (2, 3, 4)
    assert restored.shape == image.shape
    assert torch.equal(restored, image)


def test_raw_conditioning_parameters_follow_input_dtype():
    model = QueryFormer(
        num_queries=2,
        hidden_size=4,
        num_hidden_layers=0,
        num_attention_heads=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        operations=comfy.ops.disable_weight_init,
    )
    torch.nn.init.normal_(model.meta_queries)
    inputs = torch.randn(3, 5, 4, dtype=torch.bfloat16)

    output = model(inputs, torch.ones(3, 5, dtype=torch.bool))

    assert output.dtype == inputs.dtype
    assert torch.equal(output, model.meta_queries.to(inputs).unsqueeze(0).expand(3, -1, -1))


def test_sequence_pad_tokens_follow_feature_dtype():
    model = make_model(layers=0, refiners=0)
    features = [
        torch.randn(2, 32, dtype=torch.bfloat16),
        torch.randn(1, 32, dtype=torch.bfloat16),
    ]
    positions = [torch.zeros(2, 3, dtype=torch.int32), torch.zeros(1, 3, dtype=torch.int32)]
    inner_padding = [torch.tensor([False, True]), torch.tensor([False])]

    batched, _, _, _, _ = model.batch_sequences(
        features, positions, inner_padding, model.x_pad_token
    )

    assert batched.dtype == features[0].dtype
    assert torch.equal(batched[0, 1], model.x_pad_token[0].to(batched))


@pytest.mark.parametrize(
    ("editing", "semantic_length"),
    [(False, 0), (False, 3), (True, 0), (True, 3)],
)
def test_cpu_bfloat16_batch_execution_is_finite(editing, semantic_length):
    model = make_model(torch.bfloat16)
    latent = torch.randn(2, 4, 4, 6, dtype=torch.bfloat16)
    context = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    attention_mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool
    )
    semantics = torch.randn(2, semantic_length, 10, dtype=torch.bfloat16)
    if semantic_length:
        semantic_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    else:
        semantic_mask = torch.zeros(2, 0, dtype=torch.bool)
    kwargs = {}
    if editing:
        kwargs["source_latents"] = torch.randn_like(latent)

    with torch.inference_mode():
        output = model(
            latent,
            torch.tensor([0.8, 0.3], dtype=torch.bfloat16),
            context=context,
            attention_mask=attention_mask,
            semantic_features=semantics,
            semantic_mask=semantic_mask,
            **kwargs,
        )

    assert output.shape == latent.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("width", "height", "batch_size"),
    [(1024, 1024, 1), (1024, 768, 2)],
)
def test_full_resolution_geometry_and_batch_shape(width, height, batch_size):
    # Zero transformer/refiner layers keep this a geometry and sequence-layout test;
    # attention behavior is covered independently on small executable sequences.
    model = make_model(layers=0, refiners=0)
    latent = torch.randn(batch_size, 4, height // 16, width // 16)
    context = torch.randn(batch_size, 3, 8)
    attention_mask = torch.ones(batch_size, 3, dtype=torch.bool)

    with torch.inference_mode():
        output = model(
            latent,
            torch.full((batch_size,), 0.5),
            context=context,
            attention_mask=attention_mask,
        )

    assert output.shape == latent.shape
    assert torch.isfinite(output).all()


def test_meta_unload_and_assign_reload_roundtrip():
    model = make_model()
    latent = torch.randn(1, 4, 3, 5)
    context = torch.randn(1, 4, 8)
    attention_mask = torch.ones(1, 4, dtype=torch.bool)
    state_dict = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }

    model.to_empty(device="meta")
    assert all(parameter.is_meta for parameter in model.parameters())
    incompatible = model.load_state_dict(state_dict, strict=True, assign=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    assert all(not parameter.is_meta for parameter in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state_dict[key], rtol=0, atol=0)

    with torch.inference_mode():
        output = model(
            latent,
            torch.tensor([0.4]),
            context=context,
            attention_mask=attention_mask,
        )

    assert output.shape == latent.shape
    assert torch.isfinite(output).all()
