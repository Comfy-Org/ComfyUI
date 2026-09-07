import torch
import pytest
from types import SimpleNamespace

from comfy.cli_args import args

args.cpu = True

import comfy.k_diffusion.sampling
import comfy_extras.nodes_llada_image as llada_nodes
from comfy_extras.nodes_llada_image import (
    LLaDAImageEditConditioning,
    LLaDAImageScheduler,
    LLaDAImageVQConditioning,
    _set_semantic_conditioning,
    llada_image_sigmas,
    sample_llada_image_turbo,
)


class _Sampling:
    def __init__(self, variant):
        self.llada_image_variant = variant


class _Model:
    def __init__(self, variant):
        self.sampling = _Sampling(variant)

    def get_model_object(self, name):
        assert name == "model_sampling"
        return self.sampling


def test_base_sigmas_match_reference_formula():
    schedule = torch.linspace(0.001, 1.0, 6, dtype=torch.float64)[:-1]
    expected = 1.0 - (1.0 - (1.0 - schedule.pow(1.17)).pow(0.8)).pow(1.1)
    expected = torch.cat((expected.to(torch.float32), torch.zeros(1)))

    actual = llada_image_sigmas(5, "base")

    assert torch.equal(actual, expected)
    assert torch.all(actual[:-1] > actual[1:])


def test_turbo_sigmas_match_reference_shift():
    unshifted = torch.linspace(1.0, 0.0, 5, dtype=torch.float32)[:-1]
    expected = 3.0 * unshifted / (1.0 + 2.0 * unshifted)
    expected = torch.cat((expected, torch.zeros(1)))

    actual = llada_image_sigmas(4, "turbo")

    assert torch.equal(actual, expected)
    assert actual.tolist() == [1.0, 0.8999999761581421, 0.75, 0.5, 0.0]


def test_scheduler_zero_uses_variant_default_steps():
    base = LLaDAImageScheduler.execute(_Model("base"), 0)[0]
    turbo = LLaDAImageScheduler.execute(_Model("turbo"), 0)[0]

    assert len(base) == 51
    assert len(turbo) == 5


def test_turbo_sampler_uses_denoised_interpolation_and_is_seeded():
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    latent = torch.ones((1, 1, 2, 2))
    calls = []

    def model(x, sigma, **extra_args):
        calls.append((x.clone(), sigma.clone(), extra_args["seed"]))
        return x * 0.25

    first = sample_llada_image_turbo(
        model, latent.clone(), sigmas, extra_args={"seed": 12}
    )
    second = sample_llada_image_turbo(
        model, latent.clone(), sigmas, extra_args={"seed": 12}
    )
    different = sample_llada_image_turbo(
        model, latent.clone(), sigmas, extra_args={"seed": 13}
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert len(calls) == 6
    assert torch.equal(first, calls[1][0] * 0.25)


def test_turbo_sampler_single_step_matches_core_seeded_noise_source():
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    latent = torch.ones((1, 1, 2, 2))
    sampler = comfy.k_diffusion.sampling.default_noise_sampler(latent, seed=7)
    expected_first = 0.5 * (latent * 0.25) + 0.5 * sampler(sigmas[0], sigmas[1])
    expected = expected_first * 0.25

    actual = sample_llada_image_turbo(
        lambda x, sigma, **kwargs: x * 0.25,
        latent,
        sigmas,
        extra_args={"seed": 7},
    )

    assert torch.equal(actual, expected)


def test_standard_euler_matches_official_nonstochastic_turbo_update():
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    latent = torch.ones((1, 1, 2, 2))

    expected = latent.clone()
    for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
        denoised = expected * 0.25
        model_output = (expected - denoised) / sigma
        expected = expected + (sigma_next - sigma) * model_output

    actual = comfy.k_diffusion.sampling.sample_euler(
        lambda x, sigma, **kwargs: x * 0.25,
        latent,
        sigmas,
        disable=True,
    )

    assert torch.equal(actual, expected)


def test_semantic_conditioning_matches_cfg_and_edit_contract():
    positive = [[torch.randn(1, 3, 8), {}]]
    negative = [[torch.randn(1, 2, 8), {}]]
    semantic = torch.randn(1, 4, 10)
    source = torch.randn(1, 4, 2, 2)

    positive, negative = _set_semantic_conditioning(
        positive, negative, semantic, source
    )

    assert torch.equal(positive[0][1]["semantic_features"], semantic)
    assert positive[0][1]["semantic_mask"].all()
    assert negative[0][1]["semantic_features"].shape == (1, 0, 10)
    assert negative[0][1]["semantic_mask"].shape == (1, 0)
    assert torch.equal(positive[0][1]["source_latents"], source)
    assert torch.equal(negative[0][1]["source_latents"], source)


def test_load_llada_clip_rejects_wrong_encoder_before_loading():
    class Clip:
        cond_stage_model = object()
        load_called = False

        def load_model(self):
            self.load_called = True

    clip = Clip()
    with pytest.raises(ValueError, match="not an LLaDA-Image AIO text encoder"):
        llada_nodes._load_llada_clip(clip)

    assert not clip.load_called


def test_vq_conditioning_generates_expected_token_count(monkeypatch):
    positive = [[torch.randn(1, 2, 8), {}]]
    negative = [[torch.randn(1, 2, 8), {}]]

    class Tokenizer:
        @staticmethod
        def tokenize_vq(prompt, height, width):
            assert (prompt, height, width) == ("fox", 768, 1024)
            return [1, 2], [3], 24, 32

    class SigVQ:
        class Embedding:
            num_embeddings = 16

        prior_token_embedding = Embedding()

    class Model:
        dtype = torch.float32
        sigvq = SigVQ()

        @staticmethod
        def generate_vq_tokens(
            input_ids, unconditional_ids, image_token_count, cfg_scale
        ):
            assert image_token_count == 24 * 32
            assert cfg_scale == 2.0
            return torch.arange(image_token_count).remainder(16).unsqueeze(0)

        @staticmethod
        def encode_sigvq(token_ids=None, pixel_values=None):
            return torch.ones(1, token_ids.shape[1], 10), token_ids

    class Clip:
        tokenizer = Tokenizer()

    monkeypatch.setattr(
        llada_nodes,
        "_encode_prompts",
        lambda clip, prompt, negative_prompt: (positive, negative),
    )
    def load_clip(clip, vq_sequence_length):
        assert vq_sequence_length == 2 + 24 * 32
        return Model(), torch.device("cpu")

    monkeypatch.setattr(llada_nodes, "_load_llada_clip", load_clip)

    output_positive, output_negative = LLaDAImageVQConditioning.execute(
        Clip(), "fox", "", 1024, 768
    )

    assert output_positive[0][1]["semantic_features"].shape == (1, 768, 10)
    assert output_positive[0][1]["semantic_mask"].all()
    assert output_negative[0][1]["semantic_features"].shape == (1, 0, 10)


def test_load_llada_clip_reserves_vq_memory_through_model_management(monkeypatch):
    class Encoder:
        llada2 = object()

        def vq_memory_estimation(self, sequence_length):
            assert sequence_length == 1050
            return 2345678901

        def set_clip_options(self, options):
            assert options == {"execution_device": torch.device("cpu")}

    patcher = SimpleNamespace(load_device=torch.device("cpu"))
    clip = SimpleNamespace(cond_stage_model=Encoder(), patcher=patcher)
    monkeypatch.setattr(llada_nodes, "LLaDAImageTEModel", Encoder)
    calls = []
    monkeypatch.setattr(
        llada_nodes.comfy.model_management, "load_models_gpu",
        lambda models, **kwargs: calls.append((models, kwargs)),
    )

    model, device = llada_nodes._load_llada_clip(clip, 1050)

    assert model is clip.cond_stage_model.llada2
    assert device == patcher.load_device
    assert calls == [([patcher], {"memory_required": 2345678901})]


def test_vq_conditioning_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="divisible by 16"):
        LLaDAImageVQConditioning.execute(None, "", "", 1000, 768)


@pytest.mark.parametrize("model_dtype", (torch.float32, torch.bfloat16))
def test_edit_conditioning_preprocesses_both_native_paths(monkeypatch, model_dtype):
    positive = [[torch.randn(1, 2, 8), {}]]
    negative = [[torch.randn(1, 2, 8), {}]]
    calls = {}

    class VAE:
        @staticmethod
        def encode(image):
            calls["vae"] = image
            return torch.randn(
                image.shape[0], 128, image.shape[1] // 16, image.shape[2] // 16
            )

    class Model:
        dtype = model_dtype

        @staticmethod
        def encode_sigvq(pixel_values=None, token_ids=None):
            calls["sigvq"] = pixel_values
            return torch.randn(pixel_values.shape[0], 12, 10), None

    monkeypatch.setattr(
        llada_nodes,
        "_encode_prompts",
        lambda clip, prompt, negative_prompt: (positive, negative),
    )
    monkeypatch.setattr(
        llada_nodes, "_load_llada_clip", lambda clip: (Model(), torch.device("cpu"))
    )

    image = torch.rand(2, 65, 97, 3)
    output_positive, output_negative, latent = LLaDAImageEditConditioning.execute(
        object(), VAE(), image, "edit", ""
    )

    assert calls["vae"].shape == (2, 64, 96, 3)
    assert calls["sigvq"].shape == (2, 3, 32, 48)
    expected_pixels = torch.nn.functional.interpolate(
        calls["vae"].movedim(-1, 1).float() * 2.0 - 1.0,
        size=(32, 48), mode="bilinear", align_corners=False,
    ).to(model_dtype)
    assert torch.equal(calls["sigvq"], expected_pixels)
    assert calls["sigvq"].min() >= -1.0
    assert calls["sigvq"].max() <= 1.0
    assert latent["samples"].shape == (2, 128, 4, 6)
    assert output_positive[0][1]["semantic_features"].shape == (2, 12, 10)
    assert output_positive[0][1]["source_latents"].shape == (2, 128, 4, 6)
    assert output_negative[0][1]["source_latents"].shape == (2, 128, 4, 6)


def test_edit_conditioning_preserves_already_aligned_input(monkeypatch):
    positive = [[torch.randn(1, 2, 8), {}]]
    negative = [[torch.randn(1, 2, 8), {}]]
    calls = {}

    class VAE:
        @staticmethod
        def encode(image):
            calls["vae"] = image
            return torch.zeros(1, 128, 4, 4)

    class Model:
        dtype = torch.float32

        @staticmethod
        def encode_sigvq(pixel_values=None, token_ids=None):
            return torch.zeros(1, 16, 10), None

    monkeypatch.setattr(
        llada_nodes,
        "_encode_prompts",
        lambda clip, prompt, negative_prompt: (positive, negative),
    )
    monkeypatch.setattr(
        llada_nodes, "_load_llada_clip", lambda clip: (Model(), torch.device("cpu"))
    )

    def unexpected_resize(*args, **kwargs):
        raise AssertionError("an already aligned image must not be resampled")

    monkeypatch.setattr(llada_nodes.comfy.utils, "common_upscale", unexpected_resize)
    image = torch.linspace(0.0, 1.0, 64 * 64 * 3).reshape(1, 64, 64, 3)

    LLaDAImageEditConditioning.execute(object(), VAE(), image, "edit", "")

    assert torch.equal(calls["vae"], image)
