from types import SimpleNamespace

import pytest
import torch

import comfy.samplers as comfy_samplers
from comfy.model_sampling import ModelSamplingFlux


DYNAMIC_SHIFT = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 8192,
    "max_shift": 0.9,
}


def _sampling():
    settings = {
        "shift": 0.69,
        "dynamic_shift": DYNAMIC_SHIFT,
        "shift_terminal": 0.02,
    }
    return ModelSamplingFlux(SimpleNamespace(sampling_settings=settings))


def _reference_sigmas(latent_height, latent_width, steps):
    sequence_length = latent_height * latent_width
    shift = 0.5 + 0.4 * (sequence_length - 256) / (8192 - 256)
    timesteps = torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float64)
    exp_shift = torch.exp(torch.tensor(shift, dtype=torch.float64))
    sigmas = exp_shift / (exp_shift + (1.0 / timesteps - 1.0))
    scale_factor = (1.0 - sigmas[-1]) / (1.0 - 0.02)
    sigmas = 1.0 - ((1.0 - sigmas) / scale_factor)
    return torch.cat((sigmas, torch.zeros(1, dtype=torch.float64)))


def _capture_ksampler(monkeypatch, model_sampling, steps=40, denoise=None):
    captured = []
    model = SimpleNamespace(
        model_options={},
        get_model_object=lambda name: model_sampling,
    )
    sampler = comfy_samplers.KSampler(
        model,
        steps=steps,
        device=torch.device("cpu"),
        sampler="euler",
        scheduler="simple",
        denoise=denoise,
    )

    def capture_sample(
        model,
        noise,
        positive,
        negative,
        cfg,
        device,
        sampler,
        sigmas,
        model_options,
        **kwargs,
    ):
        captured.append(sigmas.clone())
        return kwargs["latent_image"]

    monkeypatch.setattr(comfy_samplers, "sample", capture_sample)
    monkeypatch.setattr(comfy_samplers, "sampler_object", lambda name: None)
    return sampler, captured


def _sample_shape(sampler, latent_height, latent_width, sigmas=None, **kwargs):
    latent = torch.empty((1, 4, latent_height, latent_width))
    return sampler.sample(
        noise=torch.zeros_like(latent),
        positive=[],
        negative=[],
        cfg=1.0,
        latent_image=latent,
        sigmas=sigmas,
        **kwargs,
    )


@pytest.mark.parametrize(
    "latent_height,latent_width", [(32, 32), (64, 64), (128, 128), (96, 64)]
)
def test_ksampler_schedule_matches_published_dynamic_reference(
    monkeypatch, latent_height, latent_width
):
    sampling = _sampling()
    sampler, captured = _capture_ksampler(monkeypatch, sampling)

    _sample_shape(sampler, latent_height, latent_width)
    actual = captured[-1].double()
    expected = _reference_sigmas(latent_height, latent_width, 40)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-4)
    assert actual[0].item() == pytest.approx(1.0)
    assert actual[-2].item() == pytest.approx(0.02, abs=1e-6)
    assert actual[-1].item() == 0.0


def test_dynamic_shift_resolution_does_not_mutate_shared_sampling(monkeypatch):
    sampling = _sampling()
    sigma_table = sampling.sigmas.clone()
    sampler, captured = _capture_ksampler(monkeypatch, sampling)

    _sample_shape(sampler, 64, 64)
    first = captured[-1].clone()
    _sample_shape(sampler, 128, 128)
    second = captured[-1].clone()
    _sample_shape(sampler, 64, 64)

    torch.testing.assert_close(captured[-1], first)
    assert first[20].item() != second[20].item()
    assert sampling.shift == 0.69
    torch.testing.assert_close(sampling.sigmas, sigma_table)


def test_fixed_flux_override_keeps_its_explicit_shift(monkeypatch):
    sampling = ModelSamplingFlux(SimpleNamespace(sampling_settings={"shift": 0.9}))
    sampler, captured = _capture_ksampler(monkeypatch, sampling)
    expected = sampler.sigmas.clone()

    _sample_shape(sampler, 128, 128)

    torch.testing.assert_close(captured[-1], expected)


def test_explicit_sigmas_bypass_dynamic_default(monkeypatch):
    sampler, captured = _capture_ksampler(monkeypatch, _sampling())
    explicit_sigmas = torch.tensor([1.0, 0.7, 0.02, 0.0])

    _sample_shape(sampler, 128, 128, sigmas=explicit_sigmas)

    torch.testing.assert_close(captured[-1], explicit_sigmas)


def test_partial_steps_slice_the_resolved_schedule(monkeypatch):
    sampler, captured = _capture_ksampler(monkeypatch, _sampling())
    expected = _reference_sigmas(128, 128, 40)[10:21]

    _sample_shape(sampler, 128, 128, start_step=10, last_step=20)

    torch.testing.assert_close(captured[-1].double(), expected, rtol=1e-4, atol=2e-4)


def test_denoise_uses_the_resolved_full_schedule_before_truncating(monkeypatch):
    sampler, captured = _capture_ksampler(monkeypatch, _sampling(), denoise=0.5)
    expected = _reference_sigmas(128, 128, 80)[-41:]

    _sample_shape(sampler, 128, 128)

    torch.testing.assert_close(captured[-1].double(), expected, rtol=1e-4, atol=2e-4)


def test_single_inference_sigma_is_preserved_during_terminal_stretch():
    sampling = _sampling().for_latent_image(torch.empty((1, 4, 128, 128)))
    sigmas = torch.tensor([1.0, 0.0])

    torch.testing.assert_close(sampling.stretch_sigmas(sigmas), sigmas)
