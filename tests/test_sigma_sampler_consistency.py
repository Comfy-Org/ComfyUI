from types import SimpleNamespace

import torch

from comfy.samplers import (
    DISCARD_PENULTIMATE_SIGMA_SAMPLERS,
    KSampler,
    calculate_sigmas,
    calculate_sigmas_for_sampler,
)
from comfy_extras.nodes_custom_sampler import BasicScheduler

MS = SimpleNamespace(sigma_min=0.0292, sigma_max=14.6146)


def test_discard_set_exposes_known_samplers():
    assert "dpm_2" in DISCARD_PENULTIMATE_SIGMA_SAMPLERS
    assert "euler" not in DISCARD_PENULTIMATE_SIGMA_SAMPLERS


def test_helper_matches_ksampler_trim():
    # KSampler builds steps+1 sigmas and drops the penultimate one for these samplers;
    # the shared helper must produce exactly that schedule.
    expected = calculate_sigmas(MS, "karras", 11)
    expected = torch.cat([expected[:-2], expected[-1:]])

    got = calculate_sigmas_for_sampler(MS, "karras", 10, "dpm_2")

    assert torch.allclose(got, expected)


def test_non_discard_sampler_stays_untrimmed():
    plain = calculate_sigmas(MS, "karras", 10)

    got = calculate_sigmas_for_sampler(MS, "karras", 10, "euler")

    assert torch.allclose(got, plain)


def test_basic_scheduler_can_match_ksampler_schedule():
    model = SimpleNamespace(get_model_object=lambda _key: MS)

    ks = KSampler(model=model, steps=10, device="cpu", sampler="dpm_2", scheduler="karras", denoise=1.0)
    out = BasicScheduler.execute(model=model, scheduler="karras", steps=10, denoise=1.0, sampler_name="dpm_2")

    assert torch.allclose(out.result[0], ks.sigmas.cpu())


def test_basic_scheduler_default_matches_old_behavior():
    model = SimpleNamespace(get_model_object=lambda _key: MS)

    out = BasicScheduler.execute(model=model, scheduler="karras", steps=10, denoise=1.0)

    assert torch.allclose(out.result[0], calculate_sigmas(MS, "karras", 10))
