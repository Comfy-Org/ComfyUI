import pytest
import torch

import comfy.patcher_extension as patcher_extension
from comfy.samplers import KSAMPLER

SIGMAS = torch.tensor([14.6, 7.0, 0.0])
NOISE = torch.zeros(1, 4, 8, 8)
LATENT = torch.zeros(1, 4, 8, 8)


class _ModelSampling:
    sigma_max = 14.6

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return noise

    def inverse_noise_scaling(self, sigma, latent):
        return latent


class _InnerModel:
    def __init__(self):
        self.model_sampling = _ModelSampling()


class _ModelPatcher:
    def __init__(self):
        self.model = _InnerModel()


class _ModelWrap:
    """Minimal stand-in for CFGGuider: only what KSAMPLER.sample and the first-step log touch."""

    def __init__(self):
        self.inner_model = _InnerModel()
        self.model_patcher = _ModelPatcher()
        self.cfg = 8.0


def _sampler_function(model, noise, sigmas, extra_args=None, callback=None, disable=False, **kwargs):
    for i in range(len(sigmas) - 1):
        if callback is not None:
            callback({"i": i, "denoised": noise, "x": noise, "sigma": sigmas[i]})
    return noise


def _failing_sampler_function(model, noise, sigmas, extra_args=None, callback=None, disable=False, **kwargs):
    raise RuntimeError("sampling failed")


class _SamplerAbort(BaseException):
    pass


def _aborting_sampler_function(model, noise, sigmas, extra_args=None, callback=None, disable=False, **kwargs):
    raise _SamplerAbort("sampling aborted")


@pytest.fixture
def extra_args():
    return {"model_options": {}}


def _register(extra_args, call_type, callback):
    patcher_extension.add_callback(call_type, callback, extra_args["model_options"], is_model_options=True)


def test_sampling_unchanged_without_lifecycle_callbacks(extra_args):
    """No registered lifecycle callbacks: the legacy per-step callback still fires and the result is returned."""
    legacy_steps = []
    sampler = KSAMPLER(_sampler_function)

    samples = sampler.sample(
        _ModelWrap(), SIGMAS, extra_args,
        lambda i, denoised, x, total_steps: legacy_steps.append((i, total_steps)),
        NOISE, latent_image=LATENT,
    )

    assert samples.shape == NOISE.shape
    assert legacy_steps == [(0, 2), (1, 2)]


def test_start_step_end_callbacks_are_delivered(extra_args):
    started, stepped, ended = [], [], []
    events = []

    def start_callback(info):
        events.append("start")
        started.append(info)

    def step_callback(info):
        events.append(f"step:{info['step']}")
        stepped.append(info)

    def end_callback(info):
        events.append("end")
        ended.append(info)

    def legacy_callback(i, denoised, x, total_steps):
        events.append(f"legacy:{i}")

    _register(extra_args, patcher_extension.CallbacksMP.ON_SAMPLER_START, start_callback)
    _register(extra_args, patcher_extension.CallbacksMP.ON_SAMPLER_STEP, step_callback)
    _register(extra_args, patcher_extension.CallbacksMP.ON_SAMPLER_END, end_callback)

    KSAMPLER(_sampler_function).sample(
        _ModelWrap(), SIGMAS, extra_args, legacy_callback, NOISE, latent_image=LATENT,
    )

    assert events == ["start", "legacy:0", "step:0", "legacy:1", "step:1", "end"]

    assert len(started) == 1
    assert started[0]["total_steps"] == 2
    assert started[0]["sample_sigmas"] == tuple(float(sigma) for sigma in SIGMAS)
    assert started[0]["noise_shape"] == tuple(NOISE.shape)
    assert started[0]["latent_shape"] == tuple(LATENT.shape)
    assert started[0]["sampler_function"] == "_sampler_function"

    assert [s["step"] for s in stepped] == [0, 1]
    assert [s["total_steps"] for s in stepped] == [2, 2]
    assert float(stepped[0]["sigma"]) == pytest.approx(float(SIGMAS[0]))
    assert float(stepped[0]["sigma_next"]) == pytest.approx(float(SIGMAS[1]))
    assert float(stepped[1]["sigma_next"]) == pytest.approx(float(SIGMAS[2]))
    assert stepped[0]["sample_sigmas"] == tuple(float(sigma) for sigma in SIGMAS)
    assert stepped[0]["x_shape"] == tuple(NOISE.shape)
    assert stepped[0]["denoised_shape"] == tuple(NOISE.shape)

    assert len(ended) == 1
    assert ended[0]["total_steps"] == 2
    assert ended[0]["sample_sigmas"] == tuple(float(sigma) for sigma in SIGMAS)
    assert ended[0]["samples_shape"] == tuple(NOISE.shape)
    assert ended[0]["sampler_function"] == "_sampler_function"


def test_end_callback_runs_when_sampling_raises(extra_args):
    ended = []

    def end_callback(info):
        ended.append(info)
        raise RuntimeError("end callback failed")

    _register(extra_args, patcher_extension.CallbacksMP.ON_SAMPLER_END, end_callback)

    with pytest.raises(RuntimeError, match="sampling failed"):
        KSAMPLER(_failing_sampler_function).sample(
            _ModelWrap(), SIGMAS, extra_args, None, NOISE, latent_image=LATENT,
        )

    assert len(ended) == 1
    assert ended[0]["samples_shape"] is None
    assert ended[0]["total_steps"] == 2
    assert ended[0]["sampler_function"] == "_failing_sampler_function"


def test_end_callback_runs_when_sampling_aborts(extra_args):
    ended = []
    _register(extra_args, patcher_extension.CallbacksMP.ON_SAMPLER_END, ended.append)

    with pytest.raises(_SamplerAbort, match="sampling aborted"):
        KSAMPLER(_aborting_sampler_function).sample(
            _ModelWrap(), SIGMAS, extra_args, None, NOISE, latent_image=LATENT,
        )

    assert len(ended) == 1
    assert ended[0]["samples_shape"] is None
    assert ended[0]["total_steps"] == 2
    assert ended[0]["sampler_function"] == "_aborting_sampler_function"


def test_end_callback_runs_when_start_callback_raises(extra_args):
    ended = []

    def start_callback(info):
        raise RuntimeError("start callback failed")

    def end_callback(info):
        ended.append(info)
        raise RuntimeError("end callback failed")

    _register(extra_args, patcher_extension.CallbacksMP.ON_SAMPLER_START, start_callback)
    _register(extra_args, patcher_extension.CallbacksMP.ON_SAMPLER_END, end_callback)

    with pytest.raises(RuntimeError, match="start callback failed"):
        KSAMPLER(_sampler_function).sample(
            _ModelWrap(), SIGMAS, extra_args, None, NOISE, latent_image=LATENT,
        )

    assert len(ended) == 1
    assert ended[0]["samples_shape"] is None
    assert ended[0]["total_steps"] == 2
    assert ended[0]["sampler_function"] == "_sampler_function"
