import asyncio
from types import SimpleNamespace

import pytest
import torch

from comfy.sd1_clip import SDTokenizer
from comfy_api.latest._sdk import (
    ClipRef,
    CondRef,
    InProcessOps,
    InProcessRefResolver,
    ModelRef,
    SamplerRef,
    SigmasRef,
    bind_runtime,
)


class _Sampling:
    sigma_max = torch.tensor(14.0)
    sigma_min = torch.tensor(0.03)

    @staticmethod
    def percent_to_sigma(percent):
        if percent == 0.0:
            return 999_999_999.9
        if percent == 1.0:
            return 0.0
        return 10.0 * (1.0 - percent)


class _Model:
    def get_model_object(self, name):
        if name != "model_sampling":
            raise KeyError(name)
        return _Sampling()


class _ClipPatcher:
    def __init__(self):
        self.model_options = {"transformer_options": {"stable": True}}


class _Clip:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.patcher = _ClipPatcher()

    def clone(self):
        return _Clip(self.tokenizer)


def _tokenizer():
    tokenizer = object.__new__(SDTokenizer)
    tokenizer.embedding_key = "clip_l"
    tokenizer.start_token = 1
    tokenizer.end_token = 2
    tokenizer.pad_token = 0
    tokenizer.inv_vocab = {0: "<pad>", 1: "<start>", 2: "<end>", 7: "word"}
    return SimpleNamespace(clip_l=tokenizer)


def test_ppm_scalar_conditioning_token_and_sampler_primitives():
    async def run():
        refs = InProcessRefResolver()
        conditioning_value = [[
            torch.ones((1, 2, 3)),
            {"pooled_output": torch.ones((1, 3)), "stable": "yes"},
        ]]
        conditioning = CondRef._wrap(await refs.create(
            "CONDITIONING", conditioning_value))
        sigmas = SigmasRef._wrap(await refs.create(
            "SIGMAS", torch.tensor([4.0, 2.0, 0.0])))
        model = ModelRef._wrap(await refs.create("MODEL", _Model()))
        clip = ClipRef._wrap(await refs.create("CLIP", _Clip(_tokenizer())))

        with bind_runtime(refs, None, InProcessOps()):
            metadata = await conditioning.with_metadata(
                width=1024, height=768, crop_w=4, crop_h=8,
                target_width=896, target_height=640,
            )
            ranged = await conditioning.with_timestep_range(0.2, 0.8)
            zeroed = await conditioning.zero_out()
            sigma = await sigmas.value_at(-2)
            sigma_tail = await sigmas.slice(1)
            normal_endpoint = await model.sigma_for_percent(0.0)
            actual_endpoint = await model.sigma_for_percent(
                0.0, actual_endpoints=True)
            sampler = await SamplerRef.named(
                "gradient_estimation", ge_gamma=3.25)
            descriptions = await clip.describe_tokens({
                "l": [[(1, 1.0), (7, -0.5), (2, 1.0)]],
            })
            selected_clip = await clip.with_attention_impl("optimized")

            with pytest.raises(ValueError, match="does not accept"):
                await SamplerRef.named("euler", eta=1.0)
            with pytest.raises(ValueError, match="start <= end"):
                await conditioning.with_timestep_range(0.8, 0.2)
            with pytest.raises(IndexError, match="outside"):
                await sigmas.value_at(9)
            with pytest.raises(ValueError, match="at least two"):
                await sigmas.slice(2)

        return {
            "metadata": await refs.resolve(metadata),
            "ranged": await refs.resolve(ranged),
            "zeroed": await refs.resolve(zeroed),
            "sigma": sigma,
            "sigma_tail": await refs.resolve(sigma_tail),
            "normal_endpoint": normal_endpoint,
            "actual_endpoint": actual_endpoint,
            "sampler": await refs.resolve(sampler),
            "descriptions": descriptions,
            "selected_clip": await refs.resolve(selected_clip),
        }

    result = asyncio.run(run())
    assert result["metadata"][0][1] == {
        "pooled_output": result["metadata"][0][1]["pooled_output"],
        "stable": "yes",
        "width": 1024,
        "height": 768,
        "crop_w": 4,
        "crop_h": 8,
        "target_width": 896,
        "target_height": 640,
    }
    assert result["ranged"][0][1]["start_percent"] == 0.2
    assert result["ranged"][0][1]["end_percent"] == 0.8
    assert torch.count_nonzero(result["zeroed"][0][0]) == 0
    assert torch.count_nonzero(
        result["zeroed"][0][1]["pooled_output"]) == 0
    assert result["sigma"] == 2.0
    assert torch.equal(result["sigma_tail"], torch.tensor([2.0, 0.0]))
    assert result["normal_endpoint"] == pytest.approx(999_999_999.9)
    assert result["actual_endpoint"] == pytest.approx(14.0)
    assert result["sampler"].extra_options == {"ge_gamma": 3.25}
    assert result["descriptions"] == {
        "l": [[
            {"id": 1, "text": "<start>", "special": True},
            {"id": 7, "text": "word", "special": False},
            {"id": 2, "text": "<end>", "special": True},
        ]],
    }
    transformer_options = result["selected_clip"].patcher.model_options[
        "transformer_options"]
    assert transformer_options["stable"] is True
    assert callable(transformer_options["optimized_attention_override"])


def test_sigmas_from_values_materializes_bounded_float32_schedule():
    async def run():
        refs = InProcessRefResolver()
        with bind_runtime(refs, None, InProcessOps()):
            sigmas = await SigmasRef.from_values([14, 3.5, 0.0])
            single = await SigmasRef.from_values([0.0])
        return await refs.resolve(sigmas), await refs.resolve(single)

    sigmas, single = asyncio.run(run())
    assert sigmas.dtype == torch.float32
    assert sigmas.device.type == "cpu"
    assert torch.equal(sigmas, torch.tensor([14.0, 3.5, 0.0]))
    assert torch.equal(single, torch.tensor([0.0]))


@pytest.mark.parametrize(
    "values,error",
    [
        ([], ValueError),
        ([0.0] * 10002, ValueError),
        ([True], TypeError),
        ([[1.0]], TypeError),
        ([float("nan")], ValueError),
        ([float("inf")], ValueError),
        ([-0.01], ValueError),
    ],
)
def test_sigmas_from_values_rejects_malformed_values(values, error):
    async def run():
        refs = InProcessRefResolver()
        with bind_runtime(refs, None, InProcessOps()):
            with pytest.raises(error):
                await SigmasRef.from_values(values)

    asyncio.run(run())


def test_model_sampling_sigmas_uses_bounded_shift_without_mutating_model(
    monkeypatch,
):
    import comfy.samplers

    class Sampling:
        def __init__(self, config=None):
            self.config = config
            self.shift = None

        def set_parameters(self, *, shift):
            self.shift = shift

    original = Sampling("original")
    model = SimpleNamespace(
        model=SimpleNamespace(model_config="config"),
        get_model_object=lambda name: original if name == "model_sampling" else None,
    )
    calls = []

    def calculate_sigmas(sampling, scheduler, steps):
        calls.append((sampling, scheduler, steps))
        return torch.linspace(float(steps), 0.0, steps + 1)

    monkeypatch.setattr(comfy.samplers, "calculate_sigmas", calculate_sigmas)

    async def run():
        refs = InProcessRefResolver()
        model_ref = ModelRef._wrap(await refs.create("MODEL", model))
        with bind_runtime(refs, None, InProcessOps()):
            sigmas = await model_ref.sampling_sigmas(
                scheduler="normal", steps=4, denoise=0.5, shift=3.25)
            result = await refs.resolve(sigmas)
            with pytest.raises(ValueError, match="denoise"):
                await model_ref.sampling_sigmas(
                    scheduler="normal", steps=4, denoise=0.0)
            return result

    result = asyncio.run(run())
    assert torch.equal(result, torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0]))
    assert calls[0][1:] == ("normal", 8)
    assert calls[0][0].shift == 3.25
    assert calls[0][0].config == "config"
    assert original.shift is None


@pytest.mark.parametrize(
    "scheduler", ["simple", "sgm_uniform", "ddim_uniform", "beta", "normal"],
)
def test_model_sampling_sigmas_matches_comfy_shifted_schedule(scheduler):
    import comfy.model_sampling
    import comfy.samplers

    config = SimpleNamespace(sampling_settings={})
    original = comfy.model_sampling.ModelSamplingFlux(config)
    model = SimpleNamespace(
        model=SimpleNamespace(model_config=config),
        get_model_object=lambda name: original if name == "model_sampling" else None,
    )

    class ShiftedSampling(type(original)):
        pass

    expected_sampling = ShiftedSampling(config)
    expected_sampling.set_parameters(shift=3.25)
    expected = comfy.samplers.calculate_sigmas(
        expected_sampling, scheduler, 8).detach().cpu()[-5:]

    async def run():
        refs = InProcessRefResolver()
        model_ref = ModelRef._wrap(await refs.create("MODEL", model))
        with bind_runtime(refs, None, InProcessOps()):
            result = await model_ref.sampling_sigmas(
                scheduler=scheduler, steps=4, denoise=0.5, shift=3.25)
        return await refs.resolve(result)

    actual = asyncio.run(run())
    assert torch.equal(actual, expected)
    assert model.get_model_object("model_sampling") is original
