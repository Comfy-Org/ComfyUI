import inspect

import pytest
import torch

import comfy.utils
import nodes
from comfy import model_management, sample, samplers
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy_api import v4

POSITIVE = [[torch.zeros(1, 2, 4), {}]]
NEGATIVE = [[torch.ones(1, 2, 4), {}]]


@pytest.fixture
def model(tiny_sd15):
    yield tiny_sd15(
        transformer_depth=[1],
        transformer_depth_output=[1, 1],
        use_spatial_transformer=True,
        context_dim=4,
        num_head_channels=16,
    )
    model_management.unload_all_models()


@pytest.fixture
def latent():
    return {
        "samples": torch.linspace(-1, 1, 2 * 4 * 8 * 8).reshape(2, 4, 8, 8),
        "batch_index": [3, 1],
    }


def api(synchronous):
    return v4.ComfyAPISync() if synchronous else v4.eng


async def finish(result):
    return await result if inspect.isawaitable(result) else result


def run_sampler(model, sampler, steps=None, sigmas=(14.6, 4.0, 1.0, 0.0), seed=7):
    """Three steps through ComfyUI's own custom-sampling path."""
    callback = None
    if steps is not None:

        def callback(index, denoised, x, total):
            steps.append((index, denoised.clone(), x.clone(), total))

    return sample.sample_custom(
        model,
        sample.prepare_noise(torch.zeros(1, 4, 8, 8), seed),
        2.0,
        sampler,
        torch.tensor(sigmas),
        POSITIVE,
        NEGATIVE,
        torch.zeros(1, 4, 8, 8),
        callback=callback,
        disable_pbar=True,
        seed=seed,
    )


def euler_written_like_a_pack(
    model, x, sigmas, extra_args=None, callback=None, disable=None, scale=1.0
):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = (x - denoised) / sigmas[i]
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        x = x + d * (sigmas[i + 1] - sigmas[i]) * scale
    return x


class ZeroNoise:
    """A NOISE a pack writes itself: a seed and `generate_noise`."""

    def __init__(self, seed=0):
        self.seed = seed

    def generate_noise(self, input_latent):
        return torch.zeros_like(input_latent["samples"])


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize("name", ["euler", "dpmpp_2m", "ddim", "uni_pc"])
async def test_named_sampler_runs_as_comfyui_runs_the_name(model, name, synchronous):
    selected = await finish(api(synchronous).sampler.named(name))
    steps, native_steps = [], []

    actual = run_sampler(model, selected, steps)
    expected = run_sampler(model, samplers.sampler_object(name), native_steps)

    assert torch.equal(actual, expected)
    torch.testing.assert_close(steps, native_steps, rtol=0, atol=0)
    assert len(steps) == 3


@pytest.mark.asyncio
async def test_named_options_are_copied_and_reach_the_sampler(model):
    options = {"eta": 0.5, "s_noise": 0.75}
    selected = await v4.eng.sampler.named("euler_ancestral", options)
    options["s_noise"] = 9.0

    expected = samplers.ksampler("euler_ancestral", {"eta": 0.5, "s_noise": 0.75})
    assert torch.equal(run_sampler(model, selected), run_sampler(model, expected))
    assert not torch.equal(
        run_sampler(model, selected),
        run_sampler(model, samplers.sampler_object("euler_ancestral")),
    )


@pytest.mark.asyncio
async def test_named_finds_samplers_added_after_import_and_rejects_unknown_names(
    model, monkeypatch
):
    monkeypatch.setattr(
        k_diffusion_sampling,
        "sample_added_later",
        euler_written_like_a_pack,
        raising=False,
    )

    added = await v4.eng.sampler.named("added_later", {"scale": 1.0})

    assert torch.equal(
        run_sampler(model, added), run_sampler(model, samplers.sampler_object("euler"))
    )
    with pytest.raises(AttributeError):
        await v4.eng.sampler.named("missing")


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_custom_loop_written_like_k_diffusion_matches_comfyui_euler(
    model, synchronous
):
    custom = await finish(api(synchronous).sampler.custom(euler_written_like_a_pack))
    steps, native_steps = [], []

    actual = run_sampler(model, custom, steps)
    expected = run_sampler(model, samplers.sampler_object("euler"), native_steps)

    assert torch.equal(actual, expected)
    torch.testing.assert_close(steps, native_steps, rtol=0, atol=0)


@pytest.mark.asyncio
async def test_custom_passes_a_copy_of_options_and_the_runs_seed(model):
    seen = []

    def loop(model, x, sigmas, extra_args, callback, disable, scale):
        seen.append((extra_args["seed"], scale))
        return euler_written_like_a_pack(model, x, sigmas, extra_args, scale=scale)

    options = {"scale": 0.5}
    custom = await v4.eng.sampler.custom(loop, options)
    options["scale"] = 2.0
    run_sampler(model, custom, seed=11)

    assert seen == [(11, 0.5)]


@pytest.mark.asyncio
async def test_custom_runs_an_unsampling_schedule(model):
    custom = await v4.eng.sampler.custom(euler_written_like_a_pack)
    steps = []

    inverted = run_sampler(model, custom, steps, sigmas=(0.5, 1.0, 4.0, 14.6))

    assert len(steps) == 3
    assert torch.isfinite(inverted).all()
    assert inverted.abs().sum() > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_sigmas_are_the_ones_ksampler_uses_for_every_scheduler(
    model, synchronous
):
    group = api(synchronous).sampler
    for scheduler in samplers.KSampler.SCHEDULERS:
        for denoise in (1.0, 0.5):
            sigmas = await finish(group.sigmas(model, scheduler, 4, denoise))
            native = samplers.KSampler(
                model, 4, "cpu", "euler", scheduler, denoise=denoise
            ).sigmas
            assert torch.equal(sigmas, native), (scheduler, denoise)
            assert sigmas.device.type == "cpu"
    assert (await finish(group.sigmas(model, "normal", 4, 0))).numel() == 0


@pytest.mark.asyncio
async def test_sigmas_find_schedulers_added_after_import_and_reject_unknown_names(
    model, monkeypatch
):
    def halves(model_sampling, steps):
        return torch.tensor([2.0**-i for i in range(steps)] + [0.0])

    monkeypatch.setitem(
        samplers.SCHEDULER_HANDLERS, "added_later", samplers.SchedulerHandler(halves)
    )

    assert torch.equal(
        await v4.eng.sampler.sigmas(model, "added_later", 3),
        torch.tensor([1.0, 0.5, 0.25, 0.0]),
    )
    with pytest.raises(ValueError):
        await v4.eng.sampler.sigmas(model, "missing", 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"denoise": 0.6},
        {"start_step": 1, "last_step": 3, "disable_noise": True},
        {"last_step": 2, "force_full_denoise": True, "denoise": 0.8},
    ],
)
async def test_sample_matches_ksampler_and_returns_a_copy(
    model, latent, options, synchronous
):
    original = dict(latent)
    original_samples = latent["samples"].clone()
    arguments = (model, 5, 4, 2.0, "dpmpp_2m", "karras", POSITIVE, NEGATIVE)

    result = await finish(
        api(synchronous).sampler.sample(*arguments, latent, **options)
    )
    (expected,) = nodes.common_ksampler(*arguments, latent, **options)

    assert torch.equal(result["samples"], expected["samples"])
    assert result["batch_index"] == [3, 1]
    assert result is not latent
    assert latent == original
    assert torch.equal(latent["samples"], original_samples)


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_sample_result_has_no_downscale_ratios(model, latent, synchronous):
    latent = {**latent, "downscale_ratio_spacial": 8, "downscale_ratio_temporal": 4}
    arguments = (model, 5, 2, 2.0, "euler", "normal", POSITIVE, NEGATIVE, latent)

    result = await finish(api(synchronous).sampler.sample(*arguments))

    assert set(result) == {"samples", "batch_index"}
    assert latent["downscale_ratio_spacial"] == 8


def test_sample_takes_the_argument_forms_of_common_ksampler(model, latent):
    sync = v4.ComfyAPISync().sampler
    head = (model, 2, 3, 1.5, "euler", "normal", POSITIVE, NEGATIVE, latent)
    named = {
        "model": model,
        "seed": 2,
        "steps": 3,
        "cfg": 1.5,
        "sampler_name": "euler",
        "scheduler": "normal",
        "positive": POSITIVE,
        "negative": NEGATIVE,
        "latent": latent,
    }
    step_options = {
        "disable_noise": False,
        "start_step": 1,
        "last_step": 2,
        "force_full_denoise": True,
    }

    for positional, keywords in (
        (head, {}),
        (head, {"denoise": 0.7}),
        ((), {**named, "denoise": 0.7}),
        (head, {"denoise": 0.7, **step_options}),
    ):
        (expected,) = nodes.common_ksampler(*positional, **keywords)
        result = sync.sample(*positional, **keywords)
        assert torch.equal(result["samples"], expected["samples"])


@pytest.mark.asyncio
async def test_sample_runs_unknown_names_as_the_first_in_comfyuis_lists(model, latent):
    arguments = (model, 1, 2, 1.5)
    conditioning = (POSITIVE, NEGATIVE, latent)

    unknown = await v4.eng.sampler.sample(
        *arguments, "missing", "missing", *conditioning
    )
    first = await v4.eng.sampler.sample(
        *arguments,
        samplers.KSampler.SAMPLERS[0],
        samplers.KSampler.SCHEDULERS[0],
        *conditioning,
    )

    assert torch.equal(unknown["samples"], first["samples"])


@pytest.mark.asyncio
async def test_sample_and_sample_custom_report_each_step_as_progress(
    model, latent, monkeypatch
):
    reported = []
    monkeypatch.setattr(
        comfy.utils,
        "PROGRESS_BAR_HOOK",
        lambda value, total, preview, node_id=None: reported.append((value, total)),
    )

    await v4.eng.sampler.sample(
        model, 1, 2, 1.0, "euler", "normal", POSITIVE, NEGATIVE, latent
    )
    await v4.eng.sampler.sample_custom(
        await v4.eng.noise.random(1),
        await v4.eng.guider.basic(model, POSITIVE),
        await v4.eng.sampler.named("euler"),
        await v4.eng.sampler.sigmas(model, "normal", 2),
        latent,
    )

    assert reported == [(1, 2), (2, 2), (1, 2), (2, 2)]


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_sample_custom_from_stock_parts_matches_ksampler(
    model, latent, synchronous
):
    eng = api(synchronous)
    noise = await finish(eng.noise.random(9))
    guider = await finish(eng.guider.cfg(model, POSITIVE, NEGATIVE, 2.5))
    sampler = await finish(eng.sampler.named("dpmpp_2m"))
    sigmas = await finish(eng.sampler.sigmas(model, "karras", 4, 0.75))

    result = await finish(
        eng.sampler.sample_custom(noise, guider, sampler, sigmas, latent)
    )
    (expected,) = nodes.common_ksampler(
        model, 9, 4, 2.5, "dpmpp_2m", "karras", POSITIVE, NEGATIVE, latent, 0.75
    )

    assert torch.equal(result["samples"], expected["samples"])
    assert result is not latent
    assert result["batch_index"] == [3, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_empty_noise_and_basic_guider_match_ksampler_without_noise_or_cfg(
    model, latent, synchronous
):
    eng = api(synchronous)

    result = await finish(
        eng.sampler.sample_custom(
            await finish(eng.noise.empty()),
            await finish(eng.guider.basic(model, POSITIVE)),
            await finish(eng.sampler.named("euler")),
            await finish(eng.sampler.sigmas(model, "normal", 3, 0.5)),
            latent,
        )
    )
    (expected,) = nodes.common_ksampler(
        model,
        0,
        3,
        1.0,
        "euler",
        "normal",
        POSITIVE,
        NEGATIVE,
        latent,
        0.5,
        disable_noise=True,
    )

    assert torch.equal(result["samples"], expected["samples"])


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_noise_generate_reads_stock_and_pack_noise(latent, synchronous):
    eng = api(synchronous)

    stock = await finish(eng.noise.generate(await finish(eng.noise.random(4)), latent))
    from_pack = await finish(eng.noise.generate(ZeroNoise(), latent))

    assert torch.equal(
        stock, sample.prepare_noise(latent["samples"], 4, latent["batch_index"])
    )
    assert torch.equal(from_pack, torch.zeros(2, 4, 8, 8))


@pytest.mark.asyncio
async def test_sample_custom_runs_pack_noise_and_seeds_the_sampler_with_it(
    model, latent
):
    seeds = []

    def loop(model, x, sigmas, extra_args, callback, disable):
        seeds.append(extra_args["seed"])
        return euler_written_like_a_pack(model, x, sigmas, extra_args)

    sampler = await v4.eng.sampler.custom(loop)
    sigmas = await v4.eng.sampler.sigmas(model, "normal", 3)
    guider = await v4.eng.guider.cfg(model, POSITIVE, NEGATIVE, 2.0)

    from_pack = await v4.eng.sampler.sample_custom(
        ZeroNoise(seed=5), guider, sampler, sigmas, latent
    )
    stock = await v4.eng.sampler.sample_custom(
        await v4.eng.noise.empty(), guider, sampler, sigmas, latent
    )

    assert torch.equal(from_pack["samples"], stock["samples"])
    assert seeds == [5, 0]
