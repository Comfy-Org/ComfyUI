import pytest
import torch

import comfy.sample
import comfy.samplers
import comfy.utils
from comfy import model_management, supported_models
from comfy.ldm.modules import attention
from comfy.model_base import BaseModel, Flux, ModelType
from comfy.model_patcher import ModelPatcher
from comfy_api import v4
from comfy_api.v4._model_hooks import (
    AttentionInfo,
    BlockInfo,
    CfgArgs,
    PostCfgArgs,
    PreCfgArgs,
    UnetArgs,
)

hooks = v4.eng.model


def patcher(native):
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in native.parameters():
            parameter.normal_(0, 0.05, generator=generator)
    return ModelPatcher(native, torch.device("cpu"), torch.device("cpu"))


@pytest.fixture
def unet():
    config = supported_models.SD15(
        {
            "image_size": 32,
            "in_channels": 4,
            "model_channels": 32,
            "out_channels": 4,
            "num_res_blocks": [1, 1],
            "transformer_depth": [1, 1],
            "transformer_depth_output": [1, 1, 1, 1],
            "transformer_depth_middle": 1,
            "channel_mult": [1, 2],
            "use_spatial_transformer": True,
            "context_dim": 16,
            "num_head_channels": 16,
            "use_linear_in_transformer": True,
            "dtype": torch.float32,
        }
    )
    return patcher(BaseModel(config, ModelType.EPS, device=torch.device("cpu")))


@pytest.fixture
def flux():
    config = supported_models.Flux(
        {
            "image_model": "flux",
            "in_channels": 16,
            "out_channels": 16,
            "patch_size": 2,
            "vec_in_dim": 8,
            "context_in_dim": 8,
            "hidden_size": 32,
            "mlp_ratio": 2.0,
            "num_heads": 2,
            "depth": 1,
            "depth_single_blocks": 1,
            "axes_dim": [4, 6, 6],
            "theta": 10000,
            "qkv_bias": True,
            "guidance_embed": False,
            "txt_ids_dims": [],
            "dtype": torch.float32,
        }
    )
    return patcher(Flux(config, device=torch.device("cpu")))


def sample(model, cfg=5.0, noise_mask=None):
    generator = torch.Generator().manual_seed(1)
    channels, context = (16, 8) if isinstance(model.model, Flux) else (4, 16)
    noise = torch.randn(1, channels, 8, 8, generator=generator)
    latent = torch.randn(1, channels, 8, 8, generator=generator)
    positive = [[torch.randn(1, 3, context, generator=generator), {}]]
    negative = [[torch.randn(1, 3, context, generator=generator), {}]]
    if channels == 16:
        positive[0][1]["pooled_output"] = torch.randn(1, 8, generator=generator)
        negative[0][1]["pooled_output"] = torch.randn(1, 8, generator=generator)
    result = comfy.sample.sample(
        model,
        noise,
        2,
        cfg,
        "euler",
        "normal",
        positive,
        negative,
        latent,
        noise_mask=noise_mask,
        disable_pbar=True,
        seed=1,
    )
    return result, latent


def keys(typed_dict):
    return set(typed_dict.__annotations__)


@pytest.mark.asyncio
async def test_cfg_hook_gets_the_declared_args_and_returns_the_guided_prediction(unet):
    seen = []

    def cfg_function(args):
        seen.append(args)
        cond, uncond = args["cond"], args["uncond"]
        assert args["sigma"].shape == (1,) and args["input"].shape == cond.shape
        return uncond

    patched = await hooks.set_cfg(unet, cfg_function)

    torch.testing.assert_close(
        sample(patched)[0], sample(unet, cfg=0.0)[0], rtol=0, atol=0
    )
    assert {frozenset(args) for args in seen} == {frozenset(keys(CfgArgs))}
    assert seen[0]["cond_scale"] == 5.0


@pytest.mark.asyncio
async def test_pre_cfg_hook_replaces_the_model_outputs_that_cfg_combines(unet):
    seen = []

    def pre_cfg_patch(args):
        seen.append(set(args))
        conds_out = args["conds_out"]
        assert args["cond_scale"] == 5.0 and args["sigma"].shape == (1,)
        assert all(out.shape == args["input"].shape for out in conds_out)
        return [conds_out[1], conds_out[1]]

    patched = await hooks.set_pre_cfg(unet, pre_cfg_patch)

    torch.testing.assert_close(
        sample(patched)[0], sample(unet, cfg=0.0)[0], rtol=0, atol=0
    )
    assert seen and all(args == keys(PreCfgArgs) for args in seen)


@pytest.mark.asyncio
async def test_post_cfg_hooks_run_in_order_on_the_denoised_result(unet):
    seen = []

    def post_cfg_function(args):
        seen.append(set(args))
        assert (
            args["sigma"].shape == (1,)
            and args["input"].shape == args["denoised"].shape
        )
        return args["uncond_denoised"]

    def second(args):
        seen.append(torch.equal(args["denoised"], args["uncond_denoised"]))
        return args["denoised"]

    patched = await hooks.set_post_cfg(unet, post_cfg_function)
    patched = await hooks.set_post_cfg(patched, second)

    torch.testing.assert_close(
        sample(patched)[0], sample(unet, cfg=0.0)[0], rtol=0, atol=0
    )
    assert seen == [keys(PostCfgArgs), True] * 2


NEGATIVE = {
    "set_cfg": (lambda args: args["uncond_denoised"], lambda args: args["cond"]),
    "set_pre_cfg": (lambda args: args["conds_out"][1], lambda args: args["conds_out"]),
    "set_post_cfg": (
        lambda args: args["uncond_denoised"],
        lambda args: args["denoised"],
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("call", sorted(NEGATIVE))
@pytest.mark.parametrize("disable", [False, True])
async def test_at_cfg_one_the_negative_is_run_only_when_asked(unet, call, disable):
    negative, result = NEGATIVE[call]
    seen = []

    def custom_cfg_function(args):
        seen.append(bool(negative(args).abs().sum()))
        return result(args)

    patched = await getattr(hooks, call)(
        unet, custom_cfg_function, disable_cfg1_optimization=disable
    )
    sample(patched, cfg=1.0)

    assert seen == [disable] * 2


@pytest.mark.asyncio
async def test_denoise_mask_hook_gets_the_schedule_and_its_mask_is_used(unet):
    seen = []

    def function(sigma, denoise_mask, extra_options):
        seen.append(set(extra_options))
        sigmas = extra_options["sigmas"]
        assert sigmas.shape == (3,) and float(sigma.max()) in sigmas.tolist()
        return torch.zeros_like(denoise_mask)

    patched = await hooks.set_denoise_mask(unet, function)
    sample(patched)
    assert seen == []
    result, latent = sample(patched, noise_mask=torch.ones(1, 1, 8, 8))

    torch.testing.assert_close(result, latent)
    assert seen == [{"sigmas"}] * 2


@pytest.mark.asyncio
async def test_unet_wrapper_calls_the_model_and_its_output_is_used(unet):
    seen = []

    sigmas = comfy.samplers.calculate_sigmas(
        unet.get_model_object("model_sampling"), "normal", 2
    ).tolist()

    def unet_wrapper(model_function, kwargs):
        seen.append(set(kwargs))
        assert kwargs["cond_or_uncond"] == [1, 0]
        assert float(kwargs["timestep"][0]) in sigmas
        return model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])

    def apply_cdtuner(model_function, kwargs):
        return model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"]) * 2

    baseline = sample(unet)[0]
    passed = await hooks.set_unet_wrapper(unet, unet_wrapper)
    doubled = await hooks.set_unet_wrapper(unet, apply_cdtuner)

    torch.testing.assert_close(sample(passed)[0], baseline, rtol=0, atol=0)
    assert not torch.allclose(sample(doubled)[0], baseline)
    assert seen and all(args == keys(UnetArgs) for args in seen)


@pytest.mark.asyncio
async def test_a_wrapper_already_on_the_model_runs_inside_the_new_one(unet):
    order = []

    def existing(apply_model, args):
        order.append(("existing", args["cond_or_uncond"], set(args["c"])))
        return apply_model(args["input"], args["timestep"], **args["c"]) * 2

    def wrapper(apply_model, args):
        order.append(("new", args["cond_or_uncond"], set(args["c"])))
        return apply_model(args["input"], args["timestep"], **args["c"])

    first = unet.clone()
    first.set_model_unet_function_wrapper(existing)
    patched = await hooks.set_unet_wrapper(first, wrapper)

    torch.testing.assert_close(sample(patched)[0], sample(first)[0], rtol=0, atol=0)
    (new, *passed), (inner, *received) = order[:2]
    assert (new, inner) == ("new", "existing") and passed == received


@pytest.mark.asyncio
@pytest.mark.parametrize("set_by", ["ComfyUI", "V4"])
async def test_a_wrapper_already_on_the_model_is_moved_loaded_and_cleaned_up_as_alone(
    unet, monkeypatch, set_by
):
    extra = ModelPatcher(
        torch.nn.Linear(1, 1), torch.device("cpu"), torch.device("cpu")
    )

    class HoldsAModel:
        def __init__(self, events, device=None):
            self.events, self.device = events, device

        def __call__(self, apply_model, args):
            self.events.append(("called", self.device))
            return apply_model(args["input"], args["timestep"], **args["c"])

        def to(self, device):
            self.events.append(("moved", device))
            return HoldsAModel(self.events, device)

        def models(self):
            return [extra]

        def cleanup(self):
            self.events.append(("cleaned", self.device))

    async def run(nested):
        monkeypatch.setattr(model_management, "current_loaded_models", [])
        events = []
        model = unet.clone()
        if set_by == "ComfyUI":
            model.set_model_unet_function_wrapper(HoldsAModel(events))
        else:
            model = await hooks.set_unet_wrapper(model, HoldsAModel(events))
        if nested:
            model = await hooks.set_unet_wrapper(model, CALLS["set_unet_wrapper"])
        sample(model)
        loaded = [entry.model for entry in model_management.current_loaded_models]
        return list(events), extra in loaded

    alone = await run(nested=False)

    assert alone[1] and {kind for kind, _ in alone[0]} == {"moved", "called", "cleaned"}
    assert await run(nested=True) == alone


class LLLitePatch:
    def __init__(self, scale):
        self.scale = scale
        self.moved = []
        self.seen = []

    def __call__(self, q, k, v, extra_options):
        self.seen.append(set(extra_options))
        block, block_index = extra_options["block"], extra_options["block_index"]
        assert block[0] in ("input", "middle", "output") and block_index == 0
        return q, k * self.scale, v * self.scale

    def to(self, device):
        self.moved.append(device)
        return self


@pytest.mark.asyncio
@pytest.mark.parametrize("call", ["set_attn1_patch", "set_attn2_patch"])
async def test_attention_patch_gets_block_info_and_its_inputs_are_used(unet, call):
    baseline = sample(unet)[0]
    same, scaled = LLLitePatch(1.0), LLLitePatch(0.0)

    torch.testing.assert_close(
        sample(await getattr(hooks, call)(unet, same))[0], baseline, rtol=0, atol=0
    )
    assert not torch.allclose(
        sample(await getattr(hooks, call)(unet, scaled))[0], baseline
    )
    assert same.seen and all(info == keys(AttentionInfo) for info in same.seen)
    assert torch.device("cpu") in same.moved


@pytest.mark.asyncio
async def test_a_patch_objects_models_are_loaded_and_it_is_cleaned_up(
    unet, monkeypatch
):
    monkeypatch.setattr(model_management, "current_loaded_models", [])
    extra = ModelPatcher(
        torch.nn.Linear(1, 1), torch.device("cpu"), torch.device("cpu")
    )
    cleaned = []

    class HoldsAModel(LLLitePatch):
        def models(self):
            return [extra]

        def cleanup(self):
            cleaned.append(True)

    sample(await hooks.set_attn1_patch(unet, HoldsAModel(1.0)))

    assert extra in [loaded.model for loaded in model_management.current_loaded_models]
    assert cleaned


@pytest.mark.asyncio
@pytest.mark.parametrize("call", ["set_attn1_output_patch", "set_attn2_output_patch"])
async def test_attention_output_patch_result_replaces_the_output(unet, call):
    seen = []

    def attn2_output_patch(out, extra_options):
        seen.append(set(extra_options))
        batch = len(extra_options["cond_or_uncond"])
        assert extra_options["original_shape"][0] == batch == out.shape[0]
        return out * 0

    patched = await getattr(hooks, call)(unet, attn2_output_patch)

    assert not torch.allclose(sample(patched)[0], sample(unet)[0])
    assert seen and all(info == keys(AttentionInfo) for info in seen)


@pytest.mark.asyncio
async def test_block_patches_get_block_info_and_their_features_are_used(unet):
    seen = []

    def block_patch(h, transformer_options):
        seen.append(set(transformer_options))
        assert transformer_options["block"][0] in ("input", "output")
        return h

    def block_patch_hsp(h, hsp, transformer_options):
        h = block_patch(h, transformer_options)
        return h, hsp * 0

    baseline = sample(unet)[0]
    same = await hooks.set_input_block_patch(unet, block_patch)
    changed = await hooks.set_output_block_patch(same, block_patch_hsp)

    torch.testing.assert_close(sample(same)[0], baseline, rtol=0, atol=0)
    assert not torch.allclose(sample(changed)[0], baseline)
    assert seen and all(info == keys(BlockInfo) for info in seen)


@pytest.mark.asyncio
async def test_each_attention_patch_runs_at_its_own_attention(unet):
    order = []

    def record(name):
        def patch(*args):
            order.append(name)
            return args[0] if len(args) == 2 else args[:3]

        return patch

    patched = unet
    for call in (
        "set_attn2_output_patch",
        "set_attn2_patch",
        "set_attn1_output_patch",
        "set_attn1_patch",
    ):
        patched = await getattr(hooks, call)(patched, record(call))
    sample(patched)

    assert order[:4] == [
        "set_attn1_patch",
        "set_attn1_output_patch",
        "set_attn2_patch",
        "set_attn2_output_patch",
    ]


@pytest.mark.asyncio
async def test_input_block_patch_result_is_what_the_skip_connection_carries(unet):
    skips = []

    def output_block_patch(h, skip, extra_options):
        skips.append(float(skip.abs().sum()))
        return h, skip

    patched = await hooks.set_input_block_patch(unet, lambda h, extra_options: h * 0)
    sample(await hooks.set_output_block_patch(patched, output_block_patch))

    assert skips and not any(skips)


@pytest.mark.asyncio
async def test_unet_hooks_are_left_off_models_that_are_not_unets(flux):
    called = []

    def minimax_attn1_patch(q, k, v, extra_options):
        called.append(True)
        return q, k, v

    patched = flux
    for call in ("set_attn1_patch", "set_attn2_patch"):
        patched = await getattr(hooks, call)(patched, minimax_attn1_patch)
    sample(patched)

    assert called == []
    assert "patches" not in patched.model_options["transformer_options"]


class AttentionOverride:
    def __init__(self, scale):
        self.scale = scale
        self.heads = []

    def __call__(
        self,
        original,
        query,
        key,
        value,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        self.heads.append(heads)
        return original(
            query * self.scale,
            key,
            value,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )


@pytest.mark.asyncio
async def test_attention_override_falls_back_to_the_attention_it_is_given(unet):
    baseline = sample(unet)[0]
    same, scaled = AttentionOverride(1.0), AttentionOverride(0.0)

    torch.testing.assert_close(
        sample(await hooks.set_attention_override(unet, same))[0],
        baseline,
        rtol=0,
        atol=0,
    )
    assert not torch.allclose(
        sample(await hooks.set_attention_override(unet, scaled))[0], baseline
    )
    assert same.heads


@pytest.mark.asyncio
async def test_an_override_gets_options_a_model_passes_by_position_as_keywords(unet):
    seen = {}

    def override(original, q, k, v, heads, **options):
        seen.update(options)
        return original(q, k, v, heads, **options)

    patched = await hooks.set_attention_override(unet, override)
    transformer_options = patched.model_options["transformer_options"]
    q, mask = torch.rand(1, 4, 8), torch.zeros(1, 4, 4)

    result = attention.optimized_attention_masked(
        q, q, q, 2, mask, None, False, transformer_options=transformer_options
    )

    expected = attention.optimized_attention_masked(q, q, q, 2, mask)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert seen["mask"] is mask
    assert seen["attn_precision"] is None and seen["skip_reshape"] is False


@pytest.mark.asyncio
async def test_an_override_may_call_comfyui_attention_with_the_options_it_got(unet):
    calls = []

    def sage_override(original, q, k, v, heads, **kwargs):
        calls.append(heads)
        return attention.attention_pytorch(q, k, v, heads, **kwargs)

    patched = await hooks.set_attention_override(unet, sage_override)

    torch.testing.assert_close(
        sample(patched)[0], sample(unet)[0], rtol=1e-4, atol=1e-4
    )
    assert calls


@pytest.mark.asyncio
async def test_an_override_already_on_the_model_runs_inside_the_new_one(unet):
    order = []

    def previous(func, *args, **kwargs):
        order.append("previous")
        return func(*args, **kwargs)

    def installed(original, *args, **kwargs):
        order.append("new")
        return original(*args, **kwargs)

    first = unet.clone()
    first.model_options["transformer_options"]["optimized_attention_override"] = (
        previous
    )
    patched = await hooks.set_attention_override(first, installed)

    torch.testing.assert_close(sample(patched)[0], sample(unet)[0], rtol=0, atol=0)
    assert order[:2] == ["new", "previous"]


CALLS = {
    "set_cfg": lambda args: args["cond"],
    "set_pre_cfg": lambda args: args["conds_out"],
    "set_post_cfg": lambda args: args["denoised"],
    "set_denoise_mask": lambda sigma, denoise_mask, extra_options: denoise_mask,
    "set_unet_wrapper": lambda apply, args: apply(
        args["input"], args["timestep"], **args["c"]
    ),
    "set_attn1_patch": lambda q, k, v, extra_options: (q, k, v),
    "set_attn2_patch": lambda q, k, v, extra_options: (q, k, v),
    "set_attn1_output_patch": lambda out, extra_options: out,
    "set_attn2_output_patch": lambda out, extra_options: out,
    "set_input_block_patch": lambda h, extra_options: h,
    "set_output_block_patch": lambda h, skip, extra_options: (h, skip),
    "set_attention_override": lambda attention, *args, **kwargs: attention(
        *args, **kwargs
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("call", sorted(CALLS))
async def test_each_call_returns_a_patched_copy_in_both_forms_and_leaves_the_model(
    unet, call
):
    unet.set_model_sampler_post_cfg_function(CALLS["set_post_cfg"])
    unet.set_model_attn1_patch(CALLS["set_attn1_patch"])
    before = comfy.utils.deepcopy_list_dict(unet.model_options)

    patched = await getattr(hooks, call)(unet, CALLS[call])
    synced = getattr(v4.ComfyAPISync().model, call)(unet, CALLS[call])

    assert unet.model_options == before
    for copy in (patched, synced):
        assert copy is not unet and copy.model is unet.model
        assert copy.model_options != before
    torch.testing.assert_close(sample(synced)[0], sample(patched)[0], rtol=0, atol=0)
