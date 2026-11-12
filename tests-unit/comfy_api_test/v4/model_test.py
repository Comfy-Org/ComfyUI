import copy
import json

import pytest
import torch

from clip_test import clip as clip_fixture
from comfy import latent_formats, model_sampling
from comfy_api import v4
from comfy_api.v4._model import Model


clip = clip_fixture


@pytest.fixture
def model(tiny_sd15):
    return tiny_sd15()


@pytest.mark.asyncio
async def test_clone_separates_patch_settings_while_sharing_weights(model):
    clone = await Model().clone(model)
    assert clone is not model
    assert clone.model is model.model
    clone.add_object_patch("latent_format", latent_formats.SDXL())
    assert not model.object_patches
    assert clone.get_model_object("latent_format").scale_factor == 0.13025
    assert model.get_model_object("latent_format").scale_factor == 0.18215


@pytest.mark.asyncio
async def test_patch_strengths_offset_and_source_ownership(model):
    key = "diffusion_model.out.0.bias"
    with torch.no_grad():
        model.model.get_parameter(key).fill_(0.125)
    original = model.model.get_parameter(key).clone()
    delta = torch.ones(3)
    patches = {(key, (0, 2, 3)): [delta], "missing": [torch.ones(1)]}
    clone = await Model().add_patches(model, patches, 0.5, 0.25)
    expected = original.clone()
    expected[2:5] = expected[2:5] * 0.25 + 0.5
    torch.testing.assert_close(
        await Model().read_weight(clone, key), expected, rtol=0, atol=0
    )
    torch.testing.assert_close(model.model.get_parameter(key), original, rtol=0, atol=0)
    assert not model.patches
    assert isinstance(patches[(key, (0, 2, 3))], list)


@pytest.mark.asyncio
@pytest.mark.parametrize("both", [True, False])
async def test_native_low_rank_lora_preserves_sources_and_metadata(model, clip, both):
    key = "diffusion_model.input_blocks.0.0.weight"
    original = model.model.get_parameter(key).clone()
    up = torch.ones(32, 2, 1, 1)
    down = torch.ones(2, 4, 3, 3) * 0.5
    weights = {
        "lora_unet_input_blocks_0_0.lora_up.weight": up,
        "lora_unet_input_blocks_0_0.lora_down.weight": down,
        "lora_unet_input_blocks_0_0.alpha": torch.tensor(2.0),
    }
    metadata = {"ss_network_module": "networks.lora", "description": "synthetic"}
    before = copy.deepcopy(weights)
    patched, text_encoder = await Model().load_lora(
        model, clip if both else None, weights, 0.5, 0.25, metadata
    )
    assert patched is not model
    assert (text_encoder is not clip) if both else text_encoder is None
    assert patched.model is model.model
    assert not model.patches
    assert not clip.patcher.patches
    torch.testing.assert_close(weights, before, rtol=0, atol=0)
    torch.testing.assert_close(model.model.get_parameter(key), original, rtol=0, atol=0)
    expected = original.float() + 0.5
    actual = await Model().read_weight(patched, key)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.005)
    assert patched.get_attachment("lora_metadata") == metadata
    assert patched.get_attachment("lora_metadata") is not metadata
    if both:
        assert text_encoder.patcher.get_attachment("lora_metadata") == metadata


@pytest.mark.asyncio
async def test_lora_can_patch_only_clip_and_leaves_absent_model_absent(clip):
    key = "clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"
    original = clip.cond_stage_model.get_parameter(key).clone()
    weights = {
        "lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight": torch.ones(
            16, 2
        ),
        "lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": torch.ones(
            2, 16
        ),
    }
    model, patched = await Model().load_lora(None, clip, weights, strength_clip=0.25)
    assert model is None
    assert patched is not clip
    assert not clip.patcher.patches
    torch.testing.assert_close(
        patched.patcher.patch_weight_to_device(key, return_weight=True),
        original + 0.5,
        rtol=0.01,
        atol=0.005,
    )


@pytest.mark.asyncio
async def test_lora_in_a_format_comfyui_renames_leaves_the_callers_keys(model):
    name = "lora_unet__blocks_0_cross_attn_k.lora_down.weight"
    weights = {name: torch.ones(2, 4)}

    await Model().load_lora(model, None, weights)

    assert list(weights) == [name]


@pytest.mark.asyncio
async def test_empty_lora_and_missing_inputs_follow_native_behavior(model):
    assert await Model().load_lora(None, None, {}) == (None, None)
    copied, missing = await Model().load_lora(model, None, {}, lora_metadata={})
    assert copied is not model
    assert missing is None
    assert not copied.patches
    assert copied.get_attachment("lora_metadata") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [False, True])
async def test_latent_round_trip_honors_selected_format_and_preserves_input(
    model, override
):
    if override:
        model.add_object_patch("latent_format", latent_formats.SDXL_Playground_2_5())
    latent = torch.linspace(-2, 2, 4 * 3 * 5).reshape(1, 4, 3, 5).transpose(2, 3)
    original = latent.clone()
    selected = model.get_model_object("latent_format")
    scaled = await Model().latent_process_in(model, latent)
    torch.testing.assert_close(scaled, selected.process_in(latent), rtol=0, atol=0)
    torch.testing.assert_close(
        await Model().latent_process_out(model, scaled), original
    )
    torch.testing.assert_close(latent, original, rtol=0, atol=0)


@pytest.mark.asyncio
async def test_sampling_round_trip_and_percent_endpoints(model):
    timesteps = torch.tensor([0.0, 1.0, 200.0, 999.0])
    sigma = await Model().sigma(model, timesteps)
    torch.testing.assert_close(await Model().timestep(model, sigma), timesteps.long())
    assert await Model().percent_to_sigma(model, 0.0) == 999999999.9
    assert await Model().percent_to_sigma(model, 1.0) == 0.0
    assert await Model().percent_to_sigma(model, 0.25) > await Model().percent_to_sigma(
        model, 0.75
    )


@pytest.mark.asyncio
async def test_info_and_schedule_use_the_same_object_patch(model):
    original = await Model().info(model)
    clone = await Model().clone(model)
    selected = model_sampling.ModelSamplingDiscrete(zsnr=True)
    selected.set_sigmas(torch.tensor([0.25, 0.5, 1.0]))
    clone.add_object_patch("model_sampling", selected)
    clone.add_object_patch("latent_format", latent_formats.SD3())
    info = await Model().info(clone)
    assert info["sigma_min"] == 0.25
    assert info["sigma_max"] == 1.0
    assert info["latent_scale_factor"] == 1.5305
    assert (info["latent_channels"], info["zsnr"], original["zsnr"]) == (
        16,
        True,
        False,
    )
    assert info["dtype"] == "float32"
    assert info["model_type"] == "EPS"
    assert await Model().info(model) == original
    torch.testing.assert_close(
        await Model().sigma(clone, torch.tensor([0.0, 2.0])), torch.tensor([0.25, 1.0])
    )
    json.dumps(info)


@pytest.mark.asyncio
async def test_weight_keys_and_detached_read(model):
    keys = await Model().weight_keys(model)
    assert keys == sorted(model.model_state_dict())
    assert "model_sampling.sigmas" in keys
    key = "diffusion_model.out.0.bias"
    parameter = model.model.get_parameter(key)
    parameter.requires_grad_(True)
    value = await Model().read_weight(model, key)
    torch.testing.assert_close(value, parameter)
    assert value.data_ptr() == parameter.data_ptr()
    assert not value.requires_grad
    with pytest.raises(AttributeError):
        await Model().read_weight(model, "absent.weight")


def same(actual, expected):
    if isinstance(expected, torch.Tensor):
        return torch.equal(actual, expected)
    return actual == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call, arguments",
    [
        ("info", ()),
        ("weight_keys", ()),
        ("percent_to_sigma", (0.25,)),
        ("sigma", (torch.tensor([1.0, 500.0]),)),
        ("timestep", (torch.tensor([0.5, 2.0]),)),
        ("latent_process_in", (torch.ones(1, 4, 2, 2),)),
        ("latent_process_out", (torch.ones(1, 4, 2, 2),)),
    ],
)
async def test_the_sync_form_of_a_call_gives_what_awaiting_it_gives(
    model, call, arguments
):
    expected = await getattr(v4.eng.model, call)(model, *arguments)

    assert same(getattr(v4.ComfyAPISync().model, call)(model, *arguments), expected)


@pytest.mark.asyncio
async def test_the_sync_forms_of_clone_and_add_patches_return_patched_clones(model):
    key = next(iter(model.model_state_dict()))
    patches = {key: (torch.ones_like(model.model_state_dict()[key]),)}
    sync = v4.ComfyAPISync().model

    clone = sync.clone(model)
    patched = sync.add_patches(model, patches)

    assert clone is not model and clone.is_clone(model)
    assert same(
        sync.read_weight(patched, key),
        await v4.eng.model.read_weight(
            await v4.eng.model.add_patches(model, patches), key
        ),
    )
    assert not model.patches
