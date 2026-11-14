import copy
import json
from pathlib import Path

import pytest
import torch

import comfy.utils
from comfy import hooks, model_management, sd1_clip
from comfy.sd import CLIP
from comfy_api import v4
from comfy_api.v4._clip import Clip


@pytest.fixture
def clip():
    config = json.loads(
        (Path(sd1_clip.__file__).parent / "sd1_clip_config.json").read_text()
    )
    config.update(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_hidden_layers=3,
        projection_dim=16,
    )

    class TinyClipModel(sd1_clip.SDClipModel):
        def __init__(self, device, dtype, model_options):
            super().__init__(
                device=device,
                dtype=dtype,
                model_options=model_options,
                textmodel_json_config=config,
                return_attention_masks=True,
            )

    class Target:
        params = {"clip_model": TinyClipModel}
        clip = sd1_clip.SD1ClipModel
        tokenizer = sd1_clip.SD1Tokenizer

    encoder = CLIP(
        Target(),
        model_options={
            "load_device": torch.device("cpu"),
            "offload_device": torch.device("cpu"),
            "dtype": torch.float32,
        },
    )
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for parameter in encoder.cond_stage_model.parameters():
            parameter.normal_(0, 0.02, generator=generator)
    yield encoder
    model_management.unload_all_models()


@pytest.mark.asyncio
async def test_tokenize_preserves_weighting_and_word_ids(clip):
    text = "a (red:1.5) cat"
    tokens = await Clip().tokenize(clip, text, return_word_ids=True)
    assert tokens == clip.tokenize(text, return_word_ids=True)
    assert any(weight == 1.5 for _, weight, _ in tokens["l"][0])
    assert any(word_id > 0 for _, _, word_id in tokens["l"][0])


@pytest.mark.asyncio
async def test_tokenize_gives_the_tokenizer_only_the_options_passed(clip, monkeypatch):
    received = []
    tokenize = clip.tokenizer.tokenize_with_weights

    def recording(text, return_word_ids=False, **options):
        received.append(options)
        return tokenize(text, return_word_ids, **options)

    monkeypatch.setattr(clip.tokenizer, "tokenize_with_weights", recording)
    image = torch.zeros(1, 4, 4, 3)

    await Clip().tokenize(clip, "cat", images=[image], llama_template="{}")
    await Clip().tokenize(clip, "cat")

    assert received == [{"images": [image], "llama_template": "{}"}, {}]


@pytest.mark.asyncio
@pytest.mark.parametrize("pooled", [False, True, "unprojected"])
@pytest.mark.parametrize("as_dict", [False, True])
async def test_unscheduled_encoding_preserves_return_modes_and_tokens(
    clip, pooled, as_dict
):
    tokens = await Clip().tokenize(clip, "a (red:1.5) cat")
    original = copy.deepcopy(tokens)
    actual = await Clip().encode_from_tokens(
        clip, tokens, return_pooled=pooled, return_dict=as_dict
    )
    expected = clip.encode_from_tokens(tokens, pooled, as_dict)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert tokens == original
    if as_dict:
        assert {"cond", "pooled_output"} <= actual.keys()
    elif pooled:
        assert isinstance(actual, tuple)
        assert len(actual) == 2
    else:
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == (1, 77, 16)


@pytest.mark.asyncio
async def test_edited_tokens_and_unprojected_pooled_output_are_used(clip):
    tokens = await Clip().tokenize(clip, "a (red:1.5) cat")
    first, projected = await Clip().encode_from_tokens(clip, tokens, return_pooled=True)
    _, unprojected = await Clip().encode_from_tokens(
        clip, tokens, return_pooled="unprojected"
    )
    assert not torch.equal(projected, unprojected)

    tokens["l"][0][2] = (tokens["l"][0][2][0], 0.0)
    changed = await Clip().encode_from_tokens(clip, tokens)
    assert not torch.equal(first, changed)


@pytest.mark.asyncio
async def test_scheduled_encoding_preserves_metadata_without_consuming_it(clip):
    tokens = await Clip().tokenize(clip, "red cat")
    metadata = {"strength": 0.75, "start_percent": 0.2}
    actual = await Clip().encode_tokens(
        clip, tokens, unprojected=True, add_dict=metadata, show_pbar=False
    )
    expected = clip.encode_from_tokens_scheduled(
        tokens, unprojected=True, add_dict=metadata, show_pbar=False
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert metadata == {"strength": 0.75, "start_percent": 0.2}
    assert actual[0][1] is not metadata
    assert "attention_mask" in actual[0][1]


@pytest.mark.asyncio
async def test_scheduled_encoding_honors_hook_intervals_and_range(clip):
    hook = hooks.WeightHook()
    hook.hook_keyframe.add(hooks.HookKeyframe(1.0, start_percent=0.0))
    hook.hook_keyframe.add(hooks.HookKeyframe(0.5, start_percent=0.5))
    group = hooks.HookGroup()
    group.add(hook)
    clip.patcher.forced_hooks = group
    clip.use_clip_schedule = True
    tokens = await Clip().tokenize(clip, "cat")

    actual = await Clip().encode_tokens(clip, tokens, show_pbar=False)
    assert [
        (entry[1]["clip_start_percent"], entry[1]["clip_end_percent"])
        for entry in actual
    ] == [(0.0, 0.5), (0.5, 1.0)]
    filtered = await Clip().encode_tokens(
        clip, tokens, add_dict={"start_percent": 0.75}, show_pbar=False
    )
    assert len(filtered) == 1
    torch.testing.assert_close(filtered[0][0], actual[1][0], rtol=0, atol=0)
    assert hook.hook_keyframe.strength == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("show_pbar", [False, True])
async def test_scheduled_encoding_reports_progress_only_when_asked(
    clip, monkeypatch, show_pbar
):
    reported = []
    monkeypatch.setattr(
        comfy.utils,
        "PROGRESS_BAR_HOOK",
        lambda value, total, preview, node_id=None: reported.append((value, total)),
    )
    hook = hooks.WeightHook()
    hook.hook_keyframe.add(hooks.HookKeyframe(1.0, start_percent=0.0))
    hook.hook_keyframe.add(hooks.HookKeyframe(0.5, start_percent=0.5))
    clip.patcher.forced_hooks = hooks.HookGroup()
    clip.patcher.forced_hooks.add(hook)
    clip.use_clip_schedule = True

    tokens = await Clip().tokenize(clip, "cat")

    await Clip().encode_tokens(clip, tokens, show_pbar=show_pbar)

    assert reported == ([(1, 2), (2, 2)] if show_pbar else [])


@pytest.mark.asyncio
async def test_clone_preserves_weights_but_has_independent_settings(clip):
    cloned = await Clip().clone(clip)
    assert cloned is not clip
    assert cloned.patcher is not clip.patcher
    assert cloned.cond_stage_model is clip.cond_stage_model
    cloned.clip_layer(-2)
    cloned.set_tokenizer_option("max_length", 120)
    assert await Clip().info(cloned) == {"layer_idx": -2}
    assert await Clip().info(clip) == {"layer_idx": None}
    assert clip.tokenizer_options == {}

    tokens = await Clip().tokenize(clip, "cat")
    original = await Clip().encode_from_tokens(clip, tokens)
    layered = await Clip().encode_from_tokens(cloned, tokens)
    assert not torch.equal(layered, original)
    torch.testing.assert_close(
        await Clip().encode_from_tokens(clip, tokens), original, rtol=0, atol=0
    )


@pytest.mark.asyncio
async def test_patching_clones_encoder_normalizes_lists_and_preserves_strengths(clip):
    key = "clip_l.transformer.text_model.final_layer_norm.bias"
    with torch.no_grad():
        clip.cond_stage_model.get_parameter(key).fill_(0.125)
    base = (await Clip().read_weight(clip, key)).clone()
    delta = torch.ones_like(base)
    patches = {key: [delta], "missing.weight": [torch.ones(1)]}
    patched = await Clip().add_patches(clip, patches, 0.5, 0.25)
    expected = clip.clone()
    expected.add_patches({key: (delta,)}, 0.5, 0.25)

    assert patched is not clip
    assert isinstance(patches[key], list)
    assert not clip.patcher.patches
    assert list(patched.patcher.patches) == [key]
    weight = await Clip().read_weight(patched, key)
    torch.testing.assert_close(
        weight, expected.patcher.patch_weight_to_device(key, return_weight=True)
    )
    torch.testing.assert_close(weight, base * 0.25 + 0.5, rtol=0, atol=0)
    torch.testing.assert_close(
        await Clip().read_weight(clip, key), base, rtol=0, atol=0
    )


@pytest.mark.asyncio
async def test_weight_inspection_preserves_native_storage_and_detaches(clip):
    keys = await Clip().weight_keys(clip)
    assert keys == sorted(clip.patcher.model_state_dict())
    key = "clip_l.transformer.text_model.final_layer_norm.bias"
    clip.cond_stage_model.get_parameter(key).requires_grad_(True)
    parameter = clip.patcher.model_state_dict()[key]
    weight = await Clip().read_weight(clip, key)
    assert weight.data_ptr() == parameter.data_ptr()
    assert not weight.requires_grad
    weight_copy = weight.clone()
    weight_copy.add_(1)
    assert not torch.equal(weight_copy, parameter)
    with pytest.raises(AttributeError):
        await Clip().read_weight(clip, "missing.weight")


@pytest.mark.asyncio
async def test_patch_key_offsets_only_change_the_selected_slice(clip):
    key = "clip_l.transformer.text_model.final_layer_norm.bias"
    with torch.no_grad():
        clip.cond_stage_model.get_parameter(key).fill_(0.125)
    original = (await Clip().read_weight(clip, key)).clone()
    patched = await Clip().add_patches(clip, {(key, (0, 2, 3)): [torch.ones(3)]})
    expected = original.clone()
    expected[2:5] += 1
    torch.testing.assert_close(
        await Clip().read_weight(patched, key), expected, rtol=0, atol=0
    )
    torch.testing.assert_close(
        await Clip().read_weight(clip, key), original, rtol=0, atol=0
    )


@pytest.mark.asyncio
async def test_public_sync_and_async_clip_produce_the_same_conditioning(clip):
    sync = v4.ComfyAPISync().clip
    tokens = sync.tokenize(clip, "a (red:1.5) cat")
    expected = await v4.eng.clip.encode_tokens(clip, tokens, show_pbar=False)
    actual = sync.encode_tokens(clip, tokens, show_pbar=False)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[0][0].shape == (1, 77, 16)


@pytest.mark.asyncio
async def test_the_sync_form_of_each_clip_call_gives_what_awaiting_it_gives(clip):
    sync, api = v4.ComfyAPISync().clip, v4.eng.clip
    key = (await api.weight_keys(clip))[0]
    tokens = await api.tokenize(clip, "a cat")
    patches = {key: (torch.ones_like(await api.read_weight(clip, key)),)}

    assert sync.info(clip) == await api.info(clip)
    assert sync.weight_keys(clip) == await api.weight_keys(clip)
    assert torch.equal(sync.read_weight(clip, key), await api.read_weight(clip, key))
    assert torch.equal(
        sync.encode_from_tokens(clip, tokens),
        await api.encode_from_tokens(clip, tokens),
    )
    assert sync.clone(clip).patcher.is_clone(clip.patcher)
    assert torch.equal(
        sync.read_weight(sync.add_patches(clip, patches), key),
        await api.read_weight(await api.add_patches(clip, patches), key),
    )
    assert not clip.patcher.patches
