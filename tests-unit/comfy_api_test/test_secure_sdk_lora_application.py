"""Real SDK LoRA policies with optional, locally provided original model files.

Set COMFY_TEST_LORA_CHECKPOINT, COMFY_TEST_LORA_CLIP and COMFY_TEST_LORA_FILE
to run the original-model regressions. These tests never download weights.
"""
import asyncio
import os
from pathlib import Path

import pytest
import torch

from comfy_api.latest import _sdk


@pytest.fixture(scope="module")
def original_lora():
    names = ("COMFY_TEST_LORA_CHECKPOINT", "COMFY_TEST_LORA_CLIP", "COMFY_TEST_LORA_FILE")
    if not all(os.environ.get(name) for name in names):
        pytest.skip("Local original checkpoint, CLIP and LoRA fixtures are required")
    import comfy.sd
    import comfy.utils
    import folder_paths

    checkpoint, clip_path, lora_path = (Path(os.environ[name]) for name in names)
    model = comfy.sd.load_checkpoint_guess_config(
        str(checkpoint), output_vae=False, output_clip=False, embedding_directory=[])[0]
    clip = comfy.sd.load_clip(
        ckpt_paths=[str(clip_path)], embedding_directory=[],
        clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,
        model_options={"load_device": torch.device("cpu"), "offload_device": torch.device("cpu")})
    previous = folder_paths.folder_names_and_paths.get("loras")
    folder_paths.folder_names_and_paths["loras"] = ([str(lora_path.parent)], {".safetensors"})
    state = comfy.utils.load_torch_file(str(lora_path), safe_load=True)
    yield model, clip, lora_path, state
    if previous is None:
        folder_paths.folder_names_and_paths.pop("loras", None)
    else:
        folder_paths.folder_names_and_paths["loras"] = previous


async def handles(original, with_clip=True):
    model, clip, path, _ = original
    refs = _sdk.InProcessRefResolver()
    model_ref = _sdk.ModelRef._wrap(await refs.create("MODEL", model))
    clip_ref = _sdk.ClipRef._wrap(await refs.create("CLIP", clip)) if with_clip else None
    asset = _sdk.AssetRef._wrap(await refs.create("ASSET", path))
    return refs, model_ref, clip_ref, asset


@pytest.mark.parametrize("options", [{}, {"skip_zero": True, "allow_extended_strengths": False}])
def test_default_zero_preserves_original_refs(original_lora, options):
    async def run():
        refs, model, clip, asset = await handles(original_lora)
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            actual = await model.apply_lora(asset, clip, 0., 0., **options)
        assert actual == (model, clip)
        assert await refs.resolve(actual[0]) is original_lora[0]
        assert await refs.resolve(actual[1]) is original_lora[1]
        refs.clear()
    asyncio.run(run())


@pytest.mark.parametrize("with_clip", [False, True])
def test_explicit_zero_loads_and_clones_like_native(original_lora, with_clip):
    import comfy.sd

    async def run():
        refs, model, clip, asset = await handles(original_lora, with_clip)
        expected_model, expected_clip = comfy.sd.load_lora_for_models(
            original_lora[0], original_lora[1] if with_clip else None, original_lora[3], 0., 0.)
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            actual_model, actual_clip = await model.apply_lora(asset, clip, 0., 0., skip_zero=False)
        resolved = await refs.resolve(actual_model)
        assert resolved is not original_lora[0]
        assert resolved.patches.keys() == expected_model.patches.keys()
        if with_clip:
            resolved_clip = await refs.resolve(actual_clip)
            assert resolved_clip is not original_lora[1]
            assert resolved_clip.patcher.patches.keys() == expected_clip.patcher.patches.keys()
        else:
            assert actual_clip is None and expected_clip is None
        refs.clear()
    asyncio.run(run())


@pytest.mark.parametrize("strength", [101., -101., 1e6, float("inf"), float("-inf"), float("nan")])
def test_explicit_strength_matches_native_weight(original_lora, strength):
    import comfy.sd

    async def run():
        refs, model, _, asset = await handles(original_lora, False)
        expected, _ = comfy.sd.load_lora_for_models(
            original_lora[0], None, original_lora[3], strength, 0.)
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            actual, actual_clip = await model.apply_lora(
                asset, None, strength, 0., allow_extended_strengths=True)
        actual = await refs.resolve(actual)
        assert actual_clip is None
        key = next(iter(expected.patches))
        left = expected.patch_weight_to_device(key, return_weight=True)
        right = actual.patch_weight_to_device(key, return_weight=True)
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
        refs.clear()
    asyncio.run(run())


@pytest.mark.parametrize("strength", [101., -101., float("inf"), float("nan")])
def test_default_strength_validation_unchanged(original_lora, strength):
    async def run():
        refs, model, _, asset = await handles(original_lora, False)
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            with pytest.raises(ValueError, match=r"finite and in \[-100, 100\]"):
                await model.apply_lora(asset, None, strength, 0.)
        refs.clear()
    asyncio.run(run())


@pytest.mark.parametrize("name", ["skip_zero", "allow_extended_strengths"])
@pytest.mark.parametrize("bad", [None, 1, "false", []])
def test_policy_requires_real_boolean(original_lora, name, bad):
    async def run():
        refs, model, clip, asset = await handles(original_lora)
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            with pytest.raises(TypeError, match="must be a boolean"):
                await model.apply_lora(asset, clip, 0., 0., **{name: bad})
        refs.clear()
    asyncio.run(run())


@pytest.mark.parametrize("bad", [True, None, "101", [], {}])
def test_extended_strength_does_not_coerce_invalid_input(original_lora, bad):
    async def run():
        refs, model, _, asset = await handles(original_lora, False)
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            with pytest.raises(TypeError, match="must be a number"):
                await model.apply_lora(asset, None, bad, 0., allow_extended_strengths=True)
        refs.clear()
    asyncio.run(run())


def test_native_policies_preserve_clip_and_asset_validation(original_lora, tmp_path):
    async def run():
        refs, model, _, asset = await handles(original_lora, False)
        outside = tmp_path / "outside.safetensors"
        outside.touch()
        escaped = _sdk.AssetRef._wrap(await refs.create("ASSET", outside))
        missing = _sdk.AssetRef._wrap(await refs.create("ASSET", original_lora[2].parent / "missing-lora.safetensors"))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            with pytest.raises(ValueError, match="escapes the loras directory"):
                await model.apply_lora(escaped, None, 0., 0., skip_zero=False)
            with pytest.raises(FileNotFoundError):
                await model.apply_lora(missing, None, 0., 0., skip_zero=False)
            with pytest.raises(ValueError, match="must be zero when clip is None"):
                await model.apply_lora(asset, None, 101., 1., allow_extended_strengths=True)
        refs.clear()
    asyncio.run(run())
