import asyncio
from copy import deepcopy

import pytest
import torch

import comfy.ldm.anima.lllite
import comfy.model_base
import comfy.model_patcher
import comfy.utils
import folder_paths
from comfy_api.latest import _sdk


class _Sampling:
    @staticmethod
    def percent_to_sigma(value):
        return 1.0 - value


class _PatchedModel:
    def __init__(self, model, model_options):
        self.model = model
        self.model_options = deepcopy(model_options)

    def set_model_post_input_patch(self, patch):
        self.set_model_patch(patch, "post_input")

    def set_model_attn1_patch(self, patch):
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch):
        self.set_model_patch(patch, "attn2_patch")

    def set_model_patch(self, patch, name):
        patches = self.model_options.setdefault(
            "transformer_options", {}).setdefault("patches", {})
        patches.setdefault(name, []).append(patch)


def _plan():
    return _sdk.ExecutionPlan(
        prompt_id="anima",
        node_id="1",
        node_type="anima-test",
        prompt={"1": {"class_type": "anima-test"}},
        extra_pnginfo={},
    )


def test_anima_lllite_uses_confined_rgb_weights_and_preserves_patch_stacking(
    tmp_path, monkeypatch,
):
    class BaseAnima:
        pass

    class SourceModel:
        def __init__(self):
            self.model = BaseAnima()
            self.model_options = {
                "model_function_wrapper": object(),
                "transformer_options": {"patches": {"existing": [object()]}},
            }

        def get_model_object(self, name):
            assert name == "model_sampling"
            return _Sampling()

        def clone(self):
            return _PatchedModel(self.model, self.model_options)

    class LLLite:
        cond_in_channels = 3

        def __init__(self, state, metadata, **kwargs):
            assert set(state) == {"weight"}
            assert metadata == {"lllite.version": "2"}
            self.loaded = False

        def load_state_dict(self, state, assign=False):
            self.loaded = True

    class CoreModelPatcher:
        def __init__(self, model, **kwargs):
            self.model = model

        @staticmethod
        def is_dynamic():
            return False

    weights_path = tmp_path / "anima.safetensors"
    weights_path.write_bytes(b"safe")
    monkeypatch.setattr(comfy.model_base, "Anima", BaseAnima)
    monkeypatch.setattr(
        comfy.ldm.anima.lllite, "AnimaLLLite", LLLite)
    monkeypatch.setattr(
        comfy.model_patcher, "CoreModelPatcher", CoreModelPatcher)
    monkeypatch.setattr(
        comfy.utils, "load_torch_file",
        lambda *args, **kwargs: (
            {"weight": torch.ones(1)}, {"lllite.version": "2"}),
    )
    monkeypatch.setattr(comfy.utils, "weight_dtype", lambda state: torch.float32)
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda name: [str(tmp_path)])

    async def run():
        refs = _sdk.InProcessRefResolver()
        context = _sdk.InProcessCtxProvider().build(_plan())
        with _sdk.bind_runtime(refs, context, _sdk.InProcessOps()):
            model = _sdk.ModelRef._wrap(
                await refs.create("MODEL", SourceModel()))
            weights = _sdk.AssetRef._wrap(
                await refs.create("ASSET", str(weights_path)))
            image = _sdk.ImageRef._wrap(
                await refs.create("IMAGE", torch.zeros(1, 32, 48, 3)))
            result_ref = await context.integrations.anima.apply_lllite(
                model, weights, image, strength=0.75,
                start_percent=0.2, end_percent=0.8,
                preserve_wrapper=False,
            )
            result = await refs.resolve(result_ref)
            assert "model_function_wrapper" not in result.model_options
            patches = result.model_options["transformer_options"]["patches"]
            assert len(patches["existing"]) == 1
            assert {"post_input", "attn1_patch", "attn2_patch", "mlp_patch"} <= set(patches)

            outside = tmp_path.parent / "outside.safetensors"
            outside.write_bytes(b"safe")
            outside_ref = _sdk.AssetRef._wrap(
                await refs.create("ASSET", str(outside)))
            with pytest.raises(ValueError, match="escapes the controlnet"):
                await context.integrations.anima.apply_lllite(
                    model, outside_ref, image)

    asyncio.run(run())


def test_anima_lllite_spatial_tile_crops_the_control_image_exactly():
    image = torch.arange(1 * 32 * 48 * 3).reshape(1, 32, 48, 3)
    tile = comfy.ldm.anima.lllite._spatial_tile_image(image, {
        "top": 1,
        "bottom": 3,
        "left": 2,
        "right": 5,
        "source_height": 4,
        "source_width": 6,
    })
    assert torch.equal(tile, image[:, 8:24, 16:40, :])
