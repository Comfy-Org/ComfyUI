"""Secure custom-node SDK seam — regression + POC.

Drives the real ``execution._async_map_node_over_list`` dispatch path with
nodes authored against the v0_0_3 ``sdk`` (refs + ctx), for both the sync and
async execute forms, and verifies:

  * output is correct (invert / scale) — no regression vs. today's in-process
    behavior;
  * the default execution backend is the in-process one (zero-overhead);
  * a registered overlay backend intercepts real node dispatch (the provider
    swap), while output stays correct.
"""
import asyncio
import json
import pathlib
import threading

import pytest
import torch

from comfy_api.latest import sdk
from comfy_api.latest._sdk import (
    BackgroundRemovalModelRef,
    CondRef,
    GuiderRef,
    ImageRef,
    InProcessCtxProvider,
    InpaintModelRef,
    InProcessExecutionBackend,
    InProcessOps,
    InProcessRefResolver,
    MaskRef,
    ModelRef,
    bind_runtime,
    ExecutionPlan,
)
from comfy_api.v0_0_3 import io


class _InvertAsync(io.ComfyNode):
    # SDK asset node: receives an ImageRef, transforms via an engine-side op,
    # never touches a buffer.
    SDK_REFS = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="_TestInvertAsync", category="test",
            inputs=[io.Image.Input("image")], outputs=[io.Image.Output()],
        )

    @classmethod
    async def execute(cls, image):
        ctx = sdk.ctx()
        await ctx.progress.update(0.0, 1.0)
        out = await image.invert()   # operation on the asset
        await ctx.progress.update(1.0, 1.0)
        return io.NodeOutput(out)


class _ScaleSyncLegacy(io.ComfyNode):
    # Legacy (non-SDK) v3 node: sync execute, receives a raw tensor. Confirms
    # the sync dispatch branch + that non-SDK nodes are unaffected by the seam.
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="_TestScaleSyncLegacy", category="test",
            inputs=[io.Image.Input("image")], outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image):
        return io.NodeOutput(image * 0.5)


class _InvertWithUi(io.ComfyNode):
    # An SDK asset node that is ALSO an output node. Rebuilding its NodeOutput
    # to resolve refs must not discard what is not a result.
    SDK_REFS = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="_TestInvertWithUi", category="test",
            inputs=[io.Image.Input("image")], outputs=[io.Image.Output()],
            is_output_node=True,
        )

    @classmethod
    async def execute(cls, image):
        out = await image.invert()
        return io.NodeOutput(out, ui={"text": ["hello"]})


class _ProgressWithPreview(io.ComfyNode):
    SDK_REFS = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="_TestProgressWithPreview", category="test",
            inputs=[io.Image.Input("image")], outputs=[io.Image.Output()],
        )

    @classmethod
    async def execute(cls, image):
        await sdk.ctx().progress.update(0.5, 1.0, preview=image)
        return io.NodeOutput(image)


async def _run_full(node_cls, image):
    import execution

    results = await execution._async_map_node_over_list(
        prompt_id="p", unique_id="1", obj=node_cls,
        input_data_all={"image": [image]}, func=node_cls.FUNCTION, v3_data=None,
    )
    return results[0]


async def _run(node_cls, image):
    out = await _run_full(node_cls, image)
    return out.result[0]


def test_sdk_node_keeps_its_ui_output():
    """Resolving output refs must preserve `ui`.

    `unwrap_outputs` rebuilds the NodeOutput to swap refs back for real
    objects. Rebuilding it from results alone dropped `ui`, which made every
    SDK_REFS node unable to be an output node: ComfyUI only emits the
    `executed` event that carries results to the frontend for nodes returning
    ui data, so such a node ran correctly and then displayed nothing.
    """
    img = torch.rand(1, 8, 8, 3)
    out = asyncio.run(_run_full(_InvertWithUi, img))
    assert torch.allclose(out.result[0], 1.0 - img), "pixels wrong"
    assert out.ui == {"text": ["hello"]}, f"ui was dropped: {out.ui!r}"


def test_progress_preview_resolves_image_ref_for_comfy(monkeypatch):
    import comfy.utils

    updates = []

    class RecordingProgressBar:
        def __init__(self, total, node_id=None):
            self.total = total
            self.node_id = node_id

        def update_absolute(self, value, total=None, preview=None):
            updates.append((self.node_id, value, total, preview))

    monkeypatch.setattr(comfy.utils, "ProgressBar", RecordingProgressBar)
    image = torch.zeros((1, 5, 7, 3), dtype=torch.float32)
    got = _output_of(_ProgressWithPreview, image)
    assert torch.equal(got, image)
    assert len(updates) == 1
    node_id, value, total, preview = updates[0]
    assert (node_id, value, total) == ("1", 0.5, 1.0)
    assert preview[0] == "PNG"
    assert preview[1].size == (7, 5)


def test_image_brokers_control_execution_and_extra_metadata(tmp_path):
    import folder_paths
    from PIL import Image

    old_output = folder_paths.get_output_directory()
    old_temp = folder_paths.get_temp_directory()
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    output_dir.mkdir()
    temp_dir.mkdir()
    folder_paths.set_output_directory(str(output_dir))
    folder_paths.set_temp_directory(str(temp_dir))

    async def run():
        refs = InProcessRefResolver()
        plan = ExecutionPlan(
            prompt_id="metadata",
            node_id="1",
            node_type="metadata-test",
            prompt={"1": {"class_type": "metadata-test"}},
            extra_pnginfo={"workflow": {"nodes": [{"id": 1}]}},
        )
        context = InProcessCtxProvider().build(plan)
        image = ImageRef._wrap(await refs.create(
            "IMAGE", torch.zeros((1, 2, 3, 3), dtype=torch.float32)))
        with bind_runtime(refs, context, InProcessOps()):
            normal = await context.output.save_images(
                image, filename_prefix="with_metadata",
                extra_metadata={"Title": "Crystools"})
            private = await context.output.save_images(
                image, filename_prefix="without_workflow",
                save_metadata=False,
                extra_metadata={"Title": "Crystools"})
            preview = await context.ui.preview_images(image)
        return normal, private, preview

    try:
        normal, private, preview = asyncio.run(run())
        normal_info = Image.open(pathlib.Path(
            output_dir, normal["images"][0]["filename"])).info
        assert json.loads(normal_info["prompt"])["1"]["class_type"] == (
            "metadata-test")
        assert json.loads(normal_info["workflow"])["nodes"][0]["id"] == 1
        assert json.loads(normal_info["Title"]) == "Crystools"

        private_info = Image.open(pathlib.Path(
            output_dir, private["images"][0]["filename"])).info
        assert "prompt" not in private_info
        assert "workflow" not in private_info
        assert json.loads(private_info["Title"]) == "Crystools"

        preview_info = Image.open(pathlib.Path(
            temp_dir, preview["images"][0]["filename"])).info
        assert "prompt" in preview_info
        assert "workflow" in preview_info
    finally:
        folder_paths.set_output_directory(old_output)
        folder_paths.set_temp_directory(old_temp)


def test_system_stats_are_bounded_resource_totals(monkeypatch):
    import comfy.model_management as model_management

    device = torch.device("cpu")
    monkeypatch.setattr(model_management, "get_torch_device", lambda: device)
    monkeypatch.setattr(
        model_management, "get_all_torch_devices", lambda: [device])
    monkeypatch.setattr(
        model_management, "get_torch_device_name", lambda value: "Test CPU")

    def total(value, torch_total_too=False):
        return (1000, 800) if torch_total_too else 4096

    def free(value, torch_free_too=False):
        return (400, 300) if torch_free_too else 1024

    monkeypatch.setattr(model_management, "get_total_memory", total)
    monkeypatch.setattr(model_management, "get_free_memory", free)
    context = InProcessCtxProvider().build(ExecutionPlan(
        prompt_id="stats", node_id="1", node_type="stats"))

    stats = asyncio.run(context.system.stats())
    assert stats == {
        "system": {"ram_total": 4096, "ram_free": 1024},
        "devices": [{
            "name": "Test CPU",
            "type": "cpu",
            "index": None,
            "vram_total": 1000,
            "vram_free": 400,
            "torch_vram_total": 800,
            "torch_vram_free": 300,
        }],
    }


def test_conditioning_spatial_crop_keeps_tile_orchestration_pack_side():
    class FakeControl:
        def __init__(self, hint, extra, previous=None):
            self.cond_hint_original = hint
            self.cond_hint = object()
            self.control_input = object()
            self.extra_concat_orig = [extra]
            self.previous_controlnet = previous

        def copy(self):
            clone = object.__new__(type(self))
            clone.__dict__ = self.__dict__.copy()
            return clone

        def set_previous_controlnet(self, previous):
            self.previous_controlnet = previous

    async def run_crop():
        refs = InProcessRefResolver()
        ops = InProcessOps()
        embedding = torch.tensor([[[1.0]]])
        pooled = torch.tensor([[2.0]])
        mask = torch.zeros((1, 8, 10))
        mask[:, 2:5, 3:7] = 1.0
        hint = torch.arange(3 * 64 * 80).reshape(1, 3, 64, 80)
        extra = torch.arange(8 * 10).reshape(1, 1, 8, 10)
        previous = FakeControl(hint + 1, extra + 1)
        control = FakeControl(hint, extra, previous)
        conditioning = [
            [embedding, {
                "area": (4, 5, 1, 2),
                "mask": mask,
                "gligen": (
                    "position", object(), [
                        (pooled, 4, 5, 1, 2),
                        (pooled, 1, 1, 7, 9),
                    ],
                ),
                "control": control,
            }],
            [embedding + 1, {"area": (1, 1, 7, 9)}],
            [embedding + 2, {
                "area": ("percentage", 0.5, 0.5, 0.25, 0.2),
            }],
        ]
        ref = CondRef._wrap(await refs.create("CONDITIONING", conditioning))
        with bind_runtime(refs, None, ops):
            cropped_ref = await ref.spatial_crop(
                x=3, y=2, width=4, height=3,
                source_width=10, source_height=8,
            )
            cropped = await refs.resolve(cropped_ref)
        return cropped, conditioning, control, previous

    cropped, original, control, previous = asyncio.run(run_crop())
    assert len(cropped) == 2
    assert cropped[0][0] is original[0][0]
    assert cropped[0][1]["area"] == (3, 4, 0, 0)
    assert cropped[0][1]["mask"].shape == (1, 3, 4)
    assert torch.all(cropped[0][1]["mask"] == 1)
    assert cropped[0][1]["gligen"][2] == [
        (original[0][1]["gligen"][2][0][0], 3, 4, 0, 0),
    ]
    assert cropped[1][1]["area"] == (3, 4, 0, 0)

    cloned = cropped[0][1]["control"]
    assert cloned is not control
    assert cloned.previous_controlnet is not previous
    assert torch.equal(
        cloned.cond_hint_original,
        control.cond_hint_original[..., 16:40, 24:56],
    )
    assert torch.equal(
        cloned.extra_concat_orig[0],
        control.extra_concat_orig[0][..., 2:5, 3:7],
    )
    assert cloned.cond_hint is None
    assert cloned.control_input is None
    assert control.cond_hint is not None
    assert control.control_input is not None


def test_scheduled_cfg_guider_accepts_closed_sigma_bounds(monkeypatch):
    import comfy.samplers

    calls = []

    def sampling_function(
        inner_model, x, timestep, uncond, cond, cfg,
        model_options=None, seed=None,
    ):
        calls.append((uncond, cond, cfg))
        return x

    monkeypatch.setattr(comfy.samplers, "sampling_function", sampling_function)

    class FakeModel:
        model_options = {}

        @staticmethod
        def is_dynamic():
            return False

    async def run():
        refs = InProcessRefResolver()
        model = ModelRef._wrap(await refs.create("MODEL", FakeModel()))
        positive = CondRef._wrap(await refs.create("CONDITIONING", []))
        negative = CondRef._wrap(await refs.create("CONDITIONING", []))
        with bind_runtime(refs, None, InProcessOps()):
            guider_ref = await model.scheduled_cfg_guider(
                positive, negative, 6.5,
                bounds={"unit": "sigma", "start": 5.42, "end": 0.28},
            )
            guider = await refs.resolve(guider_ref)
            guider.inner_model = "model"
            guider.conds = {"positive": "positive", "negative": "negative"}
            sample = torch.zeros((1, 1, 1, 1))
            guider.predict_noise(sample, torch.tensor([5.0]))
            guider.predict_noise(sample, torch.tensor([0.1]))
            with pytest.raises(ValueError, match="at least"):
                await model.scheduled_cfg_guider(
                    positive, negative, 6.5,
                    bounds={"unit": "sigma", "start": 0.28, "end": 5.42},
                )

    asyncio.run(run())
    assert calls == [
        ("negative", "positive", 6.5),
        (None, "positive", 1.0),
    ]


def test_sampling_spatial_crop_uses_patch_owned_protocol_for_model_and_guider():
    import comfy.model_patcher

    class SpatialPatch:
        def __init__(self, label):
            self.label = label
            self.calls = []

        def spatial_crop_inputs(self, **kwargs):
            self.calls.append(kwargs)
            return SpatialPatch(self.label + "-cropped")

    class FakeModelPatcher(comfy.model_patcher.ModelPatcher):
        def __del__(self):
            pass

        def __init__(self, patch):
            self.model_options = {
                "transformer_options": {
                    "patches": {
                        "first": [patch],
                        "second": [patch],
                    },
                },
            }

        def clone(self):
            clone = object.__new__(type(self))
            patches = self.model_options["transformer_options"]["patches"]
            clone.model_options = {
                "transformer_options": {
                    "patches": {
                        name: list(values) for name, values in patches.items()
                    },
                },
            }
            return clone

    class Guider:
        def __init__(self, model):
            self.model_patcher = model
            self.model_options = model.model_options
            self.cfg = 4.0

    async def run_crop():
        refs = InProcessRefResolver()
        ops = InProcessOps()
        patch = SpatialPatch("hint")
        model = FakeModelPatcher(patch)
        guider = Guider(model)
        model_ref = ModelRef._wrap(await refs.create("MODEL", model))
        guider_ref = GuiderRef._wrap(await refs.create("GUIDER", guider))
        params = {
            "regions": [(0, 0, 32, 64), (32, 0, 64, 64)],
            "source_width": 64,
            "source_height": 64,
            "target_width": 32,
            "target_height": 64,
        }
        with bind_runtime(refs, None, ops):
            cropped_model_ref = await model_ref.spatial_crop_inputs(**params)
            cropped_guider_ref = await guider_ref.spatial_crop_inputs(**params)
            cropped_model = await refs.resolve(cropped_model_ref)
            cropped_guider = await refs.resolve(cropped_guider_ref)
        return patch, model, guider, cropped_model, cropped_guider, params

    patch, model, guider, cropped_model, cropped_guider, params = asyncio.run(
        run_crop())
    assert len(patch.calls) == 2
    assert patch.calls == [params, params]
    assert cropped_model is not model
    first = cropped_model.model_options["transformer_options"]["patches"]
    assert first["first"][0] is first["second"][0]
    assert first["first"][0].label == "hint-cropped"
    original = model.model_options["transformer_options"]["patches"]
    assert original["first"][0] is patch
    assert cropped_guider is not guider
    assert cropped_guider.model_patcher is not model
    assert cropped_guider.model_options is cropped_guider.model_patcher.model_options
    guider_patches = cropped_guider.model_options[
        "transformer_options"]["patches"]
    assert guider_patches["first"][0] is guider_patches["second"][0]


def test_qwen_control_patches_crop_their_own_spatial_inputs(monkeypatch):
    import comfy.latent_formats
    from comfy_extras.nodes_model_patch import (
        DiffSynthCnetPatch,
        ZImageControlPatch,
    )

    monkeypatch.setattr(
        comfy.latent_formats.Flux, "process_in", lambda _self, value: value)

    class Vae:
        @staticmethod
        def encode(image):
            return image.movedim(-1, 1)

        @staticmethod
        def spacial_compression_encode():
            return 1

    class ControlModel:
        def __init__(self, additional_in_dim):
            self.additional_in_dim = additional_in_dim

        @staticmethod
        def process_input_latent_image(value):
            return value

    class ControlPatcher:
        def __init__(self, additional_in_dim):
            self.model = ControlModel(additional_in_dim)

    image = torch.arange(1 * 8 * 8 * 3, dtype=torch.float32).reshape(
        1, 8, 8, 3)
    inpaint = image.flip(2)
    mask = torch.zeros((1, 1, 1, 8, 8), dtype=torch.float32)
    mask[..., :4] = 1.0
    params = {
        "regions": [(0, 0, 4, 8), (4, 0, 8, 8)],
        "source_width": 8,
        "source_height": 8,
        "target_width": 4,
        "target_height": 8,
    }

    diffsynth = DiffSynthCnetPatch(
        ControlPatcher(0), Vae(), image, 0.75)
    diffsynth_crop = diffsynth.spatial_crop_inputs(**params)
    assert diffsynth_crop is not diffsynth
    assert diffsynth_crop.image.shape == (2, 8, 4, 3)
    assert diffsynth_crop.encoded_image.shape == (2, 3, 8, 4)
    assert torch.equal(diffsynth_crop.image[0], image[0, :, :4])
    assert torch.equal(diffsynth_crop.image[1], image[0, :, 4:])
    assert diffsynth.image.shape == (1, 8, 8, 3)

    zimage = ZImageControlPatch(
        ControlPatcher(1), Vae(), image, 0.5,
        inpaint_image=inpaint, mask=mask,
    )
    zimage_crop = zimage.spatial_crop_inputs(**params)
    assert zimage_crop is not zimage
    assert zimage_crop.image.shape == (2, 8, 4, 3)
    assert zimage_crop.inpaint_image.shape == (2, 8, 4, 3)
    assert zimage_crop.mask.shape == (2, 1, 1, 8, 4)
    assert zimage_crop.encoded_image.shape[0] == 2
    assert zimage.image.shape == (1, 8, 8, 3)
    assert zimage.mask.shape == (1, 1, 1, 8, 8)


def test_typed_inpaint_model_runs_host_side_primitive(monkeypatch):
    import comfy.model_management

    class FakeInpaintModel(torch.nn.Module):
        def forward(self, image, mask):
            return image * (1.0 - mask) + mask * 0.75

    monkeypatch.setattr(
        comfy.model_management, "get_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        comfy.model_management, "unet_offload_device",
        lambda: torch.device("cpu"),
    )
    cache_clears = []
    monkeypatch.setattr(
        comfy.model_management, "soft_empty_cache",
        lambda: cache_clears.append(True),
    )

    async def run_inpaint():
        refs = InProcessRefResolver()
        ops = InProcessOps()
        pixels = torch.full((2, 16, 24, 3), 0.2)
        mask = torch.zeros((1, 16, 24))
        mask[:, 4:12, 8:16] = 1.0
        model_ref = InpaintModelRef._wrap(await refs.create(
            "INPAINT_MODEL", {
                "secure_kind": "image_inpaint.big-lama",
                "model": FakeInpaintModel(),
                "architecture": "big-lama",
                "lock": threading.Lock(),
            }))
        image_ref = ImageRef._wrap(await refs.create("IMAGE", pixels))
        mask_ref = MaskRef._wrap(await refs.create("MASK", mask))
        with bind_runtime(refs, None, ops):
            output_ref = await model_ref.inpaint(image_ref, mask_ref)
            output = await refs.resolve(output_ref)
        return output

    output = asyncio.run(run_inpaint())
    assert output.shape == (2, 16, 24, 3)
    assert output.dtype == torch.float32
    assert torch.allclose(output[:, :4], torch.full_like(output[:, :4], 0.2))
    assert torch.allclose(
        output[:, 4:12, 8:16],
        torch.full_like(output[:, 4:12, 8:16], 0.75),
    )
    assert cache_clears == [True]


def test_background_removal_uses_typed_canonical_model_handle():
    class FakeBackgroundRemoval:
        @staticmethod
        def encode_image(pixels):
            return pixels[..., 0].clone()

    async def run_mask():
        refs = InProcessRefResolver()
        ops = InProcessOps()
        pixels = torch.zeros((2, 8, 10, 3), dtype=torch.float32)
        pixels[..., 0] = torch.linspace(0, 1, 80).reshape(1, 8, 10)
        model = BackgroundRemovalModelRef._wrap(await refs.create(
            "BACKGROUND_REMOVAL_MODEL", {
                "secure_kind": "background_removal.comfy",
                "model": FakeBackgroundRemoval(),
                "lock": threading.Lock(),
            }))
        image = ImageRef._wrap(await refs.create("IMAGE", pixels))
        with bind_runtime(refs, None, ops):
            mask_ref = await model.mask(image)
            mask = await refs.resolve(mask_ref)
        return pixels, mask

    pixels, mask = asyncio.run(run_mask())
    assert mask.shape == (2, 8, 10)
    assert torch.equal(mask, pixels[..., 0])


def test_deep_shrink_uses_core_patch_with_pack_visible_metadata():
    class Sampling:
        @staticmethod
        def percent_to_sigma(percent):
            return 1.0 - float(percent)

    class ModelConfig:
        unet_config = {"context_dim": 2048}

    class InnerModel:
        model_config = ModelConfig()

    class FakePatcher:
        def __init__(self):
            self.model = InnerModel()
            self.input_patch = None
            self.output_patch = None

        def get_model_object(self, name):
            assert name == "model_sampling"
            return Sampling()

        def clone(self):
            return FakePatcher()

        def set_model_input_block_patch_after_skip(self, patch):
            self.input_patch = patch

        def set_model_output_block_patch(self, patch):
            self.output_patch = patch

    async def run_patch():
        refs = InProcessRefResolver()
        ops = InProcessOps()
        original = FakePatcher()
        model = sdk.ModelRef._wrap(await refs.create("MODEL", original))
        latent = sdk.LatentRef._wrap(await refs.create("LATENT", {
            "samples": torch.zeros((1, 4, 96, 320)),
        }))
        with bind_runtime(refs, None, ops):
            context_dim = await model.unet_context_dim()
            spatial_shape = await latent.spatial_shape()
            patched_ref = await model.patch(
                "kohya_deep_shrink",
                block_number=3,
                downscale_factor=2.0,
                start_percent=0.0,
                end_percent=0.35,
                downscale_after_skip=True,
                downscale_method="bicubic",
                upscale_method="bicubic",
            )
            patched = await refs.resolve(patched_ref)
        return original, patched, context_dim, spatial_shape

    original, patched, context_dim, spatial_shape = asyncio.run(run_patch())
    assert context_dim == 2048
    assert spatial_shape == (96, 320)
    assert patched is not original
    assert original.input_patch is None
    assert callable(patched.input_patch)
    assert callable(patched.output_patch)


def test_spatial_tiled_evaluation_is_one_synchronized_model_wrapper():
    tile_contexts = []

    def existing_wrapper(apply_model, args):
        tile_contexts.append(
            args["c"]["transformer_options"]["spatial_tile"])
        return apply_model(
            args["input"], args["timestep"], **args["c"]) + 1

    class FakePatcher:
        def __init__(self, parent=None):
            self.parent = parent
            self.model_options = {
                "model_function_wrapper": existing_wrapper,
            }
            self.wrapper = existing_wrapper

        def clone(self):
            result = FakePatcher(self)
            result.model_options = dict(self.model_options)
            result.wrapper = self.wrapper
            return result

        def set_model_unet_function_wrapper(self, wrapper):
            self.wrapper = wrapper
            self.model_options["model_function_wrapper"] = wrapper

    async def run_patch():
        refs = InProcessRefResolver()
        ops = InProcessOps()
        original = FakePatcher()
        model = sdk.ModelRef._wrap(await refs.create("MODEL", original))
        with bind_runtime(refs, None, ops):
            patched_ref = await model.patch(
                "spatial_tiled_evaluation",
                rows=2,
                columns=3,
                overlap=0.25,
                overlap_x=1,
                overlap_y=1,
                blend="linear",
                preserve_existing=True,
            )
            patched = await refs.resolve(patched_ref)
        return original, patched

    original, patched = asyncio.run(run_patch())
    sample = torch.arange(96, dtype=torch.float32).reshape(1, 1, 8, 12)

    def apply_model(value, _timestep, **_conditioning):
        return value * 2

    output = patched.wrapper(apply_model, {
        "input": sample,
        "timestep": torch.ones((1,)),
        "c": {"transformer_options": {"kept": True}},
    })
    assert torch.allclose(output, sample * 2 + 1)
    assert len(tile_contexts) == 6
    assert all(context["source_height"] == 8 for context in tile_contexts)
    assert all(context["source_width"] == 12 for context in tile_contexts)
    assert original.wrapper is existing_wrapper
    assert patched is not original


def test_diffusion_delta_and_concat_latent_are_separate_core_primitives(
    tmp_path, monkeypatch,
):
    import folder_paths
    from safetensors.torch import save_file

    patch_path = tmp_path / "ic-light.safetensors"
    patch_state = {
        "input_blocks.0.0.weight": torch.ones((2, 8, 1, 1)),
        "input_blocks.0.0.bias": torch.full((2,), 0.25),
    }
    save_file(patch_state, str(patch_path))
    monkeypatch.setattr(
        folder_paths,
        "get_full_path_or_raise",
        lambda folder, name: str(patch_path)
        if (folder, name) == ("model_patches", "ic-light.safetensors")
        else (_ for _ in ()).throw(FileNotFoundError((folder, name))),
    )

    class Diffusion:
        @staticmethod
        def state_dict():
            return {
                "input_blocks.0.0.weight": torch.zeros((2, 4, 1, 1)),
                "input_blocks.0.0.bias": torch.zeros((2,)),
            }

    class LatentFormat:
        scale_factor = 0.5

    class ModelConfig:
        latent_format = LatentFormat()

    class InnerModel:
        diffusion_model = Diffusion()
        model_config = ModelConfig()

    class FakePatcher:
        def __init__(self, parent=None):
            self.parent = parent
            self.model = InnerModel()
            self.model_options = {}
            self.patches = {}
            self.wrapper = None

        def clone(self):
            result = FakePatcher(self)
            result.patches = dict(self.patches)
            result.model_options = dict(self.model_options)
            result.wrapper = self.wrapper
            return result

        def add_patches(self, patches, strength):
            self.patches.update({
                key: (value, strength) for key, value in patches.items()
            })
            return list(patches)

        def set_model_unet_function_wrapper(self, wrapper):
            self.wrapper = wrapper
            self.model_options["model_function_wrapper"] = wrapper

    async def run_patch():
        refs = InProcessRefResolver()
        ops = InProcessOps()
        original = FakePatcher()
        model = sdk.ModelRef._wrap(await refs.create("MODEL", original))
        latent_value = {
            "samples": torch.arange(2 * 4 * 2 * 3, dtype=torch.float32)
            .reshape(2, 4, 2, 3),
        }
        latent = sdk.LatentRef._wrap(await refs.create(
            "LATENT", latent_value))
        with bind_runtime(refs, None, ops):
            weighted_ref = await model.patch(
                "diffusion_weight_delta",
                model_patch="ic-light.safetensors",
                strength=1.0,
                pad_input_channels=True,
            )
            combined_ref = await weighted_ref.patch(
                "concat_latent_input", latent=latent)
            weighted = await refs.resolve(weighted_ref)
            combined = await refs.resolve(combined_ref)
        return original, weighted, combined, latent_value

    original, weighted, combined, latent_value = asyncio.run(run_patch())
    assert original.patches == {}
    assert set(weighted.patches) == {
        "diffusion_model.input_blocks.0.0.weight",
        "diffusion_model.input_blocks.0.0.bias",
    }
    weight_patch = weighted.patches[
        "diffusion_model.input_blocks.0.0.weight"][0]
    assert weight_patch[0] == "diff"
    assert weight_patch[1][1] == {"pad_weight": True}
    assert combined.wrapper is not None

    sample = torch.zeros((2, 4, 2, 3))

    def apply_model(**kwargs):
        return kwargs

    invoked = combined.wrapper(apply_model, {
        "input": sample,
        "timestep": torch.ones((2,)),
        "c": {"tag": "kept"},
    })
    expected = torch.cat([
        item.unsqueeze(0) for item in latent_value["samples"]
    ], dim=1).repeat(2, 1, 1, 1) * 0.5
    assert invoked["tag"] == "kept"
    assert torch.equal(invoked["c_concat"], expected)


def test_conditioning_masks_and_latent_composite_are_typed_primitives():
    async def run_operations():
        refs = InProcessRefResolver()
        ops = InProcessOps()
        conditioning_value = [[torch.ones((1, 2, 3)), {"tag": "source"}]]
        conditioning = CondRef._wrap(await refs.create(
            "CONDITIONING", conditioning_value))
        mask_value = torch.ones((1, 16, 24))
        mask = MaskRef._wrap(await refs.create("MASK", mask_value))
        destination_value = {"samples": torch.zeros((1, 4, 4, 5))}
        source_value = {"samples": torch.ones((1, 4, 2, 3))}
        destination = sdk.LatentRef._wrap(await refs.create(
            "LATENT", destination_value))
        source = sdk.LatentRef._wrap(await refs.create(
            "LATENT", source_value))
        with bind_runtime(refs, None, ops):
            masked_ref = await conditioning.with_mask(mask, strength=0.75)
            composite_ref = await destination.composite(source)
            masked = await refs.resolve(masked_ref)
            composite = await refs.resolve(composite_ref)
        return masked, composite, mask_value

    masked, composite, mask_value = asyncio.run(run_operations())
    assert masked[0][1]["tag"] == "source"
    assert masked[0][1]["mask_strength"] == 0.75
    assert masked[0][1]["set_area_to_bounds"] is False
    assert torch.equal(masked[0][1]["mask"], mask_value)
    assert torch.all(composite["samples"][..., :2, :3] == 1)
    assert torch.all(composite["samples"][..., 2:, :] == 0)
    assert torch.all(composite["samples"][..., :2, 3:] == 0)


def test_rgb_selection_and_latent_repeat_are_typed_primitives():
    async def run_operations():
        refs = InProcessRefResolver()
        ops = InProcessOps()
        pixels = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(
            2, 3, 4, 4)
        image = ImageRef._wrap(await refs.create("IMAGE", pixels))
        latent_value = {
            "samples": torch.arange(2 * 4 * 2 * 3).reshape(2, 4, 2, 3),
            "noise_mask": torch.arange(2 * 2 * 3).reshape(2, 2, 3),
            "batch_index": [4, 5],
        }
        latent = sdk.LatentRef._wrap(await refs.create(
            "LATENT", latent_value))
        with bind_runtime(refs, None, ops):
            rgb_ref = await image.rgb()
            repeated_ref = await latent.repeat_batch(3)
            rgb = await refs.resolve(rgb_ref)
            repeated = await refs.resolve(repeated_ref)
        return pixels, latent_value, rgb, repeated

    pixels, latent_value, rgb, repeated = asyncio.run(run_operations())
    assert torch.equal(rgb, pixels[..., :3])
    assert torch.equal(
        repeated["samples"], latent_value["samples"].repeat(3, 1, 1, 1))
    assert torch.equal(
        repeated["noise_mask"], latent_value["noise_mask"].repeat(3, 1, 1))
    assert repeated["batch_index"] == [4, 5, 6, 7, 8, 9]


def test_inpaint_primitives_delegate_to_canonical_core_nodes():
    class FakeVae:
        @staticmethod
        def spacial_compression_encode():
            return 8

        @staticmethod
        def encode(pixels):
            return pixels.movedim(-1, 1).clone()

    class FakePatcher:
        def __init__(self, parent=None):
            self.parent = parent
            self.denoise_mask = None

        def clone(self):
            return FakePatcher(self)

        def set_model_denoise_mask_function(self, function):
            self.denoise_mask = function

    async def run_operations():
        refs = InProcessRefResolver()
        ops = InProcessOps()
        pixels = torch.full((1, 16, 16, 3), 0.25)
        mask_value = torch.zeros((1, 16, 16))
        mask_value[:, 7:9, 7:9] = 1.0
        positive_value = [[torch.ones((1, 2, 3)), {"side": "positive"}]]
        negative_value = [[torch.zeros((1, 2, 3)), {"side": "negative"}]]
        image = ImageRef._wrap(await refs.create("IMAGE", pixels))
        mask = MaskRef._wrap(await refs.create("MASK", mask_value))
        vae = sdk.VaeRef._wrap(await refs.create("VAE", FakeVae()))
        positive = CondRef._wrap(await refs.create(
            "CONDITIONING", positive_value))
        negative = CondRef._wrap(await refs.create(
            "CONDITIONING", negative_value))
        latent_with_mask = sdk.LatentRef._wrap(await refs.create("LATENT", {
            "samples": torch.zeros((1, 4, 2, 2)),
            "noise_mask": mask_value,
        }))
        original_model = FakePatcher()
        model = sdk.ModelRef._wrap(await refs.create("MODEL", original_model))
        with bind_runtime(refs, None, ops):
            grown_ref = await mask.grow(1, tapered_corners=False)
            latent_mask_ref = await latent_with_mask.noise_mask()
            encoded_ref = await vae.encode_for_inpaint(
                image, mask, grow_mask_by=2)
            conditioned = await vae.encode_inpaint_conditioning(
                image, grown_ref, positive, negative, noise_mask=True)
            patched_ref = await model.patch(
                "differential_diffusion", strength=0.75)
            grown = await refs.resolve(grown_ref)
            latent_mask = await refs.resolve(latent_mask_ref)
            encoded = await refs.resolve(encoded_ref)
            conditioned_values = [
                await refs.resolve(item) for item in conditioned]
            patched = await refs.resolve(patched_ref)
        return (
            grown, latent_mask, encoded, conditioned_values,
            original_model, patched,
        )

    (
        grown, latent_mask, encoded, conditioned,
        original_model, patched,
    ) = asyncio.run(run_operations())
    assert torch.count_nonzero(grown) > 4
    assert torch.equal(latent_mask, torch.where(
        latent_mask > 0, torch.ones_like(latent_mask), latent_mask))
    assert encoded["samples"].shape == (1, 3, 16, 16)
    assert encoded["noise_mask"].shape == (1, 1, 16, 16)
    positive, negative, latent = conditioned
    assert positive[0][1]["side"] == "positive"
    assert negative[0][1]["side"] == "negative"
    assert "concat_latent_image" in positive[0][1]
    assert latent["samples"].shape == (1, 3, 16, 16)
    assert patched is not original_model
    assert patched.parent is original_model
    assert callable(patched.denoise_mask)


def _output_of(node_cls, image):
    return asyncio.run(_run(node_cls, image))


def test_default_backend_is_in_process():
    assert isinstance(sdk.providers.execution_backend, InProcessExecutionBackend)
    assert sdk.providers.overlay_active is False


def test_async_sdk_node_inverts_through_real_engine():
    img = torch.rand(1, 8, 8, 3)
    got = _output_of(_InvertAsync, img)
    assert torch.allclose(got, 1.0 - img)


def test_legacy_sync_node_scales_through_real_engine():
    img = torch.rand(1, 8, 8, 3)
    got = _output_of(_ScaleSyncLegacy, img)
    assert torch.allclose(got, img * 0.5)


def test_overlay_backend_intercepts_dispatch():
    calls = []

    class _FakeOverlayBackend:
        async def dispatch(self, plan, local_call, runtime=None):
            calls.append((plan, runtime))
            return await local_call()  # delegate -> behavior preserved

    original = sdk.providers.execution_backend
    sdk.providers.register_execution_backend(_FakeOverlayBackend())
    try:
        img = torch.rand(1, 4, 4, 3)
        got = _output_of(_InvertAsync, img)
        assert torch.allclose(got, 1.0 - img)   # still correct
        assert len(calls) == 1                    # overlay saw the real dispatch
        plan, runtime = calls[0]
        assert plan.node_type == "_InvertAsync"
        # Work-unit payload: an out-of-process backend gets the module spec,
        # the ref-wrapped inputs, and the host runtime to broker against.
        assert plan.node_module == _InvertAsync.__module__
        assert isinstance(plan.inputs["image"], sdk.ImageRef)
        assert runtime is not None and runtime.refs is not None
    finally:
        sdk.providers.execution_backend = original


if __name__ == "__main__":
    # Runnable without pytest.
    test_default_backend_is_in_process()
    test_async_sdk_node_inverts_through_real_engine()
    test_legacy_sync_node_scales_through_real_engine()
    test_overlay_backend_intercepts_dispatch()
