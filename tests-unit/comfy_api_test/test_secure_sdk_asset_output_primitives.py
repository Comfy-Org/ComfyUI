import asyncio
import hashlib
import os

import pytest
import torch
from PIL import Image, UnidentifiedImageError

from comfy_api.latest._sdk import (
    AssetRef,
    ExecutionPlan,
    ImageRef,
    InProcessCtxProvider,
    InProcessOps,
    InProcessRefResolver,
    bind_runtime,
)


def _plan(*, workflow=None):
    return ExecutionPlan(
        prompt_id="asset-output",
        node_id="1",
        node_type="asset-output-test",
        prompt={"1": {"class_type": "asset-output-test"}},
        extra_pnginfo={"workflow": workflow or {"nodes": [{"id": 1}]}},
    )


def test_image_batch_size_is_bounded_scalar_metadata():
    async def run():
        refs = InProcessRefResolver()
        context = InProcessCtxProvider().build(_plan())
        with bind_runtime(refs, context, InProcessOps()):
            hwc = ImageRef._wrap(await refs.create(
                "IMAGE", torch.zeros((2, 3, 3))))
            bhwc = ImageRef._wrap(await refs.create(
                "IMAGE", torch.zeros((2, 2, 3, 4))))
            oversized = ImageRef._wrap(await refs.create(
                "IMAGE", torch.empty((4097, 0, 0, 3))))
            assert await hwc.batch_size() == 1
            assert await bhwc.batch_size() == 2
            with pytest.raises(ValueError, match="batch size"):
                await oversized.batch_size()

    asyncio.run(run())


def test_asset_digest_streams_sha256_and_invalidates_on_change(tmp_path):
    asset_path = tmp_path / "weights.safetensors"
    asset_path.write_bytes(b"first weights")

    async def run():
        refs = InProcessRefResolver()
        context = InProcessCtxProvider().build(_plan())
        asset = AssetRef._wrap(await refs.create("ASSET", str(asset_path)))
        with bind_runtime(refs, context, InProcessOps()):
            first = await context.assets.digest(asset)
            again = await context.assets.digest(asset)
            assert first == again == hashlib.sha256(b"first weights").hexdigest()
            asset_path.write_bytes(b"second weights with another size")
            second = await context.assets.digest(asset)
            assert second == hashlib.sha256(
                b"second weights with another size").hexdigest()
            assert second != first
            with pytest.raises(ValueError, match="sha256"):
                await context.assets.digest(asset, "md5")

    asyncio.run(run())


def test_asset_load_image_decodes_one_bounded_rgb_image(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGBA", (5, 4), (64, 128, 255, 17)).save(image_path)
    invalid_path = tmp_path / "invalid.png"
    invalid_path.write_bytes(b"not an image")

    async def run():
        refs = InProcessRefResolver()
        context = InProcessCtxProvider().build(_plan())
        image_asset = AssetRef._wrap(await refs.create(
            "ASSET", str(image_path)))
        invalid_asset = AssetRef._wrap(await refs.create(
            "ASSET", str(invalid_path)))
        with bind_runtime(refs, context, InProcessOps()):
            image = await context.assets.load_image(image_asset)
            pixels = await refs.resolve(image)
            assert pixels.shape == (1, 4, 5, 3)
            assert torch.allclose(
                pixels[0, 0, 0], torch.tensor([64, 128, 255]) / 255)
            with pytest.raises(UnidentifiedImageError):
                await context.assets.load_image(invalid_asset)
            context.assets._IMAGE_PIXELS_MAX = 19
            with pytest.raises(ValueError, match="dimensions"):
                await context.assets.load_image(image_asset)

    asyncio.run(run())


def test_managed_asset_list_treats_a_new_confined_prefix_as_empty(
    tmp_path, monkeypatch,
):
    import folder_paths

    for folder in ("input", "output", "temp"):
        (tmp_path / folder).mkdir()
    monkeypatch.setattr(
        folder_paths, "get_input_directory", lambda: str(tmp_path / "input"))
    monkeypatch.setattr(
        folder_paths, "get_output_directory", lambda: str(tmp_path / "output"))
    monkeypatch.setattr(
        folder_paths, "get_temp_directory", lambda: str(tmp_path / "temp"))

    async def run():
        context = InProcessCtxProvider().build(_plan())
        assert await context.assets.list(
            "output", "new/subfolder", recursive=False) == []
        with pytest.raises(ValueError, match="escapes"):
            await context.assets.list("output", "../outside", recursive=False)

    asyncio.run(run())


def test_exact_image_names_closed_codecs_metadata_and_workflow_sidecar(
    tmp_path, monkeypatch,
):
    import folder_paths

    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output))
    expected_formats = {
        "png": ("sample.png", "PNG"),
        "jpg": ("sample.jpg", "JPEG"),
        "jpeg": ("sample.jpeg", "JPEG"),
        "webp": ("sample.webp", "WEBP"),
        "j2k": ("sample.j2k", "JPEG2000"),
        "jp2": ("sample.jp2", "JPEG2000"),
        "gif": ("sample.gif", "GIF"),
        "tiff": ("sample.tiff", "TIFF"),
        "bmp": ("sample.bmp", "BMP"),
        "avif": ("sample.avif", "AVIF"),
    }

    async def run():
        refs = InProcessRefResolver()
        workflow = {"nodes": [{"id": 7, "type": "Saved"}]}
        context = InProcessCtxProvider().build(_plan(workflow=workflow))
        image = ImageRef._wrap(await refs.create(
            "IMAGE", torch.zeros((1, 3, 4, 3), dtype=torch.float32)))
        responses = {}
        with bind_runtime(refs, context, InProcessOps()):
            assert await image.batch_size() == 1
            for image_format, (filename, _) in expected_formats.items():
                responses[image_format] = await context.output.save_images(
                    image,
                    filenames=[f"nested/{image_format}/{filename}"],
                    image_format=image_format,
                    quality=100,
                    lossless=True,
                    optimize=True,
                )
            sidecar = await context.output.save_workflow_json(
                "nested/workflow.json")
        return workflow, responses, sidecar

    workflow, responses, sidecar = asyncio.run(run())
    for image_format, (filename, pillow_format) in expected_formats.items():
        record = responses[image_format]["images"][0]
        assert record["filename"] == filename
        assert record["subfolder"] == f"nested/{image_format}"
        saved = Image.open(output / record["subfolder"] / filename)
        assert saved.format == pillow_format
        if image_format in {"jpg", "jpeg", "webp", "avif"}:
            values = [value for value in saved.getexif().values()
                      if isinstance(value, str)]
            assert any(value.lower().startswith("workflow:") for value in values)
    assert sidecar == "nested/workflow.json"
    import json
    assert json.loads((output / sidecar).read_text()) == workflow


def test_exact_image_names_fail_closed_and_never_overwrite(tmp_path, monkeypatch):
    import folder_paths

    output = tmp_path / "output"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "escape").symlink_to(outside, target_is_directory=True)
    existing = output / "existing.png"
    existing.write_bytes(b"keep me")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output))

    async def run():
        refs = InProcessRefResolver()
        context = InProcessCtxProvider().build(_plan())
        image = ImageRef._wrap(await refs.create(
            "IMAGE", torch.zeros((1, 2, 2, 3), dtype=torch.float32)))
        with bind_runtime(refs, context, InProcessOps()):
            for filename, image_format, error in (
                ("../outside.png", "png", ValueError),
                ("escape/out.png", "png", ValueError),
                ("wrong.jpg", "png", ValueError),
                ("existing.png", "png", FileExistsError),
            ):
                with pytest.raises(error):
                    await context.output.save_images(
                        image, filenames=[filename], image_format=image_format)
            with pytest.raises(ValueError, match="length"):
                await context.output.save_images(
                    image, filenames=[], image_format="png")
            with pytest.raises(ValueError, match="not supported"):
                await context.output.save_images(
                    image, filenames=["bad.jxl"], image_format="jxl")

    asyncio.run(run())
    assert existing.read_bytes() == b"keep me"
    assert not (outside / "out.png").exists()


def test_jpeg_large_broker_metadata_degrades_without_losing_the_image(
    tmp_path, monkeypatch,
):
    import folder_paths

    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output))

    async def run():
        refs = InProcessRefResolver()
        plan = _plan(workflow={"blob": "w" * 100_000})
        context = InProcessCtxProvider().build(plan)
        image = ImageRef._wrap(await refs.create(
            "IMAGE", torch.zeros((1, 2, 2, 3), dtype=torch.float32)))
        with bind_runtime(refs, context, InProcessOps()):
            kept = await context.output.save_images(
                image,
                filenames=["kept.jpg"],
                image_format="jpg",
                extra_metadata={"pack_key": "kept"},
            )
            dropped = await context.output.save_images(
                image,
                filenames=["dropped.jpg"],
                image_format="jpg",
                extra_metadata={"pack_blob": "p" * 100_000},
            )
        return kept, dropped

    kept, dropped = asyncio.run(run())
    assert kept["images"][0]["filename"] == "kept.jpg"
    assert dropped["images"][0]["filename"] == "dropped.jpg"
    kept_values = [
        value for value in Image.open(output / "kept.jpg").getexif().values()
        if isinstance(value, str)
    ]
    assert any(value.startswith("pack_key:") for value in kept_values)
    assert not any("w" * 100 in value for value in kept_values)
    assert len(Image.open(output / "dropped.jpg").getexif()) == 0


def test_empty_latent_can_declare_canonical_spatial_ratio():
    from comfy_api.latest._sdk import LatentRef

    async def run():
        refs = InProcessRefResolver()
        context = InProcessCtxProvider().build(_plan())
        with bind_runtime(refs, context, InProcessOps()):
            latent = await LatentRef.empty(
                128, 64, spatial_downscale_ratio=8)
            value = await refs.resolve(latent)
            assert tuple(value["samples"].shape) == (1, 4, 8, 16)
            assert value["downscale_ratio_spacial"] == 8
            flux2 = await LatentRef.empty(
                1024, 768, channels=128, spatial_downscale_ratio=16)
            flux2_value = await refs.resolve(flux2)
            assert tuple(flux2_value["samples"].shape) == (1, 128, 48, 64)
            assert flux2_value["downscale_ratio_spacial"] == 16
            with pytest.raises(ValueError, match="bounded range"):
                await LatentRef.empty(
                    1024, 768, channels=129, spatial_downscale_ratio=16)
            with pytest.raises(ValueError, match="bounded range"):
                await LatentRef.empty(
                    128, 64, spatial_downscale_ratio=7)

    asyncio.run(run())


def test_civitai_vendor_projection_is_closed_bounded_and_cached(monkeypatch):
    from comfy_api.latest._civitai import InProcessCivitai

    calls = []

    def fetch(_cls, path, query=None):
        calls.append((path, query))
        if path == "/api/v1/models":
            return {
                "items": [{
                    "id": 7,
                    "name": "Model",
                    "description": "must not cross the broker",
                    "modelVersions": [{
                        "id": 9,
                        "name": "Version",
                        "downloadUrl": "must not cross",
                    }],
                }],
                "metadata": {"nextPage": "must not cross"},
            }
        if path == "/api/v1/model-versions/9":
            return {
                "id": 9,
                "name": "Version",
                "files": [{
                    "name": "model.safetensors",
                    "downloadUrl": "must not cross",
                    "hashes": {"AutoV3": "ABC", "bad key!": "hidden"},
                }],
            }
        return {
            "id": 9,
            "name": "Version",
            "modelId": 7,
            "baseModel": "SDXL 1.0",
            "trainedWords": ["trigger one", "trigger two", 3, "x" * 513],
            "air": "urn:air:test",
            "model": {
                "name": "Model", "type": "Checkpoint", "nsfw": True,
            },
            "files": [],
            "images": [
                {
                    "url": "https://image.civitai.com/example.webp",
                    "meta": {
                        "prompt": "a useful example prompt",
                        "steps": 24,
                        "nested": {"sampler": "Euler"},
                    },
                    "width": "must not cross",
                },
                {"url": "http://private.invalid/image.png", "meta": {}},
                {
                    "url": "https://image.civitai.com/bad-meta.webp",
                    "meta": {"__proto__": {"bad": True}},
                },
            ],
        }

    monkeypatch.setattr(
        InProcessCivitai, "_fetch_json", classmethod(fetch))
    InProcessCivitai._CACHE.clear()

    async def run():
        context = InProcessCtxProvider().build(_plan())
        first = await context.integrations.civitai.search_models(
            "alice", "Model", limit=20, nsfw=True)
        again = await context.integrations.civitai.search_models(
            "alice", "Model", limit=20, nsfw=True)
        version = await context.integrations.civitai.model_version(9)
        by_hash = await context.integrations.civitai.model_version_by_hash(
            "a" * 64)
        refreshed = await context.integrations.civitai.model_version_by_hash(
            "a" * 64, refresh=True)
        with pytest.raises(ValueError, match="limit"):
            await context.integrations.civitai.search_models(
                "alice", limit=101)
        with pytest.raises(ValueError, match="hash"):
            await context.integrations.civitai.model_version_by_hash("../x")
        return first, again, version, by_hash, refreshed

    first, again, version, by_hash, refreshed = asyncio.run(run())
    assert first == again == {"items": [{
        "id": 7,
        "name": "Model",
        "modelVersions": [{"id": 9, "name": "Version"}],
    }]}
    assert version == {
        "id": 9,
        "name": "Version",
        "files": [{
            "name": "model.safetensors", "hashes": {"AutoV3": "ABC"},
        }],
    }
    assert by_hash == {
        "id": 9,
        "name": "Version",
        "modelId": 7,
        "baseModel": "SDXL 1.0",
        "air": "urn:air:test",
        "model": {"name": "Model", "type": "Checkpoint"},
        "files": [],
        "trainedWords": ["trigger one", "trigger two"],
        "images": [
            {
                "url": "https://image.civitai.com/example.webp",
                "meta": {
                    "prompt": "a useful example prompt",
                    "steps": 24,
                    "nested": {"sampler": "Euler"},
                },
            },
            {"url": "https://image.civitai.com/bad-meta.webp"},
        ],
    }
    assert refreshed == by_hash
    assert [path for path, _query in calls].count("/api/v1/models") == 1


def test_civitai_image_metadata_projection_is_closed_and_bounded():
    from comfy_api.latest._civitai import InProcessCivitai

    images = [
        {
            "url": f"https://image.civitai.com/{index}.webp",
            "meta": {"prompt": f"prompt {index}"},
        }
        for index in range(40)
    ]
    images[1] = {
        "url": "https://image.civitai.com/deep.webp",
        "meta": {"a": {"b": {"c": {"d": {"e": "too deep"}}}}},
    }
    images[2] = {
        "url": "http://127.0.0.1/private.png",
        "meta": {"prompt": "must not cross"},
    }
    projected = InProcessCivitai._project_version_by_hash({
        "id": 9,
        "name": "Version",
        "modelId": 7,
        "baseModel": "SDXL",
        "model": {"name": "Model"},
        "files": [],
        "images": images,
    })

    assert projected["baseModel"] == "SDXL"
    assert len(projected["images"]) == 31
    assert projected["images"][0]["meta"] == {"prompt": "prompt 0"}
    assert projected["images"][1] == {
        "url": "https://image.civitai.com/deep.webp",
    }
    assert all(
        "127.0.0.1" not in item["url"] for item in projected["images"])
    assert all(set(item) <= {"url", "meta"} for item in projected["images"])


def test_onnx_multilabel_classifier_keeps_scores_opaque_and_pages_matches(
    tmp_path, monkeypatch,
):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import folder_paths
    from comfy_api.latest import _sdk

    input_info = helper.make_tensor_value_info(
        "image", TensorProto.FLOAT, [None, 4, 4, 3])
    output_info = helper.make_tensor_value_info(
        "scores", TensorProto.FLOAT, [None, 4])
    weights = helper.make_tensor(
        "weights", TensorProto.FLOAT, [48, 4],
        np.zeros((48, 4), dtype=np.float32).ravel())
    bias = helper.make_tensor(
        "bias", TensorProto.FLOAT, [4], [0.1, 0.7, 0.3, 0.9])
    graph = helper.make_graph([
        helper.make_node("Flatten", ["image"], ["flat"], axis=1),
        helper.make_node(
            "Gemm", ["flat", "weights", "bias"], ["scores"]),
    ], "classifier", [input_info], [output_info], [weights, bias])
    model = helper.make_model(
        graph, opset_imports=[
            helper.make_opsetid("", 17),
            # Several real WD exporters retain unused provider opsets. The
            # validator confines domains on executable nodes, not dead imports.
            helper.make_opsetid("com.microsoft", 1),
        ])
    model.ir_version = 8
    model_path = tmp_path / "classifier.onnx"
    onnx.save(model, model_path)
    monkeypatch.setitem(
        folder_paths.folder_names_and_paths,
        "onnx", ([str(tmp_path)], {".onnx"}),
    )
    _sdk._ONNX_IMAGE_CLASSIFIER_CACHE.clear()

    with pytest.raises(ValueError, match="sha256"):
        _sdk.HuggingFaceWeight(
            "owner/model", "model.onnx", "onnx", revision="abc123")
    declaration = _sdk.HuggingFaceWeight(
        "owner/model", "model.onnx", "onnx", revision="abc123",
        sha256="a" * 64,
    )
    assert declaration.catalogue_name.endswith("/model.onnx")

    async def run():
        refs = InProcessRefResolver()
        context = InProcessCtxProvider().build(_plan())
        with bind_runtime(refs, context, InProcessOps()):
            classifier = await context.models.load_onnx_image_classifier(
                "classifier.onnx",
                input_layout="NHWC",
                channel_order="BGR",
                resize_mode="fit_pad",
                input_scale=255.0,
                activation="identity",
            )
            # A second bind reuses the validated runtime session.
            await context.models.load_onnx_image_classifier(
                "classifier.onnx", activation="identity")
            images = ImageRef._wrap(await refs.create(
                "IMAGE", torch.zeros((2, 2, 4, 3), dtype=torch.float32)))
            scores = await classifier.predict_scores(images)
            shape = await scores.shape()
            first = await scores.select_above(
                0, 0, 4, 0.25, offset=0, limit=2)
            second = await scores.select_above(
                0, 0, 4, 0.25, offset=first["next_offset"], limit=2)
            with pytest.raises(ValueError, match="class range"):
                await scores.select_above(0, 0, 5, 0.0)
            return shape, first, second

    shape, first, second = asyncio.run(run())
    assert shape == (2, 4)
    assert [item["index"] for item in first["items"]] == [1, 2]
    assert first["next_offset"] == 2
    assert [item["index"] for item in second["items"]] == [3]
    assert second["next_offset"] is None
    assert _sdk._ONNX_IMAGE_CLASSIFIER_CACHE.loads == 1


def test_onnx_validation_rejects_external_tensor_data(tmp_path):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from comfy_api.latest import _sdk

    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [None, 1])
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, 1])
    weight = numpy_helper.from_array(
        np.ones((1, 1), dtype=np.float32), name="weight")
    graph = helper.make_graph([
        helper.make_node("MatMul", ["input", "weight"], ["output"]),
    ], "external", [input_info], [output_info], [weight])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    model_path = tmp_path / "external.onnx"
    onnx.save_model(
        model, model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="external.data",
        size_threshold=0,
    )
    with pytest.raises(ValueError, match="external ONNX tensor"):
        _sdk._validate_onnx_weight_file(str(model_path))
