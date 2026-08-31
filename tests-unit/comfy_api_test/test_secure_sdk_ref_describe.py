from __future__ import annotations

import asyncio
import types

import pytest
import torch

from comfy_api.latest import _sdk


def test_ref_describe_projects_tensor_metadata_without_values():
    async def run():
        refs = _sdk.InProcessRefResolver()
        token = await refs.create(
            "IMAGE", torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3))
        image = _sdk.ImageRef._wrap(token)
        with _sdk.bind_runtime(
            refs, types.SimpleNamespace(), _sdk.InProcessOps(),
        ):
            return await image.describe()

    assert asyncio.run(run()) == {
        "kind": "IMAGE",
        "type": "Tensor",
        "length": 1,
        "first": "<redacted tensor slice shape=[2, 4, 3]>",
        "shape": [1, 2, 4, 3],
        "summary": "<IMAGE tensor shape=[1, 2, 4, 3] dtype=torch.float32 device=cpu>",
        "truncated": False,
    }


def test_ref_describe_never_invokes_opaque_object_behavior():
    touched = []

    class Trap:
        def __len__(self):
            touched.append("len")
            raise AssertionError

        def __iter__(self):
            touched.append("iter")
            raise AssertionError

        def __repr__(self):
            touched.append("repr")
            raise AssertionError

        @property
        def shape(self):
            touched.append("shape")
            raise AssertionError

    async def run():
        refs = _sdk.InProcessRefResolver()
        model = _sdk.ModelRef._wrap(await refs.create("MODEL", Trap()))
        with _sdk.bind_runtime(
            refs, types.SimpleNamespace(), _sdk.InProcessOps(),
        ):
            return await model.describe()

    assert asyncio.run(run()) == {
        "kind": "MODEL",
        "type": "opaque MODEL",
        "length": None,
        "first": None,
        "shape": None,
        "summary": "<opaque MODEL>",
        "truncated": False,
    }
    assert touched == []


def test_ref_describe_never_exposes_an_asset_path():
    async def run():
        refs = _sdk.InProcessRefResolver()
        asset = _sdk.AssetRef._wrap(await refs.create(
            "ASSET", "/tenant/private/models/secret.safetensors"))
        with _sdk.bind_runtime(
            refs, types.SimpleNamespace(), _sdk.InProcessOps(),
        ):
            return await asset.describe()

    description = asyncio.run(run())
    assert description["kind"] == "ASSET"
    assert description["summary"] == "<opaque ASSET>"
    assert "tenant" not in repr(description)
    assert "secret.safetensors" not in repr(description)


def test_ref_describe_bounds_and_truncates_its_projection():
    async def run(limit):
        refs = _sdk.InProcessRefResolver()
        kind = "DIAGNOSTIC_" + "X" * 80
        value = _sdk.Ref(kind=kind, id=(await refs.create(kind, object())).id)
        with _sdk.bind_runtime(
            refs, types.SimpleNamespace(), _sdk.InProcessOps(),
        ):
            return await value.describe(limit)

    description = asyncio.run(run(32))
    assert description["truncated"] is True
    assert len(description["summary"]) == 32
    assert description["summary"].endswith("…")
    with pytest.raises(ValueError, match=r"\[32, 32768\]"):
        asyncio.run(run(31))
    with pytest.raises(TypeError, match="must be an integer"):
        asyncio.run(run(True))
