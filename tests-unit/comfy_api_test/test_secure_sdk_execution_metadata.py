"""Owned metadata and execution observations; graph/model values are fixtures."""
import asyncio
import copy
import gc
import json
import weakref

import pytest
import torch

import execution
import nodes
from comfy_api.latest import _execution_metadata as em
from comfy_api.latest import _sdk
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
from comfy_execution.graph import DynamicPrompt


class TextFixture:
    RETURN_TYPES = ("CONDITIONING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING",)}, "hidden": {"secret": "AUTH_TOKEN_COMFY_ORG"}}


class SamplerFixture:
    RETURN_TYPES = ("LATENT",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positive": ("CONDITIONING",), "noise": ("NOISE",),
            "seed": ("INT",), "latent_image": ("LATENT",),
        }}


PROMPT = {
    "1": {"class_type": "TextFixture", "inputs": {"text": "real text"}, "_meta": {"title": ">>C"}},
    "2": {"class_type": "SamplerFixture", "inputs": {"positive": ["1", 0], "seed": 0}},
    "3": {"class_type": "MetadataReader", "inputs": {}, "properties": {"lm_marker_role": "meta_primary_sampler"}},
}
EXTRA = {"workflow": {"nodes": [{"id": 2, "properties": {"lm_marker_role": "meta_primary_sampler"}}]},
         "false": False, "null": None, "unicode": "日本語", "parameters": "retained"}


def context(owner, *, node_id="3", prompt_id=None, permissions=("graph.metadata",)):
    plan = _sdk.ExecutionPlan(
        owner.prompt_id if prompt_id is None else prompt_id, node_id, "MetadataReader",
        permissions=permissions, execution_metadata=owner,
    )
    return _sdk.InProcessCtxProvider().build(plan)


def read(ctx, **kwargs):
    async def run():
        with _sdk.bind_runtime(_sdk.InProcessRefResolver(), ctx, _sdk.InProcessOps()):
            return await ctx.graph.execution_metadata(**kwargs)
    return asyncio.run(run())


def test_snapshot_keeps_actual_json_links_titles_workflow_and_detaches():
    prompt = copy.deepcopy(PROMPT)
    extra = copy.deepcopy(EXTRA)
    with em.ExecutionMetadataScope("owned", DynamicPrompt(prompt), extra) as owner:
        result = read(context(owner))
        assert result["prompt"] == prompt and result["extra_pnginfo"] == extra
        assert result["node_id"] == "3" and result["prompt_id"] == "owned"
        result["prompt"]["1"]["inputs"]["text"] = "guest changed"
        result["extra_pnginfo"]["workflow"]["nodes"].clear()
        assert read(context(owner))["prompt"] == prompt
        assert read(context(owner))["extra_pnginfo"] == extra


def test_observations_preserve_order_aliases_noise_seed_and_tensor_shape():
    condition = [[torch.zeros(1, 2, 4), {}]]
    latent = {"samples": torch.zeros(1, 4, 8, 9)}
    with em.ExecutionMetadataScope("trace", DynamicPrompt(copy.deepcopy(PROMPT)), EXTRA, observe=True) as owner:
        owner.record_start("1", TextFixture, {"text": ["actual resolved text"], "secret": ["never expose"]})
        owner.record_outputs("1", TextFixture, [[condition]])
        owner.record_start("2", SamplerFixture, {
            "positive": [condition], "noise": [Noise_RandomNoise(2**63 + 7)],
            "seed": [0], "latent_image": [latent],
        })
        result = read(context(owner), include_observations=True)
        assert result["execution_order"] == ["1", "2"]
        records = result["observations"]
        output = records["1"]["outputs"][0][0]
        consumed = records["2"]["inputs"]["positive"][0]
        assert output["identity"] == consumed["identity"]
        assert records["1"]["inputs"] == {"text": ["actual resolved text"]}
        assert records["2"]["inputs"]["noise"][0]["seed"] == 2**63 + 7
        samples = records["2"]["inputs"]["latent_image"][0]["items"][0][1]
        assert samples == {"kind": "tensor", "shape": [1, 4, 8, 9], "dtype": "torch.float32"}
        assert "never expose" not in json.dumps(result)


def test_passthrough_any_conditioning_keeps_identity():
    class AnyFixture(TextFixture):
        RETURN_TYPES = ("*",)
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"value": ("*",)}}
    cond = [[torch.zeros(1, 2, 4), {}]]
    with em.ExecutionMetadataScope("any", DynamicPrompt(copy.deepcopy(PROMPT)), observe=True) as owner:
        owner.record_outputs("1", TextFixture, [[cond]])
        owner.record_start("2", AnyFixture, {"value": [cond]})
        owner.record_outputs("2", AnyFixture, [[cond]])
        result = read(context(owner), include_observations=True)["observations"]
        assert result["1"]["outputs"][0][0]["identity"] == result["2"]["outputs"][0][0]["identity"]
        assert result["2"]["inputs"]["value"][0]["identity"] == result["1"]["outputs"][0][0]["identity"]


def test_cached_record_uses_actual_values_without_claiming_execution():
    cond = [[torch.zeros(1, 2, 4), {}]]
    with em.ExecutionMetadataScope("warm", DynamicPrompt(copy.deepcopy(PROMPT)), observe=True) as owner:
        owner.record_cached("1", TextFixture, {"text": ["cached real text"]}, [[cond]])
        owner.record_start("2", SamplerFixture, {"positive": [cond], "seed": [0]})
        snapshot = read(context(owner), include_observations=True)
        assert snapshot["execution_order"] == ["2"]
        assert snapshot["observations"]["1"]["cached"] is True
        assert snapshot["observations"]["1"]["inputs"]["text"] == ["cached real text"]
        assert snapshot["observations"]["1"]["outputs"][0][0]["identity"] == snapshot["observations"]["2"]["inputs"]["positive"][0]["identity"]


@pytest.mark.parametrize("failure", ["permission", "foreign_plan", "foreign_owner", "closed", "unbound_runtime", "foreign_context", "unknown_node"])
def test_owner_and_lifetime_refusals(failure):
    owner = em.ExecutionMetadataScope("owner", DynamicPrompt(copy.deepcopy(PROMPT)), EXTRA)
    with owner:
        ctx = context(owner, permissions=() if failure == "permission" else ("graph.metadata",),
                      prompt_id="foreign" if failure == "foreign_plan" else None,
                      node_id="missing" if failure == "unknown_node" else "3")
        if failure == "closed":
            owner.close()
        if failure == "foreign_owner":
            with em.ExecutionMetadataScope("other", DynamicPrompt(copy.deepcopy(PROMPT))) as other:
                ctx.graph._execution_metadata = other
                with pytest.raises(RuntimeError, match="does not belong"):
                    read(ctx)
            return
        if failure == "unbound_runtime":
            with pytest.raises(RuntimeError, match="No active"):
                asyncio.run(ctx.graph.execution_metadata())
            return
        if failure == "foreign_context":
            async def run():
                with _sdk.bind_runtime(_sdk.InProcessRefResolver(), context(owner), _sdk.InProcessOps()):
                    with pytest.raises(RuntimeError, match="not current"):
                        await ctx.graph.execution_metadata()
            asyncio.run(run())
            return
        with pytest.raises(Exception) as caught:
            read(ctx)
        assert failure == "unknown_node" or isinstance(caught.value, (PermissionError, RuntimeError))


@pytest.mark.parametrize("bad", [None, 0, 1, "true", lambda: None])
def test_observation_flag_is_not_coerced(bad):
    with em.ExecutionMetadataScope("bool", DynamicPrompt(copy.deepcopy(PROMPT))) as owner:
        with pytest.raises(TypeError, match="boolean"):
            read(context(owner), include_observations=bad)


def test_callbacks_cannot_stand_in_for_the_owner():
    class FakeOwner:
        def snapshot(self, *_args, **_kwargs):
            raise AssertionError("fake producer executed")
    with em.ExecutionMetadataScope("real", DynamicPrompt(copy.deepcopy(PROMPT))) as owner:
        ctx = context(owner)
        ctx.graph._execution_metadata = FakeOwner()
        with pytest.raises(TypeError, match="native owner"):
            read(ctx)


@pytest.mark.parametrize("bad", [b"bytes", object(), lambda: None])
def test_non_json_hidden_metadata_keeps_serialization_error(bad):
    with em.ExecutionMetadataScope("json", DynamicPrompt(copy.deepcopy(PROMPT)), {"bad": bad}) as owner:
        with pytest.raises(TypeError):
            read(context(owner))


def test_snapshot_size_is_bounded_without_truncation():
    with em.ExecutionMetadataScope("large", DynamicPrompt(copy.deepcopy(PROMPT)), {"value": "x" * em.MAX_METADATA_BYTES}) as owner:
        with pytest.raises(ValueError, match="1 MiB"):
            read(context(owner))


def test_failed_observation_is_reported_without_replacing_prompt_snapshot():
    with em.ExecutionMetadataScope("error", DynamicPrompt(copy.deepcopy(PROMPT)), EXTRA, observe=True) as owner:
        owner.record_start("1", TextFixture, {"text": ["x" * (em.MAX_METADATA_BYTES + 1)]})
        assert read(context(owner))["extra_pnginfo"] == EXTRA
        with pytest.raises(RuntimeError, match="1 MiB"):
            read(context(owner), include_observations=True)
        assert not owner._identities


def test_close_releases_borrowed_conditioning_and_does_not_retain_models():
    class Opaque:
        def __repr__(self):
            raise AssertionError("opaque object was inspected")
    value = Opaque()
    value_ref = weakref.ref(value)
    with em.ExecutionMetadataScope("release", DynamicPrompt(copy.deepcopy(PROMPT)), observe=True) as owner:
        owner.record_outputs("1", TextFixture, [[[[torch.zeros(1, 2, 4), {}]]]])
        owner.record_outputs("2", SamplerFixture, [[value]])
        del value
        gc.collect()
        assert value_ref() is None
        assert owner._identities
    assert owner.closed and not owner._identities and not owner.observations


class ServerFixture:
    client_id = None
    last_node_id = None
    def send_sync(self, *_args):
        pass


class ScalarNodeFixture:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING",)}}
    def run(self, text):
        return (text + " resolved",)


class MetadataReaderFixture:
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    SDK_PERMISSIONS = ("graph.metadata",)
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING",)}}
    async def run(self, text):
        snapshot = await _sdk.current_context().graph.execution_metadata(include_observations=True)
        return {"result": (), "ui": {"metadata": [snapshot]}}


def test_real_executor_populates_cold_and_cached_inputs_and_closes_scope(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "MetadataScalarFixture", ScalarNodeFixture)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "MetadataReaderFixture", MetadataReaderFixture)
    prompt = {
        "1": {"class_type": "MetadataScalarFixture", "inputs": {"text": "base"}},
        "2": {"class_type": "MetadataReaderFixture", "inputs": {"text": ["1", 0]}},
    }
    executor = execution.PromptExecutor(ServerFixture(), cache_args={"ram": 0, "ram_inactive": 0})
    async def run():
        await executor.execute_async(prompt, "cold", {"extra_pnginfo": EXTRA}, ["2"])
        first = (await executor.caches.outputs.get("2")).ui["output"]["metadata"][0]
        assert first["node_id"] == "2" and first["extra_pnginfo"] == EXTRA
        assert first["execution_order"] == ["1", "2"]
        assert first["observations"]["2"]["inputs"]["text"] == ["base resolved"]
        await executor.execute_async(prompt, "warm", {"extra_pnginfo": {"second_owner": True}}, ["2"])
        second = (await executor.caches.outputs.get("2")).ui["output"]["metadata"][0]
        assert second["observations"]["1"]["cached"] is True
        assert second["observations"]["1"]["inputs"]["text"] == ["base"]
        assert second["execution_order"] == ["2"]
        assert second["prompt_id"] == "warm" and second["extra_pnginfo"] == {"second_owner": True}
        assert em.current("cold") is None and em.current("warm") is None
    try:
        asyncio.run(run())
    finally:
        executor.close()


def test_reader_task_created_before_prompt_uses_explicit_runtime_owner():
    async def run():
        ready = asyncio.Event()
        ctx = None
        async def reader():
            await ready.wait()
            assert em.current("reader") is None
            with _sdk.bind_runtime(_sdk.InProcessRefResolver(), ctx, _sdk.InProcessOps()):
                return await ctx.graph.execution_metadata()
        task = asyncio.create_task(reader())
        with em.ExecutionMetadataScope("reader", DynamicPrompt(copy.deepcopy(PROMPT)), EXTRA) as owner:
            ctx = context(owner)
            ready.set()
            result = await task
            assert result["prompt_id"] == "reader" and result["extra_pnginfo"] == EXTRA
        with _sdk.bind_runtime(_sdk.InProcessRefResolver(), ctx, _sdk.InProcessOps()):
            with pytest.raises(RuntimeError, match="not current"):
                await ctx.graph.execution_metadata()
    asyncio.run(run())
