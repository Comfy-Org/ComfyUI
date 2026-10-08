"""Tensor socket declarations outrank the shape-only compatibility fallback."""
import asyncio

import pytest
import torch

from comfy_api.latest import _sdk, io


@pytest.mark.parametrize("kind,ref_class", [("MASK", _sdk.MaskRef), ("IMAGE", _sdk.ImageRef), ("TENSOR", _sdk.TensorRef), ("SIGMAS", _sdk.SigmasRef)])
def test_declared_tensor_kind_preserves_the_original_numeric_buffer(kind, ref_class):
    async def run():
        refs = _sdk.InProcessRefResolver()
        value = torch.ones(1, 8, 8)
        result = (await _sdk.wrap_inputs(refs, {"value": value}, {"value": kind}))["value"]
        assert type(result) is ref_class and result.kind == kind
        assert await refs.resolve(result) is value
    asyncio.run(run())


def test_schema_hint_does_not_turn_an_object_into_a_tensor_or_engine_model():
    async def run():
        refs = _sdk.InProcessRefResolver()
        value = object()
        result = (await _sdk.wrap_inputs(refs, {"value": value}, {"value": "MASK"}))["value"]
        assert result.kind == "OPAQUE"
        tensor = torch.ones(1, 8, 8)
        result = (await _sdk.wrap_inputs(refs, {"value": tensor}, {"value": "MODEL"}))["value"]
        assert result.kind == "IMAGE"
        assert await refs.resolve(result) is tensor
    asyncio.run(run())


def test_list_mapping_refs_and_untyped_nested_sockets_keep_their_contract():
    async def run():
        refs = _sdk.InProcessRefResolver()
        value = torch.ones(1, 8, 8)
        existing = _sdk.MaskRef._wrap(await refs.create("MASK", value))
        result = await _sdk.wrap_inputs(refs, {"masks": [value, existing], "custom": {"tensor": value}, "control": "option"}, {"masks": "MASK", "control": ["option"]})
        assert type(result["masks"][0]) is _sdk.MaskRef
        assert result["masks"][1] is existing
        assert result["custom"]["tensor"].kind == "IMAGE"
        assert result["control"] == "option"
    asyncio.run(run())


class _DeclaredMaskProbe(io.ComfyNode):
    SDK_REFS = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="_DeclaredMaskProbe", inputs=[io.Mask.Input("mask"), io.Image.Input("image"), io.Custom("TENSOR").Input("tensor"), io.Custom("SIGMAS").Input("sigmas")], outputs=[io.String.Output()] )

    @classmethod
    async def execute(cls, mask, image, tensor, sigmas):
        assert type(mask) is _sdk.MaskRef
        assert type(image) is _sdk.ImageRef
        assert type(tensor) is _sdk.TensorRef
        assert type(sigmas) is _sdk.SigmasRef
        return io.NodeOutput("MASK,IMAGE,TENSOR,SIGMAS")


def test_real_outer_executor_passes_declared_mask_and_other_tensor_kinds():
    import execution

    async def run():
        value = torch.ones(1, 8, 8)
        results = await execution._async_map_node_over_list("declared", "1", _DeclaredMaskProbe, {name: [value] for name in ("mask", "image", "tensor", "sigmas")}, _DeclaredMaskProbe.FUNCTION)
        result = (await execution.resolve_map_node_over_list_results(results))[0]
        assert result.result == ("MASK,IMAGE,TENSOR,SIGMAS",)
    asyncio.run(run())
