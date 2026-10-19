import asyncio
from types import SimpleNamespace

import pytest
import torch
import comfy.sd
import comfy.model_patcher
from comfy_api.latest import _sdk, sdk
from comfy_api.latest._gguf_tensor import GGUFTensor, GGMLType
from comfy_api.latest._ggml_patcher import GGMLModelPatcher


def context():
    return _sdk.InProcessCtxProvider().build(_sdk.ExecutionPlan(
        prompt_id='ggml-construction', node_id='1', node_type='GGML',
        node_module=__name__, inputs={}))


@pytest.mark.parametrize('method', ['ggml_diffusion_from_state_dict', 'ggml_text_encoders_from_state_dicts'])
@pytest.mark.parametrize('precision', [None, 'default', 'float32'])
def test_constructor_routes_original_state_and_owned_ggml_ops(monkeypatch, method, precision):
    raw = torch.cat((torch.tensor([0.5], dtype=torch.float16).view(torch.uint8),
                     torch.arange(32, dtype=torch.uint8))).expand(2, -1).clone()
    weight = GGUFTensor(raw, tensor_type=GGMLType.Q8_0, tensor_shape=(2, 32))
    state = {'weight': weight}
    observed = []
    def native(state=None, model_options=None, metadata=None, **kwargs):
        if state is None:
            state = kwargs['state_dicts'][0]
        observed.append((state, model_options, metadata, kwargs))
        layer = model_options['custom_operations'].Linear(32, 2, bias=False)
        layer.load_state_dict(state)
        patcher = comfy.model_patcher.ModelPatcher(layer, torch.device('cpu'), torch.device('cpu'))
        return patcher if method == 'ggml_diffusion_from_state_dict' else SimpleNamespace(patcher=patcher)
    monkeypatch.setattr(comfy.sd, 'load_diffusion_model_state_dict', native)
    monkeypatch.setattr(comfy.sd, 'load_text_encoder_state_dicts', native)
    async def run():
        refs = _sdk.InProcessRefResolver()
        with _sdk.bind_runtime(refs, context(), _sdk.InProcessOps()):
            ref = await sdk.ValueRef.from_value(state)
            if method == 'ggml_diffusion_from_state_dict':
                result = await sdk.ctx().models.ggml_diffusion_from_state_dict(
                    ref, {'test': 'source metadata'}, dequant_dtype=precision,
                    patch_dtype='target', patch_on_device=True)
            else:
                result = await sdk.ctx().models.ggml_text_encoders_from_state_dicts([ref], 'sd3')
            value = await refs.resolve(result)
            patcher = value if method == 'ggml_diffusion_from_state_dict' else value.patcher
            assert type(patcher) is GGMLModelPatcher
            assert result.kind == ('MODEL' if method == 'ggml_diffusion_from_state_dict' else 'CLIP')
            assert patcher.model.weight.dtype is torch.uint8
            expected = torch.nn.functional.linear(torch.ones(1, 32), torch.arange(32.).mul(0.5).expand(2, -1))
            assert torch.equal(patcher.model(torch.ones(1, 32)), expected)
            assert observed[0][0] is state
            if method == 'ggml_diffusion_from_state_dict':
                assert observed[0][2] == {'test': 'source metadata'} and patcher.patch_on_device is True
            else:
                assert observed[0][3]['clip_type'] is comfy.sd.CLIPType.SD3
                assert observed[0][1]['initial_device'] == comfy.model_management.text_encoder_offload_device()
        refs.clear()
    asyncio.run(run())


@pytest.mark.parametrize('method', ['ggml_diffusion_from_state_dict', 'ggml_text_encoders_from_state_dicts'])
@pytest.mark.parametrize('case', ['released', 'foreign', 'relabeled', 'unsupported-quant'])
def test_invalid_or_unsupported_state_precedes_native_build(monkeypatch, method, case):
    def unexpected(*args, **kwargs):
        raise AssertionError('native build reached invalid input')
    monkeypatch.setattr(comfy.sd, 'load_diffusion_model_state_dict', unexpected)
    monkeypatch.setattr(comfy.sd, 'load_text_encoder_state_dicts', unexpected)
    async def run():
        refs = _sdk.InProcessRefResolver()
        foreign = _sdk.InProcessRefResolver()
        value = GGUFTensor(torch.zeros(32, dtype=torch.uint8), tensor_type=10, tensor_shape=(32,))
        owner = foreign if case == 'foreign' else refs
        raw = await owner.create('ASSET' if case == 'relabeled' else 'VALUE', {'weight': value})
        ref = _sdk.ValueRef(kind='VALUE', id=raw.id)
        if case == 'released':
            await refs.release(ref)
        with _sdk.bind_runtime(refs, context(), _sdk.InProcessOps()):
            with pytest.raises((KeyError, TypeError, ValueError, NotImplementedError)):
                if method == 'ggml_diffusion_from_state_dict':
                    await sdk.ctx().models.ggml_diffusion_from_state_dict(ref)
                else:
                    await sdk.ctx().models.ggml_text_encoders_from_state_dicts([ref], 'sd3')
        refs.clear()
        foreign.clear()
    asyncio.run(run())


@pytest.mark.parametrize('clip_type', ['', 'not-a-clip', None])
def test_clip_type_is_not_silently_changed(clip_type):
    async def run():
        refs = _sdk.InProcessRefResolver()
        with _sdk.bind_runtime(refs, context(), _sdk.InProcessOps()):
            ref = await sdk.ValueRef.from_value({})
            with pytest.raises((TypeError, ValueError)):
                await sdk.ctx().models.ggml_text_encoders_from_state_dicts([ref], clip_type)
        refs.clear()
    asyncio.run(run())


@pytest.mark.parametrize('size', [0, 5])
def test_encoder_count_is_bounded(size):
    async def run():
        refs = _sdk.InProcessRefResolver()
        with _sdk.bind_runtime(refs, context(), _sdk.InProcessOps()):
            ref = await sdk.ValueRef.from_value({})
            with pytest.raises(ValueError, match='1..4'):
                await sdk.ctx().models.ggml_text_encoders_from_state_dicts([ref] * size, 'sd3')
        refs.clear()
    asyncio.run(run())
