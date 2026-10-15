"""Opaque constructors retain native state, placement and failure behavior."""
import asyncio
from collections import OrderedDict

import pytest
import torch

import comfy.sd
from comfy.audio_encoders import audio_encoders
from comfy_api.latest import _sdk, sdk


METHODS = ['vae_from_state_dict', 'audio_encoder_from_state_dict']


def context():
    return _sdk.InProcessCtxProvider().build(_sdk.ExecutionPlan(
        prompt_id='state-models', node_id='1', node_type='StateModels',
        node_module=__name__, inputs={}))


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.uint8])
def test_native_constructor_receives_original_mapping_and_storage(monkeypatch, method, dtype):
    state = OrderedDict(weight=torch.arange(12, dtype=dtype).reshape(3, 4))
    state._metadata = {'': {'version': 3}}
    value = object()
    received = []

    def vae(*, sd):
        received.append(sd)
        return value

    def audio(sd):
        received.append(sd)
        return value

    monkeypatch.setattr(comfy.sd, 'VAE', vae)
    monkeypatch.setattr(audio_encoders, 'load_audio_encoder_from_sd', audio)

    async def run():
        refs = _sdk.InProcessRefResolver()
        with _sdk.bind_runtime(refs, context(), _sdk.InProcessOps()):
            source = await sdk.ValueRef.from_value(state)
            result = await getattr(sdk.ctx().models, method)(source)
            expected = sdk.VaeRef if method == METHODS[0] else sdk.AudioEncoderRef
            assert type(result) is expected
            assert await refs.resolve(result) is value
            assert received == [state] and received[0] is state
            assert received[0]._metadata is state._metadata
            assert received[0]['weight'].data_ptr() == state['weight'].data_ptr()
            assert received[0]['weight'].dtype is dtype
            await refs.release(result)
            with pytest.raises(KeyError):
                await refs.resolve(result)
            assert refs.clear() == 1
    asyncio.run(run())


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('state', [None, [], 3, {3: torch.ones(1)}, {'weight': 1}])
def test_non_tensor_state_refused_before_native_constructor(monkeypatch, method, state):
    def unexpected(*args, **kwargs):
        raise AssertionError('native constructor reached invalid input')

    monkeypatch.setattr(comfy.sd, 'VAE', unexpected)
    monkeypatch.setattr(audio_encoders, 'load_audio_encoder_from_sd', unexpected)

    async def run():
        refs = _sdk.InProcessRefResolver()
        with _sdk.bind_runtime(refs, context(), _sdk.InProcessOps()):
            source = await sdk.ValueRef.from_value(state)
            with pytest.raises(TypeError):
                await getattr(sdk.ctx().models, method)(source)
            assert len(refs._table) == 1
            refs.clear()
    asyncio.run(run())


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('case', ['plain', 'wrong_kind', 'relabeled', 'unknown', 'released', 'foreign'])
def test_ref_identity_and_lifetime_before_loading(monkeypatch, method, case):
    def unexpected(*args, **kwargs):
        raise AssertionError('native constructor reached an unauthorized ref')

    monkeypatch.setattr(comfy.sd, 'VAE', unexpected)
    monkeypatch.setattr(audio_encoders, 'load_audio_encoder_from_sd', unexpected)

    async def run():
        refs = _sdk.InProcessRefResolver()
        with _sdk.bind_runtime(refs, context(), _sdk.InProcessOps()):
            state = {'weight': torch.ones(1)}
            source = await sdk.ValueRef.from_value(state)
            error = TypeError
            if case == 'plain':
                source = state
            elif case == 'wrong_kind':
                source = sdk.ValueRef(kind='IMAGE', id=source.id)
            elif case == 'relabeled':
                image = await refs.create('IMAGE', state)
                source = sdk.ValueRef(kind='VALUE', id=image.id)
            elif case == 'unknown':
                source = sdk.ValueRef(kind='VALUE', id='unknown')
                error = KeyError
            elif case == 'released':
                await refs.release(source)
                error = KeyError
            elif case == 'foreign':
                other = _sdk.InProcessRefResolver()
                source = sdk.ValueRef._wrap(await other.create('VALUE', state))
                error = KeyError
            before = len(refs._table)
            with pytest.raises(error):
                await getattr(sdk.ctx().models, method)(source)
            assert len(refs._table) == before
            refs.clear()
    asyncio.run(run())


@pytest.mark.parametrize('method', METHODS)
def test_native_failure_propagates_without_output_ref(monkeypatch, method):
    failure = RuntimeError('native state load failure')

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(comfy.sd, 'VAE', fail)
    monkeypatch.setattr(audio_encoders, 'load_audio_encoder_from_sd', fail)

    async def run():
        refs = _sdk.InProcessRefResolver()
        with _sdk.bind_runtime(refs, context(), _sdk.InProcessOps()):
            source = await sdk.ValueRef.from_value({})
            with pytest.raises(RuntimeError) as caught:
                await getattr(sdk.ctx().models, method)(source)
            assert caught.value is failure
            assert len(refs._table) == 1
            refs.clear()
    asyncio.run(run())


def test_native_pixel_space_vae_encode_decode_matches_direct_core():
    async def run():
        refs = _sdk.InProcessRefResolver()
        state = {'pixel_space_vae': torch.tensor(1.0)}
        direct = comfy.sd.VAE(sd=state)
        pixels = torch.arange(192, dtype=torch.float32).reshape(1, 8, 8, 3) / 191
        with _sdk.bind_runtime(refs, context(), _sdk.InProcessOps()):
            source = await sdk.ValueRef.from_value(state)
            ref = await sdk.ctx().models.vae_from_state_dict(source)
            actual = await refs.resolve(ref)
            assert type(actual) is comfy.sd.VAE
            encoded = actual.encode(pixels.clone())
            expected = direct.encode(pixels.clone())
            assert torch.equal(encoded, expected)
            assert torch.equal(actual.decode(encoded), direct.decode(expected))
            refs.clear()
    asyncio.run(run())


def test_native_audio_loader_error_is_unchanged():
    state = {'unsupported': torch.ones(1)}
    with pytest.raises(RuntimeError) as expected:
        audio_encoders.load_audio_encoder_from_sd(state)

    async def run():
        refs = _sdk.InProcessRefResolver()
        with _sdk.bind_runtime(refs, context(), _sdk.InProcessOps()):
            source = await sdk.ValueRef.from_value(state)
            with pytest.raises(RuntimeError) as actual:
                await sdk.ctx().models.audio_encoder_from_state_dict(source)
            assert str(actual.value) == str(expected.value)
            assert len(refs._table) == 1
            refs.clear()
    asyncio.run(run())


@pytest.mark.parametrize('model_type', [audio_encoders.AudioEncoderModel,
                                       audio_encoders.SheetSage2AudioEncoder])
def test_audio_encoder_input_projection_uses_public_opaque_type(model_type):
    value = model_type.__new__(model_type)
    assert _sdk._ref_type_for(value) == (sdk.AudioEncoderRef, 'AUDIO_ENCODER')

    async def run():
        refs = _sdk.InProcessRefResolver()
        wrapped = await _sdk.wrap_inputs(refs, {'audio_encoder': value})
        assert type(wrapped['audio_encoder']) is sdk.AudioEncoderRef
        assert await refs.resolve(wrapped['audio_encoder']) is value
        refs.clear()
    asyncio.run(run())
