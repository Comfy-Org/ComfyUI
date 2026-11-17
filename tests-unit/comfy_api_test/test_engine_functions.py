"""Pack functions that call back into the engine: model.add_wrapper, model.set_unet_wrapper and
sampler.custom, on a real (tiny) Wan model, in every mode. Each must sample exactly what the same
function gives when attached with ComfyUI's own API."""
import asyncio
import pathlib
import sys

import pytest
import torch

from comfy_api.v4 import eng, io
from comfy_api.latest import _engine

import engine_loopback as loopback

PROOF = pathlib.Path(__file__).resolve().parents[3] / "poc-packs" / "proof"


@pytest.fixture(scope="module")
def tiny():
    if not (PROOF / "tiny_models.py").is_file():
        pytest.skip("the POC proof kit (tiny engine models) isn't next to this ComfyUI")
    sys.path.insert(0, str(PROOF))
    import tiny_models
    return tiny_models


@pytest.fixture(params=sorted(loopback.MODES))
def run(request):
    mode = loopback.MODES[request.param]

    def go(node, **inputs):
        return asyncio.run(mode(node, "p", inputs))[0]
    return go


def _sample(model, sampler=None, steps=3):
    """Sample the tiny Wan model as SamplerCustom does: real engine sampling, on CPU."""
    import comfy.sample
    import comfy.samplers

    cond = [[torch.ones(1, 4, 4096) * 0.01, {}]]
    latent = torch.zeros(1, 16, 1, 8, 8)
    noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(7))
    sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "simple", steps)
    sampler = sampler or comfy.samplers.ksampler("euler")
    return comfy.sample.sample_custom(model, noise, 1.0, sampler, sigmas, cond, cond, latent, disable_pbar=True, seed=7)


# --------------------------------------------------------------------------- #
# The pack's functions (module level, so a pack process can import them)
# --------------------------------------------------------------------------- #
def halve_prediction(executor, *args, **kwargs):
    out = executor(*args, **kwargs)
    return out * 0.5


def shifted_unet(apply_model, args):
    out = apply_model(args["input"], args["timestep"], **args["c"])
    return out + 0.25


def euler(model, x, sigmas, extra_args, callback=None, disable=None):
    """k-diffusion's Euler, written as pack code."""
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        d = (x - denoised) / sigmas[i]
        x = x + d * (sigmas[i + 1] - sigmas[i])
    return x


class WrapPredict(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WrapPredict", inputs=[io.Model.Input("model")], outputs=[io.Model.Output()])

    @classmethod
    async def execute(cls, model):
        return io.NodeOutput(await eng.model.add_wrapper(model, "predict_noise", halve_prediction))


class WrapUnet(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WrapUnet", inputs=[io.Model.Input("model")], outputs=[io.Model.Output()])

    @classmethod
    async def execute(cls, model):
        return io.NodeOutput(await eng.model.set_unet_wrapper(model, shifted_unet))


class PackEuler(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="PackEuler", inputs=[], outputs=[io.Sampler.Output()])

    @classmethod
    async def execute(cls):
        return io.NodeOutput(await eng.sampler.custom(euler))


# --------------------------------------------------------------------------- #
def test_a_stage_wrapper_samples_as_comfyuis_own_wrapper_does(run, tiny):
    model = tiny.load_model()
    reference = model.clone()
    reference.add_wrapper_with_key("predict_noise", None, halve_prediction)  # ComfyUI's own API
    wrapped = run(WrapPredict, model=model).result[0]
    expected, got = _sample(reference), _sample(wrapped)
    assert torch.equal(got, expected)
    assert not torch.equal(got, _sample(model))  # the wrapper did something
    assert "predict_noise" not in model.wrappers  # the input model is unchanged


def test_a_unet_wrapper_samples_as_comfyuis_own_wrapper_does(run, tiny):
    model = tiny.load_model()
    reference = model.clone()
    reference.set_model_unet_function_wrapper(shifted_unet)
    wrapped = run(WrapUnet, model=model).result[0]
    assert torch.equal(_sample(wrapped), _sample(reference))
    assert "model_function_wrapper" not in model.model_options


def test_a_pack_sampler_samples_as_comfyuis_own_sampler_does(run, tiny):
    import comfy.samplers

    model = tiny.load_model()
    sampler = run(PackEuler).result[0]
    import comfy.sample
    expected = _sample(model, comfy.samplers.ksampler("euler"))
    got = _sample(model, sampler)
    assert torch.equal(got, expected)


def test_the_pack_function_sees_engine_values_as_opaque_and_can_pass_them_back():
    seen = {}

    def fn(executor, guider, tensor, options):
        seen.update(guider=guider, options=options)
        return executor(guider, tensor * 2)

    class Guider:
        pass

    guider = Guider()
    run = _engine.pack_function(fn, engine_args=(0,))
    result = run(lambda g, t: (g, t + 1), guider, torch.ones(1), {"model": guider, "n": 3})
    assert isinstance(seen["guider"], _engine.Opaque) and isinstance(seen["options"]["model"], _engine.Opaque)
    assert seen["options"]["n"] == 3
    assert result[0] is guider and torch.equal(result[1], torch.full((1,), 3.0))


def test_a_wrapper_kind_comfyui_doesnt_have_is_refused(tiny):
    model = tiny.load_model()

    class Bad(io.ComfyNode):
        ENGINE_API = True

        @classmethod
        async def execute(cls, model):
            return io.NodeOutput(await eng.model.add_wrapper(model, "everything", halve_prediction))

    with pytest.raises(ValueError, match="wrapper kind"):
        asyncio.run(loopback.run_in_process(Bad, "p", {"model": model}))


@pytest.mark.parametrize("sigmas,ok", [
    (torch.tensor([1.0, 0.5, 0.0]), True), (torch.tensor([1.0, 0.5]), True),  # SplitSigmas' high half
    (torch.tensor([0.0]), False), (torch.tensor([1.0, 2.0, 0.0]), False), (torch.tensor([float("nan"), 0.0]), False),
    (torch.tensor([1.0, -0.1]), False), (torch.ones(2, 2), False)])
def test_pack_samplers_get_the_overlays_sigma_checks(sigmas, ok):
    if ok:
        _engine.check_sigmas(sigmas)
    else:
        with pytest.raises(ValueError):
            _engine.check_sigmas(sigmas)


INTERACT_ANSWER = {"choice": [1]}


class AskUser(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="AskUser", inputs=[], outputs=[io.String.Output()])

    @classmethod
    async def execute(cls):
        import json as _json
        answer = await eng.ui.interact("prompt-await", {"message": "continue?"}, timeout=10.0)
        return io.NodeOutput(_json.dumps(answer, sort_keys=True))


def test_asking_the_user_goes_through_the_overlays_broker(run, monkeypatch):
    import json as _json
    import threading as _threading
    import server as comfy_server
    from comfy_api.latest import interactions

    sent = []

    class FakeServer:
        client_id = "client-1"

        def send_sync(self, event, data, sid=None):
            sent.append((event, data, sid))
            token = data.get("request_id") or data.get("id") or data.get("token")
            response = {"text": "yes"} if data.get("kind") == "prompt-await" else INTERACT_ANSWER
            _threading.Timer(0.05, lambda: interactions.BROKER.respond(token, response)).start()
    monkeypatch.setattr(comfy_server.PromptServer, "instance", FakeServer(), raising=False)
    result = run(AskUser).result[0]
    assert sent and sent[0][0] == interactions.EVENT and sent[0][2] == "client-1"
    assert _json.loads(result)  # the answer came back to the node
