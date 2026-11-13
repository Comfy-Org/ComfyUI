"""V4 callbacks of a pack that runs in its own process: hooks, custom samplers, noise.

The same pack runs in ComfyUI's process and in its own, and the same sampling is compared bit for
bit. Each hook runs in two situations: inside the pack's node, which samples with
`eng.sampler.sample`, and later, when ComfyUI samples the patched MODEL the node returned.
"""

import asyncio
import gc
import os
import threading
import time

import pytest
import torch

import nodes
from comfy import model_management, supported_models
from comfy.model_base import BaseModel, ModelType
from comfy.model_management import interrupt_current_processing
from comfy.model_patcher import ModelPatcher
from comfy_api.v4 import _pack_host, _pack_processes, io
from comfy_execution.utils import CurrentNodeContext
from comfy_extras.nodes_custom_sampler import (
    BasicScheduler,
    CFGGuider,
    SamplerCustomAdvanced,
)
from conftest import needs_pack_process

pytestmark = needs_pack_process

CALLBACK_PACK = """\
import gc
import os
import threading
import time
import weakref

import torch
from comfy_api.v4 import ComfyAPISync, ComfyExtension, eng, io

SEEN = []
MOVED = weakref.WeakSet()


def seen(name, *tensors):
    SEEN.append(
        {
            "hook": name,
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
            "modes": (torch.is_inference_mode_enabled(), torch.is_grad_enabled()),
            "devices": sorted({str(tensor.device) for tensor in tensors}),
        }
    )


def cfg(args):
    seen("set_cfg", args["cond"], args["uncond"], args["input"])
    cond, uncond = args["cond"], args["uncond"]
    guided = uncond + (cond - uncond) * args["cond_scale"]
    return guided * (cond.std() / guided.std()) ** 0.7


def pre_cfg(args):
    seen("set_pre_cfg", *args["conds_out"])
    return [out * 0.9 for out in args["conds_out"]]


def post_cfg(args):
    seen("set_post_cfg", args["denoised"])
    return args["denoised"] * 0.95 + args["uncond_denoised"] * 0.05


def denoise_mask(sigma, denoise_mask, extra_options):
    seen("set_denoise_mask", sigma, denoise_mask)
    return denoise_mask * 0.5


def unet_wrapper(apply_model, args):
    seen("set_unet_wrapper", args["input"])
    return apply_model(args["input"], args["timestep"], **args["c"]) * 1.1


class Lllite:
    def __init__(self, name):
        self.name = name

    def __call__(self, q, k, v, extra_options):
        seen(self.name, q, k, v)
        return q, k * 0.9, v * 0.9

    def to(self, device_or_dtype):
        SEEN.append({"hook": "to", "argument": repr(device_or_dtype)})
        return self

    def cleanup(self):
        SEEN.append({"hook": "cleanup"})


def attn_output(name):
    def patch(out, extra_options):
        seen(name, out)
        return out * 0.9

    return patch


def input_block(h, extra_options):
    seen("set_input_block_patch", h)
    return h * 0.9


def output_block(h, skip, extra_options):
    seen("set_output_block_patch", h, skip)
    return h, skip * 0.9


def attention_override(attention, q, k, v, heads, **options):
    seen("set_attention_override", q, k, v)
    return attention(q * 0.9, k, v, heads, **options)


class HookError(Exception):
    pass


def raises_its_own_error(args):
    raise HookError("the hook failed")


def raises_value_error(args):
    raise ValueError("the hook failed")


def checks_for_cancel(args):
    ComfyAPISync().execution.check_interrupted()
    return cfg(args)


def spins(args):
    ComfyAPISync().ui.send("cancel", {})
    while True:
        time.sleep(0.01)


def dies(args):
    os._exit(5)


# Each move gives a new object, as a patch holding weights may.
class Moving:
    def __init__(self):
        MOVED.add(self)

    def __call__(self, q, k, v, extra_options):
        return q, k, v

    def to(self, device_or_dtype):
        return Moving()


def uses_its_model(model):
    def cfg_after_half(args):
        half = ComfyAPISync().model.percent_to_sigma(model, 0.5)
        return cfg(args) if float(args["sigma"][0]) < half else args["cond"]

    return cfg_after_half


class Holding(Lllite):
    def __init__(self, extra):
        super().__init__("holding")
        self.extra = extra

    def models(self):
        return [self.extra]


HOOKS = {
    "set_cfg": cfg,
    "set_pre_cfg": pre_cfg,
    "set_post_cfg": post_cfg,
    "set_denoise_mask": denoise_mask,
    "set_unet_wrapper": unet_wrapper,
    "set_attn1_patch": Lllite("set_attn1_patch"),
    "set_attn2_patch": Lllite("set_attn2_patch"),
    "set_attn1_output_patch": attn_output("set_attn1_output_patch"),
    "set_attn2_output_patch": attn_output("set_attn2_output_patch"),
    "set_input_block_patch": input_block,
    "set_output_block_patch": output_block,
    "set_attention_override": attention_override,
}
CFG_HOOKS = {
    "raises_its_own_error": raises_its_own_error,
    "raises_value_error": raises_value_error,
    "checks_for_cancel": checks_for_cancel,
    "spins": spins,
    "dies": dies,
}


# `hook` names an eng.model call, patched with its hook above, or another hook.
async def patch(model, hook, extra):
    if hook in HOOKS:
        return await getattr(eng.model, hook)(model, HOOKS[hook])
    if hook == "holds_a_model":
        return await eng.model.set_attn1_patch(model, Holding(extra))
    if hook == "moving":
        return await eng.model.set_attn1_patch(model, Moving())
    return await eng.model.set_cfg(model, uses_its_model(model) if hook == "uses_its_model" else CFG_HOOKS[hook])


def euler(model, x, sigmas, extra_args=None, callback=None, disable=None, scale=1.0):
    seen("custom", x)
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = (x - denoised) / sigmas[i]
        callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        x = x + d * (sigmas[i + 1] - sigmas[i]) * scale
    return x


class SeededNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        seen("noise")
        generator = torch.Generator().manual_seed(self.seed)
        return torch.randn(input_latent["samples"].shape, generator=generator) * 0.5


class Patch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Patch",
            inputs=[io.Model.Input("model"), io.String.Input("hook"), io.Model.Input("extra", optional=True)],
            outputs=[io.Model.Output()],
        )

    @classmethod
    async def execute(cls, model, hook, extra=None):
        return io.NodeOutput(await patch(model, hook, extra))


class PatchAndSample(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PatchAndSample",
            inputs=[
                io.Model.Input("model"),
                io.String.Input("hook"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent"),
                io.Model.Input("extra", optional=True),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    async def execute(cls, model, hook, positive, negative, latent, extra=None):
        patched = await patch(model, hook, extra)
        return io.NodeOutput(
            await eng.sampler.sample(patched, 1, 3, 5.0, "euler", "normal", positive, negative, latent)
        )


class CustomParts(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CustomParts", outputs=[io.Sampler.Output(), io.Noise.Output()]
        )

    @classmethod
    async def execute(cls):
        return io.NodeOutput(await eng.sampler.custom(euler, {"scale": 0.9}), SeededNoise(7))


class SampleCustom(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SampleCustom",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent"),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    async def execute(cls, model, positive, negative, latent):
        sampler, noise = (await CustomParts.execute()).args
        guider = await eng.guider.cfg(model, positive, negative, 5.0)
        sigmas = await eng.sampler.sigmas(model, "normal", 3)
        return io.NodeOutput(await eng.sampler.sample_custom(noise, guider, sampler, sigmas, latent))


class Seen(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Seen", outputs=[io.AnyType.Output()])

    @classmethod
    def execute(cls):
        calls = list(SEEN)
        SEEN.clear()
        return io.NodeOutput(calls)


class Moved(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Moved", outputs=[io.Int.Output()])

    @classmethod
    def execute(cls):
        gc.collect()
        return io.NodeOutput(len(MOVED))


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [Patch, PatchAndSample, CustomParts, SampleCustom, Seen, Moved]


async def comfy_entrypoint():
    return Extension()
"""
MODULE = "v4_callback_pack"


@pytest.fixture(scope="module")
def pack_process(tmp_path_factory):
    """The callback pack in its own process, shared by this module's tests: its extension and
    its process."""
    path = tmp_path_factory.mktemp("callbacks") / f"{MODULE}.py"
    path.write_text(CALLBACK_PACK, encoding="utf-8")
    extension = asyncio.run(_pack_host.load(str(path), MODULE)).comfy_entrypoint()
    yield extension, _pack_processes.processes[str(path)]
    _pack_processes.unload(str(path))


@pytest.fixture
def in_process(load_pack):
    """The callback pack loaded in ComfyUI's process; returns a node runner."""

    async def run(node_id, **inputs):
        if node_id not in nodes.NODE_CLASS_MAPPINGS:
            await load_pack(CALLBACK_PACK, MODULE)
        output = nodes.NODE_CLASS_MAPPINGS[node_id].execute(**inputs)
        if asyncio.iscoroutine(output):
            output = await output
        return output.args

    return run


@pytest.fixture
def isolated(pack_process):
    """The callback pack in its own process, with nothing seen yet; returns a node runner."""

    _, process = pack_process

    async def run(node_id, **inputs):
        output = await process.run(("node", node_id, "execute"), (inputs, {}))
        return output.args

    asyncio.run(run("Seen"))
    run.pid = process.process.pid
    return run


@pytest.fixture(params=["in_process", "pack_process"])
def pack(request):
    """The callback pack, in ComfyUI's process or in its own."""
    return request.getfixturevalue(
        "isolated" if request.param == "pack_process" else "in_process"
    )


@pytest.fixture
def unet():
    config = supported_models.SD15(
        {
            "image_size": 32,
            "in_channels": 4,
            "model_channels": 32,
            "out_channels": 4,
            "num_res_blocks": [1, 1],
            "transformer_depth": [1, 1],
            "transformer_depth_output": [1, 1, 1, 1],
            "transformer_depth_middle": 1,
            "channel_mult": [1, 2],
            "use_spatial_transformer": True,
            "context_dim": 16,
            "num_head_channels": 16,
            "use_linear_in_transformer": True,
            "dtype": torch.float32,
        }
    )
    native = BaseModel(config, ModelType.EPS, device=torch.device("cpu"))
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in native.parameters():
            parameter.normal_(0, 0.05, generator=generator)
    return ModelPatcher(native, torch.device("cpu"), torch.device("cpu"))


def conditioning(seed):
    generator = torch.Generator().manual_seed(seed)
    return [[torch.randn(1, 3, 16, generator=generator), {}]]


POSITIVE, NEGATIVE = conditioning(1), conditioning(2)
LATENT = {
    "samples": torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(3)),
    "noise_mask": torch.ones(1, 1, 8, 8),
}


def ksampler(model):
    """Sample as ComfyUI's KSampler node does, with the settings PatchAndSample uses."""
    with CurrentNodeContext("prompt", "ksampler"):
        return nodes.common_ksampler(
            model, 1, 3, 5.0, "euler", "normal", POSITIVE, NEGATIVE, LATENT
        )[0]


async def sample(pack, model, hook, situation, **extra):
    """Sample `model` patched with the pack's `hook`, inside the pack's node or later in ComfyUI."""
    if situation == INSIDE:
        (latent,) = await pack(
            "PatchAndSample",
            model=model,
            hook=hook,
            positive=POSITIVE,
            negative=NEGATIVE,
            latent=LATENT,
            **extra,
        )
        return latent["samples"]
    (patched,) = await pack("Patch", model=model, hook=hook, **extra)
    return ksampler(patched)["samples"]


HOOK_CALLS = [
    "set_cfg",
    "set_pre_cfg",
    "set_post_cfg",
    "set_denoise_mask",
    "set_unet_wrapper",
    "set_attn1_patch",
    "set_attn2_patch",
    "set_attn1_output_patch",
    "set_attn2_output_patch",
    "set_input_block_patch",
    "set_output_block_patch",
    "set_attention_override",
]
INSIDE, LATER = "inside the pack's node", "when ComfyUI samples later"
SITUATIONS = [INSIDE, LATER]


@pytest.mark.asyncio
@pytest.mark.parametrize("situation", SITUATIONS)
@pytest.mark.parametrize("call", HOOK_CALLS)
async def test_a_hook_in_a_pack_process_gives_the_tensors_it_gives_in_process(
    in_process, isolated, unet, call, situation
):
    ran = []
    for pack in (in_process, isolated):
        ran.append(await sample(pack, unet, call, situation))
        (seen,) = await pack("Seen")
        ran.append(
            [(entry["pid"], entry["thread"]) for entry in seen if entry["hook"] == call]
        )
    expected, ran_in_process, actual, ran_isolated = ran

    assert torch.equal(actual, expected)
    assert not torch.equal(actual, ksampler(unet)["samples"])
    assert ran_in_process and len(ran_isolated) == len(ran_in_process)
    assert set(ran_in_process) == {(os.getpid(), threading.current_thread().name)}
    assert set(ran_isolated) == {(isolated.pid, "MainThread")}


@pytest.mark.asyncio
@pytest.mark.parametrize("situation", SITUATIONS)
async def test_a_packs_custom_sampler_and_noise_give_the_tensors_they_give_in_process(
    in_process, isolated, unet, situation
):
    async def sample_custom(pack):
        if situation == INSIDE:
            (latent,) = await pack(
                "SampleCustom",
                model=unet,
                positive=POSITIVE,
                negative=NEGATIVE,
                latent=LATENT,
            )
            return latent["samples"]
        sampler, noise = await pack("CustomParts")
        guider = CFGGuider.execute(
            model=unet, positive=POSITIVE, negative=NEGATIVE, cfg=5.0
        ).result[0]
        sigmas = BasicScheduler.execute(
            model=unet, scheduler="normal", steps=3, denoise=1.0
        ).result[0]
        output = SamplerCustomAdvanced.execute(
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=LATENT,
        )
        return output.result[0]["samples"]

    expected = await sample_custom(in_process)
    actual = await sample_custom(isolated)
    (seen,) = await isolated("Seen")

    assert torch.equal(actual, expected)
    assert {entry["hook"] for entry in seen} == {"custom", "noise"}
    assert {entry["pid"] for entry in seen} == {isolated.pid}


@pytest.mark.asyncio
@pytest.mark.parametrize("situation", SITUATIONS)
async def test_comfyui_moves_a_packs_patch_object_and_cleans_it_up(
    pack, unet, situation
):
    await sample(pack, unet, "set_attn1_patch", situation)
    (seen,) = await pack("Seen")

    protocol = [entry for entry in seen if entry["hook"] in ("to", "cleanup")]
    assert {"hook": "to", "argument": "device(type='cpu')"} in protocol
    assert {"hook": "to", "argument": "torch.float32"} in protocol
    assert {"hook": "cleanup"} in protocol


@pytest.mark.asyncio
async def test_comfyui_loads_the_models_a_packs_patch_object_names(
    pack, unet, monkeypatch
):
    monkeypatch.setattr(model_management, "current_loaded_models", [])
    extra = ModelPatcher(
        torch.nn.Linear(1, 1), torch.device("cpu"), torch.device("cpu")
    )

    await sample(pack, unet, "holds_a_model", INSIDE, extra=extra)

    assert extra in [loaded.model for loaded in model_management.current_loaded_models]


@pytest.mark.asyncio
@pytest.mark.parametrize("situation", SITUATIONS)
@pytest.mark.parametrize("inference", [False, True])
async def test_a_hook_runs_in_the_modes_and_on_the_device_it_runs_in_in_process(
    in_process, isolated, unet, situation, inference
):
    seen = []
    for pack in (in_process, isolated):
        with torch.inference_mode(inference):
            await sample(pack, unet, "set_cfg", situation)
        (entries,) = await pack("Seen")
        seen.append(
            {
                (entry["modes"], *entry["devices"])
                for entry in entries
                if entry["hook"] == "set_cfg"
            }
        )

    # ComfyUI samples without grad.
    assert seen == [{((inference, False), "cpu")}] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("hook", ["uses_its_model", "holds_a_model"])
async def test_a_hook_can_use_comfyuis_objects_from_the_request_that_set_it_only_inside_it(
    in_process, isolated, unet, hook
):
    extra = ModelPatcher(
        torch.nn.Linear(1, 1), torch.device("cpu"), torch.device("cpu")
    )

    inside = [
        await sample(pack, unet, hook, INSIDE, extra=extra)
        for pack in (in_process, isolated)
    ]
    later = await sample(in_process, unet, hook, LATER, extra=extra)

    assert torch.equal(inside[0], inside[1]) and torch.equal(later, inside[0])
    # A pack's handles to ComfyUI's objects end with the request they were given in.
    with pytest.raises(ValueError, match="a handle from another request"):
        await sample(isolated, unet, hook, LATER, extra=extra)


class Given(io.ComfyNode):
    """Outputs the object the test gave it under `name`."""

    given = {}

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Given",
            inputs=[io.String.Input("name")],
            outputs=[io.AnyType.Output()],
        )

    @classmethod
    def execute(cls, name):
        return io.NodeOutput(cls.given[name])


class Keep(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Keep", inputs=[io.AnyType.Input("value")], is_output_node=True
        )

    @classmethod
    def execute(cls, value):
        return io.NodeOutput()


def sampling_prompt(hook: str, situation: str) -> dict:
    """A prompt that samples the given model patched with the callback pack's `hook`."""
    given = {
        name: {"class_type": "Given", "inputs": {"name": name}} for name in Given.given
    }
    inputs = {name: [name, 0] for name in ("positive", "negative")}
    if situation == INSIDE:
        sample = {
            "class_type": "PatchAndSample",
            "inputs": {
                "model": ["model", 0],
                "hook": hook,
                "latent": ["latent", 0],
                **inputs,
            },
        }
    else:
        given["patch"] = {
            "class_type": "Patch",
            "inputs": {"model": ["model", 0], "hook": hook},
        }
        settings = {
            "seed": 1,
            "steps": 3,
            "cfg": 5.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
        }
        sample = {
            "class_type": "KSampler",
            "inputs": {
                "model": ["patch", 0],
                "latent_image": ["latent", 0],
                **inputs,
                **settings,
            },
        }
    return {
        **given,
        "sample": sample,
        "keep": {"class_type": "Keep", "inputs": {"value": ["sample", 0]}},
    }


@pytest.fixture
def given(unet, monkeypatch):
    """ComfyUI's nodes that give the test's model and inputs, and keep a result."""
    monkeypatch.setattr(
        Given,
        "given",
        {"model": unet, "positive": POSITIVE, "negative": NEGATIVE, "latent": LATENT},
    )
    for node_cls in (Given, Keep):
        monkeypatch.setitem(
            nodes.NODE_CLASS_MAPPINGS, node_cls.GET_SCHEMA().node_id, node_cls
        )


def register(extension, monkeypatch):
    for node_cls in extension.nodes:
        monkeypatch.setitem(
            nodes.NODE_CLASS_MAPPINGS, node_cls.GET_SCHEMA().node_id, node_cls
        )


@pytest.fixture(params=["in_process", "pack_process"])
def prompt_with(request, given, load_pack, run_prompt, monkeypatch):
    """Run a prompt that samples `unet` patched with the callback pack's `hook`, the pack in
    ComfyUI's process or in its own; returns the executor."""
    if request.param == "pack_process":
        register(request.getfixturevalue("pack_process")[0], monkeypatch)

    async def run(hook: str, situation: str):
        if "Patch" not in nodes.NODE_CLASS_MAPPINGS:
            await load_pack(CALLBACK_PACK, MODULE)
        executor, _ = await run_prompt(sampling_prompt(hook, situation), ["keep"])
        return executor

    return run


@pytest.fixture
def own_process(given, tmp_path, monkeypatch):
    """The callback pack in a process of its own for this test, its nodes in ComfyUI's table."""
    path = tmp_path / f"{MODULE}.py"
    path.write_text(CALLBACK_PACK, encoding="utf-8")
    extension = asyncio.run(_pack_host.load(str(path), MODULE)).comfy_entrypoint()
    register(extension, monkeypatch)
    yield _pack_processes.processes[str(path)]
    _pack_processes.unload(str(path))


def outcome(executor):
    """The prompt's last status message: success, error or interrupt, with its data."""
    return executor.status_messages[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("situation", SITUATIONS)
@pytest.mark.parametrize(
    "hook, exception_type",
    [("raises_value_error", "ValueError"), ("raises_its_own_error", ".HookError")],
)
async def test_an_error_raised_in_a_packs_hook_is_the_error_of_the_node_that_sampled(
    prompt_with, situation, hook, exception_type
):
    executor = await prompt_with(hook, situation)

    event, data = outcome(executor)
    assert (event, data["node_id"]) == ("execution_error", "sample")
    assert data["node_type"] == (
        "PatchAndSample" if situation == INSIDE else "KSampler"
    )
    assert data["exception_type"].endswith(exception_type)
    assert data["exception_message"].startswith("the hook failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("situation", SITUATIONS)
@pytest.mark.parametrize(
    "hook, register",
    [
        # The user cancels after the model ran: the pack's hook is the first to see it.
        ("checks_for_cancel", "register_forward_hook"),
        # The user cancels before the model runs: ComfyUI's model raises under the pack's wrapper.
        ("set_unet_wrapper", "register_forward_pre_hook"),
    ],
)
async def test_a_cancel_during_sampling_through_a_packs_hook_ends_only_that_prompt_as_interrupted(
    prompt_with, unet, situation, hook, register
):
    forwards = []

    def cancel_at_the_second_step(module, args, output=None):
        forwards.append(True)
        if len(forwards) == 2:
            interrupt_current_processing()

    getattr(unet.model.diffusion_model, register)(cancel_at_the_second_step)
    executor = await prompt_with(hook, situation)

    event, data = outcome(executor)
    assert (event, data["node_id"]) == ("execution_interrupted", "sample")
    assert len(forwards) == 2
    assert outcome(await prompt_with(hook, situation))[0] == "execution_success"


@pytest.mark.asyncio
@pytest.mark.parametrize("situation", SITUATIONS)
async def test_a_cancel_while_a_packs_hook_computes_without_calling_kills_the_process_after_the_grace_period(
    own_process, run_prompt, monkeypatch, situation
):
    monkeypatch.setattr(_pack_host, "GRACE", 1.0)
    first = own_process.process

    started_at = time.monotonic()
    executor, server = await run_prompt(sampling_prompt("spins", situation), ["keep"])
    took = time.monotonic() - started_at
    cancel_cleared = not model_management.processing_interrupted()
    after, _ = await run_prompt(sampling_prompt("set_cfg", situation), ["keep"])

    event, data = outcome(executor)
    assert (event, data["node_id"]) == ("execution_interrupted", "sample")
    assert "cancel" in [event for event, _, _ in server.sent]
    assert took >= 1.0 and first.poll() is not None and cancel_cleared
    assert outcome(after)[0] == "execution_success"
    assert own_process.process.pid != first.pid


@pytest.mark.asyncio
@pytest.mark.parametrize("situation", SITUATIONS)
async def test_a_pack_that_dies_in_a_hook_during_sampling_fails_the_node_that_sampled_with_its_exit_code(
    own_process, run_prompt, situation
):
    executor, _ = await run_prompt(sampling_prompt("dies", situation), ["keep"])
    after, _ = await run_prompt(sampling_prompt("set_cfg", situation), ["keep"])

    event, data = outcome(executor)
    assert (event, data["node_id"], data["exception_type"]) == (
        "execution_error",
        "sample",
        "comfy_api.v4._channel.PackStopped",
    )
    assert data["exception_message"].strip() == (
        f"pack {MODULE}.py stopped (exit code 5) while running node sample"
    )
    assert outcome(after)[0] == "execution_success"


@pytest.mark.asyncio
# A dying model moves its patches in `__del__`; that must not reach the pack after the release.
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
async def test_the_objects_comfyui_gets_each_time_it_moves_a_packs_patch_are_released_in_the_pack(
    own_process, run_prompt
):
    async def moved():
        return (await own_process.run(("node", "Moved", "execute"), ({}, {}))).args[0]

    executors = [
        (await run_prompt(sampling_prompt("moving", LATER), ["keep"]))[0]
        for _ in range(3)
    ]
    held = await moved()
    del executors
    gc.collect()

    assert held >= 1
    assert await moved() == 0
