"""The VAE, CLIP, sampler, noise and guider calls, made by a pack in its own process.

Each test makes the same calls, with the arguments the in-process tests of these groups use, from
ComfyUI's process and from a pack process (async and sync forms), and compares the results.

In-process tests these twins do not repeat: those that check identity (`is`, `data_ptr`) of a value
that crosses, which a copy cannot keep; those that inject a group instance or a suspending body,
which concern ComfyUI's own process; and those whose sampler, noise or hooks are functions of the
test, which the callback tests cover as pack code (`pack_process_callbacks_test.py`).
"""

import asyncio

import pytest
import torch
from clip_test import clip as clip_fixture
from sampler_test import NEGATIVE, POSITIVE, run_sampler
from sampler_test import latent as latent_fixture
from sampler_test import model as model_fixture
from vae_test import pixels as pixels_fixture
from vae_test import vae as vae_fixture

from comfy.sd import VAE
from comfy_api import v4
from comfy_api.v4 import _pack_host, _pack_processes
from conftest import needs_pack_process

pytestmark = needs_pack_process

clip = clip_fixture
latent = latent_fixture
model = model_fixture
pixels = pixels_fixture
vae = vae_fixture

CALLER_PACK = """\
from comfy_api.v4 import CacheProvider, ComfyAPISync, ComfyExtension, eng, io


class Provider(CacheProvider):
    async def on_lookup(self, context):
        return None

    async def on_store(self, context, value):
        pass


class Call(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Call", outputs=[io.AnyType.Output()])

    @classmethod
    async def execute(cls, group, name, args, kwargs, synchronous):
        if args == ("the pack's provider",):
            args = (Provider(),)
        if synchronous:
            return io.NodeOutput(getattr(getattr(ComfyAPISync(), group), name)(*args, **kwargs))
        return io.NodeOutput(await getattr(getattr(eng, group), name)(*args, **kwargs))


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [Call]


async def comfy_entrypoint():
    return Extension()
"""


@pytest.fixture(scope="module")
def caller(tmp_path_factory):
    """A pack process whose node makes the call it is given."""
    path = tmp_path_factory.mktemp("caller") / "v4_caller_pack.py"
    path.write_text(CALLER_PACK, encoding="utf-8")
    asyncio.run(_pack_host.load(str(path), "v4_caller_pack"))
    yield _pack_processes.processes[str(path)]
    _pack_processes.unload(str(path))


@pytest.fixture(params=["async", "sync"])
def call(request, caller):
    """Make `eng.<group>.<name>(*args, **kwargs)` in the pack process, awaited or sync."""

    async def made_in_the_pack(group, name, *args, **kwargs):
        inputs = {
            "group": group,
            "name": name,
            "args": args,
            "kwargs": kwargs,
            "synchronous": request.param == "sync",
        }
        return (await caller.run(("node", "Call", "execute"), (inputs, {}))).args[0]

    return made_in_the_pack


async def in_process(group, name, *args, **kwargs):
    return await getattr(getattr(v4.eng, group), name)(*args, **kwargs)


def assert_same(actual, expected):
    """Equal values of the same types, tensors bit for bit; ComfyUI's objects by type."""
    assert type(actual) is type(expected)
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected) and actual.dtype == expected.dtype
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for item, expected_item in zip(actual, expected):
            assert_same(item, expected_item)
    elif type(expected).__module__ == "builtins":
        assert actual == expected


async def assert_calls_match(call, group, cases):
    for name, args, kwargs in cases:
        assert_same(
            await call(group, name, *args, **kwargs),
            await in_process(group, name, *args, **kwargs),
        )


@pytest.mark.asyncio
async def test_each_vae_call_from_a_pack_process_gives_what_it_gives_in_process(
    call, vae, pixels
):
    samples = await in_process("vae", "encode", vae, pixels)
    tiles = {"tile_x": 8, "tile_y": 12, "overlap": 2}

    with torch.inference_mode():
        await assert_calls_match(
            call,
            "vae",
            [
                ("encode", (vae, pixels), {}),
                ("decode", (vae, samples), {}),
                ("encode_tiled", (vae, pixels), tiles),
                (
                    "decode_tiled",
                    (vae, samples),
                    {**tiles, "tile_t": 3, "overlap_t": 1},
                ),
                ("info", (vae,), {}),
            ],
        )
    weights = {"pixel_space_vae": torch.tensor(1.0)}
    restored = await call("vae", "from_state_dict", weights)
    expected = await in_process("vae", "from_state_dict", weights)
    assert_same(
        await in_process("vae", "info", restored),
        await in_process("vae", "info", expected),
    )
    with pytest.raises(RuntimeError, match="VAE is invalid"):
        await call("vae", "encode", VAE(sd={}), torch.empty(1, 3, 8, 8))


@pytest.mark.asyncio
async def test_each_clip_call_from_a_pack_process_gives_what_it_gives_in_process(
    call, clip
):
    key = "clip_l.transformer.text_model.final_layer_norm.bias"
    tokens = await in_process("clip", "tokenize", clip, "a (red:1.5) cat")
    metadata = {"strength": 0.75, "start_percent": 0.2}

    await assert_calls_match(
        call,
        "clip",
        [
            ("tokenize", (clip, "a (red:1.5) cat"), {"return_word_ids": True}),
            ("encode_from_tokens", (clip, tokens), {}),
            ("encode_from_tokens", (clip, tokens), {"return_pooled": "unprojected"}),
            (
                "encode_from_tokens",
                (clip, tokens),
                {"return_pooled": True, "return_dict": True},
            ),
            (
                "encode_tokens",
                (clip, tokens),
                {"unprojected": True, "add_dict": metadata, "show_pbar": False},
            ),
            ("info", (clip,), {}),
            ("weight_keys", (clip,), {}),
            ("read_weight", (clip, key), {}),
        ],
    )
    patched = await call(
        "clip", "add_patches", clip, {key: [torch.ones(16)]}, 0.5, 0.25
    )
    expected = await in_process(
        "clip", "add_patches", clip, {key: [torch.ones(16)]}, 0.5, 0.25
    )
    assert_same(
        await in_process("clip", "read_weight", patched, key),
        await in_process("clip", "read_weight", expected, key),
    )
    cloned = await call("clip", "clone", clip)
    assert cloned is not clip and cloned.cond_stage_model is clip.cond_stage_model
    with pytest.raises(AttributeError):
        await call("clip", "read_weight", clip, "missing.weight")


@pytest.mark.asyncio
async def test_each_sampler_noise_and_guider_call_from_a_pack_process_gives_what_it_gives_in_process(
    call, model, latent
):
    arguments = (model, 5, 4, 2.0, "dpmpp_2m", "karras", POSITIVE, NEGATIVE, latent)

    await assert_calls_match(
        call,
        "sampler",
        [
            ("sigmas", (model, "karras", 4, 0.75), {}),
            ("sample", arguments, {}),
            (
                "sample",
                arguments,
                {"start_step": 1, "last_step": 3, "disable_noise": True},
            ),
        ],
    )
    named = await call(
        "sampler", "named", "euler_ancestral", {"eta": 0.5, "s_noise": 0.75}
    )
    expected = await in_process(
        "sampler", "named", "euler_ancestral", {"eta": 0.5, "s_noise": 0.75}
    )
    assert torch.equal(run_sampler(model, named), run_sampler(model, expected))
    with pytest.raises(AttributeError):
        await call("sampler", "named", "missing")

    async def sample_custom(make, noise, guider):
        parts = (
            await make("noise", *noise),
            await make("guider", *guider),
            await make("sampler", "named", "dpmpp_2m"),
            await make("sampler", "sigmas", model, "karras", 4, 0.75),
        )
        return await make("sampler", "sample_custom", *parts, latent)

    for noise, guider in (
        (("random", 9), ("cfg", model, POSITIVE, NEGATIVE, 2.5)),
        (("empty",), ("basic", model, POSITIVE)),
    ):
        assert_same(
            await sample_custom(call, noise, guider),
            await sample_custom(in_process, noise, guider),
        )
    assert_same(
        await call("noise", "generate", await call("noise", "random", 4), latent),
        await in_process(
            "noise", "generate", await in_process("noise", "random", 4), latent
        ),
    )


@pytest.mark.asyncio
async def test_a_pack_process_cannot_register_a_cache_provider(call):
    from comfy_execution.cache_provider import _get_cache_providers

    with pytest.raises(TypeError, match="registered in ComfyUI's process"):
        await call("caching", "register_provider", "the pack's provider")

    assert _get_cache_providers() == ()
