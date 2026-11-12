import json
import os
import pickle

import pytest
import torch
from safetensors.torch import save_file

import comfy.ops
import comfy.taesd.taesd
import comfy.utils
import folder_paths
import nodes
from comfy.ldm.flux.model import Flux
from comfy_api import v4
from comfy_api.internal.singleton import SingletonMetaclass
from comfy_api.v4._models import Models


@pytest.fixture(autouse=True)
def isolate_api_instances(monkeypatch):
    instances = dict(SingletonMetaclass._instances)
    instances.pop(v4.ComfyAPI, None)
    instances.pop(Models, None)
    monkeypatch.setattr(SingletonMetaclass, "_instances", instances)


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    registered = dict(folder_paths.folder_names_and_paths)
    for folder in (
        "checkpoints",
        "diffusion_models",
        "vae",
        "vae_approx",
        "loras",
        "embeddings",
    ):
        (tmp_path / folder).mkdir()
        registered[folder] = (
            [str(tmp_path / folder)],
            folder_paths.supported_pt_extensions,
        )
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", registered)
    monkeypatch.setattr(folder_paths, "filename_list_cache", {})
    return tmp_path


@pytest.fixture(scope="module")
def diffusion_weights():
    torch.manual_seed(0)
    model = Flux(
        in_channels=16,
        out_channels=16,
        vec_in_dim=8,
        context_in_dim=8,
        hidden_size=128,
        mlp_ratio=4.0,
        num_heads=1,
        depth=1,
        depth_single_blocks=1,
        axes_dim=[16, 56, 56],
        theta=10000,
        patch_size=2,
        qkv_bias=True,
        guidance_embed=False,
        txt_ids_dims=[],
        dtype=torch.float32,
        device=torch.device("cpu"),
        operations=comfy.ops.disable_weight_init,
    )
    return {
        key: torch.randn_like(value) * 0.02 for key, value in model.state_dict().items()
    }


@pytest.fixture(scope="module")
def vae_weights():
    torch.manual_seed(1)
    taesd = comfy.taesd.taesd.TAESD(latent_channels=4)
    return {
        key: torch.randn_like(value) * 0.1 for key, value in taesd.state_dict().items()
    }


@pytest.fixture
def checkpoint(models_dir, diffusion_weights, vae_weights):
    save_file(
        {
            f"model.diffusion_model.{key}": value
            for key, value in diffusion_weights.items()
        }
        | {f"vae.{key}": value for key, value in vae_weights.items()},
        models_dir / "checkpoints" / "tiny.safetensors",
    )
    return "tiny.safetensors"


@pytest.fixture
def pixels():
    return torch.linspace(0, 1, 16 * 16 * 3).reshape(1, 16, 16, 3)


def assert_same_weights(loaded, expected):
    assert loaded.keys() == expected.keys()
    for key, value in expected.items():
        torch.testing.assert_close(loaded[key], value, rtol=0, atol=0)


def unet_weights(model):
    return model.model.diffusion_model.state_dict()


@pytest.mark.asyncio
async def test_names_are_the_ones_native_loader_nodes_offer(models_dir):
    folder = models_dir / "loras" / "style"
    folder.mkdir()
    save_file({"weight": torch.ones(1)}, folder / "a.safetensors")
    save_file({"weight": torch.ones(1)}, models_dir / "loras" / "b.safetensors")
    (folder / "notes.txt").write_text("not a model")

    names = await v4.eng.models.list("loras")

    assert len(names) == 2
    assert names == nodes.LoraLoader.INPUT_TYPES()["required"]["lora_name"][0]
    assert v4.ComfyAPISync().models.list("loras") == names


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_read_gives_the_tensors_the_native_lora_loader_reads(
    models_dir, synchronous
):
    tensors = {
        "a": torch.arange(6.0).reshape(2, 3),
        "b": torch.ones(4, dtype=torch.int8),
    }
    save_file(tensors, models_dir / "loras" / "new.safetensors", {"author": "fixture"})
    torch.save(tensors, models_dir / "loras" / "old.pt")
    api = v4.ComfyAPISync().models if synchronous else v4.eng.models

    for name, metadata in (("new.safetensors", {"author": "fixture"}), ("old.pt", {})):
        result = (
            api.read("loras", name) if synchronous else await api.read("loras", name)
        )
        native = comfy.utils.load_torch_file(
            folder_paths.get_full_path_or_raise("loras", name), safe_load=True
        )
        assert result["metadata"] == metadata
        assert_same_weights(result["sd"], native)
        assert_same_weights(result["sd"], tensors)


@pytest.mark.asyncio
async def test_read_refuses_a_pickle_that_would_run_code(models_dir):
    marker = models_dir / "ran"

    class Payload:
        def __reduce__(self):
            return os.mkdir, (str(marker),)

    torch.save({"weight": Payload()}, models_dir / "loras" / "payload.pt")

    with pytest.raises(pickle.UnpicklingError):
        await v4.eng.models.read("loras", "payload.pt")
    assert not marker.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("weight_dtype", "options"),
    [
        ("default", None),
        ("fp8_e4m3fn", {"dtype": "float8_e4m3fn"}),
        ("fp8_e4m3fn_fast", {"dtype": "float8_e4m3fn", "fp8_optimizations": True}),
        ("fp8_e5m2", {"dtype": "float8_e5m2"}),
    ],
)
async def test_diffusion_model_loads_as_unet_loader_loads_it(
    models_dir, diffusion_weights, weight_dtype, options
):
    save_file(diffusion_weights, models_dir / "diffusion_models" / "tiny.safetensors")

    model = await v4.eng.models.load_diffusion_model("tiny.safetensors", options)
    (native,) = nodes.UNETLoader().load_unet("tiny.safetensors", weight_dtype)

    assert (
        model.model.model_config.optimizations
        == native.model.model_config.optimizations
    )
    assert model.model_dtype() == native.model_dtype()
    assert_same_weights(unet_weights(model), unet_weights(native))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dtype", ["float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"]
)
async def test_diffusion_model_weights_are_stored_in_the_dtype_named(
    models_dir, diffusion_weights, dtype
):
    save_file(diffusion_weights, models_dir / "diffusion_models" / "tiny.safetensors")

    model = await v4.eng.models.load_diffusion_model(
        "tiny.safetensors", {"dtype": dtype}
    )

    assert model.model_dtype() == getattr(torch, dtype)


@pytest.mark.asyncio
async def test_vae_file_metadata_reaches_comfyuis_loader(models_dir):
    save_file(
        {
            "vocoder.resblocks.0.convs1.0.weight": torch.empty(0),
            "audio_vae.test": torch.empty(0),
        },
        models_dir / "vae" / "audio.safetensors",
        {"config": "malformed JSON"},
    )

    with pytest.raises(json.JSONDecodeError):
        await v4.eng.models.load_vae("audio.safetensors")


@pytest.mark.asyncio
async def test_checkpoint_loads_as_checkpoint_loader_simple_loads_it(
    checkpoint, pixels
):
    model, clip, vae = await v4.eng.models.load_checkpoint(checkpoint)
    native_model, native_clip, native_vae = (
        nodes.CheckpointLoaderSimple().load_checkpoint(checkpoint)
    )
    fp8, _, _ = await v4.eng.models.load_checkpoint(
        checkpoint, {"dtype": "float8_e4m3fn"}
    )

    assert clip is native_clip is None
    assert type(model.model.model_config) is type(native_model.model.model_config)
    assert_same_weights(unet_weights(model), unet_weights(native_model))
    torch.testing.assert_close(
        vae.decode(vae.encode(pixels)), native_vae.decode(native_vae.encode(pixels))
    )
    assert fp8.model_dtype() == torch.float8_e4m3fn


@pytest.mark.asyncio
async def test_checkpoint_without_vae_weights_gives_a_vae_that_raises(
    models_dir, diffusion_weights, pixels
):
    save_file(
        {
            f"model.diffusion_model.{key}": value
            for key, value in diffusion_weights.items()
        },
        models_dir / "checkpoints" / "unet-only.safetensors",
    )

    _, clip, vae = await v4.eng.models.load_checkpoint("unet-only.safetensors")

    assert clip is None
    with pytest.raises(RuntimeError, match="VAE is invalid"):
        vae.encode(pixels)


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_vae_loads_as_vae_loader_loads_it(
    models_dir, vae_weights, pixels, synchronous
):
    save_file(vae_weights, models_dir / "vae" / "tiny.safetensors")
    api = v4.ComfyAPISync().models if synchronous else v4.eng.models

    vae = (
        api.load_vae("tiny.safetensors")
        if synchronous
        else await api.load_vae("tiny.safetensors")
    )
    (native,) = nodes.VAELoader().load_vae("tiny.safetensors")

    assert (vae.device, vae.vae_dtype) == (native.device, native.vae_dtype)
    torch.testing.assert_close(
        vae.decode(vae.encode(pixels)), native.decode(native.encode(pixels))
    )
    copy = vae.patcher.deepclone_multigpu(new_load_device=vae.device)
    assert copy.model is not vae.patcher.model


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize("missing", ["encoder", "decoder"])
async def test_a_tiny_vae_without_one_of_its_files_raises_file_not_found_naming_them(
    models_dir, vae_weights, synchronous, missing
):
    api = v4.ComfyAPISync().models if synchronous else v4.eng.models

    async def load():
        return api.load_vae("taesd") if synchronous else await api.load_vae("taesd")

    def save(part):
        prefix = f"taesd_{part}."
        save_file(
            {
                key.removeprefix(prefix): value
                for key, value in vae_weights.items()
                if key.startswith(prefix)
            },
            models_dir / "vae_approx" / f"{prefix}safetensors",
        )

    save({"encoder": "decoder", "decoder": "encoder"}[missing])
    with pytest.raises(FileNotFoundError, match="'taesd_encoder.*' or 'taesd_decoder"):
        await load()

    save(missing)
    folder_paths.filename_list_cache.clear()
    assert (await load()).latent_channels == 4


@pytest.mark.asyncio
async def test_names_cannot_reach_outside_the_folder(models_dir, vae_weights):
    save_file(vae_weights, models_dir / "outside.safetensors")
    calls = [
        ("loras", lambda name: v4.eng.models.read("loras", name)),
        ("checkpoints", v4.eng.models.load_checkpoint),
        ("diffusion_models", v4.eng.models.load_diffusion_model),
        ("vae", v4.eng.models.load_vae),
    ]

    for folder, load in calls:
        save_file(vae_weights, models_dir / folder / "inside.safetensors")
        for name in (
            str(models_dir / "outside.safetensors"),
            "../outside.safetensors",
            str(models_dir / folder / "inside.safetensors"),
        ):
            with pytest.raises(FileNotFoundError):
                await load(name)


@pytest.mark.asyncio
async def test_unrecognized_weights_raise(models_dir):
    for folder in ("checkpoints", "diffusion_models", "vae"):
        save_file(
            {"unknown": torch.ones(1)}, models_dir / folder / "unknown.safetensors"
        )

    for load in (
        v4.eng.models.load_checkpoint,
        v4.eng.models.load_diffusion_model,
        v4.eng.models.load_vae,
    ):
        with pytest.raises(RuntimeError):
            await load("unknown.safetensors")


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_each_accepted_call_form_runs_in_a_node(
    run_node, models_dir, checkpoint, diffusion_weights, synchronous
):
    save_file(diffusion_weights, models_dir / "diffusion_models" / "tiny.safetensors")
    save_file(
        {
            "diffusion_model.img_in.diff": torch.ones_like(
                diffusion_weights["img_in.weight"]
            )
        },
        models_dir / "loras" / "diff.safetensors",
    )

    executor, _ = await run_node(
        """\
names = {call}.models.list("loras")
model, clip, vae = {call}.models.load_checkpoint("tiny.safetensors")
unet = {call}.models.load_diffusion_model("tiny.safetensors", None)
fp8 = {call}.models.load_diffusion_model("tiny.safetensors", {{"dtype": "float8_e4m3fn"}})
lora = ({call}.models.read("loras", names[0]))["sd"]
patched, _ = {call}.model.load_lora(unet, None, lora, 0.5, 1.0)
before = {call}.model.read_weight(unet, "diffusion_model.img_in.weight")
after = {call}.model.read_weight(patched, "diffusion_model.img_in.weight")
assert names == ["diff.safetensors"] and clip is None and vae.latent_channels == 4
assert fp8.model_dtype() == torch.float8_e4m3fn
torch.testing.assert_close(after, before + 0.5)
""",
        synchronous,
    )

    assert executor.success, executor.status_messages
