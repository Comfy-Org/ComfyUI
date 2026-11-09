"""Rows implemented by ComfyUI's own engine extensions (comfy_api/latest/engine/), in every mode."""
import asyncio
import os

import pytest
import torch

from comfy_api.index import eng, io
from comfy_api._impl import _engine

import engine_loopback as loopback


@pytest.fixture(params=sorted(loopback.MODES))
def run(request):
    mode = loopback.MODES[request.param]

    def go(node, **inputs):
        return asyncio.run(mode(node, "p", inputs))[0]
    return go


@pytest.fixture
def folders(tmp_path, monkeypatch):
    import folder_paths

    for name in ("input", "output", "temp"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(folder_paths, "input_directory", str(tmp_path / "input"))
    monkeypatch.setattr(folder_paths, "output_directory", str(tmp_path / "output"))
    monkeypatch.setattr(folder_paths, "temp_directory", str(tmp_path / "temp"))
    return tmp_path


class Write:
    @classmethod
    async def execute(cls, data, folder):
        entry = await eng.output.write(data, "kj/test", "bin", folder=folder)
        return io.NodeOutput(entry, await eng.files.read(f"{entry['subfolder']}/{entry['filename']} [{folder}]"))


def test_output_write_then_files_read(run, folders):
    first = run(Write, data=b"\x00abc", folder="output").result
    second = run(Write, data=b"def", folder="output").result
    assert first[0] == {"filename": "test_00001_.bin", "subfolder": "kj", "type": "output"}
    assert second[0]["filename"] == "test_00002_.bin"
    assert first[1] == b"\x00abc" and (folders / "output/kj/test_00001_.bin").read_bytes() == b"\x00abc"
    temp = run(Write, data=b"t", folder="temp").result
    assert temp[0]["type"] == "temp" and (folders / "temp/kj/test_00001_.bin").exists()


class Bad:
    @classmethod
    async def execute(cls, call):
        if call == "escape-prefix":
            await eng.output.write(b"x", "../../outside", "bin")
        elif call == "escape-ref":
            await eng.files.read("../../etc/passwd [input]")
        elif call == "bad-folder":
            await eng.output.write(b"x", "a", "bin", folder="input")
        elif call == "bad-suffix":
            await eng.output.write(b"x", "a", "../png")
        return io.NodeOutput(None)


@pytest.mark.parametrize("call", ["escape-prefix", "escape-ref", "bad-folder", "bad-suffix"])
def test_files_stay_in_comfys_folders(run, folders, call):
    with pytest.raises(Exception):
        run(Bad, call=call)
    assert not any(p for p in folders.parent.glob("outside*"))


class Listing:
    @classmethod
    async def execute(cls):
        return io.NodeOutput(await eng.files.list("input", recursive=True, extensions=["png"]),
                             await eng.files.list("input"), await eng.files.exists("a.png"),
                             await eng.files.exists("nope.png"), await eng.files.exists("../x [input]"))


def test_files_list_and_exists(run, folders):
    (folders / "input/a.png").write_bytes(b"1")
    (folders / "input/b.txt").write_bytes(b"22")
    (folders / "input/sub").mkdir()
    (folders / "input/sub/c.png").write_bytes(b"333")
    pngs, top, yes, no, outside = run(Listing).result
    assert [e["ref"] for e in pngs] == ["a.png [input]", "sub/c.png [input]"]
    assert [e["name"] for e in top] == ["a.png", "b.txt"] and top[1]["size"] == 2
    assert (yes, no, outside) == (True, False, False)


class Device:
    @classmethod
    async def execute(cls):
        info = await eng.device.info()
        return io.NodeOutput(info, await eng.memory.free_bytes())


def test_device_info_and_memory(run):
    import comfy.model_management as mm

    info, free = run(Device).result
    assert info["torch_device"] == str(mm.get_torch_device())
    assert isinstance(getattr(torch, info["intermediate_dtype"]), torch.dtype)
    assert isinstance(free, int) and free >= 0


class Attention:
    @classmethod
    async def execute(cls, model, backend):
        return io.NodeOutput(await eng.model.set_attention(model, backend))


class _Patcher:
    def __init__(self):
        self.model_options = {"transformer_options": {}}
        self.load_device = "cpu"

    def clone(self):
        other = _Patcher()
        other.model_options = {"transformer_options": dict(self.model_options["transformer_options"])}
        return other

    def set_model_optimized_attention(self, fn):
        import comfy.model_patcher
        comfy.model_patcher.ModelPatcher.set_model_optimized_attention(self, fn)


def test_set_attention_uses_comfys_registry(run):
    from comfy.ldm.modules.attention import attention_pytorch

    source = _Patcher()
    patched = run(Attention, model=source, backend="pytorch").result[0]
    override = patched.model_options["transformer_options"]["optimized_attention_override"]
    q = torch.randn(1, 4, 8)
    assert torch.equal(override(None, q, q, q, 2), attention_pytorch(q, q, q, 2))
    assert "optimized_attention_override" not in source.model_options["transformer_options"]
    with pytest.raises(Exception, match="unknown attention backend"):
        run(Attention, model=_Patcher(), backend="no-such-kernel")


class Compile:
    @classmethod
    async def execute(cls, model):
        return io.NodeOutput(await eng.model.torch_compile(model, backend="cudagraphs", keys=["diffusion_model"]))


class _Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = torch.nn.Linear(2, 2)


def _real_patcher():
    import comfy.model_patcher
    return comfy.model_patcher.ModelPatcher(_Base(), torch.device("cpu"), torch.device("cpu"))


def test_torch_compile_sets_comfys_wrapper(run):
    source = _real_patcher()
    patched = run(Compile, model=source).result[0]
    from comfy_api._impl.torch_helpers.torch_compile import COMPILE_KEY
    from comfy.patcher_extension import WrappersMP

    assert patched.get_wrappers(WrappersMP.APPLY_MODEL, COMPILE_KEY)
    assert not source.get_wrappers(WrappersMP.APPLY_MODEL, COMPILE_KEY)


class Progress:
    @classmethod
    async def execute(cls, image):
        await eng.progress.update(1, 2, preview=image)
        await eng.ui.progress_text("half way")
        return io.NodeOutput(None)


def test_progress_preview_and_text(run, monkeypatch):
    import types
    import comfy.utils
    import server

    seen, texts = [], []
    monkeypatch.setattr(comfy.utils, "PROGRESS_BAR_HOOK", lambda *a: seen.append(a))
    monkeypatch.setattr(server.PromptServer, "instance", types.SimpleNamespace(
        client_id="c", send_progress_text=lambda text, node, sid=None: texts.append((text, node, sid))), raising=False)
    run(Progress, image=torch.full((1, 4, 6, 3), 0.5))
    (value, total, preview, _prompt, node), = seen
    assert (value, total, node) == (1, 2, "1") and preview[0] == "JPEG" and preview[1].size == (6, 4)
    assert texts == [("half way", "1", "c")]


class ReadModel:
    @classmethod
    async def execute(cls, name):
        return io.NodeOutput(await eng.models.read("loras", name))


def test_models_read(run, tmp_path, monkeypatch):
    import folder_paths
    import safetensors.torch

    safetensors.torch.save_file({"w": torch.arange(4.0)}, str(tmp_path / "tiny.safetensors"), metadata={"k": "v"})
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "loras", ([str(tmp_path)], {".safetensors"}))
    folder_paths.filename_list_cache.clear()
    out = run(ReadModel, name="tiny.safetensors").result[0]
    assert torch.equal(out["sd"]["w"], torch.arange(4.0)) and out["metadata"] == {"k": "v"}


def test_every_core_extension_row_is_on_eng():
    rows = [n for n, r in _engine.CALLS.items() if r.target.startswith("engine:")]
    assert len(rows) >= 15
    for name in rows:
        group, _, method = name.partition(".")
        assert asyncio.iscoroutinefunction(getattr(getattr(eng, group), method)), name
    files = [f for f in os.listdir(_engine.CORE_EXTENSIONS_DIR) if f.endswith(".py")]  # common/: common extensions
    assert sorted(files) == sorted(
        {r.target[len("engine:"):].partition(".")[0] + ".py" for r in _engine.CALLS.values() if r.target.startswith("engine:")})


class Misc:
    @classmethod
    async def execute(cls):
        await eng.ui.send("kj.event", {"a": [1, 2]})
        return io.NodeOutput(await eng.settings.get("disable_metadata"), await eng.settings.get("preview_size"))


def test_ui_send_and_settings(run, monkeypatch):
    import types
    import server
    from comfy.cli_args import args

    sent = []
    monkeypatch.setattr(server.PromptServer, "instance", types.SimpleNamespace(
        client_id="c", send_sync=lambda e, d, sid=None: sent.append((e, d, sid))), raising=False)
    monkeypatch.setattr(args, "disable_metadata", True)
    disabled, size = run(Misc).result
    assert sent == [("kj.event", {"a": [1, 2]}, "c")]
    assert disabled is True and size == args.preview_size


class BadSetting:
    @classmethod
    async def execute(cls):
        return io.NodeOutput(await eng.settings.get("api_key"))


def test_settings_are_a_whitelist(run):
    with pytest.raises(Exception, match="unknown setting"):
        run(BadSetting)


class _SamplingBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import comfy.latent_formats
        import comfy.model_sampling

        class Sampling(comfy.model_sampling.ModelSamplingDiscrete, comfy.model_sampling.EPS):
            pass
        self.diffusion_model = torch.nn.Linear(2, 2)
        self.model_sampling = Sampling()
        self.latent_format = comfy.latent_formats.SD15()


def _sampling_patcher():
    import comfy.model_patcher
    return comfy.model_patcher.ModelPatcher(_SamplingBase(), torch.device("cpu"), torch.device("cpu"))


class Sampling:
    @classmethod
    async def execute(cls, model, latent):
        sigma = await eng.model.percent_to_sigma(model, 0.5)
        t = await eng.model.timestep(model, torch.tensor([sigma]))
        back = await eng.model.sigma(model, t)
        info = await eng.model.info(model)
        lat_in = await eng.model.latent_process_in(model, latent)
        lat_out = await eng.model.latent_process_out(model, lat_in)
        return io.NodeOutput(sigma, t, back, info["sigma_min"], info["sigma_max"], lat_out)


def test_model_sampling_facts_match_the_engine(run):
    source = _sampling_patcher()
    ms = source.get_model_object("model_sampling")
    latent = torch.randn(1, 4, 8, 8)
    sigma, t, back, smin, smax, lat_out = run(Sampling, model=source, latent=latent).result
    assert sigma == float(ms.percent_to_sigma(0.5))
    assert torch.equal(t, ms.timestep(torch.tensor([sigma]))) and torch.equal(back, ms.sigma(t))
    assert (smin, smax) == (float(ms.sigma_min), float(ms.sigma_max))
    assert torch.allclose(lat_out, latent, atol=1e-6)


class Options:
    @classmethod
    async def execute(cls, model):
        patched = await eng.model.set_option(model, ["transformer_options", "kj_strength"], 0.75)
        return io.NodeOutput(patched, await eng.model.options(patched), await eng.model.options(model))


def test_set_option_copies_and_options_reads_back(run):
    source = _sampling_patcher()
    patched, after, before = run(Options, model=source).result
    assert patched.model_options["transformer_options"]["kj_strength"] == 0.75
    assert "kj_strength" not in source.model_options["transformer_options"]
    assert after["transformer_options"]["kj_strength"] == 0.75 and "kj_strength" not in before["transformer_options"]


class _Vae:
    latent_channels = 4

    def encode(self, pixels):
        return pixels.movedim(-1, 1)[:, :1] * 2

    def decode(self, samples):
        return samples

    def spacial_compression_decode(self):
        return 8

    spacial_compression_encode = spacial_compression_decode

    def temporal_compression_decode(self):
        return None


class Encode:
    @classmethod
    async def execute(cls, vae, image):
        return io.NodeOutput(await eng.vae.encode(vae, image))


def test_vae_encode(run):
    image = torch.rand(1, 4, 4, 3)
    assert torch.equal(run(Encode, vae=_Vae(), image=image).result[0], image.movedim(-1, 1)[:, :1] * 2)
