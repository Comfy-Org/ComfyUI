"""The public surface, and what engine-API nodes add to io and ui: model-folder combos, declared combo outputs, savers that write through eng.output.write."""
import asyncio
import comfy_api.v4 as api
import execution
import json
import nodes
import os
import pytest
import subprocess
import sys
import torch
from PIL import Image
from comfy_api.latest import _engine, _runtime, _ui
from comfy_api.v4 import io, ui


PUBLIC = ["ComfyExtension", "Input", "InputImpl", "Types", "eng", "io", "ui", "util"]


def test_the_public_surface_is_exactly_this():
    assert sorted(api.__all__) == sorted(PUBLIC)
    assert sorted(n for n in vars(api) if not n.startswith("_")) == sorted(PUBLIC)


def test_the_one_stub_declares_the_same_names():
    import ast
    import os

    stub = os.path.join(os.path.dirname(api.__file__), "v4.pyi")
    tree = ast.parse(open(stub).read())
    declared = next(ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "__all__")
    assert sorted(declared) == sorted(api.__all__)


def test_importing_the_api_loads_no_engine_modules():
    code = ("import sys, comfy_api.v4; "
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
            "('comfy', 'nodes', 'server', 'folder_paths', 'execution', 'comfy_extras')); print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


class LoraPick(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="TestLoraPick", inputs=[
            io.Combo.Input("lora_name", options=io.ModelFolder("loras")),
            io.Combo.Input("only_st", options=io.ModelFolder("loras", extensions=["safetensors"]))],
            outputs=[io.String.Output()])

    @classmethod
    async def execute(cls, lora_name, only_st):
        return io.NodeOutput(lora_name)


@pytest.fixture
def loras(tmp_path, monkeypatch):
    import folder_paths

    (tmp_path / "a.safetensors").write_bytes(b"x")
    (tmp_path / "b.pt").write_bytes(b"x")
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "loras", ([str(tmp_path)], {".safetensors", ".pt"}))
    folder_paths.filename_list_cache.clear()
    return tmp_path


def _options(name):
    kind, spec = LoraPick.GET_NODE_INFO_V1()["input"]["required"][name][:2]
    assert kind == "COMBO"
    return spec["options"]


def test_options_follow_the_folder(loras):
    import folder_paths

    assert _options("lora_name") == ["a.safetensors", "b.pt"]
    assert _options("only_st") == ["a.safetensors"]
    (loras / "c.safetensors").write_bytes(b"x")
    folder_paths.filename_list_cache.clear()
    assert _options("lora_name") == ["a.safetensors", "b.pt", "c.safetensors"]


def test_validation_accepts_a_file_added_later(loras):
    import execution
    import folder_paths
    import nodes

    nodes.NODE_CLASS_MAPPINGS["TestLoraPick"] = LoraPick
    try:
        (loras / "new.safetensors").write_bytes(b"x")
        folder_paths.filename_list_cache.clear()
        prompt = {"1": {"class_type": "TestLoraPick", "inputs": {"lora_name": "new.safetensors", "only_st": "a.safetensors"}}}
        valid = asyncio.run(execution.validate_inputs("p", prompt, "1", {}))
        assert valid[0] is True, valid
        prompt["1"]["inputs"]["lora_name"] = "missing.safetensors"
        assert asyncio.run(execution.validate_inputs("p", prompt, "1", {}))[0] is False
    finally:
        del nodes.NODE_CLASS_MAPPINGS["TestLoraPick"]


def test_no_folders_means_no_options(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_folder_paths(name, *args, **kwargs):
        if name == "folder_paths":
            raise ImportError("not here")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_folder_paths)
    assert io.ModelFolder("loras").resolve() == []


def selector(options):
    class Selector(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id='ComboSelectorFixture',
                outputs=[io.Combo.Output('value', options=options)],
            )

        @classmethod
        def execute(cls):
            return io.NodeOutput((options or ['normal'])[0])

    return Selector


@pytest.mark.parametrize('target,accepted', [
    (['normal', 'karras', 'ays'], True),
    (['unrelated'], False),
    ('COMBO', True),
])
def test_declared_combo_output_validates_against_legacy_input(monkeypatch, target, accepted):
    source = selector(['normal', 'karras'])

    class LegacyReceiver:
        RETURN_TYPES = ()
        OUTPUT_NODE = True
        FUNCTION = 'run'

        @classmethod
        def INPUT_TYPES(cls):
            return {'required': {'value': (target,)}}

        def run(self, value):
            return {'ui': {'text': [value]}}

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, 'ComboSelectorFixture', source)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, 'LegacyComboReceiverFixture', LegacyReceiver)
    prompt = {
        '1': {'class_type': 'ComboSelectorFixture', 'inputs': {}},
        '2': {'class_type': 'LegacyComboReceiverFixture', 'inputs': {'value': ['1', 0]}},
    }
    valid, _, outputs, errors = asyncio.run(execution.validate_prompt('combo-fixture', prompt, None))
    assert valid is accepted
    if accepted:
        assert outputs == ['2']
        assert errors == {}
    else:
        assert any(error['type'] == 'return_type_mismatch' for error in errors['2']['errors'])


@pytest.mark.parametrize('options,wire_type', [
    (['normal', 'karras'], ['normal', 'karras']),
    ([], 'COMBO'),
    (None, 'COMBO'),
])
def test_combo_projection_keeps_execution_catalog_and_other_outputs_consistent(options, wire_type):
    class MixedOutputs(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id='MixedComboFixture',
                outputs=[
                    io.Combo.Output('choice', options=options, tooltip='Select a mode', is_output_list=True),
                    io.String.Output('text'),
                    io.MatchType.Output(io.MatchType.Template('T', io.String)),
                ],
            )

        @classmethod
        def execute(cls):
            return io.NodeOutput(['normal'], 'text', 'matched')

    info = MixedOutputs.GET_NODE_INFO_V1()
    assert info['output'] == [wire_type, 'STRING', io.MatchType.io_type]
    assert MixedOutputs.RETURN_TYPES == info['output']
    assert info['output_name'] == ['choice', 'text', 'MATCHTYPE']
    assert info['output_is_list'] == [True, False, False]
    assert info['output_tooltips'] == ['Select a mode', None, None]
    assert info['output_matchtypes'] == [None, None, 'T']


@pytest.fixture
def folders(tmp_path, monkeypatch):
    import folder_paths

    for name in ("input", "output", "temp"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(folder_paths, "output_directory", str(tmp_path / "output"))
    monkeypatch.setattr(folder_paths, "temp_directory", str(tmp_path / "temp"))
    return tmp_path


class _Hidden:
    prompt = {"1": {"class_type": "X"}}
    extra_pnginfo = {"workflow": {"nodes": []}}


class _Node:
    hidden = _Hidden()


def _images(n=2):
    return torch.linspace(0, 1, n * 4 * 6 * 3).reshape(n, 4, 6, 3)


def test_saved_images_keep_comfyui_names_and_metadata(folders):
    saved = ui.ImageSaveHelper.save_images(_images(2), "run/x_%batch_num%", _ui.FolderType.output, _Node, 4)
    assert [(r.filename, r.subfolder, r["type"]) for r in saved] == [
        ("x_0_00001_.png", "run", "output"), ("x_1_00002_.png", "run", "output")]
    with Image.open(folders / "output" / "run" / "x_1_00002_.png") as img:
        assert img.size == (6, 4)
        assert json.loads(img.text["prompt"]) == _Hidden.prompt


def test_batch_counter_continues_across_saves(folders):
    ui.ImageSaveHelper.save_images(_images(2), "x", _ui.FolderType.output, None)
    again = ui.ImageSaveHelper.save_images(_images(1), "x", _ui.FolderType.output, None)
    assert again[0].filename == "x_00003_.png"


def test_preview_image_goes_to_temp(folders):
    preview = ui.PreviewImage(_images(1), cls=None)
    (entry,) = preview.as_dict()["images"]
    assert entry["type"] == "temp" and entry["filename"].startswith("ComfyUI_temp_")
    assert os.path.isfile(folders / "temp" / entry["filename"])


def test_audio_keeps_its_naming(folders):
    audio = {"waveform": torch.zeros(1, 1, 800), "sample_rate": 8000}
    (entry,) = ui.AudioSaveHelper.save_audio(audio, "a", _ui.FolderType.output, None, format="flac")
    assert entry.filename == "a_00001.flac"


class _OutOfProcessOps:
    """A stand-in for a worker's or guest's ops: eng calls leave the process (here, to the
    in-process dispatcher, so the files really land)."""

    out_of_process = True

    def __init__(self):
        self.calls = []

    async def apply(self, op, subject, params):
        self.calls.append(params["name"])
        return await _engine.dispatch(subject, **params)


def test_out_of_process_writes_wait_for_the_runner(folders):
    ops = _OutOfProcessOps()

    async def node():
        with _runtime.bind_runtime(_runtime.InProcessRefResolver(), None, ops), \
                _ui.collect_writes(disable_metadata=True) as pending:
            preview = ui.PreviewImage(_images(2), cls=_Node)
            before = [dict(e) for e in preview.as_dict()["images"]]
            assert ops.calls == [] and before[0]["filename"] == ""
            await pending.flush()
            return preview.as_dict()["images"]

    entries = asyncio.run(node())
    assert ops.calls == ["output.write"]  # one call per batch, so the counters run on
    names = [e["filename"] for e in entries]
    assert names[0].endswith("_00001_.png") and names[1].endswith("_00002_.png")
    with Image.open(folders / "temp" / names[0]) as img:
        assert "prompt" not in img.text  # the runner's disable_metadata applies


def test_out_of_process_without_a_runner_fails_clearly(folders):
    async def node():
        with _runtime.bind_runtime(_runtime.InProcessRefResolver(), None, _OutOfProcessOps()):
            ui.PreviewImage(_images(1))

    with pytest.raises(RuntimeError, match="doesn't collect"):
        asyncio.run(node())


def test_preview_3d_background_is_a_temp_file(folders):
    preview = ui.PreviewUI3D("mesh.glb", {"camera": 1}, bg_image=_images(1))
    path = preview.as_dict()["result"][2]
    assert path.startswith("temp/bg_") and os.path.isfile(folders / path)
