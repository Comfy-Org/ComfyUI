"""``ui`` savers write through ``eng.output.write``: at once in ComfyUI's process, and through the
node's runner (``_ui.collect_writes``) outside it. File names stay ComfyUI's own."""
import asyncio
import json
import os

import pytest
import torch
from PIL import Image

from comfy_api.index import ui
from comfy_api._impl import _engine, _runtime, _ui


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
