import os

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import execution
import folder_paths
import nodes


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [False, True])
async def test_checkpoint_name_from_other_platform(tmp_path, monkeypatch, missing):
    model = tmp_path / "family" / "model.safetensors"
    model.parent.mkdir()
    model.touch()
    monkeypatch.setitem(
        folder_paths.folder_names_and_paths,
        "checkpoints",
        ([str(tmp_path)], {".safetensors"}),
    )
    monkeypatch.setattr(folder_paths, "filename_list_cache", {})

    foreign_sep = "/" if os.sep == "\\" else "\\"
    filename = "missing.safetensors" if missing else model.name
    prompt = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": f"family{foreign_sep}{filename}"},
        },
    }

    valid, errors, _ = await execution.validate_inputs("test", prompt, "1", {})
    if missing:
        assert not valid
        assert errors[0]["type"] == "value_not_in_list"
    else:
        assert valid, errors
        selected = prompt["1"]["inputs"]["ckpt_name"]
        assert selected == os.path.join("family", model.name)
        assert folder_paths.get_full_path("checkpoints", selected) == str(model)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "loader,extra_inputs",
    [
        ("LoraLoader", {"clip": ["source", 1], "strength_clip": 1.0}),
        ("LoraLoaderModelOnly", {}),
    ],
)
async def test_lora_name_from_other_platform(tmp_path, monkeypatch, loader, extra_inputs):
    class ModelSource:
        RETURN_TYPES = ("MODEL", "CLIP")

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

    model = tmp_path / "family" / "model.safetensors"
    model.parent.mkdir()
    model.touch()
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "loras", ([str(tmp_path)], {".safetensors"}))
    monkeypatch.setattr(folder_paths, "filename_list_cache", {})
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "ModelSource", ModelSource)

    foreign_sep = "/" if os.sep == "\\" else "\\"
    inputs = {"model": ["source", 0], "lora_name": f"family{foreign_sep}{model.name}", "strength_model": 1.0, **extra_inputs}
    prompt = {
        "source": {"class_type": "ModelSource", "inputs": {}},
        "load": {"class_type": loader, "inputs": inputs},
    }

    valid, errors, _ = await execution.validate_inputs("test", prompt, "load", {})
    assert valid, errors
    assert inputs["lora_name"] == os.path.join("family", model.name)
    assert folder_paths.get_full_path("loras", inputs["lora_name"]) == str(model)


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="Windows does not allow a literal backslash in a filename")
async def test_exact_checkpoint_name_takes_priority(tmp_path, monkeypatch):
    exact = tmp_path / r"family\model.safetensors"
    translated = tmp_path / "family" / "model.safetensors"
    exact.touch()
    translated.parent.mkdir()
    translated.touch()
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "checkpoints", ([str(tmp_path)], {".safetensors"}))
    monkeypatch.setattr(folder_paths, "filename_list_cache", {})

    inputs = {"ckpt_name": exact.name}
    prompt = {"load": {"class_type": "CheckpointLoaderSimple", "inputs": inputs}}

    valid, errors, _ = await execution.validate_inputs("test", prompt, "load", {})
    assert valid, errors
    assert inputs["ckpt_name"] == exact.name
    assert folder_paths.get_full_path("checkpoints", inputs["ckpt_name"]) == str(exact)
