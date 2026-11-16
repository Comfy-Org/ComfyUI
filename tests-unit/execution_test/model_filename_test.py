import os

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import execution
import folder_paths


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
