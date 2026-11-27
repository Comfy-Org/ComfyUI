from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import folder_paths
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP
from comfy_extras.nodes_lora_stack import LoadLoraModel, LoadLoraTextEncoder
from comfy_extras.nodes_lt import GetICLoRAParameters
from nodes import LoraLoader


@pytest.fixture
def lora_dir(tmp_path, monkeypatch):
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "loras", ([str(tmp_path)], {".safetensors"}))
    return tmp_path


def save_lora(path, layer, metadata):
    tensors = {}
    for key in (f"lora_unet_{layer}", f"text_encoders.{layer}"):
        tensors[f"{key}.lora_down.weight"] = torch.ones(1, 2)
        tensors[f"{key}.lora_up.weight"] = torch.ones(2, 1)
    save_file(tensors, path, metadata=metadata)


@pytest.fixture
def model_and_clip():
    device = torch.device("cpu")
    model = torch.nn.Module()
    model.diffusion_model = torch.nn.ModuleDict({
        name: torch.nn.Linear(2, 2) for name in "abc"
    })
    model.model_config = SimpleNamespace(unet_config={})

    clip = CLIP(no_init=True)
    clip.cond_stage_model = torch.nn.ModuleDict({
        name: torch.nn.Linear(2, 2) for name in "abc"
    })
    clip.patcher = ModelPatcher(clip.cond_stage_model, device, device)
    clip.tokenizer = None
    clip.tokenizer_metadata = {}
    clip.layer_idx = None
    clip.tokenizer_options = {}
    clip.use_clip_schedule = False
    clip.apply_hooks_to_conds = None
    return ModelPatcher(model, device, device), clip


@pytest.mark.parametrize("stack", [False, True], ids=["sequential", "stack"])
@pytest.mark.parametrize(
    "later_metadata,expected_factor",
    [
        pytest.param([{"title": "ordinary"}], 2, id="unrelated"),
        pytest.param([None], 2, id="no-metadata"),
        pytest.param([{"ss_reference_downscale_factor": "4"}, {"title": "ordinary"}], 4, id="conflict"),
        pytest.param([{"reference_downscale_factor": "inf"}], 2, id="invalid"),
    ],
)
def test_iclora_metadata_survives_multiple_loads(lora_dir, model_and_clip, stack, later_metadata, expected_factor):
    settings = [{"reference_downscale_factor": "2"}, *later_metadata]
    layers = "abc"[:len(settings)]
    for layer, metadata in zip(layers, settings):
        save_lora(lora_dir / f"{layer}.safetensors", layer, metadata)

    model, clip = model_and_clip
    first_model, first_clip = LoraLoader().load_lora(model, clip, "a.safetensors", 1.0, 1.0)
    if stack:
        rows = [{"lora_name": f"{layer}.safetensors", "strength": 1.0, "enabled": True} for layer in layers[1:]]
        (result_model,) = LoadLoraModel.execute(first_model, rows).result
        (result_clip,) = LoadLoraTextEncoder.execute(first_clip, rows).result
    else:
        result_model, result_clip = first_model, first_clip
        for layer in layers[1:]:
            result_model, result_clip = LoraLoader().load_lora(result_model, result_clip, f"{layer}.safetensors", 1.0, 1.0)

    assert set(result_model.patches) == {f"diffusion_model.{layer}.weight" for layer in layers}
    assert set(result_clip.patcher.patches) == {f"{layer}.weight" for layer in layers}
    assert GetICLoRAParameters.execute(first_model).result == ({"reference_downscale_factor": 2},)
    assert GetICLoRAParameters.execute(result_model).result == ({"reference_downscale_factor": expected_factor},)

    with_metadata = [metadata for metadata in settings if metadata]
    for original, result in ((first_model, result_model), (first_clip.patcher, result_clip.patcher)):
        assert original.get_attachment("lora_metadata") == settings[0]
        assert original.get_attachment("lora_metadata_history") == (settings[0],)
        assert result.get_attachment("lora_metadata") == with_metadata[-1]
        assert result.get_attachment("lora_metadata_history") == tuple(reversed(with_metadata))


@pytest.mark.parametrize(
    "layer,strength_model,strength_clip",
    [("missing", 1.0, 1.0), ("a", 0.0, 1.0), ("a", 1.0, 0.0)],
    ids=["unmatched", "model-disabled", "clip-disabled"],
)
def test_iclora_history_only_uses_applied_patches(lora_dir, model_and_clip, layer, strength_model, strength_clip):
    factor_metadata = {"reference_downscale_factor": "2"}
    ordinary_metadata = {"title": "ordinary"}
    save_lora(lora_dir / "candidate.safetensors", layer, factor_metadata)
    save_lora(lora_dir / "ordinary.safetensors", "b", ordinary_metadata)

    model, clip = model_and_clip
    loader = LoraLoader()
    first_model, first_clip = loader.load_lora(model, clip, "candidate.safetensors", strength_model, strength_clip)
    final_model, final_clip = loader.load_lora(first_model, first_clip, "ordinary.safetensors", 1.0, 1.0)

    model_applied = layer == "a" and strength_model != 0
    clip_applied = layer == "a" and strength_clip != 0
    assert first_model.get_attachment("lora_metadata") == factor_metadata
    assert first_clip.patcher.get_attachment("lora_metadata") == factor_metadata
    for patcher in (first_model, final_model):
        assert GetICLoRAParameters.execute(patcher).result == ({"reference_downscale_factor": 2 if model_applied else 1},)
    assert final_clip.patcher.get_attachment("lora_metadata_history") == (
        (ordinary_metadata, factor_metadata) if clip_applied else (ordinary_metadata,)
    )


def test_iclora_parameters_accept_legacy_attachment(model_and_clip):
    model, _ = model_and_clip
    assert GetICLoRAParameters.execute(model).result == ({"reference_downscale_factor": 1},)
    model.set_attachments("lora_metadata", {"reference_downscale_factor": "2"})
    assert GetICLoRAParameters.execute(model).result == ({"reference_downscale_factor": 2},)
