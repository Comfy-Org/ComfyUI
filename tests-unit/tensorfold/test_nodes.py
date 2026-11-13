import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def node_module(comfy_environment, monkeypatch, tmp_path):
    import comfy_extras.nodes_tensorfold as module

    monkeypatch.setattr(module.folder_paths, "models_dir", str(tmp_path / "models"))
    return module


def test_v3_schemas_load_without_installed_engine(node_module, monkeypatch):
    monkeypatch.setattr(node_module.folder_paths, "get_filename_list", lambda name: ["weights.safetensors", "unsafe.ckpt"])
    nodes = asyncio.run(node_module.TensorFoldExtension().get_node_list())
    assert [node.define_schema().node_id for node in nodes] == ["TensorFoldMOSSImage"]
    assert nodes[0].RETURN_TYPES == ["LATENT", "STRING"]
    for node in nodes:
        schema = node.GET_SCHEMA()
        schema.validate()
        assert node.INPUT_TYPES()["required"]["checkpoint_name"][1]["options"] == ["weights.safetensors"]
        assert schema.is_experimental


def test_builtin_registration_is_present():
    path = Path(__file__).resolve().parents[2] / "nodes.py"
    assert path.read_text().count('"nodes_tensorfold.py"') == 1


def test_missing_runtime_is_actionable_before_any_model_unload(node_module, monkeypatch):
    def unexpected():
        raise AssertionError("models should not be unloaded")

    monkeypatch.setattr(node_module.comfy.model_management, "unload_all_models", unexpected)
    with pytest.raises(RuntimeError, match="setup CLI"):
        node_module.TensorFoldMOSSImage.execute("missing.safetensors", [], [], 1024, 1024, 50, 0)


def test_moss_node_routes_selected_device_and_reports_runtime(node_module, monkeypatch, tmp_path):
    events = []
    checkpoint = tmp_path / "moss.safetensors"
    checkpoint.touch()
    executable = tmp_path / "tf-moss-image"
    runtime = SimpleNamespace(artifact=executable, engine_version="1.2.0", source_revision="a" * 40,
                              source_patch_sha256="c" * 64, digest="b" * 64)
    monkeypatch.setattr(node_module, "load_runtime", lambda path: runtime)
    monkeypatch.setattr(node_module, "subprocess_environment", lambda device: {"CUDA_VISIBLE_DEVICES": "GPU-selected"})
    monkeypatch.setattr(node_module.folder_paths, "get_full_path_or_raise", lambda kind, name: str(checkpoint))
    monkeypatch.setattr(node_module.folder_paths, "get_temp_directory", lambda: str(tmp_path))
    management = node_module.comfy.model_management
    monkeypatch.setattr(management, "get_torch_device", lambda: torch.device("cuda:1"))
    monkeypatch.setattr(management, "throw_exception_if_processing_interrupted", lambda: events.append("interrupt"))
    monkeypatch.setattr(management, "unload_all_models", lambda: events.append("unload"))
    monkeypatch.setattr(management, "soft_empty_cache", lambda: events.append("empty_cache"))
    progress = []
    monkeypatch.setattr(node_module.comfy.utils, "ProgressBar", lambda total: SimpleNamespace(update_absolute=lambda n, t: progress.append((n, t))))
    positive, negative = object(), object()
    latent = {"samples": torch.zeros(1, 128, 2, 2)}

    def sample(*args, environment):
        assert args[:8] == (checkpoint, executable, positive, negative, 32, 32, 50, 7)
        assert args[8] == tmp_path
        assert environment == {"CUDA_VISIBLE_DEVICES": "GPU-selected"}
        assert events == ["interrupt", "unload", "empty_cache"]
        args[10](50, 50)
        return latent, {"effective_steps": 50}

    monkeypatch.setattr(node_module, "sample_moss", sample)
    output = node_module.TensorFoldMOSSImage.execute(checkpoint.name, positive, negative, 32, 32, 50, 7)
    assert output[0] is latent
    assert json.loads(output[1]) == {"effective_steps": 50, "engine_version": "1.2.0", "source_revision": "a" * 40,
                                   "source_patch_sha256": "c" * 64, "artifact_sha256": "b" * 64}
    assert progress == [(50, 50)]


def test_runtime_and_checkpoint_changes_invalidate_cached_nodes(node_module, monkeypatch, tmp_path):
    checkpoint = tmp_path / "moss.safetensors"
    checkpoint.write_bytes(b"first")
    monkeypatch.setattr(node_module.folder_paths, "get_full_path", lambda kind, name: str(checkpoint))
    before = node_module.fingerprint(checkpoint.name)
    checkpoint.write_bytes(b"replacement")
    after_checkpoint = node_module.fingerprint(checkpoint.name)
    assert before != after_checkpoint
    manifest = node_module.runtime_manifest()
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"updated":true}')
    assert node_module.fingerprint(checkpoint.name) != after_checkpoint
