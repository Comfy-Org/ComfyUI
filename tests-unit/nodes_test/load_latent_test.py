import ast
import hashlib
import os
from pathlib import Path

import pytest

import folder_paths


@pytest.fixture
def load_latent(tmp_path, monkeypatch):
    # Load the filesystem-only node interface without importing the inference stack.
    nodes_path = Path(__file__).resolve().parents[2] / "nodes.py"
    module = ast.parse(nodes_path.read_text(encoding="utf-8"), filename=str(nodes_path))
    node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "LoadLatent")
    namespace = {"folder_paths": folder_paths, "hashlib": hashlib, "os": os}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(nodes_path), "exec"), namespace)  # noqa: S102
    monkeypatch.setattr(folder_paths, "input_directory", str(tmp_path))
    return namespace["LoadLatent"]


def test_latent_choices_include_subfolders(load_latent, tmp_path):
    for name in ("root.latent", "Latent/saved.latent", "Latent/nested/saved.latent", "other/saved.latent", "Latent/image.png"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"latent")
    (tmp_path / "directory.latent").mkdir()

    choices = load_latent.INPUT_TYPES()["required"]["latent"][0]

    assert choices == sorted([
        "root.latent",
        os.path.join("Latent", "saved.latent"),
        os.path.join("Latent", "nested", "saved.latent"),
        os.path.join("other", "saved.latent"),
    ])


def test_discovered_latent_path_validates_and_tracks_changes(load_latent, tmp_path):
    path = tmp_path / "Latent" / "saved.latent"
    path.parent.mkdir()
    path.write_bytes(b"first")
    [choice] = load_latent.INPUT_TYPES()["required"]["latent"][0]

    assert folder_paths.get_annotated_filepath(choice) == str(path)
    assert load_latent.VALIDATE_INPUTS(choice) is True
    assert load_latent.IS_CHANGED(choice) == hashlib.sha256(b"first").hexdigest()
    path.write_bytes(b"second")
    assert load_latent.IS_CHANGED(choice) == hashlib.sha256(b"second").hexdigest()
    path.unlink()
    assert load_latent.VALIDATE_INPUTS(choice) == f"Invalid latent file: {choice}"


def test_empty_input_directory_has_no_latent_choices(load_latent):
    assert load_latent.INPUT_TYPES()["required"]["latent"][0] == []
