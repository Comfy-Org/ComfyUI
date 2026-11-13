"""Which folders a pack that runs in its own process can make ComfyUI list as model folders."""

import asyncio
import copy

import pytest

import folder_paths
from comfy_api.v4 import _pack_host, _pack_processes

from conftest import evaluate, needs_pack_process, start_probe

pytestmark = needs_pack_process


def process_of(pack) -> _pack_host.PackProcess:
    return _pack_host.PackProcess(str(pack), "pack")


@pytest.fixture
def model_folders(monkeypatch, tmp_path):
    """ComfyUI's table of model folders, restored after the test; a pack folder and a folder
    outside it that only ComfyUI may read."""
    monkeypatch.setattr(
        folder_paths,
        "folder_names_and_paths",
        copy.deepcopy(folder_paths.folder_names_and_paths),
    )
    folder_paths.filename_list_cache.clear()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "decoy.safetensors").write_bytes(b"")
    pack = tmp_path / "custom_nodes" / "pack"
    (pack / "models").mkdir(parents=True)
    yield pack, outside
    folder_paths.filename_list_cache.clear()


@pytest.mark.parametrize("link", ["models/out", "other/out"])
def test_a_folder_of_a_pack_with_a_link_out_of_it_cannot_be_a_model_folder(
    model_folders, link
):
    pack, outside = model_folders
    (pack / link).parent.mkdir(exist_ok=True)
    (pack / link).symlink_to(outside)
    if link.startswith("other"):
        (pack / "models" / "other").symlink_to(pack / "other")

    with pytest.raises(ValueError, match="cannot be a model folder"):
        _pack_host._folders(
            process_of(pack),
            "add_model_folder_path",
            ("v4_linked", str(pack / "models")),
            {},
        )

    assert "v4_linked" not in folder_paths.folder_names_and_paths


def test_a_folder_of_a_pack_whose_links_stay_in_it_is_a_model_folder(model_folders):
    pack, _ = model_folders
    (pack / "weights").mkdir()
    (pack / "weights" / "small.safetensors").write_bytes(b"")
    (pack / "models" / "weights").symlink_to(pack / "weights")

    _pack_host._folders(
        process_of(pack),
        "add_model_folder_path",
        ("v4_linked", str(pack / "models")),
        {},
    )

    assert folder_paths.get_filename_list("v4_linked") == ["weights/small.safetensors"]


def test_the_directories_a_pack_process_added_to_model_folders_go_when_it_is_taken_away(
    model_folders, tmp_path, monkeypatch
):
    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path / "models"))
    table = folder_paths.folder_names_and_paths
    loras = list(table["loras"][0])
    added = str(tmp_path / "models" / "more")
    process = start_probe(tmp_path)
    try:
        asyncio.run(
            evaluate(
                process,
                f"folders.add_model_folder_path('v4_new', {added!r}),"
                f"folders.add_model_folder_path('loras', {added!r})",
            )
        )
        during = (list(table["v4_new"][0]), added in table["loras"][0])
    finally:
        _pack_processes.unload(process.module_path)

    assert during == ([added], True)
    assert "v4_new" not in table
    assert table["loras"][0] == loras
