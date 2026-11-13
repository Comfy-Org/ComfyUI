import inspect
import logging
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest
import safetensors.torch
import torch
from PIL import Image, ImageFile

import comfy.sample
import comfy.utils
import folder_paths
import node_helpers
import nodes
from comfy.samplers import KSampler
from comfy_api.v4 import _pack_processes, util
from conftest import wait_until

FOLDER = "v4_util_test"


@pytest.fixture
def comfy_folders(tmp_path, monkeypatch):
    """ComfyUI's folders moved into tmp_path, with a model folder FOLDER of two directories."""
    for kind in ("output", "input", "temp", "user"):
        (tmp_path / kind).mkdir()
        monkeypatch.setattr(folder_paths, f"{kind}_directory", str(tmp_path / kind))
    first, second = tmp_path / "models" / "first", tmp_path / "models" / "second"
    (first / "sub").mkdir(parents=True)
    second.mkdir()
    for path in (
        first / "a.safetensors",
        first / "sub" / "b.safetensors",
        first / "c.txt",
    ):
        path.write_bytes(b"")
    (second / "a.safetensors").write_bytes(b"")
    (second / "d.safetensors").write_bytes(b"")
    table = {
        name: (list(paths), set(extensions))
        for name, (paths, extensions) in folder_paths.folder_names_and_paths.items()
    }
    table[FOLDER] = ([str(first), str(second)], {".safetensors"})
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", table)
    monkeypatch.setattr(folder_paths, "filename_list_cache", {})
    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path / "models"))
    return tmp_path


def test_directories_are_comfyuis_current_folders(comfy_folders):
    assert util.folders.get_output_directory() == util.folders.output_directory
    assert util.folders.output_directory == str(comfy_folders / "output")
    assert util.folders.input_directory == util.folders.get_input_directory()
    assert util.folders.get_input_directory() == str(comfy_folders / "input")
    assert util.folders.temp_directory == util.folders.get_temp_directory()
    assert util.folders.get_temp_directory() == str(comfy_folders / "temp")
    assert util.folders.get_user_directory() == str(comfy_folders / "user")
    assert util.folders.models_dir == str(comfy_folders / "models")
    assert util.folders.base_path == folder_paths.base_path
    assert [util.folders.get_directory_by_type(kind) for kind in ("temp", "user")] == [
        str(comfy_folders / "temp"),
        None,
    ]
    assert util.folders.get_system_user_directory("cache") == str(
        comfy_folders / "user" / "__cache"
    )
    with pytest.raises(ValueError):
        util.folders.get_system_user_directory("_cache")


def test_model_folder_calls_find_files_as_comfyuis_loaders_do(comfy_folders):
    first, second = (
        comfy_folders / "models" / "first",
        comfy_folders / "models" / "second",
    )

    listed = util.folders.get_filename_list(FOLDER)
    paths = util.folders.get_folder_paths(FOLDER)
    paths.append("elsewhere")

    assert listed == [
        "a.safetensors",
        "d.safetensors",
        os.path.join("sub", "b.safetensors"),
    ]
    assert util.folders.get_folder_paths(FOLDER) == [str(first), str(second)]
    assert util.folders.get_full_path(FOLDER, "a.safetensors") == str(
        first / "a.safetensors"
    )
    assert util.folders.get_full_path(FOLDER, "../first/c.txt") is None
    assert util.folders.get_full_path("no_such_folder", "a.safetensors") is None
    with pytest.raises(FileNotFoundError):
        util.folders.get_full_path_or_raise(FOLDER, "missing.safetensors")
    with pytest.raises(KeyError):
        util.folders.get_filename_list("no_such_folder")
    assert util.folders.get_folder_paths("unet") == util.folders.get_folder_paths(
        "diffusion_models"
    )
    files, directories, scanned_at = util.folders.get_filename_list_(FOLDER)
    assert files == listed
    assert set(directories) == {str(first), str(first / "sub"), str(second)}
    assert isinstance(scanned_at, float)


def test_folder_table_is_comfyuis_own_and_writes_reach_every_node(comfy_folders):
    extra = comfy_folders / "extra"
    extra.mkdir()
    (extra / "e.onnx").write_bytes(b"")
    (extra / "notes.md").write_bytes(b"")

    table = util.folders.folder_names_and_paths
    table["v4_registered"] = ([str(extra)], {".onnx"})
    util.folders.add_model_folder_path("v4_added", str(extra))
    util.folders.add_model_folder_path(FOLDER, str(extra), is_default=True)

    assert table is folder_paths.folder_names_and_paths
    assert folder_paths.get_filename_list("v4_registered") == ["e.onnx"]
    assert folder_paths.get_filename_list("v4_added") == ["e.onnx", "notes.md"]
    assert folder_paths.get_folder_paths(FOLDER)[0] == str(extra)


def test_supported_extensions_are_a_copy_a_pack_cannot_widen():
    extensions = util.folders.supported_pt_extensions
    extensions.add(".onnx")

    assert extensions - {".onnx"} == folder_paths.supported_pt_extensions
    assert ".onnx" not in util.folders.supported_pt_extensions


def test_clearing_the_cache_lists_files_comfyui_could_not_see_change(comfy_folders):
    assert "c.txt" not in util.folders.get_filename_list(FOLDER)
    util.folders.folder_names_and_paths[FOLDER][1].add(".txt")
    assert "c.txt" not in util.folders.get_filename_list(FOLDER)

    util.folders.clear_filename_list_cache(FOLDER)

    assert "c.txt" in util.folders.get_filename_list(FOLDER)


def test_clearing_the_cache_by_a_legacy_name_clears_its_folder(comfy_folders):
    table = util.folders.folder_names_and_paths
    table["diffusion_models"] = (
        [str(comfy_folders / "models" / "first")],
        {".safetensors"},
    )
    assert "c.txt" not in util.folders.get_filename_list("diffusion_models")
    table["diffusion_models"][1].add(".txt")

    util.folders.clear_filename_list_cache("unet")

    assert "c.txt" in util.folders.get_filename_list("diffusion_models")


def test_annotated_names_resolve_in_their_folder_and_never_outside(comfy_folders):
    (comfy_folders / "output" / "a.png").write_bytes(b"")

    assert util.folders.annotated_filepath("a.png [output]") == (
        "a.png",
        str(comfy_folders / "output"),
    )
    assert util.folders.annotated_filepath("a.png") == ("a.png", None)
    assert util.folders.get_annotated_filepath("a.png [output]") == str(
        comfy_folders / "output" / "a.png"
    )
    assert util.folders.get_annotated_filepath("a.png") == str(
        comfy_folders / "input" / "a.png"
    )
    assert util.folders.get_annotated_filepath("a.png", str(comfy_folders)) == str(
        comfy_folders / "a.png"
    )
    assert util.folders.exists_annotated_filepath("a.png [output]")
    assert not util.folders.exists_annotated_filepath("a.png")
    assert not util.folders.exists_annotated_filepath("../output/a.png")
    with pytest.raises(ValueError):
        util.folders.get_annotated_filepath("../output/a.png")


def test_save_path_numbers_the_next_file_after_those_already_saved(comfy_folders):
    output = comfy_folders / "output"
    (output / "renders").mkdir()
    (output / "renders" / "shot_00007_.png").write_bytes(b"")

    assert util.folders.get_save_image_path("renders/shot", str(output)) == (
        str(output / "renders"),
        "shot",
        8,
        "renders",
        "renders/shot",
    )
    folder, filename, counter, _, _ = util.folders.get_save_image_path(
        "new/%width%x%height%", str(output), 640, 480
    )
    assert (folder, filename, counter) == (str(output / "new"), "640x480", 1)
    assert util.folders.get_save_image_path("renders/other", str(output))[2] == 1
    assert (output / "new").is_dir()
    with pytest.raises(Exception, match="outside the output folder"):
        util.folders.get_save_image_path("../shot", str(output))


def test_file_filters_and_searches_inspect_names_and_folders(comfy_folders):
    first = comfy_folders / "models" / "first"
    (first / ".git").mkdir()
    (first / ".git" / "HEAD").write_bytes(b"")
    (comfy_folders / "input" / "clips" / "day").mkdir(parents=True)
    outside = comfy_folders / "outside"
    outside.mkdir()
    (first / "link").symlink_to(outside)

    files, directories = util.folders.recursive_search(
        str(first), excluded_dir_names=[".git"]
    )

    assert util.folders.filter_files_content_types(
        ["a.png", "b.mp4", "c.wav", "d.txt"], ["image", "audio"]
    ) == ["a.png", "c.wav"]
    assert util.folders.filter_files_extensions(
        ["b.PT", "a.safetensors", "c.txt"], {".pt", ".safetensors"}
    ) == ["a.safetensors", "b.PT"]
    assert util.folders.filter_files_extensions(["b", "a.txt"], set()) == ["a.txt", "b"]
    assert sorted(files) == [
        "a.safetensors",
        "c.txt",
        os.path.join("sub", "b.safetensors"),
    ]
    assert set(directories) == {str(first), str(first / "sub"), str(first / "link")}
    assert util.folders.recursive_search(str(first / "c.txt")) == ([], {})
    assert util.folders.is_within_directory(str(first), str(first / "sub" / "x"))
    assert util.folders.is_within_directory(str(first), str(first))
    assert not util.folders.is_within_directory(str(first), str(first / "link" / "x"))
    assert util.folders.get_input_subfolders() == ["clips", "clips/day"]


def test_names_comfyui_has_but_v4_does_not_declare_are_not_reachable():
    assert not hasattr(util.folders, "set_output_directory")
    assert not hasattr(util.folders, "filename_list_cache")
    assert not hasattr(util, "KSAMPLER_NAMES")


def test_constants_are_read_from_comfyui_when_used(monkeypatch):
    monkeypatch.setattr(KSampler, "SAMPLERS", [*KSampler.SAMPLERS, "v4_test_sampler"])

    assert util.MAX_RESOLUTION == nodes.MAX_RESOLUTION
    assert util.SAMPLER_NAMES is KSampler.SAMPLERS
    assert "v4_test_sampler" in util.SAMPLER_NAMES
    assert util.SCHEDULER_NAMES is KSampler.SCHEDULERS


def test_importing_util_and_using_folders_does_not_load_the_engine():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            dedent("""\
                import sys
                from comfy_api.v4 import util

                util.folders.get_output_directory()
                util.folders.models_dir
                for module in ("nodes", "comfy.samplers", "comfy.sample", "comfy.model_management"):
                    assert module not in sys.modules, module
            """),
        ],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr


def test_resizing_matches_comfyuis_interpolation_and_cropping():
    images = torch.rand(2, 3, 6, 8)
    video_latent = torch.rand(1, 4, 3, 6, 8)

    resized = util.common_upscale(images, 4, 3, "bilinear", "disabled")
    cropped = util.common_upscale(images, 3, 3, "nearest-exact", "center")

    assert torch.equal(
        resized, torch.nn.functional.interpolate(images, size=(3, 4), mode="bilinear")
    )
    assert torch.equal(
        cropped,
        torch.nn.functional.interpolate(
            images[..., 1:7], size=(3, 3), mode="nearest-exact"
        ),
    )
    assert util.common_upscale(video_latent, 4, 3, "bislerp", "center").shape == (
        1,
        4,
        3,
        3,
        4,
    )
    assert util.bislerp(images.half(), 4, 3).dtype == torch.float16
    quantized = util.lanczos(images, 4, 3) * 255
    assert torch.allclose(quantized, quantized.round(), atol=1e-4)
    assert torch.equal(util.lanczos(images + 1, 4, 3), torch.ones(2, 3, 3, 4))
    assert util.lanczos(images[:, :1], 4, 3).shape == (2, 3, 4)


def test_noise_is_the_ksampler_noise_for_the_seed_and_batch_positions():
    latent = torch.zeros(3, 4, 2, 2, dtype=torch.float16)
    expected = torch.randn(3, 4, 2, 2, generator=torch.manual_seed(7)).half()

    noise = util.prepare_noise(latent, 7)
    by_position = util.prepare_noise(latent, 7, [2, 0, 2])

    assert noise.dtype == torch.float16 and noise.device.type == "cpu"
    assert torch.equal(noise, expected)
    assert torch.equal(by_position, noise[[2, 0, 2]])


def test_noise_leaves_the_global_generator_seeded_with_the_seed():
    util.prepare_noise(torch.zeros(1, 4, 2, 2), 7)
    after = torch.rand(3)

    torch.manual_seed(7)
    torch.randn(1, 4, 2, 2)
    assert torch.equal(after, torch.rand(3))


def test_conditioning_values_are_set_on_a_copy_sharing_the_tensors(caplog):
    tensor = torch.ones(1, 2, 3)
    conditioning = [[tensor, {"strength": 1.0}]]

    replaced = util.conditioning_set_values(
        conditioning, {"strength": 0.5, "mask": "m"}
    )
    added = util.conditioning_set_values(conditioning, {"strength": 0.5}, append=True)
    ranged = util.conditioning_set_values_with_timestep_range(
        conditioning, {"strength": 2.0}, 0.25, 0.5
    )

    assert conditioning == [[tensor, {"strength": 1.0}]]
    assert replaced[0][0] is tensor
    assert replaced[0][1] == {"strength": 0.5, "mask": "m"}
    assert added[0][1] == {"strength": 1.5}
    assert [entry[1]["strength"] for entry in ranged] == [1.0, 2.0, 1.0]
    assert [(entry[1]["start_percent"], entry[1]["end_percent"]) for entry in ranged][
        1
    ] == (
        0.25,
        0.5,
    )
    with caplog.at_level(logging.WARNING):
        assert (
            util.conditioning_set_values_with_timestep_range(
                conditioning, {"strength": 2.0}, 0.5, 0.25
            )
            is conditioning
        )
    assert "start_percent" in caplog.text


def test_pillow_retries_a_truncated_image_and_restores_pillows_setting(tmp_path):
    path = tmp_path / "cut.png"
    Image.effect_noise((32, 32), 64).convert("RGB").save(path)
    path.write_bytes(path.read_bytes()[:-40])

    def load(name):
        return Image.open(name).convert("RGB")

    with pytest.raises(OSError):
        load(path)
    assert util.pillow(load, path).size == (32, 32)
    assert ImageFile.LOAD_TRUNCATED_IMAGES is False


def test_weight_files_round_trip_and_torch_files_are_unwrapped(tmp_path):
    weights = {"layer.weight": torch.arange(4.0)}
    util.save_torch_file(
        weights, str(tmp_path / "w.safetensors"), metadata={"by": "v4"}
    )
    util.save_torch_file(weights, str(tmp_path / "named.ckpt"))
    torch.save({"state_dict": weights}, tmp_path / "wrapped.pt")
    torch.save({"model": weights}, tmp_path / "single.pth")

    loaded, metadata = util.load_torch_file(
        str(tmp_path / "w.safetensors"), return_metadata=True
    )

    assert metadata == {"by": "v4"}
    assert torch.equal(loaded["layer.weight"], weights["layer.weight"])
    assert safetensors.torch.load_file(tmp_path / "named.ckpt").keys() == weights.keys()
    assert (
        util.load_torch_file(str(tmp_path / "wrapped.pt"), safe_load=True).keys()
        == weights.keys()
    )
    assert (
        util.load_torch_file(str(tmp_path / "single.pth"), return_metadata=True)[1]
        is None
    )
    assert util.load_torch_file(str(tmp_path / "single.pth")).keys() == weights.keys()


def test_tiled_scale_blends_tiles_into_the_whole_image_result():
    image = torch.rand(2, 3, 40, 70)
    advanced, inference = [], []

    class Bar:
        def update(self, steps):
            advanced.append(steps)

    def upscale(tile):
        inference.append(torch.is_inference_mode_enabled())
        return tile.repeat_interleave(2, -1).repeat_interleave(2, -2)

    tiled = util.tiled_scale(
        image, upscale, tile_x=32, tile_y=16, overlap=4, upscale_amount=2, pbar=Bar()
    )
    per_dimension = util.tiled_scale_multidim(
        image, upscale, tile=(16, 32), overlap=4, upscale_amount=2
    )

    restored = util.tiled_scale_multidim(
        tiled,
        lambda tile: tile[..., ::2, ::2],
        tile=(32, 64),
        overlap=8,
        upscale_amount=2,
        downscale=True,
        pbar=Bar(),
    )

    assert all(inference)
    assert torch.allclose(restored, image)
    assert torch.allclose(tiled, upscale(image))
    assert torch.equal(per_dimension, tiled)
    assert sum(advanced) == 4 * util.get_tiled_scale_steps(
        70, 40, tile_x=32, tile_y=16, overlap=4
    )


def same(ours, theirs):
    if isinstance(ours, torch.Tensor):
        return torch.equal(ours, theirs)
    if (
        isinstance(ours, dict)
        and ours
        and isinstance(next(iter(ours.values())), torch.Tensor)
    ):
        return ours.keys() == theirs.keys() and all(
            map(same, ours.values(), theirs.values())
        )
    if isinstance(ours, tuple):
        return len(ours) == len(theirs) and all(map(same, ours, theirs))
    return ours == theirs


def test_each_accepted_argument_form_gets_comfyuis_result(comfy_folders, tmp_path):
    """Each argument form packs write, against ComfyUI's own call."""
    output = str(comfy_folders / "output")
    models = str(comfy_folders / "models")
    (comfy_folders / "input" / "photo.png").write_bytes(b"")
    image, latent = torch.rand(1, 3, 6, 8), torch.zeros(2, 4, 2, 2)
    conditioning = [[image, {"a": 1}]]
    weights, pickled, png = (
        tmp_path / "w.safetensors",
        tmp_path / "w.pt",
        tmp_path / "p.png",
    )
    comfy.utils.save_torch_file({"w": torch.ones(2)}, str(weights), metadata={"k": "v"})
    torch.save({"w": torch.ones(2)}, pickled)
    Image.new("RGB", (2, 2)).save(png)
    cpu = torch.device("cpu")

    forms = [
        (util.folders.get_output_directory, folder_paths.get_output_directory),
        (util.folders.get_input_directory, folder_paths.get_input_directory),
        (util.folders.get_temp_directory, folder_paths.get_temp_directory),
        (util.folders.get_user_directory, folder_paths.get_user_directory),
        (util.folders.get_input_subfolders, folder_paths.get_input_subfolders),
        (
            lambda: util.folders.get_annotated_filepath("photo.png"),
            lambda: folder_paths.get_annotated_filepath("photo.png"),
        ),
        (
            lambda: util.folders.get_annotated_filepath(
                "photo.png", default_dir=output
            ),
            lambda: folder_paths.get_annotated_filepath("photo.png", output),
        ),
        (
            lambda: util.folders.get_annotated_filepath("photo.png", output),
            lambda: folder_paths.get_annotated_filepath("photo.png", output),
        ),
        (
            lambda: util.folders.exists_annotated_filepath("photo.png"),
            lambda: folder_paths.exists_annotated_filepath("photo.png"),
        ),
        (
            lambda: util.folders.annotated_filepath("photo.png [input]"),
            lambda: folder_paths.annotated_filepath("photo.png [input]"),
        ),
        (
            lambda: util.folders.get_filename_list(FOLDER),
            lambda: folder_paths.get_filename_list(FOLDER),
        ),
        (
            lambda: util.folders.get_filename_list_(FOLDER)[:2],
            lambda: folder_paths.get_filename_list_(FOLDER)[:2],
        ),
        (
            lambda: util.folders.get_full_path(FOLDER, "d.safetensors"),
            lambda: folder_paths.get_full_path(FOLDER, "d.safetensors"),
        ),
        (
            lambda: util.folders.get_full_path_or_raise(FOLDER, "d.safetensors"),
            lambda: folder_paths.get_full_path_or_raise(FOLDER, "d.safetensors"),
        ),
        (
            lambda: util.folders.get_folder_paths(FOLDER),
            lambda: folder_paths.get_folder_paths(FOLDER),
        ),
        (
            lambda: util.folders.get_save_image_path("ComfyUI", output, 8, 6),
            lambda: folder_paths.get_save_image_path("ComfyUI", output, 8, 6),
        ),
        (
            lambda: util.folders.get_save_image_path("ComfyUI", output),
            lambda: folder_paths.get_save_image_path("ComfyUI", output),
        ),
        (
            lambda: util.folders.filter_files_content_types(
                ["p.png", "a.txt"], ["image"]
            ),
            lambda: folder_paths.filter_files_content_types(
                ["p.png", "a.txt"], ["image"]
            ),
        ),
        (
            lambda: util.folders.filter_files_extensions(["a.pt", "b.txt"], [".pt"]),
            lambda: folder_paths.filter_files_extensions(["a.pt", "b.txt"], [".pt"]),
        ),
        (
            lambda: util.folders.get_directory_by_type("input"),
            lambda: folder_paths.get_directory_by_type("input"),
        ),
        (
            lambda: util.folders.get_system_user_directory("pack"),
            lambda: folder_paths.get_system_user_directory("pack"),
        ),
        (
            lambda: util.folders.recursive_search(models),
            lambda: folder_paths.recursive_search(models),
        ),
        (
            lambda: util.folders.recursive_search(models, excluded_dir_names=["sub"]),
            lambda: folder_paths.recursive_search(models, ["sub"]),
        ),
        (
            lambda: util.folders.is_within_directory(output, models),
            lambda: folder_paths.is_within_directory(output, models),
        ),
        (lambda: FOLDER in util.folders.folder_names_and_paths, lambda: True),
        (
            lambda: util.folders.folder_names_and_paths[FOLDER][0][0],
            lambda: folder_paths.get_folder_paths(FOLDER)[0],
        ),
        (
            lambda: util.folders.folder_names_and_paths.get("none", ([], set())),
            lambda: ([], set()),
        ),
        (
            lambda: list(util.folders.folder_names_and_paths.keys()),
            lambda: list(folder_paths.folder_names_and_paths),
        ),
        (
            lambda: os.path.join(util.folders.models_dir, "x"),
            lambda: os.path.join(models, "x"),
        ),
        (lambda: util.folders.base_path, lambda: folder_paths.base_path),
        (lambda: util.folders.output_directory, lambda: output),
        (
            lambda: util.folders.supported_pt_extensions,
            lambda: folder_paths.supported_pt_extensions,
        ),
        (
            lambda: util.common_upscale(image, 4, 3, "bilinear", "center"),
            lambda: comfy.utils.common_upscale(image, 4, 3, "bilinear", "center"),
        ),
        (
            lambda: util.common_upscale(image, 4, 3, "area", crop="disabled"),
            lambda: comfy.utils.common_upscale(image, 4, 3, "area", "disabled"),
        ),
        (
            lambda: util.common_upscale(
                image, 4, 3, crop="center", upscale_method="bicubic"
            ),
            lambda: comfy.utils.common_upscale(image, 4, 3, "bicubic", "center"),
        ),
        (
            lambda: util.load_torch_file(str(weights), safe_load=True),
            lambda: comfy.utils.load_torch_file(str(weights)),
        ),
        (
            lambda: util.load_torch_file(str(pickled)),
            lambda: comfy.utils.load_torch_file(str(pickled)),
        ),
        (
            lambda: util.load_torch_file(str(weights), device=cpu),
            lambda: comfy.utils.load_torch_file(str(weights), device=cpu),
        ),
        (
            lambda: util.load_torch_file(str(weights), device=cpu, safe_load=True),
            lambda: comfy.utils.load_torch_file(str(weights), device=cpu),
        ),
        (
            lambda: util.load_torch_file(
                str(weights), return_metadata=True, safe_load=True
            ),
            lambda: comfy.utils.load_torch_file(str(weights), return_metadata=True),
        ),
        (lambda: util.pillow(Image.open, png).size, lambda: Image.open(png).size),
        (
            lambda: util.conditioning_set_values(conditioning, {"a": 2}),
            lambda: node_helpers.conditioning_set_values(conditioning, {"a": 2}),
        ),
        (
            lambda: util.conditioning_set_values(conditioning, {"a": 2}, append=True),
            lambda: node_helpers.conditioning_set_values(conditioning, {"a": 2}, True),
        ),
        (lambda: util.lanczos(image, 4, 3), lambda: comfy.utils.lanczos(image, 4, 3)),
        (lambda: util.bislerp(image, 4, 3), lambda: comfy.utils.bislerp(image, 4, 3)),
        (
            lambda: util.prepare_noise(latent, 3, [1, 0]),
            lambda: comfy.sample.prepare_noise(latent, 3, [1, 0]),
        ),
        (
            lambda: util.prepare_noise(latent, 3),
            lambda: comfy.sample.prepare_noise(latent, 3),
        ),
        (
            lambda: util.get_tiled_scale_steps(8, 6, tile_x=4, tile_y=4, overlap=1),
            lambda: comfy.utils.get_tiled_scale_steps(8, 6, 4, 4, 1),
        ),
        (
            lambda: util.tiled_scale(
                image,
                torch.neg,
                tile_x=4,
                tile_y=4,
                overlap=1,
                upscale_amount=1,
                pbar=None,
            ),
            lambda: comfy.utils.tiled_scale(image, torch.neg, 4, 4, 1, 1),
        ),
        (lambda: util.MAX_RESOLUTION, lambda: nodes.MAX_RESOLUTION),
        (lambda: (util.SAMPLER_NAMES,), lambda: (KSampler.SAMPLERS,)),
        (lambda: (util.SCHEDULER_NAMES,), lambda: (KSampler.SCHEDULERS,)),
    ]
    for ours, theirs in forms:
        assert same(ours(), theirs()), inspect.getsource(ours).strip()

    util.folders.add_model_folder_path("v4_pack_models", models)
    util.folders.add_model_folder_path("v4_pack_models", output, is_default=False)
    util.folders.folder_names_and_paths["v4_pack_table"] = ([output], {".png"})
    util.save_torch_file(
        {"w": torch.ones(1)}, str(tmp_path / "m.safetensors"), metadata={"a": "b"}
    )
    assert folder_paths.get_folder_paths("v4_pack_models") == [models, output]
    assert folder_paths.get_folder_paths("v4_pack_table") == [output]
    assert comfy.utils.load_torch_file(
        str(tmp_path / "m.safetensors"), return_metadata=True
    )[1] == {"a": "b"}


WRAPPERS = [
    (util.common_upscale, comfy.utils.common_upscale),
    (util.bislerp, comfy.utils.bislerp),
    (util.lanczos, comfy.utils.lanczos),
    (util.tiled_scale, comfy.utils.tiled_scale),
    (util.tiled_scale_multidim, comfy.utils.tiled_scale_multidim),
    (util.get_tiled_scale_steps, comfy.utils.get_tiled_scale_steps),
    (util.prepare_noise, comfy.sample.prepare_noise),
    (util.conditioning_set_values, node_helpers.conditioning_set_values),
    (
        util.conditioning_set_values_with_timestep_range,
        node_helpers.conditioning_set_values_with_timestep_range,
    ),
    (util.pillow, node_helpers.pillow),
    (util.load_torch_file, comfy.utils.load_torch_file),
    (util.save_torch_file, comfy.utils.save_torch_file),
    *[
        (getattr(util.folders, name), getattr(folder_paths, name))
        for name in (
            "get_output_directory",
            "get_input_directory",
            "get_temp_directory",
            "get_user_directory",
            "get_system_user_directory",
            "get_directory_by_type",
            "get_input_subfolders",
            "get_folder_paths",
            "get_filename_list",
            "get_filename_list_",
            "get_full_path",
            "get_full_path_or_raise",
            "annotated_filepath",
            "get_annotated_filepath",
            "exists_annotated_filepath",
            "get_save_image_path",
            "add_model_folder_path",
            "filter_files_content_types",
            "filter_files_extensions",
            "recursive_search",
            "is_within_directory",
        )
    ],
]


@pytest.mark.parametrize(
    "wrapper, original", WRAPPERS, ids=[wrapper.__name__ for wrapper, _ in WRAPPERS]
)
def test_comfyui_still_accepts_every_argument_a_wrapper_passes(wrapper, original):
    # ComfyUI's name for a parameter V4 names differently.
    theirs = {util.pillow: {"function": "fn"}}.get(wrapper, {})
    passed = inspect.signature(wrapper).parameters

    inspect.signature(original).bind(
        **{theirs.get(name, name): None for name in passed}
    )


@pytest.mark.asyncio
async def test_a_node_saves_an_image_through_util_folders(run_node, comfy_folders):
    # A pack process has the folders it was started with: start it again with the moved ones.
    for process in _pack_processes.processes.values():
        process.kill()
        wait_until(lambda: not process.ready, "the pack process to stop")
    executor, _ = await run_node(
        "from comfy_api.v4 import util\n"
        "output = util.folders.get_output_directory()\n"
        "folder, name, counter, subfolder, prefix = util.folders.get_save_image_path('v4/shot', output, 8, 6)\n"
        "image = util.common_upscale(torch.rand(1, 3, 4, 4), 8, 6, 'bilinear', 'center')\n"
        "util.save_torch_file(dict(image=image), folder + '/' + name + '_' + str(counter) + '.safetensors')\n",
        synchronous=True,
    )

    assert executor.success, executor.status_messages
    saved = comfy_folders / "output" / "v4" / "shot_1.safetensors"
    assert util.load_torch_file(str(saved))["image"].shape == (1, 3, 6, 8)


def test_a_torch_file_holding_more_than_weights_is_refused_whatever_safe_load_says(
    tmp_path,
):
    import datetime
    import pickle

    path = tmp_path / "pickled.pt"
    torch.save({"w": torch.ones(1), "when": datetime.date(2020, 1, 1)}, path)

    for safe_load in (False, True):
        with pytest.raises(pickle.UnpicklingError):
            util.load_torch_file(str(path), safe_load=safe_load)


def test_tiled_scale_multidim_places_tiles_by_index_formulas():
    image = torch.arange(2 * 32 * 32, dtype=torch.float32).reshape(1, 2, 32, 32)

    def scaled(scale, formulas):
        return scale(
            image,
            lambda tile: tile.repeat_interleave(2, -1).repeat_interleave(2, -2),
            tile=(16, 16),
            overlap=0,
            upscale_amount=2,
            out_channels=2,
            index_formulas=formulas,
        )

    ours = scaled(util.tiled_scale_multidim, 1)

    torch.testing.assert_close(
        ours,
        scaled(comfy.utils.tiled_scale_multidim, 1),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert not torch.allclose(
        ours, scaled(util.tiled_scale_multidim, None), equal_nan=True
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: util.tiled_scale(torch.zeros(1, 3, 8, 8), torch.neg, tile_x=0),
        lambda: util.tiled_scale(torch.zeros(1, 3, 8, 8), torch.neg, tile_y=-4),
        lambda: util.tiled_scale(torch.zeros(1, 3, 8, 8), torch.neg, overlap=-1),
        lambda: util.tiled_scale_multidim(torch.zeros(1, 3, 8, 8), torch.neg, (4, 0)),
        lambda: util.tiled_scale_multidim(
            torch.zeros(1, 3, 8, 8), torch.neg, (4, 4), overlap=(1, -1)
        ),
        lambda: util.get_tiled_scale_steps(8, 8, tile_x=4, tile_y=0, overlap=1),
        lambda: util.get_tiled_scale_steps(8, 8, tile_x=4, tile_y=4, overlap=-1),
    ],
)
def test_a_tile_size_that_is_not_positive_or_a_negative_overlap_is_refused(call):
    with pytest.raises(ValueError, match="tile sizes must be positive"):
        call()


@pytest.mark.parametrize(
    "names", [(None, "/models/loras"), ("loras", None), ("loras", b"/models/loras")]
)
def test_a_model_folder_or_directory_that_is_not_a_string_is_refused_unchanged(names):
    before = {
        name: (list(directories), set(extensions))
        for name, (
            directories,
            extensions,
        ) in folder_paths.folder_names_and_paths.items()
    }

    with pytest.raises(TypeError, match="must be a string"):
        util.folders.add_model_folder_path(*names)

    assert {
        name: (list(directories), set(extensions))
        for name, (
            directories,
            extensions,
        ) in folder_paths.folder_names_and_paths.items()
    } == before


def test_a_model_directory_given_as_a_path_object_is_listed(tmp_path, monkeypatch):
    monkeypatch.setitem(
        folder_paths.folder_names_and_paths, "v4_path_test", ([], set())
    )
    (tmp_path / "weights.safetensors").write_bytes(b"")

    util.folders.add_model_folder_path("v4_path_test", tmp_path)

    assert util.folders.get_folder_paths("v4_path_test") == [str(tmp_path)]
    assert util.folders.get_filename_list("v4_path_test") == ["weights.safetensors"]
