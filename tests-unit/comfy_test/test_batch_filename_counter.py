import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from comfy.cli_args import args

_original_cpu = args.cpu
if not torch.cuda.is_available():
    args.cpu = True
try:
    import folder_paths
    from nodes import EmptyImage, SaveImage
    from comfy_extras.nodes_images import SaveImageAdvanced
finally:
    args.cpu = _original_cpu


@pytest.mark.parametrize(
    "prefix,existing,expected",
    [
        pytest.param("image_%batch_num%", ["image_0_00001_.png", "image_12_100000.avif"], 100001, id="mixed-suffixes-and-large-counter"),
        pytest.param("%batch_num%_image", ["12_image_00009.png", "1_image_00004_.jpg"], 10, id="leading-token"),
        pytest.param("shot_%batch_num%_view", ["shot_12_view_00010.png", "shot_12_view_extra_99999.png"], 11, id="middle-token"),
        pytest.param(
            "shot_%batch_num%_view_%batch_num%",
            ["shot_12_view_12_00007.png", "shot_12_view_13_99999.png",
             "shot_12_view_%batch_num%_88888.png", "shot_%batch_num%_view_%batch_num%_00003.mp4"],
            8, id="same-batch-for-repeated-token",
        ),
        pytest.param("%batch_num%%batch_num%", ["1212_00007.png", "1213_99999.png"], 8, id="adjacent-tokens"),
        pytest.param("实验.[v1]+(%batch_num%)", ["实验.[v1]+(12)_00007.png", "实验Av1112_99999.png"], 8, id="literal-regex-characters"),
        pytest.param("image_%batch_num%", ["image_%batch_num%_00090.mp4", "image_12_00012.png"], 91, id="literal-counter-is-greater"),
        pytest.param("image_%batch_num%", ["image_%batch_num%_00012.latent", "image_12_00090_.png"], 91, id="expanded-counter-is-greater"),
        pytest.param("image_%BATCH_NUM%", ["image_%BATCH_NUM%_00004.png", "image_12_99999.png"], 5, id="uppercase-token-is-literal"),
        pytest.param("plain", ["plain_00004_.png", "plain_00009.jpg", "plain_+12_.latent", "plainX_99999.png"], 13, id="unchanged-literal-counter-parsing"),
        pytest.param("image_%batch_num%", ["unrelated_99999.png", "image_x_00009.png", "image_2_bad.png"], 1, id="unrelated-and-invalid-names"),
    ],
)
def test_batch_filename_counter_matches_existing_outputs(tmp_path, prefix, existing, expected):
    for name in existing:
        (tmp_path / name).write_bytes(b"")

    output_dir, filename, counter, subfolder, returned_prefix = folder_paths.get_save_image_path(prefix, str(tmp_path))

    assert counter == expected
    assert Path(output_dir) == tmp_path
    assert filename == returned_prefix == prefix
    assert subfolder == ""


def test_batch_filename_counter_expands_dimensions_in_current_subfolder(tmp_path):
    output = tmp_path / "nested"
    output.mkdir()
    (output / "64x32_12_00011.png").write_bytes(b"")
    (tmp_path / "64x32_12_99999.png").write_bytes(b"")
    prefix = os.path.join("nested", "%width%x%height%_%batch_num%")

    result = folder_paths.get_save_image_path(prefix, str(tmp_path), 64, 32)

    assert result == (str(output), "64x32_%batch_num%", 12, "nested", os.path.join("nested", "64x32_%batch_num%"))


def test_batch_filename_counter_leaves_directory_token_literal(tmp_path):
    literal_dir = tmp_path / "runs_%batch_num%"
    literal_dir.mkdir()
    (literal_dir / "image_00007.png").write_bytes(b"")
    expanded_dir = tmp_path / "runs_0"
    expanded_dir.mkdir()
    (expanded_dir / "image_99999.png").write_bytes(b"")
    prefix = os.path.join("runs_%batch_num%", "image")

    assert folder_paths.get_save_image_path(prefix, str(tmp_path)) == (
        str(literal_dir), "image", 8, "runs_%batch_num%", prefix,
    )


@pytest.mark.parametrize("prefix,existing", [("Shot", "shot_00007.png"), ("Shot_%batch_num%", "shot_2_00007.png")])
def test_batch_filename_counter_uses_platform_normcase(tmp_path, prefix, existing):
    (tmp_path / existing).write_bytes(b"")
    expected = 8 if os.path.normcase("Shot") == os.path.normcase("shot") else 1
    assert folder_paths.get_save_image_path(prefix, str(tmp_path))[2] == expected


def test_batch_filename_counter_creates_missing_directory(tmp_path):
    output = tmp_path / "nested"
    prefix = os.path.join("nested", "image_%batch_num%")
    assert folder_paths.get_save_image_path(prefix, str(tmp_path)) == (
        str(output), "image_%batch_num%", 1, "nested", prefix,
    )
    assert output.is_dir()


@pytest.mark.parametrize("absolute", [False, True])
def test_batch_filename_counter_preserves_directory_containment(tmp_path, absolute):
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside"
    prefix = str(outside / "image_%batch_num%") if absolute else os.path.join("..", "outside", "image_%batch_num%")
    with pytest.raises(Exception, match="Saving image outside the output folder is not allowed"):
        folder_paths.get_save_image_path(prefix, str(output))
    assert not outside.exists()


@pytest.mark.parametrize("saver_kind", ["legacy", "advanced"])
@pytest.mark.parametrize("prefix", ["image", "image_%batch_num%", "shot_%batch_num%_view_%batch_num%"])
def test_batch_filename_saves_preserve_previous_images(tmp_path, monkeypatch, saver_kind, prefix):
    monkeypatch.setattr(folder_paths, "output_directory", str(tmp_path))
    monkeypatch.setattr(args, "fp16_intermediates", False)
    monkeypatch.setattr(args, "gpu_only", False)
    monkeypatch.setattr(args, "disable_metadata", False)
    saver = SaveImage() if saver_kind == "legacy" else SaveImageAdvanced.PREPARE_CLASS_CLONE(None)
    first_files = {}
    all_paths = []
    suffix = "_.png" if saver_kind == "legacy" else ".png"

    for run, (color, batch_size, first_counter) in enumerate([(0xFF0000, 2, 1), (0x0000FF, 3, 3)]):
        images = EmptyImage().generate(64, 32, batch_size=batch_size, color=color)[0]
        original = images.clone()
        prompt = {"run": run, "color": color}
        if saver_kind == "legacy":
            result = saver.save_images(images, prefix, prompt=prompt)
            entries = result["ui"]["images"]
            assert result["result"][0] is images
        else:
            result = saver.execute(images, prefix, {"format": "png", "bit_depth": "8-bit", "input_color_space": "sRGB"})
            entries = result.ui["images"]
            assert result.result[0] is images
        torch.testing.assert_close(images, original, rtol=0, atol=0)

        for path, original_bytes in first_files.items():
            assert path.read_bytes() == original_bytes
            with Image.open(path) as saved:
                np.testing.assert_array_equal(np.asarray(saved), np.broadcast_to([255, 0, 0], (32, 64, 3)))
                if saver_kind == "legacy":
                    assert json.loads(saved.info["prompt"]) == {"run": 0, "color": 0xFF0000}

        paths = [tmp_path / entry["subfolder"] / entry["filename"] for entry in entries]
        expected_paths = [
            tmp_path / f"{prefix.replace('%batch_num%', str(index))}_{first_counter + index:05}{suffix}"
            for index in range(batch_size)
        ]
        assert paths == expected_paths
        assert set(paths).isdisjoint(first_files)
        expected_color = [255, 0, 0] if run == 0 else [0, 0, 255]
        for path in paths:
            with Image.open(path) as saved:
                np.testing.assert_array_equal(np.asarray(saved), np.broadcast_to(expected_color, (32, 64, 3)))
                if saver_kind == "legacy":
                    assert json.loads(saved.info["prompt"]) == prompt
            if run == 0:
                first_files[path] = path.read_bytes()
        all_paths.extend(paths)

    assert set(tmp_path.glob("*.png")) == set(all_paths)
    assert len(all_paths) == len(set(all_paths)) == 5
