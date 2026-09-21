"""Tests for path_utils – asset category resolution."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.assets.services.path_utils import (
    compute_display_name,
    compute_loader_path,
    compute_logical_path,
    get_asset_category_and_relative_path,
    get_known_input_subfolder_tags_from_path,
    get_known_subfolder_tags,
    get_name_and_tags_from_asset_path,
    resolve_destination_from_tags,
)


@pytest.fixture
def fake_dirs():
    """Create temporary input, output, and temp directories."""
    with tempfile.TemporaryDirectory() as root:
        root_path = Path(root)
        input_dir = root_path / "input"
        output_dir = root_path / "output"
        temp_dir = root_path / "temp"
        models_root = root_path / "models"
        models_dir = models_root / "checkpoints"
        for d in (input_dir, output_dir, temp_dir, models_dir):
            d.mkdir(parents=True)

        with patch("app.assets.services.path_utils.folder_paths") as mock_fp:
            mock_fp.get_input_directory.return_value = str(input_dir)
            mock_fp.get_output_directory.return_value = str(output_dir)
            mock_fp.get_temp_directory.return_value = str(temp_dir)
            mock_fp.models_dir = str(models_root)

            with patch(
                "app.assets.services.path_utils.get_comfy_models_folders",
                return_value=[("checkpoints", [str(models_dir)], {".safetensors"})],
            ):
                yield {
                    "input": input_dir,
                    "output": output_dir,
                    "temp": temp_dir,
                    "models_root": models_root,
                    "models": models_dir,
                }


class TestGetAssetCategoryAndRelativePath:
    def test_input_file(self, fake_dirs):
        f = fake_dirs["input"] / "photo.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "input"
        assert rel == "photo.png"

    def test_output_file(self, fake_dirs):
        f = fake_dirs["output"] / "result.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "output"
        assert rel == "result.png"

    def test_temp_file(self, fake_dirs):
        """Regression: temp files must be categorised, not raise ValueError."""
        f = fake_dirs["temp"] / "GLSLShader_output_00004_.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "temp"
        assert rel == "GLSLShader_output_00004_.png"

    def test_temp_file_in_subfolder(self, fake_dirs):
        sub = fake_dirs["temp"] / "sub"
        sub.mkdir()
        f = sub / "ComfyUI_temp_tczip_00004_.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "temp"
        assert os.path.normpath(rel) == os.path.normpath("sub/ComfyUI_temp_tczip_00004_.png")

    def test_model_file(self, fake_dirs):
        f = fake_dirs["models"] / "model.safetensors"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "models"

    def test_model_path_tags_include_registered_model_type_only(self, fake_dirs):
        f = fake_dirs["models"] / "subdir" / "model.safetensors"
        f.parent.mkdir()
        f.touch()

        _name, tags = get_name_and_tags_from_asset_path(str(f))

        assert "models" in tags
        assert "model_type:checkpoints" in tags
        assert "checkpoints" not in tags
        assert "subdir" not in tags

    def test_model_type_preserves_registered_folder_case(self, fake_dirs):
        llm_dir = fake_dirs["models"].parent / "LLM"
        llm_dir.mkdir()
        f = llm_dir / "model.safetensors"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[("LLM", [str(llm_dir)], {".safetensors"})],
        ):
            _name, tags = get_name_and_tags_from_asset_path(str(f))

        assert "models" in tags
        assert "model_type:LLM" in tags
        assert "model_type:llm" not in tags

    def test_path_components_do_not_create_model_type_tags(self, fake_dirs):
        f = fake_dirs["models"] / "loras" / "model.safetensors"
        f.parent.mkdir()
        f.touch()

        _name, tags = get_name_and_tags_from_asset_path(str(f))

        assert "models" in tags
        assert "model_type:checkpoints" in tags
        assert "loras" not in tags
        assert "model_type:loras" not in tags

    def test_shared_root_returns_all_matching_model_type_tags(self, fake_dirs):
        shared_root = fake_dirs["models"].parent / "shared"
        shared_root.mkdir()
        f = shared_root / "foo.safetensors"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[
                ("checkpoints", [str(shared_root)], {".safetensors"}),
                ("loras", [str(shared_root)], {".safetensors"}),
            ],
        ):
            _name, tags = get_name_and_tags_from_asset_path(str(f))

        assert "models" in tags
        assert "model_type:checkpoints" in tags
        assert "model_type:loras" in tags

    def test_shared_root_model_type_tags_respect_bucket_extensions(self, fake_dirs):
        """Buckets sharing a base dir only tag files matching their extensions."""
        shared_root = fake_dirs["models"].parent / "unet"
        shared_root.mkdir()
        safetensors_file = shared_root / "wan.safetensors"
        gguf_file = shared_root / "wan.gguf"
        safetensors_file.touch()
        gguf_file.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[
                ("diffusion_models", [str(shared_root)], {".safetensors"}),
                ("unet_gguf", [str(shared_root)], {".gguf"}),
            ],
        ):
            _name, safetensors_tags = get_name_and_tags_from_asset_path(str(safetensors_file))
            _name, gguf_tags = get_name_and_tags_from_asset_path(str(gguf_file))

        assert "model_type:diffusion_models" in safetensors_tags
        assert "model_type:unet_gguf" not in safetensors_tags
        assert "model_type:unet_gguf" in gguf_tags
        assert "model_type:diffusion_models" not in gguf_tags

    def test_empty_extension_set_tags_any_extension(self, fake_dirs):
        """Custom buckets registered without extensions accept every file."""
        custom_root = fake_dirs["models"].parent / "custom_bucket"
        custom_root.mkdir()
        f = custom_root / "weights.bin"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[("custom_bucket", [str(custom_root)], set())],
        ):
            _name, tags = get_name_and_tags_from_asset_path(str(f))

        assert "models" in tags
        assert "model_type:custom_bucket" in tags

    def test_no_extension_match_keeps_models_tag_without_model_type(self, fake_dirs):
        f = fake_dirs["models"] / "notes.txt"
        f.touch()

        _name, tags = get_name_and_tags_from_asset_path(str(f))

        assert "models" in tags
        assert not any(tag.startswith("model_type:") for tag in tags)

    def test_output_backed_registered_folder_gets_model_and_output_tags(self, fake_dirs):
        output_checkpoints_dir = fake_dirs["output"] / "checkpoints"
        output_checkpoints_dir.mkdir()
        f = output_checkpoints_dir / "saved.safetensors"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[("checkpoints", [str(output_checkpoints_dir)], {".safetensors"})],
        ):
            _name, tags = get_name_and_tags_from_asset_path(str(f))

        assert "models" in tags
        assert "model_type:checkpoints" in tags
        assert "output" in tags

    def test_temp_path_tags_include_temp_not_output_or_preview(self, fake_dirs):
        f = fake_dirs["temp"] / "preview.png"
        f.touch()

        _name, tags = get_name_and_tags_from_asset_path(str(f))

        assert "temp" in tags
        assert "output" not in tags
        assert "preview:true" not in tags

    def test_known_subfolder_tags_are_centralized(self):
        assert get_known_subfolder_tags("pasted") == ["pasted"]
        assert get_known_subfolder_tags("arbitrary") == []

    def test_known_input_subfolder_tags_are_path_derived_for_direct_children(self, fake_dirs):
        f = fake_dirs["input"] / "pasted" / "image.png"
        f.parent.mkdir()
        f.touch()

        assert get_known_input_subfolder_tags_from_path(str(f)) == ["pasted"]

        _name, tags = get_name_and_tags_from_asset_path(str(f))
        assert "input" in tags
        assert "pasted" in tags

    def test_known_input_subfolder_tags_do_not_apply_to_nested_or_other_roots(self, fake_dirs):
        nested = fake_dirs["input"] / "pasted" / "session" / "image.png"
        output = fake_dirs["output"] / "pasted" / "image.png"
        for path in (nested, output):
            path.parent.mkdir(parents=True)
            path.touch()

        assert get_known_input_subfolder_tags_from_path(str(nested)) == []
        assert get_known_input_subfolder_tags_from_path(str(output)) == []

    def test_unknown_path_raises(self, fake_dirs):
        with pytest.raises(ValueError, match="not within"):
            get_asset_category_and_relative_path("/some/random/path.png")


class TestResponseStoragePaths:
    def test_input_file_path_and_display_name_include_subfolder(self, fake_dirs):
        sub = fake_dirs["input"] / "some" / "folder"
        sub.mkdir(parents=True)
        f = sub / "image.png"
        f.touch()

        assert compute_logical_path(str(f)) == "input/some/folder/image.png"
        assert compute_display_name(str(f)) == "some/folder/image.png"

    def test_output_file_path_and_display_name_include_subfolder(self, fake_dirs):
        sub = fake_dirs["output"] / "renders"
        sub.mkdir()
        f = sub / "ComfyUI_00001_.png"
        f.touch()

        assert compute_logical_path(str(f)) == "output/renders/ComfyUI_00001_.png"
        assert compute_display_name(str(f)) == "renders/ComfyUI_00001_.png"

    def test_temp_file_path_and_display_name(self, fake_dirs):
        f = fake_dirs["temp"] / "preview.png"
        f.touch()

        assert compute_logical_path(str(f)) == "temp/preview.png"
        assert compute_display_name(str(f)) == "preview.png"

    def test_exact_storage_root_has_no_display_name(self, fake_dirs):
        assert compute_logical_path(str(fake_dirs["input"])) == "input"
        assert compute_display_name(str(fake_dirs["input"])) is None

    def test_longest_matching_builtin_root_wins(self, fake_dirs, tmp_path: Path):
        nested_output = fake_dirs["input"] / "nested-output"
        nested_output.mkdir()
        f = nested_output / "image.png"
        f.touch()

        with patch("app.assets.services.path_utils.folder_paths") as mock_fp:
            mock_fp.get_input_directory.return_value = str(fake_dirs["input"])
            mock_fp.get_output_directory.return_value = str(nested_output)
            mock_fp.get_temp_directory.return_value = str(tmp_path / "temp")
            mock_fp.models_dir = str(fake_dirs["models_root"])

            assert compute_logical_path(str(f)) == "output/image.png"
            assert compute_display_name(str(f)) == "image.png"

    def test_model_file_path_is_relative_to_physical_models_root(self, fake_dirs):
        sub = fake_dirs["models"] / "flux"
        sub.mkdir()
        f = sub / "model.safetensors"
        f.touch()

        assert compute_logical_path(str(f)) == "models/checkpoints/flux/model.safetensors"
        assert compute_display_name(str(f)) == "checkpoints/flux/model.safetensors"

        name, tags = get_name_and_tags_from_asset_path(str(f))
        assert name == "model.safetensors"
        assert "models" in tags
        assert "model_type:checkpoints" in tags
        assert "checkpoints" not in tags
        assert "flux" not in tags

    @pytest.mark.parametrize(
        "folder_name",
        ["checkpoints", "clip", "vae", "diffusion_models", "loras"],
    )
    def test_output_model_folder_uses_output_storage_file_path(self, fake_dirs, folder_name):
        output_model_dir = fake_dirs["output"] / folder_name
        output_model_dir.mkdir(exist_ok=True)
        default_model_dir = fake_dirs["models_root"] / folder_name
        default_model_dir.mkdir(exist_ok=True)
        f = output_model_dir / "saved.safetensors"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[
                (folder_name, [str(default_model_dir), str(output_model_dir)], {".safetensors"})
            ],
        ):
            assert compute_logical_path(str(f)) == f"output/{folder_name}/saved.safetensors"
            assert compute_display_name(str(f)) == f"{folder_name}/saved.safetensors"

            name, tags = get_name_and_tags_from_asset_path(str(f))
            assert name == "saved.safetensors"
            assert "output" in tags
            assert "models" in tags
            assert f"model_type:{folder_name}" in tags
            assert folder_name not in tags

    def test_output_model_subfolder_uses_output_storage_file_path(self, fake_dirs):
        folder_name = "loras"
        output_model_dir = fake_dirs["output"] / folder_name
        subdir = output_model_dir / "experiments"
        subdir.mkdir(parents=True)
        f = subdir / "my_lora.safetensors"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[(folder_name, [str(output_model_dir)], {".safetensors"})],
        ):
            assert (
                compute_logical_path(str(f))
                == "output/loras/experiments/my_lora.safetensors"
            )
            assert compute_display_name(str(f)) == "loras/experiments/my_lora.safetensors"

            name, tags = get_name_and_tags_from_asset_path(str(f))
            assert name == "my_lora.safetensors"
            assert "output" in tags
            assert "models" in tags
            assert "model_type:loras" in tags
            assert "loras" not in tags
            assert "experiments" not in tags

    def test_external_model_folder_without_provenance_has_no_file_path(self, tmp_path: Path):
        external_checkpoints_dir = tmp_path / "external" / "not_named_like_category"
        external_checkpoints_dir.mkdir(parents=True)
        f = external_checkpoints_dir / "external.safetensors"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[("checkpoints", [str(external_checkpoints_dir)], {".safetensors"})],
        ):
            assert compute_logical_path(str(f)) is None
            assert compute_display_name(str(f)) is None

            name, tags = get_name_and_tags_from_asset_path(str(f))
            assert name == "external.safetensors"
            assert "models" in tags
            assert "model_type:checkpoints" in tags

    def test_same_relative_model_file_under_multiple_external_roots_has_no_storage_file_path(
        self, tmp_path: Path
    ):
        foo_dir = tmp_path / "foo"
        bar_dir = tmp_path / "bar"
        foo_dir.mkdir()
        bar_dir.mkdir()
        foo_file = foo_dir / "baz.safetensors"
        bar_file = bar_dir / "baz.safetensors"
        foo_file.touch()
        bar_file.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[("checkpoints", [str(foo_dir), str(bar_dir)], {".safetensors"})],
        ):
            assert compute_logical_path(str(foo_file)) is None
            assert compute_logical_path(str(bar_file)) is None
            assert compute_display_name(str(foo_file)) is None
            assert compute_display_name(str(bar_file)) is None

    def test_output_clip_folder_uses_output_storage_and_text_encoder_tag(self, fake_dirs):
        output_clip_dir = fake_dirs["output"] / "clip"
        output_clip_dir.mkdir()
        f = output_clip_dir / "clip_l.safetensors"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[("text_encoders", [str(output_clip_dir)], {".safetensors"})],
        ):
            assert compute_logical_path(str(f)) == "output/clip/clip_l.safetensors"
            assert compute_display_name(str(f)) == "clip/clip_l.safetensors"

            name, tags = get_name_and_tags_from_asset_path(str(f))
            assert name == "clip_l.safetensors"
            assert "output" in tags
            assert "models" in tags
            assert "model_type:text_encoders" in tags
            assert "clip" not in tags

    def test_physical_unet_folder_uses_storage_path_and_diffusion_models_tag(self, fake_dirs):
        unet_dir = fake_dirs["models_root"] / "unet"
        diffusion_models_dir = fake_dirs["models_root"] / "diffusion_models"
        unet_dir.mkdir()
        diffusion_models_dir.mkdir()
        f = unet_dir / "wan.safetensors"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[
                ("diffusion_models", [str(unet_dir), str(diffusion_models_dir)], {".safetensors"})
            ],
        ):
            assert compute_logical_path(str(f)) == "models/unet/wan.safetensors"
            assert compute_display_name(str(f)) == "unet/wan.safetensors"

            name, tags = get_name_and_tags_from_asset_path(str(f))
            assert name == "wan.safetensors"
            assert "models" in tags
            assert "model_type:diffusion_models" in tags
            assert "unet" not in tags

    def test_unregistered_file_under_physical_models_root_still_has_storage_file_path(self, fake_dirs):
        f = fake_dirs["models_root"] / "not_registered" / "orphan.bin"
        f.parent.mkdir()
        f.touch()

        assert compute_logical_path(str(f)) == "models/not_registered/orphan.bin"
        assert compute_display_name(str(f)) == "not_registered/orphan.bin"

    def test_output_checkpoint_folder_without_registration_has_only_output_tag(self, fake_dirs):
        f = fake_dirs["output"] / "checkpoints" / "saved.safetensors"
        f.parent.mkdir(exist_ok=True)
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[],
        ):
            assert compute_logical_path(str(f)) == "output/checkpoints/saved.safetensors"
            assert compute_display_name(str(f)) == "checkpoints/saved.safetensors"

            name, tags = get_name_and_tags_from_asset_path(str(f))
            assert name == "saved.safetensors"
            assert "output" in tags
            assert "models" not in tags
            assert not any(tag.startswith("model_type:") for tag in tags)

    def test_unknown_path_returns_none(self):
        assert compute_logical_path("/some/random/path.png") is None
        assert compute_display_name("/some/random/path.png") is None


class TestLoaderPath:
    """In-root loader path: relative to the storage root, model category dropped."""

    def test_model_loader_path_drops_category(self, fake_dirs):
        sub = fake_dirs["models"] / "flux"
        sub.mkdir()
        f = sub / "model.safetensors"
        f.touch()

        # logical_path keeps the category, file_path (loader) drops it
        assert compute_logical_path(str(f)) == "models/checkpoints/flux/model.safetensors"
        assert compute_loader_path(str(f)) == "flux/model.safetensors"

    def test_model_loader_path_flat_file(self, fake_dirs):
        f = fake_dirs["models"] / "model.safetensors"
        f.touch()

        assert compute_loader_path(str(f)) == "model.safetensors"

    def test_input_loader_path_keeps_subfolders(self, fake_dirs):
        sub = fake_dirs["input"] / "some" / "folder"
        sub.mkdir(parents=True)
        f = sub / "image.png"
        f.touch()

        assert compute_loader_path(str(f)) == "some/folder/image.png"

    def test_temp_loader_path(self, fake_dirs):
        f = fake_dirs["temp"] / "preview.png"
        f.touch()

        assert compute_loader_path(str(f)) == "preview.png"

    def test_unregistered_file_under_models_root_has_no_loader_path(self, fake_dirs):
        # Under models_root but not within any registered category base.
        f = fake_dirs["models_root"] / "not_registered" / "orphan.bin"
        f.parent.mkdir()
        f.touch()

        # It still has a namespaced logical_path, but no loader path.
        assert compute_logical_path(str(f)) == "models/not_registered/orphan.bin"
        assert compute_loader_path(str(f)) is None

    def test_extension_mismatch_in_registered_bucket_has_no_loader_path(self, fake_dirs):
        # Inside a registered bucket, but the bucket's extension set cannot
        # load it: no model_type tag, and no loader path either.
        f = fake_dirs["models"] / "notes.txt"
        f.touch()

        assert compute_logical_path(str(f)) == "models/checkpoints/notes.txt"
        assert compute_loader_path(str(f)) is None

    def test_shared_base_loader_path_uses_extension_matching_bucket(self, fake_dirs):
        shared_root = fake_dirs["models"].parent / "unet"
        shared_root.mkdir()
        f = shared_root / "wan.gguf"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[
                ("diffusion_models", [str(shared_root)], {".safetensors"}),
                ("unet_gguf", [str(shared_root)], {".gguf"}),
            ],
        ):
            assert compute_loader_path(str(f)) == "wan.gguf"

    def test_match_all_bucket_provides_loader_path_for_any_extension(self, fake_dirs):
        custom_root = fake_dirs["models"].parent / "custom_bucket"
        custom_root.mkdir()
        f = custom_root / "weights.bin"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[("custom_bucket", [str(custom_root)], set())],
        ):
            assert compute_loader_path(str(f)) == "weights.bin"

    def test_extra_path_model_has_loader_path_but_no_logical_path(self, tmp_path: Path):
        """Registered category base outside models_dir (extra_model_paths style).

        Loadable, so loader_path resolves; but it is not under any canonical
        storage root, so logical_path/display_name are None. This asymmetry is
        intentional: loader_path resolves every registered model-folder base,
        logical_path only resolves the canonical storage roots.
        """
        extra = tmp_path / "extra_ckpts"
        extra.mkdir()
        f = extra / "foo.safetensors"
        f.touch()

        with patch("app.assets.services.path_utils.folder_paths") as mock_fp, patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[("checkpoints", [str(extra)], {".safetensors"})],
        ):
            mock_fp.get_input_directory.return_value = str(tmp_path / "in")
            mock_fp.get_output_directory.return_value = str(tmp_path / "out")
            mock_fp.get_temp_directory.return_value = str(tmp_path / "tmp")
            mock_fp.models_dir = str(tmp_path / "models")  # extra is NOT under this

            assert compute_loader_path(str(f)) == "foo.safetensors"
            assert compute_logical_path(str(f)) is None
            assert compute_display_name(str(f)) is None

    def test_unknown_path_returns_none(self):
        assert compute_loader_path("/some/random/path.png") is None


class TestResolveDestinationFromTags:
    def test_extra_tags_are_not_path_components(self, fake_dirs):
        base_dir, subdirs = resolve_destination_from_tags(["input", "unit-tests", "foo"])

        assert base_dir == os.path.abspath(fake_dirs["input"])
        assert subdirs == []

    def test_model_upload_rejects_non_writable_registered_folders(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            checkpoints_dir = root_path / "models" / "checkpoints"
            configs_dir = root_path / "models" / "configs"
            custom_nodes_dir = root_path / "custom_nodes"
            for path in (checkpoints_dir, configs_dir, custom_nodes_dir):
                path.mkdir(parents=True)

            with patch("app.assets.services.path_utils.folder_paths") as mock_fp:
                mock_fp.folder_names_and_paths = {
                    "checkpoints": ([str(checkpoints_dir)], set()),
                    "configs": ([str(configs_dir)], set()),
                    "custom_nodes": ([str(custom_nodes_dir)], set()),
                }

                base_dir, subdirs = resolve_destination_from_tags(
                    ["models", "model_type:checkpoints"]
                )
                assert base_dir == os.path.abspath(checkpoints_dir)
                assert subdirs == []

                for folder_name in ("configs", "custom_nodes"):
                    with pytest.raises(ValueError, match="unknown model category"):
                        resolve_destination_from_tags(
                            ["models", f"model_type:{folder_name}"]
                        )
