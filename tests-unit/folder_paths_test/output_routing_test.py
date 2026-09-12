import datetime
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import folder_paths
import output_routing


SAMPLE_POLICY_PATH = Path(__file__).resolve().parents[2] / "output-policy.metadata-example.json"


def test_output_policy_routes_dimensions_to_filename_and_preserves_counter(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({
        "version": 1,
        "defaults": {"output": "archive"},
        "profiles": {"archive": {
            "folder_template": "{date:%Y-%m-%d}/{prefix_dir}",
            "filename_template": "{prefix_stem}_{width}x{height}",
        }},
    }), encoding="utf-8")
    monkeypatch.setattr(folder_paths, "output_directory", str(output_dir))

    folder_paths.configure_output_routing(str(policy_path))
    expected_subfolder = os.path.join("2026-09-05", "portraits")
    with patch.object(output_routing.datetime, "datetime", wraps=datetime.datetime) as mock_datetime:
        mock_datetime.now.return_value = datetime.datetime(2026, 9, 5, 12, 0)
        full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            "portraits/ComfyUI", str(output_dir), 640, 480
        )
        (output_dir / expected_subfolder / "ComfyUI_640x480_00001_.png").touch()
        _, _, next_counter, _, _ = folder_paths.get_save_image_path(
            "portraits/ComfyUI", str(output_dir), 640, 480
        )

    assert full_folder == str(output_dir / expected_subfolder)
    assert filename == "ComfyUI_640x480"
    assert counter == 1
    assert next_counter == 2
    assert subfolder == expected_subfolder


def test_output_policy_configures_output_directory_unless_cli_overrides(monkeypatch, tmp_path):
    policy_output_dir = tmp_path / "policy-output"
    cli_output_dir = tmp_path / "cli-output"
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({
        "version": 1,
        "output_directory": str(policy_output_dir),
    }), encoding="utf-8")
    monkeypatch.setattr(folder_paths, "output_directory", str(cli_output_dir))
    monkeypatch.setattr(folder_paths.args, "output_directory", None)

    folder_paths.configure_output_routing(str(policy_path))
    assert folder_paths.get_output_directory() == str(policy_output_dir)

    monkeypatch.setattr(folder_paths, "output_directory", str(cli_output_dir))
    monkeypatch.setattr(folder_paths.args, "output_directory", str(cli_output_dir))
    folder_paths.configure_output_routing(str(policy_path))
    assert folder_paths.get_output_directory() == str(cli_output_dir)


def test_output_policy_relative_output_directory_uses_configured_output_root(monkeypatch, tmp_path):
    configured_output_dir = tmp_path / "output"
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({
        "version": 1,
        "output_directory": "managed-output",
    }), encoding="utf-8")
    monkeypatch.setattr(folder_paths, "output_directory", str(configured_output_dir))
    monkeypatch.setattr(folder_paths.args, "output_directory", None)

    folder_paths.configure_output_routing(str(policy_path))

    assert folder_paths.get_output_directory() == str(configured_output_dir / "managed-output")


@pytest.mark.parametrize("output_directory", ["../managed-output", "C:managed-output"])
def test_output_policy_rejects_unsafe_relative_output_directory(monkeypatch, tmp_path, output_directory):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({
        "version": 1,
        "output_directory": output_directory,
    }), encoding="utf-8")
    monkeypatch.setattr(folder_paths, "output_routing_policy", output_routing.legacy_policy())

    with pytest.raises(output_routing.OutputRoutingError):
        folder_paths.configure_output_routing(str(policy_path))


def test_output_policy_without_filename_template_preserves_filename_stem(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({
        "version": 1,
        "defaults": {"output": "archive"},
        "profiles": {"archive": {"folder_template": "archive/{prefix_dir}"}},
    }), encoding="utf-8")
    monkeypatch.setattr(folder_paths, "output_directory", str(output_dir))

    folder_paths.configure_output_routing(str(policy_path))
    _, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        "portraits/ComfyUI", str(output_dir), 640, 480
    )

    assert filename == "ComfyUI"
    assert counter == 1
    assert subfolder == os.path.join("archive", "portraits")


def test_shipped_sample_policy_routes_output_and_temp_profiles():
    policy = output_routing.load_policy(str(SAMPLE_POLICY_PATH))
    context = output_routing.OutputRouteContext(
        width=640,
        height=480,
        prefix_dir="portraits",
        prefix_stem="ComfyUI",
    )
    with patch.object(output_routing.datetime, "datetime", wraps=datetime.datetime) as mock_datetime:
        mock_datetime.now.return_value = datetime.datetime(2026, 9, 5, 12, 0)
        output_folder, output_stem = output_routing.resolve_route(policy, "output", context)
        temp_folder, temp_stem = output_routing.resolve_route(policy, "temp", context)

    assert policy.output_directory == "F:\\ComfyUI\\_Output"
    assert output_folder == os.path.join("2026-09-05", "portraits")
    assert output_stem == "ComfyUI_640x480"
    assert temp_folder == os.path.join("temp", "2026-09-05", "portraits")
    assert temp_stem == "ComfyUI_640x480"


@pytest.mark.parametrize("template", ["../outside", "C:\\outside", "{unknown}", "{job_id}", "{job_short}", "{node_id}", "{workflow_id}", "{list_index}", "{date:%q}"])
def test_output_policy_rejects_unsafe_or_unknown_templates(monkeypatch, tmp_path, template):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({
        "version": 1,
        "profiles": {"invalid": {"folder_template": template}},
    }), encoding="utf-8")

    monkeypatch.setattr(folder_paths, "output_routing_policy", output_routing.legacy_policy())
    with pytest.raises(output_routing.OutputRoutingError):
        folder_paths.configure_output_routing(str(policy_path))


@pytest.mark.parametrize("template", ["", "../outside", "C:\\outside", "name/part", "name.png", "{counter:05}"])
def test_output_policy_rejects_unsafe_or_unsupported_filename_templates(monkeypatch, tmp_path, template):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({
        "version": 1,
        "profiles": {"invalid": {
            "folder_template": "archive",
            "filename_template": template,
        }},
    }), encoding="utf-8")

    monkeypatch.setattr(folder_paths, "output_routing_policy", output_routing.legacy_policy())
    with pytest.raises(output_routing.OutputRoutingError):
        folder_paths.configure_output_routing(str(policy_path))
