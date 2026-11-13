import pytest
import yaml
import os
import sys
from unittest.mock import Mock, patch, mock_open

from utils.extra_config import load_extra_path_config
import folder_paths


@pytest.fixture()
def clear_folder_paths():
    # Clear the global dictionary before each test to ensure isolation
    original = folder_paths.folder_names_and_paths.copy()
    folder_paths.folder_names_and_paths.clear()
    yield
    folder_paths.folder_names_and_paths = original


@pytest.fixture
def mock_yaml_content():
    return {
        'test_config': {
            'base_path': '~/App/',
            'checkpoints': 'subfolder1',
        }
    }


@pytest.fixture
def mock_expanded_home():
    return '/home/user'


@pytest.fixture
def yaml_config_with_appdata():
    return """
    test_config:
      base_path: '%APPDATA%/ComfyUI'
      checkpoints: 'models/checkpoints'
    """


@pytest.fixture
def mock_yaml_content_appdata(yaml_config_with_appdata):
    return yaml.safe_load(yaml_config_with_appdata)


@pytest.fixture
def mock_expandvars_appdata():
    mock = Mock()

    def expandvars(path):
        if '%APPDATA%' in path:
            if sys.platform == 'win32':
                return path.replace('%APPDATA%', 'C:/Users/TestUser/AppData/Roaming')
            else:
                return path.replace('%APPDATA%', '/Users/TestUser/AppData/Roaming')
        return path

    mock.side_effect = expandvars
    return mock


@pytest.fixture
def mock_add_model_folder_path():
    return Mock()


@pytest.fixture
def mock_expanduser(mock_expanded_home):
    def _expanduser(path):
        if path.startswith('~/'):
            return os.path.join(mock_expanded_home, path[2:])
        return path
    return _expanduser


@pytest.fixture
def mock_yaml_safe_load(mock_yaml_content):
    return Mock(return_value=mock_yaml_content)


@patch('builtins.open', new_callable=mock_open, read_data="dummy file content")
def test_load_extra_model_paths_expands_userpath(
    mock_file,
    monkeypatch,
    mock_add_model_folder_path,
    mock_expanduser,
    mock_yaml_safe_load,
    mock_expanded_home
):
    # Attach mocks used by load_extra_path_config
    monkeypatch.setattr(folder_paths, 'add_model_folder_path', mock_add_model_folder_path)
    monkeypatch.setattr(os.path, 'expanduser', mock_expanduser)
    monkeypatch.setattr(yaml, 'safe_load', mock_yaml_safe_load)

    dummy_yaml_file_name = 'dummy_path.yaml'
    load_extra_path_config(dummy_yaml_file_name)

    expected_calls = [
        ('checkpoints', os.path.join(mock_expanded_home, 'App', 'subfolder1'), False),
    ]

    assert mock_add_model_folder_path.call_count == len(expected_calls)

    # Check if add_model_folder_path was called with the correct arguments
    for actual_call, expected_call in zip(mock_add_model_folder_path.call_args_list, expected_calls):
        assert actual_call.args[0] == expected_call[0]
        assert os.path.normpath(actual_call.args[1]) == os.path.normpath(expected_call[1])  # Normalize and check the path to check on multiple OS.
        assert actual_call.args[2] == expected_call[2]

    # Check if yaml.safe_load was called
    mock_yaml_safe_load.assert_called_once()

    # Check if open was called with the correct file path
    mock_file.assert_called_once_with(dummy_yaml_file_name, 'r', encoding='utf-8')


@patch('builtins.open', new_callable=mock_open)
def test_load_extra_model_paths_expands_appdata(
    mock_file,
    monkeypatch,
    mock_add_model_folder_path,
    mock_expandvars_appdata,
    yaml_config_with_appdata,
    mock_yaml_content_appdata
):
    # Set the mock_file to return yaml with appdata as a variable
    mock_file.return_value.read.return_value = yaml_config_with_appdata

    # Attach mocks
    monkeypatch.setattr(folder_paths, 'add_model_folder_path', mock_add_model_folder_path)
    monkeypatch.setattr(os.path, 'expandvars', mock_expandvars_appdata)
    monkeypatch.setattr(yaml, 'safe_load', Mock(return_value=mock_yaml_content_appdata))

    # Mock expanduser to do nothing (since we're not testing it here)
    monkeypatch.setattr(os.path, 'expanduser', lambda x: x)

    dummy_yaml_file_name = 'dummy_path.yaml'
    load_extra_path_config(dummy_yaml_file_name)

    if sys.platform == "win32":
        expected_base_path = 'C:/Users/TestUser/AppData/Roaming/ComfyUI'
    else:
        expected_base_path = '/Users/TestUser/AppData/Roaming/ComfyUI'
    expected_calls = [
        ('checkpoints', os.path.normpath(os.path.join(expected_base_path, 'models/checkpoints')), False),
    ]

    assert mock_add_model_folder_path.call_count == len(expected_calls)

    # Check the base path variable was expanded
    for actual_call, expected_call in zip(mock_add_model_folder_path.call_args_list, expected_calls):
        assert actual_call.args == expected_call

    # Verify that expandvars was called
    assert mock_expandvars_appdata.called


@patch("builtins.open", new_callable=mock_open, read_data="dummy yaml content")
@patch("yaml.safe_load")
def test_load_extra_path_config_relative_base_path(
    mock_yaml_load, _mock_file, clear_folder_paths, monkeypatch, tmp_path
):
    """
    Test that when 'base_path' is a relative path in the YAML, it is joined to the YAML file directory, and then
    the items in the config are correctly converted to absolute paths.
    """
    sub_folder = "./my_rel_base"
    config_data = {
        "some_model_folder": {
            "base_path": sub_folder,
            "is_default": True,
            "checkpoints": "checkpoints",
            "some_key": "some_value"
        }
    }
    mock_yaml_load.return_value = config_data

    dummy_yaml_name = "dummy_file.yaml"

    def fake_abspath(path):
        if path == dummy_yaml_name:
            # If it's the YAML path, treat it like it lives in tmp_path
            return os.path.join(str(tmp_path), dummy_yaml_name)
        return os.path.join(str(tmp_path), path)  # Otherwise, do a normal join relative to tmp_path

    def fake_dirname(path):
        # We expect path to be the result of fake_abspath(dummy_yaml_name)
        if path.endswith(dummy_yaml_name):
            return str(tmp_path)
        return os.path.dirname(path)

    monkeypatch.setattr(os.path, "abspath", fake_abspath)
    monkeypatch.setattr(os.path, "dirname", fake_dirname)

    load_extra_path_config(dummy_yaml_name)

    expected_checkpoints = os.path.abspath(os.path.join(str(tmp_path), "my_rel_base", "checkpoints"))
    expected_some_value = os.path.abspath(os.path.join(str(tmp_path), "my_rel_base", "some_value"))

    actual_paths = folder_paths.folder_names_and_paths["checkpoints"][0]
    assert len(actual_paths) == 1, "Should have one path added for 'checkpoints'."
    assert actual_paths[0] == expected_checkpoints

    actual_paths = folder_paths.folder_names_and_paths["some_key"][0]
    assert len(actual_paths) == 1, "Should have one path added for 'some_key'."
    assert actual_paths[0] == expected_some_value


@patch("builtins.open", new_callable=mock_open, read_data="dummy yaml content")
@patch("yaml.safe_load")
def test_load_extra_path_config_absolute_base_path(
    mock_yaml_load, _mock_file, clear_folder_paths, monkeypatch, tmp_path
):
    """
    Test that when 'base_path' is an absolute path, each subdirectory is joined with that absolute path,
    rather than being relative to the YAML's directory.
    """
    abs_base = os.path.join(str(tmp_path), "abs_base")
    config_data = {
        "some_absolute_folder": {
            "base_path": abs_base,   # <-- absolute
            "is_default": True,
            "loras": "loras_folder",
            "embeddings": "embeddings_folder"
        }
    }
    mock_yaml_load.return_value = config_data

    dummy_yaml_name = "dummy_abs.yaml"

    def fake_abspath(path):
        if path == dummy_yaml_name:
            # If it's the YAML path, treat it like it is in tmp_path
            return os.path.join(str(tmp_path), dummy_yaml_name)
        return path  # For absolute base, we just return path directly

    def fake_dirname(path):
        return str(tmp_path) if path.endswith(dummy_yaml_name) else os.path.dirname(path)

    monkeypatch.setattr(os.path, "abspath", fake_abspath)
    monkeypatch.setattr(os.path, "dirname", fake_dirname)

    load_extra_path_config(dummy_yaml_name)

    # Expect the final paths to be <abs_base>/loras_folder and <abs_base>/embeddings_folder
    expected_loras = os.path.join(abs_base, "loras_folder")
    expected_embeddings = os.path.join(abs_base, "embeddings_folder")

    actual_loras = folder_paths.folder_names_and_paths["loras"][0]
    assert len(actual_loras) == 1, "Should have one path for 'loras'."
    assert actual_loras[0] == os.path.abspath(expected_loras)

    actual_embeddings = folder_paths.folder_names_and_paths["embeddings"][0]
    assert len(actual_embeddings) == 1, "Should have one path for 'embeddings'."
    assert actual_embeddings[0] == os.path.abspath(expected_embeddings)


@patch("builtins.open", new_callable=mock_open, read_data="dummy yaml content")
@patch("yaml.safe_load")
def test_load_extra_path_config_no_base_path(
    mock_yaml_load, _mock_file, clear_folder_paths, monkeypatch, tmp_path
):
    """
    Test that if 'base_path' is not present, each path is joined
    with the directory of the YAML file (unless it's already absolute).
    """
    config_data = {
        "some_folder_without_base": {
            "is_default": True,
            "text_encoders": "clip",
            "diffusion_models": "unet"
        }
    }
    mock_yaml_load.return_value = config_data

    dummy_yaml_name = "dummy_no_base.yaml"

    def fake_abspath(path):
        if path == dummy_yaml_name:
            return os.path.join(str(tmp_path), dummy_yaml_name)
        return os.path.join(str(tmp_path), path)

    def fake_dirname(path):
        return str(tmp_path) if path.endswith(dummy_yaml_name) else os.path.dirname(path)

    monkeypatch.setattr(os.path, "abspath", fake_abspath)
    monkeypatch.setattr(os.path, "dirname", fake_dirname)

    load_extra_path_config(dummy_yaml_name)

    expected_clip = os.path.join(str(tmp_path), "clip")
    expected_unet = os.path.join(str(tmp_path), "unet")

    actual_text_encoders = folder_paths.folder_names_and_paths["text_encoders"][0]
    assert len(actual_text_encoders) == 1, "Should have one path for 'text_encoders'."
    assert actual_text_encoders[0] == os.path.abspath(expected_clip)

    actual_diffusion = folder_paths.folder_names_and_paths["diffusion_models"][0]
    assert len(actual_diffusion) == 1, "Should have one path for 'diffusion_models'."
    assert actual_diffusion[0] == os.path.abspath(expected_unet)


@patch("yaml.safe_load")
def test_load_extra_path_config_implicit_subdirs(
    mock_yaml_load, clear_folder_paths, tmp_path
):
    """
    When base_path is set and no explicit sub-paths are declared, any subdir
    whose name matches a known category is auto-registered.
    """
    # Create real subdirs that match known categories
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "loras").mkdir()
    (tmp_path / "unknown_dir").mkdir()  # not a registered category — should be ignored

    config_data = {
        "comfyui": {
            "base_path": str(tmp_path),
        }
    }
    mock_yaml_load.return_value = config_data

    # Pre-populate only the categories we're testing so clear_folder_paths doesn't hide them
    folder_paths.folder_names_and_paths["checkpoints"] = ([], set())
    folder_paths.folder_names_and_paths["loras"] = ([], set())

    yaml_path = str(tmp_path / "extra_model_paths.yaml")
    with open(yaml_path, "w") as f:
        f.write("")  # content ignored; yaml.safe_load is mocked

    load_extra_path_config(yaml_path)

    assert str(tmp_path / "checkpoints") in folder_paths.folder_names_and_paths["checkpoints"][0]
    assert str(tmp_path / "loras") in folder_paths.folder_names_and_paths["loras"][0]
    assert "unknown_dir" not in folder_paths.folder_names_and_paths


@patch("yaml.safe_load")
def test_implicit_scan_excludes_custom_nodes(
    mock_yaml_load, clear_folder_paths, tmp_path
):
    """custom_nodes must never be auto-registered by the implicit scan."""
    (tmp_path / "custom_nodes").mkdir()
    (tmp_path / "checkpoints").mkdir()

    config_data = {"comfyui": {"base_path": str(tmp_path)}}
    mock_yaml_load.return_value = config_data

    folder_paths.folder_names_and_paths["checkpoints"] = ([], set())
    folder_paths.folder_names_and_paths["custom_nodes"] = ([], set())

    yaml_path = str(tmp_path / "extra_paths.yaml")
    with open(yaml_path, "w") as f:
        f.write("")

    load_extra_path_config(yaml_path)

    assert str(tmp_path / "checkpoints") in folder_paths.folder_names_and_paths["checkpoints"][0]
    assert str(tmp_path / "custom_nodes") not in folder_paths.folder_names_and_paths["custom_nodes"][0], \
        "custom_nodes must not be auto-registered by the implicit scan"


@patch("yaml.safe_load")
def test_load_extra_path_config_explicit_overrides_implicit(
    mock_yaml_load, clear_folder_paths, tmp_path
):
    """
    Explicit sub-path declarations take precedence; the implicit scan must not
    double-register a category that was already declared explicitly.
    """
    (tmp_path / "loras").mkdir()
    custom_loras = tmp_path / "my_custom_loras"
    custom_loras.mkdir()

    config_data = {
        "comfyui": {
            "base_path": str(tmp_path),
            "loras": "my_custom_loras",  # explicit override
        }
    }
    mock_yaml_load.return_value = config_data
    folder_paths.folder_names_and_paths["loras"] = ([], set())

    yaml_path = str(tmp_path / "extra_model_paths.yaml")
    with open(yaml_path, "w") as f:
        f.write("")

    load_extra_path_config(yaml_path)

    registered = folder_paths.folder_names_and_paths["loras"][0]
    assert str(custom_loras) in registered
    assert str(tmp_path / "loras") not in registered, "Implicit path must not override explicit"


@pytest.fixture
def save_restore_system_dirs():
    """Save and restore folder_paths system directories around a test."""
    saved = {
        "output": folder_paths.get_output_directory(),
        "input": folder_paths.get_input_directory(),
        "temp": folder_paths.get_temp_directory(),
        "user": folder_paths.get_user_directory(),
    }
    yield
    folder_paths.set_output_directory(saved["output"])
    folder_paths.set_input_directory(saved["input"])
    folder_paths.set_temp_directory(saved["temp"])
    folder_paths.set_user_directory(saved["user"])


@patch("yaml.safe_load")
def test_system_dir_keys(mock_yaml_load, save_restore_system_dirs, tmp_path):
    """System directory keys (output, input, temp, user) call set_*_directory()."""
    config_data = {
        "comfyui": {
            "base_path": str(tmp_path),
            "output": "my_output/",
            "input": "my_input/",
            "temp": "my_temp/",
            "user": "my_user/",
        }
    }
    mock_yaml_load.return_value = config_data

    yaml_path = str(tmp_path / "extra_paths.yaml")
    with open(yaml_path, "w") as f:
        f.write("")

    load_extra_path_config(yaml_path, allow_system_dirs=True)

    assert folder_paths.get_output_directory() == os.path.normpath(str(tmp_path / "my_output"))
    assert folder_paths.get_input_directory() == os.path.normpath(str(tmp_path / "my_input"))
    assert folder_paths.get_temp_directory() == os.path.normpath(str(tmp_path / "my_temp"))
    assert folder_paths.get_user_directory() == os.path.normpath(str(tmp_path / "my_user"))


@patch("yaml.safe_load")
def test_system_dir_keys_not_applied_for_legacy(mock_yaml_load, save_restore_system_dirs, tmp_path):
    """System directory keys are ignored when allow_system_dirs=False (legacy extra_model_paths.yaml)."""
    original_output = folder_paths.get_output_directory()
    config_data = {
        "comfyui": {
            "base_path": str(tmp_path),
            "output": "my_output/",
        }
    }
    mock_yaml_load.return_value = config_data

    yaml_path = str(tmp_path / "extra_model_paths.yaml")
    with open(yaml_path, "w") as f:
        f.write("")

    load_extra_path_config(yaml_path)  # allow_system_dirs defaults to False

    # output should be unchanged — treated as a model category, not a system dir
    assert folder_paths.get_output_directory() == original_output


@patch("yaml.safe_load")
def test_nested_models_block(mock_yaml_load, clear_folder_paths, tmp_path):
    """Nested models: block registers model paths relative to models/base_path."""
    config_data = {
        "comfyui": {
            "base_path": str(tmp_path),
            "models": {
                "base_path": "models/",
                "checkpoints": "checkpoints/",
                "loras": "loras/",
            },
        }
    }
    mock_yaml_load.return_value = config_data

    folder_paths.folder_names_and_paths["checkpoints"] = ([], set())
    folder_paths.folder_names_and_paths["loras"] = ([], set())

    yaml_path = str(tmp_path / "extra_paths.yaml")
    with open(yaml_path, "w") as f:
        f.write("")

    load_extra_path_config(yaml_path)

    expected_ckpt = os.path.normpath(str(tmp_path / "models" / "checkpoints"))
    expected_loras = os.path.normpath(str(tmp_path / "models" / "loras"))
    assert expected_ckpt in folder_paths.folder_names_and_paths["checkpoints"][0]
    assert expected_loras in folder_paths.folder_names_and_paths["loras"][0]


@patch("yaml.safe_load")
def test_nested_models_is_default(mock_yaml_load, clear_folder_paths, tmp_path):
    """is_default under models: applies to all model paths in that block."""
    config_data = {
        "comfyui": {
            "models": {
                "base_path": str(tmp_path),
                "is_default": True,
                "checkpoints": "checkpoints/",
            },
        }
    }
    mock_yaml_load.return_value = config_data
    folder_paths.folder_names_and_paths["checkpoints"] = ([], set())

    yaml_path = str(tmp_path / "extra_paths.yaml")
    with open(yaml_path, "w") as f:
        f.write("")

    mock_add = Mock()
    with patch.object(folder_paths, "add_model_folder_path", mock_add):
        load_extra_path_config(yaml_path)

    call = mock_add.call_args_list[0]
    assert call.args[0] == "checkpoints"
    assert call.args[2] is True, "is_default under models: must be passed as True"


@patch("yaml.safe_load")
def test_nested_models_multipath(mock_yaml_load, clear_folder_paths, tmp_path):
    """Multi-line path values inside models: register multiple paths per category."""
    config_data = {
        "comfyui": {
            "models": {
                "base_path": str(tmp_path),
                "text_encoders": "text_encoders/\nclip/",
            },
        }
    }
    mock_yaml_load.return_value = config_data
    folder_paths.folder_names_and_paths["text_encoders"] = ([], set())

    yaml_path = str(tmp_path / "extra_paths.yaml")
    with open(yaml_path, "w") as f:
        f.write("")

    load_extra_path_config(yaml_path)

    registered = folder_paths.folder_names_and_paths["text_encoders"][0]
    assert os.path.normpath(str(tmp_path / "text_encoders")) in registered
    assert os.path.normpath(str(tmp_path / "clip")) in registered


@patch("yaml.safe_load")
def test_nested_models_auto_scan(mock_yaml_load, clear_folder_paths, tmp_path):
    """models: with only base_path auto-scans for known categories that exist on disk."""
    (tmp_path / "models" / "checkpoints").mkdir(parents=True)
    (tmp_path / "models" / "loras").mkdir()

    config_data = {
        "comfyui": {
            "base_path": str(tmp_path),
            "models": {"base_path": "models/"},
        }
    }
    mock_yaml_load.return_value = config_data
    folder_paths.folder_names_and_paths["checkpoints"] = ([], set())
    folder_paths.folder_names_and_paths["loras"] = ([], set())

    yaml_path = str(tmp_path / "extra_paths.yaml")
    with open(yaml_path, "w") as f:
        f.write("")

    load_extra_path_config(yaml_path)

    assert os.path.normpath(str(tmp_path / "models" / "checkpoints")) in \
        folder_paths.folder_names_and_paths["checkpoints"][0]
    assert os.path.normpath(str(tmp_path / "models" / "loras")) in \
        folder_paths.folder_names_and_paths["loras"][0]


@patch("yaml.safe_load")
def test_explicit_custom_nodes_key(mock_yaml_load, clear_folder_paths, tmp_path):
    """Explicit custom_nodes key in a block registers the path via add_model_folder_path."""
    config_data = {
        "comfyui": {
            "base_path": str(tmp_path),
            "custom_nodes": "my_nodes/",
        }
    }
    mock_yaml_load.return_value = config_data
    folder_paths.folder_names_and_paths["custom_nodes"] = ([], set())

    yaml_path = str(tmp_path / "extra_paths.yaml")
    with open(yaml_path, "w") as f:
        f.write("")

    load_extra_path_config(yaml_path)

    assert os.path.normpath(str(tmp_path / "my_nodes")) in \
        folder_paths.folder_names_and_paths["custom_nodes"][0]


@patch("yaml.safe_load")
def test_nested_models_inherits_block_base(mock_yaml_load, clear_folder_paths, tmp_path):
    """models: block without its own base_path inherits the outer block's base_path."""
    config_data = {
        "comfyui": {
            "base_path": str(tmp_path),
            "is_default": True,
            "models": {
                "checkpoints": "models/checkpoints/",
            },
        }
    }
    mock_yaml_load.return_value = config_data
    folder_paths.folder_names_and_paths["checkpoints"] = ([], set())

    yaml_path = str(tmp_path / "extra_paths.yaml")
    with open(yaml_path, "w") as f:
        f.write("")

    load_extra_path_config(yaml_path)

    expected = os.path.normpath(str(tmp_path / "models" / "checkpoints"))
    paths = folder_paths.folder_names_and_paths["checkpoints"][0]
    assert expected in paths
    # is_default inherited: path should be at index 0
    assert paths[0] == expected
