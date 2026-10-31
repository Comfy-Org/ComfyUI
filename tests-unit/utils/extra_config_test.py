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


@pytest.mark.parametrize("yaml_content", [
    "",
    "\n  \n\n",
    "# every line is a comment\n# so there are no nodes to parse\n",
])
def test_load_extra_path_config_empty_yaml(yaml_content, clear_folder_paths, tmp_path):
    """
    Test that a config with no YAML nodes is a no-op. An empty file, a whitespace-only file,
    and a file of only comments all parse to None rather than to an empty dict, so the loader
    has to guard against that before iterating.
    """
    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    load_extra_path_config(str(yaml_path))

    assert folder_paths.folder_names_and_paths == {}
