import folder_paths

from app.assets.services.lookup import is_temp_path


def test_temp_content_is_detected_by_its_location():
    assert is_temp_path(f"{folder_paths.get_temp_directory()}/render.png") is True
