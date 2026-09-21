import io
from comfy_api.input_impl.video_types import (
    container_to_output_format,
    get_open_write_kwargs,
    video_encoder_options,
)
from comfy_api.util import VideoCodec, VideoContainer


def test_container_to_output_format_empty_string():
    """Test that an empty string input returns None. `None` arg allows default auto-detection."""
    assert container_to_output_format("") is None


def test_container_to_output_format_none():
    """Test that None input returns None."""
    assert container_to_output_format(None) is None


def test_container_to_output_format_comma_separated():
    """Test that a comma-separated list returns a valid singular format from the list."""
    comma_separated_format = "mp4,mov,m4a"
    output_format = container_to_output_format(comma_separated_format)
    assert output_format in comma_separated_format


def test_container_to_output_format_single():
    """Test that a single format string (not comma-separated list) is returned as is."""
    assert container_to_output_format("mp4") == "mp4"


def test_get_open_write_kwargs_filepath_no_format():
    """Test that 'format' kwarg is NOT set when dest is a file path."""
    kwargs_auto = get_open_write_kwargs("output.mp4", "mp4", VideoContainer.AUTO)
    assert "format" not in kwargs_auto, "Format should not be set for file paths (AUTO)"

    kwargs_specific = get_open_write_kwargs("output.avi", "mp4", "avi")
    fail_msg = "Format should not be set for file paths (Specific)"
    assert "format" not in kwargs_specific, fail_msg
    assert "options" not in kwargs_specific


def test_get_open_write_kwargs_base_options_mode():
    """Test basic kwargs for file path: mode and movflags."""
    kwargs = get_open_write_kwargs("output.mp4", "mp4", VideoContainer.AUTO)
    assert kwargs["mode"] == "w", "mode should be set to write"

    fail_msg = "movflags should preserve custom metadata tags and enable faststart for MP4 files"
    assert "movflags" in kwargs["options"], fail_msg
    assert kwargs["options"]["movflags"] == "use_metadata_tags+faststart", fail_msg


def test_get_open_write_kwargs_bytesio_auto_format():
    """Test kwargs for BytesIO dest with AUTO format."""
    dest = io.BytesIO()
    container_fmt = "mov,mp4,m4a"
    kwargs = get_open_write_kwargs(dest, container_fmt, VideoContainer.AUTO)

    assert kwargs["mode"] == "w"
    assert kwargs["options"]["movflags"] == "use_metadata_tags"

    fail_msg = (
        "Format should be a valid format from the container's format list when AUTO"
    )
    assert kwargs["format"] in container_fmt, fail_msg


def test_get_open_write_kwargs_bytesio_specific_format():
    """Test kwargs for BytesIO dest with a specific single format."""
    dest = io.BytesIO()
    container_fmt = "avi"
    to_fmt = VideoContainer.MP4
    kwargs = get_open_write_kwargs(dest, container_fmt, to_fmt)

    assert kwargs["mode"] == "w"
    assert kwargs["options"]["movflags"] == "use_metadata_tags"

    fail_msg = "Format should be the specified format (lowercased) when output format is not AUTO"
    assert kwargs["format"] == "mp4", fail_msg


def test_get_open_write_kwargs_bytesio_specific_format_list():
    """Test kwargs for BytesIO dest with a specific comma-separated format."""
    dest = io.BytesIO()
    container_fmt = "avi"
    to_fmt = "mov,mp4,m4a"  # A format string that is a list
    kwargs = get_open_write_kwargs(dest, container_fmt, to_fmt)

    assert kwargs["mode"] == "w"
    assert kwargs["options"]["movflags"] == "use_metadata_tags"

    fail_msg = "Format should be a valid format from the specified format list when output format is not AUTO"
    assert kwargs["format"] in to_fmt, fail_msg


def test_get_open_write_kwargs_does_not_pass_movflags_to_matroska_or_webm():
    for format, suffix in ((VideoContainer.MKV, "mkv"), (VideoContainer.WEBM, "webm")):
        assert "options" not in get_open_write_kwargs(f"output.{suffix}", "mp4", format)
        assert "options" not in get_open_write_kwargs(io.BytesIO(), "mp4", format)


def test_av1_zero_crf_uses_lossless_mode():
    assert video_encoder_options(VideoCodec.AV1, 0) == {"svtav1-params": "lossless=1"}
    assert video_encoder_options(VideoCodec.AV1, 30.0) == {"crf": "30.0"}
    assert video_encoder_options(VideoCodec.H264, 0) == {"crf": "0"}
    assert video_encoder_options(VideoCodec.AV1, None) == {}
