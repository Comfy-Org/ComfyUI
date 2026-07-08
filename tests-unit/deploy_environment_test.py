"""Tests for comfy.deploy_environment."""

import os

import pytest

from comfy import deploy_environment
from comfy.deploy_environment import get_deploy_environment


@pytest.fixture(autouse=True)
def _reset_cache_and_install_dir(tmp_path, monkeypatch):
    """Reset the functools cache and point the ComfyUI install dir at a tmp dir for each test."""
    get_deploy_environment.cache_clear()
    monkeypatch.setattr(deploy_environment, "_COMFY_INSTALL_DIR", str(tmp_path))
    yield
    get_deploy_environment.cache_clear()


def _write_env_file(tmp_path, content: str) -> str:
    """Write the env file with exact content (no newline translation).

    `newline=""` disables Python's text-mode newline translation so the bytes
    on disk match the literal string passed in, regardless of host OS.
    Newline-style tests (CRLF, lone CR) rely on this.
    """
    path = os.path.join(str(tmp_path), ".comfy_environment")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    return path


class TestGetDeployEnvironment:
    def test_returns_local_git_when_file_missing(self):
        assert get_deploy_environment() == "local-git"

    def test_reads_value_from_file(self, tmp_path):
        _write_env_file(tmp_path, "local-desktop2-standalone\n")
        assert get_deploy_environment() == "local-desktop2-standalone"

    def test_strips_trailing_whitespace_and_newline(self, tmp_path):
        _write_env_file(tmp_path, "  local-desktop2-standalone  \n")
        assert get_deploy_environment() == "local-desktop2-standalone"

    def test_only_first_line_is_used(self, tmp_path):
        _write_env_file(tmp_path, "first-line\nsecond-line\n")
        assert get_deploy_environment() == "first-line"

    def test_crlf_line_ending(self, tmp_path):
        # Windows editors often save text files with CRLF line endings.
        # The CR must not end up in the returned value.
        _write_env_file(tmp_path, "local-desktop2-standalone\r\n")
        assert get_deploy_environment() == "local-desktop2-standalone"

    def test_crlf_multiline_only_first_line_used(self, tmp_path):
        _write_env_file(tmp_path, "first-line\r\nsecond-line\r\n")
        assert get_deploy_environment() == "first-line"

    def test_crlf_with_surrounding_whitespace(self, tmp_path):
        _write_env_file(tmp_path, "  local-desktop2-standalone  \r\n")
        assert get_deploy_environment() == "local-desktop2-standalone"

    def test_lone_cr_line_ending(self, tmp_path):
        # Classic-Mac / some legacy editors use a bare CR.
        # Universal-newlines decoding treats it as a line terminator too.
        _write_env_file(tmp_path, "local-desktop2-standalone\r")
        assert get_deploy_environment() == "local-desktop2-standalone"

    def test_empty_file_falls_back_to_default(self, tmp_path):
        _write_env_file(tmp_path, "")
        assert get_deploy_environment() == "local-git"

    def test_empty_after_whitespace_strip_falls_back_to_default(self, tmp_path):
        _write_env_file(tmp_path, "   \n")
        assert get_deploy_environment() == "local-git"

    def test_strips_control_chars_within_first_line(self, tmp_path):
        # Embedded NUL/control chars in the value should be stripped
        # (header-injection / smuggling protection).
        _write_env_file(tmp_path, "abc\x00\x07xyz\n")
        assert get_deploy_environment() == "abcxyz"

    def test_strips_non_ascii_characters(self, tmp_path):
        _write_env_file(tmp_path, "café-é\n")
        assert get_deploy_environment() == "caf-"

    def test_caps_read_at_128_bytes(self, tmp_path):
        # A single huge line with no newline must not be fully read into memory.
        huge = "x" * 10_000
        _write_env_file(tmp_path, huge)
        result = get_deploy_environment()
        assert result == "x" * 128

    def test_result_is_cached_across_calls(self, tmp_path):
        path = _write_env_file(tmp_path, "first_value\n")
        assert get_deploy_environment() == "first_value"
        # Overwrite the file — cached value should still be returned.
        with open(path, "w", encoding="utf-8") as f:
            f.write("second_value\n")
        assert get_deploy_environment() == "first_value"

    def test_unreadable_file_falls_back_to_default(self, tmp_path, monkeypatch):
        _write_env_file(tmp_path, "should_not_be_used\n")

        def _boom(*args, **kwargs):
            raise OSError("simulated read failure")

        monkeypatch.setattr("builtins.open", _boom)
        assert get_deploy_environment() == "local-git"
