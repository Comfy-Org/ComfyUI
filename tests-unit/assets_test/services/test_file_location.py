import subprocess
import threading
from unittest.mock import Mock

import pytest

from app.assets.services.file_location import (
    build_file_manager_command,
    is_loopback_address,
    reveal_file_in_file_manager,
)


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "127.42.0.9", "::1", "::ffff:127.0.0.1"],
)
def test_accepts_loopback_addresses(address):
    assert is_loopback_address(address)


@pytest.mark.parametrize(
    "address",
    [
        None,
        "",
        "localhost",
        "0.0.0.0",
        "192.168.1.20",
        "::",
        "fe80::1",
        "::ffff:192.168.1.20",
        "fe80::1%eth0",
    ],
)
def test_rejects_non_loopback_addresses(address):
    assert not is_loopback_address(address)


def test_builds_platform_commands(tmp_path):
    source = tmp_path / "folder with spaces" / "render.mp4"
    source.parent.mkdir()
    source.write_bytes(b"video")

    assert build_file_manager_command(str(source), "win32") == [
        "explorer.exe",
        f"/select,{source}",
    ]
    assert build_file_manager_command(str(source), "darwin") == [
        "open",
        "-R",
        str(source),
    ]
    assert build_file_manager_command(str(source), "linux") == [
        "xdg-open",
        str(source.parent),
    ]


def test_reveal_launches_without_a_shell(tmp_path, monkeypatch):
    source = tmp_path / "render.png"
    source.write_bytes(b"image")
    process = Mock()
    popen = Mock(return_value=process)
    reaper = Mock()
    thread = Mock(return_value=reaper)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(threading, "Thread", thread)

    reveal_file_in_file_manager(str(source))

    popen.assert_called_once()
    args, kwargs = popen.call_args
    assert args[0]
    assert kwargs["close_fds"] is True
    assert "shell" not in kwargs
    thread.assert_called_once_with(target=process.wait, daemon=True)
    reaper.start.assert_called_once_with()


def test_reveal_rejects_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        reveal_file_in_file_manager(str(tmp_path / "missing.mp4"))
