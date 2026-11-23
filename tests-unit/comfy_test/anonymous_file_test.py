import mmap
import os
import sys
import tempfile

import pytest

from comfy.anonymous_file import create_anonymous_file

pytestmark = pytest.mark.skipif(
    os.name != "nt" and sys.platform != "linux",
    reason="anonymous files require Linux or Windows",
)


def test_sparse_file_io_mapping_and_lifetime(tmp_path, monkeypatch):
    directory = tmp_path / "你好-é"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    size = (1 << 32) + 8193
    with create_anonymous_file(size) as file:
        fd = file.fileno()
        assert not os.get_inheritable(fd)
        assert file.tell() == 0
        assert os.fstat(fd).st_size == size
        if os.name == "nt":
            assert os.fstat(fd).st_file_attributes & 0x200  # SPARSE_FILE
            assert os.fstat(fd).st_file_attributes & 0x100  # TEMPORARY
        else:
            assert os.fstat(fd).st_blocks == 0
            assert os.fstat(fd).st_nlink == 0
            assert "memfd:" in os.readlink(f"/proc/self/fd/{fd}")

        file.seek(size - 4)
        assert file.read(4) == bytes(4)
        file.seek(size - 4)
        file.write(b"HIGH")
        file.seek(size - 4)
        assert file.read(4) == b"HIGH"
        file.seek(17)
        file.write(b"\x00\n\r\xffKV")
        os.lseek(fd, 17, os.SEEK_SET)
        assert os.read(fd, 6) == b"\x00\n\r\xffKV"
        with mmap.mmap(fd, 8193, access=mmap.ACCESS_WRITE) as mapping:
            assert mapping[17:23] == b"\x00\n\r\xffKV"
            mapping[4093:4101] = b"mappedKV"
            file.seek(4093)
            assert file.read(8) == b"mappedKV"
            file.seek(8189)
            file.write(b"tail")
            assert mapping[8189:] == b"tail"
        file.truncate(23)
        file.seek(0, os.SEEK_END)
        assert file.tell() == 23
    assert file.closed
    with pytest.raises(OSError):
        os.fstat(fd)
    assert not list(directory.iterdir())


def test_empty_files_are_independent():
    with create_anonymous_file() as first, create_anonymous_file() as second:
        assert first.read() == second.read() == b""
        first.write(b"one")
        second.write(b"two")
        first.seek(0)
        second.seek(0)
        assert first.read() == b"one"
        assert second.read() == b"two"


def test_invalid_size():
    for size in (-1, 1 << 63):
        with pytest.raises(ValueError):
            create_anonymous_file(size)
    with pytest.raises(TypeError):
        create_anonymous_file(1.5)
