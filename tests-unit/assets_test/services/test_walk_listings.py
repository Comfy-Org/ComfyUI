"""What walk_listings lists, pinned directly: hidden names, symlinks (including broken,
looping and aliased ones), unreadable folders and entries. (Parity with the os.walk
walker it replaced is checked by a local harness run per OS and filesystem, not here.)"""

import errno
import os
import sys
from pathlib import Path

import pytest

from app.assets.services import file_utils
from app.assets.services.file_utils import walk_listings

needs_symlinks = pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")


def _write(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _names(walk) -> list[str]:
    return sorted(os.path.basename(p) for p in walk.files)


def test_hidden_files_and_folders_are_left_out(temp_dir: Path):
    for rel in (".hidden.png", ".cache/x.png", "sub/.also_hidden.png", "sub/.dir/y.png", "sub/seen.png"):
        _write(temp_dir / rel)
    walk = walk_listings(str(temp_dir))
    assert walk.files == [str(temp_dir / "sub" / "seen.png")]
    assert walk.dirs_listed == 2  # the root and sub; hidden folders are never listed


def test_a_missing_root_or_a_root_that_is_a_file_lists_nothing(temp_dir: Path):
    _write(temp_dir / "file.png")
    for root in (temp_dir / "nowhere", temp_dir / "file.png"):
        walk = walk_listings(str(root))
        assert (walk.files, walk.listings) == ([], {})


@needs_symlinks
def test_symlinks(temp_dir: Path):
    base = temp_dir / "tree"
    outside = temp_dir / "outside"
    _write(base / "a.png")
    _write(base / "sub" / "b.png")
    _write(outside / "o.png")
    (base / "link_to_sub").symlink_to(base / "sub")  # a second path to a folder already walked
    (base / "link_out").symlink_to(outside)  # a folder outside the tree: followed
    (base / "sub" / "loop").symlink_to(base)  # a cycle
    (base / "broken").symlink_to(base / "nowhere")  # gone: left out
    (base / "broken_dir").symlink_to(temp_dir / "gone_dir")
    (base / "file_link.png").symlink_to(base / "a.png")  # a file symlink: listed
    (base / "through_file").symlink_to(base / "a.png" / "x")  # NotADirectoryError: gone
    (base / "loop_a").symlink_to(base / "loop_b")
    (base / "loop_b").symlink_to(base / "loop_a")  # ELOOP: not gone, so kept for the scan's stat
    _write(base / "aaa" / "x.png")
    (base / "aaa" / "link_to_sub").symlink_to(base / "sub")
    walk = walk_listings(str(base))
    # sub's file is listed once, under whichever path reached it first.
    assert _names(walk) == ["a.png", "b.png", "file_link.png", "loop_a", "loop_b", "o.png", "x.png"]
    # base, aaa, sub and outside: the other paths to sub, the loop and the broken links list nothing.
    assert walk.dirs_listed == 4


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions, not root")
def test_folders_that_cant_be_listed_are_skipped(temp_dir: Path):
    _write(temp_dir / "ok" / "a.png")
    _write(temp_dir / "locked" / "b.png")
    _write(temp_dir / "unlistable" / "c.png")
    (temp_dir / "link_into_locked.png").symlink_to(temp_dir / "locked" / "b.png")  # stat: permission denied
    os.chmod(temp_dir / "locked", 0)
    os.chmod(temp_dir / "unlistable", 0o300)  # can enter, can't list
    try:
        walk = walk_listings(str(temp_dir))
        # The link is kept, so the scan's own stat counts it as permission denied.
        assert _names(walk) == ["a.png", "link_into_locked.png"]
    finally:
        os.chmod(temp_dir / "locked", 0o700)
        os.chmod(temp_dir / "unlistable", 0o700)


class _UnreadableEntry:
    """A directory entry whose type can't be read: on a filesystem whose listing has no
    file types, is_dir() and is_symlink() must stat, and on a network share that can fail
    with an I/O error. (A file deleted mid-listing doesn't raise: DirEntry swallows
    FileNotFoundError, and the scan's own stat drops it.)"""

    name = "unreadable.png"

    def __init__(self, dirpath: str) -> None:
        self.path = os.path.join(dirpath, self.name)

    def is_dir(self) -> bool:
        raise OSError(errno.EIO, "I/O error", self.path)

    def is_symlink(self) -> bool:
        raise OSError(errno.EIO, "I/O error", self.path)


class _OsWithExtraEntry:
    """Stands in for file_utils' os module (only there), appending one _UnreadableEntry
    to every listing."""

    def __getattr__(self, name):
        return getattr(os, name)

    def scandir(self, path):
        real = os.scandir(path)

        class _Entries:
            def __enter__(self):
                return iter([*real, _UnreadableEntry(str(path))])

            def __exit__(self, *_exc):
                real.close()

        return _Entries()


def test_an_entry_whose_type_cant_be_read_is_kept_and_the_folder_listed(temp_dir: Path, monkeypatch):
    for name in ("a.png", "b.png"):
        _write(temp_dir / name)
    monkeypatch.setattr(file_utils, "os", _OsWithExtraEntry())
    walk = walk_listings(str(temp_dir))
    assert str(temp_dir) in walk.listings
    # Kept as a file, as os.walk keeps an entry whose is_dir() raises; the scan's stat decides.
    assert sorted(walk.files) == sorted(str(temp_dir / n) for n in ("a.png", "b.png", "unreadable.png"))
