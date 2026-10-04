"""walk_listings replaced the os.walk walker for every scan, so on every tree shape it must
find the same files in the same order. The one intended difference: a symlink whose target
is gone is left out (the os.walk walker listed it, then the scan's stat dropped it)."""

import os
import sys
from pathlib import Path

import pytest

from app.assets.services import file_utils
from app.assets.services.file_utils import walk_listings

from ..os_walk_reference import list_files_with_os_walk

needs_symlinks = pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")


def _write(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _dangling(path: str) -> bool:
    if not os.path.islink(path):
        return False
    try:
        os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return True
    except OSError:
        return False
    return False


def _assert_parity(base: Path):
    expected = [p for p in list_files_with_os_walk(str(base)) if not _dangling(p)]
    walk = walk_listings(str(base))
    assert walk.files == expected
    return walk


def test_nested_tree(temp_dir: Path):
    for rel in ("a.png", "sub/b.png", "sub/deeper/c.png", "other/d.txt", "sub/e.safetensors"):
        _write(temp_dir / rel)
    _assert_parity(temp_dir)


def test_deep_tree(temp_dir: Path):
    path = temp_dir
    for depth in range(40):
        path = path / f"level{depth}"
        _write(path / f"f{depth}.png")
    _assert_parity(temp_dir)


def test_flat_folder(temp_dir: Path):
    for i in range(3000):
        (temp_dir / f"ComfyUI_{i:05d}_.png").write_bytes(b"x")
    _assert_parity(temp_dir)


def test_hidden_files_and_folders(temp_dir: Path):
    for rel in (".hidden.png", ".cache/x.png", "sub/.also_hidden.png", "sub/.dir/y.png", "sub/seen.png"):
        _write(temp_dir / rel)
    assert _assert_parity(temp_dir).dirs_listed == 2  # the root and sub; hidden folders are never listed


def test_empty_files_partial_downloads_and_odd_names(temp_dir: Path):
    for rel in ("empty.png", "dl.safetensors.part", "with space.png", "ünïcödé.png", "sub dir/x y.png"):
        _write(temp_dir / rel, b"" if rel == "empty.png" else b"x")
    _assert_parity(temp_dir)


def test_missing_root_and_root_that_is_a_file(temp_dir: Path):
    _assert_parity(temp_dir / "nowhere")
    _write(temp_dir / "file.png")
    _assert_parity(temp_dir / "file.png")


@needs_symlinks
def test_symlinks(temp_dir: Path):
    base = temp_dir / "tree"
    outside = temp_dir / "outside"
    _write(base / "a.png")
    _write(base / "sub" / "b.png")
    _write(outside / "o.png")
    (base / "link_to_sub").symlink_to(base / "sub")  # a second path to a folder already walked
    (base / "link_out").symlink_to(outside)  # a folder outside the tree
    (base / "sub" / "loop").symlink_to(base)  # a cycle
    (base / "broken").symlink_to(base / "nowhere")
    (base / "broken_dir").symlink_to(temp_dir / "gone_dir")
    (base / "file_link.png").symlink_to(base / "a.png")
    (base / "through_file").symlink_to(base / "a.png" / "x")  # NotADirectoryError: gone
    (base / "loop_a").symlink_to(base / "loop_b")
    (base / "loop_b").symlink_to(base / "loop_a")  # ELOOP: not gone, so kept for the scan's stat
    _write(base / "aaa" / "x.png")
    (base / "aaa" / "link_to_sub").symlink_to(base / "sub")  # which path reaches sub first must not change
    # base, aaa, sub and outside: the other paths to sub, the loop and the broken link list nothing
    walk = _assert_parity(base)
    assert walk.dirs_listed == 4
    assert str(base / "through_file") not in walk.files
    assert {str(base / "loop_a"), str(base / "loop_b")} <= set(walk.files)


@needs_symlinks
def test_symlinked_root(temp_dir: Path):
    _write(temp_dir / "real" / "a.png")
    (temp_dir / "alias").symlink_to(temp_dir / "real")
    _assert_parity(temp_dir / "alias")


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions, not root")
def test_permission_denied_folder(temp_dir: Path):
    _write(temp_dir / "ok" / "a.png")
    _write(temp_dir / "locked" / "b.png")
    _write(temp_dir / "unlistable" / "c.png")
    (temp_dir / "link_into_locked.png").symlink_to(temp_dir / "locked" / "b.png")  # stat: permission denied
    os.chmod(temp_dir / "locked", 0)
    os.chmod(temp_dir / "unlistable", 0o300)  # can enter, can't list
    try:
        walk = _assert_parity(temp_dir)
        # Kept, so the scan's own stat counts it as permission denied, as the os.walk walker did.
        assert str(temp_dir / "link_into_locked.png") in walk.files
    finally:
        os.chmod(temp_dir / "locked", 0o700)
        os.chmod(temp_dir / "unlistable", 0o700)


class _VanishingEntry:
    """A directory entry whose file was deleted after the listing named it, on a
    filesystem whose listing has no file types, so is_dir() and is_symlink() must stat."""

    name = "vanished.png"

    def __init__(self, dirpath: str) -> None:
        self.path = os.path.join(dirpath, self.name)

    def is_dir(self) -> bool:
        raise FileNotFoundError(self.path)

    def is_symlink(self) -> bool:
        raise FileNotFoundError(self.path)


def test_an_entry_vanishing_mid_listing_keeps_the_rest_of_the_folder(temp_dir: Path, monkeypatch):
    for name in ("a.png", "b.png"):
        _write(temp_dir / name)
    real_scandir = os.scandir

    class _Entries:
        def __init__(self, path):
            self._real = real_scandir(path)
            self._items = [*self._real, _VanishingEntry(str(path))]

        def __enter__(self):
            return iter(self._items)

        def __exit__(self, *_exc):
            self._real.close()

    monkeypatch.setattr(file_utils.os, "scandir", _Entries)
    walk = walk_listings(str(temp_dir))
    assert str(temp_dir) in walk.listings
    assert sorted(walk.files) == [str(temp_dir / "a.png"), str(temp_dir / "b.png")]
