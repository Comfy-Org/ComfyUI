"""walk_listings replaced the os.walk walker for every scan, so on every tree shape it must
find the same files in the same order. The one intended difference: a symlink whose target
is gone is left out (the os.walk walker listed it, then the scan's stat dropped it)."""

import os
import sys
from pathlib import Path

import pytest

from app.assets.services.file_utils import walk_listings

from ..os_walk_reference import list_files_with_os_walk

needs_symlinks = pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")


def _write(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _assert_parity(base: Path) -> None:
    expected = [p for p in list_files_with_os_walk(str(base)) if os.path.exists(p) or not os.path.islink(p)]
    walk = walk_listings(str(base))
    assert walk.files == expected
    assert walk.dirs_listed == len(walk.listings)


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
    _assert_parity(temp_dir)


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
    _assert_parity(base)


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
    os.chmod(temp_dir / "locked", 0)
    os.chmod(temp_dir / "unlistable", 0o300)  # can enter, can't list
    try:
        _assert_parity(temp_dir)
    finally:
        os.chmod(temp_dir / "locked", 0o700)
        os.chmod(temp_dir / "unlistable", 0o700)
