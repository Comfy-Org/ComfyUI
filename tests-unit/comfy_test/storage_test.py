import os
import stat
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

import comfy.storage


def test_fast_nvme_link_thresholds():
    cases = [
        ("8.0 GT/s PCIe", "4", True),
        ("16.0 GT/s PCIe", "4", True),
        ("32.0 GT/s PCIe", "2", True),
        ("8.0 GT/s PCIe", "2", False),
    ]
    for speed, width, expected in cases:
        with mock.patch.object(comfy.storage, "_read", side_effect=[speed, width]):
            assert comfy.storage._fast_nvme("nvme0n1") is expected


def test_non_nvme_is_not_fast():
    assert comfy.storage._fast_nvme("sda") is False


def test_every_model_file_must_be_on_fast_storage():
    with mock.patch.object(comfy.storage, "fast_storage", side_effect=[True, False]):
        assert comfy.storage.model_fast_disk(["first", "second"]) is False
    with mock.patch.object(comfy.storage, "fast_storage", side_effect=[True, True]):
        assert comfy.storage.model_fast_disk(["first", "second"]) is True


@pytest.mark.skipif(sys.platform == "win32", reason="os.makedev is POSIX only")
def test_virtual_device_number_checks_the_mount_source():
    cases = [
        (os.makedev(0, 63), os.makedev(9, 1)),
        (os.makedev(8, 1), os.makedev(8, 1)),
    ]
    for st_dev, checked in cases:
        with (
            mock.patch.object(comfy.storage.platform, "system", return_value="Linux"),
            mock.patch.object(comfy.storage.os, "stat", return_value=SimpleNamespace(st_dev=st_dev)),
            mock.patch.object(comfy.storage, "_mount_source_device", return_value=os.makedev(9, 1)),
            mock.patch.object(comfy.storage, "_linux_fast_storage", return_value=True) as linux_fast_storage,
        ):
            assert comfy.storage.fast_storage("/models/model.safetensors") is True
        linux_fast_storage.assert_called_once_with(checked)


@pytest.mark.skipif(sys.platform == "win32", reason="os.makedev is POSIX only")
def test_mount_source_device():
    partitions = [
        SimpleNamespace(device=device, mountpoint=mountpoint)
        for device, mountpoint in [
            ("/dev/sda1", "/"),
            ("systemd-1", "/models"),
            ("/dev/nvme0n1p1", "/models"),
            ("/dev/fuse", "/fuse"),
        ]
    ]
    nodes = {
        "/dev/sda1": SimpleNamespace(st_mode=stat.S_IFBLK, st_rdev=os.makedev(8, 1)),
        "/dev/nvme0n1p1": SimpleNamespace(st_mode=stat.S_IFBLK, st_rdev=os.makedev(9, 1)),
        "/dev/fuse": SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=os.makedev(9, 1)),
    }

    def fake_stat(path):
        if path not in nodes:
            raise FileNotFoundError(path)
        return nodes[path]

    cases = [
        ("/models/model.safetensors", os.makedev(9, 1)),
        ("/models2/model.safetensors", os.makedev(8, 1)),
        ("/fuse/model.safetensors", None),
    ]
    with (
        mock.patch.object(comfy.storage.os.path, "realpath", side_effect=lambda p: p),
        mock.patch.object(comfy.storage.psutil, "disk_partitions", return_value=partitions) as disk_partitions,
        mock.patch.object(comfy.storage.os, "stat", side_effect=fake_stat),
    ):
        for path, expected in cases:
            assert comfy.storage._mount_source_device(path) == expected
    disk_partitions.assert_called_with(all=True)


@pytest.mark.skipif(sys.platform == "win32", reason="os.makedev is POSIX only")
def test_multi_device_btrfs_needs_every_device_fast():
    sysfs = {"/sys/fs/btrfs": ["fsid"], "/sys/fs/btrfs/fsid/devices": ["nvme0n1", "sda"]}
    comfy.storage._linux_fast_storage.cache_clear()
    with (
        mock.patch.object(comfy.storage.os, "listdir", side_effect=lambda p: sysfs.get(p, [])),
        mock.patch.object(comfy.storage.os.path, "exists", side_effect=lambda p: p in ("/sys/dev/block/9:1", "/sys/fs/btrfs/fsid/devices/nvme0n1")),
        mock.patch.object(comfy.storage.os.path, "realpath", return_value="/sys/devices/nvme0n1"),
        mock.patch.object(comfy.storage, "_fast_nvme", side_effect=lambda name: name.startswith("nvme")),
    ):
        assert comfy.storage._linux_fast_storage(os.makedev(9, 1)) is False
    comfy.storage._linux_fast_storage.cache_clear()


def test_unreadable_btrfs_devices_is_unknown():
    def listdir(path):
        if path == "/sys/fs/btrfs":
            return ["fsid"]
        raise PermissionError(path)

    with (
        mock.patch.object(comfy.storage.os, "listdir", side_effect=listdir),
        mock.patch.object(comfy.storage.os.path, "exists", return_value=True),
    ):
        assert comfy.storage._btrfs_devices("nvme0n1") is None
