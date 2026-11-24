from unittest import mock

import comfy.storage
import comfy.system_memory


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
    with mock.patch.object(comfy.storage, "fast_storage", side_effect=[True, True]), mock.patch.object(comfy.storage.os.path, "getsize", return_value=10):
        with mock.patch.object(comfy.system_memory, "virtual_memory_available", return_value=20):
            assert comfy.storage.model_fast_disk(["first", "second"]) is True


def test_fast_disk_auto_detection_prefers_ram_when_model_fits():
    with mock.patch.object(comfy.storage.args, "fast_disk", False), mock.patch.object(comfy.storage.args, "disable_fast_disk", False):
        with mock.patch.object(comfy.storage, "fast_storage", return_value=True), mock.patch.object(comfy.storage.os.path, "getsize", return_value=17):
            with mock.patch.object(comfy.system_memory, "virtual_memory_available", return_value=45):
                assert comfy.storage.model_fast_disk(["model"]) is False
            with mock.patch.object(comfy.system_memory, "virtual_memory_available", return_value=20):
                assert comfy.storage.model_fast_disk(["model"]) is True
            with mock.patch.object(comfy.storage.args, "fast_disk", True):
                assert comfy.storage.model_fast_disk(["model"]) is True
                with mock.patch.object(comfy.storage.args, "disable_fast_disk", True):
                    assert comfy.storage.model_fast_disk(["model"]) is False
