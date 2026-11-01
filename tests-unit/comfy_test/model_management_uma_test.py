import builtins
import io
from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args as cli_args

original_cpu_arg = cli_args.cpu
if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_management as model_management  # noqa: E402

cli_args.cpu = original_cpu_arg


MIXED_TOPOLOGY = {
    "card1": {
        "realpath": "/sys/devices/pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:00.0/0000:03:00.0",
        "vram": 17163091968,
        "gtt": 16359088128,
    },
    "card2": {
        "realpath": "/sys/devices/pci0000:00/0000:00:01.2/0000:04:00.0/0000:05:00.0/0000:06:00.0",
        "vram": 17163091968,
        "gtt": 16359088128,
    },
    "card3": {
        "realpath": "/sys/devices/pci0000:00/0000:00:08.1/0000:1d:00.0",
        "vram": 536870912,
        "gtt": 16359088128,
    },
}


def install_fake_drm(monkeypatch, cards):
    files = {}
    realpaths = {}
    for card, values in cards.items():
        device_dir = model_management.os.path.join("/sys/class/drm", card, "device")
        files[model_management.os.path.join(device_dir, "vendor")] = values.get("vendor", "0x1002")
        files[model_management.os.path.join(device_dir, "mem_info_vram_total")] = str(values["vram"])
        files[model_management.os.path.join(device_dir, "mem_info_gtt_total")] = str(values["gtt"])
        realpaths[device_dir] = values["realpath"]

    monkeypatch.setattr(
        model_management.os,
        "listdir",
        lambda path: [*cards, "card1-DP-1"] if path == "/sys/class/drm" else [],
    )
    monkeypatch.setattr(model_management.os.path, "exists", lambda path: path in files)
    monkeypatch.setattr(model_management.os.path, "realpath", lambda path: realpaths.get(path, path))
    monkeypatch.setattr(
        builtins,
        "open",
        lambda path, *args, **kwargs: io.StringIO(files[str(path)]),
    )


def properties(bus, *, integrated, domain=0, device=0):
    return SimpleNamespace(
        pci_domain_id=domain,
        pci_bus_id=bus,
        pci_device_id=device,
        is_integrated=int(integrated),
    )


@pytest.fixture(autouse=True)
def gpu_mode(monkeypatch):
    monkeypatch.setattr(model_management, "cpu_state", model_management.CPUState.GPU)
    monkeypatch.setattr(model_management, "is_amd", lambda: True)
    monkeypatch.setattr(model_management, "is_nvidia", lambda: False)


def test_mixed_topology_maps_each_device_by_canonical_bdf(monkeypatch):
    install_fake_drm(monkeypatch, MIXED_TOPOLOGY)
    device_properties = {
        0: properties(3, integrated=False),
        1: properties(6, integrated=False),
        2: properties(29, integrated=True),
    }
    monkeypatch.setattr(
        model_management.torch.cuda,
        "get_device_properties",
        lambda device: device_properties[device.index],
    )

    assert model_management._amd_vram_gtt_totals(torch.device("cuda", 0)) == (
        17163091968,
        16359088128,
    )
    assert model_management._amd_vram_gtt_totals(torch.device("cuda", 1)) == (
        17163091968,
        16359088128,
    )
    assert model_management._amd_vram_gtt_totals(torch.device("cuda", 2)) == (
        536870912,
        16359088128,
    )
    assert model_management.is_integrated_gpu(torch.device("cuda", 0)) is False
    assert model_management.is_integrated_gpu(torch.device("cuda", 1)) is False
    assert model_management.is_integrated_gpu(torch.device("cuda", 2)) is True
    assert model_management.integrated_gpu_is_shared_heavy(torch.device("cuda", 0)) is False
    assert model_management.integrated_gpu_is_shared_heavy(torch.device("cuda", 1)) is False
    assert model_management.integrated_gpu_is_shared_heavy(torch.device("cuda", 2)) is True


def test_dedicated_heavy_integrated_gpu_stays_non_shared(monkeypatch):
    install_fake_drm(monkeypatch, {"card1": MIXED_TOPOLOGY["card1"]})
    monkeypatch.setattr(
        model_management.torch.cuda,
        "get_device_properties",
        lambda device: properties(3, integrated=True),
    )

    device = torch.device("cuda", 0)
    assert model_management.is_integrated_gpu(device) is True
    assert model_management.integrated_gpu_is_shared_heavy(device) is False


@pytest.mark.parametrize(
    ("vram", "gtt", "expected"),
    [
        (1024, 1023, False),
        (1024, 1024, True),
        (1024, 1025, True),
        (0, 1024, True),
    ],
)
def test_shared_heavy_threshold(monkeypatch, vram, gtt, expected):
    card = {**MIXED_TOPOLOGY["card3"], "vram": vram, "gtt": gtt}
    install_fake_drm(monkeypatch, {"card3": card})
    monkeypatch.setattr(
        model_management.torch.cuda,
        "get_device_properties",
        lambda device: properties(29, integrated=True),
    )

    assert model_management.integrated_gpu_is_shared_heavy(torch.device("cuda", 0)) is expected


def test_full_domain_bus_device_bdf_is_matched(monkeypatch):
    card = {
        **MIXED_TOPOLOGY["card3"],
        "realpath": "/sys/devices/pci1234:00/1234:ab:1f.0",
    }
    install_fake_drm(monkeypatch, {"card3": card})
    monkeypatch.setattr(
        model_management.torch.cuda,
        "get_device_properties",
        lambda device: properties(0xAB, integrated=True, domain=0x1234, device=0x1F),
    )

    assert model_management._amd_vram_gtt_totals(torch.device("cuda", 0)) == (
        536870912,
        16359088128,
    )


def test_multiple_candidates_without_bdf_match_do_not_fall_back(monkeypatch):
    install_fake_drm(monkeypatch, MIXED_TOPOLOGY)
    monkeypatch.setattr(
        model_management.torch.cuda,
        "get_device_properties",
        lambda device: properties(127, integrated=True),
    )

    assert model_management._amd_vram_gtt_totals(torch.device("cuda", 0)) is None


def test_non_amd_drm_candidate_is_ignored(monkeypatch):
    cards = {
        "card1": MIXED_TOPOLOGY["card1"],
        "card2": {
            **MIXED_TOPOLOGY["card2"],
            "vendor": "0x10de",
            "vram": 1,
            "gtt": 2,
        },
    }
    install_fake_drm(monkeypatch, cards)
    monkeypatch.setattr(
        model_management.torch.cuda,
        "get_device_properties",
        lambda device: properties(6, integrated=False),
    )

    assert model_management._amd_vram_gtt_totals(torch.device("cuda", 0)) == (
        17163091968,
        16359088128,
    )


def test_single_candidate_can_fall_back_when_pci_properties_are_unavailable(monkeypatch):
    install_fake_drm(monkeypatch, {"card3": MIXED_TOPOLOGY["card3"]})
    monkeypatch.setattr(
        model_management.torch.cuda,
        "get_device_properties",
        lambda device: (_ for _ in ()).throw(RuntimeError("properties unavailable")),
    )

    assert model_management._amd_vram_gtt_totals(torch.device("cuda", 0)) == (
        536870912,
        16359088128,
    )


def test_unavailable_totals_default_integrated_gpu_to_shared(monkeypatch):
    install_fake_drm(monkeypatch, {})

    assert model_management._amd_vram_gtt_totals(torch.device("cuda", 0)) is None
    assert model_management.integrated_gpu_is_shared_heavy(torch.device("cuda", 0)) is True


def test_non_integer_totals_default_integrated_gpu_to_shared(monkeypatch):
    card = {**MIXED_TOPOLOGY["card3"], "vram": "invalid", "gtt": 16359088128}
    install_fake_drm(monkeypatch, {"card3": card})

    assert model_management._amd_vram_gtt_totals(torch.device("cuda", 0)) is None
    assert model_management.integrated_gpu_is_shared_heavy(torch.device("cuda", 0)) is True


def test_cpu_state_never_reports_integrated_gpu(monkeypatch):
    monkeypatch.setattr(model_management, "cpu_state", model_management.CPUState.CPU)
    monkeypatch.setattr(
        model_management.torch.cuda,
        "get_device_properties",
        lambda device: properties(29, integrated=True),
    )

    assert model_management.is_integrated_gpu(torch.device("cuda", 0)) is False


def test_non_amd_backend_has_no_amd_totals(monkeypatch):
    install_fake_drm(monkeypatch, MIXED_TOPOLOGY)
    monkeypatch.setattr(model_management, "is_amd", lambda: False)

    assert model_management._amd_vram_gtt_totals(torch.device("cuda", 0)) is None
