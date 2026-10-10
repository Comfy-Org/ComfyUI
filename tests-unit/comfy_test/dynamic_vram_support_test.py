from types import SimpleNamespace

import comfy.model_management as model_management


def test_dynamic_vram_is_not_enabled_by_default_on_gfx1031(monkeypatch):
    monkeypatch.setattr(model_management, "is_nvidia", lambda: False)
    monkeypatch.setattr(model_management, "is_amd", lambda: True)
    monkeypatch.setattr(model_management, "rocm_version", (7, 16), raising=False)
    monkeypatch.setattr(model_management, "get_torch_device", lambda: "cuda:0")
    monkeypatch.setattr(
        model_management.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx1031:sramecc-:xnack-"),
    )

    assert not model_management.dynamic_vram_supported()


def test_dynamic_vram_remains_enabled_by_default_on_other_supported_devices(
    monkeypatch,
):
    monkeypatch.setattr(model_management, "is_nvidia", lambda: False)
    monkeypatch.setattr(model_management, "is_amd", lambda: True)
    monkeypatch.setattr(model_management, "rocm_version", (7, 16), raising=False)
    monkeypatch.setattr(model_management, "get_torch_device", lambda: "cuda:0")
    monkeypatch.setattr(
        model_management.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx1100:sramecc-:xnack-"),
    )

    assert model_management.dynamic_vram_supported()


def test_dynamic_vram_remains_disabled_by_default_on_older_rocm(monkeypatch):
    monkeypatch.setattr(model_management, "is_nvidia", lambda: False)
    monkeypatch.setattr(model_management, "is_amd", lambda: True)
    monkeypatch.setattr(model_management, "rocm_version", (7, 13), raising=False)

    assert not model_management.dynamic_vram_supported()


def test_dynamic_vram_remains_enabled_by_default_on_nvidia(monkeypatch):
    monkeypatch.setattr(model_management, "is_nvidia", lambda: True)
    monkeypatch.setattr(model_management, "is_amd", lambda: False)

    assert model_management.dynamic_vram_supported()
