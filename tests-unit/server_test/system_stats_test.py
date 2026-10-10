from types import SimpleNamespace

import server


def test_system_stats_omits_effective_path_configuration(monkeypatch):
    device = SimpleNamespace(type="cuda", index=0)
    monkeypatch.setattr(server.comfy.model_management, "get_torch_device", lambda: device)
    monkeypatch.setattr(server.comfy.model_management, "get_all_torch_devices", lambda: [device])
    monkeypatch.setattr(server.comfy.model_management, "get_torch_device_name", lambda _: "Test GPU")
    monkeypatch.setattr(server.comfy.model_management, "get_total_memory", lambda _, torch_total_too=False: (100, 90) if torch_total_too else 200)
    monkeypatch.setattr(server.comfy.model_management, "get_free_memory", lambda _, torch_free_too=False: (80, 70) if torch_free_too else 160)
    monkeypatch.setattr(server.comfy.model_management, "torch", SimpleNamespace(device=lambda _: "cpu"))
    monkeypatch.setattr(server.comfy.model_management, "torch_version", "test-torch")
    monkeypatch.setattr(server.FrontendManager, "get_required_frontend_version", lambda: "required")
    monkeypatch.setattr(server.FrontendManager, "get_installed_templates_version", lambda: "installed")
    monkeypatch.setattr(server.FrontendManager, "get_required_templates_version", lambda: "templates")
    monkeypatch.setattr(server.FrontendManager, "get_comfy_package_versions", lambda: {})
    monkeypatch.setattr(server, "get_deploy_environment", lambda: "test")
    monkeypatch.setattr(server.os, "getcwd", lambda: "F:/ComfyUI/current")
    monkeypatch.setattr(server.folder_paths, "base_path", "F:/ComfyUI/base")
    monkeypatch.setattr(server.folder_paths, "get_input_directory", lambda: "F:/ComfyUI/input")
    monkeypatch.setattr(server.folder_paths, "get_output_directory", lambda: "F:/ComfyUI/_Output")
    monkeypatch.setattr(server.folder_paths, "get_temp_directory", lambda: "F:/ComfyUI/temp")
    monkeypatch.setattr(server.folder_paths, "get_user_directory", lambda: "F:/ComfyUI/user")

    payload = server.get_system_stats()

    assert {"cwd", "base_path", "input_directory", "output_directory", "temp_directory", "user_directory"}.isdisjoint(payload["system"])
    assert payload["system"]["ram_total"] == 200
    assert payload["system"]["ram_free"] == 160
    assert payload["system"]["pytorch_version"] == "test-torch"
    assert payload["devices"] == [{
        "name": "Test GPU",
        "type": "cuda",
        "index": 0,
        "vram_total": 100,
        "vram_free": 80,
        "torch_vram_total": 90,
        "torch_vram_free": 70,
    }]
