
import pytest

from comfy_engine import _runtime


def test_no_provider_configured_is_a_no_op(monkeypatch) -> None:
    monkeypatch.delenv(_runtime.PROVIDER_ENV, raising=False)
    monkeypatch.delenv("OOP_CUSTOM_NODES", raising=False)
    assert not _runtime.load_execution_provider()
    assert _runtime.providers.load_custom_nodes


def test_provider_without_entrypoint_fails(tmp_path) -> None:
    provider = tmp_path / "provider.py"
    provider.write_text("value = 1\n")
    with pytest.raises(RuntimeError, match="register"):
        _runtime.load_execution_provider(str(provider))


def test_provider_can_replace_dispatch(monkeypatch, tmp_path) -> None:
    provider = tmp_path / "provider.py"
    provider.write_text("def register(providers):\n    providers.load_custom_nodes = False\n")
    registry = _runtime.Providers()
    monkeypatch.setattr(_runtime, "providers", registry)
    assert _runtime.load_execution_provider(str(provider))
    assert not registry.load_custom_nodes


def test_frontend_config_is_empty_by_default() -> None:
    assert _runtime.Providers().frontend_runtime_config == {}
