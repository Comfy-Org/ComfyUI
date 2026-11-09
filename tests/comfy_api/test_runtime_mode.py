import subprocess
import sys
import textwrap

import pytest

from comfy_api._impl import _runtime


def test_the_api_and_its_old_names_import_without_private_comfy_modules() -> None:
    script = textwrap.dedent(
        """
        import builtins

        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            blocked = ("comfy", "folder_paths", "nodes", "server")
            if any(name == root or name.startswith(root + ".") for root in blocked):
                raise ImportError(f"private module is unavailable: {name}")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = guarded_import
        import comfy_api.index
        from comfy_api._impl import legacy_imports
        legacy_imports.install()
        import comfy_api.v0_0_3
        from comfy_api.latest import io
        assert io is comfy_api.index.io
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)


def test_no_provider_configured_is_a_no_op(monkeypatch) -> None:
    monkeypatch.delenv(_runtime.PROVIDER_ENV, raising=False)
    monkeypatch.delenv("COMFY_PACK_ISOLATION", raising=False)  # the built-in pack-isolation provider
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
