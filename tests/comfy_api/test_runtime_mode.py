import subprocess
import sys
import textwrap

import pytest

from comfy_api.latest import _sdk


def test_versioned_v2_api_imports_without_private_comfy_modules() -> None:
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
        import comfy_api.v0_0_3
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)


def test_local_mode_allows_legacy_custom_nodes() -> None:
    assert _sdk.should_load_legacy_custom_nodes(
        secure_mode=False,
        disabled=False,
        has_whitelist=False,
    )


def test_local_mode_honors_legacy_custom_node_controls() -> None:
    assert not _sdk.should_load_legacy_custom_nodes(
        secure_mode=False,
        disabled=True,
        has_whitelist=False,
    )
    assert _sdk.should_load_legacy_custom_nodes(
        secure_mode=False,
        disabled=True,
        has_whitelist=True,
    )


@pytest.mark.parametrize("has_whitelist", [False, True])
def test_secure_mode_never_loads_legacy_custom_nodes(
    has_whitelist: bool,
) -> None:
    assert not _sdk.should_load_legacy_custom_nodes(
        secure_mode=True,
        disabled=False,
        has_whitelist=has_whitelist,
    )


def test_unconfigured_overlay_preserves_local_mode(monkeypatch) -> None:
    monkeypatch.delenv(_sdk.OVERLAY_ENV, raising=False)

    assert not _sdk.load_overlay()


def test_configured_overlay_without_entrypoint_fails_closed(tmp_path) -> None:
    overlay = tmp_path / "broken_overlay.py"
    overlay.write_text("VALUE = 1\n")

    with pytest.raises(RuntimeError, match="register\\(providers\\)"):
        _sdk.load_overlay(str(overlay))


def test_configured_overlay_registers_secure_providers(
    tmp_path,
    monkeypatch,
) -> None:
    overlay = tmp_path / "working_overlay.py"
    overlay.write_text(
        textwrap.dedent(
            """
            def register(providers):
                providers.registered_by_test = True
            """
        )
    )
    providers = _sdk.Providers()
    monkeypatch.setattr(_sdk, "providers", providers)

    assert _sdk.load_overlay(str(overlay))
    assert providers.overlay_active
    assert providers.registered_by_test


def test_extension_host_configuration_is_absent_in_local_mode() -> None:
    assert _sdk.Providers().frontend_runtime_config == {}


def test_overlay_can_publish_its_frontend_host_module() -> None:
    providers = _sdk.Providers()

    providers.register_extension_host("/isolated-host/entry.mjs")

    assert providers.frontend_runtime_config == {
        "extension_host": {"module_url": "/isolated-host/entry.mjs"}
    }
