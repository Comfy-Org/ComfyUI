import textwrap

import pytest

from comfy_api.latest import _sdk


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
