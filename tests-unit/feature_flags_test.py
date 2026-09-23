"""Tests for feature flags functionality."""

import pytest

from comfy_api.feature_flags import (
    get_connection_feature,
    supports_feature,
    get_server_features,
    CLI_FEATURE_FLAG_REGISTRY,
    SERVER_FEATURE_FLAGS,
    _coerce_flag_value,
    _parse_cli_feature_flags,
)
from comfy.comfy_api_env import (
    environment_overrides_for_base,
    get_environment_overrides,
    normalize_comfy_api_base,
)


class TestFeatureFlags:
    """Test suite for feature flags functions."""

    def test_get_server_features_returns_copy(self):
        """Test that get_server_features returns a copy of the server flags."""
        features = get_server_features()
        # Verify it's a copy by modifying it
        features["test_flag"] = True
        # Original should be unchanged
        assert "test_flag" not in SERVER_FEATURE_FLAGS

    def test_get_server_features_contains_expected_flags(self):
        """Test that server features contain expected flags."""
        features = get_server_features()
        assert "supports_preview_metadata" in features
        assert features["supports_preview_metadata"] is True
        assert "supports_model_type_tags" in features
        assert features["supports_model_type_tags"] is True
        assert "max_upload_size" in features
        assert isinstance(features["max_upload_size"], (int, float))

    def test_get_connection_feature_with_missing_sid(self):
        """Test getting feature for non-existent session ID."""
        sockets_metadata = {}
        result = get_connection_feature(sockets_metadata, "missing_sid", "some_feature")
        assert result is False  # Default value

    def test_get_connection_feature_with_custom_default(self):
        """Test getting feature with custom default value."""
        sockets_metadata = {}
        result = get_connection_feature(
            sockets_metadata, "missing_sid", "some_feature", default="custom_default"
        )
        assert result == "custom_default"

    def test_get_connection_feature_with_feature_flags(self):
        """Test getting feature from connection with feature flags."""
        sockets_metadata = {
            "sid1": {
                "feature_flags": {
                    "supports_preview_metadata": True,
                    "custom_feature": "value",
                },
            }
        }
        result = get_connection_feature(sockets_metadata, "sid1", "supports_preview_metadata")
        assert result is True

        result = get_connection_feature(sockets_metadata, "sid1", "custom_feature")
        assert result == "value"

    def test_get_connection_feature_missing_feature(self):
        """Test getting non-existent feature from connection."""
        sockets_metadata = {
            "sid1": {"feature_flags": {"existing_feature": True}}
        }
        result = get_connection_feature(sockets_metadata, "sid1", "missing_feature")
        assert result is False

    def test_supports_feature_returns_boolean(self):
        """Test that supports_feature always returns boolean."""
        sockets_metadata = {
            "sid1": {
                "feature_flags": {
                    "bool_feature": True,
                    "string_feature": "value",
                    "none_feature": None,
                },
            }
        }

        # True boolean feature
        assert supports_feature(sockets_metadata, "sid1", "bool_feature") is True

        # Non-boolean values should return False
        assert supports_feature(sockets_metadata, "sid1", "string_feature") is False
        assert supports_feature(sockets_metadata, "sid1", "none_feature") is False
        assert supports_feature(sockets_metadata, "sid1", "missing_feature") is False

    def test_supports_feature_with_missing_connection(self):
        """Test supports_feature with missing connection."""
        sockets_metadata = {}
        assert supports_feature(sockets_metadata, "missing_sid", "any_feature") is False

    def test_empty_feature_flags_dict(self):
        """Test connection with empty feature flags dictionary."""
        sockets_metadata = {"sid1": {"feature_flags": {}}}
        result = get_connection_feature(sockets_metadata, "sid1", "any_feature")
        assert result is False
        assert supports_feature(sockets_metadata, "sid1", "any_feature") is False


class TestCoerceFlagValue:
    """Test suite for _coerce_flag_value."""

    def test_registered_bool_true(self):
        assert _coerce_flag_value("show_signin_button", "true") is True
        assert _coerce_flag_value("show_signin_button", "True") is True

    def test_registered_bool_false(self):
        assert _coerce_flag_value("show_signin_button", "false") is False
        assert _coerce_flag_value("show_signin_button", "FALSE") is False

    def test_unregistered_key_stays_string(self):
        assert _coerce_flag_value("unknown_flag", "true") == "true"
        assert _coerce_flag_value("unknown_flag", "42") == "42"

    def test_bool_typo_raises(self):
        """Strict bool: typos like 'ture' or 'yes' must raise so the flag can be dropped."""
        with pytest.raises(ValueError):
            _coerce_flag_value("show_signin_button", "ture")
        with pytest.raises(ValueError):
            _coerce_flag_value("show_signin_button", "yes")
        with pytest.raises(ValueError):
            _coerce_flag_value("show_signin_button", "1")
        with pytest.raises(ValueError):
            _coerce_flag_value("show_signin_button", "")

    def test_failed_int_coercion_raises(self, monkeypatch):
        """Malformed values for typed flags must raise; caller decides what to do."""
        monkeypatch.setitem(
            CLI_FEATURE_FLAG_REGISTRY,
            "test_int_flag",
            {"type": "int", "default": 0, "description": "test"},
        )
        with pytest.raises(ValueError):
            _coerce_flag_value("test_int_flag", "not_a_number")


class TestParseCliFeatureFlags:
    """Test suite for _parse_cli_feature_flags."""

    def test_single_flag(self, monkeypatch):
        monkeypatch.setattr("comfy_api.feature_flags.args", type("Args", (), {"feature_flag": ["show_signin_button=true"]})())
        result = _parse_cli_feature_flags()
        assert result == {"show_signin_button": True}

    def test_missing_equals_defaults_to_true(self, monkeypatch):
        """Bare flag without '=' is treated as the string 'true' (and coerced if registered)."""
        monkeypatch.setattr("comfy_api.feature_flags.args", type("Args", (), {"feature_flag": ["show_signin_button", "valid=1"]})())
        result = _parse_cli_feature_flags()
        assert result == {"show_signin_button": True, "valid": "1"}

    def test_empty_key_skipped(self, monkeypatch):
        monkeypatch.setattr("comfy_api.feature_flags.args", type("Args", (), {"feature_flag": ["=value", "valid=1"]})())
        result = _parse_cli_feature_flags()
        assert result == {"valid": "1"}

    def test_invalid_bool_value_dropped(self, monkeypatch, caplog):
        """A typo'd bool value must be dropped entirely, not silently set to False
        and not stored as a raw string. A warning must be logged."""
        monkeypatch.setattr(
            "comfy_api.feature_flags.args",
            type("Args", (), {"feature_flag": ["show_signin_button=ture", "valid=1"]})(),
        )
        with caplog.at_level("WARNING"):
            result = _parse_cli_feature_flags()
        assert result == {"valid": "1"}
        assert "show_signin_button" not in result
        assert any("show_signin_button" in r.message and "drop" in r.message.lower() for r in caplog.records)


class TestCliFeatureFlagRegistry:
    """Test suite for the CLI feature flag registry."""

    def test_registry_entries_have_required_fields(self):
        for key, info in CLI_FEATURE_FLAG_REGISTRY.items():
            assert "type" in info, f"{key} missing 'type'"
            assert "default" in info, f"{key} missing 'default'"
            assert "description" in info, f"{key} missing 'description'"


class TestComfyApiEnv:
    """--comfy-api-base staging-tier detection + testenv main-host -> -registry rewrite."""

    @pytest.mark.parametrize(
        "url, expected",
        [
            # testenv friendly main host -> comfy-api -registry sibling (slash trimmed)
            ("https://pr-4398.testenvs.comfy.org", "https://pr-4398-registry.testenvs.comfy.org"),
            ("https://pr-4398.testenvs.comfy.org/", "https://pr-4398-registry.testenvs.comfy.org"),
            ("https://pr-4398-registry.testenvs.comfy.org", "https://pr-4398-registry.testenvs.comfy.org"),
            # staging + everything else -> unchanged (no -registry split)
            ("https://stagingapi.comfy.org", "https://stagingapi.comfy.org"),
            ("https://api.comfy.org", "https://api.comfy.org"),
            ("https://pr-1.testenvs.comfy.org.evil.com", "https://pr-1.testenvs.comfy.org.evil.com"),
            ("", ""),
        ],
    )
    def test_normalize_comfy_api_base(self, url, expected):
        assert normalize_comfy_api_base(url) == expected

    def test_config_for_staging_tier_else_none(self):
        # ephemeral testenv: friendly main host -> -registry, staging platform, dev Firebase env
        eph = environment_overrides_for_base("https://pr-1234.testenvs.comfy.org/")
        assert eph["comfy_api_base_url"] == "https://pr-1234-registry.testenvs.comfy.org"
        assert eph["comfy_platform_base_url"] == "https://stagingplatform.comfy.org"
        assert eph["firebase_env"] == "dev"
        # staging api host: emitted as-is
        stg = environment_overrides_for_base("https://stagingapi.comfy.org")
        assert stg["comfy_api_base_url"] == "https://stagingapi.comfy.org"
        assert stg["comfy_platform_base_url"] == "https://stagingplatform.comfy.org"
        assert stg["firebase_env"] == "dev"
        # prod / unknown: nothing
        assert environment_overrides_for_base("https://api.comfy.org") is None

    def test_environment_overrides_only_for_staging_tier(self, monkeypatch):
        def set_base(url):
            monkeypatch.setattr(
                "comfy.comfy_api_env.args",
                type("Args", (), {"comfy_api_base": url})(),
            )

        # The overrides merged into the HTTP /features response are present for staging-tier bases...
        set_base("https://stagingapi.comfy.org")
        assert "comfy_api_base_url" in get_environment_overrides()
        set_base("https://pr-7.testenvs.comfy.org")
        assert "comfy_api_base_url" in get_environment_overrides()
        # ...but never for prod.
        set_base("https://api.comfy.org")
        assert get_environment_overrides() is None

    def test_server_features_never_carry_env_overrides(self, monkeypatch):
        """The WebSocket capability handshake must stay free of routing keys."""
        monkeypatch.setattr(
            "comfy.comfy_api_env.args",
            type("Args", (), {"comfy_api_base": "https://pr-7.testenvs.comfy.org"})(),
        )
        features = get_server_features()
        assert "comfy_api_base_url" not in features
        assert "comfy_platform_base_url" not in features
        assert "firebase_env" not in features
