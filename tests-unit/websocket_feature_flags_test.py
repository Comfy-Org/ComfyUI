"""Simplified tests for WebSocket feature flags functionality."""
from comfy_api import feature_flags


class TestWebSocketFeatureFlags:
    """Test suite for WebSocket feature flags integration."""

    def test_server_feature_flags_response(self):
        """Test server feature flags are properly formatted."""
        features = feature_flags.get_server_features()

        # Check expected server features
        assert "supports_preview_metadata" in features
        assert features["supports_preview_metadata"] is True
        assert "supports_model_type_tags" in features
        assert features["supports_model_type_tags"] is True
        assert "max_upload_size" in features
        assert isinstance(features["max_upload_size"], (int, float))

    def test_progress_py_checks_feature_flags(self):
        """Test that progress.py checks feature flags before sending metadata."""
        # This simulates the check in progress.py
        client_id = "test_client"
        sockets_metadata = {"test_client": {"feature_flags": {}}}

        # The actual check would be in progress.py
        supports_metadata = feature_flags.supports_feature(
            sockets_metadata, client_id, "supports_preview_metadata"
        )

        assert supports_metadata is False

    def test_multiple_clients_different_features(self):
        """Test handling multiple clients with different feature support."""
        sockets_metadata = {
            "modern_client": {
                "feature_flags": {"supports_preview_metadata": True}
            },
            "legacy_client": {
                "feature_flags": {}
            }
        }

        # Check modern client
        assert feature_flags.supports_feature(
            sockets_metadata, "modern_client", "supports_preview_metadata"
        ) is True

        # Check legacy client
        assert feature_flags.supports_feature(
            sockets_metadata, "legacy_client", "supports_preview_metadata"
        ) is False

    def test_feature_negotiation_message_format(self):
        """Test the format of feature negotiation messages."""
        # Client message format
        client_message = {
            "type": "feature_flags",
            "data": {
                "supports_preview_metadata": True,
                "api_version": "1.0.0"
            }
        }

        # Verify structure
        assert client_message["type"] == "feature_flags"
        assert "supports_preview_metadata" in client_message["data"]

        # Server response format (what would be sent)
        server_features = feature_flags.get_server_features()
        server_message = {
            "type": "feature_flags",
            "data": server_features
        }

        # Verify structure
        assert server_message["type"] == "feature_flags"
        assert "supports_preview_metadata" in server_message["data"]
        assert server_message["data"]["supports_preview_metadata"] is True
        assert "supports_model_type_tags" in server_message["data"]
        assert server_message["data"]["supports_model_type_tags"] is True
