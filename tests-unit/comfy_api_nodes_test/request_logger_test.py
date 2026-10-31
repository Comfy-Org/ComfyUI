from unittest.mock import patch

from comfy_api_nodes.util.request_logger import log_request_response


def test_sensitive_headers_are_redacted(tmp_path):
    with patch("folder_paths.get_temp_directory", return_value=str(tmp_path)):
        log_request_response(
            operation_id="test_operation",
            request_method="GET",
            request_url="https://api.example.com/test",
            request_headers={"Cookie": "session=abc123"},
            response_status_code=200,
            response_headers={
                "Authorization": "Bearer rotated-token",
                "Set-Cookie": "session=def456",
                "Content-Type": "application/json",
            },
        )
    log_file = next((tmp_path / "api_logs").iterdir())
    content = log_file.read_text(encoding="utf-8")
    assert "abc123" not in content
    assert "def456" not in content
    assert "rotated-token" not in content
    assert "application/json" in content
