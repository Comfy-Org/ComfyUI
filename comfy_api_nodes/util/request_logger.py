import datetime
import hashlib
import json
import logging
import os
import re
from typing import Any

import folder_paths

logger = logging.getLogger(__name__)
_SENSITIVE_HEADERS = {"authorization", "x-api-key"}


def get_log_directory():
    """Ensures the API log directory exists within ComfyUI's temp directory and returns its path."""
    base_temp_dir = folder_paths.get_temp_directory()
    log_dir = os.path.join(base_temp_dir, "api_logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        logger.error("Error creating API log directory %s: %s", log_dir, str(e))
        # Fallback to base temp directory if sub-directory creation fails
        return base_temp_dir
    return log_dir


def _sanitize_filename_component(name: str) -> str:
    if not name:
        return "log"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name)  # Replace disallowed characters with underscore
    sanitized = sanitized.strip(" ._")  # Windows: trailing dots or spaces are not allowed
    if not sanitized:
        sanitized = "log"
    return sanitized


def _short_hash(*parts: str, length: int = 10) -> str:
    return hashlib.sha1(("|".join(parts)).encode("utf-8")).hexdigest()[:length]


def _build_log_filepath(log_dir: str, operation_id: str, request_url: str) -> str:
    """Build log filepath. We keep it well under common path length limits aiming for <= 240 characters total."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    slug = _sanitize_filename_component(operation_id)  # Best-effort human-readable slug from operation_id
    h = _short_hash(operation_id or "", request_url or "")  # Short hash ties log to the full operation and URL

    # Compute how much room we have for the slug given the directory length
    # Keep total path length reasonably below ~260 on Windows.
    max_total_path = 240
    prefix = f"{timestamp}_"
    suffix = f"_{h}.log"
    if not slug:
        slug = "op"
    max_filename_len = max(60, max_total_path - len(log_dir) - 1)
    max_slug_len = max(8, max_filename_len - len(prefix) - len(suffix))
    if len(slug) > max_slug_len:
        slug = slug[:max_slug_len].rstrip(" ._-")
    return os.path.join(log_dir, f"{prefix}{slug}{suffix}")


def _format_data_for_logging(data: Any) -> str:
    """Helper to format data (dict, str, bytes) for logging."""
    if isinstance(data, bytes):
        try:
            return data.decode("utf-8")  # Try to decode as text
        except UnicodeDecodeError:
            return f"[Binary data of length {len(data)} bytes]"
    elif isinstance(data, (dict, list)):
        try:
            return json.dumps(data, indent=2, ensure_ascii=False)
        except TypeError:
            return str(data)  # Fallback for non-serializable objects
    return str(data)


def _redact_headers(headers: dict) -> dict:
    return {k: ("***" if k.lower() in _SENSITIVE_HEADERS else v) for k, v in headers.items()}


def log_request_response(
    operation_id: str,
    request_method: str,
    request_url: str,
    request_headers: dict | None = None,
    request_params: dict | None = None,
    request_data: Any = None,
    response_status_code: int | None = None,
    response_headers: dict | None = None,
    response_content: Any = None,
    error_message: str | None = None,
):
    """
    Logs API request and response details to a file in the temp/api_logs directory.
    Filenames are sanitized and length-limited for cross-platform safety.
    If we still fail to write, we fall back to appending into api.log.
    """
    try:
        log_dir = get_log_directory()
        filepath = _build_log_filepath(log_dir, operation_id, request_url)

        log_content: list[str] = []
        log_content.append(f"Timestamp: {datetime.datetime.now().isoformat()}")
        log_content.append(f"Operation ID: {operation_id}")
        log_content.append("-" * 30 + " REQUEST " + "-" * 30)
        log_content.append(f"Method: {request_method}")
        log_content.append(f"URL: {request_url}")
        if request_headers:
            log_content.append(f"Headers:\n{_format_data_for_logging(_redact_headers(request_headers))}")
        if request_params:
            log_content.append(f"Params:\n{_format_data_for_logging(request_params)}")
        if request_data is not None:
            log_content.append(f"Data/Body:\n{_format_data_for_logging(request_data)}")

        log_content.append("\n" + "-" * 30 + " RESPONSE " + "-" * 30)
        if response_status_code is not None:
            log_content.append(f"Status Code: {response_status_code}")
        if response_headers:
            log_content.append(f"Headers:\n{_format_data_for_logging(response_headers)}")
        if response_content is not None:
            log_content.append(f"Content:\n{_format_data_for_logging(response_content)}")
        if error_message:
            log_content.append(f"Error:\n{error_message}")

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(log_content))
            logger.debug("API log saved to: %s", filepath)
        except Exception as e:
            logger.error("Error writing API log to %s: %s", filepath, str(e))
    except Exception as _log_e:
        logging.debug("[DEBUG] log_request_response failed: %s", _log_e)


if __name__ == '__main__':
    # Example usage (for testing the logger directly)
    logger.setLevel(logging.DEBUG)
    # Mock folder_paths for direct execution if not running within ComfyUI full context
    if not hasattr(folder_paths, 'get_temp_directory'):
        class MockFolderPaths:
            def get_temp_directory(self):
                # Create a local temp dir for testing if needed
                p = os.path.join(os.path.dirname(__file__), 'temp_test_logs')
                os.makedirs(p, exist_ok=True)
                return p
        folder_paths = MockFolderPaths()

    log_request_response(
        operation_id="test_operation_get",
        request_method="GET",
        request_url="https://api.example.com/test",
        request_headers={"Authorization": "Bearer testtoken"},
        request_params={"param1": "value1"},
        response_status_code=200,
        response_content={"message": "Success!"}
    )
    log_request_response(
        operation_id="test_operation_post_error",
        request_method="POST",
        request_url="https://api.example.com/submit",
        request_data={"key": "value", "nested": {"num": 123}},
        error_message="Connection timed out"
    )
    log_request_response(
        operation_id="test_binary_response",
        request_method="GET",
        request_url="https://api.example.com/image.png",
        response_status_code=200,
        response_content=b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...' # Sample binary data
    )
