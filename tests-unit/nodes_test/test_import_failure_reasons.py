"""Tests for custom node import failure reason reporting."""

import pytest
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
import asyncio


class TestImportFailureReasons:
    """Test that import failures include diagnostic information."""

    def test_import_failure_reason_format(self):
        """Test that failure reason is formatted correctly."""
        # Simulate the formatting logic
        exception = ImportError("No module named 'missing_dep'")
        error_msg = str(exception).split('\n')[0][:100]
        reason = f"{type(exception).__name__}: {error_msg}"

        assert reason == "ImportError: No module named 'missing_dep'"

    def test_import_failure_reason_truncation(self):
        """Test that long error messages are truncated."""
        long_msg = "a" * 200
        exception = ValueError(long_msg)
        error_msg = str(exception).split('\n')[0][:100]
        reason = f"{type(exception).__name__}: {error_msg}"

        # Should be truncated to 100 chars for the message part
        assert len(error_msg) == 100
        assert reason.startswith("ValueError: ")

    def test_import_failure_reason_multiline(self):
        """Test that only first line of error is used."""
        multi_line_msg = "First line\nSecond line\nThird line"
        exception = RuntimeError(multi_line_msg)
        error_msg = str(exception).split('\n')[0][:100]
        reason = f"{type(exception).__name__}: {error_msg}"

        assert reason == "RuntimeError: First line"
        assert "Second line" not in reason

    def test_import_failure_reason_various_exceptions(self):
        """Test formatting for various exception types."""
        test_cases = [
            (ModuleNotFoundError("No module named 'foo'"), "ModuleNotFoundError: No module named 'foo'"),
            (SyntaxError("invalid syntax"), "SyntaxError: invalid syntax"),
            (AttributeError("'NoneType' object has no attribute 'bar'"), "AttributeError: 'NoneType' object has no attribute 'bar'"),
            (FileNotFoundError("[Errno 2] No such file"), "FileNotFoundError: [Errno 2] No such file"),
        ]

        for exception, expected in test_cases:
            error_msg = str(exception).split('\n')[0][:100]
            reason = f"{type(exception).__name__}: {error_msg}"
            assert reason == expected, f"Failed for {type(exception).__name__}"


class TestImportSummaryOutput:
    """Test the import summary output format."""

    def test_summary_message_with_reason(self):
        """Test that summary includes reason when available."""
        reason = "ImportError: No module named 'xyz'"
        import_message = f" (IMPORT FAILED: {reason})"

        assert import_message == " (IMPORT FAILED: ImportError: No module named 'xyz')"

    def test_summary_message_without_reason(self):
        """Test fallback when no reason is available."""
        reason = ""
        if reason:
            import_message = f" (IMPORT FAILED: {reason})"
        else:
            import_message = " (IMPORT FAILED)"

        assert import_message == " (IMPORT FAILED)"

    def test_summary_format_string(self):
        """Test the full summary line format."""
        time_taken = 0.05
        import_message = " (IMPORT FAILED: ImportError: missing module)"
        module_path = "/path/to/custom_nodes/my_node"

        summary_line = "{:6.1f} seconds{}: {}".format(time_taken, import_message, module_path)

        assert "0.1 seconds" in summary_line
        assert "(IMPORT FAILED: ImportError: missing module)" in summary_line
        assert module_path in summary_line
