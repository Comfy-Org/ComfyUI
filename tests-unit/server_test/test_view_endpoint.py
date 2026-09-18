"""Tests for /view endpoint path traversal validation"""

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
import os
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath


class TestViewEndpointSecurity(AioHTTPTestCase):
    """Test /view endpoint path traversal validation"""

    async def get_application(self):
        """Create a minimal test application with /view endpoint"""
        app = web.Application()

        # Create a test directory with test files
        self.test_dir = tempfile.mkdtemp()
        self.test_file_valid = os.path.join(self.test_dir, "test..png")
        self.test_file_normal = os.path.join(self.test_dir, "normal.png")
        self.test_file_dotsecret = os.path.join(self.test_dir, "..secret")

        # Create test files
        Path(self.test_file_valid).touch()
        Path(self.test_file_normal).touch()
        Path(self.test_file_dotsecret).touch()

        async def view_handler(request):
            """Simplified /view endpoint handler for testing"""
            if "filename" not in request.rel_url.query:
                return web.Response(status=404)

            filename = request.rel_url.query["filename"]

            if not filename:
                return web.Response(status=400)

            # validation for security: prevent accessing arbitrary path
            # Normalize backslashes and use standard library to parse path components
            normalized = filename.replace('\\', '/')
            path = PurePosixPath(normalized)
            win_path = PureWindowsPath(normalized)
            if path.is_absolute() or win_path.is_absolute() or win_path.drive or '..' in path.parts:
                return web.Response(status=400)

            # For testing, just check if file exists in test directory
            filename = os.path.basename(filename)
            file_path = os.path.join(self.test_dir, filename)

            if os.path.isfile(file_path):
                return web.Response(status=200, text="OK")

            return web.Response(status=404)

        app.router.add_get('/view', view_handler)
        return app

    async def tearDownAsync(self):
        """Clean up test files"""
        import shutil
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @unittest_run_loop
    async def test_allows_consecutive_dots_in_filename(self):
        """Test that files with consecutive dots (e.g., test..png) are allowed"""
        resp = await self.client.request("GET", "/view?filename=test..png")
        assert resp.status == 200, "Should allow files with consecutive dots in filename"

    @unittest_run_loop
    async def test_allows_normal_filenames(self):
        """Test that normal filenames work"""
        resp = await self.client.request("GET", "/view?filename=normal.png")
        assert resp.status == 200, "Should allow normal filenames"

    @unittest_run_loop
    async def test_blocks_path_traversal_dotdot_slash(self):
        """Test that path traversal with ../ is blocked"""
        resp = await self.client.request("GET", "/view?filename=../etc/passwd")
        assert resp.status == 400, "Should block ../ path traversal"

    @unittest_run_loop
    async def test_blocks_path_traversal_slash_dotdot(self):
        """Test that path traversal with /.. is blocked"""
        resp = await self.client.request("GET", "/view?filename=folder/../etc/passwd")
        assert resp.status == 400, "Should block /.. path traversal"

    @unittest_run_loop
    async def test_blocks_absolute_paths(self):
        """Test that absolute paths starting with / are blocked"""
        resp = await self.client.request("GET", "/view?filename=/etc/passwd")
        assert resp.status == 400, "Should block absolute paths"

    @unittest_run_loop
    async def test_allows_dotdot_prefix_in_filename(self):
        """Test that filenames starting with .. but not as a path component are allowed"""
        resp = await self.client.request("GET", "/view?filename=..secret")
        assert resp.status == 200, "Should allow filenames starting with .. that aren't path traversal"

    @unittest_run_loop
    async def test_blocks_bare_dotdot(self):
        """Test that bare .. as a path component is blocked"""
        resp = await self.client.request("GET", "/view?filename=..")
        assert resp.status == 400, "Should block bare .. path component"

    @unittest_run_loop
    async def test_multiple_consecutive_dots(self):
        """Test that multiple consecutive dots in filename are allowed"""
        # Create a file with multiple dots
        test_file = os.path.join(self.test_dir, "test...png")
        Path(test_file).touch()

        resp = await self.client.request("GET", "/view?filename=test...png")
        assert resp.status == 200, "Should allow files with multiple consecutive dots"

    @unittest_run_loop
    async def test_dots_in_middle_of_filename(self):
        """Test that dots in the middle of filename are allowed"""
        # Create a file with dots in middle
        test_file = os.path.join(self.test_dir, "my..file..name.png")
        Path(test_file).touch()

        resp = await self.client.request("GET", "/view?filename=my..file..name.png")
        assert resp.status == 200, "Should allow dots in middle of filename"

    @unittest_run_loop
    async def test_blocks_backslash_path_traversal(self):
        """Test that Windows-style backslash path traversal (\\..) is blocked"""
        resp = await self.client.request("GET", "/view?filename=folder%5C..%5Csecret")
        assert resp.status == 400, "Should block backslash path traversal"

    @unittest_run_loop
    async def test_blocks_backslash_dotdot_at_start(self):
        """Test that backslash path traversal starting with ..\\ is blocked"""
        resp = await self.client.request("GET", "/view?filename=..%5Cetc%5Cpasswd")
        assert resp.status == 400, "Should block ..\\ path traversal at start"

    @unittest_run_loop
    async def test_blocks_mixed_slash_backslash_traversal(self):
        """Test that mixed forward/backslash path traversal is blocked"""
        resp = await self.client.request("GET", "/view?filename=folder/..%5Csecret")
        assert resp.status == 400, "Should block mixed slash/backslash path traversal"

    @unittest_run_loop
    async def test_blocks_backslash_absolute_path(self):
        """Test that Windows absolute paths with backslash are blocked (e.g., C:\\)"""
        resp = await self.client.request("GET", "/view?filename=%5Cetc%5Cpasswd")
        assert resp.status == 400, "Should block backslash absolute paths"

    @unittest_run_loop
    async def test_blocks_windows_drive_absolute_path(self):
        """Test that Windows drive-qualified absolute paths (C:/...) are blocked"""
        resp = await self.client.request("GET", "/view?filename=C:/Windows/secret.png")
        assert resp.status == 400, "Should block Windows drive-qualified absolute paths"

    @unittest_run_loop
    async def test_blocks_windows_drive_relative_path(self):
        """Test that Windows drive-qualified relative paths (C:secret.png) are blocked"""
        resp = await self.client.request("GET", "/view?filename=C:secret.png")
        assert resp.status == 400, "Should block Windows drive-qualified relative paths"

    @unittest_run_loop
    async def test_blocks_windows_drive_backslash_path(self):
        """Test that Windows drive paths with backslashes are blocked"""
        resp = await self.client.request("GET", "/view?filename=C:%5CWindows%5Csecret.png")
        assert resp.status == 400, "Should block Windows drive backslash paths"
