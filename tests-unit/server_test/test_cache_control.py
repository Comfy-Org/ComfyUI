"""Tests for server cache control middleware"""

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request
from typing import Dict, Any

from middleware.cache_middleware import cache_control, ONE_HOUR, ONE_DAY, IMG_EXTENSIONS

pytestmark = pytest.mark.asyncio  # Apply asyncio mark to all tests

# Test configuration data
CACHE_SCENARIOS = [
    # Image file scenarios
    {
        "name": "image_200_status",
        "path": "/test.jpg",
        "status": 200,
        "expected_cache": f"public, max-age={ONE_DAY}",
        "should_have_header": True,
    },
    {
        "name": "image_404_status",
        "path": "/missing.jpg",
        "status": 404,
        "expected_cache": f"public, max-age={ONE_HOUR}",
        "should_have_header": True,
    },
    # JavaScript/CSS scenarios
    {
        "name": "js_no_store",
        "path": "/script.js",
        "status": 200,
        "expected_cache": "no-store",
        "should_have_header": True,
    },
    {
        "name": "css_no_store",
        "path": "/styles.css",
        "status": 200,
        "expected_cache": "no-store",
        "should_have_header": True,
    },
    {
        "name": "index_json_no_store",
        "path": "/api/index.json",
        "status": 200,
        "expected_cache": "no-store",
        "should_have_header": True,
    },
    {
        "name": "localized_index_json_no_store",
        "path": "/templates/index.zh.json",
        "status": 200,
        "expected_cache": "no-store",
        "should_have_header": True,
    },
    # Non-matching files
    {
        "name": "html_no_header",
        "path": "/index.html",
        "status": 200,
        "expected_cache": None,
        "should_have_header": False,
    },
    {
        "name": "txt_no_header",
        "path": "/data.txt",
        "status": 200,
        "expected_cache": None,
        "should_have_header": False,
    },
    {
        "name": "api_endpoint_no_header",
        "path": "/api/endpoint",
        "status": 200,
        "expected_cache": None,
        "should_have_header": False,
    },
    {
        "name": "pdf_no_header",
        "path": "/file.pdf",
        "status": 200,
        "expected_cache": None,
        "should_have_header": False,
    },
]

# Status code scenarios for images
IMAGE_STATUS_SCENARIOS = [
    # Success statuses get long cache
    {"status": 200, "expected": f"public, max-age={ONE_DAY}"},
    {"status": 201, "expected": f"public, max-age={ONE_DAY}"},
    {"status": 202, "expected": f"public, max-age={ONE_DAY}"},
    {"status": 204, "expected": f"public, max-age={ONE_DAY}"},
    {"status": 206, "expected": f"public, max-age={ONE_DAY}"},
    # Permanent redirects get long cache
    {"status": 301, "expected": f"public, max-age={ONE_DAY}"},
    {"status": 308, "expected": f"public, max-age={ONE_DAY}"},
    # Temporary redirects get no cache
    {"status": 302, "expected": "no-cache"},
    {"status": 303, "expected": "no-cache"},
    {"status": 307, "expected": "no-cache"},
    # 404 gets short cache
    {"status": 404, "expected": f"public, max-age={ONE_HOUR}"},
]

# Case sensitivity test paths
CASE_SENSITIVITY_PATHS = ["/image.JPG", "/photo.PNG", "/pic.JpEg"]

# Edge case test paths
EDGE_CASE_PATHS = [
    {
        "name": "query_strings_ignored",
        "path": "/image.jpg?v=123&size=large",
        "expected": f"public, max-age={ONE_DAY}",
    },
    {
        "name": "multiple_dots_in_path",
        "path": "/image.min.jpg",
        "expected": f"public, max-age={ONE_DAY}",
    },
    {
        "name": "nested_paths_with_images",
        "path": "/static/images/photo.jpg",
        "expected": f"public, max-age={ONE_DAY}",
    },
]


class TestCacheControl:
    """Test cache control middleware functionality"""

    @pytest.fixture
    def status_handler_factory(self):
        """Create a factory for handlers that return specific status codes"""

        def factory(status: int, headers: Dict[str, str] = None):
            async def handler(request):
                return web.Response(status=status, headers=headers or {})

            return handler

        return factory

    @pytest.fixture
    def mock_handler(self, status_handler_factory):
        """Create a mock handler that returns a response with 200 status"""
        return status_handler_factory(200)

    @pytest.fixture
    def handler_with_existing_cache(self, status_handler_factory):
        """Create a handler that returns response with existing Cache-Control header"""
        return status_handler_factory(200, {"Cache-Control": "max-age=3600"})

    async def assert_cache_header(
        self,
        response: web.Response,
        expected_cache: str = None,
        should_have_header: bool = True,
    ):
        """Helper to assert cache control headers"""
        if should_have_header:
            assert "Cache-Control" in response.headers
            if expected_cache:
                assert response.headers["Cache-Control"] == expected_cache
        else:
            assert "Cache-Control" not in response.headers

    # Parameterized tests
    @pytest.mark.parametrize("scenario", CACHE_SCENARIOS, ids=lambda x: x["name"])
    async def test_cache_control_scenarios(
        self, scenario: Dict[str, Any], status_handler_factory
    ):
        """Test various cache control scenarios"""
        handler = status_handler_factory(scenario["status"])
        request = make_mocked_request("GET", scenario["path"])
        response = await cache_control(request, handler)

        assert response.status == scenario["status"]
        await self.assert_cache_header(
            response, scenario["expected_cache"], scenario["should_have_header"]
        )

    @pytest.mark.parametrize("ext", IMG_EXTENSIONS)
    async def test_all_image_extensions(self, ext: str, mock_handler):
        """Test all defined image extensions are handled correctly"""
        request = make_mocked_request("GET", f"/image{ext}")
        response = await cache_control(request, mock_handler)

        assert response.status == 200
        assert "Cache-Control" in response.headers
        assert response.headers["Cache-Control"] == f"public, max-age={ONE_DAY}"

    @pytest.mark.parametrize(
        "status_scenario", IMAGE_STATUS_SCENARIOS, ids=lambda x: f"status_{x['status']}"
    )
    async def test_image_status_codes(
        self, status_scenario: Dict[str, Any], status_handler_factory
    ):
        """Test different status codes for image requests"""
        handler = status_handler_factory(status_scenario["status"])
        request = make_mocked_request("GET", "/image.jpg")
        response = await cache_control(request, handler)

        assert response.status == status_scenario["status"]
        assert "Cache-Control" in response.headers
        assert response.headers["Cache-Control"] == status_scenario["expected"]

    @pytest.mark.parametrize("path", CASE_SENSITIVITY_PATHS)
    async def test_case_insensitive_image_extension(self, path: str, mock_handler):
        """Test that image extensions are matched case-insensitively"""
        request = make_mocked_request("GET", path)
        response = await cache_control(request, mock_handler)

        assert "Cache-Control" in response.headers
        assert response.headers["Cache-Control"] == f"public, max-age={ONE_DAY}"

    @pytest.mark.parametrize("edge_case", EDGE_CASE_PATHS, ids=lambda x: x["name"])
    async def test_edge_cases(self, edge_case: Dict[str, str], mock_handler):
        """Test edge cases like query strings, nested paths, etc."""
        request = make_mocked_request("GET", edge_case["path"])
        response = await cache_control(request, mock_handler)

        assert "Cache-Control" in response.headers
        assert response.headers["Cache-Control"] == edge_case["expected"]

    # Header preservation tests (special cases not covered by parameterization)
    async def test_js_preserves_existing_headers(self, handler_with_existing_cache):
        """Test that .js files preserve existing Cache-Control headers"""
        request = make_mocked_request("GET", "/script.js")
        response = await cache_control(request, handler_with_existing_cache)

        # setdefault should preserve existing header
        assert response.headers["Cache-Control"] == "max-age=3600"

    async def test_css_preserves_existing_headers(self, handler_with_existing_cache):
        """Test that .css files preserve existing Cache-Control headers"""
        request = make_mocked_request("GET", "/styles.css")
        response = await cache_control(request, handler_with_existing_cache)

        # setdefault should preserve existing header
        assert response.headers["Cache-Control"] == "max-age=3600"

    async def test_image_preserves_existing_headers(self, status_handler_factory):
        """Test that image cache headers preserve existing Cache-Control"""
        handler = status_handler_factory(200, {"Cache-Control": "private, no-cache"})
        request = make_mocked_request("GET", "/image.jpg")
        response = await cache_control(request, handler)

        # setdefault should preserve existing header
        assert response.headers["Cache-Control"] == "private, no-cache"

    async def test_304_not_modified_inherits_cache(self, status_handler_factory):
        """Test that 304 Not Modified doesn't set cache headers for images"""
        handler = status_handler_factory(304, {"Cache-Control": "max-age=7200"})
        request = make_mocked_request("GET", "/not-modified.jpg")
        response = await cache_control(request, handler)

        assert response.status == 304
        # Should preserve existing cache header, not override
        assert response.headers["Cache-Control"] == "max-age=7200"
