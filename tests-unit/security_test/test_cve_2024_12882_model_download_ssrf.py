"""Regression test for CVE-2024-12882 (non-blind SSRF).

The exploit chained POST /internal/models/download (server fetches an
attacker-supplied URL and writes the body into a model folder) with GET /view
(attacker reads the fetched body back). The download route was added in #4248
and removed in #5432. These tests fail if a route at that path is registered
again on the real PromptServer app, or if posting an exploit body to it makes
the server issue an outbound request.

Reference: https://nvd.nist.gov/vuln/detail/CVE-2024-12882
"""
from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

import server

EXPLOIT_PATHS = ["/internal/models/download", "/api/internal/models/download"]
EXPLOIT_BODY = {
    "url": "http://169.254.169.254/latest/meta-data/",
    "model_directory": "checkpoints",
    "folder_path": "checkpoints",
    "model_filename": "ssrf.safetensors",
}


@pytest.fixture
def prompt_server():
    prompt_server = server.PromptServer(None, MagicMock(enabled=False))
    prompt_server.client_session = MagicMock()
    prompt_server.add_routes()
    return prompt_server


def test_download_route_not_registered(prompt_server):
    app_paths = {resource.canonical for resource in prompt_server.app.router.resources()}
    internal_paths = {resource.canonical for resource in prompt_server.internal_routes.get_app().router.resources()}
    for path in EXPLOIT_PATHS:
        assert path not in app_paths
    assert "/models/download" not in internal_paths


@pytest.mark.asyncio
@pytest.mark.parametrize("path", EXPLOIT_PATHS)
async def test_download_route_does_not_fetch_url(prompt_server, path):
    async with TestClient(TestServer(prompt_server.app)) as client:
        resp = await client.post(path, json=EXPLOIT_BODY)
        assert resp.status in (404, 405)
    assert prompt_server.client_session.mock_calls == []
