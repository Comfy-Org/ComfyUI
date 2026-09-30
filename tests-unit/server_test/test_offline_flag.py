"""Tests for --offline and --disable-api-nodes"""

import subprocess
import sys
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from comfy.cli_args import args
import server


async def pong(request):
    return web.Response(text="pong")


def parse_args(*argv):
    code = "import comfy.options; comfy.options.enable_args_parsing(); from comfy.cli_args import args; print(args.offline, args.disable_api_nodes)"
    out = subprocess.run([sys.executable, "-c", code, *argv], capture_output=True, text=True, check=True)
    return out.stdout.split()[-2:]


@pytest.mark.parametrize("argv,expected", [
    ([], ["False", "False"]),
    (["--disable-api-nodes"], ["False", "True"]),
    (["--offline"], ["True", "True"]),
    (["--offline", "--disable-api-nodes"], ["True", "True"]),
])
def test_arg_parsing(argv, expected):
    assert parse_args(*argv) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("offline,disable_api_nodes,expect_csp", [
    (False, False, False),
    (False, True, False),
    (True, True, True),
])
async def test_csp_header(monkeypatch, offline, disable_api_nodes, expect_csp):
    monkeypatch.setattr(args, "offline", offline)
    monkeypatch.setattr(args, "disable_api_nodes", disable_api_nodes)
    prompt_server = server.PromptServer(None, MagicMock(enabled=False))
    prompt_server.app.router.add_get("/ping", pong)
    async with TestClient(TestServer(prompt_server.app)) as client:
        resp = await client.get("/ping")
        assert resp.status == 200
        csp = resp.headers.get("Content-Security-Policy")
    if expect_csp:
        assert "connect-src 'self' data:" in csp
    else:
        assert csp is None
