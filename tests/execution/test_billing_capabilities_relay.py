"""End-to-end coverage for Core's billing capabilities relay route.

The handler tests in `tests-unit/billing_capabilities_test.py` mount
`relay_billing_capabilities` on a standalone aiohttp app, so a route registered
on the wrong path, missing the automatic `/api` prefix, or surviving
`--disable-api-nodes` would still pass them. These tests drive a real ComfyUI
server whose upstream is a loopback stub, which also keeps a relayed upstream
404 from being mistaken for a missing Core route.
"""

import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

CAPABILITIES_ROUTE = "/api/billing/capabilities"
UNPREFIXED_CAPABILITIES_ROUTE = "/billing/capabilities"
UPSTREAM_ROUTE = "/api/billing/capabilities"
UPSTREAM_BODY = {"can_manage_subscription": True, "can_top_up": False}
UPSTREAM_REVISION = "rev-42"
UPSTREAM_CACHE_CONTROL = "private, max-age=30"
UPSTREAM_VARY = "Authorization, X-API-Key"
CALLER_SUPPLIED = "caller-supplied"
SERVER_READY_TIMEOUT = 300


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class UpstreamStub:
    """Stands in for the Comfy API so the relay never leaves loopback."""

    def __init__(self):
        self.received = []
        received = self.received

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                received.append({
                    "path": self.path,
                    "headers": {name.lower(): value for name, value in self.headers.items()},
                })
                body = json.dumps(UPSTREAM_BODY).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Capability-Revision", UPSTREAM_REVISION)
                self.send_header("Cache-Control", UPSTREAM_CACHE_CONTROL)
                self.send_header("Vary", UPSTREAM_VARY)
                self.send_header("Set-Cookie", "upstream_session=must-not-leak")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_port
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=10)


def _wait_until_serving(process: subprocess.Popen, port: int) -> None:
    deadline = time.monotonic() + SERVER_READY_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ComfyUI exited with code {process.returncode} before serving")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/system_stats", timeout=5):
                return
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    raise RuntimeError(f"ComfyUI did not start serving on port {port} within {SERVER_READY_TIMEOUT}s")


@contextmanager
def comfy_server(upstream_port: int, base_directory, *extra_args: str):
    port = _free_port()
    process = subprocess.Popen([
        sys.executable, "main.py",
        "--cpu",
        "--disable-all-custom-nodes",
        "--listen", "127.0.0.1",
        "--port", str(port),
        "--base-directory", str(base_directory),
        "--comfy-api-base", f"http://127.0.0.1:{upstream_port}",
        *extra_args,
    ])
    try:
        _wait_until_serving(process, port)
        yield port
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=60)


def _get(port: int, route: str, headers: dict | None = None):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{route}", headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.status, error.read(), dict(error.headers)


@pytest.fixture(scope="module")
def upstream():
    with UpstreamStub() as stub:
        yield stub


@pytest.fixture(scope="module")
def relay_port(upstream, tmp_path_factory):
    with comfy_server(upstream.port, tmp_path_factory.mktemp("relay")) as port:
        yield port


@pytest.fixture(autouse=True)
def clear_upstream_history(upstream):
    upstream.received.clear()


@pytest.mark.execution
class TestBillingCapabilitiesRelay:
    @pytest.mark.parametrize("route", [CAPABILITIES_ROUTE, UNPREFIXED_CAPABILITIES_ROUTE])
    def test_route_relays_upstream_payload(self, relay_port, upstream, route):
        status, body, headers = _get(relay_port, route)

        assert status == 200
        assert json.loads(body) == UPSTREAM_BODY
        assert headers["X-Capability-Revision"] == UPSTREAM_REVISION
        assert [request["path"] for request in upstream.received] == [UPSTREAM_ROUTE]

    def test_core_middleware_preserves_relayed_cache_headers(self, relay_port):
        _status, _body, headers = _get(relay_port, CAPABILITIES_ROUTE)

        assert headers["Cache-Control"] == UPSTREAM_CACHE_CONTROL
        assert headers["Vary"] == UPSTREAM_VARY
        assert "Set-Cookie" not in headers

    def test_upstream_receives_allowlisted_headers_and_core_owned_context(self, relay_port, upstream):
        status, _body, _headers = _get(relay_port, CAPABILITIES_ROUTE, {
            "Authorization": "Bearer relay-test-token",
            "X-API-Key": "relay-test-key",
            "Cookie": "session=must-not-leak",
            "X-Forwarded-Host": "attacker.example",
            "Comfy-Env": CALLER_SUPPLIED,
            "Comfy-Core-Version": CALLER_SUPPLIED,
        })

        assert status == 200
        forwarded = upstream.received[-1]["headers"]
        assert forwarded["authorization"] == "Bearer relay-test-token"
        assert forwarded["x-api-key"] == "relay-test-key"
        assert "cookie" not in forwarded
        assert "x-forwarded-host" not in forwarded
        assert forwarded["comfy-env"] != CALLER_SUPPLIED
        assert forwarded["comfy-core-version"] != CALLER_SUPPLIED

    def test_disable_api_nodes_removes_route(self, upstream, tmp_path_factory):
        with comfy_server(
            upstream.port,
            tmp_path_factory.mktemp("relay-disabled"),
            "--disable-api-nodes",
        ) as port:
            status, _body, _headers = _get(port, CAPABILITIES_ROUTE)

        assert status == 404
        assert upstream.received == []
