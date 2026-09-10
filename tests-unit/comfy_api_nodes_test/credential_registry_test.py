import asyncio
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from aiohttp import WSCloseCode, WSMsgType, web

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import execution
from comfy_api.credential_registry import CredentialRegistry, api_node_credentials
from comfy_api_nodes.util import _helpers
from comfy_api_nodes.util import client as api_client
from comfy_api_nodes.util import request_logger
from comfy_execution.utils import CurrentNodeContext
from server import PromptServer, _authenticated_client_id, _credential_transport_enabled


class _Node:
    hidden = SimpleNamespace(
        auth_token_comfy_org="snapshot-token",
        api_key_comfy_org=None,
        comfy_usage_source=None,
        unique_id="node",
    )


class _Transport:
    def __init__(self, *, peername, sslcontext=None, ssl_object=None):
        self._extra = {
            "peername": peername,
            "sslcontext": sslcontext,
            "ssl_object": ssl_object,
        }

    def get_extra_info(self, name):
        return self._extra.get(name)


@pytest.fixture(autouse=True)
def clear_credentials(monkeypatch):
    api_node_credentials.clear()
    monkeypatch.setattr(api_client, "is_processing_interrupted", lambda: False)
    monkeypatch.setattr(api_client, "_display_time_progress", lambda *args, **kwargs: None)
    yield
    api_node_credentials.clear()


def test_registry_isolates_clients_and_late_binds_updates():
    registry = CredentialRegistry()
    registry.update("client-a", "a1")
    registry.update("client-b", "b1")
    assert registry.bind_prompt("prompt-a", "client-a") is True
    assert registry.bind_prompt("prompt-b", "client-b") is True
    assert registry.bind_prompt("prompt-a", "client-b") is False

    assert registry.get_for_prompt("prompt-a").token == "a1"
    assert registry.get_for_prompt("prompt-b").token == "b1"

    first_generation = registry.get_for_prompt("prompt-a").generation
    registry.update("client-a", "a2")
    assert registry.get_for_prompt("prompt-a").token == "a2"
    assert registry.get_for_prompt("prompt-a").generation == first_generation + 1
    assert registry.get_for_prompt("prompt-b").token == "b1"


def test_registry_cleanup_on_completion_disconnect_and_ttl():
    now = [0.0]
    registry = CredentialRegistry(credential_ttl=10, prompt_ttl=20, clock=lambda: now[0])
    registry.update("client", "secret")
    registry.bind_prompt("prompt", "client")
    registry.disconnect("client")
    assert registry.get_for_prompt("prompt").token == "secret"

    registry.release_prompt("prompt")
    assert registry.get_for_prompt("prompt") is None

    registry.update("stale-client", "stale")
    registry.bind_prompt("stale-prompt", "stale-client")
    now[0] = 21
    assert registry.get_for_prompt("stale-prompt") is None


def test_connected_registry_credentials_do_not_expire():
    now = [0.0]
    registry = CredentialRegistry(credential_ttl=10, clock=lambda: now[0])
    registry.connect("client")
    registry.update("client", "secret")
    registry.bind_prompt("prompt", "client")
    registry.release_prompt("prompt")

    now[0] = 11
    registry.bind_prompt("next-prompt", "client")
    assert registry.get_for_prompt("next-prompt").token == "secret"


def test_auth_header_uses_registry_then_falls_back_and_preserves_api_key():
    with CurrentNodeContext("legacy-prompt", "node"):
        assert _helpers.get_auth_header(_Node) == {"Authorization": "Bearer snapshot-token"}

    api_node_credentials.update("client", "fresh-token")
    api_node_credentials.bind_prompt("prompt", "client")
    with CurrentNodeContext("prompt", "node"):
        assert _helpers.get_auth_header(_Node) == {"Authorization": "Bearer fresh-token"}

    api_node_credentials.update("client", None)
    with CurrentNodeContext("prompt", "node"):
        assert _helpers.get_auth_header(_Node) == {}

    api_key_node = type(
        "ApiKeyNode",
        (),
        {
            "hidden": SimpleNamespace(
                auth_token_comfy_org=None,
                api_key_comfy_org="api-key",
                comfy_usage_source=None,
                unique_id="node",
            )
        },
    )
    with CurrentNodeContext("prompt", "node"):
        assert _helpers.get_auth_header(api_key_node) == {"X-API-KEY": "api-key"}

    api_key_node.hidden.auth_token_comfy_org = "legacy-token"
    with CurrentNodeContext("prompt", "node"):
        assert _helpers.get_auth_header(api_key_node) == {"Authorization": "Bearer legacy-token"}


def test_client_request_authentication_requires_matching_session_key():
    request = SimpleNamespace(headers={"X-Comfy-Client-Id": "client", "X-Comfy-Credential-Key": "key"})
    metadata = {"client": {"credential_key": "key"}}
    assert _authenticated_client_id(request, metadata) == "client"

    request.headers["X-Comfy-Credential-Key"] = "wrong"
    assert _authenticated_client_id(request, metadata) is None


@pytest.mark.parametrize("peername", [("127.0.0.1", 8188), ("::1", 8188, 0, 0)])
def test_credential_transport_allows_direct_loopback(peername):
    request = SimpleNamespace(transport=_Transport(peername=peername), headers={})
    assert _credential_transport_enabled(request) is True


def test_credential_transport_allows_direct_tls():
    request = SimpleNamespace(
        transport=_Transport(peername=("203.0.113.10", 8188), sslcontext=object()),
        headers={},
    )
    assert _credential_transport_enabled(request) is True


def test_credential_transport_rejects_plaintext_remote_and_ignores_forwarded_proto():
    request = SimpleNamespace(
        transport=_Transport(peername=("203.0.113.10", 8188)),
        headers={"X-Forwarded-Proto": "https", "Forwarded": "proto=https"},
    )
    assert _credential_transport_enabled(request) is False


@pytest.mark.asyncio
async def test_credential_endpoint_updates_and_clears_without_returning_secret(aiohttp_client):
    prompt_server = PromptServer(None)
    prompt_server.sockets_metadata["client"] = {"credential_key": "key", "feature_flags": {}}
    route = next(route for route in prompt_server.routes if route.path == "/credentials")
    app = web.Application()
    app.router.add_post("/api/credentials", route.handler)
    http_client = await aiohttp_client(app)
    headers = {"X-Comfy-Client-Id": "client", "X-Comfy-Credential-Key": "key"}

    response = await http_client.post(
        "/api/credentials",
        headers=headers,
        json={"auth_token_comfy_org": "top-secret"},
    )
    assert response.status == 200
    response_text = await response.text()
    assert json.loads(response_text) == {"generation": 1}
    assert "top-secret" not in response_text

    response = await http_client.post(
        "/api/credentials",
        headers=headers,
        json={"auth_token_comfy_org": None},
    )
    assert response.status == 200
    assert await response.json() == {"generation": 2}

    response = await http_client.post(
        "/api/credentials",
        headers={"X-Comfy-Client-Id": "client", "X-Comfy-Credential-Key": "wrong"},
        json={"auth_token_comfy_org": "stolen"},
    )
    assert response.status == 403


@pytest.mark.asyncio
async def test_remote_plaintext_rejects_credentials_but_allows_legacy_prompt(
    aiohttp_client, monkeypatch
):
    monkeypatch.setattr("server._credential_transport_enabled", lambda request: False)

    async def validate_prompt(*args, **kwargs):
        return True, None, [], {}

    monkeypatch.setattr(execution, "validate_prompt", validate_prompt)
    prompt_server = PromptServer(None)
    monkeypatch.setattr(prompt_server.prompt_queue, "put", lambda item: True)
    routes = {route.path: route for route in prompt_server.routes}
    app = web.Application()
    app.router.add_get("/api/features", routes["/features"].handler)
    app.router.add_post("/api/credentials", routes["/credentials"].handler)
    app.router.add_post("/api/prompt", routes["/prompt"].handler)
    http_client = await aiohttp_client(app)
    credential_headers = {
        "X-Comfy-Client-Id": "client",
        "X-Comfy-Credential-Key": "key",
    }

    response = await http_client.post(
        "/api/credentials",
        headers=credential_headers,
        json={"auth_token_comfy_org": "top-secret"},
    )
    assert response.status == 403

    response = await http_client.post("/api/prompt", headers=credential_headers, json={})
    assert response.status == 403

    response = await http_client.post(
        "/api/prompt",
        json={"client_id": "legacy-client", "prompt": {}},
    )
    assert response.status == 200
    assert api_node_credentials.is_protected("legacy-client") is False

    response = await http_client.get("/api/features")
    assert response.status == 200
    assert "comfy_api_credentials" not in await response.json()


@pytest.mark.asyncio
async def test_remote_plaintext_websocket_omits_credentials_and_reconnects(
    aiohttp_client, monkeypatch
):
    monkeypatch.setattr("server._credential_transport_enabled", lambda request: False)
    prompt_server = PromptServer(None)
    route = next(route for route in prompt_server.routes if route.path == "/ws")
    app = web.Application()
    app.router.add_get("/ws", route.handler)
    http_client = await aiohttp_client(app)

    websocket = await http_client.ws_connect("/ws?clientId=legacy-client")
    status = await websocket.receive_json()
    assert status["data"]["sid"] == "legacy-client"
    assert "credential_key" not in status["data"]
    await websocket.send_json({"type": "feature_flags", "data": {}})
    server_features = await websocket.receive_json()
    assert "comfy_api_credentials" not in server_features["data"]
    assert "credential_key" not in prompt_server.sockets_metadata["legacy-client"]
    assert api_node_credentials.is_protected("legacy-client") is False
    await websocket.close()
    await asyncio.sleep(0)

    websocket = await http_client.ws_connect("/ws?clientId=legacy-client")
    status = await websocket.receive_json()
    assert status["data"]["sid"] == "legacy-client"
    assert "credential_key" not in status["data"]
    await websocket.close()


@pytest.mark.asyncio
async def test_protected_client_id_requires_websocket_credential_on_reconnect(aiohttp_client):
    prompt_server = PromptServer(None)
    route = next(route for route in prompt_server.routes if route.path == "/ws")
    app = web.Application()
    app.router.add_get("/ws", route.handler)
    http_client = await aiohttp_client(app)

    websocket = await http_client.ws_connect("/ws?clientId=client")
    status = await websocket.receive_json()
    credential_key = status["data"]["credential_key"]

    attacker = await http_client.ws_connect("/ws?clientId=client")
    await attacker.send_json({"type": "credential_auth", "data": {"credential_key": "wrong"}})
    assert (await attacker.receive()).type in {WSMsgType.CLOSE, WSMsgType.CLOSED}

    api_node_credentials.update("client", "top-secret")
    api_node_credentials.bind_prompt("prompt", "client")
    await websocket.close()
    await asyncio.sleep(0)

    websocket = await http_client.ws_connect("/ws?clientId=client")
    await websocket.send_json({"type": "credential_auth", "data": {"credential_key": "wrong"}})
    assert (await websocket.receive()).type in {WSMsgType.CLOSE, WSMsgType.CLOSED}

    websocket = await http_client.ws_connect("/ws?clientId=client")
    await websocket.send_json(
        {"type": "credential_auth", "data": {"credential_key": credential_key}}
    )
    assert (await websocket.receive_json())["data"]["sid"] == "client"
    await websocket.close()
    prompt_server.release_api_node_prompt("prompt")
    assert "client" not in prompt_server.sockets_metadata


@pytest.mark.parametrize("auth_data", [[], "credential_auth", None])
@pytest.mark.asyncio
async def test_protected_reconnect_rejects_non_object_credential_auth(auth_data, aiohttp_client):
    prompt_server = PromptServer(None)
    route = next(route for route in prompt_server.routes if route.path == "/ws")
    app = web.Application()
    app.router.add_get("/ws", route.handler)
    http_client = await aiohttp_client(app)

    websocket = await http_client.ws_connect("/ws?clientId=client")
    await websocket.receive_json()
    api_node_credentials.update("client", "top-secret")
    api_node_credentials.bind_prompt("prompt", "client")
    await websocket.close()
    await asyncio.sleep(0)

    websocket = await http_client.ws_connect("/ws?clientId=client")
    await websocket.send_str(json.dumps(auth_data))
    close_message = await websocket.receive()
    assert close_message.type == WSMsgType.CLOSE
    assert close_message.data == WSCloseCode.POLICY_VIOLATION


@pytest.mark.asyncio
async def test_unprotected_reconnect_negotiates_features_after_stale_credential_auth(aiohttp_client):
    prompt_server = PromptServer(None)
    route = next(route for route in prompt_server.routes if route.path == "/ws")
    app = web.Application()
    app.router.add_get("/ws", route.handler)
    http_client = await aiohttp_client(app)

    websocket = await http_client.ws_connect("/ws?clientId=client")
    status = await websocket.receive_json()
    stale_credential_key = status["data"]["credential_key"]
    await websocket.close()
    await asyncio.sleep(0)
    assert "client" not in prompt_server.sockets_metadata

    websocket = await http_client.ws_connect("/ws?clientId=client")
    await websocket.send_json(
        {"type": "credential_auth", "data": {"credential_key": stale_credential_key}}
    )
    await websocket.send_json(
        {"type": "feature_flags", "data": {"supports_preview_metadata": True}}
    )

    status = await websocket.receive_json()
    assert status["type"] == "status"
    assert status["data"]["credential_key"] != stale_credential_key
    server_features = await websocket.receive_json()
    assert server_features["type"] == "feature_flags"
    assert server_features["data"]["supports_preview_metadata"] is True
    assert prompt_server.sockets_metadata["client"]["feature_flags"] == {
        "supports_preview_metadata": True
    }
    await websocket.close()


def test_credentials_are_redacted_from_repr_and_request_logs():
    credential = CredentialRegistry()
    credential.update("client", "top-secret")
    credential.bind_prompt("prompt", "client")
    assert "top-secret" not in repr(credential.get_for_prompt("prompt"))
    assert request_logger._redact_headers(
        {"Authorization": "Bearer top-secret", "X-API-KEY": "api-secret", "Other": "visible"}
    ) == {"Authorization": "***", "X-API-KEY": "***", "Other": "visible"}
    assert execution.format_input_data(
        {
            "auth_token_comfy_org": ["top-secret"],
            "api_key_comfy_org": ["api-secret"],
            "prompt": ["visible"],
        }
    ) == {
        "auth_token_comfy_org": ["***"],
        "api_key_comfy_org": ["***"],
        "prompt": ["visible"],
    }


def test_execution_exception_message_and_traceback_are_redacted(caplog):
    extra_data = {
        "auth_token_comfy_org": "top-secret",
        "api_key_comfy_org": "api-secret",
    }
    try:
        raise RuntimeError("failed with top-secret and api-secret")
    except RuntimeError as ex:
        _, _, tb = sys.exc_info()
        message, traceback_lines = execution.log_execution_exception(ex, tb, extra_data)

    assert "top-secret" not in message
    assert "api-secret" not in message
    assert "top-secret" not in "".join(traceback_lines)
    assert "api-secret" not in "".join(traceback_lines)
    assert "top-secret" not in caplog.text
    assert "api-secret" not in caplog.text


@pytest.mark.asyncio
async def test_poll_retries_once_only_when_generation_changes(aiohttp_client, monkeypatch):
    requests = []

    async def poll(request):
        requests.append(request.headers.get("Authorization"))
        if len(requests) == 1:
            api_node_credentials.update("client", "new-token")
            return web.json_response({"error": "expired"}, status=401)
        return web.json_response({"status": "done"})

    app = web.Application()
    app.router.add_get("/poll", poll)
    http_client = await aiohttp_client(app)
    monkeypatch.setattr(api_client, "default_base_url", lambda: str(http_client.make_url("/")))
    monkeypatch.setattr(api_client.request_logger, "log_request_response", lambda **kwargs: None)
    api_node_credentials.update("client", "old-token")
    api_node_credentials.bind_prompt("prompt", "client")

    with CurrentNodeContext("prompt", "node"):
        result = await api_client.poll_op_raw(
            _Node,
            api_client.ApiEndpoint("/poll"),
            status_extractor=lambda response: response["status"],
            completed_statuses=["done"],
            poll_interval=0,
            max_retries_per_poll=0,
        )

    assert result == {"status": "done"}
    assert requests == ["Bearer old-token", "Bearer new-token"]


@pytest.mark.asyncio
async def test_auth_refresh_retry_does_not_consume_generic_retry_budget(aiohttp_client, monkeypatch):
    request_count = 0

    async def poll(request):
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            api_node_credentials.update("client", "new-token")
            return web.json_response({"error": "expired"}, status=401)
        if request_count == 2:
            return web.json_response({"error": "temporary"}, status=502)
        return web.json_response({"status": "done"})

    app = web.Application()
    app.router.add_get("/poll", poll)
    http_client = await aiohttp_client(app)
    monkeypatch.setattr(api_client, "default_base_url", lambda: str(http_client.make_url("/")))
    monkeypatch.setattr(api_client.request_logger, "log_request_response", lambda **kwargs: None)
    api_node_credentials.update("client", "old-token")
    api_node_credentials.bind_prompt("prompt", "client")

    with CurrentNodeContext("prompt", "node"):
        result = await api_client.poll_op_raw(
            _Node,
            api_client.ApiEndpoint("/poll"),
            status_extractor=lambda response: response["status"],
            completed_statuses=["done"],
            poll_interval=0,
            max_retries_per_poll=1,
            retry_delay_per_poll=0,
        )

    assert result == {"status": "done"}
    assert request_count == 3


@pytest.mark.asyncio
async def test_401_without_generation_change_is_not_retried(aiohttp_client, monkeypatch):
    request_count = 0

    async def poll(request):
        nonlocal request_count
        request_count += 1
        return web.json_response({"error": "expired"}, status=401)

    app = web.Application()
    app.router.add_get("/poll", poll)
    http_client = await aiohttp_client(app)
    monkeypatch.setattr(api_client, "default_base_url", lambda: str(http_client.make_url("/")))
    monkeypatch.setattr(api_client.request_logger, "log_request_response", lambda **kwargs: None)
    api_node_credentials.update("client", "old-token")
    api_node_credentials.bind_prompt("prompt", "client")

    with CurrentNodeContext("prompt", "node"):
        with pytest.raises(Exception, match="Unauthorized"):
            await api_client.poll_op_raw(
                _Node,
                api_client.ApiEndpoint("/poll"),
                status_extractor=lambda response: response["status"],
                poll_interval=0,
                max_retries_per_poll=0,
            )

    assert request_count == 1


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer endpoint-token"},
        {"X-API-KEY": "endpoint-key"},
    ],
)
@pytest.mark.asyncio
async def test_endpoint_auth_401_does_not_retry_after_registry_change(headers, aiohttp_client, monkeypatch):
    request_count = 0

    async def poll(request):
        nonlocal request_count
        request_count += 1
        api_node_credentials.update("client", "new-token")
        return web.json_response({"error": "expired"}, status=401)

    app = web.Application()
    app.router.add_get("/poll", poll)
    http_client = await aiohttp_client(app)
    monkeypatch.setattr(api_client, "default_base_url", lambda: str(http_client.make_url("/")))
    monkeypatch.setattr(api_client.request_logger, "log_request_response", lambda **kwargs: None)
    api_node_credentials.update("client", "old-token")
    api_node_credentials.bind_prompt("prompt", "client")

    with CurrentNodeContext("prompt", "node"):
        with pytest.raises(Exception, match="Unauthorized"):
            await api_client.poll_op_raw(
                _Node,
                api_client.ApiEndpoint("/poll", headers=headers),
                status_extractor=lambda response: response["status"],
                poll_interval=0,
                max_retries_per_poll=0,
            )

    assert request_count == 1


@pytest.mark.asyncio
async def test_paid_post_is_not_retried_after_token_change(aiohttp_client, monkeypatch):
    requests = []

    async def generate(request):
        requests.append(request.headers.get("Authorization"))
        api_node_credentials.update("client", "new-token")
        return web.json_response({"error": "expired"}, status=401)

    app = web.Application()
    app.router.add_post("/generate", generate)
    http_client = await aiohttp_client(app)
    monkeypatch.setattr(api_client, "default_base_url", lambda: str(http_client.make_url("/")))
    monkeypatch.setattr(api_client.request_logger, "log_request_response", lambda **kwargs: None)
    api_node_credentials.update("client", "old-token")
    api_node_credentials.bind_prompt("prompt", "client")

    with CurrentNodeContext("prompt", "node"):
        with pytest.raises(Exception, match="Unauthorized"):
            await api_client.sync_op_raw(
                _Node,
                api_client.ApiEndpoint("/generate", method="POST"),
                max_retries=0,
                retry_401_on_auth_change=True,
            )

    assert requests == ["Bearer old-token"]
