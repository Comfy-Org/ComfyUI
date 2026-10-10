import asyncio
import base64
import json
import logging
import time
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch
from aiohttp import web

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_api_nodes.util import _helpers
from comfy_api_nodes.util.client import ApiEndpoint, sync_op_raw
from comfy_api_nodes.util.download_helpers import download_url_to_bytesio


def _jwt(claims: dict) -> str:
    def enc(part: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(part).encode()).decode().rstrip("=")
    return f"{enc({'alg': 'ES256', 'typ': 'JWT'})}.{enc(claims)}.signature"


def _partner_token(expires_in: float, sid: str = "session-1", generation: int = 0) -> str:
    return _jwt({"aud": "comfy-partner-node", "sid": sid, "exp": time.time() + expires_in, "gen": generation})


def _node(auth_token: str | None = None, api_key: str | None = None) -> type:
    hidden = SimpleNamespace(
        auth_token_comfy_org=auth_token, api_key_comfy_org=api_key, comfy_usage_source=None, unique_id="1"
    )
    return type("FakeApiNode", (), {"hidden": hidden})


class FakeComfyApi:
    """comfy-api stand-in: a partner route that accepts only ``accepted`` tokens, plus the renew route."""

    def __init__(self, accepted: set[str], renewed_token: str | None, renew_status: int = 200, accept_renewed: bool = True):
        self.accepted = accepted
        self.renewed_token = renewed_token
        self.renew_status = renew_status
        self.accept_renewed = accept_renewed
        self.partner_calls: list[tuple[str | None, dict]] = []
        self.renew_calls: list[str | None] = []

    async def partner(self, request: web.Request) -> web.Response:
        auth = request.headers.get("Authorization")
        self.partner_calls.append((auth, await request.json()))
        if auth is None or auth.removeprefix("Bearer ") not in self.accepted:
            return web.json_response({"message": "token expired"}, status=401)
        return web.json_response({"ok": True})

    async def download(self, request: web.Request) -> web.Response:
        auth = request.headers.get("Authorization")
        self.partner_calls.append((auth, {}))
        if auth is None or auth.removeprefix("Bearer ") not in self.accepted:
            return web.Response(status=401)
        return web.Response(body=b"result-bytes")

    async def renew(self, request: web.Request) -> web.Response:
        self.renew_calls.append(request.headers.get("Authorization"))
        await asyncio.sleep(0.05)
        if self.renew_status != 200:
            return web.json_response({"message": "session revoked"}, status=self.renew_status)
        if self.accept_renewed:
            self.accepted.add(self.renewed_token)
        return web.json_response({"token": self.renewed_token})


async def _with_server(api: FakeComfyApi, monkeypatch, body):
    app = web.Application()
    app.router.add_post("/proxy/partner/generate", api.partner)
    app.router.add_get("/proxy/partner/result", api.download)
    app.router.add_post(_helpers.PARTNER_NODE_TOKEN_RENEW_PATH, api.renew)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    monkeypatch.setattr(args, "comfy_api_base", f"http://127.0.0.1:{port}")
    try:
        return await body()
    finally:
        await runner.cleanup()


def _generate(node: type):
    return sync_op_raw(
        node,
        ApiEndpoint("/proxy/partner/generate", "POST"),
        data={"prompt": "fennec ears"},
        monitor_progress=False,
        max_retries=0,
    )


@pytest.fixture(autouse=True)
def _clear_renewed_tokens():
    _helpers._renewed_partner_tokens.clear()
    _helpers._partner_token_renewals.clear()
    yield
    _helpers._renewed_partner_tokens.clear()
    _helpers._partner_token_renewals.clear()


@pytest.mark.parametrize(
    ("token", "is_partner"),
    [
        (_jwt({"aud": "comfy-partner-node", "sid": "s", "exp": 1}), True),
        (_jwt({"aud": ["comfy-partner-node"], "sid": "s", "exp": 1}), True),
        (_jwt({"aud": "comfy-api", "sid": "s", "exp": 1}), False),
        (_jwt({"aud": ["comfy-partner-node", "comfy-api"], "sid": "s", "exp": 1}), False),
        (_jwt({"aud": "comfy-partner-node", "exp": 1}), False),
        (_jwt({"aud": "comfy-partner-node", "sid": "s"}), False),
        ("comfyui-api-key-without-dots", False),
        ("a.%%%.c", False),
        ("a." + base64.urlsafe_b64encode(b"[1]").decode() + ".c", False),
    ],
)
def test_only_partner_node_tokens_are_recognized(token, is_partner):
    assert (_helpers._partner_token_claims(token) is not None) is is_partner


def test_token_near_expiry_is_renewed_before_the_request(monkeypatch):
    old, fresh = _partner_token(60), _partner_token(5400, generation=1)
    api = FakeComfyApi(accepted={old}, renewed_token=fresh)

    result = asyncio.run(_with_server(api, monkeypatch, lambda: _generate(_node(old))))

    assert result == {"ok": True}
    assert api.renew_calls == [f"Bearer {old}"]
    assert [auth for auth, _ in api.partner_calls] == [f"Bearer {fresh}"]


def test_token_far_from_expiry_is_sent_as_is(monkeypatch):
    token = _partner_token(5400)
    api = FakeComfyApi(accepted={token}, renewed_token=None)

    asyncio.run(_with_server(api, monkeypatch, lambda: _generate(_node(token))))

    assert api.renew_calls == []
    assert [auth for auth, _ in api.partner_calls] == [f"Bearer {token}"]


def test_401_renews_and_resends_the_post_once(monkeypatch):
    rejected, fresh = _partner_token(5400), _partner_token(5400, generation=1)
    api = FakeComfyApi(accepted=set(), renewed_token=fresh)

    result = asyncio.run(_with_server(api, monkeypatch, lambda: _generate(_node(rejected))))

    assert result == {"ok": True}
    assert api.renew_calls == [f"Bearer {rejected}"]
    assert api.partner_calls == [
        (f"Bearer {rejected}", {"prompt": "fennec ears"}),
        (f"Bearer {fresh}", {"prompt": "fennec ears"}),
    ]


def test_401_after_renewal_is_not_resent_again(monkeypatch):
    rejected, fresh = _partner_token(5400), _partner_token(5400, generation=1)
    api = FakeComfyApi(accepted=set(), renewed_token=fresh, accept_renewed=False)

    with pytest.raises(Exception, match="Unauthorized: Please login first"):
        asyncio.run(_with_server(api, monkeypatch, lambda: _generate(_node(rejected))))

    assert len(api.renew_calls) == 1
    assert len(api.partner_calls) == 2


def test_renew_refusal_surfaces_the_original_auth_error_without_logging_tokens(monkeypatch, caplog):
    rejected = _partner_token(5400)
    api = FakeComfyApi(accepted=set(), renewed_token=None, renew_status=401)

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Exception, match="Unauthorized: Please login first"):
            asyncio.run(_with_server(api, monkeypatch, lambda: _generate(_node(rejected))))

    assert len(api.renew_calls) == 1
    assert len(api.partner_calls) == 1
    assert rejected not in caplog.text
    assert rejected.split(".")[1] not in caplog.text


@pytest.mark.parametrize(
    ("auth_token", "api_key", "expected_header"),
    [
        (_jwt({"aud": "comfy-api", "sid": "s", "exp": time.time() - 10}), None, "Authorization"),
        (None, "comfyui-api-key", "X-API-KEY"),
    ],
)
def test_other_credentials_are_never_renewed(monkeypatch, auth_token, api_key, expected_header):
    api = FakeComfyApi(accepted=set(), renewed_token=_partner_token(5400))
    sent_headers = []

    async def partner(request: web.Request) -> web.Response:
        sent_headers.append(request.headers.get(expected_header))
        return web.json_response({"message": "Unauthorized"}, status=401)

    api.partner = partner

    with pytest.raises(Exception, match="Unauthorized: Please login first"):
        asyncio.run(_with_server(api, monkeypatch, lambda: _generate(_node(auth_token, api_key))))

    assert api.renew_calls == []
    assert sent_headers == [f"Bearer {auth_token}" if auth_token else api_key]


def test_concurrent_requests_share_one_renewal(monkeypatch):
    old, fresh = _partner_token(60), _partner_token(5400, generation=1)
    api = FakeComfyApi(accepted={old}, renewed_token=fresh)
    node = _node(old)

    async def three_requests():
        return await asyncio.gather(_generate(node), _generate(node), _generate(node))

    results = asyncio.run(_with_server(api, monkeypatch, three_requests))

    assert results == [{"ok": True}] * 3
    assert api.renew_calls == [f"Bearer {old}"]
    assert [auth for auth, _ in api.partner_calls] == [f"Bearer {fresh}"] * 3


def test_prompts_from_the_same_session_reuse_the_renewed_token(monkeypatch):
    first_snapshot, fresh = _partner_token(60), _partner_token(5400, generation=1)
    second_snapshot = _partner_token(30)
    api = FakeComfyApi(accepted={first_snapshot}, renewed_token=fresh)

    async def two_prompts():
        await _generate(_node(first_snapshot))
        await _generate(_node(second_snapshot))

    asyncio.run(_with_server(api, monkeypatch, two_prompts))

    assert len(api.renew_calls) == 1
    assert [auth for auth, _ in api.partner_calls] == [f"Bearer {fresh}"] * 2


def test_result_download_uses_a_renewed_token(monkeypatch):
    old, fresh = _partner_token(60), _partner_token(5400, generation=1)
    api = FakeComfyApi(accepted=set(), renewed_token=fresh)
    buf = BytesIO()

    asyncio.run(
        _with_server(
            api, monkeypatch, lambda: download_url_to_bytesio("/proxy/partner/result", buf, cls=_node(old), max_retries=0)
        )
    )

    assert buf.getvalue() == b"result-bytes"
    assert [auth for auth, _ in api.partner_calls] == [f"Bearer {fresh}"]
