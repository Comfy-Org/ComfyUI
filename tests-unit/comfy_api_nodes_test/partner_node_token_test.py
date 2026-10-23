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
from comfy_api_nodes.util.common_exceptions import ProcessingInterrupted
from comfy_api_nodes.util.download_helpers import download_url_to_bytesio


def _jwt(claims: dict) -> str:
    def enc(part: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(part).encode()).decode().rstrip("=")
    return f"{enc({'alg': 'ES256', 'typ': 'JWT'})}.{enc(claims)}.signature"


def _partner_token(expires_in: float, sid: str = "session-1", generation: int = 0, now: float | None = None) -> str:
    exp = (time.time() if now is None else now) + expires_in
    return _jwt({"aud": "comfy-partner-node", "sid": sid, "exp": exp, "gen": generation})


def _node(auth_token: str | None = None, api_key: str | None = None) -> type:
    hidden = SimpleNamespace(
        auth_token_comfy_org=auth_token, api_key_comfy_org=api_key, comfy_usage_source=None, unique_id="1"
    )
    return type("FakeApiNode", (), {"hidden": hidden})


class FakeComfyApi:
    """comfy-api stand-in: a partner route that accepts only ``accepted`` tokens, plus the renew route."""

    def __init__(
        self,
        accepted: set[str],
        renewed_token: str | None,
        renew_status: int = 200,
        accept_renewed: bool = True,
        renew_headers: dict[str, str] | None = None,
    ):
        self.accepted = accepted
        self.renewed_token = renewed_token
        self.renew_status = renew_status
        self.renew_headers = renew_headers
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
            return web.json_response({"message": "session revoked"}, status=self.renew_status, headers=self.renew_headers)
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


_CACHES = (
    _helpers._partner_token_lineages,
    _helpers._renewed_partner_tokens,
    _helpers._partner_token_renewals,
    _helpers._capped_partner_token_lineages,
    _helpers._partner_token_renew_not_before,
    _helpers._rejected_partner_tokens,
)


@pytest.fixture(autouse=True)
def _clear_renewed_tokens():
    for cache in _CACHES:
        cache.clear()
    yield
    for cache in _CACHES:
        cache.clear()


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
    now = time.time()
    rejected, fresh = _partner_token(5400, now=now), _partner_token(5400, generation=1, now=now)
    api = FakeComfyApi(accepted=set(), renewed_token=fresh)

    result = asyncio.run(_with_server(api, monkeypatch, lambda: _generate(_node(rejected))))

    assert result == {"ok": True}
    assert api.renew_calls == [f"Bearer {rejected}"]
    assert api.partner_calls == [
        (f"Bearer {rejected}", {"prompt": "fennec ears"}),
        (f"Bearer {fresh}", {"prompt": "fennec ears"}),
    ]


def test_401_resends_with_a_renewed_token_that_expires_sooner(monkeypatch):
    now = time.time()
    rejected, shorter = _partner_token(5400, now=now), _partner_token(5000, generation=1, now=now)
    api = FakeComfyApi(accepted=set(), renewed_token=shorter)

    result = asyncio.run(_with_server(api, monkeypatch, lambda: _generate(_node(rejected))))

    assert result == {"ok": True}
    assert api.renew_calls == [f"Bearer {rejected}"]
    assert [auth for auth, _ in api.partner_calls] == [f"Bearer {rejected}", f"Bearer {shorter}"]


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


@pytest.mark.parametrize(("renewed_exp_delta", "sent"), [(0, "renewed"), (-30, "snapshot")])
def test_renewal_capped_at_session_end_is_not_repeated(monkeypatch, renewed_exp_delta, sent):
    now = time.time()
    snapshot = _partner_token(120, now=now)
    renewed = _partner_token(120 + renewed_exp_delta, generation=1, now=now)
    api = FakeComfyApi(accepted={snapshot}, renewed_token=renewed)

    async def three_prompts():
        for _ in range(3):
            await _generate(_node(snapshot))

    asyncio.run(_with_server(api, monkeypatch, three_prompts))

    expected = renewed if sent == "renewed" else snapshot
    assert api.renew_calls == [f"Bearer {snapshot}"]
    assert [auth for auth, _ in api.partner_calls] == [f"Bearer {expected}"] * 3


def test_rate_limited_renewal_surfaces_the_auth_error_and_waits_for_retry_after(monkeypatch):
    rejected = _partner_token(5400)
    api = FakeComfyApi(accepted=set(), renewed_token=None, renew_status=429, renew_headers={"Retry-After": "120"})

    async def two_prompts():
        for _ in range(2):
            with pytest.raises(Exception, match="Unauthorized: Please login first"):
                await _generate(_node(rejected))

    asyncio.run(_with_server(api, monkeypatch, two_prompts))

    assert api.renew_calls == [f"Bearer {rejected}"]
    assert len(api.partner_calls) == 2


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


def test_prompts_carrying_the_same_snapshot_reuse_the_renewed_token(monkeypatch):
    snapshot, fresh = _partner_token(60), _partner_token(5400, generation=1)
    api = FakeComfyApi(accepted={snapshot}, renewed_token=fresh)

    async def two_prompts():
        await _generate(_node(snapshot))
        await _generate(_node(snapshot))

    asyncio.run(_with_server(api, monkeypatch, two_prompts))

    assert api.renew_calls == [f"Bearer {snapshot}"]
    assert [auth for auth, _ in api.partner_calls] == [f"Bearer {fresh}"] * 2


def test_a_token_claiming_another_session_id_never_gets_its_renewed_token(monkeypatch):
    victim, fresh = _partner_token(60), _partner_token(5400, generation=1)
    forged = _partner_token(30, generation=99)
    api = FakeComfyApi(accepted={victim}, renewed_token=fresh)

    async def victim_then_forged():
        await _generate(_node(victim))
        api.renewed_token, api.renew_status = None, 401
        with pytest.raises(Exception, match="Unauthorized: Please login first"):
            await _generate(_node(forged))
        api.renew_status = 200
        await _generate(_node(victim))

    asyncio.run(_with_server(api, monkeypatch, victim_then_forged))

    assert api.renew_calls == [f"Bearer {victim}", f"Bearer {forged}", f"Bearer {forged}"]
    assert [auth for auth, _ in api.partner_calls] == [f"Bearer {fresh}", f"Bearer {forged}", f"Bearer {fresh}"]


def test_renewing_a_renewed_token_keeps_the_snapshot_on_the_newest_token(monkeypatch):
    now = time.time()
    snapshot = _partner_token(60, now=now)
    first, second = _partner_token(200, generation=1, now=now), _partner_token(5400, generation=2, now=now)
    api = FakeComfyApi(accepted={snapshot}, renewed_token=first)

    async def renew_twice():
        await _generate(_node(snapshot))
        api.renewed_token = second
        await _generate(_node(snapshot))
        await _generate(_node(first))

    asyncio.run(_with_server(api, monkeypatch, renew_twice))

    assert api.renew_calls == [f"Bearer {snapshot}", f"Bearer {first}"]
    assert [auth for auth, _ in api.partner_calls] == [f"Bearer {first}", f"Bearer {second}", f"Bearer {second}"]


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


def test_cancelling_a_download_during_token_renewal_raises_processing_interrupted(monkeypatch):
    old, fresh = _partner_token(60), _partner_token(5400, generation=1)
    api = FakeComfyApi(accepted={fresh}, renewed_token=fresh)

    async def cancel_while_renewing():
        download = asyncio.create_task(
            download_url_to_bytesio("/proxy/partner/result", BytesIO(), cls=_node(old), max_retries=0)
        )
        while not api.renew_calls:
            await asyncio.sleep(0.005)
        download.cancel()
        with pytest.raises(ProcessingInterrupted):
            await download

    asyncio.run(_with_server(api, monkeypatch, cancel_while_renewing))

    assert api.partner_calls == []
