"""Every v2 operation but getHealth and assetFromPath resolves the caller first: unknown and system users get 403 forbidden."""

import pytest

from .contract import OPERATIONS, validate

EXEMPT = {("GET", "/api/v2/health"): 200, ("POST", "/api/v2/assets/from-path"): 501}


def url(path: str) -> str:
    return path.replace("{id}", "00000000-0000-4000-8000-000000000000").replace("{hash}", "blake3:" + "0" * 64)


@pytest.fixture(autouse=True)
def multi_user(fake_server):
    fake_server.user_manager.multi_user = True


@pytest.mark.asyncio
@pytest.mark.parametrize("user", ["nobody", "__system"])
@pytest.mark.parametrize("method, path", sorted(OPERATIONS - EXEMPT.keys()))
async def test_unknown_user_is_forbidden(conforming_client, v2_dirs, method, path, user):
    if method == "HEAD":
        # A HEAD response has no body to check against the declared envelope.
        resp = await conforming_client.session.head(conforming_client.make_url(url(path)), headers={"comfy-user": user})
        assert resp.status == 403
        return
    resp = await conforming_client.request(method, url(path), headers={"comfy-user": user})
    body = await resp.json()
    assert resp.status == 403, body
    validate("ErrorEnvelope", body)
    assert body["error"]["code"] == "forbidden"


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path", sorted(EXEMPT))
async def test_exempt_operations_ignore_the_user(conforming_client, v2_dirs, method, path):
    resp = await conforming_client.request(method, path, headers={"comfy-user": "nobody"})
    assert resp.status == EXEMPT[method, path], await resp.text()
