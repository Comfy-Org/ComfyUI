"""assetFromPath is reserved: always 501 not_implemented, nothing registered."""

import logging
import os

import pytest
from sqlalchemy import func, select

from app.assets.database.models import Asset
from app.database.db import create_session

from .contract import response_schema, validate

PATH = "/api/v2/assets/from-path"
NOT_IMPL = "Registering a host file is not implemented by ComfyUI; upload it with POST /api/v2/assets instead."


@pytest.fixture(autouse=True)
def no_aiohttp_errors(caplog):
    yield
    bad = [r for r in caplog.get_records("call") if r.levelno >= logging.ERROR and r.name in ("aiohttp.server", "aiohttp.web")]
    assert not bad, [r.getMessage() for r in bad]


def asset_count():
    with create_session() as session:
        return session.scalar(select(func.count()).select_from(Asset))


def assert_501(body, status):
    assert status == 501, body
    validate(response_schema("assetFromPath", 501), body)
    validate("ErrorEnvelope", body)
    assert body["error"]["code"] == "not_implemented"
    assert body["error"]["message"] == NOT_IMPL


@pytest.mark.asyncio
async def test_valid_body_is_501_and_registers_nothing(conforming_client, v2_dirs):
    target = v2_dirs / "input" / "a.png"
    target.write_bytes(b"\x89PNG-bytes")
    os.utime(target, (1_000_000_000, 1_000_000_000))
    before_count = asset_count()
    before_stat = target.stat()

    resp = await conforming_client.post(PATH, json={"path": str(target)})
    body = await resp.json()

    assert_501(body, resp.status)
    assert asset_count() == before_count
    assert target.read_bytes() == b"\x89PNG-bytes"
    after_stat = target.stat()
    assert (after_stat.st_mtime_ns, after_stat.st_size) == (before_stat.st_mtime_ns, before_stat.st_size)


@pytest.mark.parametrize("kwargs", [{}, {"data": b""}, {"data": b"not json", "headers": {"Content-Type": "application/json"}}])
@pytest.mark.asyncio
async def test_empty_or_non_json_body_is_501(conforming_client, v2_dirs, kwargs):
    before_count = asset_count()
    resp = await conforming_client.post(PATH, **kwargs)
    assert_501(await resp.json(), resp.status)
    assert asset_count() == before_count


@pytest.mark.asyncio
async def test_unknown_user_is_still_501(conforming_client, v2_dirs, fake_server):
    fake_server.user_manager.multi_user = True
    resp = await conforming_client.post(PATH, json={"path": "/etc/hostname"}, headers={"comfy-user": "nobody"})
    assert_501(await resp.json(), resp.status)


@pytest.mark.asyncio
async def test_get_from_path_is_asset_id_not_found(conforming_client, v2_dirs):
    resp = await conforming_client.get(PATH)
    body = await resp.json()
    assert resp.status == 404, body
    validate("ErrorEnvelope", body)
    assert body["error"]["code"] == "not_found"
