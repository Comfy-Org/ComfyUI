"""``api.request_failed`` tagged lines on the assets HTTP handlers.

Each of the seven handlers that swallow an unexpected exception emits one
tagged line carrying the handler's own name and the exception's TYPE. The
exception messages deliberately contain a filesystem path here: the tagged
line must never carry it, while the existing human-readable log line still
does.

The handlers are driven directly (aiohttp's ``make_mocked_request``, the
pattern ``services/test_api_routes_b.py`` already uses) with the service call
monkeypatched to raise, so no database and no running server is involved.
"""

import json
import re
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import streams, web
from aiohttp.test_utils import make_mocked_request

from app.assets.api import routes, upload
from app.assets.api.schemas_in import UploadError
from app.assets.event_log import ROUTES as VOCABULARY_ROUTES
from app.assets.event_log import TAG

EVENT_LINE_PATTERN = re.compile(
    rf"^{re.escape(TAG)} (?P<event>[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*) "
    r"(?P<fields>\{.*\})$"
)

# The closed set of handlers that emit api.request_failed, verbatim function
# names. `route` is always one of these static strings - never a URL, never a
# path parameter.
EXPECTED_ROUTES = frozenset(
    {
        "get_asset_route",
        "upload_asset",
        "update_asset_route",
        "delete_asset_route",
        "add_asset_tags",
        "delete_asset_tags",
        "parse_multipart_upload",
    }
)

# A message shaped like the ones real exceptions carry: an absolute path and a
# filename, both of which must stay out of the tagged line.
SECRET_PATH = "/home/someone/models/checkpoints/secret-model.safetensors"
BOOM = f"disk read failed: {SECRET_PATH}"
ASSET_ID = str(uuid.UUID(int=0x5EED))
VALID_HASH = "blake3:" + "ab" * 32


@pytest.fixture(autouse=True)
def autoclean_unit_test_assets():
    """Shadow the conftest autouse fixture; these tests need no ComfyUI boot."""
    yield


@pytest.fixture(autouse=True)
def enabled_assets_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes, "_ASSETS_ENABLED", True)
    monkeypatch.setattr(
        routes, "USER_MANAGER", Mock(get_request_user_id=Mock(return_value="tenant-1"))
    )


def tagged_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith(TAG)
    ]


def events_named(caplog: pytest.LogCaptureFixture, name: str) -> list[dict]:
    matched = []
    for line in tagged_lines(caplog):
        match = EVENT_LINE_PATTERN.match(line)
        if match is not None and match.group("event") == name:
            matched.append(json.loads(match.group("fields")))
    return matched


def _stream(body: bytes) -> streams.StreamReader:
    reader = streams.StreamReader(Mock(_reading_paused=False), 2**16)
    reader.feed_data(body)
    reader.feed_eof()
    return reader


def _json_request(method: str, path: str, body: dict) -> web.Request:
    return make_mocked_request(
        method,
        path,
        headers={"Content-Type": "application/json"},
        match_info={"id": ASSET_ID},
        payload=_stream(json.dumps(body).encode()),
    )


def _plain_request(method: str, path: str) -> web.Request:
    return make_mocked_request(method, path, match_info={"id": ASSET_ID})


def _multipart_hash_request() -> web.Request:
    boundary = "----assetsboundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="hash"\r\n\r\n'
        f"{VALID_HASH}\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    return make_mocked_request(
        "POST",
        "/api/assets",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        payload=_stream(body),
    )


def _raise(*_args, **_kwargs):
    raise RuntimeError(BOOM)


async def _drive_get_asset_route(monkeypatch: pytest.MonkeyPatch) -> web.Response:
    monkeypatch.setattr(routes, "get_asset_detail", _raise)
    return await routes.get_asset_route(_plain_request("GET", f"/api/assets/{ASSET_ID}"))


async def _drive_upload_asset(monkeypatch: pytest.MonkeyPatch) -> web.Response:
    parsed = SimpleNamespace(
        file_present=False,
        file_written=0,
        file_client_name=None,
        tmp_path=None,
        tags_raw=[],
        provided_name=None,
        user_metadata_raw=None,
        provided_hash=VALID_HASH,
        provided_hash_exists=True,
        provided_mime_type=None,
        provided_preview_id=None,
    )
    monkeypatch.setattr(routes, "parse_multipart_upload", AsyncMock(return_value=parsed))
    monkeypatch.setattr(routes, "create_from_hash", _raise)
    return await routes.upload_asset(_plain_request("POST", "/api/assets"))


async def _drive_update_asset_route(monkeypatch: pytest.MonkeyPatch) -> web.Response:
    monkeypatch.setattr(routes, "update_asset_metadata", _raise)
    request = _json_request("PUT", f"/api/assets/{ASSET_ID}", {"name": "renamed"})
    return await routes.update_asset_route(request)


async def _drive_delete_asset_route(monkeypatch: pytest.MonkeyPatch) -> web.Response:
    monkeypatch.setattr(routes, "delete_asset_reference", _raise)
    request = _plain_request("DELETE", f"/api/assets/{ASSET_ID}")
    return await routes.delete_asset_route(request)


async def _drive_add_asset_tags(monkeypatch: pytest.MonkeyPatch) -> web.Response:
    monkeypatch.setattr(routes, "apply_tags", _raise)
    request = _json_request("POST", f"/api/assets/{ASSET_ID}/tags", {"tags": ["blue"]})
    return await routes.add_asset_tags(request)


async def _drive_delete_asset_tags(monkeypatch: pytest.MonkeyPatch) -> web.Response:
    monkeypatch.setattr(routes, "remove_tags", _raise)
    request = _json_request("DELETE", f"/api/assets/{ASSET_ID}/tags", {"tags": ["blue"]})
    return await routes.delete_asset_tags(request)


ROUTE_DRIVERS = {
    "get_asset_route": _drive_get_asset_route,
    "upload_asset": _drive_upload_asset,
    "update_asset_route": _drive_update_asset_route,
    "delete_asset_route": _drive_delete_asset_route,
    "add_asset_tags": _drive_add_asset_tags,
    "delete_asset_tags": _drive_delete_asset_tags,
}


def test_the_seven_handler_names_are_the_event_log_route_vocabulary() -> None:
    """Guards the parametrization: a renamed handler must break here first."""
    assert EXPECTED_ROUTES == VOCABULARY_ROUTES
    assert set(ROUTE_DRIVERS) == EXPECTED_ROUTES - {"parse_multipart_upload"}
    for name, driver in ROUTE_DRIVERS.items():
        assert getattr(routes, name).__name__ == name
        assert driver.__name__ == f"_drive_{name}"


@pytest.mark.parametrize("route", sorted(ROUTE_DRIVERS))
@pytest.mark.asyncio
async def test_route_failure_emits_one_tagged_line(
    route: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO"):
        await ROUTE_DRIVERS[route](monkeypatch)

    assert events_named(caplog, "api.request_failed") == [
        {"error_type": "RuntimeError", "route": route}
    ]


@pytest.mark.parametrize("route", sorted(ROUTE_DRIVERS))
@pytest.mark.asyncio
async def test_route_failure_response_is_unchanged(
    route: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The emit rides alongside the existing behaviour, it does not alter it."""
    with caplog.at_level("INFO"):
        response = await ROUTE_DRIVERS[route](monkeypatch)

    assert isinstance(response, web.Response)
    assert response.status == 500
    body = json.loads(response.body)
    assert body["error"]["code"] == "INTERNAL"


@pytest.mark.parametrize("route", sorted(ROUTE_DRIVERS))
@pytest.mark.asyncio
async def test_route_failure_tagged_line_leaks_no_request_detail(
    route: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO"):
        await ROUTE_DRIVERS[route](monkeypatch)

    lines = tagged_lines(caplog)
    # Assert presence FIRST: "the path is absent" passes vacuously otherwise.
    assert len(events_named(caplog, "api.request_failed")) == 1
    for leak in (SECRET_PATH, "secret-model.safetensors", ASSET_ID, VALID_HASH, BOOM):
        assert all(leak not in line for line in lines)

    # The human-readable line is untouched and still carries the detail.
    human = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "ERROR" and not record.getMessage().startswith(TAG)
    ]
    assert human, "the existing logging.exception line must stay"


@pytest.mark.asyncio
async def test_parse_multipart_upload_failure_emits_one_tagged_line(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO"):
        with pytest.raises(UploadError) as raised:
            await upload.parse_multipart_upload(
                _multipart_hash_request(), check_hash_exists=_raise
            )

    assert raised.value.status == 500
    assert raised.value.code == "HASH_CHECK_FAILED"
    assert events_named(caplog, "api.request_failed") == [
        {"error_type": "RuntimeError", "route": "parse_multipart_upload"}
    ]


@pytest.mark.asyncio
async def test_parse_multipart_upload_tagged_line_leaks_no_hash_or_path(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO"):
        with pytest.raises(UploadError):
            await upload.parse_multipart_upload(
                _multipart_hash_request(), check_hash_exists=_raise
            )

    lines = tagged_lines(caplog)
    assert len(events_named(caplog, "api.request_failed")) == 1
    for leak in (SECRET_PATH, "secret-model.safetensors", VALID_HASH, BOOM):
        assert all(leak not in line for line in lines)
