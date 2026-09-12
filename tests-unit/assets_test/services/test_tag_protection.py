import json
from types import SimpleNamespace

import pytest

from app.assets.api import routes

_RECORD_ID = "00000000-0000-0000-0000-000000000001"
_SYSTEM_TAG_ENVELOPE = {
    "error": {
        "code": "SYSTEM_TAG_FORBIDDEN",
        "message": "Tag 'missing' is system-managed and cannot be modified via the API",
        "details": {"tag": "missing"},
    }
}


class _JsonRequest:
    def __init__(self, tags: list[str]) -> None:
        self.match_info = {"id": _RECORD_ID}
        self._payload = {"tags": tags}

    async def json(self) -> dict[str, list[str]]:
        return self._payload


class _UserManager:
    def get_request_user_id(self, _request: _JsonRequest) -> str:
        return "test-user"


@pytest.mark.asyncio
async def test_add_missing_tag_returns_400(monkeypatch):
    monkeypatch.setattr(routes, "USER_MANAGER", _UserManager())
    monkeypatch.setattr(
        routes,
        "apply_tags",
        lambda **_kwargs: SimpleNamespace(added=[], already_present=[], total_tags=[]),
    )

    response = await routes.add_asset_tags.__wrapped__(_JsonRequest(["missing"]))

    assert response.status == 400
    assert json.loads(response.body) == _SYSTEM_TAG_ENVELOPE


@pytest.mark.asyncio
async def test_remove_missing_tag_returns_400(monkeypatch):
    monkeypatch.setattr(routes, "USER_MANAGER", _UserManager())
    monkeypatch.setattr(
        routes,
        "remove_tags",
        lambda **_kwargs: SimpleNamespace(removed=[], not_present=[], total_tags=[]),
    )

    response = await routes.delete_asset_tags.__wrapped__(_JsonRequest(["missing"]))

    assert response.status == 400
    assert json.loads(response.body) == _SYSTEM_TAG_ENVELOPE


@pytest.mark.asyncio
async def test_other_tags_unaffected(monkeypatch):
    monkeypatch.setattr(routes, "USER_MANAGER", _UserManager())
    calls: list[tuple[str, list[str]]] = []

    def apply_tags(**kwargs):
        calls.append(("add", kwargs["tags"]))
        return SimpleNamespace(
            added=kwargs["tags"], already_present=[], total_tags=kwargs["tags"]
        )

    def remove_tags(**kwargs):
        calls.append(("remove", kwargs["tags"]))
        return SimpleNamespace(
            removed=kwargs["tags"], not_present=[], total_tags=[], protected=[]
        )

    monkeypatch.setattr(routes, "apply_tags", apply_tags)
    monkeypatch.setattr(routes, "remove_tags", remove_tags)

    add_response = await routes.add_asset_tags.__wrapped__(_JsonRequest(["manual"]))
    remove_response = await routes.delete_asset_tags.__wrapped__(
        _JsonRequest(["manual"])
    )

    assert add_response.status == 200
    assert remove_response.status == 200
    assert calls == [("add", ["manual"]), ("remove", ["manual"])]


@pytest.mark.asyncio
async def test_remove_tags_response_exposes_protected_bucket(monkeypatch):
    monkeypatch.setattr(routes, "USER_MANAGER", _UserManager())
    monkeypatch.setattr(
        routes,
        "remove_tags",
        lambda **_kwargs: SimpleNamespace(
            removed=[], not_present=[], total_tags=["auto"], protected=["auto"]
        ),
    )

    response = await routes.delete_asset_tags.__wrapped__(_JsonRequest(["auto"]))

    assert response.status == 200
    body = json.loads(response.body)
    assert body["protected"] == ["auto"]
