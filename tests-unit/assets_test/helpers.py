"""Helper functions for assets integration tests."""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TypeAlias, TypedDict

import pytest
import requests
from aiohttp import web
from aiohttp.test_utils import make_mocked_request
import sqlalchemy as sa
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from app.assets import scanner
from app.assets.api import routes
from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import mark_content_missing
from app.assets.database.queries.records import create_content, create_record
from app.database.models import Base


class _AssetItemOptional(TypedDict, total=False):
    preview_id: str


class AssetItem(_AssetItemOptional):
    id: str
    name: str


class _AssetListBodyOptional(TypedDict, total=False):
    next_cursor: str


class AssetListBody(_AssetListBodyOptional):
    assets: list[AssetItem]
    total: int
    has_more: bool


class ErrorItem(TypedDict):
    code: str


class ErrorBody(TypedDict):
    error: ErrorItem


@dataclass(frozen=True, slots=True)
class RecordSeed:
    name: str
    tags: tuple[str, ...] = ()
    size_bytes: int = 0


RouteDatabase: TypeAlias = tuple[Engine, Session]


@pytest.fixture(autouse=True)
def autoclean_unit_test_assets() -> Iterator[None]:
    yield


@pytest.fixture
def route_database(monkeypatch: pytest.MonkeyPatch) -> Iterator[RouteDatabase]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(routes, "create_session", lambda: Session(engine))
    monkeypatch.setattr(routes, "_ASSETS_ENABLED", True)
    with Session(engine) as session:
        yield engine, session
    engine.dispose()


def enrich_via_prepare_apply(
    session: Session,
    *,
    file_path: str,
    content_id: str,
    record_id: str,
    extract_metadata: bool = True,
    compute_hash: bool = False,
    progress=None,
) -> bool:
    content = session.get(AssetContent, content_id)
    row = scanner.UnenrichedContent(
        content_id,
        record_id,
        file_path,
        content is not None and content.hash is None,
        observed_size_bytes=content.size_bytes if content is not None else 0,
        observed_mtime_ns=content.mtime_ns if content is not None else None,
    )
    prepared = scanner._prepare_enrichment(row, extract_metadata, compute_hash, progress)
    if prepared is None:
        return False
    applied = scanner._apply_enrichments(session, [prepared])
    session.commit()
    return bool(applied)


def seed_record(session: Session, seed: RecordSeed) -> Asset:
    content = create_content(
        session,
        path=f"/output/{uuid.uuid4()}-{seed.name}",
        size_bytes=seed.size_bytes,
    )
    return create_record(
        session,
        content_id=content.id,
        name=seed.name,
        mime_type="image/png",
        tags=seed.tags,
    )


@pytest.fixture
def sortable_record_ids(route_database: RouteDatabase) -> tuple[str, str]:
    _, session = route_database
    older = seed_record(session, RecordSeed("z.png", ("sort-case",), 100))
    newer = seed_record(session, RecordSeed("a.png", ("sort-case",), 200))
    base_time = datetime(2026, 1, 1)
    older.created_at = base_time
    newer.created_at = base_time + timedelta(days=1)
    newer.updated_at = base_time
    older.updated_at = base_time + timedelta(days=1)
    older.last_access_time = base_time
    newer.last_access_time = base_time + timedelta(days=1)
    session.commit()
    return newer.id, older.id


async def request_assets(query: str = "") -> web.StreamResponse:
    suffix = f"?{query}" if query else ""
    return await routes.list_assets_route(
        make_mocked_request("GET", f"/api/assets{suffix}")
    )


def asset_list_body(response: web.StreamResponse) -> AssetListBody:
    assert isinstance(response, web.Response)
    body = response.body
    assert isinstance(body, bytes | bytearray)
    return json.loads(body)


def error_body(response: web.StreamResponse) -> ErrorBody:
    assert isinstance(response, web.Response)
    body = response.body
    assert isinstance(body, bytes | bytearray)
    return json.loads(body)


def trigger_sync_seed_assets(session: requests.Session, base_url: str) -> None:
    """Force a synchronous sync/seed pass by calling the seed endpoint with wait=true.

    Retries on 409 (already running) until the previous scan finishes.
    """
    deadline = time.monotonic() + 60
    while True:
        r = session.post(
            base_url + "/api/assets/seed?wait=true",
            json={"roots": ["models", "input", "output"]},
            timeout=60,
        )
        if r.status_code != 409:
            assert r.status_code == 200, f"seed endpoint returned {r.status_code}: {r.text}"
            return
        if time.monotonic() > deadline:
            raise TimeoutError("seed endpoint stuck in 409 (already running)")
        time.sleep(0.25)


def get_asset_filename(asset_hash: str, extension: str) -> str:
    return asset_hash.removeprefix("blake3:") + extension


def assert_hash_fields_consistent(
    body: Mapping[str, str | None],
    expected_hash: str | None = None,
) -> None:
    """Assert hash and asset_hash invariants on an Asset response.

    Both must be present or both absent (so a regression that drops only one
    is caught). When present, they must equal each other and, if expected_hash
    is provided, must equal that value.
    """
    hash_present = "hash" in body
    asset_hash_present = "asset_hash" in body
    assert hash_present == asset_hash_present, (
        f"hash and asset_hash must both be present or both absent: "
        f"hash present={hash_present}, asset_hash present={asset_hash_present}"
    )
    if hash_present:
        h = body["hash"]
        ah = body["asset_hash"]
        assert h == ah, f"hash and asset_hash must match: hash={h!r}, asset_hash={ah!r}"
        if expected_hash is not None:
            assert h == expected_hash, (
                f"hash must equal expected: got {h!r}, expected {expected_hash!r}"
            )


def sync_prefixes_in_session(
    session: Session,
    prefixes: list[str],
    collect_existing_paths: bool = False,
    progress=None,
    pending_verification_ids: list[str] | None = None,
    diagnostics=None,
) -> set[str] | None:
    """Observe and apply a prefix sync inside one caller-supplied session.

    Production drives this through `sync_root_safely`, which owns its own
    chunked transactions. Tests that need to inspect the result inside their own
    session compose the same two live calls here rather than production carrying
    a synchronous entry point nothing ships against.
    """
    if not prefixes:
        return set() if collect_existing_paths else None

    collected = diagnostics if diagnostics is not None else []
    observations, survivors = scanner.observe_references_on_filesystem(
        prefixes, diagnostics=collected, session=session
    )
    scanner.apply_reference_observations(
        session, observations, pending_verification_ids=pending_verification_ids
    )
    if diagnostics is None:
        scanner._publish_reference_diagnostics(collected, progress)
    return survivors if collect_existing_paths else None


def mark_contents_missing_outside_prefixes_in_session(
    session: Session, prefixes: list[str]
) -> int:
    """Mark every live content outside `prefixes` missing, in the caller's session."""
    contents = session.scalars(
        sa.select(AssetContent).where(AssetContent.is_missing.is_(False))
    )
    missing = [
        content
        for content in contents
        if not scanner._is_under_prefixes(content.path, prefixes)
    ]
    for content in missing:
        mark_content_missing(session, content.id)
    return len(missing)
