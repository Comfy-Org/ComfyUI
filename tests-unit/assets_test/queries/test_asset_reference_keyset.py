"""Keyset-pagination tiebreaker tests for list_references_page.

When multiple rows share the same primary sort value (e.g. four assets
created in the same microsecond), the secondary `ORDER BY id` is what keeps
keyset pagination from losing or repeating rows. This file exercises that
branch directly against an in-memory SQLite session — engineering identical
timestamps via HTTP is unreliable enough that we work at the query layer.
"""
import uuid
from datetime import datetime

import pytest
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetReference
from app.assets.database.queries.asset_reference import list_references_page


def _make_ref(session: Session, created_at: datetime, name: str, owner: str = "") -> AssetReference:
    asset = Asset(hash=f"blake3:{uuid.uuid4().hex}", size_bytes=1024)
    session.add(asset)
    session.flush()
    ref = AssetReference(
        id=str(uuid.uuid4()),
        asset_id=asset.id,
        owner_id=owner,
        name=name,
        file_path=f"/tmp/{name}",
        created_at=created_at,
        updated_at=created_at,
        last_access_time=created_at,
        is_missing=False,
    )
    session.add(ref)
    return ref


@pytest.mark.parametrize("order", ["desc", "asc"])
def test_tiebreaker_walks_duplicate_sort_values(session: Session, order: str):
    """Four rows with the SAME created_at must paginate cleanly under cursor
    mode — no row dropped, no row repeated, despite the primary sort column
    being non-discriminating.
    """
    shared_ts = datetime(2024, 5, 20, 12, 0, 0)  # naive UTC, like the DB stores
    refs = [_make_ref(session, shared_ts, f"tie_{i}.png") for i in range(4)]
    session.commit()

    expected_ids = sorted([r.id for r in refs], reverse=(order == "desc"))

    # Walk the cursor by hand: page size 2, take 3 pages (2 + 2 + 0).
    seen: list[str] = []
    after_value = None
    after_id = None
    for _ in range(4):  # generous loop bound; ought to be 2 iterations
        page, _tag_map, _total = list_references_page(
            session,
            limit=2,
            sort="created_at",
            order=order,
            after_cursor_value=after_value,
            after_cursor_id=after_id,
        )
        if not page:
            break
        seen.extend(p.id for p in page)
        # Use the last row's (created_at, id) as the next cursor input.
        last = page[-1]
        after_value, after_id = last.created_at, last.id
        if len(page) < 2:
            break

    assert seen == expected_ids, (
        f"keyset tiebreaker failed for order={order}: expected {expected_ids}, got {seen}"
    )


def test_tiebreaker_no_duplicates_under_mixed_collisions(session: Session):
    """Some rows share a timestamp, some don't. The cursor must still walk
    every row exactly once regardless of where ties sit relative to a
    page boundary."""
    t1 = datetime(2024, 5, 20, 12, 0, 0)
    t2 = datetime(2024, 5, 20, 12, 0, 1)
    layout = [t1, t1, t1, t2, t2]  # three rows at t1, two at t2
    refs = [_make_ref(session, ts, f"mix_{i}.png") for i, ts in enumerate(layout)]
    session.commit()

    all_ids = {r.id for r in refs}
    seen_set: set[str] = set()
    seen_list: list[str] = []
    after_value = None
    after_id = None
    for _ in range(6):
        page, _, _ = list_references_page(
            session,
            limit=2,
            sort="created_at",
            order="desc",
            after_cursor_value=after_value,
            after_cursor_id=after_id,
        )
        if not page:
            break
        for p in page:
            assert p.id not in seen_set, f"duplicate row {p.id} appeared in cursor walk"
            seen_set.add(p.id)
            seen_list.append(p.id)
        last = page[-1]
        after_value, after_id = last.created_at, last.id
        if len(page) < 2:
            break

    assert seen_set == all_ids, f"missing rows: expected {all_ids}, got {seen_set}"
