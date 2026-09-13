from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from sqlalchemy.orm import Session

from app.assets.database.models import Asset, Tag
from app.assets.database.queries.records import (
    RecordPageSpec,
    create_content,
    create_record,
    list_records_page,
    mark_content_missing,
)
from app.assets.database.queries.tags import (
    list_tag_counts_for_filtered_assets,
    list_tags_with_usage,
)


def _seed_asset(
    session: Session, path: str, name: str, tags: Sequence[str]
) -> Asset:
    content = create_content(session, path=path)
    return create_record(session, content_id=content.id, name=name, tags=list(tags))


def test_usage_counts_match_seeded_asset_tag_rows(session: Session) -> None:
    _seed_asset(session, "/models/a", "a", ["models", "checkpoint"])
    _seed_asset(session, "/models/b", "b", ["models"])
    _seed_asset(session, "/models/c", "c", ["lora"])
    session.commit()

    rows, total = list_tags_with_usage(session)

    assert dict(rows) == {"models": 2, "checkpoint": 1, "lora": 1}
    assert total == 3


def test_prefix_filters_tags_by_leading_substring(session: Session) -> None:
    _seed_asset(session, "/m/1", "1", ["model", "mask", "zebra"])
    session.commit()

    rows, total = list_tags_with_usage(session, prefix="ma")

    assert [name for name, _ in rows] == ["mask"]
    assert total == 1


def test_limit_and_offset_paginate_over_real_total(session: Session) -> None:
    for index in range(5):
        _seed_asset(session, f"/p/{index}", str(index), [f"tag{index}"])
    session.commit()

    rows, total = list_tags_with_usage(session, limit=2, offset=1, order="name_asc")

    assert [name for name, _ in rows] == ["tag1", "tag2"]
    assert total == 5


def test_include_zero_false_drops_unused_tags_and_shrinks_total(session: Session) -> None:
    _seed_asset(session, "/z/1", "1", ["used"])
    session.add(Tag(name="unused"))
    session.commit()

    with_zero, total_with = list_tags_with_usage(session, include_zero=True)
    without_zero, total_without = list_tags_with_usage(session, include_zero=False)

    assert ("unused", 0) in with_zero
    assert total_with == 2
    assert [name for name, _ in without_zero] == ["used"]
    assert total_without == 1


def test_order_is_count_desc_then_name_asc_or_name_asc(session: Session) -> None:
    _seed_asset(session, "/o/1", "1", ["gamma"])
    _seed_asset(session, "/o/2", "2", ["gamma", "beta"])
    _seed_asset(session, "/o/3", "3", ["gamma", "beta", "alpha"])
    session.commit()

    by_count, _ = list_tags_with_usage(session, order="count_desc")
    by_name, _ = list_tags_with_usage(session, order="name_asc")

    assert [name for name, _ in by_count] == ["gamma", "beta", "alpha"]
    assert [name for name, _ in by_name] == ["alpha", "beta", "gamma"]


def test_missing_content_counts_all_its_tags_including_missing(
    session: Session,
) -> None:
    _seed_asset(session, "/live", "live", ["foo"])
    gone = create_content(session, path="/gone")
    create_record(session, content_id=gone.id, name="gone", tags=["foo"])
    mark_content_missing(session, gone.id)
    session.commit()

    counts = dict(list_tags_with_usage(session)[0])

    assert counts["foo"] == 2
    assert counts["missing"] == 1


def test_histogram_counts_every_tag_on_the_filtered_assets(session: Session) -> None:
    _seed_asset(session, "/h/a", "a", ["models", "checkpoint"])
    _seed_asset(session, "/h/b", "b", ["models", "lora"])
    _seed_asset(session, "/h/c", "c", ["other"])
    session.commit()

    histogram = list_tag_counts_for_filtered_assets(session, include_tags=["models"])

    assert histogram == {"models": 2, "checkpoint": 1, "lora": 1}


def test_histogram_includes_missing_content_assets(session: Session) -> None:
    _seed_asset(session, "/hm/live", "live", ["models"])
    gone = create_content(session, path="/hm/gone")
    create_record(session, content_id=gone.id, name="gone", tags=["models"])
    mark_content_missing(session, gone.id)
    session.commit()

    histogram = list_tag_counts_for_filtered_assets(session, include_tags=["models"])

    assert histogram == {"models": 2, "missing": 1}


def test_refine_and_list_share_the_same_filtered_asset_set(session: Session) -> None:
    kept = _seed_asset(session, "/s/a", "checkpoint-a", ["models", "checkpoint"])
    _seed_asset(session, "/s/b", "checkpoint-b", ["models", "archived"])
    _seed_asset(session, "/s/c", "lora-c", ["models"])
    gone = create_content(session, path="/s/d")
    missing_match = create_record(
        session, content_id=gone.id, name="check-d", tags=["models", "checkpoint"]
    )
    mark_content_missing(session, gone.id)
    session.commit()

    spec = RecordPageSpec(
        all_tags=("models",),
        none_tags=("archived",),
        name_contains="check",
        limit=100,
    )
    records, tag_map, _ = list_records_page(session, spec)
    expected = Counter(
        tag for record in records for tag in tag_map.get(record.id, [])
    )
    histogram = list_tag_counts_for_filtered_assets(
        session,
        include_tags=["models"],
        exclude_tags=["archived"],
        name_contains="check",
        limit=100,
    )

    assert {record.id for record in records} == {kept.id, missing_match.id}
    assert histogram == dict(expected)
    assert histogram == {"models": 2, "checkpoint": 2, "missing": 1}


def test_histogram_any_tags_matches_the_list_union(session: Session) -> None:
    red = _seed_asset(session, "/an/a", "a", ["red"])
    blue = _seed_asset(session, "/an/b", "b", ["blue"])
    _seed_asset(session, "/an/c", "c", ["green"])
    session.commit()

    records, _, _ = list_records_page(
        session, RecordPageSpec(any_tags=("red", "blue"), limit=100)
    )
    histogram = list_tag_counts_for_filtered_assets(session, any_tags=["red", "blue"])

    assert {record.id for record in records} == {red.id, blue.id}
    assert histogram == {"red": 1, "blue": 1}


def test_list_refine_and_tags_agree_on_counts_including_missing(
    session: Session,
) -> None:
    _seed_asset(session, "/agree/live", "live", ["models", "checkpoint"])
    gone = create_content(session, path="/agree/gone")
    create_record(session, content_id=gone.id, name="gone", tags=["models", "lora"])
    mark_content_missing(session, gone.id)
    session.commit()

    records, tag_map, total = list_records_page(
        session, RecordPageSpec(all_tags=("models",), limit=100)
    )
    list_tally = dict(
        Counter(tag for record in records for tag in tag_map.get(record.id, []))
    )
    refine = list_tag_counts_for_filtered_assets(session, include_tags=["models"])
    global_usage = dict(list_tags_with_usage(session)[0])

    expected = {"models": 2, "checkpoint": 1, "lora": 1, "missing": 1}
    assert total == 2
    assert list_tally == expected
    assert refine == expected
    assert global_usage == expected
