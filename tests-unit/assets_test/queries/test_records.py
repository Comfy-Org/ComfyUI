import pytest
from sqlalchemy import create_engine, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import (
    RecordPageSpec,
    create_content,
    create_record,
    delete_record,
    get_record_by_id,
    list_records_page,
    mark_content_missing,
)
from app.database.models import Base


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with Session(engine) as sess:
        yield sess


def test_listing_includes_missing_by_default_with_missing_tag(session):
    content = create_content(session, path="/tmp/f1")
    record = create_record(session, content_id=content.id, name="test")
    mark_content_missing(session, content.id)
    session.commit()

    results, tag_map, total = list_records_page(session, RecordPageSpec())

    assert any(r.id == record.id for r in results)
    assert total == 1
    assert "missing" in tag_map.get(record.id, [])


def test_listing_excludes_missing_with_filter(session):
    content = create_content(session, path="/tmp/f2")
    record = create_record(session, content_id=content.id, name="test2")
    mark_content_missing(session, content.id)
    session.commit()

    results, _, total = list_records_page(
        session,
        RecordPageSpec(none_tags=("missing",)),
    )

    assert not any(r.id == record.id for r in results), "Missing record should be excluded"
    assert total == 0


def test_delete_record_never_deletes_referenced_preview(session):
    preview_content = create_content(session, path="/tmp/preview")
    preview_record = create_record(session, content_id=preview_content.id, name="preview")
    content1 = create_content(session, path="/tmp/f3")
    content2 = create_content(session, path="/tmp/f4")
    r1 = create_record(session, content_id=content1.id, name="r1")
    r2 = create_record(session, content_id=content2.id, name="r2")

    session.execute(update(Asset).where(Asset.id == r1.id).values(preview_id=preview_record.id))
    session.execute(update(Asset).where(Asset.id == r2.id).values(preview_id=preview_record.id))
    session.commit()

    delete_record(session, r1.id)
    session.commit()
    assert get_record_by_id(session, preview_record.id) is not None, "Preview must survive"

    delete_record(session, r2.id)
    session.commit()
    assert get_record_by_id(session, preview_record.id) is not None, (
        "Preview must survive even with no referrers"
    )

    content_row = session.execute(
        select(AssetContent).where(AssetContent.id == preview_content.id)
    ).scalar_one_or_none()
    assert content_row is not None, "Preview content row should survive (D-3 floor)"


def test_concurrent_create_content_same_path(tmp_path):
    db_path = str(tmp_path / "concurrent.db")
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)

    with Session(engine) as s1, Session(engine) as s2:
        create_content(s1, path="/tmp/shared")
        s1.commit()
        create_content(s2, path="/tmp/shared")
        s2.commit()

    with Session(engine) as s:
        live_rows = list(
            s.execute(
                select(AssetContent).where(
                    AssetContent.path == "/tmp/shared",
                    AssetContent.is_missing.is_(False),
                )
            ).scalars()
        )
    assert len(live_rows) == 1, f"Expected exactly one live row, got {len(live_rows)}"


def test_create_content_check_violation_surfaces_the_constraint_error(session):
    with pytest.raises(IntegrityError):
        create_content(session, path="/tmp/negative-size", size_bytes=-1)


def test_create_content_check_violation_on_mtime_surfaces(session):
    with pytest.raises(IntegrityError):
        create_content(session, path="/tmp/negative-mtime", size_bytes=0, mtime_ns=-1)


def test_create_content_uniqueness_race_returns_existing_live_row(session):
    first = create_content(session, path="/tmp/race")
    second = create_content(session, path="/tmp/race")

    assert second.id == first.id
