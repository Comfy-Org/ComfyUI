from contextlib import contextmanager
from datetime import datetime
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as SASession

import app.assets.database.queries.records as records_module
import app.assets.services.asset_management as asset_management_module
import app.assets.services.tagging as tagging_module
from app.assets.database.models import Asset, AssetTag, Base, Tag
from app.assets.database.queries.records import (
    create_content,
    create_record,
    mark_content_missing,
)
from app.assets.services.asset_management import update_asset_metadata
from app.assets.services.tagging import apply_tags

STALE = datetime(2020, 1, 1, 0, 0, 0)
RACED = "raced"
UNRELATED = "unrelated"


def _file_backed_engine(db_path):
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 0.2},
    )
    Base.metadata.create_all(engine)
    return engine


def _seed_record(session, path: str, tags=None) -> str:
    content = create_content(session, path)
    record = create_record(session, content.id, "fixture", tags=tags)
    session.commit()
    session.execute(update(Asset).where(Asset.id == record.id).values(updated_at=STALE))
    session.commit()
    return record.id


def _updated_at(session, record_id: str) -> datetime:
    session.expire_all()
    return session.scalar(select(Asset.updated_at).where(Asset.id == record_id))


def _tag_names(session, record_id: str) -> list[str]:
    return sorted(
        session.scalars(select(AssetTag.tag_name).where(AssetTag.asset_id == record_id)).all()
    )


def test_apply_tags_loses_a_tag_race_without_raising_and_reports_it_honestly(tmp_path):
    engine = _file_backed_engine(tmp_path / "tag_race.db")
    with SASession(engine) as seed_session:
        record_id = _seed_record(seed_session, "/tmp/tag-race-fixture", tags=[UNRELATED])

    def connection_a_commits_the_raced_tag() -> None:
        with SASession(engine) as connection_a:
            connection_a.add(Tag(name=RACED))
            connection_a.flush()
            connection_a.add(
                AssetTag(asset_id=record_id, tag_name=RACED, origin="manual")
            )
            connection_a.commit()

    fired: list[bool] = []

    @contextmanager
    def racing_session_factory():
        with SASession(engine) as session_b:
            real_add = session_b.add

            def add_racing_the_winner(instance, *args, **kwargs):
                if isinstance(instance, Tag) and instance.name == RACED and not fired:
                    fired.append(True)
                    connection_a_commits_the_raced_tag()
                return real_add(instance, *args, **kwargs)

            session_b.add = add_racing_the_winner
            yield session_b

    with patch("app.assets.services.tagging.create_session", racing_session_factory):
        result = apply_tags(record_id, [RACED])

    assert fired, "the interleave never fired; the test proves nothing"
    assert result.added == [], (
        "session B lost the race; it inserted nothing and must not claim it did"
    )
    assert result.already_present == [RACED], (
        "already_present reports the REQUESTED tags that were already there — never "
        "unrelated pre-existing tags the caller did not ask about"
    )
    assert sorted(result.total_tags) == [RACED, UNRELATED]

    with SASession(engine) as check:
        assert _tag_names(check, record_id) == [RACED, UNRELATED]
        assert _updated_at(check, record_id) == STALE, (
            "a race loser changed no link, so it is not an edit and must not move updated_at"
        )


def test_ensure_tag_link_reraises_when_the_parent_asset_is_missing(db_engine_fk):
    with SASession(db_engine_fk) as session:
        session.add(Tag(name="orphan-link"))
        session.flush()

        with pytest.raises(IntegrityError):
            records_module.ensure_tag_link(
                session,
                asset_id="no-such-asset",
                tag_name="orphan-link",
                origin="manual",
            )


def _drive_create_record(session, record_id):
    content = create_content(session, "/tmp/four-sites-create-record")
    create_record(session, content.id, "created", tags=["spied"])


def _drive_mark_content_missing(session, record_id):
    content_id = session.scalar(select(Asset.content_id).where(Asset.id == record_id))
    mark_content_missing(session, content_id)


def _drive_update_asset_metadata(session, record_id):
    update_asset_metadata(record_id, tags=["spied"])


def _drive_apply_tags(session, record_id):
    apply_tags(record_id, ["spied"])


@pytest.mark.parametrize(
    ("target_module", "drive"),
    [
        (records_module, _drive_create_record),
        (records_module, _drive_mark_content_missing),
        (asset_management_module, _drive_update_asset_metadata),
        (tagging_module, _drive_apply_tags),
    ],
    ids=["create_record", "mark_content_missing", "update_asset_metadata", "apply_tags"],
)
def test_every_tag_write_site_routes_through_the_conflict_safe_helper(
    session, mock_create_session, monkeypatch, target_module, drive
):
    record_id = _seed_record(session, "/tmp/four-sites-fixture")
    calls: list[str] = []
    real_ensure_tag_link = records_module.ensure_tag_link

    def spying_ensure_tag_link(*args, **kwargs):
        calls.append(kwargs["tag_name"])
        return real_ensure_tag_link(*args, **kwargs)

    monkeypatch.setattr(target_module, "ensure_tag_link", spying_ensure_tag_link)

    drive(session, record_id)

    assert calls, (
        "this site still writes asset_tags with a bare check-then-insert; a concurrent "
        "writer inserting the same link makes it raise instead of settling"
    )
