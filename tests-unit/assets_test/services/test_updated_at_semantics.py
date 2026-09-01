
import os
from datetime import datetime

import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetTag, Tag
from app.assets.database.queries.records import (
    create_content,
    create_record,
    delete_record,
    mark_content_missing,
    unset_content_missing,
)
from app.assets.helpers import to_stored_hash
from app.assets.scanner import enrich_asset
from app.assets.scanner_changes import split_content
from app.assets.services.asset_management import (
    resolve_asset_for_download,
    resolve_hash_to_path,
    update_asset_metadata,
)
from app.assets.services.tagging import apply_tags, remove_tags

STALE = datetime(2020, 1, 1, 0, 0, 0)


def _seed_record(session: Session, path: str, name: str = "fixture", tags=None, hash=None) -> Asset:
    content = create_content(session, path, hash=hash)
    record = create_record(session, content.id, name, tags=tags)
    session.commit()
    _force_stale(session, record.id)
    return record


def _seed_scannable_record(session: Session, path: str, name: str) -> Asset:
    stat_result = os.stat(path)
    content = create_content(
        session,
        path,
        size_bytes=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
    )
    record = create_record(session, content.id, name)
    session.commit()
    _force_stale(session, record.id)
    return record


def _force_stale(session: Session, record_id: str) -> None:
    session.execute(update(Asset).where(Asset.id == record_id).values(updated_at=STALE))
    session.commit()


def _updated_at(session: Session, record_id: str) -> datetime:
    session.expire_all()
    return session.get(Asset, record_id).updated_at


def _last_access(session: Session, record_id: str) -> datetime | None:
    session.expire_all()
    return session.get(Asset, record_id).last_access_time


def _write_file(temp_dir, name: str, payload: bytes = b"payload") -> str:
    file_path = temp_dir / name
    file_path.write_bytes(payload)
    return str(file_path)


def test_download_path_moves_last_access_time_but_not_updated_at(
    session, mock_create_session, temp_dir
):
    path = _write_file(temp_dir, "download.bin")
    record = _seed_record(session, path, name="download.bin")

    resolve_asset_for_download(record.id)

    assert _last_access(session, record.id) is not None, (
        "serving a download must record the access"
    )
    assert _updated_at(session, record.id) == STALE, (
        "a read moves last_access_time only; it is not an explicit user edit"
    )


def test_hash_serve_path_moves_last_access_time_but_not_updated_at(
    session, mock_create_session, temp_dir
):
    path = _write_file(temp_dir, "served.bin")
    digest = to_stored_hash("a" * 64)
    record = _seed_record(session, path, name="served.bin", hash=digest)

    assert resolve_hash_to_path(digest) is not None, "fixture must resolve"

    assert _last_access(session, record.id) is not None, (
        "serving by hash must record the access"
    )
    assert _updated_at(session, record.id) == STALE, (
        "hash-serving is a read; it is not an explicit user edit"
    )


def test_rename_moves_updated_at(session, mock_create_session, temp_dir):
    record = _seed_record(session, _write_file(temp_dir, "rename.bin"))

    update_asset_metadata(record.id, name="renamed")

    assert _updated_at(session, record.id) > STALE, "a rename is an explicit user edit"


@pytest.mark.parametrize(
    "field,kwargs",
    [
        ("user_metadata", {"user_metadata": {"note": "hello"}}),
        ("mime_type", {"mime_type": "image/png"}),
    ],
)
def test_user_field_updates_move_updated_at(
    session, mock_create_session, temp_dir, field, kwargs
):
    record = _seed_record(session, _write_file(temp_dir, f"{field}.bin"))

    update_asset_metadata(record.id, **kwargs)

    assert _updated_at(session, record.id) > STALE, (
        f"setting {field} is an explicit user edit"
    )


def test_preview_id_update_moves_updated_at(session, mock_create_session, temp_dir):
    record = _seed_record(session, _write_file(temp_dir, "subject.bin"))
    preview = _seed_record(session, _write_file(temp_dir, "preview.bin"), name="preview")

    update_asset_metadata(record.id, preview_id=preview.id)

    assert _updated_at(session, record.id) > STALE, (
        "setting preview_id is an explicit user edit"
    )


def test_manual_tag_add_moves_updated_at(session, mock_create_session, temp_dir):
    record = _seed_record(session, _write_file(temp_dir, "tag-add.bin"))

    apply_tags(record.id, ["favourite"])

    assert _updated_at(session, record.id) > STALE, (
        "adding a manual tag is an explicit user edit of the record"
    )


def test_manual_tag_remove_moves_updated_at(session, mock_create_session, temp_dir):
    record = _seed_record(
        session, _write_file(temp_dir, "tag-remove.bin"), tags=["favourite"]
    )

    remove_tags(record.id, ["favourite"])

    assert _updated_at(session, record.id) > STALE, (
        "removing a manual tag is an explicit user edit of the record"
    )


def test_tag_replacement_via_update_asset_metadata_moves_updated_at(
    session, mock_create_session, temp_dir
):
    record = _seed_record(session, _write_file(temp_dir, "tag-replace.bin"), tags=["old"])

    update_asset_metadata(record.id, tags=["new"])

    assert _updated_at(session, record.id) > STALE, (
        "replacing the manual tag set is an explicit user edit"
    )


def test_noop_tag_calls_do_not_move_updated_at(session, mock_create_session, temp_dir):
    record = _seed_record(
        session, _write_file(temp_dir, "tag-noop.bin"), tags=["already"]
    )

    apply_tags(record.id, ["already"])
    remove_tags(record.id, ["never-present"])

    assert _updated_at(session, record.id) == STALE, (
        "a tag call that changes no link is not an edit"
    )


def test_unchanged_tag_replacement_does_not_move_updated_at(
    session, mock_create_session, temp_dir
):
    record = _seed_record(
        session, _write_file(temp_dir, "tag-same.bin"), tags=["keep"]
    )

    update_asset_metadata(record.id, tags=["keep"])

    assert _updated_at(session, record.id) == STALE, (
        "replacing the manual tag set with an identical set changes nothing"
    )


def test_scanner_enrichment_does_not_move_updated_at(session, temp_dir):
    path = _write_file(temp_dir, "enrich.png", payload=b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    record = _seed_scannable_record(session, path, name="enrich.png")

    assert enrich_asset(
        session, path, record.content_id, record.id, extract_metadata=True
    ), "fixture must actually enrich"

    assert session.get(Asset, record.id).mime_type is not None, (
        "enrichment must have written a system-derived field"
    )
    assert _updated_at(session, record.id) == STALE, (
        "system-derived facts about unchanged bytes are not a user edit"
    )


def test_automatic_missing_tag_projection_does_not_move_updated_at(session, temp_dir):
    record = _seed_record(session, _write_file(temp_dir, "missing.bin"))

    mark_content_missing(session, record.content_id)
    session.commit()
    assert _updated_at(session, record.id) == STALE, (
        "the automatic missing tag is a system projection"
    )

    _force_stale(session, record.id)
    unset_content_missing(session, record.content_id)
    session.commit()
    assert _updated_at(session, record.id) == STALE, (
        "recovery is a system projection too"
    )


def test_content_split_does_not_move_updated_at_on_retired_record(
    session, temp_dir, monkeypatch
):
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(temp_dir))
    path = _write_file(temp_dir, "split.bin")
    record = _seed_record(session, path, name="split.bin")
    content = session.get(Asset, record.id).content

    split_content(session, content, os.stat(path), hash_value=to_stored_hash("b" * 64))
    session.commit()

    assert _updated_at(session, record.id) == STALE, (
        "a content split is a system projection over the retired record"
    )


def test_preview_target_delete_cascade_does_not_move_updated_at(db_engine_fk):
    with Session(db_engine_fk) as session:
        preview_content = create_content(session, "/output/preview-target.png")
        preview = create_record(session, preview_content.id, "preview-target")
        content = create_content(session, "/output/referencing.png")
        record = create_record(session, content.id, "referencing")
        session.commit()
        session.execute(
            update(Asset)
            .where(Asset.id == record.id)
            .values(preview_id=preview.id, updated_at=STALE)
        )
        session.commit()

        delete_record(session, preview.id)
        session.commit()

        session.expire_all()
        refreshed = session.get(Asset, record.id)
        assert refreshed.preview_id is None, "the FK cascade must have nulled the link"
        assert refreshed.updated_at == STALE, (
            "an FK-level SET NULL is not a user edit of the referencing record"
        )


def test_seed_helper_reports_stale_before_any_edit(session, temp_dir):
    record = _seed_record(session, _write_file(temp_dir, "seed.bin"))

    assert _updated_at(session, record.id) == STALE, "seeding must set the stale marker"
    assert isinstance(session.get(Asset, record.id), Asset)
    assert session.get(Tag, "nonexistent") is None
    assert session.get(AssetTag, (record.id, "nonexistent")) is None
