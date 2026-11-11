import logging
from unittest.mock import patch

from sqlalchemy.orm import Session

from app.assets.database.queries import create_content, create_record
from app.assets.scanner import build_asset_specs


def test_locked_file_logs_typed_error_without_path(caplog):
    locked_path = "/private/user/models/locked.safetensors"

    with (
        patch("app.assets.scanner.os.stat", side_effect=PermissionError(locked_path)),
        caplog.at_level(logging.WARNING),
    ):
        specs, tags, skipped = build_asset_specs([locked_path], set())

    assert specs == []
    assert tags == set()
    assert skipped == 0
    assert caplog.messages == [
        "Asset scan error: phase=discovery_stat error_type=permission_denied"
    ]
    assert locked_path not in caplog.text


def test_bulk_discovery_creates_a_record_for_each_path(session: Session) -> None:
    discovered = [
        ("/models/a/model.safetensors", "a/model.safetensors"),
        ("/models/b/model.safetensors", "b/model.safetensors"),
    ]

    records = [
        create_record(session, create_content(session, path).id, name, loader_path=name)
        for path, name in discovered
    ]

    assert [record.loader_path for record in records] == [name for _, name in discovered]
    assert records[0].content_id != records[1].content_id
