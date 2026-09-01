import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import event, select

import folder_paths
from app.assets.database.models import Asset, AssetContent
from comfy_execution.asset_enrichment import (
    register_cached_outputs,
    register_executed_outputs,
)


@contextmanager
def _assets_enabled(enabled: bool = True):
    with patch.dict(
        sys.modules,
        {"comfy.cli_args": types.SimpleNamespace(args=types.SimpleNamespace(enable_assets=enabled))},
    ):
        yield


def _write_output_file(name: str) -> Path:
    path = Path(folder_paths.get_output_directory()) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"cached output")
    return path


def _output_ui(name: str) -> dict:
    return {"images": [{"filename": name, "subfolder": "", "type": "output"}]}


def _wrapper(name: str, node_id: str = "1") -> dict:
    return {"meta": {"node_id": node_id}, "output": _output_ui(name)}


def test_cached_replay_binds_new_record_to_existing_content(mock_create_session):
    path = _write_output_file("cached-execution-bind.png")
    try:
        with _assets_enabled():
            executed = register_executed_outputs(_output_ui(path.name), "original-job")
        original_id = executed["images"][0]["id"]
        with mock_create_session() as session:
            original_content_id = session.get(Asset, original_id).content_id

        wrapper = _wrapper(path.name)
        with _assets_enabled():
            enriched = register_cached_outputs(wrapper, "cached-job")
        cached_id = enriched["output"]["images"][0]["id"]

        with mock_create_session() as session:
            contents = list(
                session.scalars(
                    select(AssetContent).where(AssetContent.path == os.path.abspath(path))
                )
            )
            records = list(
                session.scalars(
                    select(Asset).where(Asset.content_id == original_content_id)
                )
            )
            assert len(contents) == 1
            assert len(records) == 2
            assert original_id in {r.id for r in records}
            assert cached_id in {r.id for r in records}
            assert {r.job_id for r in records} == {"original-job", "cached-job"}
        assert "id" not in wrapper["output"]["images"][0]
    finally:
        path.unlink(missing_ok=True)


def test_cached_replay_does_not_update_existing_content(mock_create_session, db_engine):
    path = _write_output_file("cached-execution-no-update.png")
    update_statements: list[str] = []

    def capture_updates(_, __, statement, ___, ____, _____):
        if statement.lstrip().upper().startswith("UPDATE"):
            update_statements.append(statement)

    try:
        with _assets_enabled():
            executed = register_executed_outputs(_output_ui(path.name), "original-job")
        original_id = executed["images"][0]["id"]
        with mock_create_session() as session:
            original_content = session.get(
                AssetContent, session.get(Asset, original_id).content_id
            )
            original_state = (
                original_content.id,
                original_content.hash,
                original_content.size_bytes,
                original_content.path,
                original_content.mtime_ns,
                original_content.is_missing,
                original_content.created_at,
            )

        event.listen(db_engine, "before_cursor_execute", capture_updates)
        wrapper = _wrapper(path.name)
        with _assets_enabled():
            register_cached_outputs(wrapper, "cached-job")

        with mock_create_session() as session:
            content = session.get(AssetContent, original_state[0])
            current_state = (
                content.id,
                content.hash,
                content.size_bytes,
                content.path,
                content.mtime_ns,
                content.is_missing,
                content.created_at,
            )
            assert current_state == original_state
            assert update_statements == []
    finally:
        event.remove(db_engine, "before_cursor_execute", capture_updates)
        path.unlink(missing_ok=True)
