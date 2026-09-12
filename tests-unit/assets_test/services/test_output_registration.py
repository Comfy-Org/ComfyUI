import os
from pathlib import Path
from unittest.mock import patch

import folder_paths
from sqlalchemy import select

from app.assets.database.models import Asset, AssetContent
from app.assets.manager import AssetsEnabled, NoAssets
from comfy_execution.asset_enrichment import (
    register_cached_outputs,
    register_executed_outputs,
)

class _ArgsStub:
    def __init__(self, enable_assets: bool = True) -> None:
        self.enable_assets = enable_assets
        self.enable_asset_hashing = False


def _write_output_file(name: str, data: bytes) -> Path:
    path = Path(folder_paths.get_output_directory()) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _output_ui(name: str) -> dict:
    return {"images": [{"filename": name, "subfolder": "", "type": "output"}]}


def _wrapper(name: str, node_id: str = "1") -> dict:
    return {"meta": {"node_id": node_id}, "output": _output_ui(name)}


def test_executed_adapter_registers_new_output(mock_create_session):
    path = _write_output_file("adapter-executed-new.png", b"exec")
    try:
        output_ui = _output_ui(path.name)

        enriched = register_executed_outputs(
            output_ui, "exec-job", AssetsEnabled(_ArgsStub())
        )

        new_id = enriched["images"][0]["id"]
        assert "id" not in output_ui["images"][0]
        with mock_create_session() as session:
            record = session.get(Asset, new_id)
            content = session.get(AssetContent, record.content_id)
            assert record.job_id == "exec-job"
            assert content.path == os.path.abspath(path)
            assert content.is_missing is False
    finally:
        path.unlink(missing_ok=True)


def test_executed_adapter_over_existing_path_marks_old_missing(mock_create_session):
    path = _write_output_file("adapter-executed-replace.png", b"original")
    try:
        manager = AssetsEnabled(_ArgsStub())
        first = register_executed_outputs(_output_ui(path.name), "job-1", manager)
        old_id = first["images"][0]["id"]

        path.write_bytes(b"replacement")
        second = register_executed_outputs(_output_ui(path.name), "job-2", manager)
        new_id = second["images"][0]["id"]

        assert new_id != old_id
        with mock_create_session() as session:
            old_content = session.get(AssetContent, session.get(Asset, old_id).content_id)
            new_content = session.get(AssetContent, session.get(Asset, new_id).content_id)
            assert old_content.is_missing is True
            assert new_content.is_missing is False
            contents = list(
                session.scalars(
                    select(AssetContent).where(AssetContent.path == os.path.abspath(path))
                )
            )
            assert len(contents) == 2
    finally:
        path.unlink(missing_ok=True)


def test_executed_adapter_disabled_registers_nothing(mock_create_session):
    path = _write_output_file("adapter-executed-disabled.png", b"noop")
    try:
        output_ui = _output_ui(path.name)

        enriched = register_executed_outputs(output_ui, "job", NoAssets(_ArgsStub(False)))

        assert "id" not in enriched["images"][0]
        with mock_create_session() as session:
            contents = list(
                session.scalars(
                    select(AssetContent).where(AssetContent.path == os.path.abspath(path))
                )
            )
            assert contents == []
    finally:
        path.unlink(missing_ok=True)


def test_executed_adapter_registration_failure_never_raises(mock_create_session):
    path = _write_output_file("adapter-executed-error.png", b"exec")
    try:
        output_ui = _output_ui(path.name)

        with patch(
            "app.assets.manager.ingest_register_executed_output",
            side_effect=RuntimeError("boom"),
        ):
            enriched = register_executed_outputs(
                output_ui, "job", AssetsEnabled(_ArgsStub())
            )

        assert "id" not in enriched["images"][0]
    finally:
        path.unlink(missing_ok=True)


def test_cached_adapter_without_live_content_is_nonevent(mock_create_session):
    path = _write_output_file("adapter-cached-missing.png", b"new content")
    try:
        wrapper = _wrapper(path.name)

        enriched = register_cached_outputs(
            wrapper, "cached-job", AssetsEnabled(_ArgsStub())
        )

        assert "id" not in enriched["output"]["images"][0]
        with mock_create_session() as session:
            contents = list(
                session.scalars(
                    select(AssetContent).where(AssetContent.path == os.path.abspath(path))
                )
            )
            records = list(
                session.scalars(select(Asset).where(Asset.job_id == "cached-job"))
            )
            assert contents == []
            assert records == []
    finally:
        path.unlink(missing_ok=True)
