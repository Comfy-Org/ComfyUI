from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from .helpers import trigger_sync_seed_assets


def _db_path(comfy_tmp_base_dir: Path, request: pytest.FixtureRequest) -> str:
    url = request.config.getoption("--db-url")
    if url and url.startswith("sqlite:///"):
        return url[len("sqlite:///"):]
    return str(comfy_tmp_base_dir / "assets-test.sqlite3")


def _query(db_path: str, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    last_err: Exception | None = None
    for _ in range(20):
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
            try:
                return list(con.execute(sql, params))
            finally:
                con.close()
        except sqlite3.OperationalError as e:
            last_err = e
            time.sleep(0.1)
    raise AssertionError(f"sqlite read failed: {last_err}")


def _content_by_id(db_path: str, content_id: str) -> tuple[str | None, int]:
    rows = _query(
        db_path,
        "SELECT hash, is_missing FROM asset_contents WHERE id = ?",
        (content_id,),
    )
    assert rows, f"content {content_id} vanished"
    return rows[0][0], rows[0][1]


def _rows_at_path(db_path: str, path: str) -> list[tuple[Any, ...]]:
    return _query(
        db_path,
        "SELECT id, hash, is_missing FROM asset_contents WHERE path = ? "
        "ORDER BY created_at",
        (path,),
    )


def _missing_tag_count(db_path: str, content_id: str) -> int:
    rows = _query(
        db_path,
        "SELECT COUNT(*) FROM asset_tags at "
        "JOIN assets a ON a.id = at.asset_id "
        "WHERE a.content_id = ? AND at.tag_name = 'missing'",
        (content_id,),
    )
    return rows[0][0]


def _seed_until_row(http, api_base, db_path: str, path: str) -> tuple[str, str | None]:
    for _ in range(5):
        trigger_sync_seed_assets(http, api_base)
        rows = _rows_at_path(db_path, path)
        if rows:
            return rows[0][0], rows[0][1]
    raise AssertionError(f"scanned file never produced a content row: {path}")


def _drop_scanned_file(comfy_tmp_base_dir: Path) -> tuple[Path, bytes]:
    ckpt_dir = comfy_tmp_base_dir / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    disk_path = ckpt_dir / f"scanned_{uuid.uuid4().hex}.safetensors"
    payload = uuid.uuid4().bytes + b"S" * (65536 - 16)
    disk_path.write_bytes(payload)
    return disk_path, payload


@pytest.mark.hashing_on
def test_same_path_restore_recovery_characterization(
    http, api_base, comfy_tmp_base_dir, request
):
    db_path = _db_path(comfy_tmp_base_dir, request)
    disk_path, payload = _drop_scanned_file(comfy_tmp_base_dir)

    content_id, _ = _seed_until_row(http, api_base, db_path, str(disk_path))

    disk_path.unlink()
    trigger_sync_seed_assets(http, api_base)
    _, missing_after_delete = _content_by_id(db_path, content_id)
    assert missing_after_delete == 1

    disk_path.write_bytes(payload)
    trigger_sync_seed_assets(http, api_base)
    trigger_sync_seed_assets(http, api_base)

    _, is_missing = _content_by_id(db_path, content_id)

    assert is_missing == 0


@pytest.mark.hashing_on
def test_same_path_restore_recovers_hashed_row(
    http, api_base, comfy_tmp_base_dir, request
):
    db_path = _db_path(comfy_tmp_base_dir, request)
    disk_path, payload = _drop_scanned_file(comfy_tmp_base_dir)

    content_id, birth_hash = _seed_until_row(http, api_base, db_path, str(disk_path))
    assert birth_hash is not None, "scanned file was not hashed in hash mode"

    disk_path.unlink()
    trigger_sync_seed_assets(http, api_base)
    _, missing_after_delete = _content_by_id(db_path, content_id)
    assert missing_after_delete == 1

    disk_path.write_bytes(payload)
    trigger_sync_seed_assets(http, api_base)
    trigger_sync_seed_assets(http, api_base)

    recovered_hash, is_missing = _content_by_id(db_path, content_id)
    assert is_missing == 0, "same-path restore did not recover the hashed row"
    assert recovered_hash == birth_hash
    assert _missing_tag_count(db_path, content_id) == 0

    live_rows = [r for r in _rows_at_path(db_path, str(disk_path)) if r[2] == 0]
    assert len(live_rows) == 1
    assert live_rows[0][0] == content_id
