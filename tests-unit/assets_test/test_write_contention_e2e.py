import logging
import sqlite3
import threading
import time

import folder_paths
import pytest
from PIL import Image

import app.database.db as db_mod
from app.assets import lifecycle, mode
from app.assets.manager import default_asset_manager
from app.assets.services.hash_mode_state import clear_transition_queue
from app.assets.services.ingest import register_executed_output
from app.assets.services.schemas import RegisteredAsset
from comfy.cli_args import args


@pytest.fixture(autouse=True)
def autoclean_unit_test_assets():
    yield


def test_register_executed_output_waits_for_a_held_sqlite_writer(
    tmp_path, monkeypatch, caplog
) -> None:
    output_directory = tmp_path / "output"
    input_directory = tmp_path / "input"
    temp_directory = tmp_path / "temp"
    for directory in (output_directory, input_directory, temp_directory):
        directory.mkdir()

    database_path = tmp_path / "assets.db"
    monkeypatch.setattr(args, "database_url", f"sqlite:///{database_path}")
    monkeypatch.setattr(args, "enable_assets", True)
    monkeypatch.setattr(args, "enable_asset_hashing", False)
    monkeypatch.setattr(folder_paths, "output_directory", str(output_directory))
    monkeypatch.setattr(folder_paths, "input_directory", str(input_directory))
    monkeypatch.setattr(folder_paths, "temp_directory", str(temp_directory))
    monkeypatch.setattr(db_mod, "Session", None)
    monkeypatch.setattr(db_mod, "WriteSession", None)
    monkeypatch.setattr(db_mod, "_db_lock", None)
    monkeypatch.setattr(lifecycle, "start_asset_seeder", lambda: False)

    db_mod.init_db()
    manager = default_asset_manager()
    try:
        assert manager.enabled
        manager.startup()

        output_path = output_directory / "ComfyUI_00001_.png"
        Image.new("RGB", (1, 1), (255, 0, 0)).save(output_path)

        holder_ready = threading.Event()
        holder_errors: list[sqlite3.Error] = []

        def hold_write_lock() -> None:
            connection = sqlite3.connect(database_path, timeout=1)
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("INSERT INTO tags (name) VALUES (?)", ("e2e-holder",))
                holder_ready.set()
                threading.Event().wait(timeout=2)
                connection.commit()
            except sqlite3.Error as error:
                holder_errors.append(error)
                holder_ready.set()
            finally:
                connection.close()

        holder = threading.Thread(target=hold_write_lock)
        holder.start()
        try:
            assert holder_ready.wait(timeout=2)
            assert not holder_errors
            with caplog.at_level(logging.INFO):
                started = time.monotonic()
                result = register_executed_output(str(output_path), job_id="write-contention")
                elapsed = time.monotonic() - started
        finally:
            holder.join(timeout=2)

        assert not holder.is_alive()
        assert not holder_errors
        assert isinstance(result, RegisteredAsset)
        assert elapsed >= 1.5
        assert not any("Failed to register" in record.getMessage() for record in caplog.records)
    finally:
        manager.shutdown()
        clear_transition_queue()
        mode.init(None)
        db_mod.Session.kw["bind"].dispose()
        db_mod.WriteSession.kw["bind"].dispose()
        db_mod._db_lock.release(force=True)
