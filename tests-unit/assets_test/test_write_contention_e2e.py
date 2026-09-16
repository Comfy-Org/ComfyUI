import logging
import threading

import folder_paths
import pytest
from PIL import Image

import app.database.db as db_mod
from app.assets import lifecycle, mode, scanner
from app.assets.database.models import Asset
from app.assets.database.queries.records import create_content, create_record
from app.assets.manager import default_asset_manager
from app.assets.services.hash_mode_state import clear_transition_queue
from app.assets.services.ingest import register_executed_output
from app.assets.services.schemas import RegisteredAsset
from comfy.cli_args import args


@pytest.fixture(autouse=True)
def autoclean_unit_test_assets():
    yield


def test_register_executed_output_keeps_job_id_during_scanner_write_train(
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

        scanner_rows: list[scanner.UnenrichedContent] = []
        for index in range(4):
            scanner_path = output_directory / f"scanner-{index}.bin"
            scanner_path.write_bytes(f"scanner-{index}".encode())
            stat_result = scanner_path.stat()

            def seed(session, path=scanner_path, stat=stat_result) -> None:
                content = create_content(
                    session,
                    str(path),
                    size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                )
                record = create_record(session, content.id, path.name)
                scanner_rows.append(
                    scanner.UnenrichedContent(
                        content.id,
                        record.id,
                        str(path),
                        needs_hash=True,
                    )
                )

            db_mod.run_write_txn(seed)

        first_scanner_write_entered = threading.Event()
        release_first_scanner_write = threading.Event()
        second_scanner_write_committed = threading.Event()
        registration_started = threading.Event()
        original_apply = scanner._apply_enrichment
        original_run_write_txn = scanner.run_write_txn
        original_is_retryable_lock_error = db_mod._is_retryable_lock_error
        scanner_writes = 0
        scanner_writes_lock = threading.Lock()
        registration_thread_id: list[int | None] = [None]
        registration_blocked = threading.Event()

        def block_first_scanner_write(session, prepared):
            nonlocal scanner_writes
            updated = original_apply(session, prepared)
            with scanner_writes_lock:
                is_first_write = scanner_writes == 0
            if is_first_write:
                first_scanner_write_entered.set()
                assert release_first_scanner_write.wait(timeout=5)
            return updated

        def count_scanner_writes(work):
            nonlocal scanner_writes
            result = original_run_write_txn(work)
            with scanner_writes_lock:
                scanner_writes += 1
                if scanner_writes >= 2:
                    second_scanner_write_committed.set()
            return result

        def observe_registration_lock(error):
            is_retryable = original_is_retryable_lock_error(error)
            if threading.get_ident() == registration_thread_id[0] and is_retryable:
                registration_blocked.set()
            return is_retryable

        monkeypatch.setattr(scanner, "_apply_enrichment", block_first_scanner_write)
        monkeypatch.setattr(scanner, "run_write_txn", count_scanner_writes)
        monkeypatch.setattr(db_mod, "_is_retryable_lock_error", observe_registration_lock)
        scanner_result: dict[str, tuple[int, list[str]]] = {}
        registration_result: dict[str, RegisteredAsset | None] = {}

        def enrich_scanner_rows() -> None:
            scanner_result["value"] = scanner.enrich_assets_batch(
                scanner_rows,
                extract_metadata=False,
                compute_hash=True,
            )

        def register_output() -> None:
            registration_thread_id[0] = threading.get_ident()
            registration_started.set()
            registration_result["value"] = register_executed_output(
                str(output_path),
                job_id="write-contention",
            )

        scanner_worker = threading.Thread(target=enrich_scanner_rows)
        scanner_worker.start()
        try:
            assert first_scanner_write_entered.wait(timeout=5)
            registration_worker = threading.Thread(target=register_output)
            registration_worker.start()
            assert registration_started.wait(timeout=5)
            assert registration_blocked.wait(timeout=5)
            release_first_scanner_write.set()
            assert second_scanner_write_committed.wait(timeout=5)
            with caplog.at_level(logging.INFO):
                registration_worker.join(timeout=5)
        finally:
            release_first_scanner_write.set()
            scanner_worker.join(timeout=5)

        assert not scanner_worker.is_alive()
        assert not registration_worker.is_alive()
        assert scanner_result["value"] == (len(scanner_rows), [])
        assert scanner_writes == len(scanner_rows)
        result = registration_result["value"]
        assert isinstance(result, RegisteredAsset)
        with db_mod.create_session() as session:
            asset = session.get(Asset, result.id)
            assert asset is not None
            assert asset.job_id == "write-contention"
        assert not any("Failed to register" in record.getMessage() for record in caplog.records)
    finally:
        manager.shutdown()
        clear_transition_queue()
        mode.init(None)
        db_mod.Session.kw["bind"].dispose()
        db_mod.WriteSession.kw["bind"].dispose()
        db_mod._db_lock.release(force=True)
