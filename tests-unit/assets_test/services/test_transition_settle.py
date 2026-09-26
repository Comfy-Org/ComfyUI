from contextlib import contextmanager
from pathlib import Path
import threading
from unittest.mock import patch

import pytest
from sqlalchemy.dialects import sqlite
from sqlalchemy.orm import Session as SASession

from app.assets import scanner, seeder as seeder_module
from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import create_content, create_record
from app.assets.helpers import to_stored_hash
from app.assets.services import hash_mode_state
from app.assets.services.hash_mode_state import (
    clear_transition_queue,
    enqueue_transition_work,
    read_stored_mode,
    record_transition_intent,
    write_stored_mode,
)

_ATTEMPT_BUDGET = 5


class _AttemptBudgetExhausted(BaseException):
    # BaseException, not Exception: enrich_assets_batch's blanket except would swallow it.
    pass


@pytest.fixture(autouse=True)
def transition_queue():
    clear_transition_queue()
    yield
    clear_transition_queue()


def _denied(_candidate_path: str):
    raise PermissionError("denied")


def _rendered_candidate_placeholder_count(failed_count: int) -> int:
    failed_ids = [f"{index:036d}" for index in range(failed_count)]
    statement = scanner.build_unenriched_candidates_statement(
        prefixes=["/models"],
        compute_hashes=False,
        last_seen_id=failed_ids[-1],
        limit=100,
    )
    compiled = statement.compile(
        dialect=sqlite.dialect(),
        compile_kwargs={"render_postcompile": True},
    )
    rendered_count = str(compiled).count("?")

    assert compiled.positiontup is not None
    assert len(compiled.positiontup) == rendered_count
    return rendered_count


def test_unenriched_candidate_bind_count_does_not_scale_with_failed_rows() -> None:
    count_for_ten = _rendered_candidate_placeholder_count(10)
    count_for_thousand = _rendered_candidate_placeholder_count(1000)

    assert count_for_ten == count_for_thousand, (
        "rendered placeholders scaled with failed candidates: "
        f"N=10 -> {count_for_ten}; N=1000 -> {count_for_thousand}"
    )


def test_enrich_phase_settles_an_unreadable_transition_without_looping(
    session, db_engine, temp_dir: Path, monkeypatch
):
    path = temp_dir / "unreadable.safetensors"
    payload = b"bytes that can be stat'd but never hashed"
    path.write_bytes(payload)
    stat = path.stat()
    content = create_content(
        session, str(path), to_stored_hash("seed-digest"), stat.st_size, stat.st_mtime_ns
    )
    content_id = content.id
    record = create_record(session, content_id, path.name)
    record_id = record.id
    write_stored_mode(session, "off")
    monkeypatch.setattr(hash_mode_state._mode, "hashing_enabled", lambda: True)

    transition = record_transition_intent(session)
    enqueue_transition_work(session, transition)
    session.commit()

    @contextmanager
    def _create_session():
        with SASession(db_engine) as sess:
            yield sess

    attempts: list[str] = []
    real_enrich_asset = scanner.enrich_asset

    def counting_enrich_asset(*args, **kwargs):
        attempts.append(kwargs["record_id"])
        if len(attempts) > _ATTEMPT_BUDGET:
            raise _AttemptBudgetExhausted
        return real_enrich_asset(*args, **kwargs)

    seeder = seeder_module._AssetSeeder()
    seeder._compute_hashes = True
    seeder._run_gate.set()
    seeder._cancel_event.clear()

    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(temp_dir))
    monkeypatch.setattr(hash_mode_state, "snapshot_hash", _denied)
    monkeypatch.setattr(scanner, "snapshot_hash", _denied)
    monkeypatch.setattr(scanner, "enrich_asset", counting_enrich_asset)

    with patch("app.assets.seeder.create_session", _create_session), \
         patch("app.assets.scanner.create_session", _create_session):
        try:
            cancelled, _enriched = seeder._run_enrich_phase(("input",))
        except _AttemptBudgetExhausted:
            pytest.fail(
                f"the enrich phase re-selected the same record more than {_ATTEMPT_BUDGET} "
                "times: a terminally-cleared row stays hash-eligible, so counting its "
                "metadata as progress keeps it out of failed_ids and the pass never ends"
            )

    assert cancelled is False
    session.expire_all()
    assert read_stored_mode(session) == "on", (
        "one background scan must settle the transition on its own; waiting for a future "
        "prompt to queue another enrich pass leaves a quiet server wedged at 'off'"
    )
    settled = session.get(AssetContent, content_id)
    assert settled.is_missing is False, (
        "an unreadable file is not a deleted one; settling must not mark its row missing"
    )
    assert settled.hash is None
    assert attempts == [record_id], (
        "the record is attempted once, then excluded from the rest of the pass"
    )


def test_enrich_phase_reaches_healthy_candidates_after_a_full_failed_batch(
    session, db_engine, temp_dir: Path, monkeypatch
):
    for index in range(101):
        path = temp_dir / f"candidate-{index}.safetensors"
        content = create_content(session, str(path))
        create_record(session, content.id, path.name)
    session.commit()

    @contextmanager
    def _create_session():
        with SASession(db_engine) as sess:
            yield sess

    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(temp_dir))
    with patch("app.assets.scanner.create_session", _create_session):
        ordered = scanner.get_unenriched_assets_for_roots(
            ("input",), compute_hashes=False, limit=101
        )

    failed_ids = {row.record_id for row in ordered[:100]}
    healthy = ordered[100]
    attempted: list[str] = []

    def enrich_batch(rows, **_kwargs):
        attempted.extend(row.record_id for row in rows)
        failed = [row.record_id for row in rows if row.record_id in failed_ids]
        if healthy.record_id in attempted:
            with SASession(db_engine) as update_session:
                record = update_session.get(Asset, healthy.record_id)
                assert record is not None
                record.system_metadata = {"enriched": True}
                update_session.commit()
            return 1, failed, len(rows)
        return 0, failed, len(rows)

    asset_seeder = seeder_module._AssetSeeder()
    asset_seeder._run_gate.set()
    asset_seeder._cancel_event.clear()

    with (
        patch("app.assets.seeder.create_session", _create_session),
        patch("app.assets.scanner.create_session", _create_session),
        patch("app.assets.seeder.enrich_assets_batch", enrich_batch),
    ):
        cancelled, enriched = asset_seeder._run_enrich_phase(("input",))

    assert cancelled is False
    assert enriched == 1
    assert healthy.record_id in attempted


def test_enrich_phase_reaches_healthy_candidates_after_four_failed_batches(
    db_engine, monkeypatch
):
    failed_batches = [
        [scanner.UnenrichedContent(f"content-{index}", f"record-{index}", f"/{index}")]
        for index in range(4)
    ]
    healthy = scanner.UnenrichedContent("healthy-content", "healthy-record", "/healthy")
    batches = iter([*failed_batches, [healthy], []])
    attempted: list[str] = []

    @contextmanager
    def _create_session():
        with SASession(db_engine) as sess:
            yield sess

    def enrich_batch(rows, **_kwargs):
        attempted.extend(row.record_id for row in rows)
        if healthy in rows:
            return 1, [], len(rows)
        return 0, [row.record_id for row in rows], len(rows)

    asset_seeder = seeder_module._AssetSeeder()
    asset_seeder._run_gate.set()
    asset_seeder._cancel_event.clear()
    monkeypatch.setattr(
        seeder_module,
        "get_unenriched_assets_for_roots",
        lambda *_args, **_kwargs: next(batches),
    )
    monkeypatch.setattr(seeder_module, "enrich_assets_batch", enrich_batch)

    with patch("app.assets.seeder.create_session", _create_session):
        cancelled, enriched = asset_seeder._run_enrich_phase(("input",))

    assert cancelled is False
    assert enriched == 1
    assert attempted == [
        "record-0",
        "record-1",
        "record-2",
        "record-3",
        "healthy-record",
    ]


@pytest.mark.parametrize("interrupt_after", [0, 1], ids=["before-first", "partway"])
def test_enrich_phase_reoffers_rows_not_attempted_before_pause(
    db_engine,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_after: int,
) -> None:
    rows = [
        scanner.UnenrichedContent(
            f"content-{index}", f"record-{index}", f"/{index}"
        )
        for index in range(3)
    ]
    fetch_cursors: list[str | None] = []
    offered_batches: list[list[str]] = []
    attempted: list[str] = []
    pause_blocked = threading.Event()
    interruption_triggered = False

    @contextmanager
    def _create_session():
        with SASession(db_engine) as sess:
            yield sess

    asset_seeder = seeder_module._AssetSeeder()
    asset_seeder._run_gate.set()
    asset_seeder._cancel_event.clear()
    asset_seeder.set_event_sink(
        lambda event_type, _data: pause_blocked.set()
        if event_type == "assets.seed.paused"
        else None
    )

    def get_candidates(
        _roots,
        compute_hashes,
        limit=1000,
        last_seen_id=None,
    ):
        nonlocal interruption_triggered
        candidates = [
            row
            for row in rows
            if last_seen_id is None or row.record_id > last_seen_id
        ][:limit]
        fetch_cursors.append(last_seen_id)
        offered_batches.append([row.record_id for row in candidates])
        if interrupt_after == 0 and not interruption_triggered:
            interruption_triggered = True
            asset_seeder._run_gate.clear()
        return candidates

    def enrich_asset(*_args, **kwargs) -> bool:
        nonlocal interruption_triggered
        attempted.append(kwargs["record_id"])
        if len(attempted) == interrupt_after and not interruption_triggered:
            interruption_triggered = True
            asset_seeder._run_gate.clear()
        return True

    result: list[tuple[bool, int]] = []
    errors: list[BaseException] = []

    def run_enrich_phase() -> None:
        try:
            result.append(asset_seeder._run_enrich_phase(("input",)))
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(
        seeder_module, "get_unenriched_assets_for_roots", get_candidates
    )
    monkeypatch.setattr(scanner, "enrich_asset", enrich_asset)

    with (
        patch("app.assets.seeder.create_session", _create_session),
        patch("app.assets.scanner.create_session", _create_session),
    ):
        worker = threading.Thread(target=run_enrich_phase, daemon=True)
        worker.start()
        try:
            assert pause_blocked.wait(timeout=2), (
                "the enrich loop did not block at its loop-top pause checkpoint"
            )
            assert fetch_cursors == [None]
            assert worker.is_alive()
        finally:
            asset_seeder._run_gate.set()
            worker.join(timeout=2)

    assert worker.is_alive() is False
    assert errors == []
    assert result == [(False, 3)]
    assert attempted == [row.record_id for row in rows]
    assert offered_batches[1] == [
        row.record_id for row in rows[interrupt_after:]
    ]
    assert fetch_cursors[1] == (
        None if interrupt_after == 0 else rows[interrupt_after - 1].record_id
    )
