from pathlib import Path

import sqlalchemy as sa

import app.database.db as db_mod
from app.assets import scanner, seeder as seeder_module
from app.assets.database.models import AssetContent
from app.assets.database.queries import create_content, create_record, mark_content_missing
from app.assets.seeder import State, _AssetSeeder, _ScanState


def _seed_reference_observations(
    root: Path,
    count: int,
    *,
    content_hash: str | None = "blake3:old",
) -> tuple[list[str], list[scanner._ReferenceObservation], set[str]]:
    paths: list[Path] = []
    for index in range(count):
        path = root / f"reference-{index:03d}.bin"
        path.write_bytes(f"reference-{index:03d}".encode())
        paths.append(path)

    def seed(session):
        content_ids: list[str] = []
        for path in paths:
            stat_result = path.stat()
            content = create_content(
                session,
                str(path),
                hash=content_hash,
                size_bytes=stat_result.st_size,
                mtime_ns=stat_result.st_mtime_ns - 1,
            )
            create_record(
                session,
                content.id,
                path.name,
                system_metadata={"seeded": True},
            )
            content_ids.append(content.id)
        return content_ids

    content_ids = db_mod.run_write_txn(seed)
    observations = [
        scanner._ReferenceObservation(
            content_id,
            str(path),
            path.stat().st_size,
            path.stat().st_mtime_ns - 1,
            path.stat(),
        )
        for content_id, path in zip(content_ids, paths)
    ]
    return content_ids, observations, {str(path.resolve()) for path in paths}


def _content_states(content_ids: list[str]) -> list[tuple[str | None, int | None, bool]]:
    with db_mod.create_session() as session:
        return [
            (content.hash, content.mtime_ns, content.is_missing)
            for content_id in content_ids
            if (content := session.get(AssetContent, content_id)) is not None
        ]


def _reset_observed_rows(
    content_ids: list[str], observations: list[scanner._ReferenceObservation]
) -> None:
    def reset(session):
        for content_id, observation in zip(content_ids, observations):
            content = session.get(AssetContent, content_id)
            assert content is not None
            content.hash = "blake3:old"
            content.size_bytes = observation.observed_size_bytes
            content.mtime_ns = observation.observed_mtime_ns

    db_mod.run_write_txn(reset)


def _seed_outside_contents(root: Path, count: int) -> list[str]:
    def seed(session):
        content_ids: list[str] = []
        for index in range(count):
            path = root / f"outside-{index:03d}.bin"
            content = create_content(session, str(path), size_bytes=1, mtime_ns=1)
            create_record(session, content.id, path.name)
            content_ids.append(content.id)
        return content_ids

    return db_mod.run_write_txn(seed)


def _missing_count(content_ids: list[str]) -> int:
    with db_mod.create_session() as session:
        return session.scalar(
            sa.select(sa.func.count())
            .select_from(AssetContent)
            .where(
                AssetContent.id.in_(content_ids),
                AssetContent.is_missing.is_(True),
            )
        )


def _record_transactions(monkeypatch) -> list[int]:
    real_run_write_txn = scanner.run_write_txn
    transactions: list[int] = []

    def record(work):
        transactions.append(1)
        return real_run_write_txn(work)

    monkeypatch.setattr(scanner, "run_write_txn", record)
    return transactions




def test_root_sync_interrupts_between_chunks_and_publishes_committed_ids(
    tmp_path: Path, monkeypatch, session
) -> None:
    content_ids, observations, survivors = _seed_reference_observations(
        tmp_path, 60, content_hash=None
    )
    monkeypatch.setattr(scanner.mode, "hashing_enabled", lambda: True)
    monkeypatch.setattr(
        scanner,
        "observe_references_on_filesystem",
        lambda *_args, **_kwargs: (observations, survivors),
    )
    published: list[str] = []
    monkeypatch.setattr(scanner, "queue_pending_verification", published.append)
    real_run_write_txn = scanner.run_write_txn
    txn_count = 0

    def count_transaction(work):
        nonlocal txn_count
        txn_count += 1
        return real_run_write_txn(work)

    checks = 0

    def interrupt_after_first_chunk() -> bool:
        nonlocal checks
        checks += 1
        return checks > 1

    monkeypatch.setattr(scanner, "run_write_txn", count_transaction)

    result = scanner.sync_root_safely(
        "input", interrupt_check=interrupt_after_first_chunk
    )

    assert txn_count == 1
    assert result == survivors
    assert published == content_ids[: scanner.MAX_WRITE_BATCH]


def test_seeder_resumes_root_sync_chunks_after_pause(
    tmp_path: Path, monkeypatch, session
) -> None:
    content_ids, observations, survivors = _seed_reference_observations(tmp_path, 60)
    monkeypatch.setattr(
        scanner,
        "observe_references_on_filesystem",
        lambda *_args, **_kwargs: (observations, survivors),
    )
    seeder = _AssetSeeder()
    seeder._state = State.RUNNING
    seeder._scan_state = _ScanState()
    events: list[str] = []
    seeder.set_event_sink(lambda event_type, _data: events.append(event_type))

    def pause_before_root_sync(root, progress, interrupt_check=None):
        assert seeder.pause()
        return scanner.sync_root_safely(root, progress, interrupt_check)

    def resume_on_wait(timeout=None):
        if not seeder._run_gate.is_set():
            assert seeder.resume()
        return True

    monkeypatch.setattr(seeder_module, "sync_root_safely", pause_before_root_sync)
    monkeypatch.setattr(seeder._run_gate, "wait", resume_on_wait)
    monkeypatch.setattr(seeder_module, "collect_paths_for_roots", lambda _roots: [])
    monkeypatch.setattr(seeder_module, "tick_watch_list", lambda **_kwargs: None)

    seeder._run_fast_phase(("input",))

    assert _content_states(content_ids) == [
        (None, observation.stat_result.st_mtime_ns, False)
        for observation in observations
    ]
    assert events.count("assets.seed.paused") == 1


def test_root_sync_later_chunk_failure_publishes_only_prior_commits(
    tmp_path: Path, monkeypatch, session
) -> None:
    content_ids, observations, survivors = _seed_reference_observations(
        tmp_path, 60, content_hash=None
    )
    monkeypatch.setattr(scanner.mode, "hashing_enabled", lambda: True)
    monkeypatch.setattr(
        scanner,
        "observe_references_on_filesystem",
        lambda *_args, **_kwargs: (observations, survivors),
    )
    published: list[str] = []
    monkeypatch.setattr(scanner, "queue_pending_verification", published.append)
    real_run_write_txn = scanner.run_write_txn
    txn_count = 0

    def fail_second_transaction(work):
        nonlocal txn_count
        txn_count += 1
        if txn_count == 2:
            raise RuntimeError("chunk two failed")
        return real_run_write_txn(work)

    monkeypatch.setattr(scanner, "run_write_txn", fail_second_transaction)

    result = scanner.sync_root_safely("input")

    assert result == set()
    assert txn_count == 2
    assert published == content_ids[: scanner.MAX_WRITE_BATCH]


def test_missing_row_is_untouched_when_observation_is_applied_later(
    tmp_path: Path, session
) -> None:
    content_ids, observations, _survivors = _seed_reference_observations(tmp_path, 1)

    db_mod.run_write_txn(lambda session: mark_content_missing(session, content_ids[0]))
    db_mod.run_write_txn(
        lambda session: scanner.apply_reference_observations(session, observations)
    )

    assert _content_states(content_ids) == [
        ("blake3:old", observations[0].observed_mtime_ns, True)
    ]


def test_prune_marks_thirty_rows_in_two_transactions(
    tmp_path: Path, monkeypatch, session
) -> None:
    content_ids = _seed_outside_contents(tmp_path / "outside", 30)
    transactions = _record_transactions(monkeypatch)

    marked = scanner.mark_missing_outside_prefixes_safely(
        [str(tmp_path / "inside")]
    )

    assert marked == 30
    assert len(transactions) == 2
    assert _missing_count(content_ids) == 30


def test_prune_interrupts_between_chunks_and_returns_committed_count(
    tmp_path: Path, monkeypatch, session
) -> None:
    content_ids = _seed_outside_contents(tmp_path / "outside", 30)
    transactions = _record_transactions(monkeypatch)

    checks = 0

    def interrupt_after_first_chunk() -> bool:
        nonlocal checks
        checks += 1
        return checks > 1

    marked = scanner.mark_missing_outside_prefixes_safely(
        [str(tmp_path / "inside")],
        interrupt_check=interrupt_after_first_chunk,
    )

    assert marked == 25
    assert len(transactions) == 1
    assert _missing_count(content_ids) == 25


def test_prune_later_chunk_failure_returns_prior_committed_count(
    tmp_path: Path, monkeypatch, session
) -> None:
    content_ids = _seed_outside_contents(tmp_path / "outside", 30)
    real_run_write_txn = scanner.run_write_txn
    txn_count = 0

    def fail_second_transaction(work):
        nonlocal txn_count
        txn_count += 1
        if txn_count == 2:
            raise RuntimeError("chunk two failed")
        return real_run_write_txn(work)

    monkeypatch.setattr(scanner, "run_write_txn", fail_second_transaction)

    marked = scanner.mark_missing_outside_prefixes_safely(
        [str(tmp_path / "inside")]
    )

    assert marked == 25
    assert txn_count == 2
    assert _missing_count(content_ids) == 25
