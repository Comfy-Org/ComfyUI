import logging
import re
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.assets import scanner as scanner_module
from app.assets import seeder as seeder_module
from app.assets.database.models import Asset, Base
from app.assets.database.queries import create_content, create_record, mark_content_missing
from app.assets.event_log import TAG
from app.assets.scanner import SeedAssetSpec
from app.assets.seeder import Progress, ScanPhase, State, _AssetSeeder, _ScanStage, _ScanState


EVENT_LINE_PATTERN = re.compile(
    rf"^{re.escape(TAG)} (?P<event>[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*)"
    r"(?P<fields>(?: [a-z_]+=[^ =]+)*)$"
)
EventFields = dict[str, bool | int | str]

# Hang detector: a checkpoint that parks the scan fails the test, not the suite.
SCAN_JOIN_TIMEOUT = 5.0


@pytest.fixture
def scan_seeder(monkeypatch: pytest.MonkeyPatch) -> _AssetSeeder:
    instance = _AssetSeeder()
    instance._state = State.RUNNING
    instance._scan_state = _ScanState()
    instance._roots = ("models", "input")
    instance._phase = ScanPhase.FULL
    monkeypatch.setattr(seeder_module, "dependencies_available", lambda: True)
    monkeypatch.setattr(instance, "_log_scan_config", lambda roots: None)
    return instance


def parse_fields(raw: str) -> EventFields:
    fields: EventFields = {}
    for pair in raw.split():
        name, value = pair.split("=", maxsplit=1)
        if value == "true":
            fields[name] = True
        elif value == "false":
            fields[name] = False
        elif value.removeprefix("-").isdigit():
            fields[name] = int(value)
        else:
            fields[name] = value
    return fields


def tagged_events(caplog: pytest.LogCaptureFixture) -> list[tuple[str, EventFields]]:
    events: list[tuple[str, EventFields]] = []
    for record in caplog.records:
        match = EVENT_LINE_PATTERN.match(record.getMessage())
        if match is not None:
            events.append((match.group("event"), parse_fields(match.group("fields"))))
    return events


def events_named(
    caplog: pytest.LogCaptureFixture, event_name: str
) -> list[EventFields]:
    return [fields for event, fields in tagged_events(caplog) if event == event_name]


def _seed_spec(path: Path) -> SeedAssetSpec:
    stat_result = path.stat()
    return {
        "abs_path": str(path),
        "size_bytes": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "info_name": path.name,
        "tags": ["input"],
        "fname": path.name,
        "metadata": None,
        "mime_type": None,
        "job_id": None,
    }


def _configure_fast_phase(
    monkeypatch: pytest.MonkeyPatch,
    paths: list[Path],
    specs: list[SeedAssetSpec],
) -> None:
    monkeypatch.setattr(
        seeder_module, "sync_root_safely", lambda _root, _progress: set()
    )
    monkeypatch.setattr(
        seeder_module, "collect_paths_for_roots", lambda _roots: [str(path) for path in paths]
    )
    monkeypatch.setattr(
        seeder_module,
        "build_asset_specs",
        lambda *_args, **_kwargs: (specs, set(), 0),
    )
    watch_session = Mock()
    monkeypatch.setattr(seeder_module, "create_session", lambda: nullcontext(watch_session))
    monkeypatch.setattr(seeder_module, "tick_watch_list", lambda: None)


def _run_faulting_fast_phase(
    scan_seeder: _AssetSeeder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    original_fault: Exception,
    commit_failure: Exception | None = None,
) -> tuple[Engine, tuple[int, int, int]]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    paths = [tmp_path / name for name in ("first.bin", "broken.bin", "last.bin")]
    for path in paths:
        path.write_bytes(path.name.encode())
    specs = [_seed_spec(path) for path in paths]

    @contextmanager
    def database_session():
        with Session(engine) as session:
            if commit_failure is not None:
                session.connection().exec_driver_sql("BEGIN")
                monkeypatch.setattr(
                    session, "commit", Mock(side_effect=commit_failure)
                )
            yield session

    def create_record_or_raise(
        session: Session,
        *,
        content_id: str,
        name: str,
        mime_type: str | None,
        job_id: str | None,
        loader_path: str | None,
        tags: list[str],
    ) -> Asset:
        if name == "broken.bin":
            raise original_fault
        return create_record(
            session,
            content_id=content_id,
            name=name,
            mime_type=mime_type,
            job_id=job_id,
            loader_path=loader_path,
            tags=tags,
        )

    monkeypatch.setattr(scanner_module, "create_session", database_session)
    monkeypatch.setattr(scanner_module, "create_write_session", database_session)
    monkeypatch.setattr(scanner_module, "create_record", create_record_or_raise)
    monkeypatch.setattr(scanner_module.mode, "hashing_enabled", lambda: False)
    _configure_fast_phase(monkeypatch, paths, specs)
    return engine, scan_seeder._run_fast_phase(("models",))


def test_idle_status_returns_a_progress_snapshot() -> None:
    seeder = _AssetSeeder()
    seeder._last_progress = Progress(created=1)

    status = seeder.get_status()
    assert status.progress is not None
    status.progress.created = 999

    next_status = seeder.get_status()
    assert next_status.progress is not None
    assert next_status.progress.created == 1


def test_seeder_models_missing_as_content_state():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        content = create_content(session, "/models/checkpoints/model.safetensors", hash=None)
        record = create_record(
            session,
            content.id,
            "model.safetensors",
            loader_path="checkpoints/model.safetensors",
            tags=["models", "model_type:checkpoints"],
        )

        mark_content_missing(session, content.id)

        assert content.is_missing is True
        assert record.content_id == content.id


def test_multi_root_scan_emits_one_started_and_completed_without_root(
    scan_seeder: _AssetSeeder,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = iter((10.0, 10.8126))
    monkeypatch.setattr(seeder_module.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(scan_seeder, "_run_fast_phase", lambda roots: (3, 2, 5))
    monkeypatch.setattr(scan_seeder, "_run_enrich_phase", lambda roots: (False, 4))

    with caplog.at_level(logging.INFO):
        scan_seeder._run_scan()

    assert events_named(caplog, "seeder.scan_started") == [{"phase": "full"}]
    completed = events_named(caplog, "seeder.scan_completed")
    assert len(completed) == 1
    assert completed[0] == {
        "created": 3,
        "elapsed_ms": 813,
        "enrich_failed": 0,
        "enriched": 4,
        "hash_failed": 0,
        "permission_denied": 0,
        "phase": "full",
        "skipped": 2,
    }


def test_scan_completed_reports_per_scan_failure_counts(
    scan_seeder: _AssetSeeder,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    scan_seeder._scan_state = _ScanState(
        hash_failed=2,
        enrich_failed=3,
        permission_denied=1,
    )
    clock = iter((10.0, 10.5))
    monkeypatch.setattr(seeder_module.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(scan_seeder, "_run_fast_phase", lambda roots: (0, 0, 0))
    monkeypatch.setattr(scan_seeder, "_run_enrich_phase", lambda roots: (False, 0))

    with caplog.at_level(logging.INFO):
        scan_seeder._run_scan()

    completed = events_named(caplog, "seeder.scan_completed")
    assert len(completed) == 1
    assert completed[0]["hash_failed"] == 2
    assert completed[0]["enrich_failed"] == 3
    assert completed[0]["permission_denied"] == 1


def test_enrich_phase_does_not_count_returned_ids_as_failures(
    scan_seeder: _AssetSeeder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Mock()
    batches = iter(
        (
            [
                Mock(record_id="record-1"),
                Mock(record_id="record-2"),
            ],
            [],
        )
    )
    monkeypatch.setattr(seeder_module, "create_session", lambda: nullcontext(session))
    monkeypatch.setattr(seeder_module, "drain_pending_verifications", lambda _session: None)
    monkeypatch.setattr(seeder_module, "tick_watch_list", lambda: None)
    monkeypatch.setattr(seeder_module, "drain_transition_queue", lambda _session: None)
    monkeypatch.setattr(
        seeder_module,
        "get_unenriched_assets_for_roots",
        lambda *_args, **_kwargs: next(batches),
    )
    monkeypatch.setattr(
        seeder_module,
        "enrich_assets_batch",
        lambda *_args, **_kwargs: (0, ["record-1", "record-2"], 2),
    )
    monkeypatch.setattr(scan_seeder, "_check_pause_and_cancel", lambda _stage: False)

    cancelled, enriched = scan_seeder._run_enrich_phase(("models",))

    assert cancelled is False
    assert enriched == 0
    assert scan_seeder._scan_state is not None
    assert scan_seeder._scan_state.enrich_failed == 0


def test_starting_a_scan_installs_fresh_per_scan_failure_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _AssetSeeder()
    instance._scan_state = _ScanState(
        hash_failed=7,
        enrich_failed=4,
        permission_denied=2,
    )
    instance._scan_state.mark_emitted("enrich_failed")
    monkeypatch.setattr(instance, "_run_scan", lambda: None)

    started = instance.start(roots=("models",), phase=ScanPhase.FAST)

    assert started is True
    assert instance._thread is not None
    instance._thread.join(timeout=5)
    assert instance._scan_state == _ScanState()


def test_single_root_scan_emits_root_and_phase(
    scan_seeder: _AssetSeeder,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    scan_seeder._roots = ("output",)
    scan_seeder._phase = ScanPhase.FAST
    monkeypatch.setattr(scan_seeder, "_run_fast_phase", lambda roots: (0, 0, 0))

    with caplog.at_level(logging.INFO):
        scan_seeder._run_scan()

    assert events_named(caplog, "seeder.scan_started") == [
        {"phase": "fast", "root": "output"}
    ]
    completed = events_named(caplog, "seeder.scan_completed")
    assert len(completed) == 1
    assert completed[0]["phase"] == "fast"
    assert completed[0]["root"] == "output"


def test_dependency_failure_emits_no_tagged_scan_lifecycle_lines(
    scan_seeder: _AssetSeeder,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(seeder_module, "dependencies_available", lambda: False)

    with caplog.at_level(logging.INFO):
        scan_seeder._run_scan()

    assert [
        event for event, _fields in tagged_events(caplog) if event.startswith("seeder.scan_")
    ] == []


def test_scan_failure_emits_exception_type_without_message(
    scan_seeder: _AssetSeeder,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    scan_seeder._roots = ("models",)
    scan_seeder._phase = ScanPhase.ENRICH

    def fail_scan(_roots: tuple[str, ...]) -> None:
        raise FileNotFoundError("/private/models/secret.safetensors")

    monkeypatch.setattr(scan_seeder, "_log_scan_config", fail_scan)

    with caplog.at_level(logging.INFO):
        scan_seeder._run_scan()

    assert events_named(caplog, "seeder.scan_failed") == [
        {"error_type": "FileNotFoundError", "phase": "enrich", "root": "models"}
    ]
    tagged = "\n".join(record.getMessage() for record in caplog.records if TAG in record.getMessage())
    assert "/private/models/secret.safetensors" not in tagged


@pytest.mark.parametrize(
    ("stage", "phase"),
    [
        pytest.param("pruning", ScanPhase.FAST, id="pruning"),
        pytest.param("fast_scan", ScanPhase.FAST, id="fast-scan"),
        pytest.param("enrich", ScanPhase.ENRICH, id="enrich"),
        pytest.param("finalize", ScanPhase.ENRICH, id="finalize"),
    ],
)
def test_scan_cancellation_emits_the_checkpoint_stage(
    stage: str,
    phase: ScanPhase,
    scan_seeder: _AssetSeeder,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    scan_seeder._phase = phase
    original_check = scan_seeder._check_pause_and_cancel

    def cancel_at_stage(checkpoint_stage) -> bool:
        if checkpoint_stage.value == stage:
            scan_seeder._cancel_event.set()
        return original_check(checkpoint_stage)

    def run_enrich(roots) -> tuple[bool, int]:
        # The finalize checkpoint is non-blocking and never routes through
        # _check_pause_and_cancel, so its cancel has to land before it.
        if stage == "finalize":
            scan_seeder._cancel_event.set()
        return (False, 0)

    monkeypatch.setattr(scan_seeder, "_check_pause_and_cancel", cancel_at_stage)
    monkeypatch.setattr(scan_seeder, "_run_fast_phase", lambda roots: (0, 0, 0))
    monkeypatch.setattr(scan_seeder, "_run_enrich_phase", run_enrich)

    with caplog.at_level(logging.INFO):
        scan_seeder._run_scan()

    assert events_named(caplog, "seeder.scan_cancelled") == [
        {"phase": phase.value, "stage": stage}
    ]


def test_idle_reset_survives_a_raising_cancellation_emit(
    scan_seeder: _AssetSeeder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_check = scan_seeder._check_pause_and_cancel

    def cancel_at_pruning(stage) -> bool:
        if stage == _ScanStage.PRUNING:
            scan_seeder._cancel_event.set()
        return original_check(stage)

    monkeypatch.setattr(scan_seeder, "_check_pause_and_cancel", cancel_at_pruning)
    monkeypatch.setattr(seeder_module, "get_owned_prefixes", lambda: ())
    monkeypatch.setattr(
        seeder_module, "mark_missing_outside_prefixes_safely", lambda prefixes: 0
    )

    original_emit = seeder_module.emit

    def raise_on_scan_cancelled(event, **kwargs):
        if event == "seeder.scan_cancelled":
            raise RuntimeError("event bus down")
        return original_emit(event, **kwargs)

    monkeypatch.setattr(seeder_module, "emit", raise_on_scan_cancelled)

    with pytest.raises(RuntimeError, match="event bus down"):
        scan_seeder._run_scan()

    assert scan_seeder._state is State.IDLE
    assert scan_seeder._scan_state is None
    assert scan_seeder.mark_missing_outside_prefixes() == 0


def test_scan_paused_after_its_last_phase_still_completes(
    scan_seeder: _AssetSeeder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scan_seeder._phase = ScanPhase.ENRICH
    monkeypatch.setattr(seeder_module, "get_owned_prefixes", lambda: ())
    monkeypatch.setattr(
        seeder_module, "mark_missing_outside_prefixes_safely", lambda prefixes: 0
    )

    def pause_while_finishing(roots) -> tuple[bool, int]:
        scan_seeder.pause()
        return (False, 0)

    monkeypatch.setattr(scan_seeder, "_run_enrich_phase", pause_while_finishing)
    events: list[str] = []
    scan_seeder.set_event_sink(lambda event_type, data: events.append(event_type))

    scan = threading.Thread(target=scan_seeder._run_scan, daemon=True)
    scan.start()
    try:
        scan.join(timeout=SCAN_JOIN_TIMEOUT)

        assert scan.is_alive() is False, "paused scan parked at the finalize checkpoint"
        assert "assets.seed.completed" in events
        assert "assets.seed.paused" not in events
        assert scan_seeder.mark_missing_outside_prefixes() == 0
    finally:
        scan_seeder._run_gate.set()
        scan.join(timeout=SCAN_JOIN_TIMEOUT)


def test_enrich_interrupt_records_the_enrich_cancellation_stage(
    scan_seeder: _AssetSeeder,
) -> None:
    scan_seeder._cancel_event.set()

    assert scan_seeder._is_paused_or_cancelled() is True
    assert scan_seeder._scan_state is not None
    assert scan_seeder._scan_state.cancel_stage is not None
    assert scan_seeder._scan_state.cancel_stage == "enrich"


def test_prune_before_scan_emits_marked_missing_with_pruning_stage(
    scan_seeder: _AssetSeeder,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    scan_seeder._prune_first = True
    scan_seeder._phase = ScanPhase.FAST
    monkeypatch.setattr(seeder_module, "get_owned_prefixes", lambda: ())
    monkeypatch.setattr(
        seeder_module, "mark_missing_outside_prefixes_safely", lambda prefixes: 5
    )
    monkeypatch.setattr(
        seeder_module, "sync_temp_references_safely", lambda _progress: None
    )
    monkeypatch.setattr(scan_seeder, "_run_fast_phase", lambda roots: (0, 0, 0))

    with caplog.at_level(logging.INFO):
        scan_seeder._run_scan()

    assert events_named(caplog, "seeder.marked_missing") == [
        {"count": 5, "stage": "pruning"}
    ]


def test_standalone_mark_missing_emits_count_with_mark_missing_stage(
    scan_seeder: _AssetSeeder,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    scan_seeder._state = State.IDLE
    monkeypatch.setattr(seeder_module, "get_owned_prefixes", lambda: ())
    monkeypatch.setattr(
        seeder_module, "mark_missing_outside_prefixes_safely", lambda prefixes: 7
    )

    with caplog.at_level(logging.INFO):
        result = scan_seeder.mark_missing_outside_prefixes()

    assert result == 7
    assert events_named(caplog, "seeder.marked_missing") == [
        {"count": 7, "stage": "mark_missing"}
    ]


def test_standalone_mark_missing_failure_returns_none_and_emits_no_success_event(
    scan_seeder: _AssetSeeder,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    scan_seeder._state = State.IDLE
    monkeypatch.setattr(seeder_module, "get_owned_prefixes", lambda: [])

    def fail_create_session():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(scanner_module, "create_session", fail_create_session)

    with caplog.at_level(logging.INFO):
        result = scan_seeder.mark_missing_outside_prefixes()

    assert result is None
    assert events_named(caplog, "scanner.mark_missing_failed") == [
        {"error_type": "RuntimeError"}
    ]
    assert events_named(caplog, "seeder.marked_missing") == []


def test_scan_prune_failure_is_reported_and_the_scan_still_runs(
    scan_seeder: _AssetSeeder,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    scan_seeder._prune_first = True
    scan_seeder._phase = ScanPhase.FAST
    fast_phase_roots: list[tuple[str, ...]] = []
    monkeypatch.setattr(seeder_module, "get_owned_prefixes", lambda: [])
    monkeypatch.setattr(
        seeder_module, "mark_missing_outside_prefixes_safely", lambda _prefixes: None
    )
    monkeypatch.setattr(
        seeder_module, "sync_temp_references_safely", lambda _progress: None
    )

    def run_fast_phase(roots: tuple[str, ...]) -> tuple[int, int, int]:
        fast_phase_roots.append(roots)
        return 0, 0, 0

    monkeypatch.setattr(scan_seeder, "_run_fast_phase", run_fast_phase)

    with caplog.at_level(logging.INFO):
        scan_seeder._run_scan()

    assert scan_seeder._errors == [
        "Marking missing assets failed; scan continued without pruning"
    ]
    assert fast_phase_roots == [("models", "input")]
    assert events_named(caplog, "seeder.marked_missing") == []


def test_batch_insert_failure_emits_only_the_exception_type(
    scan_seeder: _AssetSeeder,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = Mock()
    monkeypatch.setattr(
        seeder_module, "sync_root_safely", lambda _root, _progress: set()
    )
    monkeypatch.setattr(
        seeder_module, "collect_paths_for_roots", lambda roots: ["asset.safetensors"]
    )
    monkeypatch.setattr(
        seeder_module,
        "build_asset_specs",
        lambda paths, existing_paths, enable_metadata_extraction, progress=None: (
            [{"tags": []}],
            {},
            0,
        ),
    )

    def fail_insert(batch, batch_tags) -> int:
        raise PermissionError("/private/models/asset.safetensors")

    monkeypatch.setattr(seeder_module, "insert_asset_specs", fail_insert)
    monkeypatch.setattr(seeder_module, "create_session", lambda: nullcontext(session))
    monkeypatch.setattr(seeder_module, "tick_watch_list", lambda: None)

    with caplog.at_level(logging.INFO):
        scan_seeder._run_fast_phase(("models",))

    assert events_named(caplog, "seeder.batch_insert_failed") == [
        {"error_type": "PermissionError"}
    ]
    tagged = "\n".join(record.getMessage() for record in caplog.records if TAG in record.getMessage())
    assert "/private/models/asset.safetensors" not in tagged


def test_batch_insert_fault_reports_the_specs_committed_before_it(
    scan_seeder: _AssetSeeder,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    with caplog.at_level(logging.INFO):
        engine, result = _run_faulting_fast_phase(
            scan_seeder,
            monkeypatch,
            tmp_path,
            OSError("forced record creation failure"),
        )

    with Session(engine) as session:
        assert {record.name for record in session.scalars(select(Asset))} == {
            "first.bin",
            "last.bin",
        }
    assert result == (2, 0, 3)
    assert scan_seeder._scan_state is not None
    assert scan_seeder._scan_state.created == 2
    assert scan_seeder._errors == [
        "Batch insert encountered an error at offset 0 after creating 2: "
        "forced record creation failure"
    ]


def test_batch_memory_error_stops_the_scan_instead_of_continuing(
    scan_seeder: _AssetSeeder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(MemoryError):
        _run_faulting_fast_phase(
            scan_seeder,
            monkeypatch,
            tmp_path,
            MemoryError("out of memory"),
        )

    assert scan_seeder._errors == []


def test_salvage_commit_failure_reports_the_original_batch_fault(
    scan_seeder: _AssetSeeder,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    original_fault = OSError("No space left on device")
    commit_failure = RuntimeError("forced salvage commit failure")

    with caplog.at_level(logging.INFO):
        engine, result = _run_faulting_fast_phase(
            scan_seeder,
            monkeypatch,
            tmp_path,
            original_fault,
            commit_failure,
        )

    with Session(engine) as session:
        assert session.scalar(select(Asset)) is None
    assert result == (0, 0, 3)
    assert scan_seeder._errors == [
        "Batch insert encountered an error at offset 0 after creating 0: "
        "No space left on device"
    ]
    assert events_named(caplog, "seeder.batch_insert_failed") == [
        {"error_type": "OSError"}
    ]
    caller_logs = [
        record
        for record in caplog.records
        if record.getMessage().startswith("Batch insert encountered an error")
    ]
    assert len(caller_logs) == 1
    assert caller_logs[0].exc_info is not None
    assert caller_logs[0].exc_info[1] is original_fault
