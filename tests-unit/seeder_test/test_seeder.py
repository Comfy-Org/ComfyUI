import json
import logging
import re
from contextlib import nullcontext
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.assets import seeder as seeder_module
from app.assets.database.models import Base
from app.assets.database.queries import create_content, create_record, mark_content_missing
from app.assets.event_log import TAG
from app.assets.seeder import Progress, ScanPhase, State, _AssetSeeder


EVENT_LINE_PATTERN = re.compile(
    rf"^{re.escape(TAG)} (?P<event>[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*) "
    r"(?P<fields>\{.*\})$"
)
EventFields = dict[str, bool | int | str]


@pytest.fixture
def scan_seeder(monkeypatch: pytest.MonkeyPatch) -> _AssetSeeder:
    instance = _AssetSeeder()
    instance._state = State.RUNNING
    instance._progress = Progress()
    instance._roots = ("models", "input")
    instance._phase = ScanPhase.FULL
    monkeypatch.setattr(seeder_module, "dependencies_available", lambda: True)
    monkeypatch.setattr(instance, "_log_scan_config", lambda roots: None)
    return instance


def tagged_events(caplog: pytest.LogCaptureFixture) -> list[tuple[str, EventFields]]:
    events: list[tuple[str, EventFields]] = []
    for record in caplog.records:
        match = EVENT_LINE_PATTERN.match(record.getMessage())
        if match is not None:
            events.append((match.group("event"), json.loads(match.group("fields"))))
    return events


def events_named(
    caplog: pytest.LogCaptureFixture, event_name: str
) -> list[EventFields]:
    return [fields for event, fields in tagged_events(caplog) if event == event_name]


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
    scan_seeder._progress = Progress(
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


def test_enrich_phase_counts_every_failed_reference(
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
    monkeypatch.setattr(seeder_module, "tick_watch_list", lambda _session: None)
    monkeypatch.setattr(seeder_module, "drain_transition_queue", lambda _session: None)
    monkeypatch.setattr(
        seeder_module,
        "get_unenriched_assets_for_roots",
        lambda *_args, **_kwargs: next(batches),
    )
    monkeypatch.setattr(
        seeder_module,
        "enrich_assets_batch",
        lambda *_args, **_kwargs: (0, ["record-1", "record-2"]),
    )
    monkeypatch.setattr(scan_seeder, "_check_pause_and_cancel", lambda _stage: False)

    cancelled, enriched = scan_seeder._run_enrich_phase(("models",))

    assert cancelled is False
    assert enriched == 0
    assert scan_seeder._progress is not None
    assert scan_seeder._progress.enrich_failed == 2


def test_starting_a_scan_installs_fresh_per_scan_failure_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _AssetSeeder()
    instance._progress = Progress(
        hash_failed=7,
        enrich_failed=4,
        permission_denied=2,
        enrich_failure_emitted=True,
    )
    monkeypatch.setattr(instance, "_run_scan", lambda: None)

    started = instance.start(roots=("models",), phase=ScanPhase.FAST)

    assert started is True
    assert instance._thread is not None
    instance._thread.join(timeout=5)
    assert instance._progress == Progress()


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

    monkeypatch.setattr(scan_seeder, "_check_pause_and_cancel", cancel_at_stage)
    monkeypatch.setattr(scan_seeder, "_run_fast_phase", lambda roots: (0, 0, 0))
    monkeypatch.setattr(scan_seeder, "_run_enrich_phase", lambda roots: (False, 0))

    with caplog.at_level(logging.INFO):
        scan_seeder._run_scan()

    assert events_named(caplog, "seeder.scan_cancelled") == [
        {"phase": phase.value, "stage": stage}
    ]


def test_enrich_interrupt_records_the_enrich_cancellation_stage(
    scan_seeder: _AssetSeeder,
) -> None:
    scan_seeder._cancel_event.set()

    assert scan_seeder._is_paused_or_cancelled() is True
    assert scan_seeder._progress is not None
    assert scan_seeder._progress.cancel_stage is not None
    assert scan_seeder._progress.cancel_stage.value == "enrich"


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
        lambda paths, existing_paths, enable_metadata_extraction: ([{"tags": []}], {}, 0),
    )

    def fail_insert(batch, batch_tags) -> int:
        raise PermissionError("/private/models/asset.safetensors")

    monkeypatch.setattr(seeder_module, "insert_asset_specs", fail_insert)
    monkeypatch.setattr(seeder_module, "create_session", lambda: nullcontext(session))
    monkeypatch.setattr(seeder_module, "tick_watch_list", lambda current_session: None)

    with caplog.at_level(logging.INFO):
        scan_seeder._run_fast_phase(("models",))

    assert events_named(caplog, "seeder.batch_insert_failed") == [
        {"error_type": "PermissionError"}
    ]
    tagged = "\n".join(record.getMessage() for record in caplog.records if TAG in record.getMessage())
    assert "/private/models/asset.safetensors" not in tagged
