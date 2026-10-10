"""A scan over models and other roots catalogues models in a pass of their own and
announces them with assets.seed.fast_complete before the other roots are walked, so a
large or slow input folder does not hold up the model library. Run through the seeder's
real scan loop on an in-memory catalog."""

import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session as SASession, sessionmaker
from sqlalchemy.pool import StaticPool

import folder_paths
from app.assets import scanner, seeder as seeder_module
from app.assets.database.models import AssetContent, Base
from app.assets.scanner_admission import _WATCH_LIST

# More models than inputs, so input counts that restarted from 0 would go backwards.
MODELS = 4
INPUTS = 3
OUTPUTS = 2


@pytest.fixture
def db_engine():
    """One in-memory catalog shared with the scan thread."""
    engine = sa.create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture(autouse=True)
def isolated_state(db_engine):
    @contextmanager
    def _create_session():
        with SASession(db_engine) as sess:
            yield sess

    _WATCH_LIST.clear()
    with patch("app.assets.scanner.create_session", _create_session), \
         patch("app.assets.seeder.create_session", _create_session), \
         patch("app.database.db.WriteSession", sessionmaker(bind=db_engine)):
        yield
    _WATCH_LIST.clear()


@pytest.fixture
def layout(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    dirs = {name: temp_dir / name for name in ("checkpoints", "input", "output", "temp")}
    for path in dirs.values():
        path.mkdir()
    for i in range(MODELS):
        (dirs["checkpoints"] / f"m{i}.safetensors").write_bytes(b"m" * (i + 1))
    for i in range(INPUTS):
        sub = dirs["input"] / f"d{i}"
        sub.mkdir()
        (sub / "a.png").write_bytes(b"i" * (i + 1))
    for i in range(OUTPUTS):
        (dirs["output"] / f"o{i}.png").write_bytes(b"o" * (i + 1))
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {
        "checkpoints": ([str(dirs["checkpoints"])], {".safetensors"}),
    })
    monkeypatch.setattr(folder_paths, "filename_list_cache", {})
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(dirs["input"]))
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(dirs["output"]))
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(dirs["temp"]))
    return dirs


def _rows(engine, under: Path) -> int:
    with SASession(engine) as sess:
        return sess.scalar(
            sa.select(sa.func.count()).select_from(AssetContent)
            .where(AssetContent.path.startswith(str(under)))
        )


class _Scan:
    """A seeder set up as start() would, recording each event with the catalog's
    model and input row counts at the moment it was sent."""

    def __init__(self, engine, layout, roots, phase=seeder_module.ScanPhase.FAST):
        self.engine = engine
        self.layout = layout
        self.events: list[tuple[str, dict, int, int]] = []
        self.progress: list[seeder_module.Progress] = []
        self.seeder = seeder_module._AssetSeeder()
        self.seeder._state = seeder_module.State.RUNNING
        self.seeder._scan_state = seeder_module._ScanState()
        self.seeder._roots = roots
        self.seeder._phase = phase
        self.seeder._prune_first = True
        self.seeder._progress_callback = self.progress.append
        self.seeder._run_gate.set()
        self.seeder.set_event_sink(self._sink)
        self.on_event = None

    def _sink(self, kind, data):
        counts = (_rows(self.engine, self.layout["checkpoints"]), _rows(self.engine, self.layout["input"]))
        self.events.append((kind, data, *counts))
        if self.on_event is not None:
            self.on_event(kind, data)

    def named(self, kind):
        return [e for e in self.events if e[0] == kind]

    def run_in_thread(self) -> threading.Thread:
        worker = threading.Thread(target=self.seeder._run_scan, daemon=True)
        worker.start()
        return worker


@pytest.fixture
def slow_input_walk(layout, monkeypatch):
    """Holds the input folder's walk open until ``release`` is set."""
    started, release = threading.Event(), threading.Event()
    real = scanner.list_files_recursively

    def walk(path, *args, **kwargs):
        if path == str(layout["input"]):
            started.set()
            assert release.wait(10)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(scanner, "list_files_recursively", walk)
    yield started, release
    release.set()


def test_models_are_catalogued_and_announced_while_the_input_walk_is_still_running(
    db_engine, layout, slow_input_walk
):
    started, release = slow_input_walk
    scan = _Scan(db_engine, layout, ("models", "input"))
    models_done = threading.Event()
    scan.on_event = lambda kind, data: (
        kind == "assets.seed.fast_complete" and data["roots"] == ["models"] and models_done.set()
    )
    worker = scan.run_in_thread()
    try:
        assert models_done.wait(10)
        assert started.wait(10)
        assert _rows(db_engine, layout["checkpoints"]) == MODELS
        assert _rows(db_engine, layout["input"]) == 0
        assert not scan.named("assets.seed.completed")
    finally:
        release.set()
        worker.join(10)

    assert not worker.is_alive()
    assert _rows(db_engine, layout["input"]) == INPUTS


def test_each_pass_sends_its_own_fast_complete_in_order(db_engine, layout):
    scan = _Scan(db_engine, layout, ("models", "input"))
    scan.seeder._run_scan()

    kinds = [e[0] for e in scan.events]
    fast = scan.named("assets.seed.fast_complete")
    assert [(e[1], e[2], e[3]) for e in fast] == [
        ({"roots": ["models"], "created": MODELS, "skipped": 0, "total": MODELS}, MODELS, 0),
        ({"roots": ["input"], "created": INPUTS, "skipped": 0, "total": INPUTS}, MODELS, INPUTS),
    ]
    assert kinds.index("assets.seed.completed") > kinds.index("assets.seed.fast_complete")
    (completed,) = scan.named("assets.seed.completed")
    assert completed[1]["created"] == MODELS + INPUTS
    assert completed[1]["total"] == MODELS + INPUTS


def test_enrich_runs_once_over_every_root_after_both_fast_passes(db_engine, layout):
    scan = _Scan(db_engine, layout, ("models", "input"), phase=seeder_module.ScanPhase.FULL)
    scan.seeder._run_scan()

    kinds = [e[0] for e in scan.events if e[0] in ("assets.seed.fast_complete", "assets.seed.enrich_complete")]
    assert kinds == ["assets.seed.fast_complete"] * 2 + ["assets.seed.enrich_complete"]
    (enrich,) = scan.named("assets.seed.enrich_complete")
    assert enrich[1]["roots"] == ["models", "input"]


@pytest.mark.parametrize("roots", [("models",), ("input",)])
def test_a_scan_of_one_root_sends_one_fast_complete(db_engine, layout, roots):
    scan = _Scan(db_engine, layout, roots)
    scan.seeder._run_scan()

    assert [e[1]["roots"] for e in scan.named("assets.seed.fast_complete")] == [list(roots)]


def test_progress_accumulates_across_the_passes(db_engine, layout):
    scan = _Scan(db_engine, layout, ("models", "input"))
    scan.seeder._run_scan()

    final = scan.progress[-1]
    assert (final.total, final.scanned, final.created) == (MODELS + INPUTS,) * 3
    for earlier, later in zip(scan.progress, scan.progress[1:]):
        assert later.total >= earlier.total
        assert later.scanned >= earlier.scanned
        assert later.created >= earlier.created


def test_skipped_files_accumulate_across_the_passes(db_engine, layout):
    _Scan(db_engine, layout, ("models", "input")).seeder._run_scan()
    rescan = _Scan(db_engine, layout, ("models", "input"))
    rescan.seeder._run_scan()

    assert rescan.progress[-1].skipped == MODELS + INPUTS


def test_a_prompt_during_the_input_walk_parks_the_scan_after_models_were_announced(
    db_engine, layout, slow_input_walk
):
    started, release = slow_input_walk
    scan = _Scan(db_engine, layout, ("models", "input"))
    parked = threading.Event()
    scan.on_event = lambda kind, _data: kind == "assets.seed.paused" and parked.set()
    worker = scan.run_in_thread()
    try:
        assert started.wait(10)
        assert scan.seeder.pause()
        release.set()
        assert parked.wait(10)
        assert [e[1]["roots"] for e in scan.named("assets.seed.fast_complete")] == [["models"]]
        assert _rows(db_engine, layout["input"]) == 0
        assert scan.seeder.resume()
    finally:
        release.set()
        worker.join(10)

    assert not worker.is_alive()
    assert [e[1]["roots"] for e in scan.named("assets.seed.fast_complete")] == [["models"], ["input"]]
    assert _rows(db_engine, layout["input"]) == INPUTS


def test_a_cancel_after_the_models_pass_skips_the_input_announcement(db_engine, layout):
    scan = _Scan(db_engine, layout, ("models", "input"))
    scan.on_event = lambda kind, data: (
        kind == "assets.seed.fast_complete" and data["roots"] == ["models"] and scan.seeder.cancel()
    )
    scan.seeder._run_scan()

    assert [e[1]["roots"] for e in scan.named("assets.seed.fast_complete")] == [["models"]]
    assert scan.named("assets.seed.cancelled")
    assert not scan.named("assets.seed.completed")
    assert _rows(db_engine, layout["input"]) == 0


def test_with_output_in_the_roots_input_and_output_share_the_second_pass(
    db_engine, layout, slow_input_walk
):
    started, release = slow_input_walk
    scan = _Scan(db_engine, layout, ("models", "input", "output"))
    worker = scan.run_in_thread()
    try:
        assert started.wait(10)
        assert [e[1]["roots"] for e in scan.named("assets.seed.fast_complete")] == [["models"]]
        assert _rows(db_engine, layout["checkpoints"]) == MODELS
    finally:
        release.set()
        worker.join(10)

    assert not worker.is_alive()
    fast = scan.named("assets.seed.fast_complete")
    assert [(e[1]["roots"], e[1]["created"]) for e in fast] == [
        (["models"], MODELS), (["input", "output"], INPUTS + OUTPUTS),
    ]
    assert _rows(db_engine, layout["output"]) == OUTPUTS


def test_the_output_pass_of_a_models_and_output_scan_is_not_the_output_only_rescan(
    db_engine, layout, monkeypatch
):
    def listing():
        raise AssertionError("the output-only rescan's listing ran")

    monkeypatch.setattr(seeder_module, "list_output_for_rescan", listing)
    scan = _Scan(db_engine, layout, ("models", "output"))
    scan.seeder._run_scan()

    assert [e[1]["roots"] for e in scan.named("assets.seed.fast_complete")] == [["models"], ["output"]]
    assert scan.named("assets.seed.completed")
    assert _rows(db_engine, layout["output"]) == OUTPUTS
