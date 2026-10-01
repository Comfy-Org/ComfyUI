"""The startup prune keeps a row whose folder is still registered under another
spelling (a symlink or other alias, or letter case), rewriting its path to today's
spelling instead of retiring it. Boots are the prune followed by the seeder's real
fast phase, on an in-memory catalog."""

import logging
import os
import posixpath
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session as SASession, sessionmaker

import folder_paths
from app.assets import mode, scanner_rehome, seeder as seeder_module
from app.assets.database.models import Asset, AssetContent, Base
from app.assets.database.queries.records import create_content, create_record, mark_content_missing
from app.assets.helpers import cached_prefix_matcher
from app.assets.scanner import mark_contents_missing_outside_prefixes, mark_missing_outside_prefixes_safely
from app.assets.scanner_admission import _WATCH_LIST

pytestmark = pytest.mark.skipif(os.name == "nt", reason="builds symlinks and simulates case folding on POSIX")

ALL_ROOTS = ("models", "input", "output")
OUTPUT_FILES = ("a.png", "b.png", os.path.join("sub", "c.png"), os.path.join("sub", "deep", "d.png"))
MODEL_FILES = ("m1.safetensors", os.path.join("sd", "m2.safetensors"))


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


class Folders:
    """The folder configuration one boot sees; ``use`` switches it between boots."""

    def __init__(self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch):
        self.base = temp_dir
        self._monkeypatch = monkeypatch
        for name in ("input", "temp"):
            (temp_dir / name).mkdir()
        monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(temp_dir / "temp"))
        self.use(output=None, models=None)

    def use(self, *, output: Path | None, models: Path | None, input: Path | None = None) -> None:
        input_dir = str(input if input is not None else self.base / "input")
        self._monkeypatch.setattr(folder_paths, "get_input_directory", lambda: input_dir)
        output_dir = str(output if output is not None else self.base / "no-output")
        self._monkeypatch.setattr(folder_paths, "get_output_directory", lambda: output_dir)
        registered = {} if models is None else {"checkpoints": ([str(models)], {".safetensors"})}
        self._monkeypatch.setattr(folder_paths, "folder_names_and_paths", registered)
        self._monkeypatch.setattr(folder_paths, "filename_list_cache", {})


@pytest.fixture
def folders(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Folders:
    return Folders(temp_dir, monkeypatch)


@pytest.fixture
def folds_case(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Compare paths the way Windows does. The filesystem stays case-sensitive, so the
    tests reach one folder under two spellings through a symlink."""
    monkeypatch.setattr(posixpath, "normcase", lambda path: os.fspath(path).lower())
    cached_prefix_matcher.cache_clear()
    yield
    cached_prefix_matcher.cache_clear()


def _populate(root: Path, names: tuple[str, ...]) -> None:
    for i, name in enumerate(names):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * (i + 1))


def _alias(target: Path, link: Path) -> Path:
    """``link`` reaching ``target``; as-is if it already does (a case-insensitive disk)."""
    if not link.exists():
        link.symlink_to(target, target_is_directory=True)
    return link


def _boot(caplog: pytest.LogCaptureFixture | None = None) -> int:
    """One startup: the prune, then the fast phase. Returns the rows it created."""
    seeder = seeder_module._AssetSeeder()
    seeder._scan_state = seeder_module._ScanState()
    seeder._roots = ALL_ROOTS
    seeder._phase = seeder_module.ScanPhase.FAST
    seeder._prune_first = True
    seeder._run_gate.set()
    seeder._cancel_event.clear()
    created: list[int] = []
    run_fast_phase = seeder._run_fast_phase

    def record_created(roots):
        result = run_fast_phase(roots)
        created.append(result[0])
        return result

    seeder._run_fast_phase = record_created
    if caplog is not None:
        caplog.clear()
        with caplog.at_level(logging.INFO):
            seeder._run_scan()
    else:
        seeder._run_scan()
    assert seeder._errors == []
    return created[0]


def _records(session) -> dict[str, tuple[str, str]]:
    """record id -> (content path, name), live rows only."""
    session.expire_all()
    rows = session.execute(
        sa.select(Asset.id, AssetContent.path, Asset.name)
        .join(AssetContent, Asset.content_id == AssetContent.id)
        .where(AssetContent.is_missing.is_(False))
    )
    return {record_id: (path, name) for record_id, path, name in rows}


def _missing_count(session) -> int:
    session.expire_all()
    return session.scalar(
        sa.select(sa.func.count()).select_from(AssetContent).where(AssetContent.is_missing.is_(True))
    )


def _rename_all(session) -> None:
    for record in session.scalars(sa.select(Asset)):
        record.name = f"kept-{record.id}"
    session.commit()


def _marked_missing_event(caplog: pytest.LogCaptureFixture) -> dict[str, str]:
    lines = [r.getMessage() for r in caplog.records if "seeder.marked_missing" in r.getMessage()]
    assert len(lines) == 1, lines
    return dict(pair.split("=", 1) for pair in lines[0].split()[2:])


def _respelled(records: dict[str, tuple[str, str]], old: Path, new: Path) -> dict[str, tuple[str, str]]:
    return {rid: (path.replace(str(old), str(new), 1), name) for rid, (path, name) in records.items()}


def _row(session, path: Path, tags: tuple[str, ...] = ("output",)) -> str:
    """A row catalogued as a file directly under the output folder."""
    content = create_content(session, path=str(path), size_bytes=1, mtime_ns=1)
    create_record(session, content_id=content.id, name=path.name, loader_path=path.name, tags=list(tags))
    session.flush()
    return content.id


def _live(session, content_id: str) -> str | None:
    session.expire_all()
    content = session.get(AssetContent, content_id)
    return None if content.is_missing else content.path


def _prune(session, *prefixes: Path) -> tuple[int, int]:
    return tuple(mark_contents_missing_outside_prefixes(session, [str(p) for p in prefixes]))


# --- whole boots -----------------------------------------------------------------------------

def test_output_seeded_through_a_symlink_keeps_its_records_when_booted_by_the_real_path(
    folders, temp_dir, session, caplog
):
    real = temp_dir / "real" / "output"
    _populate(real, OUTPUT_FILES)
    alias = _alias(temp_dir / "real", temp_dir / "alias") / "output"
    folders.use(output=alias, models=None)
    assert _boot() == len(OUTPUT_FILES)
    _rename_all(session)
    before = _records(session)

    folders.use(output=real, models=None)

    assert _boot(caplog) == 0
    assert _records(session) == _respelled(before, alias, real)
    assert _missing_count(session) == 0
    assert _marked_missing_event(caplog) == {"count": "0", "rehomed_count": str(len(OUTPUT_FILES)), "stage": "pruning"}
    assert _boot(caplog) == 0
    assert _marked_missing_event(caplog)["rehomed_count"] == "0"


def test_a_model_folder_registered_through_an_alias_keeps_its_records(folders, temp_dir, session):
    real = temp_dir / "disk" / "checkpoints"
    _populate(real, MODEL_FILES)
    alias = _alias(temp_dir / "disk", temp_dir / "mnt") / "checkpoints"
    folders.use(output=None, models=alias)
    assert _boot() == len(MODEL_FILES)
    _rename_all(session)
    before = _records(session)

    folders.use(output=None, models=real)

    assert _boot() == 0
    assert _records(session) == _respelled(before, alias, real)


def test_a_different_folder_with_the_same_layout_is_not_rehomed(folders, temp_dir, session):
    old, new = temp_dir / "old" / "output", temp_dir / "new" / "output"
    _populate(old, OUTPUT_FILES)
    _populate(new, OUTPUT_FILES)
    folders.use(output=old, models=None)
    _boot()
    old_ids = set(_records(session))

    folders.use(output=new, models=None)

    assert _boot() == len(OUTPUT_FILES)
    assert not old_ids & set(_records(session))
    assert _missing_count(session) == len(OUTPUT_FILES)


def test_a_case_only_respelling_neither_duplicates_nor_retires(folders, folds_case, temp_dir, session):
    real = temp_dir / "data" / "output"
    _populate(real, OUTPUT_FILES)
    if (temp_dir / "DATA").exists():
        # macOS: realpath keeps the case it is given, so the row keeps master's outcome
        # there. Windows' realpath returns the on-disk case.
        pytest.skip("this filesystem folds case, so DATA can't be a separate symlink")
    upper = _alias(temp_dir / "data", temp_dir / "DATA") / "output"
    folders.use(output=upper, models=None)
    _boot()
    _rename_all(session)
    before = _records(session)

    folders.use(output=real, models=None)

    assert _boot() == 0
    assert _records(session) == _respelled(before, upper, real)
    assert _missing_count(session) == 0


def test_existing_case_duplicates_keep_their_edits_visible_whichever_spelling_launches(
    folders, folds_case, temp_dir, session
):
    """Duplicates left by a case-only relaunch before this change: the edited row under one
    spelling, a scan stub under the other. Neither can move onto the other, and neither
    may retire, or the edits vanish on every other launch."""
    data = temp_dir / "data"
    _populate(data, ("f.png",))
    upper = _alias(data, temp_dir / "DATA")
    if not upper.is_symlink():
        pytest.skip("this filesystem folds case, so DATA can't be a separate symlink")
    edited = _row(session, upper / "f.png")
    session.scalars(sa.select(Asset).where(Asset.content_id == edited)).one().name = "edited"
    _row(session, data / "f.png")
    session.commit()

    for spelling in (data, upper, data, upper):
        folders.use(output=spelling, models=None)
        assert _boot() == 0
        assert "edited" in {name for _, name in _records(session).values()}
        assert _missing_count(session) == 0


def test_a_case_sensitive_sibling_folder_is_not_mistaken_for_the_same_one(folders, folds_case, temp_dir, session):
    """Windows can mark a directory case-sensitive, so output and Output can be two folders."""
    lower, upper = temp_dir / "cs" / "output", temp_dir / "cs" / "Output"
    _populate(lower, OUTPUT_FILES)
    if (temp_dir / "cs" / "OUTPUT").exists():
        pytest.skip("this filesystem cannot hold two names differing only in case")
    _populate(upper, OUTPUT_FILES)  # same names and sizes, different files
    folders.use(output=lower, models=None)
    _boot()
    old_ids = set(_records(session))

    folders.use(output=upper, models=None)

    # Not re-homed onto the other folder's files; left live, as master leaves it.
    assert _boot() == len(OUTPUT_FILES)
    assert {path for rid, (path, _) in _records(session).items() if rid in old_ids} == {
        str(lower / name) for name in OUTPUT_FILES
    }


def test_input_and_output_on_one_folder_in_two_case_spellings_stay_stable(folders, folds_case, temp_dir, session):
    shared = temp_dir / "shared"
    _populate(shared, OUTPUT_FILES)
    folders.use(output=_alias(shared, temp_dir / "SHARED"), models=None, input=shared)
    _boot()
    before = _records(session)

    for _ in range(2):
        assert _boot() == 0
        assert _records(session) == before
        assert _missing_count(session) == 0


def test_a_model_folder_inside_output_rehomes_into_the_output_folder(folders, temp_dir, session):
    out = temp_dir / "real" / "out"
    _populate(out, ("checkpoints/m.safetensors", "a.png"))
    folders.use(output=_alias(temp_dir / "real", temp_dir / "A") / "out", models=None)
    _boot()
    _rename_all(session)
    before = _records(session)

    folders.use(output=out, models=_alias(out / "checkpoints", temp_dir / "L"))
    _boot()

    after = _records(session)
    assert {rid: after[rid] for rid in before} == _respelled(before, temp_dir / "A" / "out", out)


def test_input_and_output_on_one_folder_keep_each_row_in_its_own_root(folders, temp_dir, session):
    shared = temp_dir / "shared"
    _populate(shared, ("f.png",))
    folders.use(output=_alias(shared, temp_dir / "X1"), models=None)
    _boot()
    (record_id,) = _records(session)

    folders.use(output=shared, models=None, input=_alias(shared, temp_dir / "Y"))
    _boot()

    assert _records(session)[record_id][0] == str(shared / "f.png")


@pytest.mark.parametrize("case_only", [False, True], ids=["alias", "case-only"])
def test_a_row_whose_folder_now_has_only_another_role_is_not_rehomed(
    folders, folds_case, temp_dir, session, case_only
):
    data = temp_dir / "data"
    _populate(data, ("f.png",))
    folders.use(output=_alias(data, temp_dir / ("DATA" if case_only else "A")), models=None)
    _boot()
    (old_id,) = _records(session)

    folders.use(output=None, models=None, input=data)
    _boot()

    # Retired, or for a case-only row left live as master leaves it.
    live = _records(session)
    assert (old_id in live) is case_only
    assert sorted(path for path, _ in live.values()) == sorted(
        [str(data / "f.png")] + ([str(temp_dir / "DATA" / "f.png")] if case_only else [])
    )


@pytest.mark.parametrize("hashing", [False, True], ids=["hashing-off", "hashing-on"])
def test_a_folder_split_by_the_old_bug_gets_its_history_back_when_spelled_as_before(
    folders, temp_dir, session, hashing
):
    """The old bug retired the rows under the first spelling and re-created stubs under
    a second. Spelled as before, the stubs must not take the paths recovery needs."""

    class _Hashing:
        enable_asset_hashing = hashing

    mode.init(_Hashing())
    real = temp_dir / "real" / "output"
    _populate(real, OUTPUT_FILES)
    first = _alias(temp_dir / "real", temp_dir / "first") / "output"
    second = _alias(temp_dir / "real", temp_dir / "second") / "output"
    folders.use(output=first, models=None)
    _boot()
    _rename_all(session)
    before = _records(session)
    for content in session.scalars(sa.select(AssetContent).where(AssetContent.is_missing.is_(False))).all():
        mark_content_missing(session, content.id)
        name = content.path[len(str(first)) + 1:]
        stub = create_content(session, path=str(second / name), size_bytes=content.size_bytes, mtime_ns=content.mtime_ns)
        create_record(
            session, content_id=stub.id, name=os.path.basename(name), loader_path=name.replace(os.sep, "/"),
            tags=["output"],
        )
    session.commit()

    folders.use(output=first, models=None)
    assert _boot() == 0

    assert _records(session) == before


# --- the prune, row by row -------------------------------------------------------------------

def test_a_gone_file_and_a_file_outside_every_folder_are_retired(session, temp_dir):
    _populate(temp_dir / "old", ("f.png",))
    (temp_dir / "new").mkdir()
    rows = [_row(session, temp_dir / "old" / "f.png"), _row(session, temp_dir / "old" / "gone.png")]

    assert _prune(session, temp_dir / "new") == (2, 0)
    assert [_live(session, row) for row in rows] == [None, None]


def test_an_unreadable_parent_skips_the_per_file_stat(session, temp_dir, monkeypatch):
    real = temp_dir / "real"
    _populate(real, ("f.png", "g.png"))
    alias = _alias(real, temp_dir / "share")
    for name in ("f.png", "g.png"):
        _row(session, alias / name)
    real_realpath, real_stat = os.path.realpath, os.stat
    stat_calls: list[str] = []

    def offline_share(path, *args, **kwargs):
        if os.fspath(path) == str(alias):
            raise OSError(112, "host is down")
        return real_realpath(path, *args, **kwargs)

    monkeypatch.setattr(os.path, "realpath", offline_share)
    monkeypatch.setattr(os, "stat", lambda path, *a, **k: stat_calls.append(path) or real_stat(path, *a, **k))

    assert _prune(session, real) == (2, 0)
    assert stat_calls == []


def _patch_stat(monkeypatch, override) -> None:
    real_stat = os.stat
    monkeypatch.setattr(os, "stat", lambda path, *a, **k: override(os.fspath(path)) or real_stat(path, *a, **k))


def test_a_filesystem_without_inode_numbers_is_never_rehomed(folders, session, temp_dir, monkeypatch):
    real = temp_dir / "real"
    _populate(real, ("f.png",))
    folders.use(output=real, models=None)
    row = _row(session, _alias(real, temp_dir / "share") / "f.png")
    real_stat = os.stat
    _patch_stat(monkeypatch, lambda path: os.stat_result([*list(real_stat(path))[:1], 0, *list(real_stat(path))[2:]]))

    assert _prune(session, real) == (1, 0)
    assert _live(session, row) is None


def test_a_different_file_behind_the_same_folder_is_not_rehomed(folders, session, temp_dir, monkeypatch):
    """A relative ../ symlink inside a bind-mounted folder names a different file under each spelling."""
    real = temp_dir / "real"
    _populate(real, ("f.png",))
    _populate(temp_dir / "stranger", ("f.png",))
    folders.use(output=real, models=None)
    row = _row(session, _alias(real, temp_dir / "alias") / "f.png")
    real_stat = os.stat
    _patch_stat(monkeypatch, lambda path: real_stat(temp_dir / "stranger" / "f.png") if path == str(real / "f.png") else None)

    assert _prune(session, real) == (1, 0)
    assert _live(session, row) is None


def test_a_row_owned_by_a_shallower_folder_is_left_as_spelled(folds_case, session, temp_dir):
    shallow = temp_dir / "models"
    (shallow / "output").mkdir(parents=True)
    row = _row(session, shallow / "output" / "f.png")

    assert _prune(session, shallow, _alias(shallow / "output", shallow / "Output")) == (0, 0)
    assert _live(session, row) == str(shallow / "output" / "f.png")


def test_a_row_both_case_variant_folders_fit_is_left_as_spelled(folders, folds_case, session, temp_dir):
    """Input and output on one folder in two case spellings: where case folds, a file
    there is tagged with both roles, so either spelling fits and neither is chosen."""
    data = temp_dir / "data"
    _populate(data, ("f.png",))
    upper = _alias(data, temp_dir / "DATA")
    folders.use(output=upper, models=None, input=data)
    row = _row(session, _alias(data, temp_dir / "Data") / "f.png", tags=("input", "output"))

    assert _prune(session, data, upper) == (0, 0)
    assert _live(session, row) == str(temp_dir / "Data" / "f.png")


def test_two_rows_resolving_to_one_target_both_retire(folders, session, temp_dir):
    real = temp_dir / "real"
    _populate(real, ("f.png",))
    folders.use(output=real, models=None)
    rows = [_row(session, _alias(real, temp_dir / name) / "f.png") for name in ("a1", "a2")]

    assert _prune(session, real) == (2, 0)
    assert [_live(session, row) for row in rows] == [None, None]


def test_a_live_row_at_the_target_keeps_it(folders, session, temp_dir):
    data = temp_dir / "data"
    _populate(data, ("f.png",))
    folders.use(output=data, models=None)
    occupant = _row(session, data / "f.png")
    mover_path = _alias(data, temp_dir / "alias") / "f.png"
    mover = _row(session, mover_path)

    # Decided by the plan, not left to the write's conflict fallback.
    assert scanner_rehome.plan_prune(session, [(mover, str(mover_path))], [str(data)]).moves == {}
    assert _prune(session, data) == (1, 0)
    assert (_live(session, occupant), _live(session, mover)) == (str(data / "f.png"), None)


def test_a_missing_row_with_records_at_the_target_is_left_to_recovery(folders, session, temp_dir):
    real = temp_dir / "real"
    _populate(real, ("f.png",))
    folders.use(output=real, models=None)
    parked = _row(session, real / "f.png")
    mark_content_missing(session, parked)
    mover = _row(session, _alias(real, temp_dir / "alias") / "f.png")

    assert _prune(session, real) == (1, 0)
    assert (_live(session, parked), _live(session, mover)) == (None, None)


def test_a_verbatim_prefix_resolves_against_a_plain_spelling(folders, session, temp_dir, monkeypatch):
    real = temp_dir / "real"
    _populate(real, ("f.png",))
    folders.use(output=real, models=None)
    row = _row(session, _alias(real, temp_dir / "alias") / "f.png")
    real_realpath = os.path.realpath
    # Only the registered folder resolves to the verbatim form.
    monkeypatch.setattr(
        os.path, "realpath",
        lambda path, **kw: ("\\\\?\\" if os.fspath(path) == str(real) else "") + real_realpath(path, **kw),
    )

    assert _prune(session, real) == (0, 1)
    assert _live(session, row) == str(real / "f.png")


def test_a_hung_mount_costs_a_bounded_wait(folders, folds_case, session, temp_dir, monkeypatch):
    real = temp_dir / "real"
    _populate(real, ("ok.png",))
    folders.use(output=real, models=None)
    kept = _row(session, _alias(real, temp_dir / "alias") / "ok.png")
    hung = [_row(session, temp_dir / "dead" / d / "f.png") for d in ("a", "b")]
    registered_hung = temp_dir / "dead" / "zz"
    case_only = _row(session, temp_dir / "dead" / "ZZ" / "f.png")
    monkeypatch.setattr(scanner_rehome, "STALL_SECONDS", 0.2)
    release = threading.Event()
    real_realpath = os.path.realpath

    def hard_mount(path, **kwargs):
        if os.fspath(path).startswith(str(temp_dir / "dead")) and os.fspath(path) != str(registered_hung):
            release.wait(30)
        return real_realpath(path, **kwargs)

    monkeypatch.setattr(os.path, "realpath", hard_mount)
    try:
        started = time.monotonic()
        result = _prune(session, real, registered_hung)
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert elapsed < 5
    assert result == (2, 1)
    assert [_live(session, row) for row in hung] == [None, None]
    assert _live(session, case_only) == str(temp_dir / "dead" / "ZZ" / "f.png")  # undecided, left as before
    assert _live(session, kept) == str(real / "ok.png")


@pytest.mark.parametrize("spelling", ["alias", "REAL"])
def test_a_target_taken_by_a_racing_writer_skips_that_row_only(
    folders, folds_case, session, temp_dir, monkeypatch, spelling
):
    real = temp_dir / "real"
    _populate(real, ("f.png", "g.png"))
    folders.use(output=real, models=None)
    alias = _alias(real, temp_dir / spelling)
    if not alias.is_symlink():
        pytest.skip("this filesystem folds case, so REAL can't be a separate symlink")
    raced, moved = _row(session, alias / "f.png"), _row(session, alias / "g.png")
    racer = _row(session, real / "f.png")
    # The racer's insert lands between the plan's read and the rewrite.
    monkeypatch.setattr(scanner_rehome, "_taken_paths", lambda _session, _paths: set())

    # The raced row is retired, or for a case-only row left live as master leaves it.
    retired = spelling == "alias"
    assert _prune(session, real) == (int(retired), 1)
    assert _live(session, raced) == (None if retired else str(alias / "f.png"))
    assert _live(session, moved) == str(real / "g.png")
    assert _live(session, racer) == str(real / "f.png")


def test_an_integrity_error_other_than_a_path_race_is_not_swallowed(folders, session, temp_dir, monkeypatch):
    real = temp_dir / "real"
    _populate(real, ("f.png",))
    folders.use(output=real, models=None)
    _row(session, _alias(real, temp_dir / "alias") / "f.png")
    original_execute = session.execute

    def failing_update(statement, *args, **kwargs):
        if getattr(statement, "is_update", False):
            raise IntegrityError("UPDATE", {}, Exception("FOREIGN KEY constraint failed"))
        return original_execute(statement, *args, **kwargs)

    monkeypatch.setattr(session, "execute", failing_update)

    with pytest.raises(IntegrityError):
        _prune(session, real)


def test_a_failed_prune_on_a_file_database_commits_none_of_its_moves(folders, temp_dir, monkeypatch):
    """Engines set up as app.database.db does: a later failure must not leave earlier batches written."""
    url = f"sqlite:///{temp_dir / 'assets.db'}"
    read, write = create_engine(url), create_engine(url)
    event.listen(write, "connect", lambda connection, _record: setattr(connection, "isolation_level", None))
    event.listen(write, "begin", lambda connection: connection.exec_driver_sql("BEGIN IMMEDIATE"))
    Base.metadata.create_all(read)
    real = temp_dir / "real"
    _populate(real, ("f.png", "g.png"))
    folders.use(output=real, models=None)
    alias = _alias(real, temp_dir / "alias")
    with SASession(read) as sess:
        for name in ("f.png", "g.png"):
            _row(sess, alias / name)
        sess.commit()
    monkeypatch.setattr(scanner_rehome, "_BATCH", 1)
    rewrite, calls = scanner_rehome._rewrite, []

    def second_batch_fails(session, moves):
        calls.append(moves)
        if len(calls) == 2:
            raise IntegrityError("UPDATE", {}, Exception("FOREIGN KEY constraint failed"))
        rewrite(session, moves)

    monkeypatch.setattr(scanner_rehome, "_rewrite", second_batch_fails)
    with patch("app.assets.scanner.create_session", lambda: SASession(read)), \
         patch("app.database.db.WriteSession", sessionmaker(bind=write)):
        assert mark_missing_outside_prefixes_safely([str(real)]) is None

    with SASession(read) as sess:
        assert sorted(sess.scalars(sa.select(AssetContent.path))) == [str(alias / "f.png"), str(alias / "g.png")]
