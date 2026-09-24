"""The catch sites that used to swallow a failure now classify it, emit the first
one per scan, and count every one into the scan's failure buckets."""

import errno
import logging
import os
import struct
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from sqlalchemy.orm import Session

from app.assets import scanner
from app.assets import seeder as seeder_module
from app.assets.database.models import Asset
from app.assets.database.queries import create_record
from app.assets.event_log import TAG
from app.assets.scanner import SeedAssetSpec
from app.assets.scanner_admission import _WATCH_LIST, _WatchEntry, tick_watch_list
from app.assets.seeder import _emit_failure_buckets, _ScanState
from app.assets.services.file_utils import list_files_recursively
from app.assets.services.image_dimensions import extract_image_dimensions
from app.assets.services.metadata_extract import extract_file_metadata

EventFields = dict[str, str | int]


def events(caplog: pytest.LogCaptureFixture, name: str) -> list[EventFields]:
    found: list[EventFields] = []
    for record in caplog.records:
        message = record.getMessage()
        if not message.startswith(f"{TAG} {name}"):
            continue
        fields: EventFields = {}
        for pair in message.split()[2:]:
            key, value = pair.split("=", 1)
            fields[key] = int(value) if value.removeprefix("-").isdigit() else value
        found.append(fields)
    return found


def buckets(progress: _ScanState) -> Counter[tuple[str, str]]:
    """The scan's failure counts keyed by (site, reason)."""
    counts: Counter[tuple[str, str]] = Counter()
    for (site, failure), count in progress.failure_buckets.items():
        counts[(site, failure.reason)] += count
    return counts


def fake_os(stat) -> SimpleNamespace:
    return SimpleNamespace(stat=stat, path=scanner.os.path)


def failing_stat(*errnos: int, winerror: int | None = None):
    """A stat that raises a fresh OSError per call, cycling through ``errnos``."""
    calls = iter(range(sys.maxsize))

    def stat(*_args, **_kwargs):
        code = errnos[next(calls) % len(errnos)]
        exc = OSError(code, "/private/share/secret.safetensors")
        if winerror is not None:
            exc.winerror = winerror  # type: ignore[attr-defined]
        raise exc

    return stat


def _spec(path: Path) -> SeedAssetSpec:
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


# --- reference_stat ------------------------------------------------------------------


def observe_references(monkeypatch: pytest.MonkeyPatch, stat, rows: int = 3):
    contents = [
        SimpleNamespace(id=f"content-{i}", path=f"/private/share/{i}.bin", size_bytes=1, mtime_ns=1)
        for i in range(rows)
    ]
    monkeypatch.setattr(scanner, "os", fake_os(stat))
    monkeypatch.setattr(scanner, "live_contents_under_prefixes", lambda _session, _prefixes: contents)
    progress = _ScanState()
    observations, _survivors = scanner.observe_references_on_filesystem(
        Mock(), ["/private/share"], progress
    )
    return progress, observations


def test_reference_stat_failure_is_reported_once_and_counted_per_row(monkeypatch, caplog):
    with caplog.at_level(logging.INFO):
        progress, observations = observe_references(monkeypatch, failing_stat(errno.ESTALE))

    [event] = events(caplog, "scanner.stat_failed")
    assert event["site"] == "reference"
    assert event["reason"] == "network_unavailable"
    assert event["errno_name"] == errno.errorcode[errno.ESTALE]
    assert event["exc_site"] == "assets.scanner.observe_references_on_filesystem"
    assert buckets(progress) == {("reference", "network_unavailable"): 3}
    # An unreachable file is not a deleted one: nothing is marked missing.
    assert observations == []


@pytest.mark.parametrize("winerror", [53, 67], ids=["bad-netpath", "bad-net-name"])
def test_windows_offline_share_enoent_is_reported_not_taken_for_a_deleted_file(
    winerror: int, monkeypatch, caplog
):
    with caplog.at_level(logging.INFO):
        progress, observations = observe_references(
            monkeypatch, failing_stat(errno.ENOENT, winerror=winerror), rows=2
        )

    assert [e["reason"] for e in events(caplog, "scanner.stat_failed")] == ["network_unavailable"]
    assert buckets(progress) == {("reference", "network_unavailable"): 2}
    assert observations == []


def test_a_plain_deleted_file_is_not_a_reference_failure_and_is_marked_missing(
    monkeypatch, caplog
):
    with caplog.at_level(logging.INFO):
        progress, observations = observe_references(monkeypatch, failing_stat(errno.ENOENT))

    assert events(caplog, "scanner.stat_failed") == []
    assert buckets(progress) == {}
    assert [observation.stat_result for observation in observations] == [None, None, None]


def test_a_permission_denied_reference_is_reported_and_still_skipped(monkeypatch, caplog):
    with caplog.at_level(logging.INFO):
        progress, observations = observe_references(monkeypatch, failing_stat(errno.EACCES))

    assert [e["reason"] for e in events(caplog, "scanner.stat_failed")] == ["permission_denied"]
    assert progress.permission_denied == 3
    assert observations == []


# --- seed_observation ----------------------------------------------------------------


def test_seed_observation_failure_is_reported(temp_dir: Path, monkeypatch, caplog):
    path = temp_dir / "unreadable.bin"
    path.write_bytes(b"data")

    def unreadable(_path):
        raise OSError(errno.EIO, str(path))

    monkeypatch.setattr(scanner, "snapshot_hash", unreadable)
    progress = _ScanState()
    with (
        patch("app.assets.scanner.mode.hashing_enabled", return_value=True),
        caplog.at_level(logging.INFO),
    ):
        observed = scanner.observe_asset_specs([_spec(path)], progress)

    assert observed == {str(path): None}
    [event] = events(caplog, "scanner.stat_failed")
    assert (event["site"], event["reason"]) == ("seed_observation", "io_error")
    assert buckets(progress) == {("seed_observation", "io_error"): 1}
    assert str(path) not in "\n".join(r.getMessage() for r in caplog.records if TAG in r.getMessage())


def test_seed_counts_every_failed_spec_not_just_the_first(
    session: Session, temp_dir: Path, monkeypatch
):
    paths = [temp_dir / f"{name}.bin" for name in ("ok", "bad-1", "bad-2")]
    for path in paths:
        path.write_bytes(path.name.encode())

    def create_record_or_raise(session_arg, *, name, **kwargs) -> Asset:
        if name.startswith("bad"):
            raise RuntimeError(name)
        return create_record(session_arg, name=name, **kwargs)

    monkeypatch.setattr(scanner, "create_record", create_record_or_raise)
    progress = _ScanState()

    created, first_error = scanner.seed_asset_specs(
        session, [_spec(path) for path in paths], progress=progress
    )

    assert created == 1
    assert isinstance(first_error, RuntimeError)
    assert buckets(progress) == {("batch_insert", "other"): 2}


def test_watch_list_failures_are_counted_under_their_watch_site(temp_dir: Path):
    gone = temp_dir / "gone.bin.part"
    _WATCH_LIST[:] = [_WatchEntry(str(gone), os.stat_result((0,) * 10))]
    progress = _ScanState()

    with patch("app.assets.scanner.insert_asset_specs", return_value=(0, None)):
        tick_watch_list(progress=progress)

    assert buckets(progress) == {("watch_stat", "vanished"): 1}
    _WATCH_LIST.clear()


# --- the walker ----------------------------------------------------------------------


def test_an_unreachable_root_is_reported_instead_of_listing_nothing_silently(temp_dir: Path):
    heard: list[tuple[str, type[OSError]]] = []

    listed = list_files_recursively(
        str(temp_dir / "unmounted"), lambda site, exc: heard.append((site, type(exc)))
    )

    assert listed == []
    assert heard == [("walk_root", FileNotFoundError)]


def test_a_root_that_is_a_file_lists_nothing_without_an_error(temp_dir: Path):
    not_a_dir = temp_dir / "file.bin"
    not_a_dir.write_bytes(b"x")
    heard: list[str] = []

    assert list_files_recursively(str(not_a_dir), lambda site, _exc: heard.append(site)) == []
    assert heard == []


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions as non-root"
)
def test_an_unreadable_subdirectory_is_reported_and_the_rest_still_listed(temp_dir: Path):
    readable = temp_dir / "ok.bin"
    readable.write_bytes(b"x")
    locked = temp_dir / "locked"
    locked.mkdir()
    (locked / "hidden.bin").write_bytes(b"x")
    locked.chmod(0)
    heard: list[tuple[str, type[OSError]]] = []
    try:
        listed = list_files_recursively(
            str(temp_dir), lambda site, exc: heard.append((site, type(exc)))
        )
    finally:
        locked.chmod(0o755)

    assert listed == [str(readable)]
    assert heard == [("walk_dir", PermissionError)]


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions as non-root"
)
def test_a_root_that_exists_but_cannot_be_listed_is_reported_as_the_root(temp_dir: Path):
    root = temp_dir / "locked-root"
    root.mkdir()
    root.chmod(0)
    heard: list[str] = []
    try:
        assert list_files_recursively(str(root), lambda site, _exc: heard.append(site)) == []
    finally:
        root.chmod(0o755)

    assert heard == ["walk_root"]


def test_collecting_an_unreachable_input_root_emits_root_unreachable(
    temp_dir: Path, monkeypatch, caplog
):
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(temp_dir / "gone"))
    progress = _ScanState()

    with caplog.at_level(logging.INFO):
        assert scanner.collect_paths_for_roots(("input",), progress=progress) == []

    [event] = events(caplog, "scanner.root_unreachable")
    assert (event["root"], event["reason"]) == ("input", "vanished")
    assert buckets(progress) == {("walk_root", "vanished"): 1}


# --- metadata ------------------------------------------------------------------------


def write_corrupt_safetensors(path: Path) -> None:
    header = b"{not json"
    path.write_bytes(struct.pack("<Q", len(header)) + header)


def test_a_corrupt_safetensors_header_is_reported_rather_than_read_as_no_header(temp_dir: Path):
    path = temp_dir / "model.safetensors"
    write_corrupt_safetensors(path)
    heard: list[BaseException] = []

    extract_file_metadata(str(path), on_error=heard.append)

    assert [type(exc).__name__ for exc in heard] == ["JSONDecodeError"]


def test_an_unreadable_image_is_reported(temp_dir: Path):
    path = temp_dir / "picture.png"
    path.write_bytes(b"not a png")
    heard: list[BaseException] = []

    assert extract_image_dimensions(str(path), mime_type="image/png", on_error=heard.append) is None
    assert [type(exc).__name__ for exc in heard] == ["UnidentifiedImageError"]


def test_enrichment_emits_metadata_failed_for_a_corrupt_header(temp_dir: Path, caplog):
    path = temp_dir / "model.safetensors"
    write_corrupt_safetensors(path)
    content = SimpleNamespace(hash=None, mtime_ns=path.stat().st_mtime_ns)
    record = SimpleNamespace(system_metadata=None, mime_type=None)
    session = Mock()
    session.get.side_effect = lambda _model, row_id: content if row_id == "content" else record
    progress = _ScanState()

    with caplog.at_level(logging.INFO):
        scanner.enrich_asset(
            session, str(path), "content", "record", extract_metadata=True, progress=progress
        )

    [event] = events(caplog, "scanner.metadata_failed")
    assert event["reason"] == "corrupt"
    assert event["exc_site"] == "assets.services.metadata_extract._read_safetensors_header"
    assert buckets(progress) == {("metadata", "corrupt"): 1}


# --- buckets -------------------------------------------------------------------------


def test_mixed_causes_survive_the_emit_once_event_as_buckets(monkeypatch, caplog):
    stat = failing_stat(errno.EACCES, errno.ESTALE, errno.ESTALE, errno.ESTALE)
    monkeypatch.setattr(scanner, "os", fake_os(stat))
    progress = _ScanState()
    paths = [f"/private/share/{i}.bin" for i in range(8)]

    with caplog.at_level(logging.INFO):
        scanner.build_asset_specs(paths, set(), progress=progress)
        _emit_failure_buckets(progress, "fast", None)

    assert [e["reason"] for e in events(caplog, "scanner.stat_failed")] == ["permission_denied"]
    lines = events(caplog, "scanner.failure_bucket")
    assert [(e["site"], e["reason"], e["count"]) for e in lines] == [
        ("discovery", "network_unavailable", 6),
        ("discovery", "permission_denied", 2),
    ]
    assert progress.failure_buckets == Counter()


def test_bucket_lines_are_capped_most_frequent_first(monkeypatch, caplog):
    monkeypatch.setattr(seeder_module, "_MAX_FAILURE_BUCKETS", 2)
    progress = _ScanState()
    for site, count in (("hash", 1), ("enrich", 3), ("metadata", 2)):
        for _ in range(count):
            progress.record_failure(site, OSError(errno.EIO, "x"))

    with caplog.at_level(logging.INFO):
        _emit_failure_buckets(progress, "fast", None)

    assert [(e["site"], e["count"]) for e in events(caplog, "scanner.failure_bucket")] == [
        ("enrich", 3),
        ("metadata", 2),
    ]
