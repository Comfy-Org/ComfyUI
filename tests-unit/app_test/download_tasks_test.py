import asyncio
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import folder_paths
from app.assets.downloads import destination
from app.assets.downloads.backend import (
    CancelOutcome,
    DownloadBackendUnavailable,
    DownloadPhase,
    DownloadRejected,
    DownloadSnapshot,
)
from app.assets.downloads.comfy_cli import ComfyCliBackend, _parse_envelope, _snapshot
from app.assets.downloads.routes import register_download_routes
from app.assets.downloads.service import DownloadTaskService

pytestmark = pytest.mark.asyncio

STARTED = datetime(2026, 10, 4, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def model_root(tmp_path):
    """A models tree with one lora folder, patched in for the whole test."""
    loras = tmp_path / "models" / "loras"
    loras.mkdir(parents=True)
    folders = {
        "loras": ([str(loras)], {".safetensors", ".ckpt"}),
        "configs": ([str(tmp_path / "models" / "configs")], {".yaml"}),
        "custom_nodes": ([str(tmp_path / "custom_nodes")], set()),
        "datasets": ([str(tmp_path / "datasets")], set()),
    }
    with patch.dict(folder_paths.folder_names_and_paths, folders, clear=True):
        folder_paths.invalidate_filename_list_cache("loras")
        yield loras
    folder_paths.invalidate_filename_list_cache("loras")


def snapshot(handle="abc123", phase=DownloadPhase.TRANSFERRING, destination_path="/models/loras/a.safetensors",
             completed=5, total=10, started=STARTED, error=None, cancellable=True):
    return DownloadSnapshot(
        handle=handle,
        phase=phase,
        destination=destination_path,
        bytes_completed=completed,
        bytes_total=total,
        started_at=started,
        updated_at=started + timedelta(seconds=3),
        error=error,
        cancellable=cancellable,
    )


class FakeBackend:
    key = "fake"

    def __init__(self, snapshots=()):
        self.snapshots = list(snapshots)
        self.requests = []
        self.cancelled = []
        self.cancel_outcome = CancelOutcome.CANCELLING
        self.start_error = None

    async def available(self):
        return True

    def change_hint(self):
        return None

    async def start(self, request):
        if self.start_error:
            raise self.start_error
        self.requests.append(request)
        made = snapshot(
            handle="new1",
            phase=DownloadPhase.PENDING,
            destination_path=os.path.join(request.directory, request.filename or "resolved.safetensors"),
            completed=0,
            total=None,
        )
        self.snapshots.append(made)
        return made

    async def list(self):
        return list(self.snapshots)

    async def get(self, handle):
        return next((s for s in self.snapshots if s.handle == handle), None)

    async def cancel(self, handle):
        self.cancelled.append(handle)
        return self.cancel_outcome


def make_service(backend, refresh=None):
    events = []
    service = DownloadTaskService(backend, lambda event, data: events.append((event, data)), refresh)
    return service, events


async def test_resolve_places_the_file_where_loaders_look(model_root):
    assert destination.resolve("loras", "a.safetensors") == str(model_root / "a.safetensors")
    assert destination.resolve("loras", "sub/a.safetensors") == str(model_root / "sub" / "a.safetensors")


@pytest.mark.parametrize(
    "folder,filename,code",
    [
        ("loras", "notes.txt", "UNSUPPORTED_EXTENSION"),
        ("loras", "../escape.safetensors", "INVALID_FILENAME"),
        ("loras", "../../etc/escape.safetensors", "INVALID_FILENAME"),
        ("loras", "/abs/escape.safetensors", "INVALID_FILENAME"),
        ("loras", "   ", "INVALID_FILENAME"),
        ("nope", "a.safetensors", "UNKNOWN_MODEL_FOLDER"),
    ],
)
async def test_resolve_refuses_destinations_no_loader_would_list(model_root, folder, filename, code):
    with pytest.raises(destination.DestinationError) as excinfo:
        destination.resolve(folder, filename)
    assert excinfo.value.code == code


@pytest.mark.parametrize("folder,filename", [
    ("custom_nodes", "pwned.py"),
    ("custom_nodes", "evil/__init__.py"),
    ("configs", "anything.yaml"),
])
async def test_resolve_refuses_to_write_outside_the_model_folders(model_root, folder, filename):
    """custom_nodes is imported at startup, so a download into it is remote code
    execution. It shares the upload endpoint's allowlist for exactly that reason."""
    with pytest.raises(destination.DestinationError) as excinfo:
        destination.resolve(folder, filename)
    assert excinfo.value.code == "UNKNOWN_MODEL_FOLDER"


async def test_a_folder_that_lists_every_extension_accepts_any_name(model_root, tmp_path):
    """filter_files_extensions treats an empty set as match-all, which is how
    folders registered by extra_model_paths.yaml and custom nodes arrive."""
    assert destination.resolve("datasets", "anything.bin") == str(tmp_path / "datasets" / "anything.bin")


async def test_an_alias_pointing_at_custom_nodes_is_refused_by_path(tmp_path):
    """extra_model_paths.yaml can register a second name for a directory that
    already has one, and a name-keyed exclusion would wave the alias through
    into code ComfyUI imports at startup."""
    nodes_dir = tmp_path / "custom_nodes"
    nodes_dir.mkdir()
    folders = {
        "custom_nodes": ([str(nodes_dir)], set()),
        "node_packs": ([str(nodes_dir)], set()),
    }
    with patch.dict(folder_paths.folder_names_and_paths, folders, clear=True):
        with pytest.raises(destination.DestinationError) as excinfo:
            destination.resolve("node_packs", "evil.py")

    assert excinfo.value.code == "UNKNOWN_MODEL_FOLDER"


async def test_a_download_into_a_match_all_folder_is_reported_usable(tmp_path):
    """The mirror of the resolve case: inspect must not call a file invisible
    just because its folder declares no extensions."""
    root = tmp_path / "datasets"
    root.mkdir()
    landed = root / "corpus.bin"
    landed.write_bytes(b"data")
    with patch.dict(folder_paths.folder_names_and_paths, {"datasets": ([str(root)], set())}, clear=True):
        folder_paths.invalidate_filename_list_cache("datasets")
        backend = FakeBackend([snapshot(phase=DownloadPhase.TRANSFERRED, destination_path=str(landed))])
        service, _ = make_service(backend)

        await service.sweep()
        task = await service.get_task(_only_task_id(service))

    assert task["status"] == "completed"


async def test_extra_descriptive_tags_do_not_hide_the_model_folder(model_root):
    app = web.Application()
    backend = FakeBackend()
    service, _ = make_service(backend)
    register_download_routes(app, service)

    status, _body = await _request(
        app,
        "POST",
        "/api/assets/download",
        json={"source_url": "https://e.test/a.safetensors", "tags": ["models", "sdxl", "loras"]},
    )

    assert status == 202
    assert backend.requests[0].directory == str(model_root)


async def test_an_unreadable_journal_is_not_mistaken_for_an_empty_one(tmp_path, monkeypatch):
    """Returning [] would retire every tracked task and let the route answer an
    authoritative 404, which the frontend reads as proof the download is gone."""
    backend = ComfyCliBackend(str(tmp_path))

    def denied(_path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("app.assets.downloads.comfy_cli.os.scandir", denied)

    with pytest.raises(PermissionError):
        backend._read_all_journals()


async def test_a_download_into_custom_nodes_is_refused_over_http(model_root):
    app = web.Application()
    backend = FakeBackend()
    service, _ = make_service(backend)
    register_download_routes(app, service)

    status, body = await _request(
        app,
        "POST",
        "/api/assets/download",
        json={"source_url": "https://e.test/pwned.py", "tags": ["models", "model_type:custom_nodes"]},
    )

    assert status == 400
    assert body["error"]["code"] == "UNKNOWN_MODEL_FOLDER"
    assert backend.requests == []


async def test_folder_for_path_ignores_paths_outside_the_model_folders(model_root, tmp_path):
    assert destination.folder_for_path(str(tmp_path / "custom_nodes" / "x.py")) is None


async def test_folder_for_path_identifies_externally_chosen_destinations(model_root):
    assert destination.folder_for_path(str(model_root / "x.safetensors")) == "loras"
    assert destination.folder_for_path("/somewhere/else/x.safetensors") is None


@pytest.mark.parametrize(
    "phase,expected",
    [
        (DownloadPhase.PENDING, "created"),
        (DownloadPhase.TRANSFERRING, "running"),
        (DownloadPhase.FAILED, "failed"),
        (DownloadPhase.CANCELLED, "cancelled"),
    ],
)
async def test_phase_projects_onto_the_task_status_the_browser_knows(phase, expected):
    backend = FakeBackend([snapshot(phase=phase)])
    service, _ = make_service(backend)

    await service.sweep()
    task = await service.get_task(_only_task_id(service))

    assert task["status"] == expected


async def test_transferred_download_completes_once_the_loader_lists_it(model_root):
    path = model_root / "landed.safetensors"
    path.write_bytes(b"weights")
    refreshed = []
    backend = FakeBackend()
    service, _ = make_service(backend, refresh=lambda: refreshed.append(True))
    await service.sweep()

    backend.snapshots.append(
        snapshot(phase=DownloadPhase.TRANSFERRED, destination_path=str(path), completed=7, total=7)
    )
    await service.sweep()
    task = await service.get_task(_only_task_id(service))

    assert task["status"] == "completed"
    assert task["result"] == {
        "success": True,
        "file_path": str(path),
        "filename": "landed.safetensors",
        "bytes_downloaded": 7,
    }
    assert refreshed == [True]


async def test_transferred_download_fails_when_the_file_landed_where_no_loader_looks(tmp_path, model_root):
    """BE-10028: bytes arriving is not the same as the model being usable."""
    stray = tmp_path / "outside" / "landed.safetensors"
    stray.parent.mkdir()
    stray.write_bytes(b"weights")
    backend = FakeBackend([snapshot(phase=DownloadPhase.TRANSFERRED, destination_path=str(stray))])
    service, _ = make_service(backend)

    await service.sweep()
    task = await service.get_task(_only_task_id(service))

    assert task["status"] == "failed"
    assert "outside every configured model directory" in task["error_message"]
    assert "result" not in task


async def test_a_file_in_the_right_folder_but_the_wrong_format_says_so(model_root):
    """Reachable through the externally-started path, where comfy-cli chose the
    name. Blaming placement would send the user looking in the wrong place."""
    odd = model_root / "weights.gguf"
    odd.write_bytes(b"weights")
    backend = FakeBackend([snapshot(phase=DownloadPhase.TRANSFERRED, destination_path=str(odd))])
    service, _ = make_service(backend)

    await service.sweep()
    task = await service.get_task(_only_task_id(service))

    assert task["status"] == "failed"
    assert "'loras' loaders only list" in task["error_message"]


async def test_a_download_shadowed_by_the_preferred_root_is_not_reported_complete(tmp_path):
    """get_filename_list unions every root, so a name already present in the
    preferred root makes a download into a secondary one look present while
    loaders keep opening the other file."""
    preferred = tmp_path / "models" / "loras"
    secondary = tmp_path / "extra" / "loras"
    preferred.mkdir(parents=True)
    secondary.mkdir(parents=True)
    (preferred / "same.safetensors").write_bytes(b"the file loaders actually open")
    landed = secondary / "same.safetensors"
    landed.write_bytes(b"what was just downloaded")

    folders = {"loras": ([str(preferred), str(secondary)], {".safetensors"})}
    with patch.dict(folder_paths.folder_names_and_paths, folders, clear=True):
        folder_paths.invalidate_filename_list_cache("loras")
        backend = FakeBackend([snapshot(phase=DownloadPhase.TRANSFERRED, destination_path=str(landed))])
        service, _ = make_service(backend)

        await service.sweep()
        task = await service.get_task(_only_task_id(service))

    assert task["status"] == "failed"
    assert "would never read it" in task["error_message"]


async def test_transferred_download_fails_when_the_file_is_gone(model_root):
    missing = str(model_root / "vanished.safetensors")
    backend = FakeBackend([snapshot(phase=DownloadPhase.TRANSFERRED, destination_path=missing)])
    service, _ = make_service(backend)

    await service.sweep()
    task = await service.get_task(_only_task_id(service))

    assert task["status"] == "failed"


async def test_task_ids_are_rebuilt_identically_by_a_restarted_server():
    records = [snapshot()]
    first, _ = make_service(FakeBackend(records))
    await first.sweep()

    restarted, _ = make_service(FakeBackend(records))
    await restarted.sweep()

    assert _only_task_id(first) == _only_task_id(restarted)


async def test_a_reused_handle_from_a_later_download_is_a_different_task():
    original, _ = make_service(FakeBackend([snapshot(handle="dup")]))
    await original.sweep()
    reused, _ = make_service(FakeBackend([snapshot(handle="dup", started=STARTED + timedelta(hours=1))]))
    await reused.sweep()

    assert _only_task_id(original) != _only_task_id(reused)


async def test_a_pruned_download_stops_resolving_so_the_browser_can_settle_it():
    backend = FakeBackend([snapshot()])
    service, _ = make_service(backend)
    await service.sweep()
    task_id = _only_task_id(service)

    backend.snapshots.clear()
    await service.sweep()

    assert await service.get_task(task_id) is None


async def test_polling_a_task_that_will_never_exist_does_not_re_enumerate_each_time():
    backend = FakeBackend([snapshot()])
    service, _ = make_service(backend)
    await service.sweep()
    enumerations = 0

    original = backend.list

    async def counting_list():
        nonlocal enumerations
        enumerations += 1
        return await original()

    backend.list = counting_list
    for _ in range(5):
        assert await service.get_task("11111111-2222-3333-4444-555555555555") is None

    assert enumerations == 0


async def test_concurrent_lookups_share_one_enumeration():
    """Resolving an unknown id enumerates, and enumerating spawns a subprocess,
    so a burst of requests must not become a burst of subprocesses."""
    backend = FakeBackend([snapshot()])
    service, _ = make_service(backend)
    enumerations = 0
    original = backend.list

    async def slow_list():
        nonlocal enumerations
        enumerations += 1
        await asyncio.sleep(0.05)
        return await original()

    backend.list = slow_list
    await asyncio.gather(*(service.get_task("11111111-2222-3333-4444-555555555555") for _ in range(20)))

    assert enumerations == 1


async def test_a_lookup_during_a_transport_outage_does_not_claim_the_task_is_gone():
    """404 is authoritative to the frontend: it settles a pending cancellation
    on it. Only a transport that answered may say a task does not exist."""
    backend = FakeBackend()

    async def unavailable():
        raise DownloadBackendUnavailable("comfy-cli is not installed")

    backend.list = unavailable
    service, _ = make_service(backend)

    with pytest.raises(DownloadBackendUnavailable):
        await service.get_task("11111111-2222-3333-4444-555555555555")


async def test_progress_is_published_in_the_shape_the_toast_consumes():
    """Cloud's AssetDownloadMessage documents progress as a 0.0-1.0 fraction and
    ProgressToastItem renders `progress * 100`, so a percentage here shows 2500%."""
    backend = FakeBackend([snapshot(completed=3, total=12, destination_path="/models/loras/pretty.safetensors")])
    service, events = make_service(backend)

    await service.sweep()

    event, payload = events[-1]
    assert event == "asset_download"
    assert payload["asset_name"] == "pretty.safetensors"
    assert payload["progress"] == 0.25
    assert payload["status"] == "running"


async def test_progress_stays_zero_while_the_total_is_unknown():
    backend = FakeBackend([snapshot(completed=0, total=None)])
    service, events = make_service(backend)

    await service.sweep()

    assert events[-1][1]["bytes_total"] == 0
    assert events[-1][1]["progress"] == 0.0


async def test_a_completed_download_reports_a_full_fraction(model_root):
    path = model_root / "whole.safetensors"
    path.write_bytes(b"weights")
    backend = FakeBackend()
    service, events = make_service(backend)
    await service.sweep()

    backend.snapshots.append(snapshot(phase=DownloadPhase.TRANSFERRED, destination_path=str(path)))
    await service.sweep()

    assert events[-1][1]["progress"] == 1.0
    assert events[-1][1]["status"] == "completed"


async def test_downloads_already_finished_at_startup_are_adopted_without_replaying_them(model_root):
    path = model_root / "old.safetensors"
    path.write_bytes(b"old")
    refreshed = []
    backend = FakeBackend(
        [
            snapshot(handle="done", phase=DownloadPhase.CANCELLED),
            snapshot(handle="old", phase=DownloadPhase.TRANSFERRED, destination_path=str(path)),
        ]
    )
    service, events = make_service(backend, refresh=lambda: refreshed.append(True))

    await service.sweep()

    assert events == []
    assert refreshed == []
    assert len(service._tracked) == 2
    assert (await service.get_task(service._tracked["old"].task_id))["status"] == "completed"


async def test_a_download_discovered_after_startup_is_announced_even_if_already_finished(model_root):
    """A worker that dies seconds after the agent launches it writes a terminal
    record inside one sweep gap; staying quiet about it is the PM-1883 bug."""
    backend = FakeBackend()
    service, events = make_service(backend)
    await service.sweep()

    backend.snapshots.append(snapshot(handle="quick", phase=DownloadPhase.FAILED, error="boom"))
    await service.sweep()

    assert [e[1]["status"] for e in events] == ["failed"]


async def test_submitting_lets_the_transport_name_the_file_when_the_caller_cannot(model_root):
    backend = FakeBackend()
    service, _ = make_service(backend)

    await service.start("https://example.test/download/12345", "loras", None)

    assert backend.requests[0].directory == str(model_root)
    assert backend.requests[0].filename is None


async def test_submitting_reports_the_task_immediately(model_root):
    backend = FakeBackend()
    service, events = make_service(backend)

    task = await service.start("https://example.test/a.safetensors", "loras", "a.safetensors")

    assert task["status"] == "created"
    assert events[-1][1]["task_id"] == task["task_id"]


async def test_stopping_the_sweeper_leaves_no_task_running():
    backend = FakeBackend([snapshot()])
    service, _ = make_service(backend)
    service.start_sweeping()
    await asyncio.sleep(0)

    await service.stop_sweeping()

    assert service._sweeper is None and service._sweeping is None


async def test_cancelling_an_unknown_task_reports_it_missing():
    service, _ = make_service(FakeBackend())

    assert await service.cancel_task("11111111-2222-3333-4444-555555555555") is CancelOutcome.MISSING


async def test_cancelling_forwards_the_transports_refusal():
    backend = FakeBackend([snapshot()])
    backend.cancel_outcome = CancelOutcome.NOT_CANCELLABLE
    service, _ = make_service(backend)
    await service.sweep()

    outcome = await service.cancel_task(_only_task_id(service))

    assert outcome is CancelOutcome.NOT_CANCELLABLE


async def test_a_journal_id_cannot_steer_the_read_outside_the_journal_directory(tmp_path):
    """A handle comes from a journal record's own id field, so it is only as
    trustworthy as that file."""
    (tmp_path / "secret.json").write_text('{"id": "secret"}')
    journal = tmp_path / ".comfy-downloads"
    journal.mkdir()
    backend = ComfyCliBackend(str(tmp_path))

    assert backend._read_journal("../secret") is None
    assert backend._read_journal("/etc/passwd") is None


async def test_an_absent_journal_is_a_definite_answer_not_a_failed_probe(tmp_path):
    """None means "cannot tell" and forces an enumeration every tick. comfy-cli
    only creates the directory on its first write, so a fresh install would
    spawn a subprocess every second forever."""
    backend = ComfyCliBackend(str(tmp_path / "never-used"))

    assert backend.change_hint() == (0, 0)


async def test_envelope_is_read_off_the_last_json_line():
    stdout = 'noise\n{"type":"event","x":1}\n{"schema":"envelope/1","type":"envelope","ok":true,"data":{"id":"z"}}\n'
    assert _parse_envelope(stdout)["data"] == {"id": "z"}
    assert _parse_envelope("not json at all") is None


async def test_a_foreground_record_is_reported_as_not_cancellable():
    record = {
        "id": "f1",
        "status": "downloading",
        "kind": "foreground",
        "dest": "/models/loras/a.safetensors",
        "completed_bytes": 1,
        "total_bytes": 2,
        "started_at": "2026-10-04T12:00:00+00:00",
        "updated_at": "2026-10-04T12:00:03+00:00",
    }
    assert _snapshot(record).cancellable is False
    assert _snapshot({**record, "kind": "background"}).cancellable is True


async def test_an_unrecognised_transport_status_is_not_treated_as_progress():
    built = _snapshot({"id": "f1", "status": "who-knows", "dest": "/x", "completed_bytes": 0, "total_bytes": None})
    assert built.phase is DownloadPhase.FAILED


async def _request(app, method, path, **kwargs):
    async with TestClient(TestServer(app)) as client:
        response = await client.request(method, path, **kwargs)
        return response.status, await response.json()


async def test_submit_rejects_a_body_with_no_model_folder(model_root):
    app = web.Application()
    service, _ = make_service(FakeBackend())
    register_download_routes(app, service)

    status, body = await _request(app, "POST", "/api/assets/download", json={"source_url": "https://e.test/a.safetensors", "tags": ["models"]})

    assert status == 400
    assert body["error"]["code"] == "INVALID_BODY"


async def test_submit_rejects_a_non_http_source(model_root):
    app = web.Application()
    service, _ = make_service(FakeBackend())
    register_download_routes(app, service)

    status, body = await _request(
        app, "POST", "/api/assets/download", json={"source_url": "file:///etc/passwd", "tags": ["models", "model_type:loras"]}
    )

    assert status == 400


async def test_submit_surfaces_a_transport_conflict_as_409(model_root):
    app = web.Application()
    backend = FakeBackend()
    backend.start_error = DownloadRejected("model_file_exists", "File already exists")
    service, _ = make_service(backend)
    register_download_routes(app, service)

    status, body = await _request(
        app,
        "POST",
        "/api/assets/download",
        json={"source_url": "https://e.test/a.safetensors", "tags": ["models", "model_type:loras"]},
    )

    assert status == 409
    assert body["error"]["code"] == "model_file_exists"


async def test_reading_a_task_reports_a_missing_transport_instead_of_crashing(model_root):
    """taskService treats anything but 404 as transient and retries forever, so a
    500 here pins a pending cancellation open until the page is reloaded."""
    app = web.Application()
    backend = FakeBackend()

    async def unavailable():
        raise DownloadBackendUnavailable("comfy-cli is not installed")

    backend.list = unavailable
    service, _ = make_service(backend)
    register_download_routes(app, service)

    status, body = await _request(app, "GET", "/api/tasks/11111111-2222-3333-4444-555555555555")

    assert status == 503
    assert body["error"]["code"] == "DEPENDENCY_MISSING"


async def test_reading_an_unknown_task_is_a_404(model_root):
    app = web.Application()
    service, _ = make_service(FakeBackend())
    register_download_routes(app, service)

    status, _body = await _request(app, "GET", "/api/tasks/11111111-2222-3333-4444-555555555555")

    assert status == 404


async def test_cancelling_a_finished_task_is_a_409(model_root):
    app = web.Application()
    backend = FakeBackend([snapshot(phase=DownloadPhase.FAILED)])
    backend.cancel_outcome = CancelOutcome.NOT_CANCELLABLE
    service, _ = make_service(backend)
    register_download_routes(app, service)
    await service.sweep()

    status, body = await _request(app, "DELETE", f"/api/tasks/{_only_task_id(service)}")

    assert status == 409
    assert body["error"]["code"] == "TASK_NOT_CANCELLABLE"


def _only_task_id(service):
    return next(iter(service._by_task))
