import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from aiohttp import web

import folder_paths
from app.assets.downloads import destination
from app.assets.downloads.backend import (
    CancelOutcome,
    DownloadPhase,
    DownloadRejected,
    DownloadSnapshot,
)
from app.assets.downloads.comfy_cli import _parse_envelope, _snapshot
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
    backend = FakeBackend([snapshot(phase=DownloadPhase.TRANSFERRED, destination_path=str(path), completed=7, total=7)])
    service, _ = make_service(backend, refresh=lambda: refreshed.append(True))

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
    assert "not visible" in task["error_message"]
    assert "result" not in task


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


async def test_progress_is_published_in_the_shape_the_toast_consumes():
    backend = FakeBackend([snapshot(completed=3, total=12, destination_path="/models/loras/pretty.safetensors")])
    service, events = make_service(backend)

    await service.sweep()

    event, payload = events[-1]
    assert event == "asset_download"
    assert payload["asset_name"] == "pretty.safetensors"
    assert payload["progress"] == 25.0
    assert payload["status"] == "running"


async def test_progress_stays_zero_while_the_total_is_unknown():
    backend = FakeBackend([snapshot(completed=0, total=None)])
    service, events = make_service(backend)

    await service.sweep()

    assert events[-1][1]["bytes_total"] == 0
    assert events[-1][1]["progress"] == 0.0


async def test_downloads_already_finished_at_startup_are_adopted_without_replaying_them(model_root):
    path = model_root / "old.safetensors"
    path.write_bytes(b"old")
    backend = FakeBackend(
        [
            snapshot(handle="done", phase=DownloadPhase.CANCELLED),
            snapshot(handle="old", phase=DownloadPhase.TRANSFERRED, destination_path=str(path)),
        ]
    )
    service, events = make_service(backend)

    await service.sweep()

    assert events == []
    assert len(service._tracked) == 2


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
    from aiohttp.test_utils import TestClient, TestServer

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
