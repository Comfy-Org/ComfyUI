import asyncio
import os
import re
import threading
import time
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlsplit

import pytest
from aiohttp import web

import folder_paths
from app import asset_export
from app.asset_export import (
    AssetExportManager,
    ExportEntry,
    ExportRequestError,
    assign_archive_names,
    export_location,
    matches_name_filter,
    parse_export_request,
    write_archive,
)
from app.assets.api import routes
from app.assets.services import path_utils
from app.assets.services.schemas import ExportableAssetFile

JOB_A = "11111111-1111-4111-8111-111111111111"
JOB_B = "22222222-2222-4222-8222-222222222222"
JOB_C = "44444444-4444-4444-8444-444444444444"
ASSET_ID = "33333333-3333-4333-8333-333333333333"
CREATED = datetime(2026, 9, 21, 14, 13, 20, tzinfo=timezone.utc)
QUEUED = CREATED - timedelta(minutes=3)
EXPORT_NAME_PATTERN = re.compile(r"^comfy-export-\d{4}-\d{2}-\d{2}-[0-9a-f]{8}\.zip$")


@pytest.fixture(autouse=True)
def assets_enabled(monkeypatch):
    monkeypatch.setattr(routes, "_ASSETS_ENABLED", True)


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    roots = {category: tmp_path / category for category in ("input", "output", "temp")}
    for category, root in roots.items():
        root.mkdir()
        monkeypatch.setattr(folder_paths, f"get_{category}_directory", lambda root=root: str(root))
    roots["models"] = tmp_path / "models" / "checkpoints"
    roots["models"].mkdir(parents=True)
    monkeypatch.setattr(
        path_utils,
        "get_comfy_models_folders",
        lambda: [("checkpoints", [str(roots["models"])], set())],
    )
    return roots


def write_file(directory, relative, data):
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def asset_file(path, job_id=JOB_A, asset_id=None, file_hash=None, created_at=CREATED, name=None):
    return ExportableAssetFile(
        id=asset_id or str(uuid.uuid4()),
        name=name or os.path.basename(path),
        path=str(path),
        hash=file_hash,
        job_id=job_id,
        created_at=created_at,
    )


@pytest.mark.parametrize(
    "body, code",
    [
        ([], "INVALID_REQUEST"),
        ({}, "INVALID_REQUEST"),
        ({"job_ids": []}, "INVALID_REQUEST"),
        ({"job_ids": ["not-a-uuid"]}, "INVALID_REQUEST"),
        ({"job_ids": ["AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"]}, "INVALID_REQUEST"),
        ({"asset_ids": ["nope"]}, "INVALID_REQUEST"),
        ({"job_ids": "abc"}, "INVALID_REQUEST"),
        ({"job_ids": [JOB_A], "naming_strategy": "flat"}, "INVALID_NAMING_STRATEGY"),
        ({"job_ids": [JOB_A], "job_asset_name_filters": []}, "INVALID_REQUEST"),
        ({"job_ids": [JOB_A], "job_asset_name_filters": {JOB_A: []}}, "INVALID_REQUEST"),
        ({"job_ids": [JOB_A], "job_asset_name_filters": {JOB_A: [1]}}, "INVALID_REQUEST"),
        ({"job_ids": [JOB_A], "include_previews": "yes"}, "INVALID_REQUEST"),
    ],
)
def test_parse_export_request_rejects_invalid_bodies(body, code):
    with pytest.raises(ExportRequestError) as error:
        parse_export_request(body)
    assert error.value.code == code


def test_parse_export_request_applies_contract_defaults_and_dedupes():
    request = parse_export_request({"job_ids": [JOB_A, JOB_A], "asset_ids": [ASSET_ID.upper()]})
    assert request.job_ids == [JOB_A]
    assert request.asset_ids == [ASSET_ID]
    assert request.naming_strategy == "group_by_job_time"
    assert request.job_asset_name_filters == {}
    assert request.include_previews is False


@pytest.mark.parametrize(
    "other, owner, same",
    [
        ({"job_ids": [JOB_B, JOB_A], "job_asset_name_filters": {JOB_A: ["b", "a"]}}, "alice", True),
        ({"job_ids": [JOB_A, JOB_B], "job_asset_name_filters": {JOB_A: ["a", "b"], JOB_C: ["x"]}}, "alice", True),
        ({"job_ids": [JOB_A, JOB_B], "job_asset_name_filters": {JOB_A: ["a", "b"]}}, "bob", False),
        ({"job_ids": [JOB_A, JOB_B]}, "alice", False),
        ({"job_ids": [JOB_A, JOB_B], "job_asset_name_filters": {JOB_A: ["a", "b"]}, "include_previews": True}, "alice", False),
    ],
    ids=["reordered", "filter of a job not requested", "other owner", "no filter", "previews"],
)
def test_idempotency_key_identifies_the_same_request_of_the_same_owner(other, owner, same):
    base = parse_export_request({"job_ids": [JOB_A, JOB_B], "job_asset_name_filters": {JOB_A: ["a", "b"]}})

    matches = parse_export_request(other).idempotency_key(owner) == base.idempotency_key("alice")

    assert matches is same


@pytest.mark.parametrize(
    "category, relative, name, expected",
    [
        ("output", "EXR/shot/a.exr", None, ("output", "EXR/shot/a.exr")),
        ("temp", "p.png", None, ("temp", "p.png")),
        ("input", "ref.png", None, ("input", "ref.png")),
        ("input", "0123abcd.png", "cat.png", ("input", "cat.png")),
        ("output", "renders/old.png", "new.png", ("output", "renders/new.png")),
        ("models", "x.safetensors", None, ("models", None)),
        (None, "elsewhere/x.png", None, (None, None)),
    ],
    ids=["output", "temp", "input", "upload stored under its hash", "renamed", "model", "outside every root"],
)
def test_export_location_names_the_file_after_its_asset(dirs, tmp_path, category, relative, name, expected):
    root = dirs[category] if category else tmp_path
    assert export_location(asset_file(root / relative, name=name)) == expected


@pytest.mark.parametrize(
    "names, expected",
    [
        (None, True),
        ({"a.png"}, True),
        ({"sub/a.png"}, True),
        ({"blake3:abc"}, True),
        ({"b.png", "sub/b.png"}, False),
    ],
)
def test_name_filters_match_asset_name_archive_path_or_hash(names, expected):
    file = asset_file("/out/sub/a.png", file_hash="blake3:abc")
    assert matches_name_filter(file, "sub/a.png", names) is expected


def entry(name, asset_id, job_id=JOB_A, is_preview=False, created_at=CREATED):
    return ExportEntry(asset_id, f"/abs/{name}", name, job_id, created_at, is_preview)


@pytest.mark.parametrize(
    "strategy, entries, expected",
    [
        ("preserve", [entry("a.png", "2"), entry("a.png", "1", job_id=JOB_B)], [("1", "a.png")]),
        (
            "group_by_job_id",
            [entry("a.png", "1"), entry("a.png", "2", job_id=JOB_B)],
            [("1", f"{JOB_A}/a.png"), ("2", f"{JOB_B}/a.png")],
        ),
        ("group_by_job_id", [entry("p.png", "1", is_preview=True)], [("1", f"{JOB_A}/previews/p.png")]),
        ("group_by_job_id", [entry("a.png", "1", job_id=None)], [("1", "unknown/a.png")]),
        (
            "group_by_job_time",
            [entry("b.png", "2", created_at=CREATED + timedelta(minutes=5)), entry("a.png", "1")],
            [("1", "2026-09-21T14-13-20/a.png"), ("2", "2026-09-21T14-13-20/b.png")],
        ),
        (
            "group_by_job_time",
            [
                entry("a.png", "1", job_id=JOB_B),
                entry("a.png", "2", job_id=JOB_A),
                entry("a.png", "3", job_id=JOB_C, created_at=CREATED + timedelta(seconds=1)),
            ],
            [
                ("1", "2026-09-21T14-13-21/a.png"),
                ("2", "2026-09-21T14-13-20/a.png"),
                ("3", "2026-09-21T14-13-22/a.png"),
            ],
        ),
        (
            "group_by_job_time",
            [entry("a.png", "1", job_id=None, created_at=CREATED.replace(tzinfo=None))],
            [("1", "2026-09-21T14-13-20/a.png")],
        ),
        ("asset_id", [entry("x/a.png", ASSET_ID)], [(ASSET_ID, f"{ASSET_ID}.png")]),
        ("preserve", [entry("../../evil/./a.png", "1")], [("1", "evil/a.png")]),
    ],
)
def test_assign_archive_names(strategy, entries, expected):
    assigned, collided = assign_archive_names(entries, strategy)
    assert [(e.asset_id, name) for e, name in assigned] == expected
    assert len(collided) == len(entries) - len(expected)


@pytest.mark.parametrize(
    "entries, job_create_times, expected",
    [
        ([entry("a.png", "1")], {JOB_A: QUEUED}, ["2026-09-21T14-10-20/a.png"]),
        (
            [entry("a.png", "1"), entry("b.png", "2", job_id=JOB_B)],
            {JOB_A: QUEUED},
            ["2026-09-21T14-10-20/a.png", "2026-09-21T14-13-20/b.png"],
        ),
        (
            [entry("a.png", "1", created_at=CREATED + timedelta(minutes=1)), entry("a.png", "2", job_id=JOB_B)],
            {JOB_A: CREATED},
            ["2026-09-21T14-13-20/a.png", "2026-09-21T14-13-21/a.png"],
        ),
        ([entry("a.png", "1")], {JOB_B: QUEUED}, ["2026-09-21T14-13-20/a.png"]),
    ],
    ids=[
        "creation time wins over asset time",
        "unknown job falls back to its earliest asset",
        "creation and fallback times on one second get separate folders",
        "creation times of other jobs are ignored",
    ],
)
def test_job_time_folders_prefer_the_job_creation_time(entries, job_create_times, expected):
    assigned, _ = assign_archive_names(entries, "group_by_job_time", job_create_times)
    assert [name for _, name in assigned] == expected


def test_write_archive_stores_files_and_skips_unreadable(tmp_path):
    contents = {
        "large.bin": b"x" * (asset_export.COPY_CHUNK_BYTES + 10),
        "small.bin": b"small",
        "empty.bin": b"",
    }
    assigned = [
        (ExportEntry(str(i), write_file(tmp_path, name, data), name, None, CREATED), f"out/{name}")
        for i, (name, data) in enumerate(contents.items())
    ]
    assigned += [
        (ExportEntry("8", str(tmp_path / "missing.bin"), "missing.bin", None, CREATED), "missing.bin"),
        (ExportEntry("9", str(tmp_path), "dir", None, CREATED), "dir"),
    ]
    byte_counts, outcomes = [], []
    dest = tmp_path / "out.zip"

    written = write_archive(
        assigned, str(dest), byte_counts.append, lambda e, ok: outcomes.append((e.asset_id, ok))
    )

    assert written == 3
    assert outcomes == [("0", True), ("1", True), ("2", True), ("8", False), ("9", False)]
    assert sum(byte_counts) == sum(len(data) for data in contents.values())
    with zipfile.ZipFile(dest) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == [f"out/{name}" for name in contents]
        for name, data in contents.items():
            assert archive.getinfo(f"out/{name}").compress_type == zipfile.ZIP_STORED
            assert archive.read(f"out/{name}") == data


class Harness:
    def __init__(self, files, job_create_times, **overrides):
        self.files = {f.id: f for f in files}
        self.job_create_times = job_create_times
        self.job_time_lookups = []
        self.events = []
        dependencies = {
            "list_job_files": self.list_job_files,
            "get_asset_file": self.get_asset_file,
            "get_job_create_times": self.get_job_create_times,
            "event_sink": lambda name, data: self.events.append((name, dict(data))),
            "get_user_id": lambda request: request.headers.get("comfy-user", "default"),
        }
        self.manager = AssetExportManager(**{**dependencies, **overrides})
        self.app = web.Application()
        self.manager.register_routes(self.app)

    def list_job_files(self, job_ids, include_previews):
        return [
            f for f in self.files.values()
            if f.job_id in job_ids and (include_previews or "temp" not in f.path)
        ]

    def get_job_create_times(self, job_ids):
        self.job_time_lookups.append(list(job_ids))
        return {job_id: ms for job_id, ms in self.job_create_times.items() if job_id in job_ids}

    def get_asset_file(self, asset_id):
        if asset_id not in self.files:
            raise ValueError(f"AssetReference {asset_id} not found")
        return self.files[asset_id]

    async def wait_finished(self, task_id, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            task = self.manager.get_task(task_id)
            if task is not None and task.finished:
                return task
            await asyncio.sleep(0.02)
        raise AssertionError("export did not finish")

    def archive_names(self, task):
        with zipfile.ZipFile(self.manager.export_path(task.export_name)) as archive:
            return sorted(archive.namelist())

    def final_event(self):
        return self.events[-1][1]


@pytest.fixture
def make_harness():
    def make(files=(), job_create_times=None, **overrides):
        return Harness(files, job_create_times or {}, **overrides)

    return make


async def start_export(client, body, user="default"):
    response = await client.post("/api/assets/export", json=body, headers={"comfy-user": user})
    assert response.status == 202
    data = await response.json()
    assert data["status"] == "created"
    return data["task_id"]


@pytest.mark.asyncio
async def test_export_job_outputs_end_to_end(aiohttp_client, make_harness, dirs):
    frames = {f"frame_{i:04d}.exr": os.urandom(1024 + i) for i in range(3)}
    files = [asset_file(write_file(dirs["output"], f"EXR/{name}", data)) for name, data in frames.items()]
    files.append(asset_file(write_file(dirs["temp"], "preview.png", b"preview")))
    harness = make_harness(files)
    client = await aiohttp_client(harness.app)

    task_id = await start_export(client, {"job_ids": [JOB_A], "naming_strategy": "preserve"})
    finished = await harness.wait_finished(task_id)
    assert harness.job_time_lookups == []

    task_response = await client.get(f"/api/tasks/{task_id}")
    assert task_response.status == 200
    task = await task_response.json()
    assert task["status"] == "completed"
    assert task["task_name"] == "task:export"
    assert task["idempotency_key"].startswith("export:default:preserve:false:")
    assert task["payload"]["job_ids"] == [JOB_A]
    total = sum(len(d) for d in frames.values())
    export_name = task["result"]["export_name"]
    assert EXPORT_NAME_PATTERN.match(export_name)
    assert task["result"] == {
        "success": True,
        "export_name": export_name,
        "export_size": os.path.getsize(harness.manager.export_path(export_name)),
        "uncompressed_size": total,
        "assets_total": 3,
        "assets_succeeded": 3,
    }

    statuses = [data["status"] for name, data in harness.events if name == "asset_export"]
    assert statuses[0] == "running" and statuses[-1] == "completed"
    assert all("export_name" not in data for _, data in harness.events[:-1])
    assert harness.final_event() == {
        "task_id": task_id,
        "export_name": export_name,
        "assets_total": 3,
        "assets_attempted": 3,
        "assets_failed": 0,
        "bytes_total": total,
        "bytes_processed": total,
        "progress": 1.0,
        "status": "completed",
    }

    url_response = await client.get(f"/api/assets/exports/{export_name}")
    assert url_response.status == 200
    url = urlsplit((await url_response.json())["url"])
    assert url.path == "/api/view"
    assert parse_qs(url.query) == {"filename": [export_name], "type": ["temp"], "subfolder": ["exports"]}
    assert harness.manager.export_path(export_name) == os.path.join(
        folder_paths.get_temp_directory(), "exports", export_name
    )
    assert harness.archive_names(finished) == sorted(f"EXR/{n}" for n in frames)


@pytest.mark.asyncio
async def test_export_honours_name_filters_and_previews(aiohttp_client, make_harness, dirs):
    harness = make_harness([
        asset_file(write_file(dirs["output"], "a.png", b"a")),
        asset_file(write_file(dirs["output"], "sub/b.png", b"b")),
        asset_file(write_file(dirs["output"], "c.png", b"c"), file_hash="blake3:c"),
        asset_file(write_file(dirs["output"], "d.png", b"d")),
        asset_file(write_file(dirs["temp"], "p.png", b"p")),
        asset_file(write_file(dirs["output"], "e.png", b"e"), job_id=JOB_B),
    ])
    client = await aiohttp_client(harness.app)

    task_id = await start_export(client, {
        "job_ids": [JOB_A, JOB_B],
        "job_asset_name_filters": {JOB_A: ["a.png", "sub/b.png", "blake3:c", "p.png"]},
        "include_previews": True,
        "naming_strategy": "group_by_job_id",
    })
    task = await harness.wait_finished(task_id)

    assert harness.archive_names(task) == [
        f"{JOB_A}/a.png",
        f"{JOB_A}/c.png",
        f"{JOB_A}/previews/p.png",
        f"{JOB_A}/sub/b.png",
        f"{JOB_B}/e.png",
    ]


@pytest.mark.asyncio
async def test_records_sharing_a_file_are_all_exported(aiohttp_client, make_harness, dirs):
    shared = write_file(dirs["output"], "a.png", b"a")
    harness = make_harness([asset_file(shared, job_id=JOB_A), asset_file(shared, job_id=JOB_B)])
    client = await aiohttp_client(harness.app)

    task = await harness.wait_finished(
        await start_export(client, {"job_ids": [JOB_A, JOB_B], "naming_strategy": "group_by_job_id"})
    )

    assert harness.archive_names(task) == [f"{JOB_A}/a.png", f"{JOB_B}/a.png"]


@pytest.mark.asyncio
async def test_uploaded_assets_are_named_after_the_asset(aiohttp_client, make_harness, dirs):
    upload = asset_file(write_file(dirs["input"], "0123abcd.png", b"cat"), job_id=None, name="cat.png")
    harness = make_harness([upload])
    client = await aiohttp_client(harness.app)

    task = await harness.wait_finished(
        await start_export(client, {"asset_ids": [upload.id], "naming_strategy": "preserve"})
    )

    assert harness.archive_names(task) == ["cat.png"]


@pytest.mark.asyncio
async def test_export_reports_each_asset_it_could_not_export(aiohttp_client, make_harness, dirs):
    exported = asset_file(write_file(dirs["output"], "a.png", b"a"), asset_id="00000000-0000-4000-8000-000000000001")
    renamed = asset_file(
        write_file(dirs["output"], "b.png", b"b"), asset_id="00000000-0000-4000-8000-000000000002", name="a.png"
    )
    deleted = asset_file(dirs["output"] / "gone.png", asset_id="00000000-0000-4000-8000-000000000003")
    unknown = str(uuid.uuid4())
    harness = make_harness([exported, renamed, deleted])
    client = await aiohttp_client(harness.app)

    task = await harness.wait_finished(await start_export(client, {
        "job_ids": [JOB_A],
        "asset_ids": [exported.id, unknown],
        "naming_strategy": "preserve",
    }))

    assert task.status == "completed"
    assert task.result["success"] is True
    assert task.result["assets_total"] == 2
    assert task.result["assets_succeeded"] == 1
    assert sorted(task.result["failed_assets"], key=lambda f: f["error"]) == [
        {"asset_id": deleted.id, "error": "fetch_failed", "job_id": JOB_A, "asset_name": "gone.png"},
        {"asset_id": renamed.id, "error": "filename_collision", "job_id": JOB_A, "asset_name": "a.png"},
        {"asset_id": unknown, "error": "not_found"},
    ]
    assert harness.final_event()["assets_failed"] == 1
    assert harness.archive_names(task) == ["a.png"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored, expected_error",
    [
        ("nothing", "No valid assets to export"),
        ("deleted", "All assets failed to export"),
        ("model", "One or more assets are not exportable"),
        ("outside", "One or more assets are not exportable"),
    ],
)
async def test_export_reports_failure_when_nothing_can_be_exported(
    aiohttp_client, make_harness, dirs, tmp_path, stored, expected_error
):
    files = {
        "nothing": [],
        "deleted": [asset_file(dirs["output"] / "gone.png")],
        "model": [asset_file(write_file(dirs["models"], "x.safetensors", b"m"))],
        "outside": [asset_file(write_file(tmp_path, "elsewhere/x.png", b"x"))],
    }[stored]
    harness = make_harness(files)
    client = await aiohttp_client(harness.app)

    task_id = await start_export(client, {"job_ids": [JOB_A]})
    task = await harness.wait_finished(task_id)

    assert task.status == "completed"
    assert task.result["success"] is False
    assert task.result["error"] == expected_error
    assert task.export_name is None
    assert harness.final_event()["status"] == "failed"
    assert harness.final_event()["error"] == expected_error
    export_directory = harness.manager.export_directory()
    assert not os.path.isdir(export_directory) or os.listdir(export_directory) == []
    body = await (await client.get(f"/api/tasks/{task_id}")).json()
    assert body["status"] == "completed"
    assert body["result"]["success"] is False
    assert "error_message" not in body


@pytest.mark.asyncio
async def test_unexpected_errors_fail_the_task(aiohttp_client, make_harness, dirs):
    def broken_lookup(job_ids, include_previews):
        raise RuntimeError("database is locked")

    harness = make_harness(list_job_files=broken_lookup)
    client = await aiohttp_client(harness.app)

    task_id = await start_export(client, {"job_ids": [JOB_A]})
    task = await harness.wait_finished(task_id)

    assert task.status == "failed"
    assert harness.final_event()["status"] == "failed"
    assert harness.final_event()["error"] == "database is locked"
    body = await (await client.get(f"/api/tasks/{task_id}")).json()
    assert body["error_message"] == "database is locked"
    assert "result" not in body


@pytest.mark.asyncio
async def test_export_completes_when_events_cannot_be_sent(aiohttp_client, make_harness, dirs):
    def broken_sink(name, data):
        raise ConnectionResetError()

    harness = make_harness([asset_file(write_file(dirs["output"], "a.png", b"a"))], event_sink=broken_sink)
    client = await aiohttp_client(harness.app)

    task = await harness.wait_finished(await start_export(client, {"job_ids": [JOB_A]}))

    assert task.succeeded


@pytest.mark.asyncio
async def test_asset_ids_are_exported_by_id(aiohttp_client, make_harness, dirs):
    stored = asset_file(write_file(dirs["output"], "renders/shot.exr", b"asset bytes"), asset_id=ASSET_ID)
    harness = make_harness([stored])
    client = await aiohttp_client(harness.app)

    task = await harness.wait_finished(
        await start_export(client, {"asset_ids": [ASSET_ID], "naming_strategy": "asset_id"})
    )

    with zipfile.ZipFile(harness.manager.export_path(task.export_name)) as archive:
        assert archive.namelist() == [f"{ASSET_ID}.exr"]
        assert archive.read(f"{ASSET_ID}.exr") == b"asset bytes"


@pytest.mark.asyncio
async def test_repeating_a_running_request_joins_its_export(aiohttp_client, make_harness, dirs):
    release = threading.Event()
    files = [asset_file(write_file(dirs["output"], "a.png", b"a"))]

    def slow_lookup(job_ids, include_previews):
        release.wait(5)
        return files

    harness = make_harness(files, list_job_files=slow_lookup)
    client = await aiohttp_client(harness.app)
    body = {"job_ids": [JOB_A]}

    first = await start_export(client, body)
    second = await start_export(client, body)
    other_user = await start_export(client, body, user="bob")
    release.set()
    await harness.wait_finished(first)
    after_finish = await start_export(client, body)

    assert second == first
    assert other_user != first
    assert after_finish != first


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "job_create_times, lookup_fails, expected_folder, used_fallback",
    [
        ({JOB_A: int(QUEUED.timestamp() * 1000)}, False, "2026-09-21T14-10-20", False),
        ({}, False, "2026-09-21T14-13-20", True),
        ({JOB_A: int(QUEUED.timestamp() * 1000)}, True, "2026-09-21T14-13-20", True),
    ],
    ids=["known job", "job the server forgot", "failed lookup"],
)
async def test_job_time_folders_use_the_job_creation_time(
    aiohttp_client, make_harness, dirs, job_create_times, lookup_fails, expected_folder, used_fallback
):
    def broken_lookup(job_ids):
        raise RuntimeError("queue unavailable")

    overrides = {"get_job_create_times": broken_lookup} if lookup_fails else {}
    harness = make_harness(
        [asset_file(write_file(dirs["output"], "a.png", b"a"))],
        job_create_times=job_create_times,
        **overrides,
    )
    client = await aiohttp_client(harness.app)

    task = await harness.wait_finished(await start_export(client, {"job_ids": [JOB_A]}))

    assert harness.archive_names(task) == [f"{expected_folder}/a.png"]
    assert task.result.get("used_job_time_fallback", False) is used_fallback
    assert harness.final_event().get("used_job_time_fallback", False) is used_fallback


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method, path",
    [
        ("post", "/api/assets/export"),
        ("get", "/api/assets/exports/unknown.zip"),
        ("get", f"/api/tasks/{ASSET_ID}"),
    ],
)
async def test_every_route_needs_the_assets_system(aiohttp_client, make_harness, dirs, monkeypatch, method, path):
    monkeypatch.setattr(routes, "_ASSETS_ENABLED", False)
    client = await aiohttp_client(make_harness().app)

    response = await getattr(client, method)(path, json={"job_ids": [JOB_A]})

    assert response.status == 503
    assert (await response.json())["error"]["code"] == "SERVICE_DISABLED"


@pytest.mark.asyncio
async def test_exports_are_private_to_their_owner(aiohttp_client, make_harness, dirs):
    harness = make_harness([asset_file(write_file(dirs["output"], "a.png", b"a"))])
    client = await aiohttp_client(harness.app)
    task_id = await start_export(client, {"job_ids": [JOB_A]}, user="alice")
    name = (await harness.wait_finished(task_id)).export_name

    assert (await client.get(f"/api/tasks/{task_id}", headers={"comfy-user": "bob"})).status == 404
    assert (await client.get(f"/api/assets/exports/{name}", headers={"comfy-user": "bob"})).status == 404
    assert (await client.get(f"/api/assets/exports/{name}", headers={"comfy-user": "alice"})).status == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method, path, kwargs, status, code",
    [
        ("post", "/api/assets/export", {"data": "not json"}, 400, "INVALID_REQUEST"),
        ("post", "/api/assets/export", {"json": {"job_ids": [JOB_A], "naming_strategy": "x"}}, 400, "INVALID_NAMING_STRATEGY"),
        ("get", "/api/assets/exports/bad name.zip", {}, 400, "INVALID_EXPORT_NAME"),
        ("get", "/api/assets/exports/nothing.tar", {}, 400, "INVALID_EXPORT_NAME"),
        ("get", "/api/assets/exports/unknown.zip", {}, 404, "EXPORT_NOT_FOUND"),
        ("get", "/api/tasks/not-a-uuid", {}, 404, "TASK_NOT_FOUND"),
        ("get", f"/api/tasks/{uuid.uuid4()}", {}, 404, "TASK_NOT_FOUND"),
    ],
)
async def test_invalid_requests_return_contract_errors(aiohttp_client, make_harness, dirs, method, path, kwargs, status, code):
    client = await aiohttp_client(make_harness().app)

    response = await getattr(client, method)(path, **kwargs)

    assert response.status == status
    assert (await response.json())["error"]["code"] == code


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method, path",
    [
        ("post", "/api/assets/export"),
        ("get", "/api/assets/exports/unknown.zip"),
        ("get", f"/api/tasks/{ASSET_ID}"),
    ],
)
async def test_unknown_user_is_rejected(aiohttp_client, make_harness, dirs, method, path):
    def reject(request):
        raise KeyError("Unknown user")

    client = await aiohttp_client(make_harness(get_user_id=reject).app)

    response = await getattr(client, method)(path, json={"job_ids": [JOB_A]})

    assert response.status == 401


@pytest.mark.asyncio
async def test_the_oldest_finished_export_is_forgotten_past_the_cap(aiohttp_client, make_harness, dirs, monkeypatch):
    monkeypatch.setattr(asset_export, "MAX_TRACKED_EXPORTS", 1)
    harness = make_harness([
        asset_file(write_file(dirs["output"], "a.png", b"a")),
        asset_file(write_file(dirs["output"], "b.png", b"b"), job_id=JOB_B),
    ])
    client = await aiohttp_client(harness.app)
    first = await harness.wait_finished(await start_export(client, {"job_ids": [JOB_A]}))

    second = await harness.wait_finished(await start_export(client, {"job_ids": [JOB_B]}))

    assert harness.manager.get_task(first.id) is None
    assert harness.manager.get_task(second.id) is second
    assert not os.path.exists(harness.manager.export_path(first.export_name))
    assert os.path.isfile(harness.manager.export_path(second.export_name))
