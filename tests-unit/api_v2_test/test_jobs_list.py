"""listJobs: the caller's records, newest first, with cursor paging and label filters."""

import base64
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from yarl import URL

from app.api_v2 import jobs
from app.assets.services.cursor import decode_cursor, encode_cursor_from_time

from .contract import response_schema, validate

WF = {"9": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}}}
T0 = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)
MS = int(T0.timestamp() * 1000)


@pytest.fixture
def clock(monkeypatch):
    t = {"now": T0 + timedelta(hours=1)}
    monkeypatch.setattr(jobs, "now", lambda: t["now"])
    return t


@pytest.fixture
def multi_user(fake_server):
    fake_server.user_manager.multi_user = True
    fake_server.user_manager.users |= {"alice", "bob"}


def add(v2_app, user="default", metadata=None, created_at=T0, job_id=None) -> str:
    job_id = job_id or str(uuid.uuid4())
    v2_app[jobs.JOBS_KEY].add(jobs.JobRecord(job_id, user, created_at, WF, metadata, None), set)
    return job_id


async def list_jobs(client, query="", user=None, raw=False):
    """GET /api/v2/jobs?<query>; ``raw`` sends the query bytes unchanged. Validates the body against the contract."""
    headers = {"comfy-user": user} if user else {}
    target = "/api/v2/jobs" + ("?" + query if query else "")
    resp = await client.get(URL(target, encoded=True) if raw else target, headers=headers)
    data = await resp.json()
    validate(response_schema("listJobs", resp.status), data)
    return resp, data


def ids(data):
    return [job["id"] for job in data["jobs"]]


async def expect_error(client, query, status, code, raw=False):
    resp, data = await list_jobs(client, query, raw=raw)
    assert (resp.status, data["error"]["code"]) == (status, code), (query, data)


@pytest.mark.asyncio
async def test_lists_own_jobs_newest_first_and_items_conform(conforming_client, v2_app, fake_server, clock):
    old = add(v2_app, metadata={"client": "acme"}, created_at=T0)
    new = add(v2_app, created_at=T0 + timedelta(seconds=1))
    fake_server.prompt_queue.queued.append((1, new, WF, {"create_time": MS, "client_id": new}, ["9"], {}))
    resp, data = await list_jobs(conforming_client)
    assert resp.status == 200
    validate("JobList", data)
    assert ids(data) == [new, old]
    assert "next_cursor" not in data and data["truncated"] is False
    first, second = data["jobs"]
    for item in data["jobs"]:
        validate("JobListItem", item)
        validate("Job", {k: v for k, v in item.items() if k not in ("create_time", "update_time")})
        assert item["create_time"] == item["created_at"]
    assert first["status"] == "queued" and first["queue_position"] == 1 and first["update_time"] == first["created_at"]
    assert second["status"] == "expired" and second["metadata"] == {"client": "acme"}
    assert second["urls"]["self"].endswith(f"/api/v2/jobs/{old}")


@pytest.mark.asyncio
async def test_update_time_is_completed_then_started(conforming_client, v2_app, fake_server, clock):
    job_id = add(v2_app)
    messages = [("execution_start", {"timestamp": MS + 1000}), ("execution_success", {"timestamp": MS + 5000})]
    fake_server.prompt_queue.history[job_id] = {"prompt": (1, job_id, WF, {}, ["9"]), "outputs": {}, "status": {"status_str": "success", "messages": messages}}
    _, data = await list_jobs(conforming_client)
    (item,) = data["jobs"]
    assert item["status"] == "succeeded" and item["update_time"] == item["completed_at"] != item["created_at"]
    messages.pop()
    fake_server.prompt_queue.history[job_id]["status"]["status_str"] = "error"
    _, data = await list_jobs(conforming_client)
    assert data["jobs"][0]["update_time"] == data["jobs"][0]["started_at"]


@pytest.mark.asyncio
async def test_user_isolation(conforming_client, v2_app, multi_user, clock):
    a = add(v2_app, user="alice", metadata={"client": "acme"})
    b = add(v2_app, user="bob")
    assert ids((await list_jobs(conforming_client, user="alice"))[1]) == [a]
    assert ids((await list_jobs(conforming_client, user="bob"))[1]) == [b]
    assert ids((await list_jobs(conforming_client, "metadata[client]=acme", user="bob"))[1]) == []
    resp, data = await list_jobs(conforming_client, user="mallory")
    assert (resp.status, data["error"]["code"]) == (403, "forbidden")


@pytest.mark.asyncio
async def test_retention(conforming_client, v2_app, clock):
    add(v2_app, created_at=T0)
    clock["now"] = T0 + jobs.RETENTION
    assert (await list_jobs(conforming_client))[1]["jobs"] == []


@pytest.mark.asyncio
async def test_default_limit_and_clamp(conforming_client, v2_app, clock):
    for i in range(501):
        add(v2_app, created_at=T0 + timedelta(milliseconds=i))
    for query in ("", "limit=10000", "limit=500", "limit=00000000000000000000000000000000000000000000000001" + "0" * 4400):
        resp, data = await list_jobs(conforming_client, query)
        assert resp.status == 200 and len(data["jobs"]) == 500 and "next_cursor" in data, query
    _, data = await list_jobs(conforming_client, "limit=1")
    assert len(data["jobs"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["0", "abc", "-1", "1.5", "", "+3", "1e3", "0000"])
async def test_bad_limit(conforming_client, v2_app, clock, value):
    await expect_error(conforming_client, f"limit={value}", 422, "invalid_request")


@pytest.mark.asyncio
async def test_three_page_walk_with_tied_timestamps(conforming_client, v2_app, clock):
    created = [T0, T0, T0 + timedelta(seconds=1), T0 + timedelta(seconds=1), T0 + timedelta(seconds=1), T0 + timedelta(seconds=2), T0 + timedelta(seconds=2)]
    all_ids = [add(v2_app, created_at=t) for t in created]
    expected = [job_id for _, job_id in sorted(zip(created, all_ids), reverse=True)]
    seen, query, pages = [], "limit=3", 0
    while True:
        resp, data = await list_jobs(conforming_client, query)
        assert resp.status == 200
        seen += ids(data)
        pages += 1
        if "next_cursor" not in data:
            break
        query = f"limit=3&cursor={data['next_cursor']}"
    assert pages == 3 and seen == expected and len(set(seen)) == 7
    next_cursor = (await list_jobs(conforming_client, "limit=3"))[1]["next_cursor"]
    assert "=" not in next_cursor and decode_cursor(next_cursor, ("created_at",), "desc").id == expected[2]


@pytest.mark.asyncio
@pytest.mark.parametrize("micros", [0, 123456])
async def test_next_cursor_round_trips_microseconds(conforming_client, v2_app, clock, micros):
    older = add(v2_app, created_at=T0 + timedelta(microseconds=micros))
    add(v2_app, created_at=T0 + timedelta(seconds=1, microseconds=micros))
    _, data = await list_jobs(conforming_client, "limit=1")
    resp, data = await list_jobs(conforming_client, f"limit=1&cursor={data['next_cursor']}")
    assert resp.status == 200 and ids(data) == [older]


@pytest.mark.asyncio
async def test_exactly_limit_has_no_next_cursor(conforming_client, v2_app, clock):
    add(v2_app)
    add(v2_app)
    _, data = await list_jobs(conforming_client, "limit=2")
    assert len(data["jobs"]) == 2 and "next_cursor" not in data


def b64(obj) -> str:
    raw = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", [
    "garbage!", "@@@", "", "%FF", b64(b"not json"), b64([1]), b64({"t": "2026-11-14T12:00:00Z"}), b64({"id": "x"}),
    b64({"t": "yesterday", "id": "x"}), b64({"t": "2026-11-14T12:00:00", "id": "x"}), b64({"t": 1, "id": "x"}), b64(b"\xff\xfe"),
    b64({"s": "created_at", "v": "yesterday", "id": "x", "o": "desc"}), b64({"s": "created_at", "v": "1", "id": "x", "o": "asc"}),
    b64({"s": "name", "v": "1", "id": "x", "o": "desc"}), b64({"s": "created_at", "v": str(10**20), "id": "x", "o": "desc"}), b64(b"[" * 6000),
])
async def test_bad_cursor(conforming_client, v2_app, clock, cursor):
    add(v2_app)
    await expect_error(conforming_client, f"cursor={cursor}", 400, "invalid_cursor", raw=True)


@pytest.mark.asyncio
async def test_foreign_wellformed_cursor_is_a_position(conforming_client, v2_app, clock):
    a = add(v2_app, created_at=T0)
    add(v2_app, created_at=T0 + timedelta(seconds=1))
    _, data = await list_jobs(conforming_client, "cursor=" + encode_cursor_from_time("created_at", T0 + timedelta(milliseconds=500), "whatever"))
    assert ids(data) == [a]


@pytest.mark.asyncio
async def test_metadata_filters(conforming_client, v2_app, clock):
    acme = add(v2_app, metadata={"client": "acme", "env": "prod"})
    add(v2_app, metadata={"client": "other", "env": "prod"})
    add(v2_app)
    assert ids((await list_jobs(conforming_client, "metadata[client]=acme"))[1]) == [acme]
    assert ids((await list_jobs(conforming_client, "metadata%5Bclient%5D=acme&metadata%5Benv%5D=prod"))[1]) == [acme]
    assert ids((await list_jobs(conforming_client, "metadata[client]=acme&metadata[env]=dev"))[1]) == []
    assert ids((await list_jobs(conforming_client, "metadata[client]=nobody"))[1]) == []
    assert ids((await list_jobs(conforming_client, "metadata[client]=ACME"))[1]) == []
    assert ids((await list_jobs(conforming_client, "metadata[a]=1&metadata[b]=2&metadata[c]=3"))[1]) == []


@pytest.mark.asyncio
async def test_filtered_paging(conforming_client, v2_app, clock):
    acme = [add(v2_app, metadata={"client": "acme"}, created_at=T0 + timedelta(seconds=i)) for i in range(5)]
    for i in range(5):
        add(v2_app, metadata={"client": "other"}, created_at=T0 + timedelta(seconds=i, milliseconds=500))
    seen, query = [], "limit=2&metadata[client]=acme"
    while True:
        _, data = await list_jobs(conforming_client, query)
        seen += ids(data)
        if "next_cursor" not in data:
            break
        query = f"limit=2&metadata[client]=acme&cursor={data['next_cursor']}"
    assert seen == acme[::-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "metadata[a]=1&metadata[b]=2&metadata[c]=3&metadata[d]=4",
    "metadata[a]=1&metadata[a]=2",
    "metadata[a]=1&metadata%5Ba%5D=1",
    "metadata[b%20c]=1", "metadata[a!]=1", "metadata[" + "k" * 41 + "]=1", "metadata[%C3%A9]=1",
    "metadata[a]=x%00", "metadata[a]=x%7F", "metadata[a]=%C2%85", "metadata[a]=%E2%80%AE", "metadata[a]=%E2%81%A6",
    # raw ; and bad escapes
    "metadata[a]=x;y", "metadata[a]=%zz", "metadata[a]=%2", "metadata[a;]=x", "metadata[%zz]=x",
    # empty keys
    "metadata[]=x", "metadata%5B%5D=x", "metadata%5B%5D=x;y", "metadata%5B%5D=%zz",
    # newline keys
    "metadata[%0A]=x", "metadata[a%0Ab]=x", "metadata%5Ba%0Ab%5D=x", "metadata[a%0A]=x",
    # encoded names
    "%6Detadata%5Ba%5D=x;y", "%6Detadata%5Ba%5D=%zz", "%6Detadata%5B%5D=x",
])
async def test_bad_metadata_filter(conforming_client, v2_app, clock, query):
    await expect_error(conforming_client, query, 400, "invalid_metadata_filter", raw=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("query,label", [
    ("metadata[a]=x%3By", "x;y"),
    ("metadata[a]=50%25", "50%"),
    ("metadata[a]=a%252Fb", "a%2Fb"),
    ("metadata[a]=%25zz", "%zz"),
    ("metadata[a]=%2520", "%20"),
    ("metadata[a]=a+b", "a b"),
    ("metadata[a]=%E2%9C%93", "\u2713"),
    ("%6Detadata%5Ba%5D=x%3By", "x;y"),
    ("metadata[a.b-c_D]=v&foo=a;b&bar=%zz", None),
])
async def test_encoded_metadata_values_match_exactly(conforming_client, v2_app, clock, query, label):
    key, value = ("a.b-c_D", "v") if label is None else ("a", label)
    match = add(v2_app, metadata={key: value})
    add(v2_app, metadata={"a": "x"})
    resp, data = await list_jobs(conforming_client, query, raw=True)
    assert resp.status == 200 and ids(data) == [match], (query, data)


@pytest.mark.asyncio
async def test_raw_semicolon_in_other_pair_is_ignored(conforming_client, v2_app, clock):
    job_id = add(v2_app)
    resp, data = await list_jobs(conforming_client, "foo=a;b&limit=5&&=&metadata=x", raw=True)
    assert resp.status == 200 and ids(data) == [job_id]


@pytest.mark.asyncio
async def test_status_filter(conforming_client, v2_app, fake_server, clock):
    queued = add(v2_app, created_at=T0)
    expired = add(v2_app, created_at=T0 + timedelta(seconds=1))
    running = add(v2_app, created_at=T0 + timedelta(seconds=2))
    fake_server.prompt_queue.queued.append((1, queued, WF, {"create_time": MS}, ["9"], {}))
    fake_server.prompt_queue.running.append((2, running, WF, {"create_time": MS}, ["9"], {}))
    assert ids((await list_jobs(conforming_client, "status=queued"))[1]) == [queued]
    assert ids((await list_jobs(conforming_client, "status=queued,running"))[1]) == [running, queued]
    assert ids((await list_jobs(conforming_client, "status=expired"))[1]) == [expired]
    assert ids((await list_jobs(conforming_client, "status=succeeded"))[1]) == []
    _, data = await list_jobs(conforming_client, "status=queued,expired&limit=1")
    assert ids(data) == [expired] and "next_cursor" in data
    assert ids((await list_jobs(conforming_client, f"status=queued,expired&limit=1&cursor={data['next_cursor']}"))[1]) == [queued]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["done", "", "queued,", "QUEUED", "queued,bogus"])
async def test_bad_status(conforming_client, v2_app, clock, value):
    await expect_error(conforming_client, f"status={value}", 400, "invalid_request")


@pytest.mark.asyncio
async def test_list_route_does_not_shadow_job_routes(conforming_client, v2_app, clock):
    job_id = add(v2_app)
    resp = await conforming_client.get(f"/api/v2/jobs/{job_id}")
    assert resp.status == 200 and (await resp.json())["id"] == job_id
    resp = await conforming_client.get("/api/v2/jobs/")
    assert resp.status == 404
    resp = await conforming_client.head("/api/v2/jobs")
    assert resp.status == 405
