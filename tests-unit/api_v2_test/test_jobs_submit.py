"""postJobs: validation order, core/ASSET rewrite, enqueue outcomes, idempotency keys and the job store."""

import asyncio
import json
import math
import os
import types
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
import torch
from aiohttp import FormData

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import execution  # noqa: E402
import server  # noqa: E402
from app.api_v2 import jobs  # noqa: E402
from app.assets.services.ingest import register_executed_output, register_file_in_place  # noqa: E402

from .contract import response_schema, validate  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"v2-jobs-test"
WF = {"9": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}}}
T0 = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)


def asset(info) -> dict:
    return {"__type": "core/ASSET", "info": info}


def wf_with(value) -> dict:
    return {"1": {"class_type": "LoadImage", "inputs": {"image": value}}, **WF}


@pytest.fixture
def clock(monkeypatch):
    t = {"now": T0}
    monkeypatch.setattr(jobs, "now", lambda: t["now"])
    return t


@pytest.fixture
def queue(fake_server):
    """Accepting ``enqueue_prompt``; the list collects every ``json_data`` it got."""
    calls = []

    async def enqueue(json_data, usage_source=None):
        calls.append((json_data, usage_source))
        await asyncio.sleep(0.01)
        return 200, {"prompt_id": json_data["prompt_id"], "number": len(calls), "node_errors": {}}

    fake_server.enqueue_prompt = enqueue
    return calls


def store(v2_app) -> jobs.JobStore:
    return v2_app[jobs.JOBS_KEY]


async def post(client, body=None, key=None, user=None, raw=None, headers=None):
    headers = dict(headers or {})
    if key is not None:
        headers["Idempotency-Key"] = key
    if user is not None:
        headers["comfy-user"] = user
    if raw is not None:
        resp = await client.post("/api/v2/jobs", data=raw, headers={"Content-Type": "application/json", **headers})
    else:
        resp = await client.post("/api/v2/jobs", json=body, headers=headers)
    data = await resp.json()
    validate(response_schema("postJobs", resp.status), data)
    return resp, data


def assert_error(resp, data, status, code, details=...):
    assert (resp.status, data["error"]["code"]) == (status, code), data
    if details is not ...:
        assert data["error"]["details"] == details


async def upload(client, file_path, data=PNG, content_type="image/png", tags=()) -> dict:
    fd = FormData(default_to_multipart=True)
    fd.add_field("file_path", file_path)
    fd.add_field("content_type", content_type)
    fd.add_field("tags", json.dumps(list(tags)))
    fd.add_field("file", data, filename="f.bin", content_type="application/octet-stream")
    resp = await client.post("/api/v2/assets", data=fd)
    body = await resp.json()
    assert resp.status == 201, body
    return body


def safetensors(payload=b"\0\0\0\0") -> bytes:
    header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, len(payload)]}}).encode()
    return len(header).to_bytes(8, "little") + header + payload


def register(path, data=PNG, tags=("output",)) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return register_file_in_place(abs_path=str(path), name=path.name, tags=list(tags), mime_type="image/png").ref.id


# happy path


@pytest.mark.asyncio
async def test_submit_returns_conforming_job(conforming_client, v2_app, v2_dirs, queue, clock):
    resp, body = await post(conforming_client, {"workflow": WF, "metadata": {"client": "acme"}, "priority": 5, "webhook_url": None, "inputs": None}, headers={"Comfy-Usage-Source": "sdk"})
    assert resp.status == 201, body
    validate("Job", body)
    job_id = body["id"]
    assert str(uuid.UUID(job_id)) == job_id
    base = str(conforming_client.make_url("")).rstrip("/")
    assert body["urls"] == {"self": f"{base}/api/v2/jobs/{job_id}", "events": f"{base}/api/v2/jobs/{job_id}/events", "cancel": f"{base}/api/v2/jobs/{job_id}/cancel"}
    assert body["status"] == "queued" and body["outputs"] == [] and body["outputs_reused"] is False
    assert body["metadata"] == {"client": "acme"} and body["priority"] == 5
    assert body["created_at"] == "2026-11-14T12:00:00Z" and body["expires_at"] == "2026-11-15T12:00:00Z"
    [(json_data, usage_source)] = queue
    assert json_data == {"prompt": WF, "prompt_id": job_id, "client_id": job_id, "extra_data": {}}
    assert usage_source == "sdk"
    record = store(v2_app).records[job_id]
    assert (record.user_id, record.workflow, record.metadata, record.priority, record.created_at) == ("default", WF, {"client": "acme"}, 5, T0)


@pytest.mark.asyncio
async def test_metadata_and_priority_absent_when_unset(conforming_client, v2_dirs, queue):
    for extras in ({}, {"metadata": None, "priority": None, "extra_data": None}, {"metadata": {}}):
        resp, body = await post(conforming_client, {"workflow": WF, **extras})
        assert resp.status == 201, body
        assert "metadata" not in body and "priority" not in body


@pytest.mark.asyncio
async def test_record_keeps_graph_enqueue_mutates(conforming_client, v2_app, v2_dirs, fake_server):
    async def enqueue(json_data, usage_source=None):
        json_data["prompt"]["9"]["inputs"]["filename_prefix"] = "mutated"
        return 200, {}
    fake_server.enqueue_prompt = enqueue
    resp, body = await post(conforming_client, {"workflow": WF})
    assert store(v2_app).records[body["id"]].workflow == WF


# validation, in order


BAD_BODIES = [
    (b"not json", 400, "invalid_request"),
    (b"\xff\xfe", 400, "invalid_request"),
    (b"", 400, "invalid_request"),
    (b"[]", 400, "invalid_request"),
    (b"null", 400, "invalid_request"),
    (b'"workflow"', 400, "invalid_request"),
    (b"[" * 100_000 + b"]" * 100_000, 400, "invalid_request"),
    (b'{"workflow": {"a": ' * 100_000 + b"1" + b"}}" * 100_000, 400, "invalid_request"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,status,code", BAD_BODIES, ids=range(len(BAD_BODIES)))
async def test_malformed_body(conforming_client, v2_dirs, queue, raw, status, code):
    resp, data = await post(conforming_client, raw=raw)
    assert_error(resp, data, status, code)
    assert queue == []


INVALID = [
    ({"workflow": WF, "webhook_url": "https://x"}, 422, "invalid_workflow"),
    ({"workflow": WF, "inputs": {}}, 422, "invalid_workflow"),
    ({"workflow": "not a graph", "webhook_url": "https://x"}, 422, "invalid_workflow"),
    ({}, 422, "invalid_workflow"),
    ({"workflow": None}, 422, "invalid_workflow"),
    ({"workflow": []}, 422, "invalid_workflow"),
    ({"workflow": {"nodes": [], "links": []}, "metadata": "bad"}, 422, "workflow_format_ui"),
    ({"workflow": {"nodes": [], "links": None}}, 422, "workflow_format_ui"),
    ({"workflow": WF, "metadata": "x", "priority": True}, 400, "invalid_request"),
    ({"workflow": WF, "metadata": [], "priority": True}, 400, "invalid_request"),
    ({"workflow": WF, "metadata": {f"k{i}": "v" for i in range(17)}, "priority": True}, 422, "metadata_invalid"),
    ({"workflow": WF, "priority": True, "extra_data": "x"}, 422, "invalid_request"),
    ({"workflow": WF, "priority": 1.0}, 422, "invalid_request"),
    ({"workflow": WF, "priority": "1"}, 422, "invalid_request"),
    ({"workflow": WF, "priority": 1_000_001}, 422, "invalid_request"),
    ({"workflow": WF, "priority": -1_000_001}, 422, "invalid_request"),
    ({"workflow": wf_with(asset({"id": str(uuid.uuid4())})), "extra_data": "x"}, 400, "invalid_request"),
    ({"workflow": WF, "extra_data": []}, 400, "invalid_request"),
    ({"workflow": WF, "extra_data": {"client_id": "x"}}, 400, "invalid_request"),
    ({"workflow": WF, "extra_data": {"api_key_comfy_org": 1}}, 400, "invalid_request"),
    ({"workflow": WF, "extra_data": {"auth_token_comfy_org": None}}, 400, "invalid_request"),
    ({"workflow": WF, "extra_data": {"preview_method": ["auto"]}}, 400, "invalid_request"),
    ({"workflow": WF, "extra_data": {"extra_pnginfo": "x"}}, 400, "invalid_request"),
    ({"workflow": wf_with(asset({"id": str(uuid.uuid4())}))}, 422, "missing_asset"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("body,status,code", INVALID, ids=range(len(INVALID)))
async def test_validation_order(conforming_client, v2_dirs, queue, body, status, code):
    resp, data = await post(conforming_client, body, key="")
    assert_error(resp, data, status, code)
    assert queue == []


MALFORMED_NODES = [
    {"1": None},
    {"1": []},
    {"1": {"inputs": {}}},
    {"1": {"class_type": ["SaveImage"], "inputs": {}}},
    {"1": {"class_type": "SaveImage"}},
    {"1": {"class_type": "SaveImage", "inputs": []}},
]


@pytest.mark.asyncio
@pytest.mark.parametrize("graph", MALFORMED_NODES, ids=range(len(MALFORMED_NODES)))
async def test_malformed_node_keeps_key_unclaimed(conforming_client, v2_app, v2_dirs, queue, graph):
    resp, data = await post(conforming_client, {"workflow": graph, "metadata": "bad"}, key="k")
    assert_error(resp, data, 422, "invalid_workflow")
    assert queue == [] and store(v2_app).records == {} and store(v2_app).idempotency == {}
    resp, body = await post(conforming_client, {"workflow": WF}, key="k")
    assert resp.status == 201, body


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [600, 2000, 9000])
async def test_deeply_nested_workflow(conforming_client, v2_app, v2_dirs, queue, depth):
    raw = b'{"workflow": {"1": {"class_type": "X", "inputs": {"a": ' + b"[" * depth + b"]" * depth + b"}}}}"
    resp, data = await post(conforming_client, raw=raw, key="deep")
    # Python 3.10's json.loads hits the recursion limit itself past ~1000 levels; 3.12 parses and the walk hits it.
    allowed = {(422, "invalid_workflow")} if depth < 1000 else {(422, "invalid_workflow"), (400, "invalid_request")}
    assert (resp.status, data["error"]["code"]) in allowed, data
    assert queue == [] and store(v2_app).idempotency == {}


@pytest.mark.asyncio
async def test_metadata_count_has_no_details(conforming_client, v2_dirs, queue):
    resp, data = await post(conforming_client, {"workflow": WF, "metadata": {f"k{i}": "v" for i in range(17)}})
    assert_error(resp, data, 422, "metadata_invalid", details=None)
    assert "17" in data["error"]["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key,value",
    [
        ("", "v"),
        ("a b", "v"),
        ("k" * 41, "v"),
        ("ключ", "v"),
        ("k/1", "v"),
        ("k", 1),
        ("k", None),
        ("k", ["v"]),
        ("k", "x" * 257),
        ("k", "é" * 129),
        ("k", "a\tb"),
        ("k", "a\nb"),
        ("k", "\x00"),
        ("k", "\x7f"),
        ("k", "\x85"),
        ("k", "\x9f"),
        ("k", "\u202a"),
        ("k", "\u202e"),
        ("k", "\u2066"),
        ("k", "\u2069"),
    ],
)
async def test_bad_metadata_pair_names_key(conforming_client, v2_dirs, queue, key, value):
    resp, data = await post(conforming_client, {"workflow": WF, "metadata": {"ok": "v", key: value}})
    assert_error(resp, data, 422, "metadata_invalid", details={"key": key})


@pytest.mark.asyncio
async def test_lone_surrogate_metadata_value(conforming_client, v2_dirs, queue):
    resp, data = await post(conforming_client, raw=b'{"workflow": {}, "metadata": {"k": "\\ud800"}}')
    assert_error(resp, data, 422, "metadata_invalid", details={"key": "k"})


@pytest.mark.asyncio
async def test_metadata_and_priority_limits_accepted(conforming_client, v2_dirs, queue):
    metadata = {f"k{i}": "v" for i in range(14)} | {"A" * 40: "x" * 256, "a._-9": "é" * 128}
    for priority in (-1_000_000, 0, 1_000_000):
        resp, body = await post(conforming_client, {"workflow": WF, "metadata": metadata, "priority": priority})
        assert resp.status == 201, body
        assert body["metadata"] == metadata and body["priority"] == priority


@pytest.mark.asyncio
async def test_non_ui_graph_with_nodes_key_passes(conforming_client, v2_dirs, queue):
    for workflow in ({"nodes": WF["9"], "links": WF["9"]}, {"nodes": WF["9"]}):
        resp, body = await post(conforming_client, {"workflow": workflow})
        assert resp.status == 201, body


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["", "k" * 256])
async def test_bad_idempotency_key(conforming_client, v2_dirs, queue, key):
    resp, data = await post(conforming_client, {"workflow": WF}, key=key)
    assert_error(resp, data, 400, "invalid_request")
    resp, body = await post(conforming_client, {"workflow": WF}, key="k" * 255)
    assert resp.status == 201, body


@pytest.mark.asyncio
async def test_unknown_user_forbidden(conforming_client, v2_dirs, queue, fake_server):
    fake_server.user_manager.multi_user = True
    resp, data = await post(conforming_client, {"workflow": WF}, user="mallory")
    assert_error(resp, data, 403, "forbidden")
    assert queue == []


# core/ASSET


@pytest.mark.asyncio
async def test_asset_refs_resolve_to_widget_values(conforming_client, v2_dirs, queue):
    input_asset = await upload(conforming_client, "input/sub/a.png")
    model = await upload(conforming_client, "m.safetensors", data=safetensors(), content_type="application/octet-stream", tags=("models", "model_type:checkpoints"))
    output_id = register(v2_dirs / "output" / "x.png", data=PNG + b"out")
    temp_id = register(v2_dirs / "temp" / "t.png", data=PNG + b"temp", tags=("temp",))
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": asset({"id": input_asset["id"]})}},
        "2": {"class_type": "LoadImage", "inputs": {"image": asset({"id": output_id, "hash": "ignored"})}},
        "3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": asset({"id": model["id"]})}},
        "4": {"class_type": "X", "inputs": {"deep": [{"a": [asset({"id": temp_id})]}], "plain": {"__type": "other", "info": 1}}},
    }
    resp, body = await post(conforming_client, {"workflow": workflow})
    assert resp.status == 201, body
    prompt = queue[0][0]["prompt"]
    assert prompt["1"]["inputs"]["image"] == input_asset["file_path"] == input_asset["hash"].removeprefix("blake3:") + ".png"
    assert prompt["2"]["inputs"]["image"] == "x.png [output]"
    assert prompt["3"]["inputs"]["ckpt_name"] == model["file_path"] == model["hash"].removeprefix("blake3:") + ".safetensors"
    assert prompt["4"]["inputs"] == {"deep": [{"a": ["t.png [temp]"]}], "plain": {"__type": "other", "info": 1}}


@pytest.mark.asyncio
async def test_hash_only_reference_accepted(conforming_client, v2_dirs, queue):
    uploaded = await upload(conforming_client, "input/h.png")
    resp, body = await post(conforming_client, {"workflow": wf_with(asset({"hash": uploaded["hash"]}))})
    assert resp.status == 201, body
    assert queue[0][0]["prompt"]["1"]["inputs"]["image"] == uploaded["file_path"]


@pytest.mark.asyncio
async def test_unknown_id_never_falls_back_to_hash(conforming_client, v2_dirs, queue):
    uploaded = await upload(conforming_client, "input/h.png")
    unknown = str(uuid.uuid4())
    resp, data = await post(conforming_client, {"workflow": wf_with(asset({"id": unknown, "hash": uploaded["hash"]}))})
    assert_error(resp, data, 422, "missing_asset", details={"missing_ids": [unknown]})
    assert queue == []


@pytest.mark.asyncio
async def test_unresolvable_references_listed(conforming_client, v2_dirs, queue):
    workflow = {
        "1": {"class_type": "X", "inputs": {"a": {"__type": "core/ASSET"}}},
        "2": {"class_type": "X", "inputs": {"a": asset("x")}},
        "3": {"class_type": "X", "inputs": {"a": asset({"id": 123})}},
        "4": {"class_type": "X", "inputs": {"a": asset({"id": "not-a-uuid"})}},
        "5": {"class_type": "X", "inputs": {"a": asset({"hash": "blake3:nope"})}},
        "6": {"class_type": "X", "inputs": {"a": asset({"hash": "blake3:" + "0" * 64})}},
        "7": {"class_type": "X", "inputs": {"a": asset({})}},
        "8": {"class_type": "X", "inputs": {"a": asset({"hash": 5})}},
    }
    resp, data = await post(conforming_client, {"workflow": workflow}, key="k")
    expected = ["<malformed core/ASSET>", "<malformed core/ASSET>", "123", "not-a-uuid", "blake3:nope", "blake3:" + "0" * 64, "<malformed core/ASSET>", "5"]
    assert_error(resp, data, 422, "missing_asset", details={"missing_ids": expected})
    resp, body = await post(conforming_client, {"workflow": WF}, key="k")
    assert resp.status == 201, body


@pytest.mark.asyncio
async def test_reference_with_gone_or_escaped_file_is_missing(conforming_client, v2_dirs, tmp_path, queue):
    gone = await upload(conforming_client, "input/gone.png", data=PNG + b"gone")
    os.unlink(v2_dirs / "input" / gone["file_path"])
    escaped = await upload(conforming_client, "input/esc.png", data=PNG + b"esc")
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG + b"esc")
    os.unlink(v2_dirs / "input" / escaped["file_path"])
    os.symlink(outside, v2_dirs / "input" / escaped["file_path"])
    workflow = {"1": {"class_type": "X", "inputs": {"a": asset({"id": gone["id"]}), "b": asset({"id": escaped["id"]})}}}
    resp, data = await post(conforming_client, {"workflow": workflow})
    assert_error(resp, data, 422, "missing_asset", details={"missing_ids": [gone["id"], escaped["id"]]})


# enqueue


@pytest.mark.asyncio
async def test_extra_data_secrets_reach_sensitive_tuple_slot(conforming_client, v2_dirs, fake_server, monkeypatch):
    async def validate_prompt(prompt_id, prompt, partial_execution_targets):
        return True, None, ["9"], {}

    monkeypatch.setattr(execution, "validate_prompt", validate_prompt)
    fake_server.number = 0
    fake_server.node_replace_manager = MagicMock()
    fake_server.trigger_on_prompt = lambda json_data: json_data
    fake_server.enqueue_prompt = types.MethodType(server.PromptServer.enqueue_prompt, fake_server)
    extra = {"api_key_comfy_org": "key-secret", "auth_token_comfy_org": "token-secret", "preview_method": "auto", "extra_pnginfo": {"workflow": {}}}
    resp, body = await post(conforming_client, {"workflow": WF, "extra_data": extra})
    assert resp.status == 201, body
    [item] = fake_server.prompt_queue.put_items
    assert item[1] == body["id"] and item[2] == WF
    assert item[5] == {"auth_token_comfy_org": "token-secret", "api_key_comfy_org": "key-secret"}
    assert "key-secret" not in json.dumps(item[3]) and "token-secret" not in json.dumps(item[3])
    assert item[3]["preview_method"] == "auto" and item[3]["extra_pnginfo"] == {"workflow": {}} and item[3]["client_id"] == body["id"]


@pytest.mark.asyncio
async def test_rejected_workflow_releases_key(conforming_client, v2_app, v2_dirs, fake_server, queue):
    accept = fake_server.enqueue_prompt

    async def reject(json_data, usage_source=None):
        return 400, {"error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"}, "node_errors": {"9": {"errors": ["x"]}}}

    fake_server.enqueue_prompt = reject
    resp, data = await post(conforming_client, {"workflow": WF}, key="k")
    assert_error(resp, data, 422, "invalid_workflow", details={"node_errors": {"9": {"errors": ["x"]}}})
    assert data["error"]["message"] == "Prompt outputs failed validation"
    assert store(v2_app).records == {} and store(v2_app).idempotency == {}
    fake_server.enqueue_prompt = accept
    resp, body = await post(conforming_client, {"workflow": WF}, key="k")
    assert resp.status == 201, body


@pytest.mark.asyncio
async def test_enqueue_exception_holds_key(conforming_client, v2_app, v2_dirs, fake_server, queue):
    accept = fake_server.enqueue_prompt

    async def boom(json_data, usage_source=None):
        raise RuntimeError("queue on fire")

    fake_server.enqueue_prompt = boom
    for _ in range(3):
        resp, data = await post(conforming_client, {"workflow": WF}, key=f"k{_}")
        assert_error(resp, data, 500, "upstream_error")
    assert len(store(v2_app).records) == 3
    fake_server.enqueue_prompt = accept
    resp, data = await post(conforming_client, {"workflow": WF}, key="k0")
    assert_error(resp, data, 422, "idempotency_key_reuse")
    assert queue == []


# idempotency keys


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt", range(5))
async def test_concurrent_same_key(conforming_client, v2_dirs, queue, attempt):
    uploaded = await upload(conforming_client, "input/race.png")
    body = {"workflow": wf_with(asset({"id": uploaded["id"]}))}
    results = await asyncio.gather(*(post(conforming_client, body, key="race") for _ in range(2)))
    statuses = sorted(resp.status for resp, _ in results)
    assert statuses == [201, 422]
    assert [data["error"]["code"] for resp, data in results if resp.status == 422] == ["idempotency_key_reuse"]
    assert len(queue) == 1


@pytest.mark.asyncio
async def test_keys_are_per_user(conforming_client, v2_dirs, queue, fake_server):
    fake_server.user_manager.multi_user = True
    fake_server.user_manager.users |= {"alice", "bob"}
    for user in ("alice", "bob"):
        resp, body = await post(conforming_client, {"workflow": WF}, key="same", user=user)
        assert resp.status == 201, body
    resp, data = await post(conforming_client, {"workflow": WF}, key="same", user="alice")
    assert_error(resp, data, 422, "idempotency_key_reuse")


@pytest.mark.asyncio
async def test_key_and_record_expire_after_24h(conforming_client, v2_app, v2_dirs, queue, clock):
    resp, first = await post(conforming_client, {"workflow": WF}, key="k")
    clock["now"] = T0 + timedelta(hours=24) - timedelta(seconds=1)
    resp, data = await post(conforming_client, {"workflow": WF}, key="k")
    assert_error(resp, data, 422, "idempotency_key_reuse")
    assert first["id"] in store(v2_app).records
    clock["now"] = T0 + timedelta(hours=24, seconds=1)
    resp, body = await post(conforming_client, {"workflow": WF}, key="k")
    assert resp.status == 201, body
    assert list(store(v2_app).records) == [body["id"]]


@pytest.mark.asyncio
async def test_key_capacity(conforming_client, v2_dirs, queue, clock, monkeypatch):
    monkeypatch.setattr(jobs, "MAX_KEYS", 3)
    for i in range(3):
        clock["now"] = T0 + timedelta(seconds=10 * i)
        resp, body = await post(conforming_client, {"workflow": WF}, key=f"k{i}")
        assert resp.status == 201, body
    clock["now"] = T0 + timedelta(hours=1, milliseconds=500)
    resp, data = await post(conforming_client, {"workflow": WF}, key="k3")
    assert_error(resp, data, 429, "rate_limited")
    retry = resp.headers["Retry-After"]
    assert retry.isdigit() and int(retry) == math.ceil(timedelta(hours=23, milliseconds=-500).total_seconds())
    resp, data = await post(conforming_client, {"workflow": WF}, key="k0")
    assert_error(resp, data, 422, "idempotency_key_reuse")
    resp, body = await post(conforming_client, {"workflow": WF})
    assert resp.status == 201, body
    clock["now"] = T0 + timedelta(hours=24)
    resp, body = await post(conforming_client, {"workflow": WF}, key="k3")
    assert resp.status == 201, body


# job store


def test_record_cap_evicts_oldest_terminal_first(monkeypatch):
    monkeypatch.setattr(jobs, "MAX_RECORDS", 3)
    store = jobs.JobStore()
    records = [jobs.JobRecord(f"j{i}", "u", T0 + timedelta(seconds=i), {}, None, None) for i in range(5)]
    active = {"j0", "j2"}
    for record in records[:3]:
        store.add(record, lambda: active)
    store.add(records[3], lambda: active)
    assert list(store.records) == ["j0", "j2", "j3"]
    active = {"j0", "j2", "j3"}
    store.add(records[4], lambda: active)
    assert list(store.records) == ["j2", "j3", "j4"]


def test_store_never_evicts_live_keys():
    store = jobs.JobStore()
    store.claim("u", "a", T0)
    store.claim("u", "b", T0 + timedelta(hours=1))
    store.prune(T0 + timedelta(hours=24))
    assert list(store.idempotency) == [("u", "b")]
    store.prune(T0 + timedelta(hours=24, minutes=59))
    assert list(store.idempotency) == [("u", "b")]


# getAsset job_metadata seam


@pytest.mark.asyncio
async def test_get_asset_shows_owned_job_metadata(conforming_client, v2_dirs, queue, fake_server):
    fake_server.user_manager.multi_user = True
    fake_server.user_manager.users |= {"alice", "bob"}
    resp, job = await post(conforming_client, {"workflow": WF, "metadata": {"client": "acme"}}, user="alice")
    out = v2_dirs / "output" / "o.png"
    out.write_bytes(PNG + b"job")
    ref_id = register_executed_output(str(out), job_id=job["id"]).id
    for user, expected in (("alice", {"client": "acme"}), ("bob", None)):
        resp = await conforming_client.get(f"/api/v2/assets/{ref_id}", headers={"comfy-user": user})
        body = await resp.json()
        assert resp.status == 200, body
        validate(response_schema("getAsset", 200), body)
        assert body["job_id"] == job["id"] and body.get("job_metadata") == expected
    resp = await conforming_client.get(f"/api/v2/assets/{ref_id}", headers={"comfy-user": "mallory"})
    assert resp.status == 403
