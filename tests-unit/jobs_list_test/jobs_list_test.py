"""Tests for the ``ids`` batch filter on the jobs listing endpoint.

Covers both layers:

* the pure ``comfy_execution.jobs.get_all_jobs`` filtering logic (the ``ids``
  argument narrows the result, composes with ``status_filter``, and silently
  ignores ids that match nothing), and

* the HTTP contract of ``GET /api/jobs`` for the ``ids`` query parameter
  (a valid set narrows the response, an oversized set or a malformed id is
  rejected with 400).

The HTTP layer is exercised against a small aiohttp app whose handler calls the
SAME ``parse_ids_filter`` that ``server.py`` uses (no hand-copied wiring, so it
cannot drift), driven by a fake queue. This keeps the test free of the heavy
ComfyUI runtime (torch, nodes, ...) while still testing the real parsing
contract.
"""

import pytest
from aiohttp import web

from comfy_execution.jobs import (
    JobStatus,
    JobIdsFilterError,
    MAX_JOB_IDS_FILTER,
    get_all_jobs,
    parse_ids_filter,
)

# Canonical UUID ids (the endpoint validates UUID format).
_UUID_A = "aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa"
_UUID_B = "bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb"
_UUID_C = "cccccccc-cccc-4ccc-cccc-cccccccccccc"
_UUID_MISSING = "ffffffff-ffff-4fff-ffff-ffffffffffff"


def make_queue_item(prompt_id, priority=0):
    """Build a queue tuple shaped like the real ones (5 elements, id at index 1)."""
    return (priority, prompt_id, {}, {}, [])


def make_history_item(status_str="success"):
    """Build a history item dict shaped like the real ones."""
    return {
        "prompt": (0, "", {}, {}, []),
        "status": {"status_str": status_str, "messages": []},
        "outputs": {},
    }


# ---------------------------------------------------------------------------
# Pure get_all_jobs filtering logic
# ---------------------------------------------------------------------------


def test_ids_filter_returns_only_requested():
    running = [make_queue_item(_UUID_A)]
    queued = [make_queue_item(_UUID_B)]
    history = {_UUID_C: make_history_item()}

    jobs, total = get_all_jobs(running, queued, history, ids=[_UUID_A, _UUID_C])

    returned = {j["id"] for j in jobs}
    assert returned == {_UUID_A, _UUID_C}
    assert total == 2
    assert _UUID_B not in returned


def test_ids_filter_absent_returns_all():
    running = [make_queue_item(_UUID_A)]
    queued = [make_queue_item(_UUID_B)]
    history = {_UUID_C: make_history_item()}

    jobs, total = get_all_jobs(running, queued, history)

    assert {j["id"] for j in jobs} == {_UUID_A, _UUID_B, _UUID_C}
    assert total == 3


def test_ids_filter_empty_list_returns_none():
    """A present-but-empty ids list is a zero-match filter, not "no filter".

    ``None`` means "no id filter"; ``[]`` means "restrict to nothing".
    """
    running = [make_queue_item(_UUID_A)]
    queued = [make_queue_item(_UUID_B)]

    jobs, total = get_all_jobs(running, queued, {}, ids=[])

    assert jobs == []
    assert total == 0


def test_ids_filter_unknown_id_silently_absent():
    """An id that matches nothing is simply not present (no error)."""
    running = [make_queue_item(_UUID_A)]

    jobs, total = get_all_jobs(running, [], {}, ids=[_UUID_A, _UUID_MISSING])

    assert {j["id"] for j in jobs} == {_UUID_A}
    assert total == 1


def test_ids_filter_composes_with_status():
    """ids only narrows; it composes with the status filter."""
    running = [make_queue_item(_UUID_A)]
    queued = [make_queue_item(_UUID_B)]
    history = {_UUID_C: make_history_item()}

    # Request A and C by id, but restrict to in_progress only -> just A.
    jobs, total = get_all_jobs(
        running, queued, history,
        status_filter=[JobStatus.IN_PROGRESS],
        ids=[_UUID_A, _UUID_C],
    )

    assert {j["id"] for j in jobs} == {_UUID_A}
    assert total == 1


# ---------------------------------------------------------------------------
# parse_ids_filter -- the shared parsing/validation (server.py + these tests)
# ---------------------------------------------------------------------------


def test_parse_ids_absent_is_none():
    assert parse_ids_filter(None) is None


def test_parse_ids_present_but_empty_is_empty_list():
    # `?ids=` and `?ids=,,` parse to [] -> zero-match filter, not None.
    assert parse_ids_filter("") == []
    assert parse_ids_filter(",,") == []


def test_parse_ids_dedupes_preserving_order():
    assert parse_ids_filter(f"{_UUID_A},{_UUID_B},{_UUID_A}") == [_UUID_A, _UUID_B]


def test_parse_ids_cap_counts_distinct_not_duplicates():
    # A small distinct set repeated far past the cap is still under it.
    repeated = ",".join([_UUID_A, _UUID_B] * MAX_JOB_IDS_FILTER)
    assert parse_ids_filter(repeated) == [_UUID_A, _UUID_B]
    # But more than MAX distinct ids is rejected.
    distinct = ",".join(
        f"{i:08d}-0000-4000-8000-000000000000" for i in range(MAX_JOB_IDS_FILTER + 1)
    )
    with pytest.raises(JobIdsFilterError):
        parse_ids_filter(distinct)


def test_parse_ids_invalid_raises_with_payload():
    with pytest.raises(JobIdsFilterError) as exc:
        parse_ids_filter(f"{_UUID_A},not-a-uuid")
    assert "not-a-uuid" in exc.value.payload["invalid_ids"]


# ---------------------------------------------------------------------------
# HTTP contract for the ids query parameter
# ---------------------------------------------------------------------------


class FakePromptQueue:
    """Minimal stand-in exposing the accessors get_jobs uses."""

    def __init__(self, running=None, queued=None, history=None):
        self._running = list(running or [])
        self._queued = list(queued or [])
        self._history = dict(history or {})

    def get_current_queue_volatile(self):
        return (list(self._running), list(self._queued))

    def get_history(self):
        return dict(self._history)


def make_app(prompt_queue):
    """Build an aiohttp app whose handler calls the REAL parse_ids_filter.

    No hand-copied parsing wiring, so this test cannot stay green while the
    shipped parsing in server.py regresses -- both go through parse_ids_filter.
    """

    async def get_jobs(request):
        try:
            ids_filter = parse_ids_filter(request.rel_url.query.get('ids'))
        except JobIdsFilterError as e:
            return web.json_response(e.payload, status=400)

        running, queued = prompt_queue.get_current_queue_volatile()
        history = prompt_queue.get_history()

        jobs, total = get_all_jobs(running, queued, history, ids=ids_filter)

        return web.json_response({
            'jobs': jobs,
            'pagination': {'total': total},
        })

    app = web.Application()
    app.router.add_get('/api/jobs', get_jobs)
    return app


@pytest.fixture
def queue():
    return FakePromptQueue(
        running=[make_queue_item(_UUID_A)],
        queued=[make_queue_item(_UUID_B)],
        history={_UUID_C: make_history_item()},
    )


@pytest.mark.asyncio
async def test_http_ids_filter_narrows(aiohttp_client, queue):
    client = await aiohttp_client(make_app(queue))

    resp = await client.get(f"/api/jobs?ids={_UUID_A},{_UUID_C}")
    assert resp.status == 200
    body = await resp.json()
    assert {j["id"] for j in body["jobs"]} == {_UUID_A, _UUID_C}


@pytest.mark.asyncio
async def test_http_ids_unknown_id_is_not_an_error(aiohttp_client, queue):
    client = await aiohttp_client(make_app(queue))

    resp = await client.get(f"/api/jobs?ids={_UUID_A},{_UUID_MISSING}")
    assert resp.status == 200
    body = await resp.json()
    assert {j["id"] for j in body["jobs"]} == {_UUID_A}


@pytest.mark.asyncio
async def test_http_ids_over_limit_returns_400(aiohttp_client, queue):
    client = await aiohttp_client(make_app(queue))

    # Distinct ids past the cap. (Repeats of one id are de-duped and would NOT
    # trip the cap -- see test_parse_ids_cap_counts_distinct_not_duplicates.)
    too_many = ",".join(
        f"{i:08d}-0000-4000-8000-000000000000" for i in range(MAX_JOB_IDS_FILTER + 1)
    )
    resp = await client.get(f"/api/jobs?ids={too_many}")
    assert resp.status == 400


@pytest.mark.asyncio
async def test_http_ids_invalid_id_returns_400(aiohttp_client, queue):
    client = await aiohttp_client(make_app(queue))

    resp = await client.get(f"/api/jobs?ids={_UUID_A},not-a-uuid")
    assert resp.status == 400
    body = await resp.json()
    assert "not-a-uuid" in body["invalid_ids"]


@pytest.mark.asyncio
async def test_http_ids_absent_returns_all(aiohttp_client, queue):
    client = await aiohttp_client(make_app(queue))

    resp = await client.get("/api/jobs")
    assert resp.status == 200
    body = await resp.json()
    assert {j["id"] for j in body["jobs"]} == {_UUID_A, _UUID_B, _UUID_C}


@pytest.mark.asyncio
async def test_http_ids_present_but_empty_returns_none(aiohttp_client, queue):
    """`?ids=` (present but empty) is a zero-match filter, not "return all"."""
    client = await aiohttp_client(make_app(queue))

    resp = await client.get("/api/jobs?ids=")
    assert resp.status == 200
    body = await resp.json()
    assert body["jobs"] == []
