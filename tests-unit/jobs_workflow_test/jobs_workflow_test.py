"""
Tests for the workflow payload on queue items in the jobs API.

GET /api/jobs/{job_id} must return the workflow for a running or pending
job (so the frontend can open it in a new tab before it finishes), while
GET /api/jobs (the list endpoint) keeps its lightweight shape.
"""

import pytest

from comfy_execution.jobs import (
    JobStatus,
    get_all_jobs,
    get_job,
    normalize_queue_item,
)

_RUNNING_ID = "aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa"
_PENDING_ID = "bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb"
_HISTORY_ID = "cccccccc-cccc-4ccc-cccc-cccccccccccc"
_MISSING_ID = "ffffffff-ffff-4fff-ffff-ffffffffffff"

_PROMPT = {"1": {"class_type": "KSampler", "inputs": {"seed": 1}}}
_WORKFLOW_JSON = {"id": "wf-abc", "nodes": [{"id": 1, "type": "KSampler"}], "links": []}


def make_extra_data():
    """extra_data shaped like a real submission, with the embedded workflow JSON."""
    return {
        "client_id": "client-1",
        "create_time": 1700000000000,
        "extra_pnginfo": {"workflow": _WORKFLOW_JSON},
    }


def make_queue_item(prompt_id, number=0):
    """Queue tuple with sensitive data already stripped (5 elements)."""
    return (number, prompt_id, _PROMPT, make_extra_data(), ["9"])


def make_history_item(prompt_id, number=0):
    """History entry for a job that completed successfully with no outputs."""
    return {
        "prompt": make_queue_item(prompt_id, number),
        "outputs": {},
        "status": {"status_str": "success", "completed": True, "messages": []},
    }


@pytest.fixture
def running():
    """A single running queue item."""
    return [make_queue_item(_RUNNING_ID, 0)]


@pytest.fixture
def queued():
    """A single pending queue item."""
    return [make_queue_item(_PENDING_ID, 1)]


@pytest.fixture
def history():
    """A single completed history entry."""
    return {_HISTORY_ID: make_history_item(_HISTORY_ID, 2)}


def assert_workflow_payload(job):
    """Assert the job carries the full workflow payload the frontend expects."""
    workflow = job["workflow"]
    assert workflow["prompt"] == _PROMPT
    assert workflow["extra_data"] == make_extra_data()
    # The frontend reads the graph from exactly this path.
    assert workflow["extra_data"]["extra_pnginfo"]["workflow"] == _WORKFLOW_JSON


class TestNormalizeQueueItem:
    def test_default_omits_workflow(self):
        """The default call keeps the lightweight list shape (no workflow key)."""
        job = normalize_queue_item(make_queue_item(_PENDING_ID), JobStatus.PENDING)
        assert "workflow" not in job
        assert job["id"] == _PENDING_ID
        assert job["status"] == JobStatus.PENDING
        assert job["workflow_id"] == _WORKFLOW_JSON["id"]

    def test_include_workflow_adds_prompt_and_extra_data(self):
        """include_workflow=True adds workflow.prompt and workflow.extra_data."""
        job = normalize_queue_item(
            make_queue_item(_PENDING_ID), JobStatus.PENDING, include_workflow=True
        )
        assert_workflow_payload(job)

    def test_include_workflow_keeps_existing_fields(self):
        """Adding the workflow does not alter any of the existing job fields."""
        base = normalize_queue_item(make_queue_item(_RUNNING_ID), JobStatus.IN_PROGRESS)
        with_workflow = normalize_queue_item(
            make_queue_item(_RUNNING_ID), JobStatus.IN_PROGRESS, include_workflow=True
        )
        without_workflow = {k: v for k, v in with_workflow.items() if k != "workflow"}
        assert without_workflow == base


class TestGetJob:
    def test_running_job_includes_workflow(self, running, queued, history):
        """A running job's detail includes its workflow."""
        job = get_job(_RUNNING_ID, running, queued, history)
        assert job["status"] == JobStatus.IN_PROGRESS
        assert_workflow_payload(job)

    def test_pending_job_includes_workflow(self, running, queued, history):
        """A pending job's detail includes its workflow."""
        job = get_job(_PENDING_ID, running, queued, history)
        assert job["status"] == JobStatus.PENDING
        assert_workflow_payload(job)

    def test_history_job_still_includes_workflow(self, running, queued, history):
        """A completed job's detail still includes its workflow (unchanged)."""
        job = get_job(_HISTORY_ID, running, queued, history)
        assert job["status"] == JobStatus.COMPLETED
        assert_workflow_payload(job)

    def test_unknown_job_is_none(self, running, queued, history):
        """An id present nowhere returns None."""
        assert get_job(_MISSING_ID, running, queued, history) is None


class TestGetAllJobs:
    def test_list_shape_has_no_workflow(self, running, queued, history):
        """The list endpoint never includes the workflow payload."""
        jobs, total = get_all_jobs(running, queued, history)
        assert total == 3
        assert all("workflow" not in job for job in jobs)
        assert all(job["workflow_id"] == _WORKFLOW_JSON["id"] for job in jobs)
