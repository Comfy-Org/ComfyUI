import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from pydantic import ValidationError

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_api_nodes import nodes_kling
from comfy_api_nodes.util import client
from comfy_api_nodes.util.common_exceptions import LocalNetworkError, ProcessingInterrupted


@pytest.fixture
def polling(monkeypatch):
    request = AsyncMock()
    monkeypatch.setattr(client, "sync_op_raw", request)
    monkeypatch.setattr(client, "sleep_with_interrupt", AsyncMock())
    monkeypatch.setattr(client, "_display_time_progress", lambda *a, **kw: None)
    monkeypatch.setattr(client, "is_processing_interrupted", lambda: False)
    return request


async def poll(**kwargs):
    return await client.poll_op_raw(
        type("Node", (), {}),
        client.ApiEndpoint("/proxy/partner/status/task-1"),
        status_extractor=lambda response: response["status"],
        max_poll_attempts=2,
        poll_interval=7,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_poll_budget_has_distinct_exception_and_preserves_message(polling):
    polling.return_value = {"status": "processing"}
    with pytest.raises(client.PollingTimeoutError) as error:
        await poll()
    assert str(error.value) == (
        "Polling aborted due to error: Polling timed out after 2 non-queued attempts "
        "(~14s of active polling)."
    )
    assert polling.await_count == 2


@pytest.mark.asyncio
async def test_budget_timeout_keeps_existing_cancel_callback(polling):
    polling.side_effect = [{"status": "processing"}, {"status": "processing"}, {}]
    endpoint = client.ApiEndpoint("/proxy/partner/cancel/task-1", method="POST")
    with pytest.raises(client.PollingTimeoutError):
        await poll(cancel_endpoint=endpoint)
    assert polling.await_count == 3
    assert polling.call_args.args[1] is endpoint
    assert polling.call_args.kwargs["max_retries"] == 0


@pytest.mark.asyncio
async def test_last_budget_poll_can_complete(polling):
    polling.side_effect = [{"status": "processing"}, {"status": "succeed"}]
    assert await poll() == {"status": "succeed"}
    assert polling.await_count == 2


@pytest.mark.asyncio
async def test_queued_polls_do_not_consume_budget(polling):
    polling.side_effect = [{"status": "submitted"}] * 3 + [{"status": "succeed"}]
    assert await poll() == {"status": "succeed"}
    assert polling.await_count == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError("request timeout"), Exception("401 Unauthorized")])
async def test_poll_request_errors_are_not_budget_timeouts(polling, error):
    polling.side_effect = error
    with pytest.raises(Exception) as caught:
        await poll()
    assert not isinstance(caught.value, client.PollingTimeoutError)
    assert str(error) in str(caught.value)
    assert polling.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ProcessingInterrupted("stopped"), asyncio.CancelledError(), LocalNetworkError("offline")])
async def test_poll_interrupt_and_network_error_types_are_preserved(polling, error):
    polling.side_effect = error
    with pytest.raises(type(error)) as caught:
        await poll()
    assert caught.value is error


@pytest.mark.asyncio
async def test_provider_failure_is_not_a_budget_timeout(polling):
    polling.return_value = {"status": "failed"}
    with pytest.raises(Exception, match="Task failed") as caught:
        await poll()
    assert not isinstance(caught.value, client.PollingTimeoutError)
    assert polling.await_count == 1


@pytest.fixture
def avatar(monkeypatch):
    for name in ("validate_image_dimensions", "validate_image_aspect_ratio", "validate_audio_duration"):
        monkeypatch.setattr(nodes_kling, name, lambda *a, **kw: None)
    monkeypatch.setattr(nodes_kling, "upload_image_to_comfyapi", AsyncMock(return_value="https://example.invalid/image"))
    monkeypatch.setattr(nodes_kling, "upload_audio_to_comfyapi", AsyncMock(return_value="https://example.invalid/audio"))
    submit = AsyncMock(return_value=nodes_kling.TaskStatusResponse(code=0, data={"task_id": "avatar-task-1"}))
    poller = AsyncMock(side_effect=client.PollingTimeoutError("budget exhausted"))
    download = AsyncMock(return_value=object())
    monkeypatch.setattr(nodes_kling, "sync_op", submit)
    monkeypatch.setattr(nodes_kling, "poll_op", poller)
    monkeypatch.setattr(nodes_kling, "download_url_to_video_output", download)
    return SimpleNamespace(submit=submit, poll=poller, download=download)


async def run_avatar():
    return await nodes_kling.KlingAvatarNode.execute(object(), object(), "std", 0)


@pytest.mark.asyncio
async def test_avatar_timeout_preserves_own_task_without_resubmitting(avatar):
    with pytest.raises(RuntimeError) as caught:
        await run_avatar()
    assert "Task ID: avatar-task-1." in str(caught.value)
    assert "may still complete" in str(caught.value)
    assert "Contact support" in str(caught.value)
    assert "another charge" in str(caught.value)
    avatar.submit.assert_awaited_once()
    avatar.poll.assert_awaited_once()
    assert avatar.poll.call_args.args[1].path.endswith("/avatar-task-1")
    assert avatar.poll.call_args.kwargs["max_poll_attempts"] == 800
    avatar.download.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [None, {}, {"task_id": ""}, {"task_id": "https://example.invalid/private?token=secret"},
     {"task_id": "task/other"}, {"task_id": "task?query"}, {"task_id": "task\nInjected"},
     {"task_id": "task%2Fother"}, {"task_id": "a" * 129}],
)
async def test_avatar_rejects_missing_or_unsafe_task_id_before_polling(avatar, data):
    avatar.submit.return_value = nodes_kling.TaskStatusResponse(code=0, data=data)
    with pytest.raises(RuntimeError) as caught:
        await run_avatar()
    assert str(caught.value) == "Kling did not return a valid task ID."
    avatar.submit.assert_awaited_once()
    avatar.poll.assert_not_awaited()
    avatar.download.assert_not_awaited()


def test_kling_response_model_rejects_numeric_task_id():
    with pytest.raises(ValidationError):
        nodes_kling.TaskStatusResponse(code=0, data={"task_id": 123})


@pytest.mark.asyncio
async def test_avatar_rejects_nonstring_task_id_when_model_validation_is_bypassed(avatar):
    avatar.submit.return_value = nodes_kling.TaskStatusResponse.model_construct(
        code=0, data=SimpleNamespace(task_id=123)
    )
    with pytest.raises(RuntimeError, match="Kling did not return a valid task ID"):
        await run_avatar()
    avatar.poll.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError("network timeout"), ProcessingInterrupted("stopped"), Exception("404 task not found")])
async def test_avatar_does_not_relabel_other_failures_as_poll_timeout(avatar, error):
    avatar.poll.side_effect = error
    with pytest.raises(type(error)) as caught:
        await run_avatar()
    assert caught.value is error
    avatar.submit.assert_awaited_once()
    avatar.download.assert_not_awaited()


@pytest.mark.asyncio
async def test_avatar_completion_still_downloads_result(avatar):
    avatar.poll.side_effect = None
    avatar.poll.return_value = nodes_kling.TaskStatusResponse(
        code=0, data={"task_status": "succeed", "task_result": {"videos": [{"url": "https://example.invalid/video"}]}}
    )
    output = await run_avatar()
    assert output.result == (avatar.download.return_value,)
    avatar.submit.assert_awaited_once()
    avatar.download.assert_awaited_once_with("https://example.invalid/video")
