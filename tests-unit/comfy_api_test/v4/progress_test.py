import pytest

from comfy.cli_args import args
from comfy.model_management import interrupt_current_processing
from comfy_api import v4
from comfy_execution import progress
from comfy_execution.progress import ProgressRegistry
from comfy_execution.utils import CurrentNodeContext
from protocol import BinaryEventTypes


def progress_sent_to_tab(server):
    return [
        (data["nodes"]["v4"]["value"], data["nodes"]["v4"]["max"])
        for event, data, sid in server.sent
        if event == "progress_state" and sid == "tab" and "v4" in data["nodes"]
    ]


def updates_sent(server):
    """The node's own updates, without the start and finish ComfyUI sends around them."""
    return progress_sent_to_tab(server)[1:-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_update_reports_the_running_node_to_the_client_that_queued_it(
    run_node, synchronous
):
    executor, server = await run_node("{call}.progress.update(1, 2)", synchronous)

    assert executor.success, executor.status_messages
    assert (1, 2) in progress_sent_to_tab(server)


@pytest.mark.asyncio
async def test_update_from_a_worker_thread_reports_the_node_that_started_it(run_node):
    executor, server = await run_node(
        "loop = asyncio.get_running_loop()\n"
        "def report():\n"
        "    update = eng.progress.update(1, 2)\n"
        "    asyncio.run_coroutine_threadsafe(update, loop).result()\n"
        "await asyncio.to_thread(report)"
    )

    assert executor.success, executor.status_messages
    assert (1, 2) in progress_sent_to_tab(server)


@pytest.mark.asyncio
async def test_update_shows_the_first_rgba_preview_image_as_a_clamped_rgb_jpeg(
    run_node, monkeypatch
):
    monkeypatch.setattr(args, "preview_size", 64)

    executor, server = await run_node(
        "first = torch.tensor([2.0, -1.0, 0.5, 0.0]).expand(4, 4, 4)\n"
        "preview = torch.stack([first, torch.zeros(4, 4, 4)])\n"
        "{call}.progress.update(1, 2, preview)"
    )

    assert executor.success, executor.status_messages
    [((image_format, image, max_size), metadata)] = [
        data
        for event, data, sid in server.sent
        if event == BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA and sid == "tab"
    ]
    assert (image_format, image.mode, image.size, max_size) == (
        "JPEG",
        "RGB",
        (4, 4),
        64,
    )
    assert image.getpixel((0, 0)) == (255, 0, 127)
    assert metadata["node_id"] == "v4"


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_progress_text_reaches_the_running_node_of_the_client_that_queued_it(
    run_node, synchronous
):
    executor, server = await run_node(
        '{call}.ui.progress_text("half way")', synchronous
    )

    assert executor.success, executor.status_messages
    assert (BinaryEventTypes.TEXT, b"\x00\x00\x00\x02v4half way", "tab") in server.sent


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize("client_id", [None, ""])
async def test_progress_text_without_a_client_does_not_broadcast(
    run_node, synchronous, client_id
):
    executor, server = await run_node(
        '{call}.ui.progress_text("private prompt text")',
        synchronous,
        client_id=client_id,
    )

    assert executor.success, executor.status_messages
    assert all(event != BinaryEventTypes.TEXT for event, _, _ in server.sent)


@pytest.mark.asyncio
@pytest.mark.parametrize("later_prompt", [None, "another-prompt", "v4-node"])
@pytest.mark.parametrize("operation", ["progress", "text"])
async def test_retained_context_cannot_report_progress_after_its_node_finishes(
    run_node, monkeypatch, later_prompt, operation
):
    executor, server = await run_node("pass")
    assert executor.success, executor.status_messages
    if later_prompt:
        registry = ProgressRegistry(later_prompt, None)
        monkeypatch.setattr(progress, "global_progress_registry", registry)
        if later_prompt != "v4-node":
            registry.start_progress("v4")
        server.client_id = "another-tab"
    sent = list(server.sent)
    interrupt_current_processing()

    with CurrentNodeContext("v4-node", "v4"):
        with pytest.raises(RuntimeError):
            if operation == "progress":
                await v4.eng.progress.update(1, 2)
            else:
                await v4.eng.ui.progress_text("late text")

    assert server.sent == sent
    assert await v4.eng.execution.is_interrupted()


@pytest.mark.asyncio
async def test_calls_about_the_running_node_fail_outside_one_and_keep_the_cancel():
    interrupt_current_processing()

    with pytest.raises(RuntimeError):
        await v4.eng.progress.update(1, 2)
    with pytest.raises(RuntimeError):
        await v4.eng.ui.progress_text("half way")
    assert await v4.eng.execution.is_interrupted()


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_many_small_updates_send_a_bounded_number_and_always_the_last(
    run_node, synchronous
):
    executor, server = await run_node(
        "for i in range(2000):\n    {call}.progress.update(i + 1, 2000)", synchronous
    )

    assert executor.success, executor.status_messages
    sent = progress_sent_to_tab(server)
    assert len(sent) <= 205 and sent[-1] == (2000, 2000)
    assert (1, 2000) in sent


@pytest.mark.asyncio
async def test_the_last_update_is_sent_even_within_half_a_percent_of_the_previous(
    run_node,
):
    executor, server = await run_node(
        "for i in range(1000):\n    {call}.progress.update(i + 1, 1000)"
    )

    assert executor.success, executor.status_messages
    assert updates_sent(server)[-1] == (1000, 1000)


@pytest.mark.asyncio
async def test_a_step_back_is_sent(run_node):
    executor, server = await run_node(
        "{call}.progress.update(10, 100)\n{call}.progress.update(5, 100)"
    )

    assert executor.success, executor.status_messages
    assert updates_sent(server) == [(10, 100), (5, 100)]


@pytest.mark.asyncio
async def test_steps_of_more_than_half_a_percent_are_all_sent(run_node):
    executor, server = await run_node(
        "for i in range(1, 50):\n    {call}.progress.update(6 * i, 1000)"
    )

    assert executor.success, executor.status_messages
    assert [value for value, _ in updates_sent(server)] == [6 * i for i in range(1, 50)]


@pytest.mark.asyncio
async def test_every_update_with_a_preview_is_sent(run_node):
    executor, server = await run_node(
        "image = torch.zeros(1, 4, 4, 3)\n"
        "for i in range(5):\n    {call}.progress.update(i + 1, 1000, image)"
    )

    assert executor.success, executor.status_messages
    previews = [
        event
        for event, _, _ in server.sent
        if event == BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA
    ]
    assert len(previews) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_update_reports_through_comfyuis_progress_hook_as_its_progress_bar_does(
    run_node, monkeypatch, synchronous
):
    import comfy.utils

    calls = []

    def hook(value, total, preview, **options):
        image_format, image, max_size = preview
        calls.append((value, total, (image_format, image.tobytes(), max_size), options))

    monkeypatch.setattr(comfy.utils, "PROGRESS_BAR_HOOK", hook)

    executor, server = await run_node(
        "from PIL import Image\n"
        "from comfy.cli_args import args\n"
        "image = ('JPEG', Image.new('RGB', (4, 4)), args.preview_size)\n"
        "comfy.utils.ProgressBar(2, node_id='v4').update_absolute(1, 2, image)\n"
        "{call}.progress.update(1, 2, torch.zeros(1, 4, 4, 3))",
        synchronous,
    )

    assert executor.success, executor.status_messages
    progress_bar, update = calls
    assert update == progress_bar


@pytest.mark.asyncio
async def test_a_repeated_value_is_not_sent_again(run_node):
    executor, server = await run_node(
        "{call}.progress.update(10, 100)\n{call}.progress.update(10, 100)"
    )

    assert executor.success, executor.status_messages
    assert updates_sent(server) == [(10, 100)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "update",
    ["float('nan'), 10", "5, float('nan')", "float('inf'), 10", "5, float('inf')"],
)
async def test_progress_that_is_not_a_finite_number_is_refused_and_nothing_is_sent(
    run_node, update
):
    executor, server = await run_node(
        "try:\n    {call}.progress.update(" + update + ")\n"
        "except ValueError as error:\n    assert 'finite' in str(error)\n"
        "else:\n    raise AssertionError('accepted')"
    )

    assert executor.success, executor.status_messages
    assert updates_sent(server) == []


@pytest.mark.asyncio
async def test_a_value_above_the_total_is_shown_as_the_total(run_node):
    executor, server = await run_node("{call}.progress.update(20, 10)")

    assert executor.success, executor.status_messages
    assert updates_sent(server) == [(10, 10)]
