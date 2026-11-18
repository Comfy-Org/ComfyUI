import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_prefetch as model_prefetch  # noqa: E402


@pytest.fixture
def steps(monkeypatch):
    ran = []

    def step(name, error=None):
        def run():
            ran.append(name)
            if error is not None:
                raise error
        return run

    def install(queues_error=None, buffers_error=None, watermark_error=None):
        monkeypatch.setattr(model_prefetch, "cleanup_prefetch_queues", step("queues", queues_error))
        monkeypatch.setattr(model_prefetch.comfy.model_management, "reset_cast_buffers", step("buffers", buffers_error))
        monkeypatch.setattr(model_prefetch.comfy_aimdo.model_vbar, "vbars_reset_watermark_limits", step("watermark", watermark_error))
        return ran

    return install


def test_cleanup_runs_every_step(steps):
    ran = steps()
    model_prefetch.cleanup_after_node(node_failed=False)
    assert ran == ["queues", "buffers", "watermark"]


def test_failed_queue_cleanup_still_resets_buffers_and_watermarks(steps):
    # CORE-694: a raising cleanup_prefetch_queues used to skip both resets, so
    # the next node on the worker started from the failed node's state.
    ran = steps(queues_error=RuntimeError("unpin failed"))
    with pytest.raises(RuntimeError, match="unpin failed"):
        model_prefetch.cleanup_after_node(node_failed=False)
    assert ran == ["queues", "buffers", "watermark"]


def test_cleanup_error_does_not_replace_the_node_error(steps):
    # CORE-694: raised from the finally, the cleanup error replaced the one in
    # flight, so a cancel became an execution error and an OOM stopped being
    # an OOM. With a node error in flight it is logged, not raised.
    ran = steps(queues_error=RuntimeError("unpin failed"))
    model_prefetch.cleanup_after_node(node_failed=True)
    assert ran == ["queues", "buffers", "watermark"]


def test_first_cleanup_error_wins(steps):
    steps(queues_error=RuntimeError("first"), watermark_error=ValueError("second"))
    with pytest.raises(RuntimeError, match="first"):
        model_prefetch.cleanup_after_node(node_failed=False)
