from comfy_api._exceptions import InterruptProcessingException
from comfy_api.internal.singleton import ProxiedSingleton
from comfy_execution.progress import NodeState, get_progress_state
from comfy_execution.utils import ExecutionContext, get_executing_context


def _is_running(context: ExecutionContext) -> bool:
    progress = get_progress_state()
    node = progress.nodes.get(context.node_id)
    return (
        progress.prompt_id == context.prompt_id
        and node is not None
        and node["state"] == NodeState.Running
    )


def _running_node() -> ExecutionContext:
    """The context of the node that is running where this is called, for the groups whose
    calls act on the running node. Not a call of the API: the groups call it from their own
    methods, which run in ComfyUI's process, where ComfyUI keeps this context."""
    context = get_executing_context()
    if context is None:
        raise RuntimeError("This call must be made from a node while it runs.")
    if not _is_running(context):
        raise RuntimeError("The node is no longer running in this prompt.")
    return context


class Execution(ProxiedSingleton):
    async def check_interrupted(self) -> None:
        """Raise `InterruptProcessingException` if the user cancelled the prompt.

        In a running node raising clears the cancel, so let the exception reach ComfyUI.
        Elsewhere (a route handler, a thread the pack started, a task left over from a node that
        has finished) the cancel stays pending. A node that has finished counts as running again
        once progress is reported for it through an older API.
        """
        from comfy.model_management import (
            processing_interrupted,
            throw_exception_if_processing_interrupted,
        )

        context = get_executing_context()
        if context is not None and _is_running(context):
            throw_exception_if_processing_interrupted()
        elif processing_interrupted():
            raise InterruptProcessingException()

    async def is_interrupted(self) -> bool:
        """Return True if the user cancelled the prompt. The cancel stays pending."""
        from comfy.model_management import processing_interrupted

        return processing_interrupted()
