import contextvars
from typing import Optional, NamedTuple

class ExecutionContext(NamedTuple):
    """
    Context information about the currently executing node.

    Attributes:
        node_id: The ID of the currently executing node
        list_index: The index in a list being processed (for operations on batches/lists)
    """
    prompt_id: str
    node_id: str
    list_index: Optional[int]

current_executing_context: contextvars.ContextVar[Optional[ExecutionContext]] = contextvars.ContextVar("current_executing_context", default=None)

# Set for the whole execution of a prompt, including the async tasks it spawns. Several prompts can
# be in flight at once, so per-prompt state (caches, progress) is looked up through this.
current_prompt_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("current_prompt_id", default=None)

def get_executing_context() -> Optional[ExecutionContext]:
    return current_executing_context.get(None)

class CurrentNodeContext:
    """
    Context manager for setting the current executing node context.

    Sets the current_executing_context on enter and resets it on exit.

    Example:
        with CurrentNodeContext(node_id="123", list_index=0):
            # Code that should run with the current node context set
            process_image()
    """
    def __init__(self, prompt_id: str, node_id: str, list_index: Optional[int] = None):
        self.context = ExecutionContext(
            prompt_id= prompt_id,
            node_id= node_id,
            list_index= list_index
        )
        self.token = None

    def __enter__(self):
        self.token = current_executing_context.set(self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token is not None:
            current_executing_context.reset(self.token)
