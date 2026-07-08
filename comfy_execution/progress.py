from typing import TypedDict, Dict, Optional, Tuple
from typing_extensions import override
from PIL import Image
from enum import Enum
from abc import ABC
from tqdm import tqdm
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from comfy_execution.graph import DynamicPrompt
from protocol import BinaryEventTypes
from comfy_api import feature_flags

PreviewImageTuple = Tuple[str, Image.Image, Optional[int]]

class NodeState(Enum):
    Pending = "pending"
    Running = "running"
    Finished = "finished"
    Error = "error"


class NodeProgressState(TypedDict):
    """
    A class to represent the state of a node's progress.
    """

    state: NodeState
    value: float
    max: float


class ProgressHandler(ABC):
    """
    Abstract base class for progress handlers.
    Progress handlers receive progress updates and display them in various ways.
    """

    def __init__(self, name: str):
        self.name = name
        self.enabled = True

    def set_registry(self, registry: "ProgressRegistry"):
        pass

    def start_handler(self, node_id: str, state: NodeProgressState, prompt_id: str):
        """Called when a node starts processing"""
        pass

    def update_handler(
        self,
        node_id: str,
        value: float,
        max_value: float,
        state: NodeProgressState,
        prompt_id: str,
        image: PreviewImageTuple | None = None,
    ):
        """Called when a node's progress is updated"""
        pass

    def finish_handler(self, node_id: str, state: NodeProgressState, prompt_id: str):
        """Called when a node finishes processing"""
        pass

    def reset(self):
        """Called when the progress registry is reset"""
        pass

    def enable(self):
        """Enable this handler"""
        self.enabled = True

    def disable(self):
        """Disable this handler"""
        self.enabled = False


class CLIProgressHandler(ProgressHandler):
    """
    Handler that displays progress using tqdm progress bars in the CLI.
    """

    def __init__(self):
        super().__init__("cli")
        self.progress_bars: Dict[str, tqdm] = {}

    @override
    def start_handler(self, node_id: str, state: NodeProgressState, prompt_id: str):
        # Create a new tqdm progress bar
        if node_id not in self.progress_bars:
            self.progress_bars[node_id] = tqdm(
                total=state["max"],
                desc=f"Node {node_id}",
                unit="steps",
                leave=True,
                position=len(self.progress_bars),
            )

    @override
    def update_handler(
        self,
        node_id: str,
        value: float,
        max_value: float,
        state: NodeProgressState,
        prompt_id: str,
        image: PreviewImageTuple | None = None,
    ):
        # Handle case where start_handler wasn't called
        if node_id not in self.progress_bars:
            self.progress_bars[node_id] = tqdm(
                total=max_value,
                desc=f"Node {node_id}",
                unit="steps",
                leave=True,
                position=len(self.progress_bars),
            )
            self.progress_bars[node_id].update(value)
        else:
            # Update existing progress bar
            if max_value != self.progress_bars[node_id].total:
                self.progress_bars[node_id].total = max_value
            # Calculate the update amount (difference from current position)
            current_position = self.progress_bars[node_id].n
            update_amount = value - current_position
            if update_amount > 0:
                self.progress_bars[node_id].update(update_amount)

    @override
    def finish_handler(self, node_id: str, state: NodeProgressState, prompt_id: str):
        # Complete and close the progress bar if it exists
        if node_id in self.progress_bars:
            # Ensure the bar shows 100% completion
            remaining = state["max"] - self.progress_bars[node_id].n
            if remaining > 0:
                self.progress_bars[node_id].update(remaining)
            self.progress_bars[node_id].close()
            del self.progress_bars[node_id]

    @override
    def reset(self):
        # Close all progress bars
        for bar in self.progress_bars.values():
            bar.close()
        self.progress_bars.clear()


class WebUIProgressHandler(ProgressHandler):
    """
    Handler that sends progress updates to the WebUI via WebSockets.
    """

    def __init__(self, server_instance):
        super().__init__("webui")
        self.server_instance = server_instance

    def set_registry(self, registry: "ProgressRegistry"):
        self.registry = registry

    def _send_progress_state(self, prompt_id: str, nodes: Dict[str, NodeProgressState]):
        """Send the current progress state to the client"""
        if self.server_instance is None:
            return

        # Only send info for non-pending nodes
        active_nodes = {
            node_id: {
                "value": state["value"],
                "max": state["max"],
                "state": state["state"].value,
                "node_id": node_id,
                "prompt_id": prompt_id,
                "display_node_id": self.registry.dynprompt.get_display_node_id(node_id),
                "parent_node_id": self.registry.dynprompt.get_parent_node_id(node_id),
                "real_node_id": self.registry.dynprompt.get_real_node_id(node_id),
            }
            for node_id, state in nodes.items()
            if state["state"] != NodeState.Pending
        }

        # Send a combined progress_state message with all node states
        # Include client_id to ensure message is only sent to the initiating client
        self.server_instance.send_sync(
            "progress_state", {"prompt_id": prompt_id, "nodes": active_nodes}, self.server_instance.client_id
        )

    @override
    def start_handler(self, node_id: str, state: NodeProgressState, prompt_id: str):
        # Send progress state of all nodes
        if self.registry:
            self._send_progress_state(prompt_id, self.registry.nodes)

    @override
    def update_handler(
        self,
        node_id: str,
        value: float,
        max_value: float,
        state: NodeProgressState,
        prompt_id: str,
        image: PreviewImageTuple | None = None,
    ):
        # Send progress state of all nodes
        if self.registry:
            self._send_progress_state(prompt_id, self.registry.nodes)
        if image:
            # Only send new format if client supports it
            if feature_flags.supports_feature(
                self.server_instance.sockets_metadata,
                self.server_instance.client_id,
                "supports_preview_metadata",
            ):
                metadata = {
                    "node_id": node_id,
                    "prompt_id": prompt_id,
                    "display_node_id": self.registry.dynprompt.get_display_node_id(
                        node_id
                    ),
                    "parent_node_id": self.registry.dynprompt.get_parent_node_id(
                        node_id
                    ),
                    "real_node_id": self.registry.dynprompt.get_real_node_id(node_id),
                }
                self.server_instance.send_sync(
                    BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA,
                    (image, metadata),
                    self.server_instance.client_id,
                )

    @override
    def finish_handler(self, node_id: str, state: NodeProgressState, prompt_id: str):
        # Send progress state of all nodes
        if self.registry:
            self._send_progress_state(prompt_id, self.registry.nodes)

class ProgressRegistry:
    """
    Registry that maintains node progress state and notifies registered handlers.
    """

    def __init__(self, prompt_id: str, dynprompt: "DynamicPrompt"):
        self.prompt_id = prompt_id
        self.dynprompt = dynprompt
        self.nodes: Dict[str, NodeProgressState] = {}
        self.handlers: Dict[str, ProgressHandler] = {}

    def register_handler(self, handler: ProgressHandler) -> None:
        """Register a progress handler"""
        self.handlers[handler.name] = handler

    def unregister_handler(self, handler_name: str) -> None:
        """Unregister a progress handler"""
        if handler_name in self.handlers:
            # Allow handler to clean up resources
            self.handlers[handler_name].reset()
            del self.handlers[handler_name]

    def enable_handler(self, handler_name: str) -> None:
        """Enable a progress handler"""
        if handler_name in self.handlers:
            self.handlers[handler_name].enable()

    def disable_handler(self, handler_name: str) -> None:
        """Disable a progress handler"""
        if handler_name in self.handlers:
            self.handlers[handler_name].disable()

    def ensure_entry(self, node_id: str) -> NodeProgressState:
        """Ensure a node entry exists"""
        if node_id not in self.nodes:
            self.nodes[node_id] = NodeProgressState(
                state=NodeState.Pending, value=0, max=1
            )
        return self.nodes[node_id]

    def start_progress(self, node_id: str) -> None:
        """Start progress tracking for a node"""
        entry = self.ensure_entry(node_id)
        entry["state"] = NodeState.Running
        entry["value"] = 0.0
        entry["max"] = 1.0

        # Notify all enabled handlers
        for handler in self.handlers.values():
            if handler.enabled:
                handler.start_handler(node_id, entry, self.prompt_id)

    def update_progress(
        self, node_id: str, value: float, max_value: float, image: PreviewImageTuple | None = None
    ) -> None:
        """Update progress for a node"""
        entry = self.ensure_entry(node_id)
        entry["state"] = NodeState.Running
        entry["value"] = value
        entry["max"] = max_value

        # Notify all enabled handlers
        for handler in self.handlers.values():
            if handler.enabled:
                handler.update_handler(
                    node_id, value, max_value, entry, self.prompt_id, image
                )

    def finish_progress(self, node_id: str) -> None:
        """Finish progress tracking for a node"""
        entry = self.ensure_entry(node_id)
        entry["state"] = NodeState.Finished
        entry["value"] = entry["max"]

        # Notify all enabled handlers
        for handler in self.handlers.values():
            if handler.enabled:
                handler.finish_handler(node_id, entry, self.prompt_id)

    def reset_handlers(self) -> None:
        """Reset all handlers"""
        for handler in self.handlers.values():
            handler.reset()

# Global registry instance
global_progress_registry: ProgressRegistry | None = None

def reset_progress_state(prompt_id: str, dynprompt: "DynamicPrompt") -> None:
    global global_progress_registry

    # Reset existing handlers if registry exists
    if global_progress_registry is not None:
        global_progress_registry.reset_handlers()

    # Create new registry
    global_progress_registry = ProgressRegistry(prompt_id, dynprompt)


def add_progress_handler(handler: ProgressHandler) -> None:
    registry = get_progress_state()
    handler.set_registry(registry)
    registry.register_handler(handler)


def get_progress_state() -> ProgressRegistry:
    global global_progress_registry
    if global_progress_registry is None:
        from comfy_execution.graph import DynamicPrompt

        global_progress_registry = ProgressRegistry(
            prompt_id="", dynprompt=DynamicPrompt({})
        )
    return global_progress_registry
