from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias


class FailureSource(str, Enum):
    NODE = "node"
    INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class QueuedEvent:
    prompt_id: str
    queue_number: float
    output_node_ids: tuple[str, ...]
    metadata: Mapping[str, Any]
    queued_at_ms: int


@dataclass(frozen=True, slots=True)
class StartedEvent:
    prompt_id: str
    metadata: Mapping[str, Any]
    queued_at_ms: int
    started_at_ms: int


@dataclass(frozen=True, slots=True)
class SucceededEvent:
    prompt_id: str
    metadata: Mapping[str, Any]
    history_result: Mapping[str, Any] | None
    queued_at_ms: int
    started_at_ms: int
    finished_at_ms: int
    execution_duration_ms: int


@dataclass(frozen=True, slots=True)
class FailedEvent:
    prompt_id: str
    metadata: Mapping[str, Any]
    source: FailureSource
    node_id: str | None
    node_type: str | None
    exception_type: str
    exception_message: str
    traceback: tuple[str, ...]
    executed_nodes: tuple[str, ...]
    history_result: Mapping[str, Any] | None
    history_task_done_succeeded: bool
    queued_at_ms: int
    started_at_ms: int
    error_reported_at_ms: int
    finished_at_ms: int
    execution_duration_ms: int


@dataclass(frozen=True, slots=True)
class InterruptedEvent:
    prompt_id: str
    metadata: Mapping[str, Any]
    node_id: str | None
    node_type: str | None
    executed_nodes: tuple[str, ...]
    history_result: Mapping[str, Any] | None
    queued_at_ms: int
    started_at_ms: int
    interruption_reported_at_ms: int
    finished_at_ms: int
    execution_duration_ms: int


@dataclass(frozen=True, slots=True)
class CancelledEvent:
    prompt_id: str
    metadata: Mapping[str, Any]
    queued_at_ms: int
    cancelled_at_ms: int


TerminalEvent: TypeAlias = SucceededEvent | FailedEvent | InterruptedEvent | CancelledEvent
LifecycleEvent: TypeAlias = QueuedEvent | StartedEvent | TerminalEvent


class LifecycleHandler:
    def accepts(self, event: LifecycleEvent) -> bool:
        return True

    def on_queued(self, event: QueuedEvent) -> None:
        pass

    def on_started(self, event: StartedEvent) -> None:
        pass

    def on_succeeded(self, event: SucceededEvent) -> None:
        pass

    def on_failed(self, event: FailedEvent) -> None:
        pass

    def on_interrupted(self, event: InterruptedEvent) -> None:
        pass

    def on_cancelled(self, event: CancelledEvent) -> None:
        pass
