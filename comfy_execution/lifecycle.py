from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import inspect
import logging
import queue
import threading
import time
import traceback as traceback_module
from types import MappingProxyType
from typing import Any

from comfy_api.latest._execution_lifecycle import (
    CancelledEvent,
    FailedEvent,
    FailureSource,
    InterruptedEvent,
    LifecycleEvent,
    LifecycleHandler,
    QueuedEvent,
    StartedEvent,
    SucceededEvent,
)


logger = logging.getLogger(__name__)

_EVENT_METHODS = {
    QueuedEvent: "on_queued",
    StartedEvent: "on_started",
    SucceededEvent: "on_succeeded",
    FailedEvent: "on_failed",
    InterruptedEvent: "on_interrupted",
    CancelledEvent: "on_cancelled",
}


@dataclass(frozen=True, slots=True)
class _HandlerRegistration:
    handler: LifecycleHandler
    methods: Mapping[type, Callable]


@dataclass(frozen=True, slots=True)
class _QueuedLifecycleContext:
    queued_at_ms: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _ExecutionOutcome:
    kind: str
    details: Mapping[str, Any]


_EMPTY_MAPPING: Mapping[str, Any] = MappingProxyType({})
_pending_registrations: list[_HandlerRegistration] = []
_frozen_routes: Mapping[type, tuple[tuple[LifecycleHandler, Callable], ...]] | None = None
_dispatch_queue: queue.Queue | None = None
_dispatch_thread: threading.Thread | None = None
_STOP = object()


def _register_handler(handler: LifecycleHandler) -> None:
    if _frozen_routes is not None:
        raise RuntimeError("Execution lifecycle handlers are already frozen")
    if not isinstance(handler, LifecycleHandler):
        raise TypeError("handler must be an ExecutionLifecycle.Handler instance")
    if inspect.iscoroutinefunction(handler.accepts):
        raise TypeError("LifecycleHandler.accepts must be synchronous")
    if any(registration.handler is handler for registration in _pending_registrations):
        logger.warning("Execution lifecycle handler is already registered: %s", type(handler).__name__)
        return

    methods = {}
    for event_type, method_name in _EVENT_METHODS.items():
        method = getattr(handler, method_name)
        if inspect.iscoroutinefunction(method):
            raise TypeError(f"LifecycleHandler.{method_name} must be synchronous")
        if getattr(type(handler), method_name) is not getattr(LifecycleHandler, method_name):
            methods[event_type] = method

    if not methods:
        raise TypeError("LifecycleHandler must override at least one event method")
    _pending_registrations.append(_HandlerRegistration(handler, MappingProxyType(methods)))


def _freeze_handler_routes() -> None:
    global _frozen_routes, _dispatch_queue, _dispatch_thread

    if _frozen_routes is not None:
        return

    routes = {}
    for event_type in _EVENT_METHODS:
        routes[event_type] = tuple(
            (registration.handler, registration.methods[event_type])
            for registration in _pending_registrations
            if event_type in registration.methods
        )
    _frozen_routes = MappingProxyType(routes)

    if any(routes.values()):
        _dispatch_queue = queue.Queue()
        _dispatch_thread = threading.Thread(
            target=_dispatch_events,
            name="execution-lifecycle",
            daemon=True,
        )
        _dispatch_thread.start()


def _has_any_handlers() -> bool:
    return _frozen_routes is not None and any(_frozen_routes.values())


def _has_handlers(event_type: type) -> bool:
    return _frozen_routes is not None and bool(_frozen_routes.get(event_type, ()))


def _publish(event: LifecycleEvent) -> None:
    if not _has_handlers(type(event)):
        return
    assert _dispatch_queue is not None
    _dispatch_queue.put_nowait(event)


def _dispatch_events() -> None:
    assert _dispatch_queue is not None
    dispatch_queue = _dispatch_queue
    while True:
        event = dispatch_queue.get()
        try:
            if event is _STOP:
                return
            routes = _frozen_routes.get(type(event), ()) if _frozen_routes is not None else ()
            for handler, method in routes:
                try:
                    if handler.accepts(event):
                        method(event)
                except BaseException:
                    logger.exception(
                        "Execution lifecycle handler %s failed for %s",
                        type(handler).__name__,
                        type(event).__name__,
                    )
        finally:
            dispatch_queue.task_done()


def _freeze_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Lifecycle snapshot mapping keys must be strings")
            frozen[key] = _freeze_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    raise TypeError(f"Unsupported lifecycle snapshot value: {type(value).__name__}")


def _freeze_metadata(extra_data: Any) -> Mapping[str, Any]:
    if not isinstance(extra_data, Mapping):
        return _EMPTY_MAPPING
    metadata = extra_data.get("execution_lifecycle")
    if not isinstance(metadata, Mapping):
        return _EMPTY_MAPPING
    try:
        return _freeze_value(metadata)
    except Exception:
        logger.warning("Ignoring invalid execution lifecycle metadata", exc_info=True)
        return _EMPTY_MAPPING


def _freeze_history_result(history_result: Any) -> Mapping[str, Any] | None:
    if not isinstance(history_result, Mapping):
        return None
    try:
        return _freeze_value(history_result)
    except Exception:
        logger.warning("Unable to snapshot execution lifecycle history result", exc_info=True)
        return None


def _publish_queued(item, context: _QueuedLifecycleContext) -> None:
    if not _has_handlers(QueuedEvent):
        return
    try:
        _publish(QueuedEvent(
            prompt_id=item[1],
            queue_number=float(item[0]),
            output_node_ids=tuple(sorted(str(node_id) for node_id in item[4])),
            metadata=context.metadata,
            queued_at_ms=context.queued_at_ms,
        ))
    except Exception:
        logger.exception("Unable to publish QueuedEvent")


def _publish_cancelled(item, context: _QueuedLifecycleContext, cancelled_at_ms: int) -> None:
    if not _has_handlers(CancelledEvent):
        return
    try:
        _publish(CancelledEvent(
            prompt_id=item[1],
            metadata=context.metadata,
            queued_at_ms=context.queued_at_ms,
            cancelled_at_ms=cancelled_at_ms,
        ))
    except Exception:
        logger.exception("Unable to publish CancelledEvent")


def _full_type_name(exception_type: type[BaseException]) -> str:
    if exception_type.__module__ == "builtins":
        return exception_type.__qualname__
    return f"{exception_type.__module__}.{exception_type.__qualname__}"


def _terminal_message(status_messages: Sequence) -> tuple[str, Mapping[str, Any]] | None:
    terminal_names = {"execution_error", "execution_interrupted", "execution_success"}
    for message in reversed(status_messages):
        if not isinstance(message, (tuple, list)) or len(message) != 2:
            continue
        name, details = message
        if name in terminal_names and isinstance(details, Mapping):
            return name, details
    return None


class _PromptExecutionLifecycle:
    def __init__(self, prompt_id: str, queued_context: _QueuedLifecycleContext):
        self.prompt_id = prompt_id
        self.queued_context = queued_context
        self.started_at_ms = 0
        self._started_at = 0.0
        self._outcome: _ExecutionOutcome | None = None
        self._history_result = None
        self._task_done_succeeded = False

    def __enter__(self):
        self.started_at_ms = int(time.time() * 1000)
        self._started_at = time.perf_counter()
        if _has_handlers(StartedEvent):
            try:
                _publish(StartedEvent(
                    prompt_id=self.prompt_id,
                    metadata=self.queued_context.metadata,
                    queued_at_ms=self.queued_context.queued_at_ms,
                    started_at_ms=self.started_at_ms,
                ))
            except Exception:
                logger.exception("Unable to publish StartedEvent")
        return self

    def record_executor_result(self, status_messages: Sequence, history_result: Any) -> None:
        self._history_result = history_result
        try:
            terminal = _terminal_message(status_messages)
            if terminal is None:
                self._outcome = self._internal_failure(
                    "MissingTerminalExecutionStatus",
                    "PromptExecutor returned without a terminal status message",
                )
                return

            event_name, details = terminal
            if event_name == "execution_error":
                self._outcome = _ExecutionOutcome("node_failure", details)
            elif event_name == "execution_interrupted":
                self._outcome = _ExecutionOutcome("interrupted", details)
            else:
                self._outcome = _ExecutionOutcome("success", details)
        except Exception as error:
            logger.exception("Unable to record execution lifecycle result")
            self._outcome = self._internal_failure(_full_type_name(type(error)), str(error))

    def mark_task_done_succeeded(self) -> None:
        self._task_done_succeeded = True

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is not None and (self._outcome is None or self._outcome.kind != "node_failure"):
                self._outcome = self._exception_failure(exc_type, exc, tb)
            elif self._outcome is None:
                self._outcome = self._internal_failure(
                    "MissingTerminalExecutionStatus",
                    "Execution lifecycle ended without an executor result",
                )

            if self._outcome.kind in ("success", "interrupted") and not self._task_done_succeeded:
                self._outcome = self._internal_failure(
                    "PromptQueueTaskDoneError",
                    "PromptQueue.task_done did not complete",
                )
            self._publish_terminal()
        except Exception:
            logger.exception("Unable to publish terminal execution lifecycle event")
        return False

    def _internal_failure(self, exception_type: str, message: str) -> _ExecutionOutcome:
        return _ExecutionOutcome("internal_failure", MappingProxyType({
            "exception_type": exception_type,
            "exception_message": message,
            "traceback": (),
            "timestamp": int(time.time() * 1000),
        }))

    def _exception_failure(self, exc_type, exc, tb) -> _ExecutionOutcome:
        return _ExecutionOutcome("internal_failure", MappingProxyType({
            "exception_type": _full_type_name(exc_type),
            "exception_message": str(exc),
            "traceback": tuple(traceback_module.format_exception(exc_type, exc, tb)),
            "timestamp": int(time.time() * 1000),
        }))

    def _publish_terminal(self) -> None:
        finished_at_ms = int(time.time() * 1000)
        execution_duration_ms = max(0, int((time.perf_counter() - self._started_at) * 1000))
        outcome = self._outcome
        assert outcome is not None

        if outcome.kind == "success":
            if not _has_handlers(SucceededEvent):
                return
            event = SucceededEvent(
                prompt_id=self.prompt_id,
                metadata=self.queued_context.metadata,
                history_result=_freeze_history_result(self._history_result),
                queued_at_ms=self.queued_context.queued_at_ms,
                started_at_ms=self.started_at_ms,
                finished_at_ms=finished_at_ms,
                execution_duration_ms=execution_duration_ms,
            )
        elif outcome.kind == "interrupted":
            if not _has_handlers(InterruptedEvent):
                return
            details = outcome.details
            event = InterruptedEvent(
                prompt_id=self.prompt_id,
                metadata=self.queued_context.metadata,
                node_id=_optional_string(details.get("node_id")),
                node_type=_optional_string(details.get("node_type")),
                executed_nodes=_sorted_strings(details.get("executed", ())),
                history_result=_freeze_history_result(self._history_result),
                queued_at_ms=self.queued_context.queued_at_ms,
                started_at_ms=self.started_at_ms,
                interruption_reported_at_ms=_timestamp(details, finished_at_ms),
                finished_at_ms=finished_at_ms,
                execution_duration_ms=execution_duration_ms,
            )
        else:
            if not _has_handlers(FailedEvent):
                return
            details = outcome.details
            event = FailedEvent(
                prompt_id=self.prompt_id,
                metadata=self.queued_context.metadata,
                source=FailureSource.NODE if outcome.kind == "node_failure" else FailureSource.INTERNAL,
                node_id=_optional_string(details.get("node_id")),
                node_type=_optional_string(details.get("node_type")),
                exception_type=str(details.get("exception_type", "ExecutionError")),
                exception_message=str(details.get("exception_message", "Execution failed")),
                traceback=tuple(str(line) for line in details.get("traceback", ())),
                executed_nodes=_sorted_strings(details.get("executed", ())),
                history_result=_freeze_history_result(self._history_result),
                history_task_done_succeeded=self._task_done_succeeded,
                queued_at_ms=self.queued_context.queued_at_ms,
                started_at_ms=self.started_at_ms,
                error_reported_at_ms=_timestamp(details, finished_at_ms),
                finished_at_ms=finished_at_ms,
                execution_duration_ms=execution_duration_ms,
            )
        _publish(event)


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _sorted_strings(values: Any) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple, set, frozenset)):
        return ()
    return tuple(sorted(str(value) for value in values))


def _timestamp(details: Mapping[str, Any], default: int) -> int:
    value = details.get("timestamp")
    return int(value) if isinstance(value, (int, float)) else default


def _wait_for_dispatch_for_tests(timeout: float = 1.0) -> bool:
    dispatch_queue = _dispatch_queue
    if dispatch_queue is None:
        return True
    deadline = time.monotonic() + timeout
    with dispatch_queue.all_tasks_done:
        while dispatch_queue.unfinished_tasks:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            dispatch_queue.all_tasks_done.wait(remaining)
    return True


def _reset_for_tests() -> None:
    global _pending_registrations, _frozen_routes, _dispatch_queue, _dispatch_thread
    if _dispatch_queue is not None and _dispatch_thread is not None:
        _dispatch_queue.put(_STOP)
        _dispatch_thread.join(timeout=1)
    _pending_registrations = []
    _frozen_routes = None
    _dispatch_queue = None
    _dispatch_thread = None
