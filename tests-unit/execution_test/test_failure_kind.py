"""Tests for machine-readable failure classification in execution errors.

Two invocation forms are supported on purpose, because python_embeded in the
portable Windows build ships without pytest:

    python tests-unit/execution_test/test_failure_kind.py     # standalone
    pytest tests-unit/execution_test/test_failure_kind.py     # CI

Standalone mode prints one line per case and exits non-zero on failure, so it
can be wired into a plain shell gate.

What is under test
------------------
comfy_execution/failure_kind.py plus its four call sites:

* execution.py  generic exception branch   -> out_of_memory / shape_mismatch / internal
* execution.py  interrupt branch           -> interrupted
* execution.py  execution_block_cb         -> execution_blocked
* comfy_execution/graph.py dependency cycle -> dependency_cycle

The load-bearing assertions are structural, not cosmetic:

1. failure_kind is one of ALL_FAILURE_KINDS and retryable agrees with
   RETRYABLE_KINDS -- this is what lets a client stop regex-matching English
   prose to decide whether a retry is worth attempting.
2. exception_message must NOT contain the old "TIPS:" advice block. The advice
   moved into hints; if it ever leaks back into the message, clients that show
   the message verbatim regress to displaying server opinions as exception text.
3. handle_execution_error must forward the three new keys into the
   execution_error websocket payload. An error_details dict that nobody forwards
   is invisible to every client, so this is asserted end to end via a fake
   server rather than by inspecting the dict alone.
"""

import asyncio
import os
import sys
import traceback

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

# Allow both "run from the repo root" and "run from inside this directory".
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import execution  # noqa: E402
import nodes  # noqa: E402
from comfy_execution import failure_kind as fk  # noqa: E402
from comfy_execution.graph import DynamicPrompt, ExecutionBlocker, ExecutionList  # noqa: E402


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------
class _Server:
    """Minimal ExecutionServer stand-in that records send_sync payloads."""

    def __init__(self):
        self.sent = []
        self.client_id = "test-client"
        self.last_node_id = None

    def send_sync(self, event, payload, client_id):
        self.sent.append((event, payload))

    def events(self, name):
        return [payload for event, payload in self.sent if event == name]


class _AsyncDictCache:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value):
        self.store[key] = value

    async def ensure_subcache_for(self, unique_id, children):
        return self

    def all_node_ids(self):
        return set()

    def clean_unused(self):
        pass


class _Caches:
    def __init__(self):
        self.outputs = _AsyncDictCache()
        self.objects = _AsyncDictCache()
        self.all = [self.outputs, self.objects]


class _ExecutionListStub:
    def cache_update(self, unique_id, entry):
        pass

    def get_cache(self, from_node_id, to_node_id):
        return None


class _NoProgress:
    def start_progress(self, *a, **k):
        pass

    def finish_progress(self, *a, **k):
        pass


_REGISTERED = []


def _register(class_type, exc):
    """Register a throwaway node that raises exc and return its name."""

    def run(self):
        raise exc

    node_cls = type(
        class_type,
        (),
        {
            "RETURN_TYPES": (),
            "FUNCTION": "run",
            "OUTPUT_NODE": True,
            "CATEGORY": "_test_failure_kind",
            "INPUT_TYPES": classmethod(lambda cls: {"required": {}}),
            "run": run,
        },
    )
    nodes.NODE_CLASS_MAPPINGS[class_type] = node_cls
    _REGISTERED.append(class_type)
    return class_type


def _run_via_execute(class_type):
    """Drive execution.execute() directly and return (result, error_details)."""
    original = execution.get_progress_state
    execution.get_progress_state = lambda: _NoProgress()
    try:
        result, error_details, _ex = asyncio.run(
            execution.execute(
                _Server(),
                DynamicPrompt({"1": {"class_type": class_type, "inputs": {}}}),
                _Caches(),
                "1",
                {},
                set(),
                "test-prompt",
                _ExecutionListStub(),
                {},
                {},
                {},
                None,
            )
        )
        return result, error_details
    finally:
        execution.get_progress_state = original


def _run_via_executor(prompt):
    """Drive a full PromptExecutor and return the last execution_error payload."""
    server = _Server()
    executor = execution.PromptExecutor(server, cache_args={"ram": 16.0, "ram_inactive": 1.0})
    executor.execute(prompt, "test-prompt", {"client_id": server.client_id}, ["1"])
    errors = server.events("execution_error")
    return errors[-1] if errors else None


def _check_metadata(payload, expected_kind):
    """Shared structural assertions for any payload carrying the new contract."""
    assert payload["failure_kind"] == expected_kind, payload["failure_kind"]
    assert payload["failure_kind"] in fk.ALL_FAILURE_KINDS, payload["failure_kind"]
    expected_retryable = expected_kind in fk.RETRYABLE_KINDS
    assert payload["retryable"] is expected_retryable, (payload["retryable"], expected_retryable)
    assert isinstance(payload["hints"], list) and payload["hints"], payload["hints"]
    assert all(isinstance(hint, str) and hint for hint in payload["hints"]), payload["hints"]


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------
def test_out_of_memory_is_classified_and_retryable():
    name = _register("_TestOomNode", torch.cuda.OutOfMemoryError("CUDA out of memory"))
    result, error_details = _run_via_execute(name)
    assert result == execution.ExecutionResult.FAILURE, result
    _check_metadata(error_details, fk.FAILURE_KIND_OUT_OF_MEMORY)
    assert error_details["retryable"] is True


def test_out_of_memory_message_has_no_tips_block():
    """The advice moved to hints; the message must now be the bare exception."""
    name = _register("_TestOomNodeMsg", torch.cuda.OutOfMemoryError("CUDA out of memory"))
    _result, error_details = _run_via_execute(name)
    message = error_details["exception_message"]
    assert "TIPS" not in message, message
    assert "This error means you ran out of memory" not in message, message
    assert message == "CUDA out of memory", message


def test_shape_mismatch_is_classified_and_not_retryable():
    # The node type must literally contain "Sampler": the shape hint is scoped to
    # sampler nodes because the same message from a text encoder means something
    # else entirely. Using a name without "Sampler" here would silently assert
    # the generic branch instead.
    name = _register(
        "_TestShapeSamplerNode",
        RuntimeError("mat1 and mat2 shapes cannot be multiplied (1x768 and 4096x768)"),
    )
    _result, error_details = _run_via_execute(name)
    _check_metadata(error_details, fk.FAILURE_KIND_SHAPE_MISMATCH)
    assert error_details["retryable"] is False


def test_generic_error_is_internal_and_not_retryable():
    name = _register("_TestGenericNode", RuntimeError("something else broke"))
    _result, error_details = _run_via_execute(name)
    _check_metadata(error_details, fk.FAILURE_KIND_INTERNAL)
    assert error_details["retryable"] is False


def test_interrupt_is_classified_without_exception_message():
    name = _register(
        "_TestInterruptNode",
        execution.comfy.model_management.InterruptProcessingException(),
    )
    result, error_details = _run_via_execute(name)
    assert result == execution.ExecutionResult.FAILURE, result
    _check_metadata(error_details, fk.FAILURE_KIND_INTERRUPTED)
    assert "exception_message" not in error_details, sorted(error_details)


def test_execution_blocked_event_carries_classification():
    """execution_block_cb builds its own payload; it must carry the contract too."""
    nodes.NODE_CLASS_MAPPINGS["_TestBlockerNode"] = type(
        "_TestBlockerNode",
        (),
        {
            "RETURN_TYPES": ("INT",),
            "FUNCTION": "run",
            "OUTPUT_NODE": False,
            "CATEGORY": "_test_failure_kind",
            "INPUT_TYPES": classmethod(lambda cls: {"required": {}}),
            "run": lambda self: (ExecutionBlocker("blocked on purpose"),),
        },
    )
    _REGISTERED.append("_TestBlockerNode")
    nodes.NODE_CLASS_MAPPINGS["_TestBlockedSinkNode"] = type(
        "_TestBlockedSinkNode",
        (),
        {
            "RETURN_TYPES": (),
            "FUNCTION": "run",
            "OUTPUT_NODE": True,
            "CATEGORY": "_test_failure_kind",
            "INPUT_TYPES": classmethod(lambda cls: {"required": {"x": ("INT",)}}),
            "run": lambda self, x: (),
        },
    )
    _REGISTERED.append("_TestBlockedSinkNode")

    payload = _run_via_executor(
        {
            "1": {"class_type": "_TestBlockedSinkNode", "inputs": {"x": ["2", 0]}},
            "2": {"class_type": "_TestBlockerNode", "inputs": {}},
        }
    )
    assert payload is not None, "no execution_error event was emitted"
    _check_metadata(payload, fk.FAILURE_KIND_EXECUTION_BLOCKED)
    assert payload["retryable"] is False
    assert payload["exception_message"] == "Execution Blocked: blocked on purpose"


def test_dependency_cycle_is_classified():
    """Build the cycle state by hand: add_node needs a live cache, and this is the
    same construction tests-unit/execution_test/test_execution_list.py uses."""
    execution_list = ExecutionList.__new__(ExecutionList)
    execution_list.staged_node_id = None
    execution_list.pendingNodes = {"1": True, "2": True}
    execution_list.blockCount = {"1": 1, "2": 1}
    execution_list.blocking = {"1": {"2": {0: True}}, "2": {"1": {0: True}}}
    execution_list.execution_cache = {"1": {}, "2": {}}
    execution_list.execution_cache_listeners = {"1": set(), "2": set()}
    execution_list.externalBlocks = 0
    execution_list.externalBlockResults = {}
    execution_list.unblockedEvent = asyncio.Event()
    execution_list.dynprompt = DynamicPrompt(
        {
            "1": {"class_type": "_CycleA", "inputs": {"a": ["2", 0]}},
            "2": {"class_type": "_CycleB", "inputs": {"b": ["1", 0]}},
        }
    )

    node_id, error_details, ex = asyncio.run(execution_list.stage_node_execution())
    assert node_id is None, node_id
    assert isinstance(ex, Exception), ex
    _check_metadata(error_details, fk.FAILURE_KIND_DEPENDENCY_CYCLE)
    assert error_details["retryable"] is False


def test_handle_execution_error_forwards_the_three_keys():
    """The classification is worthless unless it reaches the websocket payload."""
    name = _register("_TestForwardNode", torch.cuda.OutOfMemoryError("CUDA out of memory"))
    payload = _run_via_executor({"1": {"class_type": name, "inputs": {}}})
    assert payload is not None, "no execution_error event was emitted"
    _check_metadata(payload, fk.FAILURE_KIND_OUT_OF_MEMORY)
    assert payload["node_id"] == "1", payload["node_id"]
    assert payload["node_type"] == name, payload["node_type"]
    assert isinstance(payload["executed"], list), payload["executed"]
    assert "timestamp" in payload, sorted(payload)
    assert "TIPS" not in payload["exception_message"], payload["exception_message"]


def test_failure_metadata_shape_is_consistent():
    """Every declared kind must produce a usable tuple, or clients hit holes."""
    seen = set()
    for kind in fk.ALL_FAILURE_KINDS:
        metadata = fk.failure_metadata(kind)
        assert set(metadata) == {"failure_kind", "retryable", "hints"}, sorted(metadata)
        assert metadata["failure_kind"] == kind, metadata["failure_kind"]
        assert isinstance(metadata["retryable"], bool), metadata["retryable"]
        assert isinstance(metadata["hints"], list) and metadata["hints"], metadata["hints"]
        seen.add(kind)
    assert seen == set(fk.ALL_FAILURE_KINDS), (seen, fk.ALL_FAILURE_KINDS)
    assert len(fk.ALL_FAILURE_KINDS) == 6, fk.ALL_FAILURE_KINDS


def test_classify_exception_precedence():
    """interrupted outranks is_oom; a non-OOM error must never look retryable."""
    oom = torch.cuda.OutOfMemoryError("CUDA out of memory")
    assert fk.classify_exception(oom, is_oom=True) == fk.FAILURE_KIND_OUT_OF_MEMORY
    assert (
        fk.classify_exception(oom, is_oom=True, interrupted=True)
        == fk.FAILURE_KIND_INTERRUPTED
    )
    assert (
        fk.classify_exception(
            RuntimeError("mat1 and mat2 shapes cannot be multiplied"),
            is_oom=False,
            class_type="KSampler",
        )
        == fk.FAILURE_KIND_SHAPE_MISMATCH
    )
    # The shape hint only applies to sampler nodes; elsewhere it is a plain error.
    assert (
        fk.classify_exception(
            RuntimeError("mat1 and mat2 shapes cannot be multiplied"),
            is_oom=False,
            class_type="CLIPTextEncode",
        )
        == fk.FAILURE_KIND_INTERNAL
    )
    assert fk.classify_exception(ValueError("boom"), is_oom=False) == fk.FAILURE_KIND_INTERNAL


def test_retryable_kinds_is_exactly_out_of_memory():
    assert fk.RETRYABLE_KINDS == frozenset({fk.FAILURE_KIND_OUT_OF_MEMORY}), fk.RETRYABLE_KINDS


# --------------------------------------------------------------------------
# standalone runner
# --------------------------------------------------------------------------
_CASES = [
    test_out_of_memory_is_classified_and_retryable,
    test_out_of_memory_message_has_no_tips_block,
    test_shape_mismatch_is_classified_and_not_retryable,
    test_generic_error_is_internal_and_not_retryable,
    test_interrupt_is_classified_without_exception_message,
    test_execution_blocked_event_carries_classification,
    test_dependency_cycle_is_classified,
    test_handle_execution_error_forwards_the_three_keys,
    test_failure_metadata_shape_is_consistent,
    test_classify_exception_precedence,
    test_retryable_kinds_is_exactly_out_of_memory,
]


def main():
    failures = 0
    ran = 0
    for case in _CASES:
        ran += 1
        try:
            case()
        except Exception:
            failures += 1
            print("FAIL " + case.__name__)
            traceback.print_exc()
        else:
            print("PASS " + case.__name__)
    # Guard against a silently empty run: a gate that tests nothing is green for free.
    assert ran == len(_CASES) and ran >= 11, "ran " + str(ran) + " of " + str(len(_CASES)) + " cases"
    print("")
    print(str(ran - failures) + "/" + str(ran) + " cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
