"""Render progress for ``GET /api/wrapper/jobs/{job_id}``.

ComfyUI already tracks, per prompt and per node, how far execution has got:
``comfy_execution.progress`` keeps a registry for the prompt that is running
now (``get_progress_state()``), fed by the global progress hook in
``main.py`` (sampler steps, tiled decodes, ...) and by the executor as nodes
start and finish. This module turns that registry into one number a caller
can draw a bar from.

The overall fraction weighs each node of the prompt by how much work it is:

- a node that reported a step total (``max`` > 1, e.g. the H3 sampler's 30
  steps) weighs its step total and counts ``value`` of it as done;
- every other node weighs 1 and counts fully once finished (a cached node is
  finished at once, so it counts fully too);
- steps the prompt asks for (literal ``steps`` inputs, e.g. on the scheduler
  node) that no node has reported yet are reserved up front, so the bar does
  not race to 40% over the loaders and then crawl once the sampler starts.

It never decreases for a job (the last value is remembered, bounded), stays
below 100 while the job runs, and is 100 only once the job completed. Pure
functions over plain data, so it is testable without the server.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict

# The last fraction reported per job, so a node that re-reports a larger step
# total (or a registry swap) never moves the bar backwards.
_MAX_REMEMBERED = 512
_remembered: "OrderedDict[str, float]" = OrderedDict()
_lock = threading.Lock()

# Below 100 until the job is actually complete: the history entry, not the
# last sampler step, is what says the output exists.
IN_PROGRESS_CAP = 0.99


def _state_name(state) -> str:
    """NodeState enum or plain string -> 'pending' | 'running' | 'finished' | 'error'."""
    value = getattr(state, "value", state)
    return value if isinstance(value, str) else "pending"


def _number(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def anticipated_steps(prompt) -> float:
    """Sum of the literal integer ``steps`` inputs in the prompt (links are lists and are skipped)."""
    total = 0.0
    for node in (prompt or {}).values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        steps = inputs.get("steps") if isinstance(inputs, dict) else None
        if isinstance(steps, bool):
            continue
        if isinstance(steps, (int, float)) and steps > 0:
            total += float(steps)
    return total


def overall_fraction(prompt, nodes) -> float:
    """0..1 across the prompt's nodes (see the module docstring for the weights).

    ``prompt`` maps node id -> node dict (``inputs``); ``nodes`` maps node id ->
    ``{"state", "value", "max"}`` as the registry keeps them. Nodes the
    registry has not seen yet are pending.
    """
    done = 0.0
    total = 0.0
    stepped = 0.0
    unfinished = False
    for node_id in set(prompt or {}) | set(nodes or {}):
        entry = (nodes or {}).get(node_id) or {}
        state = _state_name(entry.get("state"))
        unfinished = unfinished or state != "finished"
        maximum = _number(entry.get("max"))
        value = min(max(_number(entry.get("value")), 0.0), maximum) if maximum > 0 else 0.0
        if maximum > 1:
            stepped += maximum
            total += maximum
            done += maximum if state == "finished" else value
            continue
        total += 1.0
        if state == "finished":
            done += 1.0
        elif state == "running" and maximum > 0:
            done += value / maximum
    # Only work still ahead can hold steps nobody reported (a sampler that never
    # calls the progress hook must not pin a finished graph below 100).
    if unfinished:
        total += max(0.0, anticipated_steps(prompt) - stepped)
    if total <= 0:
        return 0.0
    return min(1.0, max(0.0, done / total))


def _snapshot(registry, job_id):
    """(prompt, nodes, dynprompt) for ``job_id`` when the registry is tracking it, else None.

    The executor thread mutates the registry while this runs on the event
    loop, so the node states are copied first (and the copy retried if a node
    was added mid-copy).
    """
    if registry is None or str(getattr(registry, "prompt_id", "")) != str(job_id):
        return None
    dynprompt = getattr(registry, "dynprompt", None)
    prompt = getattr(dynprompt, "original_prompt", None) or {}
    for _ in range(3):
        try:
            nodes = {node_id: dict(state) for node_id, state in list(registry.nodes.items())}
            break
        except RuntimeError:
            continue
    else:
        return None
    return prompt, nodes, dynprompt


def _running_node(nodes, dynprompt):
    """The node running now (the last one to start, preferring one that reports steps)."""
    running = [(node_id, entry) for node_id, entry in nodes.items() if _state_name(entry.get("state")) == "running"]
    if not running:
        return None, None, None
    stepped = [item for item in running if _number(item[1].get("max")) > 1]
    node_id, entry = (stepped or running)[-1]
    node_class = None
    try:
        node_class = dynprompt.get_node(node_id).get("class_type") if dynprompt is not None else None
    except Exception:  # an ephemeral node already gone, or a stub without get_node
        node_class = None
    maximum = _number(entry.get("max"))
    step = {"value": int(min(max(_number(entry.get("value")), 0.0), maximum)), "max": int(maximum)} if maximum > 1 else None
    return str(node_id), node_class, step


def queue_position(job_id, queued_items):
    """1-based place among the pending prompts (1 = runs next), or None."""
    try:
        ordered = sorted(queued_items or [], key=lambda item: item[0])
    except (TypeError, IndexError):
        ordered = list(queued_items or [])
    for index, item in enumerate(ordered):
        try:
            if str(item[1]) == str(job_id):
                return index + 1
        except (TypeError, IndexError):
            continue
    return None


def _remember(job_id, fraction) -> float:
    with _lock:
        previous = _remembered.pop(job_id, 0.0)
        value = max(previous, fraction)
        _remembered[job_id] = value
        while len(_remembered) > _MAX_REMEMBERED:
            _remembered.popitem(last=False)
        return value


def forget(job_id) -> None:
    with _lock:
        _remembered.pop(job_id, None)


def _body(fraction, node=None, node_class=None, step=None, position=None):
    return {
        "fraction": round(fraction, 4),
        "percent": int(math.floor(fraction * 100 + 1e-9)),
        "node": node,
        "node_class": node_class,
        "step": step,
        "queue_position": position,
    }


def job_progress(job_id, status, registry=None, queued_items=None):
    """The ``progress`` object for a job dict, or None (failed / cancelled / unknown).

    ``registry`` is ``comfy_execution.progress.get_progress_state()``;
    ``queued_items`` the pending queue tuples (``(number, prompt_id, ...)``).
    """
    job_id = str(job_id)
    if status == "completed":
        forget(job_id)
        return _body(1.0)
    if status not in ("pending", "in_progress"):
        forget(job_id)
        return None

    fraction, node, node_class, step, position = 0.0, None, None, None, None
    if status == "pending":
        position = queue_position(job_id, queued_items)
    else:
        snapshot = _snapshot(registry, job_id)
        if snapshot is not None:
            prompt, nodes, dynprompt = snapshot
            fraction = overall_fraction(prompt, nodes)
            node, node_class, step = _running_node(nodes, dynprompt)
    fraction = _remember(job_id, min(fraction, IN_PROGRESS_CAP))
    return _body(fraction, node, node_class, step, position)
