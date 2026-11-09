from comfy_api.latest import IO
from comfy_execution.graph_utils import is_link


class LoopValidationError(Exception):
    def __init__(self, error_type, message, details, node_ids, output_ids):
        super().__init__(details)
        self.error = {
            "type": "custom_validation_failed",
            "message": message,
            "details": details,
            "extra_info": {
                "input_name": "loop boundary",
                "loop_error_type": error_type,
                "node_ids": sorted(node_ids),
                "output_ids": sorted(output_ids),
            },
        }


def _walk_graph(start_ids, edges, stop_at=(), return_stops=False):
    found = set()
    stops = set()
    pending = list(start_ids)
    while pending:
        node_id = pending.pop()
        if node_id in found:
            continue
        found.add(node_id)
        if node_id in stop_at:
            stops.add(node_id)
        else:
            pending.extend(edges[node_id])
    return stops if return_stops else found


def _loop_validation_error(error_type, message, details, actors, children, outputs):
    reached_outputs = _walk_graph(actors, children).intersection(outputs)
    return LoopValidationError(error_type, message, details, actors, reached_outputs)


def validate_loops(prompt, outputs, node_ids, start_nodes, end_nodes):
    if not start_nodes and not end_nodes:
        return {}

    node_ids = set(node_ids)
    all_children = {node_id: set() for node_id in prompt}
    all_parents = {node_id: set() for node_id in prompt}
    for node_id, node in prompt.items():
        for value in node.get("inputs", {}).values():
            if is_link(value) and value[0] in all_children:
                all_children[value[0]].add(node_id)
                all_parents[node_id].add(value[0])
    continuation = _walk_graph(outputs, all_children, start_nodes)
    continuation.difference_update(set(start_nodes).difference(node_ids))
    node_ids.update(continuation)
    node_ids.update(_walk_graph(set(end_nodes).intersection(continuation), all_parents, start_nodes))
    parents = {
        node_id: {
            value[0]
            for value in prompt[node_id].get("inputs", {}).values()
            if is_link(value) and value[0] in node_ids
        }
        for node_id in node_ids
    }

    children = {node_id: set() for node_id in node_ids}
    for node_id, node_parents in parents.items():
        for parent_id in node_parents:
            children[parent_id].add(node_id)

    start_nodes = set(start_nodes).intersection(node_ids)
    end_nodes = set(end_nodes).intersection(node_ids)
    terminal_outputs = {node_id for node_id in outputs if not children[node_id]}

    # Construct the Start DAG independently of Ends. Completed inner loops can
    # lead to later Starts which are still nested under the same outer Start.
    start_dag = {
        start_id: _walk_graph(children[start_id], children, start_nodes, return_stops=True)
        for start_id in start_nodes
    }
    start_descendants = {
        start_id: _walk_graph(start_dag[start_id], start_dag)
        for start_id in start_nodes
    }

    # Construct the End DAG in the reverse direction. Its leaves are the
    # innermost Ends and are therefore paired first.
    end_dag = {
        end_id: _walk_graph(parents[end_id], parents, end_nodes, return_stops=True)
        for end_id in end_nodes
    }

    pairs = {}
    remaining_starts = set(start_nodes)
    remaining_ends = set(end_nodes)
    while remaining_ends:
        end_id = next(
            node_id
            for node_id in sorted(remaining_ends)
            if not end_dag[node_id].intersection(remaining_ends)
        )

        candidates = _walk_graph(parents[end_id], parents, remaining_starts, return_stops=True)

        if not candidates:
            raise _loop_validation_error(
                "loop_end_without_start",
                "End Loop has no Start Loop",
                f"End Loop {end_id} has no available Start Loop",
                {end_id},
                children,
                outputs,
            )

        closest = {
            candidate
            for candidate in candidates
            if all(other == candidate or candidate in start_descendants[other] for other in candidates)
        }
        if len(closest) != 1:
            candidate_list = ", ".join(sorted(candidates))
            raise _loop_validation_error(
                "ambiguous_loop_nesting",
                "End Loop has ambiguous Start Loops",
                f"End Loop {end_id} can close multiple unrelated Start Loops: {candidate_list}",
                candidates.union((end_id,)),
                children,
                outputs,
            )

        start_id = closest.pop()
        pairs[start_id] = end_id
        remaining_starts.remove(start_id)
        remaining_ends.remove(end_id)

        # Validate the new pair immediately. Previously paired Ends are inner
        # boundaries and may be crossed; an unpaired End or output is an escape.
        escapes = _walk_graph(
            children[start_id],
            children,
            remaining_ends | terminal_outputs | {end_id},
            return_stops=True,
        )
        escapes.discard(end_id)
        if escapes:
            escape_list = ", ".join(sorted(escapes))
            raise _loop_validation_error(
                "loop_escape",
                "Loop body is not closed",
                f"Start Loop {start_id} reaches {escape_list} without passing through End Loop {end_id}",
                escapes | {start_id, end_id},
                children,
                outputs,
            )

    if remaining_starts:
        start_list = ", ".join(sorted(remaining_starts))
        raise _loop_validation_error(
            "loop_start_without_end",
            "Start Loop has no End Loop",
            f"Start Loops without End Loops: {start_list}",
            remaining_starts,
            children,
            outputs,
        )

    bodies = {}
    for start_id, end_id in pairs.items():
        body = _walk_graph(children[start_id], children, {end_id})
        body.remove(end_id)
        bodies[start_id] = body
        accumulate = prompt[end_id].get("inputs", {}).get("accumulate")
        if is_link(accumulate) and (accumulate[0] == start_id or accumulate[0] in body):
            source_id = accumulate[0]
            raise _loop_validation_error(
                "loop_accumulate_from_body",
                "End Loop accumulate depends on its loop body",
                f"End Loop {end_id} accumulate is driven by loop node {source_id} under Start Loop {start_id}",
                {start_id, end_id, source_id},
                children,
                outputs,
            )

    for start_id, end_id in pairs.items():
        body = bodies[start_id]
        prompt[start_id]["_loop_body"] = sorted(body)
        prompt[start_id]["_loop_end"] = end_id

    return pairs


def validate_node_input(
    received_type: str, input_type: str, strict: bool = False
) -> bool:
    """
    received_type and input_type are both strings of the form "T1,T2,...".

    If strict is True, the input_type must contain the received_type.
      For example, if received_type is "STRING" and input_type is "STRING,INT",
      this will return True. But if received_type is "STRING,INT" and input_type is
      "INT", this will return False.

    If strict is False, the input_type must have overlap with the received_type.
      For example, if received_type is "STRING,BOOLEAN" and input_type is "STRING,INT",
      this will return True.

    Supports pre-union type extension behaviour of ``__ne__`` overrides.
    """
    # If the types are exactly the same, we can return immediately
    # Use pre-union behaviour: inverse of `__ne__`
    # NOTE: this lets legacy '*' Any types work that override the __ne__ method of the str class.
    if not received_type != input_type:
        return True

    # If one of the types is '*', we can return True immediately; this is the 'Any' type.
    if received_type == IO.AnyType.io_type or input_type == IO.AnyType.io_type:
        return True

    # If the received type or input_type is a MatchType, we can return True immediately;
    # validation for this is handled by the frontend
    if received_type == IO.MatchType.io_type or input_type == IO.MatchType.io_type:
        return True

    # This accounts for some custom nodes that output lists of options as the type;
    # if we ever want to break them on purpose, this can be removed
    if isinstance(received_type, list) and input_type == IO.Combo.io_type:
        return True

    # Not equal, and not strings
    if not isinstance(received_type, str) or not isinstance(input_type, str):
        return False

    # Split the type strings into sets for comparison
    received_types = set(t.strip() for t in received_type.split(","))
    input_types = set(t.strip() for t in input_type.split(","))

    # If any of the types is '*', we can return True immediately; this is the 'Any' type.
    if IO.AnyType.io_type in received_types or IO.AnyType.io_type in input_types:
        return True

    if strict:
        # In strict mode, all received types must be in the input types
        return received_types.issubset(input_types)
    else:
        # In non-strict mode, there must be at least one type in common
        return len(received_types.intersection(input_types)) > 0
