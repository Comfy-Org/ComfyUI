from comfy_execution.graph_utils import is_link


def _linked_children(dynprompt):
    import nodes
    from comfy_execution.graph import get_input_info

    children = {}
    for candidate_id in dynprompt.all_node_ids():
        node = dynprompt.get_node(candidate_id)
        class_def = nodes.NODE_CLASS_MAPPINGS[node["class_type"]]
        for input_name, value in node.get("inputs", {}).items():
            if is_link(value):
                _, _, input_info = get_input_info(class_def, input_name)
                if input_info is not None and input_info.get("nonNavigable", False):
                    continue
                children.setdefault(value[0], set()).add(candidate_id)
    return children


def _walk_graph(start_ids, next_ids):
    found = set()
    stack = list(start_ids)
    while stack:
        node_id = stack.pop()
        if node_id in found:
            continue
        found.add(node_id)
        stack.extend(next_ids(node_id))
    return found


def descendants(dynprompt, node_id):
    children = _linked_children(dynprompt)
    return _walk_graph(children.get(node_id, ()), lambda child_id: children.get(child_id, ()))


def loop_projection(dynprompt, opener_id):
    children = _linked_children(dynprompt)
    reachable = _walk_graph(children.get(opener_id, ()), lambda node_id: children.get(node_id, ()))
    parent_counts = {node_id: 0 for node_id in reachable}
    incoming_states = {node_id: [] for node_id in reachable}
    for parent_id in reachable:
        for child_id in children.get(parent_id, ()):
            if child_id in reachable:
                parent_counts[child_id] += 1
    for node_id in children.get(opener_id, ()):
        incoming_states[node_id].append(((opener_id,), frozenset()))

    projected = set()
    close_nodes = set()
    variable_nodes = set()
    ready = [node_id for node_id, count in parent_counts.items() if count == 0]
    processed = 0
    while ready:
        node_id = ready.pop()
        processed += 1
        states = incoming_states[node_id]
        scopes = max((state[0] for state in states), key=len)
        closed_scopes = frozenset().union(*(state[1] for state in states))
        if any(scopes[:len(candidate_scopes)] != candidate_scopes for candidate_scopes, _ in states):
            raise ValueError(f"Node {node_id} belongs to incompatible nested loop scopes")
        if set(scopes).intersection(closed_scopes):
            raise ValueError(
                f"Node {node_id} routes around a Close Loop; all of its looped inputs must pass through the close"
            )

        if opener_id in scopes:
            projected.add(node_id)

        class_type = dynprompt.get_node(node_id)["class_type"]
        next_scopes = scopes
        next_closed_scopes = closed_scopes
        if class_type == "Loop":
            next_scopes = (*scopes, node_id)
        elif class_type == "LoopVariable" and scopes and scopes[-1] == opener_id:
            variable_nodes.add(node_id)
        elif class_type == "CloseLoop" and scopes:
            owner_id = scopes[-1]
            next_scopes = scopes[:-1]
            next_closed_scopes = closed_scopes.union((owner_id,))
            if owner_id == opener_id:
                close_nodes.add(node_id)

        for child_id in children.get(node_id, ()):
            if child_id not in reachable:
                continue
            incoming_states[child_id].append((next_scopes, next_closed_scopes))
            parent_counts[child_id] -= 1
            if parent_counts[child_id] == 0:
                ready.append(child_id)

    if processed != len(reachable):
        raise ValueError(f"Loop {opener_id} contains a dependency cycle")

    return projected, close_nodes, variable_nodes
