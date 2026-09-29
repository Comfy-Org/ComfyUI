"""The inputs a node actually has, with V3 dynamic inputs finalized.

V3 nodes build their dynamic inputs (Autogrow and friends) from the live prompt,
so an expanded input such as `routes.route_a` exists only there: the static
`INPUT_TYPES` schema has neither that name nor its flags, `lazy` included. The
scheduler, the validator and execution all have to agree on that list, so they
read it from here instead of each reaching into the V3 internals.
"""
from comfy_api.internal import _ComfyNodeInternal
from comfy_api.latest import _io


def is_v3_node(class_def) -> bool:
    """Whether this node class is a V3 (ComfyNode) definition."""
    return issubclass(class_def, _ComfyNodeInternal)


def get_finalized_inputs(class_def, live_inputs, valid_inputs=None):
    """Return `(valid_inputs, hidden_inputs, v3_data)` for a node.

    `live_inputs` is the node's input mapping from the prompt. `valid_inputs`
    lets a caller pass a schema it already read from `INPUT_TYPES`; nodes that
    are not V3 get it back unchanged.
    """
    if valid_inputs is None:
        valid_inputs = class_def.INPUT_TYPES()
    if not is_v3_node(class_def):
        return valid_inputs, {}, {}
    return _io.get_finalized_class_inputs(valid_inputs, live_inputs)
