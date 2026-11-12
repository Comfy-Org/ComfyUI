import inspect
import sys
from pathlib import Path
from typing import Any, TypedDict

from comfy_api.internal.singleton import ProxiedSingleton
from comfy_api.latest import io

from ._execution import running_node

_COMFY_SOURCES = ("nodes.py", "comfy_extras", "comfy_api_nodes")


class NodeRunOutput(TypedDict):
    """What `eng.node.run_output` returns: `result` as `eng.node.run` gives it, and `ui`."""

    result: tuple[Any, ...]
    ui: dict[str, list[Any]]


def _defined_by_comfyui(node_class: type) -> bool:
    import nodes

    # Not RELATIVE_PYTHON_MODULE: ComfyUI's loader writes that onto any class a pack lists.
    root = Path(nodes.__file__).resolve().parent
    file = getattr(sys.modules.get(node_class.__module__), "__file__", None)
    if file is None:
        return False
    source = Path(file).resolve()
    return (
        source.is_relative_to(root)
        and source.relative_to(root).parts[0] in _COMFY_SOURCES
    )


def _replaced_output_order(
    node: str, new_node: str, replacements: list[io.NodeReplace], outputs: int
) -> list[int]:
    """Which of `new_node`'s `outputs` each output of the removed `node` is, as ComfyUI re-points
    a workflow's links: an output the replacement does not map keeps its index."""
    replacement = next(entry for entry in replacements if entry.new_node_id == new_node)
    new_index = {
        output["old_idx"]: output["new_idx"]
        for output in replacement.output_mapping or []
    }
    order = [new_index.get(i, i) for i in range(max([outputs - 1, *new_index]) + 1)]
    for old_index, index in enumerate(order):
        if index >= outputs:
            raise ValueError(
                f"{node!r} output {old_index} has no counterpart in its replacement "
                f"{new_node!r}"
            )
    return order


def _resolve(
    node: str, inputs: dict[str, Any], node_id: str
) -> tuple[dict[str, Any], type, list[int] | None]:
    """The one-node prompt for `node` after ComfyUI's node replacements, the class that runs it,
    and the order of its outputs if a replacement took its place."""
    import nodes
    from server import PromptServer

    # The prompt format reserves lists for links; a list value is written {"__value__": [...]}.
    prompt = {
        node_id: {
            "class_type": node,
            "inputs": {
                name: {"__value__": value} if isinstance(value, list) else value
                for name, value in inputs.items()
            },
        }
    }
    manager = PromptServer.instance.node_replace_manager
    manager.apply_replacements(prompt)
    class_type = prompt[node_id]["class_type"]
    node_class = nodes.NODE_CLASS_MAPPINGS.get(class_type)
    if node_class is None or not _defined_by_comfyui(node_class):
        raise ValueError(f"{node!r} is not one of ComfyUI's own nodes")
    if inspect.iscoroutinefunction(getattr(node_class, node_class.FUNCTION)):
        raise ValueError(f"{node!r} runs asynchronously and cannot be run by a call")
    order = (
        None
        if class_type == node
        else _replaced_output_order(
            node,
            class_type,
            manager.get_replacement(node),
            len(node_class.RETURN_TYPES),
        )
    )
    return prompt, node_class, order


async def _run(node: str, inputs: dict[str, Any]) -> NodeRunOutput:
    from execution import get_input_data, get_output_data, validate_inputs

    context = running_node()
    prompt, node_class, order = _resolve(node, inputs, context.node_id)
    valid, errors, _ = await validate_inputs(
        context.prompt_id, prompt, context.node_id, {}
    )
    if not valid:
        raise ValueError(
            f"{node!r}: "
            + "; ".join(f"{error['message']}: {error['details']}" for error in errors)
        )
    values, _, v3_data = get_input_data(
        prompt[context.node_id]["inputs"], node_class, context.node_id
    )
    outputs, ui, expanded, _ = await get_output_data(
        context.prompt_id, context.node_id, node_class(), values, v3_data=v3_data
    )
    if expanded:
        raise ValueError(
            f"{node!r} expands into other nodes and cannot be run by a call"
        )

    is_list = getattr(node_class, "OUTPUT_IS_LIST", None) or [False] * len(outputs)
    result = tuple(
        value if listed else value[0] for value, listed in zip(outputs, is_list)
    )
    if order is not None:
        result = tuple(result[index] for index in order)
    return {"result": result, "ui": ui}


class Node(ProxiedSingleton):
    async def run(self, node: str, inputs: dict[str, Any]) -> tuple[Any, ...]:
        """Run one of ComfyUI's own nodes inside the running node and return its outputs.

        `node` is a node type such as "KSampler" (a saved workflow's `class_type`), and `inputs`
        its input values by name, each what the node takes in a workflow: a LATENT is
        `{"samples": tensor}`, not a bare tensor. As for a workflow, ComfyUI maps a removed node to its registered
        replacement (the outputs come back in the removed node's order; an output the replacement
        does not map keeps its index), converts number, text and boolean inputs with `int`,
        `float`, `str` or `bool`, validates the inputs (required inputs, number ranges, choice
        lists, the node's own validation), fills hidden inputs and runs the node once, on this
        thread. An input the node does not declare is ignored, unless the node accepts any input.
        Hidden inputs: `unique_id` is the running node's and the prompt is `{}`; every other
        hidden input (workflow, dynamic prompt, execution list, account tokens) is None, so a node
        that needs one (the loop nodes) fails. Raises `InterruptProcessingException` if the user
        cancelled the prompt.

        Not done: the result is not cached and the node's change check (`IS_CHANGED`,
        `fingerprint_inputs`) is not consulted, so the running node is cached on its own inputs
        alone. The node's lazy-input check is not called: it gets exactly the inputs given. Each
        value is one input: a node that takes lists gets one-item lists, and outputs it declares as
        lists come back as lists, so a list output cannot be passed on to a node that takes lists.
        The UI the node returns is not sent (see `run_output`); text and progress it sends while it
        runs show on the running node.

        Raises `ValueError` for an unknown node type, a node not defined by ComfyUI itself, an
        asynchronous node (most partner API nodes), invalid inputs, a node that expands into other
        nodes, or a removed node with an output its replacement has no counterpart for.
        """
        return (await _run(node, inputs))["result"]

    async def run_output(self, node: str, inputs: dict[str, Any]) -> NodeRunOutput:
        """Run an output or preview node as `run` does, and also return its UI output.

        `ui` is what the node shows (for example `{"images": [{"filename": ...,
        "subfolder": ..., "type": ...}]}`), `{}` if nothing. Return it from your
        node to show it.
        """
        return await _run(node, inputs)
