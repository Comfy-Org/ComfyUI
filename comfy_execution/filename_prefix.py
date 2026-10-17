import re


_NODE_INPUT_PATTERN = re.compile(r"%([^%.]+)\.([^%.]+)%")


def _normalize_node_name(name: str) -> str:
    return "".join(char.casefold() for char in name if char.isalnum())


def format_filename_prefix(
    filename_prefix: str, prompt: dict | None, node_value_resolver=None
) -> str:
    if not prompt:
        return filename_prefix

    nodes_by_name = {}
    for node in prompt.values():
        node_name = node.get("class_type")
        if node_name is not None:
            nodes_by_name.setdefault(_normalize_node_name(node_name), []).append(node)

    def replace(match: re.Match) -> str:
        nodes = nodes_by_name.get(_normalize_node_name(match.group(1)), [])
        if len(nodes) != 1:
            return match.group(0)

        value = nodes[0].get("inputs", {}).get(match.group(2))
        if (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and isinstance(value[0], str)
            and isinstance(value[1], int)
        ):
            value = (
                node_value_resolver(value[0], value[1]) if node_value_resolver else None
            )
        if value is None:
            return match.group(0)
        return str(value)

    return _NODE_INPUT_PATTERN.sub(replace, filename_prefix)
