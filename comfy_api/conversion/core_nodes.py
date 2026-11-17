"""ComfyUI's own nodes, as the converter calls them: class name -> node id, input order, function.

    python core_nodes.py core_nodes.json     (in the converter's container: ComfyUI on the CPU)

V1 code calls a core node as ``nodes.LoraLoader().load_lora(model, clip, name, 1.0, 1.0)``; the
converted code runs it with ``eng.node.run("LoraLoader", {...})``, naming the positional arguments
from this table.
"""
import asyncio
import inspect
import json
import sys

from describe import cpu_comfy


def positional(function):
    """The names a call binds positionally, in order (without self/cls); None when it takes *args or isn't
    a function we can read."""
    if function is None:
        return None
    try:
        signature = inspect.signature(function)  # a classmethod or a method read from the class: no cls/self
    except (TypeError, ValueError):
        return None
    names = []
    for index, parameter in enumerate(signature.parameters.values()):
        if parameter.kind is parameter.VAR_POSITIONAL:
            return None
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            if index == 0 and parameter.name in ("self", "cls"):
                continue
            names.append(parameter.name)
    return names


def main(out_path):
    cpu_comfy()
    import nodes
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    table = {}
    for node_id, cls in nodes.NODE_CLASS_MAPPINGS.items():
        try:
            if hasattr(cls, "GET_SCHEMA"):
                schema = cls.GET_SCHEMA()
                inputs = [i.id for i in schema.inputs]
                function = "execute"
            else:
                types = cls.INPUT_TYPES()
                inputs = list((types.get("required") or {})) + list((types.get("optional") or {}))
                function = getattr(cls, "FUNCTION", None)
        except Exception as error:
            print(f"skipped {node_id}: {error}", file=sys.stderr)  # noqa: T201
            continue
        entry = {"id": node_id, "inputs": inputs, "function": function, "module": cls.__module__}
        params = positional(getattr(cls, function, None) if function else None)
        if params is not None and params != inputs[:len(params)]:
            # V1 code passes arguments in the method's order, which may not be the schema's:
            # VAEEncode's schema is (pixels, vae), but its method is encode(self, vae, pixels)
            entry["params"] = params
        table.setdefault(f"{cls.__module__}.{cls.__name__}", entry)
    with open(out_path, "w") as file:
        json.dump(table, file, indent=0, sort_keys=True)
    print(f"{len(table)} core nodes")  # noqa: T201


if __name__ == "__main__":
    main(sys.argv[1])
