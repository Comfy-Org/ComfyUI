"""Run inside a fresh process with ComfyUI on the path: import one V1 pack with a stand-in server
(``packload.LoadedPack``) and write what ComfyUI would list for each of its nodes, plus the static
hints the converter needs.

    python describe.py <pack dir> <out.json>
"""
import ast
import inspect
import json
import os
import sys
import textwrap
import traceback
import types


def plain(value):
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):  # a pack's "any type" str subclass
        return str(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, dict):
        return {str(plain(k)): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return {"__unplain__": repr(value)[:200]}


def v1_info(node_id, cls, display_names):
    """server.py's node_info for a V1 class (python_module left out)."""
    types = cls.INPUT_TYPES()
    info = {"input": plain(types), "input_order": {k: list(v.keys()) for k, v in types.items()}}
    info["is_input_list"] = getattr(cls, "INPUT_IS_LIST", False)
    info["output"] = plain(cls.RETURN_TYPES)
    info["output_is_list"] = plain(cls.OUTPUT_IS_LIST) if hasattr(cls, "OUTPUT_IS_LIST") else [False] * len(cls.RETURN_TYPES)
    info["output_name"] = plain(cls.RETURN_NAMES) if hasattr(cls, "RETURN_NAMES") else info["output"]
    info["name"] = node_id
    info["display_name"] = display_names.get(node_id, node_id)
    info["description"] = cls.DESCRIPTION if hasattr(cls, "DESCRIPTION") else ""
    info["category"] = getattr(cls, "CATEGORY", "sd")
    info["output_node"] = bool(getattr(cls, "OUTPUT_NODE", False) is True)
    info["has_intermediate_output"] = bool(getattr(cls, "HAS_INTERMEDIATE_OUTPUT", False) is True)
    if hasattr(cls, "OUTPUT_TOOLTIPS"):
        info["output_tooltips"] = plain(cls.OUTPUT_TOOLTIPS)
    for flag, key in (("DEPRECATED", "deprecated"), ("EXPERIMENTAL", "experimental"), ("DEV_ONLY", "dev_only")):
        if getattr(cls, flag, False):
            info[key] = True
    if hasattr(cls, "API_NODE"):
        info["api_node"] = cls.API_NODE
    info["search_aliases"] = plain(getattr(cls, "SEARCH_ALIASES", []))
    if hasattr(cls, "ESSENTIALS_CATEGORY"):
        info["essentials_category"] = cls.ESSENTIALS_CATEGORY
    return info


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _sampler_constant(name, cls):
    """Recognize only references bound to the actual ComfyUI sampler registry.

    Resolve module/class dictionaries without invoking pack-defined attribute
    access. This supports import aliases without trusting matching suffixes.
    """
    module = sys.modules.get("comfy.samplers")
    if not isinstance(module, types.ModuleType) or not name:
        return None
    parts = name.split(".")
    if parts[0] in ("cls", "self"):
        return None
    function = getattr(cls.INPUT_TYPES, "__func__", cls.INPUT_TYPES)
    value = getattr(function, "__globals__", {}).get(parts[0])
    for part in parts[1:]:
        if not isinstance(value, (types.ModuleType, type)):
            return None
        value = vars(value).get(part)
    registry = vars(module)
    sampler = registry.get("KSampler")
    aliases = vars(sampler) if isinstance(sampler, type) else {}
    for constant, class_attr in (("SAMPLER_NAMES", "SAMPLERS"), ("SCHEDULER_NAMES", "SCHEDULERS")):
        candidates = (registry.get(constant), aliases.get(class_attr))
        if value is not None and any(value is candidate for candidate in candidates if candidate is not None):
            return constant
    return None


def input_hints(cls):
    """Which inputs list a model folder or ComfyUI's sampler/scheduler names, read from INPUT_TYPES's code."""
    hints = {}
    try:
        source = textwrap.dedent(inspect.getsource(cls.INPUT_TYPES))
        tree = ast.parse(source)
    except Exception:
        return hints
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            first = value.elts[0] if isinstance(value, ast.Tuple) and value.elts else value
            if isinstance(first, ast.Call) and (_dotted(first.func) or "").endswith("get_filename_list") \
                    and first.args and isinstance(first.args[0], ast.Constant):
                hints[key.value] = {"model_folder": first.args[0].value}
            constant = _sampler_constant(_dotted(first), cls)
            if constant:
                hints[key.value] = {"util_constant": constant}
    return hints


def method_info(cls, function=None):
    function = function or getattr(cls, "FUNCTION", None)
    out = {"function": function, "methods": [m for m in ("VALIDATE_INPUTS", "IS_CHANGED", "check_lazy_status") if hasattr(cls, m)]}
    try:
        out["module"] = cls.__module__
        out["qualname"] = cls.__qualname__
        out["file"] = inspect.getsourcefile(cls)
    except Exception:
        pass
    if function and hasattr(cls, function):
        try:
            out["params"] = list(inspect.signature(getattr(cls, function)).parameters)
        except (TypeError, ValueError):
            pass
    return out


def cpu_comfy(as_comfyui_loads_packs=False):
    """ComfyUI as main.py starts it, on the CPU: arguments parsed, writable folders under /tmp."""
    sys.path.insert(0, os.environ.get("COMFY_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.argv = ["main.py", "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch  # noqa: F401  ComfyUI has torch imported before any pack loads (a pack's sys.path changes can't shadow it)
    import folder_paths
    import tempfile
    scratch = os.environ.get("CONVERTER_SCRATCH") or os.path.join(tempfile.gettempdir(), "comfy_conversion")
    for name, setter in (("output", folder_paths.set_output_directory), ("temp", folder_paths.set_temp_directory),
                         ("input", folder_paths.set_input_directory), ("user", folder_paths.set_user_directory)):
        os.makedirs(os.path.join(scratch, name), exist_ok=True)
        setter(os.path.join(scratch, name))
    if as_comfyui_loads_packs:  # ComfyUI has imported nodes (comfy.samplers, comfy.sd...) before any pack loads
        import nodes  # noqa: F401


def main(pack_dir, out_path):
    cpu_comfy(as_comfyui_loads_packs=True)
    result = {"pack": pack_dir, "ok": False}
    try:
        from comfy_api.latest import _io
        from packload import LoadedPack
        pack = LoadedPack(pack_dir)
        nodes = {}
        for node_id, cls in pack.nodes.items():
            entry = {"class": getattr(cls, "__name__", str(cls))}
            try:
                if isinstance(cls, type) and issubclass(cls, _io.ComfyNode):
                    entry.update(kind="v3", info=plain(cls.GET_NODE_INFO_V1()), **method_info(cls, "execute"))
                else:
                    entry.update(kind="v1", info=v1_info(node_id, cls, pack.display_names), hints=input_hints(cls), **method_info(cls))
            except Exception as error:
                entry.update(kind="error", error=f"{type(error).__name__}: {error}"[:500])
            nodes[node_id] = entry
        result.update(ok=True, module=pack.module_name, nodes=nodes,
                      web_directory=os.path.relpath(pack.web_directory, pack.entry) if pack.web_directory else None,
                      routes=[list(r) for r in pack.routes()])
    except BaseException as error:
        result.update(error=f"{type(error).__name__}: {error}"[:2000], trace=traceback.format_exc()[-3000:])
    with open(out_path, "w") as file:
        json.dump(result, file)


if __name__ == "__main__":
    main(os.path.realpath(sys.argv[1]), sys.argv[2])
