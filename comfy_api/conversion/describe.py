"""Run inside a fresh process with ComfyUI on the path: import one V1 pack the way pack isolation
does (``packload.LoadedPack``: a stand-in server) and write what ComfyUI would list for each
of its nodes, plus the static hints the converter needs.

    python describe.py <pack dir> <out.json>
"""
import ast
import inspect
import json
import os
import sys
import textwrap
import time
import traceback
import types


def plain(value):
    if value is None or type(value) in (bool, int, float, str):
        return value
    standin = sys.modules.get("standin")
    if standin is not None and standin.contains_stand_in(value):  # a listing made from something that isn't here
        raise standin.StandInUsed(f"{value!r} is part of what this node lists")
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
        import nodes
        import utils.extra_config  # noqa: F401  main.py has: ``utils`` is ComfyUI's package from then on
        # nodes.init_external_custom_nodes, before it loads any: ``import model_management`` works in a pack
        inside = os.path.join(os.path.dirname(os.path.realpath(nodes.__file__)), "comfy")
        if inside not in sys.path:
            sys.path.insert(0, inside)
        if not getattr(cpu_comfy, "_builtins_loaded", False):
            import asyncio
            asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
            cpu_comfy._builtins_loaded = True


def _tree(root, skip=()):
    """Every file under root with its size and time, to see what a pack's import wrote."""
    seen = {}
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__pycache__" and os.path.join(folder, d) not in skip]
        for name in files + dirs:
            path = os.path.join(folder, name)
            try:
                status = os.lstat(path)
                seen[path] = (status.st_size, status.st_mtime_ns) if name in files else None
            except OSError:
                pass
    return seen


def _written(before, roots, skip=()):
    out = []
    for root in roots:
        for path, status in _tree(root, skip).items():
            if path not in before or before[path] != status:
                out.append(path)
    return sorted(out)[:60]


def _as_pack(result, entry, module_name):
    """The converter reads a pack as imported from /pack; a pack described in place (custom_nodes/<name>,
    where ComfyUI has it) is written down the same way."""
    for node in (result.get("nodes") or {}).values():
        module, file = node.get("module"), node.get("file")
        if module == module_name:
            node["module"] = "/pack"
        elif isinstance(module, str) and module.startswith(module_name + "."):
            node["module"] = "/pack" + module[len(module_name):]
        if isinstance(file, str) and os.path.realpath(file).startswith(entry + os.sep):
            node["file"] = "/pack/" + os.path.relpath(os.path.realpath(file), entry)
    result["entry"] = entry


def _find_stand_ins(pack_dir, plan, error, save, kept=None):
    """The pack doesn't load: find, one failure at a time, what it needs that isn't here (standin.py), loading
    it again in this process with each found. Returns whether it then loaded, the failure left, and rounds."""
    import standin
    from packload import LoadedPack, module_name_for
    entry = os.path.realpath(pack_dir)
    module_name = module_name_for(entry)
    blocking = plan.installers
    plan.installers = True  # from here on a pip/git run at import isn't run (kept in the plan only if one was)
    for attempt in range(1, 41):
        repair = standin.derive(error, entry, module_name, plan)
        if repair is None:
            return False, error, attempt - 1, traceback.format_exc()
        plan.add(repair, f"{type(error).__name__}: {error}")
        save(plan)
        error = None
        standin.forget(entry, module_name)
        try:
            pack = LoadedPack(pack_dir, plan)
        except BaseException as again:
            error = again
            continue
        if not blocking and not pack.stand_ins.blocked and not any(e["stand_in"] == ["installers"] for e in plan.evidence):
            plan.installers = False
        save(plan)
        if kept is not None:
            kept.append(pack)
        return True, None, attempt, ""
    return False, error, 40, traceback.format_exc()


def _find_caught(pack_dir, plan, pack, seconds):
    """The pack loads, having caught failures of its own imports (``try: from .nodes import * / except: print``):
    it carries on with nodes missing, or with none. Find what stands in the way of those imports, as for a pack
    that doesn't load. Returns whether the plan now lists more nodes (it is changed only then)."""
    import standin
    from packload import LoadedPack
    entry, module_name = os.path.realpath(pack_dir), pack.module_name
    start, had, until, tried = plan.to_json(), len(pack.nodes), time.monotonic() + seconds, 0
    while pack is not None and tried < 12 and time.monotonic() < until:
        repair = None
        for error, own in standin.failed_imports(pack.stand_ins.caught, entry):
            if own or not pack.nodes:
                repair = standin.derive(error, entry, module_name, plan)
                if repair is not None:
                    plan.add(repair, f"caught by the pack: {type(error).__name__}: {error}")
                    break
        if repair is None:
            break
        tried += 1
        standin.forget(entry, module_name)
        try:
            pack = LoadedPack(pack_dir, plan)
        except BaseException as error:
            kept = []
            try:
                _find_stand_ins(pack_dir, plan, error, lambda p: None, kept)
            except BaseException:
                pass
            pack = kept[0] if kept else None
    if pack is not None and len(pack.nodes) > had:
        return True
    plan.__init__(start)
    return False


def main(pack_dir, out_path, plan_path=None):
    cpu_comfy(as_comfyui_loads_packs=True)
    result = {"pack": pack_dir, "ok": False}
    try:
        import standin
        from comfy_api.latest import _io
        from packload import LoadedPack
        plan = standin.Plan.load(plan_path)
        entry = os.path.realpath(pack_dir)
        watched = [entry] + ([os.environ["COMFY_ROOT"]] if os.environ.get("CONVERTER_WATCH_CORE") and os.environ.get("COMFY_ROOT") else [])
        skip = (os.path.join(os.environ.get("COMFY_ROOT") or "/nonexistent", "custom_nodes"),)
        before = {}
        for root in watched:
            before.update(_tree(root, () if root == entry else skip))
        try:
            pack = LoadedPack(pack_dir, plan)
        except BaseException as error:
            if not plan_path:
                raise
            result.update(error=f"{type(error).__name__}: {error}"[:2000], trace=traceback.format_exc()[-3000:])
            try:
                loaded, left, attempts, trace = _find_stand_ins(pack_dir, plan, error, lambda p: p.save(plan_path))
            except BaseException as failure:
                loaded, left, attempts, trace = False, failure, -1, traceback.format_exc()
            result.update(found=loaded, attempts=attempts, stand_ins=plan.to_json())
            if left is not None:  # what is still in the way, with everything found so far stood in
                result.update(error_left=f"{type(left).__name__}: {left}"[:2000],
                              trace_left="".join(traceback.format_exception(type(left), left, left.__traceback__))[-3000:])
            with open(out_path, "w") as file:
                json.dump(result, file)
            return
        nodes, folders = {}, standin.Deriver(entry, pack.module_name, plan)
        needed = False
        for node_id, cls in pack.nodes.items():
            entry_info = {"class": getattr(cls, "__name__", str(cls))}
            try:
                if isinstance(cls, type) and issubclass(cls, _io.ComfyNode):
                    entry_info.update(kind="v3", info=plain(cls.GET_NODE_INFO_V1()), **method_info(cls, "execute"))
                else:
                    entry_info.update(kind="v1", info=v1_info(node_id, cls, pack.display_names), hints=input_hints(cls), **method_info(cls))
            except Exception as error:
                entry_info = {"class": entry_info["class"], "kind": "error", "error": f"{type(error).__name__}: {error}"[:500]}
                if isinstance(error, standin.StandInUsed):
                    entry_info["stand_in"] = True
                repair = folders._in_pack_code(error) if plan_path else None  # a folder its inputs list, not there
                if repair and plan.add(repair, f"{node_id}: {type(error).__name__}: {error}"):
                    needed = True
            nodes[node_id] = entry_info
        if needed:  # described again, in a fresh process, with those folders there
            plan.save(plan_path)
            result.update(error="its nodes list folders that aren't there", found=True, stand_ins=plan.to_json())
            with open(out_path, "w") as file:
                json.dump(result, file)
            return
        for name, error in pack.unlisted.items():
            nodes.setdefault(name, {"class": name, "kind": "error", "error": error})
        result.update(ok=True, module=pack.module_name, nodes=nodes,
                      web_directory=os.path.relpath(pack.web_directory, pack.entry) if pack.web_directory else None,
                      routes=[list(r) for r in pack.routes()])
        if plan:
            result.update(stand_ins=plan.to_json(), commands_not_run=pack.stand_ins.blocked[:40],
                          passed_over=pack.stand_ins.skipped[:40], installed=standin.installed(plan))
        made = [pack.stand_ins._path(rel) for rel in plan.mkdirs]  # folders the plan put there: not the pack's writing
        written = [os.path.relpath(p, entry) if p.startswith(entry + os.sep) else p for p in
                   _written(before, [entry], ()) + _written(before, watched[1:], skip)
                   if not any(m == p or m.startswith(p.rstrip(os.sep) + os.sep) for m in made)]
        if written:
            result["written_at_load"] = written
        _as_pack(result, pack.entry, pack.module_name)
        if plan_path and not plan.leave_caught and standin.failed_imports(pack.stand_ins.caught, entry):
            with open(out_path, "w") as file:  # as it loads now, whatever comes of looking further
                json.dump(result, file)
            was = plan.to_json()
            if _find_caught(pack_dir, plan, pack, float(os.environ.get("CONVERTER_CAUGHT_SECONDS", "90"))):
                plan.save(plan_path)  # described again, in a fresh process, with what its own imports needed
                result = {"pack": pack_dir, "ok": False, "found": True, "loads_with": was, "stand_ins": plan.to_json(),
                          "error": "it loads with nodes missing: imports of its own that fail are caught by the pack"}
    except BaseException as error:
        result.update(error=f"{type(error).__name__}: {error}"[:2000], trace=traceback.format_exc()[-3000:])
    with open(out_path, "w") as file:
        json.dump(result, file)


if __name__ == "__main__":
    main(os.path.realpath(sys.argv[1]), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    sys.stdout.flush()
    os._exit(0)  # a pack's stray thread mustn't keep the process alive
