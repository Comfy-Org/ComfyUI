"""Run inside a fresh, network-less process with ComfyUI on the path: check one converted pack.

    python verify.py <original pack dir> <converted pack dir> <describe.json> <out.json>

1. The V2 pack is valid: its declarations (v2/pyproject.toml) read as ComfyUI reads them.
2. Each node's schema is the original's: ComfyUI's listing of the converted node equals the original's.
3. Each node whose inputs can be made up (images, masks, latents, numbers, strings, choices...) gives
   the original's outputs, on the same seeded inputs.
"""
import asyncio
import copy
import inspect
import json
import os
import random
import re
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from describe import cpu_comfy, plain  # noqa: E402

NODE_SECONDS = 30



def _seed():
    import numpy as np
    import torch
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


MODEL_KINDS = {"MODEL", "CLIP", "VAE", "CONTROL_NET", "CLIP_VISION", "CLIP_VISION_OUTPUT", "UPSCALE_MODEL",
               "STYLE_MODEL", "GLIGEN", "PHOTOMAKER", "HOOKS"}


def _plain(value, depth=0):
    """Plain data (tensors, arrays, numbers, strings, and lists/dicts of them): safe to pickle across and compare."""
    import numpy as np
    import torch
    if isinstance(value, (torch.Tensor, np.ndarray, str, int, float, bool, bytes, type(None))):
        return True
    if depth < 8 and isinstance(value, dict):
        return all(isinstance(k, (str, int)) and _plain(v, depth + 1) for k, v in value.items())
    if depth < 8 and type(value) in (list, tuple):
        return all(_plain(v, depth + 1) for v in value)
    return False


class _OutputList:
    """The elements of one producer's OUTPUT_IS_LIST socket, already flattened by the executor."""

    def __init__(self, value):
        self.value = value


def _input_lists(inputs, already_list=()):
    """Each ordinary socket is one list element; a list-output link is already a sequence."""
    return {name: value if name in already_list else [value] for name, value in inputs.items()}


def make_inputs(info, provided=None):
    """Inputs for the required ones, or None with the reason a node can't be run on made-up data.
    ``provided``: values for pack-specific types, made by the pack's own nodes (``{type: value}``)."""
    import torch
    g = torch.Generator().manual_seed(1)
    made = {}
    already_list = set()
    for name, spec in (info["input"].get("required") or {}).items():
        spec = spec if isinstance(spec, list) else [spec]
        kind, options = spec[0], (spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {})
        if kind == "COMBO":  # V3's form: the choices are in the options
            kind = list(options.get("options") or [])
        if isinstance(kind, list):
            if not kind:
                return None, f"{name}: an empty choice"
            made[name] = options.get("default", kind[0]) if options.get("default", kind[0]) in kind else kind[0]
        elif kind == "INT":
            value = options.get("default", 0)
            made[name] = max(options.get("min", value), min(options.get("max", value), value))
        elif kind == "FLOAT":
            value = options.get("default", 0.0)
            made[name] = float(max(options.get("min", value), min(options.get("max", value), value)))
        elif kind == "STRING":
            made[name] = options.get("default", "a test string") or "a test string"
        elif kind == "BOOLEAN":
            made[name] = bool(options.get("default", False))
        elif kind == "IMAGE":
            made[name] = torch.rand(1, 64, 64, 3, generator=g)
        elif kind == "MASK":
            made[name] = torch.rand(1, 64, 64, generator=g)
        elif kind == "LATENT":
            made[name] = {"samples": torch.randn(1, 4, 8, 8, generator=g)}
        elif kind == "CONDITIONING":
            made[name] = [[torch.randn(1, 77, 768, generator=g), {"pooled_output": torch.randn(1, 1280, generator=g)}]]
        elif kind == "AUDIO":
            made[name] = {"waveform": torch.randn(1, 1, 8000, generator=g), "sample_rate": 8000}
        elif kind == "SIGMAS":
            made[name] = torch.linspace(14.6, 0.0, 11)
        elif provided and isinstance(kind, str) and kind in provided:
            value = provided[kind]
            if isinstance(value, _OutputList):
                if info.get("is_input_list"):
                    made[name] = copy.deepcopy(value.value)
                    already_list.add(name)
                elif len(value.value) == 1:
                    made[name] = copy.deepcopy(value.value[0])
                else:
                    return None, f"{name}: list output has {len(value.value)} items; mapped execution not verified"
            else:
                made[name] = copy.deepcopy(value)
        else:
            return None, f"{name}: needs a {kind}"
    # Supply visible optional widgets, as a UI prompt does. Cable-only, lazy,
    # model and dynamic inputs need their own fixtures and remain absent.
    for name, spec in (info["input"].get("optional") or {}).items():
        spec = spec if isinstance(spec, list) else [spec]
        if not spec:
            continue
        kind = spec[0]
        options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if options.get("forceInput") or options.get("lazy"):
            continue
        if kind == "COMBO":
            kind = options.get("options")
        if isinstance(kind, list):
            if kind and all(isinstance(value, (str, int, float, bool)) for value in kind):
                default = options.get("default", kind[0])
                made[name] = default if default in kind else kind[0]
        elif kind == "STRING":
            value = options.get("default", "")
            if isinstance(value, str):
                made[name] = value
        elif kind == "BOOLEAN":
            value = options.get("default", False)
            if isinstance(value, bool):
                made[name] = value
        elif kind in ("INT", "FLOAT"):
            value = options.get("default", 0)
            minimum, maximum = options.get("min", value), options.get("max", value)
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (value, minimum, maximum)) and minimum <= maximum:
                value = max(minimum, min(maximum, value))
                made[name] = int(value) if kind == "INT" else float(value)
    return (_input_lists(made, already_list) if info.get("is_input_list") else made), None


def _same(a, b, path="output"):
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return f"{path}: a tensor and a {type(b).__name__ if isinstance(a, torch.Tensor) else type(a).__name__}"
        if a.shape != b.shape or a.dtype != b.dtype:
            return f"{path}: {tuple(a.shape)} {a.dtype} vs {tuple(b.shape)} {b.dtype}"
        return None if torch.equal(a.cpu(), b.cpu()) else f"{path}: values differ"
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return None if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b) \
            else f"{path}: arrays differ"
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return f"{path}: keys {sorted(map(str, a))} vs {sorted(map(str, b))}"
        for k in a:
            diff = _same(a[k], b[k], f"{path}[{k!r}]")
            if diff:
                return diff
        return None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: {len(a)} items vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            diff = _same(x, y, f"{path}[{i}]")
            if diff:
                return diff
        return None
    if isinstance(a, _TypeOnly) or isinstance(b, _TypeOnly):
        return None if a == b else f"{path}: {getattr(a, 'name', type(a).__name__)} vs {getattr(b, 'name', type(b).__name__)}"
    if type(a).__module__ not in ("builtins",) or type(b).__module__ not in ("builtins",):
        return None if type(a).__name__ == type(b).__name__ else f"{path}: {type(a).__name__} vs {type(b).__name__}"
    return None if a == b else f"{path}: {a!r} vs {b!r}"[:300]


_TIMES = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T_\-]?\d{2}[:\-]?\d{2}(?:[:\-]?\d{2}(?:\.\d+)?)?)?"
                    r"|(?<!\d)\d{1,2}:\d{2}:\d{2}(?:\.\d+)?(?!\d)|(?<!\d)\d{8}[_\-]?\d{6}(?!\d)"
                    r"|(?<!\d)\d{6}[_\-]?\d{6}(?!\d)"  # YYMMDDHHMMSS (a dated file name)
                    r"|(?<!\d)1[6-9]\d{8}(?:\d{3})?(?!\d)"  # Unix time in seconds or milliseconds
                    r"|(?<!\d)20\d{12}(?!\d)")  # YYYYMMDD + 6 digits (a time seed)
# an int that is a clock reading: Unix time in seconds or milliseconds, or a YYYYMMDD... time seed
_TIME_INT = re.compile(r"^(?:1[6-9]\d{8}(?:\d{3})?|20\d{12})$")


def _mask_times(value, depth=0, roots=()):
    """The value with dates and clock times in its text replaced: a node that prints the time when it runs (a
    capture stamp, a dated file name) gives different text in the original's run and the converted pack's, seconds
    apart, though the two back-to-back runs of the original agree. ``roots``: the folders the original and the
    converted pack run from, which a node printing its own path shows differently; both read as ``<pack>``."""
    if depth > 20:
        return value
    if isinstance(value, str):
        for root in sorted(roots, key=len, reverse=True):
            value = value.replace(root, "<pack>")
        return _TIMES.sub("<time>", value)
    if isinstance(value, int) and not isinstance(value, bool) and _TIME_INT.match(str(value)):
        return "<time>"
    if isinstance(value, dict):
        return {k: _mask_times(v, depth + 1, roots) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_mask_times(v, depth + 1, roots) for v in value)
    return value


def _schema_diff(original, converted):
    a, b = dict(original), plain(converted)
    for info in (a, b):
        info.pop("python_module", None)
    diffs = []
    flags = {"api_node", "deprecated", "dev_only", "experimental", "output_node", "has_intermediate_output", "is_input_list"}
    for key in sorted(set(a) | set(b)):
        if key == "input":
            continue
        x, y = a.get(key), b.get(key)
        if key in flags:  # absent and False list the same
            same = bool(x) == bool(y)
        elif key == "output" and isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            # V3 lists a choice output as COMBO; V1 as its (possibly empty) list of choices
            same = all(p == q or (isinstance(p, list) and q == "COMBO") or (p == "COMBO" and isinstance(q, list))
                       for p, q in zip(x, y))
        elif key in ("output_name", "output_tooltips", "output_is_list") and isinstance(a.get("output"), list) \
                and isinstance(x, (list, tuple, type(None))) and isinstance(y, (list, tuple, type(None))):
            # one entry per output: more than that is never read (a V1 node with RETURN_NAMES longer than
            # RETURN_TYPES), a missing tooltip is None, a missing list flag False, a missing name the type's
            outputs = a["output"]
            fill = {"output_name": None, "output_tooltips": None, "output_is_list": False}[key]

            def per_output(values):
                values = (list(values or []) + [fill] * len(outputs))[:len(outputs)]
                if key == "output_name":
                    values = [v if v is not None else t for v, t in zip(values, outputs)]
                return values
            same = per_output(x) == per_output(y)
        elif key == "search_aliases":  # absent, [] and [None, ...] list the same
            same = not any(x or []) and not any(y or []) or x == y
        elif key == "input_order":
            same = {k: v for k, v in (x or {}).items() if k != "hidden" and v} == \
                   {k: v for k, v in (y or {}).items() if k != "hidden" and v}
        else:
            same = x == y
        if not same:
            diffs.append(f"{key}: {x!r} vs {y!r}"[:300])
    for section in ("required", "optional"):  # hidden inputs aren't listed to users; V3 output nodes add two
        x, y = (a.get("input") or {}).get(section) or {}, (b.get("input") or {}).get(section) or {}
        if list(x) != list(y):
            diffs.append(f"input.{section}: {list(x)} vs {list(y)}"[:300])
            continue
        for name in x:
            sx, sy = x[name], y[name]
            tx, ty = sx[0] if isinstance(sx, list) else sx, sy[0] if isinstance(sy, list) else sy
            ox = dict(sx[1]) if isinstance(sx, list) and len(sx) > 1 and isinstance(sx[1], dict) else {}
            oy = dict(sy[1]) if isinstance(sy, list) and len(sy) > 1 and isinstance(sy[1], dict) else {}
            for options in (ox, oy):
                options.pop("optional", None)
            if ty == "COMBO" and isinstance(tx, list) and oy.get("options") == tx:  # V3 lists a choice as COMBO
                ty = tx
                oy.pop("options", None)
            if tx == "COMBO" and isinstance(ty, list) and ox.get("options") == ty:
                tx = ty
                ox.pop("options", None)
            for key in set(ox) ^ set(oy):  # an option only one side lists as False / empty means nothing
                if not (ox.get(key) or oy.get(key)):
                    ox.pop(key, None), oy.pop(key, None)
            if tx != ty:
                diffs.append(f"input.{section}.{name}: type {tx!r} vs {ty!r}"[:300])
            elif ox != oy:
                diffs.append(f"input.{section}.{name}: options {ox} vs {oy}"[:300])
    return diffs


# --------------------------------------------------------------------------- #
# What nodes really do: file writes and network connections, seen while they run
# --------------------------------------------------------------------------- #
_AUDIT = {"active": False, "roots": (), "events": []}
_WRITE_EVENTS = {"os.remove", "os.rename", "os.mkdir", "os.rmdir", "shutil.copyfile", "shutil.copytree",
                 "shutil.move", "shutil.rmtree", "os.truncate", "os.link", "os.symlink"}


def _audit(event, args):
    if not _AUDIT["active"] or len(_AUDIT["events"]) > 500:
        return
    if event == "open":
        path, mode, flags = (list(args) + [None, None, None])[:3]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
        if not writing or not isinstance(path, (str, bytes, os.PathLike)):
            return
    elif event == "socket.connect":
        path = str(args[1]) if len(args) > 1 else "?"
    elif event in _WRITE_EVENTS:
        path = args[0] if args else "?"
    else:
        return
    frame = sys._getframe(1)
    while frame is not None:  # the nearest code that isn't a library: the pack's, or ComfyUI's own
        name = frame.f_code.co_filename
        if not ("site-packages" in name or "/lib/python3" in name or name.startswith("<") or name == __file__):
            break
        frame = frame.f_back
    if frame is None:
        return
    name = frame.f_code.co_filename
    root = next((r for r in _AUDIT["roots"] if name.startswith(r + os.sep)), None)
    if root is None:
        return  # ComfyUI writing for the pack (eng.output, the ui savers): allowed
    _AUDIT["events"].append({"event": event, "path": os.fsdecode(path) if isinstance(path, bytes) else str(path),
                             "file": os.path.relpath(name, root), "line": frame.f_lineno})


def _audited(roots):
    if not getattr(_audited, "installed", False):
        sys.addaudithook(_audit)
        _audited.installed = True
    _AUDIT["roots"] = tuple(os.path.realpath(r) for r in roots)


def _call_with_timeout(fn):
    """Each call on its own thread: a node stuck waiting (for a user, say) can't hold up the rest."""
    import threading
    box = {}

    def run():
        _AUDIT["active"] = True
        try:
            box["result"] = fn()
        except BaseException as error:
            box["error"] = error
        finally:
            _AUDIT["active"] = False
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(NODE_SECONDS)
    if thread.is_alive():
        raise TimeoutError(f"no result in {NODE_SECONDS}s")
    if "error" in box:
        raise box["error"]
    return box["result"]


def _fresh_folders():
    """Empty output and temp folders before each run, so file counters start the same for both packs."""
    import shutil
    import folder_paths
    for folder in (folder_paths.get_output_directory(), folder_paths.get_temp_directory()):
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)


def _original_output(cls, function, inputs, hidden):
    _fresh_folders()
    _seed()
    def invoke():
        call_hidden = _input_lists(hidden) if getattr(cls, "INPUT_IS_LIST", False) else hidden
        result = getattr(cls(), function)(**inputs, **call_hidden)
        if inspect.isawaitable(result):
            # Run the await inside the existing timed worker, including async
            # exceptions and cancellation. Sync outputs keep their old path.
            return asyncio.run(asyncio.wait_for(result, timeout=NODE_SECONDS))
        return result

    result = _call_with_timeout(invoke)
    if isinstance(result, dict):
        return tuple(result.get("result") or ()), result.get("ui")
    if hasattr(result, "result"):
        return tuple(result.result or ()), getattr(result, "ui", None)
    return tuple(result or ()), None


def _converted_output(cls, inputs, hidden_values, module):
    from comfy_api.latest import _io, _runtime

    _fresh_folders()
    _seed()
    clone = cls.PREPARE_CLASS_CLONE({"hidden_inputs": {_io.Hidden.prompt: hidden_values.get("prompt"),
                                                        _io.Hidden.unique_id: hidden_values.get("unique_id"),
                                                        _io.Hidden.extra_pnginfo: hidden_values.get("extra_pnginfo")}})
    plan = _runtime.ExecutionPlan(prompt_id="verify", node_id="1", node_type=cls.__name__, node_module=module)

    async def go():
        with _runtime.bind_runtime(_runtime.InProcessRefResolver(), None, _runtime.EngineCalls(), pack=module, plan=plan):
            result = clone.execute(**inputs)  # a V3 node's execute may be plain or async
            return (await result) if inspect.isawaitable(result) else result
    _AUDIT["active"] = True
    try:
        output = asyncio.run(asyncio.wait_for(go(), NODE_SECONDS))
    finally:
        _AUDIT["active"] = False
    ui = output.ui.as_dict() if hasattr(output.ui, "as_dict") else output.ui
    return tuple(output.result or ()), ui


class _TypeOnly:
    """An output that isn't data (a model, say): compared by its type's name."""
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, _TypeOnly) and other.name == self.name


def _comparable(value, depth=0):
    import numpy as np
    import torch
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value.copy()  # a copy: nodes mutate in place
    if depth < 8 and isinstance(value, dict):
        return {k: _comparable(v, depth + 1) for k, v in value.items()}
    if depth < 8 and isinstance(value, (list, tuple)):
        return type(value)(_comparable(v, depth + 1) for v in value) if type(value) in (list, tuple) else \
            [_comparable(v, depth + 1) for v in value]
    if type(value).__module__ == "builtins" and isinstance(value, (str, int, float, bool, bytes, type(None))):
        return value
    return _TypeOnly(type(value).__name__)


def _route_reply(response):
    """An aiohttp handler's answer as (status, content type, body), or what it was if not JSON/text."""
    from aiohttp import web
    if isinstance(response, web.Response):
        content_type = "application/json" if response.content_type == "application/json" else "text/plain"
        try:
            return [response.status, content_type, response.text or ""]
        except Exception:
            return [response.status, "binary", ""]
    return [None, type(response).__name__, ""]


class _RouteRequest:
    """A GET with no query or body, as a V2 route's request (the converter's _v2_routes.Request)."""

    def __init__(self, path):
        self.method, self.path, self.query, self.match_info, self.headers = "GET", path, {}, {}, {}
        self.rel_url = type("URL", (), {"path": path, "query": {}})()
        self.can_read_body = self.body_exists = False
        self.content_type = "text/plain"

    async def json(self, **kwargs):
        return None

    async def text(self):
        return ""

    async def read(self):
        return b""

    async def post(self):
        return {}


def _get_routes(describe):
    return sorted({p for m, p in describe.get("routes") or [] if m.upper() == "GET" and p.startswith("/")
                   and not any(c in p for c in "{}?#")})


def _hidden(entry_info):
    names = {name: spec[0] if isinstance(spec, list) else spec
             for name, spec in (entry_info["info"]["input"].get("hidden") or {}).items()}
    values = {"prompt": {}, "unique_id": "1", "extra_pnginfo": {}}
    return values, {name: values.get({"PROMPT": "prompt", "UNIQUE_ID": "unique_id", "EXTRA_PNGINFO": "extra_pnginfo"}.get(kind))
                    for name, kind in names.items()}


def _runnable(describe, provided=None):
    """(node id, describe entry, inputs) for each node that can run on made-up inputs; reasons for the rest."""
    runnable, skipped = [], {}
    for node_id, entry_info in describe.get("nodes", {}).items():
        if entry_info.get("kind") not in ("v1", "v3"):
            continue
        try:
            inputs, reason = make_inputs(entry_info["info"], provided)
        except Exception as error:
            inputs, reason = None, f"couldn't make inputs: {type(error).__name__}: {error}"[:200]
        if inputs is None:
            skipped[node_id] = reason
        else:
            runnable.append((node_id, entry_info, inputs))
    return runnable, skipped


def run_declarations(converted_dir, out_path):
    """Its own process: the converted pack's declarations (v2/pyproject.toml), read as ComfyUI reads them."""
    cpu_comfy()
    from comfy_api.latest import registration
    entry = os.path.join(converted_dir, "v2")
    try:
        declared = registration.declarations(entry)
        name = registration.pack_name(entry, declared)
        routes = declared["comfy"].get("routes") or []
        for route in routes:
            for key in ("method", "path", "module", "handler"):
                if not isinstance(route.get(key), str) or not route[key]:
                    raise ValueError(f"route {route!r} lacks {key}")
        report = {"valid": True, "name": name, "namespace": registration.pack_namespace(name),
                  "requirements": list(declared["project"].get("dependencies") or []),
                  "declared_routes": routes, "routes": {}}
    except BaseException as error:
        report = {"valid": False, "error": f"{type(error).__name__}: {error}"[:1500], "trace": traceback.format_exc()[-2500:]}
    json.dump(report, open(out_path, "w"))


def missing_generated_routes(describe, declared):
    """Literal routes the converter should have exposed through its generated handler."""
    expected = {(method.lower(), path) for method, path in describe.get("routes") or []
                if method.lower() in ("get", "post", "put", "patch", "delete")
                and path.startswith("/") and not any(c in path for c in "{}?#")}
    actual = {(route["method"].lower(), route["path"]) for route in declared
              if route.get("module") == "_v2_routes" and route.get("handler") == "handle"}
    return [f"{method.upper()} {path}" for method, path in sorted(expected - actual)]


def run_original(original_dir, describe_path, out_path):
    """Its own process: the original's outputs on made-up inputs, for the converted pack's to be compared with."""
    import pickle
    cpu_comfy(as_comfyui_loads_packs=True)  # the original, as ComfyUI loads it
    from packload import LoadedPack
    describe = json.load(open(describe_path))
    results = {}
    _audited([original_dir])
    try:
        original = LoadedPack(original_dir)
    except BaseException as error:
        results["__error__"] = f"{type(error).__name__}: {error}"[:500]
        original = None
    first, skipped = _runnable(describe) if original else ([], {})
    # pack-specific types the skipped nodes need, which the pack's own runnable nodes produce
    wanted = {m.group(1) for s in skipped.values() for m in [re.search(r"needs a (\S+)", s or "")] if m} - MODEL_KINDS
    provided = {}
    rounds = [first]
    done_ids = set()
    ran = []
    while rounds:
        batch = rounds.pop(0)
        ran.append(batch)
        _run_originals(original, batch, results, wanted, provided)
        done_ids |= {node_id for node_id, _, _ in batch}
        if provided and len(done_ids) < len(describe.get("nodes", {})):
            more = [r for r in _runnable(describe, provided)[0] if r[0] not in done_ids]
            if more:
                rounds.append(more)  # nodes the produced values unlock (and what those produce, in turn)
    results["__provided__"] = provided
    # the order the original's nodes ran in (rounds: chained nodes after the nodes that unlock them), for the
    # converted pack to run in the same order: nodes sharing state (a global Set before a Get) see the same state
    results["__order__"] = [node_id for b in ran for node_id, _, _ in b]
    _run_originals_done(original, describe, results)
    with open(out_path, "wb") as file:
        pickle.dump(results, file)


def _run_originals(original, batch, results, wanted, provided):
    import pickle
    for node_id, entry_info, inputs in batch:
        if node_id not in original.nodes:
            continue
        values, hidden = _hidden(entry_info)
        def once():
            if entry_info["kind"] == "v3":
                return _converted_output(original.nodes[node_id], copy.deepcopy(inputs), values, original.module_name)
            return _original_output(original.nodes[node_id], entry_info["function"], copy.deepcopy(inputs), hidden)
        try:
            start = len(_AUDIT["events"])
            output = once()
            io_seen = _AUDIT["events"][start:]  # the first run's (the second would repeat them)
            try:  # a second run of the original: what differs between the two isn't the conversion's doing
                again = _same(_comparable(output[0]), _comparable(once()[0]))
            except BaseException as error:
                again = f"the second run failed: {type(error).__name__}"
            if again:
                results[node_id] = {"nondeterministic": again[:200], "io": io_seen}
                continue
            outputs, ui = _comparable(output[0]), (sorted(output[1]) if isinstance(output[1], dict) else output[1] and [])
            try:
                pickle.dumps(outputs)
            except Exception:
                outputs = _TypeOnly("unpicklable output")
            results[node_id] = {"outputs": outputs, "ui": ui, "io": io_seen}
            for index, kind in enumerate(entry_info["info"].get("output") or []):
                if isinstance(kind, str) and kind in wanted and kind not in provided and index < len(output[0]) \
                        and _plain(output[0][index]):
                    try:
                        value = _comparable(output[0][index])
                        output_lists = entry_info["info"].get("output_is_list") or []
                        if index < len(output_lists) and output_lists[index]:
                            if not isinstance(value, (list, tuple)):
                                continue  # only plain list outputs have a proved flattening here
                            value = _OutputList(list(value))
                        pickle.dumps(value)
                        provided[kind] = value
                    except Exception:
                        pass
        except BaseException as error:
            results[node_id] = {"error": f"{type(error).__name__}: {error}"[:300],
                                "io": _AUDIT["events"][start:] if "start" in locals() else []}


def _run_originals_done(original, describe, results):
    if original is not None:  # its GET routes, as the frontend would call them
        replies = {}
        for path in _get_routes(describe):
            handler = next((r.handler for r in original.prompt_server.routes
                            if getattr(r, "path", None) == path and getattr(r, "method", "") in ("GET", "*")), None)
            if handler is None:
                continue
            try:
                replies[path] = _call_with_timeout(lambda: _route_reply(asyncio.run(handler(_RouteRequest(path)))))
            except BaseException as error:
                replies[path] = {"error": f"{type(error).__name__}: {error}"[:200]}
        results["__routes__"] = replies


def _scratch(name):
    import tempfile
    folder = os.environ.get("CONVERTER_SCRATCH") or os.path.join(tempfile.gettempdir(), "comfy_conversion")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


def _child(args, timeout):
    import subprocess
    try:
        done = subprocess.run([sys.executable, os.path.abspath(__file__)] + args, capture_output=True, text=True, timeout=timeout)
        return done.returncode, (done.stdout + done.stderr)[-1500:]
    except subprocess.TimeoutExpired:
        return -1, f"no result in {timeout}s"


def main(original_dir, converted_dir, describe_path, out_path):
    import pickle
    describe = json.load(open(describe_path))
    code, log = _child(["--declarations", converted_dir, _scratch("declarations.json")], 300)
    if code == -1:  # a stall under load: once more before calling it a failure
        code, log = _child(["--declarations", converted_dir, _scratch("declarations.json")], 300)
    try:
        report = json.load(open(_scratch("declarations.json")))
    except Exception:
        report = {"valid": False, "error": f"declarations exited {code}: {log}"}
    report["nodes"] = {}
    if not report["valid"]:
        json.dump(report, open(out_path, "w"))
        return
    missing = missing_generated_routes(describe, report.get("declared_routes") or [])
    if missing:
        report.update(valid=False, error="generated route declarations missing: " + ", ".join(missing),
                      missing_declared_routes=missing)
        json.dump(report, open(out_path, "w"))
        return
    code, log = _child(["--original", original_dir, describe_path, _scratch("expected.pickle")], 420)
    try:
        expected_all = pickle.load(open(_scratch("expected.pickle"), "rb"))
    except Exception:
        expected_all = {"__error__": f"original run exited {code}: {log[-400:]}"}
    if "__error__" in expected_all:
        report["original_error"] = expected_all["__error__"]
    cpu_comfy(as_comfyui_loads_packs=True)  # this process is ComfyUI too: it answers the pack's engine calls
    # These shipped core nodes answer sampler/scheduler/noise engine calls.
    # Normal ComfyUI startup registers them, but this harness skips startup and
    # only imports nodes.py. Keep third-party packs out of the host registry.
    import nodes
    sampler_path = os.path.join(os.path.dirname(nodes.__file__), "comfy_extras", "nodes_custom_sampler.py")
    if not asyncio.run(nodes.load_custom_node(sampler_path, ignore=set(nodes.NODE_CLASS_MAPPINGS),
                                              module_parent="comfy_extras")):
        raise RuntimeError("could not register ComfyUI's shipped sampler nodes")
    from packload import LoadedPack
    try:
        converted = LoadedPack(os.path.join(converted_dir, "v2"))
    except BaseException as error:
        report.update(valid=False, error=f"{type(error).__name__}: {error}"[:1500], trace=traceback.format_exc()[-2500:])
        json.dump(report, open(out_path, "w"))
        return
    _audited([os.path.join(converted_dir, "v2")])
    try:
        from comfy_api.latest import registration
        v2_dir = os.path.join(converted_dir, "v2")
        report["namespace"] = registration.pack_namespace(registration.pack_name(v2_dir, registration.declarations(v2_dir)))
    except Exception as error:
        report["namespace_error"] = f"{type(error).__name__}: {error}"[:200]
    expected_routes = expected_all.get("__routes__") or {}
    if expected_routes:
        import importlib
        report["routes"] = {}
        try:
            routes_module = importlib.import_module(f"{converted.module_name}._v2_routes")
        except Exception as error:
            routes_module = None
            report["routes_error"] = f"{type(error).__name__}: {error}"[:300]
        for path, want in expected_routes.items():
            if routes_module is None or isinstance(want, dict):
                report["routes"][path] = {"skipped": want.get("error") if isinstance(want, dict) else "no _v2_routes"}
                continue
            try:
                got = _call_with_timeout(lambda: asyncio.run(routes_module.handle(
                    {"method": "get", "path": path, "query": {}, "body": ""})))
                got = [got["status"], got["content_type"], got["body"]]
            except BaseException as error:
                report["routes"][path] = {"converted_failed": f"{type(error).__name__}: {error}"[:300]}
                continue
            report["routes"][path] = {"identical": got == want,
                                      "diff": None if got == want else f"{want[:2]} vs {got[:2]}"
                                      + ("" if want[2] == got[2] else " (bodies differ)")}
    runnable, skipped = _runnable(describe, expected_all.get("__provided__") or {})
    inputs_for = {node_id: inputs for node_id, _, inputs in runnable}
    # in the order the original ran them (then the rest, as describe lists them)
    order = [n for n in expected_all.get("__order__") or [] if n in describe.get("nodes", {})]
    ordered = order + [n for n in describe.get("nodes", {}) if n not in set(order)]
    for node_id, entry_info in ((n, describe["nodes"][n]) for n in ordered):
        if entry_info.get("kind") not in ("v1", "v3"):
            report["nodes"][node_id] = {"kind": entry_info.get("kind")}
            continue
        result = report["nodes"][node_id] = {"kind": entry_info["kind"]}
        cls = converted.nodes.get(node_id)
        if cls is None:
            result["missing"] = True
            continue
        try:
            result["schema"] = _schema_diff(entry_info["info"], cls.GET_NODE_INFO_V1())
        except BaseException as error:
            result["schema"] = [f"listing failed: {type(error).__name__}: {error}"[:300]]
        expected = expected_all.get(node_id)
        if node_id not in inputs_for or expected is None:
            result["run"] = {"skipped": skipped.get(node_id) or "original not run"}
            continue
        if "error" in expected:
            result["run"] = {"original_failed": expected["error"]}
            continue

        values, _ = _hidden(entry_info)
        result["original_io"] = expected.get("io") or []
        start = len(_AUDIT["events"])
        try:
            got = _converted_output(cls, copy.deepcopy(inputs_for[node_id]), values, converted.module_name)
            result["io"] = _AUDIT["events"][start:]
        except BaseException as error:
            result["io"] = _AUDIT["events"][start:]
            result["run"] = {"converted_failed": f"{type(error).__name__}: {error}"[:300],
                             "trace": traceback.format_exc()[-800:]}
            continue
        if "nondeterministic" in expected:  # it runs; its output can't be compared
            result["run"] = {"ran": True, "nondeterministic": expected["nondeterministic"]}
            continue
        outputs = _comparable(got[0])
        if isinstance(expected["outputs"], _TypeOnly) and expected["outputs"].name == "unpicklable output":
            diff = None
        else:
            diff = _same(expected["outputs"], outputs)
            roots = (original_dir.rstrip("/"), os.path.join(converted_dir, "v2").rstrip("/"))
            if diff and _same(_mask_times(expected["outputs"], 0, roots), _mask_times(outputs, 0, roots)) is None:
                result["run"] = {"ran": True, "nondeterministic": f"only dates, times or the pack's folder differ: {diff}"[:200]}
                continue
        got_ui = sorted(got[1]) if isinstance(got[1], dict) else got[1] and []
        ui_diff = None if (expected["ui"] is None) == (got_ui is None) and (expected["ui"] is None or expected["ui"] == got_ui) \
            else f"ui: {expected['ui']} vs {got_ui}"
        result["run"] = {"identical": diff is None and ui_diff is None, "diff": diff or ui_diff}
    json.dump(report, open(out_path, "w"))


if __name__ == "__main__":
    if sys.argv[1] == "--declarations":
        run_declarations(os.path.realpath(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "--original":
        run_original(os.path.realpath(sys.argv[2]), os.path.realpath(sys.argv[3]), sys.argv[4])
    else:
        main(*[os.path.realpath(a) if i < 3 else a for i, a in enumerate(sys.argv[1:5])])
    sys.stdout.flush()
    os._exit(0)  # a node's stray thread mustn't keep the container alive
