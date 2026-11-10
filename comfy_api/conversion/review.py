"""What in a converted pack a person or agent must convert or test, as notes with file and line:

    node_x.py, line 342 - File access through python file io must be manually converted: ...

The converter's rewrites fall in three groups:

  reliable  mechanical, one-to-one, and checked (schemas, pure util copies, folders, devices,
            progress and interrupts, UI events, LoRA loading, annotations): not flagged;
  test      automated, but can be wrong in ways the checks may not catch, and nodes the verifier
            couldn't run: flagged to test;
  manual    can't be converted automatically (engine uses left, file writes, frontend graph
            hooks, routes that aren't V2 routes, failed checks): flagged to convert by hand.

``classify`` turns the flags and checks into:

  verified       converted, only reliable rewrites, every node's behavior checked
  needs-testing  converted, but with things to test
  needs-manual   things to convert by hand, then test
  none           no V2 pack

    python review.py <run dir> [--mark]

re-classifies a run from its saved conversions and checks; ``--mark`` also writes each note into
the converted pack: a trailing ``# V2-MANUAL: ...`` / ``# V2-TEST: ...`` comment on the line (only
where that leaves the code exactly as it was), and the whole list in ``V2_CONVERSION.md``.
"""
import ast
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert  # noqa: E402

SKIP_FILES = {"install.py", "setup.py", "prestartup_script.py"}
SKIP_DIRS = {"tests", "test", "docs", "examples", "example_workflows", "node_modules", "web", "js"}


# --------------------------------------------------------------------------- #
# Notes: what the agent should do, by topic
# --------------------------------------------------------------------------- #
def note_engine_use(name):
    return (f"`{name}` is ComfyUI internals, which a V2 pack can't use: replace it with an `eng` call or "
            "`util` (comfy_api/index.pyi lists them); if none fits, move this code into an engine extension "
            "(v2/engine/<name>.py, class EngineOps) and call it as `await eng.ext.<name>.<method>(...)`.")


def note_engine_attr(name):
    kind, attr = name.split(".", 1)
    return (f"`{attr}` on a {kind}: in V2.1 a pack's {kind} is an opaque handle, so ComfyUI's object attributes "
            f"aren't there and this fails when the node runs. Use the `eng.{kind.lower()}` call that does this "
            "(comfy_api/index.pyi); if none fits, leave the line and name the capability the API would need.")


NOTE_SETTING_CHANGED = ("Changes one of ComfyUI's command-line settings. In V2.1 a pack can only read settings "
                        "(`eng.settings.get`), so the change can't reach ComfyUI: drop it, or name the capability "
                        "the API would need.")
NOTE_FILE_WRITE = ("File access through python file io must be manually converted: pack isolation doesn't let "
                   "a pack write files itself. Save results with `eng.output.write` (or the `ui` savers), keep "
                   "settings and state with `eng.store`, and use `tempfile` for scratch files.")
NOTE_HOOK = ("This function runs during sampling, in the engine: it can't make `eng` calls, and globals or "
             "closure state it changes aren't seen by the node. Test the patched model end to end.")
NOTE_SYS_PATH = ("The pack changes sys.path to import its own modules; the converter handled the imports it could "
                 "see. Test that the pack loads, and that it imports its converted modules.")
NOTE_MODEL_FOLDER = ("Adds a model folder when the pack loads; the converter reports it to ComfyUI through "
                     "get_model_folders. Test that the folder's files show in the node's dropdown.")
REVIEW_CALLS = {
    "_v2e.prompt_server": "Uses a stand-in for PromptServer.instance (routes, send_sync and send_progress_text "
                          "only). Test what this code does with it.",
    "_v2e.cleanup_models": "comfy.model_management.cleanup_models became a no-op (ComfyUI cleans up its own "
                           "models). Test memory use if this code relied on it.",
    "_v2e.PROGRESS_BAR_ENABLED": "comfy.utils.PROGRESS_BAR_ENABLED is always True now. Test the progress shown.",
    "_v2e.common_ksampler": "nodes.common_ksampler now runs ComfyUI's KSampler (or KSamplerAdvanced for step "
                            "options) with eng.node.run. Test the sampling result, especially denoise with steps.",
    "_v2e.load_models_gpu": "Loading models onto the GPU goes through eng.memory.load_models. Test it.",
    "_v2e.load_model_gpu": "Loading models onto the GPU goes through eng.memory.load_models. Test it.",
    "_v2e.unload_all_models": "Unloading models goes through eng.memory.unload_all. Test it.",
}
CORE_NODE = re.compile(r"_v2e\.CoreNode\('([^']+)'")
HOOK_SETTERS = re.compile(r"eng\.model\.set_\w+")
ENGINE_AT_IMPORT = re.compile(r"_v2e\.(get_torch_device|intermediate_device|offload_device|vae_offload_device|text_encoder_offload_device|vae_device|"
                              r"text_encoder_device|get_free_memory|unet_dtype|intermediate_dtype|vae_dtype|args)\b")
SYS_PATH = re.compile(r"sys\.path\.(insert|append)\(")
MODEL_FOLDER_WRITE = re.compile(r"util\.folders\.(add_model_folder_path\(|folder_names_and_paths\[[^\]]+\]\s*=)")
IMPORT_DEFAULTS = {"get_torch_device": "CUDA if present, else the CPU (not ComfyUI's own choice)",
                   "intermediate_device": "the CPU", "offload_device": "the CPU", "vae_device": "the CPU",
                   "text_encoder_device": "the CPU", "get_free_memory": "0", "unet_dtype": "float32",
                   "intermediate_dtype": "float32", "vae_dtype": "float32", "args": "None for every setting"}

# file writes: module functions (by their imported name) and methods
WRITE_FUNCTIONS = {"os.remove", "os.unlink", "os.rename", "os.replace", "os.makedirs", "os.mkdir", "os.rmdir",
                   "os.removedirs", "shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.copytree", "shutil.move",
                   "shutil.rmtree", "torch.save", "numpy.save", "numpy.savez", "numpy.savez_compressed", "numpy.savetxt",
                   "cv2.imwrite", "imageio.imwrite", "imageio.mimwrite", "imageio.mimsave", "imageio.v3.imwrite",
                   "soundfile.write", "torchaudio.save", "scipy.io.wavfile.write", "safetensors.torch.save_file",
                   "util.save_torch_file", "comfy_api.index.util.save_torch_file", "comfy_api.v4.util.save_torch_file", "comfy.utils.save_torch_file", "urllib.request.urlretrieve", "wget.download"}
WRITE_METHODS = {"write_text", "write_bytes", "imwrite", "to_csv", "tofile", "save_pretrained", "write_videofile",
                 "write_audiofile", "mkdir", "touch", "rmdir", "save_file"}


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def file_writes(tree):
    """(line, what) for each place the code writes, moves or deletes files itself."""
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for a in node.names:
                aliases[a.asname or a.name] = f"{node.module}.{a.name}"
    # in-memory buffers: saving into one isn't file access
    buffers = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
               and (_dotted(n.value.func) or "").split(".")[-1] in ("BytesIO", "StringIO")
               for t in n.targets if isinstance(t, ast.Name)}
    buffers |= {item.optional_vars.id for n in ast.walk(tree) if isinstance(n, (ast.With, ast.AsyncWith))
                for item in n.items if isinstance(item.optional_vars, ast.Name) and isinstance(item.context_expr, ast.Call)
                and (_dotted(item.context_expr.func) or "").split(".")[-1] in ("BytesIO", "StringIO")}

    def in_memory(arg):
        return (isinstance(arg, ast.Name) and arg.id in buffers) or (
            isinstance(arg, ast.Call) and (_dotted(arg.func) or "").split(".")[-1] in ("BytesIO", "StringIO"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.args[1] if _dotted(node.func) in ("torch.save", "numpy.save", "np.save") and len(node.args) > 1 \
            else node.args[0] if node.args else None
        if target is not None and in_memory(target):
            continue
        name = _dotted(node.func)
        if name:
            head, _, rest = name.partition(".")
            full = ".".join(x for x in (aliases.get(head, head), rest) if x)
            if full == "open" or full.endswith(".open") and full.split(".")[0] in ("io", "codecs", "builtins"):
                mode = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == "mode"), None)
                if isinstance(mode, ast.Constant) and isinstance(mode.value, str) and set(mode.value) & set("wax+"):
                    found.append((node.lineno, f"open(..., {mode.value!r})"))
                continue
            if full in WRITE_FUNCTIONS:
                found.append((node.lineno, full))
                continue
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in WRITE_METHODS:
                found.append((node.lineno, f".{attr}()"))
            elif attr == "save" and node.args and not isinstance(node.args[0], (ast.Dict, ast.List)):
                found.append((node.lineno, ".save(...)"))  # PIL images, models, arrays: saved to a path
            elif attr == "open" and _dotted(node.func.value) and any(
                    isinstance(a, ast.Constant) and isinstance(a.value, str) and set(a.value) & set("wax+")
                    for a in node.args[:1]) and "Image" not in (_dotted(node.func.value) or ""):
                found.append((node.lineno, "Path.open(write)"))
    return sorted(set(found))


# --------------------------------------------------------------------------- #
# Flags
# --------------------------------------------------------------------------- #
def _flag(flags, file, line, kind, topic, note):
    flags.append({"file": file, "line": line, "kind": kind, "topic": topic, "note": note})


def _import_time_lines(tree):
    """Lines that run when the pack loads or ComfyUI lists its nodes: module and class bodies,
    INPUT_TYPES and define_schema."""
    lines = set()

    def walk(body, at_import):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "define_schema":  # a converted V1 node's INPUT_TYPES isn't called
                    lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
                continue
            if isinstance(node, ast.ClassDef):
                walk(node.body, True)
                continue
            if at_import:
                lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    walk(tree.body, True)
    return lines


def _pack_files(v2):
    for root, dirs, files in os.walk(v2):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in files:
            if name.endswith(".py") and not name.startswith("_v2_") and name not in SKIP_FILES:
                yield os.path.relpath(os.path.join(root, name), v2).replace(os.sep, "/")


def review_files(v2, conversion):
    flags = []
    leftovers = conversion.get("leftovers") or {}
    for relative in sorted(_pack_files(v2)):
        try:
            source = open(os.path.join(v2, relative), encoding="utf-8").read()
            tree = ast.parse(source)
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            continue
        lines = source.splitlines()
        at_import = _import_time_lines(tree)
        for number, text in enumerate(lines, 1):
            code = text.split("#")[0]
            for call, note in REVIEW_CALLS.items():
                if call in code:
                    _flag(flags, relative, number, "test", call.split(".")[-1], note)
            for node_id in CORE_NODE.findall(code):
                _flag(flags, relative, number, "test", "calls a ComfyUI node",
                      f"Runs ComfyUI's node `{node_id}` with `eng.node.run`, naming the arguments by the node's "
                      "input order. Test this call: check the arguments line up and the node still has this method.")
            if HOOK_SETTERS.search(code):
                _flag(flags, relative, number, "test", "sampling-time function", NOTE_HOOK)
            match = ENGINE_AT_IMPORT.search(code)
            if number in at_import and match:
                what = match.group(1)
                default = IMPORT_DEFAULTS.get(what, "a default")
                _flag(flags, relative, number, "test", "engine call at import",
                      f"`{what}` here runs when the pack loads or ComfyUI lists its nodes, where the engine isn't "
                      f"available, so it gives {default}. Move it into the node's execute, and test.")
            if MODEL_FOLDER_WRITE.search(code):
                _flag(flags, relative, number, "test", "adds a model folder", NOTE_MODEL_FOLDER)
            if SYS_PATH.search(code):
                _flag(flags, relative, number, "test", "changes sys.path", NOTE_SYS_PATH)
        for number, what in file_writes(tree):
            _flag(flags, relative, number, "manual", "file write", f"File access through python file io must be "
                  f"manually converted ({what}): " + NOTE_FILE_WRITE.split(": ", 1)[1])
        names = set(leftovers.get(relative) or [])
        if names:
            module = convert._Module(source, relative)
            seen = set()
            for node in ast.walk(module.tree):
                if isinstance(node, (ast.Attribute, ast.Name)) and isinstance(getattr(node, "ctx", None), ast.Load):
                    full = module.resolve(node)
                    if full in names and (full, node.lineno) not in seen:
                        seen.add((full, node.lineno))
                        _flag(flags, relative, node.lineno, "manual", "engine use left", note_engine_use(full))
            # ``MODEL.model_options``: an engine object's attribute (the object is a Ref in V2.1)
            attrs = {n.split(".", 1)[1]: n.split(".")[0] for n in names if n.split(".")[0] in convert.ENGINE_KINDS}
            # ``comfy.cli_args.args.fast (changed)``: a ComfyUI setting the pack changes
            # (as the converted code spells it: ``_v2e.args.fast``)
            changed = {"_v2e.args" + n[len("comfy.cli_args.args"):-len(" (changed)")]
                       for n in names if n.endswith(" (changed)")}
            for node in ast.walk(module.tree):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.attr in attrs \
                        and (node.attr, node.lineno) not in seen:
                    seen.add((node.attr, node.lineno))
                    _flag(flags, relative, node.lineno, "manual", "engine object attribute",
                          note_engine_attr(f"{attrs[node.attr]}.{node.attr}"))
                elif changed and isinstance(node, ast.Attribute) and ".".join(convert._dotted(node) or []) in changed \
                        and ("changed", node.lineno) not in seen:
                    seen.add(("changed", node.lineno))
                    _flag(flags, relative, node.lineno, "manual", "changes a ComfyUI setting", NOTE_SETTING_CHANGED)
            # ``sampling function args['model']``: a sampling function's read of an engine object it isn't given
            keys = {n.split("'")[1] for n in names if n.startswith("sampling function args[") and "'" in n}
            for node in ast.walk(module.tree) if keys else ():
                key = None
                if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                    key = node.slice.value
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" \
                        and node.args and isinstance(node.args[0], ast.Constant):
                    key = node.args[0].value
                if key in keys and ("sampling", node.lineno) not in seen:
                    seen.add(("sampling", node.lineno))
                    _flag(flags, relative, node.lineno, "manual", "sampling function reads an engine object",
                          f"This sampling function reads `args[{key!r}]`, but V2.1 hands pack sampling functions tensors "
                          "and plain data only, so it isn't there. Compute what it needs from the tensors it gets, or "
                          "note the generic capability the API would need.")
    return flags


def review_routes(conversion, verify):
    flags = []
    routes = conversion.get("routes") or {}
    namespace = conversion.get("namespace") or "<namespace>"
    for item in routes.get("not_moved") or []:
        route = item.split(": a V2 route")[0]
        _flag(flags, "routes", 0, "manual", "route isn't a V2 route",
              f"Route {route} can't be a V2 route (a V2 route is a literal path: get, post, put, patch or delete). "
              "Give it a literal path that takes its values in the query or body, and update the frontend call.")
    for path in routes.get("not_in_frontend") or []:
        _flag(flags, "frontend", 0, "manual", "frontend route URL not found",
              f"The frontend's call to {path} wasn't found to rewrite (it's probably built at run time): point it "
              f"at `/ext/{namespace}{path}`.")
    for path in routes.get("uncalled") or []:
        _flag(flags, "routes", 0, "test", "route nothing in the pack calls",
              f"Nothing in the pack calls {path}: an outside app or script may. It is served at "
              f"`/ext/{namespace}{path}` now, so such a caller needs the new path.")
    for file in routes.get("files_or_streams") or []:
        _flag(flags, file, 0, "manual", "route answers with a file or stream",
              "A route here answers with a file or a stream; V2 routes answer JSON or text (up to 1 MB). Serve "
              "files from the pack's web directory, or return the data as JSON.")
    for path, result in (verify.get("routes") or {}).items():
        if result.get("converted_failed"):
            _flag(flags, "routes", 0, "manual", "route fails converted",
                  f"GET {path} fails in the converted pack: {result['converted_failed'][:150]}. Fix it, then test.")
        elif result.get("identical") is False:
            _flag(flags, "routes", 0, "test", "route answers differently",
                  f"GET {path} answers differently from the original ({result.get('diff')}). Test it.")
    return flags


FRONTEND_HOOKS = {
    "replaces graphToPrompt": re.compile(r"graphToPrompt\s*=\s*(async\s+)?(function|\()|graphToPrompt\s*=\s*async"),
    "defines virtual nodes": re.compile(r"isVirtualNode\s*[:=]\s*true"),
    "replaces queuePrompt": re.compile(r"(api|app)\.queuePrompt\s*=\s*(async\s+)?(function|\()"),
    "uses applyToGraph": re.compile(r"applyToGraph\s*(\(|=|:)"),
}


def frontend_hooks(pack_dir):
    """Frontend code that changes the prompt before it's sent: it needs a backend graph transform."""
    found = {}
    for root, dirs, files in os.walk(pack_dir):
        dirs[:] = [d for d in dirs if d not in ("node_modules", ".git", "dist", "build", "v2")]
        for name in files:
            if name.endswith((".js", ".mjs", ".ts")) and not name.endswith(".min.js"):
                try:
                    with open(os.path.join(root, name), encoding="utf-8", errors="ignore") as file:
                        source = file.read(400_000)
                except OSError:
                    continue
                for hook, pattern in FRONTEND_HOOKS.items():
                    if hook not in found and pattern.search(source):
                        found[hook] = os.path.relpath(os.path.join(root, name), pack_dir)
    return found


def review_frontend(pack_dir, namespace):
    return [{"file": file, "line": 0, "kind": "manual", "topic": f"frontend {hook}",
             "note": f"The frontend {hook} in this file. In V2 the backend builds the prompt from the workflow: "
                     "write a backend graph transform for what it does, and remove the frontend override."}
            for hook, file in frontend_hooks(pack_dir).items()]


def _class_line(v2, entry):
    """Where a node's class is defined in the converted pack: (file, line)."""
    relative = os.path.relpath(entry.get("file") or "", "/pack") if entry.get("file") else None
    name = (entry.get("qualname") or entry.get("class") or "").split(".")[-1]
    candidates = [relative] if relative else sorted(_pack_files(v2))
    if name:
        for candidate in candidates:
            try:
                for number, text in enumerate(open(os.path.join(v2, candidate), encoding="utf-8", errors="ignore"), 1):
                    if re.match(rf"\s*class\s+{re.escape(name)}\b", text):
                        return candidate.replace(os.sep, "/"), number
            except OSError:
                continue
    return (relative or "_v2_nodes.py"), 0


def review_nodes(v2, describe, verify):
    flags = []
    for node_id, entry in sorted((describe.get("nodes") or {}).items()):
        if entry.get("kind") not in ("v1", "v3"):
            continue
        node = (verify.get("nodes") or {}).get(node_id) or {}
        run = node.get("run") or {}
        file, line = _class_line(v2, entry)
        if node.get("missing"):
            _flag(flags, file, line, "manual", "node not generated",
                  f"Node `{node_id}` wasn't generated as a V2 node. Write its io.Schema by hand.")
        elif node.get("schema"):
            _flag(flags, file, line, "manual", "schema differs",
                  f"Node `{node_id}`'s V2 schema differs from the original ({node['schema'][0][:150]}). Fix its io.Schema.")
        if run.get("converted_failed"):
            _flag(flags, file, line, "manual", "node fails converted",
                  f"Node `{node_id}` fails in the converted pack: {run['converted_failed'][:150]}. Fix it, then test.")
        elif run.get("identical") is False:
            _flag(flags, file, line, "manual", "output differs",
                  f"Node `{node_id}`'s output differs from the original's ({run.get('diff')}). Find why, fix it, then test.")
        elif run.get("nondeterministic"):
            _flag(flags, file, line, "test", "output varies run to run",
                  f"Node `{node_id}` ran, but its output varies from run to run, so it wasn't compared. Test it.")
        elif "skipped" in run or run.get("original_failed") or not run:
            reason = run.get("skipped") or (f"the original fails on made-up inputs: {run['original_failed'][:80]}"
                                            if run.get("original_failed") else "not run")
            _flag(flags, file, line, "test", "behavior not checked",
                  f"Node `{node_id}`'s behavior wasn't checked ({reason}). Test it with real inputs against the original pack.")
    return flags


def review_runtime(verify, flags):
    """What the verifier saw nodes do while they ran (Python audit events): file writes the static
    detector missed become flags, and network connections ask for a permission. Returns the counts
    for measuring the static detector: (writes seen, of those already flagged)."""
    static = {(f["file"], f["line"]) for f in flags if f["topic"] == "file write"}
    seen, matched, added = 0, 0, set()
    for node_id, node in (verify.get("nodes") or {}).items():
        for event in node.get("io") or []:
            where = (event["file"].replace(os.sep, "/"), event["line"])
            if event["event"] == "socket.connect":
                if ("net",) + where not in added:
                    added.add(("net",) + where)
                    _flag(flags, where[0], where[1], "manual", "network access",
                          f"Node `{node_id}` connects to {event['path']} here (seen while it ran). Declare it in "
                          "pyproject.toml as `permissions = [\"network:<host>\"]` under [tool.comfy], for review to grant.")
                continue
            seen += 1
            if where in static:
                matched += 1
            elif where not in added:
                added.add(where)
                _flag(flags, where[0], where[1], "manual", "file write",
                      f"File access through python file io must be manually converted ({event['event']} "
                      f"{os.path.basename(event['path'])}, seen while node `{node_id}` ran): "
                      + NOTE_FILE_WRITE.split(": ", 1)[1])
    return seen, matched


def classify(record, flags):
    if record.get("class") == "none":
        return "none"
    kinds = {f["kind"] for f in flags}
    return "needs-manual" if "manual" in kinds else "needs-testing" if "test" in kinds else "verified"


# --------------------------------------------------------------------------- #
# Writing the notes into the pack
# --------------------------------------------------------------------------- #
def mark_lines(v2, flags):
    """A trailing marker comment on each flagged line, where that leaves the program the same."""
    by_file = collections.defaultdict(dict)
    for f in flags:
        if f["line"] and f["file"].endswith(".py"):
            by_file[f["file"]].setdefault(f["line"], []).append(f)
    marked = set()
    for relative, lines_flags in by_file.items():
        path = os.path.join(v2, relative)
        try:
            source = open(path, encoding="utf-8").read()
            before = ast.dump(ast.parse(source))
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            continue
        lines = source.split("\n")
        for number, these in sorted(lines_flags.items()):
            text = lines[number - 1]
            if "# V2-" in text or text.rstrip().endswith("\\"):
                continue
            tags = "; ".join(sorted({("V2-MANUAL: " if f["kind"] == "manual" else "V2-TEST: ") + f["topic"] for f in these}))
            candidate = lines[:number - 1] + [text.rstrip() + f"  # {tags} (see V2_CONVERSION.md)"] + lines[number:]
            try:
                if ast.dump(ast.parse("\n".join(candidate))) == before:  # a comment, nothing else
                    lines = candidate
                    marked.update((relative, number, id(f)) for f in these)
            except SyntaxError:
                pass
        open(path, "w", encoding="utf-8").write("\n".join(lines))
    return marked


def worklist(pack, result, triage=None):
    manual = [f for f in result["flags"] if f["kind"] == "manual"]
    test = [f for f in result["flags"] if f["kind"] == "test"]

    def item(f):
        where = f"{f['file']}, line {f['line']}" if f["line"] else f["file"]
        return f"- {where} - {f['note']}"
    out = [f"# V2 conversion worklist: {pack}", "",
           f"Class: **{result['class']}**: {len(manual)} to convert by hand, {len(test)} to test. "
           "Lines marked `# V2-MANUAL` / `# V2-TEST` in the code are these items.", ""]
    if triage and triage.get("metadata"):
        meta = triage["metadata"]
        out += [f"{meta.get('summary') or ''}", "", f"Tags: {', '.join((meta.get('tags') or []) + (meta.get('extra_tags') or []))}",
                f"Model families: {', '.join(meta.get('model_families') or []) or '-'}", ""]
    if manual:
        out += ["## Convert by hand", ""] + [item(f) for f in manual] + [""]
    if test:
        out += ["## Test", ""] + [item(f) for f in test] + [""]
    out += ["## Done by the converter", "",
            "Node schemas, the manifest, pure helpers (`util`), folders (`util.folders`), devices, progress, "
            "interrupts and UI events were converted mechanically and checked; they need no review.", ""]
    return "\n".join(out)


def review_pack(run, record, mark=False):
    work = os.path.join(run, "work", record["pack"])

    def load(name):
        try:
            return json.load(open(os.path.join(work, name)))
        except (OSError, ValueError):
            return {}
    describe, verify, triage = load("describe.json"), load("verify.json"), load("triage.json")
    conversion = record.get("conversion") or {}
    v2 = os.path.join(work, "converted", os.path.basename(os.path.normpath(record["dir"])), "v2")
    flags = []
    if record.get("class") != "none" and os.path.isdir(v2):
        flags = review_files(v2, conversion) + review_routes(conversion, verify) \
            + review_frontend(record["dir"], conversion.get("namespace")) + review_nodes(v2, describe, verify)
    seen, matched = review_runtime(verify, flags) if flags is not None else (0, 0)
    result = {"pack": record["pack"], "was": record.get("class"), "class": classify(record, flags), "flags": flags,
              "runtime_writes": {"seen": seen, "flagged_statically": matched}}
    if mark and flags:
        mark_lines(v2, flags)
        with open(os.path.join(v2, "V2_CONVERSION.md"), "w", encoding="utf-8") as file:
            file.write(worklist(record["pack"], result, triage))
    return result


def main(argv):
    run, mark = argv[0], "--mark" in argv
    out = []
    for line in open(os.path.join(run, "results.jsonl")):
        if line.strip():
            result = review_pack(run, json.loads(line), mark)
            out.append(result)
            work = os.path.join(run, "work", result["pack"])
            if os.path.isdir(work):
                json.dump(result, open(os.path.join(work, "review.json"), "w"), indent=1)
    with open(os.path.join(run, "review.jsonl"), "w") as file:
        for result in out:
            file.write(json.dumps(result) + "\n")
    seen = sum(r["runtime_writes"]["seen"] for r in out)
    matched = sum(r["runtime_writes"]["flagged_statically"] for r in out)
    print(f"file writes seen while nodes ran: {seen}; flagged by the static detector: {matched}"  # noqa: T201
          + (f" ({100 * matched / seen:.0f}% recall on executed code)" if seen else ""))
    classes = collections.Counter(r["class"] for r in out)
    print(f"{len(out)} packs")  # noqa: T201
    for name in ("verified", "needs-testing", "needs-manual", "none"):
        print(f"  {name:14} {classes[name]:5}  ({100 * classes[name] / max(len(out), 1):.0f}%)")  # noqa: T201
    topics = collections.Counter()
    for r in out:
        for key in {(f["kind"], f["topic"]) for f in r["flags"]}:
            topics[key] += 1
    print("\nflags, by packs:")  # noqa: T201
    for (kind, topic), count in topics.most_common(30):
        print(f"  {count:5}  {kind:6}  {topic}")  # noqa: T201


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", SyntaxWarning)
    main(sys.argv[1:])
