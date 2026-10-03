"""Static discipline check for the ``[comfy-event]`` diagnostic lines.

Pure :mod:`ast` analysis of every first-party module that mentions the emitter
or the tag. Nothing is imported or executed. Enforced over every emit call site:

a. the event is one string literal in the closed vocabulary
b. keyword fields come from the closed vocabulary, with no ``**`` splats, and
   the event's required fields are present
c. ``op=`` is a string literal registered in ``PERF_OPS``
d. no module other than the emitter spells out the tag or uses ``TAG``, so a
   launcher only sees lines that went through emit()
e. the call sites present in the tree match an explicit manifest
"""

from __future__ import annotations

import ast
import os
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

from comfy.diagnostics.events import (
    ALLOWED_EVENTS,
    ALLOWED_FIELDS,
    PERF_OPS,
    REQUIRED_FIELDS,
    TAG,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_SCOPE = "<module>"
EVENTS_MODULE = "comfy.diagnostics.events"
EVENTS_FILE = "comfy/diagnostics/events.py"
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
# First-party packages; top-level modules such as main.py and server.py are
# scanned too. Anything else at the root (venvs, custom_nodes, tests) is not.
FIRST_PARTY_PACKAGES = (
    "api_server",
    "app",
    "comfy",
    "comfy_api",
    "comfy_api_nodes",
    "comfy_config",
    "comfy_execution",
    "comfy_extras",
    "middleware",
    "utils",
)


class CallSite(NamedTuple):
    path: str
    function: str
    event: str
    op: str | None


# Every emit call in the tree: (file, enclosing function, event, op literal).
EXPECTED_CALL_SITES: frozenset[CallSite] = frozenset()


class Aliases(NamedTuple):
    """The names one module binds to the events module, its emit() and its TAG."""

    module: frozenset[str]
    emit: frozenset[str]
    tag: frozenset[str]
    star: bool


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _first_party_files(root: Path) -> Iterator[Path]:
    yield from root.glob("*.py")
    for package in FIRST_PARTY_PACKAGES:
        for directory, dirnames, filenames in os.walk(root / package):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]
            yield from (Path(directory) / f for f in filenames if f.endswith(".py"))


def _candidate_files(root: Path) -> tuple[str, ...]:
    """First-party modules that mention the emitter or the tag at all."""
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in _first_party_files(root)
            if "diagnostics" in (text := _read(path)) or "events" in text or TAG in text
        )
    )


def _scoped_nodes(tree: ast.Module) -> Iterator[tuple[ast.AST, str]]:
    """Yield every node paired with the name of its innermost enclosing function."""

    def walk(node: ast.AST, scope: str) -> Iterator[tuple[ast.AST, str]]:
        for child in ast.iter_child_nodes(node):
            child_scope = child.name if isinstance(child, FUNCTION_NODES) else scope
            yield child, child_scope
            yield from walk(child, child_scope)

    yield from walk(tree, MODULE_SCOPE)


def _absolute_module(relative: str, node: ast.ImportFrom) -> str:
    """The absolute dotted name a (possibly relative) from-import refers to."""
    if node.level == 0:
        return node.module or ""
    package = relative.removesuffix(".py").split("/")[:-1]
    base = package[: len(package) - (node.level - 1)]
    return ".".join([*base, *([node.module] if node.module else [])])


def _resolve_aliases(tree: ast.Module, relative: str) -> Aliases:
    module: set[str] = set()
    emit: set[str] = set()
    tag: set[str] = set()
    star = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == EVENTS_MODULE:
                    module.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            source = _absolute_module(relative, node)
            for alias in node.names:
                bound = alias.asname or alias.name
                if source == EVENTS_MODULE and alias.name == "*":
                    star = True
                elif source == EVENTS_MODULE and alias.name == "emit":
                    emit.add(bound)
                elif source == EVENTS_MODULE and alias.name == "TAG":
                    tag.add(bound)
                elif f"{source}.{alias.name}" == EVENTS_MODULE:
                    module.add(bound)
    return Aliases(frozenset(module), frozenset(emit), frozenset(tag), star)


def _dotted_name(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _is_emit_call(func: ast.expr, aliases: Aliases) -> bool:
    if isinstance(func, ast.Attribute) and func.attr == "emit":
        return _dotted_name(func.value) in aliases.module
    return isinstance(func, ast.Name) and func.id in aliases.emit


def _is_unresolvable_emit_call(func: ast.expr, aliases: Aliases) -> bool:
    return (
        bool(aliases.module)
        and isinstance(func, ast.Attribute)
        and func.attr == "emit"
        and _dotted_name(func.value) is None
    )


def _literal(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _event_of(call: ast.Call) -> str | None:
    return _literal(call.args[0]) if len(call.args) == 1 else None


def _op_of(call: ast.Call) -> str | None:
    return next((_literal(k.value) for k in call.keywords if k.arg == "op"), None)


def _emit_faults(call: ast.Call) -> Iterator[str]:
    event = _event_of(call)
    if event not in ALLOWED_EVENTS:
        yield "the event must be one string literal in ALLOWED_EVENTS"
    passed = {keyword.arg for keyword in call.keywords}
    for name in sorted(REQUIRED_FIELDS.get(event or "", frozenset()) - passed):
        yield f"required field {name!r} is missing"
    for keyword in call.keywords:
        if keyword.arg is None:
            yield "**splat fields cannot be checked statically"
        elif keyword.arg not in ALLOWED_FIELDS:
            yield f"field {keyword.arg!r} is not in ALLOWED_FIELDS"
        elif keyword.arg == "op" and _literal(keyword.value) not in PERF_OPS:
            yield f"op must be a string literal in PERF_OPS, got {ast.unparse(keyword.value)!r}"


def _tag_fault(node: ast.AST, aliases: Aliases) -> str | None:
    """Why this node could put the tag on a line without going through emit()."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and TAG in node.value:
        return f"spells out {TAG} outside emit()"
    if isinstance(node, ast.Name) and node.id in aliases.tag:
        return "uses the emitter's TAG outside emit()"
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "TAG"
        and _dotted_name(node.value) in aliases.module
    ):
        return "uses the emitter's TAG outside emit()"
    return None


def _scan_file(root: Path, relative: str) -> tuple[Counter[CallSite], list[str]]:
    tree = ast.parse(_read(root / relative), filename=relative)
    aliases = _resolve_aliases(tree, relative)
    sites: Counter[CallSite] = Counter()
    faults: list[str] = []
    if aliases.star:
        faults.append(f"{relative}: a star import of the emitter cannot be checked statically")
    for node, scope in _scoped_nodes(tree):
        tag_fault = _tag_fault(node, aliases) if relative != EVENTS_FILE else None
        if tag_fault is not None:
            faults.append(f"{relative}:{getattr(node, 'lineno', 0)}: {tag_fault}")
        if not isinstance(node, ast.Call):
            continue
        if _is_unresolvable_emit_call(node.func, aliases):
            faults.append(f"{relative}:{node.lineno}: the emit receiver cannot be resolved statically")
        elif _is_emit_call(node.func, aliases):
            faults.extend(f"{relative}:{node.lineno}: {reason}" for reason in _emit_faults(node))
            sites[CallSite(relative, scope, _event_of(node) or "?", _op_of(node))] += 1
    return sites, faults


def scan_repository(root: Path = REPO_ROOT) -> tuple[tuple[str, ...], Counter[CallSite], list[str]]:
    files = _candidate_files(root)
    sites: Counter[CallSite] = Counter()
    faults: list[str] = []
    for relative in files:
        file_sites, file_faults = _scan_file(root, relative)
        sites += file_sites
        faults.extend(file_faults)
    return files, sites, faults


FILES, SITES, FAULTS = scan_repository()


def _write(root: Path, relative: str, source: str) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return relative


def test_the_walk_finds_the_emitter_and_skips_tests():
    """Guards every other check: a broken walk would make them all vacuous."""
    assert EVENTS_FILE in FILES
    assert not any(f.startswith(("tests-unit/", "tests/", "custom_nodes/")) for f in FILES)


def test_emit_call_sites_stay_inside_the_closed_vocabulary():
    assert FAULTS == []


def test_call_sites_match_the_manifest():
    manifest = Counter(EXPECTED_CALL_SITES)
    unexpected = SITES - manifest
    missing = manifest - SITES
    assert not unexpected, f"emit call sites not in the manifest: {sorted(unexpected)}"
    assert not missing, f"manifest call sites absent from the tree: {sorted(missing)}"


def test_every_import_form_is_recognised(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "PERF_OPS", frozenset({"startup.db"}))
    relative = _write(
        tmp_path,
        "app/probe.py",
        "import comfy.diagnostics.events\n"
        "from comfy.diagnostics import events as ev\n"
        "from comfy.diagnostics.events import emit as e\n\n"
        "def probe():\n"
        '    comfy.diagnostics.events.emit("perf.timing", op="startup.db")\n'
        '    ev.emit("perf.timing", op="startup.db")\n'
        '    e("perf.timing", op="startup.db")\n',
    )

    sites, faults = _scan_file(tmp_path, relative)

    assert faults == []
    assert sites == Counter({CallSite(relative, "probe", "perf.timing", "startup.db"): 3})


def test_relative_imports_are_recognised(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "PERF_OPS", frozenset({"startup.db"}))
    relative = _write(
        tmp_path,
        "comfy/diagnostics/sibling.py",
        "from . import events\n"
        "from .events import emit\n\n"
        "def probe():\n"
        '    events.emit("perf.timing", op="startup.db")\n'
        '    emit("perf.timing", op="startup.db")\n',
    )

    sites, faults = _scan_file(tmp_path, relative)

    assert faults == []
    assert sites == Counter({CallSite(relative, "probe", "perf.timing", "startup.db"): 2})


def test_bad_call_sites_are_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "PERF_OPS", frozenset({"startup.db"}))
    relative = _write(
        tmp_path,
        "app/bad.py",
        "import sys\n"
        "from comfy.diagnostics import events\n"
        "from comfy.diagnostics.events import TAG as T\n\n"
        "def probe(name, op, fields, provider):\n"
        "    events.emit(name, op=op)\n"
        '    events.emit("perf.timing", op=op)\n'
        '    events.emit("perf.timing", op="not.registered")\n'
        '    events.emit("perf.timing", op="startup.db", path="x")\n'
        '    events.emit("perf.timing", **fields)\n'
        '    provider().emit("perf.timing")\n'
        '    print("[comfy-event] perf.timing op=x")\n'
        '    sys.stdout.write(f"{T} perf.timing\\n")\n'
        '    sys.stdout.write(events.TAG)\n'
        '    events.emit("perf.timing", duration_ms=1)\n',
    )

    _sites, faults = _scan_file(tmp_path, relative)

    # Every probe line after the imports is reported, and nothing else.
    assert sorted({int(fault.split(":")[1]) for fault in faults}) == list(range(6, 16))


def test_a_star_import_of_the_emitter_is_reported(tmp_path):
    relative = _write(tmp_path, "app/star.py", "from comfy.diagnostics.events import *\n")

    _sites, faults = _scan_file(tmp_path, relative)

    assert len(faults) == 1 and "star import" in faults[0]


def test_the_candidate_filter_only_reads_first_party_code(tmp_path):
    _write(tmp_path, "main.py", "from comfy.diagnostics import events\n")
    _write(tmp_path, "app/uses.py", "from comfy.diagnostics import events\n")
    _write(tmp_path, "app/unrelated.py", "import logging\n")
    _write(tmp_path, "tests-unit/t.py", "from comfy.diagnostics import events\n")
    _write(tmp_path, "venv/lib/x.py", "from comfy.diagnostics import events\n")
    _write(tmp_path, "custom_nodes/pack/x.py", "print('[comfy-event] x')\n")

    assert _candidate_files(tmp_path) == ("app/uses.py", "main.py")
