"""Static discipline check for the ``[assets-event]`` log lines.

Pure :mod:`ast` analysis: no module under ``app/assets`` is imported or
executed, so this check never needs a running ComfyUI. It deliberately lives at
the ``tests-unit`` root rather than under ``tests-unit/assets_test/``, whose
autouse fixture boots a ComfyUI subprocess for every test in that subtree.

Four rules are enforced over every emit call site:

a. keyword fields come from the closed vocabulary, the event is a string literal
b. no other log line anywhere carries the tag, so the tap only ever sees emits
c. the call sites present in the tree match an explicit manifest
d. ``error_type=`` values come from ``event_log.error_type()``, never a string
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import pytest

from app.assets.event_log import ALLOWED_FIELDS, EVENT_NAME_PATTERN, TAG

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_SCOPE = "<module>"
EVENT_LOG_NAME = "event_log"
LOG_METHODS = frozenset({"debug", "info", "warning", "error", "exception", "critical", "log"})
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


class CallSite(NamedTuple):
    """The identity of one emit call: file, enclosing function, event."""

    path: str
    function: str
    event: str


# The manifest of every tagged event this branch is expected to emit, mapped to
# the plan todo that lands it. ``None`` means landed: those triples must be
# present in the tree exactly as written. A number means the call site does not
# exist yet — when that todo lands it flips its own entries to ``None``, which
# also removes its pending marker below.
EXPECTED_CALL_SITES: dict[CallSite, int | None] = {
    # todo 10 - seeder lifecycle + the single assets.enabled site
    CallSite("server.py", "__init__", "assets.enabled"): 10,
    CallSite("app/assets/seeder.py", "_run_scan", "seeder.scan_started"): 10,
    CallSite("app/assets/seeder.py", "_run_scan", "seeder.scan_completed"): 10,
    CallSite("app/assets/seeder.py", "_run_scan", "seeder.scan_failed"): 10,
    CallSite("app/assets/seeder.py", "_run_scan", "seeder.scan_cancelled"): 10,
    CallSite("app/assets/seeder.py", "_run_scan", "seeder.marked_missing"): 10,
    CallSite("app/assets/seeder.py", "mark_missing_outside_prefixes", "seeder.marked_missing"): 10,
    CallSite("app/assets/seeder.py", "_run_fast_phase", "seeder.batch_insert_failed"): 10,
    # todo 11 - scanner and ingest failure paths
    CallSite("app/assets/scanner.py", "sync_root_safely", "scanner.fast_scan_failed"): 11,
    CallSite("app/assets/scanner.py", "sync_temp_references_safely", "scanner.temp_sync_failed"): 11,
    CallSite(
        "app/assets/scanner.py", "mark_missing_outside_prefixes_safely", "scanner.mark_missing_failed"
    ): 11,
    CallSite("app/assets/scanner.py", "enrich_asset", "scanner.hash_failed"): 11,
    CallSite("app/assets/scanner.py", "enrich_asset", "scanner.hash_discarded_modified"): 11,
    CallSite("app/assets/scanner.py", "enrich_assets_batch", "scanner.enrich_failed"): 11,
    CallSite(
        "app/assets/services/ingest.py", "register_cached_output", "ingest.register_output_failed"
    ): 11,
    CallSite(
        "app/assets/services/ingest.py", "register_executed_output", "ingest.register_output_failed"
    ): 11,
    CallSite(
        "app/assets/services/ingest.py", "_discard_unreferenced_content", "ingest.discard_orphan_failed"
    ): 11,
    # todo 12 - one api.request_failed per route handler, seven handlers
    CallSite("app/assets/api/routes.py", "get_asset_route", "api.request_failed"): 12,
    CallSite("app/assets/api/routes.py", "upload_asset", "api.request_failed"): 12,
    CallSite("app/assets/api/routes.py", "update_asset_route", "api.request_failed"): 12,
    CallSite("app/assets/api/routes.py", "delete_asset_route", "api.request_failed"): 12,
    CallSite("app/assets/api/routes.py", "add_asset_tags", "api.request_failed"): 12,
    CallSite("app/assets/api/routes.py", "delete_asset_tags", "api.request_failed"): 12,
    CallSite("app/assets/api/upload.py", "parse_multipart_upload", "api.request_failed"): 12,
}

REQUEST_FAILED_HANDLERS = frozenset(
    {
        ("app/assets/api/routes.py", "get_asset_route"),
        ("app/assets/api/routes.py", "upload_asset"),
        ("app/assets/api/routes.py", "update_asset_route"),
        ("app/assets/api/routes.py", "delete_asset_route"),
        ("app/assets/api/routes.py", "add_asset_tags"),
        ("app/assets/api/routes.py", "delete_asset_tags"),
        ("app/assets/api/upload.py", "parse_multipart_upload"),
    }
)

LANDED_CALL_SITES = Counter(site for site, todo in EXPECTED_CALL_SITES.items() if todo is None)
PENDING_TODOS = sorted({todo for todo in EXPECTED_CALL_SITES.values() if todo is not None})


class Aliases(NamedTuple):
    """The names one module binds to the event_log module and its functions."""

    module: frozenset[str]
    emit: frozenset[str]
    error_type: frozenset[str]


class Scan(NamedTuple):
    """Everything the AST walk learned about the tree."""

    files: tuple[str, ...]
    call_sites: Counter[CallSite]
    vocabulary: tuple[str, ...]
    event_names: tuple[str, ...]
    error_types: tuple[str, ...]
    tagged_logs: tuple[str, ...]


def _scanned_files(root: Path) -> tuple[str, ...]:
    """Every assets module, plus server.py for its single assets.enabled emit."""
    assets = sorted(p.relative_to(root).as_posix() for p in root.glob("app/assets/**/*.py"))
    return (*assets, "server.py")


def _scoped_nodes(tree: ast.Module) -> Iterator[tuple[ast.AST, str]]:
    """Yield every node paired with the name of its innermost enclosing function."""

    def walk(node: ast.AST, scope: str) -> Iterator[tuple[ast.AST, str]]:
        for child in ast.iter_child_nodes(node):
            child_scope = child.name if isinstance(child, FUNCTION_NODES) else scope
            yield child, child_scope
            yield from walk(child, child_scope)

    yield from walk(tree, MODULE_SCOPE)


def _resolve_aliases(tree: ast.Module) -> Aliases:
    module: set[str] = set()
    emit: set[str] = set()
    error_type: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[-1] == EVENT_LOG_NAME:
                    module.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            from_event_log = (node.module or "").rsplit(".", 1)[-1] == EVENT_LOG_NAME
            for alias in node.names:
                if from_event_log and alias.name == "emit":
                    emit.add(alias.asname or alias.name)
                elif from_event_log and alias.name == "error_type":
                    error_type.add(alias.asname or alias.name)
                elif not from_event_log and alias.name == EVENT_LOG_NAME:
                    module.add(alias.asname or alias.name)
    return Aliases(frozenset(module), frozenset(emit), frozenset(error_type))


def _is_emit_call(func: ast.expr, aliases: Aliases) -> bool:
    if isinstance(func, ast.Attribute) and func.attr == "emit" and isinstance(func.value, ast.Name):
        return func.value.id in aliases.module
    return isinstance(func, ast.Name) and func.id in aliases.emit


def _is_error_type_call(value: ast.expr, aliases: Aliases) -> bool:
    """True only for a call to the sanctioned event_log.error_type()."""
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    if isinstance(func, ast.Attribute) and func.attr == "error_type" and isinstance(func.value, ast.Name):
        return func.value.id in aliases.module
    return isinstance(func, ast.Name) and func.id in aliases.error_type


def _carries_tag(node: ast.Call) -> bool:
    return any(
        isinstance(child, ast.Constant) and isinstance(child.value, str) and TAG in child.value
        for child in ast.walk(node)
    )


def _is_log_call(func: ast.expr) -> bool:
    return isinstance(func, ast.Attribute) and func.attr in LOG_METHODS


def _event_of(call: ast.Call) -> str | None:
    """The literal event name, or None when it is not a plain string literal."""
    if len(call.args) != 1:
        return None
    first = call.args[0]
    if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
        return None
    return first.value


def _field_faults(call: ast.Call, aliases: Aliases) -> Iterator[tuple[str, str]]:
    """(category, reason) for every keyword that breaks rule (a) or rule (d)."""
    for keyword in call.keywords:
        if keyword.arg is None:
            yield "vocabulary", "**splat fields cannot be checked statically"
        elif keyword.arg not in ALLOWED_FIELDS:
            yield "vocabulary", f"field {keyword.arg!r} is not in ALLOWED_FIELDS"
        elif keyword.arg == "error_type" and not _is_error_type_call(keyword.value, aliases):
            yield (
                "error_types",
                "error_type= must be a call to event_log.error_type(), got "
                f"{ast.unparse(keyword.value)!r}",
            )


def _file_faults(call: ast.Call, aliases: Aliases) -> Iterator[tuple[str, str]]:
    if _is_emit_call(call.func, aliases):
        event = _event_of(call)
        if event is None or EVENT_NAME_PATTERN.match(event) is None:
            yield "event_names", "the event must be one string literal matching the event pattern"
        yield from _field_faults(call, aliases)
    elif _is_log_call(call.func) and _carries_tag(call):
        yield "tagged_logs", f"log line carries {TAG} outside event_log.emit()"


def _scan_file(root: Path, relative: str) -> tuple[Counter[CallSite], list[tuple[str, str]]]:
    tree = ast.parse((root / relative).read_text(encoding="utf-8"), filename=relative)
    aliases = _resolve_aliases(tree)
    sites: Counter[CallSite] = Counter()
    faults: list[tuple[str, str]] = []
    for node, scope in _scoped_nodes(tree):
        if not isinstance(node, ast.Call):
            continue
        if _is_emit_call(node.func, aliases):
            event = _event_of(node)
            if event is not None and EVENT_NAME_PATTERN.match(event) is not None:
                sites[CallSite(relative, scope, event)] += 1
        for category, reason in _file_faults(node, aliases):
            faults.append((category, f"{relative}:{node.lineno}: {reason}"))
    return sites, faults


def scan_repository(root: Path = REPO_ROOT) -> Scan:
    files = _scanned_files(root)
    sites: Counter[CallSite] = Counter()
    found: dict[str, list[str]] = {"vocabulary": [], "event_names": [], "error_types": [], "tagged_logs": []}
    for relative in files:
        file_sites, faults = _scan_file(root, relative)
        sites += file_sites
        for category, message in faults:
            found[category].append(message)
    return Scan(
        files=files,
        call_sites=sites,
        vocabulary=tuple(found["vocabulary"]),
        event_names=tuple(found["event_names"]),
        error_types=tuple(found["error_types"]),
        tagged_logs=tuple(found["tagged_logs"]),
    )


SCAN = scan_repository()


def test_the_walk_actually_covers_the_assets_tree() -> None:
    """Guards every other check: a broken glob would make them all vacuous."""
    assert "app/assets/event_log.py" in SCAN.files
    assert "app/assets/seeder.py" in SCAN.files
    assert "server.py" in SCAN.files
    assert len(SCAN.files) > 20


def test_emit_fields_stay_inside_the_closed_vocabulary() -> None:
    assert SCAN.vocabulary == ()


def test_emit_events_are_literals_matching_the_event_pattern() -> None:
    assert SCAN.event_names == ()


def test_error_type_values_come_from_event_log_error_type() -> None:
    assert SCAN.error_types == ()


def test_no_other_log_line_carries_the_event_tag() -> None:
    assert SCAN.tagged_logs == ()


def test_call_sites_match_the_manifest() -> None:
    unexpected = SCAN.call_sites - LANDED_CALL_SITES
    missing = LANDED_CALL_SITES - SCAN.call_sites
    assert not unexpected, (
        f"emit call sites not in the manifest: {sorted(unexpected)} — add them, or flip their "
        "EXPECTED_CALL_SITES entry from its pending todo number to None"
    )
    assert not missing, f"manifest call sites absent from the tree: {sorted(missing)}"


def test_manifest_declares_one_request_failed_site_per_route_handler() -> None:
    declared = Counter(
        (site.path, site.function) for site in EXPECTED_CALL_SITES if site.event == "api.request_failed"
    )
    assert set(declared) == REQUEST_FAILED_HANDLERS
    assert set(declared.values()) == {1}


@pytest.mark.parametrize(
    "todo",
    [
        pytest.param(t, marks=pytest.mark.xfail(strict=True, reason=f"todo {t} has not landed yet"))
        for t in PENDING_TODOS
    ],
)
def test_pending_call_sites_have_landed(todo: int) -> None:
    expected = Counter(site for site, owner in EXPECTED_CALL_SITES.items() if owner == todo)
    assert not expected - SCAN.call_sites
