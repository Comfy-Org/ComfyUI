import ast
from collections import Counter
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Literal, NamedTuple, NoReturn

import folder_paths
import pytest

from app.assets.database.queries.records import create_content, create_record
from app.assets.helpers import to_stored_hash
from app.assets.services.asset_management import (
    delete_asset_reference,
    resolve_asset_for_download,
    resolve_hash_to_path,
    update_asset_metadata,
)
from app.assets.services.ingest import (
    create_from_hash,
    register_cached_output,
    register_executed_output,
    register_file_in_place,
    upload_from_temp_path,
)
from app.assets.services.tagging import apply_tags, remove_tags
from app.database.models import Base
from assets_test.services import conftest as service_fixtures

REPO_ROOT = Path(__file__).resolve().parents[2]
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


class CreateSessionCallSite(NamedTuple):
    path: str
    function: str


READ_ONLY_CREATE_SESSION_CALL_SITES = frozenset(
    {
        CreateSessionCallSite("app/assets/api/routes.py", "list_assets_route"),
        # This commits only SELECTs and in-memory queue changes, so a writer retry
        # would acquire a write lease for no persisted work.
        CreateSessionCallSite("app/assets/lifecycle.py", "enqueue_mode_transition_work"),
        CreateSessionCallSite("app/assets/scanner.py", "get_unenriched_assets_for_roots"),
        # This reads the catalogue so the stat walk runs before the writer lease is taken.
        CreateSessionCallSite("app/assets/scanner.py", "observe_references_on_filesystem"),
        # This reads the ids to prune before bounded write transactions mark them missing.
        CreateSessionCallSite("app/assets/scanner.py", "mark_missing_outside_prefixes_safely"),
        # This preflight reads a content path and stats it before outside-transaction hashing.
        CreateSessionCallSite("app/assets/scanner_changes.py", "_preflight_pending_verification"),
        CreateSessionCallSite("app/assets/services/asset_management.py", "get_asset_detail"),
        # These qualify candidate rows with filesystem I/O before the writer lease is taken.
        CreateSessionCallSite("app/assets/services/asset_management.py", "_preflight_hash_resolution"),
        CreateSessionCallSite("app/assets/services/asset_management.py", "resolve_asset_for_download"),
        CreateSessionCallSite("app/assets/services/asset_management.py", "asset_exists"),
        CreateSessionCallSite("app/assets/services/asset_management.py", "get_preview_file_paths"),
        # This reads the transition row before hashing starts outside the writer lease.
        CreateSessionCallSite("app/assets/services/hash_mode_state.py", "_preflight_transition_entry"),
        # These preflights read decision facts before metadata or hash I/O outside the writer lease.
        CreateSessionCallSite("app/assets/services/ingest.py", "_preflight_upload_record"),
        CreateSessionCallSite("app/assets/services/ingest.py", "_preflight_settle_target"),
        CreateSessionCallSite("app/assets/services/ingest.py", "_preflight_cached_registration"),
        CreateSessionCallSite("app/assets/services/tagging.py", "list_tags"),
        CreateSessionCallSite("app/assets/services/tagging.py", "list_tag_histogram"),
    }
)


@pytest.fixture(autouse=True)
def autoclean_unit_test_assets():
    yield


@pytest.fixture(autouse=True)
def initialised_hash_mode():
    yield from service_fixtures.initialised_hash_mode.__wrapped__()


@pytest.fixture(name="db_engine")
def service_db_engine():
    return service_fixtures.db_engine.__wrapped__()


@pytest.fixture(name="session")
def service_session(db_engine, monkeypatch):
    yield from service_fixtures.session.__wrapped__(db_engine, monkeypatch)


@pytest.fixture(name="mock_create_session")
def service_mock_create_session(db_engine):
    yield from service_fixtures.mock_create_session.__wrapped__(db_engine)


@pytest.fixture(name="temp_dir")
def service_temp_dir():
    yield from service_fixtures.temp_dir.__wrapped__()


def _is_create_session_call(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Name)
        and call.func.id == "create_session"
    ) or (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "create_session"
    )


def _create_session_call_sites(path: Path) -> Counter[CreateSessionCallSite]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative_path = path.relative_to(REPO_ROOT).as_posix()
    sites: Counter[CreateSessionCallSite] = Counter()

    def walk(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            child_scope = child.name if isinstance(child, FUNCTION_NODES) else scope
            if isinstance(child, ast.Call) and _is_create_session_call(child):
                sites[CreateSessionCallSite(relative_path, scope)] += 1
            walk(child, child_scope)

    walk(tree, "<module>")
    return sites


def test_create_session_call_sites_stay_read_only() -> None:
    call_sites: Counter[CreateSessionCallSite] = Counter()
    for path in REPO_ROOT.glob("app/assets/**/*.py"):
        call_sites.update(_create_session_call_sites(path))

    expected = Counter(READ_ONLY_CREATE_SESSION_CALL_SITES)
    unexpected = call_sites - expected
    missing = expected - call_sites
    assert not unexpected, (
        f"Unexpected create_session() call sites: {sorted(unexpected)}. "
        "Use run_write_txn for writing sessions."
    )
    assert not missing, f"Read-only create_session() allowlist entries missing from the tree: {sorted(missing)}"


RuntimeCase = Literal[
    "register_executed_output",
    "register_cached_output",
    "register_file_in_place",
    "upload_from_temp_path",
    "create_from_hash",
    "update_asset_metadata",
    "delete_asset_reference",
    "resolve_asset_for_download",
    "resolve_hash_to_path",
    "apply_tags",
    "remove_tags",
]


def _assert_never(value: NoReturn) -> NoReturn:
    raise AssertionError(f"Unexpected runtime case: {value!r}")


def _seed_record(session, path: Path) -> str:
    content = create_content(session, str(path), size_bytes=path.stat().st_size)
    record = create_record(session, content.id, path.name)
    session.commit()
    return record.id


def _find_orm_instance(value, depth: int = 0):
    if isinstance(value, Base):
        return value
    if depth == 2:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            if leak := _find_orm_instance(getattr(value, field.name), depth + 1):
                return leak
    elif isinstance(value, tuple):
        for item in value:
            if leak := _find_orm_instance(item, depth + 1):
                return leak
    elif isinstance(value, list):
        for item in value:
            if leak := _find_orm_instance(item, depth + 1):
                return leak
    elif isinstance(value, dict):
        for key, item in value.items():
            if leak := _find_orm_instance(key, depth + 1):
                return leak
            if leak := _find_orm_instance(item, depth + 1):
                return leak
    return None


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("register_executed_output", id="register_executed_output PASS"),
        pytest.param("register_cached_output", id="register_cached_output PASS"),
        pytest.param("register_file_in_place", id="register_file_in_place PASS"),
        pytest.param("upload_from_temp_path", id="upload_from_temp_path PASS"),
        pytest.param("create_from_hash", id="create_from_hash PASS"),
        pytest.param("update_asset_metadata", id="update_asset_metadata PASS"),
        pytest.param("delete_asset_reference", id="delete_asset_reference PASS"),
        pytest.param("resolve_asset_for_download", id="resolve_asset_for_download PASS"),
        pytest.param("resolve_hash_to_path", id="resolve_hash_to_path PASS"),
        pytest.param("apply_tags", id="apply_tags PASS"),
        pytest.param("remove_tags", id="remove_tags PASS"),
    ],
)
def test_converted_value_results_do_not_leak_orm_instances(
    case: RuntimeCase, session, mock_create_session, monkeypatch, temp_dir, tmp_path
) -> None:
    match case:
        case "register_executed_output":
            path = Path(folder_paths.get_output_directory()) / "write-session-executed.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"executed")
            result = register_executed_output(str(path), job_id="write-session")
        case "register_cached_output":
            path = Path(folder_paths.get_output_directory()) / "write-session-cached.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"cached")
            assert register_executed_output(str(path), job_id="source") is not None
            result = register_cached_output(str(path), job_id="delivery")
        case "register_file_in_place":
            path = tmp_path / "in-place.bin"
            path.write_bytes(b"in-place")
            result = register_file_in_place(str(path), "in-place.bin", ["output"])
        case "upload_from_temp_path":
            path = tmp_path / "upload.part"
            path.write_bytes(b"upload")
            result = upload_from_temp_path(
                str(path), name="upload.bin", tags=["output"], client_filename="upload.bin"
            )
        case "create_from_hash":
            digest = "a" * 64
            path = temp_dir / "hash-source.bin"
            path.write_bytes(b"hash-source")
            monkeypatch.setattr("app.assets.mode.hashing_enabled", lambda: True)
            create_content(session, str(path), to_stored_hash(digest), path.stat().st_size)
            session.commit()
            result = create_from_hash(f"blake3:{digest}", "derived.bin")
        case "update_asset_metadata":
            path = temp_dir / "metadata.bin"
            path.write_bytes(b"metadata")
            result = update_asset_metadata(_seed_record(session, path), name="renamed")
        case "delete_asset_reference":
            path = temp_dir / "delete.bin"
            path.write_bytes(b"delete")
            result = delete_asset_reference(_seed_record(session, path))
        case "resolve_asset_for_download":
            path = temp_dir / "download.bin"
            path.write_bytes(b"download")
            result = resolve_asset_for_download(_seed_record(session, path))
        case "resolve_hash_to_path":
            digest = "b" * 64
            path = temp_dir / "hash-download.bin"
            path.write_bytes(b"hash-download")
            content = create_content(
                session, str(path), to_stored_hash(digest), path.stat().st_size
            )
            create_record(session, content.id, path.name)
            session.commit()
            result = resolve_hash_to_path(f"blake3:{digest}")
        case "apply_tags":
            path = temp_dir / "apply-tags.bin"
            path.write_bytes(b"apply-tags")
            result = apply_tags(_seed_record(session, path), ["tag"])
        case "remove_tags":
            path = temp_dir / "remove-tags.bin"
            path.write_bytes(b"remove-tags")
            content = create_content(session, str(path), size_bytes=path.stat().st_size)
            record = create_record(session, content.id, path.name, tags=["tag"])
            session.commit()
            result = remove_tags(record.id, ["tag"])
        case unreachable:
            _assert_never(unreachable)

    assert result is not None
    assert _find_orm_instance(result) is None
