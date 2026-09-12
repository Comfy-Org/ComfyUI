from __future__ import annotations

import os
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.assets.database.queries import create_content, create_record
from app.assets.scanner import get_unenriched_assets_for_roots
from app.assets.scanner_changes import is_path_under_prefixes

from .path_prefix_cases import prefix_case_paths


@contextmanager
def _reuse_session(session: Session) -> Iterator[Session]:
    yield session


def _raw_system_metadata(session: Session, record_id: str) -> object:
    return session.execute(
        sa.text("SELECT system_metadata FROM assets WHERE id = :id"),
        {"id": record_id},
    ).scalar()


def test_create_record_stores_none_system_metadata_as_sql_null(
    session: Session,
) -> None:
    content = create_content(session, "/models/no-meta.safetensors", hash=None)

    record = create_record(session, content.id, "no-meta.safetensors")
    session.commit()

    assert _raw_system_metadata(session, record.id) is None


def test_off_mode_returns_metadata_less_seeded_asset(
    session: Session, temp_dir: Path
) -> None:
    path = temp_dir / "fresh.safetensors"
    content = create_content(session, str(path), hash=None)
    record = create_record(session, content.id, path.name)
    session.commit()
    record_id = record.id

    with (
        patch("app.assets.scanner.create_session", lambda: _reuse_session(session)),
        patch(
            "app.assets.scanner.get_scan_prefixes_for_root",
            return_value=[str(temp_dir)],
        ),
    ):
        rows = get_unenriched_assets_for_roots(("models",), compute_hashes=False)

    assert record_id in {row.record_id for row in rows}


def test_off_mode_excludes_asset_with_system_metadata(
    session: Session, temp_dir: Path
) -> None:
    path = temp_dir / "enriched.safetensors"
    content = create_content(session, str(path), hash=None)
    record = create_record(
        session,
        content.id,
        path.name,
        system_metadata={"architecture": "flux"},
    )
    session.commit()
    record_id = record.id

    with (
        patch("app.assets.scanner.create_session", lambda: _reuse_session(session)),
        patch(
            "app.assets.scanner.get_scan_prefixes_for_root",
            return_value=[str(temp_dir)],
        ),
    ):
        rows = get_unenriched_assets_for_roots(("models",), compute_hashes=False)

    assert record_id not in {row.record_id for row in rows}


def test_query_pushes_limit_into_sql(session: Session, temp_dir: Path) -> None:
    for index in range(5):
        path = temp_dir / f"cand-{index}.safetensors"
        content = create_content(session, str(path), hash=None)
        create_record(session, content.id, path.name)
    session.commit()

    statements: list[str] = []

    def _capture(_conn, _cursor, statement, _params, _context, _executemany) -> None:
        statements.append(statement)

    engine = session.bind
    sa.event.listen(engine, "before_cursor_execute", _capture)
    try:
        with (
            patch("app.assets.scanner.create_session", lambda: _reuse_session(session)),
            patch(
                "app.assets.scanner.get_scan_prefixes_for_root",
                return_value=[str(temp_dir)],
            ),
        ):
            rows = get_unenriched_assets_for_roots(
                ("models",), compute_hashes=False, limit=2
            )
    finally:
        sa.event.remove(engine, "before_cursor_execute", _capture)

    assert len(rows) == 2
    content_selects = [
        s for s in statements if "asset_contents" in s.lower() and s.lower().lstrip().startswith("select")
    ]
    assert content_selects, f"expected a SELECT over asset_contents; got {statements}"
    assert any("limit" in s.lower() for s in content_selects), (
        f"enrichment query must push LIMIT into SQL; got: {content_selects}"
    )


def test_prefix_filter_matches_is_path_under_prefixes(
    session: Session, temp_dir: Path
) -> None:
    prefix = str(temp_dir)
    inside_path = os.path.join(prefix, "sub", "inside.safetensors")
    sibling_path = prefix + "-sibling" + os.sep + "outside.safetensors"

    inside = create_record(
        session, create_content(session, inside_path, hash=None).id, "inside.safetensors"
    )
    sibling = create_record(
        session, create_content(session, sibling_path, hash=None).id, "outside.safetensors"
    )
    session.commit()

    with (
        patch("app.assets.scanner.create_session", lambda: _reuse_session(session)),
        patch(
            "app.assets.scanner.get_scan_prefixes_for_root",
            return_value=[prefix],
        ),
    ):
        returned = {
            row.record_id
            for row in get_unenriched_assets_for_roots(("models",), compute_hashes=False)
        }

    assert is_path_under_prefixes(inside_path, [prefix]) is True
    assert is_path_under_prefixes(sibling_path, [prefix]) is False
    assert inside.id in returned
    assert sibling.id not in returned


def _seed_paths(session: Session, paths: list[str]) -> dict[str, str]:
    by_record: dict[str, str] = {}
    for index, path in enumerate(paths):
        content = create_content(session, path, hash=None)
        record = create_record(session, content.id, f"case-{index}.safetensors")
        by_record[record.id] = content.path
    session.commit()
    return by_record


def _candidate_paths(session: Session, prefix: str) -> set[str]:
    with (
        patch("app.assets.scanner.create_session", lambda: _reuse_session(session)),
        patch(
            "app.assets.scanner.get_scan_prefixes_for_root",
            return_value=[prefix],
        ),
    ):
        rows = get_unenriched_assets_for_roots(("models",), compute_hashes=False)
    return {row.file_path for row in rows}


def test_prefix_filter_result_set_equals_python_predicate(
    session: Session, temp_dir: Path
) -> None:
    root = str(temp_dir / "root")
    by_record = _seed_paths(session, prefix_case_paths(root))
    stored = set(by_record.values())

    returned = _candidate_paths(session, root)

    expected = {p for p in stored if is_path_under_prefixes(p, [root])}
    assert returned == expected
    assert expected and expected != stored


def test_prefix_holding_metacharacters_matches_only_literal_children(
    session: Session, temp_dir: Path
) -> None:
    root = str(temp_dir / "a_b%c*d?e[f")
    inside_path = os.path.join(root, "inside.safetensors")
    decoy_path = os.path.join(str(temp_dir), "aXbYcZdWeQf", "decoy.safetensors")
    _seed_paths(session, [inside_path, decoy_path])

    returned = _candidate_paths(session, root)

    assert is_path_under_prefixes(decoy_path, [root]) is False
    assert returned == {inside_path}
