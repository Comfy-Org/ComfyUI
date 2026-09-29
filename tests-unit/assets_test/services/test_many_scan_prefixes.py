"""Prefix filters over hundreds of scan folders.

Each prefix adds terms to one OR, and SQLite rejects an expression tree deeper than
1000, so a single statement over about 500 prefixes failed every scan. The filters now
run in batches; these pin that the batched results equal the single-statement ones.
"""

from __future__ import annotations

import logging
import ntpath
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from app.assets import helpers, scanner, seeder as seeder_module
from app.assets.database.models import Asset, AssetContent, Base
from app.assets.database.queries import create_content, create_record, mark_content_missing
from app.assets.helpers import (
    PREFIX_BATCH_SIZE,
    sql_path_under_prefix,
    sql_path_under_prefix_batches,
)
from app.assets.scanner import get_unenriched_assets_for_roots, live_references_safely
from app.assets.scanner_changes import live_contents_under_prefixes

from .path_prefix_cases import expected_prefix_case_paths, prefix_case_paths

PREFIX_COUNTS = [1, PREFIX_BATCH_SIZE, PREFIX_BATCH_SIZE + 1, 499, 500, 2000]


@contextmanager
def _reuse_session(session: Session):
    yield session


def _folders(root: Path, count: int) -> list[str]:
    return [str(root / f"b{i:05d}" / "checkpoints") for i in range(count)]


def _seed(session: Session, prefixes: list[str]) -> set[str]:
    """One live row under each prefix, plus decoys none of them match."""
    inside = set()
    for prefix in dict.fromkeys(prefixes):
        path = os.path.join(prefix, "model.safetensors")
        create_record(session, create_content(session, path).id, "model.safetensors")
        inside.add(path)
        create_content(session, prefix + "-other" + os.sep + "decoy.safetensors")
    missing = create_content(session, os.path.join(prefixes[-1], "gone.safetensors"))
    mark_content_missing(session, missing.id)
    session.commit()
    return inside


def test_batches_cover_every_prefix_in_order():
    prefixes = [f"/p{i}" for i in range(2 * PREFIX_BATCH_SIZE + 1)]

    batches = sql_path_under_prefix_batches(AssetContent.path, prefixes)

    assert len(batches) == 3
    assert sql_path_under_prefix_batches(AssetContent.path, []) == []


@pytest.mark.parametrize("count", [1, 20, PREFIX_BATCH_SIZE])
def test_up_to_one_batch_compiles_to_the_single_statement_predicate(count):
    prefixes = [f"/p{i}" for i in range(count)]
    single = sa.or_(*(sql_path_under_prefix(AssetContent.path, p) for p in prefixes))

    [batch] = sql_path_under_prefix_batches(AssetContent.path, prefixes)

    compile_kwargs = {"literal_binds": True}
    assert str(batch.compile(compile_kwargs=compile_kwargs)) == str(
        single.compile(compile_kwargs=compile_kwargs)
    )


def test_a_single_statement_over_500_prefixes_exceeds_sqlite_expression_depth(session, temp_dir):
    """The limit the batching works around; if SQLite lifts it this test says so."""
    prefixes = _folders(temp_dir, 500)
    stmt = sa.select(AssetContent.id).where(
        sa.or_(*(sql_path_under_prefix(AssetContent.path, p) for p in prefixes))
    )

    with pytest.raises(sa.exc.OperationalError, match="Expression tree is too large"):
        session.execute(stmt).all()


@pytest.mark.parametrize("count", PREFIX_COUNTS)
def test_live_contents_under_many_prefixes(session, temp_dir, count):
    prefixes = _folders(temp_dir, count)
    inside = _seed(session, prefixes)

    returned = [content.path for content in live_contents_under_prefixes(session, prefixes)]

    assert sorted(returned) == sorted(inside)


@pytest.mark.parametrize("count", PREFIX_COUNTS)
def test_live_references_under_many_prefixes(session, temp_dir, count):
    prefixes = _folders(temp_dir, count)
    inside = _seed(session, prefixes)

    with (
        patch.object(scanner, "create_session", lambda: _reuse_session(session)),
        patch.object(scanner, "get_scan_prefixes_for_root", lambda _root: prefixes),
    ):
        live = live_references_safely("models")

    assert set(live) == inside
    assert all(len(observations) == 1 for observations in live.values())


@pytest.mark.parametrize("count", PREFIX_COUNTS)
def test_unenriched_candidates_under_many_prefixes(session, temp_dir, count):
    prefixes = _folders(temp_dir, count)
    inside = _seed(session, prefixes)

    with (
        patch.object(scanner, "create_session", lambda: _reuse_session(session)),
        patch.object(scanner, "get_scan_prefixes_for_root", lambda _root: prefixes),
    ):
        rows = get_unenriched_assets_for_roots(("models",), compute_hashes=False, limit=10_000)

    assert sorted(row.file_path for row in rows) == sorted(inside)


def _nested_prefixes(temp_dir: Path) -> tuple[list[str], set[str]]:
    """Duplicate and nested prefixes whose matches straddle batch boundaries."""
    prefixes = _folders(temp_dir, 2 * PREFIX_BATCH_SIZE + 50)
    outer = str(temp_dir / "shared")
    inner = str(temp_dir / "shared" / "inner")
    prefixes[0] = outer
    prefixes[PREFIX_BATCH_SIZE + 3] = inner
    prefixes[-1] = outer
    return prefixes, {outer, inner}


def test_nested_prefixes_across_batches_yield_each_row_once(session, temp_dir):
    prefixes, _ = _nested_prefixes(temp_dir)
    inside = _seed(session, prefixes)
    assert len(inside) < len(prefixes)

    contents = [content.id for content in live_contents_under_prefixes(session, prefixes)]
    with (
        patch.object(scanner, "create_session", lambda: _reuse_session(session)),
        patch.object(scanner, "get_scan_prefixes_for_root", lambda _root: prefixes),
    ):
        live = live_references_safely("models")
        rows = get_unenriched_assets_for_roots(("models",), compute_hashes=False, limit=10_000)

    # The outer prefix also takes the inner prefix's "-other" decoy.
    under = inside | {str(temp_dir / "shared" / "inner-other" / "decoy.safetensors")}
    assert len(contents) == len(set(contents)) == len(under)
    assert set(live) == under
    assert all(len(observations) == 1 for observations in live.values())
    assert len(rows) == len({row.record_id for row in rows}) == len(inside)


@pytest.mark.parametrize("limit", [7, 150, 1000])
def test_unenriched_paging_across_batches_matches_a_single_ordered_scan(session, temp_dir, limit):
    """Keyset pages over many batches equal the pages of one ordered query."""
    prefixes, _ = _nested_prefixes(temp_dir)
    _seed(session, prefixes)
    expected = list(
        session.execute(
            sa.select(Asset.id)
            .join(AssetContent, Asset.content_id == AssetContent.id)
            .where(AssetContent.is_missing.is_(False), AssetContent.path.not_like("%-other%"))
            .order_by(Asset.id)
        ).scalars()
    )

    pages: list[list[str]] = []
    last_seen_id = None
    with (
        patch.object(scanner, "create_session", lambda: _reuse_session(session)),
        patch.object(scanner, "get_scan_prefixes_for_root", lambda _root: prefixes),
    ):
        while True:
            rows = get_unenriched_assets_for_roots(
                ("models",), compute_hashes=False, limit=limit, last_seen_id=last_seen_id
            )
            if not rows:
                break
            pages.append([row.record_id for row in rows])
            last_seen_id = rows[-1].record_id

    assert pages == [expected[i:i + limit] for i in range(0, len(expected), limit)]


def test_path_semantics_hold_in_a_later_batch(session, temp_dir):
    root = str(temp_dir / "root")
    for path, _ in prefix_case_paths(root):
        create_content(session, path)
    session.commit()
    prefixes = _folders(temp_dir / "elsewhere", 499) + [root]

    returned = {content.path for content in live_contents_under_prefixes(session, prefixes)}

    assert returned == expected_prefix_case_paths(root)


def test_windows_paths_in_a_later_batch(session, monkeypatch):
    """Drive-letter paths keep exact-or-under, the separator bound and case sensitivity."""
    monkeypatch.setattr(helpers, "os", SimpleNamespace(path=ntpath, sep="\\"))
    root = "C:\\models\\target"
    stored = {
        "C:\\models\\target": True,
        "C:\\models\\target\\ckpt.safetensors": True,
        "C:\\models\\target\\sub\\lora.safetensors": True,
        "C:\\models\\targetx\\ckpt.safetensors": False,
        "C:\\models\\target-other\\ckpt.safetensors": False,
        "C:\\Models\\Target\\ckpt.safetensors": False,
        "D:\\models\\target\\ckpt.safetensors": False,
    }
    # Stored as Windows' abspath writes them; this host's abspath would mangle them.
    session.add_all(AssetContent(path=path, size_bytes=0) for path in stored)
    session.commit()
    prefixes = [f"C:\\other\\b{i:05d}" for i in range(2 * PREFIX_BATCH_SIZE + 5)] + ["C:\\models\\target\\"]

    returned = {content.path for content in live_contents_under_prefixes(session, prefixes)}

    assert returned == {path for path, inside in stored.items() if inside}
    assert root in returned


# --- end to end: a real scan over hundreds of model folders ---


MODEL_FOLDERS = 520


@pytest.fixture
def model_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """MODEL_FOLDERS registered folders; a file in every 40th and the last.

    The prefix count is what broke the scan. Fewer files keep the fast phase's
    per-file folder checks from dominating the test's runtime.
    """
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'assets.db'}")
    Base.metadata.create_all(engine)
    monkeypatch.setattr("app.database.db.Session", sessionmaker(bind=engine))
    monkeypatch.setattr("app.database.db.WriteSession", sessionmaker(bind=engine))

    folders = [tmp_path / "models" / f"base{i:04d}" / "checkpoints" for i in range(MODEL_FOLDERS)]
    files = []
    for index, folder in enumerate(folders):
        folder.mkdir(parents=True)
        if index % 40 == 0 or index == MODEL_FOLDERS - 1:
            files.append(folder / f"model{index:04d}.safetensors")
            files[-1].write_bytes(b"\0" * 16)
    folder_strs = [str(f) for f in folders]
    monkeypatch.setattr(
        scanner,
        "get_comfy_models_folders",
        lambda: [("checkpoints", folder_strs, {".safetensors"})],
    )
    monkeypatch.setattr(
        "folder_paths.folder_names_and_paths",
        {"checkpoints": (folder_strs, {".safetensors"})},
    )
    monkeypatch.setattr("folder_paths.filename_list_cache", {})
    monkeypatch.setattr(seeder_module, "dependencies_available", lambda: True)
    yield files
    engine.dispose()


def _full_scan(caplog: pytest.LogCaptureFixture) -> list[str]:
    caplog.clear()
    seeder = seeder_module._AssetSeeder()
    with caplog.at_level(logging.INFO):
        assert seeder.start(roots=("models",), phase=seeder_module.ScanPhase.FULL)
        assert seeder.wait(timeout=120)
    return [record.getMessage() for record in caplog.records]


def _live_model_paths() -> set[str]:
    from app.database.db import create_session

    with create_session() as session:
        return set(
            session.scalars(sa.select(AssetContent.path).where(AssetContent.is_missing.is_(False)))
        )


def test_full_scan_over_520_model_folders_completes(model_files, caplog):
    messages = _full_scan(caplog)

    assert [m for m in messages if "scan_failed" in m] == []
    assert any("seeder.scan_completed" in m for m in messages), messages
    assert _live_model_paths() == {str(f) for f in model_files}
    assert get_unenriched_assets_for_roots(("models",), compute_hashes=False) == []

    # A rescan runs the per-prefix sync, which must see the file that went away.
    gone = model_files[-1]
    gone.unlink()
    messages = _full_scan(caplog)

    assert [m for m in messages if "scan_failed" in m] == []
    assert _live_model_paths() == {str(f) for f in model_files} - {str(gone)}
