from __future__ import annotations

import logging
import os
import shutil

import folder_paths
from sqlalchemy import select

from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import delete_record
from app.assets.helpers import sql_path_under_prefix
from app.assets.services.hash_mode_state import drain_transition_queue
from app.assets.services.hash_mode_state import enqueue_transition_work
from app.assets.services.hash_mode_state import record_transition_intent
from app.database.db import can_create_session, create_session
from comfy.cli_args import args

_excluded_scan_roots: set[str] = set()
_hash_mode_transition: str | None = None


def get_excluded_scan_roots() -> frozenset[str]:
    return frozenset(_excluded_scan_roots)


def record_hash_mode_transition_intent() -> None:
    global _hash_mode_transition

    with create_session() as session:
        _hash_mode_transition = record_transition_intent(session)
        session.commit()


def enqueue_mode_transition_work() -> None:
    with create_session() as session:
        enqueue_transition_work(session, _hash_mode_transition)
        session.commit()


def drain_mode_transition_work() -> None:
    with create_session() as session:
        drain_transition_queue(session)
        session.commit()


def wipe_temp_db_rows(session) -> tuple[int, int]:
    try:
        temp_root = os.path.abspath(folder_paths.get_temp_directory())
    except OSError:
        return 0, 0
    # These rows are hard-deleted, so the predicate must stay case-SENSITIVE: admitting a
    # case-different persistent directory destroys user assets.
    under_temp = sql_path_under_prefix(AssetContent.path, temp_root)

    temp_record_ids = list(
        session.scalars(
            select(Asset.id)
            .join(AssetContent, Asset.content_id == AssetContent.id)
            .where(under_temp)
        )
    )

    records_deleted = 0
    for record_id in temp_record_ids:
        delete_record(session, record_id)
        records_deleted += 1

    contents_deleted = 0
    for content in session.scalars(select(AssetContent).where(under_temp)).all():
        session.delete(content)
        contents_deleted += 1

    session.flush()
    return records_deleted, contents_deleted


def cleanup_temp_filesystem() -> bool:
    temp_dir = os.path.abspath(folder_paths.get_temp_directory())
    if not os.path.exists(temp_dir):
        return True
    try:
        shutil.rmtree(temp_dir)
        return True
    except OSError as exc:
        logging.warning(
            "Failed to remove temp directory %s: %s — excluding from scan for this process",
            temp_dir,
            exc,
        )
        _excluded_scan_roots.add(temp_dir)
        return False


def start_asset_seeder() -> bool:
    from app.assets.seeder import asset_seeder

    started = asset_seeder.start(
        roots=("models", "input", "output"),
        prune_first=True,
        compute_hashes=args.enable_asset_hashing,
    )
    if started:
        logging.info("Background asset scan initiated for models, input, output")
    return started


def run_asset_startup() -> None:
    try:
        with create_session() as session:
            wipe_temp_db_rows(session)
            session.commit()
    except Exception:
        logging.exception("Temp DB row wipe failed; skipping filesystem cleanup")
        enqueue_mode_transition_work()
        drain_mode_transition_work()
        start_asset_seeder()
        return
    cleanup_temp_filesystem()
    enqueue_mode_transition_work()
    drain_mode_transition_work()
    start_asset_seeder()


def run_startup(*, enable_assets: bool) -> None:
    try:
        if enable_assets:
            run_asset_startup()
        else:
            cleanup_temp_filesystem()
    except Exception:
        logging.exception("Asset startup maintenance failed")


def run_asset_shutdown_cleanup() -> None:
    try:
        with create_session() as session:
            wipe_temp_db_rows(session)
            session.commit()
    except Exception:
        logging.exception("Temp DB row wipe failed during shutdown")
    finally:
        cleanup_temp_filesystem()


def run_shutdown() -> None:
    try:
        if can_create_session():
            run_asset_shutdown_cleanup()
        else:
            cleanup_temp_filesystem()
    except Exception:
        logging.exception("Asset shutdown cleanup failed")
