import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, TypedDict

import folder_paths
import sqlalchemy as sa
from sqlalchemy.orm import Session
from app.assets import mode
from app.assets.database.queries import (
    mark_content_missing,
    create_content,
    create_record,
)
from app.assets.database.models import Asset, AssetContent
from app.assets.helpers import sql_path_under_prefix, to_stored_hash
from app.assets.lifecycle import get_excluded_scan_roots
from app.assets.scanner_changes import (
    clear_pending_verifications,
    detect_content_change,
    drain_pending_verifications,
    is_path_under_prefixes,
    live_contents_under_prefixes,
    pending_recovery_count,
    recover_missing_content,
)
from app.assets.scanner_admission import (
    PARTIAL_DOWNLOAD_EXTENSIONS as PARTIAL_DOWNLOAD_EXTENSIONS,
    _WATCH_LIST as _WATCH_LIST,
    _WatchEntry as _WatchEntry,
    _should_skip_extension,
    _two_stat_admit,
    tick_watch_list as tick_watch_list,
)
from app.assets.services.file_utils import get_mtime_ns, is_visible, list_files_recursively
from app.assets.services.image_dimensions import extract_image_dimensions
from app.assets.services.metadata_extract import ExtractedMetadata, extract_file_metadata
from app.assets.services.path_utils import (
    compute_loader_path,
    get_comfy_models_folders,
    get_name_and_tags_from_asset_path,
)
from app.assets.services.ingest import _discard_unreferenced_content
from app.assets.services.snapshot_hash import snapshot_hash
from app.database.db import create_session

__all__ = [
    "clear_pending_verifications",
    "drain_pending_verifications",
    "pending_recovery_count",
]


# Temp is deliberately absent: it is wiped before every scan, so walking it finds nothing.
RootType = Literal["models", "input", "output"]


class SeedAssetSpec(TypedDict):

    abs_path: str
    size_bytes: int
    mtime_ns: int
    info_name: str
    tags: list[str]
    fname: str | None
    metadata: ExtractedMetadata | None
    mime_type: str | None
    job_id: str | None


@dataclass(frozen=True, slots=True)
class UnenrichedContent:
    content_id: str
    record_id: str
    file_path: str


def get_scan_prefixes_for_root(root: RootType) -> list[str]:
    if root == "models":
        bases: list[str] = []
        for _bucket, paths, _exts in get_comfy_models_folders():
            bases.extend(paths)
        return [os.path.abspath(p) for p in bases]
    if root == "input":
        return [os.path.abspath(folder_paths.get_input_directory())]
    if root == "output":
        return [os.path.abspath(folder_paths.get_output_directory())]
    return []


def get_owned_prefixes() -> list[str]:
    """Every directory an asset may live in; references outside these are marked missing."""
    scan_roots: tuple[RootType, ...] = ("models", "input", "output")
    prefixes = [p for root in scan_roots for p in get_scan_prefixes_for_root(root)]
    return prefixes + get_temp_prefixes()


def get_temp_prefixes() -> list[str]:
    temp_dir = os.path.abspath(folder_paths.get_temp_directory())
    if temp_dir in get_excluded_scan_roots():
        return []
    return [temp_dir]


def collect_models_files() -> list[str]:
    out: list[str] = []
    for folder_name, bases, _exts in get_comfy_models_folders():
        rel_files = folder_paths.get_filename_list(folder_name) or []
        for rel_path in rel_files:
            if not all(is_visible(part) for part in Path(rel_path).parts):
                continue
            abs_path = folder_paths.get_full_path(folder_name, rel_path)
            if not abs_path:
                continue
            abs_path = os.path.abspath(abs_path)
            allowed = False
            abs_p = Path(abs_path)
            for b in bases:
                if abs_p.is_relative_to(os.path.abspath(b)):
                    allowed = True
                    break
            if allowed:
                out.append(abs_path)
    return out


def sync_references_with_filesystem(
    session,
    root: RootType,
    collect_existing_paths: bool = False,
) -> set[str] | None:
    return sync_prefixes_with_filesystem(
        session,
        get_scan_prefixes_for_root(root),
        collect_existing_paths=collect_existing_paths,
    )


def sync_prefixes_with_filesystem(
    session: Session,
    prefixes: list[str],
    collect_existing_paths: bool = False,
) -> set[str] | None:
    if not prefixes:
        return set() if collect_existing_paths else None

    survivors: set[str] = set()
    for content in live_contents_under_prefixes(session, prefixes):
        try:
            stat_result = os.stat(content.path, follow_symlinks=True)
        except FileNotFoundError:
            mark_content_missing(session, content.id)
        except PermissionError:
            logging.debug("Permission denied accessing %s", content.path)
        except OSError as e:
            logging.debug("OSError checking %s: %s", content.path, e)
            mark_content_missing(session, content.id)
        else:
            detect_content_change(
                session,
                content,
                stat_result,
                hashing_is_enabled=mode.hashing_enabled(),
            )
            survivors.add(os.path.abspath(content.path))

    return survivors if collect_existing_paths else None


def _is_under_prefixes(path: str, prefixes: list[str]) -> bool:
    return is_path_under_prefixes(path, prefixes)


def sync_root_safely(root: RootType) -> set[str]:
    """Sync a single root's references with the filesystem.

    Returns survivors (existing paths) or empty set on failure.
    """
    try:
        with create_session() as sess:
            survivors = sync_references_with_filesystem(
                sess,
                root,
                collect_existing_paths=True,
            )
            sess.commit()
            return survivors or set()
    except Exception as e:
        logging.exception("fast DB scan failed for %s: %s", root, e)
        return set()


def sync_temp_references_safely() -> None:
    """Retire temp references whose file is gone; temp is never scanned, so nothing else stats them."""
    try:
        with create_session() as sess:
            sync_prefixes_with_filesystem(sess, get_temp_prefixes())
            sess.commit()
    except Exception as e:
        logging.exception("temp reference sync failed: %s", e)


def mark_missing_outside_prefixes_safely(prefixes: list[str]) -> int:
    """Mark references as missing when outside the given prefixes.

    This is a non-destructive soft-delete. Returns count marked or 0 on failure.
    """
    try:
        with create_session() as sess:
            count = mark_contents_missing_outside_prefixes(sess, prefixes)
            sess.commit()
            return count
    except Exception as e:
        logging.exception("marking missing assets failed: %s", e)
        return 0


def mark_contents_missing_outside_prefixes(
    session: Session, prefixes: list[str]
) -> int:
    contents = session.scalars(
        sa.select(AssetContent).where(AssetContent.is_missing.is_(False))
    )
    missing = [content for content in contents if not _is_under_prefixes(content.path, prefixes)]
    for content in missing:
        mark_content_missing(session, content.id)
    return len(missing)


def collect_paths_for_roots(roots: tuple[RootType, ...]) -> list[str]:
    """Collect all file paths for the given roots."""
    paths: list[str] = []
    if "models" in roots:
        paths.extend(collect_models_files())
    if "input" in roots:
        paths.extend(list_files_recursively(folder_paths.get_input_directory()))
    if "output" in roots:
        paths.extend(list_files_recursively(folder_paths.get_output_directory()))
    return paths


def build_asset_specs(
    paths: list[str],
    existing_paths: set[str],
    enable_metadata_extraction: bool = True,
) -> tuple[list[SeedAssetSpec], set[str], int]:
    """Build asset specs from paths, returning (specs, tag_pool, skipped_count).

    Args:
        paths: List of file paths to process
        existing_paths: Set of paths that already exist in the database
        enable_metadata_extraction: If True, extract tier 1 & 2 metadata
    """
    specs: list[SeedAssetSpec] = []
    tag_pool: set[str] = set()
    skipped = 0
    candidates: list[tuple[str, os.stat_result]] = []

    for p in paths:
        abs_p = os.path.abspath(p)
        if _should_skip_extension(abs_p):
            skipped += 1
            continue
        if abs_p in existing_paths:
            skipped += 1
            continue
        try:
            stat_p = os.stat(abs_p, follow_symlinks=True)
        except OSError:
            continue
        if not stat_p.st_size:
            continue
        candidates.append((abs_p, stat_p))

    admitted_paths, _ = _two_stat_admit(candidates)
    candidate_stats = dict(candidates)
    for abs_p in admitted_paths:
        stat_p = candidate_stats[abs_p]
        name, tags = get_name_and_tags_from_asset_path(abs_p)
        rel_fname = compute_loader_path(abs_p)

        # Extract metadata (tier 1: filesystem, tier 2: safetensors header)
        metadata = None
        if enable_metadata_extraction:
            metadata = extract_file_metadata(
                abs_p,
                stat_result=stat_p,
                relative_filename=rel_fname,
            )

        mime_type = metadata.content_type if metadata else None
        specs.append(
            {
                "abs_path": abs_p,
                "size_bytes": stat_p.st_size,
                "mtime_ns": get_mtime_ns(stat_p),
                "info_name": name,
                "tags": tags,
                "fname": rel_fname,
                "metadata": metadata,
                "mime_type": mime_type,
                "job_id": None,
            }
        )
        tag_pool.update(tags)

    return specs, tag_pool, skipped


def seed_asset_specs(session: Session, specs: list[SeedAssetSpec]) -> int:
    created = 0
    created_content_ids: list[str] = []
    try:
        for spec in specs:
            path = os.path.abspath(spec["abs_path"])
            try:
                stat_result = os.stat(path, follow_symlinks=True)
            except OSError:
                logging.warning("Skipping vanished asset during scan: %s", path)
                continue
            try:
                recovery = recover_missing_content(
                    session,
                    path,
                    stat_result,
                    hashing_is_enabled=mode.hashing_enabled(),
                )
            except OSError:
                logging.warning("Skipping vanished asset during scan: %s", path)
                continue
            if recovery != "no_match":
                continue
            content = create_content(
                session,
                path=path,
                hash=None,
                size_bytes=spec["size_bytes"],
                mtime_ns=spec["mtime_ns"],
            )
            created_content_ids.append(content.id)
            existing_record = session.scalar(
                sa.select(Asset.id).where(Asset.content_id == content.id).limit(1)
            )
            if existing_record is not None:
                continue
            create_record(
                session,
                content_id=content.id,
                name=spec["info_name"],
                mime_type=spec["mime_type"],
                job_id=spec["job_id"],
                loader_path=spec["fname"],
                tags=spec["tags"],
            )
            created += 1
    except Exception:
        session.rollback()
        for content_id in created_content_ids:
            _discard_unreferenced_content(session, content_id)
        raise
    return created


def insert_asset_specs(specs: list[SeedAssetSpec], _tag_pool: set[str]) -> int:
    if not specs:
        return 0
    with create_session() as sess:
        created = seed_asset_specs(sess, specs)
        sess.commit()
        return created


def get_unenriched_assets_for_roots(
    roots: tuple[RootType, ...],
    compute_hashes: bool,
    limit: int = 1000,
) -> list[UnenrichedContent]:
    prefixes: list[str] = []
    for root in roots:
        prefixes.extend(get_scan_prefixes_for_root(root))

    if not prefixes:
        return []

    with create_session() as sess:
        query = (
            sa.select(AssetContent.id, Asset.id, AssetContent.path)
            .join(Asset, Asset.content_id == AssetContent.id)
            .where(AssetContent.is_missing.is_(False))
        )
        if compute_hashes:
            query = query.where(
                sa.or_(
                    AssetContent.hash.is_(None),
                    Asset.system_metadata.is_(None),
                )
            )
        else:
            query = query.where(Asset.system_metadata.is_(None))
        query = query.where(
            sa.or_(
                *(sql_path_under_prefix(AssetContent.path, p) for p in prefixes)
            )
        )
        rows = sess.execute(query.order_by(Asset.id).limit(limit)).all()

    return [
        UnenrichedContent(content_id, record_id, file_path)
        for content_id, record_id, file_path in rows
    ]


def enrich_asset(
    session,
    file_path: str,
    content_id: str,
    record_id: str,
    extract_metadata: bool = True,
    compute_hash: bool = False,
) -> bool:
    """Enrich a single asset with metadata and/or hash.

    Args:
        session: Database session (caller manages lifecycle)
        file_path: Absolute path to the file
        content_id: ID of the content to update
        record_id: ID of the record to update
        extract_metadata: If True, extract safetensors header and mime type
        compute_hash: If True, compute blake3 hash

    Returns:
        Whether enrichment changed the B-schema record or content
    """
    try:
        stat_p = os.stat(file_path, follow_symlinks=True)
    except OSError:
        return False

    initial_mtime_ns = get_mtime_ns(stat_p)
    rel_fname = compute_loader_path(file_path)
    mime_type: str | None = None
    metadata = None

    if extract_metadata:
        metadata = extract_file_metadata(
            file_path,
            stat_result=stat_p,
            relative_filename=rel_fname,
        )
        if metadata:
            mime_type = metadata.content_type

    content = session.get(AssetContent, content_id)

    digest: str | None = None
    stored_hash: str | None = None
    verified_stat: os.stat_result | None = None
    if compute_hash and content is not None and content.hash is None:
        try:
            snapshot = snapshot_hash(file_path)
            if snapshot is None:
                logging.warning(
                    "File modified during hashing (snapshot unstable), discarding hash: %s",
                    file_path,
                )
                return False
            digest, verified_stat = snapshot
            stored_hash = to_stored_hash(digest)
        except Exception as e:
            logging.warning("Failed to hash %s: %s", file_path, e)

    record = session.get(Asset, record_id)
    if content is None or record is None or content.mtime_ns != initial_mtime_ns:
        session.rollback()
        logging.info(
            "Content %s mtime changed during enrichment, discarding stale result",
            content_id,
        )
        return False

    # Non-NULL system_metadata permanently excludes the row from re-enrichment, so a
    # disagreement here must discard the metadata too, not just the hash.
    if verified_stat is not None and (
        get_mtime_ns(verified_stat) != initial_mtime_ns
        or verified_stat.st_size != stat_p.st_size
    ):
        session.rollback()
        logging.info(
            "Content %s changed between its metadata read and its hash read, "
            "discarding stale result",
            content_id,
        )
        return False

    if extract_metadata and metadata:
        system_metadata = metadata.to_user_metadata()
        if mime_type and mime_type.startswith("image/"):
            dims = extract_image_dimensions(file_path, mime_type=mime_type)
            if dims:
                system_metadata.update(dims)
        record.system_metadata = {**(record.system_metadata or {}), **system_metadata}

    if stored_hash:
        content.hash = stored_hash
    if mime_type:
        record.mime_type = mime_type

    session.commit()

    return stored_hash is not None or metadata is not None or mime_type is not None


def enrich_assets_batch(
    rows: list[UnenrichedContent],
    extract_metadata: bool = True,
    compute_hash: bool = False,
    interrupt_check: Callable[[], bool] | None = None,
) -> tuple[int, list[str]]:
    """Enrich a batch of assets.

    Uses a single DB session for the entire batch, committing after each
    individual asset to avoid long-held transactions while eliminating
    per-asset session creation overhead.

    Args:
        rows: List of UnenrichedReferenceRow from get_unenriched_assets_for_roots
        extract_metadata: If True, extract metadata for each asset
        compute_hash: If True, compute hash for each asset
        interrupt_check: Optional non-blocking callable that returns True if
            the operation should be interrupted (e.g. paused or cancelled)

    Returns:
        Tuple of (enriched_count, failed_reference_ids)
    """
    enriched = 0
    failed_ids: list[str] = []

    with create_session() as sess:
        for row in rows:
            if interrupt_check is not None and interrupt_check():
                break

            try:
                updated = enrich_asset(
                    sess,
                    file_path=row.file_path,
                    content_id=row.content_id,
                    record_id=row.record_id,
                    extract_metadata=extract_metadata,
                    compute_hash=compute_hash,
                )
                if updated:
                    enriched += 1
                else:
                    failed_ids.append(row.record_id)
            except Exception as e:
                logging.warning("Failed to enrich %s: %s", row.file_path, e)
                sess.rollback()
                failed_ids.append(row.record_id)

    return enriched, failed_ids
