"""Walks the asset roots and turns what it finds into catalog rows: collecting
paths, building specs, seeding new content and records, then enriching them
with metadata and hashes. Each spec is seeded inside its own savepoint, so one
file whose row conflicts cannot discard the work done for the files around it.
Enrichment candidates use ordered ID pagination, so each row is attempted at
most once per pass while failed rows remain eligible for the next pass. A pause
can end a batch early, and the cursor holds at the last row the batch attempted,
so the rows it never reached are selected again when the scan resumes.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, NamedTuple, Protocol, TypedDict

import folder_paths
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.assets import mode
from app.assets.event_log import emit, error_type
from app.assets.database.queries import (
    create_content_reporting_insert,
    is_live_path_conflict,
    mark_content_missing,
    create_record,
)
from app.assets.database.models import Asset, AssetContent
from app.assets.helpers import path_prefix_matcher, sql_path_under_prefix, to_stored_hash
from app.assets.lifecycle import get_excluded_scan_roots
from app.assets.scanner_changes import (
    clear_pending_verifications,
    detect_content_change,
    drain_pending_verifications,
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
from app.assets.services.snapshot_hash import snapshot_hash
from app.database.db import create_session, create_write_session

__all__ = [
    "clear_pending_verifications",
    "drain_pending_verifications",
    "pending_recovery_count",
]


# Temp is deliberately absent: it is wiped before every scan, so walking it finds nothing.
RootType = Literal["models", "input", "output"]


class _ScanProgress(Protocol):
    hash_failed: int
    enrich_failed: int
    permission_denied: int

    def mark_emitted(self, key: str) -> bool: ...


class SeedAssetSpec(TypedDict):

    abs_path: str
    # Walk-time diagnostics only: seeding persists the seed-time restat instead.
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


class _ReferenceObservation(NamedTuple):
    """One live row's file, stat'ed outside the write transaction.

    ``size_bytes`` and ``mtime_ns`` are the row's values when observed, not the file's:
    the write skips a row that no longer matches them. ``stat_result`` is None when the
    file is gone.
    """

    content_id: str
    size_bytes: int | None
    mtime_ns: int | None
    stat_result: os.stat_result | None


def _log_scan_error(phase: str, error: OSError) -> None:
    error_type = (
        "permission_denied" if isinstance(error, PermissionError) else "os_error"
    )
    logging.warning("Asset scan error: phase=%s error_type=%s", phase, error_type)


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


def observe_references_on_filesystem(
    session: Session, prefixes: list[str], progress: _ScanProgress | None = None
) -> tuple[list[_ReferenceObservation], set[str]]:
    """Stat every live row under ``prefixes`` without writing, so the caller can
    apply the result in a short write transaction. Also returns the paths whose
    file still exists."""
    contents = [
        (content.id, content.path, content.size_bytes, content.mtime_ns)
        for content in live_contents_under_prefixes(session, prefixes)
    ]
    observations: list[_ReferenceObservation] = []
    survivors: set[str] = set()
    for content_id, path, size_bytes, mtime_ns in contents:
        try:
            stat_result = os.stat(path, follow_symlinks=True)
        except FileNotFoundError:
            observations.append(_ReferenceObservation(content_id, size_bytes, mtime_ns, None))
        except PermissionError as e:
            _log_scan_error("reference_stat", e)
            if progress is not None:
                progress.permission_denied += 1
            logging.debug("Permission denied accessing %s", path)
        except OSError as e:
            _log_scan_error("reference_stat", e)
            logging.debug("OSError checking %s: %s", path, e)
            observations.append(_ReferenceObservation(content_id, size_bytes, mtime_ns, None))
        else:
            survivors.add(os.path.abspath(path))
            if stat_result.st_mtime_ns != mtime_ns:
                observations.append(
                    _ReferenceObservation(content_id, size_bytes, mtime_ns, stat_result)
                )
    return observations, survivors


def apply_reference_observations(
    session: Session, observations: list[_ReferenceObservation]
) -> None:
    for observation in observations:
        content = session.get(AssetContent, observation.content_id)
        # Skip a row another writer changed since it was observed; the next scan sees it afresh.
        if (
            content is None
            or content.is_missing
            or content.size_bytes != observation.size_bytes
            or content.mtime_ns != observation.mtime_ns
        ):
            continue
        if observation.stat_result is None:
            mark_content_missing(session, content.id)
            continue
        detect_content_change(
            session,
            content,
            observation.stat_result,
            hashing_is_enabled=mode.hashing_enabled(),
        )


def _sync_prefixes_in_write_txn(
    prefixes: list[str], progress: _ScanProgress | None
) -> set[str]:
    with create_session() as session:
        observations, survivors = observe_references_on_filesystem(
            session, prefixes, progress
        )
    if observations:
        with create_write_session() as session:
            apply_reference_observations(session, observations)
            session.commit()
    return survivors


def sync_root_safely(
    root: RootType, progress: _ScanProgress | None = None
) -> set[str]:
    """Sync a single root's references with the filesystem.

    Returns survivors (existing paths) or empty set on failure.
    """
    try:
        return _sync_prefixes_in_write_txn(get_scan_prefixes_for_root(root), progress)
    except Exception as exc:
        logging.exception("fast DB scan failed for %s: %s", root, exc)
        emit(
            "scanner.fast_scan_failed",
            root=root,
            error_type=error_type(exc),
        )
        return set()


def sync_temp_references_safely(
    progress: _ScanProgress | None = None,
) -> None:
    """Retire temp references whose file is gone; temp is never scanned, so nothing else stats them."""
    try:
        _sync_prefixes_in_write_txn(get_temp_prefixes(), progress)
    except Exception as exc:
        logging.exception("temp reference sync failed: %s", exc)
        emit(
            "scanner.temp_sync_failed",
            root="temp",
            error_type=error_type(exc),
        )


def mark_missing_outside_prefixes_safely(prefixes: list[str]) -> int | None:
    """Mark references as missing when outside the given prefixes.

    This is a non-destructive soft-delete. Returns the count marked, or None when
    the operation fails.
    """
    try:
        with create_session() as sess:
            count = mark_contents_missing_outside_prefixes(sess, prefixes)
            sess.commit()
            return count
    except Exception as exc:
        logging.exception("marking missing assets failed: %s", exc)
        emit(
            "scanner.mark_missing_failed",
            error_type=error_type(exc),
        )
        return None


def mark_contents_missing_outside_prefixes(
    session: Session, prefixes: list[str]
) -> int:
    contents = session.scalars(
        sa.select(AssetContent).where(AssetContent.is_missing.is_(False))
    )
    is_owned = path_prefix_matcher(prefixes)
    missing = [content for content in contents if not is_owned(content.path)]
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
    progress: _ScanProgress | None = None,
) -> tuple[list[SeedAssetSpec], set[str], int]:
    """Build asset specs from paths, returning (specs, tag_pool, skipped_count).

    Args:
        paths: List of file paths to process
        existing_paths: Set of paths that already exist in the database
        enable_metadata_extraction: If True, extract tier 1 & 2 metadata
        progress: Optional per-scan state for emit-once bookkeeping
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
        except FileNotFoundError:
            continue
        except OSError as e:
            _log_scan_error("discovery_stat", e)
            if progress is not None:
                if isinstance(e, PermissionError):
                    progress.permission_denied += 1
                if progress.mark_emitted("stat_failed:discovery"):
                    emit("scanner.stat_failed", site="discovery", error_type=error_type(e))
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


class _SpecObservation(NamedTuple):
    """A spec's file as seen before the write transaction opens.

    ``snapshot`` is None when hashing is off, or when the file changed while being hashed.
    """

    stat_result: os.stat_result
    snapshot: tuple[str, os.stat_result] | None


def observe_asset_specs(specs: list[SeedAssetSpec]) -> dict[str, _SpecObservation | None]:
    """Stat (and, in hashing mode, hash) each spec before the write transaction opens.

    ``None`` marks a path that vanished or could not be read.
    """
    hashing_is_enabled = mode.hashing_enabled()
    observed: dict[str, _SpecObservation | None] = {}
    for spec in specs:
        path = os.path.abspath(spec["abs_path"])
        try:
            stat_result = os.stat(path, follow_symlinks=True)
            snapshot = snapshot_hash(path) if hashing_is_enabled else None
        except FileNotFoundError:
            logging.warning("Skipping vanished asset during scan: %s", path)
            observed[path] = None
            continue
        except OSError as e:
            _log_scan_error("seed_observation", e)
            observed[path] = None
            continue
        if snapshot is not None:
            stat_result = snapshot[1]  # the stat the hash was verified against
        observed[path] = _SpecObservation(stat_result, snapshot)
    return observed


def seed_asset_specs(
    session: Session,
    specs: list[SeedAssetSpec],
    observed: dict[str, _SpecObservation | None] | None = None,
) -> tuple[int, Exception | None]:
    if observed is None:
        observed = observe_asset_specs(specs)
    created = 0
    first_error: Exception | None = None
    # Counted, not gated through _ScanProgress.mark_emitted like its neighbours, because this
    # function takes no progress object. Ungated, a restored archive of pre-epoch mtimes puts
    # one event per file into the closed-vocabulary stream.
    invalid_mtimes = 0
    for spec in specs:
        path = os.path.abspath(spec["abs_path"])
        try:
            with session.begin_nested():
                observation = observed[path]
                if observation is None:  # observe_asset_specs already logged why
                    continue
                stat_result = observation.stat_result
                if get_mtime_ns(stat_result) < 0:
                    logging.warning(
                        "Skipping asset with invalid mtime during scan: %s", path
                    )
                    invalid_mtimes += 1
                    continue
                recovery = recover_missing_content(
                    session,
                    path,
                    observation.snapshot,
                    hashing_is_enabled=mode.hashing_enabled(),
                )
                if recovery != "no_match":
                    continue
                content, _inserted = create_content_reporting_insert(
                    session,
                    path=path,
                    hash=None,
                    size_bytes=stat_result.st_size,
                    mtime_ns=get_mtime_ns(stat_result),
                )
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
        except IntegrityError as error:
            if is_live_path_conflict(error):
                logging.warning(
                    "Skipping asset whose row conflicts during scan: %s", path
                )
                continue
            if first_error is None:
                first_error = error
        except MemoryError:
            # Deferring this one would keep allocating for every remaining spec
            # while the process is already out of memory.
            raise
        except Exception as error:
            if first_error is None:
                first_error = error
    if invalid_mtimes:
        emit("scanner.invalid_mtime", count=invalid_mtimes)
    return created, first_error


def insert_asset_specs(
    specs: list[SeedAssetSpec], _tag_pool: set[str]
) -> tuple[int, Exception | None]:
    if not specs:
        return 0, None
    observed = observe_asset_specs(specs)
    with create_write_session() as sess:
        created, first_error = seed_asset_specs(sess, specs, observed)
        try:
            sess.commit()
        except Exception:
            if first_error is None:
                raise
            logging.exception("Failed to commit successful specs from failed asset batch")
            try:
                sess.rollback()
            except Exception:
                logging.exception("Failed to roll back asset batch after commit failure")
            return 0, first_error
        return created, first_error


def build_unenriched_candidates_statement(
    prefixes: list[str],
    compute_hashes: bool,
    last_seen_id: str | None,
    limit: int = 1000,
) -> sa.Select[tuple[str, str, str]]:
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
    if last_seen_id is not None:
        query = query.where(Asset.id > last_seen_id)
    return (
        query.where(
            sa.or_(*(sql_path_under_prefix(AssetContent.path, p) for p in prefixes))
        )
        .order_by(Asset.id.asc())
        .limit(limit)
    )


def get_unenriched_assets_for_roots(
    roots: tuple[RootType, ...],
    compute_hashes: bool,
    limit: int = 1000,
    last_seen_id: str | None = None,
) -> list[UnenrichedContent]:
    prefixes: list[str] = []
    for root in roots:
        prefixes.extend(get_scan_prefixes_for_root(root))

    if not prefixes:
        return []

    query = build_unenriched_candidates_statement(
        prefixes,
        compute_hashes,
        last_seen_id,
        limit,
    )
    with create_session() as sess:
        rows = sess.execute(query).all()

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
    progress: _ScanProgress | None = None,
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
    except FileNotFoundError:
        return False
    except OSError as e:
        _log_scan_error("enrichment_stat", e)
        if progress is not None:
            if isinstance(e, PermissionError):
                progress.permission_denied += 1
            if progress.mark_emitted("stat_failed:enrich"):
                emit("scanner.stat_failed", site="enrich", error_type=error_type(e))
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
    hash_requested = compute_hash and content is not None and content.hash is None
    if hash_requested:
        try:
            snapshot = snapshot_hash(file_path)
            if snapshot is None:
                if progress is None or progress.mark_emitted("hash_discarded_modified"):
                    emit("scanner.hash_discarded_modified")
                logging.warning(
                    "File modified during hashing (snapshot unstable), discarding hash: %s",
                    file_path,
                )
                return False
            digest, verified_stat = snapshot
            stored_hash = to_stored_hash(digest)
        except Exception as exc:
            emit_failure = progress is None
            if progress is not None:
                progress.hash_failed += 1
                emit_failure = progress.mark_emitted("hash_failed")
            if emit_failure:
                emit("scanner.hash_failed", error_type=error_type(exc))
            if isinstance(exc, OSError):
                _log_scan_error("hashing", exc)
            else:
                logging.warning("Failed to hash %s: %s", file_path, exc, exc_info=True)

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

    if hash_requested and stored_hash is None:
        return False
    return stored_hash is not None or metadata is not None or mime_type is not None


def enrich_assets_batch(
    rows: list[UnenrichedContent],
    extract_metadata: bool = True,
    compute_hash: bool = False,
    interrupt_check: Callable[[], bool] | None = None,
    progress: _ScanProgress | None = None,
) -> tuple[int, list[str], int]:
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
        Tuple of (enriched_count, failed_reference_ids, consumed_count)
    """
    enriched = 0
    failed_ids: list[str] = []
    consumed = 0

    with create_session() as sess:
        for row in rows:
            if interrupt_check is not None and interrupt_check():
                break
            consumed += 1

            try:
                updated = enrich_asset(
                    sess,
                    file_path=row.file_path,
                    content_id=row.content_id,
                    record_id=row.record_id,
                    extract_metadata=extract_metadata,
                    compute_hash=compute_hash,
                    progress=progress,
                )
                if updated:
                    enriched += 1
                else:
                    failed_ids.append(row.record_id)
            except Exception as exc:
                if progress is not None:
                    progress.enrich_failed += 1
                if progress is None or progress.mark_emitted("enrich_failed"):
                    emit("scanner.enrich_failed", error_type=error_type(exc))
                logging.warning("Failed to enrich %s: %s", row.file_path, exc)
                sess.rollback()
                failed_ids.append(row.record_id)

    return enriched, failed_ids, consumed
