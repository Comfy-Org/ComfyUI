"""Walks the asset roots and turns what it finds into catalog rows: collecting
paths, building specs, seeding new content and records, then enriching them
with metadata and hashes. Each spec is seeded inside its own savepoint, so one
file whose row conflicts cannot discard the work done for the files around it.
Enrichment counts as progress only when it produced what was asked of it — a
requested hash that could not be computed is no progress, which is what bounds
a pass over a file the server cannot read.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple, Protocol, TypedDict

import folder_paths
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.assets import mode
from app.assets.event_log import emit, error_type
from app.assets.database.queries import (
    create_content_reporting_insert,
    mark_content_missing,
    create_record,
)
from app.assets.database.models import Asset, AssetContent
from app.assets.helpers import sql_path_under_prefix, to_stored_hash
from app.assets.lifecycle import get_excluded_scan_roots
from app.assets.scanner_changes import (
    PreparedRecovery,
    clear_pending_verifications,
    detect_content_change,
    drain_pending_verifications,
    is_path_under_prefixes,
    live_contents_under_prefixes,
    pending_recovery_count,
    prepare_missing_content_recovery,
    queue_pending_recovery,
    queue_pending_verification,
    recover_missing_content,
    recover_missing_content_from_preparation,
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
from app.database.db import create_session, run_write_txn

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
    needs_hash: bool = False


class _PreparedEnrichment(NamedTuple):
    row: UnenrichedContent
    stat_result: os.stat_result
    system_metadata: dict[str, Any] | None
    mime_type: str | None
    stored_hash: str | None
    hash_requested: bool


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


def sync_references_with_filesystem(
    session,
    root: RootType,
    collect_existing_paths: bool = False,
    progress: _ScanProgress | None = None,
    pending_verification_ids: list[str] | None = None,
    diagnostics: list[OSError] | None = None,
) -> set[str] | None:
    return sync_prefixes_with_filesystem(
        session,
        get_scan_prefixes_for_root(root),
        collect_existing_paths=collect_existing_paths,
        progress=progress,
        pending_verification_ids=pending_verification_ids,
        diagnostics=diagnostics,
    )


class _ReferenceObservation(NamedTuple):

    content_id: str
    path: str
    observed_size_bytes: int | None
    observed_mtime_ns: int | None
    stat_result: os.stat_result | None


class _ReferenceDiagnostic(NamedTuple):

    path: str
    error: OSError


def _catalogued_references(
    session: Session, prefixes: list[str]
) -> list[tuple[str, str, int | None, int | None]]:
    return [
        (content.id, content.path, content.size_bytes, content.mtime_ns)
        for content in live_contents_under_prefixes(session, prefixes)
    ]


def observe_references_on_filesystem(
    prefixes: list[str],
    progress: _ScanProgress | None = None,
    diagnostics: list[_ReferenceDiagnostic] | None = None,
    session: Session | None = None,
) -> tuple[list[_ReferenceObservation], set[str]]:
    """Stat every catalogued reference without holding the writer lease.

    Returns only observations that require a write, so an unchanged root applies nothing
    and never acquires the write lock.
    """
    if session is not None:
        catalogued = _catalogued_references(session, prefixes)
    else:
        with create_session() as read_session:
            catalogued = _catalogued_references(read_session, prefixes)

    observations: list[_ReferenceObservation] = []
    survivors: set[str] = set()
    for content_id, path, size_bytes, mtime_ns in catalogued:
        try:
            stat_result = os.stat(path, follow_symlinks=True)
        except FileNotFoundError:
            observations.append(
                _ReferenceObservation(content_id, path, size_bytes, mtime_ns, None)
            )
        except PermissionError as e:
            if diagnostics is None:
                _log_scan_error("reference_stat", e)
                if progress is not None:
                    progress.permission_denied += 1
                logging.debug("Permission denied accessing %s", path)
            else:
                diagnostics.append(_ReferenceDiagnostic(path, e))
        except OSError as e:
            if diagnostics is None:
                _log_scan_error("reference_stat", e)
                logging.debug("OSError checking %s: %s", path, e)
            else:
                diagnostics.append(_ReferenceDiagnostic(path, e))
            observations.append(
                _ReferenceObservation(content_id, path, size_bytes, mtime_ns, None)
            )
        else:
            survivors.add(os.path.abspath(path))
            if stat_result.st_mtime_ns != mtime_ns:
                observations.append(
                    _ReferenceObservation(
                        content_id, path, size_bytes, mtime_ns, stat_result
                    )
                )

    return observations, survivors


def apply_reference_observations(
    session: Session,
    observations: list[_ReferenceObservation],
    pending_verification_ids: list[str] | None = None,
) -> None:
    """Apply pre-computed observations. Performs no filesystem I/O."""
    if not observations:
        return
    hashing_is_enabled = mode.hashing_enabled()
    for observation in observations:
        content = session.get(AssetContent, observation.content_id)
        if content is None:
            continue
        if (
            content.size_bytes != observation.observed_size_bytes
            or content.mtime_ns != observation.observed_mtime_ns
        ):
            continue
        if observation.stat_result is None:
            mark_content_missing(session, observation.content_id)
            continue
        detect_content_change(
            session,
            content,
            observation.stat_result,
            hashing_is_enabled=hashing_is_enabled,
            pending_verification_ids=pending_verification_ids,
        )


def sync_prefixes_with_filesystem(
    session: Session,
    prefixes: list[str],
    collect_existing_paths: bool = False,
    progress: _ScanProgress | None = None,
    pending_verification_ids: list[str] | None = None,
    diagnostics: list[_ReferenceDiagnostic] | None = None,
) -> set[str] | None:
    if not prefixes:
        return set() if collect_existing_paths else None

    observations, survivors = observe_references_on_filesystem(
        prefixes, progress=progress, diagnostics=diagnostics, session=session
    )
    apply_reference_observations(
        session, observations, pending_verification_ids=pending_verification_ids
    )

    return survivors if collect_existing_paths else None


def _publish_reference_diagnostics(
    diagnostics: list[_ReferenceDiagnostic], progress: _ScanProgress | None
) -> None:
    for path, error in diagnostics:
        _log_scan_error("reference_stat", error)
        if isinstance(error, PermissionError):
            if progress is not None:
                progress.permission_denied += 1
            logging.debug("Permission denied accessing %s", path)
        else:
            logging.debug("OSError checking %s: %s", path, error)
        if progress is None or progress.mark_emitted("stat_failed:reference_stat"):
            emit("scanner.stat_failed", site="reference_stat", error_type=error_type(error))


def _is_under_prefixes(path: str, prefixes: list[str]) -> bool:
    return is_path_under_prefixes(path, prefixes)


def sync_root_safely(
    root: RootType, progress: _ScanProgress | None = None
) -> set[str]:
    """Sync a single root's references with the filesystem.

    Returns survivors (existing paths) or empty set on failure.
    """
    try:
        diagnostics: list[_ReferenceDiagnostic] = []
        observations, survivors = observe_references_on_filesystem(
            get_scan_prefixes_for_root(root), diagnostics=diagnostics
        )

        pending_verification_ids: list[str] = []
        if observations:
            def _work(sess: Session) -> None:
                apply_reference_observations(
                    sess, observations, pending_verification_ids=pending_verification_ids
                )

            run_write_txn(_work)

        for content_id in pending_verification_ids:
            queue_pending_verification(content_id)
        _publish_reference_diagnostics(diagnostics, progress)
        return survivors
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
        diagnostics: list[_ReferenceDiagnostic] = []
        observations, _ = observe_references_on_filesystem(
            get_temp_prefixes(), diagnostics=diagnostics
        )

        pending_verification_ids: list[str] = []
        if observations:
            def _work(sess: Session) -> None:
                apply_reference_observations(
                    sess, observations, pending_verification_ids=pending_verification_ids
                )

            run_write_txn(_work)

        for content_id in pending_verification_ids:
            queue_pending_verification(content_id)
        _publish_reference_diagnostics(diagnostics, progress)
    except Exception as exc:
        logging.exception("temp reference sync failed: %s", exc)
        emit(
            "scanner.temp_sync_failed",
            root="temp",
            error_type=error_type(exc),
        )


def mark_missing_outside_prefixes_safely(prefixes: list[str]) -> int:
    """Mark references as missing when outside the given prefixes.

    This is a non-destructive soft-delete. Returns count marked or 0 on failure.
    """
    try:
        return run_write_txn(
            lambda session: mark_contents_missing_outside_prefixes(session, prefixes)
        )
    except Exception as exc:
        logging.exception("marking missing assets failed: %s", exc)
        emit(
            "scanner.mark_missing_failed",
            error_type=error_type(exc),
        )
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


def seed_asset_specs(
    session: Session,
    specs: list[SeedAssetSpec],
    prepared_recoveries: dict[str, PreparedRecovery | None] | None = None,
    pending_recovery_paths: list[str] | None = None,
) -> int:
    created = 0
    created_content_ids: list[str] = []
    try:
        for spec in specs:
            path = os.path.abspath(spec["abs_path"])
            try:
                with session.begin_nested():
                    try:
                        stat_result = os.stat(path, follow_symlinks=True)
                    except OSError:
                        logging.warning("Skipping vanished asset during scan: %s", path)
                        continue
                    if prepared_recoveries is None:
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
                    elif mode.hashing_enabled():
                        prepared = prepared_recoveries.get(path)
                        if prepared is None:
                            logging.warning("Skipping vanished asset during scan: %s", path)
                            continue
                        recovery = recover_missing_content_from_preparation(
                            session,
                            path,
                            stat_result,
                            prepared,
                            pending_recovery_paths if pending_recovery_paths is not None else [],
                        )
                    else:
                        recovery = "no_match"
                    if recovery != "no_match":
                        continue
                    content, inserted = create_content_reporting_insert(
                        session,
                        path=path,
                        hash=None,
                        size_bytes=stat_result.st_size,
                        mtime_ns=get_mtime_ns(stat_result),
                    )
                    if inserted:
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
            except IntegrityError:
                logging.warning("Skipping asset whose row conflicts during scan: %s", path)
                continue
    except Exception:
        session.rollback()
        for content_id in created_content_ids:
            _discard_unreferenced_content(session, content_id)
        raise
    return created


def insert_asset_specs(specs: list[SeedAssetSpec], _tag_pool: set[str]) -> int:
    if not specs:
        return 0
    prepared_recoveries: dict[str, PreparedRecovery | None] = {}
    if mode.hashing_enabled():
        for spec in specs:
            path = os.path.abspath(spec["abs_path"])
            try:
                prepared_recoveries[path] = prepare_missing_content_recovery(
                    path, os.stat(path, follow_symlinks=True)
                )
            except OSError:
                prepared_recoveries[path] = None

    def _work(sess: Session) -> tuple[int, list[str]]:
        pending_recovery_paths: list[str] = []
        created = seed_asset_specs(
            sess,
            specs,
            prepared_recoveries,
            pending_recovery_paths,
        )
        return created, pending_recovery_paths

    created, pending_recovery_paths = run_write_txn(_work)
    for path in pending_recovery_paths:
        queue_pending_recovery(path)
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
            sa.select(
                AssetContent.id,
                Asset.id,
                AssetContent.path,
                AssetContent.hash.is_(None).label("needs_hash"),
            )
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
        UnenrichedContent(content_id, record_id, file_path, needs_hash)
        for content_id, record_id, file_path, needs_hash in rows
    ]


def _prepare_enrichment(
    row: UnenrichedContent,
    extract_metadata: bool,
    compute_hash: bool,
    progress: _ScanProgress | None,
) -> _PreparedEnrichment | None:
    try:
        stat_result = os.stat(row.file_path, follow_symlinks=True)
    except FileNotFoundError:
        return None
    except OSError as exc:
        _log_scan_error("enrichment_stat", exc)
        if progress is not None:
            if isinstance(exc, PermissionError):
                progress.permission_denied += 1
            if progress.mark_emitted("stat_failed:enrich"):
                emit("scanner.stat_failed", site="enrich", error_type=error_type(exc))
        return None
    system_metadata: dict[str, Any] | None = None
    mime_type: str | None = None
    if extract_metadata:
        metadata = extract_file_metadata(
            row.file_path,
            stat_result=stat_result,
            relative_filename=compute_loader_path(row.file_path),
        )
        if metadata is not None:
            system_metadata = metadata.to_user_metadata()
            mime_type = metadata.content_type
            if mime_type is not None and mime_type.startswith("image/"):
                dimensions = extract_image_dimensions(row.file_path, mime_type=mime_type)
                if dimensions:
                    system_metadata.update(dimensions)
    hash_requested = compute_hash and row.needs_hash
    stored_hash: str | None = None
    if hash_requested:
        try:
            snapshot = snapshot_hash(row.file_path)
            if snapshot is None:
                if progress is None or progress.mark_emitted("hash_discarded_modified"):
                    emit("scanner.hash_discarded_modified")
                logging.warning(
                    "File modified during hashing (snapshot unstable), discarding hash: %s",
                    row.file_path,
                )
                return None
            digest, verified_stat = snapshot
            if (
                verified_stat.st_size != stat_result.st_size
                or get_mtime_ns(verified_stat) != get_mtime_ns(stat_result)
            ):
                logging.info(
                    "Content %s changed during enrichment preparation, discarding stale result",
                    row.content_id,
                )
                return None
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
                logging.warning("Failed to hash %s: %s", row.file_path, exc)
    return _PreparedEnrichment(
        row,
        stat_result,
        system_metadata,
        mime_type,
        stored_hash,
        hash_requested,
    )


def _apply_enrichment(session: Session, prepared: _PreparedEnrichment) -> bool:
    row = prepared.row
    content = session.get(AssetContent, row.content_id)
    record = session.get(Asset, row.record_id)
    if content is None or record is None:
        return False
    try:
        current_stat = os.stat(row.file_path, follow_symlinks=True)
    except OSError:
        return False
    if (
        content.mtime_ns != get_mtime_ns(prepared.stat_result)
        or current_stat.st_size != prepared.stat_result.st_size
        or get_mtime_ns(current_stat) != get_mtime_ns(prepared.stat_result)
    ):
        logging.info(
            "Content %s changed during enrichment, discarding stale result",
            row.content_id,
        )
        return False
    hash_applied = False
    if prepared.stored_hash is not None and content.hash is None:
        content.hash = prepared.stored_hash
        hash_applied = True

    if prepared.system_metadata is not None:
        record.system_metadata = {
            **(record.system_metadata or {}),
            **prepared.system_metadata,
        }
    if prepared.mime_type is not None:
        record.mime_type = prepared.mime_type

    if prepared.hash_requested and prepared.stored_hash is None:
        return False
    return hash_applied or prepared.system_metadata is not None or prepared.mime_type is not None


def enrich_assets_batch(
    rows: list[UnenrichedContent],
    extract_metadata: bool = True,
    compute_hash: bool = False,
    interrupt_check: Callable[[], bool] | None = None,
    progress: _ScanProgress | None = None,
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

    for row in rows:
        if interrupt_check is not None and interrupt_check():
            break
        try:
            prepared = _prepare_enrichment(
                row, extract_metadata, compute_hash, progress
            )
            if prepared is None:
                failed_ids.append(row.record_id)
                continue
            updated = run_write_txn(
                lambda session: _apply_enrichment(session, prepared)
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
            failed_ids.append(row.record_id)

    return enriched, failed_ids
