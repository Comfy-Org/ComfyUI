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
from typing import Any, Callable, Literal, NamedTuple, Protocol, TypedDict

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
from app.assets.helpers import sql_path_under_prefix, to_stored_hash
from app.assets.lifecycle import get_excluded_scan_roots
from app.assets.scanner_changes import (
    PreparedRecovery,
    clear_pending_verifications,
    detect_content_change,
    drain_pending_verifications,
    is_path_under_prefixes,
    live_contents_under_prefixes,
    prepare_missing_content_recovery,
    queue_pending_verification,
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
]


# Temp is deliberately absent: it is wiped before every scan, so walking it finds nothing.
RootType = Literal["models", "input", "output"]

# A scan's worth of rows in one transaction would hold the writer lock for the whole
# scan, so seeding commits in batches this size.
MAX_WRITE_BATCH = 25


class _ScanProgress(Protocol):
    hash_failed: int
    enrich_failed: int
    permission_denied: int

    def mark_emitted(self, key: str) -> bool: ...


class SeedAssetSpec(TypedDict):

    abs_path: str
    # Walk-time diagnostics only: seeding uses the stat taken by insert_asset_specs
    # before the write transaction.
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
    observed_size_bytes: int = 0
    observed_mtime_ns: int | None = None


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
    diagnostics: list[_ReferenceDiagnostic],
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
            diagnostics.append(_ReferenceDiagnostic(path, e))
        except OSError as e:
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
            content.is_missing
            or content.size_bytes != observation.observed_size_bytes
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
    root: RootType,
    progress: _ScanProgress | None = None,
    interrupt_check: Callable[[], bool] | None = None,
) -> set[str]:
    """Sync a single root's references with the filesystem.

    Returns survivors (existing paths) or empty set on failure.
    """
    try:
        diagnostics: list[_ReferenceDiagnostic] = []
        observations, survivors = observe_references_on_filesystem(
            get_scan_prefixes_for_root(root), diagnostics=diagnostics
        )

        committed_ids: list[str] = []
        try:
            for index in range(0, len(observations), MAX_WRITE_BATCH):
                if interrupt_check is not None and interrupt_check():
                    break
                chunk = observations[index : index + MAX_WRITE_BATCH]

                def _apply_chunk(
                    sess: Session, chunk: list[_ReferenceObservation] = chunk
                ) -> list[str]:
                    pending_verification_ids: list[str] = []
                    apply_reference_observations(
                        sess,
                        chunk,
                        pending_verification_ids=pending_verification_ids,
                    )
                    return pending_verification_ids

                committed_ids.extend(run_write_txn(_apply_chunk))
        finally:
            for content_id in committed_ids:
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
    interrupt_check: Callable[[], bool] | None = None,
) -> None:
    """Retire temp references whose file is gone; temp is never scanned, so nothing else stats them."""
    try:
        diagnostics: list[_ReferenceDiagnostic] = []
        observations, _ = observe_references_on_filesystem(
            get_temp_prefixes(), diagnostics=diagnostics
        )

        committed_ids: list[str] = []
        try:
            for index in range(0, len(observations), MAX_WRITE_BATCH):
                if interrupt_check is not None and interrupt_check():
                    break
                chunk = observations[index : index + MAX_WRITE_BATCH]

                def _apply_chunk(
                    sess: Session, chunk: list[_ReferenceObservation] = chunk
                ) -> list[str]:
                    pending_verification_ids: list[str] = []
                    apply_reference_observations(
                        sess,
                        chunk,
                        pending_verification_ids=pending_verification_ids,
                    )
                    return pending_verification_ids

                committed_ids.extend(run_write_txn(_apply_chunk))
        finally:
            for content_id in committed_ids:
                queue_pending_verification(content_id)

        _publish_reference_diagnostics(diagnostics, progress)
    except Exception as exc:
        logging.exception("temp reference sync failed: %s", exc)
        emit(
            "scanner.temp_sync_failed",
            root="temp",
            error_type=error_type(exc),
        )


def mark_missing_outside_prefixes_safely(
    prefixes: list[str], interrupt_check: Callable[[], bool] | None = None
) -> int | None:
    """Mark references as missing when outside the given prefixes.

    This is a non-destructive soft-delete. Returns the count committed before
    completion or interruption, or None when the prune failed having committed
    nothing — so a caller can tell a failure apart from a prune with no work.
    """
    marked_so_far = 0
    try:
        with create_session() as session:
            contents = session.scalars(
                sa.select(AssetContent).where(AssetContent.is_missing.is_(False))
            )
            content_ids = [
                content.id
                for content in contents
                if not _is_under_prefixes(content.path, prefixes)
            ]

        for index in range(0, len(content_ids), MAX_WRITE_BATCH):
            if interrupt_check is not None and interrupt_check():
                break
            chunk = content_ids[index : index + MAX_WRITE_BATCH]

            def _mark_chunk(
                session: Session, chunk: list[str] = chunk
            ) -> int:
                marked = 0
                for content_id in chunk:
                    content = session.get(AssetContent, content_id)
                    if content is None or content.is_missing:
                        continue
                    mark_content_missing(session, content_id)
                    marked += 1
                return marked

            marked_so_far += run_write_txn(_mark_chunk)
        return marked_so_far
    except Exception as exc:
        logging.exception("marking missing assets failed: %s", exc)
        emit(
            "scanner.mark_missing_failed",
            error_type=error_type(exc),
        )
        # A partial count is real committed work; only a prune that landed
        # nothing reports None, so the caller never claims count=0 for a failure.
        return marked_so_far if marked_so_far else None


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


def stat_seed_specs(specs: list[SeedAssetSpec]) -> dict[str, os.stat_result | None]:
    stats: dict[str, os.stat_result | None] = {}
    for spec in specs:
        path = os.path.abspath(spec["abs_path"])
        try:
            stats[path] = os.stat(path, follow_symlinks=True)
        except OSError:
            stats[path] = None
    return stats


def seed_asset_specs(
    session: Session,
    specs: list[SeedAssetSpec],
    stats: dict[str, os.stat_result | None],
    prepared_recoveries: dict[str, PreparedRecovery | None] | None = None,
    pending_recovery_paths: list[str] | None = None,
) -> tuple[int, Exception | None]:
    created = 0
    created_content_ids: list[str] = []
    first_error: Exception | None = None
    # Counted, not gated through _ScanProgress.mark_emitted like its neighbours, because this
    # function takes no progress object. Ungated, a restored archive of pre-epoch mtimes puts
    # one event per file into the closed-vocabulary stream.
    invalid_mtimes = 0
    try:
        for spec in specs:
            path = os.path.abspath(spec["abs_path"])
            try:
                with session.begin_nested():
                    stat_result = stats.get(path)
                    if stat_result is None:
                        logging.warning("Skipping vanished asset during scan: %s", path)
                        continue
                    if get_mtime_ns(stat_result) < 0:
                        logging.warning(
                            "Skipping asset with invalid mtime during scan: %s", path
                        )
                        invalid_mtimes += 1
                        continue
                    if prepared_recoveries is not None and mode.hashing_enabled():
                        prepared = prepared_recoveries.get(path)
                        if prepared is None:
                            logging.warning(
                                "Skipping asset whose recovery hash could not be prepared "
                                "during scan: %s",
                                path,
                            )
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
    except Exception:
        # Only a fault the per-spec handlers refuse to absorb reaches here, and it
        # takes the enclosing write transaction with it, so the content rows this
        # batch inserted are compensated before it propagates.
        session.rollback()
        for content_id in created_content_ids:
            _discard_unreferenced_content(session, content_id)
        raise
    if invalid_mtimes:
        emit("scanner.invalid_mtime", count=invalid_mtimes)
    return created, first_error


def insert_asset_specs(
    specs: list[SeedAssetSpec], _tag_pool: set[str]
) -> tuple[int, Exception | None]:
    if not specs:
        return 0, None
    stats = stat_seed_specs(specs)
    prepared_recoveries: dict[str, PreparedRecovery | None] = {}
    if mode.hashing_enabled():
        for path, stat_result in stats.items():
            if stat_result is None:
                prepared_recoveries[path] = None
                continue
            try:
                prepared_recoveries[path] = prepare_missing_content_recovery(path, stat_result)
            except OSError:
                prepared_recoveries[path] = None

    # run_write_txn owns the commit, so the batch fault has to leave the callback
    # out of band for the commit-failure path below to still be able to report it.
    batch_fault: Exception | None = None

    def _work(sess: Session) -> tuple[int, list[str]]:
        nonlocal batch_fault
        pending_recovery_paths: list[str] = []
        created, first_error = seed_asset_specs(
            sess,
            specs,
            stats,
            prepared_recoveries,
            pending_recovery_paths,
        )
        # A locked write is retried on a fresh session, so a fault recorded by an
        # attempt whose work was discarded must not outlive that attempt.
        batch_fault = first_error
        return created, pending_recovery_paths

    try:
        created, _ = run_write_txn(_work)
    except Exception:
        if batch_fault is None:
            raise
        logging.exception("Failed to commit successful specs from failed asset batch")
        return 0, batch_fault
    return created, batch_fault


def build_unenriched_candidates_statement(
    prefixes: list[str],
    compute_hashes: bool,
    last_seen_id: str | None,
    limit: int = 1000,
) -> sa.Select[tuple[str, str, str, bool, int, int | None]]:
    query = (
        sa.select(
            AssetContent.id,
            Asset.id,
            AssetContent.path,
            AssetContent.hash.is_(None).label("needs_hash"),
            AssetContent.size_bytes,
            AssetContent.mtime_ns,
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
        UnenrichedContent(
            content_id,
            record_id,
            file_path,
            needs_hash,
            observed_size_bytes,
            observed_mtime_ns,
        )
        for content_id, record_id, file_path, needs_hash, observed_size_bytes, observed_mtime_ns in rows
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
                logging.warning(
                    "Failed to hash %s: %s", row.file_path, exc, exc_info=True
                )
    return _PreparedEnrichment(
        row,
        stat_result,
        system_metadata,
        mime_type,
        stored_hash,
        hash_requested,
    )


def _apply_enrichments(
    session: Session, prepared: list[_PreparedEnrichment]
) -> list[str]:
    applied_ids: list[str] = []
    for item in prepared:
        row = item.row
        content = session.get(AssetContent, row.content_id)
        record = session.get(Asset, row.record_id)
        if (
            content is None
            or record is None
            or content.is_missing
            or content.size_bytes != row.observed_size_bytes
            or content.mtime_ns != row.observed_mtime_ns
        ):
            continue

        hash_applied = False
        if item.stored_hash is not None and content.hash is None:
            content.hash = item.stored_hash
            content.size_bytes = item.stat_result.st_size
            content.mtime_ns = get_mtime_ns(item.stat_result)
            hash_applied = True

        if item.system_metadata is not None:
            record.system_metadata = {
                **(record.system_metadata or {}),
                **item.system_metadata,
            }
        if item.mime_type is not None:
            record.mime_type = item.mime_type

        if item.hash_requested and item.stored_hash is None:
            continue
        if hash_applied or item.system_metadata is not None or item.mime_type is not None:
            applied_ids.append(row.record_id)
    return applied_ids


def enrich_assets_batch(
    rows: list[UnenrichedContent],
    extract_metadata: bool = True,
    compute_hash: bool = False,
    interrupt_check: Callable[[], bool] | None = None,
    progress: _ScanProgress | None = None,
) -> tuple[int, list[str], int]:
    """Prepares up to MAX_WRITE_BATCH rows on the scanner thread (stat, metadata,
    optional hash), then applies them in one write transaction with a per-row
    compare-and-set on the stored size/mtime; stale rows are skipped and left for
    the next sync.

    Returns:
        Tuple of (enriched_count, failed_reference_ids, consumed_count) — the
        consumed count is what the caller advances its id cursor by, so a batch
        that ends early never strands the rows it did not reach.
    """
    enriched = 0
    failed_ids: list[str] = []
    consumed = 0

    for index in range(0, len(rows), MAX_WRITE_BATCH):
        prepared_list: list[_PreparedEnrichment] = []
        interrupted = False
        for row in rows[index : index + MAX_WRITE_BATCH]:
            if interrupt_check is not None and interrupt_check():
                interrupted = True
                break
            consumed += 1

            try:
                prepared = _prepare_enrichment(
                    row, extract_metadata, compute_hash, progress
                )
            except Exception as exc:
                if progress is not None:
                    progress.enrich_failed += 1
                if progress is None or progress.mark_emitted("enrich_failed"):
                    emit("scanner.enrich_failed", error_type=error_type(exc))
                logging.warning("Failed to enrich %s: %s", row.file_path, exc)
                failed_ids.append(row.record_id)
                continue
            if prepared is None:
                failed_ids.append(row.record_id)
                continue
            prepared_list.append(prepared)

        if prepared_list:
            try:
                applied = run_write_txn(
                    lambda session, prepared_list=prepared_list: _apply_enrichments(
                        session, prepared_list
                    )
                )
            except Exception as exc:
                if progress is not None:
                    progress.enrich_failed += len(prepared_list)
                if progress is None or progress.mark_emitted("enrich_failed"):
                    emit("scanner.enrich_failed", error_type=error_type(exc))
                logging.warning(
                    "Failed to enrich %d assets: %s", len(prepared_list), exc
                )
                failed_ids.extend(item.row.record_id for item in prepared_list)
            else:
                enriched += len(applied)
                applied_ids = set(applied)
                failed_ids.extend(
                    item.row.record_id
                    for item in prepared_list
                    if item.row.record_id not in applied_ids
                )

        if interrupted:
            break

    return enriched, failed_ids, consumed
