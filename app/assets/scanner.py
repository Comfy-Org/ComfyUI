"""Walks the asset roots and turns what it finds into catalog rows: collecting
paths, building specs, seeding new content and records, then enriching them
with metadata and hashes. Each spec is seeded inside its own savepoint, so one
file whose row conflicts cannot discard the work done for the files around it.
Enrichment candidates use ordered ID pagination, so each row is attempted at
most once per pass while failed rows remain eligible for the next pass. A pause
can end a batch early, and the cursor holds at the last row the batch attempted,
so the rows it never reached are selected again when the scan resumes.
"""

import enum
import logging
import os
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Callable, Literal, NamedTuple, Protocol, TypedDict

import folder_paths
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.assets import mode
from app.assets.event_log import emit, error_kind, error_type
from app.assets.database.queries import (
    create_content_reporting_insert,
    is_live_path_conflict,
    mark_contents_missing,
    create_record,
)
from app.assets.database.models import Asset, AssetContent
from app.assets.helpers import (
    PREFIX_BATCH_SIZE,
    path_prefix_matcher,
    sql_path_under_prefix,
    sql_path_under_prefix_batches,
    stored_path_under_prefixes,
    to_stored_hash,
)
from app.assets.lifecycle import get_excluded_scan_roots
from app.assets.scanner_changes import (
    clear_pending_verifications,
    detect_content_change,
    drain_pending_verifications,
    live_contents_under_prefixes,
    missing_content_ids_by_path,
    pending_recovery_count,
    recover_missing_content,
    recover_missing_content_by_stat,
)
from app.assets.scanner_admission import (
    PARTIAL_DOWNLOAD_EXTENSIONS as PARTIAL_DOWNLOAD_EXTENSIONS,
    _WATCH_LIST as _WATCH_LIST,
    _WatchEntry as _WatchEntry,
    _should_skip_extension,
    _two_stat_admit,
    tick_watch_list as tick_watch_list,
)
from app.assets.services.file_utils import (
    RESCAN_YIELD_RUN,
    DirListings,
    ListingWalk,
    get_mtime_ns,
    is_visible,
    list_files_recursively,
    walk_listings,
)
from app.assets.services.gil import yield_gil
from app.assets.services.media_metadata import extract_media_metadata
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
    missing_marked: int
    recovered: int
    dirs_listed: int
    files_statted: int

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


# Before each write batch: blocks while the scan is paused, returns True to stop.
ShouldStop = Callable[[], bool]


def _never_stop() -> bool:
    return False


# The marking steps write in batches of WRITE_BATCH_ROWS, each its own transaction, so
# the write lock is never held for a whole pass and a foreground write (an output being
# registered) waits at most one batch. After each commit the thread sleeps about as
# long as it held the lock: a writer in SQLite's busy handler polls with growing sleeps
# (up to 100 ms), so taking the lock straight back would keep winning it.
WRITE_BATCH_ROWS = 256
WRITE_YIELD_MIN_SECONDS = 0.02
WRITE_YIELD_MAX_SECONDS = 0.1


def _write_in_batches(
    items: list,
    write_batch: Callable[[Session, list], int],
    should_stop: ShouldStop,
    committed: list[int],
) -> None:
    """Apply ``write_batch`` to ``items`` a batch per transaction, appending the count
    each batch reports to ``committed`` once it commits. A failure leaves the batches
    before it committed, and ``committed`` says how much they wrote."""
    for start in range(0, len(items), WRITE_BATCH_ROWS):
        if should_stop():
            break
        opened = time.perf_counter()
        with create_write_session() as session:
            count = write_batch(session, items[start : start + WRITE_BATCH_ROWS])
            session.commit()
        committed.append(count)
        held = time.perf_counter() - opened
        if start + WRITE_BATCH_ROWS < len(items):
            time.sleep(min(max(held, WRITE_YIELD_MIN_SECONDS), WRITE_YIELD_MAX_SECONDS))


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
    # Owned but never scanned: previews are only ever registered by Core or an upload.
    return prefixes + get_temp_prefixes() + [os.path.abspath(folder_paths.get_previews_directory())]


def get_temp_prefixes() -> list[str]:
    temp_dir = os.path.abspath(folder_paths.get_temp_directory())
    if temp_dir in get_excluded_scan_roots():
        return []
    return [temp_dir]


def collect_models_files() -> list[str]:
    out: list[str] = []
    for folder_name, bases, _exts in get_comfy_models_folders():
        try:
            rel_files = folder_paths.get_filename_list(folder_name) or []
        except OSError:
            # The cached listing raises on every call once a folder it recorded has gone
            # away; a fresh listing skips that folder and lists the rest.
            try:
                rel_files = folder_paths.get_filename_list_(folder_name)[0]
            except OSError as e:
                logging.warning("Asset scan: skipping model category %s, it can't be listed: %s", folder_name, e)
                continue
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
        if progress is not None:
            progress.files_statted += 1
        try:
            stat_result = os.stat(path, follow_symlinks=True)
        except (FileNotFoundError, NotADirectoryError):
            observations.append(_ReferenceObservation(content_id, size_bytes, mtime_ns, None))
        except PermissionError as e:
            _log_scan_error("reference_stat", e)
            if progress is not None:
                progress.permission_denied += 1
            logging.debug("Permission denied accessing %s", path)
        except OSError as e:
            # An I/O error (a flaky network share, a stale handle) says nothing about
            # whether the file still exists, so the row stays live, as _is_gone leaves it.
            _log_scan_error("reference_stat", e)
            logging.debug("OSError checking %s: %s", path, e)
        else:
            survivors.add(os.path.abspath(path))
            if stat_result.st_mtime_ns != mtime_ns:
                observations.append(
                    _ReferenceObservation(content_id, size_bytes, mtime_ns, stat_result)
                )
    return observations, survivors


def apply_reference_observations(
    session: Session, observations: list[_ReferenceObservation]
) -> int:
    """Apply the observations; returns how many rows were marked missing."""
    # One query loads the rows, so the session.get calls below never go to the database.
    list(session.scalars(sa.select(AssetContent).where(AssetContent.id.in_([o.content_id for o in observations]))))
    gone: list[str] = []
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
            gone.append(content.id)
            continue
        detect_content_change(
            session,
            content,
            observation.stat_result,
            hashing_is_enabled=mode.hashing_enabled(),
        )
    return len(mark_contents_missing(session, gone))


def _sync_prefixes(
    prefixes: list[str],
    progress: _ScanProgress | None,
    should_stop: ShouldStop,
    marked: list[int],
) -> set[str]:
    """Returns the surviving paths; ``marked`` gets each committed batch's count of rows
    marked missing, which a caller still has if a later batch raises."""
    with create_session() as session:
        observations, survivors = observe_references_on_filesystem(
            session, prefixes, progress
        )
    _write_in_batches(observations, apply_reference_observations, should_stop, marked)
    return survivors


def sync_root_safely(
    root: RootType,
    progress: _ScanProgress | None = None,
    should_stop: ShouldStop = _never_stop,
) -> set[str]:
    """Sync a single root's references with the filesystem.

    Returns survivors (existing paths) or empty set on failure.
    """
    marked: list[int] = []
    try:
        survivors = _sync_prefixes(
            get_scan_prefixes_for_root(root), progress, should_stop, marked
        )
    except Exception as exc:
        logging.exception("fast DB scan failed for %s: %s", root, exc)
        emit(
            "scanner.fast_scan_failed",
            root=root,
            error_type=error_type(exc),
            error_kind=error_kind(exc),
        )
        survivors = set()
    if progress is not None:
        progress.missing_marked += sum(marked)
    return survivors


def sync_temp_references_safely(
    progress: _ScanProgress | None = None,
    should_stop: ShouldStop = _never_stop,
) -> None:
    """Retire temp references whose file is gone; temp is never scanned, so nothing else stats them."""
    try:
        _sync_prefixes(get_temp_prefixes(), progress, should_stop, [])
    except Exception as exc:
        logging.exception("temp reference sync failed: %s", exc)
        emit(
            "scanner.temp_sync_failed",
            root="temp",
            error_type=error_type(exc),
            error_kind=error_kind(exc),
        )


def mark_missing_outside_prefixes_safely(
    prefixes: list[str], should_stop: ShouldStop = _never_stop
) -> int | None:
    """Mark references as missing when outside the given prefixes.

    This is a non-destructive soft-delete. Returns the count marked, or None when
    the operation fails; batches committed before a failure stay committed.
    """
    marked: list[int] = []
    try:
        with create_session() as sess:
            content_ids = content_ids_outside_prefixes(sess, prefixes)
        _write_in_batches(
            content_ids,
            lambda session, batch: len(mark_contents_missing(session, batch)),
            should_stop,
            marked,
        )
        return sum(marked)
    except Exception as exc:
        logging.exception("marking missing assets failed after marking %d: %s", sum(marked), exc)
        emit(
            "scanner.mark_missing_failed",
            error_type=error_type(exc),
            error_kind=error_kind(exc),
        )
        return None


def content_ids_outside_prefixes(session: Session, prefixes: list[str]) -> list[str]:
    """The live rows outside every prefix. A read: the marking re-checks each is still
    live inside its own write transaction."""
    is_owned = path_prefix_matcher(prefixes)
    rows = session.execute(
        sa.select(AssetContent.id, AssetContent.path)
        .where(AssetContent.is_missing == sa.false())
        .execution_options(yield_per=500)
    )
    return [content_id for content_id, path in rows if not is_owned(path)]


def collect_paths_for_roots(
    roots: tuple[RootType, ...],
    progress: _ScanProgress | None = None,
    should_stop: ShouldStop = _never_stop,
) -> list[str]:
    """Collect all file paths for the given roots.

    ``progress.dirs_listed`` counts the input and output walks only. Models are
    listed through folder_paths.get_filename_list, which walks the model folders
    on a cache miss and re-checks their mtimes on a hit; none of that is counted.

    ``should_stop`` is checked before each input or output directory; once it returns
    True the list is partial, so callers check it again before using the result.
    """
    paths: list[str] = []
    if "models" in roots:
        paths.extend(collect_models_files())
    if "input" in roots:
        paths.extend(list_files_recursively(folder_paths.get_input_directory(), progress, should_stop))
    if "output" in roots:
        paths.extend(list_files_recursively(folder_paths.get_output_directory(), progress, should_stop))
    return paths


def rescans_output_by_listing(roots: tuple[RootType, ...]) -> bool:
    """Whether this scan checks the catalog against directory listings rather than by
    stat'ing every live row. Only output-only scans do: the rescan queued after each prompt.

    The listing diff catches every add and delete, but nothing stats an already-cataloged
    file, so an in-place overwrite (same path; new content, size or mtime) goes undetected
    until the next scan that is not output-only, such as the startup scan. Core save nodes
    never overwrite, and reported outputs are registered at save time, so this only
    affects files written by something else.
    """
    return tuple(roots) == ("output",)


def live_references_safely(root: RootType) -> dict[str, list[_ReferenceObservation]]:
    """The live rows under ``root`` by path, read without touching the filesystem.

    Each is observed as gone (``stat_result=None``): what apply_reference_observations
    needs to retire it, should its path turn out not to be listed. Empty on failure, as
    sync_root_safely is.
    """
    prefixes = get_scan_prefixes_for_root(root)
    live: dict[str, list[_ReferenceObservation]] = {}
    if not prefixes:
        return live
    seen: set[str] = set()
    try:
        with create_session() as session:
            for under_prefixes in sql_path_under_prefix_batches(AssetContent.path, prefixes):
                stmt = sa.select(
                    AssetContent.id, AssetContent.path, AssetContent.size_bytes, AssetContent.mtime_ns
                ).where(AssetContent.is_missing.is_(False), under_prefixes)
                for content_id, path, size_bytes, mtime_ns in session.execute(stmt):
                    yield_gil(run=RESCAN_YIELD_RUN)
                    if content_id in seen:
                        continue
                    seen.add(content_id)
                    live.setdefault(os.path.abspath(path), []).append(
                        _ReferenceObservation(content_id, size_bytes, mtime_ns, None)
                    )
    except Exception as exc:
        logging.exception("fast DB scan failed for %s: %s", root, exc)
        emit(
            "scanner.fast_scan_failed",
            root=root,
            error_type=error_type(exc),
            error_kind=error_kind(exc),
        )
        return {}
    return live


def unlisted_references(
    live: dict[str, list[_ReferenceObservation]],
    listings: DirListings,
    progress: _ScanProgress | None = None,
) -> tuple[list[_ReferenceObservation], int]:
    """Split the live rows into (vanished, skipped count).

    A row its parent's listing names is present, with no further check. Every other row
    is stat'ed, and has vanished only if the stat says the file is gone: that covers a
    name the listing lacks, a removed directory, and the rows no listing can speak for
    (a hidden path, a directory that failed to list, a symlink alias the walk did not
    take). The skipped count is the rows that were stat'ed and kept.
    """
    vanished: list[_ReferenceObservation] = []
    skipped = 0
    names_by_dir: dict[str, set[str]] = {}
    for path, observations in live.items():
        yield_gil(run=RESCAN_YIELD_RUN)
        verdict = _listing_verdict(path, listings, names_by_dir)
        if verdict is ListingVerdict.LISTED:
            continue
        # Stat before retiring. A listing compares names exactly, but a case-insensitive
        # (NTFS, APFS) or Unicode-normalizing (HFS+) filesystem resolves a stored path
        # spelled differently from its entry. Rows that reach here are normally few.
        if progress is not None:
            progress.files_statted += 1
        if _is_gone(path):
            vanished.extend(observations)
        else:
            skipped += len(observations)
    return vanished, skipped


def _is_gone(path: str) -> bool:
    """True only when stat says the path does not exist. Any other error (permissions,
    I/O) leaves it undecided, and the row stays live, as the per-row stat left it."""
    try:
        os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return True
    except OSError:
        return False
    return False


class ListingVerdict(enum.Enum):
    """What this rescan's directory listings say about a cataloged path."""

    LISTED = "listed"  # its parent's listing has the name
    ABSENT = "absent"  # the nearest listed directory above it lacks the next component
    UNKNOWN = "unknown"  # no listing can say: hidden, no listed ancestor, or not walked


def _listing_verdict(
    path: str, listings: DirListings, names_by_dir: dict[str, set[str]]
) -> ListingVerdict:
    """Classify ``path`` against the listings; see ListingVerdict. An ancestor that
    still has the directory this walk did not list is UNKNOWN, not LISTED."""
    child, parent = path, os.path.dirname(path)
    while parent not in listings:
        if not is_visible(os.path.basename(child)):
            return ListingVerdict.UNKNOWN
        child, parent = parent, os.path.dirname(parent)
        if parent == child:  # reached the filesystem root without meeting a listing
            return ListingVerdict.UNKNOWN
    name = os.path.basename(child)
    if not is_visible(name):
        return ListingVerdict.UNKNOWN
    names = names_by_dir.get(parent)
    if names is None:
        files, subdirs = listings[parent]
        names = names_by_dir[parent] = {*files, *subdirs}
    if name not in names:
        return ListingVerdict.ABSENT
    return ListingVerdict.LISTED if child == path else ListingVerdict.UNKNOWN


def mark_unlisted_references_missing_safely(
    root: RootType,
    observations: list[_ReferenceObservation],
    progress: _ScanProgress | None = None,
    should_stop: ShouldStop = _never_stop,
) -> None:
    """Retire rows whose file the listing lacks, through the same guarded write
    sync_root applies to a row whose file has vanished."""
    marked: list[int] = []
    try:
        _write_in_batches(observations, apply_reference_observations, should_stop, marked)
    except Exception as exc:
        logging.exception("fast DB scan failed for %s: %s", root, exc)
        emit(
            "scanner.fast_scan_failed",
            root=root,
            error_type=error_type(exc),
            error_kind=error_kind(exc),
        )
    if progress is not None:
        progress.missing_marked += sum(marked)


def list_output_for_rescan() -> ListingWalk:
    """Walk the output root, listing every directory."""
    return walk_listings(folder_paths.get_output_directory())


def build_asset_specs(
    paths: list[str],
    existing_paths: set[str],
    enable_metadata_extraction: bool = True,
    progress: _ScanProgress | None = None,
    should_stop: ShouldStop = _never_stop,
) -> tuple[list[SeedAssetSpec], set[str], int]:
    """Build asset specs from paths, returning (specs, tag_pool, skipped_count).

    Args:
        paths: List of file paths to process
        existing_paths: Set of paths that already exist in the database
        enable_metadata_extraction: If True, extract tier 1 & 2 metadata
        progress: Optional per-scan state for emit-once bookkeeping
        should_stop: Checked before each path's stats and spec, so a large library
            can pause here; once it returns True, no specs are returned
    """
    specs: list[SeedAssetSpec] = []
    tag_pool: set[str] = set()
    skipped = 0
    candidates: list[tuple[str, os.stat_result]] = []

    for p in paths:
        if should_stop():
            return [], set(), skipped
        abs_p = os.path.abspath(p)
        if _should_skip_extension(abs_p):
            skipped += 1
            continue
        if abs_p in existing_paths:
            skipped += 1
            continue
        if progress is not None:
            progress.files_statted += 1
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
                    emit(
                        "scanner.stat_failed",
                        site="discovery",
                        error_type=error_type(e),
                        error_kind=error_kind(e),
                    )
            continue
        if not stat_p.st_size:
            continue
        candidates.append((abs_p, stat_p))

    admitted_paths, _ = _two_stat_admit(candidates, progress, should_stop)
    candidate_stats = dict(candidates)
    for abs_p in admitted_paths:
        if should_stop():
            return [], set(), skipped
        yield_gil()
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


@dataclass
class SeedCounts:
    """What one seed_asset_specs call did besides creating records."""

    recovered: int = 0


class _SpecObservation(NamedTuple):
    """A spec's file as seen before the write transaction opens.

    ``snapshot`` is None when hashing is off, or when the file changed while being hashed.
    """

    stat_result: os.stat_result
    snapshot: tuple[str, os.stat_result] | None


def observe_asset_specs(
    specs: list[SeedAssetSpec], progress: _ScanProgress | None = None
) -> dict[str, _SpecObservation | None]:
    """Stat (and, in hashing mode, hash) each spec before the write transaction opens.

    ``None`` marks a path that vanished or could not be read.
    """
    hashing_is_enabled = mode.hashing_enabled()
    observed: dict[str, _SpecObservation | None] = {}
    for spec in specs:
        path = os.path.abspath(spec["abs_path"])
        if progress is not None:
            progress.files_statted += 1
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
    counts: SeedCounts | None = None,
    missing_ids_by_path: dict[str, list[str]] | None = None,
) -> tuple[int, Exception | None]:
    """``missing_ids_by_path`` is only used with hashing off. insert_asset_specs reads it
    before its write transaction opens; when omitted, it is read here."""
    if observed is None:
        observed = observe_asset_specs(specs)
    hashing_is_enabled = mode.hashing_enabled()
    if not hashing_is_enabled and missing_ids_by_path is None:
        missing_ids_by_path = missing_content_ids_by_path(session, _observed_paths(observed))
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
                if hashing_is_enabled:
                    recovery = recover_missing_content(
                        session, path, observation.snapshot, hashing_is_enabled=True
                    )
                else:
                    recovery = recover_missing_content_by_stat(
                        session, path, stat_result, (missing_ids_by_path or {}).get(path, [])
                    )
                if recovery == "recovered" and counts is not None:
                    counts.recovered += 1
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


def _observed_paths(observed: dict[str, _SpecObservation | None]) -> list[str]:
    return [path for path, observation in observed.items() if observation is not None]


def insert_asset_specs(
    specs: list[SeedAssetSpec],
    _tag_pool: set[str],
    progress: _ScanProgress | None = None,
) -> tuple[int, Exception | None]:
    if not specs:
        return 0, None
    observed = observe_asset_specs(specs, progress)
    missing_ids_by_path = None
    if not mode.hashing_enabled():
        with create_session() as sess:
            missing_ids_by_path = missing_content_ids_by_path(sess, _observed_paths(observed))
    counts = SeedCounts()
    with create_write_session() as sess:
        created, first_error = seed_asset_specs(
            sess, specs, observed, counts, missing_ids_by_path
        )
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
        if progress is not None:
            progress.recovered += counts.recovered
        return created, first_error


def unenriched_candidates_query(
    compute_hashes: bool, last_seen_id: str | None
) -> sa.Select[tuple[str, str, str]]:
    """Every unenriched live candidate after ``last_seen_id``, in id order."""
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
    return query.order_by(Asset.id.asc())


def build_unenriched_candidates_statement(
    prefixes: list[str],
    compute_hashes: bool,
    last_seen_id: str | None,
    limit: int = 1000,
) -> sa.Select[tuple[str, str, str]]:
    """The next page of candidates under at most PREFIX_BATCH_SIZE prefixes."""
    return (
        unenriched_candidates_query(compute_hashes, last_seen_id)
        .where(sa.or_(*(sql_path_under_prefix(AssetContent.path, p) for p in prefixes)))
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

    with create_session() as sess:
        if len(prefixes) <= PREFIX_BATCH_SIZE:
            statement = build_unenriched_candidates_statement(
                prefixes,
                compute_hashes,
                last_seen_id,
                limit,
            )
            rows = sess.execute(statement).all()
        else:
            # Too many prefixes for one SQL predicate. Paging each batch separately
            # would rescan to the end of the table on every page for any batch with
            # few matches, so filter a single id-ordered pass here instead.
            is_under = stored_path_under_prefixes(prefixes)
            candidates = sess.execute(
                unenriched_candidates_query(compute_hashes, last_seen_id).execution_options(yield_per=500)
            )
            rows = list(islice((row for row in candidates if is_under(row[2])), limit))

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
    if progress is not None:
        progress.files_statted += 1
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
                emit(
                    "scanner.stat_failed",
                    site="enrich",
                    error_type=error_type(e),
                    error_kind=error_kind(e),
                )
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
        dims = extract_media_metadata(file_path, mime_type=mime_type)
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
