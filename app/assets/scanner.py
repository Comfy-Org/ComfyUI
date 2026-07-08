import logging
import os
from pathlib import Path
from typing import Callable, Literal, TypedDict

import folder_paths
from app.assets.database.queries import (
    add_missing_tag_for_asset_id,
    bulk_update_enrichment_level,
    bulk_update_is_missing,
    bulk_update_needs_verify,
    delete_orphaned_seed_asset,
    delete_references_by_ids,
    ensure_tags_exist,
    get_asset_by_hash,
    get_reference_by_id,
    get_references_for_prefixes,
    get_unenriched_references,
    mark_references_missing_outside_prefixes,
    reassign_asset_references,
    remove_missing_tag_for_asset_id,
    set_reference_system_metadata,
    update_asset_hash_and_mime,
)
from app.assets.services.bulk_ingest import (
    SeedAssetSpec,
    batch_insert_seed_assets,
)
from app.assets.services.file_utils import (
    get_mtime_ns,
    is_visible,
    list_files_recursively,
    verify_file_unchanged,
)
from app.assets.services.hashing import HashCheckpoint, compute_blake3_hash
from app.assets.services.image_dimensions import extract_image_dimensions
from app.assets.services.metadata_extract import extract_file_metadata
from app.assets.services.path_utils import (
    compute_relative_filename,
    get_comfy_models_folders,
    get_name_and_tags_from_asset_path,
)
from app.database.db import create_session


class _RefInfo(TypedDict):
    ref_id: str
    file_path: str
    exists: bool
    stat_unchanged: bool
    needs_verify: bool


class _AssetAccumulator(TypedDict):
    hash: str | None
    size_db: int
    refs: list[_RefInfo]


RootType = Literal["models", "input", "output"]


def get_prefixes_for_root(root: RootType) -> list[str]:
    if root == "models":
        bases: list[str] = []
        for _bucket, paths in get_comfy_models_folders():
            bases.extend(paths)
        return [os.path.abspath(p) for p in bases]
    if root == "input":
        return [os.path.abspath(folder_paths.get_input_directory())]
    if root == "output":
        return [os.path.abspath(folder_paths.get_output_directory())]
    return []


def get_all_known_prefixes() -> list[str]:
    """Get all known asset prefixes across all root types."""
    all_roots: tuple[RootType, ...] = ("models", "input", "output")
    return [p for root in all_roots for p in get_prefixes_for_root(root)]


def collect_models_files() -> list[str]:
    out: list[str] = []
    for folder_name, bases in get_comfy_models_folders():
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
    update_missing_tags: bool = False,
) -> set[str] | None:
    """Reconcile asset references with filesystem for a root.

    - Toggle needs_verify per reference using mtime/size stat check
    - For hashed assets with at least one stat-unchanged ref: delete stale missing refs
    - For seed assets with all refs missing: delete Asset and its references
    - Optionally add/remove 'missing' tags based on stat check in this root
    - Optionally return surviving absolute paths

    Args:
        session: Database session
        root: Root type to scan
        collect_existing_paths: If True, return set of surviving file paths
        update_missing_tags: If True, update 'missing' tags based on file status

    Returns:
        Set of surviving absolute paths if collect_existing_paths=True, else None
    """
    prefixes = get_prefixes_for_root(root)
    if not prefixes:
        return set() if collect_existing_paths else None

    rows = get_references_for_prefixes(
        session, prefixes, include_missing=update_missing_tags
    )

    by_asset: dict[str, _AssetAccumulator] = {}
    for row in rows:
        acc = by_asset.get(row.asset_id)
        if acc is None:
            acc = {"hash": row.asset_hash, "size_db": row.size_bytes, "refs": []}
            by_asset[row.asset_id] = acc

        stat_unchanged = False
        try:
            exists = True
            stat_unchanged = verify_file_unchanged(
                mtime_db=row.mtime_ns,
                size_db=acc["size_db"],
                stat_result=os.stat(row.file_path, follow_symlinks=True),
            )
        except FileNotFoundError:
            exists = False
        except PermissionError:
            exists = True
            logging.debug("Permission denied accessing %s", row.file_path)
        except OSError as e:
            exists = False
            logging.debug("OSError checking %s: %s", row.file_path, e)

        acc["refs"].append(
            {
                "ref_id": row.reference_id,
                "file_path": row.file_path,
                "exists": exists,
                "stat_unchanged": stat_unchanged,
                "needs_verify": row.needs_verify,
            }
        )

    to_set_verify: list[str] = []
    to_clear_verify: list[str] = []
    stale_ref_ids: list[str] = []
    to_mark_missing: list[str] = []
    to_clear_missing: list[str] = []
    survivors: set[str] = set()

    for aid, acc in by_asset.items():
        a_hash = acc["hash"]
        refs = acc["refs"]
        any_unchanged = any(r["stat_unchanged"] for r in refs)
        all_missing = all(not r["exists"] for r in refs)

        for r in refs:
            if not r["exists"]:
                to_mark_missing.append(r["ref_id"])
                continue
            if r["stat_unchanged"]:
                to_clear_missing.append(r["ref_id"])
                if r["needs_verify"]:
                    to_clear_verify.append(r["ref_id"])
            if not r["stat_unchanged"] and not r["needs_verify"]:
                to_set_verify.append(r["ref_id"])

        if a_hash is None:
            if refs and all_missing:
                delete_orphaned_seed_asset(session, aid)
            else:
                for r in refs:
                    if r["exists"]:
                        survivors.add(os.path.abspath(r["file_path"]))
            continue

        if any_unchanged:
            for r in refs:
                if not r["exists"]:
                    stale_ref_ids.append(r["ref_id"])
            if update_missing_tags:
                try:
                    remove_missing_tag_for_asset_id(session, asset_id=aid)
                except Exception as e:
                    logging.warning(
                        "Failed to remove missing tag for asset %s: %s", aid, e
                    )
        elif update_missing_tags:
            try:
                add_missing_tag_for_asset_id(session, asset_id=aid, origin="automatic")
            except Exception as e:
                logging.warning("Failed to add missing tag for asset %s: %s", aid, e)

        for r in refs:
            if r["exists"]:
                survivors.add(os.path.abspath(r["file_path"]))

    delete_references_by_ids(session, stale_ref_ids)
    stale_set = set(stale_ref_ids)
    to_mark_missing = [ref_id for ref_id in to_mark_missing if ref_id not in stale_set]
    bulk_update_is_missing(session, to_mark_missing, value=True)
    bulk_update_is_missing(session, to_clear_missing, value=False)
    bulk_update_needs_verify(session, to_set_verify, value=True)
    bulk_update_needs_verify(session, to_clear_verify, value=False)

    return survivors if collect_existing_paths else None


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
                update_missing_tags=True,
            )
            sess.commit()
            return survivors or set()
    except Exception as e:
        logging.exception("fast DB scan failed for %s: %s", root, e)
        return set()


def mark_missing_outside_prefixes_safely(prefixes: list[str]) -> int:
    """Mark references as missing when outside the given prefixes.

    This is a non-destructive soft-delete. Returns count marked or 0 on failure.
    """
    try:
        with create_session() as sess:
            count = mark_references_missing_outside_prefixes(sess, prefixes)
            sess.commit()
            return count
    except Exception as e:
        logging.exception("marking missing assets failed: %s", e)
        return 0


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
    compute_hashes: bool = False,
) -> tuple[list[SeedAssetSpec], set[str], int]:
    """Build asset specs from paths, returning (specs, tag_pool, skipped_count).

    Args:
        paths: List of file paths to process
        existing_paths: Set of paths that already exist in the database
        enable_metadata_extraction: If True, extract tier 1 & 2 metadata
        compute_hashes: If True, compute blake3 hashes (slow for large files)
    """
    specs: list[SeedAssetSpec] = []
    tag_pool: set[str] = set()
    skipped = 0

    for p in paths:
        abs_p = os.path.abspath(p)
        if abs_p in existing_paths:
            skipped += 1
            continue
        try:
            stat_p = os.stat(abs_p, follow_symlinks=True)
        except OSError:
            continue
        if not stat_p.st_size:
            continue
        name, tags = get_name_and_tags_from_asset_path(abs_p)
        rel_fname = compute_relative_filename(abs_p)

        # Extract metadata (tier 1: filesystem, tier 2: safetensors header)
        metadata = None
        if enable_metadata_extraction:
            metadata = extract_file_metadata(
                abs_p,
                stat_result=stat_p,
                relative_filename=rel_fname,
            )

        # Compute hash if requested
        asset_hash: str | None = None
        if compute_hashes:
            try:
                digest, _ = compute_blake3_hash(abs_p)
                asset_hash = "blake3:" + digest
            except Exception as e:
                logging.warning("Failed to hash %s: %s", abs_p, e)

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
                "hash": asset_hash,
                "mime_type": mime_type,
                "job_id": None,
            }
        )
        tag_pool.update(tags)

    return specs, tag_pool, skipped



def insert_asset_specs(specs: list[SeedAssetSpec], tag_pool: set[str]) -> int:
    """Insert asset specs into database, returning count of created refs."""
    if not specs:
        return 0
    with create_session() as sess:
        if tag_pool:
            ensure_tags_exist(sess, tag_pool)
        result = batch_insert_seed_assets(sess, specs=specs, owner_id="")
        sess.commit()
        return result.inserted_refs


# Enrichment level constants
ENRICHMENT_STUB = 0  # Fast scan: path, size, mtime only
ENRICHMENT_METADATA = 1  # Metadata extracted (safetensors header, mime type)
ENRICHMENT_HASHED = 2  # Hash computed (blake3)


def get_unenriched_assets_for_roots(
    roots: tuple[RootType, ...],
    max_level: int = ENRICHMENT_STUB,
    limit: int = 1000,
) -> list:
    """Get assets that need enrichment for the given roots.

    Args:
        roots: Tuple of root types to scan
        max_level: Maximum enrichment level to include
        limit: Maximum number of rows to return

    Returns:
        List of UnenrichedReferenceRow
    """
    prefixes: list[str] = []
    for root in roots:
        prefixes.extend(get_prefixes_for_root(root))

    if not prefixes:
        return []

    with create_session() as sess:
        return get_unenriched_references(
            sess, prefixes, max_level=max_level, limit=limit
        )


def enrich_asset(
    session,
    file_path: str,
    reference_id: str,
    asset_id: str,
    extract_metadata: bool = True,
    compute_hash: bool = False,
    interrupt_check: Callable[[], bool] | None = None,
    hash_checkpoints: dict[str, HashCheckpoint] | None = None,
) -> int:
    """Enrich a single asset with metadata and/or hash.

    Args:
        session: Database session (caller manages lifecycle)
        file_path: Absolute path to the file
        reference_id: ID of the reference to update
        asset_id: ID of the asset to update (for mime_type and hash)
        extract_metadata: If True, extract safetensors header and mime type
        compute_hash: If True, compute blake3 hash
        interrupt_check: Optional non-blocking callable that returns True if
            the operation should be interrupted (e.g. paused or cancelled)
        hash_checkpoints: Optional dict for saving/restoring hash progress
            across interruptions, keyed by file path

    Returns:
        New enrichment level achieved
    """
    new_level = ENRICHMENT_STUB

    try:
        stat_p = os.stat(file_path, follow_symlinks=True)
    except OSError:
        return new_level

    initial_mtime_ns = get_mtime_ns(stat_p)
    rel_fname = compute_relative_filename(file_path)
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
            new_level = ENRICHMENT_METADATA

    full_hash: str | None = None
    if compute_hash:
        try:
            mtime_before = get_mtime_ns(stat_p)
            size_before = stat_p.st_size

            # Restore checkpoint if available and file unchanged
            checkpoint = None
            if hash_checkpoints is not None:
                checkpoint = hash_checkpoints.get(file_path)
                if checkpoint is not None:
                    cur_stat = os.stat(file_path, follow_symlinks=True)
                    if (checkpoint.mtime_ns != get_mtime_ns(cur_stat)
                            or checkpoint.file_size != cur_stat.st_size):
                        checkpoint = None
                        hash_checkpoints.pop(file_path, None)
                    else:
                        mtime_before = get_mtime_ns(cur_stat)

            digest, new_checkpoint = compute_blake3_hash(
                file_path,
                interrupt_check=interrupt_check,
                checkpoint=checkpoint,
            )

            if digest is None:
                # Interrupted — save checkpoint for later resumption
                if hash_checkpoints is not None and new_checkpoint is not None:
                    new_checkpoint.mtime_ns = mtime_before
                    new_checkpoint.file_size = size_before
                    hash_checkpoints[file_path] = new_checkpoint
                return new_level

            # Completed — clear any saved checkpoint
            if hash_checkpoints is not None:
                hash_checkpoints.pop(file_path, None)

            stat_after = os.stat(file_path, follow_symlinks=True)
            mtime_after = get_mtime_ns(stat_after)
            if mtime_before != mtime_after:
                logging.warning("File modified during hashing, discarding hash: %s", file_path)
            else:
                full_hash = f"blake3:{digest}"
                metadata_ok = not extract_metadata or metadata is not None
                if metadata_ok:
                    new_level = ENRICHMENT_HASHED
        except Exception as e:
            logging.warning("Failed to hash %s: %s", file_path, e)

    # Optimistic guard: if the reference's mtime_ns changed since we
    # started (e.g. ingest_existing_file updated it), our results are
    # stale — discard them to avoid overwriting fresh registration data.
    ref = get_reference_by_id(session, reference_id)
    if ref is None or ref.mtime_ns != initial_mtime_ns:
        session.rollback()
        logging.info(
            "Ref %s mtime changed during enrichment, discarding stale result",
            reference_id,
        )
        return ENRICHMENT_STUB

    if extract_metadata and metadata:
        system_metadata = metadata.to_user_metadata()
        if mime_type and mime_type.startswith("image/"):
            dims = extract_image_dimensions(file_path, mime_type=mime_type)
            if dims:
                system_metadata.update(dims)
        set_reference_system_metadata(session, reference_id, system_metadata)

    if full_hash:
        existing = get_asset_by_hash(session, full_hash)
        if existing and existing.id != asset_id:
            reassign_asset_references(session, asset_id, existing.id, reference_id)
            delete_orphaned_seed_asset(session, asset_id)
            if mime_type:
                update_asset_hash_and_mime(session, existing.id, mime_type=mime_type)
        else:
            update_asset_hash_and_mime(session, asset_id, full_hash, mime_type)
    elif mime_type:
        update_asset_hash_and_mime(session, asset_id, mime_type=mime_type)

    bulk_update_enrichment_level(session, [reference_id], new_level)
    session.commit()

    return new_level


def enrich_assets_batch(
    rows: list,
    extract_metadata: bool = True,
    compute_hash: bool = False,
    interrupt_check: Callable[[], bool] | None = None,
    hash_checkpoints: dict[str, HashCheckpoint] | None = None,
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
        hash_checkpoints: Optional dict for saving/restoring hash progress
            across interruptions, keyed by file path

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
                new_level = enrich_asset(
                    sess,
                    file_path=row.file_path,
                    reference_id=row.reference_id,
                    asset_id=row.asset_id,
                    extract_metadata=extract_metadata,
                    compute_hash=compute_hash,
                    interrupt_check=interrupt_check,
                    hash_checkpoints=hash_checkpoints,
                )
                if new_level > row.enrichment_level:
                    enriched += 1
                else:
                    failed_ids.append(row.reference_id)
            except Exception as e:
                logging.warning("Failed to enrich %s: %s", row.file_path, e)
                sess.rollback()
                failed_ids.append(row.reference_id)

    return enriched, failed_ids
