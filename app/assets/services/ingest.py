"""Turns incoming bytes into catalogued assets: multipart uploads moved into a
hash-addressed destination, files registered where they already sit, and
records created from a hash the catalog already holds. Every path persists the
stat that hashing verified, so a row's recorded size and mtime describe the
same observation as its hash. A live row already at the destination is
reconciled before the write, so an upload never adopts a fresh hash onto
records created for bytes it just replaced.
"""

import contextlib
import errno
import logging
import mimetypes
import os
import shutil
import tempfile
from typing import Any, NamedTuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.assets import mode
from app.assets.database.models import Asset, AssetContent, AssetTag
from app.assets.database.queries.records import (
    create_content_reporting_insert,
    create_record,
    mark_content_missing,
)
from app.assets.event_log import emit, error_type
from app.assets.helpers import normalize_tags, to_stored_hash
from app.assets.services.file_utils import get_mtime_ns, get_size_and_mtime_ns
from app.assets.services.image_dimensions import extract_image_dimensions
from app.assets.services.lookup import (
    claim_qualified_content,
    lookup_for_view,
    refresh_qualified_content,
)
from app.assets.services.metadata_extract import extract_file_metadata
from app.assets.services.path_utils import (
    compute_loader_path,
    get_name_and_tags_from_asset_path,
    resolve_destination_from_tags,
    validate_path_within_base,
)
from app.assets.services.schemas import (
    AssetData,
    RegisteredAsset,
    ReferenceData,
    UploadResult,
    UserMetadata,
)
from app.assets.services.snapshot_hash import snapshot_hash
from app.database.db import create_session, run_write_txn


def _normalize_hash_input(hash_str: str) -> str:
    if not hash_str:
        return hash_str
    normalized = hash_str.strip().lower()
    digest = normalized.partition(":")[2] or normalized
    return to_stored_hash(digest)


def _extract_system_metadata_sync(
    locator: str,
    mime_type: str | None,
    stat_result: os.stat_result | None = None,
) -> dict[str, Any]:
    """Extract ``system_metadata`` at registration time (S29/D8).

    Mirrors the ``scanner.enrich`` pass: tier-1/tier-2 file metadata plus image
    dimensions for image MIME types, so records carry metadata at creation
    instead of waiting for the background enrich pass to fill it.
    """
    metadata = extract_file_metadata(
        locator,
        stat_result=stat_result,
        relative_filename=compute_loader_path(locator),
    )
    system_metadata = metadata.to_user_metadata()
    if mime_type and mime_type.startswith("image/"):
        dims = extract_image_dimensions(locator, mime_type=mime_type)
        if dims:
            system_metadata.update(dims)
    return system_metadata


def _discard_unreferenced_content(session: Session, content_id: str) -> None:
    """Remove a content row left orphaned by a failed registration.

    ``create_content`` inserts inside a SAVEPOINT (``begin_nested``); under
    pysqlite that insert survives the enclosing ``rollback`` because pysqlite
    has no real nested transaction. When the follow-on ``create_record`` fails
    we would otherwise leak an unreferenced content row, so delete it explicitly
    once we have confirmed no record points at it. Best-effort: cleanup errors
    are logged and swallowed so the original failure is what surfaces.
    """
    try:
        ref_count = session.scalar(
            select(func.count())
            .select_from(Asset)
            .where(Asset.content_id == content_id)
        )
        if ref_count:
            return
        content = session.get(AssetContent, content_id)
        if content is not None:
            session.delete(content)
            session.commit()
    except Exception:
        logging.exception("Failed to discard orphan content %s", content_id)


def _sanitize_filename(name: str | None, fallback: str) -> str:
    n = os.path.basename((name or "").strip() or fallback)
    return n if n else fallback


class HashMismatchError(Exception):
    pass


class UploadUnstableError(Exception):
    pass


class DependencyMissingError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


_UPLOAD_HASH_ATTEMPTS = 3


def _remove_temp_path(temp_path: str | None) -> None:
    if not temp_path or not os.path.exists(temp_path):
        return
    with contextlib.suppress(OSError):
        os.remove(temp_path)
    parent = os.path.dirname(temp_path)
    with contextlib.suppress(OSError):
        if parent and os.path.isdir(parent):
            os.rmdir(parent)


def _snapshot_hash_with_retry(path: str) -> tuple[str, os.stat_result]:
    for _ in range(_UPLOAD_HASH_ATTEMPTS):
        snapshot = snapshot_hash(path)
        if snapshot is not None:
            return snapshot
    raise UploadUnstableError("upload file changed during hashing")


def _hash_mode_dest_path(
    tags: list[str],
    digest: str,
    client_filename: str | None,
    name: str | None,
) -> str:
    base_dir, subdirs = resolve_destination_from_tags(tags)
    dest_dir = os.path.join(base_dir, *subdirs) if subdirs else base_dir
    os.makedirs(dest_dir, exist_ok=True)
    src_for_ext = (client_filename or name or "").strip()
    _ext = os.path.splitext(os.path.basename(src_for_ext))[1] if src_for_ext else ""
    ext = _ext if 0 < len(_ext) <= 16 else ""
    hashed_basename = f"{digest}{ext}"
    dest_abs = os.path.abspath(os.path.join(dest_dir, hashed_basename))
    validate_path_within_base(dest_abs, base_dir)
    return dest_abs


def _guess_upload_mime_type(
    mime_type: str | None,
    client_filename: str | None,
    name: str | None,
    fallback_basename: str,
) -> str:
    src_for_ext = (client_filename or name or "").strip()
    if mime_type:
        return mime_type
    guessed = mimetypes.guess_type(os.path.basename(src_for_ext), strict=False)[0]
    if guessed:
        return guessed
    guessed = mimetypes.guess_type(fallback_basename, strict=False)[0]
    return guessed or "application/octet-stream"


def _move_temp_to_dest(temp_path: str, dest_abs: str) -> None:
    os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
    try:
        os.replace(temp_path, dest_abs)
    except OSError as error:
        if error.errno == errno.EXDEV:
            destination_dir = os.path.dirname(dest_abs)
            destination_fd, destination_temp = tempfile.mkstemp(
                dir=destination_dir,
                prefix=f".{os.path.basename(dest_abs)}.",
                suffix=".tmp",
            )
            try:
                os.close(destination_fd)
                shutil.copy2(temp_path, destination_temp)
                os.replace(destination_temp, dest_abs)
            except BaseException:
                if os.path.exists(destination_temp):
                    os.unlink(destination_temp)
                raise
            os.unlink(temp_path)
            return
        raise RuntimeError(f"failed to move uploaded file into place: {error}") from error
    except Exception as e:
        raise RuntimeError(f"failed to move uploaded file into place: {e}") from e


def _create_upload_record(
    session: Session,
    content_id: str,
    prepared: "_PreparedUploadRecord",
) -> Asset:
    preflight = prepared.preflight
    spec = preflight.spec
    record = create_record(
        session,
        content_id,
        spec.name,
        mime_type=spec.mime_type,
        loader_path=compute_loader_path(preflight.path),
        tags=spec.tags,
        system_metadata=prepared.system_metadata,
    )
    if spec.user_metadata:
        record.user_metadata = dict(spec.user_metadata)
    if spec.preview_id:
        record.preview_id = spec.preview_id
    session.flush()
    return record


def _record_to_upload_result(
    session: Session, record: Asset, *, created_new: bool
) -> UploadResult:
    content = session.get(AssetContent, record.content_id)
    tag_names = list(
        session.scalars(
            select(AssetTag.tag_name)
            .where(AssetTag.asset_id == record.id)
            .order_by(AssetTag.tag_name)
        )
    )
    api_hash = content.hash if content else None
    ref = ReferenceData(
        id=record.id,
        name=record.name,
        file_path=content.path if content else None,
        loader_path=record.loader_path,
        user_metadata=record.user_metadata,
        preview_id=record.preview_id,
        system_metadata=record.system_metadata,
        job_id=record.job_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        last_access_time=record.last_access_time,
    )
    asset = AssetData(
        hash=api_hash,
        size_bytes=content.size_bytes if content else None,
        mime_type=record.mime_type,
    )
    return UploadResult(
        ref=ref,
        content_id=record.content_id,
        asset=asset,
        tags=tag_names,
        created_new=created_new,
    )


class _ContentFacts(NamedTuple):

    stored_hash: str
    size_bytes: int
    mtime_ns: int


def _reconcile_live_content_at_path(
    session: Session,
    locator: str,
    facts: _ContentFacts,
    *,
    content_written: bool,
) -> None:
    """Make the live content row at ``locator`` honest about the bytes there.

    ``create_content`` resolves a live-path uniqueness conflict
    (``uq_asset_contents_path_live``) by handing back the row already at that
    path, so a caller that registers a path without reconciling first gets the
    OLD row's hash and size.

    ``content_written`` is the caller's own knowledge of whether it just wrote
    bytes at ``locator``, and it is what makes an unhashed row readable at all:
    equal ``size_bytes`` is NOT proof of equal bytes, so a caller that replaced
    the file cannot adopt a fresh hash onto records that were created for the
    bytes it destroyed. Cases, in order:

    * the hash already matches - proof the bytes are identical, so the OBSERVED
      stat is the trustworthy one. Refresh it: a touch, a copy, or a same-bytes
      rewrite moves ``mtime_ns`` without changing a byte, and a row left stale
      fails ``lookup._stat_consistent``, which makes the content unservable by
      hash and silently routes the caller into creating a duplicate record.
    * the hash is ``None`` and the caller wrote nothing - "not yet hashed" is
      NOT evidence of different bytes. Scanner-seeded rows and executed outputs
      (D14a) both defer hashing to the enrich pass, so retiring here would mark
      every record on the row missing for a file that never changed. Adopt the
      hash we just computed the way ``scanner.enrich_asset`` would, unless the
      row's recorded ``size_bytes`` positively contradicts the file. A
      differing ``mtime_ns`` is deliberately not a contradiction: renaming a
      file into place, copying it, or touching it all move the mtime while the
      bytes stay identical.
    * anything else - a known but different hash, or unhashed bytes the caller
      has just overwritten. Either way the row can no longer speak for what is
      at ``locator``: retire it so the fresh ``create_content`` inserts a new
      live row and the old records keep describing the old bytes.
    """
    existing = session.scalars(
        select(AssetContent).where(
            AssetContent.path == locator,
            AssetContent.is_missing.is_(False),
        )
    ).first()
    if existing is None:
        return
    if existing.hash == facts.stored_hash:
        existing.size_bytes = facts.size_bytes
        existing.mtime_ns = facts.mtime_ns
        session.flush()
        return
    if (
        existing.hash is None
        and not content_written
        and existing.size_bytes == facts.size_bytes
    ):
        existing.hash = facts.stored_hash
        existing.mtime_ns = facts.mtime_ns
        session.flush()
        return
    mark_content_missing(session, existing.id)


def _upload_destination_or_none(
    tags: list[str] | None,
    digest: str,
    client_filename: str | None,
    name: str | None,
) -> str | None:
    if not tags:
        return None
    try:
        return _hash_mode_dest_path(tags, digest, client_filename, name)
    except ValueError:
        return None


class _UploadRecordSpec(NamedTuple):

    name: str
    tags: list[str]
    mime_type: str | None
    user_metadata: UserMetadata
    preview_id: str | None


class _FileSignature(NamedTuple):

    path: str
    size_bytes: int
    mtime_ns: int


class _UploadRecordPreflight(NamedTuple):

    content_id: str | None
    stored_hash: str | None
    path: str
    signature: _FileSignature
    spec: _UploadRecordSpec


class _PreparedUploadRecord(NamedTuple):

    preflight: _UploadRecordPreflight
    system_metadata: dict[str, Any]


class _SettleTargetPreflight(NamedTuple):

    content_id: str
    content_hash: str | None
    content_size_bytes: int
    content_mtime_ns: int | None
    signature: _FileSignature


class _PreparedSettleTarget(NamedTuple):

    preflight: _SettleTargetPreflight
    facts: _ContentFacts | None


class _CachedRegistrationPreflight(NamedTuple):

    content_id: str
    sibling_id: str | None
    sibling_metadata: dict[str, Any] | None
    signature: _FileSignature | None


class _PreflightStale(Exception):
    pass


def _file_signature(path: str) -> _FileSignature:
    stat_result = os.stat(path, follow_symlinks=True)
    return _FileSignature(path, stat_result.st_size, get_mtime_ns(stat_result))


def _file_signature_matches(signature: _FileSignature) -> bool:
    try:
        return _file_signature(signature.path) == signature
    except OSError:
        return False


def _preflight_upload_record(
    stored_hash: str | None,
    fallback_path: str | None,
    spec: _UploadRecordSpec,
) -> _UploadRecordPreflight | None:
    """Check the target before metadata I/O so it does not extend the writer lease."""
    with create_session() as session:
        content = lookup_for_view(session, stored_hash) if stored_hash else None
        if content is None:
            if fallback_path is None:
                return None
            content_id = None
            path = fallback_path
        else:
            content_id = content.id
            path = content.path
        if spec.preview_id is not None and session.get(Asset, spec.preview_id) is None:
            raise ValueError(
                f"preview_id {spec.preview_id!r} does not reference an existing asset"
            )
    return _UploadRecordPreflight(
        content_id,
        stored_hash,
        path,
        _file_signature(path),
        spec,
    )


def _prepare_upload_record(
    preflight: _UploadRecordPreflight,
) -> _PreparedUploadRecord:
    return _PreparedUploadRecord(
        preflight,
        _extract_system_metadata_sync(
            preflight.path,
            preflight.spec.mime_type,
        ),
    )


def _assert_upload_preflight_current(
    session: Session,
    preflight: _UploadRecordPreflight,
) -> None:
    if not _file_signature_matches(preflight.signature):
        raise _PreflightStale
    preview_id = preflight.spec.preview_id
    if preview_id is not None and session.get(Asset, preview_id) is None:
        raise _PreflightStale


def _create_upload_record_in_txn(
    session: Session,
    content_id: str,
    spec: _UploadRecordSpec,
    abs_path: str,
) -> Asset:
    if spec.preview_id is not None and session.get(Asset, spec.preview_id) is None:
        raise ValueError(
            f"preview_id {spec.preview_id!r} does not reference an existing asset"
        )
    record = create_record(
        session,
        content_id,
        spec.name,
        mime_type=spec.mime_type,
        loader_path=compute_loader_path(abs_path),
        tags=spec.tags,
        system_metadata=_extract_system_metadata_sync(abs_path, spec.mime_type),
    )
    if spec.user_metadata:
        record.user_metadata = dict(spec.user_metadata)
    if spec.preview_id:
        record.preview_id = spec.preview_id
    session.flush()
    return record


def _apply_reused_upload_record(
    session: Session,
    prepared: _PreparedUploadRecord,
) -> UploadResult:
    preflight = prepared.preflight
    content = lookup_for_view(session, preflight.stored_hash)
    if (
        content is None
        or content.id != preflight.content_id
        or content.path != preflight.path
    ):
        raise _PreflightStale
    _assert_upload_preflight_current(session, preflight)
    if not claim_qualified_content(session, content.id, preflight.stored_hash):
        raise _PreflightStale
    content = refresh_qualified_content(session, content.id)
    if content is None or content.path != preflight.path:
        raise _PreflightStale
    record = _create_upload_record(session, content.id, prepared)
    return _record_to_upload_result(session, record, created_new=True)


def _reuse_qualified_content_in_txn(
    session: Session,
    stored_hash: str,
    spec: _UploadRecordSpec,
) -> UploadResult | None:
    content = lookup_for_view(session, stored_hash)
    if content is None:
        return None
    if not claim_qualified_content(session, content.id, stored_hash):
        session.rollback()
        return None
    content = refresh_qualified_content(session, content.id)
    if content is None:
        session.rollback()
        return None
    record = _create_upload_record_in_txn(session, content.id, spec, content.path)
    return _record_to_upload_result(session, record, created_new=True)


def _reuse_qualified_content(
    stored_hash: str,
    spec: _UploadRecordSpec,
) -> UploadResult | None:
    for _restart in range(4):
        preflight = _preflight_upload_record(stored_hash, None, spec)
        if preflight is None:
            return None
        prepared = _prepare_upload_record(preflight)
        try:
            return run_write_txn(
                lambda session: _apply_reused_upload_record(session, prepared)
            )
        except _PreflightStale:
            continue
    logging.warning(
        "Upload preflight changed three times; falling back to in-transaction metadata extraction"
    )
    return run_write_txn(
        lambda session: _reuse_qualified_content_in_txn(session, stored_hash, spec)
    )


def _preflight_settle_target(dest_abs: str) -> _SettleTargetPreflight | None:
    """Read incumbent facts so its hash runs before, not inside, the writer lease."""
    if not os.path.isfile(dest_abs):
        return None
    with create_session() as session:
        existing = session.scalars(
            select(AssetContent).where(
                AssetContent.path == dest_abs,
                AssetContent.is_missing.is_(False),
            )
        ).first()
        if existing is None:
            return None
        signature = _file_signature(dest_abs)
        if (
            existing.hash is not None
            and existing.size_bytes == signature.size_bytes
            and existing.mtime_ns == signature.mtime_ns
        ):
            return None
        return _SettleTargetPreflight(
            existing.id,
            existing.hash,
            existing.size_bytes,
            existing.mtime_ns,
            signature,
        )


def _prepare_settle_target(
    preflight: _SettleTargetPreflight,
) -> _PreparedSettleTarget:
    try:
        digest, verified_stat = _snapshot_hash_with_retry(preflight.signature.path)
    except (UploadUnstableError, OSError):
        return _PreparedSettleTarget(preflight, None)
    return _PreparedSettleTarget(
        preflight,
        _ContentFacts(
            to_stored_hash(digest),
            verified_stat.st_size,
            verified_stat.st_mtime_ns,
        ),
    )


def _apply_settle_target(
    session: Session,
    prepared: _PreparedSettleTarget,
) -> None:
    preflight = prepared.preflight
    existing = session.get(AssetContent, preflight.content_id)
    if (
        existing is None
        or existing.path != preflight.signature.path
        or existing.is_missing
        or existing.hash != preflight.content_hash
        or existing.size_bytes != preflight.content_size_bytes
        or existing.mtime_ns != preflight.content_mtime_ns
        or not _file_signature_matches(preflight.signature)
    ):
        raise _PreflightStale
    if prepared.facts is None:
        mark_content_missing(session, existing.id)
        return
    _reconcile_live_content_at_path(
        session,
        preflight.signature.path,
        prepared.facts,
        content_written=False,
    )


def _settle_destination_before_write_in_txn(session: Session, dest_abs: str) -> None:
    if not os.path.isfile(dest_abs):
        return
    existing = session.scalars(
        select(AssetContent).where(
            AssetContent.path == dest_abs,
            AssetContent.is_missing.is_(False),
        )
    ).first()
    if existing is None:
        return
    size_bytes, mtime_ns = get_size_and_mtime_ns(dest_abs)
    if (
        existing.hash is not None
        and existing.size_bytes == size_bytes
        and existing.mtime_ns == mtime_ns
    ):
        return
    try:
        incumbent_digest, verified_stat = _snapshot_hash_with_retry(dest_abs)
    except (UploadUnstableError, OSError):
        mark_content_missing(session, existing.id)
        return
    _reconcile_live_content_at_path(
        session,
        dest_abs,
        _ContentFacts(
            to_stored_hash(incumbent_digest),
            verified_stat.st_size,
            verified_stat.st_mtime_ns,
        ),
        content_written=False,
    )


def _settle_destination_before_write(dest_abs: str) -> None:
    for _restart in range(4):
        preflight = _preflight_settle_target(dest_abs)
        if preflight is None:
            return
        prepared = _prepare_settle_target(preflight)
        try:
            run_write_txn(lambda session: _apply_settle_target(session, prepared))
            return
        except _PreflightStale:
            continue
    logging.warning(
        "Upload destination preflight changed three times; falling back to in-transaction hashing"
    )
    run_write_txn(
        lambda session: _settle_destination_before_write_in_txn(session, dest_abs)
    )
    return


def _create_content_and_upload_record(
    stored_hash: str,
    path: str,
    facts: _ContentFacts,
    content_written: bool,
    spec: _UploadRecordSpec,
) -> UploadResult:
    for _restart in range(4):
        preflight = _preflight_upload_record(None, path, spec)
        if preflight is None:
            raise RuntimeError("new upload record requires a destination path")
        prepared = _prepare_upload_record(preflight)

        def _work(session: Session) -> UploadResult:
            _assert_upload_preflight_current(session, prepared.preflight)
            _reconcile_live_content_at_path(
                session,
                path,
                facts,
                content_written=content_written,
            )
            content, inserted = create_content_reporting_insert(
                session,
                path,
                stored_hash,
                facts.size_bytes,
                facts.mtime_ns,
            )
            created_content_id = content.id if inserted else None
            try:
                record = _create_upload_record(session, content.id, prepared)
            except Exception:
                session.rollback()
                if created_content_id is not None:
                    _discard_unreferenced_content(session, created_content_id)
                raise
            return _record_to_upload_result(session, record, created_new=True)

        try:
            return run_write_txn(_work)
        except _PreflightStale:
            continue

    logging.warning(
        "Upload preflight for a new asset did not settle in 4 attempts; refusing the upload "
        "rather than persisting content facts and file metadata that describe different bytes"
    )
    raise RuntimeError(
        f"Upload preflight for {path} did not settle in 4 attempts; "
        "refusing to persist content facts and file metadata that describe different bytes"
    )


def upload_from_temp_path(
    temp_path: str,
    name: str | None = None,
    tags: list[str] | None = None,
    user_metadata: dict | None = None,
    client_filename: str | None = None,
    expected_hash: str | None = None,
    mime_type: str | None = None,
    preview_id: str | None = None,
) -> UploadResult:
    display_name = _sanitize_filename(name or client_filename, fallback="upload")
    user_metadata = user_metadata or {}

    try:
        digest, verified_stat = _snapshot_hash_with_retry(temp_path)
    except UploadUnstableError:
        _remove_temp_path(temp_path)
        raise
    stored_hash = to_stored_hash(digest)
    if expected_hash and stored_hash != _normalize_hash_input(expected_hash):
        _remove_temp_path(temp_path)
        raise HashMismatchError("Uploaded file hash does not match provided hash.")

    spec = _UploadRecordSpec(
        display_name,
        normalize_tags([*(tags or []), "uploaded"]),
        mime_type,
        user_metadata,
        preview_id,
    )
    settle_target = _upload_destination_or_none(tags, digest, client_filename, name)
    if settle_target is not None:
        _settle_destination_before_write(settle_target)
    reused = _reuse_qualified_content(stored_hash, spec)
    if reused is not None:
        _remove_temp_path(temp_path)
        return reused

    if not tags:
        _remove_temp_path(temp_path)
        raise ValueError("tags are required for new asset uploads")

    dest_abs = _hash_mode_dest_path(tags, digest, client_filename, name)
    content_type = _guess_upload_mime_type(
        mime_type, client_filename, name, os.path.basename(dest_abs)
    )
    _move_temp_to_dest(temp_path, dest_abs)
    return _create_content_and_upload_record(
        stored_hash,
        dest_abs,
        _ContentFacts(
            stored_hash,
            verified_stat.st_size,
            verified_stat.st_mtime_ns,
        ),
        True,
        _UploadRecordSpec(
            display_name,
            normalize_tags([*(tags or []), "uploaded"]),
            content_type,
            user_metadata,
            preview_id,
        ),
    )


def register_file_in_place(
    abs_path: str,
    name: str,
    tags: list[str],
    mime_type: str | None = None,
    *,
    content_written: bool = True,
) -> UploadResult:
    """Register an already-saved file in the asset database without moving it.

    Used by ``/upload/image`` after the handler writes (or reuses) bytes on disk.
    ``compare_image_hash`` same-name dedup is legacy physical behavior and stays
    outside hash-mode policy (parallel to path-form ``/view`` semantics).

    Dedup is scoped to ``abs_path``, deliberately weaker than the multipart
    endpoint's global content dedup. ``_reconcile_live_content_at_path`` already
    settles the same-path row before this returns, so a global hash match adds
    nothing there - the only thing it can still reach is a DIFFERENT file, and
    honouring that would leave the just-written ``abs_path`` untracked and hand
    back an asset describing someone else's path. Equal hashes across paths stay
    discoverable by hash, they are simply not merged.
    ``_reconcile_live_content_at_path`` is the whole dedup story here - it leaves
    at most one live row at ``abs_path``, so re-registering unchanged bytes reuses
    that content row while still writing a new record, which is what a repeat save
    through ``/upload/image`` is.

    ``content_written`` reports whether the caller just wrote bytes at
    ``abs_path``. The handler always knows - ``/upload/image`` skips the write
    exactly when ``compare_image_hash`` proves the bytes are already there - and
    it is the only sound way to reconcile an unhashed row at that path, because
    the bytes it was created for are gone by the time we are called. It defaults
    to the conservative answer so a caller that does not know retires rather
    than adopts.
    """
    locator = os.path.abspath(abs_path)
    display_name = _sanitize_filename(name, fallback=os.path.basename(locator))
    try:
        _, path_tags = get_name_and_tags_from_asset_path(locator)
    except ValueError:
        path_tags = []
    merged_tags = normalize_tags([*path_tags, *tags, "uploaded"])
    content_type = _guess_upload_mime_type(
        mime_type, name, name, os.path.basename(locator)
    )
    digest, verified_stat = _snapshot_hash_with_retry(locator)
    size_bytes, mtime_ns = verified_stat.st_size, verified_stat.st_mtime_ns
    stored_hash = to_stored_hash(digest)
    def _reconcile_work(session: Session) -> None:
        _reconcile_live_content_at_path(
            session,
            locator,
            _ContentFacts(stored_hash, size_bytes, mtime_ns),
            content_written=content_written,
        )
    run_write_txn(_reconcile_work)
    return _create_content_and_upload_record(
        stored_hash,
        locator,
        _ContentFacts(stored_hash, size_bytes, mtime_ns),
        content_written,
        _UploadRecordSpec(
            display_name,
            merged_tags,
            content_type,
            {},
            None,
        ),
    )


def create_from_hash(
    hash_str: str,
    name: str,
    tags: list[str] | None = None,
    user_metadata: dict | None = None,
    mime_type: str | None = None,
    preview_id: str | None = None,
) -> UploadResult | None:
    if not mode.hashing_enabled():
        return None

    stored_hash = _normalize_hash_input(hash_str)
    bare_digest = stored_hash.partition(":")[2] or stored_hash
    display_name = _sanitize_filename(
        name, fallback=bare_digest
    )

    result = _reuse_qualified_content(
        stored_hash,
        _UploadRecordSpec(
            display_name,
            tags or [],
            mime_type,
            user_metadata or {},
            preview_id,
        ),
    )
    if result is None:
        logging.warning("create_from_hash: no asset found for hash %s", hash_str)
    return result


def _preflight_cached_registration(
    locator: str,
) -> _CachedRegistrationPreflight | None:
    """Read cached facts before metadata I/O so it does not extend the writer lease."""
    with create_session() as session:
        existing = session.scalars(
            select(AssetContent).where(
                AssetContent.path == locator,
                AssetContent.is_missing.is_(False),
            )
        ).first()
        if existing is None:
            return None
        sibling = session.scalars(
            select(Asset)
            .where(Asset.content_id == existing.id)
            .order_by(Asset.created_at.asc(), Asset.id.asc())
            .limit(1)
        ).first()
        sibling_id = sibling.id if sibling is not None else None
        sibling_metadata = (
            dict(sibling.system_metadata)
            if sibling is not None and sibling.system_metadata is not None
            else None
        )
        content_id = existing.id
    signature = _file_signature(locator) if sibling_id is None else None
    return _CachedRegistrationPreflight(
        content_id,
        sibling_id,
        sibling_metadata,
        signature,
    )


def _apply_cached_registration(
    session: Session,
    preflight: _CachedRegistrationPreflight,
    name: str,
    path_tags: list[str],
    mime_type: str | None,
    job_id: str | None,
    locator: str,
    system_metadata: dict[str, Any] | None,
) -> RegisteredAsset:
    existing = session.get(AssetContent, preflight.content_id)
    if existing is None or existing.path != locator or existing.is_missing:
        raise _PreflightStale
    sibling = session.scalars(
        select(Asset)
        .where(Asset.content_id == existing.id)
        .order_by(Asset.created_at.asc(), Asset.id.asc())
        .limit(1)
    ).first()
    sibling_id = sibling.id if sibling is not None else None
    sibling_metadata = (
        dict(sibling.system_metadata)
        if sibling is not None and sibling.system_metadata is not None
        else None
    )
    if (
        sibling_id != preflight.sibling_id
        or sibling_metadata != preflight.sibling_metadata
        or (
            preflight.signature is not None
            and not _file_signature_matches(preflight.signature)
        )
    ):
        raise _PreflightStale
    record = create_record(
        session,
        existing.id,
        name,
        mime_type=mime_type,
        job_id=job_id,
        loader_path=compute_loader_path(locator),
        tags=path_tags,
        system_metadata=system_metadata,
    )
    return RegisteredAsset(
        id=record.id,
        content_id=record.content_id,
        job_id=record.job_id,
        name=record.name,
    )


def _register_cached_output_in_txn(
    session: Session,
    locator: str,
    job_id: str | None,
) -> RegisteredAsset | None:
    existing = session.scalars(
        select(AssetContent).where(
            AssetContent.path == locator,
            AssetContent.is_missing.is_(False),
        )
    ).first()
    if existing is None:
        logging.info(
            "Cached output registration is a non-event; no live content for %s",
            locator,
        )
        return None
    name, path_tags = get_name_and_tags_from_asset_path(locator)
    mime_type = mimetypes.guess_type(locator, strict=False)[0]
    sibling = session.scalars(
        select(Asset)
        .where(Asset.content_id == existing.id)
        .order_by(Asset.created_at.asc(), Asset.id.asc())
        .limit(1)
    ).first()
    system_metadata = (
        dict(sibling.system_metadata)
        if sibling is not None and sibling.system_metadata is not None
        else _extract_system_metadata_sync(locator, mime_type)
    )
    record = create_record(
        session,
        existing.id,
        name,
        mime_type=mime_type,
        job_id=job_id,
        loader_path=compute_loader_path(locator),
        tags=path_tags,
        system_metadata=system_metadata,
    )
    return RegisteredAsset(
        id=record.id,
        content_id=record.content_id,
        job_id=record.job_id,
        name=record.name,
    )


def register_cached_output(
    abs_path: str, job_id: str | None = None
) -> RegisteredAsset | None:
    locator = os.path.abspath(abs_path)
    try:
        for _restart in range(4):
            preflight = _preflight_cached_registration(locator)
            if preflight is None:
                logging.info(
                    "Cached output registration is a non-event; no live content for %s",
                    locator,
                )
                return None
            name, path_tags = get_name_and_tags_from_asset_path(locator)
            mime_type = mimetypes.guess_type(locator, strict=False)[0]
            system_metadata = preflight.sibling_metadata
            if preflight.sibling_id is None:
                system_metadata = _extract_system_metadata_sync(locator, mime_type)
            try:
                return run_write_txn(
                    lambda session: _apply_cached_registration(
                        session,
                        preflight,
                        name,
                        path_tags,
                        mime_type,
                        job_id,
                        locator,
                        system_metadata,
                    )
                )
            except _PreflightStale:
                continue
        logging.warning(
            "Cached-output preflight changed three times; falling back to in-transaction metadata extraction"
        )
        return run_write_txn(
            lambda session: _register_cached_output_in_txn(session, locator, job_id)
        )
    except Exception as exc:
        logging.exception("Failed to register cached output: %s", locator)
        emit(
            "ingest.register_failed",
            output_kind="cached",
            error_type=error_type(exc),
            job_id=job_id,
        )
        return None


def register_executed_output(
    abs_path: str, job_id: str | None = None
) -> RegisteredAsset | None:
    locator = os.path.abspath(abs_path)
    try:
        stat_result = os.stat(locator, follow_symlinks=True)
        size_bytes = stat_result.st_size
        mtime_ns = get_mtime_ns(stat_result)
        mime_type = mimetypes.guess_type(locator, strict=False)[0]
        name, path_tags = get_name_and_tags_from_asset_path(locator)
        system_metadata = _extract_system_metadata_sync(
            locator, mime_type, stat_result
        )
        def _work(session: Session) -> RegisteredAsset:
            created_content_id: str | None = None
            try:
                existing = session.scalars(
                    select(AssetContent).where(
                        AssetContent.path == locator,
                        AssetContent.is_missing.is_(False),
                    )
                ).first()
                if existing is not None:
                    mark_content_missing(session, existing.id)
                content, inserted = create_content_reporting_insert(
                    session, locator, None, size_bytes, mtime_ns
                )
                if inserted:
                    created_content_id = content.id
                record = create_record(
                    session,
                    content.id,
                    name,
                    mime_type=mime_type,
                    job_id=job_id,
                    loader_path=compute_loader_path(locator),
                    tags=path_tags,
                    system_metadata=system_metadata,
                )
            except Exception:
                session.rollback()
                if created_content_id is not None:
                    _discard_unreferenced_content(session, created_content_id)
                raise
            return RegisteredAsset(
                id=record.id,
                content_id=record.content_id,
                job_id=record.job_id,
                name=record.name,
            )

        return run_write_txn(_work)
    except Exception as exc:
        logging.exception("Failed to register executed output: %s", locator)
        emit(
            "ingest.register_failed",
            output_kind="executed",
            error_type=error_type(exc),
            job_id=job_id,
        )
        return None
