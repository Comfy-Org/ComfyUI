"""Semantics step 1: re-derive the reference state that comes from the path.

Recomputes ``loader_path``, the backend tags a path implies, and ``is_missing``.

It must keep reading no file contents, so an untouched model library costs one
``stat`` per file. That rules out repairing anything derived from content --
``hash``, ``size_bytes``, mime -- so a file that has changed underneath its row
goes to the existing ``needs_verify`` path instead.

It must also leave alone anything a person chose (manual and upload-origin tags,
``user_metadata``, ``preview_id``, ``deleted_at``, ``job_id``, ``name``),
references whose file is gone, and references whose path is under no root this
install currently knows about -- a misconfigured ``extra_model_paths.yaml`` must
not be able to strip tags off assets that are merely out of view.
"""

import logging
import os
from dataclasses import dataclass

from app.assets.database.queries import (
    bulk_update_is_missing,
    bulk_update_needs_verify,
)
from app.assets.database.queries.semantics import (
    AUTOMATIC_TAG_ORIGIN,
    DerivedStateRow,
    bulk_add_automatic_tags,
    bulk_remove_automatic_tags,
    bulk_set_loader_paths,
    get_file_backed_references_page,
    get_tag_origins_by_reference,
)
from app.assets.helpers import normalize_tags
from app.assets.semantics.step import (
    InterruptCheck,
    SemanticsStepInterrupted,
    StepResult,
)
from app.assets.services.file_utils import verify_file_unchanged
from app.assets.services.path_utils import (
    compute_loader_path,
    get_path_derived_tag_vocabulary,
    get_path_derived_tags_from_path,
)
from app.database.db import create_session

_ROWS_PER_TRANSACTION = 500


@dataclass
class ReprojectionSummary(StepResult):

    scanned: int = 0
    unchanged_files: int = 0
    changed_files: int = 0
    absent_files: int = 0
    unclassified_paths: int = 0
    loader_paths_rewritten: int = 0
    tags_added: int = 0
    tags_removed: int = 0
    missing_flags_cleared: int = 0
    verify_flags_set: int = 0

    @property
    def complete(self) -> bool:
        """Nothing else re-derives these rows -- the scan only builds specs for paths it has not seen -- so staying unstamped is what lets a restored root repair them."""
        return self.unclassified_paths == 0

    def __str__(self) -> str:
        return (
            f"scanned={self.scanned} unchanged={self.unchanged_files} "
            f"changed={self.changed_files} absent={self.absent_files} "
            f"unclassified={self.unclassified_paths} "
            f"loader_paths={self.loader_paths_rewritten} "
            f"tags+{self.tags_added}/-{self.tags_removed} "
            f"unflagged_missing={self.missing_flags_cleared} "
            f"flagged_verify={self.verify_flags_set}"
        )


def reproject_derived_state(
    interrupt_check: InterruptCheck | None = None,
) -> ReprojectionSummary:
    """Each batch commits on its own, so a kill mid-walk is safe to resume."""
    summary = ReprojectionSummary()
    vocabulary = get_path_derived_tag_vocabulary()
    after_id: str | None = None

    while True:
        if interrupt_check is not None and interrupt_check():
            raise SemanticsStepInterrupted(
                f"interrupted after {summary.scanned} references"
            )

        with create_session() as session:
            rows = get_file_backed_references_page(
                session, after_id=after_id, limit=_ROWS_PER_TRANSACTION
            )
            if not rows:
                return summary
            after_id = rows[-1].reference_id
            _reproject_batch(session, rows, vocabulary, summary)
            session.commit()


def _reproject_batch(
    session,
    rows: list[DerivedStateRow],
    vocabulary: set[str],
    summary: ReprojectionSummary,
) -> None:
    stored_tags = get_tag_origins_by_reference(
        session, [row.reference_id for row in rows]
    )

    loader_paths: dict[str, str | None] = {}
    tags_to_add: list[tuple[str, str]] = []
    tags_to_remove: list[tuple[str, str]] = []
    clear_missing: list[str] = []
    set_needs_verify: list[str] = []

    for row in rows:
        summary.scanned += 1
        present, unchanged = _classify_file(row)

        if not present:
            # The scanner owns retiring these, under its own missing semantics.
            summary.absent_files += 1
            continue

        try:
            derived_tags = set(
                normalize_tags(get_path_derived_tags_from_path(row.file_path))
            )
        except ValueError:
            summary.unclassified_paths += 1
            continue

        if unchanged:
            summary.unchanged_files += 1
        else:
            # Hand it to the verify path rather than re-read the file for hash and size.
            summary.changed_files += 1
            if not row.needs_verify:
                set_needs_verify.append(row.reference_id)

        loader_path = compute_loader_path(row.file_path)
        if loader_path != row.loader_path:
            loader_paths[row.reference_id] = loader_path
            summary.loader_paths_rewritten += 1

        current_tags = stored_tags.get(row.reference_id, {})
        for tag_name in sorted(derived_tags - set(current_tags)):
            tags_to_add.append((row.reference_id, tag_name))
        for tag_name, origin in sorted(current_tags.items()):
            if (
                origin == AUTOMATIC_TAG_ORIGIN
                and tag_name in vocabulary
                and tag_name not in derived_tags
            ):
                tags_to_remove.append((row.reference_id, tag_name))

        if row.is_missing:
            # Nothing else un-flags this: the scanner excludes already-missing
            # references from its temp reconciliation.
            clear_missing.append(row.reference_id)

    bulk_set_loader_paths(session, loader_paths)
    bulk_add_automatic_tags(session, tags_to_add)
    bulk_remove_automatic_tags(session, tags_to_remove)
    bulk_update_is_missing(session, clear_missing, value=False)
    bulk_update_needs_verify(session, set_needs_verify, value=True)

    summary.tags_added += len(tags_to_add)
    summary.tags_removed += len(tags_to_remove)
    summary.missing_flags_cleared += len(clear_missing)
    summary.verify_flags_set += len(set_needs_verify)


def _classify_file(row: DerivedStateRow) -> tuple[bool, bool]:
    """Mirrors the scanner's stat handling: permission denied means present-but-unreadable, any other OS error means gone."""
    try:
        stat_result = os.stat(row.file_path, follow_symlinks=True)
    except FileNotFoundError:
        return False, False
    except PermissionError:
        logging.debug("Permission denied accessing %s", row.file_path)
        return True, False
    except OSError as error:
        logging.debug("OSError checking %s: %s", row.file_path, error)
        return False, False

    return True, verify_file_unchanged(
        mtime_db=row.mtime_ns,
        size_db=row.size_bytes,
        stat_result=stat_result,
    )
