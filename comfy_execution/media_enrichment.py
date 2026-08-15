"""Enrich executed-node output entries with media metadata.

Attaches a ``metadata`` object (``kind``/``width``/``height``, plus
``duration``/``fps``/``frame_count`` for videos) to each file-type output
entry at output-processing time, so consumers of the ``executed`` message
and ``/history`` can read media properties without probing the files
themselves. Unlike asset enrichment this is NOT gated on ``--enable-assets``:
the properties serve history/websocket consumers that don't run the assets
system at all.
"""
import logging
import mimetypes
import os

import folder_paths


def resolve_output_entry_path(entry) -> str | None:
    """Resolve a file-type output entry to an absolute path inside its base dir.

    Returns ``None`` for non-file entries (no ``filename``/``type``, or
    non-string fields), unknown folder types, paths that escape the
    folder-type base directory (symlinks are resolved before the containment
    check), and paths with no file on disk.
    """
    if not isinstance(entry, dict) or "filename" not in entry or "type" not in entry:
        return None
    filename = entry["filename"]
    subfolder = entry.get("subfolder", "")
    if not isinstance(filename, str) or not isinstance(subfolder, str) or not isinstance(entry["type"], str):
        return None
    base = folder_paths.get_directory_by_type(entry["type"])
    if base is None:
        return None
    # realpath (not abspath) so a symlink planted inside the base directory
    # can't smuggle an outside target past the containment check; the base is
    # resolved too since output/temp dirs are themselves commonly symlinks.
    base_real = os.path.realpath(base)
    real_path = os.path.realpath(os.path.join(base_real, subfolder, filename))
    try:
        if os.path.commonpath([base_real, real_path]) != base_real:
            logging.warning("Output entry path escapes base directory: %s", filename)
            return None
    except ValueError:
        logging.warning("Output entry path escapes base directory: %s", filename)
        return None
    if not os.path.isfile(real_path):
        return None
    return real_path


def enrich_output_with_media_metadata(output_ui: dict) -> dict:
    """Attach a ``metadata`` media-properties object to file output entries.

    Runs once per produced output at output-processing time. Returns a new
    dict; entries are copied before modification, and an entry that already
    carries a ``metadata`` key is left untouched. Best-effort: unreadable
    files, non-media types, and per-entry errors leave that entry unchanged
    and never block execution.
    """
    try:
        from app.assets.services.media_metadata import extract_media_metadata
    except Exception:
        # Broad on purpose: this enrichment is best-effort and is called
        # unguarded from the output-processing path, so a dependency failing
        # to import for any reason must degrade to a no-op, not block outputs.
        logging.warning("Media metadata extraction unavailable; skipping enrichment", exc_info=True)
        return output_ui

    enriched = {}
    for key, entries in output_ui.items():
        if not isinstance(entries, list):
            enriched[key] = entries
            continue
        new_entries = []
        for entry in entries:
            try:
                if isinstance(entry, dict) and "metadata" not in entry:
                    abs_path = resolve_output_entry_path(entry)
                    if abs_path is not None:
                        mime_type = mimetypes.guess_type(entry["filename"], strict=False)[0]
                        media = extract_media_metadata(abs_path, mime_type=mime_type)
                        if media:
                            entry = dict(entry)
                            entry["metadata"] = media
            except Exception:
                filename = entry.get("filename") if isinstance(entry, dict) else None
                logging.warning(
                    "Failed to enrich output entry with media metadata: %s",
                    filename, exc_info=True,
                )
            new_entries.append(entry)
        enriched[key] = new_entries
    return enriched
