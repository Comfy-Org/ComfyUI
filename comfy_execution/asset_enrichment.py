"""Enrich executed-node output entries with asset id."""
import logging

from comfy_execution.media_enrichment import resolve_output_entry_path


def enrich_output_with_assets(output_ui: dict) -> dict:
    """Register file-type output entries as assets and inject their ``id``.

    Runs at output-processing time, once per produced output, when
    --enable-assets is set. Returns a new dict; entries without a resolvable
    on-disk file path are left unchanged. Errors are caught per-entry so a
    failure never blocks execution or the other entries.
    """
    from comfy.cli_args import args
    if not args.enable_assets:
        return output_ui

    from app.assets.services.ingest import register_file_in_place, DependencyMissingError

    enriched = {}
    for key, entries in output_ui.items():
        if not isinstance(entries, list):
            enriched[key] = entries
            continue
        new_entries = []
        for entry in entries:
            if not isinstance(entry, dict) or "filename" not in entry or "type" not in entry:
                new_entries.append(entry)
                continue
            try:
                abs_path = resolve_output_entry_path(entry)
                if abs_path is None:
                    new_entries.append(entry)
                    continue

                # Register unconditionally: the file was just produced, and
                # register_file_in_place re-hashes so an overwritten path can
                # never carry a stale id.
                result = register_file_in_place(
                    abs_path=abs_path,
                    name=entry["filename"],
                    tags=[entry["type"]],
                )

                entry = dict(entry)
                entry["id"] = result.ref.id
            except DependencyMissingError:
                logging.warning("Asset enrichment skipped (blake3 not available): %s", entry.get("filename"))
            except Exception:
                logging.warning("Failed to enrich output entry with asset id: %s", entry.get("filename"), exc_info=True)
            new_entries.append(entry)
        enriched[key] = new_entries
    return enriched
