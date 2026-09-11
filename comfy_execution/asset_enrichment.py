"""Enrich executed-node output entries with asset id."""
import logging
import os


def enrich_output_with_assets(output_ui: dict) -> dict:
    """Register file-type output entries as assets and inject their ``id``.

    Runs at output-processing time, once per produced output. Returns a new
    dict; entries without a resolvable on-disk file path are left unchanged.
    Errors are caught per-entry so a failure never blocks execution or the
    other entries.
    """
    import folder_paths
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
                base = folder_paths.get_directory_by_type(entry["type"])
                if base is None:
                    new_entries.append(entry)
                    continue
                base_abs = os.path.abspath(base)
                abs_path = os.path.abspath(os.path.join(base_abs, entry.get("subfolder") or "", entry["filename"]))
                try:
                    if os.path.commonpath([base_abs, abs_path]) != base_abs:
                        raise ValueError("escapes base")
                except ValueError:
                    logging.warning("Asset enrichment skipped (path escapes base): %s", entry.get("filename"))
                    new_entries.append(entry)
                    continue
                if not os.path.isfile(abs_path):
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
