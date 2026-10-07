"""Registers the files a prompt produced as assets, at the moment the execution
result is emitted. Each output is matched to a path inside its declared base
directory and skipped when it escapes that directory or is not on disk, and a
run served from cache replays the same registration so cached results still
yield assets. Registering at emission rather than by inspecting the cache
afterwards keeps this independent of any cache eviction policy.
"""

import copy
import logging
import os
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from app.assets.manager import AssetManager
    from app.assets.services.schemas import RegisteredAsset
    from comfy_execution.server_protocol import ExecutionServer
    from execution import CacheEntry


def _resolve_output_path(entry: dict) -> str | None:
    """Resolve an output entry to an absolute, in-base, on-disk file path.

    Returns ``None`` (skip, no registration) when the type is unknown, the
    resolved path escapes its base directory, or the file does not exist.
    """
    import folder_paths

    base = folder_paths.get_directory_by_type(entry["type"])
    if base is None:
        return None
    base_abs = os.path.abspath(base)
    abs_path = os.path.abspath(os.path.join(base_abs, entry.get("subfolder") or "", entry["filename"]))
    try:
        if os.path.commonpath([base_abs, abs_path]) != base_abs:
            return None
    except ValueError:
        return None
    if not os.path.isfile(abs_path):
        return None
    return abs_path


def _enrich_in_place(
    output_ui: dict,
    job_id: str | None,
    register: Callable[[str, str | None], "RegisteredAsset | None"],
) -> None:
    """S10.6: producers that write the same output path are not coalesced (unsupported)."""
    for entries in output_ui.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or "filename" not in entry or "type" not in entry:
                continue
            try:
                abs_path = _resolve_output_path(entry)
                if abs_path is None:
                    continue
                result = register(abs_path, job_id)
                if result is not None:
                    entry["id"] = result.id
                    if result.preview_id:
                        entry["preview_id"] = result.preview_id
                    elif _is_own_image_preview(abs_path):
                        entry["preview_id"] = result.id
            except Exception:
                logging.warning("Asset registration failed for output: %s", entry.get("filename"), exc_info=True)


def _is_own_image_preview(abs_path: str) -> bool:
    from app.assets.services.preview_rules import own_preview_kind

    return own_preview_kind(None, abs_path) == "image"


async def generate_output_previews(output_ui: dict) -> None:
    """Make previews for registered entries that need one; a cached replay never does."""
    from app.assets.previews import generate_previews, has_preview_generator

    pending: list[tuple[dict, str]] = []
    for entries in output_ui.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or "id" not in entry:
                continue
            # A linked preview stays; an entry that is only its own preview can get a better one.
            if entry.get("preview_id") not in (None, entry["id"]):
                continue
            try:
                abs_path = _resolve_output_path(entry)
            except Exception:  # not a file entry, e.g. a custom node's own {"id": ...} rows
                continue
            if abs_path is not None and has_preview_generator(abs_path):
                pending.append((entry, abs_path))
    if not pending:
        return
    try:
        linked = await generate_previews([(entry["id"], path) for entry, path in pending], "output")
    except Exception:
        logging.warning("Preview generation failed for outputs", exc_info=True)
        return
    for entry, _ in pending:
        if entry["id"] in linked:
            entry["preview_id"] = linked[entry["id"]]


def _strip_ids(output_ui: dict) -> None:
    for entries in output_ui.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                entry.pop("id", None)
                entry.pop("preview_id", None)


def register_executed_outputs(output_ui: dict, job_id: str, asset_manager: "AssetManager") -> dict:
    enriched = copy.deepcopy(output_ui)
    if not asset_manager.enabled:
        return enriched

    _enrich_in_place(enriched, job_id, asset_manager.register_executed_output)
    return enriched


def register_cached_outputs(ui_wrapper: dict | None, job_id: str, asset_manager: "AssetManager") -> dict | None:
    if ui_wrapper is None:
        return None

    enriched = copy.deepcopy(ui_wrapper)
    output_ui = enriched.get("output")
    if not isinstance(output_ui, dict):
        return enriched
    _strip_ids(output_ui)

    if not asset_manager.enabled:
        return enriched

    _enrich_in_place(output_ui, job_id, asset_manager.register_cached_output)
    return enriched


def emit_cached_output(server: "ExecutionServer", node_id: str, display_node_id: str, cached: "CacheEntry", prompt_id: str, ui_outputs: dict, asset_manager: "AssetManager") -> None:
    if node_id in ui_outputs:
        return
    enriched = register_cached_outputs(cached.ui, prompt_id, asset_manager)
    if enriched is not None:
        ui_outputs[node_id] = enriched
    if server.client_id is None:
        return
    output = enriched.get("output") if enriched is not None else None
    server.send_sync(
        "executed",
        {"node": node_id, "display_node": display_node_id, "output": output, "prompt_id": prompt_id},
        server.client_id,
    )
