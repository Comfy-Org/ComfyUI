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
            except Exception:
                logging.warning("Asset registration failed for output: %s", entry.get("filename"), exc_info=True)


def _strip_ids(output_ui: dict) -> None:
    for entries in output_ui.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                entry.pop("id", None)


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
