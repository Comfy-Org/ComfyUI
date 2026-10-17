"""Registers the files a prompt produced as assets, at the moment the execution
result is emitted. Each output is matched to a path inside its declared base
directory and skipped when it escapes that directory or is not on disk, and a
run served from cache replays the same registration so cached results still
yield assets. Registering at emission rather than by inspecting the cache
afterwards keeps this independent of any cache eviction policy. With assets on
or off, each emitted file entry also gets a ``version`` for clients' /view URLs.
"""

import contextlib
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
    register: Callable[[str, str | None], "RegisteredAsset | None"] | None,
) -> None:
    """Register each file entry as an asset (when ``register`` is given) and stamp it with a
    ``version`` that changes with the file's mtime or size.

    S10.6: producers that write the same output path are not coalesced (unsupported)."""
    for entries in output_ui.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or "filename" not in entry or "type" not in entry:
                continue
            entry.pop("version", None)
            try:
                abs_path = _resolve_output_path(entry)
                if abs_path is None:
                    continue
                with contextlib.suppress(OSError):
                    st = os.stat(abs_path)
                    entry["version"] = f"{st.st_mtime_ns:x}-{st.st_size:x}"
                if register is not None and (result := register(abs_path, job_id)) is not None:
                    entry["id"] = result.id
                    if result.preview_id:
                        entry["preview_id"] = result.preview_id
            except Exception:
                logging.warning("Output enrichment failed for output: %s", entry.get("filename"), exc_info=True)


def take_asset_previews(output_ui: dict) -> dict[str, dict]:
    """Pop the previews save nodes named for their outputs, by output path, so none reaches clients or the cache."""
    previews: dict[str, dict] = {}
    for entries in output_ui.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or "asset_preview" not in entry:
                continue
            preview = entry.pop("asset_preview")
            try:
                abs_path = _resolve_output_path(entry)
            except Exception:
                continue
            if abs_path is not None:
                previews[abs_path] = preview
    return previews


def _strip_ids(output_ui: dict) -> None:
    for entries in output_ui.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                entry.pop("id", None)
                entry.pop("preview_id", None)


def register_executed_outputs(output_ui: dict, job_id: str, asset_manager: "AssetManager", asset_previews: dict[str, dict] | None = None) -> dict:
    enriched = copy.deepcopy(output_ui)
    previews = asset_previews or {}
    _enrich_in_place(
        enriched,
        job_id,
        (lambda path, job: asset_manager.register_executed_output(path, job, preview_ref=previews.get(path)))
        if asset_manager.enabled
        else None,
    )
    return enriched


def register_cached_outputs(ui_wrapper: dict | None, job_id: str, asset_manager: "AssetManager") -> dict | None:
    if ui_wrapper is None:
        return None

    enriched = copy.deepcopy(ui_wrapper)
    output_ui = enriched.get("output")
    if not isinstance(output_ui, dict):
        return enriched
    _strip_ids(output_ui)
    _enrich_in_place(output_ui, job_id, asset_manager.register_cached_output if asset_manager.enabled else None)
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
