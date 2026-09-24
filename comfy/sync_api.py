"""
Synchronous run API — POST /api/sync-run accepts files + workflow, runs to completion, returns output.
"""
import asyncio
import json
import logging
import os
import tempfile
import time
import uuid



def _find_input_nodes(prompt: dict) -> list[tuple[str, str, str]]:
    """Find LoadImage/LoadVideo nodes in the workflow.
    Returns [(node_id, class_type, input_name), ...] for image/video input widgets.
    """
    image_classes = {"LoadImage", "LoadVideo", "LoadImageMask", "VHS_LoadVideo"}
    results = []
    for node_id, node_data in prompt.items():
        class_type = node_data.get("class_type", "")
        if class_type in image_classes:
            inputs = node_data.get("inputs", {})
            for key in inputs:
                if key in ("image", "video", "file", "directory"):
                    results.append((node_id, class_type, key))
    return results


def _patch_workflow_inputs(prompt: dict, file_mapping: dict[tuple[str, str], str]) -> dict:
    """Patch workflow node inputs to point to uploaded files."""
    import copy
    prompt = copy.deepcopy(prompt)
    for (node_id, input_name), filename in file_mapping.items():
        if node_id in prompt:
            prompt[node_id]["inputs"][input_name] = filename
    return prompt


async def _wait_for_prompt(prompt_queue, prompt_id: str, timeout: int = 300):
    """Wait until a prompt completes. Returns (outputs_dict, status)."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            history = prompt_queue.get_history(prompt_id=prompt_id)
            if history and prompt_id in history:
                return history[prompt_id], "completed"
        except Exception:
            pass

        try:
            queue = prompt_queue.get_current_queue()
            running = queue[0] if queue else []
            pending = queue[1] if len(queue) > 1 else []
            if prompt_id not in [str(r[1]) for r in running] and prompt_id not in [str(p[1]) for p in pending]:
                history = prompt_queue.get_history(prompt_id=prompt_id)
                if history and prompt_id in history:
                    return history[prompt_id], "completed"
        except Exception:
            pass

        await asyncio.sleep(0.5)

    return None, "timeout"


def _collect_outputs(history_entry: dict, temp_dir: str | None = None) -> dict:
    """Extract output files/flat data from a history entry."""
    outputs = history_entry.get("outputs", {})
    result = {"images": [], "videos": [], "files": [], "json": []}

    for node_id, node_output in outputs.items():
        for key in ("images", "videos", "gifs", "files"):
            for item in node_output.get(key, []):
                entry = {
                    "filename": item.get("filename", ""),
                    "subfolder": item.get("subfolder", ""),
                    "type": item.get("type", key),
                }
                result.setdefault(key if key != "gifs" else "videos", []).append(entry)

        for key, val in node_output.items():
            if key not in ("images", "videos", "gifs", "files", "audio"):
                result["json"].append({"node_id": node_id, "key": key, "value": val})

    return result


async def handle_sync_run(request, prompt_queue, output_dir, input_dir):
    """
    POST /api/sync-run

    Multipart form:
      - workflow: JSON file or string containing the workflow
      - images: one or more image/video files
      - timeout: integer seconds (default 300)
      - response_type: "files" (multipart output) or "json" (base64 in JSON)

    Returns: multipart/mixed with output files, or JSON with base64-encoded outputs.
    """
    from aiohttp import web

    reader = await request.multipart()

    workflow_data = None
    params_data: dict[str, str] = {}
    uploaded_files: list[tuple[str, bytes, str]] = []  # (field_name, data, original_filename)
    timeout_val = 300
    response_type = "files"

    async for part in reader:
        field_name = part.name
        if field_name == "workflow":
            workflow_data = await part.text()
        elif field_name == "timeout":
            timeout_val = int(await part.text())
        elif field_name == "response_type":
            response_type = (await part.text()).strip().lower()
        elif field_name == "params":
            params_data = json.loads(await part.text())
        elif field_name in ("images", "videos", "files"):
            filename = part.filename or "upload"
            data = await part.read()
            uploaded_files.append((field_name, data, filename))

    if not workflow_data:
        return web.json_response({"error": "workflow field is required"}, status=400)

    try:
        prompt = json.loads(workflow_data)
    except json.JSONDecodeError:
        return web.json_response({"error": "invalid workflow JSON"}, status=400)

    temp_dir = tempfile.mkdtemp(dir=input_dir, prefix="sync_")
    run_id = os.path.basename(temp_dir)

    file_mapping: dict[tuple[str, str], str] = {}
    file_paths: list[str] = []

    try:
        input_nodes = _find_input_nodes(prompt)

        for field_name, data, orig_filename in uploaded_files:
            safe_name = f"sync_{run_id}_{orig_filename}"
            dest_path = os.path.join(input_dir, safe_name)
            with open(dest_path, "wb") as f:
                f.write(data)
            file_paths.append(dest_path)

            if input_nodes:
                node_id, class_type, input_name = input_nodes[len(file_mapping)]
                file_mapping[(node_id, input_name)] = safe_name
                logging.info(f"[SyncAPI] Mapped {safe_name} -> node {node_id}.{input_name}")

        if file_mapping:
            prompt = _patch_workflow_inputs(prompt, file_mapping)

        # Apply params — format: {"node_id.input_name": "value"}
        if params_data:
            for key, value in params_data.items():
                parts = key.split(".", 1)
                if len(parts) == 2 and parts[0] in prompt:
                    prompt[parts[0]]["inputs"][parts[1]] = str(value)
                    logging.info(f"[SyncAPI] Param {key} = {value}")

        prompt_id = str(uuid.uuid4())
        extra_data = {"create_time": int(time.time() * 1000)}

        # Validate
        from execution import validate_prompt
        valid = await validate_prompt(prompt_id, prompt, None)

        if not valid[0]:
            error_info = valid[1]
            return web.json_response({
                "error": error_info.get("message", "validation failed"),
                "details": error_info
            }, status=400)

        outputs_to_execute = valid[2]
        prompt_queue.put((0, prompt_id, prompt, extra_data, outputs_to_execute, {}))
        logging.info(f"[SyncAPI] Submitted prompt {prompt_id}, waiting...")

        history_entry, status = await _wait_for_prompt(prompt_queue, prompt_id, timeout_val)

        if history_entry is None:
            return web.json_response({"error": f"prompt timed out after {timeout_val}s"}, status=504)

        collected = _collect_outputs(history_entry)

        if response_type == "json":
            import base64
            from folder_paths import get_output_directory
            out_dir = output_dir or get_output_directory()
            encoded_files = []
            for kind in ("images", "videos", "files"):
                for f in collected[kind]:
                    filepath = os.path.join(out_dir, f.get("subfolder", ""), f["filename"])
                    if os.path.isfile(filepath):
                        with open(filepath, "rb") as fh:
                            encoded_files.append({
                                "filename": f["filename"],
                                "type": kind,
                                "base64": base64.b64encode(fh.read()).decode("utf-8"),
                            })
            return web.json_response({"status": "completed", "outputs": encoded_files})

        # Return as multipart file response
        boundary = f"comfyui-sync-{uuid.uuid4().hex[:16]}"
        parts = []
        from folder_paths import get_output_directory
        out_dir = output_dir or get_output_directory()

        for kind in ("images", "videos", "files"):
            for f in collected[kind]:
                filepath = os.path.join(out_dir, f.get("subfolder", ""), f["filename"])
                if os.path.isfile(filepath):
                    with open(filepath, "rb") as fh:
                        content = fh.read()
                    content_type = {
                        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".webp": "image/webp", ".mp4": "video/mp4", ".gif": "image/gif",
                    }.get(os.path.splitext(f["filename"])[1].lower(), "application/octet-stream")
                    parts.append(
                        f"--{boundary}\r\n"
                        f"Content-Disposition: attachment; filename=\"{f['filename']}\"\r\n"
                        f"Content-Type: {content_type}\r\n\r\n"
                    )
                    parts.append(content)
                    parts.append(b"\r\n")

        if not parts:
            return web.json_response({"error": "no output files generated"}, status=500)

        parts.append(f"--{boundary}--\r\n".encode())
        body = b""
        for p in parts:
            body += p if isinstance(p, bytes) else p.encode()

        resp = web.Response(
            body=body,
            content_type=f"multipart/mixed; boundary={boundary}",
        )
        return resp

    finally:
        for fp in file_paths:
            try:
                os.remove(fp)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass
