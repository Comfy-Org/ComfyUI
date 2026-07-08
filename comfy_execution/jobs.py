"""
Job utilities for the /api/jobs endpoint.
Provides normalization and helper functions for job status tracking.
"""

import uuid
from typing import Callable, Optional

from comfy_api.internal import prune_dict


# Result of classifying a job for cancellation.
# 'running'  -> job is currently executing (interrupt it)
# 'pending'  -> job is queued but not started (dequeue it)
# 'terminal' -> job already finished (present in history); cancel is a no-op
# 'unknown'  -> job id is not present anywhere
CANCEL_RUNNING = 'running'
CANCEL_PENDING = 'pending'
CANCEL_TERMINAL = 'terminal'
CANCEL_UNKNOWN = 'unknown'


class JobStatus:
    """Job status constants."""
    PENDING = 'pending'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'

    ALL = [PENDING, IN_PROGRESS, COMPLETED, FAILED, CANCELLED]


def validate_job_id(value) -> str:
    """Validate a client-supplied job (prompt) id.

    Job ids must be UUIDs in the canonical lowercase hyphenated form. The id
    is stored and compared verbatim everywhere downstream — history keys,
    websocket events, and /interrupt matching — so accepting another spelling
    would silently rewrite the client's id and then miss every exact-match
    lookup. Rejecting loudly beats that.

    Returns the id unchanged. Raises ValueError when the value is not a
    string in canonical UUID form.
    """
    if not isinstance(value, str):
        raise ValueError(f"job id must be a string, got {type(value).__name__}")
    if str(uuid.UUID(value)) != value:
        raise ValueError("job id must be a UUID in canonical lowercase hyphenated form")
    return value


# Media types that can be previewed in the frontend
PREVIEWABLE_MEDIA_TYPES = frozenset({'images', 'video', 'audio', '3d', 'text'})

# 3D file extensions for preview fallback (no dedicated media_type exists)
THREE_D_EXTENSIONS = frozenset({'.obj', '.fbx', '.gltf', '.glb', '.usdz'})


def has_3d_extension(filename: str) -> bool:
    lower = filename.lower()
    return any(lower.endswith(ext) for ext in THREE_D_EXTENSIONS)


def normalize_output_item(item):
    """Normalize a single output list item for the jobs API.

    Returns the normalized item, or None to exclude it.
    String items with 3D extensions become {filename, type, subfolder} dicts.
    """
    if item is None:
        return None
    if isinstance(item, str):
        if has_3d_extension(item):
            return {'filename': item, 'type': 'output', 'subfolder': '', 'mediaType': '3d'}
        return None
    if isinstance(item, dict):
        return item
    return None


def normalize_outputs(outputs: dict) -> dict:
    """Normalize raw node outputs for the jobs API.

    Transforms string 3D filenames into file output dicts and removes
    None items. All other items (non-3D strings, dicts, etc.) are
    preserved as-is.
    """
    normalized = {}
    for node_id, node_outputs in outputs.items():
        if not isinstance(node_outputs, dict):
            normalized[node_id] = node_outputs
            continue
        normalized_node = {}
        for media_type, items in node_outputs.items():
            if media_type == 'animated' or not isinstance(items, list):
                normalized_node[media_type] = items
                continue
            normalized_items = []
            for item in items:
                if item is None:
                    continue
                norm = normalize_output_item(item)
                normalized_items.append(norm if norm is not None else item)
            normalized_node[media_type] = normalized_items
        normalized[node_id] = normalized_node
    return normalized

# Text preview truncation limit (1024 characters) to prevent preview_output bloat
TEXT_PREVIEW_MAX_LENGTH = 1024


def _create_text_preview(value: str) -> dict:
    """Create a text preview dict with optional truncation.

    Returns:
        dict with 'content' and optionally 'truncated' flag
    """
    if len(value) <= TEXT_PREVIEW_MAX_LENGTH:
        return {'content': value}
    return {
        'content': value[:TEXT_PREVIEW_MAX_LENGTH],
        'truncated': True
    }


def _extract_job_metadata(extra_data: dict) -> tuple[Optional[int], Optional[str]]:
    """Extract create_time and workflow_id from extra_data.

    Returns:
        tuple: (create_time, workflow_id)
    """
    create_time = extra_data.get('create_time')
    extra_pnginfo = extra_data.get('extra_pnginfo', {})
    workflow_id = extra_pnginfo.get('workflow', {}).get('id')
    return create_time, workflow_id


def is_previewable(media_type: str, item: dict) -> bool:
    """
    Check if an output item is previewable.
    Matches frontend logic in ComfyUI_frontend/src/stores/queueStore.ts
    Maintains backwards compatibility with existing logic.

    Priority:
    1. media_type is 'images', 'video', 'audio', or '3d'
    2. format field starts with 'video/' or 'audio/'
    3. filename has a 3D extension (.obj, .fbx, .gltf, .glb, .usdz)
    """
    if media_type in PREVIEWABLE_MEDIA_TYPES:
        return True

    # Check format field (MIME type).
    # Maintains backwards compatibility with how custom node outputs are handled in the frontend.
    fmt = item.get('format', '')
    if fmt and (fmt.startswith('video/') or fmt.startswith('audio/')):
        return True

    # Check for 3D files by extension
    filename = item.get('filename', '').lower()
    if any(filename.endswith(ext) for ext in THREE_D_EXTENSIONS):
        return True

    return False


def normalize_queue_item(item: tuple, status: str) -> dict:
    """Convert queue item tuple to unified job dict.

    Expects item with sensitive data already removed (5 elements).
    """
    priority, prompt_id, _, extra_data, _ = item
    create_time, workflow_id = _extract_job_metadata(extra_data)

    return prune_dict({
        'id': prompt_id,
        'status': status,
        'priority': priority,
        'create_time': create_time,
        'outputs_count': 0,
        'workflow_id': workflow_id,
    })


def normalize_history_item(prompt_id: str, history_item: dict, include_outputs: bool = False) -> dict:
    """Convert history item dict to unified job dict.

    History items have sensitive data already removed (prompt tuple has 5 elements).
    """
    prompt_tuple = history_item['prompt']
    priority, _, prompt, extra_data, _ = prompt_tuple
    create_time, workflow_id = _extract_job_metadata(extra_data)

    status_info = history_item.get('status', {})
    status_str = status_info.get('status_str') if status_info else None

    outputs = history_item.get('outputs', {})
    outputs_count, preview_output = get_outputs_summary(outputs)

    execution_error = None
    execution_start_time = None
    execution_end_time = None
    was_interrupted = False
    if status_info:
        messages = status_info.get('messages', [])
        for entry in messages:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                event_name, event_data = entry[0], entry[1]
                if isinstance(event_data, dict):
                    if event_name == 'execution_start':
                        execution_start_time = event_data.get('timestamp')
                    elif event_name in ('execution_success', 'execution_error', 'execution_interrupted'):
                        execution_end_time = event_data.get('timestamp')
                        if event_name == 'execution_error':
                            execution_error = event_data
                        elif event_name == 'execution_interrupted':
                            was_interrupted = True

    if status_str == 'success':
        status = JobStatus.COMPLETED
    elif status_str == 'error':
        status = JobStatus.CANCELLED if was_interrupted else JobStatus.FAILED
    else:
        status = JobStatus.COMPLETED

    job = prune_dict({
        'id': prompt_id,
        'status': status,
        'priority': priority,
        'create_time': create_time,
        'execution_start_time': execution_start_time,
        'execution_end_time': execution_end_time,
        'execution_error': execution_error,
        'outputs_count': outputs_count,
        'preview_output': preview_output,
        'workflow_id': workflow_id,
    })

    if include_outputs:
        job['outputs'] = normalize_outputs(outputs)
        job['execution_status'] = status_info
        job['workflow'] = {
            'prompt': prompt,
            'extra_data': extra_data,
        }

    return job


def get_outputs_summary(outputs: dict) -> tuple[int, Optional[dict]]:
    """
    Count outputs and find preview in a single pass.
    Returns (outputs_count, preview_output).

    Preview priority (matching frontend):
    1. type="output" with previewable media
    2. Any previewable media
    """
    count = 0
    preview_output = None
    fallback_preview = None

    for node_id, node_outputs in outputs.items():
        if not isinstance(node_outputs, dict):
            continue
        for media_type, items in node_outputs.items():
            # 'animated' is a boolean flag, not actual output items
            if media_type == 'animated' or not isinstance(items, list):
                continue

            for item in items:
                if not isinstance(item, dict):
                    # Handle text outputs (non-dict items like strings or tuples)
                    normalized = normalize_output_item(item)
                    if normalized is None:
                        # Not a 3D file string — check for text preview
                        if media_type == 'text':
                            count += 1
                            if preview_output is None:
                                if isinstance(item, tuple):
                                    text_value = item[0] if item else ''
                                else:
                                    text_value = str(item)
                                text_preview = _create_text_preview(text_value)
                                enriched = {
                                    **text_preview,
                                    'nodeId': node_id,
                                    'mediaType': media_type
                                }
                                if fallback_preview is None:
                                    fallback_preview = enriched
                        continue
                    # normalize_output_item returned a dict (e.g. 3D file)
                    item = normalized

                count += 1

                if preview_output is not None:
                    continue

                if is_previewable(media_type, item):
                    enriched = {
                        **item,
                        'nodeId': node_id,
                    }
                    if 'mediaType' not in item:
                        enriched['mediaType'] = media_type
                    if item.get('type') == 'output':
                        preview_output = enriched
                    elif fallback_preview is None:
                        fallback_preview = enriched

    return count, preview_output or fallback_preview


def apply_sorting(jobs: list[dict], sort_by: str, sort_order: str) -> list[dict]:
    """Sort jobs list by specified field and order."""
    reverse = (sort_order == 'desc')

    if sort_by == 'execution_duration':
        def get_sort_key(job):
            start = job.get('execution_start_time', 0)
            end = job.get('execution_end_time', 0)
            return end - start if end and start else 0
    else:
        def get_sort_key(job):
            return job.get('create_time', 0)

    return sorted(jobs, key=get_sort_key, reverse=reverse)


def get_job(prompt_id: str, running: list, queued: list, history: dict) -> Optional[dict]:
    """
    Get a single job by prompt_id from history or queue.

    Args:
        prompt_id: The prompt ID to look up
        running: List of currently running queue items
        queued: List of pending queue items
        history: Dict of history items keyed by prompt_id

    Returns:
        Job dict with full details, or None if not found
    """
    if prompt_id in history:
        return normalize_history_item(prompt_id, history[prompt_id], include_outputs=True)

    for item in running:
        if item[1] == prompt_id:
            return normalize_queue_item(item, JobStatus.IN_PROGRESS)

    for item in queued:
        if item[1] == prompt_id:
            return normalize_queue_item(item, JobStatus.PENDING)

    return None


def get_all_jobs(
    running: list,
    queued: list,
    history: dict,
    status_filter: Optional[list[str]] = None,
    workflow_id: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    limit: Optional[int] = None,
    offset: int = 0
) -> tuple[list[dict], int]:
    """
    Get all jobs (running, pending, completed) with filtering and sorting.

    Args:
        running: List of currently running queue items
        queued: List of pending queue items
        history: Dict of history items keyed by prompt_id
        status_filter: List of statuses to include (from JobStatus.ALL)
        workflow_id: Filter by workflow ID
        sort_by: Field to sort by ('created_at', 'execution_duration')
        sort_order: 'asc' or 'desc'
        limit: Maximum number of items to return
        offset: Number of items to skip

    Returns:
        tuple: (jobs_list, total_count)
    """
    jobs = []

    if status_filter is None:
        status_filter = JobStatus.ALL

    if JobStatus.IN_PROGRESS in status_filter:
        for item in running:
            jobs.append(normalize_queue_item(item, JobStatus.IN_PROGRESS))

    if JobStatus.PENDING in status_filter:
        for item in queued:
            jobs.append(normalize_queue_item(item, JobStatus.PENDING))

    history_statuses = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
    requested_history_statuses = history_statuses & set(status_filter)
    if requested_history_statuses:
        for prompt_id, history_item in history.items():
            job = normalize_history_item(prompt_id, history_item)
            if job.get('status') in requested_history_statuses:
                jobs.append(job)

    if workflow_id:
        jobs = [j for j in jobs if j.get('workflow_id') == workflow_id]

    jobs = apply_sorting(jobs, sort_by, sort_order)

    total_count = len(jobs)

    if offset > 0:
        jobs = jobs[offset:]
    if limit is not None:
        jobs = jobs[:limit]

    return (jobs, total_count)


def classify_job_for_cancel(prompt_id: str, running: list, queued: list, history: dict) -> str:
    """Classify a job id for cancellation.

    Returns one of CANCEL_RUNNING, CANCEL_PENDING, CANCEL_TERMINAL, CANCEL_UNKNOWN.

    Queue items are tuples whose second element (index 1) is the prompt_id.
    History is a dict keyed by prompt_id, so a job present there has already
    finished and cancelling it is a no-op.
    """
    for item in running:
        if item[1] == prompt_id:
            return CANCEL_RUNNING
    for item in queued:
        if item[1] == prompt_id:
            return CANCEL_PENDING
    if prompt_id in history:
        return CANCEL_TERMINAL
    return CANCEL_UNKNOWN


def cancel_job(
    prompt_id: str,
    running: list,
    queued: list,
    history: dict,
    interrupt: Callable[[str], bool],
    dequeue: Callable[[str], bool],
) -> str:
    """Cancel a single job by id, regardless of state.

    Maps the cancel onto the runtime's existing mechanics:
      - a running job is interrupted via ``interrupt``
      - a pending job is removed from the queue via ``dequeue``
      - a job that already finished (terminal) is a no-op
      - an unknown id is a no-op (callers that need fail-fast behaviour should
        validate ids up front with ``classify_job_for_cancel``)

    Both ``interrupt`` and ``dequeue`` take the prompt id and return whether
    they acted on a job that was *actually* in that state, so the value returned
    here reflects what truly happened rather than the (possibly stale)
    classification. This matters around the narrow TOCTOU windows where a job
    changes state between the caller's snapshot and the action:

      - a job classified RUNNING may have finished before ``interrupt`` fires:
        ``interrupt`` returns False and this returns CANCEL_UNKNOWN (no-op).
      - a job classified PENDING may have started executing before ``dequeue``
        fires: ``dequeue`` returns False, ``interrupt`` then catches the now-
        running job and this returns CANCEL_RUNNING. If it had simply finished
        instead, both return False and this returns CANCEL_UNKNOWN.

    ``interrupt`` must be atomic — interrupt the job only if it is still the one
    running — so a cancel can never land on an unrelated prompt that started in
    the meantime (see ``execution.PromptQueue.interrupt_if_running``).
    """
    classification = classify_job_for_cancel(prompt_id, running, queued, history)
    if classification == CANCEL_RUNNING:
        return CANCEL_RUNNING if interrupt(prompt_id) else CANCEL_UNKNOWN
    if classification == CANCEL_PENDING:
        if dequeue(prompt_id):
            return CANCEL_PENDING
        # Left the pending queue between classification and dequeue: if it
        # started executing, interrupt the now-running job; otherwise it has
        # already finished and the cancel is a genuine no-op.
        return CANCEL_RUNNING if interrupt(prompt_id) else CANCEL_UNKNOWN
    # CANCEL_TERMINAL and CANCEL_UNKNOWN are intentional no-ops.
    return classification
