"""Comfy API v2 (``/api/v2/*``), served natively; the contract is ``openapi-v2.yaml``.

Registered only when the assets system is enabled. ``server`` is duck-typed: v2 uses
its ``prompt_queue``, ``user_manager``, ``asset_manager``, ``enqueue_prompt``,
``cancel_job_by_id``, ``sid_listeners`` and ``sockets_metadata``.

``openapi-v2.yaml`` is never edited here; it arrives through the cloud sync PR. The
coverage test ``tests-unit/api_v2_test/test_contract_coverage.py`` fails on any contract
operation that has no route here.
"""

from aiohttp import web

from .jobs import JOBS_KEY, JobStore
from .routes import ROUTES, V2_SERVER_KEY

# Routes served beyond the contract's operations. Every v2 route is a contract operation.
ROUTE_EXTENSIONS: dict[tuple[str, str], str] = {}

# Additive fields and query parameters kept for comfy-api-proxy compatibility.
FIELD_EXTENSIONS: dict[str, str] = {
    "priority": "Job field: the advisory priority the submitter gave, shown only to the submitter.",
    "outputs_reused": "Job field: outputs ComfyUI served from its cache instead of executing.",
    "status": "listJobs query parameter: filter by job status.",
    "truncated": "listJobs response field: always false, listing pages with cursors instead.",
    "create_time": "listJobs item field: items are full Job objects plus their creation time.",
    "update_time": "listJobs item field: items are full Job objects plus their last update time.",
    "core/ASSET": "Workflow asset references may carry only info.hash, without info.id.",
}


def register(app: web.Application, server) -> None:
    app[V2_SERVER_KEY] = server
    app[JOBS_KEY] = JobStore()
    app.add_routes(ROUTES)
