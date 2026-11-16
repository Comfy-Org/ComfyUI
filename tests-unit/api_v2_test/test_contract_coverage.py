"""Every contract operation has a route, every route is a contract operation, and every
status a handler can return is declared by its operation."""

import re

import jsonschema
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

import app.api_v2 as api_v2

from . import contract
from .conftest import ConformingClient

# Every status each handler can return, read from app/api_v2. 500 is aiohttp's answer to an
# unhandled exception; 403 forbidden comes from routes.user_id for unknown users; getAssetContent's
# 403/404/416/206 also come from aiohttp's FileResponse (unreadable or vanished file, bad Range).
IMPLEMENTED_STATUSES = {
    "getHealth": {200, 500},
    "postAssets": {200, 201, 403, 409, 413, 422, 500},
    "assetFromHash": {200, 403, 404, 422, 500},
    "assetFromPath": {501, 500},
    "headAssetByHash": {200, 403, 404, 500},
    "getAsset": {200, 403, 404, 500},
    "getAssetContent": {200, 206, 403, 404, 416, 500},
    "deleteAsset": {204, 403, 404, 500},
    "listJobs": {200, 400, 403, 422, 500},
    "postJobs": {201, 400, 403, 422, 429, 500},
    "getJob": {200, 403, 404, 500},
    "cancelJob": {200, 403, 404, 500},
    "getJobWorkflow": {200, 403, 404, 500},
    "getJobLogs": {204, 403, 404, 500},
    "getJobEvents": {200, 403, 404, 429, 500},
}


def route_inventory():
    app = web.Application()
    api_v2.register(app, object())
    routes = set()
    for route in app.router.routes():
        path = route.resource.canonical
        if path.startswith("/api/v2"):
            routes.add((route.method, re.sub(r"\{(\w+):.*?\}(?=/|$)", r"{\1}", path)))
    return routes


def test_routes_are_exactly_the_contract_operations():
    assert route_inventory() == contract.OPERATIONS | contract.ROUTE_EXTENSIONS
    assert len(contract.OPERATIONS) == 15
    assert {("GET", "/api/v2/health"), ("POST", "/api/v2/assets/from-path")} <= contract.OPERATIONS


def test_route_extensions_registry():
    assert contract.ROUTE_EXTENSIONS == set()
    assert set(api_v2.ROUTE_EXTENSIONS) == contract.ROUTE_EXTENSIONS
    assert all(reason.strip() for reason in api_v2.ROUTE_EXTENSIONS.values())


def test_field_extensions_have_reasons():
    assert api_v2.FIELD_EXTENSIONS
    assert all(name and reason.strip() for name, reason in api_v2.FIELD_EXTENSIONS.items())


def test_implemented_statuses_are_declared():
    operations = {op["operationId"]: op for item in contract.SPEC["paths"].values() for m, op in item.items() if m in contract.HTTP_METHODS}
    assert set(IMPLEMENTED_STATUSES) == set(operations)
    undeclared = {(op_id, s) for op_id, statuses in IMPLEMENTED_STATUSES.items() for s in statuses if str(s) not in operations[op_id]["responses"]}
    assert not undeclared, f"statuses the implementation returns but the contract does not declare: {sorted(undeclared)}"


@pytest.mark.asyncio
@pytest.mark.parametrize("status, body, error", [(418, {"status": "healthy"}, AssertionError), (200, {"status": "sideways"}, jsonschema.ValidationError)])
async def test_conforming_client_rejects_nonconforming_responses(status, body, error):
    async def health(request):
        return web.json_response(body, status=status)

    app = web.Application()
    app.router.add_get("/api/v2/health", health, allow_head=False)
    async with ConformingClient(TestServer(app)) as client:
        with pytest.raises(error):
            await client.get("/api/v2/health")
