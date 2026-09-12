from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, assert_never

import pytest
import requests
from _pytest.mark.structures import ParameterSet


RouteKind = Literal[
    "head_asset_hash",
    "list_assets",
    "get_asset",
    "get_asset_content",
    "create_from_hash",
    "upload_asset",
    "update_asset",
    "delete_asset",
    "get_tags",
    "add_asset_tags",
    "delete_asset_tags",
    "get_tags_refine",
    "seed_assets",
    "get_seed_status",
    "cancel_seed",
    "prune_missing_assets",
]


@dataclass(frozen=True, slots=True)
class RouteSpec:
    method: str
    path: str
    kind: RouteKind
    xfail_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SmokeContext:
    http: requests.Session
    api_base: str
    seeded_asset: SmokeAsset
    make_asset_bytes: Callable[[str, int], bytes]


class SmokeAsset(TypedDict):
    id: str
    hash: str


ROUTES_PY = Path(__file__).resolve().parents[2] / "app/assets/api/routes.py"

ROUTE_DECORATOR_RE = re.compile(
    r"@ROUTES\.(?P<method>get|post|put|delete|head)\((?:f)?[\"'](?P<path>.*?)[\"']\)",
    re.DOTALL,
)
NESTED_PARAM_RE = re.compile(r"\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\:\{[^{}]+\}\}")
PATH_PARAM_RE = re.compile(r"\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\:[^{}]+\}")


ROUTE_SPECS: tuple[RouteSpec, ...] = (
    RouteSpec("HEAD", "/api/assets/hash/{hash}", "head_asset_hash"),
    RouteSpec("GET", "/api/assets", "list_assets"),
    RouteSpec("GET", "/api/assets/{id}", "get_asset"),
    RouteSpec("GET", "/api/assets/{id}/content", "get_asset_content"),
    RouteSpec("POST", "/api/assets/from-hash", "create_from_hash"),
    RouteSpec("POST", "/api/assets", "upload_asset"),
    RouteSpec("PUT", "/api/assets/{id}", "update_asset"),
    RouteSpec("DELETE", "/api/assets/{id}", "delete_asset"),
    RouteSpec("GET", "/api/tags", "get_tags"),
    RouteSpec("POST", "/api/assets/{id}/tags", "add_asset_tags"),
    RouteSpec("DELETE", "/api/assets/{id}/tags", "delete_asset_tags"),
    RouteSpec("GET", "/api/assets/tags/refine", "get_tags_refine"),
    RouteSpec("POST", "/api/assets/seed", "seed_assets"),
    RouteSpec("GET", "/api/assets/seed/status", "get_seed_status"),
    RouteSpec("POST", "/api/assets/seed/cancel", "cancel_seed"),
    RouteSpec("POST", "/api/assets/prune", "prune_missing_assets"),
)

EXPECTED_ROUTE_KEYS = {f"{spec.method} {spec.path}" for spec in ROUTE_SPECS}


def _normalize_route_path(raw_path: str) -> str:
    path = raw_path.replace("{{", "{").replace("}}", "}")
    path = NESTED_PARAM_RE.sub(r"{\g<name>}", path)
    return PATH_PARAM_RE.sub(r"{\g<name>}", path)


def parse_route_keys(source: str) -> set[str]:
    return {
        f"{match.group('method').upper()} {_normalize_route_path(match.group('path'))}"
        for match in ROUTE_DECORATOR_RE.finditer(source)
    }


def assert_routes_covered(route_keys: Collection[str], covered_keys: Collection[str]) -> None:
    route_set = set(route_keys)
    covered_set = set(covered_keys)
    missing = sorted(route_set - covered_set)
    extra = sorted(covered_set - route_set)
    if missing or extra:
        raise AssertionError(f"route coverage mismatch: missing={missing}, extra={extra}")


def _route_key(spec: RouteSpec) -> str:
    return f"{spec.method} {spec.path}"


def _make_route_param(spec: RouteSpec) -> ParameterSet:
    marks = ()
    if spec.xfail_reason is not None:
        marks = (pytest.mark.xfail(strict=True, reason=spec.xfail_reason),)
    return pytest.param(spec, marks=marks, id=_route_key(spec))


ROUTE_PARAMS: tuple[ParameterSet, ...] = tuple(_make_route_param(spec) for spec in ROUTE_SPECS)


@pytest.fixture
def smoke_context(
    http: requests.Session,
    api_base: str,
    seeded_asset: SmokeAsset,
    make_asset_bytes: Callable[[str, int], bytes],
) -> SmokeContext:
    return SmokeContext(
        http=http,
        api_base=api_base,
        seeded_asset=seeded_asset,
        make_asset_bytes=make_asset_bytes,
    )


@pytest.fixture
def seeded_asset(
    asset_factory: Callable[[str, Sequence[str], dict[str, object], bytes], dict[str, object]],
    make_asset_bytes: Callable[[str, int], bytes],
) -> SmokeAsset:
    name = f"surface-smoke-{uuid.uuid4().hex}.safetensors"
    body = asset_factory(
        name,
        ["models", "model_type:checkpoints", "unit-tests", "smoke"],
        {"purpose": "smoke"},
        make_asset_bytes(name, 512),
    )
    return {
        "id": str(body["id"]),
        "hash": str(body["hash"]),
    }


def _request_for_spec(spec: RouteSpec, ctx: SmokeContext) -> requests.Response:
    seeded = ctx.seeded_asset
    asset_id = str(seeded["id"])
    asset_hash = str(seeded["hash"])

    match spec.kind:
        case "head_asset_hash":
            return ctx.http.head(f"{ctx.api_base}/api/assets/hash/{asset_hash}", timeout=120)
        case "list_assets":
            return ctx.http.get(f"{ctx.api_base}/api/assets", params={"limit": "1"}, timeout=120)
        case "get_asset":
            return ctx.http.get(f"{ctx.api_base}/api/assets/{asset_id}", timeout=120)
        case "get_asset_content":
            return ctx.http.get(f"{ctx.api_base}/api/assets/{asset_id}/content", timeout=120)
        case "create_from_hash":
            return ctx.http.post(
                f"{ctx.api_base}/api/assets/from-hash",
                json={
                    "hash": asset_hash,
                    "name": f"smoke-copy-{uuid.uuid4().hex}.safetensors",
                    "tags": ["models", "unit-tests", "smoke"],
                    "user_metadata": {"purpose": "smoke"},
                },
                timeout=120,
            )
        case "upload_asset":
            name = f"smoke-upload-{uuid.uuid4().hex}.bin"
            return ctx.http.post(
                f"{ctx.api_base}/api/assets",
                files={"file": (name, ctx.make_asset_bytes(name, 512), "application/octet-stream")},
                data={
                    "tags": json.dumps(["input", "unit-tests", "smoke"]),
                    "name": name,
                    "user_metadata": json.dumps({"purpose": "smoke"}),
                },
                timeout=120,
            )
        case "update_asset":
            return ctx.http.put(
                f"{ctx.api_base}/api/assets/{asset_id}",
                json={"name": f"smoke-renamed-{uuid.uuid4().hex}.safetensors"},
                timeout=120,
            )
        case "delete_asset":
            return ctx.http.delete(f"{ctx.api_base}/api/assets/{asset_id}", timeout=120)
        case "get_tags":
            return ctx.http.get(f"{ctx.api_base}/api/tags", params={"limit": "1"}, timeout=120)
        case "add_asset_tags":
            return ctx.http.post(
                f"{ctx.api_base}/api/assets/{asset_id}/tags",
                json={"tags": ["smoke-tag"]},
                timeout=120,
            )
        case "delete_asset_tags":
            return ctx.http.delete(
                f"{ctx.api_base}/api/assets/{asset_id}/tags",
                json={"tags": ["unit-tests"]},
                timeout=120,
            )
        case "get_tags_refine":
            return ctx.http.get(
                f"{ctx.api_base}/api/assets/tags/refine",
                params={"limit": "1"},
                timeout=120,
            )
        case "seed_assets":
            return ctx.http.post(
                f"{ctx.api_base}/api/assets/seed?wait=true",
                json={"roots": ["models"]},
                timeout=120,
            )
        case "get_seed_status":
            return ctx.http.get(f"{ctx.api_base}/api/assets/seed/status", timeout=120)
        case "cancel_seed":
            return ctx.http.post(f"{ctx.api_base}/api/assets/seed/cancel", timeout=120)
        case "prune_missing_assets":
            return ctx.http.post(f"{ctx.api_base}/api/assets/prune", timeout=120)
        case _:
            assert_never(spec.kind)


def test_route_coverage_guard_rejects_extra_route() -> None:
    with pytest.raises(AssertionError, match=r"GET /api/extra"):
        assert_routes_covered(
            ["GET /api/assets", "POST /api/assets", "GET /api/extra"],
            ["GET /api/assets", "POST /api/assets"],
        )


def test_route_coverage_matches_routes_py() -> None:
    assert_routes_covered(
        parse_route_keys(ROUTES_PY.read_text(encoding="utf-8")),
        sorted(EXPECTED_ROUTE_KEYS),
    )


@pytest.mark.parametrize("spec", ROUTE_PARAMS)
def test_api_surface(spec: RouteSpec, smoke_context: SmokeContext) -> None:
    response = _request_for_spec(spec, smoke_context)
    assert response.status_code < 500, f"{_route_key(spec)} returned {response.status_code}"
