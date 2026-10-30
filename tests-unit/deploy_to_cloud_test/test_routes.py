"""Tests for GET /api/deploy/summary and POST /api/deploy/bundle, against a real fake comfy-builder server"""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
import torch
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer, unused_port

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import folder_paths  # noqa: E402
import server  # noqa: E402
from app import deploy_to_cloud  # noqa: E402
from app import comfy_api_env  # noqa: E402

BEARER = "Bearer eyJhbGciOiJSUzI1NiJ9.test.sig"
API_KEY = "comfyui-" + "0123456789abcdef" * 4
SHA = "a" * 40
CREATED = {"id": "0b9a3f4e-1c2d-4e5f-8a9b-0c1d2e3f4a5b", "workspaceId": "ws 1/a", "expiresAt": "2026-10-17T12:00:00Z"}
WORKFLOW = {"last_node_id": 1, "nodes": [{"id": 1, "type": "KSampler"}]}

# Every `error` value the routes can return: error_of rejects any other, and
# test_expected_error_codes_are_all_exercised requires each one to be asserted by some test.
EXPECTED_ERROR_CODES = {
    "offline",
    "missing_credentials",
    "multiple_credentials",
    "invalid_api_key_format",
    "invalid_body",
    "unauthorized",
    "too_large",
    "too_many_bundles",
    "builder_busy",
    "builder_rejected",
    "builder_unavailable",
}


def write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def installation(tmp_path, monkeypatch):
    """A realistic local install: cnr, git and file custom nodes, a disabled pack, and models in two folders."""
    base = tmp_path / "ComfyUI"
    nodes = base / "custom_nodes"
    write(nodes / "ComfyUI-KJNodes" / ".tracking")
    write(nodes / "ComfyUI-KJNodes" / "pyproject.toml", '[project]\nname = "comfyui-kjnodes"\nversion = "1.1.2"\n')
    gitdir = nodes / "ComfyUI-Impact-Pack" / ".git"
    write(gitdir / "HEAD", "ref: refs/heads/main\n")
    write(gitdir / "refs" / "heads" / "main", SHA + "\n")
    write(gitdir / "config", '[remote "origin"]\n\turl = https://user:tok@github.com/ltdrdata/ComfyUI-Impact-Pack.git\n')
    write(nodes / "my_local_nodes" / "__init__.py")
    write(nodes / "simple_node.py")
    write(nodes / ".disabled" / "old_pack" / "__init__.py")
    models = base / "models"
    for rel in ("checkpoints/sd_xl_base_1.0.safetensors", "checkpoints/flux/flux1-dev-Q4.gguf",
                "checkpoints/readme.txt", "loras/detail.safetensors"):
        write(models / rel)
    monkeypatch.setattr(folder_paths, "base_path", str(base))
    monkeypatch.setattr(folder_paths, "filename_list_cache", {})
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {
        "custom_nodes": ([str(nodes)], set()),
        "configs": ([str(models / "configs")], {".yaml"}),
        # .gguf is registered for checkpoints by GGUF loader packs in real installs
        "checkpoints": ([str(models / "checkpoints")], set(folder_paths.supported_pt_extensions) | {".gguf"}),
        "loras": ([str(models / "loras")], set(folder_paths.supported_pt_extensions)),
    })
    return base


@pytest.fixture
def builder_state():
    return {"status": 201, "body": CREATED, "headers": {}, "delay": 0, "requests": [], "redirect_hits": 0}


@pytest_asyncio.fixture
async def builder(builder_state, monkeypatch):
    """Fake comfy-builder on a real localhost socket, recording what it receives."""
    async def create(request):
        builder_state["requests"].append({"headers": dict(request.headers), "body": await request.json()})
        await asyncio.sleep(builder_state["delay"])
        if builder_state["status"] == 302:
            raise web.HTTPFound("/redirect-target")
        headers = builder_state["headers"]
        if isinstance(builder_state["body"], bytes):
            return web.Response(body=builder_state["body"], status=builder_state["status"], content_type="text/html", headers=headers)
        return web.json_response(builder_state["body"], status=builder_state["status"], headers=headers)

    async def redirect_target(request):
        builder_state["redirect_hits"] += 1
        return web.json_response(CREATED, status=201)

    app = web.Application()
    app.router.add_post("/builder/v1/import-bundles", create)
    app.router.add_route("*", "/redirect-target", redirect_target)
    async with TestServer(app) as srv:
        monkeypatch.setattr(deploy_to_cloud, "builder_base_url", lambda: str(srv.make_url("/builder")))
        yield builder_state


@pytest_asyncio.fixture
async def client(monkeypatch):
    monkeypatch.setattr(args, "offline", False)
    monkeypatch.setattr(args, "comfy_api_base", "https://api.comfy.org")
    prompt_server = server.PromptServer(None, MagicMock(enabled=False))
    prompt_server.add_routes()
    await prompt_server.setup()
    try:
        async with TestClient(TestServer(prompt_server.app)) as c:
            yield c
    finally:
        await prompt_server.client_session.close()


async def error_of(resp, status):
    assert resp.status == status
    code = (await resp.json())["error"]
    assert code in EXPECTED_ERROR_CODES
    return code


# --- summary ---

@pytest.mark.asyncio
async def test_summary_returns_only_counts(client, installation, monkeypatch):
    def no_pip():
        raise AssertionError("summary must not collect pip packages")
    monkeypatch.setattr(deploy_to_cloud, "pip_packages", no_pip)
    resp = await client.get("/api/deploy/summary")
    assert resp.status == 200
    body = await resp.json()
    assert set(body) == {"custom_nodes", "models"}
    assert body == {"custom_nodes": 4, "models": 3}


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path", [
    ("GET", "/deploy/summary"),
    ("GET", "/api/api/deploy/summary"),
    ("POST", "/deploy/bundle"),
    ("POST", "/api/api/deploy/bundle"),
])
async def test_no_route_aliases(client, method, path):
    resp = await client.request(method, path)
    assert resp.status in (404, 405)


@pytest.mark.asyncio
async def test_summary_offline(client, monkeypatch):
    monkeypatch.setattr(args, "offline", True)
    assert await error_of(await client.get("/api/deploy/summary"), 403) == "offline"


# --- bundle: success ---

@pytest.mark.asyncio
async def test_bundle_success_with_bearer(client, builder, installation, tmp_path):
    resp = await client.post("/api/deploy/bundle", json={"workflow": WORKFLOW, "name": "My install"},
                             headers={"Authorization": BEARER})
    assert resp.status == 200
    assert await resp.json() == {
        "url": "https://platform.comfy.org/profile/builds/new?step=import"
               "&bundle=0b9a3f4e-1c2d-4e5f-8a9b-0c1d2e3f4a5b&workspace=ws+1%2Fa&bundle_workspace=ws+1%2Fa",
        "bundle_id": CREATED["id"],
        "workspace_id": CREATED["workspaceId"],
        "expires_at": CREATED["expiresAt"],
    }
    [received] = builder["requests"]
    (tmp_path / "bundle-request.json").write_text(json.dumps(received["body"], indent=2))
    assert received["headers"]["Authorization"] == BEARER
    assert "X-API-Key" not in received["headers"]
    assert received["headers"]["Content-Type"].startswith("application/json")
    body = received["body"]
    assert body["source"] == "comfyui"
    assert body["workflow"] == WORKFLOW
    assert body["name"] == "My install"
    assert body["snapshot"]["type"] == "comfyui-desktop-2-snapshot"
    assert body["snapshot"]["installationName"] == "ComfyUI"
    assert {(n["type"], n["dirName"]) for n in body["snapshot"]["snapshots"][0]["customNodes"]} == {
        ("cnr", "ComfyUI-KJNodes"), ("git", "ComfyUI-Impact-Pack"), ("git", "my_local_nodes"),
        ("file", "simple_node.py"), ("git", "old_pack"),
    }
    assert body["models"] == [
        {"directory": "checkpoints", "filename": "flux/flux1-dev-Q4.gguf"},
        {"directory": "checkpoints", "filename": "sd_xl_base_1.0.safetensors"},
        {"directory": "loras", "filename": "detail.safetensors"},
    ]


@pytest.mark.asyncio
async def test_bundle_success_with_api_key_and_no_optional_fields(client, builder, installation):
    resp = await client.post("/api/deploy/bundle", json={"workflow": None}, headers={"X-API-Key": API_KEY})
    assert resp.status == 200
    [received] = builder["requests"]
    assert received["headers"]["X-API-Key"] == API_KEY
    assert "Authorization" not in received["headers"]
    assert set(received["body"]) == {"snapshot", "models", "source"}


@pytest.mark.asyncio
async def test_proxy_basic_auth_is_ignored_and_api_key_forwarded_alone(client, builder, installation):
    resp = await client.post("/api/deploy/bundle", json={},
                             headers={"Authorization": "Basic dXNlcjpwYXNz", "X-API-Key": API_KEY})
    assert resp.status == 200
    [received] = builder["requests"]
    assert received["headers"]["X-API-Key"] == API_KEY
    assert "Authorization" not in received["headers"]


@pytest.mark.asyncio
async def test_staging_tier_returns_staging_platform_url(client, builder, installation, monkeypatch):
    monkeypatch.setattr(args, "comfy_api_base", "https://stagingapi.comfy.org")
    resp = await client.post("/api/deploy/bundle", json={}, headers={"Authorization": BEARER})
    assert (await resp.json())["url"].startswith("https://stagingplatform.comfy.org/profile/builds/new?")


# --- bundle: rejected locally, before collection or network ---

LOCAL_ERRORS = [
    ({}, {}, 401, "missing_credentials"),
    ({"Authorization": "Basic abc"}, {}, 401, "missing_credentials"),
    ({"Authorization": BEARER, "X-API-Key": API_KEY}, {}, 400, "multiple_credentials"),
    ({"X-API-Key": "comfyui-short"}, {}, 400, "invalid_api_key_format"),
    ({"X-API-Key": "sk-" + "0" * 64}, {}, 400, "invalid_api_key_format"),
    ({"Authorization": "Basic dXNlcjpwYXNz", "X-API-Key": "comfyui-short"}, {}, 400, "invalid_api_key_format"),
    ({"Authorization": BEARER}, [1, 2], 400, "invalid_body"),
    ({"Authorization": BEARER}, "not json", 400, "invalid_body"),
    ({"Authorization": BEARER}, {"workflow": [1]}, 400, "invalid_body"),
    ({"Authorization": BEARER}, {"name": "x" * 201}, 400, "invalid_body"),
    ({"Authorization": BEARER}, {"name": 5}, 400, "invalid_body"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("headers,body,status,code", LOCAL_ERRORS)
async def test_bundle_local_rejections_skip_collection_and_network(client, builder, monkeypatch, headers, body, status, code):
    def no_collect():
        raise AssertionError("collect must not run")
    monkeypatch.setattr(deploy_to_cloud, "collect", no_collect)
    kwargs = {"data": body} if isinstance(body, str) else {"json": body}
    resp = await client.post("/api/deploy/bundle", headers=headers, **kwargs)
    assert await error_of(resp, status) == code
    assert builder["requests"] == []


@pytest.mark.asyncio
async def test_bundle_offline_makes_no_request(client, builder, monkeypatch):
    monkeypatch.setattr(args, "offline", True)
    resp = await client.post("/api/deploy/bundle", json={}, headers={"Authorization": BEARER})
    assert await error_of(resp, 403) == "offline"
    assert builder["requests"] == []


@pytest.mark.asyncio
async def test_name_of_200_chars_is_accepted(client, builder, installation):
    resp = await client.post("/api/deploy/bundle", json={"name": "é" * 200}, headers={"Authorization": BEARER})
    assert resp.status == 200


# --- bundle: builder outcomes ---

BUILDER_ERRORS = [
    (401, {"error": "UNAUTHORIZED", "message": "no"}, 401, "unauthorized"),
    (403, {"error": "FORBIDDEN", "message": "no"}, 403, "unauthorized"),
    (413, {"error": "IMPORT_BUNDLE_TOO_LARGE", "message": "big"}, 413, "too_large"),
    (429, {"error": "IMPORT_BUNDLE_LIMIT", "message": "20"}, 429, "too_many_bundles"),
    (429, {"error": "IMPORT_BUNDLE_IN_PROGRESS", "message": "wait"}, 503, "builder_busy"),
    (429, b"<html>429 Too Many Requests</html>", 503, "builder_busy"),
    (503, {"error": "IMPORT_BUNDLE_BUSY", "message": "busy"}, 503, "builder_busy"),
    (503, {"error": "IMPORT_BUNDLE_STORAGE_UNAVAILABLE", "message": "down"}, 502, "builder_unavailable"),
    (503, b"<html>503 Service Unavailable</html>", 502, "builder_unavailable"),
    (408, {"error": "IMPORT_BUNDLE_TIMEOUT", "message": "slow"}, 502, "builder_unavailable"),
    (415, {"error": "UNSUPPORTED_MEDIA_TYPE", "message": "json"}, 502, "builder_unavailable"),
    (400, {"error": "INVALID_IMPORT_BUNDLE", "message": "bad"}, 502, "builder_rejected"),
    (400, b"<html>400 Bad Request</html>", 502, "builder_rejected"),
    (404, {"error": "NOT_FOUND"}, 502, "builder_unavailable"),
    (405, {"error": "METHOD_NOT_ALLOWED"}, 502, "builder_unavailable"),
    (500, {"error": "INTERNAL"}, 502, "builder_unavailable"),
    (302, None, 502, "builder_unavailable"),
    (201, {"id": "x"}, 502, "builder_unavailable"),  # 201 without a usable id/workspace never yields a link
    (201, "not an object", 502, "builder_unavailable"),
    (201, b"<html>created</html>", 502, "builder_unavailable"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("builder_status,builder_body,status,code", BUILDER_ERRORS)
async def test_bundle_builder_outcomes(client, builder, installation, builder_status, builder_body, status, code):
    builder["status"], builder["body"] = builder_status, builder_body
    resp = await client.post("/api/deploy/bundle", json={}, headers={"Authorization": BEARER})
    assert await error_of(resp, status) == code
    assert len(builder["requests"]) == 1
    assert builder["redirect_hits"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("builder_status,builder_error,retry_after,expected", [
    (429, "IMPORT_BUNDLE_IN_PROGRESS", "5", "5"),
    (503, "IMPORT_BUNDLE_BUSY", "5", "5"),
    (429, "IMPORT_BUNDLE_IN_PROGRESS", None, None),
    (503, "IMPORT_BUNDLE_BUSY", "Wed, 21 Oct 2026 07:28:00 GMT", None),
    (429, "IMPORT_BUNDLE_IN_PROGRESS", "-1", None),
    (429, "IMPORT_BUNDLE_LIMIT", "5", None),
    (503, "IMPORT_BUNDLE_STORAGE_UNAVAILABLE", "5", None),
])
async def test_retry_after_passes_through_only_digits_on_builder_busy(client, builder, installation, builder_status, builder_error, retry_after, expected):
    builder["status"], builder["body"] = builder_status, {"error": builder_error}
    builder["headers"] = {} if retry_after is None else {"Retry-After": retry_after}
    resp = await client.post("/api/deploy/bundle", json={}, headers={"Authorization": BEARER})
    assert resp.headers.get("Retry-After") == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("builder_body,message", [
    ({"error": "INVALID_IMPORT_BUNDLE", "message": "m" * 600, "extra": 1}, "m" * 500),
    (b"<html>400 Bad Request</html>", ""),
])
async def test_builder_rejection_forwards_only_a_truncated_message(client, builder, installation, builder_body, message):
    builder["status"], builder["body"] = 400, builder_body
    resp = await client.post("/api/deploy/bundle", json={}, headers={"Authorization": BEARER})
    assert resp.status == 502
    assert await resp.json() == {"error": "builder_rejected", "message": message}


@pytest.mark.asyncio
async def test_builder_connection_refused(client, installation, monkeypatch):
    monkeypatch.setattr(deploy_to_cloud, "builder_base_url", lambda: f"http://127.0.0.1:{unused_port()}/builder")
    resp = await client.post("/api/deploy/bundle", json={}, headers={"Authorization": BEARER})
    assert await error_of(resp, 502) == "builder_unavailable"


@pytest.mark.asyncio
async def test_builder_timeout(client, builder, installation, monkeypatch):
    monkeypatch.setattr(deploy_to_cloud, "_UPLOAD_TIMEOUT", deploy_to_cloud.aiohttp.ClientTimeout(total=0.2))
    builder["delay"] = 2
    resp = await client.post("/api/deploy/bundle", json={}, headers={"Authorization": BEARER})
    assert await error_of(resp, 502) == "builder_unavailable"


@pytest.mark.asyncio
async def test_credentials_and_payload_never_logged(client, builder, installation, caplog):
    caplog.set_level("DEBUG")
    await client.post("/api/deploy/bundle", json={"name": "secret-name"}, headers={"Authorization": BEARER})
    builder["status"] = 500
    await client.post("/api/deploy/bundle", json={}, headers={"X-API-Key": API_KEY})
    assert CREATED["id"] in caplog.text
    for secret in (BEARER.split()[1], API_KEY, "secret-name", "ComfyUI-KJNodes"):
        assert secret not in caplog.text


def test_expected_error_codes_are_all_exercised():
    asserted = {"offline"} | {c for *_, c in LOCAL_ERRORS} | {c for *_, c in BUILDER_ERRORS}
    assert asserted == EXPECTED_ERROR_CODES


# --- comfy_api_env tier mapping ---

@pytest.mark.parametrize("base,builder_url,platform_url", [
    ("https://api.comfy.org", "https://platformapi.comfy.org/builder", "https://platform.comfy.org"),
    ("", "https://platformapi.comfy.org/builder", "https://platform.comfy.org"),
    ("https://stagingapi.comfy.org", "https://stagingplatformapi.comfy.org/builder", "https://stagingplatform.comfy.org"),
    ("https://pr-7.testenvs.comfy.org", "https://stagingplatformapi.comfy.org/builder", "https://stagingplatform.comfy.org"),
])
def test_builder_and_platform_urls_follow_api_tier(monkeypatch, base, builder_url, platform_url):
    monkeypatch.setattr(args, "comfy_api_base", base)
    assert comfy_api_env.builder_base_url() == builder_url
    assert comfy_api_env.platform_base_url() == platform_url
