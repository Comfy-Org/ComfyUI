"""v2 test app: ``register(app, FakeServer)`` over the real asset DB.

``FakeServer`` provides exactly the duck-typed surface ``app.api_v2`` uses.
Later tests extend it by setting attributes, e.g. ``fake_server.enqueue_prompt = ...``,
appending to ``fake_server.prompt_queue.history``/``running``/``queued``, or
``fake_server.user_manager.multi_user = True`` (users: ``fake_server.user_manager.users``).
"""

import re
from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import folder_paths
from app.api_v2 import register
from app.assets import mode
from app.assets.manager import AssetsEnabled
from comfy.cli_args import args

from .contract import HTTP_METHODS, SPEC, resolve, response_schema, validate

# (METHOD, raw-path regex, operation); literal paths first so from-hash wins over {id}.
_OPERATION_PATTERNS = sorted(
    (
        (method.upper(), re.compile("^" + re.sub(r"\\\{[^}]+\\\}", "[^/]+", re.escape(path)) + "$"), op)
        for path, item in SPEC["paths"].items()
        for method, op in item.items()
        if method in HTTP_METHODS
    ),
    key=lambda p: p[1].pattern.count("[^/]+"),
)


async def assert_conforms(resp) -> None:
    """Fail unless ``resp``'s status is declared by its contract operation and a JSON body
    matches that status's declared application/json schema. Non-contract paths are skipped."""
    op = next((op for method, pattern, op in _OPERATION_PATTERNS if method == resp.method and pattern.match(resp.url.raw_path)), None)
    if op is None:
        return
    declared = op["responses"].get(str(resp.status))
    assert declared is not None, f"{op['operationId']} returned undeclared status {resp.status}: {await resp.text()}"
    declared = resolve(declared["$ref"]) if "$ref" in declared else declared
    if resp.content_type == "application/json" and "application/json" in (declared.get("content") or {}):
        validate(response_schema(op["operationId"], resp.status), await resp.json())


class ConformingClient(TestClient):
    async def _request(self, method, path, **kwargs):
        resp = await super()._request(method, path, **kwargs)
        await assert_conforms(resp)
        return resp


class FakeUserManager:
    """``UserManager.get_request_user_id`` without the users file."""

    def __init__(self):
        self.multi_user = False
        self.users = {"default"}

    def get_request_user_id(self, request):
        user = "default"
        if self.multi_user and "comfy-user" in request.headers:
            user = request.headers["comfy-user"]
            if user.startswith(folder_paths.SYSTEM_USER_PREFIX):
                raise KeyError("Unknown user: " + user)
        if user not in self.users:
            raise KeyError("Unknown user: " + user)
        return user


class FakePromptQueue:
    """``running``/``queued`` are queue tuples; ``history`` maps prompt id to its history item."""

    def __init__(self):
        self.running = []
        self.queued = []
        self.history = {}
        self.put_items = []

    def get_current_queue_volatile(self):
        return list(self.running), list(self.queued)

    def get_history(self, prompt_id=None, max_items=None, offset=-1, map_function=None):
        if prompt_id is not None:
            items = {prompt_id: self.history[prompt_id]} if prompt_id in self.history else {}
        else:
            items = dict(self.history)
        return {k: map_function(v) for k, v in items.items()} if map_function else items

    def put(self, item):
        self.put_items.append(item)


class FakeServer:
    def __init__(self):
        self.prompt_queue = FakePromptQueue()
        self.user_manager = FakeUserManager()
        self.asset_manager = AssetsEnabled(args)
        self.sid_listeners = {}
        self.sockets_metadata = {}
        self.cancelled = []

    async def enqueue_prompt(self, json_data, usage_source=None):
        raise NotImplementedError("set fake_server.enqueue_prompt in the test")

    def cancel_job_by_id(self, job_id):
        raise NotImplementedError("set fake_server.cancel_job_by_id in the test")


@pytest.fixture
def v2_dirs(tmp_path, monkeypatch):
    """input/, output/, temp/ and checkpoints and loras model folders in ``tmp_path``."""
    for name in ("input", "output", "temp", "models/checkpoints", "models/loras"):
        (tmp_path / name).mkdir(parents=True)
    monkeypatch.setattr(folder_paths, "input_directory", str(tmp_path / "input"))
    monkeypatch.setattr(folder_paths, "output_directory", str(tmp_path / "output"))
    monkeypatch.setattr(folder_paths, "temp_directory", str(tmp_path / "temp"))
    for folder in ("checkpoints", "loras"):
        monkeypatch.setitem(
            folder_paths.folder_names_and_paths,
            folder,
            ([str(tmp_path / "models" / folder)], folder_paths.supported_pt_extensions),
        )
    return tmp_path


@pytest.fixture(autouse=True)
def hashing():
    """``--enable-asset-hashing`` on; a test turns it off with ``hashing.enable_asset_hashing = False``."""
    flags = SimpleNamespace(enable_asset_hashing=True)
    mode.init(flags)
    yield flags
    mode.init(None)


@pytest.fixture
def fake_server():
    return FakeServer()


@pytest.fixture
def v2_app(file_db, fake_server):
    app = web.Application()
    register(app, fake_server)
    return app


@pytest_asyncio.fixture
async def v2_client(v2_app):
    async with TestClient(TestServer(v2_app)) as client:
        yield client


@pytest_asyncio.fixture
async def conforming_client(v2_app):
    """``v2_client`` that fails the test on any response that does not conform to the contract."""
    async with ConformingClient(TestServer(v2_app)) as client:
        yield client
