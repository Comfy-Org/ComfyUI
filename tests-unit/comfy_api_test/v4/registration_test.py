import builtins
import importlib.util
import json
import logging
import os
import sys

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import folder_paths
import nodes
import server
from app import extension_packs
from comfy_api import v4
from comfy_api.v4 import _packs, _services
from comfy_api.v4.util import folders
from comfy_execution import cache_provider
from execution import validate_prompt

PACK = """\
import json

from comfy_api.v4 import ComfyExtension, io

EXTRA_NODES = []


class Node(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        {schema}

    @classmethod
    def execute(cls):
        return io.NodeOutput(1)


async def echo(request):
    view = {{key: request[key] for key in ("method", "path", "match_info", "query", "remote")}}
    view["x-test"] = request["headers"].get("x-test")
    view["body"] = request["body"].decode()
    return {{
        "status": 200,
        "headers": {{"content-type": "application/json"}},
        "body": json.dumps({{"pack": {name!r}, **view}}).encode(),
    }}


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [Node] + EXTRA_NODES

    async def get_routes(self):
        return [{{"method": method, "path": path, "handler": echo}} for method, path in {routes!r}]


async def comfy_entrypoint():
    return Extension()

{extra}
"""

SCHEMA = "return io.Schema(node_id={node_id!r}, outputs=[io.Int.Output()])"
DISPLAYED = (
    "return io.Schema(node_id={node_id!r}, display_name='Shown', is_output_node=True,"
    " outputs=[io.Int.Output()])"
)
WEB = 'WEB_DIRECTORY = "./web"'


def write_pack(pack, routes, name=None, node_id=None, schema=None, extra="", web=False):
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "pyproject.toml").write_text(f'[project]\nname = "{name or pack.name}"\n')
    if web:
        (pack / "web").mkdir(exist_ok=True)
        (pack / "web" / f"{pack.name}.js").write_text(f"// {pack.name}")
        extra = WEB + "\n" + extra
    schema = schema or SCHEMA.format(node_id=node_id or f"{pack.name}Node")
    (pack / "__init__.py").write_text(
        PACK.format(name=pack.name, routes=routes, schema=schema, extra=extra),
        encoding="utf-8",
    )


@pytest.fixture
def load(tmp_path, monkeypatch):
    """Write a pack folder under `root` (default: the test's folder) and load it."""

    async def load(folder, routes, root="", **options):
        pack = tmp_path / root / folder
        write_pack(pack, routes, **options)
        monkeypatch.setitem(sys.modules, str(pack).replace(".", "_x_"), None)
        return await nodes.load_custom_node(str(pack))

    return load


def loaded_module(pack):
    """The module ComfyUI's loader made for the pack in this folder."""
    return sys.modules[str(pack).replace(".", "_x_")]


def serving(host):
    host.add_routes()
    return TestClient(TestServer(host.app))


async def answer(client, method, path, **kwargs):
    response = await client.request(method, path, **kwargs)
    text = await response.text()
    return response.status, json.loads(text) if response.status == 200 else text


async def answered_by(client, method, path):
    """The pack whose handler answered, or None if something else did."""
    response = await client.request(method, path)
    if response.content_type != "application/json":
        return None
    body = await response.json()
    return body.get("pack") if isinstance(body, dict) else None


async def valid(node_id):
    prompt = {"1": {"class_type": node_id, "inputs": {}}}
    return await validate_prompt("p", prompt, ["1"])


@pytest.mark.asyncio
async def test_each_pack_answers_its_own_routes_at_both_addresses(host, load):
    alpha = [
        ("POST", "/alpha/items/{item}"),
        ("GET", "/alpha/list"),
        ("GET", "/alpha/{name}"),
    ]
    assert await load("alpha", alpha)
    assert await load("beta", [("GET", "/beta/list")])

    async with serving(host) as client:
        status, body = await answer(
            client,
            "POST",
            "/alpha/items/7?tag=a&tag=b",
            data=b'{"x": 1}',
            headers=[("X-Test", "yes"), ("X-Test", "again")],
        )
        assert status == 200
        assert body == {
            "pack": "alpha",
            "method": "POST",
            "path": "/alpha/items/7",
            "match_info": {"item": "7"},
            "query": [["tag", "a"], ["tag", "b"]],
            "x-test": "yes, again",
            "body": '{"x": 1}',
            "remote": "127.0.0.1",
        }
        assert (await answer(client, "GET", "/api/alpha/list"))[1]["match_info"] == {}
        assert (await answer(client, "GET", "/alpha/x"))[1]["match_info"] == {
            "name": "x"
        }
        assert (await answer(client, "GET", "/beta/list"))[1]["pack"] == "beta"
        assert (await answer(client, "GET", "/api/beta/list"))[1]["pack"] == "beta"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first, second, request_path",
    [
        ("/alpha/list", "/alpha/list", "/alpha/list"),
        ("/alpha/list", "/api/alpha/list", "/api/alpha/list"),
        ("/alpha/list", "/alpha/{anything}", "/alpha/list"),
        ("/alpha/{item}", "/alpha/list", "/alpha/list"),
        ("/alpha/list", "/{everything:.*}", "/alpha/list"),
        ("/p/{pack}/settings", "/p/beta/{key}", "/p/beta/settings"),
        ("/p/beta/{key}", "/p/{pack}/settings", "/p/beta/settings"),
        ("/files/{name}", "/{kind}/info", "/files/info"),
        ("/a/{rest:.*}", "/a/b/{c}", "/a/b/c"),
        ("/a/{x}", "/a/{rest:.*}", "/a/1"),
    ],
)
async def test_a_route_another_pack_serves_disables_the_later_pack(
    host, load, first, second, request_path, caplog
):
    assert await load("alpha", [("GET", first)])

    with caplog.at_level(logging.WARNING):
        assert not await load("beta", [("GET", "/beta/own"), ("GET", second)])
    assert "Pack 'beta'" in caplog.text and "is already served" in caplog.text
    assert "betaNode" not in nodes.NODE_CLASS_MAPPINGS

    async with serving(host) as client:
        assert await answered_by(client, "GET", request_path) == "alpha"
        assert (await answer(client, "GET", "/beta/own"))[0] == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first, second, first_request, second_request",
    [
        ("/p/{id}/info", "/p/{id}/data", "/p/1/info", "/p/1/data"),
        ("/a/{x}", "/a/{x}/{y}", "/a/1", "/a/1/2"),
        ("/a/{rest:.*}", "/b/{c}", "/a/x/y", "/b/1"),
        ("/a/{x}", "/a/{x}/{rest:.*}", "/a/1", "/a/1/"),
    ],
)
async def test_patterns_no_request_could_reach_both_belong_to_different_packs(
    host, load, first, second, first_request, second_request
):
    assert await load("alpha", [("GET", first)])
    assert await load("beta", [("GET", second)])

    async with serving(host) as client:
        assert await answered_by(client, "GET", first_request) == "alpha"
        assert await answered_by(client, "GET", second_request) == "beta"


@pytest.mark.asyncio
async def test_routes_of_different_methods_on_one_path_belong_to_different_packs(
    host, load
):
    assert await load("alpha", [("GET", "/shared/state")])
    assert await load("beta", [("POST", "/shared/state")])

    async with serving(host) as client:
        assert (await answer(client, "GET", "/shared/state"))[1]["pack"] == "alpha"
        assert (await answer(client, "POST", "/shared/state"))[1]["pack"] == "beta"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method, path, request_path",
    [
        ("GET", "/system_stats", "/system_stats"),
        ("GET", "/settings", "/settings"),
        ("GET", "/users", "/users"),
        ("GET", "/userdata", "/userdata?dir=workflows"),
        ("GET", "/i18n", "/i18n"),
        ("GET", "/global_subgraphs", "/global_subgraphs"),
        ("GET", "/node_replacements", "/node_replacements"),
        ("GET", "/userdata/{file}", "/api/userdata/x.json"),
        ("GET", "/internal/logs", "/internal/logs"),
        ("POST", "/internal/anything", "/internal/anything"),
        ("GET", "/templates/index.json", "/templates/index.json"),
        ("GET", "/api/deploy/summary", "/api/deploy/summary"),
        ("GET", "/docs/{page}", "/docs/x"),
        ("GET", "/extensions/alpha/{file}", "/extensions/alpha/alpha.js"),
        ("GET", "/index.html", "/index.html"),
    ],
)
async def test_a_pack_with_a_route_comfyui_serves_is_disabled_and_comfyui_answers_it(
    host, load, method, path, request_path, caplog
):
    assert await load("alpha", [], web=True)
    assert await load("beta", [(method, path), ("GET", "/beta/own")], web=True)

    with caplog.at_level(logging.WARNING):
        async with serving(host) as client:
            assert await answered_by(client, method, request_path) is None
            assert (await answer(client, "GET", "/beta/own"))[0] == 404
            assert (await answer(client, "GET", "/api/beta/own"))[0] == 404
            listed = await (await client.get("/extensions")).json()
            files = [
                (await client.get(f"/extensions/{pack}/{pack}.js")).status
                for pack in ("alpha", "beta")
            ]
    assert "Pack 'beta'" in caplog.text and "is already served" in caplog.text
    assert "betaNode" not in nodes.NODE_CLASS_MAPPINGS
    assert "/extensions/alpha/alpha.js" in listed
    assert "/extensions/beta/beta.js" not in listed
    assert files == [200, 404]


@pytest.mark.asyncio
async def test_a_route_under_a_disabled_packs_web_folder_address_still_answers(host, load):
    async def other_answers(request):
        return web.Response(text="other")

    assert await load("beta", [("GET", "/system_stats")], web=True)
    host.app.router.add_post("/extensions/beta/action", other_answers)

    async with serving(host) as client:
        assert (await client.get("/extensions/beta/beta.js")).status == 404
        assert await (await client.post("/extensions/beta/action")).text() == "other"


@pytest.mark.asyncio
async def test_a_route_a_sub_application_matched_by_domain_serves_disables_the_pack(
    host, load, caplog
):
    async def other_answers(request):
        return web.Response(text="other")

    assert await load("alpha", [("GET", "/alpha/list")])
    other = web.Application()
    other.router.add_get("/alpha/list", other_answers)
    host.app.add_domain("other.example", other)

    with caplog.at_level(logging.WARNING):
        async with serving(host) as client:
            there = await client.get("/alpha/list", headers={"Host": "other.example"})
            assert await there.text() == "other"
            assert (await answer(client, "GET", "/alpha/list"))[0] == 404
    assert "Pack 'alpha'" in caplog.text and "is already served" in caplog.text


@pytest.mark.asyncio
async def test_a_get_route_whose_head_requests_are_served_already_disables_the_pack(
    host, load, caplog
):
    async def other_answers(request):
        return web.Response(headers={"X-By": "other"})

    assert await load("alpha", [("GET", "/alpha/list")])
    host.app.router.add_head("/alpha/list", other_answers)

    with caplog.at_level(logging.WARNING):
        async with serving(host) as client:
            assert (await client.head("/alpha/list")).headers["X-By"] == "other"
    assert "Pack 'alpha'" in caplog.text and "is already served" in caplog.text


@pytest.mark.asyncio
async def test_comfyui_starts_with_a_web_root_it_cannot_list_when_no_pack_has_routes(
    host, load, tmp_path
):
    host.web_root = str(tmp_path / "web_root")
    (tmp_path / "web_root").mkdir()
    (tmp_path / "web_root" / "index.html").write_text("front")
    (tmp_path / "web_root").chmod(0o311)
    if os.access(host.web_root, os.R_OK):
        pytest.skip("this user can list any folder")
    assert await load("alpha", [])

    try:
        async with serving(host) as client:
            assert await (await client.get("/index.html")).text() == "front"
    finally:
        (tmp_path / "web_root").chmod(0o755)


@pytest.mark.asyncio
async def test_a_route_with_an_empty_path_does_not_stop_comfyui_starting(host, load):
    assert await load("alpha", [("GET", "/alpha/list")])
    host.app.router.add_get("", lambda request: web.Response())

    async with serving(host) as client:
        assert await answered_by(client, "GET", "/alpha/list") == "alpha"


OLDER_API_ROUTE = """
from aiohttp import web
from server import PromptServer


@PromptServer.instance.routes.get("/shared/state")
async def state(request):
    return web.json_response({"pack": "older"})


NODE_CLASS_MAPPINGS = {}
"""


@pytest.mark.asyncio
async def test_a_route_an_older_api_pack_loaded_later_serves_disables_the_pack(
    host, load, tmp_path
):
    assert await load("beta", [("GET", "/shared/state")])
    (tmp_path / "older.py").write_text(OLDER_API_ROUTE)
    assert await nodes.load_custom_node(str(tmp_path / "older.py"))

    async with serving(host) as client:
        assert await answered_by(client, "GET", "/shared/state") == "older"
    assert "betaNode" not in nodes.NODE_CLASS_MAPPINGS


NO_HANDLER = """
class Extension(Extension):
    async def get_routes(self):
        return [{"method": "GET", "path": "/beta/list"}]


async def comfy_entrypoint():
    return Extension()
"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "routes, extra, message",
    [
        ([("GET", "/beta/ok"), ("PSOT", "/beta/save")], "", "'PSOT' is not one of"),
        ([("GET", "/beta/ok"), ("get", "/beta/save")], "", "'get' is not one of"),
        ([("GET", "/beta/ok"), ("GET", "beta/save")], "", "does not start with '/'"),
        ([("GET", "/beta/ok"), ("GET", None)], "", "None does not start with '/'"),
        ([("GET", "/beta/ok"), ("GET", "/beta/{x")], "", "route path '/beta/{x'"),
        (
            [("GET", "/beta/ok"), ("GET", "/beta/{x:(?P<y>a)?b}")],
            "",
            "has a named group of its own",
        ),
        (
            [("GET", "/beta/ok"), ("GET", "/beta/ok")],
            "",
            "overlaps another of its routes",
        ),
        ([("GET", "/beta/ok"), ("GET", "/api/beta/ok")], "", "overlaps another of its"),
        ([("GET", "/beta/{a}"), ("GET", "/beta/{b}")], "", "overlaps another of its"),
        ([], NO_HANDLER, "GET /beta/list has no handler"),
    ],
)
async def test_a_bad_or_repeated_route_disables_the_pack_and_comfyui_starts(
    host, load, routes, extra, message, caplog
):
    with caplog.at_level(logging.WARNING):
        assert not await load("beta", routes, extra=extra)
    assert "Pack 'beta'" in caplog.text and message in caplog.text

    async with serving(host) as client:
        assert (await answer(client, "GET", "/beta/ok"))[0] == 404
        assert (await answer(client, "GET", "/beta/list"))[0] == 404
        assert (await answer(client, "GET", "/system_stats"))[0] == 200


@pytest.mark.asyncio
async def test_a_pack_whose_id_a_loaded_pack_holds_is_disabled_whatever_its_folder(
    host, load, caplog
):
    assert await load("alpha", [("GET", "/alpha/list")], name="comfyui-alpha")
    assert await load(
        "alpha", [("GET", "/folder/list")], root="same_folder", node_id="A2"
    )
    with caplog.at_level(logging.WARNING):
        assert not await load(
            "alpha_fork", [("GET", "/fork/list")], name="ComfyUI-Alpha", node_id="A3"
        )
    assert "its id 'comfyui-alpha' is already held by" in caplog.text
    assert "A3" not in nodes.NODE_CLASS_MAPPINGS

    async with serving(host) as client:
        assert (await answer(client, "GET", "/alpha/list"))[1]["pack"] == "alpha"
        assert (await answer(client, "GET", "/folder/list"))[0] == 200
        assert (await answer(client, "GET", "/fork/list"))[0] == 404


@pytest.mark.asyncio
async def test_folders_whose_names_differ_after_a_dot_hold_two_packs(host, load):
    assert await load("Pack-1.5", [("GET", "/one")], name="pack-one", node_id="One")
    assert await load("Pack-1.6", [("GET", "/two")], name="pack-two", node_id="Two")

    async with serving(host) as client:
        assert await answered_by(client, "GET", "/one") == "Pack-1.5"
        assert await answered_by(client, "GET", "/two") == "Pack-1.6"


@pytest.mark.asyncio
async def test_a_pack_loaded_after_comfyui_starts_serving_cannot_have_routes(
    host, load, caplog
):
    assert await load("alpha", [("GET", "/alpha/list")])
    async with serving(host):
        with caplog.at_level(logging.WARNING):
            assert not await load("beta", [("GET", "/beta/list")])
        assert await load("gamma", [])
    assert "before ComfyUI starts serving" in caplog.text


@pytest.mark.asyncio
async def test_a_pack_loaded_after_comfyui_serves_no_pack_cannot_have_routes(
    host, load
):
    async with serving(host):
        assert not await load("beta", [("GET", "/beta/list")])


@pytest.mark.asyncio
async def test_a_pack_with_routes_loads_when_comfyui_runs_without_a_server(
    host, load, monkeypatch
):
    monkeypatch.delattr(server.PromptServer, "instance")
    assert await load("beta", [("GET", "/beta/list")])
    assert "betaNode" in nodes.NODE_CLASS_MAPPINGS


FAILS_ON_LOAD = """
async def _on_load(self):
    raise RuntimeError("second copy fails")


Extension.on_load = _on_load
"""


@pytest.mark.asyncio
async def test_a_failed_second_load_of_a_loaded_pack_leaves_it_working(host, load):
    assert await load("alpha", [("GET", "/alpha/list")])
    assert not await load("alpha", [("GET", "/alpha/list")], extra=FAILS_ON_LOAD)

    assert "alphaNode" in nodes.NODE_CLASS_MAPPINGS
    async with serving(host) as client:
        assert (await answer(client, "GET", "/alpha/list"))[1]["pack"] == "alpha"


SCHEMA_FAILS_WHEN_BUILT_AGAIN = """
class Flaky(io.ComfyNode):
    built = 0

    @classmethod
    def define_schema(cls):
        cls.built += 1
        if cls.built > 1:
            raise RuntimeError("schema fails when built again")
        return io.Schema(node_id="Flaky", outputs=[io.Int.Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput(1)


EXTRA_NODES = [Flaky]
"""


@pytest.mark.asyncio
async def test_a_pack_that_fails_after_it_was_recorded_leaves_no_node_route_or_folder(
    host, load
):
    assert not await load(
        "beta",
        [("GET", "/beta/list")],
        schema=DISPLAYED.format(node_id="BetaNode"),
        extra=SCHEMA_FAILS_WHEN_BUILT_AGAIN,
        web=True,
    )

    assert "BetaNode" not in nodes.NODE_CLASS_MAPPINGS
    assert "BetaNode" not in nodes.NODE_DISPLAY_NAME_MAPPINGS
    assert (await valid("BetaNode"))[1]["type"] == "missing_node_type"
    assert "beta" not in nodes.LOADED_MODULE_DIRS
    assert "beta" not in nodes.EXTENSION_WEB_DIRS
    assert await load("gamma", [("GET", "/beta/list")])


@pytest.mark.asyncio
async def test_a_pack_whose_schema_fails_records_no_route_and_leaves_no_folder(
    host, load
):
    broken = 'raise RuntimeError("broken schema")'
    assert not await load("beta", [("GET", "/beta/list")], schema=broken, web=True)
    assert "beta" not in nodes.LOADED_MODULE_DIRS
    assert "beta" not in nodes.EXTENSION_WEB_DIRS
    assert await load("gamma", [("GET", "/beta/list")])

    async with serving(host) as client:
        assert (await answer(client, "GET", "/beta/list"))[1]["pack"] == "gamma"


BROKEN_SECOND_NODE = """
class Broken(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        raise RuntimeError("broken schema")


EXTRA_NODES = [Broken]
"""

ORIGINAL = """
class Original:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    RETURN_TYPES = ("INT",)
    FUNCTION = "f"
    def f(self):
        return (1,)

NODE_CLASS_MAPPINGS = {"BetaNode": Original}
"""


@pytest.mark.asyncio
async def test_a_pack_with_a_broken_schema_leaves_the_node_it_would_replace(
    host, load, tmp_path
):
    (tmp_path / "original.py").write_text(ORIGINAL)
    assert await nodes.load_custom_node(str(tmp_path / "original.py"))
    original = nodes.NODE_CLASS_MAPPINGS["BetaNode"]
    assert not await load(
        "beta",
        [],
        schema=DISPLAYED.format(node_id="BetaNode"),
        extra=BROKEN_SECOND_NODE,
    )
    assert nodes.NODE_CLASS_MAPPINGS.get("BetaNode") is original


@pytest.mark.asyncio
async def test_a_disabled_pack_takes_away_its_nodes_and_folders_and_nothing_of_another(
    host, load, tmp_path
):
    assert await load(
        "alpha", [("GET", "/system_stats")], name="first", node_id="FirstNode", web=True
    )
    older = tmp_path / "other" / "alpha"
    (older / "web").mkdir(parents=True)
    (older / "__init__.py").write_text(WEB + ORIGINAL.replace("BetaNode", "FirstNode"))
    assert await nodes.load_custom_node(str(older))
    assert await load("beta", [], schema=DISPLAYED.format(node_id="BetaNode"), web=True)
    second = nodes.NODE_CLASS_MAPPINGS["FirstNode"]

    async with serving(host):
        pass

    assert nodes.NODE_CLASS_MAPPINGS["FirstNode"] is second
    assert nodes.LOADED_MODULE_DIRS["alpha"] == str(tmp_path / "other" / "alpha")
    assert nodes.EXTENSION_WEB_DIRS["alpha"] == str(
        tmp_path / "other" / "alpha" / "web"
    )
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["BetaNode"] == "Shown"
    assert "beta" in nodes.LOADED_MODULE_DIRS and "beta" in nodes.EXTENSION_WEB_DIRS


@pytest.mark.asyncio
async def test_a_refused_pack_does_not_serve_its_browser_code(host, load):
    assert await load("alpha", [("GET", "/shared")])
    assert not await load("beta", [("GET", "/shared")], web=True)

    async with serving(host) as client:
        listed = await (await client.get("/extensions")).json()
    assert not any("/extensions/beta/" in p for p in listed)
    assert "beta" not in nodes.LOADED_MODULE_DIRS


@pytest.mark.asyncio
async def test_a_refused_same_identity_pack_does_not_take_the_loaded_packs_folders(
    host, load
):
    assert await load("alpha", [("GET", "/one")], node_id="A1", web=True)
    first = dict(nodes.LOADED_MODULE_DIRS), dict(nodes.EXTENSION_WEB_DIRS)
    assert not await load(
        "alpha", [("GET", "/two")], root="copy", node_id="A2", web=True
    )

    assert (nodes.LOADED_MODULE_DIRS, nodes.EXTENSION_WEB_DIRS) == first
    async with serving(host) as client:
        served = await (await client.get("/extensions/alpha/alpha.js")).text()
    assert served == "// alpha"


BAD_TOML = '[project]\nname = "gamma"\nversion = 1.0\n'


@pytest.mark.asyncio
@pytest.mark.parametrize("toml", ["[project\nname = 'x'\n", BAD_TOML])
async def test_a_pack_with_a_pyproject_the_loader_tolerates_loads_named_by_its_folder(
    host, load, tmp_path, toml
):
    (tmp_path / "gamma").mkdir()
    (tmp_path / "gamma" / "pyproject.toml").write_text(toml)
    assert await load("gamma", [])
    assert not await load("gamma", [], root="copy", name="gamma", node_id="Copy")


@pytest.mark.asyncio
async def test_a_pack_disabled_at_start_gives_back_the_node_it_replaced(
    host, load, tmp_path
):
    (tmp_path / "original.py").write_text(ORIGINAL)
    assert await nodes.load_custom_node(str(tmp_path / "original.py"))
    original = nodes.NODE_CLASS_MAPPINGS["BetaNode"]
    assert await load(
        "beta", [("GET", "/system_stats")], schema=DISPLAYED.format(node_id="BetaNode")
    )

    async with serving(host):
        pass

    assert nodes.NODE_CLASS_MAPPINGS.get("BetaNode") is original
    assert "BetaNode" not in nodes.NODE_DISPLAY_NAME_MAPPINGS


@pytest.mark.asyncio
async def test_a_pack_that_fails_while_its_nodes_register_gives_back_the_node_it_replaced(
    host, load, tmp_path
):
    (tmp_path / "original.py").write_text(ORIGINAL)
    assert await nodes.load_custom_node(str(tmp_path / "original.py"))
    original = nodes.NODE_CLASS_MAPPINGS["BetaNode"]
    assert not await load(
        "beta",
        [],
        schema=DISPLAYED.format(node_id="BetaNode"),
        extra=SCHEMA_FAILS_WHEN_BUILT_AGAIN,
    )

    assert nodes.NODE_CLASS_MAPPINGS.get("BetaNode") is original


@pytest.mark.asyncio
async def test_a_web_root_entry_aiohttp_cannot_parse_does_not_stop_comfyui_starting(
    host, load, tmp_path
):
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "notes{draft").mkdir()
    host.web_root = str(tmp_path / "web")
    assert await load("alpha", [("GET", "/alpha/list")])

    async with serving(host) as client:
        assert (await answer(client, "GET", "/alpha/list"))[1]["pack"] == "alpha"


ROUTES_NOT_A_LIST = """
async def _get_routes(self):
    return None


Extension.get_routes = _get_routes
"""

ROUTE_NOT_A_DICT = """
from aiohttp import web


async def _get_routes(self):
    return [web.get("/beta/list", echo)]


Extension.get_routes = _get_routes
"""


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", [ROUTES_NOT_A_LIST, ROUTE_NOT_A_DICT])
async def test_a_pack_whose_routes_are_not_route_dicts_is_refused_and_leaves_no_folder(
    host, load, extra, caplog
):
    assert not await load("beta", [], web=True, extra=extra)

    assert "beta" not in nodes.LOADED_MODULE_DIRS
    assert "beta" not in nodes.EXTENSION_WEB_DIRS
    assert "betaNode" not in nodes.NODE_CLASS_MAPPINGS
    assert "Pack 'beta'" in caplog.text


STATUS = """
async def teapot(request):
    return {"status": 418, "headers": {"x-probe": "1"}, "body": b"short and stout"}


class Extension(Extension):
    async def get_routes(self):
        return [{"method": "GET", "path": "/beta/teapot", "handler": teapot}]


async def comfy_entrypoint():
    return Extension()
"""


@pytest.mark.asyncio
async def test_a_route_handlers_status_and_headers_reach_the_client(host, load):
    assert await load("beta", [], extra=STATUS)

    async with serving(host) as client:
        response = await client.get("/api/beta/teapot")
        assert response.status == 418 and response.headers["x-probe"] == "1"
        assert await response.read() == b"short and stout"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["beta-named", "Beta-Named"])
async def test_a_refused_pack_gives_back_the_web_folder_its_pyproject_names(
    host, load, tmp_path, monkeypatch, name
):
    assert await load("alpha", [("GET", "/shared")])
    pack = tmp_path / "beta"
    write_pack(pack, [("GET", "/shared")])
    (pack / "web").mkdir()
    (pack / "web" / "beta.js").write_text("// beta")
    (pack / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\n\n[tool.comfy]\nweb = "web"\n'
    )
    monkeypatch.setitem(sys.modules, str(pack).replace(".", "_x_"), None)

    assert not await nodes.load_custom_node(str(pack))
    assert name not in nodes.EXTENSION_WEB_DIRS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pyproject, reason",
    [
        (None, "it has no pyproject.toml"),
        ("[project]\nversion = '1.0'\n", "has no [project] name"),
        ("[project]\nname = 'beta pack'\n", "is not a registry id"),
        ("[project]\nname = 'comfy-core'\n", "is reserved"),
        ("[project]\nname = 'Color'\n", "is a field every node saves"),
    ],
)
async def test_a_v4_pack_whose_pyproject_gives_no_valid_id_is_refused_and_leaves_nothing(
    host, load, tmp_path, monkeypatch, caplog, pyproject, reason
):
    pack = tmp_path / "beta"
    write_pack(pack, [("GET", "/beta/own")], web=True)
    if pyproject is None:
        (pack / "pyproject.toml").unlink()
    else:
        (pack / "pyproject.toml").write_text(pyproject)
    monkeypatch.setitem(sys.modules, str(pack).replace(".", "_x_"), None)

    with caplog.at_level(logging.ERROR):
        assert not await nodes.load_custom_node(str(pack))

    assert f"V4 pack {pack} is not loaded: " in caplog.text and reason in caplog.text
    assert "betaNode" not in nodes.NODE_CLASS_MAPPINGS
    assert "beta" not in nodes.LOADED_MODULE_DIRS
    assert "beta" not in nodes.EXTENSION_WEB_DIRS
    async with serving(host) as client:
        assert (await answer(client, "GET", "/beta/own"))[0] == 404


@pytest.mark.asyncio
async def test_a_one_file_v4_pack_is_refused_for_it_has_no_pyproject(
    host, tmp_path, monkeypatch, caplog
):
    pack = tmp_path / "beta.py"
    pack.write_text(
        PACK.format(name="beta", routes=[], schema=SCHEMA.format(node_id="betaNode"), extra="")
    )
    monkeypatch.setitem(sys.modules, str(pack.with_suffix("")), None)

    with caplog.at_level(logging.ERROR):
        assert not await nodes.load_custom_node(str(pack))

    assert "it has no pyproject.toml" in caplog.text
    assert "betaNode" not in nodes.NODE_CLASS_MAPPINGS


@pytest.mark.asyncio
async def test_a_pack_whose_id_a_pack_verified_before_loading_holds_is_disabled(
    host, load, tmp_path, monkeypatch, caplog
):
    holder = tmp_path / "web_side" / "alpha"
    beta = tmp_path / "beta"
    monkeypatch.setattr(
        extension_packs,
        "_checked",
        (frozenset({str(holder), str(beta)}), ({str(holder): ("alpha", "1.0")}, {})),
    )

    with caplog.at_level(logging.WARNING):
        assert not await load("beta", [], name="alpha", node_id="Refused")
        assert await load("alpha", [], root="web_side", node_id="Holder")

    assert f"its id 'alpha' is already held by {holder}" in caplog.text
    assert "Refused" not in nodes.NODE_CLASS_MAPPINGS
    assert "Holder" in nodes.NODE_CLASS_MAPPINGS


@pytest.mark.asyncio
async def test_a_pack_disabled_at_start_gives_back_a_v4_packs_node_it_replaced(
    host, load
):
    assert await load("alpha", [("GET", "/alpha/own")], node_id="Shared")
    first = nodes.NODE_CLASS_MAPPINGS["Shared"]
    assert await load("beta", [("GET", "/system_stats")], node_id="Shared")

    async with serving(host):
        pass

    assert nodes.NODE_CLASS_MAPPINGS.get("Shared") is first


@pytest.mark.asyncio
async def test_two_packs_disabled_at_start_leave_neither_ones_node(host, load):
    assert await load("alpha", [("GET", "/system_stats")], node_id="Shared")
    assert await load("beta", [("GET", "/settings")], node_id="Shared")

    async with serving(host):
        pass

    assert "Shared" not in nodes.NODE_CLASS_MAPPINGS


@pytest.mark.asyncio
async def test_two_packs_disabled_at_start_give_back_the_node_the_first_replaced(
    host, load, tmp_path
):
    (tmp_path / "original.py").write_text(ORIGINAL)
    assert await nodes.load_custom_node(str(tmp_path / "original.py"))
    original = nodes.NODE_CLASS_MAPPINGS["BetaNode"]
    assert await load("alpha", [("GET", "/system_stats")], node_id="BetaNode")
    assert await load("beta", [("GET", "/settings")], node_id="BetaNode")

    async with serving(host):
        pass

    assert nodes.NODE_CLASS_MAPPINGS.get("BetaNode") is original


SHARED = """
from comfy_api.v4 import io


class SharedNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="SharedNode", display_name="Shared", outputs=[io.Int.Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput(1)
"""


@pytest.mark.asyncio
async def test_a_pack_disabled_at_start_leaves_a_node_class_another_loaded_pack_also_lists(
    host, load, tmp_path, monkeypatch
):
    (tmp_path / "shared_nodes.py").write_text(SHARED)
    spec = importlib.util.spec_from_file_location(
        "shared_nodes", tmp_path / "shared_nodes.py"
    )
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    monkeypatch.setitem(sys.modules, "shared_nodes", shared)
    extra = "import shared_nodes\nEXTRA_NODES = [shared_nodes.SharedNode]"
    assert await load("alpha", [("GET", "/system_stats")], extra=extra)
    assert await load("beta", [("GET", "/beta/own")], extra=extra)

    async with serving(host) as client:
        assert (await answer(client, "GET", "/beta/own"))[1]["pack"] == "beta"

    assert "betaNode" in nodes.NODE_CLASS_MAPPINGS
    assert nodes.NODE_CLASS_MAPPINGS.get("SharedNode") is shared.SharedNode


@pytest.mark.asyncio
async def test_a_refused_single_file_pack_does_not_serve_its_browser_code(
    host, load, tmp_path, monkeypatch
):
    assert await load("alpha", [("GET", "/shared")])
    root = tmp_path / "files"
    (root / "web").mkdir(parents=True)
    (root / "web" / "beta.js").write_text("// beta")
    folder = tmp_path / "beta_src"
    write_pack(folder, [("GET", "/shared")], extra='WEB_DIRECTORY = "./web"')
    pack = root / "beta.py"
    pack.write_text((folder / "__init__.py").read_text())
    monkeypatch.setitem(sys.modules, str(pack.with_suffix("")), None)

    assert not await nodes.load_custom_node(str(pack))

    assert str(root / "web") not in nodes.EXTENSION_WEB_DIRS.values()


@pytest.mark.asyncio
async def test_a_disabled_pack_gives_back_the_display_name_of_the_node_it_replaced(
    host, load, tmp_path
):
    (tmp_path / "original.py").write_text(
        """
class Original:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    RETURN_TYPES = ("INT",)
    FUNCTION = "f"
    def f(self):
        return (1,)

NODE_CLASS_MAPPINGS = {"BetaNode": Original}
NODE_DISPLAY_NAME_MAPPINGS = {"BetaNode": "Original's name"}
"""
    )
    assert await nodes.load_custom_node(str(tmp_path / "original.py"))
    assert await load(
        "beta",
        [("GET", "/system_stats")],
        schema="return io.Schema(node_id='BetaNode', display_name='Beta', outputs=[io.Int.Output()])",
    )
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["BetaNode"] == "Beta"

    async with serving(host):
        pass

    assert nodes.NODE_DISPLAY_NAME_MAPPINGS.get("BetaNode") == "Original's name"


@pytest.mark.asyncio
async def test_a_pack_that_fails_after_it_was_recorded_leaves_an_earlier_pack(
    host, load
):
    assert await load("first", [("GET", "/first/list")])
    assert not await load(
        "later", [("GET", "/later/list")], extra=SCHEMA_FAILS_WHEN_BUILT_AGAIN
    )

    assert "firstNode" in nodes.NODE_CLASS_MAPPINGS
    async with serving(host) as client:
        assert (await answer(client, "GET", "/first/list"))[1]["pack"] == "first"


@pytest.mark.asyncio
async def test_a_pack_disabled_at_start_gives_back_its_replaced_node_after_another_failed(
    host, load
):
    assert await load("original", [], node_id="Shared")
    first = nodes.NODE_CLASS_MAPPINGS["Shared"]
    assert await load("override", [("GET", "/system_stats")], node_id="Shared")
    assert not await load(
        "unrelated", [("GET", "/unrelated/a")], extra=SCHEMA_FAILS_WHEN_BUILT_AGAIN
    )

    async with serving(host):
        pass

    assert nodes.NODE_CLASS_MAPPINGS["Shared"] is first


RAISING_HANDLER = """
import asyncio

from aiohttp import web

from comfy_api.v4 import InterruptProcessingException


async def raising(request):
    raise {error}


class Extension(Extension):
    async def get_routes(self):
        return [{{"method": "GET", "path": "/beta/stop", "handler": raising}}]
"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        "InterruptProcessingException()",
        "web.HTTPFound('/')",
        "RuntimeError('failed')",
        "asyncio.TimeoutError()",
    ],
    ids=["a cancel", "an aiohttp response", "an error", "a timeout"],
)
async def test_a_handler_that_raises_gets_a_500(host, load, error):
    assert await load("beta", [], extra=RAISING_HANDLER.format(error=error))

    async with serving(host) as client:
        assert (await answer(client, "GET", "/beta/stop"))[0] == 500
        assert (await answer(client, "GET", "/api/beta/stop"))[0] == 500


KEPT_ROUTE = """
ROUTE = {"method": "GET", "path": "/beta/kept", "handler": echo}


class Extension(Extension):
    async def get_routes(self):
        return [ROUTE]
"""


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [{"path": "no-slash"}, {"path": "/system_stats"}])
async def test_a_route_the_pack_changes_after_loading_is_served_as_it_was_checked(
    host, load, change, tmp_path
):
    assert await load("beta", [], extra=KEPT_ROUTE)
    loaded_module(tmp_path / "beta").ROUTE.update(change)

    async with serving(host) as client:
        assert (await answer(client, "GET", "/beta/kept"))[1]["pack"] == "beta"


@pytest.mark.asyncio
async def test_flushing_a_pack_answers_404_to_its_routes_and_leaves_another_packs(
    host, load
):
    assert await load("alpha", [("GET", "/alpha/{item}")])
    assert await load("beta", [("GET", "/beta/list")])

    async with serving(host) as client:
        _packs._flush_pack_resources([("alpha", "alpha"), ("not-loaded", "not-loaded")])
        flushed = [
            (await answer(client, "GET", path))[0]
            for path in ("/alpha/1", "/api/alpha/1")
        ]
        beta = await answered_by(client, "GET", "/beta/list")
        _packs._flush_all_pack_resources()
        all_flushed = (await answer(client, "GET", "/beta/list"))[0]

    assert flushed == [404, 404] and beta == "beta" and all_flushed == 404


@pytest.mark.asyncio
async def test_flushing_a_pack_removes_its_nodes_and_folders_and_nothing_of_another_pack(
    host, load
):
    alpha = DISPLAYED.format(node_id="AlphaNode")
    assert await load("alpha", [("GET", "/alpha/list")], schema=alpha, web=True)
    beta = DISPLAYED.format(node_id="BetaNode")
    assert await load("beta", [("GET", "/beta/list")], schema=beta, web=True)

    async with serving(host) as client:
        _packs._flush_pack_resources([("alpha", "alpha")])

        assert "AlphaNode" not in nodes.NODE_CLASS_MAPPINGS
        assert "AlphaNode" not in nodes.NODE_DISPLAY_NAME_MAPPINGS
        assert (await valid("AlphaNode"))[1]["type"] == "missing_node_type"
        assert "alpha" not in nodes.LOADED_MODULE_DIRS
        assert "alpha" not in nodes.EXTENSION_WEB_DIRS
        assert (await valid("BetaNode"))[0]
        assert nodes.NODE_DISPLAY_NAME_MAPPINGS["BetaNode"] == "Shown"
        assert "beta" in nodes.LOADED_MODULE_DIRS and "beta" in nodes.EXTENSION_WEB_DIRS
        listed = await (await client.get("/extensions")).json()
        assert not any(path.startswith("/extensions/alpha/") for path in listed)

        assert await load("alpha", [], schema=alpha, web=True)
        assert (await valid("AlphaNode"))[0]
        assert "alpha" in nodes.EXTENSION_WEB_DIRS


@pytest.mark.asyncio
async def test_a_flush_leaves_entries_another_pack_has_since_taken(
    host, load, tmp_path
):
    assert await load("alpha", [], name="first", node_id="SharedNode")
    assert await load("alpha", [], root="other", name="second", node_id="SharedNode")
    second = nodes.NODE_CLASS_MAPPINGS["SharedNode"]

    _packs._flush_pack_resources([("first", "alpha")])

    assert nodes.NODE_CLASS_MAPPINGS["SharedNode"] is second
    assert nodes.LOADED_MODULE_DIRS["alpha"] == str(tmp_path / "other" / "alpha")


@pytest.mark.asyncio
async def test_a_flush_takes_away_only_the_pack_with_the_named_id(host, load):
    assert await load("alpha", [], name="first", node_id="AlphaNode")
    assert await load("alpha", [], root="other", name="second", node_id="BetaNode")

    _packs._flush_pack_resources([("first", "alpha")])

    assert "AlphaNode" not in nodes.NODE_CLASS_MAPPINGS
    assert "BetaNode" in nodes.NODE_CLASS_MAPPINGS


@pytest.mark.asyncio
async def test_the_web_folder_of_a_pack_taken_away_is_no_longer_served(host, load):
    assert await load("alpha", [], web=True)
    assert await load("beta", [], web=True)

    async with serving(host) as client:
        before = [
            (await client.get(f"/extensions/{name}/{name}.js")).status
            for name in ("alpha", "beta")
        ]
        _packs._flush_pack_resources([("alpha", "alpha")])
        after = [
            (await client.get(f"/extensions/{name}/{name}.js")).status
            for name in ("alpha", "beta")
        ]

    assert before == [200, 200]
    assert after == [404, 200]


@pytest.mark.asyncio
async def test_a_web_folder_that_is_not_a_v4_packs_is_served_as_before(
    host, load, tmp_path, monkeypatch
):
    other = tmp_path / "other_web"
    other.mkdir()
    (other / "other.js").write_text("")
    monkeypatch.setitem(nodes.EXTENSION_WEB_DIRS, "other", str(other))
    assert await load("alpha", [], web=True)

    async with serving(host) as client:
        del nodes.EXTENSION_WEB_DIRS["other"]
        response = await client.get("/extensions/other/other.js")

    assert response.status == 200


@pytest.mark.asyncio
async def test_a_web_folder_an_older_api_pack_registered_under_a_v4_packs_key_is_served(
    host, load, tmp_path, monkeypatch
):
    other = tmp_path / "other_web"
    other.mkdir()
    (other / "other.js").write_text("")
    assert await load("alpha", [], web=True)
    monkeypatch.setitem(nodes.EXTENSION_WEB_DIRS, "alpha", str(other))

    async with serving(host) as client:
        response = await client.get("/extensions/alpha/other.js")

    assert response.status == 200


REGISTERS = """
from comfy_api.v4 import CacheProvider, engine
from comfy_api.v4.util import folders

folders.add_model_folder_path("v4_registered", __file__ + ".models")


class Provider(CacheProvider):
    async def on_lookup(self, context):
        return None

    async def on_store(self, context, value):
        pass


PROVIDER = Provider()


async def on_load(self):
    await engine.caching.register_provider(PROVIDER)
    await engine.node_replacement.register(io.NodeReplace("LoadImage", "V4RemovedNode"))
    {then}


Extension.on_load = on_load
"""


@pytest.fixture
def registered(host, monkeypatch):
    """Whether a pack's cache provider, node replacement and model folder are registered with
    ComfyUI now."""
    monkeypatch.setattr(
        folder_paths, "folder_names_and_paths", dict(folder_paths.folder_names_and_paths)
    )
    monkeypatch.setattr(cache_provider, "_providers", [])
    monkeypatch.setattr(cache_provider, "_providers_snapshot", ())
    monkeypatch.setattr(_services, "_provider_uses", [])

    def registered():
        providers = [type(p).__name__ for p in cache_provider._get_cache_providers()]
        replaced = host.node_replace_manager.has_replacement("V4RemovedNode")
        return providers, replaced, "v4_registered" in folder_paths.folder_names_and_paths

    return registered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "routes, then",
    [
        pytest.param([("GET", "/alpha/list")], "pass", id="its route is served already"),
        pytest.param([], "raise RuntimeError('on_load fails')", id="its on_load fails"),
    ],
)
async def test_a_pack_that_does_not_load_leaves_nothing_it_registered(
    host, load, registered, routes, then
):
    assert await load("alpha", [("GET", "/alpha/list")])

    assert not await load("beta", routes, extra=REGISTERS.format(then=then))

    assert registered() == ([], False, False)


@pytest.mark.asyncio
async def test_a_pack_taken_away_takes_what_it_registered_with_it(
    host, load, registered, tmp_path
):
    assert await load("alpha", [], extra=REGISTERS.format(then="pass"))
    assert registered() == (["Provider"], True, True)

    _packs._remove_pack(str(tmp_path / "alpha"))

    assert registered() == ([], False, False)


@pytest.mark.asyncio
async def test_a_pack_disabled_at_start_takes_what_it_registered_with_it(
    host, load, registered
):
    assert await load(
        "alpha", [("GET", "/prompt")], extra=REGISTERS.format(then="pass")
    )
    assert registered() == (["Provider"], True, True)

    host.add_routes()

    assert registered() == ([], False, False)


@pytest.mark.asyncio
async def test_what_a_loaded_pack_registers_later_stays_when_another_pack_fails(
    host, load, registered
):
    assert await load("alpha", [])
    provider = type("Later", (v4.CacheProvider,), {"on_lookup": None, "on_store": None})()
    await v4.engine.caching.register_provider(provider)
    await v4.engine.node_replacement.register(v4.io.NodeReplace("LoadImage", "V4RemovedNode"))
    folders.add_model_folder_path("v4_registered", "later")

    assert not await load("beta", [], extra="raise RuntimeError('import fails')")

    assert registered() == (["Later"], True, True)


FROM_A_THREAD = """
import asyncio
import threading


async def on_load(self):
    register = engine.caching.register_provider(PROVIDER)
    if {own}:
        thread = threading.Thread(target=asyncio.run, args=(register,))
        thread.start()
        thread.join()
    else:
        await asyncio.get_running_loop().run_in_executor(None, asyncio.run, register)
    raise RuntimeError("on_load fails")


Extension.on_load = on_load
"""

LATER = """
import asyncio

LATER = []


async def later():
    await asyncio.sleep(0)
    await engine.caching.register_provider(PROVIDER)


async def on_load(self):
    LATER.append(asyncio.create_task(later()))
    raise RuntimeError("on_load fails")


Extension.on_load = on_load
"""


@pytest.mark.asyncio
async def test_a_pack_that_does_not_load_leaves_nothing_a_task_it_started_registers(
    host, load, registered, tmp_path
):
    assert not await load("beta", [], extra=REGISTERS.format(then="pass") + LATER)
    await loaded_module(tmp_path / "beta").LATER[0]

    assert registered() == ([], False, False)


@pytest.mark.asyncio
@pytest.mark.parametrize("own", [True, False], ids=["its own", "the loop's executor"])
async def test_what_is_registered_on_a_thread_that_is_not_the_loaders_is_no_packs_and_stays(
    host, load, registered, own
):
    assert await load("alpha", [])

    assert not await load(
        "beta", [], extra=REGISTERS.format(then="pass") + FROM_A_THREAD.format(own=own)
    )

    assert registered() == (["Provider"], False, False)


SHARES_A_REPLACEMENT = """
import builtins

from comfy_api.v4 import engine


async def on_load(self):
    await engine.node_replacement.register(builtins.v4_shared_replacement)
    raise RuntimeError("on_load fails")


Extension.on_load = on_load
"""


@pytest.mark.asyncio
async def test_a_pack_that_does_not_load_leaves_a_replacement_registered_before_it(
    host, load, monkeypatch
):
    shared = v4.io.NodeReplace("LoadImage", "V4RemovedNode")
    monkeypatch.setattr(builtins, "v4_shared_replacement", shared, raising=False)
    host.node_replace_manager.register(shared)

    assert not await load("beta", [], extra=SHARES_A_REPLACEMENT)

    assert host.node_replace_manager.has_replacement("V4RemovedNode")


SHARES_A_PROVIDER = """
import builtins

from comfy_api.v4 import engine


async def on_load(self):
    await engine.caching.register_provider({provider})
    {then}


Extension.on_load = on_load
"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    ["builtins.v4_shared_provider", "type(builtins.v4_shared_provider)()"],
    ids=["the same provider", "a provider equal to it"],
)
async def test_a_pack_that_does_not_load_leaves_a_provider_a_loaded_pack_registered(
    host, load, registered, monkeypatch, provider
):
    alike = {
        "on_lookup": None,
        "on_store": None,
        "__eq__": lambda self, other: type(other) is type(self),
        "__hash__": lambda self: 0,
    }
    shared = type("Shared", (v4.CacheProvider,), alike)()
    monkeypatch.setattr(builtins, "v4_shared_provider", shared, raising=False)
    pack = SHARES_A_PROVIDER.replace("{provider}", provider)
    assert await load("alpha", [], extra=pack.format(then="pass"))

    assert not await load("beta", [], extra=pack.format(then="raise RuntimeError('fails')"))

    assert registered()[0] == ["Shared"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    ["builtins.v4_shared_provider", "type(builtins.v4_shared_provider)()"],
    ids=["the same provider", "a provider equal to it"],
)
async def test_a_provider_two_packs_registered_stays_until_both_are_taken_away(
    host, load, registered, monkeypatch, tmp_path, provider
):
    alike = {
        "on_lookup": None,
        "on_store": None,
        "__eq__": lambda self, other: type(other) is type(self),
        "__hash__": lambda self: 0,
    }
    shared = type("Shared", (v4.CacheProvider,), alike)()
    monkeypatch.setattr(builtins, "v4_shared_provider", shared, raising=False)
    pack = SHARES_A_PROVIDER.replace("{provider}", provider).format(then="pass")
    assert await load("alpha", [], extra=pack)
    assert await load("beta", [], extra=pack)

    _packs._remove_pack(str(tmp_path / "alpha"))
    with_one_left = registered()[0]
    _packs._remove_pack(str(tmp_path / "beta"))

    assert (with_one_left, registered()[0]) == (["Shared"], [])


@pytest.mark.asyncio
@pytest.mark.parametrize("again", ["another pack", "nobody's pack"])
async def test_a_provider_unregistered_and_registered_again_is_not_taken_back_with_the_first_pack(
    host, load, registered, monkeypatch, tmp_path, again
):
    alike = {
        "on_lookup": None,
        "on_store": None,
        "__eq__": lambda self, other: type(other) is type(self),
        "__hash__": lambda self: 0,
    }
    shared = type("Shared", (v4.CacheProvider,), alike)()
    monkeypatch.setattr(builtins, "v4_shared_provider", shared, raising=False)
    pack = SHARES_A_PROVIDER.replace("{provider}", "builtins.v4_shared_provider")
    assert await load("alpha", [], extra=pack.format(then="pass"))
    await v4.engine.caching.unregister_provider(shared)
    if again == "another pack":
        assert await load("beta", [], extra=pack.format(then="pass"))
    else:
        await v4.engine.caching.register_provider(type(shared)())

    _packs._remove_pack(str(tmp_path / "alpha"))

    assert registered()[0] == ["Shared"]


LATER_STILL = """
import asyncio

LATER = []
GO = asyncio.Event()


async def later():
    await GO.wait()
    await engine.caching.register_provider(PROVIDER)


async def on_load(self):
    LATER.append(asyncio.create_task(later()))


Extension.on_load = on_load
"""


@pytest.mark.asyncio
async def test_a_pack_taken_away_leaves_nothing_a_task_it_started_registers_afterwards(
    host, load, registered, tmp_path
):
    assert await load("alpha", [], extra=REGISTERS.format(then="pass") + LATER_STILL)
    _packs._remove_pack(str(tmp_path / "alpha"))

    pack = loaded_module(tmp_path / "alpha")
    pack.GO.set()
    await pack.LATER[0]

    assert registered() == ([], False, False)
