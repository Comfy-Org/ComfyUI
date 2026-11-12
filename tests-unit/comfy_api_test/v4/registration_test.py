import importlib.util
import json
import logging
import sys
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import nodes
import server
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


@pytest.fixture
def host(monkeypatch):
    for attribute in (
        "NODE_CLASS_MAPPINGS",
        "NODE_DISPLAY_NAME_MAPPINGS",
        "LOADED_MODULE_DIRS",
        "EXTENSION_WEB_DIRS",
    ):
        monkeypatch.setattr(nodes, attribute, dict(getattr(nodes, attribute)))
    monkeypatch.setattr(server.PromptServer, "instance", None, raising=False)
    return server.PromptServer(None, MagicMock(enabled=False))


def write_pack(pack, routes, name=None, node_id=None, schema=None, extra="", web=False):
    pack.mkdir(parents=True, exist_ok=True)
    if name is not None:
        (pack / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n')
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
    assert "Pack 'beta'" in caplog.text and "is already served" in caplog.text
    assert "betaNode" not in nodes.NODE_CLASS_MAPPINGS
    assert "/extensions/alpha/alpha.js" in listed
    assert "/extensions/beta/beta.js" not in listed


@pytest.mark.asyncio
async def test_a_sub_application_matched_by_domain_does_not_stop_comfyui_starting(
    host, load
):
    assert await load("alpha", [("GET", "/alpha/list")])
    other = web.Application()
    other.router.add_get("/alpha/list", lambda request: web.Response())
    host.app.add_domain("other.example", other)

    async with serving(host) as client:
        assert await answered_by(client, "GET", "/alpha/list") == "alpha"


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
async def test_a_pack_is_disabled_only_if_a_loaded_pack_has_its_name_and_its_folder(
    host, load, caplog
):
    assert await load("alpha", [("GET", "/alpha/list")], name="comfyui-alpha")
    assert await load("alpha_fork", [("GET", "/fork/list")], name="comfyui-alpha")
    assert await load(
        "alpha", [("GET", "/folder/list")], root="same_folder", node_id="A2"
    )
    with caplog.at_level(logging.WARNING):
        assert not await load(
            "alpha",
            [("GET", "/copy/list")],
            root="copy",
            name="comfyui-alpha",
            node_id="A3",
        )
    assert "same name and folder name" in caplog.text

    async with serving(host) as client:
        assert (await answer(client, "GET", "/alpha/list"))[1]["pack"] == "alpha"
        assert (await answer(client, "GET", "/fork/list"))[1]["pack"] == "alpha_fork"
        assert (await answer(client, "GET", "/folder/list"))[0] == 200
        assert (await answer(client, "GET", "/copy/list"))[0] == 404


@pytest.mark.asyncio
async def test_folders_whose_names_differ_after_a_dot_hold_two_packs(host, load):
    assert await load("Pack-1.5", [("GET", "/one")], name="pack", node_id="One")
    assert await load("Pack-1.6", [("GET", "/two")], name="pack", node_id="Two")

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
async def test_a_refused_pack_gives_back_the_web_folder_its_pyproject_names(
    host, load, tmp_path, monkeypatch
):
    assert await load("alpha", [("GET", "/shared")])
    pack = tmp_path / "beta"
    write_pack(pack, [("GET", "/shared")])
    (pack / "web").mkdir()
    (pack / "web" / "beta.js").write_text("// beta")
    (pack / "pyproject.toml").write_text(
        '[project]\nname = "beta-named"\n\n[tool.comfy]\nweb = "web"\n'
    )
    monkeypatch.setitem(sys.modules, str(pack).replace(".", "_x_"), None)

    assert not await nodes.load_custom_node(str(pack))
    assert "beta-named" not in nodes.EXTENSION_WEB_DIRS


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


INTERRUPTED_HANDLER = """
from comfy_api.v4 import InterruptProcessingException


async def interrupted(request):
    raise InterruptProcessingException()


class Extension(Extension):
    async def get_routes(self):
        return [{"method": "GET", "path": "/beta/stop", "handler": interrupted}]
"""


@pytest.mark.asyncio
async def test_a_handler_stopped_by_a_cancel_gets_a_500(host, load):
    assert await load("beta", [], extra=INTERRUPTED_HANDLER)

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
    host, load, change
):
    assert await load("beta", [], extra=KEPT_ROUTE)
    [module] = [
        module
        for module in list(sys.modules.values())
        if getattr(module, "ROUTE", None) is not None
        and "beta" in getattr(module, "__file__", "")
    ]
    module.ROUTE.update(change)

    async with serving(host) as client:
        assert (await answer(client, "GET", "/beta/kept"))[1]["pack"] == "beta"
