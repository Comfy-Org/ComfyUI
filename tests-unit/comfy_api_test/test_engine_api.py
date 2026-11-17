"""The engine call: table, dispatcher, extensions, and both execution modes."""
import asyncio
import hashlib
import importlib
import inspect
import os
import subprocess
import sys

import pytest
import torch

from comfy_api.v4 import eng, io
from comfy_api.latest import _engine, _runtime as _sdk

import engine_loopback as loopback  # noqa: E402

COMFY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class _Patcher:
    """Stands in for a ModelPatcher where only dispatch is under test."""

    def __init__(self):
        self.model_options = {"transformer_options": {}}
        self.load_device = "cpu"
        self.hook = None
        self.patches = []

    def clone(self):
        other = _Patcher()
        other.patches = list(self.patches)
        return other

    def set_model_sampler_post_cfg_function(self, fn):
        self.hook = fn

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        self.patches.append((patches, strength_patch, strength_model))
        return list(patches)


class _Clip:
    def tokenize(self, text, **kwargs):
        return {"l": [[(1, 1.0), (len(text), 1.0)]]}

    def encode_from_tokens_scheduled(self, tokens, add_dict={}):
        return [[torch.ones(1, 2, 3), {"tokens": len(tokens["l"][0]), **add_dict}]]


class _TupleNode:
    """A V1 node shape."""
    FUNCTION = "go"

    def go(self, value):
        return (value * 2, "done")


class _ModernNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Modern", inputs=[], outputs=[])

    @classmethod
    def execute(cls, value):
        return io.NodeOutput(value + 1)


class _CallingNode:
    @classmethod
    async def execute(cls, clip):
        tokens = await eng.clip.tokenize(clip, "hello")
        cond = await eng.clip.encode_tokens(clip, tokens, add_dict={"width": 8})
        rows = [[cond[0][0] * 2, {**cond[0][1], "doubled": True}]]
        return io.NodeOutput(rows)


class _HookNode:
    @classmethod
    async def execute(cls, model, scale):
        def hook(args):
            return args["denoised"] * scale

        return io.NodeOutput(await eng.model.set_post_cfg(model, hook))


class _RunNode:
    @classmethod
    async def execute(cls, value):
        doubled, status = await eng.node.run("EngineTestTuple", {"value": value})
        return io.NodeOutput(doubled, status)


@pytest.fixture(params=sorted(loopback.MODES))
def run(request):
    mode = loopback.MODES[request.param]

    def go(node, pack="p", **inputs):
        return asyncio.run(mode(node, pack, inputs))

    return go


def test_every_row_resolves_to_real_engine_code():
    import comfy.model_patcher
    import comfy.sd

    classes = {"MODEL": comfy.model_patcher.ModelPatcher, "CLIP": comfy.sd.CLIP, "VAE": comfy.sd.VAE}
    for name, row in _engine.CALLS.items():
        if row.target.startswith("engine:"):
            stem, _, method = row.target[len("engine:"):].partition(".")
            assert callable(getattr(_engine.core_extension(stem), method)), name
        elif row.kind is None:
            module, attr = row.target.split(":")
            assert callable(getattr(importlib.import_module(module), attr)), name
        elif name == "clip.read_weight":
            assert callable(comfy.model_patcher.ModelPatcher.patch_weight_to_device)
        elif name == "model.latent_process_in":
            import comfy.latent_formats
            assert callable(comfy.latent_formats.LatentFormat.process_in)
        else:
            assert callable(getattr(classes[row.kind], row.target)), name


def test_a_call_with_data_and_a_structure_the_pack_built(run):
    (result, _) = run(_CallingNode, clip=_Clip())
    rows = result.result[0]
    assert rows[0][1] == {"tokens": 2, "width": 8, "doubled": True}
    assert torch.equal(rows[0][0], torch.ones(1, 2, 3) * 2)


def test_a_sampling_function(run):
    source = _Patcher()
    (result, _) = run(_HookNode, model=source, scale=3.0)
    patched = result.result[0]
    assert patched is not source and source.hook is None
    out = patched.hook({"denoised": torch.ones(2), "model": object(), "sigma": 1.0})
    assert torch.equal(out, torch.ones(2) * 3)


def test_a_sampling_function_sees_only_data():
    seen = {}
    wrapped = _engine._sampling_function(lambda args: seen.update(args))
    tensor = torch.ones(1)
    wrapped({
        "denoised": tensor, "model": object(), "sigma": 1.0,
        "model_options": {"patches": {"a": [print]}, "transformer_options": {"sample_sigmas": tensor}},
    })
    assert seen["denoised"] is tensor and "model" not in seen
    assert seen["model_options"]["transformer_options"]["sample_sigmas"] is tensor
    assert seen["model_options"]["patches"] == {"a": []}


def test_node_run_v1_and_modern(run, monkeypatch):
    import nodes

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "EngineTestTuple", _TupleNode)
    (result, _) = run(_RunNode, value=4)
    assert list(result.result) == [8, "done"]

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "EngineTestModern", _ModernNode)
    assert asyncio.run(_engine.core_extension("node").run("EngineTestModern", {"value": 1})) == (2,)


class _HiddenV1:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}, "hidden": {"prompt": "PROMPT", "node": "UNIQUE_ID"}}

    FUNCTION = "go"

    def go(self, value, prompt=None, node=None):
        return {"ui": {"images": [value, prompt, node]}, "result": (value + 1,)}


class _HiddenModern(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="EngineTestHiddenModern", inputs=[io.Int.Input("value")],
                         outputs=[io.Int.Output()], hidden=[io.Hidden.unique_id])

    @classmethod
    def execute(cls, value):
        return io.NodeOutput(value, ui={"node": cls.hidden.unique_id})


def test_node_run_output_returns_ui_and_fills_hidden_inputs(monkeypatch):
    import nodes

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "EngineTestHiddenV1", _HiddenV1)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "EngineTestHiddenModern", _HiddenModern)
    plan = _sdk.ExecutionPlan(prompt_id="p", node_id="7", node_type="n", prompt={"1": {}})

    async def go():
        with _sdk.bind_runtime(_sdk.InProcessRefResolver(), None, _sdk.EngineCalls(), plan=plan):
            return (await eng.node.run_output("EngineTestHiddenV1", {"value": 1}),
                    await eng.node.run("EngineTestHiddenV1", {"value": 1}),
                    await eng.node.run_output("EngineTestHiddenModern", {"value": 2}))

    first, tuple_only, modern = asyncio.run(go())
    assert first == {"result": (2,), "ui": {"images": [1, {"1": {}}, "7"]}}
    assert tuple_only == (2,)
    assert modern == {"result": (2,), "ui": {"node": "7"}}


class _SeesNode:
    """Reports what the pack's code was handed, and what an eng call gave back."""

    @classmethod
    async def execute(cls, model, image, latent, clip):
        tokens = await eng.clip.tokenize(clip, "hi")
        cond = await eng.clip.encode_tokens(clip, tokens)
        cloned = await eng.model.clone(model)
        return io.NodeOutput({
            "image": type(image).__name__, "latent": type(latent["samples"]).__name__,
            "cond": type(cond[0][0]).__name__, "tokens": type(tokens).__name__,
            "model_is_ref": isinstance(model, _sdk.Ref), "clone_is_ref": isinstance(cloned, _sdk.Ref),
        }, cloned)


def test_pack_code_gets_values_in_both_modes(run):
    source = _Patcher()
    (result, _) = run(_SeesNode, model=source, image=torch.ones(1, 2, 2, 3),
                      latent={"samples": torch.zeros(1, 4, 2, 2)}, clip=_Clip())
    seen, cloned = result.result
    assert {k: seen[k] for k in ("image", "latent", "cond")} == {"image": "Tensor", "latent": "Tensor", "cond": "Tensor"}
    assert seen["tokens"] == "dict"
    assert isinstance(cloned, _Patcher) and cloned is not source


def test_in_process_hands_the_pack_the_engine_objects_themselves():
    source = _Patcher()
    (result, _) = asyncio.run(loopback.run_in_process(
        _SeesNode, "p", {"model": source, "image": torch.ones(1), "latent": {"samples": torch.ones(1)}, "clip": _Clip()}))
    seen, _cloned = result.result
    assert seen["model_is_ref"] is False and seen["clone_is_ref"] is False


def test_every_row_is_a_method_on_eng():
    for name, row in _engine.CALLS.items():
        group, _, method = name.partition(".")
        fn = getattr(getattr(eng, group), method)
        assert inspect.iscoroutinefunction(fn), name
        assert fn.__doc__ == row.doc, name


def _ext(name):
    """``eng.ext.<file>.<method>`` for a dotted name, as pack code would write it."""
    stem, _, method = name.rpartition(".")
    return getattr(getattr(eng.ext, stem), method)


def test_ext_is_only_for_extensions():
    with pytest.raises(AttributeError, match="eng.model.<name>"):
        eng.ext.model.clone
    assert not hasattr(eng, "internal_call")


def test_extension_named_like_a_group_is_refused(tmp_path, monkeypatch):
    root, key = _pack(tmp_path, file_name="model.py")
    monkeypatch.setattr(_engine, "ALLOWED_EXTENSIONS", [key])
    with pytest.raises(_engine.ExtensionNotAllowed, match="eng group"):
        _engine.load_pack_extensions("packs.one", root)


def test_eng_has_no_handle_vocabulary():
    for word in ("read", "write", "is_data", "Handle"):
        assert not hasattr(eng, word), word


def test_progress_goes_to_the_running_node(monkeypatch):
    import comfy.utils

    seen = []
    monkeypatch.setattr(comfy.utils, "PROGRESS_BAR_HOOK", lambda *a: seen.append(a))
    plan = _sdk.ExecutionPlan(prompt_id="p", node_id="7", node_type="n")

    async def go():
        with _sdk.bind_runtime(_sdk.InProcessRefResolver(), None, _sdk.EngineCalls(), plan=plan):
            await eng.progress.update(1, 4)

    asyncio.run(go())
    assert seen == [(1, 4, None, "p", "7")]


def test_unknown_and_private_names_are_rejected():
    class Node:
        @classmethod
        async def execute(cls, name):
            return io.NodeOutput(await _ext(name)())

    for name in ("nope", "os.system", "video_attention.enhance_a_video", "a.b.c", "tool._private"):
        with pytest.raises((_engine.CallNotSupported, AttributeError)):
            asyncio.run(loopback.run_in_process(Node, "p", {"name": name}))


def test_wrong_handle_kind_is_rejected():
    class Node:
        @classmethod
        async def execute(cls, clip):
            return io.NodeOutput(await eng.vae.decode(clip, 1))

    with pytest.raises(TypeError, match="VAE"):
        asyncio.run(loopback.run_in_process(Node, "p", {"clip": _Clip()}))


def test_the_wire_refuses_a_tensor():
    with pytest.raises(TypeError):
        loopback._wire({"t": torch.ones(1)})


# --------------------------------------------------------------------------- #
# Extensions
# --------------------------------------------------------------------------- #
EXTENSION = b"""
class EngineOps:
    def echo(self, value):
        return value
    def _private(self):
        return 1
"""


def _pack(tmp_path, name="pack", file_name="tool.py", source=EXTENSION):
    directory = tmp_path / name / "v2" / "engine"
    directory.mkdir(parents=True)
    (directory / file_name).write_bytes(source)
    digest = "sha256:" + hashlib.sha256(source).hexdigest()
    return str(tmp_path / name), (file_name, digest)


@pytest.fixture(autouse=True)
def _clean_extensions(monkeypatch):
    monkeypatch.setattr(_engine, "_extensions", {})
    monkeypatch.setattr(_engine, "_pack_extensions", {})
    monkeypatch.delenv(_engine.DEV_EXTENSIONS_ENV, raising=False)


def _call(name, pack, *args):
    class Node:
        @classmethod
        async def execute(cls):
            return io.NodeOutput(await _ext(name)(*args))

    (result, _) = asyncio.run(loopback.run_in_process(Node, pack, {}))
    return result.result[0]


def test_listed_extension_loads_and_runs(tmp_path, monkeypatch):
    root, key = _pack(tmp_path)
    monkeypatch.setattr(_engine, "ALLOWED_EXTENSIONS", [key])
    assert _engine.load_pack_extensions("packs.one", root) == ["tool"]
    assert _call("tool.echo", "packs.one.nodes.sub", 5) == 5


def test_changed_file_is_refused_at_load(tmp_path, monkeypatch):
    root, key = _pack(tmp_path)
    monkeypatch.setattr(_engine, "ALLOWED_EXTENSIONS", [(key[0], "sha256:" + "0" * 64)])
    with pytest.raises(_engine.ExtensionNotAllowed):
        _engine.load_pack_extensions("packs.one", root)


def test_right_hash_under_another_name_is_refused(tmp_path, monkeypatch):
    root, key = _pack(tmp_path, file_name="other.py")
    monkeypatch.setattr(_engine, "ALLOWED_EXTENSIONS", [("tool.py", key[1])])
    with pytest.raises(_engine.ExtensionNotAllowed):
        _engine.load_pack_extensions("packs.one", root)


def test_a_second_file_or_folder_is_refused(tmp_path, monkeypatch):
    root, key = _pack(tmp_path)
    monkeypatch.setattr(_engine, "ALLOWED_EXTENSIONS", [key])
    (tmp_path / "pack" / "v2" / "engine" / "helpers").mkdir()
    with pytest.raises(_engine.ExtensionNotAllowed):
        _engine.load_pack_extensions("packs.one", root)


def _dispatch_as(pack, name):
    async def go():
        with _sdk.bind_runtime(_sdk.InProcessRefResolver(), None, _sdk.EngineCalls(), pack=pack):
            return await _engine.dispatch(None, name)
    return asyncio.run(go())


def test_a_pack_reaches_only_the_extensions_it_ships(tmp_path, monkeypatch):
    root, key = _pack(tmp_path)
    monkeypatch.setattr(_engine, "ALLOWED_EXTENSIONS", [key])
    _engine.load_pack_extensions("packs.one", root)
    with pytest.raises(_engine.CallNotSupported):
        _call("tool.echo", "packs.two", 1)
    with pytest.raises(AttributeError):  # private methods aren't reachable as eng.ext names at all
        _call("tool._private", "packs.one")
    with pytest.raises(_engine.CallNotSupported):  # nor through the wire name
        _dispatch_as("packs.one", "tool._private")


def test_two_packs_share_a_file_and_two_versions_coexist(tmp_path, monkeypatch):
    version_two = EXTENSION.replace(b"return value", b"return value * 10")
    one, key_one = _pack(tmp_path, "one")
    same, _ = _pack(tmp_path, "same")
    two, key_two = _pack(tmp_path, "two", source=version_two)
    monkeypatch.setattr(_engine, "ALLOWED_EXTENSIONS", [key_one, key_two])
    _engine.load_pack_extensions("one", one)
    _engine.load_pack_extensions("same", same)
    _engine.load_pack_extensions("two", two)
    assert _engine._pack_extensions["one"] == _engine._pack_extensions["same"]
    assert len(_engine._extensions) == 2
    assert _call("tool.echo", "one", 2) == 2
    assert _call("tool.echo", "two", 2) == 20


def test_the_developer_override(tmp_path, monkeypatch):
    root, _ = _pack(tmp_path)
    monkeypatch.setattr(_engine, "ALLOWED_EXTENSIONS", [])
    monkeypatch.setenv(_engine.DEV_EXTENSIONS_ENV, "1")
    assert _engine.load_pack_extensions("packs.one", root) == ["tool"]


def test_real_pack_load_fails_on_an_unlisted_extension(tmp_path, monkeypatch):
    import nodes

    root, _ = _pack(tmp_path)
    with open(os.path.join(root, "__init__.py"), "w") as file:
        file.write("NODE_CLASS_MAPPINGS = {'ShouldNotLoad': object}\n")
    monkeypatch.setattr(_engine, "ALLOWED_EXTENSIONS", [])
    assert asyncio.run(nodes.load_custom_node(root)) is False
    assert "ShouldNotLoad" not in nodes.NODE_CLASS_MAPPINGS


@pytest.mark.parametrize("module", [_engine, _sdk])
def test_the_api_has_no_pack_or_isolation_names(module):
    with open(inspect.getsourcefile(module), encoding="utf-8") as file:
        source = file.read().lower()
    for word in ("sandbox", "secure", "guest", "overlay", "gvisor", "kjnodes", "rgthree", "rpc"):
        assert word not in source, word


def test_generated_stubs_are_current():
    subprocess.run(
        [sys.executable, os.path.join(COMFY_ROOT, "comfy_api", "generate_engine_stubs.py"), "--check"],
        check=True,
    )


# --------------------------------------------------------------------------- #
# Extensions add no dependencies to ComfyUI
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("line", [
    "import sageattention", "from flash_attn import flash_attn_func",
    "import importlib\nm = importlib.import_module('triton')", "m = __import__('some_other_pack')",
    "import importlib\nname = 'x'\nm = importlib.import_module(name)", "from . import helpers",
])
def test_an_extension_that_imports_outside_comfyui_is_refused_even_in_dev_mode(tmp_path, monkeypatch, line):
    source = (line + "\nclass EngineOps:\n    pass\n").encode()
    root, _ = _pack(tmp_path, source=source)
    monkeypatch.setenv(_engine.DEV_EXTENSIONS_ENV, "1")
    with pytest.raises(_engine.ExtensionNotAllowed, match="doesn't ship"):
        _engine.load_pack_extensions("packs.dep", root)


def test_an_extension_may_use_comfyui_and_its_requirements(tmp_path, monkeypatch):
    source = b"import torch\nimport numpy\nimport einops\nfrom PIL import Image\nimport comfy.model_management\n" \
             b"import folder_paths\nimport json\nclass EngineOps:\n    pass\n"
    root, _ = _pack(tmp_path, source=source)
    monkeypatch.setenv(_engine.DEV_EXTENSIONS_ENV, "1")
    assert _engine.load_pack_extensions("packs.ok", root) == ["tool"]


def test_core_extensions_follow_the_same_rule():
    for name in sorted(os.listdir(_engine.CORE_EXTENSIONS_DIR)):
        if name.endswith(".py"):
            path = os.path.join(_engine.CORE_EXTENSIONS_DIR, name)
            with open(path, "rb") as file:
                _engine.check_extension_imports(name, file.read(), path)


ROUTES_PYPROJECT = """
[project]
name = "Route Pack"
version = "1.0"

[[tool.comfy.routes]]
method = "get"
path = "/tags"
module = "routes"
handler = "tags"
"""

ROUTES_MODULE = """
import json


def tags(request):
    return {"status": 200, "body": json.dumps({"q": request["query"].get("q"), "method": request["method"]})}
"""


def test_declared_routes_are_served_on_comfyuis_server(tmp_path):
    import asyncio
    import importlib.util
    import json
    import sys

    from comfy_api.latest import registration

    root = tmp_path / "route_pack"
    root.mkdir()
    (root / "pyproject.toml").write_text(ROUTES_PYPROJECT)
    (root / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n")
    (root / "routes.py").write_text(ROUTES_MODULE)
    package = str(root).replace(".", "_x_")
    spec = importlib.util.spec_from_file_location(package, root / "__init__.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)

    class Routes:
        table = {}

        def route(self, method, url):
            def add(handler):
                self.table[(method, url)] = handler
                return handler
            return add

    class Server:
        routes = Routes()

    class Request:
        content_length = None
        query = {"q": "cat"}

        async def text(self):
            return ""

    served = registration.install_routes(str(root), package, Server())
    url = f"/ext/{registration.pack_namespace('route-pack')}/tags"
    assert served == [f"GET {url}"]
    response = asyncio.run(Server.routes.table[("GET", url)](Request()))
    assert response.status == 200
    assert json.loads(response.text) == {"q": "cat", "method": "get"}
