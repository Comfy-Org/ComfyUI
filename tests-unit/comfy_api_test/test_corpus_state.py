"""The rest of the API on corpus packs and small packs:

- eng.store: ComfyUI-Apex-Artist's preset node, its JSON file replaced by the store;
- eng.cache: Y7-SBS's grid cache, its module-level dict replaced by the cache;
- under pack isolation: a pack's own types between its nodes (as Impact's SEGS: namedtuples with
  numpy in them), numpy outputs, one pack's node calling another pack's node, model folders
  (as ComfyUI_essentials' "luts") and node replacements, all from manifests.
"""
import asyncio
import json
import os
import pathlib
import random
import re
import textwrap
import threading
from typing import Any, Dict, List, Optional

import pytest
import torch

from comfy_api._impl import _engine, _runtime

import engine_loopback as loopback
from corpus_ports import corpus_file, original


@pytest.fixture(params=sorted(loopback.MODES))
def run(request):
    mode = loopback.MODES[request.param]

    def go(node, **inputs):
        return asyncio.run(mode(node, "corpus-state", inputs))[0]
    return go


@pytest.fixture
def user_dir(tmp_path, monkeypatch):
    import folder_paths
    pack_storage = pytest.importorskip("comfy_secure_nodes.v21.pack_storage")
    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(tmp_path / "user"))
    monkeypatch.setattr(_runtime, "providers", _runtime.Providers())
    _runtime.load_execution_provider("comfy_secure_nodes.v21.in_process")
    monkeypatch.setattr(pack_storage, "_STORE", None)  # the store is made once per process, under the user folder
    monkeypatch.delenv("COMFY_SECURE_STORAGE_ROOT", raising=False)
    os.makedirs(tmp_path / "user")
    return tmp_path / "user"


def _stored(pack, key):
    pack_storage = pytest.importorskip("comfy_secure_nodes.v21.pack_storage")
    text = pack_storage.pack_store().get("default", pack, key)
    return None if text is None else json.loads(text)


# --------------------------------------------------------------------------- #
# eng.store: Apex's presets
# --------------------------------------------------------------------------- #
CASES = [(0, "Random", "Random", "Disabled", "Random"), (42, "Disabled", "Random", "Random", "Disabled"),
         (7, "Random", "Disabled", "Random", "Random")]


def _apex_original(folder):
    try:
        source = corpus_file("comfyui-apex-artist", "apex_prompt.py")
    except FileNotFoundError:
        pytest.skip("ComfyUI-Apex-Artist isn't in the pack corpus here")
    namespace = {"os": os, "json": json, "random": random, "re": re, "threading": threading, "Dict": Dict,
                 "List": List, "Optional": Optional, "Any": Any, "__file__": str(folder / "apex_prompt.py")}
    return original(source, ["ApexPromptPreset"], namespace)["ApexPromptPreset"]


def test_apex_presets_live_in_the_store_and_give_the_originals_prompts(run, user_dir, tmp_path):
    (tmp_path / "apex").mkdir()
    V1 = _apex_original(tmp_path / "apex")
    from corpus_ports.apex_prompt import ApexPromptPresetV2
    for seed, env, light, style, lens in CASES:
        expected = V1().combine_prompts("a [red|blue] car", seed, env, light, style, lens)
        got = run(ApexPromptPresetV2, input_text="a [red|blue] car", seed=seed, environment_preset=env,
                  lighting_preset=light, style_preset=style, camera_lens_preset=lens).result
        assert tuple(got) == expected
    stored = _stored("corpus-state", "presets")
    assert stored == json.loads((tmp_path / "apex" / "prompt_presets.json").read_text())
    assert (user_dir / "secure-node-storage").is_dir()  # the overlay's storage, shared by every mode

    # A user edits the presets: the original's file, the port's store. Both read the edit.
    edited = stored
    first = next(iter(edited["Apex Style"]))
    edited["Apex Style"][first]["prompt"] = "EDITED STYLE"
    pack_storage = pytest.importorskip("comfy_secure_nodes.v21.pack_storage")
    pack_storage.pack_store().set("default", "corpus-state", "presets", json.dumps(edited))
    (tmp_path / "apex" / "prompt_presets.json").write_text(json.dumps(edited))
    expected = V1().combine_prompts("x", 0, "Disabled", "Disabled", first, "Disabled")
    got = run(ApexPromptPresetV2, input_text="x", seed=0, style_preset=first).result
    assert tuple(got) == expected and "EDITED STYLE" in got[0]


def test_store_scopes_and_limits(user_dir):
    from comfy_api.index import eng

    plan = _runtime.ExecutionPlan(prompt_id="p", node_id="5", node_type="n",
                                  extra_pnginfo={"workflow": {"id": "wf-1"}})

    async def go():
        with _runtime.bind_runtime(_runtime.InProcessRefResolver(), None, _runtime.EngineCalls(), pack="pk", plan=plan):
            await eng.store.set("k", {"a": 1})
            await eng.store.set("k", "workflow value", scope="workflow")
            await eng.store.set("k", 3, scope="node")
            got = (await eng.store.get("k"), await eng.store.get("k", "workflow"), await eng.store.get("k", "node"))
            assert await eng.store.keys() == ["k"]
            assert await eng.store.delete("k") and await eng.store.get("k", default="gone") == "gone"
            with pytest.raises(ValueError, match="exceeds"):
                await eng.store.set("big", "x" * (2 << 20))
            with pytest.raises(TypeError):
                await eng.store.set("bad", torch.ones(1))
            return got

    assert asyncio.run(go()) == ({"a": 1}, "workflow value", 3)


# --------------------------------------------------------------------------- #
# eng.cache: Y7's grids
# --------------------------------------------------------------------------- #
def _cache_module():
    """The globals of the cache extension the dispatcher runs (core extensions are loaded by source)."""
    return type(_engine.core_extension("cache")).get.__globals__


def test_y7_grids_come_from_the_cache(run, monkeypatch):
    from corpus_ports.y7_grid import Y7Grid, build_grid_gs

    cache_ext = _cache_module()
    monkeypatch.setitem(cache_ext, "_entries", type(cache_ext["_entries"])())
    try:
        source = corpus_file("comfyui-y7-sbs-2dto3d", "nodes/sbs.py")
        v1 = original(source, ["get_grid_gs", "_GRID_CACHE_GS"], {"torch": torch})
    except FileNotFoundError:
        v1 = None
    image = torch.rand(1, 6, 10, 3)
    first = run(Y7Grid, image=image).result
    assert len(cache_ext["_entries"]) == 1
    second = run(Y7Grid, image=image).result
    assert len(cache_ext["_entries"]) == 1  # the second run was served from the cache
    expected = v1["get_grid_gs"](6, 10, torch.float16, image.device) if v1 else build_grid_gs(6, 10, torch.float16, image.device)
    for got in (first, second):
        assert torch.equal(got[0], expected[0]) and torch.equal(got[1], expected[1])


def test_cache_ttl_and_budget(monkeypatch):
    from comfy_api.index import eng

    cache_ext = _cache_module()
    monkeypatch.setitem(cache_ext, "_entries", type(cache_ext["_entries"])())
    monkeypatch.setenv("COMFY_PACK_CACHE_MB", "1")

    async def go():
        with _runtime.bind_runtime(_runtime.InProcessRefResolver(), None, _runtime.EngineCalls(), pack="pk"):
            await eng.cache.set("a", torch.zeros(100_000))  # 400 KB
            await eng.cache.set("b", torch.zeros(100_000))
            await eng.cache.set("c", torch.zeros(100_000))  # over 1 MB: "a" goes
            kept = [k for k in "abc" if await eng.cache.get(k) is not None]
            await eng.cache.set("t", 1, ttl=0.01)
            await asyncio.sleep(0.05)
            return kept, await eng.cache.get("t", default="expired")

    assert asyncio.run(go()) == (["b", "c"], "expired")


# --------------------------------------------------------------------------- #
# Under pack isolation
# --------------------------------------------------------------------------- #
SEGS_PACK = '''
import collections
import os
import numpy as np
import torch

SEG = collections.namedtuple("SEG", ["cropped_mask", "crop_region", "label"])  # as Impact's SEGS holds


class MaskToSegs:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",)}}
    RETURN_TYPES = ("SEGS",)
    FUNCTION = "run"
    CATEGORY = "test"

    def run(self, mask):
        array = mask[0].numpy() > 0.5
        ys, xs = np.nonzero(array)
        region = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        crop = array[region[1]:region[3], region[0]:region[2]].astype(np.float32)
        return ((mask.shape[1:], [SEG(crop, region, "blob")]),)


class SegsToMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"segs": ("SEGS",)}}
    RETURN_TYPES = ("MASK", "INT")
    FUNCTION = "run"
    CATEGORY = "test"
    OUTPUT_NODE = True

    def run(self, segs):
        (h, w), items = segs
        out = np.zeros((h, w), dtype=np.float32)
        for seg in items:
            x0, y0, x1, y1 = seg.crop_region
            out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], seg.cropped_mask)
        return {"ui": {"pid": [os.getpid()]}, "result": (torch.from_numpy(out)[None], os.getpid())}


NODE_CLASS_MAPPINGS = {"MaskToSegs": MaskToSegs, "SegsToMask": SegsToMask}
'''

B_PACK = '''
import os
import numpy as np
from comfy_api.index import ComfyExtension, io


class BDouble(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="BDouble", inputs=[io.Image.Input("image")],
                         outputs=[io.Image.Output(), io.Int.Output(), io.Custom("NUMPY").Output()])

    @classmethod
    def execute(cls, image):
        return io.NodeOutput(image * 2, os.getpid(), np.arange(4))


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [BDouble]

    async def get_model_folders(self):
        return {"luts": ["cube"]}   # as ComfyUI_essentials registers its LUT folder

    async def get_node_replacements(self):
        return [io.NodeReplace(new_node_id="BDouble", old_node_id="OldDouble")]


async def comfy_entrypoint():
    return Extension()
'''

A_PACK = '''
import os
from comfy_api.index import ComfyExtension, eng, io


class ACallsB(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="ACallsB", is_output_node=True, inputs=[io.Image.Input("image")],
                         outputs=[io.Image.Output()])

    @classmethod
    async def execute(cls, image):
        doubled, b_pid, numbers = await eng.node.run("BDouble", {"image": image})
        return io.NodeOutput(doubled, ui={"a_pid": [os.getpid()], "b_pid": [b_pid], "numbers": numbers.tolist()})


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [ACallsB]


async def comfy_entrypoint():
    return Extension()
'''


@pytest.fixture
def isolated(tmp_path, monkeypatch, user_dir):
    import folder_paths
    import nodes

    isolation = pytest.importorskip("comfy_secure_nodes.v21.isolation")
    packs = pytest.importorskip("comfy_secure_nodes.v21.isolation.packs")
    server = pytest.importorskip("comfy_secure_nodes.v21.isolation.server")
    monkeypatch.setenv(isolation.ENV, "true")
    monkeypatch.setenv(server.RUN_DIR_ENV, str(tmp_path / "run"))
    monkeypatch.delenv(_runtime.PROVIDER_ENV, raising=False)
    monkeypatch.setattr(packs, "_cache_dir", lambda: str(tmp_path / "index"))
    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path / "models"))
    os.makedirs(tmp_path / "index")
    monkeypatch.setattr(_engine, "_extensions", {})
    monkeypatch.setattr(_engine, "_pack_extensions", {})
    saved, before = _runtime.providers.execution_backend, dict(nodes.NODE_CLASS_MAPPINGS)
    folders = dict(folder_paths.folder_names_and_paths)
    assert _runtime.load_execution_provider("comfy_secure_nodes.v21") is True
    backend = _runtime.providers.execution_backend
    yield backend, tmp_path / "custom_nodes"
    backend.stop()
    _runtime.providers.execution_backend = saved
    nodes.NODE_CLASS_MAPPINGS.clear()
    nodes.NODE_CLASS_MAPPINGS.update(before)
    folder_paths.folder_names_and_paths.clear()
    folder_paths.folder_names_and_paths.update(folders)


def _pack(root, name, code, v2=True, depends_on=None):
    server = pytest.importorskip("comfy_secure_nodes.v21.isolation.server")
    manifest = pytest.importorskip("comfy_secure_nodes.v21.manifest")
    pack = root / name
    pack.mkdir(parents=True)
    (pack / "__init__.py").write_text(textwrap.dedent(code))
    if v2:
        data = server.PackServer.manifest(str(pack))
        if depends_on:
            data["depends_on"] = depends_on
        (pack / manifest.FILE).write_text(json.dumps(data))
    return pack


def _run_prompt(prompt, outputs):
    import execution

    class Server:
        client_id = last_node_id = last_prompt_id = None

        def send_sync(self, *args, **kwargs):
            pass

    executor = execution.PromptExecutor(Server(), cache_type=execution.CacheType.NONE,
                                        cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
    executor.execute(prompt, "corpus-state", {}, outputs)
    return executor


def _dispatch(backend, node_type, inputs):
    plan = _runtime.ExecutionPlan(prompt_id="p", node_id="1", node_type=node_type, node_module="", inputs=inputs)
    runtime = _runtime.Runtime(refs=_runtime.InProcessRefResolver(), ops=_runtime.EngineCalls(), pack="", plan=plan)
    out = asyncio.run(backend.dispatch(plan, None, runtime))
    return tuple(out.result) if hasattr(out, "result") else tuple(out)  # V1 nodes return their tuple


def test_a_packs_own_types_stay_in_its_process_between_its_nodes(isolated):
    boundary = pytest.importorskip("comfy_secure_nodes.v21.isolation.boundary")

    backend, root = isolated
    _pack(root, "segs_pack", SEGS_PACK, v2=False)  # a V1 pack, unmodified
    backend.setup_packs([str(root)])
    backend.packs.ensure_started("segs_pack")  # registers its (V1) nodes
    mask = torch.zeros(1, 8, 8)
    mask[0, 2:5, 3:7] = 1.0
    segs = _dispatch(backend, "MaskToSegs", {"mask": mask})[0]
    shape, items = segs
    assert isinstance(items[0], boundary.PackValue) and items[0].type == "SEG"  # stayed in the pack
    result = _dispatch(backend, "SegsToMask", {"segs": segs})
    assert torch.equal(result[0], mask)  # the next node got the namedtuples, numpy and all
    assert result[1] != os.getpid()


def test_one_packs_node_runs_another_packs_node_in_its_process(isolated):
    import folder_paths

    backend, root = isolated
    _pack(root, "pack_b", B_PACK)
    _pack(root, "pack_a", A_PACK, depends_on=[{"pack": "pack_b"}])
    backend.setup_packs([str(root)])
    assert not backend.packs.started and not backend.packs.missing_dependencies
    assert "luts" in folder_paths.folder_names_and_paths  # pack_b's model folder, from its manifest
    assert ".cube" in folder_paths.folder_names_and_paths["luts"][1]

    prompt = {"1": {"class_type": "EmptyImage", "inputs": {"width": 4, "height": 4, "batch_size": 1, "color": 0x404040}},
              "2": {"class_type": "ACallsB", "inputs": {"image": ["1", 0]}}}
    executor = _run_prompt(prompt, ["2"])
    assert executor.success, executor.status_messages
    ui = executor.history_result["outputs"]["2"]
    assert ui["a_pid"][0] != ui["b_pid"][0] != os.getpid()  # A, B and ComfyUI: three processes
    assert ui["numbers"] == [0, 1, 2, 3]  # numpy crossed as data
    assert sorted(backend.packs.started) == ["pack_a", "pack_b"]


def test_model_folders_list_files_and_replacements_rename_old_nodes(isolated, monkeypatch):
    import folder_paths
    from app.node_replace_manager import NodeReplaceManager
    import server as comfy_server
    from comfy_api.index import io

    backend, root = isolated
    manager = NodeReplaceManager()
    monkeypatch.setattr(comfy_server.PromptServer, "instance", type("S", (), {"node_replace_manager": manager})(),
                        raising=False)
    _pack(root, "pack_b", B_PACK)
    backend.setup_packs([str(root)])
    luts = pathlib.Path(folder_paths.models_dir) / "luts"
    luts.mkdir(parents=True)
    (luts / "film.cube").write_text("LUT_3D_SIZE 2\n")
    (luts / "notes.txt").write_text("not a lut")
    assert io.ModelFolder("luts").resolve() == ["film.cube"]
    prompt = {"1": {"class_type": "OldDouble", "inputs": {"image": ["9", 0]}}, "9": {"class_type": "EmptyImage", "inputs": {}}}
    manager.apply_replacements(prompt)
    assert prompt["1"]["class_type"] == "BDouble"
    assert not backend.packs.started  # none of this ran the pack
