"""V2 packs' manifests (comfy-pack.json): ComfyUI starts without importing them, their nodes are
listed anyway, and what the manifest declares is in place before the pack runs."""
import hashlib
import json
import os
import pathlib
import sys
import textwrap

import pytest
import torch

from comfy_api._impl import _engine, _runtime, manifest
from comfy_api._impl import isolation
from comfy_api._impl.isolation import packs, server, weights

PACK = '''
import os
open(os.path.join(os.path.dirname(__file__), "imported"), "a").write(f"{os.getpid()}\\n")

from comfy_api.index import ComfyExtension, io


class ManifestDouble(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="ManifestDouble", display_name="Double (manifest)", category="test",
                         inputs=[io.Image.Input("image"),
                                 io.Combo.Input("lora", options=io.ModelFolder("loras"), optional=True),
                                 io.Float.Input("factor", default=2.0, min=0.0, max=10.0)],
                         outputs=[io.Image.Output(), io.Int.Output()])

    @classmethod
    async def execute(cls, image, factor, lora=None):
        return io.NodeOutput(image * factor, os.getpid())


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [ManifestDouble]


async def comfy_entrypoint():
    return Extension()
'''

PYPROJECT = '''
[project]
name = "manifest-pack"
version = "1.2.3"
description = "A pack for the manifest tests"
dependencies = []

[tool.comfy]
PublisherId = "comfy-tests"
DisplayName = "Manifest Pack"

[[tool.comfy.routes]]
method = "get"
path = "/hello"
module = "routes"
handler = "hello"
'''

ROUTES = '''
import json
import os


def hello(request):
    return {"status": 200, "body": json.dumps({"hello": request["query"].get("name"), "pid": os.getpid()}),
            "content_type": "application/json"}
'''


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    """A V2 pack and its manifest, made the way a publisher makes it (pack.sh manifest)."""
    root = tmp_path_factory.mktemp("manifest-packs")
    pack = root / "custom_nodes" / "manifest_pack"
    pack.mkdir(parents=True)
    (pack / "__init__.py").write_text(textwrap.dedent(PACK))
    (pack / "pyproject.toml").write_text(textwrap.dedent(PYPROJECT))
    (pack / "routes.py").write_text(textwrap.dedent(ROUTES))
    data = server.PackServer.manifest(str(pack))
    (pack / manifest.FILE).write_text(json.dumps(data))
    (pack / "imported").unlink()  # making the manifest imported it; ComfyUI must not
    return root, pack, data


@pytest.fixture
def isolated(made, tmp_path, monkeypatch):
    import nodes

    monkeypatch.setenv(isolation.ENV, "true")
    monkeypatch.setenv(server.RUN_DIR_ENV, str(tmp_path / "run"))
    monkeypatch.delenv(_runtime.PROVIDER_ENV, raising=False)
    monkeypatch.setattr(packs, "_cache_dir", lambda: str(tmp_path / "index"))
    os.makedirs(tmp_path / "index")
    monkeypatch.setattr(_engine, "_extensions", {})
    monkeypatch.setattr(_engine, "_pack_extensions", {})
    saved = _runtime.providers.execution_backend
    before = dict(nodes.NODE_CLASS_MAPPINGS)
    assert _runtime.load_execution_provider() is True
    backend = _runtime.providers.execution_backend
    yield backend
    backend.stop()
    _runtime.providers.execution_backend = saved
    nodes.NODE_CLASS_MAPPINGS.clear()
    nodes.NODE_CLASS_MAPPINGS.update(before)


def _run(prompt, outputs):
    import execution

    class Server:
        client_id = last_node_id = last_prompt_id = None

        def send_sync(self, *args, **kwargs):
            pass

    executor = execution.PromptExecutor(Server(), cache_type=execution.CacheType.NONE,
                                        cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
    executor.execute(prompt, "manifest-test", {}, outputs)
    return executor


PROMPT = {
    "1": {"class_type": "EmptyImage", "inputs": {"width": 4, "height": 4, "batch_size": 1, "color": 0}},
    "2": {"class_type": "ManifestDouble", "inputs": {"image": ["1", 0], "factor": 3.0}},
    "3": {"class_type": "PreviewImage", "inputs": {"images": ["2", 0]}},
}


def test_the_manifest_carries_identity_and_every_nodes_schema(made):
    _, _, data = made
    assert data["format"] == manifest.FORMAT
    assert (data["name"], data["version"], data["publisher"], data["display_name"]) == \
        ("manifest-pack", "1.2.3", "comfy-tests", "Manifest Pack")
    assert data["permissions"] == [] and data["weights"] == [] and data["depends_on"] == []
    node = data["nodes"]["ManifestDouble"]
    assert node["display_name"] == "Double (manifest)" and node["async"] is True
    schema = manifest.decode_schema(node["schema"])
    assert schema.node_id == "ManifestDouble"
    assert [i.id for i in schema.inputs] == ["image", "lora", "factor"]
    json.dumps(data)  # plain JSON throughout


def test_comfyui_starts_without_importing_a_v2_pack_and_lists_its_nodes(made, isolated):
    import nodes

    root, pack, _ = made
    isolated.setup_packs([str(root / "custom_nodes")])
    assert not (pack / "imported").exists(), "the pack's code ran at startup"
    assert not isolated.packs.started
    assert not any("manifest_pack" in name for name in sys.modules)
    cls = nodes.NODE_CLASS_MAPPINGS["ManifestDouble"]  # listed before the pack ever runs
    info = cls.GET_NODE_INFO_V1()
    assert info["display_name"] == "Double (manifest)"
    assert info["input"]["required"]["factor"][1]["default"] == 2.0


def test_a_render_starts_the_pack_and_runs_its_node_there(made, isolated):
    root, pack, _ = made
    isolated.setup_packs([str(root / "custom_nodes")])
    executor = _run(PROMPT, ["3"])
    assert executor.success, executor.status_messages
    imported_by = {int(line) for line in (pack / "imported").read_text().split()}
    assert os.getpid() not in imported_by and len(imported_by) == 1  # in its own process only
    assert sorted(isolated.packs.started) == ["manifest_pack"]


def test_a_pack_whose_dependency_isnt_installed_says_so_when_used(made, isolated, tmp_path):
    root, pack, data = made
    other = tmp_path / "custom_nodes" / "needs_more"
    other.mkdir(parents=True)
    (other / "__init__.py").write_text("raise ImportError('never imported')\n")
    needs = dict(data, name="needs-more", depends_on=[{"pack": "not-installed", "version": ">=2"}],
                 nodes={"NeedsMore": dict(data["nodes"]["ManifestDouble"],
                                          schema=dict(data["nodes"]["ManifestDouble"]["schema"]))})
    needs["nodes"]["NeedsMore"]["schema"]["attrs"] = dict(needs["nodes"]["NeedsMore"]["schema"]["attrs"], node_id="NeedsMore")
    (other / manifest.FILE).write_text(json.dumps(needs))
    isolated.setup_packs([str(tmp_path / "custom_nodes")])
    assert isolated.packs.missing_dependencies["needs_more"] == ["not-installed >=2"]
    with pytest.raises(RuntimeError, match="needs not-installed >=2"):
        isolated.packs.ensure_started("needs_more")


def test_declared_weights_are_verified_and_fetched_before_a_pack_runs(tmp_path, monkeypatch):
    import folder_paths

    folder = tmp_path / "models" / "loras"
    folder.mkdir(parents=True)
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "loras", ([str(folder)], {".bin", ".safetensors"}))
    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(tmp_path))
    (folder / "present.bin").write_bytes(b"weights")
    good = hashlib.sha256(b"weights").hexdigest()
    fetched = []

    def download(weight):  # stands in for the network; installs where the overlay's downloader does
        fetched.append(weight["filename"])
        path = folder / "huggingface" / weight["repo_id"] / (weight.get("revision") or "main") / weight["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fetched")
        return str(path)

    declared = [{"folder": "loras", "filename": "present.bin", "sha256": good},
                {"folder": "loras", "filename": "missing.bin", "repo_id": "org/repo", "revision": "abc",
                 "sha256": hashlib.sha256(b"fetched").hexdigest()}]
    paths = weights.ensure("p", declared, download=download)
    assert fetched == ["missing.bin"]
    assert paths[1].endswith(os.path.join("huggingface", "org", "repo", "abc", "missing.bin"))
    assert weights.ensure("p", declared, download=download) == paths and fetched == ["missing.bin"]  # verified once
    (folder / "present.bin").write_bytes(b"tampered")
    with pytest.raises(weights.WeightsError, match="isn't the file"):
        weights.ensure("p", declared, download=download)
    with pytest.raises(weights.WeightsError, match="doesn't say where"):
        weights.ensure("p", [{"folder": "loras", "filename": "nowhere.bin"}])


def test_the_overlays_downloader_verifies_what_it_installs(tmp_path, monkeypatch):
    """The lifted downloader end to end, with Hugging Face replaced by a local file."""
    import folder_paths
    import huggingface_hub
    import safetensors.torch
    from comfy_api._impl.isolation import hf_weights

    folder = tmp_path / "models" / "loras"
    folder.mkdir(parents=True)
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "loras", ([str(folder)], {".safetensors"}))
    source = tmp_path / "hub" / "lora.safetensors"
    source.parent.mkdir()
    safetensors.torch.save_file({"w": torch.ones(2)}, str(source))
    good = hashlib.sha256(source.read_bytes()).hexdigest()

    def fake_hf_hub_download(*, repo_id, filename, repo_type, revision, endpoint, token, dry_run=False):
        if dry_run:
            return type("Info", (), {"file_size": source.stat().st_size})()
        return str(source)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    weight = {"folder": "loras", "filename": "lora.safetensors", "repo_id": "org/lora", "revision": "main", "sha256": good}
    path = weights._download(weight)
    assert path == str(folder / "huggingface" / "org" / "lora" / "main" / "lora.safetensors")
    source.write_bytes(b"not a safetensors file")
    monkeypatch.setattr(hf_weights._InProcessModels, "_HF_VERIFIED_WEIGHTS", {})
    with pytest.raises(Exception):
        weights._download(dict(weight, filename="lora.safetensors", repo_id="org/other", sha256=None))


def test_a_manifest_that_cant_be_acted_on_is_refused():
    base = {"format": manifest.FORMAT, "name": "ok", "version": "1", "nodes": {}}
    manifest.check(base)
    for bad in ({"name": "Has Spaces"}, {"weights": [{"folder": "loras", "filename": "../../etc/passwd"}]},
                {"weights": [{"folder": "loras", "filename": "a.bin", "sha256": "nothex"}]},
                {"depends_on": [{"version": ">=1"}]}, {"nodes": {"X": {}}}, {"format": "other"}):
        with pytest.raises(manifest.ManifestError):
            manifest.check(dict(base, **bad))


def test_model_folder_names_are_plain_names_everywhere(monkeypatch, tmp_path):
    # a pack names model folders; ComfyUI places them under its models directory. A path would let it read anywhere.
    base = {"format": manifest.FORMAT, "name": "ok", "version": "1", "nodes": {}}
    for folder in ("../../etc", "/etc", "a/b", "..", "a\\b", "", ".hidden"):
        with pytest.raises(manifest.ManifestError):
            manifest.check(dict(base, model_folders={folder: None}))
        with pytest.raises(manifest.ManifestError):
            manifest.check(dict(base, weights=[{"folder": folder, "filename": "a.bin"}]))
        with pytest.raises(manifest.ManifestError):
            manifest.check_folder_name(folder)
    manifest.check(dict(base, model_folders={"my_models": None, "sam2.1": [".pt"]}))
    import folder_paths
    from comfy_api._impl import registration
    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {})
    with pytest.raises(manifest.ManifestError):
        registration._model_folder("../outside", None)  # the engine refuses it even without a manifest check
    assert folder_paths.folder_names_and_paths == {}
    registration._model_folder("pack_models", None)
    assert folder_paths.folder_names_and_paths["pack_models"][0] == [str(tmp_path / "pack_models")]


def test_every_conversion_in_the_pack_corpus_reads_as_a_manifest():
    corpus = pathlib.Path(__file__).resolve().parents[3] / "overlay-r2" / "pack-db" / "packs"
    entries = sorted(p.parent for p in corpus.glob("*/*/*/v2/secure-nodes.json"))
    if not entries:
        pytest.skip("the pack corpus isn't next to this ComfyUI")
    for entry in entries:
        data = manifest.read(str(entry))
        for node in data["nodes"].values():
            manifest.decode_schema(node["schema"])


SAME_AS_CODE = r'''
import json, os, sys
from comfy_api._impl import manifest
from comfy_api._impl.isolation import proxy
from comfy_api._impl.isolation.server import LoadedPack
entry = os.path.realpath(sys.argv[1])
data = manifest.generate(entry)
pack = LoadedPack(entry)
differ = []
for node_id, node in data["nodes"].items():
    real = pack.nodes[node_id].GET_NODE_INFO_V1()
    cls = proxy.make_proxy(node_id, {"kind": "manifest", "schema": node["schema"], "is_async": node["async"],
                                     "methods": node["methods"]}, "p", "p", lambda n: {})
    mine = cls.GET_NODE_INFO_V1()
    for info in (real, mine):
        info.pop("python_module", None)
    if json.dumps(real, sort_keys=True, default=str) != json.dumps(mine, sort_keys=True, default=str):
        differ.append(node_id)
print(json.dumps({"nodes": len(data["nodes"]), "differ": differ}))
'''


def _same_as_code(entry, tmp_path):
    import subprocess
    env = dict(os.environ, PYTHONPATH=str(pathlib.Path(__file__).resolve().parents[2]),
               COMFY_DEV_ENGINE_EXTENSIONS="1", **{server.RUN_DIR_ENV: str(tmp_path / "run")})
    done = subprocess.run([sys.executable, "-c", SAME_AS_CODE, str(entry)], env=env, capture_output=True,
                          text=True, timeout=600, cwd=str(pathlib.Path(__file__).resolve().parents[2]))
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_a_node_registered_from_the_manifest_is_the_node_its_code_declares(made, tmp_path):
    _, pack, _ = made
    result = _same_as_code(pack, tmp_path)
    assert result == {"nodes": 1, "differ": []}


def test_every_kjnodes_node_registers_from_its_manifest_as_its_code_declares(tmp_path):
    kj = pathlib.Path(__file__).resolve().parents[3] / "kjnodes-eng"
    if not (kj / "__init__.py").is_file():
        pytest.skip("the KJNodes port isn't next to this ComfyUI")
    result = _same_as_code(kj, tmp_path)
    assert result["nodes"] >= 250 and result["differ"] == []


def test_a_v2_packs_routes_are_declared_and_served_under_its_namespace(made, isolated, monkeypatch):
    """[[tool.comfy.routes]] -> the manifest; ComfyUI serves them at /ext/<namespace>/ (from the
    manifest's name, wherever the pack is installed) without importing the pack; a request starts it."""
    import asyncio
    import types
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    root, pack, data = made
    assert data["routes"] == [{"method": "get", "path": "/hello", "module": "routes", "handler": "hello"}]
    namespace = packs.pack_namespace("manifest-pack")
    (pack / "imported").unlink(missing_ok=True)  # an earlier test may have run the pack
    stand_in = types.SimpleNamespace(routes=web.RouteTableDef(), app=web.Application())
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=stand_in)))
    isolated.setup_packs([str(root / "custom_nodes")])
    assert isolated.packs.packs["manifest_pack"].namespace == namespace
    assert not (pack / "imported").exists() and not isolated.packs.started

    async def go():
        app = web.Application()
        app.add_routes(stand_in.routes)
        async with TestClient(TestServer(app)) as client:
            unknown = await client.get(f"/ext/{namespace}/nothing-here")
            assert unknown.status == 404 and not isolated.packs.started  # undeclared: the pack isn't started
            return await (await client.get(f"/ext/{namespace}/hello?name=ben")).json()
    reply = asyncio.run(go())
    assert reply["hello"] == "ben" and reply["pid"] != os.getpid()
    assert sorted(isolated.packs.started) == ["manifest_pack"]


def test_a_route_must_be_a_literal_path():
    with pytest.raises(manifest.ManifestError):
        manifest.check({"format": manifest.FORMAT, "name": "x", "version": "1", "nodes": {},
                        "routes": [{"method": "get", "path": "/items/{id}", "module": "routes", "handler": "item"}]})
