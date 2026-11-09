"""comfy_api.index: the one public API. Changing this list is an API change."""
import subprocess
import sys

import comfy_api.index as api

PUBLIC = ["ComfyExtension", "Input", "InputImpl", "Types", "eng", "io", "ui", "util"]


def test_the_public_surface_is_exactly_this():
    assert sorted(api.__all__) == sorted(PUBLIC)
    assert sorted(n for n in vars(api) if not n.startswith("_")) == sorted(PUBLIC)


def test_the_one_stub_declares_the_same_names():
    import ast
    import os

    stub = os.path.join(os.path.dirname(api.__file__), "index.pyi")
    tree = ast.parse(open(stub).read())
    declared = next(ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "__all__")
    assert sorted(declared) == sorted(api.__all__)


def test_the_api_folder_has_only_the_release_layout():
    import os

    root = os.path.dirname(api.__file__)
    entries = sorted(e for e in os.listdir(root) if e != "__pycache__")
    assert entries == ["README.md", "_impl", "docs", "generate_engine_stubs.py", "index.py", "index.pyi",
                       "py.typed"]


def test_old_import_names_are_the_same_objects():
    """V3 packs import comfy_api.latest (and v0_0_x, input, util...); ComfyUI serves those names."""
    code = """
import comfy_api.index as api
from comfy_api._impl import legacy_imports
legacy_imports.install()
import comfy_api.latest as latest
from comfy_api.latest import io, IO, ui, UI, ComfyExtension, Input, InputImpl, Types, eng, ComfyAPI, ComfyAPISync
from comfy_api.latest._io import ComfyNode
from comfy_api.latest._util.video_types import VideoContainer
from comfy_api.v0_0_2 import ComfyAPISync as Sync2, io as io2
from comfy_api.input import VideoInput
from comfy_api.input_impl import VideoFromFile
from comfy_api.util import VideoCodec
from comfy_api.torch_helpers import set_torch_compile_wrapper
from comfy_api._impl import _api, _io, _types
assert io is IO is io2 is api.io and ui is UI is api.ui and eng is api.eng
assert ComfyExtension is api.ComfyExtension and Input is api.Input and Types is api.Types
assert ComfyAPI is _api.ComfyAPI and ComfyAPISync is Sync2 is _api.ComfyAPISync
assert ComfyNode is _io.ComfyNode and VideoContainer is _types.VideoContainer
assert VideoInput is api.Input.Video and VideoFromFile is api.InputImpl.VideoFromFile
assert VideoCodec is api.Types.VideoCodec
print("ok")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "ok", out.stdout + out.stderr


def test_importing_the_api_loads_no_engine_modules():
    code = ("import sys, comfy_api.index; "
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
            "('comfy', 'nodes', 'server', 'folder_paths', 'execution', 'comfy_extras')); print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr
