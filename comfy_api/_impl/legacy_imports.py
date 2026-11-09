"""Old import names for packs written before ``comfy_api.index``.

V3 packs import ``comfy_api.latest`` (and ``comfy_api.v0_0_2`` / ``v0_0_3``, ``comfy_api.input``,
``comfy_api.input_impl``, ``comfy_api.util``...). Those modules don't exist in the release
layout; this finder serves each old name as the module that replaced it, so the same objects
come back and ``isinstance`` checks keep working. ComfyUI installs it before it loads custom
nodes. Nothing in ComfyUI itself uses these names.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import types

# old module -> its module in the release layout
MODULES = {
    "comfy_api.latest._io": "comfy_api._impl._io",
    "comfy_api.latest._io_public": "comfy_api._impl._io_public",
    "comfy_api.latest.io": "comfy_api._impl._io_public",
    "comfy_api.latest._ui": "comfy_api._impl._ui",
    "comfy_api.latest._ui_public": "comfy_api._impl._ui_public",
    "comfy_api.latest.ui": "comfy_api._impl._ui_public",
    "comfy_api.latest._input": "comfy_api._impl._input",
    "comfy_api.latest._input_impl": "comfy_api._impl._input_impl",
    "comfy_api.latest._util": "comfy_api._impl._types",
    "comfy_api.latest._caching": "comfy_api._impl._caching",
    "comfy_api.latest._execution_metadata": "comfy_api._impl._execution_metadata",
    "comfy_api.latest._engine": "comfy_api._impl._engine",
    "comfy_api.latest._runtime": "comfy_api._impl._runtime",
    "comfy_api.latest.util": "comfy_api._impl.util",
    "comfy_api.input": "comfy_api._impl._input",
    "comfy_api.input_impl": "comfy_api._impl._input_impl",
    "comfy_api.util": "comfy_api._impl._types",
    "comfy_api.internal": "comfy_api._impl.internal",
    "comfy_api.torch_helpers": "comfy_api._impl.torch_helpers",
    "comfy_api.feature_flags": "comfy_api._impl.feature_flags",
}
# old API packages, served as one module with the names they exported
PACKAGES = ("comfy_api.latest", "comfy_api.v0_0_3", "comfy_api.v0_0_2", "comfy_api.v0_0_1")


def _package(name: str) -> types.ModuleType:
    from comfy_api import index
    from comfy_api._impl import _api

    module = types.ModuleType(name, f"Old name for comfy_api.index (see {__name__}).")
    module.__path__ = []  # a package: its old submodules resolve through this finder
    names = dict(
        ComfyAPI=_api.ComfyAPI, ComfyAPISync=_api.ComfyAPISync, Caching=_api.Caching,
        ComfyExtension=index.ComfyExtension, Input=index.Input, InputImpl=index.InputImpl,
        Types=index.Types, io=index.io, IO=index.io, ui=index.ui, UI=index.ui, eng=index.eng)
    if name == "comfy_api.latest":
        names["ComfyAPI_latest"] = _api.ComfyAPI
    else:
        names[f"ComfyAPIAdapter_{name.rsplit('.', 1)[1]}"] = _api.ComfyAPI
    vars(module).update(names)
    module.__all__ = sorted(names)
    return module


def _target(fullname: str) -> str | None:
    for old, new in MODULES.items():
        if fullname == old or fullname.startswith(old + "."):
            return new + fullname[len(old):]
    return None


class _Alias(importlib.abc.Loader):
    def __init__(self, make):
        self._make = make

    def create_module(self, spec):
        return self._make()

    def exec_module(self, module):
        pass


class LegacyImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in PACKAGES:
            loader = _Alias(lambda: _package(fullname))
            return importlib.util.spec_from_loader(fullname, loader, is_package=True)
        new = _target(fullname)
        if new is None:
            return None
        return importlib.util.spec_from_loader(fullname, _Alias(lambda: importlib.import_module(new)))


def install() -> None:
    if not any(isinstance(finder, LegacyImportFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, LegacyImportFinder())
