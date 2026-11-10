"""The ComfyUI custom node API. Everything a node pack uses comes from here:

    from comfy_engine import ComfyExtension, eng, io, ui, util

==================  ==================================================================
``io``              Declaring nodes: ``io.ComfyNode``, ``io.Schema``, input and output
                    types, ``io.ModelFolder`` for model-file dropdowns.
``eng``             The engine: one async method per supported call
                    (``await eng.model.clone(model)``, ``eng.vae.decode``,
                    ``eng.node.run``...), plus ``eng.ext.<file>.<method>()`` for
                    the engine extensions a pack ships.
``ui``              What a node shows in the UI (previews, text).
``util``            Pure tensor/data helpers (``common_upscale``, ``prepare_noise``...)
                    that behave the same wherever a node runs.
``ComfyExtension``  How a pack registers its nodes.
``Types``           Shared value types (video, 3D, ...).
``Input``           Input value types (``ImageInput``, ``VideoInput``...).
``InputImpl``       Concrete input values (``VideoFromFile``...).
==================  ==================================================================

This file is the whole public surface. Its types and documentation are in ``__init__.pyi``,
the API's one stub, generated from the engine table. Names that aren't listed here are internal and
may change; ``tests-unit/comfy_engine_test/test_public_api.py`` locks this list. Nothing here
imports ComfyUI's engine (``comfy``, ``nodes``, ``server``, ``folder_paths``), so a node
pack written against it runs the same in ComfyUI's process, in a worker process, or in a
sandbox.
"""
from . import io
from . import ui
from . import util
from comfy_api.latest import ComfyExtension, Input, InputImpl, Types
from ._engine import eng

__all__ = [
    "ComfyExtension",
    "Input",
    "InputImpl",
    "Types",
    "eng",
    "io",
    "ui",
    "util",
]
