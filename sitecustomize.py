"""Diagnostic compatibility shim for old ComfyUI + current H3 Flow.

Older MiniMax-H3 block-replacement callbacks do not include ``layout`` in the
callback args. Current MiniMax-H3-Flow-Aligned-Regenerate uses that field only
for layout/attention context bookkeeping and otherwise crashes with
``KeyError: 'layout'`` before the first transformer evaluation.

This shim patches only the Flow layout wrapper at import time. When an old-core
callback has no layout, it passes straight through to the previously installed
wrapper/original block. No MiniMax-H3 numerical code is changed.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys

_TARGET = "h3_flow_regenerate.attention"
_MARKER = "_comfy_h3_flow_legacy_layout_compat"


def _patch_flow_attention(module) -> None:
    if getattr(module, _MARKER, False):
        return

    def make_layout_block_wrapper(layer, metrics, previous=None, *, record_layout=True):
        def wrapper(args, extra):
            layout = args.get("layout")

            # Old ComfyUI H3 block callback ABI: no layout field. Layout recording
            # is diagnostic/context plumbing only, so preserve the old numerical
            # execution path exactly.
            if layout is None:
                if previous is not None:
                    return previous(args, extra)
                return extra["original_block"](args)

            transformer = args["transformer_options"]
            old = transformer.get("h3_flow_attention_context")
            transformer["h3_flow_attention_context"] = {"layout": layout, "layer": layer}
            if layer == 0 and record_layout:
                metrics.event("packed_layout", **module.layout_summary(layout))
            try:
                if previous is not None:
                    return previous(args, extra)
                return extra["original_block"](args)
            finally:
                if old is None:
                    transformer.pop("h3_flow_attention_context", None)
                else:
                    transformer["h3_flow_attention_context"] = old

        return wrapper

    module.make_layout_block_wrapper = make_layout_block_wrapper
    setattr(module, _MARKER, True)
    print("[H3 Flow legacy-layout compat] old H3 block callbacks without layout are accepted")


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def create_module(self, spec):
        create = getattr(self._wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self._wrapped.exec_module(module)
        _patch_flow_attention(module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None or isinstance(spec.loader, _PatchLoader):
            return spec
        spec.loader = _PatchLoader(spec.loader)
        return spec


sys.meta_path.insert(0, _PatchFinder())
