import asyncio
import importlib
import pickle
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from comfy_api import latest, v3
from comfy_api.internal import _ComfyNodeInternal, _NodeOutputInternal
from comfy_api.internal.singleton import SingletonMetaclass


@pytest.mark.parametrize("first", ["latest", "v3"])
def test_private_imports_preserve_identity_in_either_order(first):
    script = dedent(f"""\
        import importlib
        import sys

        first = importlib.import_module("comfy_api.{first}._input_impl.video_types")
        from comfy_api import latest, v3
        from comfy_api.v3._input_impl.video_types import VideoFromComponents
        import comfy_api.v3._util.video_types

        assert VideoFromComponents is first.VideoFromComponents
        assert VideoFromComponents is latest.VideoFromComponents
        assert v3.VideoFromComponents is latest.VideoFromComponents
        assert v3._util.video_types is latest._util.video_types

        originals = {{name: module for name, module in sys.modules.items()
                     if name.startswith("comfy_api.latest.")}}
        for name, original in originals.items():
            renamed = name.replace("comfy_api.latest.", "comfy_api.v3.", 1)
            assert importlib.import_module(renamed) is original, renamed
    """)
    subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        capture_output=True,
        text=True,
    )


def test_private_module_mutations_and_pickles_share_the_original_namespace(monkeypatch):
    renamed = importlib.import_module("comfy_api.v3._util.video_types")
    original = importlib.import_module("comfy_api.latest._util.video_types")
    replacement = object()
    monkeypatch.setattr(renamed, "VideoComponents", replacement)
    assert original.VideoComponents is replacement

    curve = v3._input.curve_types.LinearCurve([(0.0, 0.0), (1.0, 2.0)])
    restored = pickle.loads(pickle.dumps(curve))
    assert type(restored) is latest._input.curve_types.LinearCurve
    assert restored.interp(0.25) == 0.5


def test_numbered_root_uses_the_same_injected_services(monkeypatch):
    instances = dict(SingletonMetaclass._instances)
    for cls in (latest.ComfyAPI, v3.ComfyAPI, latest.ComfyAPI.Execution):
        instances.pop(cls, None)
    monkeypatch.setattr(SingletonMetaclass, "_instances", instances)

    class Execution(latest.ComfyAPI.Execution):
        async def set_progress(self, value, max_value, **kwargs):
            return value / max_value

    service = Execution()
    latest.ComfyAPI.Execution.inject_instance(service)
    assert v3.ComfyAPI().execution is service
    assert latest.ComfyAPI().execution is service
    assert asyncio.run(v3.ComfyAPI().execution.set_progress(1, 4)) == 0.25
    assert v3.ComfyAPISync().execution.set_progress(1, 4) == 0.25
    assert v3.ComfyAPI().VERSION == "3.0.0"
    assert latest.ComfyAPI().VERSION == "latest"


def test_private_schema_types_remain_native_node_and_output_types():
    from comfy_api.v3._io import ComfyNode, NodeOutput, Schema

    class Add(ComfyNode):
        @classmethod
        def define_schema(cls):
            return Schema(
                node_id="V3AliasAdd",
                inputs=[v3.io.Int.Input("value")],
                outputs=[latest.io.Int.Output()],
            )

        @classmethod
        def execute(cls, value):
            return NodeOutput(value + 1)

    assert issubclass(Add, _ComfyNodeInternal)
    assert isinstance(Add.define_schema(), latest.io.Schema)
    result = Add.execute(2)
    assert isinstance(result, _NodeOutputInternal)
    assert isinstance(result, latest.io.NodeOutput)
    assert result.result == (3,)
