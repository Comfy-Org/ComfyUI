import asyncio

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import nodes
from comfy_api.latest import io
from comfy_api.latest._io import _DynamicGroup

# Startup only logs an import failure and carries on. The sweep below is only meaningful over a
# complete node list, so here a failure is fatal instead. nodes_replacements.py is exempt: it needs
# a running PromptServer, and registers no nodes of its own, so skipping it costs no coverage.
ALLOWED_IMPORT_FAILURES = {"nodes_replacements.py"}


async def _load_bundled_nodes():
    return await nodes.init_builtin_extra_nodes() + await nodes.init_builtin_api_nodes()


@pytest.fixture(scope="module")
def bundled_nodes():
    class_mappings = dict(nodes.NODE_CLASS_MAPPINGS)
    display_name_mappings = dict(nodes.NODE_DISPLAY_NAME_MAPPINGS)
    try:
        # Both halves of a default startup, so the sweep below sees every node a user gets. The
        # loaded modules themselves stay resident; only the mappings are put back.
        import_failed = asyncio.run(_load_bundled_nodes())
        assert set(import_failed) <= ALLOWED_IMPORT_FAILURES, \
            "a bundled module failed to import, so the checks below would skip its nodes"
        yield nodes.NODE_CLASS_MAPPINGS
    finally:
        nodes.NODE_CLASS_MAPPINGS.clear()
        nodes.NODE_CLASS_MAPPINGS.update(class_mappings)
        nodes.NODE_DISPLAY_NAME_MAPPINGS.clear()
        nodes.NODE_DISPLAY_NAME_MAPPINGS.update(display_name_mappings)


def test_both_halves_of_a_default_startup_are_loaded(bundled_nodes):
    # The sweep below is only as good as its node list, and the api nodes are a third of it. Pin one
    # from each half so a glob that stops matching shows up here instead of quietly shrinking it.
    assert "KSampler" in bundled_nodes
    assert "CameraAngle" in bundled_nodes
    assert "OpenAIGPTImage1" in bundled_nodes


def test_camera_nodes_register_under_the_ids_workflows_reference(bundled_nodes):
    # CreateCameraInfo has shipped since v0.23.0 and just moved out of nodes_gaussian_splat.py into
    # nodes_camera.py, so saved workflows already carry its id. CameraAngle is new, and the frontend
    # picker ships against this id and display name.
    assert "CreateCameraInfo" in bundled_nodes
    assert "CameraAngle" in bundled_nodes
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["CameraAngle"] == "Compose Camera Angle Prompt"


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def test_a_dynamic_group_widget_is_detectable():
    class Grouped(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=cls.__name__, inputs=[
                _DynamicGroup.Input("rows", template=[io.Float.Input("x")]),
            ], outputs=[])

    assert _DynamicGroup.io_type in _strings(Grouped.INPUT_TYPES())


def test_no_bundled_node_offers_a_dynamic_group_widget(bundled_nodes):
    # comfyui-frontend-package 1.55.x has no DynamicGroup renderer, so a bundled node offering one
    # would show an unusable widget. Frontend support landed in 1.57.0; drop this once the pin in
    # requirements.txt reaches a version that has it.
    offenders = sorted(
        name for name, node in bundled_nodes.items()
        if _DynamicGroup.io_type in _strings(node.INPUT_TYPES())
    )
    assert offenders == []
