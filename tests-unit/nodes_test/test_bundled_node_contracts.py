import asyncio

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import nodes
from comfy_api.latest._io import _DynamicGroup

# nodes_replacements.py only registers node replacements, and needs a running PromptServer to do
# it. It contributes no nodes of its own, so skipping it here costs no coverage.
ALLOWED_IMPORT_FAILURES = {"nodes_replacements.py"}


async def _load_bundled_nodes():
    return await nodes.init_builtin_extra_nodes() + await nodes.init_builtin_api_nodes()


@pytest.fixture(scope="module")
def bundled_nodes():
    class_mappings = dict(nodes.NODE_CLASS_MAPPINGS)
    display_name_mappings = dict(nodes.NODE_DISPLAY_NAME_MAPPINGS)
    # Both halves of a default startup, so the sweep below sees every node a user gets.
    import_failed = asyncio.run(_load_bundled_nodes())
    assert set(import_failed) <= ALLOWED_IMPORT_FAILURES, \
        "a bundled module failed to import, so the checks below would skip its nodes"
    try:
        yield nodes.NODE_CLASS_MAPPINGS
    finally:
        nodes.NODE_CLASS_MAPPINGS.clear()
        nodes.NODE_CLASS_MAPPINGS.update(class_mappings)
        nodes.NODE_DISPLAY_NAME_MAPPINGS.clear()
        nodes.NODE_DISPLAY_NAME_MAPPINGS.update(display_name_mappings)


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


def test_no_bundled_node_offers_a_dynamic_group_widget(bundled_nodes):
    # The bundled frontend has no DynamicGroup renderer, so a bundled node offering one would show
    # an unusable widget. The input stays internal until the frontend ships support for it.
    offenders = sorted(
        name for name, node in bundled_nodes.items()
        if _DynamicGroup.io_type in _strings(node.INPUT_TYPES())
    )
    assert offenders == []
