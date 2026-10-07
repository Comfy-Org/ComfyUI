import asyncio

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import nodes
from comfy_api.latest._io import _DynamicGroup


@pytest.fixture(scope="module")
def bundled_nodes():
    asyncio.run(nodes.init_builtin_extra_nodes())
    return nodes.NODE_CLASS_MAPPINGS


def test_camera_nodes_keep_the_ids_saved_workflows_reference(bundled_nodes):
    # CreateCameraInfo moved out of nodes_gaussian_splat.py into nodes_camera.py, and CameraAngle
    # was renamed for display only. Both node ids are already in saved workflows.
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
