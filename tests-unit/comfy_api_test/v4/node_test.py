import os
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

import folder_paths
import nodes
from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder
from conftest import CANCEL
from model_test import model as model_fixture
from protocol import BinaryEventTypes

model = model_fixture

COMFY_ROOT = os.path.dirname(os.path.abspath(nodes.__file__))
SCALE = (
    '"ImageScale", {{"image": torch.rand(1, 6, 8, 3), '
    '"upscale_method": "nearest-exact", "width": 4, "height": 3, "crop": "disabled"}}'
)


def error(executor):
    [data] = [
        data for event, data in executor.status_messages if event == "execution_error"
    ]
    return data["exception_type"], data["exception_message"].strip()


@pytest.fixture
def load_comfy_nodes(monkeypatch):
    """Load one of ComfyUI's own node modules as ComfyUI does at start."""
    for attribute in (
        "NODE_CLASS_MAPPINGS",
        "NODE_DISPLAY_NAME_MAPPINGS",
        "LOADED_MODULE_DIRS",
        "EXTENSION_WEB_DIRS",
    ):
        monkeypatch.setattr(nodes, attribute, dict(getattr(nodes, attribute)))

    async def load(path, module_parent="comfy_extras"):
        monkeypatch.setitem(sys.modules, os.path.splitext(path)[0], None)
        assert await nodes.load_custom_node(path, module_parent=module_parent)

    return load


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_run_returns_a_built_in_nodes_outputs_as_a_tuple(run_node, synchronous):
    executor, server = await run_node(
        f"[image] = {{call}}.node.run({SCALE})\nassert image.shape == (1, 3, 4, 3)",
        synchronous,
    )

    assert executor.success, executor.status_messages


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_run_loads_an_image_by_file_name(
    run_node, monkeypatch, tmp_path, synchronous
):
    monkeypatch.setattr(folder_paths, "input_directory", str(tmp_path))
    Image.new("RGBA", (5, 4), (255, 0, 0, 128)).save(tmp_path / "red.png")

    executor, server = await run_node(
        'image, mask = {call}.node.run("LoadImage", {{"image": "red.png"}})\n'
        "assert image.shape == (1, 4, 5, 3) and image[0, 0, 0, 0] == 1.0\n"
        "assert mask.shape == (1, 4, 5) and 0.49 < mask[0, 0, 0] < 0.5",
        synchronous,
    )

    assert executor.success, executor.status_messages


@pytest.mark.asyncio
# The body takes a model the test made in ComfyUI's memory; a pack process can only be given one.
@pytest.mark.parametrize("run_node", ["in_process"], indirect=True)
async def test_run_samples_with_ksampler_and_keyword_inputs(
    run_node, monkeypatch, model
):
    monkeypatch.setitem(
        sys.modules, "node_run_test_model", SimpleNamespace(model=model)
    )

    executor, server = await run_node(
        "from node_run_test_model import model\n"
        "cond = [[torch.zeros(1, 1, 768), {{}}]]\n"
        '[latent] = {call}.node.run("KSampler", {{'
        '"model": model, "seed": 3, "steps": 1, "cfg": 2.0, '
        '"sampler_name": "euler", "scheduler": "normal", "positive": cond, "negative": cond, '
        '"latent_image": {{"samples": torch.zeros(1, 4, 8, 8)}}, "denoise": 1.0}})\n'
        'assert latent["samples"].shape == (1, 4, 8, 8) and latent["samples"].abs().sum() > 0'
    )

    assert executor.success, executor.status_messages


@pytest.mark.asyncio
async def test_run_fills_a_v3_nodes_hidden_unique_id_with_the_running_node(
    run_node, load_comfy_nodes
):
    await load_comfy_nodes(os.path.join(COMFY_ROOT, "comfy_extras", "nodes_images.py"))

    executor, server = await run_node(
        'size = {call}.node.run("GetImageSize", {{"image": torch.rand(2, 12, 16, 3)}})\n'
        "assert size == (16, 12, 2)"
    )

    assert executor.success, executor.status_messages
    [text] = [data for event, data, _ in server.sent if event == BinaryEventTypes.TEXT]
    assert text.startswith(b"\x00\x00\x00\x02v4width: 16")


@pytest.mark.asyncio
async def test_run_returns_outputs_declared_as_lists_as_lists(
    run_node, load_comfy_nodes
):
    await load_comfy_nodes(os.path.join(COMFY_ROOT, "comfy_extras", "nodes_images.py"))

    executor, server = await run_node(
        '[tiles] = {call}.node.run("SplitImageToTileList", {{"image": torch.rand(1, 128, 128, 3), '
        '"tile_width": 64, "tile_height": 64, "overlap": 0}})\n'
        "assert [tile.shape for tile in tiles] == [(1, 64, 64, 3)] * 4"
    )

    assert executor.success, executor.status_messages


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_run_output_saves_and_returns_the_ui_without_a_workflow(
    run_node, monkeypatch, tmp_path, synchronous
):
    monkeypatch.setattr(folder_paths, "output_directory", str(tmp_path))

    executor, server = await run_node(
        'output = {call}.node.run_output("SaveImage", '
        '{{"images": torch.rand(1, 4, 5, 3), "filename_prefix": "v4"}})\n'
        'assert output["ui"] == {{"images": '
        '[{{"filename": "v4_00001_.png", "subfolder": "", "type": "output"}}]}}\n'
        '[images] = output["result"]\n'
        "assert images.shape == (1, 4, 5, 3)",
        synchronous,
    )

    assert executor.success, executor.status_messages
    assert "executed" not in [event for event, _, _ in server.sent]
    with Image.open(tmp_path / "v4_00001_.png") as saved:
        assert saved.info["prompt"] == "{}" and "workflow" not in saved.info


@pytest.mark.asyncio
async def test_run_maps_a_removed_node_to_its_registered_replacement(
    run_node, load_comfy_nodes, monkeypatch
):
    for module in ("nodes_latent.py", "nodes_post_processing.py"):
        await load_comfy_nodes(os.path.join(COMFY_ROOT, "comfy_extras", module))
    monkeypatch.delitem(nodes.NODE_CLASS_MAPPINGS, "ImageBatch")

    # The replacement ComfyUI registers (comfy_extras/nodes_replacements.py), through the API.
    executor, server = await run_node(
        "{call}.node_replacement.register(io.NodeReplace(\n"
        '    new_node_id="BatchImagesNode", old_node_id="ImageBatch", input_mapping=[\n'
        '        {{"new_id": "images.image0", "old_id": "image1"}},\n'
        '        {{"new_id": "images.image1", "old_id": "image2"}},\n'
        "    ]\n"
        "))\n"
        '[batch] = {call}.node.run("ImageBatch", '
        '{{"image1": torch.zeros(1, 4, 4, 3), "image2": torch.ones(1, 4, 4, 3)}})\n'
        "assert batch.shape == (2, 4, 4, 3) and batch[1].min() == 1.0"
    )

    assert executor.success, executor.status_messages


def replace_old_size(output_mapping):
    """Register OldSize, removed, as replaced by GetImageSize with `output_mapping`."""
    return (
        "from comfy_api.v4 import io as v4io\n"
        + "await eng.node_replacement.register(v4io.NodeReplace('GetImageSize', 'OldSize', "
        + "input_mapping=[{{'new_id': 'image', 'old_id': 'image'}}], "
        + f"output_mapping={output_mapping!r}))\n".replace("{", "{{").replace("}", "}}")
        + "image = torch.rand(2, 12, 16, 3)\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output_mapping, outputs",
    [
        ([{"old_idx": 0, "new_idx": 1}, {"old_idx": 1, "new_idx": 2}], (12, 2, 2)),
        ([{"old_idx": 2, "new_idx": 0}, {"old_idx": 0, "new_idx": 1}], (12, 12, 16)),
    ],
)
async def test_run_returns_a_removed_nodes_outputs_where_a_workflows_links_would_point(
    run_node, load_comfy_nodes, output_mapping, outputs
):
    await load_comfy_nodes(os.path.join(COMFY_ROOT, "comfy_extras", "nodes_images.py"))

    executor, server = await run_node(
        replace_old_size(output_mapping)
        + "assert await eng.node.run('OldSize', {{'image': image}}) == "
        + f"{outputs!r}\n"
        + "output = await eng.node.run_output('OldSize', {{'image': image}})\n"
        + f"assert output['result'] == {outputs!r}"
    )

    assert executor.success, executor.status_messages


@pytest.mark.asyncio
async def test_run_refuses_a_removed_node_whose_output_has_no_replacement(
    run_node, load_comfy_nodes
):
    await load_comfy_nodes(os.path.join(COMFY_ROOT, "comfy_extras", "nodes_images.py"))

    executor, server = await run_node(
        replace_old_size([{"old_idx": 0, "new_idx": 1}, {"old_idx": 1, "new_idx": 3}])
        + "await eng.node.run('OldSize', {{'image': image}})"
    )

    assert error(executor) == (
        "ValueError",
        "'OldSize' output 1 has no counterpart in its replacement 'GetImageSize'",
    )
    assert BinaryEventTypes.TEXT not in [event for event, _, _ in server.sent]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "inputs, message",
    [
        ('"upscale_method": "bogus", "width": 4', "Value not in list"),
        ('"upscale_method": "nearest-exact", "width": -1', "smaller than min"),
        ('"upscale_method": "nearest-exact"', "Required input is missing"),
    ],
)
async def test_run_rejects_inputs_a_workflow_would_reject(run_node, inputs, message):
    executor, server = await run_node(
        '{call}.node.run("ImageScale", {{"image": torch.rand(1, 6, 8, 3), '
        + inputs
        + ', "height": 3, "crop": "disabled"}})'
    )

    exception_type, exception_message = error(executor)
    assert exception_type == "ValueError" and message in exception_message


class TableNode:
    """A pack's node put straight into ComfyUI's node table, as some packs do."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "run"

    def run(self):
        return (1,)


Generated = type("Generated", (TableNode,), {"__module__": "nodes_built_at_run_time"})


class TableNodeV3(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="TableNodeV3", outputs=[io.Int.Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput(1)


@pytest.mark.asyncio
async def test_run_counts_every_node_comfyui_loads_itself_as_its_own(monkeypatch):
    from comfy_api.v4._node import _defined_by_comfyui

    for attribute in (
        "NODE_CLASS_MAPPINGS",
        "NODE_DISPLAY_NAME_MAPPINGS",
        "LOADED_MODULE_DIRS",
        "EXTENSION_WEB_DIRS",
    ):
        monkeypatch.setattr(nodes, attribute, dict(getattr(nodes, attribute)))
    await nodes.init_builtin_extra_nodes()
    await nodes.init_builtin_api_nodes()

    foreign = [
        node_id
        for node_id, node_class in nodes.NODE_CLASS_MAPPINGS.items()
        if not _defined_by_comfyui(node_class)
    ]
    assert len(nodes.NODE_CLASS_MAPPINGS) > 500 and foreign == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "node", ["V4Node", "NoSuchNode", "TableNode", "TableNodeV3", "Generated"]
)
async def test_run_refuses_a_node_that_is_not_comfyuis_own(run_node, monkeypatch, node):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "TableNode", TableNode)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "TableNodeV3", TableNodeV3)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Generated", Generated)

    executor, server = await run_node(f'{{call}}.node.run("{node}", {{{{}}}})')

    assert error(executor) == (
        "ValueError",
        f"'{node}' is not one of ComfyUI's own nodes",
    )


ALIAS_PACK = """\
import nodes

NODE_CLASS_MAPPINGS = {"AliasInvert": nodes.ImageInvert}
"""


@pytest.mark.asyncio
@pytest.mark.parametrize("node", ["ImageInvert", "AliasInvert"])
async def test_run_runs_a_built_in_node_a_pack_also_lists(
    run_node, load_pack, monkeypatch, node
):
    monkeypatch.setattr(
        nodes.ImageInvert, "RELATIVE_PYTHON_MODULE", None, raising=False
    )
    await load_pack(ALIAS_PACK, "alias_pack")

    executor, server = await run_node(
        f'[inverted] = {{call}}.node.run("{node}", {{{{"image": torch.zeros(1, 2, 2, 3)}}}})\n'
        "assert inverted.min() == 1.0"
    )

    assert executor.success, executor.status_messages


@pytest.mark.asyncio
async def test_run_refuses_a_packs_subclass_of_a_built_in_node(
    run_node, load_comfy_nodes, monkeypatch
):
    await load_comfy_nodes(os.path.join(COMFY_ROOT, "comfy_extras", "nodes_images.py"))
    subclass = type("PackImageSize", (nodes.NODE_CLASS_MAPPINGS["GetImageSize"],), {})
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "PackImageSize", subclass)

    executor, server = await run_node(
        '{call}.node.run("PackImageSize", {{"image": torch.rand(1, 4, 4, 3)}})'
    )

    assert error(executor) == (
        "ValueError",
        "'PackImageSize' is not one of ComfyUI's own nodes",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_run_refuses_an_asynchronous_partner_api_node(
    run_node, load_comfy_nodes, synchronous
):
    await load_comfy_nodes(
        os.path.join(COMFY_ROOT, "comfy_api_nodes", "nodes_ideogram.py"),
        "comfy_api_nodes",
    )

    executor, server = await run_node(
        '{call}.node.run("IdeogramV3", {{}})', synchronous
    )

    assert error(executor) == (
        "ValueError",
        "'IdeogramV3' runs asynchronously and cannot be run by a call",
    )


@pytest.mark.asyncio
async def test_run_refuses_a_node_that_expands_into_other_nodes(run_node, monkeypatch):
    def expand(self, image):
        graph = GraphBuilder()
        empty = graph.node("EmptyImage", width=8, height=8)
        return {"result": (empty.out(0),), "expand": graph.finalize()}

    monkeypatch.setattr(nodes.ImageInvert, "invert", expand)

    executor, server = await run_node(
        '{call}.node.run("ImageInvert", {{"image": torch.rand(1, 4, 4, 3)}})'
    )

    assert error(executor) == (
        "ValueError",
        "'ImageInvert' expands into other nodes and cannot be run by a call",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_run_stops_when_the_user_cancelled(run_node, synchronous):
    executor, server = await run_node(
        CANCEL + f"{{call}}.node.run({SCALE})\n" + '{call}.ui.send("ran", {{}})',
        synchronous,
    )

    assert "execution_interrupted" in [event for event, _ in executor.status_messages]
    assert "ran" not in [event for event, _, _ in server.sent]


@pytest.mark.asyncio
async def test_run_refuses_a_node_of_a_pack_in_comfyuis_custom_nodes_folder(
    run_node, monkeypatch
):
    module = SimpleNamespace(
        __file__=os.path.join(COMFY_ROOT, "custom_nodes", "pack", "__init__.py")
    )
    monkeypatch.setitem(sys.modules, "v4_custom_nodes_pack", module)

    class PackNode:
        __module__ = "v4_custom_nodes_pack"
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"
        CATEGORY = "test"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

        def run(self):
            return (1,)

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "PackNode", PackNode)

    executor, server = await run_node("await eng.node.run('PackNode', {{}})")

    assert error(executor) == (
        "ValueError",
        "'PackNode' is not one of ComfyUI's own nodes",
    )


@pytest.mark.asyncio
async def test_run_returns_every_output_of_a_removed_node_with_more_outputs(
    run_node, load_comfy_nodes
):
    await load_comfy_nodes(os.path.join(COMFY_ROOT, "comfy_extras", "nodes_images.py"))

    executor, server = await run_node(
        replace_old_size([{"old_idx": 3, "new_idx": 0}])
        + "assert await eng.node.run('OldSize', {{'image': image}}) == (16, 12, 2, 16)"
    )

    assert executor.success, executor.status_messages
