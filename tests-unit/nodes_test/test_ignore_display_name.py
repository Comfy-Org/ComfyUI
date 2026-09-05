import sys

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import nodes


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _restore_node_mappings():
    class_mappings = dict(nodes.NODE_CLASS_MAPPINGS)
    display_name_mappings = dict(nodes.NODE_DISPLAY_NAME_MAPPINGS)
    try:
        yield
    finally:
        nodes.NODE_CLASS_MAPPINGS.clear()
        nodes.NODE_CLASS_MAPPINGS.update(class_mappings)
        nodes.NODE_DISPLAY_NAME_MAPPINGS.clear()
        nodes.NODE_DISPLAY_NAME_MAPPINGS.update(display_name_mappings)
        sys.modules.pop("test_v1_custom_node", None)
        sys.modules.pop("test_v3_custom_node", None)


async def test_load_custom_node_skips_display_names_for_ignored_nodes(tmp_path, monkeypatch):
    v1_module = tmp_path / "test_v1_custom_node.py"
    v1_module.write_text(
        "NODE_CLASS_MAPPINGS = {\"LeakTest\": object}\n"
        "NODE_DISPLAY_NAME_MAPPINGS = {\"LeakTest\": \"Leak Test\"}\n",
    )

    v3_module = tmp_path / "test_v3_custom_node.py"
    v3_module.write_text(
        "from comfy_api.latest import ComfyExtension\n\n"
        "class LeakTestV3Node:\n"
        "    @classmethod\n"
        "    def GET_SCHEMA(cls):\n"
        "        class Schema:\n"
        "            node_id = \"LeakTestV3\"\n"
        "            display_name = \"Leak Test V3\"\n\n"
        "        return Schema()\n\n\n"
        "class TestExtension(ComfyExtension):\n"
        "    async def get_node_list(self):\n"
        "        return [LeakTestV3Node]\n\n\n"
        "async def comfy_entrypoint():\n"
        "    return TestExtension()\n",
    )

    monkeypatch.syspath_prepend(str(tmp_path))

    assert await nodes.load_custom_node(str(v1_module), ignore={"LeakTest"})
    assert await nodes.load_custom_node(str(v3_module), ignore={"LeakTestV3"})

    assert "LeakTest" not in nodes.NODE_DISPLAY_NAME_MAPPINGS
    assert "LeakTestV3" not in nodes.NODE_DISPLAY_NAME_MAPPINGS
