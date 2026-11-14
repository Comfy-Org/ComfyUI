"""Which web folder ComfyUI serves for a pack that runs in its own process."""

import pytest

import nodes
from comfy.cli_args import args
from comfy_api.v4 import _pack_processes

from conftest import needs_pack_process

pytestmark = needs_pack_process

PACK = """
from comfy_api.v4 import ComfyExtension, io

WEB_DIRECTORY = {web!r}


class Extension(ComfyExtension):
    async def get_node_list(self):
        return []


async def comfy_entrypoint():
    return Extension()
"""


@pytest.fixture
def pack_folder(tmp_path, monkeypatch, node_tables):
    monkeypatch.setattr(args, "pack_process", ["v4_web_pack"])
    folder = tmp_path / "custom_nodes" / "v4_web_pack"
    (folder / "js").mkdir(parents=True)
    yield folder
    _pack_processes.unload(str(folder))


@pytest.mark.asyncio
@pytest.mark.parametrize("web", ["/", "../..", "js/../../.."])
async def test_a_web_folder_outside_the_pack_is_not_served(pack_folder, web):
    (pack_folder / "__init__.py").write_text(PACK.format(web=web), encoding="utf-8")

    assert await nodes.load_custom_node(str(pack_folder))

    assert "v4_web_pack" not in nodes.EXTENSION_WEB_DIRS


@pytest.mark.asyncio
async def test_a_web_folder_inside_the_pack_is_served(pack_folder):
    (pack_folder / "__init__.py").write_text(PACK.format(web="./js"), encoding="utf-8")

    assert await nodes.load_custom_node(str(pack_folder))

    assert nodes.EXTENSION_WEB_DIRS["v4_web_pack"] == str(pack_folder / "js")
