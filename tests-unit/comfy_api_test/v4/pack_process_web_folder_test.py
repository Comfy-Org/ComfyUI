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
    (folder / "pyproject.toml").write_text('[project]\nname = "v4_web_pack"\n')
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


PYPROJECT = '[project]\nname = "{name}"\n\n[tool.comfy]\nweb = "{web}"\n'


@pytest.mark.asyncio
@pytest.mark.parametrize("web", ["/", "../..", "js/../../.."])
async def test_a_web_folder_its_pyproject_names_outside_the_pack_is_not_served(
    pack_folder, web
):
    (pack_folder / "__init__.py").write_text(PACK.format(web=None), encoding="utf-8")
    (pack_folder / "pyproject.toml").write_text(
        PYPROJECT.format(name="v4_web_named", web=web), encoding="utf-8"
    )

    assert await nodes.load_custom_node(str(pack_folder))

    assert "v4_web_named" not in nodes.EXTENSION_WEB_DIRS


@pytest.mark.asyncio
async def test_a_web_folder_its_pyproject_names_inside_the_pack_is_served(pack_folder):
    (pack_folder / "__init__.py").write_text(PACK.format(web=None), encoding="utf-8")
    (pack_folder / "pyproject.toml").write_text(
        PYPROJECT.format(name="v4_web_named", web="js"), encoding="utf-8"
    )

    assert await nodes.load_custom_node(str(pack_folder))

    assert nodes.EXTENSION_WEB_DIRS["v4_web_named"] == str(pack_folder / "js")


@pytest.mark.asyncio
async def test_a_pyproject_the_pack_changes_after_the_loader_read_it_does_not_keep_a_folder_served(
    pack_folder, monkeypatch
):
    from comfy_config import config_parser

    (pack_folder / "__init__.py").write_text(PACK.format(web=None), encoding="utf-8")
    pyproject = pack_folder / "pyproject.toml"
    pyproject.write_text(PYPROJECT.format(name="v4_web_named", web="/"), encoding="utf-8")
    read = config_parser.extract_node_configuration

    def read_then_changed(path):
        project = read(path)
        inside = PYPROJECT.format(name="v4_web_named", web="js")
        pyproject.write_text(inside, encoding="utf-8")
        return project

    monkeypatch.setattr(config_parser, "extract_node_configuration", read_then_changed)

    assert await nodes.load_custom_node(str(pack_folder))

    assert "v4_web_named" not in nodes.EXTENSION_WEB_DIRS


@pytest.mark.asyncio
async def test_a_web_folder_of_another_pack_that_was_overwritten_is_put_back(
    pack_folder, monkeypatch
):
    (pack_folder / "__init__.py").write_text(PACK.format(web=None), encoding="utf-8")
    (pack_folder / "pyproject.toml").write_text(
        PYPROJECT.format(name="v4_web_other", web="/"), encoding="utf-8"
    )
    monkeypatch.setitem(nodes.EXTENSION_WEB_DIRS, "v4_web_other", "/the/other/packs/js")

    assert await nodes.load_custom_node(str(pack_folder))

    assert nodes.EXTENSION_WEB_DIRS["v4_web_other"] == "/the/other/packs/js"
