import pytest
from aiohttp import web
from unittest.mock import patch
from app.custom_node_manager import CustomNodeManager
import json

pytestmark = (
    pytest.mark.asyncio
)  # This applies the asyncio mark to all test functions in the module


@pytest.fixture
def custom_node_manager():
    return CustomNodeManager()


@pytest.fixture
def app(custom_node_manager):
    app = web.Application()
    routes = web.RouteTableDef()
    custom_node_manager.add_routes(
        routes, app, [("ComfyUI-TestExtension1", "ComfyUI-TestExtension1")]
    )
    app.add_routes(routes)
    return app


async def test_get_workflow_templates(aiohttp_client, app, tmp_path):
    client = await aiohttp_client(app)
    # Setup temporary custom nodes file structure with 1 workflow file
    custom_nodes_dir = tmp_path / "custom_nodes"
    example_workflows_dir = (
        custom_nodes_dir / "ComfyUI-TestExtension1" / "example_workflows"
    )
    example_workflows_dir.mkdir(parents=True)
    template_file = example_workflows_dir / "workflow1.json"
    template_file.write_text("")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("COMFY_CUSTOM_NODE_EXAMPLE_WORKFLOWS_ENABLED", "true")

    try:
        with patch(
            "folder_paths.folder_names_and_paths",
            {"custom_nodes": ([str(custom_nodes_dir)], None)},
        ):
            response = await client.get("/workflow_templates")
            assert response.status == 200
            workflows_dict = await response.json()
            assert isinstance(workflows_dict, dict)
            assert "ComfyUI-TestExtension1" in workflows_dict
            assert isinstance(workflows_dict["ComfyUI-TestExtension1"], list)
            assert workflows_dict["ComfyUI-TestExtension1"][0] == "workflow1"
    finally:
        monkeypatch.undo()


async def test_get_workflow_templates_hidden_by_default(aiohttp_client, app, tmp_path):
    client = await aiohttp_client(app)
    custom_nodes_dir = tmp_path / "custom_nodes"
    example_workflows_dir = (
        custom_nodes_dir / "ComfyUI-TestExtension1" / "example_workflows"
    )
    example_workflows_dir.mkdir(parents=True)
    (example_workflows_dir / "workflow1.json").write_text("")

    with patch(
        "folder_paths.folder_names_and_paths",
        {"custom_nodes": ([str(custom_nodes_dir)], None)},
    ):
        response = await client.get("/workflow_templates")
        assert response.status == 200
        workflows_dict = await response.json()
        assert workflows_dict == {}


async def test_get_workflow_templates_disabled_by_env(
    aiohttp_client, app, tmp_path, monkeypatch
):
    client = await aiohttp_client(app)
    custom_nodes_dir = tmp_path / "custom_nodes"
    example_workflows_dir = (
        custom_nodes_dir / "ComfyUI-TestExtension1" / "example_workflows"
    )
    example_workflows_dir.mkdir(parents=True)
    (example_workflows_dir / "workflow1.json").write_text("")

    monkeypatch.setenv("COMFY_CUSTOM_NODE_EXAMPLE_WORKFLOWS_ENABLED", "false")

    with patch(
        "folder_paths.folder_names_and_paths",
        {"custom_nodes": ([str(custom_nodes_dir)], None)},
    ):
        response = await client.get("/workflow_templates")
        assert response.status == 200
        workflows_dict = await response.json()
        assert workflows_dict == {}


async def test_get_workflow_templates_allowlist(
    aiohttp_client, app, tmp_path, monkeypatch
):
    client = await aiohttp_client(app)
    custom_nodes_dir = tmp_path / "custom_nodes"

    ext1_dir = custom_nodes_dir / "ComfyUI-TestExtension1" / "example_workflows"
    ext2_dir = custom_nodes_dir / "ComfyUI-TestExtension2" / "example_workflows"
    ext1_dir.mkdir(parents=True)
    ext2_dir.mkdir(parents=True)
    (ext1_dir / "workflow1.json").write_text("")
    (ext2_dir / "workflow2.json").write_text("")

    monkeypatch.setenv(
        "COMFY_CUSTOM_NODE_EXAMPLE_WORKFLOWS_ALLOWLIST",
        "ComfyUI-TestExtension2",
    )

    with patch(
        "folder_paths.folder_names_and_paths",
        {"custom_nodes": ([str(custom_nodes_dir)], None)},
    ):
        response = await client.get("/workflow_templates")
        assert response.status == 200
        workflows_dict = await response.json()
        assert "ComfyUI-TestExtension1" not in workflows_dict
        assert workflows_dict["ComfyUI-TestExtension2"] == ["workflow2"]


async def test_build_translations_empty_when_no_locales(custom_node_manager, tmp_path):
    custom_nodes_dir = tmp_path / "custom_nodes"
    custom_nodes_dir.mkdir(parents=True)

    with patch("folder_paths.get_folder_paths", return_value=[str(custom_nodes_dir)]):
        translations = custom_node_manager.build_translations()
        assert translations == {}


async def test_build_translations_loads_all_files(custom_node_manager, tmp_path):
    # Setup test directory structure
    custom_nodes_dir = tmp_path / "custom_nodes" / "test-extension"
    locales_dir = custom_nodes_dir / "locales" / "en"
    locales_dir.mkdir(parents=True)

    # Create test translation files
    main_content = {"title": "Test Extension"}
    (locales_dir / "main.json").write_text(json.dumps(main_content))

    node_defs = {"node1": "Node 1"}
    (locales_dir / "nodeDefs.json").write_text(json.dumps(node_defs))

    commands = {"cmd1": "Command 1"}
    (locales_dir / "commands.json").write_text(json.dumps(commands))

    settings = {"setting1": "Setting 1"}
    (locales_dir / "settings.json").write_text(json.dumps(settings))

    with patch(
        "folder_paths.get_folder_paths", return_value=[tmp_path / "custom_nodes"]
    ):
        translations = custom_node_manager.build_translations()

        assert translations == {
            "en": {
                "title": "Test Extension",
                "nodeDefs": {"node1": "Node 1"},
                "commands": {"cmd1": "Command 1"},
                "settings": {"setting1": "Setting 1"},
            }
        }


async def test_build_translations_handles_invalid_json(custom_node_manager, tmp_path):
    # Setup test directory structure
    custom_nodes_dir = tmp_path / "custom_nodes" / "test-extension"
    locales_dir = custom_nodes_dir / "locales" / "en"
    locales_dir.mkdir(parents=True)

    # Create valid main.json
    main_content = {"title": "Test Extension"}
    (locales_dir / "main.json").write_text(json.dumps(main_content))

    # Create invalid JSON file
    (locales_dir / "nodeDefs.json").write_text("invalid json{")

    with patch(
        "folder_paths.get_folder_paths", return_value=[tmp_path / "custom_nodes"]
    ):
        translations = custom_node_manager.build_translations()

        assert translations == {
            "en": {
                "title": "Test Extension",
            }
        }


async def test_build_translations_merges_multiple_extensions(
    custom_node_manager, tmp_path
):
    # Setup test directory structure for two extensions
    custom_nodes_dir = tmp_path / "custom_nodes"
    ext1_dir = custom_nodes_dir / "extension1" / "locales" / "en"
    ext2_dir = custom_nodes_dir / "extension2" / "locales" / "en"
    ext1_dir.mkdir(parents=True)
    ext2_dir.mkdir(parents=True)

    # Create translation files for extension 1
    ext1_main = {"title": "Extension 1", "shared": "Original"}
    (ext1_dir / "main.json").write_text(json.dumps(ext1_main))

    # Create translation files for extension 2
    ext2_main = {"description": "Extension 2", "shared": "Override"}
    (ext2_dir / "main.json").write_text(json.dumps(ext2_main))

    with patch("folder_paths.get_folder_paths", return_value=[str(custom_nodes_dir)]):
        translations = custom_node_manager.build_translations()

        assert translations == {
            "en": {
                "title": "Extension 1",
                "description": "Extension 2",
                "shared": "Override",  # Second extension should override first
            }
        }
