"""Tests for the custom node startup error tracking introduced for
Comfy-Org/ComfyUI-Launcher#303.

Covers:
- load_custom_node populates NODE_STARTUP_ERRORS with the correct source
  for each module_parent (custom_nodes / comfy_extras / comfy_api_nodes).
- Composite keying prevents collisions between modules with the same name
  in different sources.
- record_node_startup_error stores the expected fields.
- pyproject.toml metadata is attached when present and omitted when absent.
"""
import textwrap

import pytest

import nodes


@pytest.fixture(autouse=True)
def _clear_startup_errors():
    nodes.NODE_STARTUP_ERRORS.clear()
    yield
    nodes.NODE_STARTUP_ERRORS.clear()


def _write_broken_module(tmp_path, name: str) -> str:
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent("""\
        # Deliberately broken module to exercise startup-error tracking.
        raise RuntimeError("boom from " + __name__)
    """))
    return str(path)


def test_record_node_startup_error_fields(tmp_path):
    err = ValueError("kaboom")
    nodes.record_node_startup_error(
        module_path=str(tmp_path / "my_pack"),
        source="custom_nodes",
        phase="import",
        error=err,
        tb="traceback-text",
    )
    assert "custom_nodes:my_pack" in nodes.NODE_STARTUP_ERRORS
    entry = nodes.NODE_STARTUP_ERRORS["custom_nodes:my_pack"]
    assert entry["source"] == "custom_nodes"
    assert entry["module_name"] == "my_pack"
    assert entry["phase"] == "import"
    assert entry["error"] == "kaboom"
    assert entry["traceback"] == "traceback-text"
    assert entry["module_path"].endswith("my_pack")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module_parent",
    ["custom_nodes", "comfy_extras", "comfy_api_nodes"],
)
async def test_load_custom_node_records_source(tmp_path, module_parent):
    # `source` in the entry should be the same string as `module_parent`.
    module_path = _write_broken_module(tmp_path, "broken_pack")

    success = await nodes.load_custom_node(module_path, module_parent=module_parent)
    assert success is False

    key = f"{module_parent}:broken_pack"
    assert key in nodes.NODE_STARTUP_ERRORS, nodes.NODE_STARTUP_ERRORS
    entry = nodes.NODE_STARTUP_ERRORS[key]
    assert entry["source"] == module_parent
    assert entry["module_name"] == "broken_pack"
    assert entry["phase"] == "import"
    assert "boom from" in entry["error"]
    assert "RuntimeError" in entry["traceback"]


@pytest.mark.asyncio
async def test_load_custom_node_collision_across_sources(tmp_path):
    # Same module name registered as both a custom node and a comfy_extra;
    # composite keying should keep both entries.
    cn_dir = tmp_path / "cn"
    extras_dir = tmp_path / "extras"
    cn_dir.mkdir()
    extras_dir.mkdir()
    cn_path = _write_broken_module(cn_dir, "nodes_audio")
    extras_path = _write_broken_module(extras_dir, "nodes_audio")

    assert await nodes.load_custom_node(cn_path, module_parent="custom_nodes") is False
    assert await nodes.load_custom_node(extras_path, module_parent="comfy_extras") is False

    assert "custom_nodes:nodes_audio" in nodes.NODE_STARTUP_ERRORS
    assert "comfy_extras:nodes_audio" in nodes.NODE_STARTUP_ERRORS
    assert (
        nodes.NODE_STARTUP_ERRORS["custom_nodes:nodes_audio"]["module_path"]
        != nodes.NODE_STARTUP_ERRORS["comfy_extras:nodes_audio"]["module_path"]
    )


@pytest.mark.asyncio
async def test_load_custom_node_attaches_pyproject_metadata(tmp_path):
    pack_dir = tmp_path / "MyCoolPack"
    pack_dir.mkdir()
    (pack_dir / "__init__.py").write_text("raise RuntimeError('boom')\n")
    (pack_dir / "pyproject.toml").write_text(textwrap.dedent("""\
        [project]
        name = "comfyui-mycoolpack"
        version = "1.2.3"

        [project.urls]
        Repository = "https://github.com/example/comfyui-mycoolpack"

        [tool.comfy]
        PublisherId = "example"
        DisplayName = "My Cool Pack"
    """))

    success = await nodes.load_custom_node(str(pack_dir), module_parent="custom_nodes")
    assert success is False

    entry = nodes.NODE_STARTUP_ERRORS["custom_nodes:MyCoolPack"]
    assert "pyproject" in entry, entry
    py = entry["pyproject"]

    # Shape must mirror PyProjectConfig 1:1 so consumers can parse it back
    # through the same pydantic model used by comfy_config.config_parser.
    project = py["project"]
    assert project["name"] == "comfyui-mycoolpack"
    assert project["version"] == "1.2.3"
    assert project["urls"]["repository"] == "https://github.com/example/comfyui-mycoolpack"

    tool_comfy = py["tool_comfy"]
    assert tool_comfy["publisher_id"] == "example"
    assert tool_comfy["display_name"] == "My Cool Pack"


def test_prune_empty_drops_empty_leaves_only():
    src = {
        "keep_str": "x",
        "drop_empty_str": "",
        "drop_none": None,
        "drop_empty_list": [],
        "drop_empty_dict": {},
        "keep_zero": 0,
        "keep_false": False,
        "nested": {
            "drop_me": "",
            "keep_me": "y",
            "deeper": {"only_empties": ""},
        },
        "list_of_dicts": [{"a": ""}, {"a": "z"}],
    }
    result = nodes._prune_empty(src)
    assert result == {
        "keep_str": "x",
        "keep_zero": 0,
        "keep_false": False,
        "nested": {"keep_me": "y"},
        "list_of_dicts": [{"a": "z"}],
    }


@pytest.mark.asyncio
async def test_load_custom_node_no_pyproject_skips_metadata(tmp_path):
    # Single-file extras-style module: no pyproject.toml exists alongside it,
    # so the entry must not contain a 'pyproject' key.
    module_path = _write_broken_module(tmp_path, "lonely")
    assert await nodes.load_custom_node(module_path, module_parent="comfy_extras") is False
    entry = nodes.NODE_STARTUP_ERRORS["comfy_extras:lonely"]
    assert "pyproject" not in entry


@pytest.mark.asyncio
async def test_load_custom_node_arbitrary_module_parent_passes_through(tmp_path):
    # `source` is a free-form string — an unknown module_parent (e.g. a future
    # node-source bucket) should be recorded as-is, not coerced or rejected.
    module_path = _write_broken_module(tmp_path, "future_pack")
    assert await nodes.load_custom_node(module_path, module_parent="future_source") is False
    entry = nodes.NODE_STARTUP_ERRORS["future_source:future_pack"]
    assert entry["source"] == "future_source"


# ---------------------------------------------------------------------------
# Tests for the public reshape/filter helper (nodes.filter_node_startup_errors).
# The HTTP route is a thin wrapper around this helper, so unit-testing it
# directly avoids spinning up an aiohttp app while still covering every
# query-param branch.
# ---------------------------------------------------------------------------


def _seed(*, source, module_name, pack_id=None, module_path="/abs/path"):
    """Insert a synthetic entry directly into NODE_STARTUP_ERRORS."""
    entry = {
        "source": source,
        "module_name": module_name,
        "module_path": module_path,
        "error": "boom",
        "traceback": "tb",
        "phase": "import",
    }
    if pack_id is not None:
        entry["pyproject"] = {"project": {"name": pack_id}}
    nodes.NODE_STARTUP_ERRORS[f"{source}:{module_name}"] = entry


def test_filter_node_startup_errors_strips_module_path_and_groups_by_source():
    _seed(source="custom_nodes", module_name="A", module_path="/x/A")
    _seed(source="comfy_extras", module_name="B", module_path="/x/B")
    grouped = nodes.filter_node_startup_errors()
    assert set(grouped) == {"custom_nodes", "comfy_extras"}
    assert "module_path" not in grouped["custom_nodes"]["A"]
    assert "module_path" not in grouped["comfy_extras"]["B"]


def test_filter_node_startup_errors_source_filter():
    _seed(source="custom_nodes", module_name="A")
    _seed(source="comfy_extras", module_name="B")
    grouped = nodes.filter_node_startup_errors(source="comfy_extras")
    assert set(grouped) == {"comfy_extras"}
    assert set(grouped["comfy_extras"]) == {"B"}
    # Non-matching source filter returns an empty dict, not an error.
    assert nodes.filter_node_startup_errors(source="nope") == {}
    # An explicit empty-string filter is treated as a real value (matches
    # entries whose source is literally ""), NOT silently as "no filter".
    # The HTTP route layer is responsible for coalescing `?source=` to None
    # before calling this helper; this assertion locks that contract in.
    assert nodes.filter_node_startup_errors(source="") == {}


def test_filter_node_startup_errors_module_name_filter():
    _seed(source="custom_nodes", module_name="A")
    _seed(source="comfy_extras", module_name="A")  # same name, different source
    _seed(source="custom_nodes", module_name="C")
    grouped = nodes.filter_node_startup_errors(module_name="A")
    # Both A entries (from different sources) survive the filter and stay in
    # their respective source buckets.
    assert set(grouped) == {"custom_nodes", "comfy_extras"}
    assert set(grouped["custom_nodes"]) == {"A"}
    assert set(grouped["comfy_extras"]) == {"A"}


def test_filter_node_startup_errors_pack_id_filter_matches_only_pyproject_entries():
    _seed(source="custom_nodes", module_name="A", pack_id="comfyui-foo")
    _seed(source="custom_nodes", module_name="B", pack_id="comfyui-bar")
    _seed(source="comfy_extras", module_name="C")  # no pyproject at all
    grouped = nodes.filter_node_startup_errors(pack_id="comfyui-foo")
    assert set(grouped) == {"custom_nodes"}
    assert set(grouped["custom_nodes"]) == {"A"}
    # An entry without a parsed pyproject can never match a pack_id filter.
    assert nodes.filter_node_startup_errors(pack_id="anything-else") == {}


def test_filter_node_startup_errors_filters_combine_with_and():
    _seed(source="custom_nodes", module_name="A", pack_id="comfyui-foo")
    _seed(source="comfy_extras", module_name="A", pack_id="comfyui-foo")
    grouped = nodes.filter_node_startup_errors(
        source="comfy_extras", pack_id="comfyui-foo"
    )
    assert set(grouped) == {"comfy_extras"}
    assert set(grouped["comfy_extras"]) == {"A"}
