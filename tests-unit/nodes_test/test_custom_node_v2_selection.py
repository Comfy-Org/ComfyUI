from pathlib import Path

import pytest

import folder_paths
import nodes


@pytest.mark.asyncio
async def test_external_custom_nodes_select_one_entrypoint_per_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_nodes = tmp_path / "custom_nodes"
    converted = custom_nodes / "converted-pack"
    converted_v2 = converted / "v2"
    legacy = custom_nodes / "legacy-pack"
    converted_v2.mkdir(parents=True)
    legacy.mkdir(parents=True)
    (converted / "__init__.py").write_text("", encoding="utf-8")
    (converted_v2 / "__init__.py").write_text("", encoding="utf-8")
    (legacy / "__init__.py").write_text("", encoding="utf-8")

    loaded: list[tuple[Path, str | None]] = []

    async def record_load(
        module_path: str,
        _ignore: set[str],
        module_parent: str,
        module_name: str | None = None,
    ) -> bool:
        assert module_parent == "custom_nodes"
        loaded.append((Path(module_path), module_name))
        return True

    monkeypatch.setattr(
        folder_paths,
        "get_folder_paths",
        lambda _folder_name: [str(custom_nodes)],
    )
    monkeypatch.setattr(nodes, "load_custom_node", record_load)
    monkeypatch.setattr(nodes.args, "disable_all_custom_nodes", False)
    monkeypatch.setattr(nodes.args, "enable_manager", False)
    monkeypatch.setattr(nodes.sys, "path", nodes.sys.path.copy())

    await nodes.init_external_custom_nodes()

    assert sorted(loaded) == [
        (converted_v2, "converted-pack"),
        (legacy, "legacy-pack"),
    ]
