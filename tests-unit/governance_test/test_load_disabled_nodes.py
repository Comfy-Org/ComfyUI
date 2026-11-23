from pathlib import Path

import pytest
import yaml

from app.governance import load_disabled_nodes


VALID_CONFIGS = [
    ("disabled_nodes:\n  - A\n  - B\n", {"A", "B"}),
    ("disabled_nodes: []\n", set()),
    ("disabled_nodes:\n  - A\n  - A\n", {"A"}),
]

INVALID_CONFIGS = [
    ("disabled_nodes: [\n", yaml.YAMLError),
    ("- A\n", ValueError),
    ("other: []\n", ValueError),
    ('disabled_nodes: "A"\n', ValueError),
    ("disabled_nodes: [1]\n", ValueError),
]


@pytest.mark.parametrize("config,expected", VALID_CONFIGS)
def test_load_disabled_nodes_returns_expected_set(tmp_path: Path, config: str, expected: set[str]) -> None:
    config_path = tmp_path / "disabled_nodes.yaml"
    config_path.write_text(config, encoding="utf-8")

    assert load_disabled_nodes(str(config_path)) == expected


def test_load_disabled_nodes_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_disabled_nodes(str(tmp_path / "missing.yaml"))


@pytest.mark.parametrize("config,expected_error", INVALID_CONFIGS)
def test_load_disabled_nodes_rejects_invalid_config(
    tmp_path: Path,
    config: str,
    expected_error: type[Exception],
) -> None:
    config_path = tmp_path / "disabled_nodes.yaml"
    config_path.write_text(config, encoding="utf-8")

    with pytest.raises(expected_error):
        load_disabled_nodes(str(config_path))
