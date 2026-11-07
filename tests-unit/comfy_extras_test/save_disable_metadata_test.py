import io as pyio
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import av
import pytest
import torch

import folder_paths
from comfy.cli_args import args
from comfy_api.latest import IO

previous_nodes = sys.modules.get("nodes")
previous_server = sys.modules.get("server")
sys.modules["nodes"] = MagicMock(MAX_RESOLUTION=16384)
sys.modules["server"] = MagicMock()
from comfy_extras.nodes_images import SaveSVGNode  # noqa: E402
from comfy_extras.nodes_video import SaveWEBM  # noqa: E402
for name, previous in (("nodes", previous_nodes), ("server", previous_server)):
    if previous is None:
        sys.modules.pop(name)
    else:
        sys.modules[name] = previous

HIDDEN = SimpleNamespace(prompt={"1": {"class_type": "secret-prompt"}}, extra_pnginfo={"workflow": {"nodes": []}})
SVG = b'<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"></svg>'


@pytest.fixture
def output_dir(tmp_path):
    saved = folder_paths.get_output_directory()
    folder_paths.set_output_directory(str(tmp_path))
    yield tmp_path
    folder_paths.set_output_directory(saved)


@pytest.mark.parametrize("disable_metadata", [False, True])
def test_save_webm_respects_disable_metadata(output_dir, monkeypatch, disable_metadata):
    monkeypatch.setattr(args, "disable_metadata", disable_metadata)
    monkeypatch.setattr(SaveWEBM, "hidden", HIDDEN, raising=False)

    SaveWEBM.execute(torch.rand(2, 16, 16, 3), "vp9", 8.0, "t", 32.0)

    with av.open(str(next(output_dir.glob("*.webm")))) as container:
        keys = {k.lower() for k in container.metadata}
    assert ("prompt" in keys) is not disable_metadata
    assert ("workflow" in keys) is not disable_metadata


@pytest.mark.parametrize("disable_metadata", [False, True])
def test_save_svg_respects_disable_metadata(output_dir, monkeypatch, disable_metadata):
    monkeypatch.setattr(args, "disable_metadata", disable_metadata)
    monkeypatch.setattr(SaveSVGNode, "hidden", HIDDEN, raising=False)

    SaveSVGNode.execute(IO.SVG.Type([pyio.BytesIO(SVG)]), "t")

    content = next(output_dir.glob("*.svg")).read_text()
    assert ("secret-prompt" in content) is not disable_metadata
    assert ("<metadata>" in content) is not disable_metadata
