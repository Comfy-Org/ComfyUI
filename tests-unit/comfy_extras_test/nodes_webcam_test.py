"""Regression test for https://github.com/Comfy-Org/ComfyUI/issues/15756

WebcamCapture.load_capture resolved the incoming ``[temp]``-annotated image
name via folder_paths.get_annotated_filepath(), then handed the resulting
*absolute* path to LoadImage.load_image(), which resolves it again. The
second resolution sees a plain absolute path with no annotation suffix, so
it defaults to checking containment against the input directory instead of
the temp directory the file actually lives in, and the path-traversal guard
in folder_paths.get_annotated_filepath() rejects it.

nodes.LoadImage.load_image() pulls in torch/PIL/av for the actual image
decode, which is irrelevant to this bug. It is stubbed here with a class
that reproduces only the one line that matters: resolving the ``image``
argument through folder_paths.get_annotated_filepath().
"""
import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

import folder_paths
from comfy.options import enable_args_parsing

enable_args_parsing()


class _FakeLoadImage:
    def load_image(self, image):
        return folder_paths.get_annotated_filepath(image)


mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_nodes.LoadImage = _FakeLoadImage


@pytest.fixture
def webcam_capture_cls():
    with patch.dict(sys.modules, {"nodes": mock_nodes}):
        sys.modules.pop("comfy_extras.nodes_webcam", None)
        module = importlib.import_module("comfy_extras.nodes_webcam")
    try:
        yield module.WebcamCapture
    finally:
        sys.modules.pop("comfy_extras.nodes_webcam", None)


@pytest.fixture
def sandbox(tmp_path):
    base = os.path.realpath(str(tmp_path))
    input_dir = os.path.join(base, "input")
    temp_dir = os.path.join(base, "temp")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    orig_input = folder_paths.get_input_directory()
    orig_temp = folder_paths.get_temp_directory()

    folder_paths.set_input_directory(input_dir)
    folder_paths.set_temp_directory(temp_dir)

    yield {"input": input_dir, "temp": temp_dir}

    folder_paths.set_input_directory(orig_input)
    folder_paths.set_temp_directory(orig_temp)


def test_load_capture_resolves_temp_annotated_image(sandbox, webcam_capture_cls):
    result = webcam_capture_cls().load_capture(image="webcam/capture.png [temp]")
    assert result == os.path.join(sandbox["temp"], "webcam", "capture.png")
