import inspect
import sys
from unittest.mock import MagicMock

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

_stubs = {"nodes": MagicMock(MAX_RESOLUTION=16384), "server": MagicMock()}
_saved = {name: sys.modules.get(name) for name in _stubs}
sys.modules.update(_stubs)
try:
    import comfy_extras.nodes_wan as nodes_wan
    from comfy_api.latest import io
finally:
    for _name, _original in _saved.items():
        if _original is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _original


NODE_CLASSES = [
    obj for obj in vars(nodes_wan).values()
    if inspect.isclass(obj) and issubclass(obj, io.ComfyNode) and obj is not io.ComfyNode and obj.__module__ == nodes_wan.__name__
]


@pytest.mark.parametrize("node_cls", NODE_CLASSES, ids=lambda c: c.__name__)
def test_optional_inputs_have_execute_defaults(node_cls):
    # The executor leaves unconnected optional inputs out of the call entirely.
    params = inspect.signature(node_cls.execute).parameters
    for i in node_cls.define_schema().inputs:
        if i.optional and i.id in params:
            assert params[i.id].default is not inspect.Parameter.empty, f"{node_cls.__name__}.{i.id}"


def test_phantom_subject_to_video_without_images():
    positive = [[torch.zeros(1, 4, 8), {}]]
    negative = [[torch.zeros(1, 4, 8), {}]]

    out = nodes_wan.WanPhantomSubjectToVideo.execute(positive, negative, MagicMock(), 64, 32, 9, 1)

    pos, neg_text, neg_img_text, latent = out.result
    assert latent["samples"].shape == (1, 16, 3, 4, 8)
    assert pos is positive
    assert neg_text is negative
    assert neg_img_text is negative
