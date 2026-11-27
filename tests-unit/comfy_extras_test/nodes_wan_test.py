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
    import comfy.latent_formats
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


def test_phantom_subject_to_video_with_images():
    positive = [[torch.zeros(1, 4, 8), {}]]
    negative = [[torch.zeros(1, 4, 8), {}]]
    images = torch.stack([torch.full((32, 64, 3), 0.25), torch.full((32, 64, 3), 0.75)])
    vae = MagicMock()
    vae.encode.side_effect = lambda px: torch.full((1, 16, 1, 4, 8), px.mean().item())

    out = nodes_wan.WanPhantomSubjectToVideo.execute(positive, negative, vae, 64, 32, 9, 1, images=images)

    pos, neg_text, neg_img_text, _ = out.result
    encoded = [c.args[0] for c in vae.encode.call_args_list]
    assert len(encoded) == 2
    for px, value in zip(encoded, (0.25, 0.75)):
        assert px.shape == (1, 32, 64, 3)
        assert torch.allclose(px, torch.full_like(px, value))

    concat = pos[0][1]["time_dim_concat"]
    assert concat.shape == (1, 16, 2, 4, 8)
    assert torch.allclose(concat[:, :, 0], torch.full((1, 16, 4, 8), 0.25))
    assert torch.allclose(concat[:, :, 1], torch.full((1, 16, 4, 8), 0.75))
    assert torch.equal(neg_text[0][1]["time_dim_concat"], concat)
    expected_neg = comfy.latent_formats.Wan21().process_out(torch.zeros_like(concat))
    assert torch.equal(neg_img_text[0][1]["time_dim_concat"], expected_neg)
