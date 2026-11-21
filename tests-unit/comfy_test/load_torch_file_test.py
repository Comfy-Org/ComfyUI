import pytest
import torch

import comfy.utils


class _CustomObject:
    """Not on the weights_only allowlist: loading it requires full unpickling."""

    marker = "payload"


@pytest.fixture()
def tensor_ckpt(tmp_path):
    path = tmp_path / "tensors.pt"
    torch.save({"tensor": torch.zeros(2)}, path)
    return str(path)


@pytest.fixture()
def pickled_ckpt(tmp_path):
    path = tmp_path / "pickled.pt"
    torch.save({"obj": _CustomObject()}, path)
    return str(path)


@pytest.mark.parametrize("safe_load", [True, False])
def test_plain_tensor_dict_loads_in_every_mode(tensor_ckpt, safe_load):
    sd = comfy.utils.load_torch_file(tensor_ckpt, safe_load=safe_load)
    assert "tensor" in sd


def test_plain_tensor_dict_loads_with_defaults(tensor_ckpt):
    sd = comfy.utils.load_torch_file(tensor_ckpt)
    assert "tensor" in sd


@pytest.mark.parametrize("kwargs", [{}, {"safe_load": True}])
def test_safe_modes_reject_pickled_objects(pickled_ckpt, kwargs):
    with pytest.raises(Exception):
        comfy.utils.load_torch_file(pickled_ckpt, **kwargs)


def test_safe_load_false_loads_pickled_objects(pickled_ckpt):
    # safe_load is the weights_only toggle: opting out must allow full
    # unpickling, as it did before the parameter was left unwired.
    sd = comfy.utils.load_torch_file(pickled_ckpt, safe_load=False)
    assert sd["obj"].marker == "payload"
