import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.model_patcher import ModelPatcher, get_key_weight


class QuantizedLikeLinear(torch.nn.Linear):
    """Lists a scale tensor in the state dict without exposing it as an attribute, like the quantized ops do."""

    def state_dict(self, *args, destination=None, prefix="", **kwargs):
        sd = super().state_dict(*args, destination=destination, prefix=prefix, **kwargs)
        sd[f"{prefix}weight_scale"] = torch.ones(1)
        return sd


def _model():
    return torch.nn.Sequential(QuantizedLikeLinear(2, 2), torch.nn.Linear(2, 2))


def test_get_key_weight_returns_none_for_state_dict_only_keys():
    model = _model()

    assert get_key_weight(model, "0.weight")[0] is model[0].weight
    assert get_key_weight(model, "0.weight_scale") == (None, None, None)


def test_get_key_patches_includes_state_dict_only_keys():
    model = _model()
    patcher = ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))

    patches = patcher.get_key_patches()

    assert patches["0.weight"][0][0] is model[0].weight
    assert patches["1.weight"][0][0] is model[1].weight
    assert patches["0.weight_scale"][0][0] is None
