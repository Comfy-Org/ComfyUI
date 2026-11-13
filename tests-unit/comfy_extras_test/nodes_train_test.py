import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_extras.nodes_train import TrainLoraNode, _create_weight_adapter


@pytest.mark.parametrize("alpha", [1.0, 16.0])
def test_new_lora_adapter_uses_alpha(alpha):
    module = torch.nn.Linear(8, 4)
    adapter, params = _create_weight_adapter(module, "layer", {}, "LoRA", torch.float32, 4, alpha)
    assert float(adapter.alpha.detach()) == alpha
    assert float(params["layer.alpha"].detach()) == alpha


def test_existing_lora_keeps_saved_alpha():
    module = torch.nn.Linear(8, 4)
    existing = {
        "layer.lora_up.weight": torch.zeros(4, 2),
        "layer.lora_down.weight": torch.zeros(2, 8),
        "layer.alpha": torch.tensor(2.0),
    }
    adapter, _ = _create_weight_adapter(module, "layer", existing, "LoRA", torch.float32, 2, 16.0)
    assert float(adapter.alpha.detach()) == 2.0


def test_alpha_input_defaults_to_previous_behavior():
    schema = TrainLoraNode.define_schema()
    alpha = next(i for i in schema.inputs if i.id == "alpha")
    assert alpha.default == 1.0
    assert alpha.optional


def test_saved_alpha_round_trips_on_resume():
    module = torch.nn.Linear(8, 4)
    _, params = _create_weight_adapter(module, "layer", {}, "LoRA", torch.float32, 2, 16.0)
    saved = {k: v.detach() for k, v in params.items()}
    adapter, _ = _create_weight_adapter(module, "layer", saved, "LoRA", torch.float32, 2, 1.0)
    assert float(adapter.alpha.detach()) == 16.0
