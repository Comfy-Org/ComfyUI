import importlib

import torch


def _tiny_config(qwen35):
    return qwen35.Qwen35Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        conv_kernel_size=2,
        layer_types=["linear_attention"],
        final_norm=False,
    )


def _module():
    operations = importlib.import_module("comfy.ops")
    qwen35 = importlib.import_module("comfy.text_encoders.qwen35")
    model_management = importlib.import_module("comfy.model_management")
    module = qwen35.GatedDeltaNet(
        _tiny_config(qwen35),
        device=torch.device("cpu"),
        dtype=torch.float32,
        ops=operations.manual_cast,
    )
    for index, parameter in enumerate(module.parameters()):
        torch.nn.init.constant_(parameter, (index + 1) * 0.001)
    return module.eval(), model_management


def test_gated_delta_net_casts_unmanaged_parameters_for_forward(monkeypatch):
    args = importlib.import_module("comfy.cli_args").args
    if not torch.cuda.is_available():
        monkeypatch.setattr(args, "cpu", True)
    module, model_management = _module()
    a_log_id = id(module.A_log)
    dt_bias_id = id(module.dt_bias)
    state_before = {
        name: value.detach().clone() for name, value in module.state_dict().items()
    }
    calls = []
    original_cast = model_management.cast_to_device

    def capture_cast(tensor, device, dtype, copy=False):
        if tensor is module.A_log or tensor is module.dt_bias:
            calls.append((id(tensor), torch.device(device), dtype, copy))
        return original_cast(tensor, device, dtype, copy=copy)

    monkeypatch.setattr(model_management, "cast_to_device", capture_cast)
    execution_device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    x = torch.linspace(
        -0.25,
        0.25,
        steps=24,
        dtype=torch.float32,
        device=execution_device,
    ).reshape(1, 3, 8)

    output, present_state = module(x)

    assert present_state is None
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert output.device == x.device
    assert torch.isfinite(output).all()
    assert calls == [
        (a_log_id, x.device, torch.float32, False),
        (dt_bias_id, x.device, torch.float32, False),
    ]
    assert id(module.A_log) == a_log_id
    assert id(module.dt_bias) == dt_bias_id
    assert module.A_log.device.type == "cpu"
    assert module.dt_bias.device.type == "cpu"
    state_after = module.state_dict()
    assert tuple(state_after) == tuple(state_before)
    for name, value in state_before.items():
        assert torch.equal(state_after[name], value)
