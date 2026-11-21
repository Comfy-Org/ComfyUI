from unittest import mock

import torch

import comfy.model_patcher as model_patcher
import comfy.ops as ops


class SingleLayerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = ops.disable_weight_init.Linear(16, 16, bias=True, dtype=torch.float32)
        torch.nn.init.constant_(self.layer.weight, 1.0)
        torch.nn.init.constant_(self.layer.bias, 0.5)


class ThreeLayerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = ops.disable_weight_init.Linear(50, 50, bias=False, dtype=torch.float32)
        self.l2 = ops.disable_weight_init.Linear(30, 25, bias=False, dtype=torch.float32)
        self.l3 = ops.disable_weight_init.Linear(50, 50, bias=False, dtype=torch.float32)
        torch.nn.init.constant_(self.l1.weight, 1.0)
        torch.nn.init.constant_(self.l2.weight, 1.0)
        torch.nn.init.constant_(self.l3.weight, 1.0)


class MovableWrapper:
    def __init__(self, scale):
        self.scale = scale
        self.device = None

    def __call__(self, weight):
        return weight * self.scale

    def move_to(self, device):
        self.device = device
        return 0


def test_unpatch_full_load_restores_cast_flags_and_weight_wrappers():
    model = SingleLayerModel()
    cpu = torch.device("cpu")
    patcher = model_patcher.ModelPatcher(model, load_device=cpu, offload_device=cpu)
    wrapper = MovableWrapper(10.0)
    patcher.set_model_compute_dtype(torch.float32)
    patcher.add_weight_wrapper("layer.weight", wrapper)
    patcher.add_weight_wrapper("layer.bias", wrapper)

    patcher.patch_model(device_to=cpu, lowvram_model_memory=0, load_weights=True)
    x = torch.ones(1, 16)
    assert torch.allclose(model.layer(x), torch.full((1, 16), 165.0))
    assert model.layer.comfy_cast_weights is True
    assert model.layer.comfy_force_cast_weights is True
    assert wrapper.device == cpu

    wrapper.device = None
    patcher.unpatch_model(device_to=cpu, unpatch_weights=True)
    assert not hasattr(model.layer, "prev_comfy_cast_weights")
    assert not hasattr(model.layer, "comfy_force_cast_weights")
    assert model.layer.comfy_cast_weights is False
    assert model.layer.weight_function == []
    assert model.layer.bias_function == []
    assert wrapper.device == cpu
    assert torch.allclose(model.layer(x), torch.full((1, 16), 16.5))


def test_lowvram_load_applies_lora_after_full_load_with_compute_dtype():
    model = SingleLayerModel()
    cpu = torch.device("cpu")
    patcher1 = model_patcher.ModelPatcher(model, load_device=cpu, offload_device=cpu)
    patcher1.set_model_compute_dtype(torch.float32)
    patcher1.patch_model(device_to=cpu, lowvram_model_memory=0, load_weights=True)
    patcher1.unpatch_model(device_to=cpu, unpatch_weights=True)

    patcher2 = model_patcher.ModelPatcher(model, load_device=cpu, offload_device=cpu)
    patcher2.add_patches(
        {
            "layer.weight": (torch.ones_like(model.layer.weight),),
            "layer.bias": (torch.ones_like(model.layer.bias),),
        },
        strength_patch=1.0,
        strength_model=1.0,
    )
    patcher2.load(device_to=cpu, lowvram_model_memory=1, full_load=False)

    x = torch.ones(1, 16)
    assert model.model_lowvram is True
    assert patcher2.lowvram_patch_counter() == 2
    assert torch.allclose(model.layer(x), torch.full((1, 16), 33.5))


def test_partially_unload_preserves_prev_cast_weights_and_wrapper_order():
    model = SingleLayerModel()
    cpu = torch.device("cpu")
    patcher = model_patcher.ModelPatcher(model, load_device=cpu, offload_device=cpu)
    patcher.set_model_compute_dtype(torch.float32)
    patcher.add_patches(
        {
            "layer.weight": (torch.ones_like(model.layer.weight),),
            "layer.bias": (torch.ones_like(model.layer.bias),),
        },
        strength_patch=1.0,
        strength_model=1.0,
    )
    patcher.add_weight_wrapper("layer.weight", lambda w: w * 2.0)
    patcher.add_weight_wrapper("layer.bias", lambda b: b * 2.0)

    patcher.load(device_to=cpu, full_load=True)
    x = torch.ones(1, 16)
    expected = model.layer(x)
    assert torch.allclose(expected, torch.full((1, 16), 67.0))
    assert model.layer.prev_comfy_cast_weights is False

    patcher.partially_unload(cpu, memory_to_free=10000)
    assert model.layer.prev_comfy_cast_weights is False
    assert torch.allclose(model.layer(x), expected)

    patcher.unpatch_model(device_to=cpu, unpatch_weights=True)
    assert not hasattr(model.layer, "prev_comfy_cast_weights")
    assert model.layer.comfy_cast_weights is False


def test_incremental_partially_load_does_not_double_apply_lora_or_inflate_counter():
    model = ThreeLayerModel()
    cpu = torch.device("cpu")
    patcher = model_patcher.ModelPatcher(model, load_device=cpu, offload_device=cpu)
    patcher.add_patches(
        {
            "l1.weight": (torch.ones_like(model.l1.weight),),
            "l2.weight": (torch.ones_like(model.l2.weight),),
        },
        strength_patch=1.0,
        strength_model=1.0,
    )

    x_l1 = torch.ones(1, 50)
    x_l2 = torch.ones(1, 30)

    patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    assert patcher.lowvram_patch_counter() == 2
    assert torch.allclose(model.l1(x_l1), torch.full((1, 50), 100.0))
    assert torch.allclose(model.l2(x_l2), torch.full((1, 25), 60.0))

    model2 = ThreeLayerModel()
    model2.l2.pre_quant_scale = torch.nn.Parameter(torch.full((30,), 2.0))
    patcher2 = model_patcher.ModelPatcher(model2, load_device=cpu, offload_device=cpu)
    patcher2.add_patches(
        {
            "l2.weight": (torch.ones_like(model2.l2.weight),),
            "l2.pre_quant_scale": (torch.ones_like(model2.l2.pre_quant_scale),),
        },
        strength_patch=1.0,
        strength_model=1.0,
    )

    patcher2.partially_load(cpu, extra_memory=14000)
    assert model2.l2.comfy_patched_weights is True
    assert patcher2.lowvram_patch_counter() == 0
    assert torch.allclose(model2.l2.pre_quant_scale, torch.full((30,), 3.0))
    assert torch.allclose(model2.l2(x_l2), torch.full((1, 25), 60.0))

    with mock.patch.object(model2.l2, "to", wraps=model2.l2.to) as l2_to:
        patcher2.partially_load(cpu, extra_memory=18000)
    l2_to.assert_called_once_with(cpu)
    assert model2.l2.comfy_patched_weights is False
    assert patcher2.lowvram_patch_counter() == 1
    assert torch.allclose(model2.l2.pre_quant_scale, torch.full((30,), 3.0))
    assert torch.allclose(model2.l2(x_l2), torch.full((1, 25), 60.0))

    patcher2.partially_load(cpu, extra_memory=20000)
    assert model2.model_lowvram is False
    assert patcher2.lowvram_patch_counter() == 0
    assert torch.allclose(model2.l2.pre_quant_scale, torch.full((30,), 3.0))
    assert torch.allclose(model2.l2(x_l2), torch.full((1, 25), 60.0))
