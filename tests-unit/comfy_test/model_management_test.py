from unittest import mock

import pytest
import torch

import comfy.model_management as model_management


class Tensor:
    device = torch.device("cpu")
    nbytes = 4096

    def is_pinned(self):
        return False

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return 0x13730


class CudaRuntime:
    def __init__(self):
        self.register_calls = []

    def cudaHostRegister(self, ptr, size, flags):
        self.register_calls.append((ptr, size, flags))
        return 0


def run_pin(budget, registerable):
    runtime = CudaRuntime()
    tensor = Tensor()
    with (
        mock.patch.object(model_management, "MAX_PINNED_MEMORY", 8192),
        mock.patch.object(model_management, "PINNED_MEMORY", {}),
        mock.patch.object(model_management, "TOTAL_PINNED_MEMORY", 0),
        mock.patch.object(model_management, "ensure_pin_budget", return_value=budget) as budget_check,
        mock.patch.object(model_management, "ensure_pin_registerable", return_value=registerable) as registerable_check,
        mock.patch.object(model_management.comfy.memory_management, "extra_ram_release"),
        mock.patch.object(torch.cuda, "cudart", return_value=runtime),
    ):
        result = model_management.pin_memory(tensor)
        pinned = dict(model_management.PINNED_MEMORY)
        total = model_management.TOTAL_PINNED_MEMORY
    return result, runtime.register_calls, pinned, total, budget_check, registerable_check


@pytest.mark.parametrize(
    ("budget", "registerable", "expected_result", "expected_calls"),
    [
        (False, True, False, []),
        (True, False, False, []),
        (True, True, True, [(0x13730, Tensor.nbytes, 1)]),
    ],
)
def test_static_pin_respects_budgets(budget, registerable, expected_result, expected_calls):
    result, calls, pinned, total, budget_check, registerable_check = run_pin(budget, registerable)

    assert result is expected_result
    assert calls == expected_calls
    assert pinned == ({0x13730: Tensor.nbytes} if expected_result else {})
    assert total == (Tensor.nbytes if expected_result else 0)
    budget_check.assert_called_once_with(Tensor.nbytes)
    if budget:
        registerable_check.assert_called_once_with(Tensor.nbytes)
    else:
        registerable_check.assert_not_called()


def test_high_ram_only_bypasses_ram_pressure_budget():
    with (
        mock.patch.object(model_management.args, "high_ram", True),
        mock.patch.object(model_management, "free_pins") as free_pins,
    ):
        assert model_management.ensure_pin_budget(Tensor.nbytes) is True
        free_pins.assert_not_called()

    result, calls, pinned, total, _, registerable_check = run_pin(True, False)
    assert result is False
    assert calls == []
    assert pinned == {}
    assert total == 0
    registerable_check.assert_called_once_with(Tensor.nbytes)
