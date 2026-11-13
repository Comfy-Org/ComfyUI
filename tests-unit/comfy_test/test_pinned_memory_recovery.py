import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from comfy.cli_args import args

args.cpu = True
import comfy.pinned_memory as pm

free_registrations = pm.comfy.model_management.free_registrations


class Buffer:

    def __init__(self):
        self.size = 64
        self.views = weakref.WeakSet()

    def extend(self, size, register):
        self.size += size

    def truncate(self, offset, do_unregister):
        if offset >= self.size:
            raise RuntimeError('HostBuffer.truncate failed')
        self.size = offset


class View:

    def __init__(self, buf, offset=0, size=0):
        self.buf, self.offset, self.nbytes = buf, offset, size
        buf.views.add(self)

    def __getitem__(self, key):
        return View(self.buf, key.start, key.stop - key.start)

    def untyped_storage(self):
        return self

    def data_ptr(self):
        assert self.offset + self.nbytes <= self.buf.size, 'registration uses evicted memory'
        return self.offset


@pytest.fixture
def state(monkeypatch):
    buf = Buffer()
    module = SimpleNamespace(_pin_state={'weights': (buf, [], [0], [0], [0], {})})
    mm = pm.comfy.model_management
    monkeypatch.setattr(mm, 'DISABLE_PINNED_MEMORY', False)
    monkeypatch.setattr(mm, 'TOTAL_PINNED_MEMORY', 0)
    monkeypatch.setattr(mm, 'ensure_pin_budget', lambda *a, **k: True)
    monkeypatch.setattr(mm, 'ensure_pin_registerable', lambda *a, **k: True)
    monkeypatch.setattr(mm, 'discard_cuda_async_error', lambda: None)
    monkeypatch.setattr(mm, 'free_registrations', lambda *a, **k: None)
    monkeypatch.setattr(pm.comfy.memory_management, 'extra_ram_release', lambda *a: None)
    monkeypatch.setattr(pm.comfy_aimdo.torch, 'hostbuf_to_tensor', View)
    return buf, module, mm


def test_registration_retry_reallocates_after_eviction(state, monkeypatch):
    buf, module, mm = state
    register = Mock(side_effect=[1, 0])
    monkeypatch.setattr(pm.torch.cuda, 'cudart', lambda: SimpleNamespace(cudaHostRegister=register))

    def evict(*a, **k):
        # Only the committed 64-byte buffer may be evicted; the temporary view
        # must have been discarded before changing the shared allocation.
        assert buf.size == 64
        assert not buf.views
        buf.truncate(0, do_unregister=True)
    monkeypatch.setattr(mm, 'free_registrations', evict)
    assert pm.pin_memory(module, size=32)
    assert module._pins['weights']['pin'].offset == 0
    assert buf.size == 32
    assert mm.TOTAL_PINNED_MEMORY == 32


def test_second_registration_failure_returns_fallback_without_leaking(state, monkeypatch):
    buf, module, mm = state
    monkeypatch.setattr(pm.torch.cuda, 'cudart', lambda: SimpleNamespace(cudaHostRegister=lambda *a: 1))
    monkeypatch.setattr(mm, 'free_registrations', lambda *a, **k: buf.truncate(0, True))
    assert not pm.pin_memory(module, size=32)
    assert buf.size == 0
    assert mm.TOTAL_PINNED_MEMORY == 0
    assert 'pin' not in module._pins['weights']


def test_first_registration_succeeds(state, monkeypatch):
    buf, module, mm = state
    monkeypatch.setattr(pm.torch.cuda, 'cudart', lambda: SimpleNamespace(cudaHostRegister=lambda *a: 0))
    assert pm.pin_memory(module, size=32)
    assert buf.size == 96
    assert mm.TOTAL_PINNED_MEMORY == 32


@pytest.mark.parametrize('partial_growth', [False, True])
def test_allocation_failure_rolls_back_only_existing_extension(state, monkeypatch, partial_growth):
    buf, module, mm = state

    def fail(size, register):
        if partial_growth:
            buf.size += size
        raise RuntimeError('allocation failed')
    monkeypatch.setattr(buf, 'extend', fail)
    assert not pm.pin_memory(module, size=32)
    assert buf.size == 64
    assert mm.TOTAL_PINNED_MEMORY == 0


def test_rollback_error_is_not_swallowed_or_retried(state, monkeypatch):
    buf, module, mm = state
    monkeypatch.setattr(pm.torch.cuda, 'cudart', lambda: SimpleNamespace(cudaHostRegister=lambda *a: 1))
    truncate = Mock(side_effect=RuntimeError('rollback failed'))
    monkeypatch.setattr(buf, 'truncate', truncate)
    with pytest.raises(RuntimeError, match='rollback failed'):
        pm.pin_memory(module, size=32)
    truncate.assert_called_once()
    assert mm.TOTAL_PINNED_MEMORY == 0


def test_registration_retry_without_eviction_uses_same_offset(state, monkeypatch):
    buf, module, mm = state
    register = Mock(side_effect=[1, 0])
    monkeypatch.setattr(pm.torch.cuda, 'cudart', lambda: SimpleNamespace(cudaHostRegister=register))
    assert pm.pin_memory(module, size=32)
    assert [c.args[0] for c in register.call_args_list] == [64, 64]
    assert buf.size == 96
    assert module._pin_state['weights'][1] == [(module, 64)]
    assert module._pin_state['weights'][3] == [32]
    assert mm.TOTAL_PINNED_MEMORY == 32


@pytest.mark.parametrize('partial_growth', [False, True])
def test_retry_allocation_failure_rolls_back(state, monkeypatch, partial_growth):
    buf, module, mm = state
    extend = buf.extend
    attempts = []

    def allocate(size, register):
        attempts.append(buf.size)
        if len(attempts) == 1:
            return extend(size, register)
        if partial_growth:
            extend(size, register)
        raise RuntimeError('retry allocation failed')
    monkeypatch.setattr(buf, 'extend', allocate)
    monkeypatch.setattr(pm.torch.cuda, 'cudart', lambda: SimpleNamespace(cudaHostRegister=lambda *a: 1))
    monkeypatch.setattr(mm, 'free_registrations', lambda *a, **k: buf.truncate(0, True))
    assert not pm.pin_memory(module, size=32)
    assert attempts == [64, 0]
    assert buf.size == 0
    assert not buf.views
    assert mm.TOTAL_PINNED_MEMORY == 0


def test_registration_retry_after_fast_pin_eviction_policy(state, monkeypatch):
    buf, module, mm = state
    subset = 'weights-fast'
    pin_state = module._pin_state
    pin_state[subset] = pin_state.pop('weights')
    pin_state.update(active=False, current_prompt=False)
    _, stack, _, pinned_size, _, buckets = pin_state[subset]
    victim = SimpleNamespace(_pin_state=pin_state, _pins={subset: {}})
    victim_pin = victim._pins[subset]
    victim_pin.update(pin=View(buf, 0, 64), registered=True, stack_index=0)
    stack.append((victim, 0))
    pinned_size[0] = 64
    pm._add_to_bucket(victim, victim_pin, buckets, 64, 0)
    patcher = SimpleNamespace(model=SimpleNamespace(dynamic_pins={'cpu': pin_state}), load_device='cpu', is_dynamic=lambda: True)

    def evict(size, subsets):
        assert subset in subsets
        stack.pop()
        victim_pin['balancer_entry'][-1] = None
        del victim_pin['pin']
        buf.truncate(0, do_unregister=True)
        pinned_size[0] -= 64
        mm.TOTAL_PINNED_MEMORY -= 64
        return 64
    patcher.partially_unload_ram = evict
    monkeypatch.setattr(mm, 'current_loaded_models', [SimpleNamespace(model=patcher)])
    monkeypatch.setattr(mm, 'MAX_PINNED_MEMORY', 1024)
    monkeypatch.setattr(mm, 'REGISTERABLE_PIN_HYSTERESIS', 0)
    monkeypatch.setattr(mm, 'TOTAL_PINNED_MEMORY', 64)
    monkeypatch.setattr(mm, 'free_registrations', free_registrations)
    register = Mock(side_effect=[1, 0])
    monkeypatch.setattr(pm.torch.cuda, 'cudart', lambda: SimpleNamespace(cudaHostRegister=register))
    assert pm.pin_memory(module, subset=subset, size=32)
    assert [c.args[0] for c in register.call_args_list] == [64, 0]
    assert 'pin' not in victim_pin
    assert stack == [(module, 0)]
    assert buf.size == 32
    assert pinned_size == [32]
    assert mm.TOTAL_PINNED_MEMORY == 32
