import ctypes
import importlib.util
import sys
import types
import weakref
from pathlib import Path

import pytest

from comfy.pin_order import PrefetchPinOrder, prefetch_budget_checked, prefetch_pin_state
from comfy import windows_dxgi
from comfy.windows_dxgi import DXGINonLocalBudgetProvider, DXGIUnavailable, LivePinBudget, VideoMemoryInfo, create_dxgi_budget_provider, safe_headroom


GiB = 1024 ** 3


class SyntheticProvider:
    def __init__(self, budget, usage):
        self.budget = budget
        self.usage = usage

    def query(self):
        return VideoMemoryInfo(self.budget, self.usage)


class FakeAdapter:
    def __init__(self, luid):
        self.luid = luid
        self.closed = False
        self.queried_node = None

    def query(self, node_index=0):
        self.queried_node = node_index
        return VideoMemoryInfo(16 * GiB, 8 * GiB)

    def close(self):
        self.closed = True


class FakeBlock:
    def __init__(self, index, pin_state=None):
        self.index = index
        self._v = object()
        self._pin_state = {"prefetch_orders": weakref.WeakSet()} if pin_state is None else pin_state

    def modules(self):
        return [self]


def test_dxgi_headroom_reserves_space_below_budget():
    info = VideoMemoryInfo(16 * GiB, 8 * GiB)
    assert safe_headroom(info, 2 * GiB) == 6 * GiB
    assert safe_headroom(VideoMemoryInfo(16 * GiB, 15 * GiB), 2 * GiB) == 0


def test_dxgi_ctypes_layouts_match_windows_abi():
    assert ctypes.sizeof(windows_dxgi._GUID) == 16
    assert ctypes.sizeof(windows_dxgi._LUID) == 8
    assert ctypes.sizeof(windows_dxgi._DXGI_QUERY_VIDEO_MEMORY_INFO) == 32
    if sys.platform == "win32":
        assert ctypes.sizeof(windows_dxgi._DXGI_ADAPTER_DESC1) == 312


def test_live_budget_evicts_with_hysteresis_and_rechecks():
    provider = SyntheticProvider(16 * GiB, 13 * GiB)
    budget = LivePinBudget(provider, reserve=2 * GiB, hysteresis=GiB // 2)
    evictions = []

    def evict(size):
        evictions.append(size)
        provider.usage -= size
        return size

    assert budget.ensure(2 * GiB, evict)
    assert evictions == [GiB + GiB // 2]
    assert budget.last_headroom == 2 * GiB + GiB // 2
    assert budget.query_count == 2


def test_live_budget_tracks_query_cost(monkeypatch):
    timestamps = iter((100, 150, 200, 275))
    monkeypatch.setattr(windows_dxgi.time, "perf_counter_ns", lambda: next(timestamps))
    budget = LivePinBudget(SyntheticProvider(16 * GiB, 8 * GiB), reserve=2 * GiB, hysteresis=GiB // 2)

    assert budget.ensure(GiB, lambda size: 0)
    assert budget.ensure(GiB, lambda size: 0)
    assert budget.query_count == 2
    assert budget.query_ns == 125
    assert budget.max_query_ns == 75


def test_budget_increase_does_not_evict_or_churn():
    provider = SyntheticProvider(12 * GiB, 8 * GiB)
    budget = LivePinBudget(provider, reserve=2 * GiB, hysteresis=GiB // 2)
    evictions = []
    assert budget.ensure(GiB, lambda size: evictions.append(size))
    provider.budget = 16 * GiB
    assert budget.ensure(4 * GiB, lambda size: evictions.append(size))
    assert evictions == []


def test_exact_cuda_luid_selects_the_matching_dxgi_adapter():
    first = FakeAdapter(b"first000")
    match = FakeAdapter(b"match000")
    provider = DXGINonLocalBudgetProvider(2, lambda index: b"match000", lambda: [first, match])
    assert provider.query().budget == 16 * GiB
    assert first.closed
    assert not match.closed
    provider.close()
    assert match.closed


def test_cuda_node_mask_mapping_reaches_dxgi_query_node():
    adapter = FakeAdapter(b"match000")
    provider = DXGINonLocalBudgetProvider(0, lambda index: (b"match000", 3), lambda: [adapter])
    provider.query()
    assert adapter.queried_node == 3
    provider.close()


def test_ambiguous_cuda_luid_match_fails_conservative():
    adapters = [FakeAdapter(b"same0000"), FakeAdapter(b"same0000")]
    with pytest.raises(DXGIUnavailable, match="matched 2"):
        DXGINonLocalBudgetProvider(0, lambda index: b"same0000", lambda: adapters)
    assert all(adapter.closed for adapter in adapters)


def test_missing_cuda_luid_match_fails_conservative():
    adapter = FakeAdapter(b"other000")
    with pytest.raises(DXGIUnavailable, match="matched 0"):
        DXGINonLocalBudgetProvider(0, lambda index: b"cuda0000", lambda: [adapter])
    assert adapter.closed


def test_non_windows_does_not_create_dxgi_policy(monkeypatch):
    monkeypatch.setattr("comfy.windows_dxgi.platform.system", lambda: "Linux")
    assert create_dxgi_budget_provider(0) is None


def test_prefetch_order_prefers_upcoming_over_consumed_blocks():
    blocks = [FakeBlock(index) for index in range(8)]
    order = PrefetchPinOrder(blocks, window=3)
    for _ in range(4):
        order.advance()
    assert order.state(blocks[3])[0]
    assert order.state(blocks[4])[0]
    assert order.state(blocks[5])[0]
    assert not order.state(blocks[2])[0]
    assert order.state(blocks[2])[1] > order.state(blocks[7])[1]
    order.close()


def test_overlapping_prefetch_orders_keep_independent_state():
    pin_state = {"prefetch_orders": weakref.WeakSet()}
    shared = FakeBlock(0, pin_state)
    first_only = FakeBlock(1, pin_state)
    second_only = FakeBlock(2, pin_state)
    first = PrefetchPinOrder([shared, first_only])
    second = PrefetchPinOrder([second_only, shared])

    first.advance()
    second.advance()
    assert prefetch_pin_state(shared) == (True, 0)
    first.advance()
    assert prefetch_pin_state(shared) == (True, 1)

    first.budget_checked = True
    assert not prefetch_budget_checked(shared)
    second.budget_checked = True
    assert prefetch_budget_checked(shared)

    first.close()
    assert prefetch_pin_state(shared) == (True, 1)
    second.close()
    assert prefetch_pin_state(shared) is None
    assert not prefetch_budget_checked(shared)
    assert not hasattr(shared, "_pin_prefetch_order")


def test_fifty_sequential_blocks_cycle_through_constrained_live_budget(monkeypatch):
    pinned, _, _ = _load_pinned_memory(monkeypatch)
    blocks = [FakeBlock(index) for index in range(50)]
    order = PrefetchPinOrder(blocks, window=3)
    provider = SyntheticProvider(16 * GiB, 8 * GiB)
    budget = LivePinBudget(provider, reserve=2 * GiB, hysteresis=GiB)
    registered = set()
    pinned_before_transfer = []
    evicted = []
    peak_registered = 0

    def evict(size):
        candidates = sorted(registered, reverse=True, key=lambda block: pinned.pin_eviction_priority(order.state(block)))
        freed = 0
        for block in candidates:
            registered.remove(block)
            evicted.append(block.index)
            provider.usage -= GiB
            freed += GiB
            if freed >= size:
                break
        return freed

    for block in blocks:
        order.advance()
        if block not in registered:
            assert budget.ensure(GiB, evict)
            provider.usage += GiB
            registered.add(block)
        pinned_before_transfer.append(block in registered)
        peak_registered = max(peak_registered, len(registered))
        assert provider.usage + budget.reserve <= provider.budget

    assert all(pinned_before_transfer)
    assert peak_registered <= 6
    assert evicted
    assert len(registered) <= 6
    order.close()


def test_live_budget_smaller_than_preferred_band_evicts_farthest_first(monkeypatch):
    pinned, _, _ = _load_pinned_memory(monkeypatch)
    blocks = [FakeBlock(index) for index in range(3)]
    order = PrefetchPinOrder(blocks, window=3)
    order.advance()
    provider = SyntheticProvider(4 * GiB, 3 * GiB)
    budget = LivePinBudget(provider, reserve=2 * GiB, hysteresis=0)
    registered = set(blocks)
    evicted = []

    def evict(size):
        candidates = sorted(registered, reverse=True, key=lambda block: pinned.pin_eviction_priority(order.state(block)))
        freed = 0
        for block in candidates:
            registered.remove(block)
            evicted.append(block.index)
            provider.usage -= GiB
            freed += GiB
            if freed >= size:
                break
        return freed

    assert budget.ensure(0, evict)
    assert evicted == [2]
    assert blocks[0] in registered
    assert blocks[1] in registered
    order.close()


def _load_pinned_memory(monkeypatch):
    import comfy
    import comfy.pin_order

    model_management = types.ModuleType("comfy.model_management")
    model_management.TOTAL_PINNED_MEMORY = 0
    model_management.ensure_pin_registerable = lambda *args, **kwargs: True
    model_management.has_live_pin_budget = lambda device: True
    model_management.free_registrations = lambda *args, **kwargs: True
    model_management.discard_cuda_async_error = lambda: None

    memory_management = types.ModuleType("comfy.memory_management")
    memory_management.RAM_CACHE_HEADROOM = 0
    memory_management.extra_ram_release = lambda size: None
    memory_management.vram_aligned_size = lambda values: sum(value.nbytes for value in values if value is not None)

    utils = types.ModuleType("comfy.utils")
    utils.bit_reverse_range = lambda value, bits: value

    cli_args = types.ModuleType("comfy.cli_args")
    cli_args.args = types.SimpleNamespace(disable_pinned_memory=False)

    aimdo = types.ModuleType("comfy_aimdo")
    aimdo.__path__ = []
    aimdo_host_buffer = types.ModuleType("comfy_aimdo.host_buffer")
    aimdo_torch = types.ModuleType("comfy_aimdo.torch")

    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    monkeypatch.setitem(sys.modules, "comfy.memory_management", memory_management)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)
    monkeypatch.setitem(sys.modules, "comfy.cli_args", cli_args)
    monkeypatch.setitem(sys.modules, "comfy_aimdo", aimdo)
    monkeypatch.setitem(sys.modules, "comfy_aimdo.host_buffer", aimdo_host_buffer)
    monkeypatch.setitem(sys.modules, "comfy_aimdo.torch", aimdo_torch)
    monkeypatch.setattr(comfy, "model_management", model_management, raising=False)
    monkeypatch.setattr(comfy, "memory_management", memory_management, raising=False)
    monkeypatch.setattr(comfy, "utils", utils, raising=False)

    path = Path(__file__).parents[2] / "comfy" / "pinned_memory.py"
    spec = importlib.util.spec_from_file_location("test_pinned_memory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, model_management, cli_args.args


class FakePin:
    def __init__(self, size=1024, pointer=1234):
        self.nbytes = size
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer


class FakeModule:
    pass


def _retained_pin(registered):
    module = FakeModule()
    pin = FakePin()
    module_pin = {"pin": pin, "registered": registered, "stack_index": 0}
    module._pins = {"weights": module_pin}
    module._pin_state = {
        "device": types.SimpleNamespace(index=0),
        "prefetch_orders": weakref.WeakSet(),
        "weights": (None, [(module, 0)], [0], [pin.nbytes if registered else 0], [0], {}),
    }
    return module, pin, module_pin


def test_lowvram_source_copies_all_active_prefetch_orders(monkeypatch):
    pinned, _, _ = _load_pinned_memory(monkeypatch)
    pin_state = {"prefetch_orders": weakref.WeakSet()}
    source = FakeBlock(0, pin_state)
    target = FakeBlock(3, pin_state)
    first = PrefetchPinOrder([source, FakeBlock(1, pin_state)])
    second = PrefetchPinOrder([FakeBlock(2, pin_state), source])
    first.advance()
    second.advance()

    pinned.copy_prefetch_order(source, target)

    assert first.state(target) == (True, 0)
    assert second.state(target) == (True, 1)
    assert prefetch_pin_state(target) == (True, 0)
    first.close()
    assert prefetch_pin_state(target) == (True, 1)
    second.close()


def test_registered_pin_is_reconsidered_after_budget_shrink(monkeypatch):
    pinned, model_management, _ = _load_pinned_memory(monkeypatch)
    module, pin, module_pin = _retained_pin(True)
    model_management.TOTAL_PINNED_MEMORY = pin.nbytes
    model_management.ensure_pin_registerable = lambda *args, **kwargs: False
    monkeypatch.setattr(pinned, "_host_unregister", lambda value: 0)

    assert pinned.get_pin(module) is pin
    assert not module_pin["registered"]
    assert model_management.TOTAL_PINNED_MEMORY == 0
    assert module._pin_state["weights"][3][0] == 0


def test_requested_pin_survives_when_farther_preferred_pin_can_be_evicted(monkeypatch):
    pinned, model_management, _ = _load_pinned_memory(monkeypatch)
    requested, requested_pin, requested_state = _retained_pin(True)
    farther, farther_pin, farther_state = _retained_pin(True)
    for module in (requested, farther):
        module._v = object()
        module.modules = lambda module=module: [module]
    order = PrefetchPinOrder([requested, farther], window=3)
    order.advance()
    model_management.TOTAL_PINNED_MEMORY = requested_pin.nbytes + farther_pin.nbytes
    monkeypatch.setattr(pinned, "_host_unregister", lambda value: 0)

    def ensure(size, **kwargs):
        assert kwargs["protected"] == {(requested, "weights")}
        return pinned.unregister_pin(farther, "weights") == farther_pin.nbytes

    model_management.ensure_pin_registerable = ensure

    assert pinned.get_pin(requested) is requested_pin
    assert requested_state["registered"]
    assert not farther_state["registered"]
    assert model_management.TOTAL_PINNED_MEMORY == requested_pin.nbytes
    order.close()


def test_prefetch_boundary_check_avoids_per_module_budget_query(monkeypatch):
    pinned, model_management, _ = _load_pinned_memory(monkeypatch)
    module, pin, module_pin = _retained_pin(True)
    module._v = object()
    module.modules = lambda: [module]
    order = PrefetchPinOrder([module])
    order.advance()
    order.budget_checked = True
    checks = []
    model_management.ensure_pin_registerable = lambda *args, **kwargs: checks.append(True)

    assert pinned.get_pin(module) is pin
    assert module_pin["registered"]
    assert checks == []
    order.close()


def test_in_flight_pin_cannot_be_unregistered(monkeypatch):
    pinned, model_management, _ = _load_pinned_memory(monkeypatch)
    module, pin, module_pin = _retained_pin(True)
    module_pin["in_flight"] = types.SimpleNamespace(query=lambda: False)
    model_management.TOTAL_PINNED_MEMORY = pin.nbytes
    unregister_calls = []
    monkeypatch.setattr(pinned, "_host_unregister", lambda value: unregister_calls.append(value))

    assert pinned.unregister_pin(module, "weights") == 0
    assert module_pin["registered"]
    assert unregister_calls == []
    assert model_management.TOTAL_PINNED_MEMORY == pin.nbytes


def test_ram_pressure_cannot_unload_in_flight_pin(monkeypatch):
    pinned, _, _ = _load_pinned_memory(monkeypatch)

    class HostBuffer:
        def truncate(self, *args, **kwargs):
            raise AssertionError("in-flight host buffer was truncated")

    module = types.SimpleNamespace(_pins={"weights": {"registered": True}})
    stack = [(module, 0)]
    pin_state = {
        "weights": (HostBuffer(), stack, [-1], [1024], [0], {}),
    }
    monkeypatch.setattr(pinned, "pin_eviction_state", lambda *args: (True, None))

    assert pinned.partially_unload_ram(pin_state, 1024, subsets=["weights"]) == 0
    assert stack == [(module, 0)]


def test_unregister_failure_keeps_registration_and_ledger(monkeypatch):
    pinned, model_management, _ = _load_pinned_memory(monkeypatch)
    module, pin, module_pin = _retained_pin(True)
    model_management.TOTAL_PINNED_MEMORY = pin.nbytes
    monkeypatch.setattr(pinned, "_host_unregister", lambda value: 1)

    assert pinned.unregister_pin(module, "weights") == 0
    assert module_pin["registered"]
    assert model_management.TOTAL_PINNED_MEMORY == pin.nbytes
    assert module._pin_state["weights"][3][0] == pin.nbytes


def test_non_live_policy_keeps_registered_pin_without_reconsidering(monkeypatch):
    pinned, model_management, _ = _load_pinned_memory(monkeypatch)
    module, pin, module_pin = _retained_pin(True)
    model_management.has_live_pin_budget = lambda device: False
    checks = []
    model_management.ensure_pin_registerable = lambda *args, **kwargs: checks.append(True)

    assert pinned.get_pin(module) is pin
    assert module_pin["registered"]
    assert checks == []


def test_non_live_re_register_failure_keeps_original_single_attempt(monkeypatch):
    pinned, model_management, _ = _load_pinned_memory(monkeypatch)
    module, pin, module_pin = _retained_pin(False)
    model_management.has_live_pin_budget = lambda device: False
    register_calls = []
    monkeypatch.setattr(pinned, "_host_register", lambda value, size: register_calls.append((value, size)) or 1)

    assert pinned.get_pin(module) is pin
    assert not module_pin["registered"]
    assert len(register_calls) == 1
    assert model_management.TOTAL_PINNED_MEMORY == 0


def test_non_live_prefetch_order_steals_consumed_before_upcoming_pin(monkeypatch):
    pinned, model_management, _ = _load_pinned_memory(monkeypatch)
    model_management.has_live_pin_budget = lambda device: False
    blocks = [FakeBlock(index) for index in range(5)]
    order = PrefetchPinOrder(blocks, window=3)
    for _ in range(3):
        order.advance()

    consumed = blocks[1]
    current = blocks[2]
    upcoming = blocks[3]
    consumed_pin = FakePin(pointer=111)
    upcoming_pin = FakePin(pointer=333)
    consumed._pins = {"weights": {"pin": consumed_pin, "registered": True, "stack_index": 0}}
    upcoming._pins = {"weights": {"pin": upcoming_pin, "registered": True, "stack_index": 1}}
    current._pins = {"weights": {}}
    stack = [(consumed, 0), (upcoming, consumed_pin.nbytes)]
    buckets = {}
    pinned._add_to_bucket(consumed, consumed._pins["weights"], buckets, consumed_pin.nbytes, 100)
    pinned._add_to_bucket(upcoming, upcoming._pins["weights"], buckets, upcoming_pin.nbytes, 200)

    assert pinned._steal_pin(current, stack, buckets, consumed_pin.nbytes, 0, "weights")
    assert current._pins["weights"]["pin"] is consumed_pin
    assert upcoming._pins["weights"]["pin"] is upcoming_pin
    assert "pin" not in consumed._pins["weights"]
    order.close()


def test_severe_pressure_steals_far_future_before_current_pin(monkeypatch):
    pinned, model_management, _ = _load_pinned_memory(monkeypatch)
    model_management.has_live_pin_budget = lambda device: True
    blocks = [FakeBlock(index) for index in range(3)]
    order = PrefetchPinOrder(blocks, window=3)
    order.advance()

    current, incoming, far_future = blocks
    current_pin = FakePin(pointer=111)
    far_future_pin = FakePin(pointer=333)
    current._pins = {"weights": {"pin": current_pin, "registered": True, "stack_index": 0}}
    incoming._pins = {"weights": {}}
    far_future._pins = {"weights": {"pin": far_future_pin, "registered": True, "stack_index": 1}}
    stack = [(current, 0), (far_future, current_pin.nbytes)]
    buckets = {}
    pinned._add_to_bucket(current, current._pins["weights"], buckets, current_pin.nbytes, 100)
    pinned._add_to_bucket(far_future, far_future._pins["weights"], buckets, far_future_pin.nbytes, 200)

    assert pinned._steal_pin(incoming, stack, buckets, current_pin.nbytes, 0, "weights")
    assert incoming._pins["weights"]["pin"] is far_future_pin
    assert current._pins["weights"]["pin"] is current_pin
    assert "pin" not in far_future._pins["weights"]
    order.close()


def test_eviction_hierarchy_prefers_stale_then_generic_then_upcoming(monkeypatch):
    pinned, _, _ = _load_pinned_memory(monkeypatch)

    assert pinned.pin_eviction_priority((False, 50)) > pinned.pin_eviction_priority(None)
    assert pinned.pin_eviction_priority(None) > pinned.pin_eviction_priority((True, 2))
    assert pinned.pin_eviction_priority((True, 2)) > pinned.pin_eviction_priority((True, 0))


def test_register_failure_evicts_retries_and_updates_ledger_once(monkeypatch):
    pinned, model_management, _ = _load_pinned_memory(monkeypatch)
    module, pin, module_pin = _retained_pin(False)
    results = iter([1, 0])
    evictions = []
    monkeypatch.setattr(pinned, "_host_register", lambda value, size: next(results))
    model_management.free_registrations = lambda size, **kwargs: evictions.append(size) or True

    assert pinned.get_pin(module) is pin
    assert module_pin["registered"]
    assert evictions == [pin.nbytes]
    assert model_management.TOTAL_PINNED_MEMORY == pin.nbytes
    assert module._pin_state["weights"][3][0] == pin.nbytes


def test_double_register_failure_falls_back_pageable_without_accounting(monkeypatch):
    pinned, model_management, _ = _load_pinned_memory(monkeypatch)
    module, pin, module_pin = _retained_pin(False)
    monkeypatch.setattr(pinned, "_host_register", lambda value, size: 1)

    assert pinned.get_pin(module) is pin
    assert not module_pin["registered"]
    assert model_management.TOTAL_PINNED_MEMORY == 0
    assert module._pin_state["weights"][3][0] == 0
    assert pinned.PIN_SCHEDULER_STATS["register_failures"] == 2
    assert pinned.PIN_SCHEDULER_STATS["pageable_prefetches"] == 1


def test_disable_pinned_memory_keeps_existing_behavior(monkeypatch):
    pinned, model_management, args = _load_pinned_memory(monkeypatch)
    module, pin, module_pin = _retained_pin(False)
    args.disable_pinned_memory = True
    checks = []
    model_management.ensure_pin_registerable = lambda *args, **kwargs: checks.append(True)

    assert pinned.get_pin(module) is pin
    assert not module_pin["registered"]
    assert checks == []
