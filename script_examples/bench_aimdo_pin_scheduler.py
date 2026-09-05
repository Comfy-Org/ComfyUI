"""Exercise AIMDO pin scheduling with real CUDA host registrations and H2D copies.

The synthetic-baseline policy disables live DXGI budgeting and expects the
injected Windows registration ceiling to return cudaErrorMemoryAllocation.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import types
import weakref


MiB = 1024 ** 2
GiB = 1024 ** 3
RESULT_PREFIX = "AIMDO_PIN_SCHEDULER_RESULT="
COMFY_ROOT = Path(__file__).resolve().parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=50)
    parser.add_argument("--block-mib", type=int, default=4)
    parser.add_argument("--policy", choices=("synthetic", "synthetic-baseline", "native", "heuristic"), required=True)
    parser.add_argument("--delay-cycles", type=int, default=5_000_000)
    parser.add_argument("--output")
    parser.add_argument("--i-understand-this-uses-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_understand_this_uses_gpu:
        parser.error("pass --i-understand-this-uses-gpu after the required idle-GPU preflight")
    if args.blocks < 6 or args.block_mib <= 0 or args.delay_cycles < 0:
        parser.error("blocks must be at least 6 and sizes/delays must be non-negative")
    return args


class SyntheticBudgetProvider:
    def __init__(self, model_management, capacity):
        self.model_management = model_management
        self.capacity = capacity
        self.external_usage = 4 * GiB
        self.reserve = model_management.WINDOWS_PIN_SAFETY_RESERVE

    def query(self):
        from comfy.windows_dxgi import VideoMemoryInfo

        return VideoMemoryInfo(
            self.external_usage + self.reserve + self.capacity,
            self.external_usage + self.model_management.TOTAL_PINNED_MEMORY,
        )

    def close(self):
        pass


class Block:
    def __init__(self, torch, index, pin_state):
        self.index = index
        self._v = object()
        self._pin_state = pin_state
        self.weight = torch.empty(0)
        self.bias = None

    def modules(self):
        return (self,)


def empty_bucket(host_buffer):
    return (host_buffer.HostBuffer(0, 0, 0), [], [-1], [0], [0], {})


def make_pin_state(host_buffer, device, total_bytes):
    def bucket():
        return (host_buffer.HostBuffer(0, 0, total_bytes), [], [-1], [0], [0], {})

    return {
        "device": device,
        "active": True,
        "current_prompt": True,
        "prefetch_orders": weakref.WeakSet(),
        "weights": bucket(),
        "weights-loaded": empty_bucket(host_buffer),
        "patches": empty_bucket(host_buffer),
        "patches-loaded": empty_bucket(host_buffer),
    }


def make_patcher(pinned_memory, pin_state, device):
    model = types.SimpleNamespace(dynamic_pins={device: pin_state})
    patcher = types.SimpleNamespace(model=model, load_device=device)
    patcher.is_dynamic = lambda: True

    def unregister_inactive_pins(self, ram_to_unload, subsets=("weights-loaded", "patches-loaded", "weights", "patches"), protected=None, prefetch_only=False):
        state = self.model.dynamic_pins[self.load_device]
        return pinned_memory.unregister_inactive_pins(state, ram_to_unload, subsets, protected=protected, prefetch_only=prefetch_only)

    def partially_unload_ram(self, ram_to_unload, subsets=("weights-loaded", "patches-loaded", "weights", "patches"), protected=None):
        state = self.model.dynamic_pins[self.load_device]
        return pinned_memory.partially_unload_ram(state, ram_to_unload, subsets, protected=protected)

    patcher.unregister_inactive_pins = types.MethodType(unregister_inactive_pins, patcher)
    patcher.partially_unload_ram = types.MethodType(partially_unload_ram, patcher)
    return patcher


def registered_bytes(blocks):
    return sum(
        block._pins["weights"]["pin"].nbytes
        for block in blocks
        if block.__dict__.get("_pins", {}).get("weights", {}).get("registered")
    )


def run(args):
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "backend:native,garbage_collection_threshold:0.95,expandable_segments:False",
    )
    sys.argv = [sys.argv[0], "--enable-dynamic-vram", "--async-offload", "2"]

    import comfy.options

    comfy.options.enable_args_parsing()
    import comfy_aimdo.control as aimdo_control

    if not aimdo_control.init():
        raise RuntimeError("AIMDO native library initialization failed")

    import torch
    import comfy.model_management as model_management
    import comfy.pin_order as pin_order
    import comfy.pinned_memory as pinned_memory
    import comfy.system_memory
    import comfy_aimdo.host_buffer as host_buffer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.device >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {args.device} is unavailable")
    if args.policy == "native" and platform.system() != "Windows":
        raise RuntimeError("native DXGI policy is Windows-only")
    if args.policy.startswith("synthetic") and platform.system() != "Windows":
        raise RuntimeError("synthetic Windows policies are Windows-only")
    if args.policy == "heuristic" and platform.system() == "Windows":
        raise RuntimeError("heuristic policy is for the non-Windows run")

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    if not aimdo_control.init_device(args.device):
        raise RuntimeError(f"AIMDO device {args.device} initialization failed")

    block_bytes = args.block_mib * MiB
    pin_state = make_pin_state(host_buffer, device, args.blocks * block_bytes)
    blocks = [Block(torch, index, pin_state) for index in range(args.blocks)]
    patcher = make_patcher(pinned_memory, pin_state, device)
    loaded = types.SimpleNamespace(model=patcher)
    order = pin_order.PrefetchPinOrder(blocks, window=3)
    streams = [torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)]
    pending = []
    observations = []
    failures = []
    peak_registered = 0
    hard_protection_checks = 0
    original_max = model_management.MAX_PINNED_MEMORY
    original_available = comfy.system_memory.virtual_memory_available
    original_stats = dict(pinned_memory.PIN_SCHEDULER_STATS)
    original_host_register = pinned_memory._host_register
    provider = None
    synthetic_cuda_ooms = []
    budget_violations = []

    def capacity_for(index):
        if args.policy == "native":
            return None
        if index < args.blocks // 3:
            return 6 * block_bytes
        if index < (2 * args.blocks) // 3:
            return 2 * block_bytes
        return 4 * block_bytes

    def validate(entry):
        entry["event"].synchronize()
        error_count = int((entry["output"] != entry["expected"]).sum().item())
        if error_count:
            failures.append(f"block {entry['index']} H2D mismatch in {error_count} bytes")

    current_capacity = capacity_for(0)

    def capacity_limited_host_register(pin, size):
        if model_management.TOTAL_PINNED_MEMORY + size > current_capacity:
            synthetic_cuda_ooms.append({
                "block": order.current,
                "registered_bytes": model_management.TOTAL_PINNED_MEMORY,
                "requested_bytes": size,
                "capacity_bytes": current_capacity,
                "cuda_error": "cudaErrorMemoryAllocation",
                "cuda_error_code": 2,
            })
            return 2
        return original_host_register(pin, size)

    model_management.current_loaded_models.append(loaded)
    model_management.TOTAL_PINNED_MEMORY = 0
    for key in pinned_memory.PIN_SCHEDULER_STATS:
        pinned_memory.PIN_SCHEDULER_STATS[key] = 0
    comfy.system_memory.virtual_memory_available = lambda: 1 << 60
    if args.policy == "synthetic":
        provider = SyntheticBudgetProvider(model_management, capacity_for(0))
        model_management.set_pin_budget_provider(device, provider)
        pinned_memory._host_register = capacity_limited_host_register
    elif args.policy == "synthetic-baseline":
        model_management.set_pin_budget_provider(device, None)
        model_management.MAX_PINNED_MEMORY = args.blocks * block_bytes
        pinned_memory._host_register = capacity_limited_host_register
    elif args.policy == "heuristic":
        model_management.MAX_PINNED_MEMORY = capacity_for(0)

    torch.cuda.synchronize(device)
    try:
        for index, block in enumerate(blocks):
            while len(pending) >= 2:
                validate(pending.pop(0))

            for entry in pending:
                if not entry["event"].query():
                    hard_protection_checks += 1
                    state = entry["block"]._pins["weights"]
                    if not state.get("registered"):
                        failures.append(f"block {entry['index']} was unregistered during H2D")

            order.advance()
            capacity = capacity_for(index)
            current_capacity = capacity
            if provider is not None:
                provider.capacity = capacity
                order.budget_checked = model_management.ensure_pin_registerable(0, device=device)
            elif capacity is not None and args.policy != "synthetic-baseline":
                model_management.MAX_PINNED_MEMORY = capacity

            low_ram_injected = index == args.blocks // 2
            if low_ram_injected:
                comfy.system_memory.virtual_memory_available = lambda: 0
            try:
                pinned_memory.pin_memory(block, size=block_bytes)
            finally:
                comfy.system_memory.virtual_memory_available = lambda: 1 << 60

            module_pin = block.__dict__.get("_pins", {}).get("weights", {})
            pin = module_pin.get("pin")
            if pin is None:
                failures.append(f"block {index} obtained no host buffer")
                continue
            registered = bool(module_pin.get("registered"))
            pin.fill_(index % 251)
            stream = streams[index % len(streams)]
            output = torch.empty(block_bytes, dtype=torch.uint8, device=device)
            with torch.cuda.stream(stream):
                if args.delay_cycles:
                    torch.cuda._sleep(args.delay_cycles)
                output.copy_(pin, non_blocking=True)
                event = torch.cuda.Event()
                event.record(stream)
            pinned_memory.mark_modules_in_flight([block], stream)
            pending.append({
                "index": index,
                "block": block,
                "event": event,
                "output": output,
                "expected": index % 251,
            })

            current_registered = registered_bytes(blocks)
            peak_registered = max(peak_registered, current_registered)
            if capacity is not None and current_registered > capacity:
                violation = {
                    "block": index,
                    "registered_bytes": current_registered,
                    "capacity_bytes": capacity,
                }
                budget_violations.append(violation)
                if args.policy != "synthetic-baseline":
                    failures.append(
                        f"block {index} registered {current_registered} bytes above capacity {capacity}"
                    )
            if capacity is None or capacity >= block_bytes:
                if not registered:
                    failures.append(f"block {index} was pageable although one block fit")
            observations.append({
                "block": index,
                "capacity_bytes": capacity,
                "registered_bytes": current_registered,
                "registered_before_h2d": registered,
                "preferred": order.preferred_indices(),
                "low_ram_injected": low_ram_injected,
            })

        while pending:
            validate(pending.pop(0))

        query_stats = model_management.pin_budget_query_stats(device)
        if args.policy == "synthetic-baseline" and not synthetic_cuda_ooms:
            failures.append("baseline did not reach the injected cudaErrorMemoryAllocation boundary")
        if args.policy != "synthetic-baseline" and synthetic_cuda_ooms:
            failures.append("scheduler attempted registration above the injected Windows capacity")
        status = "pass"
        if failures:
            status = "fail"
        elif args.policy == "synthetic-baseline":
            status = "expected_cuda_oom"
        result = {
            "schema": 1,
            "status": status,
            "failures": failures,
            "platform": platform.platform(),
            "policy": args.policy,
            "aimdo_version": importlib.metadata.version("comfy-aimdo"),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "blocks": args.blocks,
            "block_bytes": block_bytes,
            "preferred_window": order.window,
            "peak_registered_bytes": peak_registered,
            "final_registered_bytes": registered_bytes(blocks),
            "hard_protection_checks": hard_protection_checks,
            "synthetic_cuda_ooms": synthetic_cuda_ooms,
            "budget_violations": budget_violations,
            "scheduler_stats": dict(pinned_memory.PIN_SCHEDULER_STATS),
            "budget_query_stats": None if query_stats is None else {
                "count": query_stats[0],
                "total_ns": query_stats[1],
                "max_ns": query_stats[2],
            },
            "observations": observations,
        }
    finally:
        torch.cuda.synchronize(device)
        order.close()
        for block in blocks:
            state = block.__dict__.get("_pins", {}).get("weights")
            if state is not None and state.get("registered"):
                pinned_memory.unregister_pin(block, "weights")
        model_management.current_loaded_models.remove(loaded)
        model_management.clear_pin_budget_providers()
        model_management.MAX_PINNED_MEMORY = original_max
        model_management.TOTAL_PINNED_MEMORY = 0
        pinned_memory._host_register = original_host_register
        comfy.system_memory.virtual_memory_available = original_available
        for key, value in original_stats.items():
            pinned_memory.PIN_SCHEDULER_STATS[key] = value
        del blocks, pin_state, streams
        gc.collect()
        torch.cuda.synchronize(device)
        aimdo_control.deinit()
    return result


def main():
    args = parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(RESULT_PREFIX + json.dumps(result, separators=(",", ":")))
    return 0 if result["status"] in ("pass", "expected_cuda_oom") else 1


if __name__ == "__main__":
    raise SystemExit(main())
