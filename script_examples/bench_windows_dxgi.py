"""Measure the Windows DXGI NON_LOCAL budget query used by AIMDO pin scheduling."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path


COMFY_ROOT = Path(__file__).resolve().parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from comfy.windows_dxgi import create_dxgi_budget_provider


def percentile(values, percent):
    position = (len(values) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark IDXGIAdapter3.QueryVideoMemoryInfo latency for AIMDO.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output")
    parser.add_argument("--i-understand-this-initializes-cuda", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Windows":
        parser.error("the DXGI NON_LOCAL query is only available on Windows")
    if args.device < 0:
        parser.error("--device must be non-negative")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if not args.i_understand_this_initializes_cuda:
        parser.error("pass --i-understand-this-initializes-cuda after the required GPU preflight")
    return args


def main():
    args = parse_args()
    provider = create_dxgi_budget_provider(args.device)
    if provider is None:
        raise RuntimeError("the CUDA device could not be mapped unambiguously to a DXGI adapter")
    adapter_description = provider.adapter.description
    node_index = provider.node_index

    try:
        for _ in range(args.warmup):
            provider.query()

        samples_ns = []
        info = None
        for _ in range(args.iterations):
            start = time.perf_counter_ns()
            info = provider.query()
            samples_ns.append(time.perf_counter_ns() - start)
    finally:
        provider.close()

    samples_ns.sort()
    result = {
        "schema": 1,
        "platform": platform.platform(),
        "device": args.device,
        "adapter": adapter_description,
        "node_index": node_index,
        "warmup_queries": args.warmup,
        "measured_queries": args.iterations,
        "query_latency_us": {
            "median": round(statistics.median(samples_ns) / 1000, 3),
            "p95": round(percentile(samples_ns, 95) / 1000, 3),
            "p99": round(percentile(samples_ns, 99) / 1000, 3),
            "maximum": round(max(samples_ns) / 1000, 3),
            "total_ms": round(sum(samples_ns) / 1_000_000, 3),
        },
        "last_info": {
            "budget": info.budget,
            "current_usage": info.current_usage,
            "available_for_reservation": info.available_for_reservation,
            "current_reservation": info.current_reservation,
        },
        "metric_notes": {
            "measured_queries": "Explicit provider.query calls made by this probe.",
            "query_latency_us": "Python wall time around one QueryVideoMemoryInfo call; excludes CUDA LUID mapping and adapter creation.",
            "production_counts": "Use AIMDO detail logs for cumulative scheduler query count and time during a real workload.",
        },
    }
    output = json.dumps(result, indent=2)
    if args.output is not None:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
