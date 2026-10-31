"""Compare FP32 single-token GQA operators with seeded synthetic tensors.

Run from the repository root: python script_examples/benchmark_fp32_gqa.py
No model, weights, or workflow is needed. Timings cover only attention operators.
"""

import argparse
import importlib.metadata
import json
import time

import comfy_kitchen
import torch
import torch.nn.functional as functional


def measure(operation, device, repeats):
    """Return an output, synchronized mean runtime, and extra allocated memory."""
    for _ in range(3):
        operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = None
        result = operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_extra_bytes = torch.cuda.max_memory_allocated(device) - baseline
    else:
        peak_extra_bytes = None
    elapsed = time.perf_counter() - started
    return result, {
        "seconds_per_call": elapsed / repeats,
        "peak_extra_allocated_bytes": peak_extra_bytes,
        "finite": bool(torch.isfinite(result).all()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    device = torch.device(args.device)
    torch.manual_seed(2608)
    torch.set_num_threads(2)
    key = torch.randn((2, 8, 4278, 128), device=device, dtype=torch.float32)
    value = torch.randn_like(key)
    query = torch.randn((2, 32, 1, 128), device=device, dtype=torch.float32)
    cases = []
    for length in (776, 4278):
        keys = key[:, :, :length, :]
        values = value[:, :, :length, :]
        lengths = torch.full((2,), length, dtype=torch.int32, device=device)
        for query_scale in (0.1, 1.0, 10.0):
            queries = query * query_scale
            baseline, baseline_metrics = measure(
                lambda: functional.scaled_dot_product_attention(
                    queries, keys, values, dropout_p=0.0,
                    is_causal=False, enable_gqa=True), device, args.repeats)
            grouped, grouped_metrics = measure(
                lambda: comfy_kitchen.flash_attention_decode_gqa(
                    queries, keys, values, lengths, causal=False),
                device, args.repeats)
            expected = baseline.transpose(1, 2).reshape(2, 1, 4096)
            delta = (expected - grouped).abs()
            cases.append({
                "prefix_length": length,
                "query_scale": query_scale,
                "sdpa_enable_gqa": baseline_metrics,
                "comfy_kitchen_gqa": grouped_metrics,
                "max_abs_error": delta.max().item(),
                "mean_abs_error": delta.mean().item(),
                "close_at_1e_minus_5": bool(torch.allclose(
                    expected, grouped, atol=1e-5, rtol=1e-5)),
            })
            del baseline, grouped, expected, delta
    result = {
        "torch": torch.__version__,
        "comfy_kitchen": importlib.metadata.version("comfy-kitchen"),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "seed": 2608,
        "repeats": args.repeats,
        "query_shape": list(query.shape),
        "key_value_shape": list(key.shape),
        "cases": cases,
        "claim_limit": "Synthetic operator benchmark; not a full-model or audio quality test.",
    }
    print(json.dumps(result, indent=2))  # noqa: T201
    return 0 if all(case["close_at_1e_minus_5"] for case in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
