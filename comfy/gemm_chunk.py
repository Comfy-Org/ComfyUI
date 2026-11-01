"""Split large GEMMs on the ROCm builds that get them wrong.

Some ROCm builds return wrong results for GEMMs with more than 2**19 rows or
more than 65536 batches (ROCm/TheRock#8554, ROCm/TheRock#9552). A model that
pushes a million voxels through a Linear, or a million vertices through a
batched matrix product, then gets silently wrong numbers instead of an error.

The fault is measured against the CPU, so the split switches itself off once
the installed libraries are correct.
"""

import logging

import torch
import torch.nn.functional as F

from comfy.cli_args import args

# The reported faults start at 2**19 rows and 65536 batches; split well below both.
ROW_SPLIT = 1 << 18
BATCH_SPLIT = 1 << 15

# 65536 batches is the last count the faulty builds get right, so an override
# cannot be larger than that without reintroducing the fault.
MAX_SPLIT = 1 << 16

_IS_ROCM = torch.version.hip is not None

# --rocm-gemm-chunk overrides the measurement: 0 turns the split off, a positive
# number is the chunk size to use for both rows and batches.
_OVERRIDE = args.rocm_gemm_chunk
if _OVERRIDE is not None:
    _OVERRIDE = min(max(_OVERRIDE, 0), MAX_SPLIT)

_measured = False
_row_split = 0
_batch_split = 0


def _faulty_rows(device):
    """True when the device's GEMMs go wrong past 2**19 rows."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn((1 << 19) + 1, 8, generator=g)
    w = torch.randn(4, 8, generator=g)
    gpu = F.linear(x.to(device), w.to(device)).cpu()
    return not torch.allclose(gpu, F.linear(x, w), rtol=1e-2, atol=1e-2)


def _faulty_batches(device):
    """True when the device's batched GEMMs go wrong past 65536 batches."""
    g = torch.Generator().manual_seed(0)
    a = torch.randn(65537, 4, 4, generator=g)
    b = torch.randn(65537, 4, 4, generator=g)
    gpu = torch.bmm(a.to(device), b.to(device)).cpu()
    return not torch.allclose(gpu, torch.bmm(a, b), rtol=1e-2, atol=1e-2)


def measure(device):
    """Compare the device's large GEMMs against the CPU and record how far to split them.

    Called once, before the first model runs, so the split can be turned off on a
    ROCm build whose libraries are correct.
    """
    global _measured, _row_split, _batch_split
    if _measured or _OVERRIDE is not None or not _IS_ROCM or device.type != "cuda":
        return
    try:
        faulty_rows = _faulty_rows(device)
        faulty_batches = _faulty_batches(device)
    except Exception as e:
        logging.warning("Could not compare the GPU's large GEMM results against the CPU: {}".format(e))
        return
    _measured = True
    if faulty_rows:
        _row_split = ROW_SPLIT
    if faulty_batches:
        _batch_split = BATCH_SPLIT
    if faulty_rows or faulty_batches:
        logging.warning(
            "The GPU's BLAS returns wrong results for large GEMMs (ROCm/TheRock#8554, ROCm/TheRock#9552); "
            "splitting them into chunks of %d rows and %d batches.", _row_split, _batch_split)
    else:
        logging.info("The GPU's large GEMM results match the CPU.")


def row_split():
    """Rows a single GEMM may have before it has to be split; 0 when the backend gets them right."""
    if _OVERRIDE is not None:
        return _OVERRIDE
    if not _measured and _IS_ROCM:
        # Not checked yet: assume the fault is there so nothing comes out silently wrong.
        return ROW_SPLIT
    return _row_split


def batch_split():
    """Batches a single GEMM may have before it has to be split; 0 when the backend gets them right."""
    if _OVERRIDE is not None:
        return _OVERRIDE
    if not _measured and _IS_ROCM:
        return BATCH_SPLIT
    return _batch_split


def _chunks(size, limit):
    """The (start, end) ranges to split a size into."""
    if limit == 0 or size <= limit:
        return [(0, size)]
    return [(s, min(s + limit, size)) for s in range(0, size, limit)]


def _split_needed(tensor):
    """Only the GPU's BLAS is affected; a CPU tensor is always computed correctly."""
    return tensor.device.type != "cpu"


def linear(input, weight, bias=None):
    """F.linear, split into row chunks when the backend gets large GEMMs wrong."""
    limit = row_split()
    if limit == 0 or not _split_needed(input):
        return F.linear(input, weight, bias)
    rows = input.numel() // input.shape[-1]
    if rows <= limit:
        return F.linear(input, weight, bias)
    flat = input.reshape(-1, input.shape[-1])
    first = F.linear(flat[:limit], weight, bias)
    out_features = first.shape[-1]
    out = torch.empty((flat.shape[0], out_features), dtype=first.dtype, device=first.device)
    out[:limit] = first
    del first
    for s, e in _chunks(flat.shape[0], limit)[1:]:
        out[s:e] = F.linear(flat[s:e], weight, bias)
    return out.reshape(*input.shape[:-1], out_features)


def bmm(a, b):
    """torch.bmm, split over the batch count and over the rows of each batch element."""
    rows = row_split()
    batches = batch_split()
    if not _split_needed(a):
        return torch.bmm(a, b)
    B, M, _ = a.shape
    if (rows == 0 or M <= rows) and (batches == 0 or B <= batches):
        return torch.bmm(a, b)
    out = torch.empty((B, M, b.shape[-1]), dtype=a.dtype, device=a.device)
    for b0, b1 in _chunks(B, batches):
        for s0, s1 in _chunks(M, rows):
            out[b0:b1, s0:s1] = torch.bmm(a[b0:b1, s0:s1], b[b0:b1])
    return out


def einsum(equation, *operands):
    """torch.einsum, split over the leading batch dim when the backend gets large GEMMs wrong.

    Every operand shares its leading dim as a batch dim, and that dim comes first
    in the output.
    """
    limit = batch_split()
    if limit == 0 or not _split_needed(operands[0]) or operands[0].shape[0] <= limit:
        return torch.einsum(equation, *operands)
    total = operands[0].shape[0]
    first = torch.einsum(equation, *[o[:limit] for o in operands])
    out = torch.empty((total,) + tuple(first.shape[1:]), dtype=first.dtype, device=first.device)
    out[:limit] = first
    del first
    for s, e in _chunks(total, limit)[1:]:
        out[s:e] = torch.einsum(equation, *[o[s:e] for o in operands])
    return out
