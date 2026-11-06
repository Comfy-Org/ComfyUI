"""Host-owned GGUF storage views; parsing and name transforms stay in the guest."""
from enum import IntEnum
import json
import math
import mmap
import os
import stat
import sys

import torch


class GGMLType(IntEnum):
    F32 = 0
    F16 = 1
    Q8_0 = 8
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    TQ1_0 = 34
    TQ2_0 = 35
    MXFP4 = 39
    NVFP4 = 40


_LAYOUTS = {
    GGMLType.F32: (1, 4, torch.float32), GGMLType.F16: (1, 2, torch.float16),
    GGMLType.Q8_0: (32, 34, torch.uint8), GGMLType.Q4_K: (256, 144, torch.uint8),
    GGMLType.Q5_K: (256, 176, torch.uint8), GGMLType.Q6_K: (256, 210, torch.uint8),
    GGMLType.I8: (1, 1, torch.int8), GGMLType.I16: (1, 2, torch.int16),
    GGMLType.I32: (1, 4, torch.int32), GGMLType.I64: (1, 8, torch.int64),
    GGMLType.F64: (1, 8, torch.float64), GGMLType.BF16: (1, 2, torch.uint8),
    GGMLType.IQ2_XXS: (256, 66, torch.uint8), GGMLType.IQ2_XS: (256, 74, torch.uint8),
    GGMLType.IQ3_XXS: (256, 98, torch.uint8), GGMLType.IQ1_S: (256, 50, torch.uint8),
    GGMLType.IQ4_NL: (32, 18, torch.uint8), GGMLType.IQ3_S: (256, 110, torch.uint8),
    GGMLType.IQ2_S: (256, 82, torch.uint8), GGMLType.IQ4_XS: (256, 136, torch.uint8),
    GGMLType.IQ1_M: (256, 56, torch.uint8), GGMLType.TQ1_0: (256, 54, torch.uint8),
    GGMLType.TQ2_0: (256, 66, torch.uint8), GGMLType.MXFP4: (32, 17, torch.uint8),
    GGMLType.NVFP4: (64, 36, torch.uint8),
}


class GGUFTensor(torch.Tensor):
    """Quantized storage plus logical shape, retaining calcuis tensor semantics.

    clone/detach intentionally keep the wrapper; casting retains its metadata.
    The native copy_ refusal is deliberately suppressed by that tensor's API.
    This class supplies storage semantics, not forward-time dequantization ops.
    """
    @staticmethod
    def __new__(cls, raw, *, tensor_type, tensor_shape):
        return torch.Tensor._make_subclass(cls, raw, require_grad=False)

    def __init__(self, raw, *, tensor_type, tensor_shape):
        self.tensor_type = tensor_type
        self.tensor_shape = torch.Size(tensor_shape)
        self.patches = []

    @property
    def shape(self):
        return getattr(self, 'tensor_shape', self.size())

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        result.tensor_type = getattr(self, 'tensor_type', None)
        result.tensor_shape = getattr(self, 'tensor_shape', result.size())
        result.patches = getattr(self, 'patches', []).copy()
        return result

    def copy_(self, *args, **kwargs):
        try:
            return super().copy_(*args, **kwargs)
        except Exception:
            return None


def _shape(value, field):
    if (type(value) is not list or not 1 <= len(value) <= 8
            or any(type(n) is not int or not 1 <= n <= 2**63 - 1 for n in value)):
        raise ValueError(f'GGUF {field} must contain 1..8 positive integer dimensions')
    return tuple(value)


def tensor_state(path, table, max_bytes):
    if type(max_bytes) is not int or not 1 <= max_bytes <= 2**63 - 1:
        raise ValueError('GGUF mapping requires a positive finite max_bytes')
    if (type(table) is not dict or set(table) != {'format', 'byteorder', 'tensors'}
            or table['format'] != 'gguf-tensor-table/1'
            or table['byteorder'] != 'little' or sys.byteorder != 'little'):
        raise ValueError('GGUF mapping requires a little-endian gguf-tensor-table/1')
    entries = table['tensors']
    if type(entries) is not list or not 1 <= len(entries) <= 4096:
        raise ValueError('GGUF table requires 1..4096 tensors')
    if len(json.dumps(table, ensure_ascii=True).encode()) > 512 * 1024:
        raise ValueError('GGUF tensor table exceeds 512 KiB')
    selected = []
    names = set()
    for entry in entries:
        if type(entry) is not dict or set(entry) != {'name', 'offset', 'tensor_type',
                                                    'shape', 'storage_shape'}:
            raise ValueError('invalid GGUF tensor descriptor fields')
        name = entry['name']
        if type(name) is not str or not name or len(name.encode()) > 1024 or name in names:
            raise ValueError('GGUF tensor names must be nonempty and unique')
        names.add(name)
        if type(entry['tensor_type']) is not int:
            raise ValueError('GGUF tensor_type must be its integer GGML type')
        try:
            qtype = GGMLType(entry['tensor_type'])
        except ValueError:
            raise NotImplementedError(f"unsupported GGML tensor type {entry['tensor_type']}") from None
        block, width, dtype = _LAYOUTS[qtype]
        shape = _shape(entry['shape'], 'shape')
        storage = _shape(entry['storage_shape'], 'storage_shape')
        count = math.prod(shape)
        if count % block:
            raise ValueError(f'GGUF {qtype.name} element count is not block aligned')
        nbytes = count // block * width
        if math.prod(storage) * torch.empty((), dtype=dtype).element_size() != nbytes:
            raise ValueError('GGUF storage shape does not match its logical byte count')
        offset = entry['offset']
        if type(offset) is not int or offset < 0 or nbytes > max_bytes:
            raise ValueError('GGUF tensor byte range exceeds max_bytes')
        selected.append((name, offset, nbytes, dtype, qtype, shape, storage))
    ranges = sorted((row[1], row[1] + row[2]) for row in selected)
    if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
        raise ValueError('overlapping GGUF tensor storage is unsupported')
    with open(path, 'rb') as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise ValueError('GGUF asset exceeds the selected regular-file byte limit')
        if source.read(4) != b'GGUF':
            raise ValueError('GGUF asset has an unsupported magic or byte order')
        if any(end > info.st_size for _, end in ranges):
            raise ValueError('GGUF tensor byte range exceeds the asset')
        # Each storage keeps its memoryview/mapping alive. No global cache or
        # raw weight export is introduced, and writes cannot change the file.
        mapping = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_COPY)
    result = {}
    for name, offset, nbytes, dtype, qtype, shape, storage in selected:
        raw = torch.frombuffer(memoryview(mapping)[offset:offset + nbytes], dtype=dtype).reshape(storage)
        if qtype in {GGMLType.F16, GGMLType.F32}:
            raw = raw.view(shape)
        if qtype == GGMLType.BF16 and len(shape) != 2:
            bits = raw.view(torch.uint16).to(torch.int32) << 16
            result[name] = bits.view(torch.float32).to(torch.float16).reshape(shape)
        else:
            result[name] = GGUFTensor(raw, tensor_type=qtype, tensor_shape=shape)
    quantized = [tensor for tensor in result.values()
                 if getattr(tensor, 'tensor_type', None) not in {None, GGMLType.F32, GGMLType.F16}]
    if quantized:
        max(quantized, key=lambda tensor: tensor.numel()).is_largest_weight = True
    return result
