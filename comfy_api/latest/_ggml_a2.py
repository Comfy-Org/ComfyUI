"""Selected IQ/ternary/FP4 compatibility math, with call-owned lookups only."""
import torch

from ._gguf_tensor import GGMLType
from ._ggml_codebooks import BOOKS

TYPES = frozenset({GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ3_XXS, GGMLType.IQ1_S,
                   GGMLType.IQ4_NL, GGMLType.IQ3_S, GGMLType.IQ2_S, GGMLType.IQ4_XS,
                   GGMLType.IQ1_M, GGMLType.TQ1_0, GGMLType.TQ2_0, GGMLType.MXFP4,
                   GGMLType.NVFP4})


def _little(data):
    values = data.to(torch.int64)
    shifts = torch.arange(data.shape[-1], device=data.device, dtype=torch.int64) * 8
    return (values << shifts).sum(dim=-1)


def _bits(value, shifts, mask):
    lanes = torch.tensor(shifts, device=value.device, dtype=torch.int64)
    return (value.to(torch.int64).unsqueeze(-1) >> lanes) & mask


def _grid(kind, device):
    shape, values, encoded = BOOKS[kind]
    width = (len(values) - 1).bit_length()
    stride = 8 // (8 // width)
    codes = torch.tensor(list(encoded), device=device, dtype=torch.int64)
    indices = _bits(codes, tuple(range(0, 8, stride)), (1 << width) - 1)
    table = torch.tensor(values + (0,) * ((1 << width) - len(values)),
                         device=device, dtype=torch.float32)
    return table[indices.flatten()].reshape(shape)


def _signs(value):
    return 1.0 - 2.0 * _bits(value, tuple(range(8)), 1).to(torch.float32)


def _parity_signs(indices):
    parity = _bits(indices, tuple(range(7)), 1).sum(dim=-1) & 1
    return _signs(indices | (parity << 7))


def _half(data, dtype):
    return data.contiguous().view(torch.float16).to(dtype=dtype)


def _fp4_values(indices):
    values = torch.tensor((0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12),
                          dtype=torch.float32, device=indices.device)
    return values[indices.long()]


def dequantize_blocks(qtype, blocks, dtype):
    n = blocks.shape[0]
    if qtype in {GGMLType.IQ4_NL, GGMLType.IQ4_XS}:
        values = torch.tensor((-127, -104, -83, -65, -49, -35, -22, -10,
                                1, 13, 25, 38, 53, 69, 89, 113),
                              device=blocks.device, dtype=torch.float32)
        scale = _half(blocks[:, :2], dtype)
        if qtype == GGMLType.IQ4_NL:
            packed = blocks[:, 2:]
            indices = torch.cat((packed & 15, packed >> 4), dim=-1)
            return scale * values[indices.long()].to(dtype=dtype)
        high = _little(blocks[:, 2:4].reshape(n, 1, 2))
        low = _bits(blocks[:, 4:8], (0, 4), 15).reshape(n, 8)
        groups = (low | (_bits(high, tuple(range(0, 16, 2)), 3).reshape(n, 8) << 4)) - 32
        scales = (scale * groups.to(dtype=dtype)).unsqueeze(-1)
        packed = blocks[:, 8:].reshape(n, 8, 16)
        indices = torch.cat((packed & 15, packed >> 4), dim=-1)
        return (scales * values[indices.long()].to(dtype=dtype)).reshape(n, 256)
    if qtype in {GGMLType.TQ1_0, GGMLType.TQ2_0}:
        scale = _half(blocks[:, -2:], dtype)
        if qtype == GGMLType.TQ2_0:
            packed = blocks[:, :64].reshape(n, 2, 32)
            lanes = torch.stack(tuple(packed >> shift for shift in (0, 2, 4, 6)), dim=2)
            values = (lanes & 3).reshape(n, 256) - 1
        else:
            def trits(data, powers):
                factors = torch.tensor(powers, dtype=torch.uint8, device=blocks.device)
                return (data.unsqueeze(1) * factors.view(1, -1, 1)).reshape(n, -1)
            lanes = torch.cat((trits(blocks[:, :32], (1, 3, 9, 27, 81)),
                               trits(blocks[:, 32:48], (1, 3, 9, 27, 81)),
                               trits(blocks[:, 48:52], (1, 3, 9, 27))), dim=1)
            # Keep the selected native uint8 arithmetic, including wrapping.
            # Replacing it by signed/spec math would change node results.
            values = ((lanes * 3) >> 8) - 1
        return scale * values
    if qtype == GGMLType.MXFP4:
        exponent = blocks[:, :1].to(torch.int32)
        bits = torch.where(exponent < 2, torch.full_like(exponent, 2097152) << exponent,
                           (exponent - 1) << 23)
        scale = bits.view(torch.float32)
        packed = blocks[:, 1:]
        indices = torch.cat((packed & 15, packed >> 4), dim=-1)
        return (scale * _fp4_values(indices)).to(dtype=dtype)
    if qtype == GGMLType.NVFP4:
        encoded = blocks[:, :4].to(torch.int32)
        exponent, mantissa = (encoded >> 3) & 15, (encoded & 7).to(torch.float32)
        normal = (1.0 + mantissa / 8.0) * torch.pow(
            torch.tensor(2.0, dtype=torch.float32, device=blocks.device), exponent.float() - 7.0)
        scale = torch.where(exponent == 0, mantissa * 2.0**-9, normal)
        scale = torch.where((encoded == 0) | (encoded == 127), torch.zeros_like(scale), scale) * 0.5
        packed = blocks[:, 4:].reshape(n, 4, 8)
        indices = torch.cat((packed & 15, packed >> 4), dim=-1)
        return (scale.unsqueeze(-1) * _fp4_values(indices)).reshape(n, 64).to(dtype=dtype)
    if qtype not in TYPES:
        raise NotImplementedError(f'unsupported GGML forward tensor type {qtype!r}')
    if qtype == GGMLType.IQ1_M:
        scales = _little(blocks[:, 48:].reshape(n, 4, 2))
        word = ((scales[:, 0] & 0xF000) >> 12) | ((scales[:, 1] & 0xF000) >> 8)
        word |= ((scales[:, 2] & 0xF000) >> 4) | (scales[:, 3] & 0xF000)
        scale = word.to(torch.int16).contiguous().view(torch.float16).float().reshape(n, 1)
        local = _bits(scales, (0, 3, 6, 9), 7).reshape(n, 16)
        factor = (scale * (2 * local + 1).float()).reshape(n, 8, 2, 1, 1)
        high = _bits(blocks[:, 32:48], (0, 4), 15)
        indices = blocks[:, :32].long() | ((high & 7).reshape(n, 32) << 8)
        delta = torch.where((high & 8) == 0, 0.125, -0.125).reshape(n, 8, 2, 2, 1)
        values = _grid('iq1', blocks.device)[indices].reshape(n, 8, 2, 2, 8)
        return (factor * (values + delta)).reshape(n, 256).to(dtype=dtype)
    scale = _half(blocks[:, :2], torch.float32)
    if qtype == GGMLType.IQ1_S:
        high = _little(blocks[:, 34:].reshape(n, 8, 2))
        factor = (scale * (2 * ((high >> 12) & 7) + 1).float()).reshape(n, 8, 1, 1)
        delta = torch.where((high & 32768) == 0, 0.125, -0.125).reshape(n, 8, 1, 1)
        indices = blocks[:, 2:34].long() | (_bits(high, (0, 3, 6, 9), 7).reshape(n, 32) << 8)
        values = _grid('iq1', blocks.device)[indices].reshape(n, 8, 4, 8)
        return (factor * (values + delta)).reshape(n, 256).to(dtype=dtype)
    if qtype == GGMLType.IQ2_XXS:
        words = _little(blocks[:, 2:].reshape(n, 8, 2, 4))
        factor = (scale * (0.5 + (words[..., 1] >> 28).float()) * 0.25).reshape(n, 8, 1, 1)
        signs = _parity_signs(_bits(words[..., 1], (0, 7, 14, 21), 127))
        values = _grid('iq2_xxs', blocks.device)[_bits(words[..., 0], (0, 8, 16, 24), 255)]
    elif qtype == GGMLType.IQ2_XS:
        words = _little(blocks[:, 2:66].reshape(n, 32, 2))
        scales = _bits(blocks[:, 66:], (0, 4), 15).reshape(n, 16).float()
        factor = (scale * (0.5 + scales) * 0.25).reshape(n, 16, 1, 1)
        signs = _parity_signs(words >> 9).reshape(n, 16, 2, 8)
        values = _grid('iq2_xs', blocks.device)[words & 511].reshape(n, 16, 2, 8)
    elif qtype == GGMLType.IQ2_S:
        scales = _bits(blocks[:, 74:], (0, 4), 15).reshape(n, 16).float()
        factor = (scale * (0.5 + scales) * 0.25).reshape(n, 16, 1, 1)
        high = _bits(blocks[:, 66:74], (0, 2, 4, 6), 3).reshape(n, 32)
        indices = blocks[:, 2:34].long() | (high << 8)
        signs = _signs(blocks[:, 34:66]).reshape(n, 16, 2, 8)
        values = _grid('iq2_s', blocks.device)[indices].reshape(n, 16, 2, 8)
    elif qtype == GGMLType.IQ3_XXS:
        words = _little(blocks[:, 66:].reshape(n, 8, 4))
        factor = (scale * (0.5 + (words >> 28).float()) * 0.5).reshape(n, 8, 1, 1)
        signs = _parity_signs(_bits(words, (0, 7, 14, 21), 127))
        values = _grid('iq3_xxs', blocks.device)[blocks[:, 2:66].long()].reshape(n, 8, 4, 8)
    else:  # IQ3_S
        scales = _bits(blocks[:, 106:], (0, 4), 15).reshape(n, 8).float()
        factor = (scale * (1 + 2 * scales)).reshape(n, 8, 1, 1)
        high = _bits(blocks[:, 66:74], tuple(range(8)), 1).reshape(n, 64)
        indices = blocks[:, 2:66].long() | (high << 8)
        signs = _signs(blocks[:, 74:106]).reshape(n, 8, 4, 8)
        values = _grid('iq3_s', blocks.device)[indices].reshape(n, 8, 4, 8)
    return (factor * values * signs).reshape(n, 256).to(dtype=dtype)
