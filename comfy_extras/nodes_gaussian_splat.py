# Generic utility nodes for the SPLAT type (3D gaussian splats)

import gzip
import logging
import math
import struct
from io import BytesIO

import numpy as np
import torch
from typing_extensions import override
from scipy.ndimage import map_coordinates, minimum as _ndi_minimum, maximum as _ndi_maximum
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import comfy.model_management
import comfy.utils
from comfy_api.latest import ComfyExtension, IO, Types
from comfy_extras.nodes_save_3d import pack_variable_mesh_batch
from server import PromptServer

_C0 = 0.28209479177387814  # SH band-0 constant: DC coefficient -> base RGB


def _srgb_to_linear(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c.clamp_min(0) + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c.clamp_min(0) ** (1 / 2.4) - 0.055)


def _real_len(g: Types.SPLAT, i: int) -> int:
    # Real splat count of batch item i (honors variable-length `counts`).
    return int(g.counts[i].item()) if g.counts is not None else g.positions.shape[1]


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    # "#RRGGBB" -> (r,g,b) in [0,1]; falls back to black.
    h = h.lstrip("#")
    if len(h) != 6:
        return (0.0, 0.0, 0.0)
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _quantile(x, q):
    # torch.quantile errors above 2**24 elements; stride-subsample large inputs for the estimate.
    lim = 1 << 24
    if x.numel() > lim:
        x = x[:: x.numel() // lim + 1]
    return torch.quantile(x, q)


def _gaussian_ply_bytes(positions, scales, rotations, opacities, sh) -> bytes:
    """Serialize render-ready gaussian tensors as a binary 3DGS .ply.

    positions (N,3) world; scales (N,3) linear; rotations (N,4) quat wxyz; opacities (N,1) in [0,1];
    sh (N,K,3) SH coefficients. Activated values are inverted to the standard 3D gaussian splat storage convention
    (log scale, logit opacity).
    """
    xyz = positions.cpu().numpy().astype(np.float32)
    n = xyz.shape[0]
    if n == 0:
        raise ValueError("SplatToFile3D: gaussian is empty")
    normals = np.zeros_like(xyz)
    f = sh.cpu().numpy().astype(np.float32)                  # (N, K, 3)
    f_dc = f[:, 0, :]                                        # (N, 3)
    f_rest = f[:, 1:, :].transpose(0, 2, 1).reshape(n, -1)   # (N, 3*(K-1)) channel-major
    op = opacities.cpu().numpy().astype(np.float32).reshape(n, 1).clip(1e-6, 1 - 1e-6)
    op = np.log(op / (1.0 - op))                             # inverse sigmoid (logit)
    scale = np.log(scales.cpu().numpy().astype(np.float32).clip(min=1e-8))
    rot = rotations.cpu().numpy().astype(np.float32)         # (N, 4)

    attrs = (['x', 'y', 'z', 'nx', 'ny', 'nz']
             + [f'f_dc_{i}' for i in range(3)]
             + [f'f_rest_{i}' for i in range(f_rest.shape[1])]
             + ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)])
    elements = np.empty(n, dtype=[(a, 'f4') for a in attrs])
    elements[:] = list(map(tuple, np.concatenate([xyz, normals, f_dc, f_rest, op, scale, rot], axis=1)))

    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
    header += "".join(f"property float {a}\n" for a in attrs) + "end_header\n"
    return header.encode('ascii') + elements.tobytes()


# .ksplat (mkkellogg SplatBuffer) level 0, SH degree 0: 4096-byte header, one 1024-byte section header,
# then N 44-byte records. Bucketing/quantization only exist at levels >= 1. See SplatBuffer.js.
_KSPLAT_HEADER_BYTES = 4096
_KSPLAT_SECTION_HEADER_BYTES = 1024
_KSPLAT_BYTES_PER_SPLAT = 44  # center 12 + scale 12 + rotation 16 + color(RGBA u8) 4
_KSPLAT_VERSION = (0, 1)      # SplatBuffer CurrentMajor/MinorVersion


def _gaussian_ksplat_bytes(positions, scales, rotations, opacities, sh) -> bytes:
    """Serialize gaussian tensors as a level-0, SH degree-0 .ksplat (linear scale, opacity in color alpha).

    positions (N,3) world; scales (N,3) linear; rotations (N,4) wxyz; opacities (N,1) in [0,1]; sh (N,K,3).
    """
    xyz = positions.cpu().numpy().astype(np.float32)
    n = xyz.shape[0]
    if n == 0:
        raise ValueError("SplatToFile3D: gaussian is empty")
    scale = scales.cpu().numpy().astype(np.float32)
    rot = rotations.cpu().numpy().astype(np.float32)                      # wxyz, mirrors the .ply rot order
    rot = rot / np.linalg.norm(rot, axis=1, keepdims=True).clip(1e-12)
    rgb = np.clip(sh[:, 0, :].cpu().numpy().astype(np.float32) * _C0 + 0.5, 0, 1)
    op = opacities.cpu().numpy().astype(np.float32).reshape(n, 1).clip(0, 1)
    rgba = np.round(np.concatenate([rgb, op], axis=1) * 255.0).astype(np.uint8)   # (N, 4) RGBA

    # 44-byte record: float center(3) + scale(3) + rot(4), then uint8 rgba(4).
    floats = np.concatenate([xyz, scale, rot], axis=1).astype('<f4')     # (N, 10)
    rec = np.empty(n, dtype=[('f', '<f4', 10), ('c', 'u1', 4)])
    rec['f'] = floats
    rec['c'] = rgba
    splat_data = rec.tobytes()

    header = bytearray(_KSPLAT_HEADER_BYTES)
    header[0] = _KSPLAT_VERSION[0]
    header[1] = _KSPLAT_VERSION[1]
    struct.pack_into('<I', header, 4, 1)        # maxSectionCount
    struct.pack_into('<I', header, 8, 1)        # sectionCount
    struct.pack_into('<I', header, 12, n)       # maxSplatCount
    struct.pack_into('<I', header, 16, n)       # splatCount
    struct.pack_into('<H', header, 20, 0)       # compressionLevel
    struct.pack_into('<fff', header, 24, 0.0, 0.0, 0.0)   # sceneCenter
    struct.pack_into('<ff', header, 36, 0.0, 0.0)         # min/max SH coeff (unused at degree 0)

    section = bytearray(_KSPLAT_SECTION_HEADER_BYTES)
    struct.pack_into('<I', section, 0, n)       # splatCount
    struct.pack_into('<I', section, 4, n)       # maxSplatCount
    # offsets 8..24: bucketSize/bucketCount/bucketBlockSize/bucketStorageSizeBytes/compressionScaleRange = 0
    struct.pack_into('<I', section, 28, n * _KSPLAT_BYTES_PER_SPLAT)   # storageSizeBytes
    struct.pack_into('<I', section, 32, 0)      # fullBucketCount
    struct.pack_into('<I', section, 36, 0)      # partiallyFilledBucketCount
    struct.pack_into('<H', section, 40, 0)      # sphericalHarmonicsDegree

    return bytes(header) + bytes(section) + splat_data


# .spz (Niantic) version 2, gzip-wrapped, SH degree 0: a 16-byte header then per-attribute arrays
# (positions 24-bit fixed point, then 1B alpha, 3B color, 3B scale, 3B rotation per splat). The
# quantizations below invert spark's SpzReader decode formulas.
_SPZ_MAGIC = 0x5053474E               # "NGSP"
_SPZ_VERSION = 2
_SPZ_FRACTIONAL_BITS = 12             # position fixed-point precision (~0.24mm at unit scale)
_SPZ_COLOR_SCALE = _C0 / 0.15         # contrast factor applied when decoding color bytes


def _gaussian_spz_bytes(positions, scales, rotations, opacities, sh) -> bytes:
    """Serialize gaussian tensors as a gzip-compressed .spz (Niantic v2, SH degree 0, base color only).

    positions (N,3) world; scales (N,3) linear; rotations (N,4) wxyz; opacities (N,1) in [0,1]; sh (N,K,3).
    """
    xyz = positions.cpu().numpy().astype(np.float32)
    n = xyz.shape[0]
    if n == 0:
        raise ValueError("SplatToFile3D: gaussian is empty")

    # Positions: fixed point, masked to 24 bits, little-endian 3-byte words.
    fixed = 1 << _SPZ_FRACTIONAL_BITS
    qi = np.clip(np.round(xyz * fixed), -(1 << 23), (1 << 23) - 1).astype(np.int32)
    qu = (qi & 0xFFFFFF).astype(np.uint32)
    pos = np.stack([qu & 0xFF, (qu >> 8) & 0xFF, (qu >> 16) & 0xFF], axis=-1).reshape(n, 9).astype(np.uint8)

    alpha = np.round(opacities.cpu().numpy().astype(np.float32).reshape(n) * 255.0).clip(0, 255).astype(np.uint8)

    rgb = sh[:, 0, :].cpu().numpy().astype(np.float32) * _C0 + 0.5
    col = np.round(((rgb - 0.5) / _SPZ_COLOR_SCALE + 0.5) * 255.0).clip(0, 255).astype(np.uint8)   # (N,3)

    sln = np.log(scales.cpu().numpy().astype(np.float32).clip(min=1e-9))
    scb = np.round((sln + 10.0) * 16.0).clip(0, 255).astype(np.uint8)     # (N,3) inverts exp(b/16-10)

    rot = rotations.cpu().numpy().astype(np.float32)                      # wxyz
    rot = rot / np.linalg.norm(rot, axis=1, keepdims=True).clip(1e-12)
    rot[rot[:, 0] < 0] *= -1.0                                            # canonical w >= 0 (w dropped on decode)
    rotb = np.round((rot[:, 1:4] + 1.0) * 127.5).clip(0, 255).astype(np.uint8)   # (N,3) x,y,z

    header = bytearray(16)
    struct.pack_into('<I', header, 0, _SPZ_MAGIC)
    struct.pack_into('<I', header, 4, _SPZ_VERSION)
    struct.pack_into('<I', header, 8, n)
    header[12] = 0                       # shDegree
    header[13] = _SPZ_FRACTIONAL_BITS
    header[14] = 0                       # flags
    header[15] = 0                       # reserved

    raw = (bytes(header) + pos.tobytes() + alpha.tobytes()
           + col.tobytes() + scb.tobytes() + rotb.tobytes())
    return gzip.compress(raw)


# ---- Readers: splat file bytes -> (positions, scales linear, rotations wxyz, opacities [0,1], sh (N,K,3)) ----
# Inverse of the writers above and of spark's loaders. ksplat/splat/spz carry base color only (SH degree 0
# -> K=1); .ply round-trips full SH. None of the formats flip axes, so import is the identity of export.
_PLY_DTYPES = {'char': 'i1', 'uchar': 'u1', 'short': 'i2', 'ushort': 'u2', 'int': 'i4', 'uint': 'u4',
               'float': 'f4', 'double': 'f8', 'int8': 'i1', 'uint8': 'u1', 'int16': 'i2', 'uint16': 'u2',
               'int32': 'i4', 'uint32': 'u4', 'float32': 'f4', 'float64': 'f8'}
_KSPLAT_COMPRESSION = {  # level -> (bytesPerCenter, scale, rotation, color, shComponent, defaultScaleRange)
    0: (12, 12, 16, 4, 4, 1), 1: (6, 6, 8, 4, 2, 32767), 2: (6, 6, 8, 4, 1, 32767)}
_KSPLAT_SH_COMPONENTS = {0: 0, 1: 9, 2: 24, 3: 45}


def _rgb_to_sh_dc(rgb):
    return ((np.asarray(rgb, np.float32) - 0.5) / _C0)[:, None, :]   # (N,3) base color -> (N,1,3) SH DC


def _norm_quat(q):
    return q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-12)


def _parse_ply_gaussian(data: bytes):
    end = data.find(b'end_header')
    if end < 0:
        raise ValueError("File3DToSplat: not a PLY (missing end_header)")
    header = data[:end].decode('ascii', 'replace')
    body = end + len(b'end_header')
    body += 2 if data[body:body + 2] == b'\r\n' else 1
    count, props, in_vertex = 0, [], False
    for line in header.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == 'format' and p[1] != 'binary_little_endian':
            raise ValueError(f"File3DToSplat: unsupported PLY format '{p[1]}' (need binary_little_endian)")
        if p[0] == 'element':
            in_vertex = p[1] == 'vertex'
            if in_vertex:
                count = int(p[2])
        elif p[0] == 'property' and in_vertex:
            if p[1] == 'list':
                raise ValueError("File3DToSplat: PLY vertex has list properties (unsupported)")
            props.append((p[2], '<' + _PLY_DTYPES[p[1]]))
    arr = np.frombuffer(data, np.dtype(props), count=count, offset=body)
    names = arr.dtype.names
    c = lambda k: arr[k].astype(np.float32)
    n = count

    xyz = np.stack([c('x'), c('y'), c('z')], 1)
    if 'scale_0' in names:
        scale = np.exp(np.stack([c('scale_0'), c('scale_1'), c('scale_2')], 1))   # 3DGS stores log scale
    else:
        scale = np.full((n, 3), 0.01, np.float32)
    if 'rot_0' in names:
        rot = _norm_quat(np.stack([c('rot_0'), c('rot_1'), c('rot_2'), c('rot_3')], 1))   # wxyz
    else:
        rot = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))
    opacity = 1.0 / (1.0 + np.exp(-c('opacity'))) if 'opacity' in names else np.ones(n, np.float32)

    if 'f_dc_0' in names:
        dc = np.stack([c('f_dc_0'), c('f_dc_1'), c('f_dc_2')], 1)                 # (N,3)
        rest = sorted((k for k in names if k.startswith('f_rest_')), key=lambda s: int(s.split('_')[-1]))
        if rest:
            r = np.stack([c(k) for k in rest], 1)                                 # (N, 3*(K-1)) channel-major
            kk = r.shape[1] // 3 + 1
            r = r.reshape(n, 3, kk - 1).transpose(0, 2, 1)                        # -> (N, K-1, 3)
            sh = np.concatenate([dc[:, None, :], r], 1)
        else:
            sh = dc[:, None, :]
    elif 'red' in names:
        sh = _rgb_to_sh_dc(np.stack([c('red'), c('green'), c('blue')], 1) / 255.0)
    else:
        sh = np.zeros((n, 1, 3), np.float32)
    return xyz, scale, rot, opacity, sh


def _parse_splat_gaussian(data: bytes):
    # antimatter15 .splat: 32-byte records (f32 xyz, f32 scale, u8 rgba, u8 quat as (b-128)/128 wxyz).
    if len(data) % 32 != 0:
        raise ValueError("File3DToSplat: .splat size is not a multiple of 32 bytes")
    rec = np.frombuffer(data, np.dtype([('xyz', '<f4', 3), ('scale', '<f4', 3),
                                        ('rgba', 'u1', 4), ('quat', 'u1', 4)]))
    rgba = rec['rgba'].astype(np.float32) / 255.0
    rot = _norm_quat((rec['quat'].astype(np.float32) - 128.0) / 128.0)            # wxyz
    return (rec['xyz'].astype(np.float32), rec['scale'].astype(np.float32), rot,
            rgba[:, 3].copy(), _rgb_to_sh_dc(rgba[:, :3]))


def _parse_ksplat_gaussian(data: bytes):
    # mkkellogg SplatBuffer: 4096-byte header, N section headers, then per-section splat data. Supports
    # levels 0 (float) / 1 (half + bucketed positions) / 2 (half, uint8 SH). SH is skipped (base color kept).
    if data[0] != 0:
        raise ValueError(f"File3DToSplat: unsupported .ksplat version {data[0]}.{data[1]}")
    max_sections = struct.unpack_from('<I', data, 4)[0]
    level = struct.unpack_from('<H', data, 20)[0]
    if level not in _KSPLAT_COMPRESSION:
        raise ValueError(f"File3DToSplat: invalid .ksplat compression level {level}")
    bc, bs, br, bcol, bshc, default_range = _KSPLAT_COMPRESSION[level]

    parts = []
    base = 4096 + max_sections * 1024
    for s in range(max_sections):
        so = 4096 + s * 1024
        cnt = struct.unpack_from('<I', data, so + 0)[0]
        sec_max = struct.unpack_from('<I', data, so + 4)[0]
        bucket_size = struct.unpack_from('<I', data, so + 8)[0]
        bucket_count = struct.unpack_from('<I', data, so + 12)[0]
        block_size = struct.unpack_from('<f', data, so + 16)[0]
        bucket_store = struct.unpack_from('<H', data, so + 20)[0]
        scale_range = struct.unpack_from('<I', data, so + 24)[0] or default_range
        full_buckets = struct.unpack_from('<I', data, so + 32)[0]
        partial_buckets = struct.unpack_from('<I', data, so + 36)[0]
        sh_components = _KSPLAT_SH_COMPONENTS.get(struct.unpack_from('<H', data, so + 40)[0], 0)
        bytes_per_splat = bc + bs + br + bcol + sh_components * bshc
        meta_bytes = partial_buckets * 4
        buckets_store = bucket_store * bucket_count + meta_bytes
        data_base = base + buckets_store

        if cnt > 0:
            ct, ft = ('<f4', '<f4') if level == 0 else ('<u2', '<f2')
            fields = [('center', ct, 3), ('scale', ft, 3), ('rot', ft, 4), ('color', 'u1', 4)]
            if sh_components:
                fields.append(('sh', '<f2' if level == 1 else ('<f4' if level == 0 else 'u1'), sh_components))
            rec = np.frombuffer(data, np.dtype(fields), count=cnt, offset=data_base)
            colf = rec['color'].astype(np.float32) / 255.0
            rot = _norm_quat(rec['rot'].astype(np.float32))                       # wxyz
            scale = rec['scale'].astype(np.float32)
            if level == 0:
                xyz = rec['center'].astype(np.float32)
            else:
                buckets = np.frombuffer(data, '<f4', count=bucket_count * 3, offset=base + meta_bytes).reshape(-1, 3)
                idx = np.empty(cnt, np.int64)
                full_splats = full_buckets * bucket_size
                nf = min(full_splats, cnt)
                idx[:nf] = np.arange(nf) // bucket_size
                if cnt > full_splats:
                    lengths = np.frombuffer(data, '<u4', count=partial_buckets, offset=base)
                    idx[full_splats:] = np.repeat(full_buckets + np.arange(partial_buckets), lengths)[:cnt - full_splats]
                xyz = (rec['center'].astype(np.float32) - scale_range) * (block_size / 2.0 / scale_range) + buckets[idx]
            parts.append((xyz, scale, rot, colf[:, 3].copy(), _rgb_to_sh_dc(colf[:, :3])))
        base += bytes_per_splat * sec_max + buckets_store

    if not parts:
        raise ValueError("File3DToSplat: .ksplat has no splats")
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(5))


def _parse_spz_gaussian(data: bytes):
    # Niantic .spz (gzip-wrapped), versions 1-3. Base color only (SH skipped). See spark's SpzReader.
    raw = gzip.decompress(data)
    if struct.unpack_from('<I', raw, 0)[0] != _SPZ_MAGIC:
        raise ValueError("File3DToSplat: invalid .spz (bad magic)")
    version = struct.unpack_from('<I', raw, 4)[0]
    n = struct.unpack_from('<I', raw, 8)[0]
    frac_bits = raw[13]
    off = 16

    if version == 1:
        xyz = np.frombuffer(raw, '<f2', count=n * 3, offset=off).astype(np.float32).reshape(n, 3)
        off += n * 6
    elif version in (2, 3):
        b = np.frombuffer(raw, np.uint8, count=n * 9, offset=off).reshape(n, 3, 3).astype(np.int64)
        v = (b[..., 2] << 16) | (b[..., 1] << 8) | b[..., 0]
        v = np.where(v & 0x800000, v - 0x1000000, v)                             # sign-extend 24-bit
        xyz = (v / (1 << frac_bits)).astype(np.float32)
        off += n * 9
    else:
        raise ValueError(f"File3DToSplat: unsupported .spz version {version}")

    alpha = np.frombuffer(raw, np.uint8, count=n, offset=off).astype(np.float32) / 255.0
    off += n
    cb = np.frombuffer(raw, np.uint8, count=n * 3, offset=off).reshape(n, 3).astype(np.float32)
    off += n * 3
    rgb = (cb / 255.0 - 0.5) * _SPZ_COLOR_SCALE + 0.5
    sb = np.frombuffer(raw, np.uint8, count=n * 3, offset=off).reshape(n, 3).astype(np.float32)
    off += n * 3
    scale = np.exp(sb / 16.0 - 10.0)

    if version == 3:                                                             # smallest-three quaternion
        qb = np.frombuffer(raw, np.uint8, count=n * 4, offset=off).reshape(n, 4).astype(np.int64)
        combined = qb[:, 0] | (qb[:, 1] << 8) | (qb[:, 2] << 16) | (qb[:, 3] << 24)
        largest = (combined >> 30) & 3
        q = np.zeros((n, 4), np.float32)                                         # x,y,z,w
        remaining, sumsq = combined.copy(), np.zeros(n, np.float64)
        for comp in (3, 2, 1, 0):
            active = comp != largest
            value = (remaining & 0x1FF).astype(np.float64)
            sign = (remaining >> 9) & 1
            remaining = np.where(active, remaining >> 10, remaining)
            val = (1.0 / math.sqrt(2)) * (value / 0x1FF)
            val = np.where(sign == 1, -val, val)
            q[active, comp] = val[active]
            sumsq += np.where(active, val * val, 0.0)
        q[np.arange(n), largest] = np.sqrt(np.clip(1.0 - sumsq, 0, None))
        rot = _norm_quat(np.stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]], 1))      # xyzw -> wxyz
    else:
        qb = np.frombuffer(raw, np.uint8, count=n * 3, offset=off).reshape(n, 3).astype(np.float32)
        xq = qb / 127.5 - 1.0
        w = np.sqrt(np.clip(1.0 - (xq ** 2).sum(1), 0, None))
        rot = _norm_quat(np.concatenate([w[:, None], xq], 1))                    # wxyz
    return xyz, scale, rot, alpha, _rgb_to_sh_dc(rgb)


_GAUSSIAN_PARSERS = {"ply": _parse_ply_gaussian, "splat": _parse_splat_gaussian,
                     "ksplat": _parse_ksplat_gaussian, "spz": _parse_spz_gaussian}


def _detect_splat_format(data: bytes) -> str:
    if data[:3] == b'ply':
        return "ply"
    if data[:2] == b'\x1f\x8b':            # gzip -> spz
        return "spz"
    if len(data) >= 2 and data[0] == 0 and data[1] >= 1:   # ksplat version 0.x header
        return "ksplat"
    if len(data) % 32 == 0:
        return "splat"
    raise ValueError("File3DToSplat: could not determine splat format from contents")


def _gaussian_item(g: Types.SPLAT, i: int, device):
    # Slice batch item i to its real length, as float32 torch tensors on `device` (SH DC -> base RGB).
    end = _real_len(g, i)
    to = lambda a: a.to(device=device, dtype=torch.float32)
    xyz = to(g.positions[i, :end])
    rgb = (to(g.sh[i, :end, 0, :]) * _C0 + 0.5).clamp(0, 1)
    opacity = to(g.opacities[i, :end]).reshape(-1)
    scale = to(g.scales[i, :end])
    rot = to(g.rotations[i, :end])
    return xyz, rgb, opacity, scale, rot


def _quat_to_mat(q):
    # q: (N, 4) wxyz, normalized -> (N, 3, 3)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def _quat_mul(a, b):
    # Hamilton product a (x) b, wxyz.
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _euler_to_quat(rx, ry, rz):
    # Degrees, applied as Rz @ Ry @ Rx (rotate about X, then Y, then Z in world). Returns wxyz.
    c, s = np.cos(np.radians([rx, ry, rz]) / 2.0), np.sin(np.radians([rx, ry, rz]) / 2.0)
    qx = torch.tensor([c[0], s[0], 0.0, 0.0], dtype=torch.float32)
    qy = torch.tensor([c[1], 0.0, s[1], 0.0], dtype=torch.float32)
    qz = torch.tensor([c[2], 0.0, 0.0, s[2]], dtype=torch.float32)
    return _quat_mul(_quat_mul(qz, qy), qx)


def _mat_to_quat(m):
    # Rotation matrix (..., 3, 3) -> quaternion (..., 4) wxyz. Batched; builds the four candidate quaternions
    # and keeps the one with the largest component (numerically stable across all rotations).
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    m21, m12 = m[..., 2, 1], m[..., 1, 2]
    m02, m20 = m[..., 0, 2], m[..., 2, 0]
    m10, m01 = m[..., 1, 0], m[..., 0, 1]
    q2 = torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                      1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1)   # 4 * (w^2, x^2, y^2, z^2)
    cand = torch.stack([
        torch.stack([q2[..., 0], m21 - m12, m02 - m20, m10 - m01], -1),
        torch.stack([m21 - m12, q2[..., 1], m10 + m01, m02 + m20], -1),
        torch.stack([m02 - m20, m10 + m01, q2[..., 2], m12 + m21], -1),
        torch.stack([m10 - m01, m02 + m20, m12 + m21, q2[..., 3]], -1),
    ], -2)                                                            # (...,4,4) candidates, rows = wxyz
    sel = q2.argmax(-1)
    q = torch.gather(cand, -2, sel[..., None, None].expand(sel.shape + (1, 4)))[..., 0, :]
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class SplatToFile3D(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SplatToFile3D",
            display_name="Create 3D File (from Splat)",
            search_aliases=["gaussian to ply", "splat to file", "export gaussian"],
            category="3d/splat",
            description="Serialize a gaussian splat to a File3D object for Save / Preview 3D nodes. "
                        "Supports one item per batch only.",
            inputs=[
                IO.Splat.Input("splat"),
                IO.Combo.Input("format", options=["ply", "ksplat", "spz"],  # TODO: add "splat" when we have a writer for it
                               tooltip="ply: standard 3D Gaussian Splat with full spherical harmonics. "
                                       "ksplat: mkkellogg SplatBuffer (level 0, uncompressed), base color only "
                                       "spz: Niantic gzip-compressed (~10x smaller), base color only "
                                       ),
            ],
            outputs=[IO.File3DSplatAny.Output(display_name="model_3d")],
        )

    @classmethod
    def execute(cls, splat, format="ply") -> IO.NodeOutput:
        if splat.positions.shape[0] > 1:
            logging.warning("SplatToFile3D supports one item per batch only. Got %d; using first.", splat.positions.shape[0])
        end = _real_len(splat, 0)
        writer = {"ksplat": _gaussian_ksplat_bytes, "spz": _gaussian_spz_bytes}.get(format, _gaussian_ply_bytes)
        data = writer(splat.positions[0, :end], splat.scales[0, :end],
                      splat.rotations[0, :end], splat.opacities[0, :end], splat.sh[0, :end])
        return IO.NodeOutput(Types.File3D(BytesIO(data), file_format=format))


class File3DToSplat(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="File3DToSplat",
            display_name="Get Splat",
            search_aliases=["load splat", "ply to splat", "import splat", "file to splat"],
            category="3d/splat",
            description="Parse a splat File3D into a gaussian splat. Inverse of Create 3D File (from Splat). "
                        "Supported format:  PLY, SPLAT, KSPLAT, SPZ. PLY carries full spherical harmonics, "
                        "the other formats are base color only. Format is auto-detected from the file contents.",
            inputs=[
                IO.MultiType.Input(
                    IO.File3DAny.Input("model_3d"),
                    types=[IO.File3DSplatAny, IO.File3DPLY, IO.File3DSPLAT, IO.File3DKSPLAT, IO.File3DSPZ],
                    tooltip="A gaussian splat 3D file",
                ),
            ],
            outputs=[IO.Splat.Output(display_name="splat")],
        )

    @classmethod
    def execute(cls, model_3d: Types.File3D) -> IO.NodeOutput:
        data = model_3d.get_bytes()
        fmt = (model_3d.format or "").lower()
        parser = _GAUSSIAN_PARSERS.get(fmt) or _GAUSSIAN_PARSERS[_detect_splat_format(data)]
        xyz, scale, rot, opacity, sh = parser(data)

        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
        splat = Types.SPLAT(
            t(xyz)[None],                              # (1, N, 3)
            t(scale)[None],                            # (1, N, 3) linear
            t(rot)[None],                              # (1, N, 4) wxyz
            t(opacity).reshape(1, -1, 1),              # (1, N, 1)
            t(sh)[None],                               # (1, N, K, 3)
        )
        return IO.NodeOutput(splat)


def _view_matrix_t(yaw_deg, pitch_deg, device):
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    cy, sy, cp, sp = math.cos(y), math.sin(y), math.cos(p), math.sin(p)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], device=device)
    Rx = torch.tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], device=device)
    return Rx @ Ry


def _camera_basis(camera_info, dev):
    # Look-at basis in the splat frame, named by their projection rows: right = image +x, up = image +y
    # (down, since yflip=1), fwd = view/depth axis (eye -> scene). Load3D is three.js (right-handed, Y-up,
    # camera looks down -Z); the splat is 3DGS (Y-down, Z-forward). World -> splat is a 180 deg rotation
    # about X: (x, y, z) -> (x, -y, -z) (det +1, no mirror, no axis swap).
    pos, tgt = camera_info.get("position", {}), camera_info.get("target", {})
    m = lambda d: torch.tensor([float(d.get("x", 0.0)), -float(d.get("y", 0.0)), -float(d.get("z", 0.0))], device=dev)
    eye, target = m(pos), m(tgt)
    mv = lambda v: torch.stack([v[0], -v[1], -v[2]])             # same world->splat map, for direction vectors
    n = lambda v: v / v.norm().clamp_min(1e-8)
    q = camera_info.get("quaternion")
    if q:                                                        # exact camera world rotation (incl. roll)
        qwxyz = torch.tensor([float(q.get("w", 1.0)), float(q.get("x", 0.0)),
                              float(q.get("y", 0.0)), float(q.get("z", 0.0))], device=dev)
        R = _quat_to_mat(qwxyz[None])[0]                         # columns = camera world axes; looks down local -Z
        right = n(mv(R[:, 0]))                                   # camera +X -> image right
        up = n(mv(-R[:, 1]))                                     # camera +Y is image up; image-down row is its negative
        fwd = n(mv(-R[:, 2]))                                    # camera looks down local -Z -> view direction
        return eye, target, right, up, fwd
    fwd = n(target - eye)                                        # no quaternion: orbit-consistent, roll-free
    yaw = math.degrees(math.atan2(-float(fwd[0]), float(fwd[2])))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(fwd[1])))))
    W = _view_matrix_t(yaw, pitch, dev)
    return eye, target, W[0], W[1], W[2]


def _lookat_quat_wxyz(position, target, dev):
    # three.js lookAt in world frame: camera local +Z = (eye - target), up = world +Y. Returns wxyz.
    z = position - target
    z = z / z.norm().clamp_min(1e-8)
    up0 = torch.tensor([0.0, 1.0, 0.0], device=dev)
    if z.dot(up0).abs() > 0.999:                                # looking straight up/down
        up0 = torch.tensor([0.0, 0.0, 1.0], device=dev)
    x = torch.linalg.cross(up0, z)
    x = x / x.norm().clamp_min(1e-8)
    y = torch.linalg.cross(z, x)
    R = torch.stack([x, y, z], dim=1)                           # columns = camera world axes
    return _mat_to_quat(R[None])[0]


def _lookat_camera_info(position, target, fov, dev, zoom=1.0, camera_type="perspective", roll=0.0):
    # Build a camera_info from a world-space (right-handed, Y-up) eye + look-at target; up = world +Y.
    pos = torch.as_tensor(position, dtype=torch.float32, device=dev)
    tgt = torch.as_tensor(target, dtype=torch.float32, device=dev)
    q = _lookat_quat_wxyz(pos, tgt, dev)
    if roll:                                                    # roll about the view axis (camera local Z)
        a = math.radians(roll)
        qz = torch.tensor([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)], device=dev)
        q = _quat_mul(q[None], qz[None])[0]
    xyz = lambda v: {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
    return {"position": xyz(pos), "target": xyz(tgt),
            "quaternion": {"x": float(q[1]), "y": float(q[2]), "z": float(q[3]), "w": float(q[0])},
            "fov": float(fov), "cameraType": str(camera_type), "zoom": float(zoom)}


def _quat_camera_info(position, quat_xyzw, fov, dev, zoom=1.0, camera_type="perspective"):
    # camera_info from an explicit world position + camera-rotation quaternion (three.js: looks down local -Z).
    pos = torch.as_tensor(position, dtype=torch.float32, device=dev)
    qx, qy, qz, qw = (float(c) for c in quat_xyzw)
    qwxyz = torch.tensor([qw, qx, qy, qz], dtype=torch.float32, device=dev)
    qwxyz = qwxyz / qwxyz.norm().clamp_min(1e-8)
    R = _quat_to_mat(qwxyz[None])[0]
    tgt = pos - R[:, 2]                                         # look one unit down local -Z
    xyz = lambda v: {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
    return {"position": xyz(pos), "target": xyz(tgt),
            "quaternion": {"x": float(qwxyz[1]), "y": float(qwxyz[2]), "z": float(qwxyz[3]), "w": float(qwxyz[0])},
            "fov": float(fov), "cameraType": str(camera_type), "zoom": float(zoom)}


def _orbit_camera_info(yaw, pitch, distance, fov, pivot_splat, dev):
    # Orbit helper for RenderSplat's default camera: yaw/pitch about `pivot_splat` (splat frame) at `distance`.
    # World<->splat is the (x,-y,-z) map, so _camera_basis recovers exactly _view_matrix_t(yaw, pitch).
    y, p = math.radians(yaw), math.radians(pitch)
    cy, sy, cp, sp = math.cos(y), math.sin(y), math.cos(p), math.sin(p)
    fwd_splat = torch.tensor([-cp * sy, sp, cp * cy], device=dev)    # == _view_matrix_t(yaw, pitch)[2]
    m = lambda v: torch.stack([v[0], -v[1], -v[2]])                  # splat<->world (its own inverse)
    return _lookat_camera_info(m(pivot_splat - distance * fwd_splat), m(pivot_splat), fov, dev)


def _orbit_camera_info_yaw(camera_info, angle_deg, dev):
    # Turntable: rigidly rotate a camera_info about world +Y around its target by angle_deg. Returns a new dict.
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    v = lambda d: torch.tensor([float(d.get("x", 0.0)), float(d.get("y", 0.0)), float(d.get("z", 0.0))], device=dev)
    pos, tgt = v(camera_info.get("position", {})), v(camera_info.get("target", {}))
    Ry = torch.tensor([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], device=dev)
    new_pos = tgt + Ry @ (pos - tgt)
    q = camera_info.get("quaternion") or {}
    qcur = torch.tensor([float(q.get("w", 1.0)), float(q.get("x", 0.0)),
                         float(q.get("y", 0.0)), float(q.get("z", 0.0))], device=dev)
    qy = torch.tensor([math.cos(a / 2), 0.0, math.sin(a / 2), 0.0], device=dev)   # world +Y rotation
    qn = _quat_mul(qy[None], qcur[None])[0]
    xyz = lambda t: {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])}
    return {**camera_info, "position": xyz(new_pos),
            "quaternion": {"x": float(qn[1]), "y": float(qn[2]), "z": float(qn[3]), "w": float(qn[0])}}


def _gauss_blur(x, sigma, dev):
    # Separable Gaussian blur of (1, C, H, W). Used to denoise the screen-space normal map.
    r = max(1, int(round(3 * sigma)))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=dev, dtype=torch.float32) / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    x = torch.nn.functional.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), padding=(0, r), groups=c)
    x = torch.nn.functional.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), padding=(r, 0), groups=c)
    return x


def _render_gaussian(xyz, rgb, opacity, scale, rot, width, height, splat_scale, bg, camera_info,
                     sharpen=1.0, headlight_shading=0.0, render_style="color"):
    # Perspective-correct anisotropic gaussian splat rasterizer. Each splat is weighted by its 3D Gaussian's
    # peak along each pixel's ray (AAA / Hahlbohm), composited front-to-back across depth slabs. `render_style`
    # selects the image: color / clay / depth / normal. Returns (image HxWx3, coverage mask HxW) on CPU.
    dev = comfy.model_management.get_torch_device()
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
    idev, idtype = comfy.model_management.intermediate_device(), comfy.model_management.intermediate_dtype()
    xyz, rgb, opacity = t(xyz), t(rgb).clamp(0, 1), t(opacity).reshape(-1)
    scale, rot = t(scale) * float(splat_scale), t(rot)
    do_linear = render_style == "color"  # colour blends in linear light, re-encoded at the end
    if do_linear:
        rgb = _srgb_to_linear(rgb)
    flat = width * height
    bg_t = t(bg)
    bg_comp = _srgb_to_linear(bg_t) if do_linear else bg_t  # background blended in the same space as the splats
    need_depth = render_style == "depth"
    need_normal = render_style in ("normal", "clay") or headlight_shading > 0

    def background_only():  # no splats to rasterize -> just the background + empty mask
        img = bg_t.expand(height, width, 3) if render_style == "color" else torch.zeros(height, width, 3, device=dev)
        return img.to(idev, idtype), torch.zeros(height, width, device=idev, dtype=idtype)

    if xyz.shape[0] == 0:  # empty input (e.g. all culled by opacity_threshold)
        return background_only()

    eye, target, right, up, fwd = _camera_basis(camera_info, dev)  # all camera state comes from camera_info
    W = torch.stack([right, up, fwd], 0)                           # rows = camera axes (world -> camera)
    cam = (xyz - eye) @ W.T
    fov = float(camera_info.get("fov", 0) or 0) or 35.0
    zoom = float(camera_info.get("zoom", 1.0) or 1.0)              # three.js digital zoom: scales the focal length
    is_ortho = str(camera_info.get("cameraType", "")).lower().startswith("ortho")
    xc, yc, zc = cam.unbind(-1)

    keep = zc > 1e-2
    xc, yc, zc, rgb, opacity, scale, rot = (a[keep] for a in (xc, yc, zc, rgb, opacity, scale, rot))
    if xc.shape[0] == 0:  # nothing in front of the camera -> background only
        return background_only()
    if render_style == "clay":
        rgb = torch.full_like(rgb, 0.75)  # neutral albedo -> shading shows pure geometry

    f = (min(width, height) / 2) / math.tan(math.radians(fov) / 2) * zoom  # fov over the smaller axis, x camera zoom
    cx0, cy0 = width / 2, height / 2

    # Camera-space 3D covariance per splat: Sigma = (W Rq) diag(scale^2) (W Rq)^T, plus a tiny relative
    # regularizer for a stable inverse (a pixel-size Mip low-pass would over-thicken flat surfels and blur).
    Mw = W[None] @ _quat_to_mat(rot)  # (N,3,3) world -> camera
    cam_cov = (Mw * scale.square()[:, None, :]) @ Mw.transpose(1, 2)
    cam_cov = cam_cov + (cam_cov.diagonal(dim1=-2, dim2=-1).mean(-1) * 1e-3)[:, None, None] * torch.eye(3, device=dev)

    # Perspective-correct weighting: peak of the 3D Gaussian along each pixel ray. Precompute Si, Si@mu, mu^T Si mu.
    mu = torch.stack([xc, yc, zc], -1)
    si = torch.linalg.inv(cam_cov)
    simu = (si @ mu[:, :, None])[:, :, 0]  # (N,3)
    musimu = (mu * simu).sum(-1)           # (N,)
    s00, s01, s02 = si[:, 0, 0], si[:, 0, 1], si[:, 0, 2]
    s11, s12, s22 = si[:, 1, 1], si[:, 1, 2], si[:, 2, 2]
    simu0, simu1, simu2 = simu.unbind(-1)
    if need_normal:  # surfel normal = thinnest axis, oriented toward camera
        nrm = Mw[torch.arange(Mw.shape[0], device=dev), :, scale.argmin(-1)]  # (N,3) camera-space normal
        nrm = nrm * torch.where(nrm[:, 2:3] > 0, -1.0, 1.0)                   # flip so nz <= 0 (faces camera)

    # Screen centre (exact) + footprint radius from the affine 2D projection (used only to size the kernel).
    # The image is +y-down, so the projection's y row is unflipped - it matches the splat frame's +Y.
    jm = torch.zeros(xc.shape[0], 2, 3, device=dev)
    if is_ortho:                                              # parallel projection: screen = s * (xc, yc)
        s = f / float((target - eye).norm().clamp_min(1e-6))  # pixels per world unit at the target plane
        cx, cy = cx0 + s * xc, cy0 + s * yc
        jm[:, 0, 0] = s
        jm[:, 1, 1] = s
    else:  # perspective: screen = f * (xc, yc) / zc
        invz = 1.0 / zc
        cx, cy = cx0 + f * xc * invz, cy0 + f * yc * invz
        jm[:, 0, 0], jm[:, 0, 2] = f * invz, -f * xc * invz.square()
        jm[:, 1, 1], jm[:, 1, 2] = f * invz, -f * yc * invz.square()
    cov2 = jm @ cam_cov @ jm.transpose(1, 2)
    a, b, c = cov2[:, 0, 0], cov2[:, 0, 1], cov2[:, 1, 1]
    max_eig = (a + c) * 0.5 + (((a - c) * 0.5).square() + b * b).clamp_min(0).sqrt()
    radius = 3.0 * max_eig.clamp_min(1e-8).sqrt()
    K = int(min(max(24, min(width, height) // 16), max(2, math.ceil(_quantile(radius, 0.995).item()))))

    # Per-splat kernel size: bucket splats by radius into a coarse ladder of window sizes (global K stays the cap) so
    # small splats (the bulk of it) use a small window.
    levels = [L for L in (16, 64, 256) if L < K] + [K]
    levels_t = torch.tensor(levels, device=dev, dtype=torch.float32)
    grids = []
    for L in levels:
        rng = torch.arange(-L, L + 1, device=dev, dtype=torch.float32)
        gy, gx = torch.meshgrid(rng, rng, indexing="ij")
        grids.append((gx.reshape(-1), gy.reshape(-1)))
    blevel = torch.bucketize(radius * (4.0 / 3.0), levels_t).clamp_(max=len(levels) - 1)  # window >= ~4 sigma

    n = zc.shape[0]
    ns = int(min(256, max(1, n // 1000)))                      # depth slabs: 1 per ~1000 splats, capped
    nl = len(levels)
    order = torch.argsort(zc)                                  # front (small zc) -> back -> defines the slabs
    bounds = torch.linspace(0, n, ns + 1, device=dev).round().long()
    rank = torch.empty(n, dtype=torch.long, device=dev)
    rank[order] = torch.arange(n, device=dev)                  # depth rank of each splat
    slab_id = (torch.searchsorted(bounds, rank, right=True) - 1).clamp_(0, ns - 1)
    key = slab_id * nl + blevel                                # group by slab, then kernel level (order-free within)
    order = torch.argsort(key)
    key = key[order]

    cxr, cyr = cx[order].round(), cy[order].round()
    s00, s01, s02 = s00[order], s01[order], s02[order]
    s11, s12, s22 = s11[order], s12[order], s22[order]
    s01b, s02b, s12b = s01 * 2, s02 * 2, s12 * 2               # doubled cross terms for the fused quadratic forms
    simu0, simu1, simu2, musimu = simu0[order], simu1[order], simu2[order], musimu[order]
    opacity, rgb = opacity[order], rgb[order]
    zc_o = zc[order] if need_depth else None
    nrm_o = nrm[order] if need_normal else None
    mux_o, muy_o, muz_o = (xc[order], yc[order], zc[order]) if is_ortho else (None, None, None)

    # Pack the per-splat scalars into one tensor so each chunk slices once
    common = [cxr, cyr, s00, s11, s22, s01b, s02b, s12b, opacity]
    pstack = torch.stack(common + ([s02, s12, mux_o, muy_o, muz_o] if is_ortho else [simu0, simu1, simu2, musimu]))

    # Precompute the (slab, level) run table on-GPU and pull it to the CPU once
    starts = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), (key[1:] != key[:-1]).nonzero().flatten() + 1])
    ks = key[starts]
    run_lo = starts.tolist() + [n]
    run_lev = (ks % nl).tolist()
    run_slab = torch.div(ks, nl, rounding_mode="floor").tolist()
    slab_runs = [[] for _ in range(ns)]
    for r in range(len(run_lev)):
        slab_runs[run_slab[r]].append((run_lo[r], run_lo[r + 1], run_lev[r]))

    def splat(lo, hi, ox, oy):  # -> pixel idx (m,M), alpha (m,M); weight = 3D Gaussian peak along each pixel's ray
        cols = pstack[:, lo:hi, None].unbind(0)
        cxr_, cyr_, a00, a11, a22, b01, b02, b12, opa = cols[:9]   # a* = Si components; b* = 2 * cross terms
        px = cxr_ + ox[None, :]
        py = cyr_ + oy[None, :]
        valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        if is_ortho:  # parallel ray (0,0,1) from screen point (X, Y, 0); rz constant per splat
            c02, c12, mx, my, mz = cols[9:]
            rx = (px - cx0) / s - mx
            ry = (py - cy0) / s - my
            rz = -mz
            a22rz = a22 * rz
            inx = torch.addcmul(b02 * rz, a00, rx).addcmul_(b01, ry)          # a00 rx + b01 ry + b02 rz
            rSr = torch.addcmul(a22rz * rz, rx, inx).addcmul_(ry, torch.addcmul(b12 * rz, a11, ry))
            dsr = torch.addcmul(a22rz, c02, rx).addcmul_(c12, ry)
            q = torch.addcdiv(rSr, dsr * dsr, a22.clamp_min(1e-12), value=-1).clamp_min_(0)
        else:  # perspective ray (dx,dy,1) through the camera origin
            su0, su1, su2, mus = cols[9:]
            dx, dy = (px - cx0) / f, (py - cy0) / f
            dsid = torch.addcmul(a22, dx, torch.addcmul(b02, a00, dx))        # a22 + dx*(a00 dx + b02)
            dsid = dsid.addcmul_(dy, torch.addcmul(b12, a11, dy))             # + dy*(a11 dy + b12)
            dsid = dsid.addcmul_(b01 * dx, dy)                               # + (2 s01) dx dy
            dsimu = torch.addcmul(su2, dx, su0).addcmul_(dy, su1)
            q = torch.addcdiv(mus, dsimu * dsimu, dsid.clamp_min(1e-12), value=-1).clamp_min_(0)
        alpha = (opa * torch.exp(-0.5 * q) * valid).clamp_(0, 0.999)
        idx = py.long().clamp(0, height - 1) * width + px.long().clamp(0, width - 1)
        return idx, alpha

    # Front-to-back compositing over the depth slabs set up above. Within a slab the accumulation is a pure
    # sum (order-independent), so splats are grouped by kernel level and each level uses its own tight window.
    sharp = sharpen != 1.0  # winner-take-more colour blend: dominant splat shows more
    cacc = torch.zeros((flat, 3), device=dev)
    trans = torch.ones((flat,), device=dev)
    a_buf = torch.zeros((flat,), device=dev)                            # sum alpha -> colour/depth/normal weight (alpha-weighted mean)
    tau_buf = torch.zeros((flat,), device=dev)                          # sum -ln(1-alpha) -> slab opacity = 1-prod(1-alpha)
    crgb = torch.zeros((flat, 3), device=dev)                           # sum alpha^p * rgb -> slab colour
    wbuf = torch.zeros((flat,), device=dev) if sharp else None          # sum alpha^p -> colour normalizer (sharp only)
    dacc = torch.zeros((flat,), device=dev) if need_depth else None     # front-weighted depth
    nacc = torch.zeros((flat, 3), device=dev) if need_normal else None  # front-weighted camera-space normal
    zslab = torch.zeros((flat,), device=dev) if need_depth else None
    nslab = torch.zeros((flat, 3), device=dev) if need_normal else None
    stale = 0  # consecutive fully-occluded slabs -> early-out
    for si in range(ns):
        runs = slab_runs[si]
        if not runs:
            continue
        a_buf.zero_()
        tau_buf.zero_()
        crgb.zero_()
        if sharp:
            wbuf.zero_()
        if need_depth:
            zslab.zero_()
        if need_normal:
            nslab.zero_()
        for r_lo, r_hi, li in runs:            # contiguous same-kernel-level runs in this slab
            ox, oy = grids[li]
            ch = max(2048, 10_000_000 // ox.shape[0])          # splats/chunk, bounded by this level's kernel size
            for lo in range(r_lo, r_hi, ch):
                hi = min(lo + ch, r_hi)
                idx, alpha = splat(lo, hi, ox, oy)
                idx, af = idx.reshape(-1), alpha.reshape(-1)
                a_buf.index_add_(0, idx, af)
                tau_buf.index_add_(0, idx, (-torch.log1p(-alpha)).reshape(-1))      # -ln(1-alpha), correct opacity merge
                apw = alpha.pow(sharpen) if sharp else alpha                        # bias colour toward the highest-alpha splat
                crgb.index_add_(0, idx, (apw[:, :, None] * rgb[lo:hi, None, :]).reshape(-1, 3))
                if sharp:
                    wbuf.index_add_(0, idx, apw.reshape(-1))
                if need_depth:
                    zslab.index_add_(0, idx, (alpha * zc_o[lo:hi, None]).reshape(-1))
                if need_normal:
                    nslab.index_add_(0, idx, (alpha[:, :, None] * nrm_o[lo:hi, None, :]).reshape(-1, 3))
        slab_a = 1 - torch.exp(-tau_buf)   # 1 - prod(1-alpha): true opacity of the slab's splats
        front = trans * slab_a
        denom = wbuf if sharp else a_buf
        cacc.addcmul_(front[:, None], crgb / denom.clamp_min(1e-8)[:, None])  # cacc += front * (crgb/denom)
        if need_depth or need_normal:
            ainv = a_buf.clamp_min(1e-8)   # alpha-weighted-mean normalizer (depth/normal only)
            if need_depth:
                dacc.addcmul_(front, zslab / ainv)
            if need_normal:
                nacc.addcmul_(front[:, None], nslab / ainv[:, None])
        trans.mul_(1 - slab_a)
        if si % 8 == 7:                    # checkpoint every 8 slabs (a per-slab GPU sync would cost more)
            if float(front.max()) < 1e-3:  # this checkpoint slab is fully occluded by what is in front
                stale += 1
                if stale >= 2:             # two occluded checkpoints running -> the rest are too -> stop
                    break
            else:
                stale = 0

    cov = 1 - trans
    covg = cov.reshape(height, width)
    covm = covg > 0.5 if render_style in ("depth", "normal") else None  # silhouette mask (depth/normal styles only)
    depth_map = (dacc / cov.clamp_min(1e-6)).reshape(height, width) if need_depth else None
    nrm_map = None
    if need_normal:
        # Per-splat surfel normals are jittery, so do a masked blur
        nb = nacc.reshape(height, width, 3).permute(2, 0, 1)[None]
        cb = cov.reshape(1, 1, height, width)
        nb, cb = _gauss_blur(nb, 1.2, dev), _gauss_blur(cb, 1.2, dev)
        normal = (nb / cb.clamp_min(1e-6))[0].permute(1, 2, 0)
        nrm_map = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    if render_style == "depth":  # near = bright, far = dark, 0 off-object
        d = torch.zeros(height, width, device=dev)
        if bool(covm.any()):
            lo, hi = depth_map[covm].min(), depth_map[covm].max()
            d = torch.where(covm, ((hi - depth_map) / (hi - lo).clamp_min(1e-6)).clamp(0, 1), d)
        img = d[:, :, None].expand(height, width, 3)
    elif render_style == "normal":  # OpenGL normal map: +X right, +Y up, +Z to viewer
        enc = (nrm_map * t([1.0, -1.0, -1.0]) * 0.5 + 0.5).clamp(0, 1)
        img = enc * covm[:, :, None]
    else:  # color / clay
        img = cacc.reshape(height, width, 3)
        if render_style == "clay":  # studio key light + ambient -> sculpted matte look
            kl = t([-0.4, -0.7, -0.6])  # key from screen upper-left, angled toward the viewer
            kl = kl / kl.norm()
            hl = (0.5 * (nrm_map * kl).sum(-1) + 0.5).clamp(0, 1)  # half-Lambert: soft terminator, no harsh dark side
            img = img * (0.35 + 0.65 * hl * hl)[:, :, None]        # ambient floor + diffuse key
        elif headlight_shading > 0:  # camera headlight: darken faces turned from view
            k = float(headlight_shading)
            ndotl = (-nrm_map[:, :, 2]).clamp(0, 1)
            img = img * (1 - 0.6 * k + 0.6 * k * ndotl)[:, :, None]
        img = img.addcmul_(trans.reshape(height, width, 1), bg_comp)
        if do_linear:  # back to display space after linear compositing
            img = _linear_to_srgb(img)
    return img.clamp(0, 1).to(idev, idtype), covg.clamp(0, 1).to(idev, idtype)


class RenderSplat(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RenderSplat",
            display_name="Render Splat",
            search_aliases=["splat to image", "render splat", "gaussian turntable"],
            category="3d/splat",
            description="Render a gaussian splat as an image with an anisotropic EWA rasterizer (oriented "
                        "elliptical splats, antialiased, depth-sorted front-to-back). The camera comes from a "
                        "camera_info input (Load / Preview 3D, or a Create Camera Info node); leave it empty to "
                        "auto-frame the splat. Set frames greater than 1 for a turntable batch of images to feed a Video node.",
            inputs=[
                IO.Splat.Input("splat"),
                IO.Int.Input("width", default=1024, min=64, max=2048, step=8),
                IO.Int.Input("height", default=1024, min=64, max=2048, step=8),
                IO.Int.Input("frames", default=1, min=-240, max=240,
                             tooltip="-1, 0, 1 = single still image; >1 = turntable, the camera orbits over a full "
                                     "360 turn (works with any camera_info). Negative value orbits the other way."),
                IO.Float.Input("splat_scale", default=1.0, min=0.1, max=5.0, step=0.05, advanced=True,
                               tooltip="Multiplier on each splat's projected footprint (lower = crisper points, "
                                       "higher = softer/fuller surface)."),
                IO.Float.Input("sharpen", default=2.0, min=1.0, max=8.0, step=0.5,
                               tooltip="Sharpen overlapping splats: 1.0 = physically-correct blend; higher biases "
                                       "each pixel toward its dominant (nearest) splat for crisper texture, without "
                                       "shrinking splats or opening gaps. Non-physical above 1."),
                IO.Float.Input("headlight_shading", default=0.0, min=0.0, max=3.0, step=0.05, advanced=True,
                               tooltip="Diffuse shading from a light at the camera (headlight), using the splat surfel "
                                       "normals: darkens surfaces that turn away from view to reveal form/curvature. "
                                       "0 = flat albedo, 1 = strongest shading."),
                IO.Float.Input("opacity_threshold", default=0.0, min=0.0, max=1.0, step=0.01, advanced=True,
                               tooltip="Cull gaussians with opacity below this (removes faint floaters)."),
                IO.Combo.Input("render_style", options=["color", "clay", "depth", "normal"],
                               tooltip="What the image output shows: color, clay (neutral-albedo shaded), "
                                       "depth (near=bright), normal (OpenGL normal map)."),
                IO.Color.Input("background", default="#000000"),
                IO.Image.Input("bg_image", optional=True,
                               tooltip="Optional background plate composited behind the splat (overrides the solid "
                                       "background colour). Resized to the render size; a batch is used per frame, "
                                       "a single image for all. color/clay only."),
                IO.Load3DCamera.Input("camera_info", optional=True,
                                      tooltip="Camera to render from - a Load3D / Preview3D camera or a Create Camera "
                                              "Info node. If empty, the splat is auto-framed from a default 3/4 view."),
            ],
            outputs=[IO.Image.Output(display_name="image"), IO.Mask.Output(display_name="mask")],
        )

    @classmethod
    def execute(cls, splat, width, height, frames, splat_scale, sharpen, headlight_shading,
                opacity_threshold, background, render_style, camera_info=None, bg_image=None) -> IO.NodeOutput:
        bg = _hex_to_rgb(background)
        bg_imgs = None
        if bg_image is not None:  # resize the plate(s) to the render size: (B,H,W,3)
            bi = bg_image[... , :3].movedim(-1, 1)  # (B,3,H,W)
            bi = comfy.utils.common_upscale(bi, width, height, "bicubic", "disabled")
            bg_imgs = bi.movedim(1, -1).clamp(0, 1)
        n_frames = abs(int(frames)) or 1         # magnitude = frame count (0 -> single still)
        orbit_dir = -1.0 if frames < 0 else 1.0  # sign = orbit direction
        imgs, masks = [], []
        device = comfy.model_management.get_torch_device()
        total = splat.positions.shape[0] * n_frames
        pbar = comfy.utils.ProgressBar(total) if total > 1 else None
        k = 0
        for i in range(splat.positions.shape[0]):
            xyz, rgb, opacity, scale, rot = _gaussian_item(splat, i, device)
            if opacity_threshold > 0:
                keep = opacity >= opacity_threshold
                xyz, rgb, opacity, scale, rot = xyz[keep], rgb[keep], opacity[keep], scale[keep], rot[keep]
            base_cam = camera_info
            if base_cam is None:  # no camera -> default 3/4 view, auto-framed on the splat
                center = xyz.mean(0) if xyz.shape[0] else torch.zeros(3, device=device)
                extent = (_quantile((xyz - center).norm(dim=-1), 0.99).clamp_min(1e-4) if xyz.shape[0]
                          else torch.tensor(1.0, device=device))
                dist = float(extent / (math.tan(math.radians(35.0) / 2) * 0.9))
                base_cam = _orbit_camera_info(35.0, 30.0, dist, 35.0, center, device)
            for fr in range(n_frames):
                cam_fr = (base_cam if n_frames == 1
                          else _orbit_camera_info_yaw(base_cam, orbit_dir * 360.0 * fr / n_frames, device))
                bg_k = bg_imgs[k % bg_imgs.shape[0]] if bg_imgs is not None else bg   # per-frame plate, or solid colour
                img, mask = _render_gaussian(xyz, rgb, opacity, scale, rot, width, height, splat_scale, bg_k, cam_fr,
                                             sharpen=sharpen, headlight_shading=headlight_shading,
                                             render_style=render_style)
                imgs.append(img)
                masks.append(mask)
                k += 1
                if pbar is not None:
                    pbar.update(1)
        return IO.NodeOutput(torch.stack(imgs), torch.stack(masks))


class CreateCameraInfo(IO.ComfyNode):  # TODO: move to better file
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CreateCameraInfo",
            display_name="Create Camera Info",
            search_aliases=["camera position", "make camera info", "orbit camera", "look at camera"],
            category="3d",
            description="Build a camera_info"
                        "Mode 'orbit' aims with yaw/pitch/distance around the target; "
                        "'look_at' places the camera at world position. Coordinates are the viewer's world space (right-handed,Y-up).",
            inputs=[
                IO.DynamicCombo.Input("mode", options=[
                    IO.DynamicCombo.Option("orbit", [
                        IO.Float.Input("yaw", default=35.0, min=-360.0, max=360.0, step=1.0),
                        IO.Float.Input("pitch", default=30.0, min=-89.0, max=89.0, step=1.0),
                        IO.Float.Input("distance", default=4.0, min=0.01, max=1000.0, step=0.01,
                                       tooltip="Camera distance from the target."),
                    ]),
                    IO.DynamicCombo.Option("look_at", [
                        IO.Float.Input("position_x", default=4.0, min=-1000.0, max=1000.0, step=0.01,
                                       tooltip="Camera position in world space (right-handed, Y-up)."),
                        IO.Float.Input("position_y", default=4.0, min=-1000.0, max=1000.0, step=0.01),
                        IO.Float.Input("position_z", default=4.0, min=-1000.0, max=1000.0, step=0.01),
                    ]),
                    IO.DynamicCombo.Option("quaternion", [
                        IO.Float.Input("position_x", default=4.0, min=-1000.0, max=1000.0, step=0.01,
                                       tooltip="Camera position in world space (right-handed, Y-up)."),
                        IO.Float.Input("position_y", default=4.0, min=-1000.0, max=1000.0, step=0.01),
                        IO.Float.Input("position_z", default=4.0, min=-1000.0, max=1000.0, step=0.01),
                        IO.Float.Input("quat_x", default=0.0, min=-1.0, max=1.0, step=0.001),
                        IO.Float.Input("quat_y", default=0.0, min=-1.0, max=1.0, step=0.001),
                        IO.Float.Input("quat_z", default=0.0, min=-1.0, max=1.0, step=0.001),
                        IO.Float.Input("quat_w", default=1.0, min=-1.0, max=1.0, step=0.001,
                                       tooltip="Camera world-rotation quaternion (three.js: looks down local -Z). Normalized for you."),
                    ]),
                ], tooltip="How to define the camera: orbit angles, an explicit position, or a position + quaternion."),
                IO.Float.Input("target_x", default=0.0, min=-1000.0, max=1000.0, step=0.01, advanced=True,
                               tooltip="Look-at point (orbit pivot / aim). In orbit mode, move it to pan/translate the "
                                       "whole camera. Ignored in quaternion mode. Defaults to the origin."),
                IO.Float.Input("target_y", default=0.0, min=-1000.0, max=1000.0, step=0.01, advanced=True),
                IO.Float.Input("target_z", default=0.0, min=-1000.0, max=1000.0, step=0.01, advanced=True),
                IO.Float.Input("roll", default=0.0, min=-180.0, max=180.0, step=1.0,
                               tooltip="Camera roll about the view axis, degrees."),
                IO.Float.Input("fov", default=35.0, min=1.0, max=120.0, step=1.0,
                               tooltip="Vertical field of view in degrees."),
                IO.Float.Input("zoom", default=1.0, min=0.01, max=100.0, step=0.01,
                               tooltip="Digital zoom (focal-length multiplier). >1 zooms in without moving the camera."),
                IO.Combo.Input("camera_type", options=["perspective", "orthographic"],
                               tooltip="Projection used by Render Splat: perspective (foreshortening) or orthographic (parallel)."),
            ],
            outputs=[IO.Load3DCamera.Output(display_name="camera_info")],
        )

    @classmethod
    def execute(cls, mode, target_x, target_y, target_z, roll, fov, zoom=1.0, camera_type="perspective") -> IO.NodeOutput:
        dev = comfy.model_management.get_torch_device()
        kind = mode["mode"]
        if kind == "quaternion":  # explicit world position + camera rotation
            position = [mode["position_x"], mode["position_y"], mode["position_z"]]
            quat = [mode["quat_x"], mode["quat_y"], mode["quat_z"], mode["quat_w"]]
            return IO.NodeOutput(_quat_camera_info(position, quat, fov, dev, zoom=zoom, camera_type=camera_type))
        target = [target_x, target_y, target_z]  # orbit pivot / aim; move it to pan the whole camera
        if kind == "orbit":  # yaw/pitch/distance about the target (world Y-up)
            y, p = math.radians(mode["yaw"]), math.radians(mode["pitch"])
            cy, sy, cp, sp = math.cos(y), math.sin(y), math.cos(p), math.sin(p)
            d = mode["distance"]
            position = [target_x + d * cp * sy, target_y + d * sp, target_z + d * cp * cy]
        else:  # look_at: explicit world-space camera position
            position = [mode["position_x"], mode["position_y"], mode["position_z"]]
        return IO.NodeOutput(_lookat_camera_info(position, target, fov, dev, zoom=zoom, camera_type=camera_type, roll=roll))


class TransformSplat(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TransformSplat",
            display_name="Transform Splat",
            search_aliases=["move splat", "rotate splat", "scale splat", "gaussian transform"],
            category="3d/splat",
            description="Translate, rotate, and scale a gaussian splat. "
                        "Non-uniform scale also reshapes every individual splat, slower process.",
            inputs=[
                IO.Splat.Input("splat"),
                IO.Float.Input("translate_x", default=0.0, min=-100.0, max=100.0, step=0.01),
                IO.Float.Input("translate_y", default=0.0, min=-100.0, max=100.0, step=0.01),
                IO.Float.Input("translate_z", default=0.0, min=-100.0, max=100.0, step=0.01),
                IO.Float.Input("rotate_x", default=0.0, min=-360.0, max=360.0, step=1.0),
                IO.Float.Input("rotate_y", default=0.0, min=-360.0, max=360.0, step=1.0),
                IO.Float.Input("rotate_z", default=0.0, min=-360.0, max=360.0, step=1.0),
                IO.Float.Input("scale_x", default=1.0, min=0.01, max=100.0, step=0.01),
                IO.Float.Input("scale_y", default=1.0, min=0.01, max=100.0, step=0.01),
                IO.Float.Input("scale_z", default=1.0, min=0.01, max=100.0, step=0.01),
            ],
            outputs=[IO.Splat.Output(display_name="splat")],
        )

    @classmethod
    def execute(cls, splat, translate_x, translate_y, translate_z,
                rotate_x, rotate_y, rotate_z, scale_x, scale_y, scale_z) -> IO.NodeOutput:
        pos = splat.positions
        dev, dt = pos.device, pos.dtype
        q_rot = _euler_to_quat(rotate_x, rotate_y, rotate_z).to(device=dev, dtype=dt)
        R = _quat_to_mat(q_rot[None])[0]                            # (3, 3) node rotation
        D = torch.tensor([scale_x, scale_y, scale_z], dtype=dt, device=dev)
        A = D[:, None] * R                                          # diag(D) @ R: per-axis scale after rotation
        t = torch.tensor([translate_x, translate_y, translate_z], dtype=dt, device=dev)

        positions = pos @ A.T + t                                   # rotate, scale per-axis, then translate
        if scale_x == scale_y == scale_z:                           # uniform: rotation/scale factor out cleanly
            scales = splat.scales * scale_x
            rotations = _quat_mul(q_rot.expand_as(splat.rotations), splat.rotations)
            rotations = rotations / rotations.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        else:                                                       # non-uniform: transform Sigma = A R s^2 R^T A^T, re-extract
            rg = _quat_to_mat(splat.rotations.reshape(-1, 4))       # (M,3,3) per-splat rotation
            s2 = splat.scales.reshape(-1, 3).square()
            cov = (rg * s2[:, None, :]) @ rg.transpose(-1, -2)      # Sigma
            cov = A @ cov @ A.T                                     # A Sigma A^T (A broadcast over splats)
            lam, V = torch.linalg.eigh(cov)                         # symmetric -> eigenvalues (asc), orthonormal axes
            V = V * torch.where(torch.linalg.det(V) < 0, -1.0, 1.0)[..., None, None]   # keep a proper rotation
            scales = lam.clamp_min(0).sqrt().reshape(splat.scales.shape)
            rotations = _mat_to_quat(V).reshape(splat.rotations.shape)
        out = Types.SPLAT(positions, scales, rotations, splat.opacities, splat.sh,
                             counts=getattr(splat, "counts", None))
        return IO.NodeOutput(out)


class GetSplatCount(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GetSplatCount",
            display_name="Get Splat Count",
            search_aliases=["splat count", "gaussian count", "number of splats", "splat info"],
            category="3d/splat",
            description="Returns the number of splats summed across the batch.",
            inputs=[IO.Splat.Input("splat")],
            outputs=[IO.Splat.Output(display_name="splat"),
                     IO.Int.Output(display_name="count"),
                     ],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, splat) -> IO.NodeOutput:
        count = sum(_real_len(splat, i) for i in range(splat.positions.shape[0]))
        if cls.hidden.unique_id:  # show the count inline on the node
            PromptServer.instance.send_progress_text(f"{count:,} splats", cls.hidden.unique_id)
        return IO.NodeOutput(splat, count)


def _pad_stack(items, n):
    # Stack a list of (Lᵢ, *tail) tensors into (B, n, *tail), zero-padding each row up to n.
    tail = items[0].shape[1:]
    out = items[0].new_zeros((len(items), n, *tail))
    for i, t in enumerate(items):
        out[i, :t.shape[0]] = t
    return out


def _merge_gaussians(gaussians: list) -> Types.SPLAT:
    # Concatenate SPLAT batches along the splat dimension (per item), padding SH to the highest degree.
    gs = [g for g in gaussians if g is not None]
    if not gs:
        raise ValueError("MergeSplat: no gaussians to merge")
    b = gs[0].positions.shape[0]
    for g in gs:
        if g.positions.shape[0] != b:
            raise ValueError(f"MergeSplat: batch size mismatch ({b} vs {g.positions.shape[0]}).")
    max_k = max(g.sh.shape[2] for g in gs)

    pos_b, scl_b, rot_b, op_b, sh_b, lengths = [], [], [], [], [], []
    for i in range(b):
        pos_i, scl_i, rot_i, op_i, sh_i = [], [], [], [], []
        for g in gs:
            end = _real_len(g, i)
            pos_i.append(g.positions[i, :end])
            scl_i.append(g.scales[i, :end])
            rot_i.append(g.rotations[i, :end])
            op_i.append(g.opacities[i, :end])
            sh = g.sh[i, :end]       # (end, K, 3)
            if sh.shape[1] < max_k:  # zero-pad lower-degree SH
                sh = torch.cat([sh, sh.new_zeros(sh.shape[0], max_k - sh.shape[1], sh.shape[2])], dim=1)
            sh_i.append(sh)
        pos_b.append(torch.cat(pos_i))
        scl_b.append(torch.cat(scl_i))
        rot_b.append(torch.cat(rot_i))
        op_b.append(torch.cat(op_i))
        sh_b.append(torch.cat(sh_i))
        lengths.append(pos_b[-1].shape[0])

    n = max(lengths)
    counts = None
    if len(set(lengths)) > 1:
        counts = torch.tensor(lengths, device=gs[0].positions.device, dtype=torch.int64)
    return Types.SPLAT(_pad_stack(pos_b, n), _pad_stack(scl_b, n), _pad_stack(rot_b, n),
                          _pad_stack(op_b, n), _pad_stack(sh_b, n), counts=counts)


class MergeSplat(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        # Autogrow: a splat0/splat1/... input list that grows a fresh slot as you connect splats.
        splats = IO.Autogrow.TemplatePrefix(IO.Splat.Input("splat"), prefix="splat", min=2, max=32)
        return IO.Schema(
            node_id="MergeSplat",
            display_name="Merge Splats",
            search_aliases=["union splat", "densify gaussian", "combine splat", "merge gaussian"],
            category="3d/splat",
            description="Concatenate any number of gaussian splats into one. Unioning several decodes of the same "
                        "latent at different seeds densifies the surface, this can improve surface quality when meshing.",
            inputs=[IO.Autogrow.Input("splats", template=splats)],
            outputs=[IO.Splat.Output(display_name="splat")],
        )

    @classmethod
    def execute(cls, splats: IO.Autogrow.Type) -> IO.NodeOutput:
        gs = [v for v in splats.values() if v is not None]
        if not gs:
            raise ValueError("MergeSplat: connect at least one splat.")
        return IO.NodeOutput(_merge_gaussians(gs))


def _inverse_covariance(scale, quat):
    # Per-splat Sigma^-1 = R diag(1/s^2) R^T. scale (N,3) linear std, quat (N,4) wxyz -> (N,3,3).
    q = quat / quat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=1).reshape(-1, 3, 3)
    inv_s2 = 1.0 / scale.clamp_min(1e-8) ** 2                       # (N, 3)
    return torch.einsum("nij,nj,nkj->nik", R, inv_s2, R)


def _splat_density(xyz, opacity, scale, quat, rgb, res, kernel, device, color_sharpen=1.0, chunk=4096, progress=None,
                   col_dtype=torch.float16):
    # Splat each gaussian as its oriented-covariance disk (3-sigma, opacity-weighted) into a density grid,
    # plus a colour volume. Each gaussian uses a voxel window sized to its OWN 3-sigma (capped at `kernel`).
    # Colour is weighted by w^color_sharpen: >1 biases each voxel toward its dominant gaussian (crisper
    # texture). Returns (density, colour numerator, colour normaliser, origin, voxel).
    pad = 4.0 * scale.median()
    lo = xyz.amin(0) - pad
    hi = xyz.amax(0) + pad
    voxel = ((hi - lo).max() / res).clamp_min(1e-8)
    dx, dy, dz = (torch.ceil((hi - lo) / voxel).long() + 1).tolist()

    sinv = _inverse_covariance(scale, quat)
    kreq = torch.ceil(3.0 * scale.amax(-1) / voxel).long().clamp(1, int(kernel))  # per-gaussian half-width
    sharp = color_sharpen != 1.0
    vol = torch.zeros(dx * dy * dz, device=device)                                          # Sum(w) density (surface)
    colvol = torch.zeros(dx * dy * dz, 3, device=device, dtype=col_dtype)                   # Sum(w^p * rgb) colour numerator
    wcol = torch.zeros(dx * dy * dz, device=device, dtype=col_dtype) if sharp else None     # Sum(w^p) normaliser (p>1)
    n, done = xyz.shape[0], 0
    for k in range(1, int(kernel) + 1):
        sel = (kreq == k).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        rng = torch.arange(-k, k + 1, device=device, dtype=torch.float32)
        off = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)  # (M, 3)
        for st in range(0, sel.numel(), chunk):
            gi = sel[st:st + chunk]
            cc = xyz[gi]
            idx = ((cc - lo) / voxel).round()[:, None, :] + off[None]      # (b, M, 3) voxel coords
            d = (lo + idx * voxel) - cc[:, None, :]                        # world offset to voxel center
            quad = torch.einsum("bmi,bij,bmj->bm", d, sinv[gi], d)
            wgt = opacity[gi, None] * torch.exp(-0.5 * quad)
            wgt = torch.where(quad < 9.0, wgt, torch.zeros_like(wgt))      # clip beyond 3 sigma
            ii = idx.long()
            ix = ii[..., 0].clamp(0, dx - 1)
            iy = ii[..., 1].clamp(0, dy - 1)
            iz = ii[..., 2].clamp(0, dz - 1)
            flat = (ix * (dy * dz) + iy * dz + iz).reshape(-1)
            vol.index_add_(0, flat, wgt.reshape(-1))
            wp = wgt.pow(color_sharpen) if sharp else wgt                  # winner-take-more colour weight
            colvol.index_add_(0, flat, (wp[..., None] * rgb[gi, None, :]).reshape(-1, 3).to(col_dtype))
            if sharp:
                wcol.index_add_(0, flat, wp.reshape(-1).to(col_dtype))
            done += gi.numel()
            if progress is not None:
                progress(min(1.0, done / max(1, n)))
    colnorm = (wcol if sharp else vol).reshape(dx, dy, dz)                 # p==1 -> Sum(w) == density
    return vol.reshape(dx, dy, dz), colvol.reshape(dx, dy, dz, 3), colnorm, lo.cpu().numpy(), float(voxel)


def _connected_components_gpu(faces, nv):
    # FastSV connected components: grandparent hooking + shortcutting, ~O(log nv) iterations.
    # Returns per-vertex component labels (min node id, not densified).
    a = torch.cat([faces[:, 0], faces[:, 1]])                              # 2F edge endpoints: (v0,v1),(v1,v2)
    b = torch.cat([faces[:, 1], faces[:, 2]])
    f = torch.arange(nv, device=faces.device)
    while True:
        gp = f[f]                                                          # grandparent
        ga, gb = gp[a], gp[b]
        new = f.clone()
        new.scatter_reduce_(0, f[a], gb, "amin", include_self=True)        # stochastic hooking onto roots
        new.scatter_reduce_(0, f[b], ga, "amin", include_self=True)
        new.scatter_reduce_(0, a, gb, "amin", include_self=True)           # aggressive hooking, both directions
        new.scatter_reduce_(0, b, ga, "amin", include_self=True)
        new = new[new]                                                     # shortcut (path compression)
        if torch.equal(new, f):
            return f
        f = new


def _clean_components_gpu(verts, faces, min_verts, device):
    # GPU port of _clean_components: FastSV components + scatter reductions. Byte-identical to the numpy path
    vt = torch.as_tensor(verts, device=device)
    ft = torch.as_tensor(faces, device=device)
    nv = vt.shape[0]
    _, label = torch.unique(_connected_components_gpu(ft, nv), return_inverse=True)  # dense 0..ncomp-1
    ncomp = int(label.max()) + 1
    flabel = label[ft[:, 0]]                                                         # component id per face
    keep = torch.bincount(label, minlength=ncomp) >= min_verts                       # per-component vertex-count gate
    if int(keep.sum()) > 1:
        fcount = torch.bincount(flabel, minlength=ncomp)
        largest = int(torch.where(keep, fcount, fcount.new_tensor(-1)).argmax())
        v0, v1, v2 = vt[ft[:, 0]], vt[ft[:, 1]], vt[ft[:, 2]]
        cvol = torch.zeros(ncomp, device=device).scatter_add_(0, flabel, (v0 * torch.linalg.cross(v1, v2)).sum(-1))
        idx3 = label[:, None].expand(-1, 3)  # per-component vertex bbox
        cmin = torch.full((ncomp, 3), float("inf"), device=device).scatter_reduce_(0, idx3, vt, "amin", include_self=True)
        cmax = torch.full((ncomp, 3), float("-inf"), device=device).scatter_reduce_(0, idx3, vt, "amax", include_self=True)
        tol = 1e-4 * (cmax[largest] - cmin[largest]).max()
        enclosed = (cmin >= cmin[largest] - tol).all(1) & (cmax <= cmax[largest] + tol).all(1)
        inner = enclosed & (torch.sign(cvol) != torch.sign(cvol[largest])) & (torch.arange(ncomp, device=device) != largest)
        keep &= ~inner
    faces_k = ft[keep[flabel]]
    if faces_k.shape[0] == 0:
        return verts[:0], faces[:0]
    used = torch.unique(faces_k)  # sorted, matches np.unique
    remap = torch.full((nv,), -1, dtype=torch.int64, device=device)
    remap[used] = torch.arange(used.shape[0], device=device)
    return vt[used].cpu().numpy(), remap[faces_k].cpu().numpy()


def _clean_components(verts, faces, min_verts, device=None):
    # Drop floaters (components with < min_verts vertices) and inner shells - the surfel shell density
    # extracts a double wall (outer + inner cavity surface). GPU path (FastSV CC + scatter reductions, ~13x
    # faster) when an accelerator has headroom; else numpy/scipy. Both produce byte-identical output.
    if device is not None and not comfy.model_management.is_device_cpu(device) and \
            comfy.model_management.get_free_memory(device) > 10 * faces.size * 8:   # peak ~8.4x faces bytes
        return _clean_components_gpu(verts, faces, min_verts, device)
    nv = len(verts)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], 0)
    ncomp, label = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv)), directed=False)
    flabel = label[faces[:, 0]]                              # component id per face
    keep = np.bincount(label, minlength=ncomp) >= min_verts  # per-component vertex-count gate
    if keep.sum() > 1:
        fcount = np.bincount(flabel, minlength=ncomp)
        largest = np.where(keep, fcount, -1).argmax()
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        cvol = np.bincount(flabel, weights=np.einsum("ij,ij->i", v0, np.cross(v1, v2)), minlength=ncomp)  # 6*signed vol
        cidx = np.arange(ncomp)                                  # per-component vertex bbox via ndimage (~6x faster than ufunc.at)
        cmin = np.stack([_ndi_minimum(verts[:, a], label, cidx) for a in range(3)], 1)
        cmax = np.stack([_ndi_maximum(verts[:, a], label, cidx) for a in range(3)], 1)
        tol = 1e-4 * (cmax[largest] - cmin[largest]).max()
        enclosed = (cmin >= cmin[largest] - tol).all(1) & (cmax <= cmax[largest] + tol).all(1)
        inner = enclosed & (np.sign(cvol) != np.sign(cvol[largest])) & (np.arange(ncomp) != largest)
        keep &= ~inner
    faces = faces[keep[flabel]]
    if len(faces) == 0:
        return verts[:0], faces
    used = np.unique(faces)
    remap = np.full(nv, -1, np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[faces]


def _surface_nets(vol, level, voxel, origin, device):
    # Vectorized Surface Nets: one dual vertex per sign-changing cell at its edge-crossing mean, quads wound CCW-outward.
    # Returns verts (V,3), faces (F,3).
    vol = vol.to(device=device, dtype=torch.float32)
    dx, dy, dz = vol.shape
    origin_t = torch.as_tensor(origin, device=device, dtype=torch.float32)
    empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64))
    if dx < 2 or dy < 2 or dz < 2:
        return empty

    # Active = cells whose 8 corners aren't all in/all out.
    inside = vol >= level  # (dx,dy,dz) bool
    cs8 = [inside[ox:ox + dx - 1, oy:oy + dy - 1, oz:oz + dz - 1]
           for ox, oy, oz in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                              (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))]
    any_in = cs8[0] | cs8[1] | cs8[2] | cs8[3] | cs8[4] | cs8[5] | cs8[6] | cs8[7]
    all_in = cs8[0] & cs8[1] & cs8[2] & cs8[3] & cs8[4] & cs8[5] & cs8[6] & cs8[7]
    active = any_in & ~all_in  # (cx,cy,cz) straddling cells
    nv = int(active.sum())
    if nv == 0:
        return empty

    # Active cells only (a thin shell): each dual vertex = mean of its 12 edges' zero-crossings.
    del any_in, all_in, cs8                                             # corner bool grids no longer needed
    ac = active.nonzero(as_tuple=False)                                 # (nv,3) cell min-corner indices
    offs = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                         [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], device=device)
    offf = offs.to(torch.float32)
    edges = torch.tensor([[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
                          [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]], device=device)
    e0, e1 = edges[:, 0], edges[:, 1]
    oe0, oe1 = offf[e0], offf[e1]                                       # (12,3) edge endpoints

    cstep = 1 << 18                                                     # chunk to bound peak memory (CPU RAM too)
    loc = []
    for st in range(0, nv, cstep):
        ci = ac[st:st + cstep, None, :] + offs[None]                    # (m,8,3)
        cval = vol[ci[..., 0], ci[..., 1], ci[..., 2]]                  # (m,8) corner values
        csl = cval >= level
        v0, v1 = cval[:, e0], cval[:, e1]                               # (m,12)
        cross = (csl[:, e0] != csl[:, e1])[..., None].to(torch.float32)
        denom = v1 - v0
        t = torch.where(denom.abs() > 1e-12, (level - v0) / denom, torch.full_like(denom, 0.5)).clamp(0, 1)
        pts = torch.lerp(oe0, oe1, t[..., None])                        # (m,12,3) local crossings (fused interp)
        loc.append((pts * cross).sum(1) / cross.sum(1).clamp_min(1.0))  # (m,3) in [0,1]
    local = torch.cat(loc, 0) if len(loc) > 1 else loc[0]               # (nv,3)
    verts = origin_t + (ac.to(torch.float32) + local) * voxel           # world space
    del loc, local, ac

    vid = torch.full((dx - 1, dy - 1, dz - 1), -1, dtype=torch.int32, device=device)
    vid[active] = torch.arange(nv, dtype=torch.int32, device=device)
    del active

    # Each straddling grid edge -> one quad from its 4 cells; `sol` (low-end sign) picks outward winding.
    faces = []

    def emit(cr, sol, a, b, d, c):
        valid = cr & (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
        if not bool(valid.any()):
            return
        a, b, c, d, sol = a[valid], b[valid], c[valid], d[valid], sol[valid]
        p2, p4 = torch.where(sol, b, c), torch.where(sol, c, b)  # reverse quad winding where ~sol
        faces.append(torch.stack([a, p2, d], 1))
        faces.append(torch.stack([a, d, p4], 1))

    a = inside[0:dx - 1, 1:dy - 1, 1:dz - 1]
    emit(a != inside[1:dx, 1:dy - 1, 1:dz - 1], a,
         vid[:, 0:dy - 2, 0:dz - 2], vid[:, 1:dy - 1, 0:dz - 2],
         vid[:, 1:dy - 1, 1:dz - 1], vid[:, 0:dy - 2, 1:dz - 1])
    a = inside[1:dx - 1, 0:dy - 1, 1:dz - 1]
    emit(a != inside[1:dx - 1, 1:dy, 1:dz - 1], a,
         vid[0:dx - 2, :, 0:dz - 2], vid[0:dx - 2, :, 1:dz - 1],
         vid[1:dx - 1, :, 1:dz - 1], vid[1:dx - 1, :, 0:dz - 2])
    a = inside[1:dx - 1, 1:dy - 1, 0:dz - 1]
    emit(a != inside[1:dx - 1, 1:dy - 1, 1:dz], a,
         vid[0:dx - 2, 0:dy - 2, :], vid[1:dx - 1, 0:dy - 2, :],
         vid[1:dx - 1, 1:dy - 1, :], vid[0:dx - 2, 1:dy - 1, :])

    if not faces:
        return empty
    return verts.cpu().numpy().astype(np.float32), torch.cat(faces, 0).cpu().numpy().astype(np.int64)


def _otsu_level(values, bins=256):
    # Otsu threshold: the density value that best splits inside/outside (max between-class variance).
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) * 0.5
    w = np.cumsum(hist)  # background-class weight at each split
    mu = np.cumsum(hist * centers)
    wf = w[-1] - w  # foreground-class weight
    mb = mu / np.where(w > 0, w, 1.0)
    mf = (mu[-1] - mu) / np.where(wf > 0, wf, 1.0)
    var_b = w * wf * (mb - mf) ** 2  # between-class variance
    var_b[(w <= 0) | (wf <= 0)] = -1.0
    return float(centers[int(np.argmax(var_b))])


def _taubin_smooth(verts, faces, iters, lam=0.5, mu=-0.53):
    # Taubin lambda|mu smoothing: low-pass the mesh surface without the shrinkage of a Laplacian blur
    # (the mu inflation pass cancels the lambda pass's volume loss). Uniform (umbrella) weights.
    if iters <= 0 or len(verts) == 0 or len(faces) == 0:
        return verts
    nv = len(verts)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], 0)
    e = np.concatenate([e, e[:, ::-1]], 0)  # symmetric adjacency
    adj = coo_matrix((np.ones(len(e), np.float32), (e[:, 0], e[:, 1])), shape=(nv, nv)).tocsr()
    adj.data[:] = 1.0
    deg = np.clip(np.asarray(adj.sum(1)).ravel(), 1.0, None).astype(np.float32)[:, None]
    v = verts.astype(np.float32)  # fp32 matvec: ~2x faster, sub-micron drift on unit-scale verts
    for _ in range(int(iters)):
        for fac in (lam, mu):
            v = v + np.float32(fac) * ((adj @ v) / deg - v)  # fac * (mean(neighbours) - v)
    return np.ascontiguousarray(v)


def _sample_vertex_colours_gpu(colvol, colnorm, verts, origin, voxel, device):
    # GPU trilinear sampling of the colour numerator (3ch) and normaliser (1ch) at vertex grid-coords
    # reproduces scipy map_coordinates(order=1, mode='nearest'). Returns col (V,3) numpy.
    dx, dy, dz = colnorm.shape
    vt = torch.as_tensor(verts, device=device, dtype=torch.float32)
    org = torch.as_tensor(origin, device=device, dtype=torch.float32)
    gi = (vt - org) / voxel                                                   # (V,3) grid-index coords (x,y,z)
    size = torch.tensor([dx, dy, dz], device=device, dtype=torch.float32)
    g = 2.0 * gi / (size - 1).clamp_min(1.0) - 1.0                            # -> [-1,1] (align_corners)
    grid = torch.stack([g[:, 2], g[:, 1], g[:, 0]], -1)[None, None, None]     # (1,1,1,V,3): grid_sample order (W=z,H=y,D=x)

    def samp(v):                                                             # (dx,dy,dz,C) cpu fp16 -> (C,V) fp32 on device
        inp = v.to(device).permute(3, 0, 1, 2)[None].float()
        o = torch.nn.functional.grid_sample(inp, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return o[0, :, 0, 0, :]
    num = samp(colvol)                                                        # (3,V)
    den = samp(colnorm[..., None])                                            # (1,V)
    return (num / den.clamp_min(1e-8)).T.cpu().numpy()                        # (V,3)


def _gaussian_to_mesh(g: Types.SPLAT, i, res, kernel, taubin, level_bias, min_component, min_opacity, color_sharpen, device, progress=None):
    # Mesh one splat: density + colour grids -> Surface Nets -> floater removal -> Taubin smoothing ->
    # volume-sampled colours. Returns (verts, faces int64, colors in [0,1]), or None if no surface.
    rep = progress if progress is not None else (lambda *_: None)

    end = _real_len(g, i)
    xyz = g.positions[i, :end].to(device=device, dtype=torch.float32)
    scale = g.scales[i, :end].to(device=device, dtype=torch.float32)
    quat = g.rotations[i, :end].to(device=device, dtype=torch.float32)
    opacity = g.opacities[i, :end].reshape(-1).to(device=device, dtype=torch.float32)
    rgb = (g.sh[i, :end, 0, :].to(device=device, dtype=torch.float32) * _C0 + 0.5).clamp(0, 1)

    keep = opacity >= min_opacity
    xyz, scale, quat, opacity, rgb = xyz[keep], scale[keep], quat[keep], opacity[keep], rgb[keep]
    if xyz.shape[0] == 0:
        return None

    vol, colvol, colnorm, origin, voxel = _splat_density(xyz, opacity, scale, quat, rgb, res, kernel, device,
                                                         color_sharpen=color_sharpen,
                                                         progress=lambda f: rep(0.25 * f))   # density build: 0 -> 25%
    # Colour: sample on the GPU (grid_sample) when there's headroom
    colour_gpu = not comfy.model_management.is_device_cpu(device) and comfy.model_management.get_free_memory(device) > 6 * vol.numel() * 4
    if colour_gpu:
        colvol_cpu, colnorm_cpu = colvol.cpu(), colnorm.half().cpu()  # park colours (fp16) off-GPU during meshing
        colvol_np = colnorm_np = None
    else:
        colvol_np = colvol.cpu().numpy().astype(np.float32)  # Sum(w^p * rgb) colour numerator (fp16 grid -> fp32)
        colnorm_np = colnorm.cpu().numpy().astype(np.float32)  # Sum(w^p) colour normaliser
    del colvol, colnorm                                      # free the colour grids before iso-surfacing
    rep(0.40)

    vmin, vmax = float(vol.min()), float(vol.max())
    occ = vol[vol > vmax * 1e-3]                             # occupied voxels (skip the empty-space peak)
    if occ.numel() == 0:
        return None
    # Otsu picks the inside/outside split principledly; `level_bias` nudges it (1.0 = auto). Clamp strictly
    # inside the data range so a bias can't push the iso off the histogram.
    level = min(max(_otsu_level(occ.cpu().numpy()) * level_bias, vmin + 1e-6 * (vmax - vmin)),
                vmax - 1e-6 * (vmax - vmin))

    # Iso-surface on the accelerator when there's headroom: ~15x faster than CPU, identical output. Chunked
    # Surface Nets peaks at ~3-3.5x the density grid, so fall back to CPU for large grids / tight VRAM.
    sn_dev = device
    if not comfy.model_management.is_device_cpu(device) and comfy.model_management.get_free_memory(device) < 6 * vol.numel() * 4:
        sn_dev = torch.device("cpu")
        vol = vol.cpu()
    verts, faces = _surface_nets(vol, level, voxel, origin, sn_dev)
    del vol
    rep(0.55)
    if min_component > 0 and len(faces) > 0:
        verts, faces = _clean_components(verts, faces, min_component, device)
    if len(verts) == 0 or len(faces) == 0:
        return None

    # Taubin smooths the blocky iso without shrinking it (unlike blurring the density, which rounds features).
    verts = _taubin_smooth(verts, faces, taubin)
    rep(0.7)

    # Colour each vertex from the co-splatted colour volume: trilinearly sample the numerator Sum(w^p*rgb)
    # and normaliser Sum(w^p) separately, then divide. Normalising AFTER interpolation keeps zero-density
    # edge voxels from pulling colours toward black, and matches the gaussians that formed the surface.
    if colour_gpu:
        col = _sample_vertex_colours_gpu(colvol_cpu, colnorm_cpu, verts, origin, voxel, device)
    else:
        coords = ((verts - origin) / voxel).T  # (3, V) grid-index coords, matching volume axes
        num = np.stack([map_coordinates(colvol_np[..., c], coords, order=1, mode="nearest") for c in range(3)], -1)
        den = map_coordinates(colnorm_np, coords, order=1, mode="nearest")
        col = num / np.clip(den, 1e-8, None)[:, None]
    rep(1.0)

    # The unlit material's COLOR_0 is linear and the viewer sRGB-encodes it on output; the splat colours
    # are display (sRGB) values, so convert sRGB -> linear here to land at the same brightness as the splat.
    col = np.clip(col, 0, 1)
    col = np.where(col <= 0.04045, col / 12.92, ((col + 0.055) / 1.055) ** 2.4).astype(np.float32)

    # Splat +Y is glTF's -Y: rotate 180 deg about X (negate Y,Z) to land upright. Proper rotation, so
    # winding is kept; done after colouring (which works in the splat frame).
    verts = np.ascontiguousarray(verts * np.array([1.0, -1.0, -1.0], dtype=np.float32))
    return (torch.from_numpy(verts), torch.from_numpy(faces), torch.from_numpy(col))


class SplatToMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SplatToMesh",
            display_name="Extract Mesh from Splat",
            search_aliases=["splat to mesh", "gaussian surface nets", "splat surface", "mesh splat"],
            category="3d/splat",
            description="Extract a coloured mesh from a gaussian splat.",
            inputs=[
                IO.Splat.Input("splat"),
                IO.Int.Input("resolution", default=384, min=64, max=768, step=16,
                             tooltip="Density-grid resolution along the longest axis. Higher = finer surface, "
                                     "more VRAM/time (grows with resolution^3)."),
                IO.Int.Input("kernel", default=5, min=1, max=8,
                             tooltip="Max splat half-width in voxels. Each gaussian is rasterized over a window "
                                     "sized to its own 3-sigma, capped here - small surfels stay cheap, large ones "
                                     "aren't truncated. Raise if sparse splats leave gaps."),
                IO.Int.Input("smooth", default=0, min=0, max=60, advanced = True,
                             tooltip="Taubin mesh-smoothing iterations. Smooths the surface without shrinking it "
                                     "(volume-preserving), unlike blurring the density. 0 = raw surface."),
                IO.Float.Input("level", default=0.4, min=0.0, max=2.0, step=0.01,
                               tooltip="Iso-surface level. Auto-picked by Otsu; this biases it (1.0 = auto, lower = "
                                       "fatter/more-connected surface, higher = thinner/tighter)."),
                IO.Int.Input("min_component", default=500, min=0, max=100000, step=50, advanced=True,
                             tooltip="Drop connected components smaller than this many vertices (0 = keep all). "
                                     "Removes detached floater blobs and the inner shell of the double wall."),
                IO.Float.Input("min_opacity", default=0.02, min=0.0, max=1.0, step=0.01, advanced=True,
                               tooltip="Ignore gaussians fainter than this before meshing."),
                IO.Float.Input("color_sharpen", default=2.0, min=1.0, max=8.0, step=0.5,
                               tooltip="Crisp up the vertex texture: 1.0 = physically-correct blend; higher biases "
                                       "each voxel's colour toward its dominant gaussian instead of averaging "
                                       "neighbours (de-smears the texture). Colour only - geometry is unchanged."),
            ],
            outputs=[IO.Mesh.Output(display_name="mesh")],
        )

    @classmethod
    def execute(cls, splat, resolution, kernel, smooth, level, min_component, min_opacity, color_sharpen) -> IO.NodeOutput:
        device = comfy.model_management.get_torch_device()
        b = splat.positions.shape[0]
        prec = 1000  # each splat owns a 0..prec block of the bar; its callback advances within that block
        pbar = comfy.utils.ProgressBar(b * prec)

        verts_l, faces_l, colors_l = [], [], []
        for i in range(b):
            cb = lambda f, base=i * prec: pbar.update_absolute(base + int(min(max(f, 0.0), 1.0) * prec))
            res = _gaussian_to_mesh(splat, i, resolution, kernel, smooth, level, min_component, min_opacity, color_sharpen, device, cb)
            if res is None:
                logging.warning("SplatToMesh: splat %d produced no surface; emitting an empty mesh.", i)
                v, f, c = torch.zeros((0, 3)), torch.zeros((0, 3), dtype=torch.int64), torch.zeros((0, 3))
            else:
                v, f, c = res
            verts_l.append(v)
            faces_l.append(f)
            colors_l.append(c)
            pbar.update_absolute((i + 1) * prec)  # snap to block end (covers empty / early-out splats)
        # unlit: render flat (emissive-like) so SaveGLB matches the splat instead of lighting/washing it.
        return IO.NodeOutput(pack_variable_mesh_batch(verts_l, faces_l, colors=colors_l, unlit=True))


class GaussianExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [SplatToFile3D, File3DToSplat, RenderSplat, CreateCameraInfo, TransformSplat,
                GetSplatCount, MergeSplat, SplatToMesh]


async def comfy_entrypoint() -> GaussianExtension:
    return GaussianExtension()
