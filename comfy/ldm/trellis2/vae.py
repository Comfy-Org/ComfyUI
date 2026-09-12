import torch
import torch.nn as nn
import torch.nn.functional as F
from fractions import Fraction
from typing import List, Any, Dict, Optional, overload, Union
import comfy.ops
from comfy.ldm.trellis2.flexgemm import TorchHashMap, sparse_submanifold_conv3d

ops = comfy.ops.disable_weight_init


def pixel_shuffle_3d(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    B, C, H, W, D = x.shape
    C_ = C // scale_factor**3
    x = x.reshape(B, C_, scale_factor, scale_factor, scale_factor, H, W, D)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(B, C_, H*scale_factor, W*scale_factor, D*scale_factor)
    return x

class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        super(SparseConv3d, self).__init__()
        sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride, dilation, padding, bias, indice_key)

    def forward(self, x):
        return sparse_conv3d_forward(self, x)


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size, ) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, ) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation, ) * 3
    self.cache_neighbor_map = True

    self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    # Permute weight (Co, Ci, Kd, Kh, Kw) -> (Co, Kd, Kh, Kw, Ci)
    self.weight = nn.Parameter(self.weight.permute(0, 2, 3, 4, 1).contiguous())


def sparse_conv3d_forward(self, x):
    # check if neighbor map is already computed
    Co, Kd, Kh, Kw, Ci = self.weight.shape
    neighbor_cache_key = f'SubMConv3d_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{self.dilation}'
    neighbor_cache = x.get_spatial_cache(neighbor_cache_key)
    feats = x.feats
    weight = comfy.ops.cast_to(self.weight, feats.dtype, feats.device)
    bias = comfy.ops.cast_to(self.bias, feats.dtype, feats.device) if self.bias is not None else None

    out, neighbor_cache_ = sparse_submanifold_conv3d(
        x.feats,
        x.coords,
        x.spatial_shape,
        weight,
        bias,
        neighbor_cache,
        self.dilation,
        self.cache_neighbor_map,
    )

    if neighbor_cache is None and neighbor_cache_ is not None:
        x.register_spatial_cache(neighbor_cache_key, neighbor_cache_)

    out = x.replace(out)
    return out

class SparseConvNeXtBlock3d(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.channels = channels

        self.norm = ops.LayerNorm(channels, elementwise_affine=True, eps=1e-6)
        self.conv = SparseConv3d(channels, channels, 3)
        self.mlp = nn.Sequential(
            ops.Linear(channels, int(channels * mlp_ratio)),
            nn.SiLU(inplace=True),
            ops.Linear(int(channels * mlp_ratio), channels),
        )

    def _forward(self, x):
        h = self.conv(x)
        h = h.replace(self.norm(h.feats))
        h = h.replace(self.mlp(h.feats))
        h.feats.add_(x.feats.to(h.feats))
        return h

    def forward(self, x):
        return self._forward(x)

class SparseSpatial2Channel(nn.Module):
    def __init__(self, factor: int = 2):
        super(SparseSpatial2Channel, self).__init__()
        self.factor = factor

    def forward(self, x):
        DIM = x.coords.shape[-1] - 1
        cache = x.get_spatial_cache(f'spatial2channel_{self.factor}')
        if cache is None:
            coord = list(x.coords.unbind(dim=-1))
            for i in range(DIM):
                coord[i+1] = coord[i+1] // self.factor
            subidx = x.coords[:, 1:] % self.factor
            subidx = sum([subidx[..., i] * self.factor ** i for i in range(DIM)])

            MAX = [(s + self.factor - 1) // self.factor for s in x.spatial_shape]
            OFFSET = [1]
            for size in reversed(MAX):
                OFFSET.append(OFFSET[-1] * size)
            OFFSET.reverse()
            code = sum([c * o for c, o in zip(coord, OFFSET)])
            code, idx = code.unique(return_inverse=True)

            new_coords = torch.stack(
                [code // OFFSET[0]] +
                [(code // OFFSET[i+1]) % MAX[i] for i in range(DIM)],
                dim=-1
            )
        else:
            new_coords, idx, subidx = cache

        new_feats = torch.zeros(new_coords.shape[0] * self.factor ** DIM, x.feats.shape[1], device=x.feats.device, dtype=x.feats.dtype)
        new_feats[idx * self.factor ** DIM + subidx] = x.feats

        out = SparseTensor(new_feats.reshape(new_coords.shape[0], -1), new_coords, None if x._shape is None else torch.Size([x._shape[0], x._shape[1] * self.factor ** DIM]))
        out._scale = tuple([s * self.factor for s in x._scale])
        out._spatial_cache = dict(x._spatial_cache)

        if cache is None:
            x.register_spatial_cache(f'spatial2channel_{self.factor}', (new_coords, idx, subidx))
            out.register_spatial_cache(f'channel2spatial_{self.factor}', (x.coords, idx, subidx))
            out.register_spatial_cache('shape', torch.Size(MAX))

        return out

class SparseChannel2Spatial(nn.Module):
    def __init__(self, factor: int = 2):
        super(SparseChannel2Spatial, self).__init__()
        self.factor = factor

    def _subdivision_plan(self, x, subdivision):
        DIM = x.coords.shape[-1] - 1
        cache = x.get_spatial_cache(f'channel2spatial_{self.factor}')
        if cache is None:
            if subdivision is None:
                raise ValueError('Cache not found. Provide subdivision tensor or pair SparseChannel2Spatial with SparseSpatial2Channel.')
            flat_idx = subdivision.feats.flatten().nonzero(as_tuple=True)[0]
            idx = torch.div(flat_idx, self.factor ** DIM, rounding_mode='floor')
            subidx = flat_idx.remainder(self.factor ** DIM)
            new_coords = x.coords[idx]
            new_coords[:, 1:] *= self.factor
            for i in range(DIM):
                new_coords[:, i+1] += subidx // self.factor ** i % self.factor
            return new_coords, flat_idx, False
        else:
            new_coords, idx, subidx = cache
            flat_idx = idx * self.factor ** DIM + subidx
            return new_coords, flat_idx, True

    def _apply_plan(self, x, new_coords, flat_idx, inherit_cache):
        DIM = x.coords.shape[-1] - 1
        x_feats = x.feats.reshape(x.feats.shape[0] * self.factor ** DIM, -1)
        new_feats = x_feats[flat_idx]
        out = SparseTensor(new_feats, new_coords, None if x._shape is None else torch.Size([x._shape[0], x._shape[1] // self.factor ** DIM]))
        out._scale = tuple([s / self.factor for s in x._scale])
        if inherit_cache:           # only keep cache when subdiv following it
            out._spatial_cache = dict(x._spatial_cache)
        return out

    def forward(self, x, subdivision = None):
        new_coords, flat_idx, inherit_cache = self._subdivision_plan(x, subdivision)
        return self._apply_plan(x, new_coords, flat_idx, inherit_cache)

class SparseResBlockC2S3d(nn.Module):
    def __init__(self, channels: int, out_channels: Optional[int] = None, pred_subdiv: bool = True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.pred_subdiv = pred_subdiv

        self.norm1 = ops.LayerNorm(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = ops.LayerNorm(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = SparseConv3d(channels, self.out_channels * 8, 3)
        self.conv2 = SparseConv3d(self.out_channels, self.out_channels, 3)
        if pred_subdiv:
            self.to_subdiv = SparseLinear(channels, 8)
        self.updown = SparseChannel2Spatial(2)

    def forward(self, x, subdiv = None):
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats, inplace=True))
        h = self.conv1(h)
        subdiv_binarized = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        new_coords, flat_idx, inherit_cache = self.updown._subdivision_plan(h, subdiv_binarized)
        h = self.updown._apply_plan(h, new_coords, flat_idx, inherit_cache)
        skip_feats = x.feats.reshape(x.feats.shape[0] * 8, -1)[flat_idx]
        del x, new_coords, flat_idx, subdiv_binarized
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats, inplace=True))
        h = self.conv2(h)
        skip_repeat = self.out_channels // (self.channels // 8)
        h.feats.view(h.feats.shape[0], skip_feats.shape[1], skip_repeat).add_(skip_feats.to(h.feats.dtype).unsqueeze(-1))
        if self.pred_subdiv:
            return h, subdiv
        else:
            return h


class VarLenTensor:

    def __init__(self, feats: torch.Tensor, layout: List[slice]=None):
        self.feats = feats
        self.layout = layout if layout is not None else [slice(0, feats.shape[0])]
        self._cache = {}

    @staticmethod
    def from_tensor_list(tensor_list: List[torch.Tensor]) -> 'VarLenTensor':
        """
        Create a VarLenTensor from a list of tensors.
        """
        feats = torch.cat(tensor_list, dim=0)
        layout = []
        start = 0
        for tensor in tensor_list:
            layout.append(slice(start, start + tensor.shape[0]))
            start += tensor.shape[0]
        return VarLenTensor(feats, layout)

    def __len__(self) -> int:
        return len(self.layout)

    @property
    def shape(self) -> torch.Size:
        return torch.Size([len(self.layout), *self.feats.shape[1:]])

    def dim(self) -> int:
        return len(self.shape)

    @property
    def ndim(self) -> int:
        return self.dim()

    @property
    def dtype(self):
        return self.feats.dtype

    @property
    def device(self):
        return self.feats.device

    @property
    def seqlen(self) -> torch.LongTensor:
        if 'seqlen' not in self._cache:
            self._cache['seqlen'] = torch.tensor([l.stop - l.start for l in self.layout], dtype=torch.long, device=self.device)
        return self._cache['seqlen']

    @property
    def batch_boardcast_map(self) -> torch.LongTensor:
        """
        Get the broadcast map for the varlen tensor.
        """
        if 'batch_boardcast_map' not in self._cache:
            self._cache['batch_boardcast_map'] = torch.repeat_interleave(
                torch.arange(len(self.layout), device=self.device),
                self.seqlen,
            )
        return self._cache['batch_boardcast_map']

    @overload
    def to(self, dtype: torch.dtype, *, non_blocking: bool = False, copy: bool = False) -> 'VarLenTensor': ...

    @overload
    def to(self, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None, *, non_blocking: bool = False, copy: bool = False) -> 'VarLenTensor': ...

    def to(self, *args, **kwargs) -> 'VarLenTensor':
        device = None
        dtype = None
        if len(args) == 2:
            device, dtype = args
        elif len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
            else:
                device = args[0]
        if 'dtype' in kwargs:
            assert dtype is None, "to() received multiple values for argument 'dtype'"
            dtype = kwargs['dtype']
        if 'device' in kwargs:
            assert device is None, "to() received multiple values for argument 'device'"
            device = kwargs['device']
        non_blocking = kwargs.get('non_blocking', False)
        copy = kwargs.get('copy', False)

        new_feats = self.feats.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        return self.replace(new_feats)

    def type(self, dtype):
        new_feats = self.feats.type(dtype)
        return self.replace(new_feats)

    def cpu(self) -> 'VarLenTensor':
        new_feats = self.feats.cpu()
        return self.replace(new_feats)

    def cuda(self) -> 'VarLenTensor':
        new_feats = self.feats.cuda()
        return self.replace(new_feats)

    def half(self) -> 'VarLenTensor':
        new_feats = self.feats.half()
        return self.replace(new_feats)

    def float(self) -> 'VarLenTensor':
        new_feats = self.feats.float()
        return self.replace(new_feats)

    def detach(self) -> 'VarLenTensor':
        new_feats = self.feats.detach()
        return self.replace(new_feats)

    def reshape(self, *shape) -> 'VarLenTensor':
        new_feats = self.feats.reshape(self.feats.shape[0], *shape)
        return self.replace(new_feats)

    def unbind(self, dim: int) -> List['VarLenTensor']:
        return varlen_unbind(self, dim)

    def replace(self, feats: torch.Tensor) -> 'VarLenTensor':
        new_tensor = VarLenTensor(
            feats=feats,
            layout=self.layout,
        )
        # Shallow-copy so derived tensors don't share-by-reference the cache
        # dict — see SparseTensor.replace for rationale.
        new_tensor._cache = dict(self._cache)
        return new_tensor

    def __neg__(self) -> 'VarLenTensor':
        return self.replace(-self.feats)

    def __elemwise__(self, other: Union[torch.Tensor, 'VarLenTensor'], op: callable) -> 'VarLenTensor':
        if isinstance(other, torch.Tensor):
            can_broadcast_batch = other.ndim <= self.ndim and all(
                source == 1 or source == target
                for source, target in zip(reversed(other.shape), reversed(self.shape))
            )
            if can_broadcast_batch:
                other = torch.broadcast_to(other, self.shape)
                other = other[self.batch_boardcast_map]
        if isinstance(other, VarLenTensor):
            other = other.feats
        new_feats = op(self.feats, other)
        new_tensor = self.replace(new_feats)
        return new_tensor

    def __add__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, torch.add)

    def __radd__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, torch.add)

    def __sub__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, torch.sub)

    def __rsub__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, lambda x, y: torch.sub(y, x))

    def __mul__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, torch.mul)

    def __rmul__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, torch.mul)

    def __truediv__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, torch.div)

    def __rtruediv__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, lambda x, y: torch.div(y, x))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = range(*idx.indices(self.shape[0]))
        elif isinstance(idx, list):
            assert all(isinstance(i, int) for i in idx), f"Only integer indices are supported: {idx}"
        elif isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                assert idx.shape == (self.shape[0],), f"Invalid index shape: {idx.shape}"
                idx = idx.nonzero().squeeze(1)
            elif idx.dtype in [torch.int32, torch.int64]:
                assert len(idx.shape) == 1, f"Invalid index shape: {idx.shape}"
            else:
                raise ValueError(f"Unknown index type: {idx.dtype}")
        else:
            raise ValueError(f"Unknown index type: {type(idx)}")

        new_feats = []
        new_layout = []
        start = 0
        for new_idx, old_idx in enumerate(idx):
            new_feats.append(self.feats[self.layout[old_idx]])
            new_layout.append(slice(start, start + len(new_feats[-1])))
            start += len(new_feats[-1])
        new_feats = torch.cat(new_feats, dim=0).contiguous()
        new_tensor = VarLenTensor(feats=new_feats, layout=new_layout)
        return new_tensor

    def __repr__(self) -> str:
        return f"VarLenTensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"

def varlen_unbind(input: VarLenTensor, dim: int) -> Union[List[VarLenTensor]]:

    if dim == 0:
        return [input[i] for i in range(len(input))]
    else:
        feats = input.feats.unbind(dim)
        return [input.replace(f) for f in feats]


class SparseTensor(VarLenTensor):

    @overload
    def __init__(self, feats: torch.Tensor, coords: torch.Tensor, shape: Optional[torch.Size] = None, **kwargs): ...

    @overload
    def __init__(self, data, shape: Optional[torch.Size] = None, **kwargs): ...

    def __init__(self, *args, **kwargs):
        method_id = 0
        if len(args) != 0:
            method_id = 0 if isinstance(args[0], torch.Tensor) else 1
        else:
            method_id = 1 if 'data' in kwargs else 0

        if method_id == 0:
            feats, coords, shape = args + (None,) * (3 - len(args))
            if 'feats' in kwargs:
                feats = kwargs['feats']
                del kwargs['feats']
            if 'coords' in kwargs:
                coords = kwargs['coords']
                del kwargs['coords']
            if 'shape' in kwargs:
                shape = kwargs['shape']
                del kwargs['shape']

            self.data = {
                'feats': feats,
                'coords': coords,
            }
        elif method_id == 1:
            data, shape = args + (None,) * (2 - len(args))
            if 'data' in kwargs:
                data = kwargs['data']
                del kwargs['data']
            if 'shape' in kwargs:
                shape = kwargs['shape']
                del kwargs['shape']

            self.data = data

        self._shape = shape
        self._scale = kwargs.get('scale', (Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)))
        self._spatial_cache = kwargs.get('spatial_cache', {})

    @staticmethod
    def from_tensor_list(feats_list: List[torch.Tensor], coords_list: List[torch.Tensor]) -> 'SparseTensor':
        """
        Create a SparseTensor from a list of tensors.
        """
        feats = torch.cat(feats_list, dim=0)
        coords = []
        for i, coord in enumerate(coords_list):
            coord = torch.cat([torch.full_like(coord[:, :1], i), coord[:, 1:]], dim=1)
            coords.append(coord)
        coords = torch.cat(coords, dim=0)
        return SparseTensor(feats, coords)

    def __len__(self) -> int:
        return len(self.layout)

    def __cal_shape(self, feats, coords):
        shape = []
        shape.append(coords[:, 0].max().item() + 1)
        shape.extend([*feats.shape[1:]])
        return torch.Size(shape)

    def __cal_layout(self, coords, batch_size):
        seq_len = torch.bincount(coords[:, 0], minlength=batch_size)
        offset = torch.cumsum(seq_len, dim=0)
        layout = [slice((offset[i] - seq_len[i]).item(), offset[i].item()) for i in range(batch_size)]
        return layout

    def __cal_spatial_shape(self, coords):
        return torch.Size((coords[:, 1:].max(0)[0] + 1).tolist())

    @property
    def shape(self) -> torch.Size:
        if self._shape is None:
            self._shape = self.__cal_shape(self.feats, self.coords)
        return self._shape

    @property
    def layout(self) -> List[slice]:
        layout = self.get_spatial_cache('layout')
        if layout is None:
            layout = self.__cal_layout(self.coords, self.shape[0])
            self.register_spatial_cache('layout', layout)
        return layout

    @property
    def spatial_shape(self) -> torch.Size:
        spatial_shape = self.get_spatial_cache('shape')
        if spatial_shape is None:
            spatial_shape = self.__cal_spatial_shape(self.coords)
            self.register_spatial_cache('shape', spatial_shape)
        return spatial_shape

    @property
    def feats(self) -> torch.Tensor:
        return self.data['feats']

    @feats.setter
    def feats(self, value: torch.Tensor):
        self.data['feats'] = value

    @property
    def coords(self) -> torch.Tensor:
        return self.data['coords']

    @coords.setter
    def coords(self, value: torch.Tensor):
        self.data['coords'] = value

    @property
    def dtype(self):
        return self.feats.dtype

    @property
    def device(self):
        return self.feats.device

    @property
    def seqlen(self) -> torch.LongTensor:
        seqlen = self.get_spatial_cache('seqlen')
        if seqlen is None:
            seqlen = torch.tensor([l.stop - l.start for l in self.layout], dtype=torch.long, device=self.device)
            self.register_spatial_cache('seqlen', seqlen)
        return seqlen

    @property
    def batch_boardcast_map(self) -> torch.LongTensor:
        """
        Get the broadcast map for the varlen tensor.
        """
        batch_boardcast_map = self.get_spatial_cache('batch_boardcast_map')
        if batch_boardcast_map is None:
            batch_boardcast_map = torch.repeat_interleave(
                torch.arange(len(self.layout), device=self.device),
                self.seqlen,
            )
            self.register_spatial_cache('batch_boardcast_map', batch_boardcast_map)
        return batch_boardcast_map

    @overload
    def to(self, dtype: torch.dtype, *, non_blocking: bool = False, copy: bool = False) -> 'SparseTensor': ...

    @overload
    def to(self, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None, *, non_blocking: bool = False, copy: bool = False) -> 'SparseTensor': ...

    def to(self, *args, **kwargs) -> 'SparseTensor':
        device = None
        dtype = None
        if len(args) == 2:
            device, dtype = args
        elif len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
            else:
                device = args[0]
        if 'dtype' in kwargs:
            assert dtype is None, "to() received multiple values for argument 'dtype'"
            dtype = kwargs['dtype']
        if 'device' in kwargs:
            assert device is None, "to() received multiple values for argument 'device'"
            device = kwargs['device']
        non_blocking = kwargs.get('non_blocking', False)
        copy = kwargs.get('copy', False)

        new_feats = self.feats.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        new_coords = self.coords.to(device=device, non_blocking=non_blocking, copy=copy)
        return self.replace(new_feats, new_coords)

    def type(self, dtype):
        new_feats = self.feats.type(dtype)
        return self.replace(new_feats)

    def cpu(self) -> 'SparseTensor':
        new_feats = self.feats.cpu()
        new_coords = self.coords.cpu()
        return self.replace(new_feats, new_coords)

    def cuda(self) -> 'SparseTensor':
        new_feats = self.feats.cuda()
        new_coords = self.coords.cuda()
        return self.replace(new_feats, new_coords)

    def half(self) -> 'SparseTensor':
        new_feats = self.feats.half()
        return self.replace(new_feats)

    def float(self) -> 'SparseTensor':
        new_feats = self.feats.float()
        return self.replace(new_feats)

    def detach(self) -> 'SparseTensor':
        new_coords = self.coords.detach()
        new_feats = self.feats.detach()
        return self.replace(new_feats, new_coords)

    def reshape(self, *shape) -> 'SparseTensor':
        new_feats = self.feats.reshape(self.feats.shape[0], *shape)
        return self.replace(new_feats)

    def unbind(self, dim: int) -> List['SparseTensor']:
        return sparse_unbind(self, dim)

    def replace(self, feats: torch.Tensor, coords: Optional[torch.Tensor] = None) -> 'SparseTensor':
        new_data = {
            'feats': feats,
            'coords': self.data['coords'] if coords is None else coords,
        }
        return SparseTensor(
            new_data,
            shape=torch.Size([self._shape[0]] + list(feats.shape[1:])) if self._shape is not None else None,
            scale=self._scale,
            # Shallow-copy the cache: each derived tensor gets its own dict, so
            # adding/overwriting entries on one doesn't leak to siblings.
            # Cached tensors themselves are still shared by reference (safe
            # because they're read-only after populate).
            spatial_cache=dict(self._spatial_cache),
        )

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = range(*idx.indices(self.shape[0]))
        elif isinstance(idx, list):
            assert all(isinstance(i, int) for i in idx), f"Only integer indices are supported: {idx}"
        elif isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                assert idx.shape == (self.shape[0],), f"Invalid index shape: {idx.shape}"
                idx = idx.nonzero().squeeze(1)
            elif idx.dtype in [torch.int32, torch.int64]:
                assert len(idx.shape) == 1, f"Invalid index shape: {idx.shape}"
            else:
                raise ValueError(f"Unknown index type: {idx.dtype}")
        else:
            raise ValueError(f"Unknown index type: {type(idx)}")

        new_coords = []
        new_feats = []
        new_layout = []
        new_shape = torch.Size([len(idx)] + list(self.shape[1:]))
        start = 0
        for new_idx, old_idx in enumerate(idx):
            new_coords.append(self.coords[self.layout[old_idx]].clone())
            new_coords[-1][:, 0] = new_idx
            new_feats.append(self.feats[self.layout[old_idx]])
            new_layout.append(slice(start, start + len(new_coords[-1])))
            start += len(new_coords[-1])
        new_coords = torch.cat(new_coords, dim=0).contiguous()
        new_feats = torch.cat(new_feats, dim=0).contiguous()
        new_tensor = SparseTensor(feats=new_feats, coords=new_coords, shape=new_shape)
        new_tensor.register_spatial_cache('layout', new_layout)
        return new_tensor

    def register_spatial_cache(self, key, value) -> None:
        """
        Register a spatial cache.
        The spatial cache can be any thing you want to cache.
        The registery and retrieval of the cache is based on current scale.
        """
        scale_key = str(self._scale)
        if scale_key not in self._spatial_cache:
            self._spatial_cache[scale_key] = {}
        self._spatial_cache[scale_key][key] = value

    def get_spatial_cache(self, key=None):
        """
        Get a spatial cache.
        """
        scale_key = str(self._scale)
        cur_scale_cache = self._spatial_cache.get(scale_key, {})
        if key is None:
            return cur_scale_cache
        return cur_scale_cache.get(key, None)

    def __repr__(self) -> str:
        return f"SparseTensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"

    def _comfy_cache_tensors(self):
        return self.data, self._spatial_cache

def sparse_cat(inputs: List[SparseTensor], dim: int = 0) -> SparseTensor:
    if dim == 0:
        start = 0
        coords = []
        for input in inputs:
            coords.append(input.coords.clone())
            coords[-1][:, 0] += start
            start += input.shape[0]
        coords = torch.cat(coords, dim=0)
        feats = torch.cat([input.feats for input in inputs], dim=0)
        output = SparseTensor(
            coords=coords,
            feats=feats,
        )
    else:
        feats = torch.cat([input.feats for input in inputs], dim=dim)
        output = inputs[0].replace(feats)

    return output


def sparse_unbind(input: SparseTensor, dim: int) -> List[SparseTensor]:
    if dim == 0:
        return [input[i] for i in range(input.shape[0])]
    else:
        feats = input.feats.unbind(dim)
        return [input.replace(f) for f in feats]

# allow operations.Linear inheritance
class SparseLinear:
    def __new__(cls, in_features, out_features, bias=True, device=None, dtype=None, operations=ops, *args, **kwargs):
        class _SparseLinear(operations.Linear):
            def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
                super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

            def forward(self, input: VarLenTensor) -> VarLenTensor:
                return input.replace(super().forward(input.feats))

        return _SparseLinear(in_features, out_features, bias=bias, device=device, dtype=dtype, *args, **kwargs)

class SparseUnetVaeDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.pred_subdiv = pred_subdiv

        self.output_layer = SparseLinear(model_channels[-1], out_channels)
        self.from_latent = SparseLinear(latent_channels, model_channels[0])

        self.blocks = nn.ModuleList([])
        for i in range(len(num_blocks)):
            self.blocks.append(nn.ModuleList([]))
            for j in range(num_blocks[i]):
                self.blocks[-1].append(
                    globals()[block_type[i]](
                        model_channels[i],
                        **block_args[i],
                    )
                )
            if i < len(num_blocks) - 1:
                up_block = globals()[up_block_type[i]](
                    model_channels[i],
                    model_channels[i+1],
                    pred_subdiv=pred_subdiv,
                    **block_args[i],
                )
                if num_blocks[i + 1] == 0:
                    up_block.conv2.cache_neighbor_map = False
                self.blocks[-1].append(up_block)

    def forward(self, x: SparseTensor, guide_subs: Optional[List[SparseTensor]] = None, return_subs: bool = False) -> SparseTensor:
        h = self.from_latent(x)
        subs = []
        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    if self.pred_subdiv:
                        h, sub = block(h)
                        subs.append(SparseTensor(feats=sub.feats, coords=sub.coords, shape=sub.shape, scale=sub._scale))
                    else:
                        h = block(h, subdiv=guide_subs[i] if guide_subs is not None else None)
                else:
                    h = block(h)
        h = SparseTensor(
            feats=F.layer_norm(h.feats, h.feats.shape[-1:]),
            coords=h.coords,
            shape=h.shape,
            scale=h._scale,
        )
        h = self.output_layer(h)
        if return_subs:
            return h, subs
        else:
            return h

    def upsample(self, x: SparseTensor, upsample_times: int) -> torch.Tensor:
        h = self.from_latent(x)
        for i, res in enumerate(self.blocks):
            if i == upsample_times:
                return h.coords
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    h, sub = block(h)
                else:
                    h = block(h)

class FlexiDualGridVaeDecoder(SparseUnetVaeDecoder):
    def __init__(
        self,
        resolution: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        voxel_margin: float = 0.5,
    ):
        self.resolution = resolution
        self.voxel_margin = voxel_margin
        super().__init__(
            7,
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            up_block_type,
            block_args,
        )

    def set_resolution(self, resolution: int) -> None:
        self.resolution = resolution

    def forward(self, x: SparseTensor, gt_intersected: SparseTensor = None, **kwargs):
        decoded = super().forward(x, **kwargs)
        out_list = list(decoded) if isinstance(decoded, tuple) else [decoded]
        h = out_list[0]
        vertices = h.replace((1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - self.voxel_margin)
        intersected = h.replace(h.feats[..., 3:6] > 0)
        quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
        mesh = [flexible_dual_grid_to_mesh(
            v.coords[:, 1:], v.feats, i.feats, q.feats,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            grid_size=self.resolution,
        ) for v, i, q in zip(vertices, intersected, quad_lerp)]
        out_list[0] = mesh
        return out_list[0] if len(out_list) == 1 else tuple(out_list)

def flexible_dual_grid_to_mesh(
    coords: torch.Tensor,
    dual_vertices: torch.Tensor,
    intersected_flag: torch.Tensor,
    split_weight: Union[torch.Tensor, None],
    aabb: Union[list, tuple, torch.Tensor],
    voxel_size: Union[float, list, tuple, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, torch.Tensor] = None,
):

    device = coords.device
    # Small constant index tables — built per call (stateless), not cached on the function.
    edge_neighbor_voxel_offset = torch.tensor([
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],     # x-axis
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],     # y-axis
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],     # z-axis
    ], dtype=torch.int, device=device).unsqueeze(0)
    quad_split_1 = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long, device=device)
    quad_split_2 = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long, device=device)

    aabb = torch.tensor(aabb, dtype=torch.float32, device=device)

    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size] * 3
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        if isinstance(grid_size, int):
            grid_size = [grid_size] * 3
        grid_size = torch.tensor(grid_size, dtype=torch.int32, device=device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size

    # Extract mesh
    N = dual_vertices.shape[0]
    _, H, D = int(grid_size[0].item()), int(grid_size[1].item()), int(grid_size[2].item())
    flat_keys = coords[:, 0].long() * (H * D)
    flat_keys.add_(coords[:, 1].long() * D)
    flat_keys.add_(coords[:, 2].long())
    values = torch.arange(N, dtype=torch.int32, device=device)
    torch_hashmap = TorchHashMap(flat_keys, values)

    # Find connected voxels — direct gather instead of materializing the full [N, 3, 4, 3]
    n_idx, axis_idx = intersected_flag.nonzero(as_tuple=True)                          # (M,), (M,)
    offsets_per_axis = edge_neighbor_voxel_offset[0]        # (3, 4, 3)
    connected_voxel = coords[n_idx].unsqueeze(1) + offsets_per_axis[axis_idx]          # (M, 4, 3)
    M = connected_voxel.shape[0]
    # flatten connected voxel coords and lookup. In-place to avoid extra memory allocation.
    H, D = int(grid_size[1].item()), int(grid_size[2].item())
    cv = connected_voxel.reshape(-1, 3)
    conn_flat = cv[:, 0].long() * (H * D)
    conn_flat.add_(cv[:, 1].long() * D)
    conn_flat.add_(cv[:, 2].long())

    conn_indices = torch_hashmap.lookup_flat(conn_flat).reshape(M, 4).int()
    connected_voxel_valid = (conn_indices >= 0).all(dim=1)   # -1 = missing neighbor
    quad_indices = conn_indices[connected_voxel_valid].int()                             # (L, 4)

    # Chain in-place — each op in the original allocated a fresh N*3 fp32 tensor.
    mesh_vertices = coords.float()
    mesh_vertices.add_(dual_vertices).mul_(voxel_size).add_(aabb[0].reshape(1, 3))
    if split_weight is None:
        # if split 1
        atempt_triangles_0 = quad_indices[:, quad_split_1]
        normals0 = torch.cross(mesh_vertices[atempt_triangles_0[:, 1]] - mesh_vertices[atempt_triangles_0[:, 0]], mesh_vertices[atempt_triangles_0[:, 2]] - mesh_vertices[atempt_triangles_0[:, 0]])
        normals1 = torch.cross(mesh_vertices[atempt_triangles_0[:, 2]] - mesh_vertices[atempt_triangles_0[:, 1]], mesh_vertices[atempt_triangles_0[:, 3]] - mesh_vertices[atempt_triangles_0[:, 1]])
        align0 = (normals0 * normals1).sum(dim=1, keepdim=True).abs()
        # if split 2
        atempt_triangles_1 = quad_indices[:, quad_split_2]
        normals0 = torch.cross(mesh_vertices[atempt_triangles_1[:, 1]] - mesh_vertices[atempt_triangles_1[:, 0]], mesh_vertices[atempt_triangles_1[:, 2]] - mesh_vertices[atempt_triangles_1[:, 0]])
        normals1 = torch.cross(mesh_vertices[atempt_triangles_1[:, 2]] - mesh_vertices[atempt_triangles_1[:, 1]], mesh_vertices[atempt_triangles_1[:, 3]] - mesh_vertices[atempt_triangles_1[:, 1]])
        align1 = (normals0 * normals1).sum(dim=1, keepdim=True).abs()
        # select split
        mesh_triangles = torch.where(align0 > align1, atempt_triangles_0, atempt_triangles_1).reshape(-1, 3)
    else:
        split_weight_ws = split_weight[quad_indices]
        split_weight_ws_02 = split_weight_ws[:, 0] * split_weight_ws[:, 2]
        split_weight_ws_13 = split_weight_ws[:, 1] * split_weight_ws[:, 3]
        mesh_triangles = torch.where(
            split_weight_ws_02 > split_weight_ws_13,
            quad_indices[:, quad_split_1],
            quad_indices[:, quad_split_2]
        ).reshape(-1, 3)

    return mesh_vertices, mesh_triangles

class ChannelLayerNorm32(ops.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, DIM-1, *range(1, DIM-1)).contiguous()
        return x

class UpsampleBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = ops.Conv3d(in_channels, out_channels * 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pixel_shuffle_3d(self.conv(x), 2)

class ResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.norm1 = ChannelLayerNorm32(channels)
        self.norm2 = ChannelLayerNorm32(self.out_channels)
        self.conv1 = ops.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = ops.Conv3d(self.out_channels, self.out_channels, 3, padding=1)
        self.skip_connection = ops.Conv3d(channels, self.out_channels, 1) if channels != self.out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h


class SparseStructureDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle

        self.input_layer = ops.Conv3d(latent_channels, channels[0], 3, padding=1)

        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], channels[0])
            for _ in range(num_res_blocks_middle)
        ])

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(ch, ch)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.blocks.append(
                    UpsampleBlock3d(ch, channels[i+1])
                )

        self.out_layer = nn.Sequential(
            ChannelLayerNorm32(channels[-1]),
            nn.SiLU(),
            ops.Conv3d(channels[-1], out_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)
        h = self.out_layer(h)
        return h


class ShapeVae(nn.Module):
    """Decoder bundle from the Trellis2 shape checkpoint: structure decoder
    (32^3 SS latent → 64^3 dense occupancy) and shape decoder (sparse latent →
    mesh + per-stage subdivisions)."""

    def __init__(self):
        super().__init__()
        self.shape_dec = FlexiDualGridVaeDecoder(
            resolution=256,
            model_channels=[1024, 512, 256, 128, 64],
            latent_channels=32,
            num_blocks=[4, 16, 8, 4, 0],
            block_type=["SparseConvNeXtBlock3d"] * 5,
            up_block_type=["SparseResBlockC2S3d"] * 4,
            block_args=[{}, {}, {}, {}, {}],
        )
        self.struct_dec = SparseStructureDecoder(
            out_channels=1,
            latent_channels=8,
            num_res_blocks=2,
            num_res_blocks_middle=2,
            channels=[512, 128, 32],
        )
        self.register_buffer("resolution", torch.tensor(1024.0), persistent=False)

    def decode_structure(self, x: torch.Tensor) -> torch.Tensor:
        return self.struct_dec(x)

    def decode_shape_slat(self, slat: 'SparseTensor', resolution: int):
        self.shape_dec.set_resolution(resolution)
        return self.shape_dec(slat, return_subs=True)

    def upsample_shape(self, slat: 'SparseTensor', upsample_times: int) -> torch.Tensor:
        return self.shape_dec.upsample(slat, upsample_times)


class TextureVae(nn.Module):
    """Decoder bundle from the Trellis2 texture checkpoint: sparse 3D
    per-voxel color decoder, guided by subdivisions from a prior shape decode."""

    def __init__(self):
        super().__init__()
        self.txt_dec = SparseUnetVaeDecoder(
            out_channels=6,
            model_channels=[1024, 512, 256, 128, 64],
            latent_channels=32,
            num_blocks=[4, 16, 8, 4, 0],
            block_type=["SparseConvNeXtBlock3d"] * 5,
            up_block_type=["SparseResBlockC2S3d"] * 4,
            block_args=[{}, {}, {}, {}, {}],
            pred_subdiv=False,
        )
        self.register_buffer("resolution", torch.tensor(1024.0), persistent=False)

    def decode_tex_slat(self, slat: 'SparseTensor', subs):
        return self.txt_dec(slat, guide_subs=subs) * 0.5 + 0.5
