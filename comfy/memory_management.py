import math
import ctypes
import dataclasses
import torch
from typing import NamedTuple

import comfy_aimdo.host_buffer
from comfy.quant_ops import QuantizedTensor


class TensorFileSlice(NamedTuple):
    file_ref: object
    lock: object
    offset: int
    size: int


def read_tensor_file_slice_into(tensor, destination, stream=None, destination2=None):

    if isinstance(tensor, QuantizedTensor):
        if not read_tensor_file_slice_into(tensor._qdata,
                                           destination._qdata if destination is not None else None, stream=stream,
                                           destination2=(destination2._qdata if destination2 is not None else None)):
            return False

        if destination is not None:
            dst_orig_dtype = destination._params.orig_dtype
            destination._params.copy_from(tensor._params, non_blocking=False)
            destination._params = dataclasses.replace(destination._params, orig_dtype=dst_orig_dtype)
        if destination2 is not None:
            dst_orig_dtype = destination2._params.orig_dtype
            destination2._params.copy_from(destination._params if destination is not None else tensor._params, non_blocking=True)
            destination2._params = dataclasses.replace(destination2._params, orig_dtype=dst_orig_dtype)
        return True

    info = getattr(tensor.untyped_storage(), "_comfy_tensor_file_slice", None)
    if info is None:
        return False

    if destination is not None and destination.device.type != "cpu" and destination2 is None:
        destination2 = destination
        destination = None

    file_obj = info.file_ref
    if (file_obj is None
            or (destination is None and destination2 is None)
            or (destination is not None and (destination.device.type != "cpu" or destination.numel() * destination.element_size() < info.size))
            or (destination2 is not None and (destination2.device.type == "cpu" or destination2.numel() * destination2.element_size() < info.size))
            or tensor.numel() * tensor.element_size() != info.size
            or tensor.storage_offset() != 0
            or not tensor.is_contiguous()):
        return False

    if info.size == 0:
        return True

    if destination is None:
        stream_ptr = getattr(stream, "cuda_stream", 0) if stream is not None else 0
        comfy_aimdo.host_buffer.read_file_to_device(file_obj, info.offset, info.size,
                                                    stream_ptr, destination2.data_ptr(),
                                                    destination2.device.index,
                                                    mark_cold=False)
        return True

    hostbuf = getattr(destination.untyped_storage(), "_comfy_hostbuf", None)
    if hostbuf is not None:
        stream_ptr = getattr(stream, "cuda_stream", 0) if stream is not None else 0
        device_ptr = destination2.data_ptr() if destination2 is not None else 0
        with info.lock:
            hostbuf.read_file_slice(file_obj, info.offset, info.size,
                                    offset=destination.data_ptr() - hostbuf.get_raw_address(),
                                    stream=stream_ptr,
                                    device_ptr=device_ptr,
                                    device=None if destination2 is None else destination2.device.index)
        return True

    if not hasattr(file_obj, "seek") or not hasattr(file_obj, "readinto"):
        return False

    buf_type = ctypes.c_ubyte * info.size
    view = memoryview(buf_type.from_address(destination.data_ptr()))

    try:
        with info.lock:
            file_obj.seek(info.offset)
            done = 0
            while done < info.size:
                try:
                    n = file_obj.readinto(view[done:])
                except OSError:
                    return False
                if n <= 0:
                    return False
                done += n
        return True
    finally:
        view.release()

class TensorGeometry(NamedTuple):
    shape: any
    dtype: torch.dtype

    def element_size(self):
        info = torch.finfo(self.dtype) if self.dtype.is_floating_point else torch.iinfo(self.dtype)
        return info.bits // 8

    def numel(self):
        return math.prod(self.shape)

def tensors_to_geometries(tensors, dtype=None):
    geometries = []
    for t in tensors:
        if t is None or isinstance(t, QuantizedTensor):
            geometries.append(t)
            continue
        tdtype = t.dtype
        if hasattr(t, "_model_dtype"):
            tdtype = t._model_dtype
        if dtype is not None:
            tdtype = dtype
        geometries.append(TensorGeometry(shape=t.shape, dtype=tdtype))
    return geometries

def vram_aligned_size(tensor):
    if isinstance(tensor, list):
        return sum([vram_aligned_size(t) for t in tensor])

    if isinstance(tensor, QuantizedTensor):
        inner_tensors, _ = tensor.__tensor_flatten__()
        return vram_aligned_size([ getattr(tensor, attr) for attr in inner_tensors ])

    if tensor is None:
        return 0

    size = tensor.numel() * tensor.element_size()
    aligment_req = 1024
    return (size + aligment_req - 1) // aligment_req * aligment_req

def interpret_gathered_like(tensors, gathered):
    offset = 0
    dest_views = []

    if gathered.dim() != 1 or gathered.element_size() != 1:
        raise ValueError(f"Buffer must be 1D and single-byte (got {gathered.dim()}D {gathered.dtype})")

    for tensor in tensors:

        if tensor is None:
            dest_views.append(None)
            continue

        if isinstance(tensor, QuantizedTensor):
            inner_tensors, qt_ctx = tensor.__tensor_flatten__()
            templates = { attr: getattr(tensor, attr) for attr in inner_tensors }
        else:
            templates = { "data": tensor }

        actuals = {}
        for attr, template in templates.items():
            size = template.numel() * template.element_size()
            if offset + size > gathered.numel():
                raise ValueError(f"Buffer too small: needs {offset + size} bytes, but only has {gathered.numel()}. ")
            actuals[attr] = gathered[offset:offset+size].view(dtype=template.dtype).view(template.shape)
            offset += vram_aligned_size(template)

        if isinstance(tensor, QuantizedTensor):
            dest_views.append(QuantizedTensor.__tensor_unflatten__(actuals, qt_ctx, 0, 0))
        else:
            dest_views.append(actuals["data"])

    return dest_views

aimdo_enabled = False

extra_ram_release_callback = None
RAM_CACHE_HEADROOM = 0

def set_ram_cache_release_state(callback, headroom):
    global extra_ram_release_callback
    global RAM_CACHE_HEADROOM
    extra_ram_release_callback = callback
    RAM_CACHE_HEADROOM = max(0, int(headroom))

def extra_ram_release(target, free_active=False):
    if extra_ram_release_callback is None:
        return 0
    return extra_ram_release_callback(target, free_active=free_active)
