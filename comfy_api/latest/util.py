"""Pure helpers for pack code: ``from comfy_api.v4 import util``.

Copies of ComfyUI functions that only do tensor/data work, so a node can use them wherever it runs,
including where ComfyUI's own ``comfy`` modules aren't importable. Each is tested against the
ComfyUI original (tests-unit/comfy_api_test/test_util.py), so the copies can't drift.

For anything that needs the engine (models, devices), use ``eng``. ``folders`` is where ComfyUI keeps
files, for libraries that load or save files themselves.
"""
import itertools
import logging
import math

import numpy as np
import torch
from PIL import Image, ImageFile, UnidentifiedImageError

from ._util_constants import SAMPLER_NAMES, SCHEDULER_NAMES  # generated from comfy.samplers

#: ComfyUI's largest image dimension (``nodes.MAX_RESOLUTION``).
MAX_RESOLUTION = 16384

__all__ = ["MAX_RESOLUTION", "SAMPLER_NAMES", "SCHEDULER_NAMES", "common_upscale", "bislerp", "lanczos",
           "prepare_noise", "conditioning_set_values", "conditioning_set_values_with_timestep_range", "pillow",
           "folders", "tiled_scale", "tiled_scale_multidim", "get_tiled_scale_steps", "load_torch_file",
           "save_torch_file"]


class _Folders:
    """``util.folders``: where ComfyUI keeps files, by path, for libraries that read or write files
    themselves (a detector that loads its own weights, a video encoder). Use it anywhere: at import,
    in a node, in a route handler.

    The names are ``folder_paths``' own; the ones below are documented, and its other public names
    work too, as ComfyUI has them. A node that reads a model or input file as data can use
    ``eng.models.read`` and ``eng.files.read`` instead, and ``eng.output.write`` to save. A model
    folder a pack adds here is in this process; ``ComfyExtension.get_model_folders`` is how ComfyUI
    lists it (the converter generates that from the folders a pack adds when it loads).
    """

    @staticmethod
    def _fp():
        import folder_paths
        return folder_paths

    def __getattr__(self, name):
        if name.startswith("_") and name not in ("__file__",):
            raise AttributeError(name)
        return getattr(self._fp(), name)

    def __setattr__(self, name, value):
        setattr(self._fp(), name, value)

    @property
    def base_path(self) -> str:
        """ComfyUI's folder."""
        return self._fp().base_path

    @property
    def output_directory(self) -> str:
        """The output folder, as ``get_output_directory()``."""
        return self._fp().get_output_directory()

    @property
    def input_directory(self) -> str:
        """The input folder, as ``get_input_directory()``."""
        return self._fp().get_input_directory()

    @property
    def temp_directory(self) -> str:
        """The temp folder, as ``get_temp_directory()``."""
        return self._fp().get_temp_directory()

    @property
    def folder_names_and_paths(self) -> dict:
        """``{model folder: ([directories], {extensions})}``, ComfyUI's own table."""
        return self._fp().folder_names_and_paths

    @property
    def supported_pt_extensions(self) -> set:
        """The file extensions ComfyUI treats as model files."""
        return self._fp().supported_pt_extensions

    @property
    def models_dir(self) -> str:
        """The models folder (``models/``)."""
        return self._fp().models_dir

    def get_input_directory(self) -> str:
        """The input folder."""
        return self._fp().get_input_directory()

    def get_output_directory(self) -> str:
        """The output folder."""
        return self._fp().get_output_directory()

    def get_temp_directory(self) -> str:
        """The temp folder (previews; emptied when ComfyUI starts)."""
        return self._fp().get_temp_directory()

    def get_user_directory(self) -> str:
        """The user folder."""
        return self._fp().get_user_directory()

    def get_directory_by_type(self, type_name: str):
        """``"input"``, ``"output"`` or ``"temp"``'s folder; None for another name."""
        return self._fp().get_directory_by_type(type_name)

    def get_folder_paths(self, folder: str) -> list:
        """The directories of a model folder (``"checkpoints"``, ``"loras"``...)."""
        return list(self._fp().get_folder_paths(folder))

    def get_filename_list(self, folder: str) -> list:
        """The files in a model folder, as model dropdowns list them."""
        return list(self._fp().get_filename_list(folder))

    def get_full_path(self, folder: str, filename: str):
        """A model file's path, or None if it isn't there."""
        return self._fp().get_full_path(folder, filename)

    def get_full_path_or_raise(self, folder: str, filename: str) -> str:
        """A model file's path; FileNotFoundError if it isn't there."""
        return self._fp().get_full_path_or_raise(folder, filename)

    def get_annotated_filepath(self, name: str, default_dir=None) -> str:
        """The path of a file ref such as ``"photo.png"``, ``"a.png [output]"`` or ``"b.png [temp]"``."""
        return self._fp().get_annotated_filepath(name, default_dir)

    def exists_annotated_filepath(self, name: str) -> bool:
        """Whether a file ref names a file that exists."""
        return self._fp().exists_annotated_filepath(name)

    def get_save_image_path(self, filename_prefix: str, output_dir: str, image_width: int = 0, image_height: int = 0):
        """``(folder, filename, counter, subfolder, filename_prefix)`` for saving the next file with
        ComfyUI's naming, as ``SaveImage`` does."""
        return self._fp().get_save_image_path(filename_prefix, output_dir, image_width, image_height)

    def add_model_folder_path(self, folder: str, path: str, is_default: bool = False) -> None:
        """Add a directory to a model folder, in this process (use ``ComfyExtension.get_model_folders``
        to add a folder ComfyUI lists)."""
        self._fp().add_model_folder_path(folder, path, is_default)

    def __repr__(self) -> str:
        return "<util.folders>"


folders = _Folders()


# --- from comfy/utils.py ---------------------------------------------------- #
def bislerp(samples, width, height):
    def slerp(b1, b2, r):
        '''slerps batches b1, b2 according to ratio r, batches should be flat e.g. NxC'''

        c = b1.shape[-1]

        #norms
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        #normalize
        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms

        #zero when norms are zero
        b1_normalized[b1_norms.expand(-1,c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1,c) == 0.0] = 0.0

        #slerp
        dot = (b1_normalized*b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        #technically not mathematically correct, but more pleasing?
        res = (torch.sin((1.0-r.squeeze(1))*omega)/so).unsqueeze(1)*b1_normalized + (torch.sin(r.squeeze(1)*omega)/so).unsqueeze(1) * b2_normalized
        res *= (b1_norms * (1.0-r) + b2_norms * r).expand(-1,c)

        #edge cases for same or polar opposites
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5]
        res[dot < 1e-5 - 1] = (b1 * (1.0-r) + b2 * r)[dot < 1e-5 - 1]
        return res

    def generate_bilinear_data(length_old, length_new, device):
        coords_1 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1,1,1,-1))
        coords_1 = torch.nn.functional.interpolate(coords_1, size=(1, length_new), mode="bilinear")
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)

        coords_2 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1,1,1,-1)) + 1
        coords_2[:,:,:,-1] -= 1
        coords_2 = torch.nn.functional.interpolate(coords_2, size=(1, length_new), mode="bilinear")
        coords_2 = coords_2.to(torch.int64)
        return ratios, coords_1, coords_2

    orig_dtype = samples.dtype
    samples = samples.float()
    n,c,h,w = samples.shape
    h_new, w_new = (height, width)

    #linear w
    ratios, coords_1, coords_2 = generate_bilinear_data(w, w_new, samples.device)
    coords_1 = coords_1.expand((n, c, h, -1))
    coords_2 = coords_2.expand((n, c, h, -1))
    ratios = ratios.expand((n, 1, h, -1))

    pass_1 = samples.gather(-1,coords_1).movedim(1, -1).reshape((-1,c))
    pass_2 = samples.gather(-1,coords_2).movedim(1, -1).reshape((-1,c))
    ratios = ratios.movedim(1, -1).reshape((-1,1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h, w_new, c).movedim(-1, 1)

    #linear h
    ratios, coords_1, coords_2 = generate_bilinear_data(h, h_new, samples.device)
    coords_1 = coords_1.reshape((1,1,-1,1)).expand((n, c, -1, w_new))
    coords_2 = coords_2.reshape((1,1,-1,1)).expand((n, c, -1, w_new))
    ratios = ratios.reshape((1,1,-1,1)).expand((n, 1, -1, w_new))

    pass_1 = result.gather(-2,coords_1).movedim(1, -1).reshape((-1,c))
    pass_2 = result.gather(-2,coords_2).movedim(1, -1).reshape((-1,c))
    ratios = ratios.movedim(1, -1).reshape((-1,1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
    return result.to(orig_dtype)

def lanczos(samples, width, height):
    #the below API is strict and expects grayscale to be squeezed
    if samples.ndim == 4:
        samples = samples.squeeze(1) if samples.shape[1] == 1 else samples.movedim(1, -1)
    images = [Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(t).movedim(-1, 0) if (t := np.array(image).astype(np.float32) / 255.0).ndim == 3 else torch.from_numpy(t) for image in images]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)


def common_upscale(samples, width, height, upscale_method, crop):
        orig_shape = tuple(samples.shape)
        if len(orig_shape) > 4:
            samples = samples.reshape(samples.shape[0], samples.shape[1], -1, samples.shape[-2], samples.shape[-1])
            samples = samples.movedim(2, 1)
            samples = samples.reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])
        if crop == "center":
            old_width = samples.shape[-1]
            old_height = samples.shape[-2]
            old_aspect = old_width / old_height
            new_aspect = width / height
            x = 0
            y = 0
            if old_aspect > new_aspect:
                x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
            s = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
        else:
            s = samples

        if upscale_method == "bislerp":
            out = bislerp(s, width, height)
        elif upscale_method == "lanczos":
            out = lanczos(s, width, height)
        else:
            out = torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

        if len(orig_shape) == 4:
            return out

        out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
        return out.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))


# --- from comfy/sample.py --------------------------------------------------- #
def prepare_noise_inner(latent_image, generator, noise_inds=None):
    if noise_inds is None:
        return torch.randn(latent_image.size(), dtype=torch.float32, layout=latent_image.layout, generator=generator, device="cpu").to(dtype=latent_image.dtype)

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1]+1):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=torch.float32, layout=latent_image.layout, generator=generator, device="cpu").to(dtype=latent_image.dtype)
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    return torch.cat(noises, axis=0)


def prepare_noise(latent_image, seed, noise_inds=None):
    """Random noise for a latent and a seed, as ComfyUI's samplers make it (``comfy.sample.prepare_noise``)."""
    generator = torch.manual_seed(seed)
    if latent_image.is_nested:
        raise TypeError("nested latents (audio+video) need ComfyUI's own prepare_noise; use eng.node.run")
    return prepare_noise_inner(latent_image, generator, noise_inds)


# --- from node_helpers.py --------------------------------------------------- #
def conditioning_set_values(conditioning, values={}, append=False):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            val = values[k]
            if append:
                old_val = n[1].get(k, None)
                if old_val is not None:
                    val = old_val + val

            n[1][k] = val
        c.append(n)

    return c

def conditioning_set_values_with_timestep_range(conditioning, values={}, start_percent=0.0, end_percent=1.0):
    """
    Apply values to conditioning only during [start_percent, end_percent], keeping the
    original conditioning active outside that range. Respects existing per-entry ranges.
    """
    if start_percent > end_percent:
        logging.warning(f"start_percent ({start_percent}) must be <= end_percent ({end_percent})")
        return conditioning

    EPS = 1e-5 # the sampler gates entries with strict > / <, shift boundaries slightly to ensure only one conditioning is active per timestep
    c = []
    for t in conditioning:
        cond_start = t[1].get("start_percent", 0.0)
        cond_end   = t[1].get("end_percent",   1.0)
        intersect_start = max(start_percent, cond_start)
        intersect_end   = min(end_percent,   cond_end)

        if intersect_start >= intersect_end: # no overlap: emit unchanged
            c.append(t)
            continue

        if intersect_start > cond_start: # part before the requested range
            c.extend(conditioning_set_values([t], {"start_percent": cond_start, "end_percent": intersect_start - EPS}))

        c.extend(conditioning_set_values([t], {**values, "start_percent": intersect_start, "end_percent": intersect_end}))

        if intersect_end < cond_end: # part after the requested range
            c.extend(conditioning_set_values([t], {"start_percent": intersect_end + EPS, "end_percent": cond_end}))
    return c


def pillow(fn, arg):
    prev_value = None
    try:
        x = fn(arg)
    except (OSError, UnidentifiedImageError, ValueError): #PIL issues #4472 and #2445, also fixes ComfyUI issue #3416
        prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = fn(arg)
    finally:
        if prev_value is not None:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
    return x


# --- from comfy/utils.py: tiled processing ---------------------------------- #
def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    rows = 1 if height <= tile_y else math.ceil((height - overlap) / (tile_y - overlap))
    cols = 1 if width <= tile_x else math.ceil((width - overlap) / (tile_x - overlap))
    return rows * cols

@torch.inference_mode()
def tiled_scale_multidim(samples, function, tile=(64, 64), overlap=8, upscale_amount=4, out_channels=3, output_device="cpu", downscale=False, index_formulas=None, pbar=None):
    dims = len(tile)

    if not (isinstance(upscale_amount, (tuple, list))):
        upscale_amount = [upscale_amount] * dims

    if not (isinstance(overlap, (tuple, list))):
        overlap = [overlap] * dims

    if index_formulas is None:
        index_formulas = upscale_amount

    if not (isinstance(index_formulas, (tuple, list))):
        index_formulas = [index_formulas] * dims

    def get_upscale(dim, val):
        up = upscale_amount[dim]
        if callable(up):
            return up(val)
        else:
            return up * val

    def get_downscale(dim, val):
        up = upscale_amount[dim]
        if callable(up):
            return up(val)
        else:
            return val / up

    def get_upscale_pos(dim, val):
        up = index_formulas[dim]
        if callable(up):
            return up(val)
        else:
            return up * val

    def get_downscale_pos(dim, val):
        up = index_formulas[dim]
        if callable(up):
            return up(val)
        else:
            return val / up

    if downscale:
        get_scale = get_downscale
        get_pos = get_downscale_pos
    else:
        get_scale = get_upscale
        get_pos = get_upscale_pos

    def mult_list_upscale(a):
        out = []
        for i in range(len(a)):
            out.append(round(get_scale(i, a[i])))
        return out

    output = torch.empty([samples.shape[0], out_channels] + mult_list_upscale(samples.shape[2:]), device=output_device)

    for b in range(samples.shape[0]):
        s = samples[b:b+1]

        # handle entire input fitting in a single tile
        if all(s.shape[d+2] <= tile[d] for d in range(dims)):
            output[b:b+1] = function(s).to(output_device)
            if pbar is not None:
                pbar.update(1)
            continue

        out = output[b:b+1].zero_()
        out_div = torch.zeros([s.shape[0], 1] + mult_list_upscale(s.shape[2:]), device=output_device)

        positions = [range(0, s.shape[d+2] - overlap[d], tile[d] - overlap[d]) if s.shape[d+2] > tile[d] else [0] for d in range(dims)]

        for it in itertools.product(*positions):
            s_in = s
            upscaled = []

            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap[d], it[d]))
                l = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(get_pos(d, pos)))

            ps = function(s_in).to(output_device)
            mask = torch.ones([1, 1] + list(ps.shape[2:]), device=output_device)

            for d in range(2, dims + 2):
                feather = round(get_scale(d - 2, overlap[d - 2]))
                if feather >= mask.shape[d]:
                    continue
                for t in range(feather):
                    a = (t + 1) / feather
                    mask.narrow(d, t, 1).mul_(a)
                    mask.narrow(d, mask.shape[d] - 1 - t, 1).mul_(a)

            o = out
            o_d = out_div
            ps_view = ps
            mask_view = mask
            for d in range(dims):
                l = min(ps_view.shape[d + 2], o.shape[d + 2] - upscaled[d])
                o = o.narrow(d + 2, upscaled[d], l)
                o_d = o_d.narrow(d + 2, upscaled[d], l)
                if l < ps_view.shape[d + 2]:
                    ps_view = ps_view.narrow(d + 2, 0, l)
                    mask_view = mask_view.narrow(d + 2, 0, l)

            o.add_(ps_view * mask_view)
            o_d.add_(mask_view)

            if pbar is not None:
                pbar.update(1)

        out.div_(out_div)
    return output

def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap = 8, upscale_amount = 4, out_channels = 3, output_device="cpu", pbar = None):
    return tiled_scale_multidim(samples, function, (tile_y, tile_x), overlap=overlap, upscale_amount=upscale_amount, out_channels=out_channels, output_device=output_device, pbar=pbar)


# --- from comfy/utils.py: weight files ------------------------------------- #
def load_torch_file(ckpt, safe_load=False, device=None, return_metadata=False):
    """A ``.safetensors`` or torch weight file as ``{name: tensor}`` (and its metadata), as
    ``comfy.utils.load_torch_file`` reads it. For a file of the pack's own; models in ComfyUI's
    folders can be read with ``eng.models.read``."""
    import safetensors
    if device is None:
        device = torch.device("cpu")
    metadata = None
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        with safetensors.safe_open(ckpt, framework="pt", device=device.type) as f:
            sd = {k: f.get_tensor(k) for k in f.keys()}
            if return_metadata:
                metadata = f.metadata()
    else:
        pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        elif len(pl_sd) == 1:
            key = list(pl_sd.keys())[0]
            sd = pl_sd[key]
            if not isinstance(sd, dict):
                sd = pl_sd
        else:
            sd = pl_sd
    return (sd, metadata) if return_metadata else sd


def save_torch_file(sd, ckpt, metadata=None):
    """Save ``{name: tensor}`` as a ``.safetensors`` file."""
    import safetensors.torch
    if metadata is not None:
        safetensors.torch.save_file(sd, ckpt, metadata=metadata)
    else:
        safetensors.torch.save_file(sd, ckpt)
