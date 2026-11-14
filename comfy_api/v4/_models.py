from __future__ import annotations

from typing import Literal, TypedDict

import comfy.utils
import folder_paths
import torch

from comfy_api.internal.singleton import ProxiedSingleton
from comfy_api.latest import io

WeightDtype = Literal["float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"]

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


class ModelFile(TypedDict):
    sd: dict[str, torch.Tensor]
    metadata: dict[str, str]


class CheckpointOptions(TypedDict, total=False):
    dtype: WeightDtype


class DiffusionModelOptions(TypedDict, total=False):
    dtype: WeightDtype
    fp8_optimizations: bool


def _model_options(
    options: CheckpointOptions | DiffusionModelOptions | None,
) -> dict[str, object]:
    options = options or {}
    model_options: dict[str, object] = {}
    if "dtype" in options:
        model_options["dtype"] = _DTYPES[options["dtype"]]
    if options.get("fp8_optimizations"):
        model_options["fp8_optimizations"] = True
    return model_options


class Models(ProxiedSingleton):
    """Model files named by folder and file name, as ComfyUI's loader nodes take them.

    A name is relative to its folder: an absolute path or `..` cannot reach outside it. A name
    that is not a file in the folder raises `FileNotFoundError`, except the other names
    `load_vae` takes from VAELoader, such as "pixel_space".
    """

    async def list(self, folder: str) -> list[str]:
        """Return the sorted file names in a model folder such as "loras", subfolders included.

        Only files with an extension ComfyUI accepts for the folder are listed, from every
        directory configured for it. A name in a subfolder uses this machine's separator: `\\` on
        Windows, `/` elsewhere. Raises `KeyError` for a folder ComfyUI does not know.
        """
        return folder_paths.get_filename_list(folder)

    async def read(self, folder: str, name: str) -> ModelFile:
        """Read a file's tensors without building a model.

        Tensors may map the file read-only: clone one before changing it in place.
        Pickle-based formats load with torch's weights-only unpickler; from a pickle, a
        `state_dict` entry, or a lone entry that is a dict, is returned in place of the whole file,
        and non-tensor values are kept. `metadata` is a safetensors file's header metadata, and
        empty for other formats.
        """
        sd, metadata = comfy.utils.load_torch_file(
            folder_paths.get_full_path_or_raise(folder, name),
            safe_load=True,
            return_metadata=True,
        )
        return {"sd": sd, "metadata": metadata or {}}

    async def load_checkpoint(
        self, name: str, options: CheckpointOptions | None = None
    ) -> tuple[io.Model.Type, io.Clip.Type | None, io.Vae.Type]:
        """Load (model, clip, vae) from the checkpoints folder, as CheckpointLoaderSimple does.

        `dtype` sets the stored weight type of the diffusion model only, and is ignored for a file
        that holds nothing but a diffusion model. `clip` is None for a file without a text encoder.
        A file without VAE weights gives a VAE that raises `RuntimeError` when used. Raises
        `RuntimeError` if ComfyUI does not recognize the file as a model.
        """
        import comfy.sd

        model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
            folder_paths.get_full_path_or_raise("checkpoints", name),
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            model_options=_model_options(options),
        )
        return model, clip, vae

    async def load_diffusion_model(
        self, name: str, options: DiffusionModelOptions | None = None
    ) -> io.Model.Type:
        """Load a model from the diffusion_models folder, as UNETLoader does.

        `dtype` sets the stored weight type. `fp8_optimizations` multiplies in fp8, on devices that
        support it, wherever the stored weights are float8_e4m3fn, set by `dtype` or already so in
        the file, as UNETLoader's "fp8_e4m3fn_fast" does. Raises
        `RuntimeError` if ComfyUI does not recognize the file as a diffusion model.
        """
        import comfy.sd

        return comfy.sd.load_diffusion_model(
            folder_paths.get_full_path_or_raise("diffusion_models", name),
            model_options=_model_options(options),
        )

    async def load_vae(self, name: str) -> io.Vae.Type:
        """Load a VAE by a name the VAELoader node offers, as that node does.

        Raises `FileNotFoundError` for a file that is not found (for a tiny VAE, its encoder or
        decoder in `vae_approx`), and `RuntimeError` if ComfyUI does not recognize the weights as a
        VAE.
        """
        from nodes import VAELoader

        try:
            return VAELoader().load_vae(name)[0]
        except StopIteration:
            # VAELoader finds a tiny VAE's two files with next(), which stops when one is missing.
            raise FileNotFoundError(
                f"Model in folder 'vae_approx' with filename '{name}_encoder.*' or "
                f"'{name}_decoder.*' not found."
            ) from None
