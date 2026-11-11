# Credits:
# Original Flux code can be found on: https://github.com/black-forest-labs/flux
# Chroma Radiance adaption referenced from https://github.com/lodestone-rock/flow

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from einops import repeat
import comfy.ldm.common_dit

from comfy.ldm.flux.layers import EmbedND, DoubleStreamBlock, SingleStreamBlock

from comfy.ldm.chroma.model import Chroma, ChromaParams
from comfy.ldm.chroma.layers import (
    Approximator,
)
from .layers import (
    NerfEmbedder,
    NerfGLUBlock,
    NerfFinalLayer,
    NerfFinalLayerConv,
)


@dataclass
class ChromaRadianceParams(ChromaParams):
    patch_size: int
    nerf_hidden_size: int
    nerf_mlp_ratio: int
    nerf_depth: int
    nerf_max_freqs: int
    # Setting nerf_tile_size to 0 disables tiling.
    nerf_tile_size: int
    # Currently one of linear (legacy) or conv.
    nerf_final_head_type: str
    # None means use the same dtype as the model.
    nerf_embedder_dtype: Optional[torch.dtype]
    use_x0: bool
    # Use sequential txt_ids instead of zeros
    use_sequential_txt_ids: bool

class ChromaRadiance(Chroma):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        if operations is None:
            raise RuntimeError("Attempt to create ChromaRadiance object without setting operations")
        nn.Module.__init__(self)
        self.dtype = dtype
        params = ChromaRadianceParams(**kwargs)
        self.params = params
        self.patch_size = params.patch_size
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.in_dim = params.in_dim
        self.out_dim = params.out_dim
        self.hidden_dim = params.hidden_dim
        self.n_layers = params.n_layers
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in_patch = operations.Conv2d(
            params.in_channels,
            params.hidden_size,
            kernel_size=params.patch_size,
            stride=params.patch_size,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.txt_in = operations.Linear(params.context_in_dim, self.hidden_size, dtype=dtype, device=device)
        # set as nn identity for now, will overwrite it later.
        self.distilled_guidance_layer = Approximator(
                    in_dim=self.in_dim,
                    hidden_dim=self.hidden_dim,
                    out_dim=self.out_dim,
                    n_layers=self.n_layers,
                    dtype=dtype, device=device, operations=operations
                )

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    modulation=False,
                    dtype=dtype, device=device, operations=operations
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    modulation=False,
                    dtype=dtype, device=device, operations=operations,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        # pixel channel concat with DCT
        self.nerf_image_embedder = NerfEmbedder(
            in_channels=params.in_channels,
            hidden_size_input=params.nerf_hidden_size,
            max_freqs=params.nerf_max_freqs,
            dtype=params.nerf_embedder_dtype or dtype,
            device=device,
            operations=operations,
        )

        self.nerf_blocks = nn.ModuleList([
            NerfGLUBlock(
                hidden_size_s=params.hidden_size,
                hidden_size_x=params.nerf_hidden_size,
                mlp_ratio=params.nerf_mlp_ratio,
                dtype=dtype,
                device=device,
                operations=operations,
            ) for _ in range(params.nerf_depth)
        ])

        if params.nerf_final_head_type == "linear":
            self.nerf_final_layer = NerfFinalLayer(
                params.nerf_hidden_size,
                out_channels=params.in_channels,
                dtype=dtype,
                device=device,
                operations=operations,
            )
        elif params.nerf_final_head_type == "conv":
            self.nerf_final_layer_conv = NerfFinalLayerConv(
                params.nerf_hidden_size,
                out_channels=params.in_channels,
                dtype=dtype,
                device=device,
                operations=operations,
            )
        else:
            errstr = f"Unsupported nerf_final_head_type {params.nerf_final_head_type}"
            raise ValueError(errstr)

        self.skip_mmdit = []
        self.skip_dit = []
        self.lite = False

        if params.use_x0:
            self.register_buffer("__x0__", torch.tensor([]))

        if params.use_sequential_txt_ids:
            self.register_buffer("__sequential__", torch.tensor([]))

    @property
    def _nerf_final_layer(self) -> nn.Module:
        if self.params.nerf_final_head_type == "linear":
            return self.nerf_final_layer
        if self.params.nerf_final_head_type == "conv":
            return self.nerf_final_layer_conv
        # Impossible to get here as we raise an error on unexpected types on initialization.
        raise NotImplementedError

    def img_in(self, img: Tensor) -> Tensor:
        img = self.img_in_patch(img) # -> [B, Hidden, H/P, W/P]
        # flatten into a sequence for the transformer.
        return img.flatten(2).transpose(1, 2) # -> [B, NumPatches, Hidden]

    def forward_nerf(
        self,
        img_orig: Tensor,
        img_out: Tensor,
        params: ChromaRadianceParams,
    ) -> Tensor:
        B, C, H, W = img_orig.shape
        num_patches = img_out.shape[1]
        patch_size = params.patch_size

        # Store the raw pixel values of each patch for the NeRF head later.
        # unfold creates patches: [B, C * P * P, NumPatches]
        nerf_pixels = nn.functional.unfold(img_orig, kernel_size=patch_size, stride=patch_size)
        nerf_pixels = nerf_pixels.transpose(1, 2) # -> [B, NumPatches, C * P * P]

        # Reshape for per-patch processing
        nerf_hidden = img_out.reshape(B * num_patches, params.hidden_size)
        nerf_pixels = nerf_pixels.reshape(B * num_patches, C, patch_size**2).transpose(1, 2)

        if params.nerf_tile_size > 0 and num_patches > params.nerf_tile_size:
            # Enable tiling if nerf_tile_size isn't 0 and we actually have more patches than
            # the tile size.
            img_dct = self.forward_tiled_nerf(nerf_hidden, nerf_pixels, B, C, num_patches, patch_size, params)
        else:
            # Get DCT-encoded pixel embeddings [pixel-dct]
            img_dct = self.nerf_image_embedder(nerf_pixels)

            # Pass through the dynamic MLP blocks (the NeRF)
            for block in self.nerf_blocks:
                img_dct = block(img_dct, nerf_hidden)

        # Reassemble the patches into the final image.
        img_dct = img_dct.transpose(1, 2) # -> [B*NumPatches, C, P*P]
        # Reshape to combine with batch dimension for fold
        img_dct = img_dct.reshape(B, num_patches, -1) # -> [B, NumPatches, C*P*P]
        img_dct = img_dct.transpose(1, 2) # -> [B, C*P*P, NumPatches]
        img_dct = nn.functional.fold(
            img_dct,
            output_size=(H, W),
            kernel_size=patch_size,
            stride=patch_size,
        )
        return self._nerf_final_layer(img_dct)

    def forward_tiled_nerf(
        self,
        nerf_hidden: Tensor,
        nerf_pixels: Tensor,
        batch: int,
        channels: int,
        num_patches: int,
        patch_size: int,
        params: ChromaRadianceParams,
    ) -> Tensor:
        """
        Processes the NeRF head in tiles to save memory.
        nerf_hidden has shape [B, L, D]
        nerf_pixels has shape [B, L, C * P * P]
        """
        tile_size = params.nerf_tile_size
        output_tiles = []
        # Iterate over the patches in tiles. The dimension L (num_patches) is at index 1.
        for i in range(0, num_patches, tile_size):
            end = min(i + tile_size, num_patches)

            # Slice the current tile from the input tensors
            nerf_hidden_tile = nerf_hidden[i * batch:end * batch]
            nerf_pixels_tile = nerf_pixels[i * batch:end * batch]

            # get DCT-encoded pixel embeddings [pixel-dct]
            img_dct_tile = self.nerf_image_embedder(nerf_pixels_tile)

            # pass through the dynamic MLP blocks (the NeRF)
            for block in self.nerf_blocks:
                img_dct_tile = block(img_dct_tile, nerf_hidden_tile)

            output_tiles.append(img_dct_tile)

        # Concatenate the processed tiles along the patch dimension
        return torch.cat(output_tiles, dim=0)

    def radiance_get_override_params(self, overrides: dict) -> ChromaRadianceParams:
        params = self.params
        if not overrides:
            return params
        params_dict = {k: getattr(params, k) for k in params.__dataclass_fields__}
        nullable_keys = frozenset(("nerf_embedder_dtype",))
        bad_keys = tuple(k for k in overrides if k not in params_dict)
        if bad_keys:
            e = f"Unknown key(s) in transformer_options chroma_radiance_options: {', '.join(bad_keys)}"
            raise ValueError(e)
        bad_keys = tuple(
            k
            for k, v in overrides.items()
            if not isinstance(v, type(getattr(params, k))) and (v is not None or k not in nullable_keys)
        )
        if bad_keys:
            e = f"Invalid value(s) in transformer_options chroma_radiance_options: {', '.join(bad_keys)}"
            raise ValueError(e)
        # At this point it's all valid keys and values so we can merge with the existing params.
        params_dict |= overrides
        return params.__class__(**params_dict)

    def _apply_x0_residual(self, predicted, noisy, timesteps):

        # non zero during training to prevent 0 div
        eps = 0.0
        return (noisy - predicted) / (timesteps.view(-1,1,1,1) + eps)

    def _forward(
        self,
        x: Tensor,
        timestep: Tensor,
        context: Tensor,
        guidance: Optional[Tensor],
        control: Optional[dict]=None,
        transformer_options: dict={},
        **kwargs: dict,
    ) -> Tensor:
        bs, c, h, w = x.shape
        img = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))

        if img.ndim != 4:
            raise ValueError("Input img tensor must be in [B, C, H, W] format.")
        if context.ndim != 3:
            raise ValueError("Input txt tensors must have 3 dimensions.")

        params = self.radiance_get_override_params(transformer_options.get("chroma_radiance_options", {}))

        h_len = (img.shape[-2] // self.patch_size)
        w_len = (img.shape[-1] // self.patch_size)

        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        # Radiance after 2026-05-22 uses sequential txt_ids instead of zeros
        if params.use_sequential_txt_ids:
            txt_ids[:, :, 0] = torch.arange(context.shape[1], device=x.device, dtype=x.dtype).unsqueeze(0).expand(bs, -1)

        img_out = self.forward_orig(
            img,
            img_ids,
            context,
            txt_ids,
            timestep,
            guidance,
            control,
            transformer_options,
            attn_mask=kwargs.get("attention_mask", None),
        )

        out = self.forward_nerf(img, img_out, params)[:, :, :h, :w]

        # If x0 variant → v-pred, just return this instead
        if hasattr(self, "__x0__"):
            out = self._apply_x0_residual(out, img, timestep)
        return out

