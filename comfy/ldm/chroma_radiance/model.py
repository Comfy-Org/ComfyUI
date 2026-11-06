# Credits:
# Original Flux code can be found on: https://github.com/black-forest-labs/flux
# Chroma Radiance adaption referenced from https://github.com/lodestone-rock/flow

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from einops import repeat
import comfy.ldm.common_dit
import comfy.patcher_extension

from comfy.ldm.flux.layers import EmbedND, timestep_embedding, DoubleStreamBlock, SingleStreamBlock

from comfy.ldm.chroma.layers import (
    Approximator,
    ChromaModulationOut,
)
from .layers import (
    NerfEmbedder,
    NerfGLUBlock,
    NerfFinalLayer,
    NerfFinalLayerConv,
)


@dataclass
class ChromaRadianceParams:
    # Fields from ChromaParams (now independent)
    in_channels: int
    out_channels: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list
    theta: int
    qkv_bias: bool
    in_dim: int
    out_dim: int
    hidden_dim: int
    n_layers: int
    txt_ids_dims: list
    vec_in_dim: int
    # ChromaRadiance-specific fields
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

class ChromaRadiance(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        if operations is None:
            raise RuntimeError("Attempt to create ChromaRadiance object without setting operations")
        super().__init__()
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

    def get_modulations(self, tensor: torch.Tensor, block_type: str, *, idx: int = 0):
        # This function slices up the modulations tensor which has the following layout:
        #   single     : num_single_blocks * 3 elements
        #   double_img : num_double_blocks * 6 elements
        #   double_txt : num_double_blocks * 6 elements
        #   final      : 2 elements
        if block_type == "final":
            return (tensor[:, -2:-1, :], tensor[:, -1:, :])
        single_block_count = self.params.depth_single_blocks
        double_block_count = self.params.depth
        offset = 3 * idx
        if block_type == "single":
            return ChromaModulationOut.from_offset(tensor, offset)
        # Double block modulations are 6 elements so we double 3 * idx.
        offset *= 2
        if block_type in {"double_img", "double_txt"}:
            # Advance past the single block modulations.
            offset += 3 * single_block_count
            if block_type == "double_txt":
                # Advance past the double block img modulations.
                offset += 6 * double_block_count
            return (
                ChromaModulationOut.from_offset(tensor, offset),
                ChromaModulationOut.from_offset(tensor, offset + 3),
            )
        raise ValueError("Bad block_type")

    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        guidance: Tensor = None,
        control = None,
        transformer_options={},
        attn_mask: Tensor = None,
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})

        # running on sequences img
        img = self.img_in(img)

        # distilled vector guidance
        mod_index_length = 344
        distill_timestep = timestep_embedding(timesteps.detach().clone(), 16).to(img.device, img.dtype)
        # guidance = guidance *
        distil_guidance = timestep_embedding(guidance.detach().clone(), 16).to(img.device, img.dtype)

        # get all modulation index
        modulation_index = timestep_embedding(torch.arange(mod_index_length, device=img.device), 32).to(img.device, img.dtype)
        # we need to broadcast the modulation index here so each batch has all of the index
        modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1).to(img.device, img.dtype)
        # and we need to broadcast timestep and guidance along too
        timestep_guidance = torch.cat([distill_timestep, distil_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1).to(img.dtype).to(img.device, img.dtype)
        # then and only then we could concatenate it together
        input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1).to(img.device, img.dtype)

        mod_vectors = self.distilled_guidance_layer(input_vec)

        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.double_blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.double_blocks):
            transformer_options["block_index"] = i
            if i not in self.skip_mmdit:
                double_mod = (
                    self.get_modulations(mod_vectors, "double_img", idx=i),
                    self.get_modulations(mod_vectors, "double_txt", idx=i),
                )
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"], out["txt"] = block(img=args["img"],
                                                       txt=args["txt"],
                                                       vec=args["vec"],
                                                       pe=args["pe"],
                                                       attn_mask=args.get("attn_mask"),
                                                       transformer_options=args.get("transformer_options"))
                        return out

                    out = blocks_replace[("double_block", i)]({"img": img,
                                                               "txt": txt,
                                                               "vec": double_mod,
                                                               "pe": pe,
                                                               "attn_mask": attn_mask,
                                                               "transformer_options": transformer_options},
                                                              {"original_block": block_wrap})
                    txt = out["txt"]
                    img = out["img"]
                else:
                    img, txt = block(img=img,
                                     txt=txt,
                                     vec=double_mod,
                                     pe=pe,
                                     attn_mask=attn_mask,
                                     transformer_options=transformer_options)

                if control is not None: # Controlnet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img += add

        img = torch.cat((txt, img), 1)

        transformer_options["total_blocks"] = len(self.single_blocks)
        transformer_options["block_type"] = "single"
        for i, block in enumerate(self.single_blocks):
            transformer_options["block_index"] = i
            if i not in self.skip_dit:
                single_mod = self.get_modulations(mod_vectors, "single", idx=i)
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"],
                                           vec=args["vec"],
                                           pe=args["pe"],
                                           attn_mask=args.get("attn_mask"),
                                           transformer_options=args.get("transformer_options"))
                        return out

                    out = blocks_replace[("single_block", i)]({"img": img,
                                                               "vec": single_mod,
                                                               "pe": pe,
                                                               "attn_mask": attn_mask,
                                                               "transformer_options": transformer_options},
                                                              {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=single_mod, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options)

                if control is not None: # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, txt.shape[1] :, ...] += add

        img = img[:, txt.shape[1] :, ...]
        return img

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

    def forward(self, x, timestep, context, guidance, control=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, guidance, control, transformer_options, **kwargs)

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
