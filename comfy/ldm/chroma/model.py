#Original code can be found on: https://github.com/black-forest-labs/flux

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from einops import rearrange
import comfy.patcher_extension
import comfy.ldm.common_dit

from comfy.ldm.flux.layers import (
    EmbedND,
    timestep_embedding,
    DoubleStreamBlock,
    SingleStreamBlock,
)

from .layers import (
    LastLayer,
    Approximator,
    ChromaModulationOut,
)


@dataclass
class ChromaParams:
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
    patch_size: int
    qkv_bias: bool
    in_dim: int
    out_dim: int
    hidden_dim: int
    n_layers: int
    txt_ids_dims: list
    vec_in_dim: int



class Chroma(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        params = ChromaParams(**kwargs)
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
        self.img_in = operations.Linear(self.in_channels, self.hidden_size, bias=True, dtype=dtype, device=device)
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
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, modulation=False, dtype=dtype, device=device, operations=operations)
                for _ in range(params.depth_single_blocks)
            ]
        )

        if final_layer:
            self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels, dtype=dtype, device=device, operations=operations)

        self.skip_mmdit = []
        self.skip_dit = []
        self.lite = False

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
        transformer_options = transformer_options.copy()
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
        transformer_options["img_slice"] = [txt.shape[1], img.shape[1]]
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
        if hasattr(self, "final_layer"):
            final_mod = self.get_modulations(mod_vectors, "final")
            img = self.final_layer(img, vec=final_mod)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward(self, x, timestep, context, guidance=None, ref_latents=None, control=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, guidance, ref_latents, control, transformer_options, **kwargs)

    def _forward(self, x, timestep, context, guidance=None, ref_latents=None, control=None, transformer_options={}, **kwargs):
        bs, c, h_orig, w_orig = x.shape

        h_len = ((h_orig + (self.patch_size // 2)) // self.patch_size)
        w_len = ((w_orig + (self.patch_size // 2)) // self.patch_size)
        img, img_ids = comfy.ldm.common_dit.process_img(x, patch_size=self.patch_size, transformer_options=transformer_options)
        if img.ndim != 3 or context.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        img_tokens = img.shape[1]
        if ref_latents is not None:
            h = 0
            w = 0
            index = 0
            ref_latents_method = kwargs.get("ref_latents_method", "offset")
            for ref in ref_latents:
                if ref_latents_method == "index":
                    index += 1
                    h_offset = 0
                    w_offset = 0
                elif ref_latents_method == "uxo":
                    index = 0
                    h_offset = h_len * self.patch_size + h
                    w_offset = w_len * self.patch_size + w
                    h += ref.shape[-2]
                    w += ref.shape[-1]
                else:
                    index = 1
                    h_offset = 0
                    w_offset = 0
                    if ref.shape[-2] + h > ref.shape[-1] + w:
                        w_offset = w
                    else:
                        h_offset = h
                    h = max(h, ref.shape[-2] + h_offset)
                    w = max(w, ref.shape[-1] + w_offset)

                kontext, kontext_ids = comfy.ldm.common_dit.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset, patch_size=self.patch_size, transformer_options=transformer_options)
                img = torch.cat([img, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        out = self.forward_orig(img, img_ids, context, txt_ids, timestep, guidance, control, transformer_options, attn_mask=kwargs.get("attention_mask", None))
        out = out[:, :img_tokens]
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=self.patch_size, pw=self.patch_size)[:,:,:h_orig,:w_orig]
