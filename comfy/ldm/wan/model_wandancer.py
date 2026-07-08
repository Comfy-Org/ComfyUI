import torch
import torch.nn as nn
import comfy
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.flux.layers import EmbedND

from .model import AudioInjector_WAN, WanModel, MLPProj, Head, sinusoidal_embedding_1d


class MusicSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, device=None, dtype=None, operations=None):
        assert dim % num_heads == 0
        super().__init__()
        self.embed_dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.k_proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.v_proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.out_proj = operations.Linear(dim, dim, device=device, dtype=dtype)

    def forward(self, x, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.q_proj(x).view(b, s, n, d)
        q = apply_rope1(q, freqs)

        k = self.k_proj(x).view(b, s, n, d)
        k = apply_rope1(k, freqs)

        x = optimized_attention(
            q.view(b, s, n * d),
            k.view(b, s, n * d),
            self.v_proj(x).view(b, s, n * d),
            heads=self.num_heads,
        )

        return self.out_proj(x)


class MusicEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, device=None, dtype=None, operations=None):
        super().__init__()
        self.self_attn = MusicSelfAttention(dim, num_heads, device=device, dtype=dtype, operations=operations)

        self.linear1 = operations.Linear(dim, ffn_dim, device=device, dtype=dtype)
        self.linear2 = operations.Linear(ffn_dim, dim, device=device, dtype=dtype)

        self.norm1 = operations.LayerNorm(dim, device=device, dtype=dtype)
        self.norm2 = operations.LayerNorm(dim, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), freqs=freqs)
        x = x + self.linear2(torch.nn.functional.gelu(self.linear1(self.norm2(x)))) # ffn
        return x


class WanDancerModel(WanModel):
    def __init__(self,
                 model_type='wandancer',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=5120,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=40,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 in_dim_ref_conv=None,
                 image_model=None,
                 device=None, dtype=None, operations=None,
                 audio_inject_layers=[0, 4, 8, 12, 16, 20, 24, 27],
                 music_dim = 256,
                 music_heads = 4,
                 music_feature_dim = 35,
                 music_latent_dim = 256
                 ):

        super().__init__(model_type='i2v', patch_size=patch_size, text_len=text_len, in_dim=in_dim, dim=dim, ffn_dim=ffn_dim, freq_dim=freq_dim, text_dim=text_dim, out_dim=out_dim,
                         num_heads=num_heads, num_layers=num_layers, window_size=window_size, qk_norm=qk_norm, cross_attn_norm=cross_attn_norm, eps=eps, image_model=image_model, in_dim_ref_conv=in_dim_ref_conv,
                         device=device, dtype=dtype, operations=operations)

        self.dtype = dtype
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}

        self.patch_embedding_global = operations.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size, device=operation_settings.get("device"), dtype=torch.float32)
        self.img_emb_refimage = MLPProj(1280, dim, operation_settings=operation_settings)
        self.head_global = Head(dim, out_dim, patch_size, eps, operation_settings=operation_settings)

        self.music_injector = AudioInjector_WAN(
            dim=self.dim,
            num_heads=self.num_heads,
            inject_layer=audio_inject_layers,
            root_net=self,
            enable_adain=False,
            dtype=dtype, device=device, operations=operations
        )

        self.music_projection = operations.Linear(music_feature_dim, music_latent_dim, device=device, dtype=dtype)
        self.music_encoder = nn.ModuleList([MusicEncoderLayer(dim=music_dim, num_heads=music_heads, ffn_dim=1024, device=device, dtype=dtype, operations=operations) for _ in range(2)])
        music_head_dim = music_dim // music_heads
        self.music_rope_embedder = EmbedND(dim=music_head_dim, theta=10000.0, axes_dim=[music_head_dim])

    def forward_orig(self, x, t, context, clip_fea=None, clip_fea_ref=None, freqs=None, audio_embed=None, fps=30, audio_inject_scale=1.0, transformer_options={}, **kwargs):
        # embeddings
        if int(fps + 0.5) != 30:
            x = self.patch_embedding_global(x.float()).to(x.dtype)
        else:
            x = self.patch_embedding(x.float()).to(x.dtype)

        grid_sizes = x.shape[2:]
        latent_frames = grid_sizes[0]
        transformer_options["grid_sizes"] = grid_sizes
        x = x.flatten(2).transpose(1, 2)
        seq_len = x.size(1)

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
        e = e.reshape(t.shape[0], -1, e.shape[-1])
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        full_ref = None
        if self.ref_conv is not None: # model has the weight, but this wasn't used in the original pipeline
            full_ref = kwargs.get("reference_latent", None)
            if full_ref is not None:
                full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
                x = torch.concat((full_ref, x), dim=1)

        # context
        context = self.text_embedding(context)

        audio_emb = None
        if audio_embed is not None: # encode music feature，[1, frame_num, 35] -> [1, F*8, dim]
            music_feature = self.music_projection(audio_embed)

            music_seq_len = music_feature.shape[1]
            music_ids = torch.arange(music_seq_len, device=music_feature.device, dtype=music_feature.dtype).reshape(1, -1, 1) # create 1D position IDs
            music_freqs = self.music_rope_embedder(music_ids).movedim(1, 2)

            # apply encoder layers
            for layer in self.music_encoder:
                music_feature = layer(music_feature, music_freqs)

            # interpolate
            audio_emb = torch.nn.functional.interpolate(music_feature.unsqueeze(1), size=(latent_frames * 8, self.dim), mode='bilinear').squeeze(1)

        context_img_len = 0
        if self.img_emb is not None and clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.cat([context_clip, context], dim=1)
            context_img_len += clip_fea.shape[-2]
        if self.img_emb_refimage is not None and clip_fea_ref is not None:
            context_clip_ref = self.img_emb_refimage(clip_fea_ref)
            context = torch.cat([context_clip_ref, context], dim=1)
            context_img_len += clip_fea_ref.shape[-2]

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len, transformer_options=args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len, transformer_options=transformer_options)
            if audio_emb is not None:
                x = self.music_injector(x, i, audio_emb, audio_emb_global=None, seq_len=seq_len, scale=audio_inject_scale)

        # head
        if int(fps + 0.5) != 30:
            x = self.head_global(x, e)
        else:
            x = self.head(x, e)

        if full_ref is not None:
            x = x[:, full_ref.shape[1]:]

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

    def _forward(self, x, timestep, context, clip_fea=None, time_dim_concat=None, transformer_options={}, clip_fea_ref=None, fps=30, audio_inject_scale=1.0, **kwargs):
        bs, c, t, h, w = x.shape
        x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)

        t_len = t
        if time_dim_concat is not None:
            time_dim_concat = comfy.ldm.common_dit.pad_to_patch_size(time_dim_concat, self.patch_size)
            x = torch.cat([x, time_dim_concat], dim=2)
            t_len = x.shape[2]

        freqs = self.rope_encode(t_len, h, w, device=x.device, dtype=x.dtype, fps=fps, transformer_options=transformer_options)
        return self.forward_orig(x, timestep, context, clip_fea=clip_fea, clip_fea_ref=clip_fea_ref, freqs=freqs, fps=fps, audio_inject_scale=audio_inject_scale, transformer_options=transformer_options, **kwargs)[:, :, :t, :h, :w]

    def rope_encode(self, t, h, w, t_start=0, steps_t=None, steps_h=None, steps_w=None, fps=30, device=None, dtype=None, transformer_options={}):
        patch_size = self.patch_size
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2])

        if steps_t is None:
            steps_t = t_len
        if steps_h is None:
            steps_h = h_len
        if steps_w is None:
            steps_w = w_len

        h_start = 0
        w_start = 0
        rope_options = transformer_options.get("rope_options", None)
        if rope_options is not None:
            t_len = (t_len - 1.0) * rope_options.get("scale_t", 1.0) + 1.0
            h_len = (h_len - 1.0) * rope_options.get("scale_y", 1.0) + 1.0
            w_len = (w_len - 1.0) * rope_options.get("scale_x", 1.0) + 1.0

            t_start += rope_options.get("shift_t", 0.0)
            h_start += rope_options.get("shift_y", 0.0)
            w_start += rope_options.get("shift_x", 0.0)

        img_ids = torch.zeros((steps_t, steps_h, steps_w, 3), device=device, dtype=dtype)

        if int(fps + 0.5) != 30:
            time_scale = 30.0 / fps # how many time units each frame represents relative to 30fps
            positions_new = torch.arange(steps_t, device=device, dtype=dtype) * time_scale + t_start
            total_frames_at_30fps = int(time_scale * steps_t + 0.5)
            positions_new[-1] = t_start + (total_frames_at_30fps - 1)

            img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + positions_new.reshape(-1, 1, 1)
        else:
            img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(t_start, t_start + (t_len - 1), steps=steps_t, device=device, dtype=dtype).reshape(-1, 1, 1)

        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(h_start, h_start + (h_len - 1), steps=steps_h, device=device, dtype=dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(w_start, w_start + (w_len - 1), steps=steps_w, device=device, dtype=dtype).reshape(1, 1, -1)
        img_ids = img_ids.reshape(1, -1, img_ids.shape[-1])

        freqs = self.rope_embedder(img_ids).movedim(1, 2)
        return freqs
