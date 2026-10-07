"""Kandinsky 6 video+audio transformer for ComfyUI."""

import math

import torch
from torch import nn

import comfy.ldm.common_dit
import comfy.model_prefetch
import comfy.ops
import comfy.patcher_extension
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.modules.attention import ComfyAttention

from comfy.ldm.kandinsky5.model import (
    TimeEmbeddings,
    TextEmbeddings,
    VisualEmbeddings,
    Modulation,
    OutLayer,
    TransformerEncoderBlock,
    TransformerDecoderBlock,
    attention,
    apply_gate_sum,
    apply_scale_shift_norm,
    get_shift_scale_gate,
)
from .core_contract import DIT_CONFIG, GENERATION_DEFAULTS


_DEFAULT_PATCH_SIZE = tuple(int(value) for value in DIT_CONFIG["patch_size"])
_DEFAULT_VISUAL_INPUT_DIM = (
    2 * int(DIT_CONFIG["in_visual_dim"]) + 1
) * math.prod(_DEFAULT_PATCH_SIZE)
_DEFAULT_AUDIO_HEAD_DIM = sum(int(value) for value in DIT_CONFIG["axes_dims_a"])
_DEFAULT_ROPE_SCALE = tuple(
    float(value) for value in GENERATION_DEFAULTS["scale_factor"]
)


class AsymCrossAttention(nn.Module):
    def __init__(self, q_dim, head_dim, kv_dim, operation_settings=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        assert q_dim % head_dim == 0
        self.num_heads = q_dim // head_dim
        self.head_dim = head_dim

        operations = operation_settings.get("operations")
        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")

        self.to_query = operations.Linear(q_dim, q_dim, bias=True, device=device, dtype=dtype)
        self.to_key = operations.Linear(kv_dim, q_dim, bias=True, device=device, dtype=dtype)
        self.to_value = operations.Linear(kv_dim, q_dim, bias=True, device=device, dtype=dtype)
        self.query_norm = operations.RMSNorm(head_dim, device=device, dtype=dtype)
        self.key_norm = operations.RMSNorm(head_dim, device=device, dtype=dtype)
        self.out_layer = operations.Linear(q_dim, q_dim, bias=True, device=device, dtype=dtype)

    def forward(self, x, context, rope_q=None, rope_kv=None, transformer_options={}):
        q = self.to_query(x).view(*x.shape[:-1], self.num_heads, -1)
        k = self.to_key(context).view(*context.shape[:-1], self.num_heads, -1)
        v = self.to_value(context).view(*context.shape[:-1], self.num_heads, -1)
        q = self.query_norm(q)
        k = self.key_norm(k)
        if rope_q is not None:
            q = apply_rope1(q, rope_q)
        if rope_kv is not None:
            k = apply_rope1(k, rope_kv)
        out = attention(q, k, v, self.num_heads,
                        preferred_attention=self.comfy_attention,
                        transformer_options=transformer_options)
        return self.out_layer(out)


class OutLayerAudio(nn.Module):
    """Audio output head."""

    def __init__(self, model_dim_a, time_dim_a, in_audio_dim, operation_settings=None):
        super().__init__()
        self.modulation = Modulation(time_dim_a, model_dim_a, 2, operation_settings=operation_settings)
        operations = operation_settings.get("operations")
        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")
        self.norm = operations.LayerNorm(model_dim_a, elementwise_affine=False, device=device, dtype=dtype)
        self.out_layer = operations.Linear(model_dim_a, in_audio_dim, bias=True, device=device, dtype=dtype)

    def forward(self, audio_embed, time_embed):
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        audio_embed = apply_scale_shift_norm(
            self.norm, audio_embed, scale.unsqueeze(1), shift.unsqueeze(1)
        )
        audio_embed = self.norm(audio_embed)
        return self.out_layer(audio_embed)


class FusedTransformerDecoderBlock(nn.Module):
    """One fused video+audio block."""

    def __init__(self, model_dim, time_dim, ff_dim, head_dim,
                 model_dim_a, time_dim_a, ff_dim_a, head_dim_a,
                 cross_gates=False, fix_modulation=False, ca_rope=False,
                 operation_settings=None):
        super().__init__()
        self.model_dim = model_dim
        self.model_dim_a = model_dim_a
        self.cross_gates = cross_gates
        self.fix_modulation = fix_modulation
        self.ca_rope = ca_rope

        self.videoT = TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim,
                                              operation_settings=operation_settings)
        self.audioT = TransformerDecoderBlock(model_dim_a, time_dim_a, ff_dim_a, head_dim_a,
                                              operation_settings=operation_settings)

        self.va_cross_attention = AsymCrossAttention(model_dim, head_dim, model_dim_a,
                                                     operation_settings=operation_settings)
        self.av_cross_attention = AsymCrossAttention(model_dim_a, head_dim_a, model_dim,
                                                     operation_settings=operation_settings)

        if not cross_gates:
            self.va_modulation = Modulation(time_dim, model_dim, 3, operation_settings=operation_settings)
            self.av_modulation = Modulation(time_dim_a, model_dim_a, 3, operation_settings=operation_settings)
        else:
            self.va_modulation = Modulation(time_dim, model_dim * 2 + model_dim_a, 1, operation_settings=operation_settings)
            self.av_modulation = Modulation(time_dim_a, model_dim_a * 2 + model_dim, 1, operation_settings=operation_settings)

        operations = operation_settings.get("operations")
        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")
        self.va_normalization = operations.LayerNorm(model_dim, elementwise_affine=False, device=device, dtype=dtype)
        self.av_normalization = operations.LayerNorm(model_dim_a, elementwise_affine=False, device=device, dtype=dtype)

    def forward(self, visual_embed, audio_embed, video_text_embed, audio_text_embed,
                video_time_embed, audio_time_embed, visual_rope, audio_rope,
                transformer_options={}):
        v_self, v_cross, v_ff = torch.chunk(
            self.videoT.visual_modulation(video_time_embed),
            3,
            dim=-1,
        )
        shift, scale, gate = get_shift_scale_gate(v_self)
        out = apply_scale_shift_norm(
            self.videoT.self_attention_norm, visual_embed, scale, shift
        )
        out = self.videoT.self_attention(out, visual_rope, transformer_options=transformer_options)
        visual_embed = apply_gate_sum(visual_embed, out, gate)

        shift, scale, gate_v_cross = get_shift_scale_gate(v_cross)
        visual_out_t = apply_scale_shift_norm(
            self.videoT.cross_attention_norm, visual_embed, scale, shift
        )
        visual_before = visual_out_t
        visual_out_t = self.videoT.cross_attention(visual_out_t, video_text_embed,
                                                   transformer_options=transformer_options)

        a_self, a_cross, a_ff = torch.chunk(
            self.audioT.visual_modulation(audio_time_embed),
            3,
            dim=-1,
        )
        shift, scale, gate = get_shift_scale_gate(a_self)
        out = apply_scale_shift_norm(
            self.audioT.self_attention_norm, audio_embed, scale, shift
        )
        out = self.audioT.self_attention(out, audio_rope, transformer_options=transformer_options)
        audio_embed = apply_gate_sum(audio_embed, out, gate)

        shift, scale, gate_a_cross = get_shift_scale_gate(a_cross)
        audio_out = apply_scale_shift_norm(
            self.audioT.cross_attention_norm, audio_embed, scale, shift
        )
        audio_before = audio_out

        audio_out_t = self.audioT.cross_attention(audio_out, audio_text_embed,
                                                  transformer_options=transformer_options)
        audio_embed = apply_gate_sum(audio_embed, audio_out_t, gate_a_cross)

        va_input = video_time_embed if self.fix_modulation else audio_time_embed
        av_input = audio_time_embed if self.fix_modulation else video_time_embed
        if not self.cross_gates:
            va_shift, va_scale, va_gate = get_shift_scale_gate(
                self.va_modulation(va_input)
            )
            av_shift, av_scale, av_gate = get_shift_scale_gate(
                self.av_modulation(av_input)
            )
        else:
            va_shift, va_scale, va_gate = torch.split(
                self.va_modulation(va_input),
                [self.model_dim, self.model_dim, self.model_dim_a],
                dim=-1,
            )
            av_shift, av_scale, av_gate = torch.split(
                self.av_modulation(av_input),
                [self.model_dim_a, self.model_dim_a, self.model_dim],
                dim=-1,
            )
            # Unlike get_shift_scale_gate(), torch.split() leaves these as
            # [B, D]. Add the token axis explicitly: broadcasting [B, D]
            # against [B, L, D] silently becomes [B, B, D] when CFG batches
            # conditional and unconditional samples together (B > 1).
            va_shift, va_scale, va_gate = (
                value.unsqueeze(1) for value in (va_shift, va_scale, va_gate)
            )
            av_shift, av_scale, av_gate = (
                value.unsqueeze(1) for value in (av_shift, av_scale, av_gate)
            )

        visual_embed = apply_gate_sum(visual_embed, visual_out_t, gate_v_cross)

        visual_out_a = apply_scale_shift_norm(
            self.va_normalization, visual_embed, va_scale, va_shift
        )
        audio_out_v = apply_scale_shift_norm(
            self.av_normalization, audio_embed, av_scale, av_shift
        )

        rope_q_va = visual_rope if self.ca_rope else None
        rope_kv_va = audio_rope if self.ca_rope else None
        rope_q_av = audio_rope if self.ca_rope else None
        rope_kv_av = visual_rope if self.ca_rope else None

        visual_out_a = self.va_cross_attention(visual_out_a, audio_before, rope_q=rope_q_va, rope_kv=rope_kv_va,
                                               transformer_options=transformer_options)
        audio_out_v = self.av_cross_attention(audio_out_v, visual_before, rope_q=rope_q_av, rope_kv=rope_kv_av,
                                              transformer_options=transformer_options)

        va_gate_applied = av_gate if self.cross_gates else va_gate
        av_gate_applied = va_gate if self.cross_gates else av_gate
        visual_embed = apply_gate_sum(
            visual_embed, visual_out_a, va_gate_applied
        )
        audio_embed = apply_gate_sum(
            audio_embed, audio_out_v, av_gate_applied
        )

        shift, scale, gate = get_shift_scale_gate(v_ff)
        out = apply_scale_shift_norm(
            self.videoT.feed_forward_norm, visual_embed, scale, shift
        )
        visual_embed = apply_gate_sum(
            visual_embed, self.videoT.feed_forward(out), gate
        )

        shift, scale, gate = get_shift_scale_gate(a_ff)
        out = apply_scale_shift_norm(
            self.audioT.feed_forward_norm, audio_embed, scale, shift
        )
        audio_embed = apply_gate_sum(
            audio_embed, self.audioT.feed_forward(out), gate
        )

        return visual_embed, audio_embed


class Kandinsky6(nn.Module):
    def __init__(
        self,
        in_visual_dim=int(DIT_CONFIG["in_visual_dim"]),
        out_visual_dim=int(DIT_CONFIG["out_visual_dim"]),
        in_audio_dim=int(DIT_CONFIG["in_audio_dim"]),
        out_audio_dim=None,
        n_grid=1,
        in_text_dim=int(DIT_CONFIG["in_text_dim"]),
        in_text_dim2=int(DIT_CONFIG["in_text_dim2"]),
        time_dim=int(DIT_CONFIG["time_dim"]),
        model_dim=int(DIT_CONFIG["model_dim"]),
        ff_dim=int(DIT_CONFIG["ff_dim"]),
        visual_embed_dim=_DEFAULT_VISUAL_INPUT_DIM,
        time_dim_a=int(DIT_CONFIG["time_dim_a"]),
        model_dim_a=int(DIT_CONFIG["model_dim_a"]),
        ff_dim_a=int(DIT_CONFIG["ff_dim_a"]),
        head_dim_a=_DEFAULT_AUDIO_HEAD_DIM,
        patch_size=_DEFAULT_PATCH_SIZE,
        num_text_blocks=int(DIT_CONFIG["num_text_blocks"]),
        num_visual_blocks=int(DIT_CONFIG["num_visual_blocks"]),
        axes_dims=tuple(int(value) for value in DIT_CONFIG["axes_dims"]),
        rope_scale_factor=_DEFAULT_ROPE_SCALE,
        freqs_scaling=float(DIT_CONFIG["audio_freqs_scaling"]),
        cross_gates=bool(DIT_CONFIG["cross_gates"]),
        fix_modulation=bool(DIT_CONFIG["fix_modulation"]),
        ca_rope=bool(DIT_CONFIG["ca_rope"]),
        visual_token_type_num_embeddings=int(
            DIT_CONFIG["visual_token_type_num_embeddings"]
        ),
        dtype=None, device=None, operations=None, **kwargs
    ):
        super().__init__()
        head_dim = sum(axes_dims)
        self.patch_size = patch_size
        self.model_dim = model_dim
        self.in_visual_dim = in_visual_dim
        self.in_audio_dim = in_audio_dim
        # A distilled release expands both output heads into n_grid DX grids;
        # the forward collapses them to a single velocity for standard sampling.
        self.n_grid = int(n_grid)
        self.out_audio_dim = in_audio_dim if out_audio_dim is None else int(out_audio_dim)
        self.rope_scale_factor = rope_scale_factor
        self.freqs_scaling = float(freqs_scaling)
        self.visual_token_type_num_embeddings = int(
            visual_token_type_num_embeddings or 0
        )
        self.dtype = dtype
        self.device = device
        op = {"operations": operations, "device": device, "dtype": dtype}

        self.visual_embeddings = VisualEmbeddings(visual_embed_dim, model_dim, patch_size, operation_settings=op)
        if self.visual_token_type_num_embeddings > 0:
            self.visual_token_type_embeddings = operations.Embedding(
                self.visual_token_type_num_embeddings,
                model_dim,
                device=device,
                dtype=dtype,
            )
        self.video_time_embeddings = TimeEmbeddings(model_dim, time_dim, operation_settings=op)
        self.video_text_embeddings = TextEmbeddings(in_text_dim, model_dim, operation_settings=op)
        self.video_pooled_text_embeddings = TextEmbeddings(in_text_dim2, time_dim, operation_settings=op)
        self.video_text_transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(model_dim, time_dim, ff_dim, head_dim, operation_settings=op)
             for _ in range(num_text_blocks)]
        )
        self.out_layer = OutLayer(model_dim, time_dim, out_visual_dim, patch_size, operation_settings=op)

        self.audio_embeddings = TextEmbeddings(in_audio_dim, model_dim_a, operation_settings=op)
        self.audio_time_embeddings = TimeEmbeddings(model_dim_a, time_dim_a, operation_settings=op)
        self.audio_text_embeddings = TextEmbeddings(in_text_dim, model_dim_a, operation_settings=op)
        self.audio_pooled_text_embeddings = TextEmbeddings(in_text_dim2, time_dim_a, operation_settings=op)
        self.audio_text_transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(model_dim_a, time_dim_a, ff_dim_a, head_dim_a, operation_settings=op)
             for _ in range(num_text_blocks)]
        )
        self.audio_outLayer = OutLayerAudio(model_dim_a, time_dim_a, self.out_audio_dim, operation_settings=op)

        self.visual_blocks = nn.ModuleList(
            [FusedTransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim,
                                          model_dim_a, time_dim_a, ff_dim_a, head_dim_a,
                                          cross_gates=cross_gates, fix_modulation=fix_modulation,
                                          ca_rope=ca_rope, operation_settings=op)
             for _ in range(num_visual_blocks)]
        )

        self.rope_embedder_3d = EmbedND(dim=head_dim, theta=10000.0, axes_dim=list(axes_dims))
        self.rope_embedder_1d = EmbedND(dim=head_dim, theta=10000.0, axes_dim=[head_dim])
        self.audio_rope_embedder_1d = EmbedND(dim=head_dim_a, theta=10000.0, axes_dim=[head_dim_a])

    def rope_encode_3d(self, t, h, w, device=None, dtype=None):
        patch_size = self.patch_size
        t_len = (t + (patch_size[0] // 2)) // patch_size[0]
        h_len = (h + (patch_size[1] // 2)) // patch_size[1]
        w_len = (w + (patch_size[2] // 2)) // patch_size[2]
        steps_t, steps_h, steps_w = t_len, h_len, w_len

        sf = self.rope_scale_factor
        t_len = (t_len - 1.0) / sf[0] + 1.0
        h_len = (h_len - 1.0) / sf[1] + 1.0
        w_len = (w_len - 1.0) / sf[2] + 1.0

        # Positions and trigonometry must not inherit the bf16 latent dtype.
        position_dtype = torch.float32
        img_ids = torch.zeros(
            (steps_t, steps_h, steps_w, 3), device=device, dtype=position_dtype
        )
        img_ids[:, :, :, 0] += torch.linspace(
            0, t_len - 1, steps=steps_t, device=device, dtype=position_dtype
        ).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] += torch.linspace(
            0, h_len - 1, steps=steps_h, device=device, dtype=position_dtype
        ).reshape(1, -1, 1)
        img_ids[:, :, :, 2] += torch.linspace(
            0, w_len - 1, steps=steps_w, device=device, dtype=position_dtype
        ).reshape(1, 1, -1)
        img_ids = img_ids.reshape(1, -1, img_ids.shape[-1])
        return self.rope_embedder_3d(img_ids).movedim(1, 2)

    def rope_encode_1d(self, embedder, seq_len, scale=1.0, device=None, dtype=None):
        ids = (
            torch.arange(seq_len, device=device, dtype=torch.float32) * scale
        ).reshape(1, -1, 1)
        return embedder(ids).movedim(1, 2)

    def forward(self, *args, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                kwargs.get("transformer_options", {}),
            ),
        ).execute(*args, **kwargs)

    def _forward(self, x_V, x_A, timestep, context, pooled,
                 context_a=None, pooled_a=None, freqs_scaling=None,
                 visual_token_type_ids=None,
                 visual_reference_tail=False,
                 transformer_options={}, **kwargs):
        if isinstance(timestep, (list, tuple)):
            t_V, t_A = timestep
        else:
            t_V = t_A = timestep
        if context_a is None:
            context_a = context
        if pooled_a is None:
            pooled_a = pooled
        dtype = x_V.dtype
        if freqs_scaling is None:
            freqs_scaling = self.freqs_scaling

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})

        tev = self.video_time_embeddings(t_V, dtype) + self.video_pooled_text_embeddings(pooled)
        tea = self.audio_time_embeddings(t_A, dtype) + self.audio_pooled_text_embeddings(pooled_a)

        _, _, t_len, h, w = x_V.shape
        x_V = comfy.ldm.common_dit.pad_to_patch_size(x_V, self.patch_size)
        visual_embed = self.visual_embeddings(x_V)
        if (
            hasattr(self, "visual_token_type_embeddings")
            and visual_token_type_ids is not None
        ):
            type_ids = visual_token_type_ids.to(device=visual_embed.device)
            if type_ids.ndim == 1:
                if type_ids.shape[0] != visual_embed.shape[1]:
                    raise ValueError(
                        "visual_token_type_ids length must match visual latent "
                        f"frames: type_ids={tuple(type_ids.shape)}, "
                        f"visual={tuple(visual_embed.shape)}"
                    )
                type_embed = self.visual_token_type_embeddings(type_ids)[
                    None, :, None, None, :
                ]
            elif type_ids.ndim == 2:
                if tuple(type_ids.shape) != tuple(visual_embed.shape[:2]):
                    raise ValueError(
                        "batched visual_token_type_ids must match batch and "
                        f"frames: type_ids={tuple(type_ids.shape)}, "
                        f"visual={tuple(visual_embed.shape)}"
                    )
                type_embed = self.visual_token_type_embeddings(type_ids)[
                    :, :, None, None, :
                ]
            else:
                raise ValueError(
                    "visual_token_type_ids must have shape (T,) or (B, T)"
                )
            visual_embed = visual_embed + type_embed.to(dtype=visual_embed.dtype)
        visual_shape = visual_embed.shape[:-1]
        visual_embed = visual_embed.flatten(1, -2)
        if visual_reference_tail:
            if t_len < 2:
                raise ValueError(
                    "Kandinsky 6 I2VA reference-tail RoPE needs at least two frames."
                )
            rope_v = self.rope_encode_3d(
                t_len - 1, h, w, device=x_V.device, dtype=dtype
            )
            tokens_per_frame = visual_shape[2] * visual_shape[3]
            rope_v = torch.cat(
                (rope_v, rope_v[:, :tokens_per_frame]), dim=1
            )
        else:
            rope_v = self.rope_encode_3d(
                t_len, h, w, device=x_V.device, dtype=dtype
            )

        audio_embed = self.audio_embeddings(x_A)
        rope_a = self.rope_encode_1d(self.audio_rope_embedder_1d, x_A.shape[1],
                                     scale=freqs_scaling, device=x_V.device, dtype=dtype)

        ctx_v = self.video_text_embeddings(context)
        rope_text_v = self.rope_encode_1d(
            self.rope_embedder_1d,
            context.shape[1],
            device=x_V.device,
            dtype=dtype,
        )
        for block in self.video_text_transformer_blocks:
            ctx_v = block(ctx_v, tev, rope_text_v, transformer_options=transformer_options)

        ctx_a = self.audio_text_embeddings(context_a)
        rope_text_a = self.rope_encode_1d(
            self.audio_rope_embedder_1d,
            context_a.shape[1],
            device=x_V.device,
            dtype=dtype,
        )
        for block in self.audio_text_transformer_blocks:
            ctx_a = block(ctx_a, tea, rope_text_a, transformer_options=transformer_options)

        transformer_options["total_blocks"] = len(self.visual_blocks)
        prefetch = comfy.model_prefetch.make_prefetch_queue(
            list(self.visual_blocks), x_V.device, transformer_options
        )
        for i, block in enumerate(self.visual_blocks):
            transformer_options["block_index"] = i
            comfy.model_prefetch.prefetch_queue_pop(prefetch, x_V.device, block, dtype)
            if ("double_block", i) in blocks_replace:
                def block_wrap(args, _block=block):
                    v, a = _block(args["visual"], args["audio"], args["ctx_v"], args["ctx_a"],
                                  args["tev"], args["tea"], args["rope_v"], args["rope_a"],
                                  transformer_options=args.get("transformer_options"))
                    return {"visual": v, "audio": a}
                out = blocks_replace[("double_block", i)](
                    {"visual": visual_embed, "audio": audio_embed, "ctx_v": ctx_v, "ctx_a": ctx_a,
                     "tev": tev, "tea": tea, "rope_v": rope_v, "rope_a": rope_a,
                     "transformer_options": transformer_options},
                    {"original_block": block_wrap},
                )
                visual_embed, audio_embed = out["visual"], out["audio"]
            else:
                visual_embed, audio_embed = block(
                    visual_embed, audio_embed, ctx_v, ctx_a, tev, tea, rope_v, rope_a,
                    transformer_options=transformer_options,
                )
        comfy.model_prefetch.prefetch_queue_pop(prefetch, x_V.device, None)

        visual_embed = visual_embed.reshape(*visual_shape, -1)
        v_out = self.out_layer(visual_embed, tev)
        # Drop the patch padding added before the visual embeddings.
        v_out = v_out[:, :, :t_len, :h, :w]
        a_out = self.audio_outLayer(audio_embed, tea)
        return v_out, a_out


class Kandinsky6NativeAVDiT(Kandinsky6):
    """Adapt ComfyUI's generic multimodal call to the canonical K6 call.

    ComfyUI unpacks a nested ``LATENT`` before calling the diffusion model, so
    ``x`` is a two-item sequence containing the video and audio streams. A
    noisy video latent gets a zero visual condition plus mask appended; for
    I2VA the ``k6_reference`` conditioning carries the encoded reference,
    which is appended as a clean tail frame and cropped from the output.
    """

    def _legacy_forward(self, *args, **kwargs):
        return super()._forward(*args, **kwargs)

    def _forward(
        self,
        x,
        timestep,
        context,
        y=None,
        control=None,
        transformer_options=None,
        freqs_scaling=None,
        k6_reference=None,
        collapse_grids=True,
        **kwargs,
    ):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(
                "Kandinsky 6 expects a joint LATENT with video and audio streams."
            )

        video, audio = x
        if video.ndim != 5 or video.shape[1] != self.in_visual_dim:
            raise ValueError(
                "Kandinsky 6 expects video latents shaped "
                f"[B, {self.in_visual_dim}, T, H, W], got {tuple(video.shape)}."
            )
        if audio.ndim != 3 or audio.shape[-1] != self.in_audio_dim:
            raise ValueError(
                "Kandinsky 6 expects audio latents shaped "
                f"[B, T, {self.in_audio_dim}], got {tuple(audio.shape)}."
            )
        if video.shape[0] != audio.shape[0]:
            raise ValueError("Kandinsky 6 video and audio batch sizes must match.")
        if y is None:
            raise ValueError("Kandinsky 6 requires CLIP-L pooled conditioning.")

        # For I2VA the clean reference frame is appended as a tail inside the
        # model and cropped from the output, so the sampler only ever sees the
        # generated latent.
        reference_tail = k6_reference is not None
        visual_token_type_ids = None
        if reference_tail:
            if self.visual_token_type_num_embeddings < 2:
                raise ValueError(
                    "Kandinsky 6 I2VA requires two visual token-type embeddings."
                )
            if k6_reference.ndim != 5 or k6_reference.shape[1] != self.in_visual_dim:
                raise ValueError(
                    "Kandinsky 6 I2VA expects a reference latent shaped "
                    f"[B, {self.in_visual_dim}, T, H, W], got {tuple(k6_reference.shape)}."
                )
            if k6_reference.shape[0] == 1 and video.shape[0] != 1:
                k6_reference = k6_reference.expand(video.shape[0], -1, -1, -1, -1)
            if k6_reference.shape[0] != video.shape[0]:
                raise ValueError(
                    "Kandinsky 6 I2VA reference batch size must match the video "
                    f"batch: reference={k6_reference.shape[0]}, video={video.shape[0]}."
                )
            generated_frames = video.shape[2]
            video = torch.cat((video, k6_reference), dim=2)
            visual_token_type_ids = torch.zeros(
                video.shape[2], dtype=torch.long, device=video.device
            )
            visual_token_type_ids[-1] = 1

        visual_cond = torch.zeros_like(video)
        visual_mask = torch.zeros_like(video[:, :1])
        if reference_tail:
            visual_mask[:, :, -1] = 1
        video_input = torch.cat((video, visual_cond, visual_mask), dim=1)
        transformer_options = {} if transformer_options is None else transformer_options

        v_out, a_out = self._legacy_forward(
            video_input,
            audio,
            (timestep, timestep),
            context,
            y,
            context_a=context,
            pooled_a=y,
            freqs_scaling=freqs_scaling,
            visual_token_type_ids=visual_token_type_ids,
            visual_reference_tail=reference_tail,
            transformer_options=transformer_options,
            **kwargs,
        )
        if reference_tail:
            v_out = v_out[:, :, :generated_frames]
        # The native distilled output is the n_grid velocity grids. Standard
        # samplers consume one velocity, so collapse by default; the PiFlow
        # guider passes collapse_grids=False to read the raw grids.
        if self.n_grid > 1 and collapse_grids:
            v_out = self._collapse_dx_grids(v_out)
            a_out = self._collapse_dx_grids_audio(a_out)
        return v_out, a_out

    def _collapse_dx_grids(self, v_out):
        # A distilled head emits n_grid velocity grids stacked on the channel
        # axis. Average them into one velocity so a standard flow-matching
        # sampler can consume the output without the PiFlow machinery.
        b, c, t, h, w = v_out.shape
        base_c = c // self.n_grid
        return v_out.view(b, self.n_grid, base_c, t, h, w).mean(dim=1)

    def _collapse_dx_grids_audio(self, a_out):
        b, t, c = a_out.shape
        base_c = c // self.n_grid
        return a_out.view(b, t, self.n_grid, base_c).mean(dim=2)
