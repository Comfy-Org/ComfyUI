from typing import Tuple
import torch
import torch.nn as nn
from comfy.ldm.lightricks.model import (
    ADALN_BASE_PARAMS_COUNT,
    ADALN_CROSS_ATTN_PARAMS_COUNT,
    CrossAttention,
    FeedForward,
    AdaLayerNormSingle,
    PixArtAlphaTextProjection,
    NormSingleLinearTextProjection,
    LTXVModel,
    apply_cross_attention_adaln,
    compute_prompt_timestep,
)
from comfy.ldm.lightricks.symmetric_patchifier import AudioPatchifier
from comfy.ldm.lightricks.embeddings_connector import Embeddings1DConnector
import comfy.ldm.common_dit
import comfy.model_prefetch

class CompressedTimestep:
    """Store video timestep embeddings in compressed form using per-frame indexing."""
    __slots__ = ('data', 'batch_size', 'num_frames', 'patches_per_frame', 'feature_dim')

    def __init__(self, tensor: torch.Tensor, patches_per_frame: int, per_frame: bool = False):
        """
        tensor: [batch, num_tokens, feature_dim] (per-token, default) or
                [batch, num_frames, feature_dim] (per_frame=True, already compressed).
        patches_per_frame: spatial patches per frame; pass None to disable compression.
        """
        self.batch_size, n, self.feature_dim = tensor.shape
        if per_frame:
            self.patches_per_frame = patches_per_frame
            self.num_frames = n
            self.data = tensor
        elif patches_per_frame is not None and n >= patches_per_frame and n % patches_per_frame == 0:
            self.patches_per_frame = patches_per_frame
            self.num_frames = n // patches_per_frame
            # All patches in a frame are identical — keep only the first.
            self.data = tensor.view(self.batch_size, self.num_frames, patches_per_frame, self.feature_dim)[:, :, 0, :].contiguous()
        else:
            self.patches_per_frame = 1
            self.num_frames = n
            self.data = tensor

    def expand(self):
        """Expand back to original tensor."""
        if self.patches_per_frame == 1:
            return self.data

        # [batch, frames, feature_dim] -> [batch, frames, patches_per_frame, feature_dim] -> [batch, tokens, feature_dim]
        expanded = self.data.unsqueeze(2).expand(self.batch_size, self.num_frames, self.patches_per_frame, self.feature_dim)
        return expanded.reshape(self.batch_size, -1, self.feature_dim)

    def expand_for_computation(self, scale_shift_table: torch.Tensor, batch_size: int, indices: slice = slice(None, None)):
        """Compute ada values on compressed per-frame data, then expand spatially."""
        num_ada_params = scale_shift_table.shape[0]

        # No compression - compute directly
        if self.patches_per_frame == 1:
            num_tokens = self.data.shape[1]
            dim_per_param = self.feature_dim // num_ada_params
            reshaped = self.data.reshape(batch_size, num_tokens, num_ada_params, dim_per_param)[:, :, indices, :]
            table_values = scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=self.data.device, dtype=self.data.dtype)
            ada_values = (table_values + reshaped).unbind(dim=2)
            return ada_values

        # Compressed: compute on per-frame data then expand spatially
        # Reshape: [batch, frames, feature_dim] -> [batch, frames, num_ada_params, dim_per_param]
        frame_reshaped = self.data.reshape(batch_size, self.num_frames, num_ada_params, -1)[:, :, indices, :]
        table_values = scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(
            device=self.data.device, dtype=self.data.dtype
        )
        frame_ada = (table_values + frame_reshaped).unbind(dim=2)

        # Expand each ada parameter spatially: [batch, frames, dim] -> [batch, frames, patches, dim] -> [batch, tokens, dim]
        return tuple(
            frame_val.unsqueeze(2).expand(batch_size, self.num_frames, self.patches_per_frame, -1)
            .reshape(batch_size, -1, frame_val.shape[-1])
            for frame_val in frame_ada
        )

class BasicAVTransformerBlock(nn.Module):
    def __init__(
        self,
        v_dim,
        a_dim,
        v_heads,
        a_heads,
        vd_head,
        ad_head,
        v_context_dim=None,
        a_context_dim=None,
        attn_precision=None,
        apply_gated_attention=False,
        cross_attention_adaln=False,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()

        self.attn_precision = attn_precision
        self.cross_attention_adaln = cross_attention_adaln

        self.attn1 = CrossAttention(
            query_dim=v_dim,
            heads=v_heads,
            dim_head=vd_head,
            context_dim=None,
            attn_precision=self.attn_precision,
            apply_gated_attention=apply_gated_attention,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.audio_attn1 = CrossAttention(
            query_dim=a_dim,
            heads=a_heads,
            dim_head=ad_head,
            context_dim=None,
            attn_precision=self.attn_precision,
            apply_gated_attention=apply_gated_attention,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.attn2 = CrossAttention(
            query_dim=v_dim,
            context_dim=v_context_dim,
            heads=v_heads,
            dim_head=vd_head,
            attn_precision=self.attn_precision,
            apply_gated_attention=apply_gated_attention,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.audio_attn2 = CrossAttention(
            query_dim=a_dim,
            context_dim=a_context_dim,
            heads=a_heads,
            dim_head=ad_head,
            attn_precision=self.attn_precision,
            apply_gated_attention=apply_gated_attention,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        # Q: Video, K,V: Audio
        self.audio_to_video_attn = CrossAttention(
            query_dim=v_dim,
            context_dim=a_dim,
            heads=a_heads,
            dim_head=ad_head,
            attn_precision=self.attn_precision,
            apply_gated_attention=apply_gated_attention,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        # Q: Audio, K,V: Video
        self.video_to_audio_attn = CrossAttention(
            query_dim=a_dim,
            context_dim=v_dim,
            heads=a_heads,
            dim_head=ad_head,
            attn_precision=self.attn_precision,
            apply_gated_attention=apply_gated_attention,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.ff = FeedForward(
            v_dim, dim_out=v_dim, glu=True, dtype=dtype, device=device, operations=operations
        )
        self.audio_ff = FeedForward(
            a_dim, dim_out=a_dim, glu=True, dtype=dtype, device=device, operations=operations
        )

        num_ada_params = ADALN_CROSS_ATTN_PARAMS_COUNT if cross_attention_adaln else ADALN_BASE_PARAMS_COUNT
        self.scale_shift_table = nn.Parameter(torch.empty(num_ada_params, v_dim, device=device, dtype=dtype))
        self.audio_scale_shift_table = nn.Parameter(
            torch.empty(num_ada_params, a_dim, device=device, dtype=dtype)
        )

        if cross_attention_adaln:
            self.prompt_scale_shift_table = nn.Parameter(torch.empty(2, v_dim, device=device, dtype=dtype))
            self.audio_prompt_scale_shift_table = nn.Parameter(torch.empty(2, a_dim, device=device, dtype=dtype))

        self.scale_shift_table_a2v_ca_audio = nn.Parameter(
            torch.empty(5, a_dim, device=device, dtype=dtype)
        )
        self.scale_shift_table_a2v_ca_video = nn.Parameter(
            torch.empty(5, v_dim, device=device, dtype=dtype)
        )

    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice = slice(None, None)
    ):
        if isinstance(timestep, CompressedTimestep):
            return timestep.expand_for_computation(scale_shift_table, batch_size, indices)

        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        num_scale_shift_values: int = 4,
    ):
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :],
            batch_size,
            scale_shift_timestep,
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :],
            batch_size,
            gate_timestep,
        )

        return (*scale_shift_ada_values, *gate_ada_values)

    def _apply_text_cross_attention(
        self, x, context, attn, scale_shift_table, prompt_scale_shift_table,
        timestep, prompt_timestep, attention_mask, transformer_options,
    ):
        """Apply text cross-attention, with optional ADaLN modulation."""
        if self.cross_attention_adaln:
            shift_q, scale_q, gate = self.get_ada_values(
                scale_shift_table, x.shape[0], timestep, slice(6, 9)
            )
            return apply_cross_attention_adaln(
                x, context, attn, shift_q, scale_q, gate,
                prompt_scale_shift_table, prompt_timestep,
                attention_mask, transformer_options,
            )
        return attn(
            comfy.ldm.common_dit.rms_norm(x), context=context,
            mask=attention_mask, transformer_options=transformer_options,
        )

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor], v_context=None, a_context=None, attention_mask=None, v_timestep=None, a_timestep=None,
        v_pe=None, a_pe=None, v_cross_pe=None, a_cross_pe=None, v_cross_scale_shift_timestep=None, a_cross_scale_shift_timestep=None,
        v_cross_gate_timestep=None, a_cross_gate_timestep=None, transformer_options=None, self_attention_mask=None,
        v_prompt_timestep=None, a_prompt_timestep=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        run_vx = transformer_options.get("run_vx", True)
        run_ax = transformer_options.get("run_ax", True)

        vx, ax = x
        run_ax = run_ax and ax.numel() > 0
        run_a2v = run_vx and transformer_options.get("a2v_cross_attn", True) and ax.numel() > 0
        run_v2a = run_ax and transformer_options.get("v2a_cross_attn", True)

        # video
        if run_vx:
            # video self-attention
            vshift_msa, vscale_msa = (self.get_ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(0, 2)))
            norm_vx = comfy.ldm.common_dit.rms_norm(vx) * (1 + vscale_msa) + vshift_msa
            del vshift_msa, vscale_msa
            attn1_out = self.attn1(norm_vx, pe=v_pe, mask=self_attention_mask, transformer_options=transformer_options)
            del norm_vx
            # video cross-attention
            vgate_msa = self.get_ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(2, 3))[0]
            vx.addcmul_(attn1_out, vgate_msa)
            del vgate_msa, attn1_out
            vx.add_(self._apply_text_cross_attention(
                vx, v_context, self.attn2, self.scale_shift_table,
                getattr(self, 'prompt_scale_shift_table', None),
                v_timestep, v_prompt_timestep, attention_mask, transformer_options,)
            )

        # audio
        if run_ax:
            # audio self-attention
            ashift_msa, ascale_msa = (self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(0, 2)))
            norm_ax = comfy.ldm.common_dit.rms_norm(ax) * (1 + ascale_msa) + ashift_msa
            del ashift_msa, ascale_msa
            attn1_out = self.audio_attn1(norm_ax, pe=a_pe, transformer_options=transformer_options)
            del norm_ax
            # audio cross-attention
            agate_msa = self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(2, 3))[0]
            ax.addcmul_(attn1_out, agate_msa)
            del agate_msa, attn1_out
            ax.add_(self._apply_text_cross_attention(
                ax, a_context, self.audio_attn2, self.audio_scale_shift_table,
                getattr(self, 'audio_prompt_scale_shift_table', None),
                a_timestep, a_prompt_timestep, attention_mask, transformer_options,)
            )

        # video - audio cross attention.
        if run_a2v or run_v2a:
            vx_norm3 = comfy.ldm.common_dit.rms_norm(vx)
            ax_norm3 = comfy.ldm.common_dit.rms_norm(ax)

            # audio to video cross attention
            if run_a2v:
                scale_ca_audio_hidden_states_a2v, shift_ca_audio_hidden_states_a2v = self.get_ada_values(
                    self.scale_shift_table_a2v_ca_audio[:4, :], ax.shape[0], a_cross_scale_shift_timestep)[:2]
                scale_ca_video_hidden_states_a2v_v, shift_ca_video_hidden_states_a2v_v = self.get_ada_values(
                    self.scale_shift_table_a2v_ca_video[:4, :], vx.shape[0], v_cross_scale_shift_timestep)[:2]

                vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_a2v_v) + shift_ca_video_hidden_states_a2v_v
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_a2v) + shift_ca_audio_hidden_states_a2v
                del scale_ca_video_hidden_states_a2v_v, shift_ca_video_hidden_states_a2v_v, scale_ca_audio_hidden_states_a2v, shift_ca_audio_hidden_states_a2v

                a2v_out = self.audio_to_video_attn(vx_scaled, context=ax_scaled, pe=v_cross_pe, k_pe=a_cross_pe, transformer_options=transformer_options)
                del vx_scaled, ax_scaled

                gate_out_a2v = self.get_ada_values(self.scale_shift_table_a2v_ca_video[4:, :], vx.shape[0], v_cross_gate_timestep)[0]
                vx.addcmul_(a2v_out, gate_out_a2v)
                del gate_out_a2v, a2v_out

            # video to audio cross attention
            if run_v2a:
                scale_ca_audio_hidden_states_v2a, shift_ca_audio_hidden_states_v2a = self.get_ada_values(
                    self.scale_shift_table_a2v_ca_audio[:4, :], ax.shape[0], a_cross_scale_shift_timestep)[2:4]
                scale_ca_video_hidden_states_v2a, shift_ca_video_hidden_states_v2a = self.get_ada_values(
                    self.scale_shift_table_a2v_ca_video[:4, :], vx.shape[0], v_cross_scale_shift_timestep)[2:4]

                ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_v2a) + shift_ca_audio_hidden_states_v2a
                vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_v2a) + shift_ca_video_hidden_states_v2a
                del scale_ca_video_hidden_states_v2a, shift_ca_video_hidden_states_v2a, scale_ca_audio_hidden_states_v2a, shift_ca_audio_hidden_states_v2a

                v2a_out = self.video_to_audio_attn(ax_scaled, context=vx_scaled, pe=a_cross_pe, k_pe=v_cross_pe, transformer_options=transformer_options)
                del ax_scaled, vx_scaled

                gate_out_v2a = self.get_ada_values(self.scale_shift_table_a2v_ca_audio[4:, :], ax.shape[0], a_cross_gate_timestep)[0]
                ax.addcmul_(v2a_out, gate_out_v2a)
                del gate_out_v2a, v2a_out

            del vx_norm3, ax_norm3

        # video feedforward
        if run_vx:
            vshift_mlp, vscale_mlp = self.get_ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(3, 5))
            vx_scaled = comfy.ldm.common_dit.rms_norm(vx) * (1 + vscale_mlp) + vshift_mlp
            del vshift_mlp, vscale_mlp

            ff_out = self.ff(vx_scaled)
            del vx_scaled

            vgate_mlp = self.get_ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(5, 6))[0]
            vx.addcmul_(ff_out, vgate_mlp)
            del vgate_mlp, ff_out

        # audio feedforward
        if run_ax:
            ashift_mlp, ascale_mlp = self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(3, 5))
            ax_scaled = comfy.ldm.common_dit.rms_norm(ax) * (1 + ascale_mlp) + ashift_mlp
            del ashift_mlp, ascale_mlp

            ff_out = self.audio_ff(ax_scaled)
            del ax_scaled

            agate_mlp = self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(5, 6))[0]
            ax.addcmul_(ff_out, agate_mlp)
            del agate_mlp, ff_out

        return vx, ax


class LTXAVModel(LTXVModel):
    """LTXAV model for audio-video generation."""

    def __init__(
        self,
        in_channels=128,
        audio_in_channels=128,
        cross_attention_dim=4096,
        audio_cross_attention_dim=2048,
        attention_head_dim=128,
        audio_attention_head_dim=64,
        num_attention_heads=32,
        audio_num_attention_heads=32,
        caption_channels=3840,
        num_layers=48,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[20, 2048, 2048],
        audio_positional_embedding_max_pos=[20],
        causal_temporal_positioning=False,
        vae_scale_factors=(8, 32, 32),
        use_middle_indices_grid=False,
        timestep_scale_multiplier=1000.0,
        av_ca_timestep_scale_multiplier=1.0,
        apply_gated_attention=False,
        caption_proj_before_connector=False,
        cross_attention_adaln=False,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        # Store audio-specific parameters
        self.audio_in_channels = audio_in_channels
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.audio_attention_head_dim = audio_attention_head_dim
        self.audio_num_attention_heads = audio_num_attention_heads
        self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
        self.apply_gated_attention = apply_gated_attention

        # Calculate audio dimensions
        self.audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim
        self.audio_out_channels = audio_in_channels

        # Audio-specific constants
        self.num_audio_channels = 8
        self.audio_frequency_bins = 16

        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier

        super().__init__(
            in_channels=in_channels,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            caption_channels=caption_channels,
            num_layers=num_layers,
            positional_embedding_theta=positional_embedding_theta,
            positional_embedding_max_pos=positional_embedding_max_pos,
            causal_temporal_positioning=causal_temporal_positioning,
            vae_scale_factors=vae_scale_factors,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            caption_proj_before_connector=caption_proj_before_connector,
            cross_attention_adaln=cross_attention_adaln,
            dtype=dtype,
            device=device,
            operations=operations,
            **kwargs,
        )

    def _init_model_components(self, device, dtype, **kwargs):
        """Initialize LTXAV-specific components."""
        # Audio-specific projections
        self.audio_patchify_proj = self.operations.Linear(
            self.audio_in_channels, self.audio_inner_dim, bias=True, dtype=dtype, device=device
        )

        # Audio-specific AdaLN
        audio_embedding_coefficient = ADALN_CROSS_ATTN_PARAMS_COUNT if self.cross_attention_adaln else ADALN_BASE_PARAMS_COUNT
        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=audio_embedding_coefficient,
            use_additional_conditions=False,
            dtype=dtype,
            device=device,
            operations=self.operations,
        )

        if self.cross_attention_adaln:
            self.audio_prompt_adaln_single = AdaLayerNormSingle(
                self.audio_inner_dim,
                embedding_coefficient=2,
                use_additional_conditions=False,
                dtype=dtype,
                device=device,
                operations=self.operations,
            )
        else:
            self.audio_prompt_adaln_single = None

        num_scale_shift_values = 4
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            use_additional_conditions=False,
            embedding_coefficient=num_scale_shift_values,
            dtype=dtype,
            device=device,
            operations=self.operations,
        )
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            use_additional_conditions=False,
            embedding_coefficient=1,
            dtype=dtype,
            device=device,
            operations=self.operations,
        )
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            use_additional_conditions=False,
            embedding_coefficient=num_scale_shift_values,
            dtype=dtype,
            device=device,
            operations=self.operations,
        )
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            use_additional_conditions=False,
            embedding_coefficient=1,
            dtype=dtype,
            device=device,
            operations=self.operations,
        )

        # Audio caption projection
        if self.caption_proj_before_connector:
            if self.caption_projection_first_linear:
                self.audio_caption_projection = NormSingleLinearTextProjection(
                    in_features=self.caption_channels,
                    hidden_size=self.audio_inner_dim,
                    dtype=dtype,
                    device=device,
                    operations=self.operations,
                )
            else:
                self.audio_caption_projection = lambda a: a
        else:
            self.audio_caption_projection = PixArtAlphaTextProjection(
                in_features=self.caption_channels,
                hidden_size=self.audio_inner_dim,
                dtype=dtype,
                device=device,
                operations=self.operations,
            )

        connector_split_rope = kwargs.get("rope_type", "split") == "split"
        connector_gated_attention = kwargs.get("connector_apply_gated_attention", False)
        attention_head_dim = kwargs.get("connector_attention_head_dim", 128)
        num_attention_heads = kwargs.get("connector_num_attention_heads", 30)
        num_layers = kwargs.get("connector_num_layers", 2)

        self.audio_embeddings_connector = Embeddings1DConnector(
            attention_head_dim=kwargs.get("audio_connector_attention_head_dim", attention_head_dim),
            num_attention_heads=kwargs.get("audio_connector_num_attention_heads", num_attention_heads),
            num_layers=num_layers,
            split_rope=connector_split_rope,
            double_precision_rope=True,
            apply_gated_attention=connector_gated_attention,
            dtype=dtype,
            device=device,
            operations=self.operations,
        )

        self.video_embeddings_connector = Embeddings1DConnector(
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            split_rope=connector_split_rope,
            double_precision_rope=True,
            apply_gated_attention=connector_gated_attention,
            dtype=dtype,
            device=device,
            operations=self.operations,
        )

    def preprocess_text_embeds(self, context, unprocessed=False):
        # LTXv2 fully processed context has dimension of self.caption_channels * 2
        # LTXv2.3 fully processed context has dimension of self.cross_attention_dim + self.audio_cross_attention_dim
        if not unprocessed:
            if context.shape[-1] in (self.cross_attention_dim + self.audio_cross_attention_dim, self.caption_channels * 2):
                return context
        if context.shape[-1] == self.cross_attention_dim + self.audio_cross_attention_dim:
            context_vid = context[:, :, :self.cross_attention_dim]
            context_audio = context[:, :, self.cross_attention_dim:]
        else:
            context_vid = context
            context_audio = context
        if self.caption_proj_before_connector:
            context_vid = self.caption_projection(context_vid)
            context_audio = self.audio_caption_projection(context_audio)
        out_vid = self.video_embeddings_connector(context_vid)[0]
        out_audio = self.audio_embeddings_connector(context_audio)[0]
        return torch.concat((out_vid, out_audio), dim=-1)

    def _init_transformer_blocks(self, device, dtype, **kwargs):
        """Initialize transformer blocks for LTXAV."""
        self.transformer_blocks = nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    v_dim=self.inner_dim,
                    a_dim=self.audio_inner_dim,
                    v_heads=self.num_attention_heads,
                    a_heads=self.audio_num_attention_heads,
                    vd_head=self.attention_head_dim,
                    ad_head=self.audio_attention_head_dim,
                    v_context_dim=self.cross_attention_dim,
                    a_context_dim=self.audio_cross_attention_dim,
                    apply_gated_attention=self.apply_gated_attention,
                    cross_attention_adaln=self.cross_attention_adaln,
                    dtype=dtype,
                    device=device,
                    operations=self.operations,
                )
                for _ in range(self.num_layers)
            ]
        )

    def _init_output_components(self, device, dtype):
        """Initialize output components for LTXAV."""
        # Video output components
        super()._init_output_components(device, dtype)
        # Audio output components
        self.audio_scale_shift_table = nn.Parameter(
            torch.empty(2, self.audio_inner_dim, dtype=dtype, device=device)
        )
        self.audio_norm_out = self.operations.LayerNorm(
            self.audio_inner_dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.audio_proj_out = self.operations.Linear(
            self.audio_inner_dim, self.audio_out_channels, dtype=dtype, device=device
        )
        self.a_patchifier = AudioPatchifier(1, start_end=True)

    def separate_audio_and_video_latents(self, x, audio_length):
        """Separate audio and video latents from combined input."""
        # vx = x[:, : self.in_channels]
        # ax = x[:, self.in_channels :]
        #
        # ax = ax.reshape(ax.shape[0], -1)
        # ax = ax[:, : audio_length * self.num_audio_channels * self.audio_frequency_bins]
        #
        # ax = ax.reshape(
        #     ax.shape[0], self.num_audio_channels, audio_length, self.audio_frequency_bins
        # )

        vx = x[0]
        ax = x[1] if len(x) > 1 else torch.zeros(
            (vx.shape[0], self.num_audio_channels, 0, self.audio_frequency_bins),
            device=vx.device, dtype=vx.dtype
        )
        return vx, ax

    def recombine_audio_and_video_latents(self, vx, ax, target_shape=None):
        if ax.numel() == 0:
            return vx
        else:
            return [vx, ax]
        """Recombine audio and video latents for output."""
        # if ax.device != vx.device or ax.dtype != vx.dtype:
        #     logging.warning("Audio and video latents are on different devices or dtypes.")
        #     ax = ax.to(device=vx.device, dtype=vx.dtype)
        #     logging.warning(f"Audio audio latent moved to device: {ax.device}, dtype: {ax.dtype}")
        #
        # ax = ax.reshape(ax.shape[0], -1)
        # # pad to f x h x w of the video latents
        # divisor = vx.shape[-1] * vx.shape[-2] * vx.shape[-3]
        # if target_shape is None:
        #     repetitions = math.ceil(ax.shape[-1] / divisor)
        # else:
        #     repetitions = target_shape[1] - vx.shape[1]
        # padded_len = repetitions * divisor
        # ax = F.pad(ax, (0, padded_len - ax.shape[-1]))
        # ax = ax.reshape(ax.shape[0], -1, vx.shape[-3], vx.shape[-2], vx.shape[-1])
        # return torch.cat([vx, ax], dim=1)

    def _process_input(self, x, keyframe_idxs, denoise_mask, **kwargs):
        """Process input for LTXAV - separate audio and video, then patchify."""
        audio_length = kwargs.get("audio_length", 0)
        # Separate audio and video latents
        vx, ax = self.separate_audio_and_video_latents(x, audio_length)

        has_spatial_mask = False
        if denoise_mask is not None:
            # check if any frame has spatial variation (inpainting)
            for frame_idx in range(denoise_mask.shape[2]):
                frame_mask = denoise_mask[0, 0, frame_idx]
                if frame_mask.numel() > 0 and frame_mask.min() != frame_mask.max():
                    has_spatial_mask = True
                    break

        [vx, v_pixel_coords, additional_args] = super()._process_input(
            vx, keyframe_idxs, denoise_mask, **kwargs
        )
        additional_args["has_spatial_mask"] = has_spatial_mask

        ax, a_latent_coords = self.a_patchifier.patchify(ax)

        # Inject reference audio for ID-LoRA in-context conditioning
        ref_audio = kwargs.get("ref_audio", None)
        ref_audio_seq_len = 0
        if ref_audio is not None:
            ref_tokens = ref_audio["tokens"].to(dtype=ax.dtype, device=ax.device)
            if ref_tokens.shape[0] < ax.shape[0]:
                ref_tokens = ref_tokens.expand(ax.shape[0], -1, -1)
            ref_audio_seq_len = ref_tokens.shape[1]
            B = ax.shape[0]

            # Compute negative temporal positions matching ID-LoRA convention:
            # offset by -(end_of_last_token + time_per_latent) so reference ends just before t=0
            p = self.a_patchifier
            tpl = p.hop_length * p.audio_latent_downsample_factor / p.sample_rate
            ref_start = p._get_audio_latent_time_in_sec(0, ref_audio_seq_len, torch.float32, ax.device)
            ref_end = p._get_audio_latent_time_in_sec(1, ref_audio_seq_len + 1, torch.float32, ax.device)
            time_offset = ref_end[-1].item() + tpl
            ref_start = (ref_start - time_offset).unsqueeze(0).expand(B, -1).unsqueeze(1)
            ref_end = (ref_end - time_offset).unsqueeze(0).expand(B, -1).unsqueeze(1)
            ref_pos = torch.stack([ref_start, ref_end], dim=-1)

            additional_args["ref_audio_seq_len"] = ref_audio_seq_len
            additional_args["target_audio_seq_len"] = ax.shape[1]
            ax = torch.cat([ref_tokens, ax], dim=1)
            a_latent_coords = torch.cat([ref_pos.to(a_latent_coords), a_latent_coords], dim=2)

        ax = self.audio_patchify_proj(ax)

        # additional_args.update({"av_orig_shape": list(x.shape)})
        return [vx, ax], [v_pixel_coords, a_latent_coords], additional_args

    def _prepare_timestep(self, timestep, batch_size, hidden_dtype, **kwargs):
        """Prepare timestep embeddings."""
        grid_mask = kwargs.get("grid_mask", None)
        orig_shape = kwargs.get("orig_shape")
        has_spatial_mask = kwargs.get("has_spatial_mask", None)
        v_patches_per_frame = None
        if not has_spatial_mask and orig_shape is not None and len(orig_shape) == 5:
            v_patches_per_frame = orig_shape[3] * orig_shape[4]

        # Used by compute_prompt_timestep and the audio cross-attention paths.
        timestep_scaled = (timestep[:, grid_mask] if grid_mask is not None else timestep) * self.timestep_scale_multiplier

        # When patches in a frame share a timestep (no spatial mask), project one row per frame instead of one per token
        per_frame_path = v_patches_per_frame is not None and (timestep.numel() // batch_size) % v_patches_per_frame == 0
        if per_frame_path:
            per_frame = timestep.reshape(batch_size, -1, v_patches_per_frame)[:, :, 0]
            if grid_mask is not None:
                # All-or-nothing per frame when has_spatial_mask=False.
                per_frame = per_frame[:, grid_mask[::v_patches_per_frame]]
            ts_input = per_frame * self.timestep_scale_multiplier
        else:
            ts_input = timestep_scaled

        v_timestep, v_embedded_timestep = self.adaln_single(
            ts_input.flatten(),
            {"resolution": None, "aspect_ratio": None},
            batch_size=batch_size,
            hidden_dtype=hidden_dtype,
        )
        v_timestep = CompressedTimestep(v_timestep.view(batch_size, -1, v_timestep.shape[-1]), v_patches_per_frame, per_frame=per_frame_path)
        v_embedded_timestep = CompressedTimestep(v_embedded_timestep.view(batch_size, -1, v_embedded_timestep.shape[-1]), v_patches_per_frame, per_frame=per_frame_path)

        v_prompt_timestep = compute_prompt_timestep(
            self.prompt_adaln_single, timestep_scaled, batch_size, hidden_dtype
        )

        # Prepare audio timestep
        a_timestep = kwargs.get("a_timestep")
        ref_audio_seq_len = kwargs.get("ref_audio_seq_len", 0)
        if ref_audio_seq_len > 0 and a_timestep is not None:
            # Reference tokens must have timestep=0, expand scalar/1D timestep to per-token so ref=0 and target=sigma.
            target_len = kwargs.get("target_audio_seq_len")
            if a_timestep.dim() <= 1:
                a_timestep = a_timestep.view(-1, 1).expand(batch_size, target_len)
            ref_ts = torch.zeros(batch_size, ref_audio_seq_len, *a_timestep.shape[2:], device=a_timestep.device, dtype=a_timestep.dtype)
            a_timestep = torch.cat([ref_ts, a_timestep], dim=1)
        if a_timestep is not None:
            a_timestep_scaled = a_timestep * self.timestep_scale_multiplier
            a_timestep_flat = a_timestep_scaled.flatten()
            timestep_flat = timestep_scaled.flatten()
            av_ca_factor = self.av_ca_timestep_scale_multiplier / self.timestep_scale_multiplier

            # Cross-attention timesteps - compress these too
            av_ca_audio_scale_shift_timestep, _ = self.av_ca_audio_scale_shift_adaln_single(
                a_timestep_flat,
                {"resolution": None, "aspect_ratio": None},
                batch_size=batch_size,
                hidden_dtype=hidden_dtype,
            )
            av_ca_video_scale_shift_timestep, _ = self.av_ca_video_scale_shift_adaln_single(
                timestep_flat,
                {"resolution": None, "aspect_ratio": None},
                batch_size=batch_size,
                hidden_dtype=hidden_dtype,
            )
            av_ca_a2v_gate_noise_timestep, _ = self.av_ca_a2v_gate_adaln_single(
                a_timestep_scaled.max().expand_as(timestep_flat) * av_ca_factor,
                {"resolution": None, "aspect_ratio": None},
                batch_size=batch_size,
                hidden_dtype=hidden_dtype,
            )
            av_ca_v2a_gate_noise_timestep, _ = self.av_ca_v2a_gate_adaln_single(
                timestep_scaled.max().expand_as(a_timestep_flat) * av_ca_factor,
                {"resolution": None, "aspect_ratio": None},
                batch_size=batch_size,
                hidden_dtype=hidden_dtype,
            )

            # Compress cross-attention timesteps (only video side, audio is too small to benefit)
            # v_patches_per_frame is None for spatial masks, set for temporal masks or no mask
            cross_av_timestep_ss = [
                av_ca_audio_scale_shift_timestep.view(batch_size, -1, av_ca_audio_scale_shift_timestep.shape[-1]),
                CompressedTimestep(av_ca_video_scale_shift_timestep.view(batch_size, -1, av_ca_video_scale_shift_timestep.shape[-1]), v_patches_per_frame),  # video - compressed if possible
                CompressedTimestep(av_ca_a2v_gate_noise_timestep.view(batch_size, -1, av_ca_a2v_gate_noise_timestep.shape[-1]), v_patches_per_frame),  # video - compressed if possible
                av_ca_v2a_gate_noise_timestep.view(batch_size, -1, av_ca_v2a_gate_noise_timestep.shape[-1]),
            ]

            a_timestep, a_embedded_timestep = self.audio_adaln_single(
                a_timestep_flat,
                {"resolution": None, "aspect_ratio": None},
                batch_size=batch_size,
                hidden_dtype=hidden_dtype,
            )
            # Audio timesteps
            a_timestep = a_timestep.view(batch_size, -1, a_timestep.shape[-1])
            a_embedded_timestep = a_embedded_timestep.view(batch_size, -1, a_embedded_timestep.shape[-1])

            a_prompt_timestep = compute_prompt_timestep(
                self.audio_prompt_adaln_single, a_timestep_scaled, batch_size, hidden_dtype
            )
        else:
            a_timestep = timestep_scaled
            a_embedded_timestep = kwargs.get("embedded_timestep")
            cross_av_timestep_ss = []
            a_prompt_timestep = None

        return [v_timestep, a_timestep, cross_av_timestep_ss, v_prompt_timestep, a_prompt_timestep], [
            v_embedded_timestep,
            a_embedded_timestep,
        ], None

    def _prepare_context(self, context, batch_size, x, attention_mask=None):
        vx = x[0]
        ax = x[1]
        video_dim = vx.shape[-1]
        audio_dim = ax.shape[-1]

        v_context_dim = self.caption_channels if self.caption_proj_before_connector is False else video_dim
        a_context_dim = self.caption_channels if self.caption_proj_before_connector is False else audio_dim

        v_context, a_context = torch.split(
            context, [v_context_dim, a_context_dim], len(context.shape) - 1
        )

        v_context, attention_mask = super()._prepare_context(
            v_context, batch_size, vx, attention_mask
        )
        if self.caption_proj_before_connector is False:
            a_context = self.audio_caption_projection(a_context)
        a_context = a_context.view(batch_size, -1, audio_dim)

        return [v_context, a_context], attention_mask

    def _prepare_positional_embeddings(self, pixel_coords, frame_rate, x_dtype):
        v_pixel_coords = pixel_coords[0]
        v_pe = super()._prepare_positional_embeddings(v_pixel_coords, frame_rate, x_dtype)

        a_latent_coords = pixel_coords[1]
        a_pe = self._precompute_freqs_cis(
            a_latent_coords,
            dim=self.audio_inner_dim,
            out_dtype=x_dtype,
            max_pos=self.audio_positional_embedding_max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.audio_num_attention_heads,
        )

        # calculate positional embeddings for the middle of the token duration, to use in av cross attention layers.
        max_pos = max(
            self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0]
        )
        v_pixel_coords = v_pixel_coords.to(torch.float32)
        v_pixel_coords[:, 0] = v_pixel_coords[:, 0] * (1.0 / frame_rate)
        av_cross_video_freq_cis = self._precompute_freqs_cis(
            v_pixel_coords[:, 0:1, :],
            dim=self.audio_cross_attention_dim,
            out_dtype=x_dtype,
            max_pos=[max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=self.audio_num_attention_heads,
        )
        av_cross_audio_freq_cis = self._precompute_freqs_cis(
            a_latent_coords[:, 0:1, :],
            dim=self.audio_cross_attention_dim,
            out_dtype=x_dtype,
            max_pos=[max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=self.audio_num_attention_heads,
        )

        return [(v_pe, av_cross_video_freq_cis), (a_pe, av_cross_audio_freq_cis)]

    def _process_transformer_blocks(
        self, x, context, attention_mask, timestep, pe, transformer_options={}, self_attention_mask=None, **kwargs
    ):
        vx = x[0]
        ax = x[1]
        v_context = context[0]
        a_context = context[1]
        v_timestep = timestep[0]
        a_timestep = timestep[1]
        v_pe, av_cross_video_freq_cis = pe[0]
        a_pe, av_cross_audio_freq_cis = pe[1]

        (
            av_ca_audio_scale_shift_timestep,
            av_ca_video_scale_shift_timestep,
            av_ca_a2v_gate_noise_timestep,
            av_ca_v2a_gate_noise_timestep,
        ) = timestep[2]

        v_prompt_timestep = timestep[3]
        a_prompt_timestep = timestep[4]

        """Process transformer blocks for LTXAV."""
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.transformer_blocks), vx.device, transformer_options)

        # Process transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, vx.device, block)
            if ("double_block", i) in blocks_replace:

                def block_wrap(args):
                    out = {}
                    out["img"] = block(
                        args["img"],
                        v_context=args["v_context"],
                        a_context=args["a_context"],
                        attention_mask=args["attention_mask"],
                        v_timestep=args["v_timestep"],
                        a_timestep=args["a_timestep"],
                        v_pe=args["v_pe"],
                        a_pe=args["a_pe"],
                        v_cross_pe=args["v_cross_pe"],
                        a_cross_pe=args["a_cross_pe"],
                        v_cross_scale_shift_timestep=args["v_cross_scale_shift_timestep"],
                        a_cross_scale_shift_timestep=args["a_cross_scale_shift_timestep"],
                        v_cross_gate_timestep=args["v_cross_gate_timestep"],
                        a_cross_gate_timestep=args["a_cross_gate_timestep"],
                        transformer_options=args["transformer_options"],
                        self_attention_mask=args.get("self_attention_mask"),
                        v_prompt_timestep=args.get("v_prompt_timestep"),
                        a_prompt_timestep=args.get("a_prompt_timestep"),
                    )
                    return out

                out = blocks_replace[("double_block", i)](
                    {
                        "img": (vx, ax),
                        "v_context": v_context,
                        "a_context": a_context,
                        "attention_mask": attention_mask,
                        "v_timestep": v_timestep,
                        "a_timestep": a_timestep,
                        "v_pe": v_pe,
                        "a_pe": a_pe,
                        "v_cross_pe": av_cross_video_freq_cis,
                        "a_cross_pe": av_cross_audio_freq_cis,
                        "v_cross_scale_shift_timestep": av_ca_video_scale_shift_timestep,
                        "a_cross_scale_shift_timestep": av_ca_audio_scale_shift_timestep,
                        "v_cross_gate_timestep": av_ca_a2v_gate_noise_timestep,
                        "a_cross_gate_timestep": av_ca_v2a_gate_noise_timestep,
                        "transformer_options": transformer_options,
                        "self_attention_mask": self_attention_mask,
                        "v_prompt_timestep": v_prompt_timestep,
                        "a_prompt_timestep": a_prompt_timestep,
                    },
                    {"original_block": block_wrap},
                )
                vx, ax = out["img"]
            else:
                vx, ax = block(
                    (vx, ax),
                    v_context=v_context,
                    a_context=a_context,
                    attention_mask=attention_mask,
                    v_timestep=v_timestep,
                    a_timestep=a_timestep,
                    v_pe=v_pe,
                    a_pe=a_pe,
                    v_cross_pe=av_cross_video_freq_cis,
                    a_cross_pe=av_cross_audio_freq_cis,
                    v_cross_scale_shift_timestep=av_ca_video_scale_shift_timestep,
                    a_cross_scale_shift_timestep=av_ca_audio_scale_shift_timestep,
                    v_cross_gate_timestep=av_ca_a2v_gate_noise_timestep,
                    a_cross_gate_timestep=av_ca_v2a_gate_noise_timestep,
                    transformer_options=transformer_options,
                    self_attention_mask=self_attention_mask,
                    v_prompt_timestep=v_prompt_timestep,
                    a_prompt_timestep=a_prompt_timestep,
                )

        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, vx.device, None)

        return [vx, ax]

    def _process_output(self, x, embedded_timestep, keyframe_idxs, **kwargs):
        vx = x[0]
        ax = x[1]
        v_embedded_timestep = embedded_timestep[0]
        a_embedded_timestep = embedded_timestep[1]

        # Trim reference audio tokens before unpatchification
        ref_audio_seq_len = kwargs.get("ref_audio_seq_len", 0)
        if ref_audio_seq_len > 0:
            ax = ax[:, ref_audio_seq_len:]
            if a_embedded_timestep.shape[1] > 1:
                a_embedded_timestep = a_embedded_timestep[:, ref_audio_seq_len:]

        # Expand compressed video timestep if needed
        if isinstance(v_embedded_timestep, CompressedTimestep):
            v_embedded_timestep = v_embedded_timestep.expand()

        vx = super()._process_output(vx, v_embedded_timestep, keyframe_idxs, **kwargs)

        # Process audio output
        a_scale_shift_values = (
            self.audio_scale_shift_table[None, None].to(device=a_embedded_timestep.device, dtype=a_embedded_timestep.dtype)
            + a_embedded_timestep[:, :, None]
        )
        a_shift, a_scale = a_scale_shift_values[:, :, 0], a_scale_shift_values[:, :, 1]

        ax = self.audio_norm_out(ax)
        ax = ax * (1 + a_scale) + a_shift
        ax = self.audio_proj_out(ax)

        # Unpatchify audio
        ax = self.a_patchifier.unpatchify(
            ax, channels=self.num_audio_channels, freq=self.audio_frequency_bins
        )

        # Recombine audio and video
        original_shape = kwargs.get("av_orig_shape")
        return self.recombine_audio_and_video_latents(vx, ax, original_shape)

    def forward(
        self,
        x,
        timestep,
        context,
        attention_mask=None,
        frame_rate=25,
        transformer_options={},
        keyframe_idxs=None,
        **kwargs,
    ):
        """
        Forward pass for LTXAV model.

        Args:
            x: Combined audio-video input tensor
            timestep: Tuple of (video_timestep, audio_timestep) or single timestep
            context: Context tensor (e.g., text embeddings)
            attention_mask: Attention mask tensor
            frame_rate: Frame rate for temporal processing
            transformer_options: Additional options for transformer blocks
            keyframe_idxs: Keyframe indices for temporal processing
            **kwargs: Additional keyword arguments including audio_length

        Returns:
            Combined audio-video output tensor
        """
        # Handle timestep format
        if isinstance(timestep, (tuple, list)) and len(timestep) == 2:
            v_timestep, a_timestep = timestep
            kwargs["a_timestep"] = a_timestep
            timestep = v_timestep
        else:
            kwargs["a_timestep"] = timestep

        # Call parent forward method
        return super().forward(
            x,
            timestep,
            context,
            attention_mask,
            frame_rate,
            transformer_options,
            keyframe_idxs,
            **kwargs,
        )
