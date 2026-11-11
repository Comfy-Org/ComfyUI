"""LTX 2.4 DurationHead: predicts the natural shot duration (in seconds) from
the caption connector token outputs, without running the diffusion pipeline.
"""

import torch
import torch.nn.functional as F
from torch import nn


class AttentionPooler(nn.Module):
    """Cross-attend ``num_queries`` learnable tokens against ``tokens``."""

    def __init__(self, hidden_dim=256, num_queries=1, num_heads=4):
        super().__init__()
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.empty(num_queries, hidden_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, tokens):
        queries = self.query_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        pooled, _ = self.cross_attn(queries, tokens, tokens, need_weights=False)
        return pooled


class DurationHead(nn.Module):
    """Predict duration in seconds from one or both connector outputs."""

    def __init__(
        self,
        video_cross_attention_dim=4096,
        audio_cross_attention_dim=2048,
        pooler_hidden_dim=256,
        num_queries=1,
        num_pooler_heads=4,
        mlp_hidden=256,
    ):
        super().__init__()
        self.video_input_proj = nn.Linear(video_cross_attention_dim, pooler_hidden_dim)
        self.video_modality_emb = nn.Parameter(torch.empty(pooler_hidden_dim))
        self.audio_input_proj = nn.Linear(audio_cross_attention_dim, pooler_hidden_dim)
        self.audio_modality_emb = nn.Parameter(torch.empty(pooler_hidden_dim))
        self.attention_pooler = AttentionPooler(
            hidden_dim=pooler_hidden_dim, num_queries=num_queries, num_heads=num_pooler_heads)
        self.mlp_hidden = nn.Linear(pooler_hidden_dim * num_queries, mlp_hidden)
        self.mlp_out = nn.Linear(mlp_hidden, 1)

    def forward(self, video_tokens=None, audio_tokens=None):
        """``video_tokens``: (B, T_v, 4096), ``audio_tokens``: (B, T_a, 2048);
        at least one required. Returns duration in seconds, shape (B,)."""
        token_groups = []
        if video_tokens is not None:
            token_groups.append(self.video_input_proj(video_tokens) + self.video_modality_emb)
        if audio_tokens is not None:
            token_groups.append(self.audio_input_proj(audio_tokens) + self.audio_modality_emb)
        if not token_groups:
            raise ValueError("DurationHead requires at least one of video_tokens / audio_tokens")
        pooled = self.attention_pooler(torch.cat(token_groups, dim=1))
        pooled = pooled.reshape(pooled.shape[0], -1)
        hidden = F.gelu(self.mlp_hidden(pooled), approximate="tanh")
        return self.mlp_out(hidden).squeeze(-1).exp()


def normalize_state_dict(sd):
    for prefix in ("model.diffusion_model.duration_head.", "duration_head."):
        stripped = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if stripped:
            return stripped
    return sd


def seconds_to_num_frames(seconds, frame_rate, min_seconds, max_seconds, time_scale=8):
    """Convert seconds to a frame count clamped to ``[min_seconds, max_seconds]``
    and snapped (floor) to the VAE's ``8k + 1`` causal temporal grid; snapping
    that undershoots the minimum bumps up to the next grid point instead."""
    min_frames = max(1, round(min_seconds * frame_rate))
    max_frames = round(max_seconds * frame_rate)
    raw_frames = max(min_frames, min(round(seconds * frame_rate), max_frames))
    frames = (raw_frames - 1) // time_scale * time_scale + 1
    if frames < min_frames:
        frames = min(-(-(min_frames - 1) // time_scale) * time_scale + 1, max_frames)
    return frames
