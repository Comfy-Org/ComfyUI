"""MiniMax H3 _forward should pull text-token tags to host once per sampling run, not once per step."""

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ops as comfy_ops  # noqa: E402
from comfy.ldm.minimax.model import MiniMaxH3Model  # noqa: E402


class _CountingTags:
    """Stands in for the real tensor payload["text_token_tags"] and counts host pulls."""

    def __init__(self, values):
        self._values = values
        self.calls = 0

    def view(self, *shape):
        self.calls += 1
        return self

    def tolist(self):
        return list(self._values)


def _make_model():
    return MiniMaxH3Model(
        hidden_size=8,
        num_layers=0,
        token_refiner_num_layers=1,
        num_attention_heads=1,
        attention_head_dim=8,
        ffn_hidden_size=8,
        latents_dim=2,
        audio_latents_dim=2,
        text_dim=8,  # == hidden_size so _forward skips the token refiner entirely
        timestep_input_dim=16,
        time_embed_hidden_size=8,
        time_embed_dim=8,
        rope_inv_freq_len=4,
        operations=comfy_ops.disable_weight_init,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _forward_inputs(model, text_len=4, latent_t=1, lat_h=2, lat_w=2, audio_t=2):
    video_x = torch.zeros(1, model.latents_dim, latent_t, lat_h, lat_w)
    audio_x = torch.zeros(1, model.audio_latents_dim, 2, audio_t)
    context = torch.zeros(1, text_len, model.hidden_size)
    timestep = torch.tensor([500.0])
    return video_x, audio_x, context, timestep


def test_text_token_tags_cached_across_forward_calls():
    model = _make_model()
    tags = _CountingTags([0, 0, 1, 1])
    payload = {"text_token_tags": tags}
    video_x, audio_x, context, timestep = _forward_inputs(model)

    model._forward([video_x, audio_x], timestep, context, minimax_payload=payload)
    model._forward([video_x, audio_x], timestep, context, minimax_payload=payload)

    assert tags.calls == 1
