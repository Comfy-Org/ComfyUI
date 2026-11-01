import logging

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.sd
from comfy.ldm.minimax.vae import LATENTS_MEAN, LATENTS_STD


class SmallVideoVAE(torch.nn.Module):
    def __init__(self, operations, num_layers, still_frame):
        super().__init__()
        self.tokens_chunk_size = 5
        self.token_overlap = 2
        self.vae_ratio_t = 4
        self.clip_length = 17
        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN))
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD))


def test_minimax_video_vae_loads_without_checkpoint_stats(monkeypatch, caplog):
    monkeypatch.setattr(comfy.sd.comfy.ldm.minimax.vae, "MiniMaxH3VideoVAE", SmallVideoVAE)
    sd = {
        "decoder.transformer_blocks.0.scale1": torch.empty(1),
        "encoder.down.5.block.0.conv1.weight": torch.empty(1),
    }

    with caplog.at_level(logging.WARNING):
        vae = comfy.sd.VAE(sd, device=torch.device("cpu"), dtype=torch.float32)

    assert "Missing VAE keys" not in caplog.text
    torch.testing.assert_close(vae.first_stage_model.latents_mean, torch.tensor(LATENTS_MEAN))
    torch.testing.assert_close(vae.first_stage_model.latents_std, torch.tensor(LATENTS_STD))
