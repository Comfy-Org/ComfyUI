import os
from typing import Literal

import torch
import torch.nn as nn

from ..bigvgan_v2.bigvgan import BigVGAN as BigVGANv2
from ..bigvgan_v2.bigvgan import load_hparams_from_json
from ..bigvgan_v2.env import AttrDict
from .distributions import DiagonalGaussianDistribution
from .vae import VAE


def build_bigvgan_v2(vocoder_config: dict | AttrDict) -> BigVGANv2:
    """Build an inference-ready BigVGAN-v2 from its serialized configuration."""
    model = BigVGANv2(vocoder_config if isinstance(vocoder_config, AttrDict) else AttrDict(vocoder_config))
    model.remove_weight_norm()
    return model


def load_bigvgan_v2(
    vocoder_ckpt_path: str,
    *,
    vocoder_config: dict | None = None,
) -> BigVGANv2:
    """Load BigVGAN-v2 from a local directory (avoids HubMixin/huggingface_hub API drift)."""
    model_file = os.path.join(vocoder_ckpt_path, "bigvgan_generator.pt")
    if vocoder_config is None:
        config_file = os.path.join(vocoder_ckpt_path, "config.json")
        if not os.path.isfile(config_file) or not os.path.isfile(model_file):
            raise FileNotFoundError(
                f"Expected config.json and bigvgan_generator.pt under {vocoder_ckpt_path}"
            )
        h = load_hparams_from_json(config_file)
    else:
        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"BigVGAN checkpoint does not exist: {model_file}")
        h = AttrDict(vocoder_config)

    model = BigVGANv2(h)
    checkpoint_dict = torch.load(model_file, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(checkpoint_dict["generator"])
    except RuntimeError:
        model.remove_weight_norm()
        model.load_state_dict(checkpoint_dict["generator"])
    model.remove_weight_norm()
    return model


class AutoEncoderModule(nn.Module):

    def __init__(
        self,
        *,
        vae_ckpt_path: str | None = None,
        vocoder_ckpt_path: str | None = None,
        vocoder_config: dict | None = None,
        mode: Literal["44k"],
        need_vae_encoder: bool = True,
        need_vae_decoder: bool = True,
    ):
        super().__init__()
        if mode != "44k":
            raise ValueError(f"Unknown model: {mode}")
        self.vae: VAE = VAE(data_dim=128, embed_dim=40, hidden_dim=512).eval()
        if vae_ckpt_path is not None:
            vae_state_dict = torch.load(vae_ckpt_path, weights_only=True, map_location="cpu")
            self.vae.load_state_dict(vae_state_dict)
        self.vae.remove_weight_norm()

        if need_vae_decoder:
            self.vocoder = load_bigvgan_v2(
                vocoder_ckpt_path,
                vocoder_config=vocoder_config,
            ).eval()
        else:
            del self.vae.decoder

        if not need_vae_encoder:
            del self.vae.encoder

        for param in self.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        return self.vae.encode(x)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)

    @torch.inference_mode()
    def vocode(self, spec: torch.Tensor) -> torch.Tensor:
        return self.vocoder(spec)
