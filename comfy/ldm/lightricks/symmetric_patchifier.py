from abc import ABC, abstractmethod
from typing import Tuple

import torch
from einops import rearrange
from torch import Tensor


def latent_to_pixel_coords(
    latent_coords: Tensor, scale_factors: Tuple[int, int, int], causal_fix: bool = False
) -> Tensor:
    """
    Converts latent coordinates to pixel coordinates by scaling them according to the VAE's
    configuration.
    Args:
        latent_coords (Tensor): A tensor of shape [batch_size, 3, num_latents]
        containing the latent corner coordinates of each token.
        scale_factors (Tuple[int, int, int]): The scale factors of the VAE's latent space.
        causal_fix (bool): Whether to take into account the different temporal scale
            of the first frame. Default = False for backwards compatibility.
    Returns:
        Tensor: A tensor of pixel coordinates corresponding to the input latent coordinates.
    """
    pixel_coords = torch.stack(
        [latent_coords[:, i] * scale_factors[i] for i in range(latent_coords.shape[1])],
        dim=1,
    )
    if causal_fix:
        # Fix temporal scale for first frame to 1 due to causality
        pixel_coords[:, 0, ...] = (pixel_coords[:, 0, ...] + 1 - scale_factors[0]).clamp(min=0)
    return pixel_coords


class Patchifier(ABC):
    def __init__(self, patch_size: int, start_end: bool=False):
        super().__init__()
        self._patch_size = (1, patch_size, patch_size)
        self.start_end = start_end

    @abstractmethod
    def patchify(
        self, latents: Tensor, frame_rates: Tensor, scale_grid: bool
    ) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def unpatchify(
        self,
        latents: Tensor,
        output_height: int,
        output_width: int,
        output_num_frames: int,
        out_channels: int,
    ) -> Tuple[Tensor, Tensor]:
        pass

    @property
    def patch_size(self):
        return self._patch_size

    def get_latent_coords(
        self, latent_num_frames, latent_height, latent_width, batch_size, device
    ):
        """
        Return a tensor of shape [batch_size, 3, num_patches] containing the
            top-left corner latent coordinates of each latent patch.
        The tensor is repeated for each batch element.
        """
        latent_sample_coords = torch.meshgrid(
            torch.arange(0, latent_num_frames, self._patch_size[0], device=device),
            torch.arange(0, latent_height, self._patch_size[1], device=device),
            torch.arange(0, latent_width, self._patch_size[2], device=device),
            indexing="ij",
        )
        latent_sample_coords_start = torch.stack(latent_sample_coords, dim=0)

        latent_sample_coords_start = latent_sample_coords_start.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        latent_sample_coords_start = rearrange(
            latent_sample_coords_start, "b c f h w -> b c (f h w)", b=batch_size
        )
        if self.start_end:
            latent_sample_coords_end = torch.stack(
                [latent_sample_coords[i] + self._patch_size[i] for i in range(3)], dim=0
            )
            latent_sample_coords_end = latent_sample_coords_end.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
            latent_sample_coords_end = rearrange(
                latent_sample_coords_end, "b c f h w -> b c (f h w)", b=batch_size
            )

            latent_coords = torch.stack((latent_sample_coords_start, latent_sample_coords_end), dim=-1)
        else:
            latent_coords = latent_sample_coords_start
        return latent_coords


class SymmetricPatchifier(Patchifier):
    def patchify(
        self,
        latents: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        b, _, f, h, w = latents.shape
        latent_coords = self.get_latent_coords(f, h, w, b, latents.device)
        latents = rearrange(
            latents,
            "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
            p1=self._patch_size[0],
            p2=self._patch_size[1],
            p3=self._patch_size[2],
        )
        return latents, latent_coords

    def unpatchify(
        self,
        latents: Tensor,
        output_height: int,
        output_width: int,
        output_num_frames: int,
        out_channels: int,
    ) -> Tuple[Tensor, Tensor]:
        output_height = output_height // self._patch_size[1]
        output_width = output_width // self._patch_size[2]
        latents = rearrange(
            latents,
            "b (f h w) (c p q) -> b c f (h p) (w q) ",
            f=output_num_frames,
            h=output_height,
            w=output_width,
            p=self._patch_size[1],
            q=self._patch_size[2],
        )
        return latents


class AudioPatchifier(Patchifier):
    def __init__(self, patch_size: int,
        sample_rate=16000,
        hop_length=160,
        audio_latent_downsample_factor=4,
        is_causal=True,
        start_end=False,
        shift = 0
    ):
        super().__init__(patch_size, start_end=start_end)
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal
        self.shift = shift

    def copy_with_shift(self, shift):
        return AudioPatchifier(
            self.patch_size, self.sample_rate, self.hop_length, self.audio_latent_downsample_factor,
            self.is_causal, self.start_end, shift
        )

    def _get_audio_latent_time_in_sec(self, start_latent, end_latent: int, dtype: torch.dtype, device=torch.device):
        audio_latent_frame = torch.arange(start_latent, end_latent, dtype=dtype, device=device)
        audio_mel_frame = audio_latent_frame * self.audio_latent_downsample_factor
        if self.is_causal:
            audio_mel_frame = (audio_mel_frame + 1 - self.audio_latent_downsample_factor).clip(min=0)
        return audio_mel_frame * self.hop_length / self.sample_rate


    def patchify(self, audio_latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # audio_latents: (batch, channels, time, freq)
        b, _, t, _ = audio_latents.shape
        audio_latents = rearrange(
            audio_latents,
            "b c t f -> b t (c f)",
        )

        audio_latents_start_timings = self._get_audio_latent_time_in_sec(self.shift, t + self.shift, torch.float32, audio_latents.device)
        audio_latents_start_timings = audio_latents_start_timings.unsqueeze(0).expand(b, -1).unsqueeze(1)

        if self.start_end:
            audio_latents_end_timings = self._get_audio_latent_time_in_sec(self.shift + 1, t + self.shift + 1, torch.float32, audio_latents.device)
            audio_latents_end_timings = audio_latents_end_timings.unsqueeze(0).expand(b, -1).unsqueeze(1)

            audio_latents_timings = torch.stack([audio_latents_start_timings, audio_latents_end_timings], dim=-1)
        else:
            audio_latents_timings = audio_latents_start_timings
        return audio_latents, audio_latents_timings

    def unpatchify(self, audio_latents: torch.Tensor, channels: int, freq: int) -> torch.Tensor:
        # audio_latents: (batch, time, freq * channels)
        audio_latents = rearrange(
            audio_latents, "b t (c f) -> b c t f", c=channels, f=freq
        )
        return audio_latents
