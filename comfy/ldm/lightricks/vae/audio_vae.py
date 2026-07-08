import json
from dataclasses import dataclass
import math
import torch
import torchaudio

from comfy.ldm.mmaudio.vae.distributions import DiagonalGaussianDistribution
from comfy.ldm.lightricks.symmetric_patchifier import AudioPatchifier
from comfy.ldm.lightricks.vae.causal_audio_autoencoder import (
    CausalityAxis,
    CausalAudioAutoencoder,
)
from comfy.ldm.lightricks.vocoders.vocoder import Vocoder, VocoderWithBWE

LATENT_DOWNSAMPLE_FACTOR = 4


@dataclass(frozen=True)
class AudioVAEComponentConfig:
    """Container for model component configuration extracted from metadata."""

    autoencoder: dict
    vocoder: dict

    @classmethod
    def from_metadata(cls, metadata: dict) -> "AudioVAEComponentConfig":
        assert metadata is not None and "config" in metadata, "Metadata is required for audio VAE"

        raw_config = metadata["config"]
        if isinstance(raw_config, str):
            parsed_config = json.loads(raw_config)
        else:
            parsed_config = raw_config

        audio_config = parsed_config.get("audio_vae")
        vocoder_config = parsed_config.get("vocoder")

        assert audio_config is not None, "Audio VAE config is required for audio VAE"
        assert vocoder_config is not None, "Vocoder config is required for audio VAE"

        return cls(autoencoder=audio_config, vocoder=vocoder_config)

class AudioLatentNormalizer:
    """Applies per-channel statistics in patch space and restores original layout."""

    def __init__(self, patchfier: AudioPatchifier, statistics_processor: torch.nn.Module):
        self.patchifier = patchfier
        self.statistics = statistics_processor

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        channels = latents.shape[1]
        freq = latents.shape[3]
        patched, _ = self.patchifier.patchify(latents)
        normalized = self.statistics.normalize(patched)
        return self.patchifier.unpatchify(normalized, channels=channels, freq=freq)

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        channels = latents.shape[1]
        freq = latents.shape[3]
        patched, _ = self.patchifier.patchify(latents)
        denormalized = self.statistics.un_normalize(patched)
        return self.patchifier.unpatchify(denormalized, channels=channels, freq=freq)


class AudioPreprocessor:
    """Prepares raw waveforms for the autoencoder by matching training conditions."""

    def __init__(self, target_sample_rate: int, mel_bins: int, mel_hop_length: int, n_fft: int):
        self.target_sample_rate = target_sample_rate
        self.mel_bins = mel_bins
        self.mel_hop_length = mel_hop_length
        self.n_fft = n_fft

    def resample(self, waveform: torch.Tensor, source_rate: int) -> torch.Tensor:
        if source_rate == self.target_sample_rate:
            return waveform
        return torchaudio.functional.resample(waveform, source_rate, self.target_sample_rate)

    def waveform_to_mel(
        self, waveform: torch.Tensor, waveform_sample_rate: int, device
    ) -> torch.Tensor:
        waveform = self.resample(waveform, waveform_sample_rate)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.mel_hop_length,
            f_min=0.0,
            f_max=self.target_sample_rate / 2.0,
            n_mels=self.mel_bins,
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
        ).to(device)

        mel = mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.permute(0, 1, 3, 2).contiguous()


class AudioVAE(torch.nn.Module):
    """High-level Audio VAE wrapper exposing encode and decode entry points."""

    def __init__(self, metadata: dict):
        super().__init__()

        component_config = AudioVAEComponentConfig.from_metadata(metadata)

        self.autoencoder = CausalAudioAutoencoder(config=component_config.autoencoder)
        if "bwe" in component_config.vocoder:
            self.vocoder = VocoderWithBWE(config=component_config.vocoder)
        else:
            self.vocoder = Vocoder(config=component_config.vocoder)

        autoencoder_config = self.autoencoder.get_config()
        self.normalizer = AudioLatentNormalizer(
            AudioPatchifier(
                patch_size=1,
                audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
                sample_rate=autoencoder_config["sampling_rate"],
                hop_length=autoencoder_config["mel_hop_length"],
                is_causal=autoencoder_config["is_causal"],
            ),
            self.autoencoder.per_channel_statistics,
        )

        self.preprocessor = AudioPreprocessor(
            target_sample_rate=autoencoder_config["sampling_rate"],
            mel_bins=autoencoder_config["mel_bins"],
            mel_hop_length=autoencoder_config["mel_hop_length"],
            n_fft=autoencoder_config["n_fft"],
        )

    def encode(self, audio, sample_rate=44100) -> torch.Tensor:
        """Encode a waveform dictionary into normalized latent tensors."""

        waveform = audio
        waveform_sample_rate = sample_rate
        input_device = waveform.device
        expected_channels = self.autoencoder.encoder.in_channels
        if waveform.shape[1] != expected_channels:
            if waveform.shape[1] == 1:
                waveform = waveform.expand(-1, expected_channels, *waveform.shape[2:])
            else:
                raise ValueError(
                    f"Input audio must have {expected_channels} channels, got {waveform.shape[1]}"
                )

        mel_spec = self.preprocessor.waveform_to_mel(
            waveform, waveform_sample_rate, device=waveform.device
        )

        latents = self.autoencoder.encode(mel_spec)
        posterior = DiagonalGaussianDistribution(latents)
        latent_mode = posterior.mode()

        normalized = self.normalizer.normalize(latent_mode)
        return normalized.to(input_device)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode normalized latent tensors into an audio waveform."""
        original_shape = latents.shape

        latents = self.normalizer.denormalize(latents)

        target_shape = self.target_shape_from_latents(original_shape)
        mel_spec = self.autoencoder.decode(latents, target_shape=target_shape)

        waveform = self.run_vocoder(mel_spec)
        return waveform

    def target_shape_from_latents(self, latents_shape):
        batch, _, time, _ = latents_shape
        target_length = time * LATENT_DOWNSAMPLE_FACTOR
        if self.autoencoder.causality_axis != CausalityAxis.NONE:
            target_length -= LATENT_DOWNSAMPLE_FACTOR - 1
        return (
            batch,
            self.autoencoder.decoder.out_ch,
            target_length,
            self.autoencoder.mel_bins,
        )

    def num_of_latents_from_frames(self, frames_number: int, frame_rate: int) -> int:
        return math.ceil((float(frames_number) / frame_rate) * self.latents_per_second)

    def run_vocoder(self, mel_spec: torch.Tensor) -> torch.Tensor:
        audio_channels = self.autoencoder.decoder.out_ch
        vocoder_input = mel_spec.transpose(2, 3)

        if audio_channels == 1:
            vocoder_input = vocoder_input.squeeze(1)
        elif audio_channels != 2:
            raise ValueError(f"Unsupported audio_channels: {audio_channels}")

        return self.vocoder(vocoder_input)

    @property
    def sample_rate(self) -> int:
        return int(self.autoencoder.sampling_rate)

    @property
    def mel_hop_length(self) -> int:
        return int(self.autoencoder.mel_hop_length)

    @property
    def mel_bins(self) -> int:
        return int(self.autoencoder.mel_bins)

    @property
    def latent_channels(self) -> int:
        return int(self.autoencoder.decoder.z_channels)

    @property
    def latent_frequency_bins(self) -> int:
        return int(self.mel_bins // LATENT_DOWNSAMPLE_FACTOR)

    @property
    def latents_per_second(self) -> float:
        return self.sample_rate / self.mel_hop_length / LATENT_DOWNSAMPLE_FACTOR

    @property
    def output_sample_rate(self) -> int:
        output_rate = getattr(self.vocoder, "output_sample_rate", None)
        if output_rate is not None:
            return int(output_rate)
        upsample_factor = getattr(self.vocoder, "upsample_factor", None)
        if upsample_factor is None:
            raise AttributeError(
                "Vocoder is missing upsample_factor; cannot infer output sample rate"
            )
        return int(self.sample_rate * upsample_factor / self.mel_hop_length)

    def memory_required(self, input_shape):
        return self.device_manager.patcher.model_size()
