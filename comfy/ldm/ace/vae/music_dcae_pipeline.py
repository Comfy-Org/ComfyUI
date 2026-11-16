# Original from: https://github.com/ace-step/ACE-Step/blob/main/music_dcae/music_dcae_pipeline.py
import torch
from .autoencoder_dc import AutoencoderDC
import comfy.audio

import torchvision.transforms as transforms
from .music_vocoder import ADaMoSHiFiGANV1


class MusicDCAE(torch.nn.Module):
    def __init__(self, source_sample_rate=None, dcae_config={}, vocoder_config={}):
        super(MusicDCAE, self).__init__()

        self.dcae = AutoencoderDC(**dcae_config)
        self.vocoder = ADaMoSHiFiGANV1(**vocoder_config)

        if source_sample_rate is None:
            self.source_sample_rate = 48000
        else:
            self.source_sample_rate = source_sample_rate

        self.transform = transforms.Compose([
            transforms.Normalize(0.5, 0.5),
        ])
        self.min_mel_value = -11.0
        self.max_mel_value = 3.0
        self.audio_chunk_size = int(round((1024 * 512 / 44100 * 48000)))
        self.mel_chunk_size = 1024
        self.time_dimention_multiple = 8
        self.latent_chunk_size = self.mel_chunk_size // self.time_dimention_multiple
        self.scale_factor = 0.1786
        self.shift_factor = -1.9091

    def forward_mel(self, audios):
        mels = []
        for i in range(len(audios)):
            image = self.vocoder.mel_transform(audios[i])
            mels.append(image)
        mels = torch.stack(mels)
        return mels

    @torch.no_grad()
    def encode(self, audios, audio_lengths=None, sr=None):
        if audio_lengths is None:
            audio_lengths = torch.tensor([audios.shape[2]] * audios.shape[0])
            audio_lengths = audio_lengths.to(audios.device)

        if sr is None:
            sr = self.source_sample_rate

        if sr != 44100:
            audios = comfy.audio.resample(audios, sr, 44100)

        max_audio_len = audios.shape[-1]
        if max_audio_len % (8 * 512) != 0:
            audios = torch.nn.functional.pad(audios, (0, 8 * 512 - max_audio_len % (8 * 512)))

        mels = self.forward_mel(audios)
        mels = (mels - self.min_mel_value) / (self.max_mel_value - self.min_mel_value)
        mels = self.transform(mels)
        latents = []
        for mel in mels:
            latent = self.dcae.encoder(mel.unsqueeze(0))
            latents.append(latent)
        latents = torch.cat(latents, dim=0)
        latents = (latents - self.shift_factor) * self.scale_factor
        return latents

    @torch.no_grad()
    def decode(self, latents, audio_lengths=None, sr=None):
        latents = latents / self.scale_factor + self.shift_factor

        pred_wavs = []

        for latent in latents:
            mels = self.dcae.decoder(latent.unsqueeze(0))
            mels = mels * 0.5 + 0.5
            mels = mels * (self.max_mel_value - self.min_mel_value) + self.min_mel_value
            wav = self.vocoder.decode(mels[0]).squeeze(1)

            if sr is not None:
                wav = comfy.audio.resample(wav, 44100, sr)
            else:
                sr = 44100
            pred_wavs.append(wav)

        if audio_lengths is not None:
            pred_wavs = [wav[:, :length].cpu() for wav, length in zip(pred_wavs, audio_lengths)]
        return torch.stack(pred_wavs)

    def forward(self, audios, audio_lengths=None, sr=None):
        latents, latent_lengths = self.encode(audios=audios, audio_lengths=audio_lengths, sr=sr)
        sr, pred_wavs = self.decode(latents=latents, audio_lengths=audio_lengths, sr=sr)
        return sr, pred_wavs, latents, latent_lengths
