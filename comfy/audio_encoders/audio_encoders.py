from .wav2vec2 import Wav2Vec2Model
from .whisper import WhisperLargeV3
from .parakeet import ParakeetTDT
from .sheetsage2 import SheetSage2
from .sheetsage2_abc import events_to_abc
import comfy.model_management
import comfy.ops
import comfy.storage
import comfy.utils
import json
import logging
import comfy.audio
import os
import torch


class AudioEncoderModel():
    def __init__(self, config, fast_disk=False):
        self.load_device = comfy.model_management.text_encoder_device()
        offload_device = comfy.model_management.text_encoder_offload_device()
        self.dtype = comfy.model_management.text_encoder_dtype(self.load_device)
        model_type = config.pop("model_type")
        self.model_sample_rate = config.pop("model_sample_rate", 16000)
        if model_type == "sheetsage2":
            self.dtype = torch.bfloat16 if comfy.model_management.should_use_bf16(self.load_device) else torch.float32
        model_config = dict(config)
        model_config.update({
            "dtype": self.dtype,
            "device": offload_device,
            "operations": comfy.ops.manual_cast
        })

        if model_type == "wav2vec2":
            self.model = Wav2Vec2Model(**model_config)
        elif model_type == "whisper3":
            self.model = WhisperLargeV3(**model_config)
        elif model_type == "sheetsage2":
            self.model = SheetSage2(**model_config)
        elif model_type == "parakeet_tdt":
            self.model = ParakeetTDT(**model_config)
        self.model.eval()
        self.patcher = comfy.model_patcher.CoreModelPatcher(self.model, load_device=self.load_device, offload_device=offload_device, fast_disk=fast_disk)
        comfy.model_management.archive_model_dtypes(self.model)

    def load_sd(self, sd):
        return self.model.load_state_dict(sd, strict=False, assign=self.patcher.is_dynamic())

    def get_sd(self):
        return self.model.state_dict()

    def encode_audio(self, audio, sample_rate):
        comfy.model_management.load_model_gpu(self.patcher)
        audio = comfy.audio.resample(audio, sample_rate, self.model_sample_rate)
        out, all_layers = self.model(audio.to(self.load_device))
        outputs = {}
        outputs["encoded_audio"] = out
        outputs["encoded_audio_all_layers"] = all_layers
        outputs["audio_samples"] = audio.shape[2]
        return outputs


class SheetSage2AudioEncoder(AudioEncoderModel):
    def generate_abc(self, audio, sample_rate, melody_only=True):
        audio = comfy.audio.resample(audio.float().mean(dim=1), sample_rate, self.model_sample_rate)
        comfy.model_management.load_model_gpu(self.patcher)
        scores = []
        for waveform in audio:
            events = self.model.transcribe(waveform[None].to(self.load_device))
            scores.append(events_to_abc(events, waveform.shape[-1] / self.model_sample_rate, melody_only=melody_only))
        return scores


class ParakeetAudioEncoder(AudioEncoderModel):
    def __init__(self, config, fast_disk=False):
        super().__init__(config, fast_disk=fast_disk)
        with open(os.path.join(os.path.dirname(__file__), "parakeet_vocab.json"), encoding="utf-8") as f:
            self.vocab = json.load(f)

    def transcribe(self, audio, sample_rate):
        audio = comfy.audio.resample(audio.float().mean(dim=1), sample_rate, self.model_sample_rate)
        comfy.model_management.load_model_gpu(self.patcher)
        texts, timestamps = [], []
        for waveform in audio:
            words = []
            for token, frame, duration in self.model.transcribe(waveform[None].to(self.load_device)):
                piece = self.vocab[token]
                if piece.startswith("<") and piece.endswith(">"):
                    continue
                if piece.startswith("\u2581") or not words:
                    words.append([piece.lstrip("\u2581"), frame, frame + duration])
                else:
                    words[-1][0] += piece
                    words[-1][2] = frame + duration
            texts.append(" ".join(word for word, _, _ in words))
            timestamps.append("\n".join(f"[{start * 0.08:.2f} - {end * 0.08:.2f}] {word}" for word, start, end in words))
        return texts, timestamps


def load_audio_encoder_from_sd(sd, prefix=""):
    sd = comfy.utils.state_dict_prefix_replace(sd, {"wav2vec2.": ""})
    if "encoder.layer_norm.bias" in sd: #wav2vec2
        embed_dim = sd["encoder.layer_norm.bias"].shape[0]
        if embed_dim == 1024:# large
            config = {
                "model_type": "wav2vec2",
                "embed_dim": 1024,
                "num_heads": 16,
                "num_layers": 24,
                "conv_norm": True,
                "conv_bias": True,
                "do_normalize": True,
                "do_stable_layer_norm": True
                }
        elif embed_dim == 768: # base
            config = {
                "model_type": "wav2vec2",
                "embed_dim": 768,
                "num_heads": 12,
                "num_layers": 12,
                "conv_norm": False,
                "conv_bias": False,
                "do_normalize": False, # chinese-wav2vec2-base has this False
                "do_stable_layer_norm": False
            }
        else:
            raise RuntimeError("ERROR: audio encoder file is invalid or unsupported embed_dim: {}".format(embed_dim))
    elif "model.encoder.embed_positions.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"model.": ""})
        config = {
            "model_type": "whisper3",
        }
    elif "encoder.feature_extractor.mel_mean" in sd and "decoder.layernorm_embedding.weight" in sd and "layer_weight" in sd:
        config = {"model_type": "sheetsage2", "model_sample_rate": 24000}
    elif "joint.head.weight" in sd and "encoder.subsampling.linear.weight" in sd:
        config = {"model_type": "parakeet_tdt"}
    else:
        raise RuntimeError("ERROR: audio encoder not supported.")

    fast_disk = comfy.storage.state_dict_fast_disk(sd)
    audio_encoder_class = {"sheetsage2": SheetSage2AudioEncoder, "parakeet_tdt": ParakeetAudioEncoder}.get(config["model_type"], AudioEncoderModel)
    audio_encoder = audio_encoder_class(config, fast_disk=fast_disk)
    m, u = audio_encoder.load_sd(sd)
    if len(m) > 0:
        logging.warning("missing audio encoder: {}".format(m))
    if len(u) > 0:
        logging.warning("unexpected audio encoder: {}".format(u))

    return audio_encoder
