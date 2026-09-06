from .wav2vec2 import Wav2Vec2Model
from .whisper import WhisperLargeV3
import comfy.model_management
import comfy.ops
import comfy.utils
import logging
import torchaudio


class AudioEncoderModel():
    def __init__(self, config):
        self.load_device = comfy.model_management.text_encoder_device()
        offload_device = comfy.model_management.text_encoder_offload_device()
        self.dtype = comfy.model_management.text_encoder_dtype(self.load_device)
        model_type = config.pop("model_type")
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
        self.model.eval()
        self.patcher = comfy.model_patcher.CoreModelPatcher(self.model, load_device=self.load_device, offload_device=offload_device)
        self.model_sample_rate = 16000
        comfy.model_management.archive_model_dtypes(self.model)

    def load_sd(self, sd):
        return self.model.load_state_dict(sd, strict=False, assign=self.patcher.is_dynamic())

    def get_sd(self):
        return self.model.state_dict()

    def encode_audio(self, audio, sample_rate):
        comfy.model_management.load_model_gpu(self.patcher)
        audio = torchaudio.functional.resample(audio, sample_rate, self.model_sample_rate)
        out, all_layers = self.model(audio.to(self.load_device))
        outputs = {}
        outputs["encoded_audio"] = out
        outputs["encoded_audio_all_layers"] = all_layers
        outputs["audio_samples"] = audio.shape[2]
        return outputs


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
        # Full whisper checkpoints include decoder weights; discard them since
        # WhisperLargeV3 is encoder-only.
        sd = {k: v for k, v in sd.items() if not k.startswith("decoder.")}
        config = {
            "model_type": "whisper3",
        }
    else:
        raise RuntimeError("ERROR: audio encoder not supported.")

    audio_encoder = AudioEncoderModel(config)
    m, u = audio_encoder.load_sd(sd)
    if len(m) > 0:
        # torchaudio registers the mel-spectrogram window and filterbank as
        # buffers whose values are deterministically computed from the config
        # at init time; they are never saved in standard whisper checkpoints.
        # Suppress warnings for these expected-missing buffers so that users
        # are only alerted about genuinely unexpected missing weights.
        whisper_computed_buffers = {
            "feature_extractor.mel_spectrogram.spectrogram.window",
            "feature_extractor.mel_spectrogram.mel_scale.fb",
        }
        significant_missing = [k for k in m if k not in whisper_computed_buffers]
        if significant_missing:
            logging.warning("missing audio encoder: {}".format(significant_missing))
    if len(u) > 0:
        logging.warning("unexpected audio encoder: {}".format(u))

    return audio_encoder
