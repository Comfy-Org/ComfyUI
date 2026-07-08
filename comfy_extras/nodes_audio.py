import av
import torchaudio
import torch
import comfy.model_management
import folder_paths
import os
import hashlib
import node_helpers
import logging
from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO, UI

class EmptyLatentAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyLatentAudio",
            display_name="Empty Latent Audio",
            category="model/latent",
            essentials_category="Audio",
            inputs=[
                IO.Float.Input("seconds", default=47.6, min=1.0, max=1000.0, step=0.1),
                IO.Int.Input(
                    "batch_size", default=1, min=1, max=4096, tooltip="The number of latent images in the batch.",
                ),
            ],
            outputs=[IO.Latent.Output()],
        )

    @classmethod
    def execute(cls, seconds, batch_size) -> IO.NodeOutput:
        length = round((seconds * 44100 / 2048) / 2) * 2
        latent = torch.zeros([batch_size, 64, length], device=comfy.model_management.intermediate_device())
        return IO.NodeOutput({"samples": latent, "type": "audio", "downscale_ratio_temporal": 2048})

    generate = execute  # TODO: remove


class ConditioningStableAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ConditioningStableAudio",
            category="model/conditioning/stable audio",
            inputs=[
                IO.Conditioning.Input("positive"),
                IO.Conditioning.Input("negative"),
                IO.Float.Input("seconds_start", default=0.0, min=0.0, max=1000.0, step=0.1),
                IO.Float.Input("seconds_total", default=47.0, min=0.0, max=1000.0, step=0.1),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, seconds_start, seconds_total) -> IO.NodeOutput:
        positive = node_helpers.conditioning_set_values(positive, {"seconds_start": seconds_start, "seconds_total": seconds_total})
        negative = node_helpers.conditioning_set_values(negative, {"seconds_start": seconds_start, "seconds_total": seconds_total})
        return IO.NodeOutput(positive, negative)

    append = execute  # TODO: remove


class VAEEncodeAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VAEEncodeAudio",
            search_aliases=["audio to latent"],
            display_name="VAE Encode Audio",
            category="model/latent",
            inputs=[
                IO.Audio.Input("audio"),
                IO.Vae.Input("vae"),
            ],
            outputs=[IO.Latent.Output()],
        )

    @classmethod
    def execute(cls, vae, audio) -> IO.NodeOutput:
        if audio is None:
            raise ValueError("VAEEncodeAudio: input audio is None (source video may have no audio track).")
        sample_rate = audio["sample_rate"]
        vae_sample_rate = getattr(vae, "audio_sample_rate", 44100)
        if vae_sample_rate != sample_rate:
            waveform = torchaudio.functional.resample(audio["waveform"], sample_rate, vae_sample_rate)
        else:
            waveform = audio["waveform"]

        t = vae.encode(waveform.movedim(1, -1))
        return IO.NodeOutput({"samples": t})

    encode = execute  # TODO: remove


def vae_decode_audio(vae, samples, tile=None, overlap=None):
    if tile is not None:
        audio = vae.decode_tiled(samples["samples"], tile_x=tile, tile_y=tile, overlap=overlap).movedim(-1, 1)
    else:
        audio = vae.decode(samples["samples"]).movedim(-1, 1)

    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio /= std
    vae_sample_rate = getattr(vae, "audio_sample_rate_output", getattr(vae, "audio_sample_rate", 44100))
    return {"waveform": audio, "sample_rate": vae_sample_rate if "sample_rate" not in samples else samples["sample_rate"]}


class VAEDecodeAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VAEDecodeAudio",
            search_aliases=["latent to audio"],
            display_name="VAE Decode Audio",
            category="model/latent",
            inputs=[
                IO.Latent.Input("samples"),
                IO.Vae.Input("vae"),
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, vae, samples) -> IO.NodeOutput:
        return IO.NodeOutput(vae_decode_audio(vae, samples))

    decode = execute  # TODO: remove


class VAEDecodeAudioTiled(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VAEDecodeAudioTiled",
            search_aliases=["latent to audio"],
            display_name="VAE Decode Audio (Tiled)",
            category="model/latent",
            inputs=[
                IO.Latent.Input("samples"),
                IO.Vae.Input("vae"),
                IO.Int.Input("tile_size", default=512, min=32, max=8192, step=8),
                IO.Int.Input("overlap", default=64, min=0, max=1024, step=8),
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, vae, samples, tile_size, overlap) -> IO.NodeOutput:
        return IO.NodeOutput(vae_decode_audio(vae, samples, tile_size, overlap))


class SaveAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveAudio",
            search_aliases=["export flac"],
            display_name="Save Audio (FLAC) (DEPRECATED)",
            category="audio",
            essentials_category="Audio",
            inputs=[
                IO.Audio.Input("audio"),
                IO.String.Input("filename_prefix", default="audio/ComfyUI"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_deprecated=True,
            is_output_node=True,
            outputs=[IO.Audio.Output("audio")]
        )

    @classmethod
    def execute(cls, audio, filename_prefix="ComfyUI", format="flac") -> IO.NodeOutput:
        if audio is None:
            raise ValueError("SaveAudio: input audio is None (source video may have no audio track).")
        return IO.NodeOutput(
            audio,
            ui=UI.AudioSaveHelper.get_save_audio_ui(audio, filename_prefix=filename_prefix, cls=cls, format=format)
        )


class SaveAudioMP3(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveAudioMP3",
            search_aliases=["export mp3"],
            display_name="Save Audio (MP3) (DEPRECATED)",
            category="audio",
            essentials_category="Audio",
            inputs=[
                IO.Audio.Input("audio"),
                IO.String.Input("filename_prefix", default="audio/ComfyUI"),
                IO.Combo.Input("quality", options=["V0", "128k", "320k"], default="V0"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_deprecated=True,
            is_output_node=True,
            outputs=[IO.Audio.Output("audio")]
        )

    @classmethod
    def execute(cls, audio, filename_prefix="ComfyUI", format="mp3", quality="128k") -> IO.NodeOutput:
        if audio is None:
            raise ValueError("SaveAudioMP3: input audio is None (source video may have no audio track).")
        return IO.NodeOutput(
            audio,
            ui=UI.AudioSaveHelper.get_save_audio_ui(
                audio, filename_prefix=filename_prefix, cls=cls, format=format, quality=quality
            )
        )


class SaveAudioOpus(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveAudioOpus",
            search_aliases=["export opus"],
            display_name="Save Audio (Opus) (DEPRECATED)",
            category="audio",
            inputs=[
                IO.Audio.Input("audio"),
                IO.String.Input("filename_prefix", default="audio/ComfyUI"),
                IO.Combo.Input("quality", options=["64k", "96k", "128k", "192k", "320k"], default="128k"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_deprecated=True,
            is_output_node=True,
            outputs=[IO.Audio.Output("audio")]
        )

    @classmethod
    def execute(cls, audio, filename_prefix="ComfyUI", format="opus", quality="V3") -> IO.NodeOutput:
        if audio is None:
            raise ValueError("SaveAudioOpus: input audio is None (source video may have no audio track).")
        return IO.NodeOutput(
            audio,
            ui=UI.AudioSaveHelper.get_save_audio_ui(
                audio, filename_prefix=filename_prefix, cls=cls, format=format, quality=quality
            )
        )


class SaveAudioAdvanced(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveAudioAdvanced",
            search_aliases=["save audio", "export audio", "output audio", "write audio", "flac", "mp3", "opus"],
            display_name="Save Audio (Advanced)",
            description="Saves the input audio to your ComfyUI output directory.",
            category="audio",
            inputs=[
                IO.Audio.Input("audio", tooltip="The audio to save."),
                IO.String.Input(
                    "filename_prefix",
                    default="audio/ComfyUI",
                    tooltip=("The prefix for the file to save. May include formatting tokens such as %date:yyyy-MM-dd%."),
                ),
                IO.DynamicCombo.Input(
                    "format",
                    options=[
                        IO.DynamicCombo.Option("flac", []),
                        IO.DynamicCombo.Option("mp3", [
                            IO.Combo.Input("quality", options=["V0", "128k", "320k"], default="V0"),
                        ]),
                        IO.DynamicCombo.Option("opus", [
                            IO.Combo.Input("quality", options=["64k", "96k", "128k", "192k", "320k"], default="128k"),
                        ]),
                    ],
                    tooltip="The file format in which to save the audio.",
                ),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[IO.Audio.Output("audio")],
        )

    @classmethod
    def execute(cls, audio, filename_prefix: str, format: dict) -> IO.NodeOutput:
        file_format = format.get("format", None)
        quality = format.get("quality", None)
        if quality:
            ui=UI.AudioSaveHelper.get_save_audio_ui(audio, filename_prefix=filename_prefix, cls=cls, format=file_format, quality=quality)
        else:
            ui=UI.AudioSaveHelper.get_save_audio_ui(audio, filename_prefix=filename_prefix, cls=cls, format=file_format)
        return IO.NodeOutput(audio, ui=ui)


class PreviewAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PreviewAudio",
            search_aliases=["play audio"],
            display_name="Preview Audio",
            category="audio",
            inputs=[
                IO.Audio.Input("audio"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[IO.Audio.Output("audio")]
        )

    @classmethod
    def execute(cls, audio) -> IO.NodeOutput:
        if audio is None:
            raise ValueError("PreviewAudio: input audio is None (source video may have no audio track).")
        return IO.NodeOutput(audio, ui=UI.PreviewAudio(audio, cls=cls))

    save_flac = execute  # TODO: remove


def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to float 32 bits PCM format."""
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / (2 ** 15)
    elif wav.dtype == torch.int32:
        return wav.float() / (2 ** 31)
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")

def load(filepath: str) -> tuple[torch.Tensor, int]:
    with av.open(filepath) as af:
        if not af.streams.audio:
            raise ValueError("No audio stream found in the file.")

        stream = af.streams.audio[0]
        sr = stream.codec_context.sample_rate
        n_channels = stream.channels

        frames = []
        length = 0
        for frame in af.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != n_channels:
                buf = buf.view(-1, n_channels).t()

            frames.append(buf)
            length += buf.shape[1]

        if not frames:
            raise ValueError("No audio frames decoded.")

        wav = torch.cat(frames, dim=1)
        wav = f32_pcm(wav)
        return wav, sr

class LoadAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        os.makedirs(input_dir, exist_ok=True)
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"])
        return IO.Schema(
            node_id="LoadAudio",
            search_aliases=["import audio", "open audio", "audio file"],
            display_name="Load Audio",
            category="audio",
            essentials_category="Audio",
            inputs=[
                IO.Combo.Input("audio", upload=IO.UploadType.audio, options=sorted(files)),
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, audio) -> IO.NodeOutput:
        audio_path = folder_paths.get_annotated_filepath(audio)
        waveform, sample_rate = load(audio_path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return IO.NodeOutput(audio)

    @classmethod
    def fingerprint_inputs(cls, audio):
        image_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def validate_inputs(cls, audio):
        if not folder_paths.exists_annotated_filepath(audio):
            return "Invalid audio file: {}".format(audio)
        return True

    load = execute  # TODO: remove


class RecordAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecordAudio",
            search_aliases=["microphone input", "audio capture", "voice input"],
            display_name="Record Audio",
            category="audio",
            inputs=[
                IO.Custom("AUDIO_RECORD").Input("audio"),
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, audio) -> IO.NodeOutput:
        audio_path = folder_paths.get_annotated_filepath(audio)

        waveform, sample_rate = load(audio_path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return IO.NodeOutput(audio)

    load = execute  # TODO: remove


class TrimAudioDuration(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TrimAudioDuration",
            search_aliases=["cut audio", "audio clip", "shorten audio"],
            display_name="Trim Audio Duration",
            description="Trim audio tensor into chosen time range.",
            category="audio",
            inputs=[
                IO.Audio.Input("audio"),
                IO.Float.Input(
                    "start_index",
                    default=0.0,
                    min=-0xffffffffffffffff,
                    max=0xffffffffffffffff,
                    step=0.01,
                    tooltip="Start time in seconds, can be negative to count from the end (supports sub-seconds).",
                ),
                IO.Float.Input(
                    "duration",
                    default=60.0,
                    min=0.0,
                    step=0.01,
                    tooltip="Duration in seconds",
                ),
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, audio, start_index, duration) -> IO.NodeOutput:
        if audio is None:
            return IO.NodeOutput(None)
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        audio_length = waveform.shape[-1]

        if audio_length == 0:
            return IO.NodeOutput(audio)

        if start_index < 0:
            start_frame = audio_length + int(round(start_index * sample_rate))
        else:
            start_frame = int(round(start_index * sample_rate))
        start_frame = max(0, min(start_frame, audio_length))

        end_frame = start_frame + int(round(duration * sample_rate))
        end_frame = max(0, min(end_frame, audio_length))

        if start_frame >= end_frame:
            raise ValueError("TrimAudioDuration: Start time must be less than end time and be within the audio length.")

        return IO.NodeOutput({"waveform": waveform[..., start_frame:end_frame], "sample_rate": sample_rate})

    trim = execute  # TODO: remove


class SplitAudioChannels(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SplitAudioChannels",
            search_aliases=["stereo to mono"],
            display_name="Split Audio Channels",
            description="Separates the audio into left and right channels.",
            category="audio",
            inputs=[
                IO.Audio.Input("audio"),
            ],
            outputs=[
                IO.Audio.Output(display_name="left"),
                IO.Audio.Output(display_name="right"),
            ],
        )

    @classmethod
    def execute(cls, audio) -> IO.NodeOutput:
        if audio is None:
            return IO.NodeOutput(None, None)
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        if waveform.shape[1] != 2:
            raise ValueError(f"AudioSplit: Input audio must be stereo (2 channels), got {waveform.shape[1]} channel(s).")

        left_channel = waveform[..., 0:1, :]
        right_channel = waveform[..., 1:2, :]

        return IO.NodeOutput({"waveform": left_channel, "sample_rate": sample_rate}, {"waveform": right_channel, "sample_rate": sample_rate})

    separate = execute  # TODO: remove

class JoinAudioChannels(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JoinAudioChannels",
            display_name="Join Audio Channels",
            description="Joins left and right mono audio channels into a stereo audio.",
            category="audio",
            inputs=[
                IO.Audio.Input("audio_left"),
                IO.Audio.Input("audio_right"),
            ],
            outputs=[
                IO.Audio.Output(display_name="audio"),
            ],
        )

    @classmethod
    def execute(cls, audio_left, audio_right) -> IO.NodeOutput:
        if audio_left is None and audio_right is None:
            return IO.NodeOutput(None)
        if audio_left is None:
            return IO.NodeOutput(audio_right)
        if audio_right is None:
            return IO.NodeOutput(audio_left)
        waveform_left = audio_left["waveform"]
        sample_rate_left = audio_left["sample_rate"]
        waveform_right = audio_right["waveform"]
        sample_rate_right = audio_right["sample_rate"]

        if waveform_left.shape[1] != 1 or waveform_right.shape[1] != 1:
            raise ValueError("AudioJoin: Both input audios must be mono.")

        # Handle different sample rates by resampling to the higher rate
        waveform_left, waveform_right, output_sample_rate = match_audio_sample_rates(
            waveform_left, sample_rate_left, waveform_right, sample_rate_right
        )

        # Handle different lengths by trimming to the shorter length
        length_left = waveform_left.shape[-1]
        length_right = waveform_right.shape[-1]

        if length_left != length_right:
            min_length = min(length_left, length_right)
            if length_left > min_length:
                logging.info(f"JoinAudioChannels: Trimming left channel from {length_left} to {min_length} samples.")
                waveform_left = waveform_left[..., :min_length]
            if length_right > min_length:
                logging.info(f"JoinAudioChannels: Trimming right channel from {length_right} to {min_length} samples.")
                waveform_right = waveform_right[..., :min_length]

        # Join the channels into stereo
        left_channel = waveform_left[..., 0:1, :]
        right_channel = waveform_right[..., 0:1, :]
        stereo_waveform = torch.cat([left_channel, right_channel], dim=1)

        return IO.NodeOutput({"waveform": stereo_waveform, "sample_rate": output_sample_rate})


def match_audio_sample_rates(waveform_1, sample_rate_1, waveform_2, sample_rate_2):
    if sample_rate_1 != sample_rate_2:
        if sample_rate_1 > sample_rate_2:
            waveform_2 = torchaudio.functional.resample(waveform_2, sample_rate_2, sample_rate_1)
            output_sample_rate = sample_rate_1
            logging.info(f"Resampling audio2 from {sample_rate_2}Hz to {sample_rate_1}Hz for merging.")
        else:
            waveform_1 = torchaudio.functional.resample(waveform_1, sample_rate_1, sample_rate_2)
            output_sample_rate = sample_rate_2
            logging.info(f"Resampling audio1 from {sample_rate_1}Hz to {sample_rate_2}Hz for merging.")
    else:
        output_sample_rate = sample_rate_1
    return waveform_1, waveform_2, output_sample_rate


class AudioConcat(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="AudioConcat",
            search_aliases=["join audio", "combine audio", "append audio"],
            display_name="Concatenate Audio",
            description="Concatenates the audio1 to audio2 in the specified direction.",
            category="audio",
            inputs=[
                IO.Audio.Input("audio1"),
                IO.Audio.Input("audio2"),
                IO.Combo.Input(
                    "direction",
                    options=['after', 'before'],
                    default="after",
                    tooltip="Whether to append audio2 after or before audio1.",
                )
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, audio1, audio2, direction) -> IO.NodeOutput:
        if audio1 is None and audio2 is None:
            return IO.NodeOutput(None)
        if audio1 is None:
            return IO.NodeOutput(audio2)
        if audio2 is None:
            return IO.NodeOutput(audio1)
        waveform_1 = audio1["waveform"]
        waveform_2 = audio2["waveform"]
        sample_rate_1 = audio1["sample_rate"]
        sample_rate_2 = audio2["sample_rate"]

        if waveform_1.shape[1] == 1:
            waveform_1 = waveform_1.repeat(1, 2, 1)
            logging.info("AudioConcat: Converted mono audio1 to stereo by duplicating the channel.")
        if waveform_2.shape[1] == 1:
            waveform_2 = waveform_2.repeat(1, 2, 1)
            logging.info("AudioConcat: Converted mono audio2 to stereo by duplicating the channel.")

        waveform_1, waveform_2, output_sample_rate = match_audio_sample_rates(waveform_1, sample_rate_1, waveform_2, sample_rate_2)

        if direction == 'after':
            concatenated_audio = torch.cat((waveform_1, waveform_2), dim=2)
        elif direction == 'before':
            concatenated_audio = torch.cat((waveform_2, waveform_1), dim=2)

        return IO.NodeOutput({"waveform": concatenated_audio, "sample_rate": output_sample_rate})

    concat = execute  # TODO: remove


class AudioMerge(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="AudioMerge",
            search_aliases=["mix audio", "overlay audio", "layer audio"],
            display_name="Merge Audio",
            description="Combine two audio tracks by overlaying their waveforms.",
            category="audio",
            inputs=[
                IO.Audio.Input("audio1"),
                IO.Audio.Input("audio2"),
                IO.Combo.Input(
                    "merge_method",
                    options=["add", "mean", "subtract", "multiply"],
                    tooltip="The method used to combine the audio waveforms.",
                )
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, audio1, audio2, merge_method) -> IO.NodeOutput:
        if audio1 is None and audio2 is None:
            return IO.NodeOutput(None)
        if audio1 is None:
            return IO.NodeOutput(audio2)
        if audio2 is None:
            return IO.NodeOutput(audio1)
        waveform_1 = audio1["waveform"]
        waveform_2 = audio2["waveform"]
        sample_rate_1 = audio1["sample_rate"]
        sample_rate_2 = audio2["sample_rate"]

        waveform_1, waveform_2, output_sample_rate = match_audio_sample_rates(waveform_1, sample_rate_1, waveform_2, sample_rate_2)

        length_1 = waveform_1.shape[-1]
        length_2 = waveform_2.shape[-1]

        if length_1 == 0 or length_2 == 0:
            return IO.NodeOutput({"waveform": waveform_1, "sample_rate": output_sample_rate})

        if length_2 > length_1:
            logging.info(f"AudioMerge: Trimming audio2 from {length_2} to {length_1} samples to match audio1 length.")
            waveform_2 = waveform_2[..., :length_1]
        elif length_2 < length_1:
            logging.info(f"AudioMerge: Padding audio2 from {length_2} to {length_1} samples to match audio1 length.")
            pad_shape = list(waveform_2.shape)
            pad_shape[-1] = length_1 - length_2
            pad_tensor = torch.zeros(pad_shape, dtype=waveform_2.dtype, device=waveform_2.device)
            waveform_2 = torch.cat((waveform_2, pad_tensor), dim=-1)

        if merge_method == "add":
            waveform = waveform_1 + waveform_2
        elif merge_method == "subtract":
            waveform = waveform_1 - waveform_2
        elif merge_method == "multiply":
            waveform = waveform_1 * waveform_2
        elif merge_method == "mean":
            waveform = (waveform_1 + waveform_2) / 2

        max_val = waveform.abs().max()
        if max_val > 1.0:
            waveform = waveform / max_val

        return IO.NodeOutput({"waveform": waveform, "sample_rate": output_sample_rate})

    merge = execute  # TODO: remove


class AudioAdjustVolume(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="AudioAdjustVolume",
            search_aliases=["audio gain", "loudness", "audio level"],
            display_name="Adjust Audio Volume",
            category="audio",
            description="Adjust the volume of the audio by a specified amount in decibels (dB).",
            inputs=[
                IO.Audio.Input("audio"),
                IO.Int.Input(
                    "volume",
                    default=1,
                    min=-100,
                    max=100,
                    tooltip="Volume adjustment in decibels (dB). 0 = no change, +6 = double, -6 = half, etc",
                )
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, audio, volume) -> IO.NodeOutput:
        if audio is None:
            return IO.NodeOutput(None)
        if volume == 0:
            return IO.NodeOutput(audio)
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        gain = 10 ** (volume / 20)
        waveform = waveform * gain

        return IO.NodeOutput({"waveform": waveform, "sample_rate": sample_rate})

    adjust_volume = execute  # TODO: remove


class EmptyAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyAudio",
            search_aliases=["blank audio"],
            display_name="Empty Audio",
            category="audio",
            inputs=[
                IO.Float.Input(
                    "duration",
                    default=60.0,
                    min=0.0,
                    max=0xffffffffffffffff,
                    step=0.01,
                    tooltip="Duration of the empty audio clip in seconds",
                ),
                IO.Int.Input(
                    "sample_rate",
                    default=44100,
                    tooltip="Sample rate of the empty audio clip.",
                    min=1,
                    max=192000,
                    advanced=True,
                ),
                IO.Int.Input(
                    "channels",
                    default=2,
                    min=1,
                    max=2,
                    tooltip="Number of audio channels (1 for mono, 2 for stereo).",
                    advanced=True,
                ),
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, duration, sample_rate, channels) -> IO.NodeOutput:
        num_samples = int(round(duration * sample_rate))
        waveform = torch.zeros((1, channels, num_samples), dtype=torch.float32)
        return IO.NodeOutput({"waveform": waveform, "sample_rate": sample_rate})

    create_empty_audio = execute  # TODO: remove


class AudioEqualizer3Band(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="AudioEqualizer3Band",
            search_aliases=["eq", "bass boost", "treble boost", "equalizer"],
            display_name="Audio Equalizer (3-Band)",
            category="audio",
            is_experimental=True,
            inputs=[
                IO.Audio.Input("audio"),
                IO.Float.Input("low_gain_dB", default=0.0, min=-24.0, max=24.0, step=0.1, tooltip="Gain for Low frequencies (Bass)"),
                IO.Int.Input("low_freq", default=100, min=20, max=500, tooltip="Cutoff frequency for Low shelf"),
                IO.Float.Input("mid_gain_dB", default=0.0, min=-24.0, max=24.0, step=0.1, tooltip="Gain for Mid frequencies"),
                IO.Int.Input("mid_freq", default=1000, min=200, max=4000, tooltip="Center frequency for Mids"),
                IO.Float.Input("mid_q", default=0.707, min=0.1, max=10.0, step=0.1, tooltip="Q factor (bandwidth) for Mids"),
                IO.Float.Input("high_gain_dB", default=0.0, min=-24.0, max=24.0, step=0.1, tooltip="Gain for High frequencies (Treble)"),
                IO.Int.Input("high_freq", default=5000, min=1000, max=15000, tooltip="Cutoff frequency for High shelf"),
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, audio, low_gain_dB, low_freq, mid_gain_dB, mid_freq, mid_q, high_gain_dB, high_freq) -> IO.NodeOutput:
        if audio is None:
            return IO.NodeOutput(None)
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        if waveform.shape[-1] == 0:
            return IO.NodeOutput(audio)

        eq_waveform = waveform.clone()

        # 1. Apply Low Shelf (Bass)
        if low_gain_dB != 0:
            eq_waveform = torchaudio.functional.bass_biquad(
                eq_waveform,
                sample_rate,
                gain=low_gain_dB,
                central_freq=float(low_freq),
                Q=0.707
            )

        # 2. Apply Peaking EQ (Mids)
        if mid_gain_dB != 0:
            eq_waveform = torchaudio.functional.equalizer_biquad(
                eq_waveform,
                sample_rate,
                center_freq=float(mid_freq),
                gain=mid_gain_dB,
                Q=mid_q
            )

        # 3. Apply High Shelf (Treble)
        if high_gain_dB != 0:
            eq_waveform = torchaudio.functional.treble_biquad(
                eq_waveform,
                sample_rate,
                gain=high_gain_dB,
                central_freq=float(high_freq),
                Q=0.707
            )

        return IO.NodeOutput({"waveform": eq_waveform, "sample_rate": sample_rate})


class AudioExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            EmptyLatentAudio,
            VAEEncodeAudio,
            VAEDecodeAudio,
            VAEDecodeAudioTiled,
            SaveAudio,
            SaveAudioMP3,
            SaveAudioOpus,
            SaveAudioAdvanced,
            LoadAudio,
            PreviewAudio,
            ConditioningStableAudio,
            RecordAudio,
            TrimAudioDuration,
            SplitAudioChannels,
            JoinAudioChannels,
            AudioConcat,
            AudioMerge,
            AudioAdjustVolume,
            EmptyAudio,
            AudioEqualizer3Band,
        ]

async def comfy_entrypoint() -> AudioExtension:
    return AudioExtension()
