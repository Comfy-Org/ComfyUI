import folder_paths
import comfy.audio_encoders.audio_encoders
import comfy.utils
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class AudioEncoderLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AudioEncoderLoader",
            display_name="Load Audio Encoder",
            category="model/loaders",
            inputs=[
                io.Combo.Input(
                    "audio_encoder_name",
                    options=folder_paths.get_filename_list("audio_encoders"),
                ),
            ],
            outputs=[io.AudioEncoder.Output()],
        )

    @classmethod
    def execute(cls, audio_encoder_name) -> io.NodeOutput:
        audio_encoder_name = folder_paths.get_full_path_or_raise("audio_encoders", audio_encoder_name)
        sd = comfy.utils.load_torch_file(audio_encoder_name, safe_load=True)
        audio_encoder = comfy.audio_encoders.audio_encoders.load_audio_encoder_from_sd(sd)
        if audio_encoder is None:
            raise RuntimeError("ERROR: audio encoder file is invalid and does not contain a valid model.")
        return io.NodeOutput(audio_encoder)


class AudioEncoderEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AudioEncoderEncode",
            category="model/conditioning",
            inputs=[
                io.AudioEncoder.Input("audio_encoder"),
                io.Audio.Input("audio"),
            ],
            outputs=[io.AudioEncoderOutput.Output()],
        )

    @classmethod
    def execute(cls, audio_encoder, audio) -> io.NodeOutput:
        output = audio_encoder.encode_audio(audio["waveform"], audio["sample_rate"])
        return io.NodeOutput(output)


class SheetSage2AudioToABC(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SheetSage2AudioToABC",
            display_name="SheetSage2 Audio to ABC",
            category="model/conditioning/yue2",
            description="Transcribes vocal and instrumental melodies from music into ABC notation. Connect abc output to YuE2 Generate Music node and use the matching mode.",
            inputs=[
                io.AudioEncoder.Input("audio_encoder"),
                io.Audio.Input("audio"),
                io.Combo.Input("mode", options=["melody", "full"], tooltip="full: generates melody and chords; melody: generates melody only, recommended for covers."),
            ],
            outputs=[io.String.Output(display_name="abc", is_output_list=True)],
        )

    @classmethod
    def execute(cls, audio_encoder, audio, mode):
        return io.NodeOutput(audio_encoder.generate_abc(audio["waveform"], audio["sample_rate"], melody_only=mode == "melody"))


class AudioEncoder(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            AudioEncoderLoader,
            AudioEncoderEncode,
            SheetSage2AudioToABC,
        ]


async def comfy_entrypoint() -> AudioEncoder:
    return AudioEncoder()
