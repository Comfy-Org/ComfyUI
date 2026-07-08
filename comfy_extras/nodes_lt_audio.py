import folder_paths
import comfy.utils
import comfy.model_management
import torch

from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_audio import VAEEncodeAudio

class LTXVAudioVAELoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXVAudioVAELoader",
            display_name="Load LTXV Audio VAE",
            category="model/loaders",
            inputs=[
                io.Combo.Input(
                    "ckpt_name",
                    options=folder_paths.get_filename_list("checkpoints"),
                    tooltip="Audio VAE checkpoint to load.",
                )
            ],
            outputs=[io.Vae.Output(display_name="Audio VAE")],
        )

    @classmethod
    def execute(cls, ckpt_name: str) -> io.NodeOutput:
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        sd, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)
        sd = comfy.utils.state_dict_prefix_replace(sd, {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."}, filter_keys=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()

        return io.NodeOutput(vae)


class LTXVAudioVAEEncode(VAEEncodeAudio):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXVAudioVAEEncode",
            display_name="LTXV Audio VAE Encode",
            category="model/latent/ltxv",
            inputs=[
                io.Audio.Input("audio", tooltip="The audio to be encoded."),
                io.Vae.Input(
                    id="audio_vae",
                    display_name="Audio VAE",
                    tooltip="The Audio VAE model to use for encoding.",
                ),
            ],
            outputs=[io.Latent.Output(display_name="Audio Latent")],
        )

    @classmethod
    def execute(cls, audio, audio_vae) -> io.NodeOutput:
        return super().execute(audio_vae, audio)


class LTXVAudioVAEDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXVAudioVAEDecode",
            display_name="LTXV Audio VAE Decode",
            category="model/latent/ltxv",
            inputs=[
                io.Latent.Input("samples", tooltip="The latent to be decoded."),
                io.Vae.Input(
                    id="audio_vae",
                    display_name="Audio VAE",
                    tooltip="The Audio VAE model used for decoding the latent.",
                ),
            ],
            outputs=[io.Audio.Output(display_name="Audio")],
        )

    @classmethod
    def execute(cls, samples, audio_vae) -> io.NodeOutput:
        audio_latent = samples["samples"]
        if audio_latent.is_nested:
            audio_latent = audio_latent.unbind()[-1]
        audio = audio_vae.decode(audio_latent).movedim(-1, 1).to(audio_latent.device)
        output_audio_sample_rate = audio_vae.first_stage_model.output_sample_rate
        return io.NodeOutput(
            {
                "waveform": audio,
                "sample_rate": int(output_audio_sample_rate),
            }
        )


class LTXVEmptyLatentAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXVEmptyLatentAudio",
            display_name="LTXV Empty Latent Audio",
            category="model/latent/ltxv",
            inputs=[
                io.Int.Input(
                    "frames_number",
                    default=97,
                    min=1,
                    max=1000,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Number of frames.",
                ),
                io.Int.Input(
                    "frame_rate",
                    default=25,
                    min=1,
                    max=1000,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Number of frames per second.",
                ),
                io.Int.Input(
                    "batch_size",
                    default=1,
                    min=1,
                    max=4096,
                    display_mode=io.NumberDisplay.number,
                    tooltip="The number of latent audio samples in the batch.",
                ),
                io.Vae.Input(
                    id="audio_vae",
                    display_name="Audio VAE",
                    tooltip="The Audio VAE model to get configuration from.",
                ),
            ],
            outputs=[io.Latent.Output(display_name="Latent")],
        )

    @classmethod
    def execute(
        cls,
        frames_number: int,
        frame_rate: int,
        batch_size: int,
        audio_vae,
    ) -> io.NodeOutput:
        """Generate empty audio latents matching the reference pipeline structure."""

        assert audio_vae is not None, "Audio VAE model is required"

        z_channels = audio_vae.latent_channels
        audio_freq = audio_vae.first_stage_model.latent_frequency_bins

        num_audio_latents = audio_vae.first_stage_model.num_of_latents_from_frames(frames_number, frame_rate)

        audio_latents = torch.zeros(
            (batch_size, z_channels, num_audio_latents, audio_freq),
            device=comfy.model_management.intermediate_device(),
        )

        return io.NodeOutput(
            {
                "samples": audio_latents,
                "type": "audio",
            }
        )


class LTXAVTextEncoderLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXAVTextEncoderLoader",
            display_name="Load LTXV Audio Text Encoder",
            category="model/loaders",
            description="Recipes:\nltxav: gemma 3 12B",
            inputs=[
                io.Combo.Input(
                    "text_encoder",
                    options=folder_paths.get_filename_list("text_encoders"),
                ),
                io.Combo.Input(
                    "ckpt_name",
                    options=folder_paths.get_filename_list("checkpoints"),
                ),
                io.Combo.Input(
                    "device",
                    options=["default", "cpu"],
                    advanced=True,
                )
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, text_encoder, ckpt_name, device="default"):
        clip_type = comfy.sd.CLIPType.LTXV

        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", text_encoder)
        clip_path2 = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)
        return io.NodeOutput(clip)


class LTXVAudioExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LTXVAudioVAELoader,
            LTXVAudioVAEEncode,
            LTXVAudioVAEDecode,
            LTXVEmptyLatentAudio,
            LTXAVTextEncoderLoader,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    return LTXVAudioExtension()
