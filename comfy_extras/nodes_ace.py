import torch
from typing_extensions import override

import comfy.model_management
import node_helpers
from comfy_api.latest import ComfyExtension, IO


class TextEncodeAceStepAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TextEncodeAceStepAudio",
            category="model/conditioning/ace",
            inputs=[
                IO.Clip.Input("clip"),
                IO.String.Input("tags", multiline=True, dynamic_prompts=True),
                IO.String.Input("lyrics", multiline=True, dynamic_prompts=True),
                IO.Float.Input("lyrics_strength", default=1.0, min=0.0, max=10.0, step=0.01),
            ],
            outputs=[IO.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, clip, tags, lyrics, lyrics_strength) -> IO.NodeOutput:
        tokens = clip.tokenize(tags, lyrics=lyrics)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning = node_helpers.conditioning_set_values(conditioning, {"lyrics_strength": lyrics_strength})
        return IO.NodeOutput(conditioning)

class TextEncodeAceStepAudio15(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TextEncodeAceStepAudio1.5",
            category="model/conditioning/ace",
            inputs=[
                IO.Clip.Input("clip"),
                IO.String.Input("tags", multiline=True, dynamic_prompts=True),
                IO.String.Input("lyrics", multiline=True, dynamic_prompts=True),
                IO.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                IO.Int.Input("bpm", default=120, min=10, max=300),
                IO.Float.Input("duration", default=120.0, min=0.0, max=2000.0, step=0.1),
                IO.Combo.Input("timesignature", options=['2', '3', '4', '6']),
                IO.Combo.Input("language", options=['ar', 'az', 'bg', 'bn', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fr', 'he', 'hi', 'hr', 'ht', 'hu', 'id', 'is', 'it', 'ja', 'ko', 'la', 'lt', 'ms', 'ne', 'nl', 'no', 'pa', 'pl', 'pt', 'ro', 'ru', 'sa', 'sk', 'sr', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'yue', 'zh', 'unknown'], default='en'),
                IO.Combo.Input("keyscale", options=[f"{root} {quality}" for quality in ["major", "minor"] for root in ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]]),
                IO.Boolean.Input("generate_audio_codes", default=True, tooltip="Enable the LLM that generates audio codes. This can be slow but will increase the quality of the generated audio. Turn this off if you are giving the model an audio reference.", advanced=True),
                IO.Float.Input("cfg_scale", default=2.0, min=0.0, max=100.0, step=0.1, advanced=True),
                IO.Float.Input("temperature", default=0.85, min=0.0, max=2.0, step=0.01, advanced=True),
                IO.Float.Input("top_p", default=0.9, min=0.0, max=2000.0, step=0.01, advanced=True),
                IO.Int.Input("top_k", default=0, min=0, max=100, advanced=True),
                IO.Float.Input("min_p", default=0.000, min=0.0, max=1.0, step=0.001, advanced=True),
            ],
            outputs=[IO.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, clip, tags, lyrics, seed, bpm, duration, timesignature, language, keyscale, generate_audio_codes, cfg_scale, temperature, top_p, top_k, min_p) -> IO.NodeOutput:
        tokens = clip.tokenize(tags, lyrics=lyrics, bpm=bpm, duration=duration, timesignature=int(timesignature), language=language, keyscale=keyscale, seed=seed, generate_audio_codes=generate_audio_codes, cfg_scale=cfg_scale, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return IO.NodeOutput(conditioning)


class EmptyAceStepLatentAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyAceStepLatentAudio",
            display_name="Empty Ace Step 1.0 Latent Audio",
            category="model/latent/ace",
            inputs=[
                IO.Float.Input("seconds", default=120.0, min=1.0, max=1000.0, step=0.1),
                IO.Int.Input(
                    "batch_size", default=1, min=1, max=4096, tooltip="The number of latent images in the batch."
                ),
            ],
            outputs=[IO.Latent.Output()],
        )

    @classmethod
    def execute(cls, seconds, batch_size) -> IO.NodeOutput:
        length = int(seconds * 44100 / 512 / 8)
        latent = torch.zeros([batch_size, 8, 16, length], device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
        return IO.NodeOutput({"samples": latent, "type": "audio"})


class EmptyAceStep15LatentAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyAceStep1.5LatentAudio",
            display_name="Empty Ace Step 1.5 Latent Audio",
            category="model/latent/ace",
            inputs=[
                IO.Float.Input("seconds", default=120.0, min=1.0, max=1000.0, step=0.01),
                IO.Int.Input(
                    "batch_size", default=1, min=1, max=4096, tooltip="The number of latent images in the batch."
                ),
            ],
            outputs=[IO.Latent.Output()],
        )

    @classmethod
    def execute(cls, seconds, batch_size) -> IO.NodeOutput:
        length = round((seconds * 48000 / 1920))
        latent = torch.zeros([batch_size, 64, length], device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
        return IO.NodeOutput({"samples": latent, "type": "audio", "downscale_ratio_temporal": 1764})

class ReferenceAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ReferenceTimbreAudio",
            display_name="Set Reference Audio",
            category="model/conditioning",
            is_experimental=True,
            description="This node sets the reference audio for ace step 1.5",
            inputs=[
                IO.Conditioning.Input("conditioning"),
                IO.Latent.Input("latent", optional=True),
            ],
            outputs=[
                IO.Conditioning.Output(),
            ]
        )

    @classmethod
    def execute(cls, conditioning, latent=None) -> IO.NodeOutput:
        if latent is not None:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_audio_timbre_latents": [latent["samples"]]}, append=True)
        return IO.NodeOutput(conditioning)

class AceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            TextEncodeAceStepAudio,
            EmptyAceStepLatentAudio,
            TextEncodeAceStepAudio15,
            EmptyAceStep15LatentAudio,
            ReferenceAudio,
        ]

async def comfy_entrypoint() -> AceExtension:
    return AceExtension()
