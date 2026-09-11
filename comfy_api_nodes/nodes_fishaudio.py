import json
import re
import uuid

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.fishaudio import (
    FishAudioASRRequest,
    FishAudioASRResponse,
    FishAudioCreateModelRequest,
    FishAudioCreateModelResponse,
    FishAudioProsody,
    FishAudioTTSRequest,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    audio_bytes_to_audio_input,
    audio_ndarray_to_bytesio,
    audio_tensor_to_contiguous_ndarray,
    sync_op,
    sync_op_raw,
    validate_string,
)

FISHAUDIO_VOICE = "FISHAUDIO_VOICE"

FISHAUDIO_VOICES = [
    ("802e3bc2b27e49c2995d23ef70e6ac89", "Energetic Male (en)"),
    ("b545c585f631496c914815291da4e893", "Friendly Women (en)"),
    ("933563129e564b19a115bedd57b7406a", "Sarah (en)"),
    ("8d21b053e2804e2a890e1cf62f267b6f", "Verity (en)"),
    ("f48d143a59a946ab87c0130fd081f349", "Polo (en)"),
    ("bf322df2096a46f18c579d0baa36f41d", "Adrian (en)"),
    ("98655a12fa944e26b274c535e5e03842", "E-girl (en)"),
    ("0327fdb5da9e4fd782899a8058c8ae2b", "Narrator (en)"),
    ("5212eb29e500460391d03af42af6552e", "Warm Conversational Voice (en)"),
    ("5c8dc6a69c0b4edfb32634db6384bf34", "Warm Storyteller (en)"),
    ("7a18a1851d2649108c48ec9f2c80eb2c", "Dramatic Character Male (en)"),
    ("59cb5986671546eaa6ca8ae6f29f6d22", "News Narrator (zh)"),
    ("bf6c479f5a384b8d857310030035824b", "Lively Female (zh)"),
    ("faccba1a8ac54016bcfc02761285e67f", "Gentle Female (zh)"),
    ("5161d41404314212af1254556477c17d", "Energetic Female (ja)"),
    ("0089dce5fefb4c6ba9b9f2f0debe1ddc", "Calm Female (ja)"),
    ("45c5d3723c9c42f598e4776dcfd5f02d", "Calm Male (ja)"),
]

FISHAUDIO_VOICE_MAP = {label: voice_id for voice_id, label in FISHAUDIO_VOICES}

MAX_REFERENCE_AUDIO_SECONDS = 270


def _rewrite_voice_tags(text: str, voice_count: int) -> tuple[str, set[int]]:
    referenced: set[int] = set()

    def repl(match: re.Match) -> str:
        index = int(match.group(1))
        if index < 1 or index > voice_count:
            raise ValueError(
                f"@Voice{index} does not match any connected voice ({voice_count} connected)."
            )
        referenced.add(index)
        return f"<|speaker:{index - 1}|>"

    rewritten = re.sub(r"(?<!\S)@voice([0-9]+)\b", repl, text, flags=re.IGNORECASE)
    return rewritten, referenced


def _tts_option_inputs() -> list:
    return [
        IO.Float.Input(
            "temperature",
            default=0.7,
            min=0.0,
            max=1.0,
            step=0.01,
            display_mode=IO.NumberDisplay.slider,
            tooltip="Expressiveness. Higher values are more varied, lower values are more consistent.",
        ),
        IO.Float.Input(
            "top_p",
            default=0.7,
            min=0.01,
            max=1.0,
            step=0.01,
            display_mode=IO.NumberDisplay.slider,
            tooltip="Diversity via nucleus sampling.",
        ),
        IO.Float.Input(
            "speed",
            default=1.0,
            min=0.5,
            max=2.0,
            step=0.01,
            display_mode=IO.NumberDisplay.slider,
            tooltip="Speaking rate. 1.0 is normal, <1.0 slower, >1.0 faster.",
        ),
        IO.Float.Input(
            "volume",
            default=0.0,
            min=-10.0,
            max=10.0,
            step=0.5,
            display_mode=IO.NumberDisplay.slider,
            tooltip="Volume adjustment in decibels. 0 is no change.",
        ),
        IO.Boolean.Input(
            "normalize",
            default=True,
            tooltip="Normalize numbers and text for English and Chinese, "
            "improving stability for numbers and dates.",
        ),
    ]


def _multi_speaker_inputs() -> list:
    return [
        IO.Autogrow.Input(
            "voices",
            template=IO.Autogrow.TemplatePrefix(
                IO.Custom(FISHAUDIO_VOICE).Input("voice"),
                prefix="voice",
                min=0,
                max=5,
            ),
            tooltip="Voices for synthesis. Leave empty for the default voice. "
            "With two or more voices, mark speaker changes in the text with @Voice1, @Voice2, etc.",
        ),
        *_tts_option_inputs(),
    ]


class FishAudioVoiceSelector(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="FishAudioVoiceSelector",
            display_name="Fish Audio Voice Selector",
            category="partner/audio/Fish Audio",
            description="Select a voice from the Fish Audio library for text-to-speech generation.",
            inputs=[
                IO.DynamicCombo.Input(
                    "voice",
                    options=[
                        *(IO.DynamicCombo.Option(label, []) for _, label in FISHAUDIO_VOICES),
                        IO.DynamicCombo.Option(
                            "custom",
                            [
                                IO.String.Input(
                                    "voice_id",
                                    default="",
                                    tooltip="Voice model ID from fish.audio, e.g. the ID in "
                                    "https://fish.audio/m/<id>/.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Choose a voice, or 'custom' to enter any fish.audio voice model ID.",
                ),
            ],
            outputs=[
                IO.Custom(FISHAUDIO_VOICE).Output(display_name="voice"),
            ],
            is_api_node=False,
        )

    @classmethod
    def execute(cls, voice: dict) -> IO.NodeOutput:
        selected = voice["voice"]
        if selected == "custom":
            voice_id = voice["voice_id"].strip()
            if not voice_id:
                raise ValueError("Custom voice ID is empty.")
            return IO.NodeOutput(voice_id)
        voice_id = FISHAUDIO_VOICE_MAP.get(selected)
        if not voice_id:
            raise ValueError(f"Unknown voice: {selected}")
        return IO.NodeOutput(voice_id)


class FishAudioTextToSpeech(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="FishAudioTextToSpeech",
            display_name="Fish Audio Text to Speech",
            category="partner/audio/Fish Audio",
            description="Convert text to speech. Supports emotion cues in the text "
            "([happy], [whispering] on s2.1-pro; (happy) on s1) and multi-speaker dialogue "
            "via @Voice1/@Voice2 tags with multiple connected voices.",
            inputs=[
                IO.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    tooltip="The text to convert to speech. With two or more voices connected, "
                    "mark speaker changes with @Voice1, @Voice2, etc.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option("s2.1-pro", _multi_speaker_inputs()),
                        IO.DynamicCombo.Option(
                            "s1",
                            [
                                IO.Custom(FISHAUDIO_VOICE).Input(
                                    "voice",
                                    optional=True,
                                    tooltip="Voice for synthesis. Leave unconnected for the default voice.",
                                ),
                                *_tts_option_inputs(),
                            ],
                        ),
                    ],
                    tooltip="Model to use for text-to-speech.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[
                IO.Audio.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["text"]),
                expr="""
                (
                  $t := widgets.text;
                  $type($t) = "string"
                    ? (
                        $bytes := $length($t) + 2 * $count($match($t, /[^\\x00-\\x7F]/));
                        {"type":"usd","usd": $bytes * 21.45 / 1000000, "format":{"approximate":true}}
                      )
                    : {"type":"usd","usd": 0.02145, "format":{"approximate":true, "suffix":"/1K bytes"}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        text: str,
        model: dict,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(text, field_name="text", min_length=1)
        model_name = model["model"]
        if model_name == "s1":
            voices = [model["voice"]] if model.get("voice") else []
        else:
            voices = [model["voices"][key] for key in model["voices"]]
        rewritten, referenced = _rewrite_voice_tags(text, len(voices))
        if len(voices) >= 2:
            missing = [i for i in range(1, len(voices) + 1) if i not in referenced]
            if missing:
                raise ValueError(
                    "With multiple voices, the text must mark speaker changes with tags for "
                    "each connected voice; missing: " + ", ".join(f"@Voice{i}" for i in missing)
                )
        reference_id: str | list[str] | None = None
        if len(voices) == 1:
            reference_id = voices[0]
        elif voices:
            reference_id = voices
        request = FishAudioTTSRequest(
            text=rewritten,
            reference_id=reference_id,
            temperature=model["temperature"],
            top_p=model["top_p"],
            prosody=FishAudioProsody(speed=model["speed"], volume=model["volume"]),
            normalize=model["normalize"],
        )
        response = await sync_op_raw(
            cls,
            ApiEndpoint(
                path="/proxy/fishaudio/v1/tts",
                method="POST",
                headers={"model": model_name},
            ),
            data=request,
            as_binary=True,
        )
        return IO.NodeOutput(audio_bytes_to_audio_input(response))


class FishAudioSpeechToText(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="FishAudioSpeechToText",
            display_name="Fish Audio Speech to Text",
            category="partner/audio/Fish Audio",
            description="Transcribe audio to text with automatic language detection.",
            inputs=[
                IO.Audio.Input(
                    "audio",
                    tooltip="Audio to transcribe.",
                ),
                IO.String.Input(
                    "language",
                    default="",
                    tooltip="ISO 639-1 language hint (e.g. 'en', 'zh'). "
                    "The language is auto-detected regardless.",
                ),
                IO.Boolean.Input(
                    "precise_timestamps",
                    default=False,
                    tooltip="Return word-level timestamped segments.",
                ),
            ],
            outputs=[
                IO.String.Output(id="text", display_name="text"),
                IO.String.Output(id="language_code", display_name="language_code"),
                IO.String.Output(id="segments_json", display_name="segments_json"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.00858,"format":{"approximate":true,"suffix":"/minute"}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        audio: Input.Audio,
        language: str,
        precise_timestamps: bool,
    ) -> IO.NodeOutput:
        audio_data_np = audio_tensor_to_contiguous_ndarray(audio["waveform"])
        audio_bytes_io = audio_ndarray_to_bytesio(audio_data_np, audio["sample_rate"], "mp4", "aac")
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/fishaudio/v1/asr", method="POST"),
            response_model=FishAudioASRResponse,
            data=FishAudioASRRequest(
                language=language.strip() or None,
                ignore_timestamps=not precise_timestamps,
            ),
            files={"audio": ("audio.mp4", audio_bytes_io, "audio/mp4")},
            content_type="multipart/form-data",
        )
        segments_json = json.dumps(
            [s.model_dump(exclude_none=True) for s in (response.segments or [])],
            indent=2,
        )
        return IO.NodeOutput(response.text or "", response.language_code or "", segments_json)


class FishAudioInstantVoiceClone(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="FishAudioInstantVoiceClone",
            display_name="Fish Audio Instant Voice Clone",
            category="partner/audio/Fish Audio",
            description="Create a private cloned voice from audio samples, instantly usable "
            "for text-to-speech. Provide 1-20 recordings, 10-30 seconds each recommended, "
            "under 270 seconds in total.",
            inputs=[
                IO.Autogrow.Input(
                    "files",
                    template=IO.Autogrow.TemplatePrefix(
                        IO.Audio.Input("audio"),
                        prefix="audio",
                        min=1,
                        max=20,
                    ),
                    tooltip="Audio recordings for voice cloning.",
                ),
                IO.Boolean.Input(
                    "enhance_audio_quality",
                    default=True,
                    tooltip="Enhance reference audio quality before training.",
                ),
            ],
            outputs=[
                IO.Custom(FISHAUDIO_VOICE).Output(display_name="voice"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(expr="""{"type":"usd","usd":0}"""),
        )

    @classmethod
    async def execute(
        cls,
        files: IO.Autogrow.Type,
        enhance_audio_quality: bool,
    ) -> IO.NodeOutput:
        total_seconds = 0.0
        for key in files:
            audio = files[key]
            total_seconds += audio["waveform"].shape[-1] / audio["sample_rate"]
        if total_seconds >= MAX_REFERENCE_AUDIO_SECONDS:
            raise ValueError(
                f"Total reference audio is {total_seconds:.0f} seconds; "
                f"it must be under {MAX_REFERENCE_AUDIO_SECONDS} seconds."
            )
        file_tuples: list[tuple[str, tuple[str, bytes, str]]] = []
        for key in files:
            audio = files[key]
            audio_data_np = audio_tensor_to_contiguous_ndarray(audio["waveform"])
            audio_bytes_io = audio_ndarray_to_bytesio(audio_data_np, audio["sample_rate"], "mp4", "aac")
            file_tuples.append(("voices", (f"{key}.mp4", audio_bytes_io.getvalue(), "audio/mp4")))
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/fishaudio/model", method="POST"),
            response_model=FishAudioCreateModelResponse,
            data=FishAudioCreateModelRequest(
                title=str(uuid.uuid4()),
                enhance_audio_quality=enhance_audio_quality,
            ),
            files=file_tuples,
            content_type="multipart/form-data",
        )
        return IO.NodeOutput(response.id)


class FishAudioExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            FishAudioVoiceSelector,
            FishAudioTextToSpeech,
            FishAudioSpeechToText,
            FishAudioInstantVoiceClone,
        ]


async def comfy_entrypoint() -> FishAudioExtension:
    return FishAudioExtension()
