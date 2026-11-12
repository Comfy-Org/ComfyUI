import base64
import re

from pydantic import BaseModel
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.breezeblue import (
    BreezeSavedVoice,
    BreezeSaveVoiceRequest,
    BreezeTextToSpeechRequest,
    BreezeVoiceCloneRequest,
    BreezeVoiceCloneResponse,
    BreezeVoiceDesignRequest,
    BreezeVoiceDesignResponse,
    BreezeVoiceLocalizeRequest,
    BreezeVoicePreviewJob,
    BreezeVoicePreviewJobStatus,
    BreezeVoiceRemixRequest,
    BreezeVoiceSettings,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    audio_bytes_to_audio_input,
    audio_ndarray_to_bytesio,
    audio_tensor_to_contiguous_ndarray,
    poll_op,
    sync_op,
    sync_op_raw,
    validate_audio_duration,
    validate_string,
)

BREEZE_VOICE = "BREEZE_VOICE"

BREEZE_MODELS = {
    "Breeze TTS 2": "breeze-tts-2",
    "Breeze TTS 2 Multilingual": "breeze-tts-2-multilingual",
}

BREEZE_LANGUAGES = {
    "Afrikaans": "af",
    "Albanian": "sq",
    "Arabic": "ar",
    "Basque": "eu",
    "Bengali": "bn",
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Chinese": "zh",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Estonian": "et",
    "Filipino": "tl",
    "Finnish": "fi",
    "French": "fr",
    "Galician": "gl",
    "German": "de",
    "Greek": "el",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Kannada": "kn",
    "Korean": "ko",
    "Lithuanian": "lt",
    "Malay": "ms",
    "Marathi": "mr",
    "Nepali": "ne",
    "Norwegian": "no",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish": "es",
    "Swahili": "sw",
    "Swedish": "sv",
    "Tamil": "ta",
    "Telugu": "te",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Vietnamese": "vi",
}

BREEZE_TTS2_LANGUAGES = ["English", "Chinese"]

BREEZE_VOICES = {
    "English": [
        ("Elizabeth", "voc_z6ufhy9629mg"),
        ("Julian", "voc_xeh3w54cqvnp"),
        ("Amara", "voc_rpabqqxus4p5"),
        ("Spark", "voc_x9gmqw84fead"),
        ("Chloe", "voc_8rsb3nhb7645"),
        ("Deema", "voc_9k38qhkc6qrm"),
    ],
    "Chinese": [
        ("灵动旅人", "voc_kgj4fsfnqmf4"),
        ("晨曦讲解员", "voc_xb3zjqcmwp44"),
        ("幻变小博士", "voc_mcqmgt3pp2ee"),
        ("青葱电台", "voc_rgr8satyqpwh"),
        ("云端悦音", "voc_hwvhpeezxup2"),
        ("弈林老叟", "voc_5g85wvn7gjde"),
    ],
    "Japanese": [
        ("島影の記録者", "voc_k5q8vby96mps"),
        ("若き調査員", "voc_2wd5j5wthapz"),
        ("深夜の灯火", "voc_attg42quyjxv"),
        ("暁の記録者", "voc_mumkt79acmuf"),
        ("古の匠の語り部", "voc_jqmzs44t37fc"),
    ],
    "Korean": [
        ("추억을 읽는 소년", "voc_gvxhufjatd5r"),
        ("든든한 길잡이", "voc_x4j8n2f9mbx3"),
        ("포근한 섬마을 할망", "voc_pj9kcrhdg9qs"),
        ("추억을 넘기는 손길", "voc_vs94ezaqjyhg"),
    ],
    "French": [
        ("Gérard", "voc_88zh5bnvm772"),
        ("Agnès", "voc_crfjzkqe9ekh"),
        ("Soizic", "voc_7czwp98nyvum"),
        ("Bastien", "voc_fp26jfs6c24z"),
    ],
    "German": [
        ("Birte", "voc_tmgea3src6zv"),
        ("Kai", "voc_rk4m9h4dzwvu"),
        ("Petra", "voc_s9ag425fpc49"),
        ("Birgit", "voc_vhzrhqceahuj"),
    ],
    "Portuguese": [
        ("Inês", "voc_hrj7fsfjhrx3"),
        ("Genoveva", "voc_pnx2xjexe9cu"),
        ("Pedro", "voc_p2um5x8chq7k"),
        ("Vera", "voc_qvscejy24hsw"),
    ],
    "Spanish": [
        ("Darío", "voc_wfd8yxp5hnjz"),
        ("Remedios", "voc_w76j7e9r9qfg"),
        ("Lola", "voc_qzt6e6yspj8v"),
        ("César", "voc_vfj2dkcc3925"),
    ],
}

BREEZE_REMIX_PRESETS = {
    "Age: teenage": "AGE_02_02",
    "Age: young-adult": "AGE_03_01",
    "Age: middle-aged": "AGE_04_01",
    "Age: elderly": "AGE_05_01",
    "Gender: male imitating female": "GEN_01_01",
    "Gender: female imitating male": "GEN_02_01",
    "Pitch: higher pitch": "PIT_01_02",
    "Pitch: lower pitch": "PIT_02_01",
    "Pace: faster": "PAC_01_01",
    "Pace: slower": "PAC_02_01",
    "Energy: energetic": "ENE_01_01",
    "Energy: low-energy": "ENE_02_02",
    "Emotion: excited": "EMO_01_01",
    "Emotion: angry": "EMO_02_02",
    "Emotion: sad": "EMO_03_01",
    "Emotion: afraid": "EMO_04_02",
    "Emotion: awkward": "EMO_05_01",
    "Emotion: sarcastic": "EMO_06_01",
    "Emotion: restrained grief": "EMO_07_02",
    "Emotion: painful": "EMO_08_01",
    "Role-play: stage actor": "ROL_01_01",
    "Role-play: news anchor": "ROL_02_02",
    "Role-play: bedtime storyteller": "ROL_03_01",
    "Delivery: laughing speech": "DEL_01_01",
    "Delivery: crying speech": "DEL_02_01",
    "Delivery: whispered": "DEL_03_01",
    "Delivery: singing": "DEL_04_01",
}

BREEZE_DOUBLE_UNIT_CHARS = (
    "[\\u1100-\\u11ff\\u3005-\\u3007\\u3041-\\u3096\\u309d-\\u309f\\u30a1-\\u30fa\\u30fc-\\u30ff\\u3131-\\u318e"
    "\\u31f0-\\u31ff\\u3400-\\u4dbf\\u4e00-\\u9fff\\ua960-\\ua97f\\uac00-\\ud7a3\\ud7b0-\\ud7ff\\uf900-\\ufaff\\uff66-\\uff9d]"
    "|\\ud82c[\\udc00-\\udd6f]|[\\ud840-\\ud8bf][\\udc00-\\udfff]"
)


def _text_price_badge(widget: str) -> IO.PriceBadge:
    return IO.PriceBadge(
        depends_on=IO.PriceBadgeDepends(widgets=[widget]),
        expr=f"""
        (
          $t := widgets.{widget};
          $type($t) = "string" and $length($t) > 0
            ? {{"type":"usd","usd": ($length($t) + $count($match($t, /{BREEZE_DOUBLE_UNIT_CHARS}/))) * 0.0000429}}
            : {{"type":"usd","usd": 0.0429, "format":{{"suffix":"/1K chars"}}}}
        )
        """,
    )


def _language_input(tooltip: str) -> IO.DynamicCombo.Input:
    names = BREEZE_TTS2_LANGUAGES + [name for name in BREEZE_LANGUAGES if name not in BREEZE_TTS2_LANGUAGES]
    return IO.DynamicCombo.Input(
        "language",
        options=[
            IO.DynamicCombo.Option(
                name,
                [IO.Combo.Input("model", options=list(BREEZE_MODELS))] if name in BREEZE_TTS2_LANGUAGES else [],
            )
            for name in names
        ],
        tooltip=tooltip,
    )


def _language_model(language: dict) -> tuple[str, str]:
    name = language["language"]
    if name in BREEZE_TTS2_LANGUAGES:
        return BREEZE_LANGUAGES[name], BREEZE_MODELS[language["model"]]
    return BREEZE_LANGUAGES[name], BREEZE_MODELS["Breeze TTS 2 Multilingual"]


def _strength_input(tooltip: str) -> IO.Float.Input:
    return IO.Float.Input(
        "strength",
        default=4.0,
        min=1.0,
        max=10.0,
        step=0.1,
        display_mode=IO.NumberDisplay.slider,
        tooltip=tooltip,
    )


def _seed_input() -> IO.Int.Input:
    return IO.Int.Input(
        "seed",
        default=42,
        min=0,
        max=2147483647,
        control_after_generate=True,
        tooltip="Seed controls whether the node should re-run; results are non-deterministic regardless of seed.",
    )


async def _download_preview(cls: type[IO.ComfyNode], generated_voice_id: str) -> Input.Audio:
    audio = await sync_op_raw(
        cls,
        ApiEndpoint(
            path=f"/proxy/breezeblue/v1/voice-previews/{generated_voice_id}/stream",
            query_params={"output_format": "wav"},
        ),
        as_binary=True,
        asset_urls=True,
    )
    return audio_bytes_to_audio_input(audio)


async def _save_voice(cls: type[IO.ComfyNode], generated_voice_id: str, request: BreezeSaveVoiceRequest) -> str:
    response = await sync_op(
        cls,
        ApiEndpoint(path=f"/proxy/breezeblue/v1/voice-previews/{generated_voice_id}/save", method="POST"),
        response_model=BreezeSavedVoice,
        data=request,
    )
    return response.voice_id


async def _preview_job(cls: type[IO.ComfyNode], kind: str, request: BaseModel) -> BreezeVoicePreviewJobStatus:
    job = await sync_op(
        cls,
        ApiEndpoint(path=f"/proxy/breezeblue/v1/voice-previews/{kind}", method="POST"),
        response_model=BreezeVoicePreviewJob,
        data=request,
    )
    result = await poll_op(
        cls,
        ApiEndpoint(path=f"/proxy/breezeblue/v1/voice-previews/{kind}/{job.generation_job_id}"),
        response_model=BreezeVoicePreviewJobStatus,
        status_extractor=lambda r: r.status,
        completed_statuses=["ready", "failed", "cancelled"],
        failed_statuses=[],
        queued_statuses=["admitted", "queued"],
        poll_interval=2.0,
    )
    if result.status != "ready" or not result.generated_voice_id:
        detail = f"{result.error.code}: {result.error.detail}" if result.error else "no preview was produced"
        raise Exception(f"Breeze voice {kind} {result.status}, {detail}")
    return result


class BreezeVoiceSelector(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="BreezeVoiceSelector",
            display_name="Breeze Voice Selector",
            category="partner/audio/Breeze",
            description="Pick a featured Breeze voice, or enter any Breeze voice id.",
            inputs=[
                IO.DynamicCombo.Input(
                    "language",
                    options=[
                        *(
                            IO.DynamicCombo.Option(language, [IO.Combo.Input("voice", options=[n for n, _ in voices])])
                            for language, voices in BREEZE_VOICES.items()
                        ),
                        IO.DynamicCombo.Option(
                            "custom",
                            [
                                IO.String.Input(
                                    "voice_id",
                                    default="",
                                    tooltip="Any Breeze voice id, e.g. a created voice shown by Preview as Text.",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
            outputs=[
                IO.Custom(BREEZE_VOICE).Output(display_name="voice"),
            ],
            is_api_node=False,
        )

    @classmethod
    def execute(cls, language: dict) -> IO.NodeOutput:
        if language["language"] == "custom":
            voice_id = language["voice_id"].strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]+", voice_id):
                raise ValueError("Custom voice id must be a Breeze voice id: letters, digits, '_' and '-' only.")
            return IO.NodeOutput(voice_id)
        return IO.NodeOutput(dict(BREEZE_VOICES[language["language"]])[language["voice"]])


class BreezeTextToSpeech(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="BreezeTextToSpeech",
            display_name="Breeze Text to Speech",
            category="partner/audio/Breeze",
            description="Convert text to speech with Breeze TTS 2.",
            inputs=[
                IO.Custom(BREEZE_VOICE).Input("voice"),
                IO.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    tooltip="Up to 1000 characters. Audio tags such as (laughs) or (sighs) are performed, not read.",
                ),
                _language_input(
                    "Language of the text. English and Chinese can use either model; other languages use "
                    "Breeze TTS 2 Multilingual. A voice sounds best in its own language: for another language, "
                    "use a copy made with Breeze Voice Localize.",
                ),
                IO.DynamicCombo.Input(
                    "delivery",
                    options=[
                        IO.DynamicCombo.Option(
                            "neutral",
                            [
                                IO.Boolean.Input(
                                    "speaker_boost",
                                    default=False,
                                    tooltip="Keep the voice closer to its original sound.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "directed",
                            [
                                IO.String.Input(
                                    "instructions",
                                    multiline=True,
                                    default="",
                                    tooltip="How to perform the text, written in the same language as the text.",
                                ),
                                _strength_input("How strongly the voice follows the instructions."),
                            ],
                        ),
                    ],
                ),
                IO.Float.Input(
                    "speed",
                    default=1.0,
                    min=0.5,
                    max=2.0,
                    step=0.05,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Changes the tempo without changing the pitch.",
                ),
                _seed_input(),
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
            price_badge=_text_price_badge("text"),
        )

    @classmethod
    async def execute(
        cls,
        voice: str,
        text: str,
        language: dict,
        delivery: dict,
        speed: float,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(text, field_name="text", min_length=1)
        validate_string(text, strip_whitespace=False, field_name="text", max_length=1000)
        language_code, model_id = _language_model(language)
        if delivery["delivery"] == "directed":
            instructions = delivery["instructions"].strip() or None
            voice_settings = BreezeVoiceSettings(guidance_scale=delivery["strength"], speed=speed)
        else:
            instructions = None
            voice_settings = BreezeVoiceSettings(use_speaker_boost=delivery["speaker_boost"], speed=speed)
        response = await sync_op_raw(
            cls,
            ApiEndpoint(
                path=f"/proxy/breezeblue/v1/text-to-speech/{voice}",
                method="POST",
                query_params={"output_format": "wav"},
            ),
            data=BreezeTextToSpeechRequest(
                text=text,
                model_id=model_id,
                language_code=language_code,
                instructions=instructions,
                voice_settings=voice_settings,
            ),
            as_binary=True,
            asset_urls=True,
        )
        return IO.NodeOutput(audio_bytes_to_audio_input(response))


class BreezeVoiceDesign(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="BreezeVoiceDesign",
            display_name="Breeze Voice Design",
            category="partner/audio/Breeze",
            description="Create a new voice from a text description.",
            inputs=[
                IO.String.Input(
                    "description",
                    multiline=True,
                    default="",
                    tooltip="Describe the voice, 3 to 500 characters: age, gender, accent, tone, pace.",
                ),
                IO.String.Input(
                    "preview_text",
                    multiline=True,
                    default="",
                    tooltip="The line the new voice speaks in the preview, up to 500 characters. "
                    "Write it in the selected language: the voice is built from this preview.",
                ),
                _language_input(
                    "Language of the preview text. English and Chinese can use either model; other languages use "
                    "Breeze TTS 2 Multilingual.",
                ),
                _strength_input("How closely the voice follows the description."),
                _seed_input(),
            ],
            outputs=[
                IO.Custom(BREEZE_VOICE).Output(display_name="voice"),
                IO.Audio.Output(display_name="preview"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=_text_price_badge("preview_text"),
        )

    @classmethod
    async def execute(
        cls,
        description: str,
        preview_text: str,
        language: dict,
        strength: float,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(description, field_name="description", min_length=3)
        validate_string(description, strip_whitespace=False, field_name="description", max_length=500)
        validate_string(preview_text, field_name="preview_text", min_length=1)
        validate_string(preview_text, strip_whitespace=False, field_name="preview_text", max_length=500)
        language_code, model_id = _language_model(language)
        response = await sync_op(
            cls,
            ApiEndpoint(
                path="/proxy/breezeblue/v1/voice-previews/design",
                method="POST",
                query_params={"output_format": "wav"},
            ),
            response_model=BreezeVoiceDesignResponse,
            data=BreezeVoiceDesignRequest(
                voice_description=description,
                text=preview_text,
                model_id=model_id,
                language_code=language_code,
                guidance_scale=strength,
            ),
        )
        preview = response.previews[0]
        voice_id = await _save_voice(
            cls,
            preview.generated_voice_id,
            BreezeSaveVoiceRequest(voice_name=description.strip()[:80], language_code=language_code),
        )
        return IO.NodeOutput(voice_id, audio_bytes_to_audio_input(base64.b64decode(preview.audio_base_64)))


class BreezeVoiceClone(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="BreezeVoiceClone",
            display_name="Breeze Voice Clone",
            category="partner/audio/Breeze",
            description="Create a voice from a recording of one speaker.",
            inputs=[
                IO.Audio.Input(
                    "audio",
                    tooltip="One speaker, at least 3 seconds. Only the first 60 seconds are used.",
                ),
                IO.Boolean.Input("remove_background_noise", default=False),
                _seed_input(),
            ],
            outputs=[
                IO.Custom(BREEZE_VOICE).Output(display_name="voice"),
                IO.Audio.Output(display_name="preview"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"range_usd","min_usd":0.0031317,"max_usd":0.0126126,"format":{"approximate":true}}""",
            ),
        )

    @classmethod
    async def execute(cls, audio: Input.Audio, remove_background_noise: bool, seed: int) -> IO.NodeOutput:
        validate_audio_duration(audio, min_duration=3)
        sample_rate = int(audio["sample_rate"])
        waveform = audio["waveform"][..., : 60 * sample_rate]
        reference = audio_ndarray_to_bytesio(audio_tensor_to_contiguous_ndarray(waveform), sample_rate, "mp4", "aac")
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/breezeblue/v1/voice-previews/clone", method="POST"),
            response_model=BreezeVoiceCloneResponse,
            data=BreezeVoiceCloneRequest(remove_background_noise=remove_background_noise),
            files=[("files", ("reference.mp4", reference.getvalue(), "audio/mp4"))],
            content_type="multipart/form-data",
        )
        preview = await _download_preview(cls, response.generated_voice_id)
        voice_id = await _save_voice(cls, response.generated_voice_id, BreezeSaveVoiceRequest())
        return IO.NodeOutput(voice_id, preview)


class BreezeVoiceRemix(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="BreezeVoiceRemix",
            display_name="Breeze Voice Remix",
            category="partner/audio/Breeze",
            description="Create a changed copy of a voice in the same language, from a description or a preset.",
            inputs=[
                IO.Custom(BREEZE_VOICE).Input("voice"),
                IO.DynamicCombo.Input(
                    "change",
                    options=[
                        IO.DynamicCombo.Option(
                            "custom",
                            [
                                IO.String.Input(
                                    "prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Describe how the voice should change, e.g. older and husky.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "preset",
                            [IO.Combo.Input("preset", options=list(BREEZE_REMIX_PRESETS))],
                        ),
                    ],
                ),
                _strength_input("Higher values change the voice more."),
                _seed_input(),
            ],
            outputs=[
                IO.Custom(BREEZE_VOICE).Output(display_name="voice"),
                IO.Audio.Output(display_name="preview"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"range_usd","min_usd":0.0033891,"max_usd":0.0072072,"format":{"approximate":true}}""",
            ),
        )

    @classmethod
    async def execute(cls, voice: str, change: dict, strength: float, seed: int) -> IO.NodeOutput:
        if change["change"] == "custom":
            validate_string(change["prompt"], field_name="prompt", min_length=1)
            request = BreezeVoiceRemixRequest(voice_id=voice, prompt=change["prompt"], guidance_scale=strength)
        else:
            request = BreezeVoiceRemixRequest(
                voice_id=voice,
                preset_id=BREEZE_REMIX_PRESETS[change["preset"]],
                guidance_scale=strength,
            )
        result = await _preview_job(cls, "remix", request)
        preview = await _download_preview(cls, result.generated_voice_id)
        voice_id = await _save_voice(
            cls,
            result.generated_voice_id,
            BreezeSaveVoiceRequest(language_code=result.language_code),
        )
        return IO.NodeOutput(voice_id, preview)


class BreezeVoiceLocalize(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="BreezeVoiceLocalize",
            display_name="Breeze Voice Localize",
            category="partner/audio/Breeze",
            description="Create a copy of a voice that speaks another language natively.",
            inputs=[
                IO.Custom(BREEZE_VOICE).Input("voice"),
                IO.Combo.Input(
                    "language",
                    options=list(BREEZE_LANGUAGES),
                    default="English",
                    tooltip="Language the new voice speaks.",
                ),
                _seed_input(),
            ],
            outputs=[
                IO.Custom(BREEZE_VOICE).Output(display_name="voice"),
                IO.Audio.Output(display_name="preview"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"range_usd","min_usd":0.0036036,"max_usd":0.0108537,"format":{"approximate":true}}""",
            ),
        )

    @classmethod
    async def execute(cls, voice: str, language: str, seed: int) -> IO.NodeOutput:
        language_code = BREEZE_LANGUAGES[language]
        result = await _preview_job(cls, "localize", BreezeVoiceLocalizeRequest(voice_id=voice, language_code=language_code))
        preview = await _download_preview(cls, result.generated_voice_id)
        voice_id = await _save_voice(
            cls,
            result.generated_voice_id,
            BreezeSaveVoiceRequest(voice_name=f"{voice} ({language})", language_code=language_code),
        )
        return IO.NodeOutput(voice_id, preview)


class BreezeExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            BreezeVoiceSelector,
            BreezeTextToSpeech,
            BreezeVoiceDesign,
            BreezeVoiceClone,
            BreezeVoiceRemix,
            BreezeVoiceLocalize,
        ]


async def comfy_entrypoint() -> BreezeExtension:
    return BreezeExtension()
