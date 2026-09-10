import os

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.cambai import (
    CambAIDialogueItem,
    CambAIPollResult,
    CambAITaskResponse,
    CambAITextToSoundRequest,
    CambAITranslateRequest,
    CambAITranslateResult,
    CambAITTSRequest,
    CambAIVoiceCloneResponse,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    audio_bytes_to_audio_input,
    audio_ndarray_to_bytesio,
    audio_tensor_to_contiguous_ndarray,
    poll_op,
    sync_op,
    sync_op_raw,
    validate_string,
)

CAMBAI_API_BASE = "https://client.camb.ai/apis"
CAMBAI_VOICE = "CAMBAI_VOICE"
CAMBAI_GENDER_MAP = {"male": 0, "female": 1, "other": 2, "prefer not to say": 9}


def _cambai_endpoint(route: str, method: str = "GET") -> ApiEndpoint:
    api_key = os.environ.get("CAMBAI_API_KEY", "")
    return ApiEndpoint(
        path=f"{CAMBAI_API_BASE}/{route}",
        method=method,
        headers={"x-api-key": api_key},
    )

CAMBAI_LANGUAGES_TTS = [
    "en-us", "es-es", "fr-fr", "de-de", "it-it", "pt-br",
    "zh-cn", "ja-jp", "ko-kr", "ar-sa", "hi-in", "ru-ru",
    "nl-nl", "pl-pl", "tr-tr", "sv-se",
]

CAMBAI_LANGUAGE_MAP = {
    "English": 1, "Spanish": 54, "French": 76, "German": 31,
    "Italian": 83, "Portuguese": 112, "Chinese": 139, "Japanese": 88,
    "Korean": 93, "Arabic": 4, "Hindi": 73, "Russian": 116,
    "Dutch": 103, "Polish": 110, "Turkish": 133, "Swedish": 125,
}

CAMBAI_TRANSCRIPTION_LANGUAGE_MAP = {
    "English": 1, "Spanish": 54, "French": 76, "German": 31,
    "Italian": 83, "Portuguese": 112, "Chinese": 139, "Japanese": 88,
    "Korean": 93, "Arabic": 4, "Hindi": 73, "Russian": 116,
}


class CambAIVoiceSelector(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="CambAIVoiceSelector",
            display_name="CAMB AI Voice Selector",
            category="api node/audio/CAMB AI",
            description="Select a CAMB AI voice by ID for text-to-speech generation.",
            inputs=[
                IO.Int.Input(
                    "voice_id",
                    default=147320,
                    min=1,
                    max=999999999,
                    tooltip="Voice ID to use for CAMB AI TTS.",
                ),
            ],
            outputs=[
                IO.Custom(CAMBAI_VOICE).Output(display_name="voice"),
            ],
            is_api_node=False,
        )

    @classmethod
    def execute(cls, voice_id: int) -> IO.NodeOutput:
        return IO.NodeOutput(voice_id)


class CambAIVoiceClone(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="CambAIVoiceClone",
            display_name="CAMB AI Voice Clone",
            category="api node/audio/CAMB AI",
            description="Create a custom cloned voice from an audio sample.",
            inputs=[
                IO.Audio.Input(
                    "audio",
                    tooltip="Audio sample of the voice to clone.",
                ),
                IO.String.Input(
                    "voice_name",
                    default="My Custom Voice",
                    tooltip="Name for the cloned voice.",
                ),
                IO.Combo.Input(
                    "gender",
                    options=["male", "female", "other", "prefer not to say"],
                    default="male",
                    tooltip="Gender of the voice to clone.",
                ),
            ],
            outputs=[
                IO.Custom(CAMBAI_VOICE).Output(display_name="voice"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        audio: Input.Audio,
        voice_name: str,
        gender: str,
    ) -> IO.NodeOutput:
        audio_data_np = audio_tensor_to_contiguous_ndarray(audio["waveform"])
        audio_bytes_io = audio_ndarray_to_bytesio(audio_data_np, audio["sample_rate"], "wav", "pcm_s16le")

        response = await sync_op(
            cls,
            _cambai_endpoint("create-custom-voice", "POST"),
            response_model=CambAIVoiceCloneResponse,
            data=None,
            files={
                "voice_name": (None, voice_name),
                "gender": (None, str(CAMBAI_GENDER_MAP[gender])),
                "file": ("voice.wav", audio_bytes_io.getvalue(), "audio/wav"),
            },
            content_type="multipart/form-data",
        )
        return IO.NodeOutput(response.voice_id)


class CambAITextToSpeech(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="CambAITextToSpeech",
            display_name="CAMB AI Text to Speech",
            category="api node/audio/CAMB AI",
            description="Convert text to speech using CAMB AI TTS models.",
            inputs=[
                IO.Custom(CAMBAI_VOICE).Input(
                    "voice",
                    tooltip="Voice to use for speech synthesis. Connect from Voice Selector or Voice Clone.",
                ),
                IO.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    tooltip="The text to convert to speech.",
                ),
                IO.Combo.Input(
                    "language",
                    options=CAMBAI_LANGUAGES_TTS,
                    default="en-us",
                    tooltip="Language for speech synthesis.",
                ),
                IO.Combo.Input(
                    "model",
                    options=["mars-flash", "mars-pro", "mars-instruct"],
                    default="mars-flash",
                    tooltip="TTS model to use.",
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
        )

    @classmethod
    async def execute(
        cls,
        voice: int,
        text: str,
        language: str,
        model: str,
    ) -> IO.NodeOutput:
        validate_string(text, min_length=1)
        request = CambAITTSRequest(
            text=text,
            voice_id=voice,
            language=language,
            speech_model=model,
        )
        response = await sync_op_raw(
            cls,
            _cambai_endpoint("tts-stream", "POST"),
            data=request,
            as_binary=True,
        )
        return IO.NodeOutput(audio_bytes_to_audio_input(response))


class CambAITranslation(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="CambAITranslation",
            display_name="CAMB AI Translation",
            category="api node/text/CAMB AI",
            description="Translate text between languages using CAMB AI.",
            inputs=[
                IO.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    tooltip="Text to translate.",
                ),
                IO.Combo.Input(
                    "source_language",
                    options=list(CAMBAI_LANGUAGE_MAP.keys()),
                    default="English",
                    tooltip="Source language.",
                ),
                IO.Combo.Input(
                    "target_language",
                    options=list(CAMBAI_LANGUAGE_MAP.keys()),
                    default="Spanish",
                    tooltip="Target language.",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="text"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        text: str,
        source_language: str,
        target_language: str,
    ) -> IO.NodeOutput:
        validate_string(text, min_length=1)
        src_id = CAMBAI_LANGUAGE_MAP[source_language]
        tgt_id = CAMBAI_LANGUAGE_MAP[target_language]

        request = CambAITranslateRequest(
            source_language=src_id,
            target_language=tgt_id,
            texts=[text],
        )
        response = await sync_op(
            cls,
            _cambai_endpoint("translate", "POST"),
            response_model=CambAITaskResponse,
            data=request,
        )

        poll_result = await poll_op(
            cls,
            _cambai_endpoint(f"translate/{response.task_id}"),
            response_model=CambAIPollResult,
            status_extractor=lambda x: x.status,
        )

        if not poll_result.run_id:
            raise ValueError("No run_id returned from CAMB AI translation task.")

        result = await sync_op(
            cls,
            _cambai_endpoint(f"translation-result/{poll_result.run_id}"),
            response_model=CambAITranslateResult,
        )

        if result.texts and len(result.texts) > 0:
            return IO.NodeOutput(result.texts[0])
        return IO.NodeOutput("")


class CambAITranscription(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="CambAITranscription",
            display_name="CAMB AI Transcription",
            category="api node/audio/CAMB AI",
            description="Transcribe audio to text using CAMB AI.",
            inputs=[
                IO.Audio.Input(
                    "audio",
                    tooltip="Audio to transcribe.",
                ),
                IO.Combo.Input(
                    "language",
                    options=list(CAMBAI_TRANSCRIPTION_LANGUAGE_MAP.keys()),
                    default="English",
                    tooltip="Language of the audio.",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="text"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        audio: Input.Audio,
        language: str,
    ) -> IO.NodeOutput:
        lang_id = CAMBAI_TRANSCRIPTION_LANGUAGE_MAP[language]
        audio_data_np = audio_tensor_to_contiguous_ndarray(audio["waveform"])
        audio_bytes_io = audio_ndarray_to_bytesio(audio_data_np, audio["sample_rate"], "wav", "pcm_s16le")

        response = await sync_op(
            cls,
            _cambai_endpoint("transcribe", "POST"),
            response_model=CambAITaskResponse,
            data=None,
            files={
                "language": (None, str(lang_id)),
                "media_file": ("audio.wav", audio_bytes_io.getvalue(), "audio/wav"),
            },
            content_type="multipart/form-data",
        )

        poll_result = await poll_op(
            cls,
            _cambai_endpoint(f"transcribe/{response.task_id}"),
            response_model=CambAIPollResult,
            status_extractor=lambda x: x.status,
        )

        if not poll_result.run_id:
            raise ValueError("No run_id returned from CAMB AI transcription task.")

        result_raw = await sync_op_raw(
            cls,
            _cambai_endpoint(f"transcription-result/{poll_result.run_id}"),
        )

        transcript = result_raw.get("transcript", [])
        dialogues = [CambAIDialogueItem(**item) for item in transcript]
        text = " ".join(item.text for item in dialogues)
        return IO.NodeOutput(text)


class CambAITextToSound(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="CambAITextToSound",
            display_name="CAMB AI Text to Sound",
            category="api node/audio/CAMB AI",
            description="Generate sound effects or music from a text description using CAMB AI.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text description of the sound to generate.",
                ),
                IO.Combo.Input(
                    "audio_type",
                    options=["sound", "music"],
                    default="sound",
                    tooltip="Type of audio to generate.",
                ),
                IO.Float.Input(
                    "duration",
                    default=5.0,
                    min=0.5,
                    max=30.0,
                    step=0.5,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Duration of generated audio in seconds.",
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
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        audio_type: str,
        duration: float,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1)
        request = CambAITextToSoundRequest(
            prompt=prompt,
            audio_type=audio_type,
            duration=duration,
        )
        response = await sync_op(
            cls,
            _cambai_endpoint("text-to-sound", "POST"),
            response_model=CambAITaskResponse,
            data=request,
        )

        poll_result = await poll_op(
            cls,
            _cambai_endpoint(f"text-to-sound/{response.task_id}"),
            response_model=CambAIPollResult,
            status_extractor=lambda x: x.status,
        )

        if not poll_result.run_id:
            raise ValueError("No run_id returned from CAMB AI text-to-sound task.")

        audio_bytes = await sync_op_raw(
            cls,
            _cambai_endpoint(f"text-to-sound-result/{poll_result.run_id}"),
            as_binary=True,
        )
        return IO.NodeOutput(audio_bytes_to_audio_input(audio_bytes))


class CambAIExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            CambAIVoiceSelector,
            CambAIVoiceClone,
            CambAITextToSpeech,
            CambAITranslation,
            CambAITranscription,
            CambAITextToSound,
        ]


async def comfy_entrypoint() -> CambAIExtension:
    return CambAIExtension()
