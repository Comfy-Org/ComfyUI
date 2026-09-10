import uuid

import torch
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.heygen import (
    HEYGEN_AVATAR_MAP,
    HEYGEN_AVATAR_OPTIONS,
    HEYGEN_TRANSLATE_LANGUAGES,
    HEYGEN_VOICE_GENERAL_MAP,
    HEYGEN_VOICE_GENERAL_OPTIONS,
    HEYGEN_VOICE_TTS_MAP,
    HEYGEN_VOICE_TTS_OPTIONS,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    audio_bytes_to_audio_input,
    download_url_as_bytesio,
    download_url_to_image_tensor,
    download_url_to_video_output,
    downscale_image_tensor_by_max_side,
    get_number_of_images,
    poll_op_raw,
    sync_op_raw,
    upload_audio_to_comfyapi,
    upload_image_to_comfyapi,
    upload_images_to_comfyapi,
    upload_video_to_comfyapi,
    validate_string,
)
from server import PromptServer

_VIDEOS_PATH = "/proxy/heygen/v3/videos"
_TRANSLATIONS_PATH = "/proxy/heygen/v3/video-translations"
_SPEECH_PATH = "/proxy/heygen/v3/voices/speech"
_AVATARS_PATH = "/proxy/heygen/v3/avatars"
_LOOKS_PATH = "/proxy/heygen/v3/avatars/looks"

_DEFAULT_VOICE_OPTION = "(avatar's default voice)"

_AVATARS_BY_ENGINE = {
    e: [label for label, (_aid, _atype, engines) in HEYGEN_AVATAR_MAP.items() if e in engines]
    for e in ("avatar_iv", "avatar_iii", "avatar_v")
}


async def _apply_speech_source(cls: type[IO.ComfyNode], payload: dict, speech: dict, require_voice: bool) -> None:
    """Fill script/audio speech fields of a /v3/videos payload from the DynamicCombo dict."""
    if speech["speech"] == "audio":
        payload["audio_url"] = await upload_audio_to_comfyapi(
            cls, speech["audio"], container_format="mp3", codec_name="libmp3lame", mime_type="audio/mpeg"
        )
    elif speech["speech"] == "script":
        validate_string(speech["text"], strip_whitespace=True, min_length=1, max_length=5000)
        payload["script"] = speech["text"]
        voice_id = speech.get("custom_voice_id", "").strip()
        if not voice_id and speech["voice"] != _DEFAULT_VOICE_OPTION:
            voice_id = HEYGEN_VOICE_GENERAL_MAP[speech["voice"]]
        if voice_id:
            payload["voice_id"] = voice_id
        elif require_voice:
            raise ValueError("A voice is required when driving the video with a text script.")
        speed = speech.get("voice_speed", 1.0)
        if speed != 1.0:
            payload["voice_settings"] = {"speed": round(speed, 2)}


async def _create_and_poll_video(cls: type[IO.ComfyNode], payload: dict) -> dict:
    """POST a /v3/videos payload, poll until terminal, and return the final video data."""
    created = await sync_op_raw(
        cls,
        ApiEndpoint(path=_VIDEOS_PATH, method="POST", headers={"Idempotency-Key": uuid.uuid4().hex}),
        data=payload,
    )
    video_id = (created.get("data") or {}).get("video_id")
    if not video_id:
        raise ValueError(f"HeyGen did not return a video_id: {created}")
    final = await poll_op_raw(
        cls,
        ApiEndpoint(path=f"{_VIDEOS_PATH}/{video_id}"),
        status_extractor=lambda r: (r.get("data") or {}).get("status"),
        queued_statuses=["pending", "waiting"],
        poll_interval=5.0,
    )
    data = final["data"]
    if not data.get("video_url"):
        raise ValueError(f"HeyGen returned no video_url for video {video_id}.")
    return data


async def _resolve_avatar(
    cls: type[IO.ComfyNode], avatar_label: str, custom_avatar_id: str, engine_choice: str
) -> tuple[str, str | None]:
    """Resolve (avatar_id, engine_type) from the combo/override + engine widgets."""
    custom_avatar_id = custom_avatar_id.strip()
    if custom_avatar_id:
        look = (
            await sync_op_raw(
                cls,
                ApiEndpoint(path=f"{_LOOKS_PATH}/{custom_avatar_id}"),
                final_label_on_success=None,
            )
        ).get("data") or {}
        avatar_id = custom_avatar_id
        avatar_label = look.get("name") or custom_avatar_id
        supported = look.get("supported_api_engines") or []
    else:
        avatar_id, avatar_type, supported = HEYGEN_AVATAR_MAP[avatar_label]

    if engine_choice == "auto":
        engine = next((e for e in ("avatar_iv", "avatar_iii", "avatar_v") if e in supported), None)
    else:
        engine = engine_choice
        if supported and engine not in supported:
            raise ValueError(
                f"Avatar '{avatar_label}' does not support the {engine} engine "
                f"(supported: {', '.join(supported)}). Set engine to 'auto' to pick "
                "a compatible engine automatically."
            )
    return avatar_id, engine


class HeyGenTalkingPhotoNode(IO.ComfyNode):
    """Animate a still image of a person into a lip-synced talking video."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="HeyGenTalkingPhotoNode",
            display_name="HeyGen Talking Photo",
            category="partner/video/HeyGen",
            description="Animate any image of a person into a lip-synced talking video "
            "(HeyGen Avatar IV). Drive it with a text script or your own audio.",
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="Image of a person to animate. Downscaled automatically if larger than 2K.",
                ),
                IO.DynamicCombo.Input(
                    "speech",
                    display_name="speech source",
                    options=[
                        IO.DynamicCombo.Option(
                            "script",
                            [
                                IO.String.Input(
                                    "text",
                                    multiline=True,
                                    default="",
                                    tooltip="Text for the avatar to speak (up to 5000 characters). "
                                    "The generated speech must be at least 1 second long.",
                                ),
                                IO.Combo.Input(
                                    "voice",
                                    options=HEYGEN_VOICE_GENERAL_OPTIONS,
                                    tooltip="Voice for the script (HeyGen's most popular voices).",
                                ),
                                IO.String.Input(
                                    "custom_voice_id",
                                    default="",
                                    optional=True,
                                    tooltip="Optional HeyGen voice ID. When set, overrides the voice selected above. "
                                    "Any voice from HeyGen's library (2000+) can be used.",
                                ),
                                IO.Float.Input(
                                    "voice_speed",
                                    default=1.0,
                                    min=0.5,
                                    max=1.5,
                                    step=0.05,
                                    optional=True,
                                    tooltip="Speech speed multiplier.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "audio",
                            [
                                IO.Audio.Input(
                                    "audio",
                                    tooltip="Audio for the avatar to lip-sync, up to 10 minutes.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Drive the avatar with a text script (HeyGen text-to-speech) or your own audio.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["720p", "1080p"],
                    default="1080p",
                    optional=True,
                    tooltip="Output video resolution.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["auto", "16:9", "9:16", "1:1", "4:5", "5:4"],
                    default="auto",
                    optional=True,
                    tooltip="Output aspect ratio. 'auto' follows the input image.",
                ),
                IO.Combo.Input(
                    "expressiveness",
                    options=["low", "medium", "high"],
                    default="low",
                    optional=True,
                    tooltip="How expressive the animated face and gestures are.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    optional=True,
                    tooltip="Not sent to HeyGen; change it to force a re-run.",
                ),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.0715,"format":{"suffix":"/second"}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        speech: dict,
        resolution: str = "1080p",
        aspect_ratio: str = "auto",
        expressiveness: str = "low",
        seed: int = 0,
    ) -> IO.NodeOutput:
        image = downscale_image_tensor_by_max_side(image, max_side=2000)
        image_url = await upload_image_to_comfyapi(cls, image, mime_type="image/png", total_pixels=None)
        payload = {
            "type": "image",
            "image": {"type": "url", "url": image_url},
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "expressiveness": expressiveness,
            "title": "ComfyUI Talking Photo",
        }
        await _apply_speech_source(cls, payload, speech, require_voice=True)
        video = await _create_and_poll_video(cls, payload)
        return IO.NodeOutput(await download_url_to_video_output(video["video_url"]))


class HeyGenAvatarVideoNode(IO.ComfyNode):
    """Generate a presenter video from a HeyGen avatar look."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="HeyGenAvatarVideoNode",
            display_name="HeyGen Avatar Video",
            category="partner/video/HeyGen",
            description="Generate a talking-presenter video from a HeyGen avatar. "
            "Includes HeyGen's most popular public avatars; any look ID can be supplied as an override.",
            inputs=[
                IO.DynamicCombo.Input(
                    "engine",
                    options=[
                        IO.DynamicCombo.Option(
                            "auto",
                            [
                                IO.Combo.Input(
                                    "avatar",
                                    options=HEYGEN_AVATAR_OPTIONS,
                                    tooltip="Avatar look to present the video (curated from HeyGen's "
                                    "public library). The best engine the look supports is chosen "
                                    "automatically.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "avatar_iv",
                            [
                                IO.Combo.Input(
                                    "avatar",
                                    options=_AVATARS_BY_ENGINE["avatar_iv"],
                                    tooltip="Avatar looks that support the Avatar IV engine.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "avatar_iii",
                            [
                                IO.Combo.Input(
                                    "avatar",
                                    options=_AVATARS_BY_ENGINE["avatar_iii"],
                                    tooltip="Avatar looks that support the Avatar III engine.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "avatar_v",
                            [
                                IO.Combo.Input(
                                    "avatar",
                                    options=_AVATARS_BY_ENGINE["avatar_v"],
                                    tooltip="Avatar looks that support the Avatar V engine.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Rendering engine; each choice lists only the avatars that support it. "
                    "'auto' offers every avatar and picks its best engine (Avatar IV preferred). "
                    "Avatar V is highest fidelity, Avatar III is the most affordable.",
                ),
                IO.String.Input(
                    "custom_avatar_id",
                    default="",
                    optional=True,
                    tooltip="Optional HeyGen avatar look ID. When set, overrides the avatar selected above. "
                    "Any of HeyGen's 3000+ public looks (or your private avatars) can be used.",
                ),
                IO.DynamicCombo.Input(
                    "speech",
                    display_name="speech source",
                    options=[
                        IO.DynamicCombo.Option(
                            "script",
                            [
                                IO.String.Input(
                                    "text",
                                    multiline=True,
                                    default="",
                                    tooltip="Text for the avatar to speak (up to 5000 characters). "
                                    "The generated speech must be at least 1 second long.",
                                ),
                                IO.Combo.Input(
                                    "voice",
                                    options=[_DEFAULT_VOICE_OPTION] + HEYGEN_VOICE_GENERAL_OPTIONS,
                                    tooltip="Voice for the script. The default option uses the voice HeyGen assigned to the avatar.",
                                ),
                                IO.String.Input(
                                    "custom_voice_id",
                                    default="",
                                    optional=True,
                                    tooltip="Optional HeyGen voice ID. When set, overrides the voice selected above. "
                                    "Any voice from HeyGen's library (2000+) can be used.",
                                ),
                                IO.Float.Input(
                                    "voice_speed",
                                    default=1.0,
                                    min=0.5,
                                    max=1.5,
                                    step=0.05,
                                    optional=True,
                                    tooltip="Speech speed multiplier.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "audio",
                            [
                                IO.Audio.Input(
                                    "audio",
                                    tooltip="Audio for the avatar to lip-sync, up to 10 minutes.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Drive the avatar with a text script (HeyGen text-to-speech) or your own audio.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["720p", "1080p"],
                    default="1080p",
                    optional=True,
                    tooltip="Output video resolution.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["auto", "16:9", "9:16", "1:1", "4:5", "5:4"],
                    default="auto",
                    optional=True,
                    tooltip="Output aspect ratio. 'auto' follows the avatar's source footage.",
                ),
                IO.String.Input(
                    "background_color",
                    default="",
                    optional=True,
                    tooltip="Optional solid background color as a hex code (e.g. '#00ff00'). "
                    "Leave empty for the avatar's own background.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    optional=True,
                    tooltip="Not sent to HeyGen; change it to force a re-run.",
                ),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["engine"]),
                expr="""
                widgets.engine = "avatar_iii"
                  ? {"type":"range_usd","min_usd":0.0143,"max_usd":0.023595,"format":{"suffix":"/second"}}
                  : widgets.engine = "avatar_v"
                  ? {"type":"usd","usd":0.1716,"format":{"suffix":"/second"}}
                  : widgets.engine = "avatar_iv"
                  ? {"type":"range_usd","min_usd":0.055055,"max_usd":0.115115,"format":{"suffix":"/second"}}
                  : {"type":"range_usd","min_usd":0.0143,"max_usd":0.1716,"format":{"suffix":"/second"}}
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        engine: dict,
        speech: dict,
        custom_avatar_id: str = "",
        resolution: str = "1080p",
        aspect_ratio: str = "auto",
        background_color: str = "",
        seed: int = 0,
    ) -> IO.NodeOutput:
        avatar_id, engine_type = await _resolve_avatar(cls, engine["avatar"], custom_avatar_id, engine["engine"])
        payload = {
            "type": "avatar",
            "avatar_id": avatar_id,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "title": "ComfyUI Avatar Video",
        }
        if engine_type:
            payload["engine"] = {"type": engine_type}
        background_color = background_color.strip()
        if background_color:
            if not background_color.startswith("#"):
                raise ValueError("background_color must be a hex color code like '#00ff00'.")
            payload["background"] = {"type": "color", "value": background_color}
        await _apply_speech_source(cls, payload, speech, require_voice=False)
        video = await _create_and_poll_video(cls, payload)
        return IO.NodeOutput(await download_url_to_video_output(video["video_url"]))


class HeyGenCreateAvatarNode(IO.ComfyNode):
    """Create a reusable HeyGen avatar from a photo or a text prompt."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="HeyGenCreateAvatarNode",
            display_name="HeyGen Create Avatar",
            category="partner/video/HeyGen",
            description="Create your own reusable HeyGen avatar from a photo of a person or "
            "from a text prompt (a generated character). Feed the resulting avatar_id into "
            "HeyGen Avatar Video's custom_avatar_id — and save the ID somewhere to reuse the "
            "avatar in future workflows.",
            inputs=[
                IO.DynamicCombo.Input(
                    "source",
                    options=[
                        IO.DynamicCombo.Option(
                            "prompt",
                            [
                                IO.String.Input(
                                    "prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Description of the avatar to generate (up to 1000 characters).",
                                ),
                                IO.Autogrow.Input(
                                    "reference_images",
                                    template=IO.Autogrow.TemplateNames(
                                        IO.Image.Input("ref_image"),
                                        names=[f"ref_image_{i}" for i in range(1, 4)],
                                        min=0,
                                    ),
                                    tooltip="Up to 3 reference images guiding the generated look.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "photo",
                            [
                                IO.Image.Input(
                                    "identity_photo",
                                    tooltip="Photo of the person to turn into an avatar. "
                                    "Downscaled automatically if larger than 2K.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Generate a new character from a text prompt, or create the avatar "
                    "from a connected photo of a person.",
                ),
            ],
            outputs=[
                IO.String.Output(
                    display_name="avatar_id",
                    tooltip="Avatar look ID. Pass it to HeyGen Avatar Video's custom_avatar_id; "
                    "save it to reuse the avatar later.",
                ),
                IO.Image.Output(display_name="preview"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["source"]),
                expr="""{"type":"usd","usd": widgets.source = "photo" ? 1.8876 : 1.43}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        source: dict,
    ) -> IO.NodeOutput:
        payload: dict = {"name": "ComfyUI Avatar"}
        if source["source"] == "photo":
            image = downscale_image_tensor_by_max_side(source["identity_photo"], max_side=2000)
            image_url = await upload_image_to_comfyapi(cls, image, mime_type="image/png", total_pixels=None)
            payload["type"] = "photo"
            payload["file"] = {"type": "url", "url": image_url}
        else:
            validate_string(source["prompt"], strip_whitespace=True, min_length=1, max_length=1000)
            payload["type"] = "prompt"
            payload["prompt"] = source["prompt"]
            ref_tensors = [t for t in (source.get("reference_images") or {}).values() if t is not None]
            if ref_tensors:
                n_images = sum(get_number_of_images(t) for t in ref_tensors)
                if n_images > 3:
                    raise ValueError(f"HeyGen accepts at most 3 reference images; got {n_images}.")
                scaled = [downscale_image_tensor_by_max_side(t, max_side=2000) for t in ref_tensors]
                ref_urls = await upload_images_to_comfyapi(
                    cls, scaled, max_images=3, mime_type="image/png", total_pixels=None
                )
                payload["reference_images"] = [{"type": "url", "url": u} for u in ref_urls]
        created = await sync_op_raw(
            cls,
            ApiEndpoint(path=_AVATARS_PATH, method="POST"),
            data=payload,
        )
        look_id = ((created.get("data") or {}).get("avatar_item") or {}).get("id")
        if not look_id:
            raise ValueError(f"HeyGen did not return an avatar: {created}")
        final = await poll_op_raw(
            cls,
            ApiEndpoint(path=f"{_LOOKS_PATH}/{look_id}"),
            # A missing status means the look needed no training and is ready.
            status_extractor=lambda r: (r.get("data") or {}).get("status") or "completed",
            failed_statuses=["failed", "pending_consent"],
            poll_interval=5.0,
        )
        data = final["data"]
        if data.get("preview_image_url"):
            preview = await download_url_to_image_tensor(data["preview_image_url"])
        else:
            preview = torch.zeros(1, 64, 64, 3)
        PromptServer.instance.send_progress_text(
            f"Please save the avatar_id for reuse.\n\navatar_id: {look_id}",
            cls.hidden.unique_id,
        )
        return IO.NodeOutput(look_id, preview)


class HeyGenVideoTranslateNode(IO.ComfyNode):
    """Translate a spoken video into another language with voice cloning and lip sync."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="HeyGenVideoTranslateNode",
            display_name="HeyGen Video Translate",
            category="partner/video/HeyGen",
            description="Translate a spoken video into another language. Clones the original "
            "speaker's voice and re-animates the mouth to match the translated speech.",
            inputs=[
                IO.Video.Input(
                    "video",
                    tooltip="Video with speech to translate.",
                ),
                IO.Combo.Input(
                    "output_language",
                    options=HEYGEN_TRANSLATE_LANGUAGES,
                    tooltip="Target language for the translated video.",
                ),
                IO.Combo.Input(
                    "mode",
                    options=["speed", "precision"],
                    default="speed",
                    tooltip="'speed' is faster; 'precision' produces higher-quality lip sync at a higher price.",
                ),
                IO.Boolean.Input(
                    "translate_audio_only",
                    default=False,
                    optional=True,
                    tooltip="Only swap the audio track, keeping the original mouth movements (no lip sync).",
                ),
                IO.Int.Input(
                    "speaker_count",
                    default=0,
                    min=0,
                    max=10,
                    optional=True,
                    tooltip="Number of speakers in the video. 0 = detect automatically.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    optional=True,
                    tooltip="Not sent to HeyGen; change it to force a re-run.",
                ),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["mode", "translate_audio_only"]),
                expr="""{"type":"usd","usd": widgets.mode = "precision" ? 0.03575 """
                """: widgets.translate_audio_only = true ? 0.013585 : 0.019305,"""
                """"format":{"suffix":"/second"}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        output_language: str,
        mode: str,
        translate_audio_only: bool = False,
        speaker_count: int = 0,
        seed: int = 0,
    ) -> IO.NodeOutput:
        video_url = await upload_video_to_comfyapi(cls, video)
        payload = {
            "video": {"type": "url", "url": video_url},
            "output_languages": [output_language],
            "mode": mode,
            "translate_audio_only": translate_audio_only,
            "title": "ComfyUI Video Translate",
        }
        if speaker_count > 0:
            payload["speaker_num"] = speaker_count
        created = await sync_op_raw(
            cls,
            ApiEndpoint(path=_TRANSLATIONS_PATH, method="POST"),
            data=payload,
        )
        translation_ids = (created.get("data") or {}).get("video_translation_ids") or []
        if not translation_ids:
            raise ValueError(f"HeyGen did not return a translation ID: {created}")
        final = await poll_op_raw(
            cls,
            ApiEndpoint(path=f"{_TRANSLATIONS_PATH}/{translation_ids[0]}"),
            status_extractor=lambda r: (r.get("data") or {}).get("status"),
            queued_statuses=["pending"],
            poll_interval=5.0,
        )
        data = final["data"]
        if not data.get("video_url"):
            raise ValueError(f"HeyGen returned no video_url for translation {translation_ids[0]}.")
        return IO.NodeOutput(await download_url_to_video_output(data["video_url"]))


class HeyGenTextToSpeechNode(IO.ComfyNode):
    """Synthesize speech audio from text with HeyGen's Starfish TTS engine."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="HeyGenTextToSpeechNode",
            display_name="HeyGen Text to Speech",
            category="partner/audio/HeyGen",
            description="Generate speech audio from text using HeyGen's Starfish TTS engine. "
            "Includes HeyGen's most popular voices across 17 languages.",
            inputs=[
                IO.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    tooltip="Text to synthesize (up to 5000 characters). The generated speech "
                    "must be at least 1 second long.",
                ),
                IO.Combo.Input(
                    "voice",
                    options=HEYGEN_VOICE_TTS_OPTIONS,
                    tooltip="Voice to use (curated from HeyGen's most popular Starfish-compatible voices).",
                ),
                IO.String.Input(
                    "custom_voice_id",
                    default="",
                    optional=True,
                    tooltip="Optional HeyGen voice ID. When set, overrides the voice selected above. "
                    "The voice must support the Starfish engine.",
                ),
                IO.Float.Input(
                    "speed",
                    default=1.0,
                    min=0.5,
                    max=2.0,
                    step=0.05,
                    optional=True,
                    tooltip="Speech speed multiplier.",
                ),
                IO.Boolean.Input(
                    "ssml",
                    default=False,
                    optional=True,
                    tooltip="Treat the text as SSML markup (for pauses, emphasis, and pronunciation control).",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    optional=True,
                    tooltip="Not sent to HeyGen; change it to force a re-run.",
                ),
            ],
            outputs=[IO.Audio.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.00095381,"format":{"approximate":true,"suffix":"/second"}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        text: str,
        voice: str,
        custom_voice_id: str = "",
        speed: float = 1.0,
        ssml: bool = False,
        seed: int = 0,
    ) -> IO.NodeOutput:
        validate_string(text, strip_whitespace=True, min_length=1, max_length=5000)
        payload = {
            "text": text,
            "voice_id": custom_voice_id.strip() or HEYGEN_VOICE_TTS_MAP[voice],
            "speed": round(speed, 2),
        }
        if ssml:
            payload["input_type"] = "ssml"
        response = await sync_op_raw(
            cls,
            ApiEndpoint(path=_SPEECH_PATH, method="POST"),
            data=payload,
        )
        audio_url = (response.get("data") or {}).get("audio_url")
        if not audio_url:
            raise ValueError(f"HeyGen did not return an audio_url: {response}")
        audio_bytes = await download_url_as_bytesio(audio_url)
        return IO.NodeOutput(audio_bytes_to_audio_input(audio_bytes.getvalue()))


class HeyGenExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            HeyGenTalkingPhotoNode,
            HeyGenAvatarVideoNode,
            HeyGenCreateAvatarNode,
            HeyGenVideoTranslateNode,
            HeyGenTextToSpeechNode,
        ]


async def comfy_entrypoint() -> HeyGenExtension:
    return HeyGenExtension()
