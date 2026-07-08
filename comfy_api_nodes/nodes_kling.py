"""Kling API Nodes

For source of truth on the allowed permutations of request fields, please reference:
- [Compatibility Table](https://app.klingai.com/global/dev/document-api/apiReference/model/skillsMap)
"""

import logging
import math
import re

import torch
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input, InputImpl
from comfy_api_nodes.apis import (
    KlingCameraControl,
    KlingCameraConfig,
    KlingCameraControlType,
    KlingVideoGenDuration,
    KlingVideoGenMode,
    KlingVideoGenAspectRatio,
    KlingVideoGenModelName,
    KlingText2VideoRequest,
    KlingText2VideoResponse,
    KlingImage2VideoRequest,
    KlingImage2VideoResponse,
    KlingVideoExtendRequest,
    KlingVideoExtendResponse,
    KlingLipSyncVoiceLanguage,
    KlingLipSyncInputObject,
    KlingLipSyncRequest,
    KlingLipSyncResponse,
    KlingVirtualTryOnModelName,
    KlingVirtualTryOnRequest,
    KlingVirtualTryOnResponse,
    KlingVideoResult,
    KlingImageResult,
    KlingImageGenerationsRequest,
    KlingImageGenerationsResponse,
    KlingImageGenImageReferenceType,
    KlingImageGenAspectRatio,
    KlingVideoEffectsRequest,
    KlingVideoEffectsResponse,
    KlingDualCharacterEffectsScene,
    KlingSingleImageEffectsScene,
    KlingDualCharacterEffectInput,
    KlingSingleImageEffectInput,
    KlingCharacterEffectModelName,
    KlingSingleImageEffectModelName,
)
from comfy_api_nodes.apis.kling import (
    ImageToVideoWithAudioRequest,
    KlingAvatarRequest,
    MotionControlRequest,
    MultiPromptEntry,
    OmniImageParamImage,
    OmniParamImage,
    OmniParamVideo,
    OmniProFirstLastFrameRequest,
    OmniProImageRequest,
    OmniProReferences2VideoRequest,
    OmniProText2VideoRequest,
    Kling3TurboSettings,
    Kling3TurboText2VideoRequest,
    Kling3TurboContent,
    Kling3TurboImage2VideoRequest,
    Kling3TurboCreateResponse,
    Kling3TurboQueryResponse,
    TaskStatusResponse,
    TextToVideoWithAudioRequest,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_image_tensor,
    download_url_to_video_output,
    get_number_of_images,
    poll_op,
    sync_op,
    tensor_to_base64_string,
    upload_audio_to_comfyapi,
    upload_image_to_comfyapi,
    upload_images_to_comfyapi,
    upload_video_to_comfyapi,
    validate_audio_duration,
    validate_image_aspect_ratio,
    validate_image_dimensions,
    validate_string,
    validate_video_dimensions,
    validate_video_duration,
)


def _generate_storyboard_inputs(count: int) -> list:
    inputs = []
    for i in range(1, count + 1):
        inputs.extend(
            [
                IO.String.Input(
                    f"storyboard_{i}_prompt",
                    multiline=True,
                    default="",
                    tooltip=f"Prompt for storyboard segment {i}. Max 512 characters.",
                ),
                IO.Int.Input(
                    f"storyboard_{i}_duration",
                    default=4,
                    min=1,
                    max=15,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip=f"Duration for storyboard segment {i} in seconds.",
                ),
            ]
        )
    return inputs


KLING_API_VERSION = "v1"
PATH_TEXT_TO_VIDEO = f"/proxy/kling/{KLING_API_VERSION}/videos/text2video"
PATH_IMAGE_TO_VIDEO = f"/proxy/kling/{KLING_API_VERSION}/videos/image2video"
PATH_VIDEO_EXTEND = f"/proxy/kling/{KLING_API_VERSION}/videos/video-extend"
PATH_LIP_SYNC = f"/proxy/kling/{KLING_API_VERSION}/videos/lip-sync"
PATH_VIDEO_EFFECTS = f"/proxy/kling/{KLING_API_VERSION}/videos/effects"
PATH_CHARACTER_IMAGE = f"/proxy/kling/{KLING_API_VERSION}/images/generations"
PATH_VIRTUAL_TRY_ON = f"/proxy/kling/{KLING_API_VERSION}/images/kolors-virtual-try-on"
PATH_IMAGE_GENERATIONS = f"/proxy/kling/{KLING_API_VERSION}/images/generations"

MAX_PROMPT_LENGTH_T2V = 2500
MAX_PROMPT_LENGTH_I2V = 500
MAX_PROMPT_LENGTH_IMAGE_GEN = 500
MAX_NEGATIVE_PROMPT_LENGTH_IMAGE_GEN = 200
MAX_PROMPT_LENGTH_LIP_SYNC = 120

AVERAGE_DURATION_T2V = 319
AVERAGE_DURATION_I2V = 164
AVERAGE_DURATION_LIP_SYNC = 455
AVERAGE_DURATION_VIRTUAL_TRY_ON = 19
AVERAGE_DURATION_IMAGE_GEN = 32
AVERAGE_DURATION_VIDEO_EFFECTS = 320
AVERAGE_DURATION_VIDEO_EXTEND = 320


MODE_TEXT2VIDEO = {
    "standard mode / 5s duration / kling-v1-6": ("std", "5", "kling-v1-6"),
    "standard mode / 10s duration / kling-v1-6": ("std", "10", "kling-v1-6"),
    "pro mode / 5s duration / kling-v2-master": ("pro", "5", "kling-v2-master"),
    "pro mode / 10s duration / kling-v2-master": ("pro", "10", "kling-v2-master"),
    "standard mode / 5s duration / kling-v2-master": ("std", "5", "kling-v2-master"),
    "standard mode / 10s duration / kling-v2-master": ("std", "10", "kling-v2-master"),
    "pro mode / 5s duration / kling-v2-1-master": ("pro", "5", "kling-v2-1-master"),
    "pro mode / 10s duration / kling-v2-1-master": ("pro", "10", "kling-v2-1-master"),
    "pro mode / 5s duration / kling-v2-5-turbo": ("pro", "5", "kling-v2-5-turbo"),
    "pro mode / 10s duration / kling-v2-5-turbo": ("pro", "10", "kling-v2-5-turbo"),
}
"""
Mapping of mode strings to their corresponding (mode, duration, model_name) tuples.
Only includes config combos that support the `image_tail` request field.

See: [Kling API Docs Capability Map](https://app.klingai.com/global/dev/document-api/apiReference/model/skillsMap)
"""


MODE_START_END_FRAME = {
    "pro mode / 5s duration / kling-v1-5": ("pro", "5", "kling-v1-5"),
    "pro mode / 10s duration / kling-v1-5": ("pro", "10", "kling-v1-5"),
    "pro mode / 5s duration / kling-v1-6": ("pro", "5", "kling-v1-6"),
    "pro mode / 10s duration / kling-v1-6": ("pro", "10", "kling-v1-6"),
    "pro mode / 5s duration / kling-v2-1": ("pro", "5", "kling-v2-1"),
    "pro mode / 10s duration / kling-v2-1": ("pro", "10", "kling-v2-1"),
    "pro mode / 5s duration / kling-v2-5-turbo": ("pro", "5", "kling-v2-5-turbo"),
    "pro mode / 10s duration / kling-v2-5-turbo": ("pro", "10", "kling-v2-5-turbo"),
}
"""
Returns a mapping of mode strings to their corresponding (mode, duration, model_name) tuples.
Only includes config combos that support the `image_tail` request field.

See: [Kling API Docs Capability Map](https://app.klingai.com/global/dev/document-api/apiReference/model/skillsMap)
"""


VOICES_CONFIG = {
    # English voices
    "Melody": ("girlfriend_4_speech02", "en"),
    "Sunny": ("genshin_vindi2", "en"),
    "Sage": ("zhinen_xuesheng", "en"),
    "Ace": ("AOT", "en"),
    "Blossom": ("ai_shatang", "en"),
    "Peppy": ("genshin_klee2", "en"),
    "Dove": ("genshin_kirara", "en"),
    "Shine": ("ai_kaiya", "en"),
    "Anchor": ("oversea_male1", "en"),
    "Lyric": ("ai_chenjiahao_712", "en"),
    "Tender": ("chat1_female_new-3", "en"),
    "Siren": ("chat_0407_5-1", "en"),
    "Zippy": ("cartoon-boy-07", "en"),
    "Bud": ("uk_boy1", "en"),
    "Sprite": ("cartoon-girl-01", "en"),
    "Candy": ("PeppaPig_platform", "en"),
    "Beacon": ("ai_huangzhong_712", "en"),
    "Rock": ("ai_huangyaoshi_712", "en"),
    "Titan": ("ai_laoguowang_712", "en"),
    "Grace": ("chengshu_jiejie", "en"),
    "Helen": ("you_pingjing", "en"),
    "Lore": ("calm_story1", "en"),
    "Crag": ("uk_man2", "en"),
    "Prattle": ("laopopo_speech02", "en"),
    "Hearth": ("heainainai_speech02", "en"),
    "The Reader": ("reader_en_m-v1", "en"),
    "Commercial Lady": ("commercial_lady_en_f-v1", "en"),
    # Chinese voices
    "阳光少年": ("genshin_vindi2", "zh"),
    "懂事小弟": ("zhinen_xuesheng", "zh"),
    "运动少年": ("tiyuxi_xuedi", "zh"),
    "青春少女": ("ai_shatang", "zh"),
    "温柔小妹": ("genshin_klee2", "zh"),
    "元气少女": ("genshin_kirara", "zh"),
    "阳光男生": ("ai_kaiya", "zh"),
    "幽默小哥": ("tiexin_nanyou", "zh"),
    "文艺小哥": ("ai_chenjiahao_712", "zh"),
    "甜美邻家": ("girlfriend_1_speech02", "zh"),
    "温柔姐姐": ("chat1_female_new-3", "zh"),
    "职场女青": ("girlfriend_2_speech02", "zh"),
    "活泼男童": ("cartoon-boy-07", "zh"),
    "俏皮女童": ("cartoon-girl-01", "zh"),
    "稳重老爸": ("ai_huangyaoshi_712", "zh"),
    "温柔妈妈": ("you_pingjing", "zh"),
    "严肃上司": ("ai_laoguowang_712", "zh"),
    "优雅贵妇": ("chengshu_jiejie", "zh"),
    "慈祥爷爷": ("zhuxi_speech02", "zh"),
    "唠叨爷爷": ("uk_oldman3", "zh"),
    "唠叨奶奶": ("laopopo_speech02", "zh"),
    "和蔼奶奶": ("heainainai_speech02", "zh"),
    "东北老铁": ("dongbeilaotie_speech02", "zh"),
    "重庆小伙": ("chongqingxiaohuo_speech02", "zh"),
    "四川妹子": ("chuanmeizi_speech02", "zh"),
    "潮汕大叔": ("chaoshandashu_speech02", "zh"),
    "台湾男生": ("ai_taiwan_man2_speech02", "zh"),
    "西安掌柜": ("xianzhanggui_speech02", "zh"),
    "天津姐姐": ("tianjinjiejie_speech02", "zh"),
    "新闻播报男": ("diyinnansang_DB_CN_M_04-v2", "zh"),
    "译制片男": ("yizhipiannan-v1", "zh"),
    "撒娇女友": ("tianmeixuemei-v1", "zh"),
    "刀片烟嗓": ("daopianyansang-v1", "zh"),
    "乖巧正太": ("mengwa-v1", "zh"),
}


def normalize_omni_prompt_references(prompt: str) -> str:
    """
    Rewrites Kling Omni-style placeholders used in the app, like:

        @image, @image1, @image2, ... @imageN
        @video, @video1, @video2, ... @videoN

    into the API-compatible form:

        <<<image_1>>>, <<<image_2>>>, ...
        <<<video_1>>>, <<<video_2>>>, ...

    This is a UX shim for ComfyUI so users can type the same syntax as in the Kling app.
    """
    if not prompt:
        return prompt

    def _image_repl(match):
        return f"<<<image_{match.group('idx') or '1'}>>>"

    def _video_repl(match):
        return f"<<<video_{match.group('idx') or '1'}>>>"

    # (?<!\w) avoids matching e.g. "test@image.com"
    # (?!\w) makes sure we only match @image / @image<digits> and not @imageFoo
    prompt = re.sub(r"(?<!\w)@image(?P<idx>\d*)(?!\w)", _image_repl, prompt)
    return re.sub(r"(?<!\w)@video(?P<idx>\d*)(?!\w)", _video_repl, prompt)


async def finish_omni_video_task(cls: type[IO.ComfyNode], response: TaskStatusResponse) -> IO.NodeOutput:
    if response.code:
        raise RuntimeError(
            f"Kling request failed. Code: {response.code}, Message: {response.message}, Data: {response.data}"
        )
    final_response = await poll_op(
        cls,
        ApiEndpoint(path=f"/proxy/kling/v1/videos/omni-video/{response.data.task_id}"),
        response_model=TaskStatusResponse,
        status_extractor=lambda r: (r.data.task_status if r.data else None),
    )
    return IO.NodeOutput(await download_url_to_video_output(final_response.data.task_result.videos[0].url))


def is_valid_camera_control_configs(configs: list[float]) -> bool:
    """Verifies that at least one camera control configuration is non-zero."""
    return any(not math.isclose(value, 0.0) for value in configs)


def is_valid_task_creation_response(response: KlingText2VideoResponse) -> bool:
    """Verifies that the initial response contains a task ID."""
    return bool(response.data.task_id)


def is_valid_video_response(response: KlingText2VideoResponse) -> bool:
    """Verifies that the response contains a task result with at least one video."""
    return (
        response.data is not None
        and response.data.task_result is not None
        and response.data.task_result.videos is not None
        and len(response.data.task_result.videos) > 0
    )


def is_valid_image_response(response: KlingVirtualTryOnResponse) -> bool:
    """Verifies that the response contains a task result with at least one image."""
    return (
        response.data is not None
        and response.data.task_result is not None
        and response.data.task_result.images is not None
        and len(response.data.task_result.images) > 0
    )


def validate_prompts(prompt: str, negative_prompt: str, max_length: int) -> bool:
    """Verifies that the positive prompt is not empty and that neither promt is too long."""
    if not prompt:
        raise ValueError("Positive prompt is empty")
    if len(prompt) > max_length:
        raise ValueError(f"Positive prompt is too long: {len(prompt)} characters")
    if negative_prompt and len(negative_prompt) > max_length:
        raise ValueError(
            f"Negative prompt is too long: {len(negative_prompt)} characters"
        )
    return True


def validate_task_creation_response(response) -> None:
    """Validates that the Kling task creation request was successful."""
    if not is_valid_task_creation_response(response):
        error_msg = f"Kling initial request failed. Code: {response.code}, Message: {response.message}, Data: {response.data}"
        logging.error(error_msg)
        raise Exception(error_msg)


def validate_video_result_response(response) -> None:
    """Validates that the Kling task result contains a video."""
    if not is_valid_video_response(response):
        error_msg = f"Kling task {response.data.task_id} succeeded but no video data found in response."
        logging.error("Error: %s.\nResponse: %s", error_msg, response)
        raise Exception(error_msg)


def validate_image_result_response(response) -> None:
    """Validates that the Kling task result contains an image."""
    if not is_valid_image_response(response):
        error_msg = f"Kling task {response.data.task_id} succeeded but no image data found in response."
        logging.error("Error: %s.\nResponse: %s", error_msg, response)
        raise Exception(error_msg)


def validate_input_image(image: torch.Tensor) -> None:
    """
    Validates the input image adheres to the expectations of the Kling API:
    - The image resolution should not be less than 300*300px
    - The aspect ratio of the image should be between 1:2.5 ~ 2.5:1

    See: https://app.klingai.com/global/dev/document-api/apiReference/model/imageToVideo
    """
    validate_image_dimensions(image, min_width=300, min_height=300)
    validate_image_aspect_ratio(image, (1, 2.5), (2.5, 1))


def get_video_from_response(response) -> KlingVideoResult:
    """Returns the first video object from the Kling video generation task result.
    Will raise an error if the response is not valid.
    """
    video = response.data.task_result.videos[0]
    logging.info(
        "Kling task %s succeeded. Video URL: %s", response.data.task_id, video.url
    )
    return video


def get_video_url_from_response(response) -> str | None:
    """Returns the first video url from the Kling video generation task result.
    Will not raise an error if the response is not valid.
    """
    if response and is_valid_video_response(response):
        return str(get_video_from_response(response).url)
    else:
        return None


def get_images_from_response(response) -> list[KlingImageResult]:
    """Returns the list of image objects from the Kling image generation task result.
    Will raise an error if the response is not valid.
    """
    images = response.data.task_result.images
    logging.info("Kling task %s succeeded. Images: %s", response.data.task_id, images)
    return images


def get_images_urls_from_response(response) -> str | None:
    """Returns the list of image urls from the Kling image generation task result.
    Will not raise an error if the response is not valid. If there is only one image, returns the url as a string. If there are multiple images, returns a list of urls.
    """
    if response and is_valid_image_response(response):
        images = get_images_from_response(response)
        image_urls = [str(image.url) for image in images]
        return "\n".join(image_urls)
    else:
        return None


async def image_result_to_node_output(
    images: list[KlingImageResult],
) -> torch.Tensor:
    """
    Converts a KlingImageResult to a tuple containing a [B, H, W, C] tensor.
    If multiple images are returned, they will be stacked along the batch dimension.
    """
    if len(images) == 1:
        return await download_url_to_image_tensor(str(images[0].url))
    else:
        return torch.cat([await download_url_to_image_tensor(str(image.url)) for image in images])


async def execute_text2video(
    cls: type[IO.ComfyNode],
    prompt: str,
    negative_prompt: str,
    cfg_scale: float,
    model_name: str,
    model_mode: str,
    duration: str,
    aspect_ratio: str,
    camera_control: KlingCameraControl | None = None,
) -> IO.NodeOutput:
    validate_prompts(prompt, negative_prompt, MAX_PROMPT_LENGTH_T2V)
    task_creation_response = await sync_op(
        cls,
        ApiEndpoint(path=PATH_TEXT_TO_VIDEO, method="POST"),
        response_model=KlingText2VideoResponse,
        data=KlingText2VideoRequest(
            prompt=prompt if prompt else None,
            negative_prompt=negative_prompt if negative_prompt else None,
            duration=KlingVideoGenDuration(duration),
            mode=KlingVideoGenMode(model_mode),
            model_name=model_name,
            cfg_scale=cfg_scale,
            aspect_ratio=KlingVideoGenAspectRatio(aspect_ratio),
            camera_control=camera_control,
        ),
    )

    validate_task_creation_response(task_creation_response)

    task_id = task_creation_response.data.task_id
    final_response = await poll_op(
        cls,
        ApiEndpoint(path=f"{PATH_TEXT_TO_VIDEO}/{task_id}"),
        response_model=KlingText2VideoResponse,
        estimated_duration=AVERAGE_DURATION_T2V,
        status_extractor=lambda r: (r.data.task_status.value if r.data and r.data.task_status else None),
    )
    validate_video_result_response(final_response)

    video = get_video_from_response(final_response)
    return IO.NodeOutput(await download_url_to_video_output(str(video.url)), str(video.id), str(video.duration))


async def execute_image2video(
    cls: type[IO.ComfyNode],
    start_frame: torch.Tensor,
    prompt: str,
    negative_prompt: str,
    model_name: str,
    cfg_scale: float,
    model_mode: str,
    aspect_ratio: str,
    duration: str,
    camera_control: KlingCameraControl | None = None,
    end_frame: torch.Tensor | None = None,
) -> IO.NodeOutput:
    validate_prompts(prompt, negative_prompt, MAX_PROMPT_LENGTH_I2V)
    validate_input_image(start_frame)

    if camera_control is not None:
        # Camera control type for image 2 video is always `simple`
        camera_control.type = KlingCameraControlType.simple

    if model_mode == "std" and model_name == KlingVideoGenModelName.kling_v2_5_turbo.value:
        model_mode = "pro"  # October 5: currently "std" mode is not supported for this model

    task_creation_response = await sync_op(
        cls,
        ApiEndpoint(path=PATH_IMAGE_TO_VIDEO, method="POST"),
        response_model=KlingImage2VideoResponse,
        data=KlingImage2VideoRequest(
            model_name=KlingVideoGenModelName(model_name),
            image=tensor_to_base64_string(start_frame),
            image_tail=(
                tensor_to_base64_string(end_frame)
                if end_frame is not None
                else None
            ),
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            cfg_scale=cfg_scale,
            mode=KlingVideoGenMode(model_mode),
            duration=KlingVideoGenDuration(duration),
            camera_control=camera_control,
        ),
    )

    validate_task_creation_response(task_creation_response)
    task_id = task_creation_response.data.task_id

    final_response = await poll_op(
        cls,
        ApiEndpoint(path=f"{PATH_IMAGE_TO_VIDEO}/{task_id}"),
        response_model=KlingImage2VideoResponse,
        estimated_duration=AVERAGE_DURATION_I2V,
        status_extractor=lambda r: (r.data.task_status.value if r.data and r.data.task_status else None),
    )
    validate_video_result_response(final_response)

    video = get_video_from_response(final_response)
    return IO.NodeOutput(await download_url_to_video_output(str(video.url)), str(video.id), str(video.duration))


async def execute_video_effect(
    cls: type[IO.ComfyNode],
    dual_character: bool,
    effect_scene: KlingDualCharacterEffectsScene | KlingSingleImageEffectsScene,
    model_name: str,
    duration: KlingVideoGenDuration,
    image_1: torch.Tensor,
    image_2: torch.Tensor | None = None,
    model_mode: KlingVideoGenMode | None = None,
) -> tuple[InputImpl.VideoFromFile, str, str]:
    if dual_character:
        request_input_field = KlingDualCharacterEffectInput(
            model_name=model_name,
            mode=model_mode,
            images=[
                tensor_to_base64_string(image_1),
                tensor_to_base64_string(image_2),
            ],
            duration=duration,
        )
    else:
        request_input_field = KlingSingleImageEffectInput(
            model_name=model_name,
            image=tensor_to_base64_string(image_1),
            duration=duration,
        )

    task_creation_response = await sync_op(
        cls,
        endpoint=ApiEndpoint(path=PATH_VIDEO_EFFECTS, method="POST"),
        response_model=KlingVideoEffectsResponse,
        data=KlingVideoEffectsRequest(
            effect_scene=effect_scene,
            input=request_input_field,
        ),
    )

    validate_task_creation_response(task_creation_response)
    task_id = task_creation_response.data.task_id

    final_response = await poll_op(
        cls,
        ApiEndpoint(path=f"{PATH_VIDEO_EFFECTS}/{task_id}"),
        response_model=KlingVideoEffectsResponse,
        estimated_duration=AVERAGE_DURATION_VIDEO_EFFECTS,
        status_extractor=lambda r: (r.data.task_status.value if r.data and r.data.task_status else None),
    )
    validate_video_result_response(final_response)

    video = get_video_from_response(final_response)
    return await download_url_to_video_output(str(video.url)), str(video.id), str(video.duration)


async def execute_lipsync(
    cls: type[IO.ComfyNode],
    video: Input.Video,
    audio: Input.Audio | None = None,
    voice_language: str | None = None,
    model_mode: str | None = None,
    text: str | None = None,
    voice_speed: float | None = None,
    voice_id: str | None = None,
) -> IO.NodeOutput:
    if text:
        validate_string(text, field_name="Text", max_length=MAX_PROMPT_LENGTH_LIP_SYNC)
    validate_video_dimensions(video, 720, 1920)
    validate_video_duration(video, 2, 10)

    # Upload video to Comfy API and get download URL
    video_url = await upload_video_to_comfyapi(cls, video)
    logging.info("Uploaded video to Comfy API. URL: %s", video_url)

    # Upload the audio file to Comfy API and get download URL
    if audio:
        audio_url = await upload_audio_to_comfyapi(
            cls, audio, container_format="mp3", codec_name="libmp3lame", mime_type="audio/mpeg"
        )
        logging.info("Uploaded audio to Comfy API. URL: %s", audio_url)
    else:
        audio_url = None

    task_creation_response = await sync_op(
        cls,
        ApiEndpoint(PATH_LIP_SYNC, "POST"),
        response_model=KlingLipSyncResponse,
        data=KlingLipSyncRequest(
            input=KlingLipSyncInputObject(
                video_url=video_url,
                mode=model_mode,
                text=text,
                voice_language=voice_language,
                voice_speed=voice_speed,
                audio_type="url",
                audio_url=audio_url,
                voice_id=voice_id,
            ),
        ),
    )

    validate_task_creation_response(task_creation_response)
    task_id = task_creation_response.data.task_id

    final_response = await poll_op(
        cls,
        ApiEndpoint(path=f"{PATH_LIP_SYNC}/{task_id}"),
        response_model=KlingLipSyncResponse,
        estimated_duration=AVERAGE_DURATION_LIP_SYNC,
        status_extractor=lambda r: (r.data.task_status.value if r.data and r.data.task_status else None),
    )
    validate_video_result_response(final_response)

    video = get_video_from_response(final_response)
    return IO.NodeOutput(await download_url_to_video_output(str(video.url)), str(video.id), str(video.duration))


class KlingCameraControls(IO.ComfyNode):
    """Kling Camera Controls Node"""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingCameraControls",
            display_name="Kling Camera Controls",
            category="partner/video/Kling",
            description="Allows specifying configuration options for Kling Camera Controls and motion control effects.",
            inputs=[
                IO.Combo.Input("camera_control_type", options=KlingCameraControlType),
                IO.Float.Input(
                    "horizontal_movement",
                    default=0.0,
                    min=-10.0,
                    max=10.0,
                    step=0.25,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Controls camera's movement along horizontal axis (x-axis). Negative indicates left, positive indicates right",
                ),
                IO.Float.Input(
                    "vertical_movement",
                    default=0.0,
                    min=-10.0,
                    max=10.0,
                    step=0.25,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Controls camera's movement along vertical axis (y-axis). Negative indicates downward, positive indicates upward.",
                ),
                IO.Float.Input(
                    "pan",
                    default=0.5,
                    min=-10.0,
                    max=10.0,
                    step=0.25,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Controls camera's rotation in vertical plane (x-axis). Negative indicates downward rotation, positive indicates upward rotation.",
                ),
                IO.Float.Input(
                    "tilt",
                    default=0.0,
                    min=-10.0,
                    max=10.0,
                    step=0.25,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Controls camera's rotation in horizontal plane (y-axis). Negative indicates left rotation, positive indicates right rotation.",
                ),
                IO.Float.Input(
                    "roll",
                    default=0.0,
                    min=-10.0,
                    max=10.0,
                    step=0.25,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Controls camera's rolling amount (z-axis). Negative indicates counterclockwise, positive indicates clockwise.",
                ),
                IO.Float.Input(
                    "zoom",
                    default=0.0,
                    min=-10.0,
                    max=10.0,
                    step=0.25,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Controls change in camera's focal length. Negative indicates narrower field of view, positive indicates wider field of view.",
                ),
            ],
            outputs=[IO.Custom("CAMERA_CONTROL").Output(display_name="camera_control")],
        )

    @classmethod
    def validate_inputs(
        cls,
        horizontal_movement: float,
        vertical_movement: float,
        pan: float,
        tilt: float,
        roll: float,
        zoom: float,
    ) -> bool | str:
        if not is_valid_camera_control_configs(
            [
                horizontal_movement,
                vertical_movement,
                pan,
                tilt,
                roll,
                zoom,
            ]
        ):
            return "Invalid camera control configs: at least one of the values must be non-zero"
        return True

    @classmethod
    def execute(
        cls,
        camera_control_type: str,
        horizontal_movement: float,
        vertical_movement: float,
        pan: float,
        tilt: float,
        roll: float,
        zoom: float,
    ) -> IO.NodeOutput:
        return IO.NodeOutput(
            KlingCameraControl(
                type=KlingCameraControlType(camera_control_type),
                config=KlingCameraConfig(
                    horizontal=horizontal_movement,
                    vertical=vertical_movement,
                    pan=pan,
                    roll=roll,
                    tilt=tilt,
                    zoom=zoom,
                ),
            )
        )


class KlingTextToVideoNode(IO.ComfyNode):
    """Kling Text to Video Node"""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        modes = list(MODE_TEXT2VIDEO.keys())
        return IO.Schema(
            node_id="KlingTextToVideoNode",
            display_name="Kling Text to Video",
            category="partner/video/Kling",
            description="Kling Text to Video Node",
            inputs=[
                IO.String.Input("prompt", multiline=True, tooltip="Positive text prompt"),
                IO.String.Input("negative_prompt", multiline=True, tooltip="Negative text prompt"),
                IO.Float.Input("cfg_scale", default=1.0, min=0.0, max=1.0),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=KlingVideoGenAspectRatio,
                    default="16:9",
                ),
                IO.Combo.Input(
                    "mode",
                    options=modes,
                    default=modes[8],
                    tooltip="The configuration to use for the video generation following the format: mode / duration / model_name.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="video_id"),
                IO.String.Output(display_name="duration"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["mode"]),
                expr="""
                (
                  $m := widgets.mode;
                  $contains($m,"v2-5-turbo")
                    ? ($contains($m,"10") ? {"type":"usd","usd":0.7} : {"type":"usd","usd":0.35})
                    : $contains($m,"v2-1-master")
                      ? ($contains($m,"10s") ? {"type":"usd","usd":2.8} : {"type":"usd","usd":1.4})
                      : $contains($m,"v2-master")
                        ? ($contains($m,"10s") ? {"type":"usd","usd":2.8} : {"type":"usd","usd":1.4})
                        : $contains($m,"v1-6")
                          ? (
                              $contains($m,"pro")
                                ? ($contains($m,"10s") ? {"type":"usd","usd":0.98} : {"type":"usd","usd":0.49})
                                : ($contains($m,"10s") ? {"type":"usd","usd":0.56} : {"type":"usd","usd":0.28})
                            )
                          : $contains($m,"v1")
                            ? (
                                $contains($m,"pro")
                                  ? ($contains($m,"10s") ? {"type":"usd","usd":0.98} : {"type":"usd","usd":0.49})
                                  : ($contains($m,"10s") ? {"type":"usd","usd":0.28} : {"type":"usd","usd":0.14})
                              )
                            : {"type":"usd","usd":0.14}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        mode: str,
        aspect_ratio: str,
    ) -> IO.NodeOutput:
        model_mode, duration, model_name = MODE_TEXT2VIDEO[mode]
        return await execute_text2video(
            cls,
            prompt=prompt,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale,
            model_mode=model_mode,
            aspect_ratio=aspect_ratio,
            model_name=model_name,
            duration=duration,
        )


class OmniProTextToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingOmniProTextToVideoNode",
            display_name="Kling 3.0 Omni Text to Video",
            category="partner/video/Kling",
            description="Use text prompts to generate videos with the latest Kling model.",
            inputs=[
                IO.Combo.Input("model_name", options=["kling-v3-omni", "kling-video-o1"]),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A text prompt describing the video content. "
                    "This can include both positive and negative descriptions. "
                    "Ignored when storyboards are enabled.",
                ),
                IO.Combo.Input("aspect_ratio", options=["16:9", "9:16", "1:1"]),
                IO.Int.Input("duration", default=5, min=3, max=15, display_mode=IO.NumberDisplay.slider),
                IO.Combo.Input("resolution", options=["4k", "1080p", "720p"], default="1080p", optional=True),
                IO.DynamicCombo.Input(
                    "storyboards",
                    options=[
                        IO.DynamicCombo.Option("disabled", []),
                        IO.DynamicCombo.Option("1 storyboard", _generate_storyboard_inputs(1)),
                        IO.DynamicCombo.Option("2 storyboards", _generate_storyboard_inputs(2)),
                        IO.DynamicCombo.Option("3 storyboards", _generate_storyboard_inputs(3)),
                        IO.DynamicCombo.Option("4 storyboards", _generate_storyboard_inputs(4)),
                        IO.DynamicCombo.Option("5 storyboards", _generate_storyboard_inputs(5)),
                        IO.DynamicCombo.Option("6 storyboards", _generate_storyboard_inputs(6)),
                    ],
                    tooltip="Generate a series of video segments with individual prompts and durations. "
                    "Ignored for o1 model.",
                    optional=True,
                ),
                IO.Boolean.Input("generate_audio", default=False, optional=True),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["duration", "resolution", "model_name", "generate_audio"]),
                expr="""
                (
                  $res := widgets.resolution;
                  $mode := $res = "4k" ? "4k" : ($res = "720p" ? "std" : "pro");
                  $isV3 := $contains(widgets.model_name, "v3");
                  $audio := $isV3 and widgets.generate_audio;
                  $rates := $audio
                    ? {"std": 0.112, "pro": 0.14, "4k": 0.42}
                    : {"std": 0.084, "pro": 0.112, "4k": 0.42};
                  {"type":"usd","usd": $lookup($rates, $mode) * widgets.duration}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model_name: str,
        prompt: str,
        aspect_ratio: str,
        duration: int,
        resolution: str = "1080p",
        storyboards: dict | None = None,
        generate_audio: bool = False,
        seed: int = 0,
    ) -> IO.NodeOutput:
        _ = seed
        if model_name == "kling-video-o1":
            if duration not in (5, 10):
                raise ValueError("kling-video-o1 only supports durations of 5 or 10 seconds.")
            if generate_audio:
                raise ValueError("kling-video-o1 does not support audio generation.")
            if resolution == "4k":
                raise ValueError("kling-video-o1 does not support 4k resolution.")
        stories_enabled = storyboards is not None and storyboards["storyboards"] != "disabled"
        if stories_enabled and model_name == "kling-video-o1":
            raise ValueError("kling-video-o1 does not support storyboards.")
        validate_string(prompt, strip_whitespace=True, min_length=0 if stories_enabled else 1, max_length=2500)

        multi_shot = None
        multi_prompt_list = None
        if stories_enabled:
            count = int(storyboards["storyboards"].split()[0])
            multi_shot = True
            multi_prompt_list = []
            for i in range(1, count + 1):
                sb_prompt = storyboards[f"storyboard_{i}_prompt"]
                sb_duration = storyboards[f"storyboard_{i}_duration"]
                validate_string(sb_prompt, field_name=f"storyboard_{i}_prompt", min_length=1, max_length=512)
                multi_prompt_list.append(
                    MultiPromptEntry(
                        index=i,
                        prompt=sb_prompt,
                        duration=str(sb_duration),
                    )
                )
            total_storyboard_duration = sum(int(e.duration) for e in multi_prompt_list)
            if total_storyboard_duration != duration:
                raise ValueError(
                    f"Total storyboard duration ({total_storyboard_duration}s) "
                    f"must equal the global duration ({duration}s)."
                )

        if resolution == "4k":
            mode = "4k"
        elif resolution == "1080p":
            mode = "pro"
        else:
            mode = "std"
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/kling/v1/videos/omni-video", method="POST"),
            response_model=TaskStatusResponse,
            data=OmniProText2VideoRequest(
                model_name=model_name,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                duration=str(duration),
                mode=mode,
                multi_shot=multi_shot,
                multi_prompt=multi_prompt_list,
                shot_type="customize" if multi_shot else None,
                sound="on" if generate_audio else "off",
            ),
        )
        return await finish_omni_video_task(cls, response)


class OmniProFirstLastFrameNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingOmniProFirstLastFrameNode",
            display_name="Kling 3.0 Omni First-Last-Frame to Video",
            category="partner/video/Kling",
            description="Use a start frame, an optional end frame, or reference images with the latest Kling model.",
            inputs=[
                IO.Combo.Input("model_name", options=["kling-v3-omni", "kling-video-o1"]),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A text prompt describing the video content. "
                    "This can include both positive and negative descriptions. "
                    "Ignored when storyboards are enabled.",
                ),
                IO.Int.Input("duration", default=5, min=3, max=15, display_mode=IO.NumberDisplay.slider),
                IO.Image.Input("first_frame"),
                IO.Image.Input(
                    "end_frame",
                    optional=True,
                    tooltip="An optional end frame for the video. "
                    "This cannot be used simultaneously with 'reference_images'. "
                    "Does not work with storyboards.",
                ),
                IO.Image.Input(
                    "reference_images",
                    optional=True,
                    tooltip="Up to 6 additional reference images.",
                ),
                IO.Combo.Input("resolution", options=["4k", "1080p", "720p"], default="1080p", optional=True),
                IO.DynamicCombo.Input(
                    "storyboards",
                    options=[
                        IO.DynamicCombo.Option("disabled", []),
                        IO.DynamicCombo.Option("1 storyboard", _generate_storyboard_inputs(1)),
                        IO.DynamicCombo.Option("2 storyboards", _generate_storyboard_inputs(2)),
                        IO.DynamicCombo.Option("3 storyboards", _generate_storyboard_inputs(3)),
                        IO.DynamicCombo.Option("4 storyboards", _generate_storyboard_inputs(4)),
                        IO.DynamicCombo.Option("5 storyboards", _generate_storyboard_inputs(5)),
                        IO.DynamicCombo.Option("6 storyboards", _generate_storyboard_inputs(6)),
                    ],
                    tooltip="Generate a series of video segments with individual prompts and durations. "
                    "Only supported for kling-v3-omni.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=False,
                    optional=True,
                    tooltip="Generate audio for the video. Only supported for kling-v3-omni.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["duration", "resolution", "model_name", "generate_audio"]),
                expr="""
                (
                  $res := widgets.resolution;
                  $mode := $res = "4k" ? "4k" : ($res = "720p" ? "std" : "pro");
                  $isV3 := $contains(widgets.model_name, "v3");
                  $audio := $isV3 and widgets.generate_audio;
                  $rates := $audio
                    ? {"std": 0.112, "pro": 0.14, "4k": 0.42}
                    : {"std": 0.084, "pro": 0.112, "4k": 0.42};
                  {"type":"usd","usd": $lookup($rates, $mode) * widgets.duration}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model_name: str,
        prompt: str,
        duration: int,
        first_frame: Input.Image,
        end_frame: Input.Image | None = None,
        reference_images: Input.Image | None = None,
        resolution: str = "1080p",
        storyboards: dict | None = None,
        generate_audio: bool = False,
        seed: int = 0,
    ) -> IO.NodeOutput:
        _ = seed
        if model_name == "kling-video-o1":
            if duration > 10:
                raise ValueError("kling-video-o1 does not support durations greater than 10 seconds.")
            if generate_audio:
                raise ValueError("kling-video-o1 does not support audio generation.")
            if resolution == "4k":
                raise ValueError("kling-video-o1 does not support 4k resolution.")
        stories_enabled = storyboards is not None and storyboards["storyboards"] != "disabled"
        if stories_enabled and model_name == "kling-video-o1":
            raise ValueError("kling-video-o1 does not support storyboards.")
        prompt = normalize_omni_prompt_references(prompt)
        validate_string(prompt, strip_whitespace=True, min_length=0 if stories_enabled else 1, max_length=2500)
        if end_frame is not None and reference_images is not None:
            raise ValueError("The 'end_frame' input cannot be used simultaneously with 'reference_images'.")
        if end_frame is not None and stories_enabled:
            raise ValueError("The 'end_frame' input cannot be used simultaneously with storyboards.")
        if (
            model_name == "kling-video-o1"
            and duration not in (5, 10)
            and end_frame is None
            and reference_images is None
        ):
            raise ValueError(
                "Duration is only supported for 5 or 10 seconds if there is no end frame or reference images."
            )

        multi_shot = None
        multi_prompt_list = None
        if stories_enabled:
            count = int(storyboards["storyboards"].split()[0])
            multi_shot = True
            multi_prompt_list = []
            for i in range(1, count + 1):
                sb_prompt = storyboards[f"storyboard_{i}_prompt"]
                sb_duration = storyboards[f"storyboard_{i}_duration"]
                validate_string(sb_prompt, field_name=f"storyboard_{i}_prompt", min_length=1, max_length=512)
                multi_prompt_list.append(
                    MultiPromptEntry(
                        index=i,
                        prompt=sb_prompt,
                        duration=str(sb_duration),
                    )
                )
            total_storyboard_duration = sum(int(e.duration) for e in multi_prompt_list)
            if total_storyboard_duration != duration:
                raise ValueError(
                    f"Total storyboard duration ({total_storyboard_duration}s) "
                    f"must equal the global duration ({duration}s)."
                )

        validate_image_dimensions(first_frame, min_width=300, min_height=300)
        validate_image_aspect_ratio(first_frame, (1, 2.5), (2.5, 1))
        image_list: list[OmniParamImage] = [
            OmniParamImage(
                image_url=(await upload_images_to_comfyapi(cls, first_frame, wait_label="Uploading first frame"))[0],
                type="first_frame",
            )
        ]
        if end_frame is not None:
            validate_image_dimensions(end_frame, min_width=300, min_height=300)
            validate_image_aspect_ratio(end_frame, (1, 2.5), (2.5, 1))
            image_list.append(
                OmniParamImage(
                    image_url=(await upload_images_to_comfyapi(cls, end_frame, wait_label="Uploading end frame"))[0],
                    type="end_frame",
                )
            )
        if reference_images is not None:
            if get_number_of_images(reference_images) > 6:
                raise ValueError("The maximum number of reference images allowed is 6.")
            for i in reference_images:
                validate_image_dimensions(i, min_width=300, min_height=300)
                validate_image_aspect_ratio(i, (1, 2.5), (2.5, 1))
            for i in await upload_images_to_comfyapi(cls, reference_images, wait_label="Uploading reference frame(s)"):
                image_list.append(OmniParamImage(image_url=i))
        if resolution == "4k":
            mode = "4k"
        elif resolution == "1080p":
            mode = "pro"
        else:
            mode = "std"
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/kling/v1/videos/omni-video", method="POST"),
            response_model=TaskStatusResponse,
            data=OmniProFirstLastFrameRequest(
                model_name=model_name,
                prompt=prompt,
                duration=str(duration),
                image_list=image_list,
                mode=mode,
                sound="on" if generate_audio else "off",
                multi_shot=multi_shot,
                multi_prompt=multi_prompt_list,
                shot_type="customize" if multi_shot else None,
            ),
        )
        return await finish_omni_video_task(cls, response)


class OmniProImageToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingOmniProImageToVideoNode",
            display_name="Kling 3.0 Omni Image to Video",
            category="partner/video/Kling",
            description="Use up to 7 reference images to generate a video with the latest Kling model.",
            inputs=[
                IO.Combo.Input("model_name", options=["kling-v3-omni", "kling-video-o1"]),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A text prompt describing the video content. "
                    "This can include both positive and negative descriptions. "
                    "Ignored when storyboards are enabled.",
                ),
                IO.Combo.Input("aspect_ratio", options=["16:9", "9:16", "1:1"]),
                IO.Int.Input("duration", default=5, min=3, max=15, display_mode=IO.NumberDisplay.slider),
                IO.Image.Input(
                    "reference_images",
                    tooltip="Up to 7 reference images.",
                ),
                IO.Combo.Input("resolution", options=["4k", "1080p", "720p"], default="1080p", optional=True),
                IO.DynamicCombo.Input(
                    "storyboards",
                    options=[
                        IO.DynamicCombo.Option("disabled", []),
                        IO.DynamicCombo.Option("1 storyboard", _generate_storyboard_inputs(1)),
                        IO.DynamicCombo.Option("2 storyboards", _generate_storyboard_inputs(2)),
                        IO.DynamicCombo.Option("3 storyboards", _generate_storyboard_inputs(3)),
                        IO.DynamicCombo.Option("4 storyboards", _generate_storyboard_inputs(4)),
                        IO.DynamicCombo.Option("5 storyboards", _generate_storyboard_inputs(5)),
                        IO.DynamicCombo.Option("6 storyboards", _generate_storyboard_inputs(6)),
                    ],
                    tooltip="Generate a series of video segments with individual prompts and durations. "
                    "Only supported for kling-v3-omni.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=False,
                    optional=True,
                    tooltip="Generate audio for the video. Only supported for kling-v3-omni.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["duration", "resolution", "model_name", "generate_audio"]),
                expr="""
                (
                  $res := widgets.resolution;
                  $mode := $res = "4k" ? "4k" : ($res = "720p" ? "std" : "pro");
                  $isV3 := $contains(widgets.model_name, "v3");
                  $audio := $isV3 and widgets.generate_audio;
                  $rates := $audio
                    ? {"std": 0.112, "pro": 0.14, "4k": 0.42}
                    : {"std": 0.084, "pro": 0.112, "4k": 0.42};
                  {"type":"usd","usd": $lookup($rates, $mode) * widgets.duration}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model_name: str,
        prompt: str,
        aspect_ratio: str,
        duration: int,
        reference_images: Input.Image,
        resolution: str = "1080p",
        storyboards: dict | None = None,
        generate_audio: bool = False,
        seed: int = 0,
    ) -> IO.NodeOutput:
        _ = seed
        if model_name == "kling-video-o1":
            if duration > 10:
                raise ValueError("kling-video-o1 does not support durations greater than 10 seconds.")
            if generate_audio:
                raise ValueError("kling-video-o1 does not support audio generation.")
            if resolution == "4k":
                raise ValueError("kling-video-o1 does not support 4k resolution.")
        stories_enabled = storyboards is not None and storyboards["storyboards"] != "disabled"
        if stories_enabled and model_name == "kling-video-o1":
            raise ValueError("kling-video-o1 does not support storyboards.")
        prompt = normalize_omni_prompt_references(prompt)
        validate_string(prompt, strip_whitespace=True, min_length=0 if stories_enabled else 1, max_length=2500)

        multi_shot = None
        multi_prompt_list = None
        if stories_enabled:
            count = int(storyboards["storyboards"].split()[0])
            multi_shot = True
            multi_prompt_list = []
            for i in range(1, count + 1):
                sb_prompt = storyboards[f"storyboard_{i}_prompt"]
                sb_duration = storyboards[f"storyboard_{i}_duration"]
                validate_string(sb_prompt, field_name=f"storyboard_{i}_prompt", min_length=1, max_length=512)
                multi_prompt_list.append(
                    MultiPromptEntry(
                        index=i,
                        prompt=sb_prompt,
                        duration=str(sb_duration),
                    )
                )
            total_storyboard_duration = sum(int(e.duration) for e in multi_prompt_list)
            if total_storyboard_duration != duration:
                raise ValueError(
                    f"Total storyboard duration ({total_storyboard_duration}s) "
                    f"must equal the global duration ({duration}s)."
                )

        if get_number_of_images(reference_images) > 7:
            raise ValueError("The maximum number of reference images is 7.")
        for i in reference_images:
            validate_image_dimensions(i, min_width=300, min_height=300)
            validate_image_aspect_ratio(i, (1, 2.5), (2.5, 1))
        image_list: list[OmniParamImage] = []
        for i in await upload_images_to_comfyapi(cls, reference_images, wait_label="Uploading reference image"):
            image_list.append(OmniParamImage(image_url=i))
        if resolution == "4k":
            mode = "4k"
        elif resolution == "1080p":
            mode = "pro"
        else:
            mode = "std"
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/kling/v1/videos/omni-video", method="POST"),
            response_model=TaskStatusResponse,
            data=OmniProReferences2VideoRequest(
                model_name=model_name,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                duration=str(duration),
                image_list=image_list,
                mode=mode,
                sound="on" if generate_audio else "off",
                multi_shot=multi_shot,
                multi_prompt=multi_prompt_list,
                shot_type="customize" if multi_shot else None,
            ),
        )
        return await finish_omni_video_task(cls, response)


class OmniProVideoToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingOmniProVideoToVideoNode",
            display_name="Kling 3.0 Omni Video to Video",
            category="partner/video/Kling",
            description="Use a video and up to 4 reference images to generate a video with the latest Kling model.",
            inputs=[
                IO.Combo.Input("model_name", options=["kling-v3-omni", "kling-video-o1"]),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A text prompt describing the video content. "
                    "This can include both positive and negative descriptions.",
                ),
                IO.Combo.Input("aspect_ratio", options=["16:9", "9:16", "1:1"]),
                IO.Int.Input("duration", default=3, min=3, max=10, display_mode=IO.NumberDisplay.slider),
                IO.Video.Input("reference_video", tooltip="Video to use as a reference."),
                IO.Boolean.Input("keep_original_sound", default=True),
                IO.Image.Input(
                    "reference_images",
                    tooltip="Up to 4 additional reference images.",
                    optional=True,
                ),
                IO.Combo.Input("resolution", options=["1080p", "720p"], optional=True),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["duration", "resolution"]),
                expr="""
                (
                  $mode := (widgets.resolution = "720p") ? "std" : "pro";
                  $rates := {"std": 0.126, "pro": 0.168};
                  {"type":"usd","usd": $lookup($rates, $mode) * widgets.duration}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model_name: str,
        prompt: str,
        aspect_ratio: str,
        duration: int,
        reference_video: Input.Video,
        keep_original_sound: bool,
        reference_images: Input.Image | None = None,
        resolution: str = "1080p",
        seed: int = 0,
    ) -> IO.NodeOutput:
        _ = seed
        prompt = normalize_omni_prompt_references(prompt)
        validate_string(prompt, min_length=1, max_length=2500)
        validate_video_duration(reference_video, min_duration=3.0, max_duration=10.05)
        validate_video_dimensions(reference_video, min_width=720, min_height=720, max_width=2160, max_height=2160)
        image_list: list[OmniParamImage] = []
        if reference_images is not None:
            if get_number_of_images(reference_images) > 4:
                raise ValueError("The maximum number of reference images allowed with a video input is 4.")
            for i in reference_images:
                validate_image_dimensions(i, min_width=300, min_height=300)
                validate_image_aspect_ratio(i, (1, 2.5), (2.5, 1))
            for i in await upload_images_to_comfyapi(cls, reference_images, wait_label="Uploading reference image"):
                image_list.append(OmniParamImage(image_url=i))
        video_list = [
            OmniParamVideo(
                video_url=await upload_video_to_comfyapi(cls, reference_video, wait_label="Uploading reference video"),
                refer_type="feature",
                keep_original_sound="yes" if keep_original_sound else "no",
            )
        ]
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/kling/v1/videos/omni-video", method="POST"),
            response_model=TaskStatusResponse,
            data=OmniProReferences2VideoRequest(
                model_name=model_name,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                duration=str(duration),
                image_list=image_list if image_list else None,
                video_list=video_list,
                mode="pro" if resolution == "1080p" else "std",
            ),
        )
        return await finish_omni_video_task(cls, response)


class OmniProEditVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingOmniProEditVideoNode",
            display_name="Kling 3.0 Omni Edit Video",
            category="partner/video/Kling",
            essentials_category="Video Generation",
            description="Edit an existing video with the latest model from Kling.",
            inputs=[
                IO.Combo.Input("model_name", options=["kling-v3-omni", "kling-video-o1"]),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A text prompt describing the video content. "
                    "This can include both positive and negative descriptions.",
                ),
                IO.Video.Input("video", tooltip="Video for editing. The output video length will be the same."),
                IO.Boolean.Input("keep_original_sound", default=True),
                IO.Image.Input(
                    "reference_images",
                    tooltip="Up to 4 additional reference images.",
                    optional=True,
                ),
                IO.Combo.Input("resolution", options=["1080p", "720p"], optional=True),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["resolution"]),
                expr="""
                (
                  $mode := (widgets.resolution = "720p") ? "std" : "pro";
                  $rates := {"std": 0.126, "pro": 0.168};
                  {"type":"usd","usd": $lookup($rates, $mode), "format":{"suffix":"/second"}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model_name: str,
        prompt: str,
        video: Input.Video,
        keep_original_sound: bool,
        reference_images: Input.Image | None = None,
        resolution: str = "1080p",
        seed: int = 0,
    ) -> IO.NodeOutput:
        _ = seed
        prompt = normalize_omni_prompt_references(prompt)
        validate_string(prompt, min_length=1, max_length=2500)
        validate_video_duration(video, min_duration=3.0, max_duration=10.05)
        validate_video_dimensions(video, min_width=720, min_height=720, max_width=2160, max_height=2160)
        image_list: list[OmniParamImage] = []
        if reference_images is not None:
            if get_number_of_images(reference_images) > 4:
                raise ValueError("The maximum number of reference images allowed with a video input is 4.")
            for i in reference_images:
                validate_image_dimensions(i, min_width=300, min_height=300)
                validate_image_aspect_ratio(i, (1, 2.5), (2.5, 1))
            for i in await upload_images_to_comfyapi(cls, reference_images, wait_label="Uploading reference image"):
                image_list.append(OmniParamImage(image_url=i))
        video_list = [
            OmniParamVideo(
                video_url=await upload_video_to_comfyapi(cls, video, wait_label="Uploading base video"),
                refer_type="base",
                keep_original_sound="yes" if keep_original_sound else "no",
            )
        ]
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/kling/v1/videos/omni-video", method="POST"),
            response_model=TaskStatusResponse,
            data=OmniProReferences2VideoRequest(
                model_name=model_name,
                prompt=prompt,
                aspect_ratio=None,
                duration=None,
                image_list=image_list if image_list else None,
                video_list=video_list,
                mode="pro" if resolution == "1080p" else "std",
            ),
        )
        return await finish_omni_video_task(cls, response)


class OmniProImageNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingOmniProImageNode",
            display_name="Kling 3.0 Omni Image",
            category="partner/image/Kling",
            description="Create or edit images with the latest model from Kling.",
            inputs=[
                IO.Combo.Input("model_name", options=["kling-v3-omni", "kling-image-o1"]),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A text prompt describing the image content. "
                    "This can include both positive and negative descriptions.",
                ),
                IO.Combo.Input("resolution", options=["1K", "2K", "4K"]),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9"],
                ),
                IO.Combo.Input(
                    "series_amount",
                    options=["disabled", "2", "3", "4", "5", "6", "7", "8", "9"],
                    tooltip="Generate a series of images. Not supported for kling-image-o1.",
                ),
                IO.Image.Input(
                    "reference_images",
                    tooltip="Up to 10 additional reference images.",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["resolution", "series_amount", "model_name"]),
                expr="""
                (
                  $prices := {"1k": 0.028, "2k": 0.028, "4k": 0.056};
                  $base := $lookup($prices, widgets.resolution);
                  $isO1 := widgets.model_name = "kling-image-o1";
                  $mult := ($isO1 or widgets.series_amount = "disabled") ? 1 : $number(widgets.series_amount);
                  {"type":"usd","usd": $base * $mult}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model_name: str,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        series_amount: str = "disabled",
        reference_images: Input.Image | None = None,
        seed: int = 0,
    ) -> IO.NodeOutput:
        _ = seed
        if model_name == "kling-image-o1" and resolution == "4K":
            raise ValueError("4K resolution is not supported for kling-image-o1 model.")
        prompt = normalize_omni_prompt_references(prompt)
        validate_string(prompt, min_length=1, max_length=2500)
        image_list: list[OmniImageParamImage] = []
        if reference_images is not None:
            if get_number_of_images(reference_images) > 10:
                raise ValueError("The maximum number of reference images is 10.")
            for i in reference_images:
                validate_image_dimensions(i, min_width=300, min_height=300)
                validate_image_aspect_ratio(i, (1, 2.5), (2.5, 1))
            for i in await upload_images_to_comfyapi(cls, reference_images, wait_label="Uploading reference image"):
                image_list.append(OmniImageParamImage(image=i))
        use_series = series_amount != "disabled"
        if use_series and model_name == "kling-image-o1":
            raise ValueError("kling-image-o1 does not support series generation.")
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/kling/v1/images/omni-image", method="POST"),
            response_model=TaskStatusResponse,
            data=OmniProImageRequest(
                model_name=model_name,
                prompt=prompt,
                resolution=resolution.lower(),
                aspect_ratio=aspect_ratio,
                image_list=image_list if image_list else None,
                result_type="series" if use_series else None,
                series_amount=int(series_amount) if use_series else None,
            ),
        )
        if response.code:
            raise RuntimeError(
                f"Kling request failed. Code: {response.code}, Message: {response.message}, Data: {response.data}"
            )
        final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/kling/v1/images/omni-image/{response.data.task_id}"),
            response_model=TaskStatusResponse,
            status_extractor=lambda r: (r.data.task_status if r.data else None),
        )
        images = final_response.data.task_result.series_images or final_response.data.task_result.images
        tensors = [await download_url_to_image_tensor(img.url) for img in images]
        return IO.NodeOutput(torch.cat(tensors, dim=0))


class KlingCameraControlT2VNode(IO.ComfyNode):
    """
    Kling Text to Video Camera Control Node. This node is a text to video node, but it supports controlling the camera.
    Duration, mode, and model_name request fields are hard-coded because camera control is only supported in pro mode with the kling-v1-5 model at 5s duration as of 2025-05-02.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingCameraControlT2VNode",
            display_name="Kling Text to Video (Camera Control)",
            category="partner/video/Kling",
            description="Transform text into cinematic videos with professional camera movements that simulate real-world cinematography. Control virtual camera actions including zoom, rotation, pan, tilt, and first-person view, while maintaining focus on your original text.",
            inputs=[
                IO.String.Input("prompt", multiline=True, tooltip="Positive text prompt"),
                IO.String.Input("negative_prompt", multiline=True, tooltip="Negative text prompt"),
                IO.Float.Input("cfg_scale", default=0.75, min=0.0, max=1.0),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=KlingVideoGenAspectRatio,
                    default="16:9",
                ),
                IO.Custom("CAMERA_CONTROL").Input(
                    "camera_control",
                    tooltip="Can be created using the Kling Camera Controls node. Controls the camera movement and motion during the video generation.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="video_id"),
                IO.String.Output(display_name="duration"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.14}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        aspect_ratio: str,
        camera_control: KlingCameraControl | None = None,
    ) -> IO.NodeOutput:
        return await execute_text2video(
            cls,
            model_name=KlingVideoGenModelName.kling_v1,
            cfg_scale=cfg_scale,
            model_mode=KlingVideoGenMode.std,
            aspect_ratio=KlingVideoGenAspectRatio(aspect_ratio),
            duration=KlingVideoGenDuration.field_5,
            prompt=prompt,
            negative_prompt=negative_prompt,
            camera_control=camera_control,
        )


class KlingImage2VideoNode(IO.ComfyNode):
    """Kling Image to Video Node"""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingImage2VideoNode",
            display_name="Kling Image(First Frame) to Video",
            category="partner/video/Kling",
            inputs=[
                IO.Image.Input("start_frame", tooltip="The reference image used to generate the video."),
                IO.String.Input("prompt", multiline=True, tooltip="Positive text prompt"),
                IO.String.Input("negative_prompt", multiline=True, tooltip="Negative text prompt"),
                IO.Combo.Input(
                    "model_name",
                    options=KlingVideoGenModelName,
                    default="kling-v2-master",
                ),
                IO.Float.Input("cfg_scale", default=0.8, min=0.0, max=1.0),
                IO.Combo.Input("mode", options=KlingVideoGenMode, default=KlingVideoGenMode.std),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=KlingVideoGenAspectRatio,
                    default=KlingVideoGenAspectRatio.field_16_9,
                ),
                IO.Combo.Input("duration", options=KlingVideoGenDuration, default=KlingVideoGenDuration.field_5),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="video_id"),
                IO.String.Output(display_name="duration"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["mode", "model_name", "duration"]),
                expr="""
                (
                  $mode := widgets.mode;
                  $model := widgets.model_name;
                  $dur := widgets.duration;
                  $contains($model,"v2-5-turbo")
                    ? ($contains($dur,"10") ? {"type":"usd","usd":0.7} : {"type":"usd","usd":0.35})
                    : ($contains($model,"v2-1-master") or $contains($model,"v2-master"))
                      ? ($contains($dur,"10") ? {"type":"usd","usd":2.8} : {"type":"usd","usd":1.4})
                      : ($contains($model,"v2-1") or $contains($model,"v1-6") or $contains($model,"v1-5"))
                        ? (
                            $contains($mode,"pro")
                              ? ($contains($dur,"10") ? {"type":"usd","usd":0.98} : {"type":"usd","usd":0.49})
                              : ($contains($dur,"10") ? {"type":"usd","usd":0.56} : {"type":"usd","usd":0.28})
                          )
                        : $contains($model,"v1")
                          ? (
                              $contains($mode,"pro")
                                ? ($contains($dur,"10") ? {"type":"usd","usd":0.98} : {"type":"usd","usd":0.49})
                                : ($contains($dur,"10") ? {"type":"usd","usd":0.28} : {"type":"usd","usd":0.14})
                            )
                          : {"type":"usd","usd":0.14}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        start_frame: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        model_name: str,
        cfg_scale: float,
        mode: str,
        aspect_ratio: str,
        duration: str,
        camera_control: KlingCameraControl | None = None,
        end_frame: torch.Tensor | None = None,
    ) -> IO.NodeOutput:
        return await execute_image2video(
            cls,
            start_frame=start_frame,
            prompt=prompt,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale,
            model_name=model_name,
            aspect_ratio=aspect_ratio,
            model_mode=mode,
            duration=duration,
            camera_control=camera_control,
            end_frame=end_frame,
        )


class KlingCameraControlI2VNode(IO.ComfyNode):
    """
    Kling Image to Video Camera Control Node. This node is a image to video node, but it supports controlling the camera.
    Duration, mode, and model_name request fields are hard-coded because camera control is only supported in pro mode with the kling-v1-5 model at 5s duration as of 2025-05-02.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingCameraControlI2VNode",
            display_name="Kling Image to Video (Camera Control)",
            category="partner/video/Kling",
            description="Transform still images into cinematic videos with professional camera movements that simulate real-world cinematography. Control virtual camera actions including zoom, rotation, pan, tilt, and first-person view, while maintaining focus on your original image.",
            inputs=[
                IO.Image.Input(
                    "start_frame",
                    tooltip="Reference Image - URL or Base64 encoded string, cannot exceed 10MB, resolution not less than 300*300px, aspect ratio between 1:2.5 ~ 2.5:1. Base64 should not include data:image prefix.",
                ),
                IO.String.Input("prompt", multiline=True, tooltip="Positive text prompt"),
                IO.String.Input("negative_prompt", multiline=True, tooltip="Negative text prompt"),
                IO.Float.Input("cfg_scale", default=0.75, min=0.0, max=1.0),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=KlingVideoGenAspectRatio,
                    default=KlingVideoGenAspectRatio.field_16_9,
                ),
                IO.Custom("CAMERA_CONTROL").Input(
                    "camera_control",
                    tooltip="Can be created using the Kling Camera Controls node. Controls the camera movement and motion during the video generation.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="video_id"),
                IO.String.Output(display_name="duration"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.49}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        start_frame: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        aspect_ratio: str,
        camera_control: KlingCameraControl,
    ) -> IO.NodeOutput:
        return await execute_image2video(
            cls,
            model_name=KlingVideoGenModelName.kling_v1_5,
            start_frame=start_frame,
            cfg_scale=cfg_scale,
            model_mode=KlingVideoGenMode.pro,
            aspect_ratio=KlingVideoGenAspectRatio(aspect_ratio),
            duration=KlingVideoGenDuration.field_5,
            prompt=prompt,
            negative_prompt=negative_prompt,
            camera_control=camera_control,
        )


class KlingStartEndFrameNode(IO.ComfyNode):
    """
    Kling First Last Frame Node. This node allows creation of a video from a first and last frame. It calls the normal image to video endpoint, but only allows the subset of input options that support the `image_tail` request field.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        modes = list(MODE_START_END_FRAME.keys())
        return IO.Schema(
            node_id="KlingStartEndFrameNode",
            display_name="Kling Start-End Frame to Video",
            category="partner/video/Kling",
            description="Generate a video sequence that transitions between your provided start and end images. The node creates all frames in between, producing a smooth transformation from the first frame to the last.",
            inputs=[
                IO.Image.Input(
                    "start_frame",
                    tooltip="Reference Image - URL or Base64 encoded string, cannot exceed 10MB, resolution not less than 300*300px, aspect ratio between 1:2.5 ~ 2.5:1. Base64 should not include data:image prefix.",
                ),
                IO.Image.Input(
                    "end_frame",
                    tooltip="Reference Image - End frame control. URL or Base64 encoded string, cannot exceed 10MB, resolution not less than 300*300px. Base64 should not include data:image prefix.",
                ),
                IO.String.Input("prompt", multiline=True, tooltip="Positive text prompt"),
                IO.String.Input("negative_prompt", multiline=True, tooltip="Negative text prompt"),
                IO.Float.Input("cfg_scale", default=0.5, min=0.0, max=1.0),
                IO.Combo.Input("aspect_ratio", options=["16:9", "9:16", "1:1"]),
                IO.Combo.Input(
                    "mode",
                    options=modes,
                    default=modes[6],
                    tooltip="The configuration to use for the video generation following the format: mode / duration / model_name.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="video_id"),
                IO.String.Output(display_name="duration"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["mode"]),
                expr="""
                (
                  $m := widgets.mode;
                  $contains($m,"v2-5-turbo")
                    ? ($contains($m,"10") ? {"type":"usd","usd":0.7} : {"type":"usd","usd":0.35})
                    : $contains($m,"v2-1")
                      ? ($contains($m,"10s") ? {"type":"usd","usd":0.98} : {"type":"usd","usd":0.49})
                      : $contains($m,"v2-master")
                        ? ($contains($m,"10s") ? {"type":"usd","usd":2.8} : {"type":"usd","usd":1.4})
                        : $contains($m,"v1-6")
                          ? (
                              $contains($m,"pro")
                                ? ($contains($m,"10s") ? {"type":"usd","usd":0.98} : {"type":"usd","usd":0.49})
                                : ($contains($m,"10s") ? {"type":"usd","usd":0.56} : {"type":"usd","usd":0.28})
                            )
                          : $contains($m,"v1")
                            ? (
                                $contains($m,"pro")
                                  ? ($contains($m,"10s") ? {"type":"usd","usd":0.98} : {"type":"usd","usd":0.49})
                                  : ($contains($m,"10s") ? {"type":"usd","usd":0.28} : {"type":"usd","usd":0.14})
                              )
                            : {"type":"usd","usd":0.14}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        start_frame: torch.Tensor,
        end_frame: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        aspect_ratio: str,
        mode: str,
    ) -> IO.NodeOutput:
        mode, duration, model_name = MODE_START_END_FRAME[mode]
        return await execute_image2video(
            cls,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_name=model_name,
            start_frame=start_frame,
            cfg_scale=cfg_scale,
            model_mode=mode,
            aspect_ratio=aspect_ratio,
            duration=duration,
            end_frame=end_frame,
        )


class KlingVideoExtendNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingVideoExtendNode",
            display_name="Kling Video Extend",
            category="partner/video/Kling",
            description="Kling Video Extend Node. Extend videos made by other Kling nodes. The video_id is created by using other Kling Nodes.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Positive text prompt for guiding the video extension",
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    tooltip="Negative text prompt for elements to avoid in the extended video",
                ),
                IO.Float.Input("cfg_scale", default=0.5, min=0.0, max=1.0),
                IO.String.Input(
                    "video_id",
                    force_input=True,
                    tooltip="The ID of the video to be extended. Supports videos generated by text-to-video, image-to-video, and previous video extension operations. Cannot exceed 3 minutes total duration after extension.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="video_id"),
                IO.String.Output(display_name="duration"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.28}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        video_id: str,
    ) -> IO.NodeOutput:
        validate_prompts(prompt, negative_prompt, MAX_PROMPT_LENGTH_T2V)
        task_creation_response = await sync_op(
            cls,
            ApiEndpoint(path=PATH_VIDEO_EXTEND, method="POST"),
            response_model=KlingVideoExtendResponse,
            data=KlingVideoExtendRequest(
                prompt=prompt if prompt else None,
                negative_prompt=negative_prompt if negative_prompt else None,
                cfg_scale=cfg_scale,
                video_id=video_id,
            ),
        )

        validate_task_creation_response(task_creation_response)
        task_id = task_creation_response.data.task_id

        final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"{PATH_VIDEO_EXTEND}/{task_id}"),
            response_model=KlingVideoExtendResponse,
            estimated_duration=AVERAGE_DURATION_VIDEO_EXTEND,
            status_extractor=lambda r: (r.data.task_status.value if r.data and r.data.task_status else None),
        )
        validate_video_result_response(final_response)

        video = get_video_from_response(final_response)
        return IO.NodeOutput(await download_url_to_video_output(str(video.url)), str(video.id), str(video.duration))


class KlingDualCharacterVideoEffectNode(IO.ComfyNode):
    """Kling Dual Character Video Effect Node"""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingDualCharacterVideoEffectNode",
            display_name="Kling Dual Character Video Effects",
            category="partner/video/Kling",
            description="Achieve different special effects when generating a video based on the effect_scene. First image will be positioned on left side, second on right side of the composite.",
            inputs=[
                IO.Image.Input("image_left", tooltip="Left side image"),
                IO.Image.Input("image_right", tooltip="Right side image"),
                IO.Combo.Input(
                    "effect_scene",
                    options=[i.value for i in KlingDualCharacterEffectsScene],
                ),
                IO.Combo.Input(
                    "model_name",
                    options=[i.value for i in KlingCharacterEffectModelName],
                    default="kling-v1",
                ),
                IO.Combo.Input(
                    "mode",
                    options=[i.value for i in KlingVideoGenMode],
                    default="std",
                ),
                IO.Combo.Input(
                    "duration",
                    options=[i.value for i in KlingVideoGenDuration],
                ),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="duration"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["mode", "model_name", "duration"]),
                expr="""
                (
                  $mode := widgets.mode;
                  $model := widgets.model_name;
                  $dur := widgets.duration;
                  ($contains($model,"v1-6") or $contains($model,"v1-5"))
                    ? (
                        $contains($mode,"pro")
                          ? ($contains($dur,"10") ? {"type":"usd","usd":0.98} : {"type":"usd","usd":0.49})
                          : ($contains($dur,"10") ? {"type":"usd","usd":0.56} : {"type":"usd","usd":0.28})
                      )
                    : $contains($model,"v1")
                      ? (
                          $contains($mode,"pro")
                            ? ($contains($dur,"10") ? {"type":"usd","usd":0.98} : {"type":"usd","usd":0.49})
                            : ($contains($dur,"10") ? {"type":"usd","usd":0.28} : {"type":"usd","usd":0.14})
                        )
                      : {"type":"usd","usd":0.14}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        image_left: torch.Tensor,
        image_right: torch.Tensor,
        effect_scene: KlingDualCharacterEffectsScene,
        model_name: KlingCharacterEffectModelName,
        mode: KlingVideoGenMode,
        duration: KlingVideoGenDuration,
    ) -> IO.NodeOutput:
        video, _, duration = await execute_video_effect(
            cls,
            dual_character=True,
            effect_scene=effect_scene,
            model_name=model_name,
            model_mode=mode,
            duration=duration,
            image_1=image_left,
            image_2=image_right,
        )
        return IO.NodeOutput(video, duration)


class KlingSingleImageVideoEffectNode(IO.ComfyNode):
    """Kling Single Image Video Effect Node"""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingSingleImageVideoEffectNode",
            display_name="Kling Video Effects",
            category="partner/video/Kling",
            description="Achieve different special effects when generating a video based on the effect_scene.",
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip=" Reference Image. URL or Base64 encoded string (without data:image prefix). File size cannot exceed 10MB, resolution not less than 300*300px, aspect ratio between 1:2.5 ~ 2.5:1",
                ),
                IO.Combo.Input(
                    "effect_scene",
                    options=[i.value for i in KlingSingleImageEffectsScene],
                ),
                IO.Combo.Input(
                    "model_name",
                    options=[i.value for i in KlingSingleImageEffectModelName],
                ),
                IO.Combo.Input(
                    "duration",
                    options=[i.value for i in KlingVideoGenDuration],
                ),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="video_id"),
                IO.String.Output(display_name="duration"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["effect_scene"]),
                expr="""
                (
                  ($contains(widgets.effect_scene,"dizzydizzy") or $contains(widgets.effect_scene,"bloombloom"))
                    ? {"type":"usd","usd":0.49}
                    : {"type":"usd","usd":0.28}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        effect_scene: KlingSingleImageEffectsScene,
        model_name: KlingSingleImageEffectModelName,
        duration: KlingVideoGenDuration,
    ) -> IO.NodeOutput:
        return IO.NodeOutput(
            *(
                await execute_video_effect(
                    cls,
                    dual_character=False,
                    effect_scene=effect_scene,
                    model_name=model_name,
                    duration=duration,
                    image_1=image,
                )
            )
        )


class KlingLipSyncAudioToVideoNode(IO.ComfyNode):
    """Kling Lip Sync Audio to Video Node. Syncs mouth movements in a video file to the audio content of an audio file."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingLipSyncAudioToVideoNode",
            display_name="Kling Lip Sync Video with Audio",
            category="partner/video/Kling",
            essentials_category="Video Generation",
            description="Kling Lip Sync Audio to Video Node. Syncs mouth movements in a video file to the audio content of an audio file. When using, ensure that the audio contains clearly distinguishable vocals and that the video contains a distinct face. The audio file should not be larger than 5MB. The video file should not be larger than 100MB, should have height/width between 720px and 1920px, and should be between 2s and 10s in length.",
            inputs=[
                IO.Video.Input("video"),
                IO.Audio.Input("audio"),
                IO.Combo.Input(
                    "voice_language",
                    options=[i.value for i in KlingLipSyncVoiceLanguage],
                    default="en",
                ),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="video_id"),
                IO.String.Output(display_name="duration"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.1,"format":{"approximate":true}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        audio: Input.Audio,
        voice_language: str,
    ) -> IO.NodeOutput:
        return await execute_lipsync(
            cls,
            video=video,
            audio=audio,
            voice_language=voice_language,
            model_mode="audio2video",
        )


class KlingLipSyncTextToVideoNode(IO.ComfyNode):
    """Kling Lip Sync Text to Video Node. Syncs mouth movements in a video file to a text prompt."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingLipSyncTextToVideoNode",
            display_name="Kling Lip Sync Video with Text",
            category="partner/video/Kling",
            description="Kling Lip Sync Text to Video Node. Syncs mouth movements in a video file to a text prompt. The video file should not be larger than 100MB, should have height/width between 720px and 1920px, and should be between 2s and 10s in length.",
            inputs=[
                IO.Video.Input("video"),
                IO.String.Input(
                    "text",
                    multiline=True,
                    tooltip="Text Content for Lip-Sync Video Generation. Required when mode is text2video. Maximum length is 120 characters.",
                ),
                IO.Combo.Input(
                    "voice",
                    options=list(VOICES_CONFIG.keys()),
                    default="Melody",
                ),
                IO.Float.Input(
                    "voice_speed",
                    default=1,
                    min=0.8,
                    max=2.0,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Speech Rate. Valid range: 0.8~2.0, accurate to one decimal place.",
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="video_id"),
                IO.String.Output(display_name="duration"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.1,"format":{"approximate":true}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        text: str,
        voice: str,
        voice_speed: float,
    ) -> IO.NodeOutput:
        voice_id, voice_language = VOICES_CONFIG[voice]
        return await execute_lipsync(
            cls,
            video=video,
            text=text,
            voice_language=voice_language,
            voice_id=voice_id,
            voice_speed=voice_speed,
            model_mode="text2video",
        )


class KlingVirtualTryOnNode(IO.ComfyNode):
    """Kling Virtual Try On Node."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingVirtualTryOnNode",
            display_name="Kling Virtual Try On",
            category="partner/image/Kling",
            description="Kling Virtual Try On Node. Input a human image and a cloth image to try on the cloth on the human. You can merge multiple clothing item pictures into one image with a white background.",
            inputs=[
                IO.Image.Input("human_image"),
                IO.Image.Input("cloth_image"),
                IO.Combo.Input(
                    "model_name",
                    options=[i.value for i in KlingVirtualTryOnModelName],
                    default="kolors-virtual-try-on-v1",
                ),
            ],
            outputs=[
                IO.Image.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.7}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        human_image: torch.Tensor,
        cloth_image: torch.Tensor,
        model_name: KlingVirtualTryOnModelName,
    ) -> IO.NodeOutput:
        task_creation_response = await sync_op(
            cls,
            ApiEndpoint(path=PATH_VIRTUAL_TRY_ON, method="POST"),
            response_model=KlingVirtualTryOnResponse,
            data=KlingVirtualTryOnRequest(
                human_image=tensor_to_base64_string(human_image),
                cloth_image=tensor_to_base64_string(cloth_image),
                model_name=model_name,
            ),
        )

        validate_task_creation_response(task_creation_response)
        task_id = task_creation_response.data.task_id

        final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"{PATH_VIRTUAL_TRY_ON}/{task_id}"),
            response_model=KlingVirtualTryOnResponse,
            estimated_duration=AVERAGE_DURATION_VIRTUAL_TRY_ON,
            status_extractor=lambda r: (r.data.task_status.value if r.data and r.data.task_status else None),
        )
        validate_image_result_response(final_response)

        images = get_images_from_response(final_response)
        return IO.NodeOutput(await image_result_to_node_output(images))


class KlingImageGenerationNode(IO.ComfyNode):
    """Kling Image Generation Node. Generate an image from a text prompt with an optional reference image."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingImageGenerationNode",
            display_name="Kling 3.0 Image",
            category="partner/image/Kling",
            description="Kling Image Generation Node. Generate an image from a text prompt with an optional reference image.",
            inputs=[
                IO.String.Input("prompt", multiline=True, tooltip="Positive text prompt"),
                IO.String.Input("negative_prompt", multiline=True, tooltip="Negative text prompt"),
                IO.Combo.Input(
                    "image_type",
                    options=[i.value for i in KlingImageGenImageReferenceType],
                    advanced=True,
                ),
                IO.Float.Input(
                    "image_fidelity",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Reference intensity for user-uploaded images",
                    advanced=True,
                ),
                IO.Float.Input(
                    "human_fidelity",
                    default=0.45,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Subject reference similarity",
                    advanced=True,
                ),
                IO.Combo.Input("model_name", options=["kling-v3", "kling-v2", "kling-v1-5"]),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=[i.value for i in KlingImageGenAspectRatio],
                    default="16:9",
                ),
                IO.Int.Input(
                    "n",
                    default=1,
                    min=1,
                    max=9,
                    tooltip="Number of generated images",
                ),
                IO.Image.Input("image", optional=True),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model_name", "n"], inputs=["image"]),
                expr="""
                (
                  $m := widgets.model_name;
                  $base :=
                    $contains($m,"kling-v1-5")
                      ? (inputs.image.connected ? 0.028 : 0.014)
                      : $contains($m,"kling-v3") ? 0.028 : 0.014;
                  {"type":"usd","usd": $base * widgets.n}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model_name: str,
        prompt: str,
        negative_prompt: str,
        image_type: KlingImageGenImageReferenceType,
        image_fidelity: float,
        human_fidelity: float,
        n: int,
        aspect_ratio: KlingImageGenAspectRatio,
        image: torch.Tensor | None = None,
        seed: int = 0,
    ) -> IO.NodeOutput:
        _ = seed
        validate_string(prompt, field_name="prompt", min_length=1, max_length=MAX_PROMPT_LENGTH_IMAGE_GEN)
        validate_string(negative_prompt, field_name="negative_prompt", max_length=MAX_PROMPT_LENGTH_IMAGE_GEN)
        task_creation_response = await sync_op(
            cls,
            ApiEndpoint(path=PATH_IMAGE_GENERATIONS, method="POST"),
            response_model=KlingImageGenerationsResponse,
            data=KlingImageGenerationsRequest(
                model_name=model_name,
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=tensor_to_base64_string(image) if image is not None else None,
                image_reference=image_type if image is not None else None,
                image_fidelity=image_fidelity,
                human_fidelity=human_fidelity,
                n=n,
                aspect_ratio=aspect_ratio,
            ),
        )

        validate_task_creation_response(task_creation_response)
        task_id = task_creation_response.data.task_id

        final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"{PATH_IMAGE_GENERATIONS}/{task_id}"),
            response_model=KlingImageGenerationsResponse,
            estimated_duration=AVERAGE_DURATION_IMAGE_GEN,
            status_extractor=lambda r: (r.data.task_status.value if r.data and r.data.task_status else None),
        )
        validate_image_result_response(final_response)

        images = get_images_from_response(final_response)
        return IO.NodeOutput(await image_result_to_node_output(images))


class TextToVideoWithAudio(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingTextToVideoWithAudio",
            display_name="Kling 2.6 Text to Video with Audio",
            category="partner/video/Kling",
            inputs=[
                IO.Combo.Input("model_name", options=["kling-v2-6"]),
                IO.String.Input("prompt", multiline=True, tooltip="Positive text prompt."),
                IO.Combo.Input("mode", options=["pro"]),
                IO.Combo.Input("aspect_ratio", options=["16:9", "9:16", "1:1"]),
                IO.Combo.Input("duration", options=[5, 10]),
                IO.Boolean.Input("generate_audio", default=True, advanced=True),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["duration", "generate_audio"]),
                expr="""{"type":"usd","usd": 0.07 * widgets.duration * (widgets.generate_audio ? 2 : 1)}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model_name: str,
        prompt: str,
        mode: str,
        aspect_ratio: str,
        duration: int,
        generate_audio: bool,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=2500)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/kling/v1/videos/text2video", method="POST"),
            response_model=TaskStatusResponse,
            data=TextToVideoWithAudioRequest(
                model_name=model_name,
                prompt=prompt,
                mode=mode,
                aspect_ratio=aspect_ratio,
                duration=str(duration),
                sound="on" if generate_audio else "off",
            ),
        )
        if response.code:
            raise RuntimeError(
                f"Kling request failed. Code: {response.code}, Message: {response.message}, Data: {response.data}"
            )
        final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/kling/v1/videos/text2video/{response.data.task_id}"),
            response_model=TaskStatusResponse,
            status_extractor=lambda r: (r.data.task_status if r.data else None),
        )
        return IO.NodeOutput(await download_url_to_video_output(final_response.data.task_result.videos[0].url))


class ImageToVideoWithAudio(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingImageToVideoWithAudio",
            display_name="Kling 2.6 Image(First Frame) to Video with Audio",
            category="partner/video/Kling",
            inputs=[
                IO.Combo.Input("model_name", options=["kling-v2-6"]),
                IO.Image.Input("start_frame"),
                IO.String.Input("prompt", multiline=True, tooltip="Positive text prompt."),
                IO.Combo.Input("mode", options=["pro"]),
                IO.Combo.Input("duration", options=[5, 10]),
                IO.Boolean.Input("generate_audio", default=True, advanced=True),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["duration", "generate_audio"]),
                expr="""{"type":"usd","usd": 0.07 * widgets.duration * (widgets.generate_audio ? 2 : 1)}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model_name: str,
        start_frame: Input.Image,
        prompt: str,
        mode: str,
        duration: int,
        generate_audio: bool,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=2500)
        validate_image_dimensions(start_frame, min_width=300, min_height=300)
        validate_image_aspect_ratio(start_frame, (1, 2.5), (2.5, 1))
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/kling/v1/videos/image2video", method="POST"),
            response_model=TaskStatusResponse,
            data=ImageToVideoWithAudioRequest(
                model_name=model_name,
                image=(await upload_images_to_comfyapi(cls, start_frame))[0],
                prompt=prompt,
                mode=mode,
                duration=str(duration),
                sound="on" if generate_audio else "off",
            ),
        )
        if response.code:
            raise RuntimeError(
                f"Kling request failed. Code: {response.code}, Message: {response.message}, Data: {response.data}"
            )
        final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/kling/v1/videos/image2video/{response.data.task_id}"),
            response_model=TaskStatusResponse,
            status_extractor=lambda r: (r.data.task_status if r.data else None),
        )
        return IO.NodeOutput(await download_url_to_video_output(final_response.data.task_result.videos[0].url))


class MotionControl(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingMotionControl",
            display_name="Kling Motion Control",
            category="partner/video/Kling",
            inputs=[
                IO.String.Input("prompt", multiline=True),
                IO.Image.Input("reference_image"),
                IO.Video.Input(
                    "reference_video",
                    tooltip="Motion reference video used to drive movement/expression.\n"
                    "Duration limits depend on character_orientation:\n"
                    " - image: 3–10s (max 10s)\n"
                    " - video: 3–30s (max 30s)",
                ),
                IO.Boolean.Input("keep_original_sound", default=True),
                IO.Combo.Input(
                    "character_orientation",
                    options=["video", "image"],
                    tooltip="Controls where the character's facing/orientation comes from.\n"
                    "video: movements, expressions, camera moves, and orientation "
                    "follow the motion reference video (other details via prompt).\n"
                    "image: movements and expressions still follow the motion reference video, "
                    "but the character orientation matches the reference image (camera/other details via prompt).",
                ),
                IO.Combo.Input("mode", options=["pro", "std"]),
                IO.Combo.Input("model", options=["kling-v3", "kling-v2-6"], optional=True),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["mode", "model"]),
                expr="""
                (
                  $prices := {
                    "kling-v3": {"std": 0.126, "pro": 0.168},
                    "kling-v2-6": {"std": 0.07, "pro": 0.112}
                  };
                  $modelPrices := $lookup($prices, widgets.model);
                  {"type":"usd","usd": $lookup($modelPrices, widgets.mode), "format":{"suffix":"/second"}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        reference_image: Input.Image,
        reference_video: Input.Video,
        keep_original_sound: bool,
        character_orientation: str,
        mode: str,
        model: str = "kling-v2-6",
    ) -> IO.NodeOutput:
        validate_string(prompt, max_length=2500)
        validate_image_dimensions(reference_image, min_width=340, min_height=340)
        validate_image_aspect_ratio(reference_image, (1, 2.5), (2.5, 1))
        if character_orientation == "image":
            validate_video_duration(reference_video, min_duration=3, max_duration=10)
        else:
            validate_video_duration(reference_video, min_duration=3, max_duration=30)
        validate_video_dimensions(reference_video, min_width=340, min_height=340, max_width=3850, max_height=3850)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/kling/v1/videos/motion-control", method="POST"),
            response_model=TaskStatusResponse,
            data=MotionControlRequest(
                prompt=prompt,
                image_url=(await upload_images_to_comfyapi(cls, reference_image))[0],
                video_url=await upload_video_to_comfyapi(cls, reference_video),
                keep_original_sound="yes" if keep_original_sound else "no",
                character_orientation=character_orientation,
                mode=mode,
                model_name=model,
            ),
        )
        if response.code:
            raise RuntimeError(
                f"Kling request failed. Code: {response.code}, Message: {response.message}, Data: {response.data}"
            )
        final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/kling/v1/videos/motion-control/{response.data.task_id}"),
            response_model=TaskStatusResponse,
            status_extractor=lambda r: (r.data.task_status if r.data else None),
        )
        return IO.NodeOutput(await download_url_to_video_output(final_response.data.task_result.videos[0].url))


def build_turbo_shot_prompt(multi_prompt: list[MultiPromptEntry]) -> str:
    """Render storyboard entries into the Turbo multi-shot prompt 'shot n, m, words; ...'."""
    return "; ".join(f"shot {i}, {int(e.duration)}, {e.prompt}" for i, e in enumerate(multi_prompt, 1)) + ";"


def _turbo_video_url(response: Kling3TurboQueryResponse) -> str:
    """Extract the result video URL from a /tasks response (data[].outputs[] where type == 'video')."""
    task = response.data[0] if response.data else None
    if task and task.outputs:
        for output in task.outputs:
            if output.type == "video" and output.url:
                return output.url
    raise RuntimeError(f"Kling 3.0 Turbo task finished without a video output: {response.model_dump()}")


async def execute_kling_turbo(
    cls: type[IO.ComfyNode],
    *,
    prompt: str,
    resolution: str,
    aspect_ratio: str,
    duration: int,
    start_frame: torch.Tensor | None,
) -> IO.NodeOutput:
    """Create + poll a Kling 3.0 Turbo task. Image-to-video when start_frame is given, else text-to-video."""
    if start_frame is not None:
        validate_image_dimensions(start_frame, min_width=300, min_height=300)
        validate_image_aspect_ratio(start_frame, (1, 2.5), (2.5, 1))
        contents = [Kling3TurboContent(type="first_frame", url=tensor_to_base64_string(start_frame))]
        if prompt:
            contents.insert(0, Kling3TurboContent(type="prompt", text=prompt))
        create = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/kling/image-to-video/kling-3.0-turbo", method="POST"),
            response_model=Kling3TurboCreateResponse,
            data=Kling3TurboImage2VideoRequest(
                contents=contents,
                settings=Kling3TurboSettings(resolution=resolution, duration=duration),  # i2v: no aspect_ratio
            ),
        )
    else:
        create = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/kling/text-to-video/kling-3.0-turbo", method="POST"),
            response_model=Kling3TurboCreateResponse,
            data=Kling3TurboText2VideoRequest(
                prompt=prompt,
                settings=Kling3TurboSettings(resolution=resolution, aspect_ratio=aspect_ratio, duration=duration),
            ),
        )
    if not (create.data and create.data.id):
        raise RuntimeError(f"Kling 3.0 Turbo create failed. Code: {create.code}, Message: {create.message}")
    final_response = await poll_op(
        cls,
        ApiEndpoint(path="/proxy/kling/tasks", query_params={"task_ids": create.data.id}),
        response_model=Kling3TurboQueryResponse,
        status_extractor=lambda r: (r.data[0].status if r.data else None),
    )
    return IO.NodeOutput(await download_url_to_video_output(_turbo_video_url(final_response)))


class KlingVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingVideoNode",
            display_name="Kling 3.0 Video",
            category="partner/video/Kling",
            description="Generate videos with Kling V3. "
            "Supports text-to-video and image-to-video with optional storyboard multi-prompt and audio generation.",
            inputs=[
                IO.DynamicCombo.Input(
                    "multi_shot",
                    options=[
                        IO.DynamicCombo.Option(
                            "disabled",
                            [
                                IO.String.Input("prompt", multiline=True, default=""),
                                IO.String.Input("negative_prompt", multiline=True, default=""),
                                IO.Int.Input(
                                    "duration",
                                    default=5,
                                    min=3,
                                    max=15,
                                    display_mode=IO.NumberDisplay.slider,
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option("1 storyboard", _generate_storyboard_inputs(1)),
                        IO.DynamicCombo.Option("2 storyboards", _generate_storyboard_inputs(2)),
                        IO.DynamicCombo.Option("3 storyboards", _generate_storyboard_inputs(3)),
                        IO.DynamicCombo.Option("4 storyboards", _generate_storyboard_inputs(4)),
                        IO.DynamicCombo.Option("5 storyboards", _generate_storyboard_inputs(5)),
                        IO.DynamicCombo.Option("6 storyboards", _generate_storyboard_inputs(6)),
                    ],
                    tooltip="Generate a series of video segments with individual prompts and durations.",
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=True,
                    tooltip="'kling-3.0-turbo' always generates native audio, so the audio toggle is ignored.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "kling-v3",
                            [
                                IO.Combo.Input("resolution", options=["4k", "1080p", "720p"], default="1080p"),
                                IO.Combo.Input(
                                    "aspect_ratio",
                                    options=["16:9", "9:16", "1:1"],
                                    tooltip="Ignored in image-to-video mode.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "kling-3.0-turbo",
                            [
                                IO.Combo.Input("resolution", options=["1080p", "720p"], default="720p"),
                                IO.Combo.Input(
                                    "aspect_ratio",
                                    options=["16:9", "9:16", "1:1"],
                                    tooltip="Ignored in image-to-video mode.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Model and generation settings.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                            "results are non-deterministic regardless of seed.",
                ),
                IO.Image.Input(
                    "start_frame",
                    optional=True,
                    tooltip="Optional start frame image. When connected, switches to image-to-video mode.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(
                    widgets=[
                        "model",
                        "model.resolution",
                        "generate_audio",
                        "multi_shot",
                        "multi_shot.duration",
                        "multi_shot.storyboard_1_duration",
                        "multi_shot.storyboard_2_duration",
                        "multi_shot.storyboard_3_duration",
                        "multi_shot.storyboard_4_duration",
                        "multi_shot.storyboard_5_duration",
                        "multi_shot.storyboard_6_duration",
                    ],
                ),
                expr="""
                (
                  $res := $lookup(widgets, "model.resolution");
                  $ms := widgets.multi_shot;
                  $isSb := $ms != "disabled";
                  $n := $isSb ? $number($substring($ms, 0, 1)) : 0;
                  $d1 := $lookup(widgets, "multi_shot.storyboard_1_duration");
                  $d2 := $n >= 2 ? $lookup(widgets, "multi_shot.storyboard_2_duration") : 0;
                  $d3 := $n >= 3 ? $lookup(widgets, "multi_shot.storyboard_3_duration") : 0;
                  $d4 := $n >= 4 ? $lookup(widgets, "multi_shot.storyboard_4_duration") : 0;
                  $d5 := $n >= 5 ? $lookup(widgets, "multi_shot.storyboard_5_duration") : 0;
                  $d6 := $n >= 6 ? $lookup(widgets, "multi_shot.storyboard_6_duration") : 0;
                  $dur := $isSb ? $d1 + $d2 + $d3 + $d4 + $d5 + $d6 : $lookup(widgets, "multi_shot.duration");
                  widgets.model = "kling-3.0-turbo"
                    ? {"type":"usd","usd": ($res = "1080p" ? 0.14 : 0.112) * $dur}
                    : (
                        $rates := {
                          "4k": {"off": 0.42, "on": 0.42},
                          "1080p": {"off": 0.112, "on": 0.168},
                          "720p": {"off": 0.084, "on": 0.126}
                        };
                        $audio := widgets.generate_audio ? "on" : "off";
                        $rate := $lookup($lookup($rates, $res), $audio);
                        {"type":"usd","usd": $rate * $dur}
                      )
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        multi_shot: dict,
        generate_audio: bool,
        model: dict,
        seed: int,
        start_frame: Input.Image | None = None,
    ) -> IO.NodeOutput:
        _ = seed
        if model["resolution"] == "4k":
            mode = "4k"
        elif model["resolution"] == "1080p":
            mode = "pro"
        else:
            mode = "std"
        custom_multi_shot = False
        if multi_shot["multi_shot"] == "disabled":
            shot_type = None
        else:
            shot_type = "customize"
            custom_multi_shot = True

        multi_prompt_list = None
        if shot_type == "customize":
            count = int(multi_shot["multi_shot"].split()[0])
            multi_prompt_list = []
            for i in range(1, count + 1):
                sb_prompt = multi_shot[f"storyboard_{i}_prompt"]
                sb_duration = multi_shot[f"storyboard_{i}_duration"]
                validate_string(sb_prompt, field_name=f"storyboard_{i}_prompt", min_length=1, max_length=512)
                multi_prompt_list.append(
                    MultiPromptEntry(
                        index=i,
                        prompt=sb_prompt,
                        duration=str(sb_duration),
                    )
                )
            duration = sum(int(e.duration) for e in multi_prompt_list)
            if duration < 3 or duration > 15:
                raise ValueError(
                    f"Total storyboard duration ({duration}s) must be between 3 and 15 seconds."
                )
        else:
            duration = multi_shot["duration"]
            validate_string(multi_shot["prompt"], min_length=1, max_length=2500)

        if model["model"] == "kling-3.0-turbo":
            turbo_prompt = build_turbo_shot_prompt(multi_prompt_list) if custom_multi_shot else multi_shot["prompt"]
            return await execute_kling_turbo(
                cls,
                prompt=turbo_prompt,
                resolution=model["resolution"],
                aspect_ratio=model["aspect_ratio"],
                duration=duration,
                start_frame=start_frame,
            )

        if start_frame is not None:
            validate_image_dimensions(start_frame, min_width=300, min_height=300)
            validate_image_aspect_ratio(start_frame, (1, 2.5), (2.5, 1))
            image_url = await upload_image_to_comfyapi(cls, start_frame, wait_label="Uploading start frame")
            response = await sync_op(
                cls,
                ApiEndpoint(path="/proxy/kling/v1/videos/image2video", method="POST"),
                response_model=TaskStatusResponse,
                data=ImageToVideoWithAudioRequest(
                    model_name=model["model"],
                    image=image_url,
                    prompt=None if custom_multi_shot else multi_shot["prompt"],
                    negative_prompt=None if custom_multi_shot else multi_shot["negative_prompt"],
                    mode=mode,
                    duration=str(duration),
                    sound="on" if generate_audio else "off",
                    multi_shot=True if shot_type else None,
                    multi_prompt=multi_prompt_list,
                    shot_type=shot_type,
                ),
            )
            poll_path = f"/proxy/kling/v1/videos/image2video/{response.data.task_id}"
        else:
            response = await sync_op(
                cls,
                ApiEndpoint(path="/proxy/kling/v1/videos/text2video", method="POST"),
                response_model=TaskStatusResponse,
                data=TextToVideoWithAudioRequest(
                    model_name=model["model"],
                    aspect_ratio=model["aspect_ratio"],
                    prompt=None if custom_multi_shot else multi_shot["prompt"],
                    negative_prompt=None if custom_multi_shot else multi_shot["negative_prompt"],
                    mode=mode,
                    duration=str(duration),
                    sound="on" if generate_audio else "off",
                    multi_shot=True if shot_type else None,
                    multi_prompt=multi_prompt_list,
                    shot_type=shot_type,
                ),
            )
            poll_path = f"/proxy/kling/v1/videos/text2video/{response.data.task_id}"

        if response.code:
            raise RuntimeError(
                f"Kling request failed. Code: {response.code}, Message: {response.message}, Data: {response.data}"
            )
        final_response = await poll_op(
            cls,
            ApiEndpoint(path=poll_path),
            response_model=TaskStatusResponse,
            status_extractor=lambda r: (r.data.task_status if r.data else None),
        )
        return IO.NodeOutput(await download_url_to_video_output(final_response.data.task_result.videos[0].url))


class KlingFirstLastFrameNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingFirstLastFrameNode",
            display_name="Kling 3.0 First-Last-Frame to Video",
            category="partner/video/Kling",
            description="Generate videos with Kling V3 using first and last frames.",
            inputs=[
                IO.String.Input("prompt", multiline=True, default=""),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=3,
                    max=15,
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Image.Input("first_frame"),
                IO.Image.Input("end_frame"),
                IO.Boolean.Input("generate_audio", default=True),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "kling-v3",
                            [
                                IO.Combo.Input("resolution", options=["4k", "1080p", "720p"], default="1080p"),
                            ],
                        ),
                    ],
                    tooltip="Model and generation settings.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(
                    widgets=["model.resolution", "generate_audio", "duration"],
                ),
                expr="""
                (
                  $rates := {
                    "4k": {"off": 0.42, "on": 0.42},
                    "1080p": {"off": 0.112, "on": 0.168},
                    "720p": {"off": 0.084, "on": 0.126}
                  };
                  $res := $lookup(widgets, "model.resolution");
                  $audio := widgets.generate_audio ? "on" : "off";
                  $rate := $lookup($lookup($rates, $res), $audio);
                  {"type":"usd","usd": $rate * widgets.duration}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        duration: int,
        first_frame: Input.Image,
        end_frame: Input.Image,
        generate_audio: bool,
        model: dict,
        seed: int,
    ) -> IO.NodeOutput:
        _ = seed
        validate_string(prompt, min_length=1, max_length=2500)
        validate_image_dimensions(first_frame, min_width=300, min_height=300)
        validate_image_aspect_ratio(first_frame, (1, 2.5), (2.5, 1))
        validate_image_dimensions(end_frame, min_width=300, min_height=300)
        validate_image_aspect_ratio(end_frame, (1, 2.5), (2.5, 1))
        image_url = await upload_image_to_comfyapi(cls, first_frame, wait_label="Uploading first frame")
        image_tail_url = await upload_image_to_comfyapi(cls, end_frame, wait_label="Uploading end frame")
        if model["resolution"] == "4k":
            mode = "4k"
        elif model["resolution"] == "1080p":
            mode = "pro"
        else:
            mode = "std"
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/kling/v1/videos/image2video", method="POST"),
            response_model=TaskStatusResponse,
            data=ImageToVideoWithAudioRequest(
                model_name=model["model"],
                image=image_url,
                image_tail=image_tail_url,
                prompt=prompt,
                mode=mode,
                duration=str(duration),
                sound="on" if generate_audio else "off",
            ),
        )
        if response.code:
            raise RuntimeError(
                f"Kling request failed. Code: {response.code}, Message: {response.message}, Data: {response.data}"
            )
        final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/kling/v1/videos/image2video/{response.data.task_id}"),
            response_model=TaskStatusResponse,
            status_extractor=lambda r: (r.data.task_status if r.data else None),
        )
        return IO.NodeOutput(await download_url_to_video_output(final_response.data.task_result.videos[0].url))


class KlingAvatarNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingAvatarNode",
            display_name="Kling Avatar 2.0",
            category="partner/video/Kling",
            description="Generate broadcast-style digital human videos from a single photo and an audio file.",
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="Avatar reference image. "
                    "Width and height must be at least 300px. Aspect ratio must be between 1:2.5 and 2.5:1.",
                ),
                IO.Audio.Input(
                    "sound_file",
                    tooltip="Audio input. Must be between 2 and 300 seconds in duration.",
                ),
                IO.Combo.Input("mode", options=["std", "pro"]),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    optional=True,
                    tooltip="Optional prompt to define avatar actions, emotions, and camera movements.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["mode"]),
                expr="""
                (
                  $prices := {"std": 0.056, "pro": 0.112};
                  {"type":"usd","usd": $lookup($prices, widgets.mode), "format":{"suffix":"/second"}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        sound_file: Input.Audio,
        mode: str,
        seed: int,
        prompt: str = "",
    ) -> IO.NodeOutput:
        validate_image_dimensions(image, min_width=300, min_height=300)
        validate_image_aspect_ratio(image, (1, 2.5), (2.5, 1))
        validate_audio_duration(sound_file, min_duration=2, max_duration=300)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/kling/v1/videos/avatar/image2video", method="POST"),
            response_model=TaskStatusResponse,
            data=KlingAvatarRequest(
                image=await upload_image_to_comfyapi(cls, image),
                sound_file=await upload_audio_to_comfyapi(
                    cls, sound_file, container_format="mp3", codec_name="libmp3lame", mime_type="audio/mpeg"
                ),
                prompt=prompt or None,
                mode=mode,
            ),
        )
        if response.code:
            raise RuntimeError(
                f"Kling request failed. Code: {response.code}, Message: {response.message}, Data: {response.data}"
            )
        final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/kling/v1/videos/avatar/image2video/{response.data.task_id}"),
            response_model=TaskStatusResponse,
            status_extractor=lambda r: (r.data.task_status if r.data else None),
            max_poll_attempts=800,
        )
        return IO.NodeOutput(await download_url_to_video_output(final_response.data.task_result.videos[0].url))


class KlingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            KlingCameraControls,
            KlingTextToVideoNode,
            KlingImage2VideoNode,
            KlingCameraControlI2VNode,
            KlingCameraControlT2VNode,
            KlingStartEndFrameNode,
            KlingVideoExtendNode,
            KlingLipSyncAudioToVideoNode,
            KlingLipSyncTextToVideoNode,
            KlingVirtualTryOnNode,
            KlingImageGenerationNode,
            KlingSingleImageVideoEffectNode,
            KlingDualCharacterVideoEffectNode,
            OmniProTextToVideoNode,
            OmniProFirstLastFrameNode,
            OmniProImageToVideoNode,
            OmniProVideoToVideoNode,
            OmniProEditVideoNode,
            OmniProImageNode,
            TextToVideoWithAudio,
            ImageToVideoWithAudio,
            MotionControl,
            KlingVideoNode,
            KlingFirstLastFrameNode,
            KlingAvatarNode,
        ]


async def comfy_entrypoint() -> KlingExtension:
    return KlingExtension()
