from typing import Optional

import torch
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension
from comfy_api_nodes.apis.minimax import (
    MinimaxFileRetrieveResponse,
    MiniMaxModel,
    MinimaxTaskResultResponse,
    MinimaxVideoGenerationRequest,
    MinimaxVideoGenerationResponse,
    SubjectReferenceItem,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    poll_op,
    sync_op,
    upload_images_to_comfyapi,
    validate_string,
)

I2V_AVERAGE_DURATION = 114
T2V_AVERAGE_DURATION = 234


async def _generate_mm_video(
    cls: type[IO.ComfyNode],
    *,
    prompt_text: str,
    seed: int,
    model: str,
    image: Optional[torch.Tensor] = None,  # used for ImageToVideo
    subject: Optional[torch.Tensor] = None,  # used for SubjectToVideo
    average_duration: Optional[int] = None,
) -> IO.NodeOutput:
    if image is None:
        validate_string(prompt_text, field_name="prompt_text")
    image_url = None
    if image is not None:
        image_url = (await upload_images_to_comfyapi(cls, image, max_images=1))[0]

    # TODO: figure out how to deal with subject properly, API returns invalid params when using S2V-01 model
    subject_reference = None
    if subject is not None:
        subject_url = (await upload_images_to_comfyapi(cls, subject, max_images=1))[0]
        subject_reference = [SubjectReferenceItem(image=subject_url)]

    response = await sync_op(
        cls,
        ApiEndpoint(path="/proxy/minimax/video_generation", method="POST"),
        response_model=MinimaxVideoGenerationResponse,
        data=MinimaxVideoGenerationRequest(
            model=MiniMaxModel(model),
            prompt=prompt_text,
            callback_url=None,
            first_frame_image=image_url,
            subject_reference=subject_reference,
            prompt_optimizer=None,
        ),
    )

    task_id = response.task_id
    if not task_id:
        raise Exception(f"MiniMax generation failed: {response.base_resp}")

    task_result = await poll_op(
        cls,
        ApiEndpoint(path="/proxy/minimax/query/video_generation", query_params={"task_id": task_id}),
        response_model=MinimaxTaskResultResponse,
        status_extractor=lambda x: x.status.value,
        estimated_duration=average_duration,
    )

    file_id = task_result.file_id
    if file_id is None:
        raise Exception("Request was not successful. Missing file ID.")
    file_result = await sync_op(
        cls,
        ApiEndpoint(path="/proxy/minimax/files/retrieve", query_params={"file_id": int(file_id)}),
        response_model=MinimaxFileRetrieveResponse,
    )

    file_url = file_result.file.download_url
    if file_url is None:
        raise Exception(f"No video was found in the response. Full response: {file_result.model_dump()}")
    if file_result.file.backup_download_url:
        try:
            return IO.NodeOutput(await download_url_to_video_output(file_url, timeout=10, max_retries=2))
        except Exception:  # if we have a second URL to retrieve the result, try again using that one
            return IO.NodeOutput(
                await download_url_to_video_output(file_result.file.backup_download_url, max_retries=3)
            )
    return IO.NodeOutput(await download_url_to_video_output(file_url))


class MinimaxTextToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="MinimaxTextToVideoNode",
            display_name="MiniMax Text to Video",
            category="partner/video/MiniMax",
            description="Generates videos synchronously based on a prompt, and optional parameters.",
            inputs=[
                IO.String.Input(
                    "prompt_text",
                    multiline=True,
                    default="",
                    tooltip="Text prompt to guide the video generation",
                ),
                IO.Combo.Input(
                    "model",
                    options=["T2V-01", "T2V-01-Director"],
                    default="T2V-01",
                    tooltip="Model to use for video generation",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                    optional=True,
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
                expr="""{"type":"usd","usd":0.43}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt_text: str,
        model: str = "T2V-01",
        seed: int = 0,
    ) -> IO.NodeOutput:
        return await _generate_mm_video(
            cls,
            prompt_text=prompt_text,
            seed=seed,
            model=model,
            image=None,
            subject=None,
            average_duration=T2V_AVERAGE_DURATION,
        )


class MinimaxImageToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="MinimaxImageToVideoNode",
            display_name="MiniMax Image to Video",
            category="partner/video/MiniMax",
            description="Generates videos synchronously based on an image and prompt, and optional parameters.",
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="Image to use as first frame of video generation",
                ),
                IO.String.Input(
                    "prompt_text",
                    multiline=True,
                    default="",
                    tooltip="Text prompt to guide the video generation",
                ),
                IO.Combo.Input(
                    "model",
                    options=["I2V-01-Director", "I2V-01", "I2V-01-live"],
                    default="I2V-01",
                    tooltip="Model to use for video generation",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                    optional=True,
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
                expr="""{"type":"usd","usd":0.43}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        prompt_text: str,
        model: str = "I2V-01",
        seed: int = 0,
    ) -> IO.NodeOutput:
        return await _generate_mm_video(
            cls,
            prompt_text=prompt_text,
            seed=seed,
            model=model,
            image=image,
            subject=None,
            average_duration=I2V_AVERAGE_DURATION,
        )


class MinimaxSubjectToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="MinimaxSubjectToVideoNode",
            display_name="MiniMax Subject to Video",
            category="partner/video/MiniMax",
            description="Generates videos synchronously based on an image and prompt, and optional parameters.",
            inputs=[
                IO.Image.Input(
                    "subject",
                    tooltip="Image of subject to reference for video generation",
                ),
                IO.String.Input(
                    "prompt_text",
                    multiline=True,
                    default="",
                    tooltip="Text prompt to guide the video generation",
                ),
                IO.Combo.Input(
                    "model",
                    options=["S2V-01"],
                    default="S2V-01",
                    tooltip="Model to use for video generation",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                    optional=True,
                ),
            ],
            outputs=[IO.Video.Output()],
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
        subject: torch.Tensor,
        prompt_text: str,
        model: str = "S2V-01",
        seed: int = 0,
    ) -> IO.NodeOutput:
        return await _generate_mm_video(
            cls,
            prompt_text=prompt_text,
            seed=seed,
            model=model,
            image=None,
            subject=subject,
            average_duration=T2V_AVERAGE_DURATION,
        )


class MinimaxHailuoVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="MinimaxHailuoVideoNode",
            display_name="MiniMax Hailuo Video",
            category="partner/video/MiniMax",
            description="Generates videos from prompt, with optional start frame using the new MiniMax Hailuo-02 model.",
            inputs=[
                IO.String.Input(
                    "prompt_text",
                    multiline=True,
                    default="",
                    tooltip="Text prompt to guide the video generation.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                    optional=True,
                ),
                IO.Image.Input(
                    "first_frame_image",
                    tooltip="Optional image to use as the first frame to generate a video.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "prompt_optimizer",
                    default=True,
                    tooltip="Optimize prompt to improve generation quality when needed.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "duration",
                    options=[6, 10],
                    default=6,
                    tooltip="The length of the output video in seconds.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["768P", "1080P"],
                    default="768P",
                    tooltip="The dimensions of the video display. 1080p is 1920x1080, 768p is 1366x768.",
                    optional=True,
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
                depends_on=IO.PriceBadgeDepends(widgets=["resolution", "duration"]),
                expr="""
                (
                  $prices := {
                    "768p": {"6": 0.28, "10": 0.56},
                    "1080p": {"6": 0.49}
                  };
                  $resPrices := $lookup($prices, $lowercase(widgets.resolution));
                  $price := $lookup($resPrices, $string(widgets.duration));
                  {"type":"usd","usd": $price ? $price : 0.43}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt_text: str,
        seed: int = 0,
        first_frame_image: Optional[torch.Tensor] = None,  # used for ImageToVideo
        prompt_optimizer: bool = True,
        duration: int = 6,
        resolution: str = "768P",
        model: str = "MiniMax-Hailuo-02",
    ) -> IO.NodeOutput:
        if first_frame_image is None:
            validate_string(prompt_text, field_name="prompt_text")

        if model == "MiniMax-Hailuo-02" and resolution.upper() == "1080P" and duration != 6:
            raise Exception(
                "When model is MiniMax-Hailuo-02 and resolution is 1080P, duration is limited to 6 seconds."
            )

        # upload image, if passed in
        image_url = None
        if first_frame_image is not None:
            image_url = (await upload_images_to_comfyapi(cls, first_frame_image, max_images=1))[0]

        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/minimax/video_generation", method="POST"),
            response_model=MinimaxVideoGenerationResponse,
            data=MinimaxVideoGenerationRequest(
                model=MiniMaxModel(model),
                prompt=prompt_text,
                callback_url=None,
                first_frame_image=image_url,
                prompt_optimizer=prompt_optimizer,
                duration=duration,
                resolution=resolution,
            ),
        )

        task_id = response.task_id
        if not task_id:
            raise Exception(f"MiniMax generation failed: {response.base_resp}")

        average_duration = 120 if resolution == "768P" else 240
        task_result = await poll_op(
            cls,
            ApiEndpoint(path="/proxy/minimax/query/video_generation", query_params={"task_id": task_id}),
            response_model=MinimaxTaskResultResponse,
            status_extractor=lambda x: x.status.value,
            estimated_duration=average_duration,
        )

        file_id = task_result.file_id
        if file_id is None:
            raise Exception("Request was not successful. Missing file ID.")
        file_result = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/minimax/files/retrieve", query_params={"file_id": int(file_id)}),
            response_model=MinimaxFileRetrieveResponse,
        )

        file_url = file_result.file.download_url
        if file_url is None:
            raise Exception(f"No video was found in the response. Full response: {file_result.model_dump()}")

        if file_result.file.backup_download_url:
            try:
                return IO.NodeOutput(await download_url_to_video_output(file_url, timeout=10, max_retries=2))
            except Exception:  # if we have a second URL to retrieve the result, try again using that one
                return IO.NodeOutput(
                    await download_url_to_video_output(file_result.file.backup_download_url, max_retries=3)
                )
        return IO.NodeOutput(await download_url_to_video_output(file_url))


class MinimaxExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            MinimaxTextToVideoNode,
            MinimaxImageToVideoNode,
            # MinimaxSubjectToVideoNode,
            MinimaxHailuoVideoNode,
        ]


async def comfy_entrypoint() -> MinimaxExtension:
    return MinimaxExtension()
