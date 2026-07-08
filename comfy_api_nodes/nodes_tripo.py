from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input, Types
from comfy_api_nodes.apis.tripo import (
    TripoAnimateRetargetRequest,
    TripoAnimateRigRequest,
    TripoConvertModelRequest,
    TripoFileEmptyReference,
    TripoFileReference,
    TripoImageToModelRequest,
    TripoImportModelRequest,
    TripoModelVersion,
    TripoMultiviewToModelRequest,
    TripoOrientation,
    TripoP1ImageToModelRequest,
    TripoP1MultiviewToModelRequest,
    TripoP1TextToModelRequest,
    TripoRefineModelRequest,
    TripoStyle,
    TripoTaskResponse,
    TripoTaskStatus,
    TripoTaskType,
    TripoTextToModelRequest,
    TripoTextureModelRequest,
    TripoTexturePrompt,
    TripoUrlReference,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_file_3d,
    poll_op,
    sync_op,
    upload_3d_model_to_comfyapi,
    upload_images_to_comfyapi,
)


def get_model_url_from_response(response: TripoTaskResponse) -> str:
    if response.data is not None:
        for key in ["pbr_model", "model", "base_model"]:
            if getattr(response.data.output, key, None) is not None:
                return getattr(response.data.output, key)
    raise RuntimeError(f"Failed to get model url from response: {response}")


async def poll_until_finished(
    node_cls: type[IO.ComfyNode],
    response: TripoTaskResponse,
    average_duration: int | None = None,
) -> IO.NodeOutput:
    """Polls the Tripo API endpoint until the task reaches a terminal state, then returns the response."""
    if response.code != 0:
        raise RuntimeError(f"Failed to generate mesh: {response.error}")
    task_id = response.data.task_id
    response_poll = await poll_op(
        node_cls,
        poll_endpoint=ApiEndpoint(path=f"/proxy/tripo/v2/openapi/task/{task_id}"),
        response_model=TripoTaskResponse,
        completed_statuses=[TripoTaskStatus.SUCCESS],
        failed_statuses=[
            TripoTaskStatus.FAILED,
            TripoTaskStatus.CANCELLED,
            TripoTaskStatus.UNKNOWN,
            TripoTaskStatus.BANNED,
            TripoTaskStatus.EXPIRED,
        ],
        status_extractor=lambda x: x.data.status,
        progress_extractor=lambda x: x.data.progress,
        price_extractor=lambda x: x.data.consumed_credit * 0.01 if x.data.consumed_credit else None,
        estimated_duration=average_duration,
    )
    if response_poll.data.status == TripoTaskStatus.SUCCESS:
        url = get_model_url_from_response(response_poll)
        file_glb = await download_url_to_file_3d(url, "glb", task_id=task_id)
        return IO.NodeOutput(f"{task_id}.glb", task_id, file_glb)
    raise RuntimeError(f"Failed to generate mesh: {response_poll}")


class TripoTextToModelNode(IO.ComfyNode):
    """
    Generates 3D models synchronously based on a text prompt using Tripo's API.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoTextToModelNode",
            display_name="Tripo: Text to Model",
            category="partner/3d/Tripo",
            inputs=[
                IO.String.Input("prompt", multiline=True),
                IO.String.Input("negative_prompt", multiline=True, optional=True),
                IO.Combo.Input(
                    "model_version", options=TripoModelVersion, default=TripoModelVersion.v2_5_20250123, optional=True
                ),
                IO.Combo.Input("style", options=TripoStyle, default="None", optional=True),
                IO.Boolean.Input("texture", default=True, optional=True),
                IO.Boolean.Input("pbr", default=True, optional=True),
                IO.Int.Input("image_seed", default=42, optional=True, advanced=True),
                IO.Int.Input("model_seed", default=42, optional=True, advanced=True),
                IO.Int.Input("texture_seed", default=42, optional=True, advanced=True),
                IO.Combo.Input(
                    "texture_quality",
                    default="standard",
                    options=["standard", "detailed"],
                    optional=True,
                    advanced=True,
                ),
                IO.Int.Input("face_limit", default=-1, min=-1, max=2000000, optional=True, advanced=True),
                IO.Boolean.Input("quad", default=False, optional=True, advanced=True),
                IO.Combo.Input(
                    "geometry_quality",
                    default="standard",
                    options=["standard", "detailed"],
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MODEL_TASK_ID").Output(display_name="model task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(
                    widgets=[
                        "model_version",
                        "texture",
                        "pbr",
                        "quad",
                        "texture_quality",
                        "geometry_quality",
                    ],
                ),
                expr="""
                (
                  $isV14 := $contains(widgets.model_version,"v1.4");
                  $isV3OrLater := $contains(widgets.model_version,"v3.");
                  $withTexture := widgets.texture or widgets.pbr;
                  $isHdTexture := (widgets.texture_quality = "detailed");
                  $isDetailedGeometry := (widgets.geometry_quality = "detailed");
                  $credits := $isV14 ? 20 : (
                    ($withTexture ? 20 : 10)
                    + (widgets.quad ? 5 : 0)
                    + ($isHdTexture ? 10 : 0)
                    + (($isDetailedGeometry and $isV3OrLater) ? 20 : 0)
                  );
                  {"type":"usd","usd": $round($credits * 0.01, 2), "format": {"approximate": true}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        negative_prompt: str | None = None,
        model_version=None,
        style: str | None = None,
        texture: bool | None = None,
        pbr: bool | None = None,
        image_seed: int | None = None,
        model_seed: int | None = None,
        texture_seed: int | None = None,
        texture_quality: str | None = None,
        geometry_quality: str | None = None,
        face_limit: int | None = None,
        quad: bool | None = None,
    ) -> IO.NodeOutput:
        style_enum = None if style == "None" else style
        if not prompt:
            raise RuntimeError("Prompt is required")
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoTextToModelRequest(
                type=TripoTaskType.TEXT_TO_MODEL,
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                model_version=model_version,
                style=style_enum,
                texture=texture,
                pbr=pbr,
                image_seed=image_seed,
                model_seed=model_seed,
                texture_seed=texture_seed,
                texture_quality=texture_quality,
                face_limit=face_limit if face_limit != -1 else None,
                geometry_quality=geometry_quality,
                auto_size=True,
                quad=quad,
            ),
        )
        return await poll_until_finished(cls, response, average_duration=80)


class TripoImageToModelNode(IO.ComfyNode):
    """
    Generates 3D models synchronously based on a single image using Tripo's API.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoImageToModelNode",
            display_name="Tripo: Image to Model",
            category="partner/3d/Tripo",
            inputs=[
                IO.Image.Input("image"),
                IO.Combo.Input(
                    "model_version",
                    options=TripoModelVersion,
                    tooltip="The model version to use for generation",
                    optional=True,
                ),
                IO.Combo.Input("style", options=TripoStyle, default="None", optional=True),
                IO.Boolean.Input("texture", default=True, optional=True),
                IO.Boolean.Input("pbr", default=True, optional=True),
                IO.Int.Input("model_seed", default=42, optional=True, advanced=True),
                IO.Combo.Input(
                    "orientation",
                    options=TripoOrientation,
                    default=TripoOrientation.DEFAULT,
                    optional=True,
                    advanced=True,
                ),
                IO.Int.Input("texture_seed", default=42, optional=True, advanced=True),
                IO.Combo.Input(
                    "texture_quality",
                    default="standard",
                    options=["standard", "detailed"],
                    optional=True,
                    advanced=True,
                ),
                IO.Combo.Input(
                    "texture_alignment",
                    default="original_image",
                    options=["original_image", "geometry"],
                    optional=True,
                    advanced=True,
                ),
                IO.Int.Input("face_limit", default=-1, min=-1, max=500000, optional=True, advanced=True),
                IO.Boolean.Input("quad", default=False, optional=True, advanced=True),
                IO.Combo.Input(
                    "geometry_quality",
                    default="standard",
                    options=["standard", "detailed"],
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MODEL_TASK_ID").Output(display_name="model task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(
                    widgets=[
                        "model_version",
                        "texture",
                        "pbr",
                        "quad",
                        "texture_quality",
                        "geometry_quality",
                    ],
                ),
                expr="""
                (
                  $isV14 := $contains(widgets.model_version,"v1.4");
                  $isV3OrLater := $contains(widgets.model_version,"v3.");
                  $withTexture := widgets.texture or widgets.pbr;
                  $isHdTexture := (widgets.texture_quality = "detailed");
                  $isDetailedGeometry := (widgets.geometry_quality = "detailed");
                  $credits := $isV14 ? 30 : (
                    ($withTexture ? 30 : 20)
                    + (widgets.quad ? 5 : 0)
                    + ($isHdTexture ? 10 : 0)
                    + (($isDetailedGeometry and $isV3OrLater) ? 20 : 0)
                  );
                  {"type":"usd","usd": $round($credits * 0.01, 2), "format": {"approximate": true}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        model_version: str | None = None,
        style: str | None = None,
        texture: bool | None = None,
        pbr: bool | None = None,
        model_seed: int | None = None,
        orientation=None,
        texture_seed: int | None = None,
        texture_quality: str | None = None,
        geometry_quality: str | None = None,
        texture_alignment: str | None = None,
        face_limit: int | None = None,
        quad: bool | None = None,
    ) -> IO.NodeOutput:
        style_enum = None if style == "None" else style
        if image is None:
            raise RuntimeError("Image is required")
        tripo_file = TripoFileReference(
            root=TripoUrlReference(
                url=(await upload_images_to_comfyapi(cls, image, max_images=1))[0],
                type="jpeg",
            )
        )
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoImageToModelRequest(
                type=TripoTaskType.IMAGE_TO_MODEL,
                file=tripo_file,
                model_version=model_version,
                style=style_enum,
                texture=texture,
                pbr=pbr,
                model_seed=model_seed,
                orientation=orientation,
                geometry_quality=geometry_quality,
                texture_alignment=texture_alignment,
                texture_seed=texture_seed,
                texture_quality=texture_quality,
                face_limit=face_limit if face_limit != -1 else None,
                auto_size=True,
                quad=quad,
            ),
        )
        return await poll_until_finished(cls, response, average_duration=80)


class TripoMultiviewToModelNode(IO.ComfyNode):
    """
    Generates 3D models synchronously based on up to four images (front, left, back, right) using Tripo's API.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoMultiviewToModelNode",
            display_name="Tripo: Multiview to Model",
            category="partner/3d/Tripo",
            inputs=[
                IO.Image.Input("image"),
                IO.Image.Input("image_left", optional=True),
                IO.Image.Input("image_back", optional=True),
                IO.Image.Input("image_right", optional=True),
                IO.Combo.Input(
                    "model_version",
                    options=TripoModelVersion,
                    optional=True,
                    tooltip="The model version to use for generation",
                ),
                IO.Combo.Input(
                    "orientation",
                    options=TripoOrientation,
                    default=TripoOrientation.DEFAULT,
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input("texture", default=True, optional=True),
                IO.Boolean.Input("pbr", default=True, optional=True),
                IO.Int.Input("model_seed", default=42, optional=True, advanced=True),
                IO.Int.Input("texture_seed", default=42, optional=True, advanced=True),
                IO.Combo.Input(
                    "texture_quality",
                    default="standard",
                    options=["standard", "detailed"],
                    optional=True,
                    advanced=True,
                ),
                IO.Combo.Input(
                    "texture_alignment",
                    default="original_image",
                    options=["original_image", "geometry"],
                    optional=True,
                    advanced=True,
                ),
                IO.Int.Input("face_limit", default=-1, min=-1, max=500000, optional=True, advanced=True),
                IO.Boolean.Input(
                    "quad",
                    default=False,
                    optional=True,
                    advanced=True,
                    tooltip="This parameter is deprecated and does nothing.",
                ),
                IO.Combo.Input(
                    "geometry_quality",
                    default="standard",
                    options=["standard", "detailed"],
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MODEL_TASK_ID").Output(display_name="model task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(
                    widgets=[
                        "model_version",
                        "texture",
                        "pbr",
                        "texture_quality",
                        "geometry_quality",
                    ],
                ),
                expr="""
                (
                  $isV14 := $contains(widgets.model_version,"v1.4");
                  $isV3OrLater := $contains(widgets.model_version,"v3.");
                  $withTexture := widgets.texture or widgets.pbr;
                  $isHdTexture := (widgets.texture_quality = "detailed");
                  $isDetailedGeometry := (widgets.geometry_quality = "detailed");
                  $credits := $isV14 ? 30 : (
                    ($withTexture ? 30 : 20)
                    + ($isHdTexture ? 10 : 0)
                    + (($isDetailedGeometry and $isV3OrLater) ? 20 : 0)
                  );
                  {"type":"usd","usd": $round($credits * 0.01, 2), "format": {"approximate": true}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        image_left: Input.Image | None = None,
        image_back: Input.Image | None = None,
        image_right: Input.Image | None = None,
        model_version: str | None = None,
        orientation: str | None = None,
        texture: bool | None = None,
        pbr: bool | None = None,
        model_seed: int | None = None,
        texture_seed: int | None = None,
        texture_quality: str | None = None,
        geometry_quality: str | None = None,
        texture_alignment: str | None = None,
        face_limit: int | None = None,
        quad: bool | None = None,
    ) -> IO.NodeOutput:
        if image is None:
            raise RuntimeError("front image for multiview is required")
        images = []
        image_dict = {"image": image, "image_left": image_left, "image_back": image_back, "image_right": image_right}
        if image_left is None and image_back is None and image_right is None:
            raise RuntimeError("At least one of left, back, or right image must be provided for multiview")
        for image_name in ["image", "image_left", "image_back", "image_right"]:
            image_ = image_dict[image_name]
            if image_ is not None:
                images.append(
                    TripoFileReference(
                        root=TripoUrlReference(
                            url=(await upload_images_to_comfyapi(cls, image_, max_images=1))[0], type="jpeg"
                        )
                    )
                )
            else:
                images.append(TripoFileEmptyReference())
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoMultiviewToModelRequest(
                type=TripoTaskType.MULTIVIEW_TO_MODEL,
                files=images,
                model_version=model_version,
                orientation=orientation,
                texture=texture,
                pbr=pbr,
                model_seed=model_seed,
                texture_seed=texture_seed,
                texture_quality=texture_quality,
                geometry_quality=geometry_quality,
                texture_alignment=texture_alignment,
                face_limit=face_limit if face_limit != -1 else None,
                quad=None,
            ),
        )
        return await poll_until_finished(cls, response, average_duration=80)


class TripoTextureNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoTextureNode",
            display_name="Tripo: Texture model",
            category="partner/3d/Tripo",
            inputs=[
                IO.Custom("MODEL_TASK_ID").Input("model_task_id"),
                IO.Boolean.Input("texture", default=True, optional=True),
                IO.Boolean.Input("pbr", default=True, optional=True),
                IO.Int.Input("texture_seed", default=42, optional=True, advanced=True),
                IO.Combo.Input(
                    "texture_quality",
                    default="standard",
                    options=["standard", "detailed"],
                    optional=True,
                    advanced=True,
                ),
                IO.Combo.Input(
                    "texture_alignment",
                    default="original_image",
                    options=["original_image", "geometry"],
                    optional=True,
                    advanced=True,
                ),
                IO.String.Input(
                    "texture_prompt",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Optional text guidance for texturing. Required in practice for imported "
                    "models (Tripo: Import Model), which carry no source image to infer colors from.",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MODEL_TASK_ID").Output(display_name="model task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["texture_quality"]),
                expr="""
                (
                  $tq := widgets.texture_quality;
                  {"type":"usd","usd": ($contains($tq,"detailed") ? 0.2 : 0.1), "format": {"approximate": true}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model_task_id,
        texture: bool | None = None,
        pbr: bool | None = None,
        texture_seed: int | None = None,
        texture_quality: str | None = None,
        texture_alignment: str | None = None,
        texture_prompt: str = "",
    ) -> IO.NodeOutput:
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoTextureModelRequest(
                original_model_task_id=model_task_id,
                texture=texture,
                pbr=pbr,
                texture_seed=texture_seed,
                texture_quality=texture_quality,
                texture_alignment=texture_alignment,
                texture_prompt=TripoTexturePrompt(text=texture_prompt.strip()) if texture_prompt.strip() else None,
            ),
        )
        return await poll_until_finished(cls, response, average_duration=80)


class TripoRefineNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoRefineNode",
            display_name="Tripo: Refine Draft model",
            category="partner/3d/Tripo",
            description="Refine a draft model created by v1.4 Tripo models only.",
            inputs=[
                IO.Custom("MODEL_TASK_ID").Input("model_task_id", tooltip="Must be a v1.4 Tripo model"),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MODEL_TASK_ID").Output(display_name="model task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.3, "format": {"approximate": true}}""",
            ),
        )

    @classmethod
    async def execute(cls, model_task_id) -> IO.NodeOutput:
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoRefineModelRequest(draft_model_task_id=model_task_id),
        )
        return await poll_until_finished(cls, response, average_duration=240)


class TripoRigNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoRigNode",
            display_name="Tripo: Rig model",
            category="partner/3d/Tripo",
            inputs=[IO.Custom("MODEL_TASK_ID").Input("original_model_task_id")],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("RIG_TASK_ID").Output(display_name="rig task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.25, "format": {"approximate": true}}""",
            ),
        )

    @classmethod
    async def execute(cls, original_model_task_id) -> IO.NodeOutput:
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoAnimateRigRequest(original_model_task_id=original_model_task_id, out_format="glb", spec="tripo"),
        )
        return await poll_until_finished(cls, response, average_duration=180)


class TripoRetargetNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoRetargetNode",
            display_name="Tripo: Retarget rigged model",
            category="partner/3d/Tripo",
            inputs=[
                IO.Custom("RIG_TASK_ID").Input("original_model_task_id"),
                IO.Combo.Input(
                    "animation",
                    options=[
                        "preset:idle",
                        "preset:walk",
                        "preset:run",
                        "preset:dive",
                        "preset:climb",
                        "preset:jump",
                        "preset:slash",
                        "preset:shoot",
                        "preset:hurt",
                        "preset:fall",
                        "preset:turn",
                        "preset:quadruped:walk",
                        "preset:hexapod:walk",
                        "preset:octopod:walk",
                        "preset:serpentine:march",
                        "preset:aquatic:march",
                    ],
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("RETARGET_TASK_ID").Output(display_name="retarget task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.1, "format": {"approximate": true}}""",
            ),
        )

    @classmethod
    async def execute(cls, original_model_task_id, animation: str) -> IO.NodeOutput:
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoAnimateRetargetRequest(
                original_model_task_id=original_model_task_id,
                animation=animation,
                out_format="glb",
                bake_animation=True,
            ),
        )
        return await poll_until_finished(cls, response, average_duration=30)


class TripoConversionNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoConversionNode",
            display_name="Tripo: Convert model",
            category="partner/3d/Tripo",
            inputs=[
                IO.Custom("MODEL_TASK_ID,RIG_TASK_ID,RETARGET_TASK_ID").Input("original_model_task_id"),
                IO.Combo.Input("format", options=["GLTF", "USDZ", "FBX", "OBJ", "STL", "3MF"]),
                IO.Boolean.Input("quad", default=False, optional=True, advanced=True),
                IO.Int.Input(
                    "face_limit",
                    default=-1,
                    min=-1,
                    max=2000000,
                    optional=True,
                    advanced=True,
                ),
                IO.Int.Input(
                    "texture_size",
                    default=4096,
                    min=128,
                    max=4096,
                    optional=True,
                    advanced=True,
                ),
                IO.Combo.Input(
                    "texture_format",
                    options=["BMP", "DPX", "HDR", "JPEG", "OPEN_EXR", "PNG", "TARGA", "TIFF", "WEBP"],
                    default="JPEG",
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input("force_symmetry", default=False, optional=True, advanced=True),
                IO.Boolean.Input("flatten_bottom", default=False, optional=True, advanced=True),
                IO.Float.Input(
                    "flatten_bottom_threshold",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input("pivot_to_center_bottom", default=False, optional=True, advanced=True),
                IO.Float.Input(
                    "scale_factor",
                    default=1.0,
                    min=0.0,
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input("with_animation", default=False, optional=True, advanced=True),
                IO.Boolean.Input("pack_uv", default=False, optional=True, advanced=True),
                IO.Boolean.Input("bake", default=False, optional=True, advanced=True),
                IO.String.Input("part_names", default="", optional=True, advanced=True),  # comma-separated list
                IO.Combo.Input(
                    "fbx_preset",
                    options=["blender", "mixamo", "3dsmax"],
                    default="blender",
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input("export_vertex_colors", default=False, optional=True, advanced=True),
                IO.Combo.Input(
                    "export_orientation",
                    options=["align_image", "default"],
                    default="default",
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input("animate_in_place", default=False, optional=True, advanced=True),
            ],
            outputs=[],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(
                    widgets=[
                        "quad",
                        "face_limit",
                        "texture_size",
                        "texture_format",
                        "flatten_bottom",
                        "flatten_bottom_threshold",
                        "pivot_to_center_bottom",
                        "scale_factor",
                    ],
                ),
                expr="""
                (
                    $face := (widgets.face_limit != null) ? widgets.face_limit : -1;
                    $texSize := (widgets.texture_size != null) ? widgets.texture_size : 4096;
                    $flatThresh := (widgets.flatten_bottom_threshold != null) ? widgets.flatten_bottom_threshold : 0;
                    $scale := (widgets.scale_factor != null) ? widgets.scale_factor : 1;
                    $texFmt := (widgets.texture_format != "" ? widgets.texture_format : "jpeg");
                    $advanced :=
                      widgets.quad or
                      widgets.flatten_bottom or
                      widgets.pivot_to_center_bottom or
                      ($face != -1) or
                      ($texSize != 4096) or
                      ($flatThresh != 0) or
                      ($scale != 1) or
                      ($texFmt != "jpeg");
                    {"type":"usd","usd": ($advanced ? 0.1 : 0.05), "format": {"approximate": true}}
                )
                """,
            ),
        )

    @classmethod
    def validate_inputs(cls, input_types):
        # The min and max of input1 and input2 are still validated because
        # we didn't take `input1` or `input2` as arguments
        if input_types["original_model_task_id"] not in ("MODEL_TASK_ID", "RIG_TASK_ID", "RETARGET_TASK_ID"):
            return "original_model_task_id must be MODEL_TASK_ID, RIG_TASK_ID or RETARGET_TASK_ID type"
        return True

    @classmethod
    async def execute(
        cls,
        original_model_task_id,
        format: str,
        quad: bool,
        force_symmetry: bool,
        face_limit: int,
        flatten_bottom: bool,
        flatten_bottom_threshold: float,
        texture_size: int,
        texture_format: str,
        pivot_to_center_bottom: bool,
        scale_factor: float,
        with_animation: bool,
        pack_uv: bool,
        bake: bool,
        part_names: str,
        fbx_preset: str,
        export_vertex_colors: bool,
        export_orientation: str,
        animate_in_place: bool,
    ) -> IO.NodeOutput:
        if not original_model_task_id:
            raise RuntimeError("original_model_task_id is required")

        # Parse part_names from comma-separated string to list
        part_names_list = None
        if part_names and part_names.strip():
            part_names_list = [name.strip() for name in part_names.split(",") if name.strip()]

        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoConvertModelRequest(
                original_model_task_id=original_model_task_id,
                format=format,
                quad=quad if quad else None,
                force_symmetry=force_symmetry if force_symmetry else None,
                face_limit=face_limit if face_limit != -1 else None,
                flatten_bottom=flatten_bottom if flatten_bottom else None,
                flatten_bottom_threshold=flatten_bottom_threshold if flatten_bottom_threshold != 0.0 else None,
                texture_size=texture_size if texture_size != 4096 else None,
                texture_format=texture_format if texture_format != "JPEG" else None,
                pivot_to_center_bottom=pivot_to_center_bottom if pivot_to_center_bottom else None,
                scale_factor=scale_factor if scale_factor != 1.0 else None,
                with_animation=with_animation if with_animation else None,
                pack_uv=pack_uv if pack_uv else None,
                bake=bake if bake else None,
                part_names=part_names_list,
                fbx_preset=fbx_preset if fbx_preset != "blender" else None,
                export_vertex_colors=export_vertex_colors if export_vertex_colors else None,
                export_orientation=export_orientation if export_orientation != "default" else None,
                animate_in_place=animate_in_place if animate_in_place else None,
            ),
        )
        return await poll_until_finished(cls, response, average_duration=30)


class TripoImportModelNode(IO.ComfyNode):
    """Imports an external 3D model into Tripo, producing a MODEL_TASK_ID for post-processing nodes."""

    SUPPORTED_FORMATS = ("glb", "fbx", "obj", "stl")

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoImportModelNode",
            display_name="Tripo: Import Model",
            category="partner/3d/Tripo",
            description="Import an external 3D model (e.g. from Rodin, Hunyuan3D or a local file) into Tripo "
            "to use it with Tripo's post-processing nodes: Texture, Rig, Convert. "
            "GLB is recommended: textures survive import only when embedded in the file. "
            "Note that texturing an imported model requires a texture prompt.",
            inputs=[
                IO.MultiType.Input(
                    "model_3d",
                    types=[IO.File3DGLB, IO.File3DFBX, IO.File3DOBJ, IO.File3DSTL, IO.File3DAny],
                    tooltip="3D model to import (GLB / FBX / OBJ / STL, up to 150 MB). "
                    "OBJ and STL files carry no embedded textures.",
                ),
            ],
            outputs=[
                IO.Custom("MODEL_TASK_ID").Output(display_name="model task_id"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"text","text":"Free"}""",
            ),
        )

    @classmethod
    async def execute(cls, model_3d: Types.File3D) -> IO.NodeOutput:
        file_format = (model_3d.format or "").lstrip(".").lower()
        if file_format == "gltf":
            raise ValueError(
                "GLTF (.gltf) references external files and cannot be imported. Export a single-file GLB instead."
            )
        if file_format not in cls.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported 3D format '{file_format or 'unknown'}'. "
                f"Tripo import supports: {', '.join(f.upper() for f in cls.SUPPORTED_FORMATS)}."
            )
        size = len(model_3d.get_bytes())
        if size > 150 * 1024 * 1024:
            raise ValueError(f"Model file is {size / (1024 * 1024):.1f} MB; Tripo import allows up to 150 MB.")

        url = await upload_3d_model_to_comfyapi(cls, model_3d, file_format)
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/import", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoImportModelRequest(url=url, format=file_format),
        )
        if response.code != 0:
            raise RuntimeError(f"Failed to import model: {response.error}")

        task_id = response.data.task_id
        response_poll = await poll_op(
            cls,
            poll_endpoint=ApiEndpoint(path=f"/proxy/tripo/v2/openapi/task/{task_id}"),
            response_model=TripoTaskResponse,
            failed_statuses=[
                TripoTaskStatus.FAILED,
                TripoTaskStatus.CANCELLED,
                TripoTaskStatus.UNKNOWN,
                TripoTaskStatus.BANNED,
                TripoTaskStatus.EXPIRED,
            ],
            status_extractor=lambda x: x.data.status,
            progress_extractor=lambda x: x.data.progress,
            estimated_duration=10,
        )
        if response_poll.data.status != TripoTaskStatus.SUCCESS:
            raise RuntimeError(f"Failed to import model: {response_poll}")
        return IO.NodeOutput(task_id)


def _p1_price_expr(*, geometry_credits: int, textured_credits: int, detailed_credits: int) -> str:
    return (
        "("
        "  $mode := widgets.output_mode;"
        '  $detailed := $lookup(widgets, "output_mode.texture_quality") = "detailed";'
        f'  $credits := $mode = "geometry only" ? {geometry_credits} : ($detailed ? {detailed_credits} : {textured_credits});'
        '  {"type":"usd","usd": $credits * 0.01, "format": {"approximate": true}}'
        ")"
    )


def _p1_textured_inputs(*, include_image_alignment: bool) -> list:
    """Inputs shown inside the 'Textured' branch of the P1 output_mode DynamicCombo."""
    inputs: list = [
        IO.Boolean.Input("pbr", default=True, tooltip="Include PBR maps. When on, base texture is forced on too."),
        IO.Combo.Input("texture_quality", options=["standard", "detailed"], default="standard"),
    ]
    if include_image_alignment:
        inputs.extend(
            [
                IO.Combo.Input(
                    "texture_alignment",
                    options=["original_image", "geometry"],
                    default="original_image",
                    tooltip="Prioritize visual fidelity to the source image, or alignment to the mesh geometry.",
                ),
                IO.Combo.Input(
                    "orientation",
                    options=["default", "align_image"],
                    default="default",
                    tooltip="Rotate the output to match the source image. Only applies when textured.",
                ),
            ]
        )
    inputs.append(IO.Int.Input("texture_seed", default=42, advanced=True))
    return inputs


def _build_p1_output_mode(*, include_image_alignment: bool) -> IO.DynamicCombo.Input:
    return IO.DynamicCombo.Input(
        "output_mode",
        options=[
            IO.DynamicCombo.Option("Geometry only", []),
            IO.DynamicCombo.Option("Textured", _p1_textured_inputs(include_image_alignment=include_image_alignment)),
        ],
        tooltip='"Geometry only" returns an untextured mesh. "Textured" adds color/PBR maps.',
    )


def _resolve_p1_texture_fields(output_mode: dict) -> dict:
    """Translate the output_mode DynamicCombo payload into P1 request fields.

    pbr=true forces texture=true server-side, but we send both explicitly so the
    intent is visible in the request body and logs.
    """
    mode = output_mode["output_mode"]
    if mode == "Geometry only":
        return {"texture": False, "pbr": False}
    out = {
        "texture": True,
        "pbr": bool(output_mode.get("pbr", True)),
        "texture_quality": output_mode.get("texture_quality", "standard"),
        "texture_seed": output_mode.get("texture_seed"),
    }
    if "texture_alignment" in output_mode:
        out["texture_alignment"] = output_mode["texture_alignment"]
    if "orientation" in output_mode:
        out["orientation"] = output_mode["orientation"]
    return out


def _p1_common_inputs() -> list:
    """Inputs shared by all P1 nodes (placed after output_mode)."""
    return [
        IO.Int.Input(
            "face_limit",
            default=-1,
            min=-1,
            max=20000,
            optional=True,
            advanced=True,
            tooltip="Target face count, 48-20000. -1 lets Tripo pick adaptively.",
        ),
        IO.Int.Input("model_seed", default=42, optional=True, advanced=True),
        IO.Boolean.Input(
            "auto_size",
            default=False,
            optional=True,
            advanced=True,
            tooltip="Scale the output to approximate real-world meters.",
        ),
        IO.Boolean.Input(
            "export_uv",
            default=True,
            optional=True,
            advanced=True,
            tooltip="UV unwrap during generation. Turn off for faster geometry-only runs.",
        ),
        IO.Boolean.Input(
            "compress_geometry",
            default=False,
            optional=True,
            advanced=True,
            tooltip="Apply geometry-based compression. Decompress before editing.",
        ),
    ]


def _build_p1_request_kwargs(
    *,
    output_mode: dict,
    face_limit: int,
    model_seed: int,
    auto_size: bool,
    export_uv: bool,
    compress_geometry: bool,
) -> dict:
    """Common P1 request fields shared by all three node types."""
    kwargs: dict = {
        "model_seed": model_seed,
        "face_limit": face_limit if face_limit != -1 else None,
        "auto_size": auto_size,
        "export_uv": export_uv,
        "compress": "geometry" if compress_geometry else None,
    }
    kwargs.update(_resolve_p1_texture_fields(output_mode))
    return kwargs


class TripoP1TextToModelNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoP1TextToModelNode",
            display_name="Tripo P1: Text to Model",
            category="partner/3d/Tripo",
            description="Tripo P1 text-to-3D. Optimized for low-poly, game-ready meshes with stable topology.",
            inputs=[
                IO.String.Input("prompt", multiline=True, tooltip="Up to 1024 characters."),
                IO.String.Input("negative_prompt", multiline=True, optional=True, tooltip="Up to 255 characters."),
                _build_p1_output_mode(include_image_alignment=False),
                IO.Int.Input("image_seed", default=42, optional=True, advanced=True),
                *_p1_common_inputs(),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MODEL_TASK_ID").Output(display_name="model task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["output_mode", "output_mode.texture_quality"]),
                expr=_p1_price_expr(geometry_credits=30, textured_credits=40, detailed_credits=50),
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        output_mode: dict,
        negative_prompt: str | None = None,
        image_seed: int | None = None,
        face_limit: int = -1,
        model_seed: int | None = None,
        auto_size: bool = False,
        export_uv: bool = True,
        compress_geometry: bool = False,
    ) -> IO.NodeOutput:
        if not prompt:
            raise RuntimeError("Prompt is required")
        common = _build_p1_request_kwargs(
            output_mode=output_mode,
            face_limit=face_limit,
            model_seed=model_seed,
            auto_size=auto_size,
            export_uv=export_uv,
            compress_geometry=compress_geometry,
        )
        request = TripoP1TextToModelRequest(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image_seed=image_seed,
            **common,
        )
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=request,
        )
        return await poll_until_finished(cls, response, average_duration=60)


class TripoP1ImageToModelNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoP1ImageToModelNode",
            display_name="Tripo P1: Image to Model",
            category="partner/3d/Tripo",
            description="Tripo P1 image-to-3D. Optimized for low-poly, game-ready meshes.",
            inputs=[
                IO.Image.Input("image"),
                _build_p1_output_mode(include_image_alignment=True),
                IO.Boolean.Input(
                    "enable_image_autofix",
                    default=False,
                    optional=True,
                    advanced=True,
                    tooltip="Pre-process the input image for better generation quality.",
                ),
                *_p1_common_inputs(),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MODEL_TASK_ID").Output(display_name="model task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["output_mode", "output_mode.texture_quality"]),
                expr=_p1_price_expr(geometry_credits=40, textured_credits=50, detailed_credits=60),
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        output_mode: dict,
        enable_image_autofix: bool = False,
        face_limit: int = -1,
        model_seed: int | None = None,
        auto_size: bool = False,
        export_uv: bool = True,
        compress_geometry: bool = False,
    ) -> IO.NodeOutput:
        if image is None:
            raise RuntimeError("Image is required")
        tripo_file = TripoFileReference(
            root=TripoUrlReference(
                url=(await upload_images_to_comfyapi(cls, image, max_images=1))[0],
                type="jpeg",
            )
        )
        common = _build_p1_request_kwargs(
            output_mode=output_mode,
            face_limit=face_limit,
            model_seed=model_seed,
            auto_size=auto_size,
            export_uv=export_uv,
            compress_geometry=compress_geometry,
        )
        request = TripoP1ImageToModelRequest(
            file=tripo_file,
            enable_image_autofix=enable_image_autofix,
            **common,
        )
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=request,
        )
        return await poll_until_finished(cls, response, average_duration=60)


class TripoP1MultiviewToModelNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoP1MultiviewToModelNode",
            display_name="Tripo P1: Multiview to Model",
            category="partner/3d/Tripo",
            description="Tripo P1 multiview-to-3D from 2-4 reference images in [front, left, back, right] order. "
            "Front is required; any combination of the other three may be omitted.",
            inputs=[
                IO.Image.Input("image", tooltip="Front view (0°). Required."),
                IO.Image.Input(
                    "image_left",
                    optional=True,
                    tooltip="Left view (90°), i.e. the subject's left side.",
                ),
                IO.Image.Input("image_back", optional=True, tooltip="Back view (180°)."),
                IO.Image.Input(
                    "image_right",
                    optional=True,
                    tooltip="Right view (270°), i.e. the subject's right side.",
                ),
                _build_p1_output_mode(include_image_alignment=True),
                *_p1_common_inputs(),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MODEL_TASK_ID").Output(display_name="model task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["output_mode", "output_mode.texture_quality"]),
                expr=_p1_price_expr(geometry_credits=40, textured_credits=50, detailed_credits=60),
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        output_mode: dict,
        image_left: Input.Image | None = None,
        image_back: Input.Image | None = None,
        image_right: Input.Image | None = None,
        face_limit: int = -1,
        model_seed: int | None = None,
        auto_size: bool = False,
        export_uv: bool = True,
        compress_geometry: bool = False,
    ) -> IO.NodeOutput:
        views = [image, image_left, image_back, image_right]
        if sum(1 for v in views if v is not None) < 2:
            raise RuntimeError("Tripo P1 multiview requires at least 2 images (front plus one of left/back/right).")

        files: list[TripoFileReference] = []
        for view in views:
            if view is None:
                files.append(TripoFileReference(root=TripoFileEmptyReference()))
                continue
            url = (await upload_images_to_comfyapi(cls, view, max_images=1))[0]
            files.append(TripoFileReference(root=TripoUrlReference(url=url, type="jpeg")))

        common = _build_p1_request_kwargs(
            output_mode=output_mode,
            face_limit=face_limit,
            model_seed=model_seed,
            auto_size=auto_size,
            export_uv=export_uv,
            compress_geometry=compress_geometry,
        )
        request = TripoP1MultiviewToModelRequest(files=files, **common)
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=request,
        )
        return await poll_until_finished(cls, response, average_duration=80)


class TripoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            TripoTextToModelNode,
            TripoImageToModelNode,
            TripoMultiviewToModelNode,
            TripoP1TextToModelNode,
            TripoP1ImageToModelNode,
            TripoP1MultiviewToModelNode,
            TripoImportModelNode,
            TripoTextureNode,
            TripoRefineNode,
            TripoRigNode,
            TripoRetargetNode,
            TripoConversionNode,
        ]


async def comfy_entrypoint() -> TripoExtension:
    return TripoExtension()
