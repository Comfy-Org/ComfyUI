import json
from pathlib import Path
from urllib.parse import urlparse

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input, Types
from comfy_api_nodes.apis.tripo import (
    TRIPO_BIPED_ANIMATIONS,
    TripoAnimatePrerigcheckRequest,
    TripoAnimateRetargetRequest,
    TripoAnimation,
    TripoAnimateRigRequest,
    TripoConvertModelRequest,
    TripoEditMultiviewImageRequest,
    TripoFileEmptyReference,
    TripoFileReference,
    TripoGenerateMultiviewImageRequest,
    TripoHighpolyToLowpolyRequest,
    TripoImageToModelRequest,
    TripoImportModelRequest,
    TripoMeshCompletionRequest,
    TripoMeshSegmentationRequest,
    TripoModelVersion,
    TripoMultiviewEditPrompt,
    TripoMultiviewToModelRequest,
    TripoOrientation,
    TripoOutFormat,
    TripoP1ImageToModelRequest,
    TripoP1MultiviewToModelRequest,
    TripoP1TextToModelRequest,
    TripoRigModelVersion,
    TripoRigType,
    TripoSpec,
    TripoStyle,
    TripoTaskResponse,
    TripoTaskStatus,
    TripoTaskType,
    TripoTextToModelRequest,
    TripoTextureModelRequest,
    TripoTextureModelVersion,
    TripoTexturePrompt,
    TripoUrlReference,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_file_3d,
    download_url_to_image_tensor,
    poll_op,
    sync_op,
    upload_3d_model_to_comfyapi,
    upload_images_to_comfyapi,
)

MULTIVIEW_KEYS = ("front_view_url", "left_view_url", "back_view_url", "right_view_url")
SEED_MAX = 2**31 - 1
TEXTURE_SOURCE_TYPES_WITH_IMAGE = ("text_to_model", "image_to_model", "multiview_to_model", "texture_model")


FACE_LIMIT_TOOLTIP = (
    "Maximum face count. -1 lets Tripo pick adaptively (about 1.4M faces on v3.x standard, 2M on detailed). "
    "Tripo clamps silently: v2.5 at 500,000, quad meshes at 150,000."
)


def get_model_url_from_response(response: TripoTaskResponse) -> str:
    if response.data is not None:
        for key in ["pbr_model", "model", "base_model"]:
            if getattr(response.data.output, key, None) is not None:
                return getattr(response.data.output, key)
    raise RuntimeError(f"Failed to get model url from response: {response}")


async def poll_task(
    node_cls: type[IO.ComfyNode],
    response: TripoTaskResponse,
    average_duration: int | None = None,
) -> TripoTaskResponse:
    """Polls the Tripo API endpoint until the task reaches a terminal state, then returns the response."""
    if response.code != 0:
        raise RuntimeError(f"Failed to create Tripo task: {response}")
    response_poll = await poll_op(
        node_cls,
        poll_endpoint=ApiEndpoint(path=f"/proxy/tripo/v2/openapi/task/{response.data.task_id}"),
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
        estimated_duration=average_duration,
    )
    if response_poll.data.status != TripoTaskStatus.SUCCESS:
        raise RuntimeError(f"Tripo task failed: {response_poll}")
    return response_poll


async def poll_until_finished(
    node_cls: type[IO.ComfyNode],
    response: TripoTaskResponse,
    average_duration: int | None = None,
) -> tuple[str, Types.File3D]:
    """Polls the Tripo API endpoint until the task reaches a terminal state, then downloads the model."""
    response_poll = await poll_task(node_cls, response, average_duration)
    task_id = response_poll.data.task_id
    url = get_model_url_from_response(response_poll)
    file_format = Path(urlparse(url).path).suffix.lstrip(".").lower() or "glb"
    return task_id, await download_url_to_file_3d(url, file_format, task_id=task_id)


async def check_riggable(node_cls: type[IO.ComfyNode], model_task_id: str) -> tuple[bool, str]:
    response = await sync_op(
        node_cls,
        endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
        response_model=TripoTaskResponse,
        data=TripoAnimatePrerigcheckRequest(original_model_task_id=model_task_id),
    )
    output = (await poll_task(node_cls, response, average_duration=5)).data.output
    return bool(output.riggable), output.rig_type or output.topology or ""


async def upload_image_reference(node_cls: type[IO.ComfyNode], image: Input.Image) -> TripoFileReference:
    url = (await upload_images_to_comfyapi(node_cls, image, max_images=1))[0]
    return TripoFileReference(root=TripoUrlReference(url=url, type="jpeg"))


async def multiview_output(node_cls: type[IO.ComfyNode], response: TripoTaskResponse, with_task_id: bool) -> IO.NodeOutput:
    response_poll = await poll_task(node_cls, response, average_duration=25)
    views = response_poll.data.output.generate_multiview_image or {}
    if any(not views.get(key) for key in MULTIVIEW_KEYS):
        raise RuntimeError(f"Tripo returned incomplete multiview images: {response_poll}")
    images = [await download_url_to_image_tensor(views[key], cls=node_cls) for key in MULTIVIEW_KEYS]
    return IO.NodeOutput(response_poll.data.task_id, *images) if with_task_id else IO.NodeOutput(*images)


def part_names_from_glb(model: Types.File3D) -> list[str]:
    data = model.get_bytes()
    if data[:4] != b"glTF":
        return []
    chunk_length = int.from_bytes(data[12:16], "little")
    document = json.loads(data[20 : 20 + chunk_length])
    return [node["name"] for node in document.get("nodes", []) if node.get("name")]


def split_part_names(part_names: str) -> list[str] | None:
    return list(dict.fromkeys(name.strip() for name in part_names.split(",") if name.strip())) or None


def glb_output(task_id: str, model: Types.File3D) -> IO.NodeOutput:
    if model.format != "glb":
        raise RuntimeError(f"Tripo returned a {model.format.upper()} file where GLB was expected")
    return IO.NodeOutput(f"{task_id}.glb", task_id, model)


def check_smart_low_poly_face_limit(smart_low_poly: bool | None, face_limit: int | None, quad: bool | None) -> None:
    if smart_low_poly and face_limit not in (None, -1) and not 500 <= face_limit <= (10000 if quad else 20000):
        raise ValueError(
            "With smart_low_poly, face_limit must be between 500 and 20,000 for triangles or 500 and 10,000 for quads."
        )


def glb_or_fbx_output(task_id: str, model: Types.File3D) -> IO.NodeOutput:
    if model.format not in ("glb", "fbx"):
        raise RuntimeError(f"Tripo returned a file of type {model.format.upper() or 'unknown'} where GLB or FBX was expected")
    return IO.NodeOutput(
        f"{task_id}.{model.format}",
        task_id,
        model if model.format == "glb" else None,
        model if model.format == "fbx" else None,
    )


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
                IO.String.Input("negative_prompt", multiline=True, optional=True, tooltip="Up to 255 characters."),
                IO.Combo.Input(
                    "model_version", options=TripoModelVersion, default=TripoModelVersion.v3_1_20260211, optional=True
                ),
                IO.Combo.Input(
                    "style",
                    options=TripoStyle,
                    default="None",
                    optional=True,
                    tooltip="No longer supported by Tripo and ignored. Kept for older workflows.",
                ),
                IO.Boolean.Input(
                    "texture",
                    default=True,
                    optional=True,
                    tooltip="Generate texture maps. Off returns bare geometry and ignores pbr.",
                ),
                IO.Boolean.Input(
                    "pbr",
                    default=True,
                    optional=True,
                    tooltip="PBR material maps (base color, metallic, roughness, normal). Requires texture.",
                ),
                IO.Int.Input("image_seed", default=42, min=0, max=SEED_MAX, optional=True, advanced=True),
                IO.Int.Input("model_seed", default=42, min=0, max=SEED_MAX, optional=True, advanced=True),
                IO.Int.Input("texture_seed", default=42, min=0, max=SEED_MAX, optional=True, advanced=True),
                IO.Combo.Input(
                    "texture_quality",
                    default="standard",
                    options=["standard", "detailed", "extreme"],
                    optional=True,
                    advanced=True,
                    tooltip="detailed = HD textures, extreme = 8K Ultra textures.",
                ),
                IO.Int.Input("face_limit", default=-1, min=-1, max=2000000, optional=True, advanced=True, tooltip=FACE_LIMIT_TOOLTIP),
                IO.Boolean.Input(
                    "quad",
                    default=False,
                    optional=True,
                    advanced=True,
                    tooltip="Quad mesh output. Tripo delivers quad meshes as FBX, so the result "
                    "arrives on the FBX output and the GLB output stays empty.",
                ),
                IO.Combo.Input(
                    "geometry_quality",
                    default="standard",
                    options=["standard", "detailed"],
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "smart_low_poly",
                    default=False,
                    optional=True,
                    advanced=True,
                    tooltip="Low-poly mesh with clean, hand-crafted style topology (500-20,000 faces, quad 500-10,000). "
                    "Best for simple subjects; complex ones may fail.",
                ),
                IO.Boolean.Input(
                    "auto_size",
                    default=True,
                    optional=True,
                    advanced=True,
                    tooltip="Scale textured models to their real-world size in meters. Tripo stores the size as the model's "
                    "scene transform and bakes it in when the model is converted, rigged or retargeted; ignored without texture.",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MODEL_TASK_ID").Output(display_name="model task_id"),
                IO.File3DGLB.Output(display_name="GLB", tooltip="Empty when quad is enabled."),
                IO.File3DFBX.Output(display_name="FBX", tooltip="Only populated when quad is enabled."),
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
                        "quad",
                        "smart_low_poly",
                        "texture_quality",
                        "geometry_quality",
                    ],
                ),
                expr="""
                (
                  $isV3OrLater := $contains(widgets.model_version,"v3.");
                  $tq := widgets.texture_quality;
                  $textureAddon := widgets.texture ? ($tq = "extreme" ? 20 : ($tq = "detailed" ? 10 : 0)) : 0;
                  $geometryAddon := (widgets.geometry_quality = "detailed" and $isV3OrLater) ? 20 : 0;
                  $credits := (widgets.texture ? 20 : 10)
                    + (widgets.quad ? 5 : 0)
                    + (widgets.smart_low_poly ? 10 : 0)
                    + $textureAddon
                    + $geometryAddon;
                  {"type":"usd","usd": $credits * 0.01, "format": {"approximate": true}}
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
        smart_low_poly: bool | None = None,
        auto_size: bool = True,
    ) -> IO.NodeOutput:
        if not prompt.strip():
            raise RuntimeError("Prompt is required")
        check_smart_low_poly_face_limit(smart_low_poly, face_limit, quad)
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoTextToModelRequest(
                type=TripoTaskType.TEXT_TO_MODEL,
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                model_version=model_version,
                texture=texture,
                pbr=False if texture is False else pbr,
                image_seed=image_seed,
                model_seed=model_seed,
                texture_seed=texture_seed,
                texture_quality=texture_quality,
                face_limit=face_limit if face_limit != -1 else None,
                geometry_quality=geometry_quality,
                auto_size=auto_size,
                quad=quad,
                smart_low_poly=smart_low_poly,
            ),
        )
        return glb_or_fbx_output(*await poll_until_finished(cls, response, average_duration=80))


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
                IO.Combo.Input(
                    "style",
                    options=TripoStyle,
                    default="None",
                    optional=True,
                    tooltip="No longer supported by Tripo and ignored. Kept for older workflows.",
                ),
                IO.Boolean.Input(
                    "texture",
                    default=True,
                    optional=True,
                    tooltip="Generate texture maps. Off returns bare geometry and ignores pbr.",
                ),
                IO.Boolean.Input(
                    "pbr",
                    default=True,
                    optional=True,
                    tooltip="PBR material maps (base color, metallic, roughness, normal). Requires texture.",
                ),
                IO.Int.Input("model_seed", default=42, min=0, max=SEED_MAX, optional=True, advanced=True),
                IO.Combo.Input(
                    "orientation",
                    options=TripoOrientation,
                    default=TripoOrientation.DEFAULT,
                    optional=True,
                    advanced=True,
                ),
                IO.Int.Input("texture_seed", default=42, min=0, max=SEED_MAX, optional=True, advanced=True),
                IO.Combo.Input(
                    "texture_quality",
                    default="standard",
                    options=["standard", "detailed", "extreme"],
                    optional=True,
                    advanced=True,
                    tooltip="detailed = HD textures, extreme = 8K Ultra textures.",
                ),
                IO.Combo.Input(
                    "texture_alignment",
                    default="original_image",
                    options=["original_image", "geometry"],
                    optional=True,
                    advanced=True,
                ),
                IO.Int.Input("face_limit", default=-1, min=-1, max=2000000, optional=True, advanced=True, tooltip=FACE_LIMIT_TOOLTIP),
                IO.Boolean.Input(
                    "quad",
                    default=False,
                    optional=True,
                    advanced=True,
                    tooltip="Quad mesh output. Tripo delivers quad meshes as FBX, so the result "
                    "arrives on the FBX output and the GLB output stays empty.",
                ),
                IO.Combo.Input(
                    "geometry_quality",
                    default="standard",
                    options=["standard", "detailed"],
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "smart_low_poly",
                    default=False,
                    optional=True,
                    advanced=True,
                    tooltip="Low-poly mesh with clean, hand-crafted style topology (500-20,000 faces, quad 500-10,000). "
                    "Best for simple subjects; complex ones may fail.",
                ),
                IO.Boolean.Input(
                    "auto_size",
                    default=True,
                    optional=True,
                    advanced=True,
                    tooltip="Scale textured models to their real-world size in meters. Tripo stores the size as the model's "
                    "scene transform and bakes it in when the model is converted, rigged or retargeted; ignored without texture.",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MODEL_TASK_ID").Output(display_name="model task_id"),
                IO.File3DGLB.Output(display_name="GLB", tooltip="Empty when quad is enabled."),
                IO.File3DFBX.Output(display_name="FBX", tooltip="Only populated when quad is enabled."),
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
                        "quad",
                        "smart_low_poly",
                        "texture_quality",
                        "geometry_quality",
                    ],
                ),
                expr="""
                (
                  $isV3OrLater := $contains(widgets.model_version,"v3.");
                  $tq := widgets.texture_quality;
                  $textureAddon := widgets.texture ? ($tq = "extreme" ? 20 : ($tq = "detailed" ? 10 : 0)) : 0;
                  $geometryAddon := (widgets.geometry_quality = "detailed" and $isV3OrLater) ? 20 : 0;
                  $credits := (widgets.texture ? 30 : 20)
                    + (widgets.quad ? 5 : 0)
                    + (widgets.smart_low_poly ? 10 : 0)
                    + $textureAddon
                    + $geometryAddon;
                  {"type":"usd","usd": $credits * 0.01, "format": {"approximate": true}}
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
        smart_low_poly: bool | None = None,
        auto_size: bool = True,
    ) -> IO.NodeOutput:
        if image is None:
            raise RuntimeError("Image is required")
        check_smart_low_poly_face_limit(smart_low_poly, face_limit, quad)
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
                texture=texture,
                pbr=False if texture is False else pbr,
                model_seed=model_seed,
                orientation=orientation,
                geometry_quality=geometry_quality,
                texture_alignment=texture_alignment,
                texture_seed=texture_seed,
                texture_quality=texture_quality,
                face_limit=face_limit if face_limit != -1 else None,
                auto_size=auto_size,
                quad=quad,
                smart_low_poly=smart_low_poly,
            ),
        )
        return glb_or_fbx_output(*await poll_until_finished(cls, response, average_duration=80))


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
                IO.Boolean.Input(
                    "texture",
                    default=True,
                    optional=True,
                    tooltip="Generate texture maps. Off returns bare geometry and ignores pbr.",
                ),
                IO.Boolean.Input(
                    "pbr",
                    default=True,
                    optional=True,
                    tooltip="PBR material maps (base color, metallic, roughness, normal). Requires texture.",
                ),
                IO.Int.Input("model_seed", default=42, min=0, max=SEED_MAX, optional=True, advanced=True),
                IO.Int.Input("texture_seed", default=42, min=0, max=SEED_MAX, optional=True, advanced=True),
                IO.Combo.Input(
                    "texture_quality",
                    default="standard",
                    options=["standard", "detailed", "extreme"],
                    optional=True,
                    advanced=True,
                    tooltip="detailed = HD textures, extreme = 8K Ultra textures.",
                ),
                IO.Combo.Input(
                    "texture_alignment",
                    default="original_image",
                    options=["original_image", "geometry"],
                    optional=True,
                    advanced=True,
                ),
                IO.Int.Input("face_limit", default=-1, min=-1, max=2000000, optional=True, advanced=True, tooltip=FACE_LIMIT_TOOLTIP),
                IO.Boolean.Input(
                    "quad",
                    default=False,
                    optional=True,
                    advanced=True,
                    tooltip="Quad mesh output. Tripo delivers quad meshes as FBX, so the result "
                    "arrives on the FBX output and the GLB output stays empty.",
                ),
                IO.Combo.Input(
                    "geometry_quality",
                    default="standard",
                    options=["standard", "detailed"],
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "smart_low_poly",
                    default=False,
                    optional=True,
                    advanced=True,
                    tooltip="Low-poly mesh with clean, hand-crafted style topology (500-20,000 faces, quad 500-10,000). "
                    "Best for simple subjects; complex ones may fail.",
                ),
                IO.Boolean.Input(
                    "auto_size",
                    default=False,
                    optional=True,
                    advanced=True,
                    tooltip="Scale textured models to their real-world size in meters. Tripo stores the size as the model's "
                    "scene transform and bakes it in when the model is converted, rigged or retargeted; ignored without texture.",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MODEL_TASK_ID").Output(display_name="model task_id"),
                IO.File3DGLB.Output(display_name="GLB", tooltip="Empty when quad is enabled."),
                IO.File3DFBX.Output(display_name="FBX", tooltip="Only populated when quad is enabled."),
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
                        "quad",
                        "smart_low_poly",
                        "texture_quality",
                        "geometry_quality",
                    ],
                ),
                expr="""
                (
                  $isV3OrLater := $contains(widgets.model_version,"v3.");
                  $tq := widgets.texture_quality;
                  $textureAddon := widgets.texture ? ($tq = "extreme" ? 20 : ($tq = "detailed" ? 10 : 0)) : 0;
                  $geometryAddon := (widgets.geometry_quality = "detailed" and $isV3OrLater) ? 20 : 0;
                  $credits := (widgets.texture ? 30 : 20)
                    + (widgets.quad ? 5 : 0)
                    + (widgets.smart_low_poly ? 10 : 0)
                    + $textureAddon
                    + $geometryAddon;
                  {"type":"usd","usd": $credits * 0.01, "format": {"approximate": true}}
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
        smart_low_poly: bool | None = None,
        auto_size: bool = False,
    ) -> IO.NodeOutput:
        if image is None:
            raise RuntimeError("front image for multiview is required")
        images = []
        image_dict = {"image": image, "image_left": image_left, "image_back": image_back, "image_right": image_right}
        if image_left is None and image_back is None and image_right is None:
            raise RuntimeError("At least one of left, back, or right image must be provided for multiview")
        check_smart_low_poly_face_limit(smart_low_poly, face_limit, quad)
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
                pbr=False if texture is False else pbr,
                model_seed=model_seed,
                texture_seed=texture_seed,
                texture_quality=texture_quality,
                geometry_quality=geometry_quality,
                texture_alignment=texture_alignment,
                face_limit=face_limit if face_limit != -1 else None,
                auto_size=auto_size,
                quad=quad,
                smart_low_poly=smart_low_poly,
            ),
        )
        return glb_or_fbx_output(*await poll_until_finished(cls, response, average_duration=80))


class TripoImageToMultiviewNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoImageToMultiviewNode",
            display_name="Tripo: Image to Multiview",
            category="partner/3d/Tripo",
            description="Generates front, left, back and right views of the subject. Feed them into "
            "Tripo: Multiview to Model, or refine them first with Tripo: Edit Multiview.",
            inputs=[IO.Image.Input("image")],
            outputs=[
                IO.Custom("MULTIVIEW_TASK_ID").Output(display_name="multiview task_id"),
                IO.Image.Output(display_name="front"),
                IO.Image.Output(display_name="left"),
                IO.Image.Output(display_name="back"),
                IO.Image.Output(display_name="right"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.1, "format": {"approximate": true}}""",
            ),
        )

    @classmethod
    async def execute(cls, image: Input.Image) -> IO.NodeOutput:
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoGenerateMultiviewImageRequest(file=await upload_image_reference(cls, image)),
        )
        return await multiview_output(cls, response, with_task_id=True)


class TripoEditMultiviewNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoEditMultiviewNode",
            display_name="Tripo: Edit Multiview",
            category="partner/3d/Tripo",
            description="Edits the views of a Tripo: Image to Multiview result with per-view text instructions. "
            "Views without an instruction stay unchanged. Feed the images into Tripo: Multiview to Model; "
            "an edited set cannot be edited again.",
            inputs=[
                IO.Custom("MULTIVIEW_TASK_ID").Input("multiview_task_id"),
                IO.String.Input("front_prompt", default="", multiline=True, optional=True),
                IO.String.Input("left_prompt", default="", multiline=True, optional=True),
                IO.String.Input("back_prompt", default="", multiline=True, optional=True),
                IO.String.Input("right_prompt", default="", multiline=True, optional=True),
            ],
            outputs=[
                IO.Image.Output(display_name="front"),
                IO.Image.Output(display_name="left"),
                IO.Image.Output(display_name="back"),
                IO.Image.Output(display_name="right"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["front_prompt", "left_prompt", "back_prompt", "right_prompt"]),
                expr="""
                (
                  $prompts := [widgets.front_prompt, widgets.left_prompt, widgets.back_prompt, widgets.right_prompt];
                  $edited := $count($filter($prompts, function($p) { $length($trim($p)) > 0 }));
                  {"type":"usd","usd": $edited * 0.05, "format": {"approximate": true}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        multiview_task_id,
        front_prompt: str = "",
        left_prompt: str = "",
        back_prompt: str = "",
        right_prompt: str = "",
    ) -> IO.NodeOutput:
        views = zip(("front", "left", "back", "right"), (front_prompt, left_prompt, back_prompt, right_prompt))
        prompts = [TripoMultiviewEditPrompt(view=view, prompt=prompt.strip()) for view, prompt in views if prompt.strip()]
        if not prompts:
            raise ValueError("Provide an edit instruction for at least one view.")
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoEditMultiviewImageRequest(original_task_id=multiview_task_id, prompts=prompts),
        )
        return await multiview_output(cls, response, with_task_id=False)


class TripoTextureNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoTextureNode",
            display_name="Tripo: Texture model",
            category="partner/3d/Tripo",
            inputs=[
                IO.Custom("MODEL_TASK_ID,SEGMENT_TASK_ID").Input("model_task_id"),
                IO.Boolean.Input(
                    "texture",
                    default=True,
                    optional=True,
                    tooltip="Ignored: this node always generates textures. Kept for older workflows.",
                ),
                IO.Boolean.Input(
                    "pbr",
                    default=True,
                    optional=True,
                    tooltip="PBR material maps (base color, metallic, roughness, normal); off gives a plain color texture.",
                ),
                IO.Int.Input("texture_seed", default=42, min=0, max=SEED_MAX, optional=True, advanced=True),
                IO.Combo.Input(
                    "texture_quality",
                    default="standard",
                    options=["standard", "detailed", "extreme"],
                    optional=True,
                    advanced=True,
                    tooltip="detailed = HD textures, extreme = 8K Ultra textures.",
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
                    "models (Tripo: Import Model), which carry no source image to infer colors from. "
                    "Cannot be combined with reference images.",
                ),
                IO.Combo.Input(
                    "model_version",
                    options=TripoTextureModelVersion,
                    default=TripoTextureModelVersion.v3_0_20250812,
                    optional=True,
                    tooltip="Texture model: v3.0 for meshes generated with v3.x, v2.5 for meshes generated with v2.5.",
                ),
                IO.Image.Input(
                    "style_image",
                    optional=True,
                    tooltip="Reference image for the artistic style of the textures. Only used together with texture_prompt.",
                ),
                IO.DynamicCombo.Input(
                    "reference",
                    options=[
                        IO.DynamicCombo.Option("none", []),
                        IO.DynamicCombo.Option(
                            "image",
                            [IO.Image.Input("reference_image", tooltip="Single reference image the textures should follow.")],
                        ),
                        IO.DynamicCombo.Option(
                            "multiview",
                            [
                                IO.Image.Input("image_front", tooltip="Front view (0°)."),
                                IO.Image.Input("image_left", tooltip="Left view (90°)."),
                                IO.Image.Input("image_back", tooltip="Back view (180°)."),
                                IO.Image.Input("image_right", tooltip="Right view (270°)."),
                            ],
                        ),
                    ],
                    optional=True,
                    tooltip="Reference images guiding the textures. Cannot be combined with texture_prompt or style_image.",
                ),
                IO.String.Input(
                    "part_names",
                    default="",
                    optional=True,
                    advanced=True,
                    tooltip="Comma-separated part names from Tripo: Segment Model to texture. Empty textures every part.",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MODEL_TASK_ID").Output(display_name="model task_id"),
                IO.File3DGLB.Output(display_name="GLB", tooltip="Empty when the source is a quad mesh or an FBX import."),
                IO.File3DFBX.Output(
                    display_name="FBX",
                    tooltip="Tripo returns FBX for quad meshes and FBX imports; empty otherwise.",
                ),
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
                  {"type":"usd","usd": ($tq = "extreme" ? 0.3 : ($tq = "detailed" ? 0.2 : 0.1)), "format": {"approximate": true}}
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
        model_version: str | None = None,
        style_image: Input.Image | None = None,
        reference: dict | None = None,
        part_names: str = "",
    ) -> IO.NodeOutput:
        text = texture_prompt.strip()
        mode = reference["reference"] if reference else "none"
        if mode != "none" and (text or style_image is not None):
            raise ValueError("Reference images cannot be combined with texture_prompt or style_image.")
        if style_image is not None and not text:
            raise ValueError("style_image requires a texture_prompt.")
        if not text:
            source = await sync_op(
                cls,
                endpoint=ApiEndpoint(path=f"/proxy/tripo/v2/openapi/task/{model_task_id}"),
                response_model=TripoTaskResponse,
            )
            if source.data.type not in TEXTURE_SOURCE_TYPES_WITH_IMAGE:
                raise ValueError(
                    "This model has no source image to texture from (imported, segmented, completed or retopologized). "
                    "Give a texture_prompt; Tripo accepts reference images only for models it generated itself."
                )
        if mode == "image":
            prompt = TripoTexturePrompt(image=await upload_image_reference(cls, reference["reference_image"]))
        elif mode == "multiview":
            views = [reference[k] for k in ("image_front", "image_left", "image_back", "image_right")]
            urls = await upload_images_to_comfyapi(cls, views, max_images=4)
            prompt = TripoTexturePrompt(images=[TripoFileReference(root=TripoUrlReference(url=u, type="jpeg")) for u in urls])
        elif text:
            prompt = TripoTexturePrompt(
                text=text,
                style_image=await upload_image_reference(cls, style_image) if style_image is not None else None,
            )
        else:
            prompt = None
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoTextureModelRequest(
                original_model_task_id=model_task_id,
                model_version=model_version,
                texture=True,
                pbr=pbr,
                texture_seed=texture_seed,
                texture_quality=texture_quality,
                texture_alignment=texture_alignment,
                texture_prompt=prompt,
                part_names=split_part_names(part_names),
            ),
        )
        return glb_or_fbx_output(*await poll_until_finished(cls, response, average_duration=80))


class TripoRigNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoRigNode",
            display_name="Tripo: Rig model",
            category="partner/3d/Tripo",
            inputs=[
                IO.Custom("MODEL_TASK_ID").Input("original_model_task_id"),
                IO.Combo.Input(
                    "model_version",
                    options=TripoRigModelVersion,
                    default=TripoRigModelVersion.v1_0_20240301,
                    optional=True,
                    tooltip="v1.0: humanoid (biped) characters only, 90+ animation presets. "
                    "v2.5: non-humanoid creatures (quadruped, hexapod, octopod, avian, serpentine, aquatic).",
                ),
                IO.Combo.Input(
                    "rig_type",
                    options=["auto", *[t.value for t in TripoRigType]],
                    default="auto",
                    optional=True,
                    tooltip="Skeleton type. 'auto' runs Tripo's free rig check first and uses the recommended type.",
                ),
                IO.Combo.Input(
                    "spec",
                    options=TripoSpec,
                    default=TripoSpec.TRIPO,
                    optional=True,
                    tooltip="Bone naming: Tripo native or Mixamo-compatible. Tripo cannot retarget its animation presets "
                    "onto a v1.0 rig made with the mixamo spec; use tripo for Tripo: Retarget rigged model.",
                ),
                IO.Combo.Input(
                    "out_format",
                    options=TripoOutFormat,
                    default=TripoOutFormat.GLB,
                    optional=True,
                    tooltip="Output file format; the result arrives on the matching output.",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("RIG_TASK_ID").Output(display_name="rig task_id"),
                IO.File3DGLB.Output(display_name="GLB", tooltip="Populated when out_format is glb."),
                IO.File3DFBX.Output(display_name="FBX", tooltip="Populated when out_format is fbx."),
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
    async def execute(
        cls,
        original_model_task_id,
        model_version: str = "v1.0-20240301",
        rig_type: str = "auto",
        spec: str = "tripo",
        out_format: str = "glb",
    ) -> IO.NodeOutput:
        if rig_type == "auto":
            riggable, rig_type = await check_riggable(cls, original_model_task_id)
            if not riggable:
                raise ValueError("Tripo reports that this model cannot be rigged.")
        if rig_type != "biped" and model_version == "v1.0-20240301":
            raise ValueError(f"Rig model v1.0-20240301 only supports biped skeletons; use v2.5-20260210 for {rig_type}.")
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoAnimateRigRequest(
                original_model_task_id=original_model_task_id,
                model_version=model_version,
                rig_type=rig_type,
                out_format=out_format,
                spec=spec,
            ),
        )
        return glb_or_fbx_output(*await poll_until_finished(cls, response, average_duration=180))


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
                    options=[*[a.value for a in TripoAnimation], *TRIPO_BIPED_ANIMATIONS],
                    tooltip="preset:* animations work with both rig models; "
                    "preset:biped:* animations require a rig made with model v1.0-20240301.",
                ),
                IO.Combo.Input(
                    "out_format",
                    options=TripoOutFormat,
                    default=TripoOutFormat.GLB,
                    optional=True,
                    tooltip="Output file format; the result arrives on the matching output.",
                ),
                IO.Boolean.Input(
                    "export_with_geometry",
                    default=True,
                    optional=True,
                    advanced=True,
                    tooltip="Include the mesh in the export; off exports the animated skeleton only.",
                ),
                IO.Boolean.Input(
                    "animate_in_place",
                    default=False,
                    optional=True,
                    advanced=True,
                    tooltip="Play the animation in place, without root displacement.",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("RETARGET_TASK_ID").Output(display_name="retarget task_id"),
                IO.File3DGLB.Output(display_name="GLB", tooltip="Populated when out_format is glb."),
                IO.File3DFBX.Output(display_name="FBX", tooltip="Populated when out_format is fbx."),
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
    async def execute(
        cls,
        original_model_task_id,
        animation: str,
        out_format: str = "glb",
        export_with_geometry: bool = True,
        animate_in_place: bool = False,
    ) -> IO.NodeOutput:
        rig = await sync_op(
            cls,
            endpoint=ApiEndpoint(path=f"/proxy/tripo/v2/openapi/task/{original_model_task_id}"),
            response_model=TripoTaskResponse,
        )
        rig_input = rig.data.input or {}
        if rig_input.get("spec") == "mixamo" and str(rig_input.get("model_version", "")).startswith("v1.0"):
            raise ValueError("Tripo cannot retarget animation presets onto a v1.0 rig made with the mixamo spec.")
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoAnimateRetargetRequest(
                original_model_task_id=original_model_task_id,
                animation=animation,
                out_format=out_format,
                export_with_geometry=export_with_geometry,
                animate_in_place=animate_in_place,
            ),
        )
        return glb_or_fbx_output(*await poll_until_finished(cls, response, average_duration=30))


class TripoRigCheckNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoRigCheckNode",
            display_name="Tripo: Rig Check",
            category="partner/3d/Tripo",
            description="Checks whether a model can be rigged and which skeleton type Tripo recommends.",
            inputs=[IO.Custom("MODEL_TASK_ID").Input("model_task_id")],
            outputs=[
                IO.Boolean.Output(display_name="riggable"),
                IO.String.Output(
                    display_name="rig_type",
                    tooltip="Recommended skeleton: biped, quadruped, hexapod, octopod, avian, serpentine or aquatic; "
                    "'others' when the model is not riggable.",
                ),
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
    async def execute(cls, model_task_id) -> IO.NodeOutput:
        riggable, rig_type = await check_riggable(cls, model_task_id)
        return IO.NodeOutput(riggable, rig_type)


class TripoSegmentNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoSegmentNode",
            display_name="Tripo: Segment Model",
            category="partner/3d/Tripo",
            description="Splits a model into parts. The part names feed Tripo: Complete Mesh Parts, "
            "Tripo: Retopology and Tripo: Convert model.",
            inputs=[IO.Custom("MODEL_TASK_ID").Input("model_task_id")],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("SEGMENT_TASK_ID").Output(display_name="segment task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
                IO.String.Output(display_name="part_names", tooltip="Comma-separated names of the parts."),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.4, "format": {"approximate": true}}""",
            ),
        )

    @classmethod
    async def execute(cls, model_task_id) -> IO.NodeOutput:
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoMeshSegmentationRequest(original_model_task_id=model_task_id),
        )
        task_id, model = await poll_until_finished(cls, response, average_duration=160)
        return IO.NodeOutput(f"{task_id}.glb", task_id, model, ",".join(part_names_from_glb(model)))


class TripoMeshCompleteNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoMeshCompleteNode",
            display_name="Tripo: Complete Mesh Parts",
            category="partner/3d/Tripo",
            description="Completes the parts of a segmented model and repairs missing regions.",
            inputs=[
                IO.Custom("SEGMENT_TASK_ID").Input("segment_task_id"),
                IO.String.Input(
                    "part_names",
                    default="",
                    optional=True,
                    tooltip="Comma-separated part names to complete. Empty completes every part.",
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
                expr="""{"type":"usd","usd":0.5, "format": {"approximate": true}}""",
            ),
        )

    @classmethod
    async def execute(cls, segment_task_id, part_names: str = "") -> IO.NodeOutput:
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoMeshCompletionRequest(
                original_model_task_id=segment_task_id,
                part_names=split_part_names(part_names),
            ),
        )
        return glb_output(*await poll_until_finished(cls, response, average_duration=240))


class TripoRetopologyNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoRetopologyNode",
            display_name="Tripo: Retopology",
            category="partner/3d/Tripo",
            description="Rebuilds a low-poly mesh with clean topology from a high-poly model.",
            inputs=[
                IO.Custom("MODEL_TASK_ID,SEGMENT_TASK_ID").Input("model_task_id"),
                IO.Int.Input(
                    "face_limit",
                    default=-1,
                    min=-1,
                    max=20000,
                    tooltip="Target face count: 500-20,000 triangles or 500-10,000 quads. -1 lets Tripo choose.",
                ),
                IO.Boolean.Input(
                    "quad",
                    default=False,
                    tooltip="Quad mesh output. Tripo delivers quad meshes as FBX, so the result "
                    "arrives on the FBX output and the GLB output stays empty.",
                ),
                IO.Boolean.Input(
                    "bake",
                    default=True,
                    optional=True,
                    advanced=True,
                    tooltip="Bake the source textures onto the low-poly mesh.",
                ),
                IO.String.Input(
                    "part_names",
                    default="",
                    optional=True,
                    advanced=True,
                    tooltip="Comma-separated part names from Tripo: Segment Model. Empty processes the whole model.",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MODEL_TASK_ID").Output(display_name="model task_id"),
                IO.File3DGLB.Output(display_name="GLB", tooltip="Empty when quad is enabled."),
                IO.File3DFBX.Output(display_name="FBX", tooltip="Only populated when quad is enabled."),
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
    async def execute(
        cls,
        model_task_id,
        face_limit: int = -1,
        quad: bool = False,
        bake: bool = True,
        part_names: str = "",
    ) -> IO.NodeOutput:
        if face_limit != -1 and not 500 <= face_limit <= (10000 if quad else 20000):
            raise ValueError("face_limit must be between 500 and 20,000 for triangles or 500 and 10,000 for quads.")
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoHighpolyToLowpolyRequest(
                original_model_task_id=model_task_id,
                face_limit=face_limit if face_limit != -1 else None,
                quad=quad,
                bake=bake,
                part_names=split_part_names(part_names),
            ),
        )
        return glb_or_fbx_output(*await poll_until_finished(cls, response, average_duration=200))


class TripoConversionNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoConversionNode",
            display_name="Tripo: Convert model",
            category="partner/3d/Tripo",
            inputs=[
                IO.Custom("MODEL_TASK_ID,RIG_TASK_ID,RETARGET_TASK_ID,SEGMENT_TASK_ID").Input("original_model_task_id"),
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
                    max=8192,
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
                    default=0.01,
                    min=0.01,
                    max=1.0,
                    optional=True,
                    advanced=True,
                    tooltip="Flattening depth used with flatten_bottom.",
                ),
                IO.Boolean.Input("pivot_to_center_bottom", default=False, optional=True, advanced=True),
                IO.Float.Input(
                    "scale_factor",
                    default=1.0,
                    min=0.01,
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "with_animation",
                    default=True,
                    optional=True,
                    advanced=True,
                    tooltip="Keep the skeleton and animation of rigged or retargeted models.",
                ),
                IO.Boolean.Input("pack_uv", default=False, optional=True, advanced=True),
                IO.Boolean.Input(
                    "bake",
                    default=True,
                    optional=True,
                    advanced=True,
                    tooltip="Bake advanced materials into the base textures for broader compatibility.",
                ),
                IO.String.Input("part_names", default="", optional=True, advanced=True),  # comma-separated list
                IO.Combo.Input(
                    "fbx_preset",
                    options=["blender", "mixamo", "3dsmax", "bake_scale"],
                    default="blender",
                    optional=True,
                    advanced=True,
                    tooltip="FBX compatibility preset. bake_scale bakes the scale transform into the geometry.",
                ),
                IO.Boolean.Input("export_vertex_colors", default=False, optional=True, advanced=True),
                IO.Combo.Input(
                    "export_orientation",
                    options=["default", "+x", "-x", "+y", "-y"],
                    default="default",
                    optional=True,
                    advanced=True,
                    tooltip="Forward axis of the exported model. default keeps Tripo's +x.",
                ),
                IO.Boolean.Input("animate_in_place", default=False, optional=True, advanced=True),
            ],
            outputs=[
                IO.File3DAny.Output(
                    display_name="model_3d",
                    tooltip="Converted model in the requested format. OBJ is delivered by Tripo as a ZIP archive "
                    "(mesh, material and textures).",
                ),
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
                        "quad",
                        "face_limit",
                        "texture_size",
                        "texture_format",
                        "flatten_bottom",
                        "pivot_to_center_bottom",
                        "scale_factor",
                    ],
                ),
                expr="""
                (
                    $face := (widgets.face_limit != null) ? widgets.face_limit : -1;
                    $texSize := (widgets.texture_size != null) ? widgets.texture_size : 4096;
                    $scale := (widgets.scale_factor != null) ? widgets.scale_factor : 1;
                    $texFmt := (widgets.texture_format != "" ? widgets.texture_format : "jpeg");
                    $advanced :=
                      widgets.quad or
                      widgets.flatten_bottom or
                      widgets.pivot_to_center_bottom or
                      ($face != -1) or
                      ($texSize != 4096) or
                      ($scale != 1) or
                      ($texFmt != "jpeg");
                    {"type":"usd","usd": ($advanced ? 0.1 : 0.05), "format": {"approximate": true}}
                )
                """,
            ),
        )

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
        part_names_list = split_part_names(part_names)

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
                flatten_bottom_threshold=flatten_bottom_threshold if flatten_bottom else None,
                texture_size=texture_size if texture_size != 4096 else None,
                texture_format=texture_format if texture_format != "JPEG" else None,
                pivot_to_center_bottom=pivot_to_center_bottom if pivot_to_center_bottom else None,
                scale_factor=scale_factor if scale_factor != 1.0 else None,
                with_animation=with_animation,
                pack_uv=pack_uv if pack_uv else None,
                bake=bake,
                part_names=part_names_list,
                fbx_preset=fbx_preset if fbx_preset != "blender" else None,
                export_vertex_colors=export_vertex_colors if export_vertex_colors else None,
                export_orientation=export_orientation if export_orientation != "default" else None,
                animate_in_place=animate_in_place if animate_in_place else None,
            ),
        )
        _, model = await poll_until_finished(cls, response, average_duration=30)
        return IO.NodeOutput(model)


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


def _p1_price_expr(*, geometry_credits: int, textured_credits: int, detailed_credits: int, extreme_credits: int) -> str:
    return (
        "("
        "  $mode := widgets.output_mode;"
        '  $tq := $lookup(widgets, "output_mode.texture_quality");'
        f'  $textured := $tq = "extreme" ? {extreme_credits} : ($tq = "detailed" ? {detailed_credits} : {textured_credits});'
        f'  $credits := $mode = "geometry only" ? {geometry_credits} : $textured;'
        '  {"type":"usd","usd": $credits * 0.01, "format": {"approximate": true}}'
        ")"
    )


def _p1_textured_inputs(*, include_image_alignment: bool) -> list:
    """Inputs shown inside the 'Textured' branch of the P1 output_mode DynamicCombo."""
    inputs: list = [
        IO.Boolean.Input("pbr", default=True, tooltip="Include PBR maps. When on, base texture is forced on too."),
        IO.Combo.Input(
            "texture_quality",
            options=["standard", "detailed", "extreme"],
            default="standard",
            tooltip="detailed = HD textures, extreme = 8K Ultra textures.",
        ),
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
    inputs.append(IO.Int.Input("texture_seed", default=42, min=0, max=SEED_MAX, advanced=True))
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
        IO.Int.Input("model_seed", default=42, min=0, max=SEED_MAX, optional=True, advanced=True),
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
            tooltip="Apply meshopt geometry compression (EXT_meshopt_compression). Smaller files, "
            "but ComfyUI's 3D preview cannot display them; decompress before editing.",
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
                IO.Int.Input("image_seed", default=42, min=0, max=SEED_MAX, optional=True, advanced=True),
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
                expr=_p1_price_expr(geometry_credits=30, textured_credits=40, detailed_credits=50, extreme_credits=60),
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
        if not prompt.strip():
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
        return glb_output(*await poll_until_finished(cls, response, average_duration=60))


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
                expr=_p1_price_expr(geometry_credits=40, textured_credits=50, detailed_credits=60, extreme_credits=70),
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
        return glb_output(*await poll_until_finished(cls, response, average_duration=60))


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
                expr=_p1_price_expr(geometry_credits=40, textured_credits=50, detailed_credits=60, extreme_credits=70),
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
        return glb_output(*await poll_until_finished(cls, response, average_duration=80))


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
            TripoImageToMultiviewNode,
            TripoEditMultiviewNode,
            TripoTextureNode,
            TripoRigCheckNode,
            TripoRigNode,
            TripoRetargetNode,
            TripoSegmentNode,
            TripoMeshCompleteNode,
            TripoRetopologyNode,
            TripoConversionNode,
        ]


async def comfy_entrypoint() -> TripoExtension:
    return TripoExtension()
