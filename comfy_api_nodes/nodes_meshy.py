from io import BytesIO

import numpy as np
import torch
from PIL import Image
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.meshy import (
    InputShouldRemesh,
    InputShouldTexture,
    MeshyAnimationRequest,
    MeshyAnimationResult,
    MeshyImageToModelRequest,
    MeshyModelResult,
    MeshyMultiImageToModelRequest,
    MeshyRefineTask,
    MeshyReliefRequest,
    MeshyReliefResult,
    MeshyRiggedResult,
    MeshyRiggingRequest,
    MeshySmartTopologyImageToModelRequest,
    MeshySmartTopologyTextToModelRequest,
    MeshyTaskResponse,
    MeshyTextToModelRequest,
    MeshyTextureRequest,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_bytesio,
    download_url_to_file_3d,
    download_url_to_image_tensor,
    get_output_consumers,
    poll_op,
    sync_op,
    upload_image_to_comfyapi,
    upload_images_to_comfyapi,
    validate_string,
)

RELIEF_FORMATS = ("glb", "stl", "obj", "fbx", "usdz", "3mf")


def validate_ultra_mode(model: str, ultra_mode: bool, ultra_resolution: str = "2k") -> None:
    if not ultra_mode:
        return
    if model not in ("meshy-7.1", "meshy-7", "latest"):
        raise ValueError("ultra_mode requires the meshy-7.1, meshy-7 or latest model")
    if ultra_resolution == "4k" and model not in ("meshy-7.1", "latest"):
        raise ValueError("4k ultra_resolution requires the meshy-7.1 or latest model")


def depth_map_to_tensor(data: BytesIO) -> torch.Tensor:
    depth = np.array(Image.open(data))
    depth = torch.from_numpy(depth.astype(np.float32) / np.iinfo(depth.dtype).max)
    return depth[None, :, :, None].repeat(1, 1, 1, 3)


def relief_bend_inputs() -> list[IO.Input]:
    return [
        IO.Combo.Input(
            "bend_axis",
            options=["xz", "yz"],
            tooltip="xz curls the left and right edges, yz the top and bottom edges.",
        ),
        IO.Float.Input(
            "bend_angle",
            default=60.0,
            min=1.0,
            max=360.0,
            step=1.0,
            tooltip="Arc the bend spans in degrees; 360 closes the relief into a cylinder.",
        ),
    ]


class MeshyTextToModelNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MeshyTextToModelNode",
            display_name="Meshy: Text to Model",
            category="partner/3d/Meshy",
            inputs=[
                IO.Combo.Input("model", options=["meshy-7.1", "meshy-7", "meshy-6", "latest"]),
                IO.String.Input("prompt", multiline=True, default=""),
                IO.Combo.Input("style", options=["realistic"]),
                IO.DynamicCombo.Input(
                    "should_remesh",
                    options=[
                        IO.DynamicCombo.Option(
                            "true",
                            [
                                IO.Combo.Input("topology", options=["triangle", "quad"]),
                                IO.Int.Input(
                                    "target_polycount",
                                    default=300000,
                                    min=100,
                                    max=300000,
                                    display_mode=IO.NumberDisplay.number,
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option("false", []),
                    ],
                    tooltip="When set to false, returns an unprocessed triangular mesh.",
                ),
                IO.Combo.Input("symmetry_mode", options=["auto", "on", "off"], advanced=True),
                IO.Combo.Input(
                    "pose_mode",
                    options=["", "A-pose", "T-pose"],
                    tooltip="Specify the pose mode for the generated model.",
                    advanced=True,
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
                IO.Boolean.Input(
                    "ultra_mode",
                    default=False,
                    tooltip="Run an extra refinement pass for higher-fidelity geometry with finer surface detail.",
                ),
                IO.Combo.Input(
                    "ultra_resolution",
                    options=["2k", "4k"],
                    tooltip="Resolution of the ultra pass: 2k runs it at 2048³, 4k at 4096³ for the finest "
                    "surface detail. 4k requires meshy-7.1 or latest. Used only when ultra_mode is enabled.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MESHY_TASK_ID").Output(display_name="meshy_task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
                IO.File3DFBX.Output(display_name="FBX"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model", "ultra_mode"]),
                expr="""
                (
                  $credits := 20
                    + ((widgets.ultra_mode and widgets.model in ["meshy-7.1", "meshy-7", "latest"]) ? 5 : 0);
                  {"type":"usd","usd": $credits * 0.0572}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        style: str,
        should_remesh: InputShouldRemesh,
        symmetry_mode: str,
        pose_mode: str,
        seed: int,
        ultra_mode: bool,
        ultra_resolution: str = "2k",
    ) -> IO.NodeOutput:
        validate_string(prompt, field_name="prompt", min_length=1, max_length=600)
        validate_ultra_mode(model, ultra_mode, ultra_resolution)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/meshy/openapi/v2/text-to-3d", method="POST"),
            response_model=MeshyTaskResponse,
            data=MeshyTextToModelRequest(
                prompt=prompt,
                art_style=style,
                ai_model=model,
                topology=should_remesh.get("topology", None),
                target_polycount=should_remesh.get("target_polycount", None),
                should_remesh=should_remesh["should_remesh"] == "true",
                symmetry_mode=symmetry_mode,
                pose_mode=pose_mode.lower(),
                ultra_mode=True if ultra_mode and model == "meshy-7" else None,
                geometry_resolution=ultra_resolution if ultra_mode and model != "meshy-7" else None,
                seed=seed,
            ),
        )
        task_id = response.result
        result = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/meshy/openapi/v2/text-to-3d/{task_id}"),
            response_model=MeshyModelResult,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
        )
        return IO.NodeOutput(
            f"{task_id}.glb",
            task_id,
            await download_url_to_file_3d(result.model_urls.glb, "glb", task_id=task_id),
            await download_url_to_file_3d(result.model_urls.fbx, "fbx", task_id=task_id),
        )


class MeshyRefineNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MeshyRefineNode",
            display_name="Meshy: Refine Draft Model",
            category="partner/3d/Meshy",
            description="Refine a previously created draft model.",
            inputs=[
                IO.Combo.Input("model", options=["meshy-7.1", "meshy-7", "meshy-6", "latest"]),
                IO.Custom("MESHY_TASK_ID").Input("meshy_task_id"),
                IO.Boolean.Input(
                    "enable_pbr",
                    default=False,
                    tooltip="Generate PBR Maps (metallic, roughness, normal) in addition to the base color.",
                    advanced=True,
                ),
                IO.String.Input(
                    "texture_prompt",
                    default="",
                    multiline=True,
                    tooltip="Provide a text prompt to guide the texturing process. "
                    "Maximum 600 characters. Cannot be used at the same time as 'texture_image'.",
                ),
                IO.Image.Input(
                    "texture_image",
                    tooltip="Only one of 'texture_image' or 'texture_prompt' may be used at the same time.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "texture_resolution",
                    options=["2k", "4k", "8k"],
                    tooltip="Base color texture resolution. Higher resolutions capture more surface detail.",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MESHY_TASK_ID").Output(display_name="meshy_task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
                IO.File3DFBX.Output(display_name="FBX"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["texture_resolution"]),
                expr="""
                (
                  $credits := widgets.texture_resolution = "8k" ? 15 : 10;
                  {"type":"usd","usd": $credits * 0.0572}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        meshy_task_id: str,
        enable_pbr: bool,
        texture_prompt: str,
        texture_resolution: str,
        texture_image: Input.Image | None = None,
    ) -> IO.NodeOutput:
        if texture_prompt and texture_image is not None:
            raise ValueError("texture_prompt and texture_image cannot be used at the same time")
        texture_image_url = None
        if texture_prompt:
            validate_string(texture_prompt, field_name="texture_prompt", max_length=600)
        if texture_image is not None:
            texture_image_url = (await upload_images_to_comfyapi(cls, texture_image, wait_label="Uploading texture"))[0]
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/meshy/openapi/v2/text-to-3d", method="POST"),
            response_model=MeshyTaskResponse,
            data=MeshyRefineTask(
                preview_task_id=meshy_task_id,
                enable_pbr=enable_pbr,
                texture_resolution=texture_resolution,
                texture_prompt=texture_prompt if texture_prompt else None,
                texture_image_url=texture_image_url,
                ai_model=model,
            ),
        )
        task_id = response.result
        result = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/meshy/openapi/v2/text-to-3d/{task_id}"),
            response_model=MeshyModelResult,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
        )
        return IO.NodeOutput(
            f"{task_id}.glb",
            task_id,
            await download_url_to_file_3d(result.model_urls.glb, "glb", task_id=task_id),
            await download_url_to_file_3d(result.model_urls.fbx, "fbx", task_id=task_id),
        )


class MeshyImageToModelNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MeshyImageToModelNode",
            display_name="Meshy: Image to Model",
            category="partner/3d/Meshy",
            inputs=[
                IO.Combo.Input("model", options=["meshy-7.1", "meshy-7", "meshy-6", "latest"]),
                IO.Image.Input("image"),
                IO.DynamicCombo.Input(
                    "should_remesh",
                    options=[
                        IO.DynamicCombo.Option(
                            "true",
                            [
                                IO.Combo.Input("topology", options=["triangle", "quad"]),
                                IO.Int.Input(
                                    "target_polycount",
                                    default=300000,
                                    min=100,
                                    max=300000,
                                    display_mode=IO.NumberDisplay.number,
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option("false", []),
                    ],
                    tooltip="When set to false, returns an unprocessed triangular mesh.",
                ),
                IO.Combo.Input("symmetry_mode", options=["auto", "on", "off"]),
                IO.DynamicCombo.Input(
                    "should_texture",
                    options=[
                        IO.DynamicCombo.Option(
                            "true",
                            [
                                IO.Boolean.Input(
                                    "enable_pbr",
                                    default=False,
                                    tooltip="Generate PBR Maps (metallic, roughness, normal) "
                                    "in addition to the base color.",
                                ),
                                IO.String.Input(
                                    "texture_prompt",
                                    default="",
                                    multiline=True,
                                    tooltip="Provide a text prompt to guide the texturing process. "
                                    "Maximum 600 characters. Cannot be used at the same time as 'texture_image'.",
                                ),
                                IO.Image.Input(
                                    "texture_image",
                                    tooltip="Only one of 'texture_image' or 'texture_prompt' "
                                    "may be used at the same time.",
                                    optional=True,
                                ),
                                IO.Combo.Input(
                                    "texture_resolution",
                                    options=["2k", "4k", "8k"],
                                    tooltip="Base color texture resolution. "
                                    "Higher resolutions capture more surface detail.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option("false", []),
                    ],
                    tooltip="Determines whether textures are generated. "
                    "Setting it to false skips the texture phase and returns a mesh without textures.",
                ),
                IO.Combo.Input(
                    "pose_mode",
                    options=["", "A-pose", "T-pose"],
                    tooltip="Specify the pose mode for the generated model.",
                    advanced=True,
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
                IO.Boolean.Input(
                    "ultra_mode",
                    default=False,
                    tooltip="Run an extra refinement pass for higher-fidelity geometry with finer surface detail.",
                ),
                IO.Combo.Input(
                    "ultra_resolution",
                    options=["2k", "4k"],
                    tooltip="Resolution of the ultra pass: 2k runs it at 2048³, 4k at 4096³ for the finest "
                    "surface detail. 4k requires meshy-7.1 or latest. Used only when ultra_mode is enabled.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MESHY_TASK_ID").Output(display_name="meshy_task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
                IO.File3DFBX.Output(display_name="FBX"),
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
                    widgets=["model", "should_texture", "should_texture.texture_resolution", "ultra_mode"],
                ),
                expr="""
                (
                  $textured := widgets.should_texture = "true";
                  $resolution := $textured ? $lookup(widgets, "should_texture.texture_resolution") : "2k";
                  $credits := ($textured ? 30 : 20)
                    + ($resolution = "8k" ? 5 : 0)
                    + ((widgets.ultra_mode and widgets.model in ["meshy-7.1", "meshy-7", "latest"]) ? 5 : 0);
                  {"type":"usd","usd": $credits * 0.0572}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        image: Input.Image,
        should_remesh: InputShouldRemesh,
        symmetry_mode: str,
        should_texture: InputShouldTexture,
        pose_mode: str,
        seed: int,
        ultra_mode: bool,
        ultra_resolution: str = "2k",
    ) -> IO.NodeOutput:
        validate_ultra_mode(model, ultra_mode, ultra_resolution)
        texture = should_texture["should_texture"] == "true"
        texture_image_url = texture_prompt = None
        if texture:
            if should_texture["texture_prompt"] and should_texture["texture_image"] is not None:
                raise ValueError("texture_prompt and texture_image cannot be used at the same time")
            if should_texture["texture_prompt"]:
                validate_string(should_texture["texture_prompt"], field_name="texture_prompt", max_length=600)
                texture_prompt = should_texture["texture_prompt"]
            if should_texture["texture_image"] is not None:
                texture_image_url = (
                    await upload_images_to_comfyapi(
                        cls, should_texture["texture_image"], wait_label="Uploading texture"
                    )
                )[0]
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/meshy/openapi/v1/image-to-3d", method="POST"),
            response_model=MeshyTaskResponse,
            data=MeshyImageToModelRequest(
                image_url=(await upload_images_to_comfyapi(cls, image, wait_label="Uploading base image"))[0],
                ai_model=model,
                topology=should_remesh.get("topology", None),
                target_polycount=should_remesh.get("target_polycount", None),
                symmetry_mode=symmetry_mode,
                should_remesh=should_remesh["should_remesh"] == "true",
                should_texture=texture,
                enable_pbr=should_texture.get("enable_pbr", None),
                texture_resolution=should_texture.get("texture_resolution", None),
                pose_mode=pose_mode.lower(),
                ultra_mode=True if ultra_mode and model == "meshy-7" else None,
                geometry_resolution=ultra_resolution if ultra_mode and model != "meshy-7" else None,
                texture_prompt=texture_prompt,
                texture_image_url=texture_image_url,
                seed=seed,
            ),
        )
        task_id = response.result
        result = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/meshy/openapi/v1/image-to-3d/{task_id}"),
            response_model=MeshyModelResult,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
        )
        return IO.NodeOutput(
            f"{task_id}.glb",
            task_id,
            await download_url_to_file_3d(result.model_urls.glb, "glb", task_id=task_id),
            await download_url_to_file_3d(result.model_urls.fbx, "fbx", task_id=task_id),
        )


class MeshyMultiImageToModelNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MeshyMultiImageToModelNode",
            display_name="Meshy: Multi-Image to Model",
            category="partner/3d/Meshy",
            inputs=[
                IO.Combo.Input("model", options=["meshy-7.1", "meshy-7", "meshy-6", "latest"]),
                IO.Autogrow.Input(
                    "images",
                    template=IO.Autogrow.TemplatePrefix(IO.Image.Input("image"), prefix="image", min=2, max=4),
                ),
                IO.DynamicCombo.Input(
                    "should_remesh",
                    options=[
                        IO.DynamicCombo.Option(
                            "true",
                            [
                                IO.Combo.Input("topology", options=["triangle", "quad"]),
                                IO.Int.Input(
                                    "target_polycount",
                                    default=300000,
                                    min=100,
                                    max=300000,
                                    display_mode=IO.NumberDisplay.number,
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option("false", []),
                    ],
                    tooltip="When set to false, returns an unprocessed triangular mesh.",
                ),
                IO.Combo.Input("symmetry_mode", options=["auto", "on", "off"], advanced=True),
                IO.DynamicCombo.Input(
                    "should_texture",
                    options=[
                        IO.DynamicCombo.Option(
                            "true",
                            [
                                IO.Boolean.Input(
                                    "enable_pbr",
                                    default=False,
                                    tooltip="Generate PBR Maps (metallic, roughness, normal) "
                                    "in addition to the base color.",
                                ),
                                IO.String.Input(
                                    "texture_prompt",
                                    default="",
                                    multiline=True,
                                    tooltip="Provide a text prompt to guide the texturing process. "
                                    "Maximum 600 characters. Cannot be used at the same time as 'texture_image'.",
                                ),
                                IO.Image.Input(
                                    "texture_image",
                                    tooltip="Only one of 'texture_image' or 'texture_prompt' "
                                    "may be used at the same time.",
                                    optional=True,
                                ),
                                IO.Combo.Input(
                                    "texture_resolution",
                                    options=["2k", "4k", "8k"],
                                    tooltip="Base color texture resolution. "
                                    "Higher resolutions capture more surface detail.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option("false", []),
                    ],
                    tooltip="Determines whether textures are generated. "
                    "Setting it to false skips the texture phase and returns a mesh without textures.",
                ),
                IO.Combo.Input(
                    "pose_mode",
                    options=["", "A-pose", "T-pose"],
                    tooltip="Specify the pose mode for the generated model.",
                    advanced=True,
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
                IO.Boolean.Input(
                    "ultra_mode",
                    default=False,
                    tooltip="Run an extra refinement pass at 2048³ for higher-fidelity geometry "
                    "with finer surface detail.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MESHY_TASK_ID").Output(display_name="meshy_task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
                IO.File3DFBX.Output(display_name="FBX"),
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
                    widgets=["model", "should_texture", "should_texture.texture_resolution", "ultra_mode"],
                ),
                expr="""
                (
                  $textured := widgets.should_texture = "true";
                  $resolution := $textured ? $lookup(widgets, "should_texture.texture_resolution") : "2k";
                  $credits := ($textured ? 30 : 20)
                    + ($resolution = "8k" ? 5 : 0)
                    + ((widgets.ultra_mode and widgets.model in ["meshy-7.1", "meshy-7", "latest"]) ? 5 : 0);
                  {"type":"usd","usd": $credits * 0.0572}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        images: IO.Autogrow.Type,
        should_remesh: InputShouldRemesh,
        symmetry_mode: str,
        should_texture: InputShouldTexture,
        pose_mode: str,
        seed: int,
        ultra_mode: bool = False,
    ) -> IO.NodeOutput:
        validate_ultra_mode(model, ultra_mode)
        texture = should_texture["should_texture"] == "true"
        texture_image_url = texture_prompt = None
        if texture:
            if should_texture["texture_prompt"] and should_texture["texture_image"] is not None:
                raise ValueError("texture_prompt and texture_image cannot be used at the same time")
            if should_texture["texture_prompt"]:
                validate_string(should_texture["texture_prompt"], field_name="texture_prompt", max_length=600)
                texture_prompt = should_texture["texture_prompt"]
            if should_texture["texture_image"] is not None:
                texture_image_url = (
                    await upload_images_to_comfyapi(
                        cls, should_texture["texture_image"], wait_label="Uploading texture"
                    )
                )[0]
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/meshy/openapi/v1/multi-image-to-3d", method="POST"),
            response_model=MeshyTaskResponse,
            data=MeshyMultiImageToModelRequest(
                image_urls=await upload_images_to_comfyapi(
                    cls, list(images.values()), wait_label="Uploading base images"
                ),
                ai_model=model,
                topology=should_remesh.get("topology", None),
                target_polycount=should_remesh.get("target_polycount", None),
                symmetry_mode=symmetry_mode,
                should_remesh=should_remesh["should_remesh"] == "true",
                should_texture=texture,
                enable_pbr=should_texture.get("enable_pbr", None),
                texture_resolution=should_texture.get("texture_resolution", None),
                pose_mode=pose_mode.lower(),
                ultra_mode=True if ultra_mode and model == "meshy-7" else None,
                geometry_resolution="2k" if ultra_mode and model != "meshy-7" else None,
                texture_prompt=texture_prompt,
                texture_image_url=texture_image_url,
                seed=seed,
            ),
        )
        task_id = response.result
        result = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/meshy/openapi/v1/multi-image-to-3d/{task_id}"),
            response_model=MeshyModelResult,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
        )
        return IO.NodeOutput(
            f"{task_id}.glb",
            task_id,
            await download_url_to_file_3d(result.model_urls.glb, "glb", task_id=task_id),
            await download_url_to_file_3d(result.model_urls.fbx, "fbx", task_id=task_id),
        )


class MeshyTextToModelSmartTopologyNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MeshyTextToModelSmartTopologyNode",
            display_name="Meshy: Text to Model (Smart Topology)",
            category="partner/3d/Meshy",
            description="Generate a game-ready triangle mesh directly at a target face count.",
            search_aliases=["meshy t2.5", "low poly"],
            inputs=[
                IO.Combo.Input("model", options=["meshy-t2.5"]),
                IO.String.Input("prompt", multiline=True, default=""),
                IO.Int.Input(
                    "target_polycount",
                    default=15000,
                    min=100,
                    max=30000,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Approximate number of triangles in the generated mesh.",
                ),
                IO.Combo.Input("pose_mode", options=["", "A-pose", "T-pose"], advanced=True),
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
                IO.Custom("MESHY_TASK_ID").Output(display_name="meshy_task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
                IO.File3DFBX.Output(display_name="FBX"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(expr='{"type":"usd","usd":0.286}'),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        target_polycount: int,
        pose_mode: str,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, field_name="prompt", min_length=1, max_length=800)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/meshy/openapi/v2/text-to-3d", method="POST"),
            response_model=MeshyTaskResponse,
            data=MeshySmartTopologyTextToModelRequest(
                prompt=prompt,
                ai_model=model,
                target_polycount=target_polycount,
                pose_mode=pose_mode.lower(),
                seed=seed,
            ),
        )
        task_id = response.result
        result = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/meshy/openapi/v2/text-to-3d/{task_id}"),
            response_model=MeshyModelResult,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
        )
        return IO.NodeOutput(
            task_id,
            await download_url_to_file_3d(result.model_urls.glb, "glb"),
            await download_url_to_file_3d(result.model_urls.fbx, "fbx"),
        )


class MeshyImageToModelSmartTopologyNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MeshyImageToModelSmartTopologyNode",
            display_name="Meshy: Image to Model (Smart Topology)",
            category="partner/3d/Meshy",
            description="Generate a game-ready triangle mesh directly at a target face count.",
            search_aliases=["meshy t2.5", "low poly"],
            inputs=[
                IO.Combo.Input("model", options=["meshy-t2.5"]),
                IO.Image.Input("image"),
                IO.Int.Input(
                    "target_polycount",
                    default=15000,
                    min=100,
                    max=30000,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Approximate number of triangles in the generated mesh.",
                ),
                IO.DynamicCombo.Input(
                    "should_texture",
                    options=[
                        IO.DynamicCombo.Option(
                            "true",
                            [
                                IO.Boolean.Input(
                                    "enable_pbr",
                                    default=False,
                                    tooltip="Generate PBR Maps (metallic, roughness, normal) "
                                    "in addition to the base color.",
                                ),
                                IO.String.Input(
                                    "texture_prompt",
                                    default="",
                                    multiline=True,
                                    tooltip="Provide a text prompt to guide the texturing process. "
                                    "Maximum 800 characters. Cannot be used at the same time as 'texture_image'.",
                                ),
                                IO.Image.Input(
                                    "texture_image",
                                    tooltip="Only one of 'texture_image' or 'texture_prompt' "
                                    "may be used at the same time.",
                                    optional=True,
                                ),
                                IO.Combo.Input("texture_resolution", options=["2k", "4k", "8k"]),
                            ],
                        ),
                        IO.DynamicCombo.Option("false", []),
                    ],
                ),
                IO.Combo.Input("pose_mode", options=["", "A-pose", "T-pose"], advanced=True),
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
                IO.Custom("MESHY_TASK_ID").Output(display_name="meshy_task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
                IO.File3DFBX.Output(display_name="FBX"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["should_texture", "should_texture.texture_resolution"]),
                expr="""
                (
                  $credits := widgets.should_texture = "true"
                    ? ($lookup(widgets, "should_texture.texture_resolution") = "8k" ? 20 : 15)
                    : 5;
                  {"type":"usd","usd": $credits * 0.0572}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        image: Input.Image,
        target_polycount: int,
        should_texture: InputShouldTexture,
        pose_mode: str,
        seed: int,
    ) -> IO.NodeOutput:
        texture = should_texture["should_texture"] == "true"
        texture_image_url = texture_prompt = None
        if texture:
            if should_texture["texture_prompt"] and should_texture["texture_image"] is not None:
                raise ValueError("texture_prompt and texture_image cannot be used at the same time")
            if should_texture["texture_prompt"]:
                validate_string(should_texture["texture_prompt"], field_name="texture_prompt", max_length=800)
                texture_prompt = should_texture["texture_prompt"]
            if should_texture["texture_image"] is not None:
                texture_image_url = await upload_image_to_comfyapi(
                    cls, should_texture["texture_image"], wait_label="Uploading texture"
                )
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/meshy/openapi/v1/image-to-3d", method="POST"),
            response_model=MeshyTaskResponse,
            data=MeshySmartTopologyImageToModelRequest(
                image_url=await upload_image_to_comfyapi(cls, image, wait_label="Uploading base image"),
                ai_model=model,
                target_polycount=target_polycount,
                should_texture=texture,
                enable_pbr=should_texture.get("enable_pbr", None),
                texture_resolution=should_texture.get("texture_resolution", None),
                texture_prompt=texture_prompt,
                texture_image_url=texture_image_url,
                pose_mode=pose_mode.lower(),
                seed=seed,
            ),
        )
        task_id = response.result
        result = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/meshy/openapi/v1/image-to-3d/{task_id}"),
            response_model=MeshyModelResult,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
        )
        return IO.NodeOutput(
            task_id,
            await download_url_to_file_3d(result.model_urls.glb, "glb"),
            await download_url_to_file_3d(result.model_urls.fbx, "fbx"),
        )


class MeshyRigModelNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MeshyRigModelNode",
            display_name="Meshy: Rig Model",
            category="partner/3d/Meshy",
            description="Provides a rigged character in standard formats. "
            "Auto-rigging is currently not suitable for untextured meshes, non-humanoid assets, "
            "or humanoid assets with unclear limb and body structure.",
            inputs=[
                IO.Custom("MESHY_TASK_ID").Input("meshy_task_id"),
                IO.Float.Input(
                    "height_meters",
                    min=0.1,
                    max=15.0,
                    default=1.7,
                    tooltip="The approximate height of the character model in meters. "
                    "This aids in scaling and rigging accuracy.",
                ),
                IO.Image.Input(
                    "texture_image",
                    tooltip="The model's UV-unwrapped base color texture image.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MESHY_RIGGED_TASK_ID").Output(display_name="rig_task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
                IO.File3DFBX.Output(display_name="FBX"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd": 0.286}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        meshy_task_id: str,
        height_meters: float,
        texture_image: Input.Image | None = None,
    ) -> IO.NodeOutput:
        texture_image_url = None
        if texture_image is not None:
            texture_image_url = (await upload_images_to_comfyapi(cls, texture_image, wait_label="Uploading texture"))[0]
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/meshy/openapi/v1/rigging", method="POST"),
            response_model=MeshyTaskResponse,
            data=MeshyRiggingRequest(
                input_task_id=meshy_task_id,
                height_meters=height_meters,
                texture_image_url=texture_image_url,
            ),
        )
        task_id = response.result
        result = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/meshy/openapi/v1/rigging/{task_id}"),
            response_model=MeshyRiggedResult,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
        )
        return IO.NodeOutput(
            f"{task_id}.glb",
            task_id,
            await download_url_to_file_3d(result.result.rigged_character_glb_url, "glb", task_id=task_id),
            await download_url_to_file_3d(result.result.rigged_character_fbx_url, "fbx", task_id=task_id),
        )


class MeshyAnimateModelNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MeshyAnimateModelNode",
            display_name="Meshy: Animate Model",
            category="partner/3d/Meshy",
            description="Apply a specific animation action to a previously rigged character.",
            inputs=[
                IO.Custom("MESHY_RIGGED_TASK_ID").Input("rig_task_id"),
                IO.Int.Input(
                    "action_id",
                    default=0,
                    min=0,
                    max=696,
                    tooltip="Visit https://docs.meshy.ai/en/api/animation-library for a list of available values.",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.File3DGLB.Output(display_name="GLB"),
                IO.File3DFBX.Output(display_name="FBX"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd": 0.1716}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        rig_task_id: str,
        action_id: int,
    ) -> IO.NodeOutput:
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/meshy/openapi/v1/animations", method="POST"),
            response_model=MeshyTaskResponse,
            data=MeshyAnimationRequest(
                rig_task_id=rig_task_id,
                action_id=action_id,
            ),
        )
        task_id = response.result
        result = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/meshy/openapi/v1/animations/{task_id}"),
            response_model=MeshyAnimationResult,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
        )
        return IO.NodeOutput(
            f"{task_id}.glb",
            await download_url_to_file_3d(result.result.animation_glb_url, "glb", task_id=task_id),
            await download_url_to_file_3d(result.result.animation_fbx_url, "fbx", task_id=task_id),
        )


class MeshyTextureNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MeshyTextureNode",
            display_name="Meshy: Texture Model",
            category="partner/3d/Meshy",
            inputs=[
                IO.Combo.Input("model", options=["meshy-7", "meshy-6", "latest"]),
                IO.Custom("MESHY_TASK_ID").Input("meshy_task_id"),
                IO.Boolean.Input(
                    "enable_original_uv",
                    default=True,
                    tooltip="Use the original UV of the model instead of generating new UVs. "
                    "When enabled, Meshy preserves existing textures from the uploaded model. "
                    "If the model has no original UV, the quality of the output might not be as good.",
                    advanced=True,
                ),
                IO.Boolean.Input("pbr", default=False, advanced=True),
                IO.String.Input(
                    "text_style_prompt",
                    default="",
                    multiline=True,
                    tooltip="Describe your desired texture style of the object using text. Maximum 600 characters."
                    "Maximum 600 characters. Cannot be used at the same time as 'image_style'.",
                ),
                IO.Image.Input(
                    "image_style",
                    optional=True,
                    tooltip="A 2d image to guide the texturing process. "
                    "Can not be used at the same time with 'text_style_prompt'.",
                ),
                IO.Combo.Input(
                    "texture_resolution",
                    options=["2k", "4k", "8k"],
                    tooltip="Base color texture resolution. Higher resolutions capture more surface detail.",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MESHY_TASK_ID").Output(display_name="meshy_task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
                IO.File3DFBX.Output(display_name="FBX"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["texture_resolution"]),
                expr="""
                (
                  $credits := widgets.texture_resolution = "8k" ? 15 : 10;
                  {"type":"usd","usd": $credits * 0.0572}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        meshy_task_id: str,
        enable_original_uv: bool,
        pbr: bool,
        text_style_prompt: str,
        texture_resolution: str,
        image_style: Input.Image | None = None,
    ) -> IO.NodeOutput:
        if text_style_prompt and image_style is not None:
            raise ValueError("text_style_prompt and image_style cannot be used at the same time")
        if not text_style_prompt and image_style is None:
            raise ValueError("Either text_style_prompt or image_style is required")
        image_style_url = None
        if image_style is not None:
            image_style_url = (await upload_images_to_comfyapi(cls, image_style, wait_label="Uploading style"))[0]
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/meshy/openapi/v1/retexture", method="POST"),
            response_model=MeshyTaskResponse,
            data=MeshyTextureRequest(
                input_task_id=meshy_task_id,
                ai_model=model,
                enable_original_uv=enable_original_uv,
                enable_pbr=pbr,
                texture_resolution=texture_resolution,
                text_style_prompt=text_style_prompt if text_style_prompt else None,
                image_style_url=image_style_url,
            ),
        )
        task_id = response.result
        result = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/meshy/openapi/v1/retexture/{task_id}"),
            response_model=MeshyModelResult,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
        )
        return IO.NodeOutput(
            f"{task_id}.glb",
            task_id,
            await download_url_to_file_3d(result.model_urls.glb, "glb", task_id=task_id),
            await download_url_to_file_3d(result.model_urls.fbx, "fbx", task_id=task_id),
        )


class MeshyTextureMultiViewNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MeshyTextureMultiViewNode",
            display_name="Meshy: Texture Model (Multi-View)",
            category="partner/3d/Meshy",
            description="Texture a previously created model using 1 to 4 reference views of the same object.",
            inputs=[
                IO.Combo.Input("model", options=["meshy-7"]),
                IO.Custom("MESHY_TASK_ID").Input("meshy_task_id"),
                IO.Autogrow.Input(
                    "multiview_images",
                    template=IO.Autogrow.TemplatePrefix(IO.Image.Input("image"), prefix="image", min=1, max=4),
                    tooltip="Reference views of the same object. The first image is the primary (front) view; "
                    "the order of the remaining views does not matter.",
                ),
                IO.Boolean.Input(
                    "enable_original_uv",
                    default=True,
                    tooltip="Use the original UV of the model instead of generating new UVs. "
                    "When enabled, Meshy preserves existing textures from the uploaded model. "
                    "If the model has no original UV, the quality of the output might not be as good.",
                    advanced=True,
                ),
                IO.Boolean.Input("pbr", default=False, advanced=True),
                IO.Combo.Input(
                    "texture_resolution",
                    options=["2k", "4k", "8k"],
                    tooltip="Base color texture resolution. Higher resolutions capture more surface detail.",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MESHY_TASK_ID").Output(display_name="meshy_task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
                IO.File3DFBX.Output(display_name="FBX"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["texture_resolution"]),
                expr="""
                (
                  $credits := widgets.texture_resolution = "8k" ? 15 : 10;
                  {"type":"usd","usd": $credits * 0.0572}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        meshy_task_id: str,
        multiview_images: IO.Autogrow.Type,
        enable_original_uv: bool,
        pbr: bool,
        texture_resolution: str,
    ) -> IO.NodeOutput:
        reference_views = list(multiview_images.values())
        view_count = sum(v.shape[0] if len(v.shape) > 3 else 1 for v in reference_views)
        if not 1 <= view_count <= 4:
            raise ValueError("multiview_images must contain 1 to 4 images")
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/meshy/openapi/v1/retexture", method="POST"),
            response_model=MeshyTaskResponse,
            data=MeshyTextureRequest(
                input_task_id=meshy_task_id,
                ai_model=model,
                enable_original_uv=enable_original_uv,
                enable_pbr=pbr,
                texture_resolution=texture_resolution,
                multiview_image_urls=await upload_images_to_comfyapi(
                    cls, reference_views, max_images=4, wait_label="Uploading reference views"
                ),
            ),
        )
        task_id = response.result
        result = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/meshy/openapi/v1/retexture/{task_id}"),
            response_model=MeshyModelResult,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
        )
        return IO.NodeOutput(
            f"{task_id}.glb",
            task_id,
            await download_url_to_file_3d(result.model_urls.glb, "glb", task_id=task_id),
            await download_url_to_file_3d(result.model_urls.fbx, "fbx", task_id=task_id),
        )


class MeshyTextureN1Node(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MeshyTextureN1Node",
            display_name="Meshy: Texture Model (N1)",
            category="partner/3d/Meshy",
            description="Texture a previously created model with Meshy N1 from 1 to 4 reference views of the same object.",
            search_aliases=["meshy n1"],
            inputs=[
                IO.Combo.Input("model", options=["meshy-n1"]),
                IO.Custom("MESHY_TASK_ID").Input("meshy_task_id"),
                IO.Image.Input(
                    "front_view",
                    tooltip="The view Meshy treats as the front. One image textures the front and leaves the back "
                    "plain; add back and side views in extra_views for a faithful back.",
                ),
                IO.Autogrow.Input(
                    "extra_views",
                    template=IO.Autogrow.TemplatePrefix(IO.Image.Input("view"), prefix="view", min=0, max=3),
                    optional=True,
                    tooltip="Further views of the same object, in any order. Together with front_view at most 4 images.",
                ),
                IO.Boolean.Input(
                    "enable_original_uv",
                    default=True,
                    tooltip="Use the original UV of the model instead of generating new UVs. "
                    "When enabled, Meshy preserves existing textures from the uploaded model. "
                    "If the model has no original UV, the quality of the output might not be as good.",
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Custom("MESHY_TASK_ID").Output(display_name="meshy_task_id"),
                IO.File3DGLB.Output(display_name="GLB"),
                IO.File3DFBX.Output(display_name="FBX"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(expr='{"type":"usd","usd":0.572}'),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        meshy_task_id: str,
        front_view: Input.Image,
        extra_views: IO.Autogrow.Type,
        enable_original_uv: bool,
    ) -> IO.NodeOutput:
        views = [front_view, *extra_views.values()]
        if sum(v.shape[0] for v in views) > 4:
            raise ValueError("front_view and extra_views must contain at most 4 images in total")
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/meshy/openapi/v1/retexture", method="POST"),
            response_model=MeshyTaskResponse,
            data=MeshyTextureRequest(
                input_task_id=meshy_task_id,
                ai_model=model,
                enable_original_uv=enable_original_uv,
                enable_pbr=False,
                texture_resolution="2k",
                multiview_image_urls=await upload_images_to_comfyapi(
                    cls, views, max_images=4, wait_label="Uploading reference views"
                ),
            ),
        )
        task_id = response.result
        result = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/meshy/openapi/v1/retexture/{task_id}"),
            response_model=MeshyModelResult,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
        )
        return IO.NodeOutput(
            task_id,
            await download_url_to_file_3d(result.model_urls.glb, "glb"),
            await download_url_to_file_3d(result.model_urls.fbx, "fbx"),
        )


class MeshyImageToReliefNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MeshyImageToReliefNode",
            display_name="Meshy: Image to Relief",
            category="partner/3d/Meshy",
            description="Turn a photo into a bas-relief model and a 16-bit depth map for 3D printing, laser engraving "
            "and CNC carving. Only the connected outputs are generated and downloaded; with no model output "
            "connected the task returns just the depth map.",
            search_aliases=["bas-relief", "depth map", "emboss"],
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="Dark artwork on a light background is raised. Transparency is dropped before upload, "
                    "so flatten a transparent logo onto a white background first.",
                ),
                IO.DynamicCombo.Input(
                    "relief_type",
                    options=[
                        IO.DynamicCombo.Option(
                            "low",
                            [IO.Float.Input("carving_depth_mm", default=3.0, min=0.1, max=60.0, step=0.1)],
                        ),
                        IO.DynamicCombo.Option(
                            "high",
                            [IO.Float.Input("carving_depth_mm", default=6.0, min=0.1, max=60.0, step=0.1)],
                        ),
                    ],
                    tooltip="low: shallow carving for laser engraving, CNC and plaques. "
                    "high: deeper carving for 3D-printed pieces.",
                ),
                IO.Float.Input(
                    "size_mm",
                    default=60.0,
                    min=20.0,
                    max=250.0,
                    step=1.0,
                    tooltip="Length of the photo's longer side on the model, measured flat before any bend.",
                ),
                IO.Combo.Input(
                    "carving_mode",
                    options=["emboss", "engrave"],
                    tooltip="emboss raises the nearer parts of the photo, engrave cuts them in.",
                ),
                IO.DynamicCombo.Input(
                    "base",
                    options=[
                        IO.DynamicCombo.Option(
                            "plate",
                            [IO.Float.Input("base_thickness_mm", default=1.0, min=0.1, max=10.0, step=0.1)],
                        ),
                        IO.DynamicCombo.Option(
                            "none",
                            [
                                IO.Float.Input(
                                    "removal_threshold",
                                    default=0.01,
                                    min=0.01,
                                    max=1.0,
                                    step=0.01,
                                    tooltip="Depth below which the background is cut away, on the 0 to 1 depth map. "
                                    "0.3 keeps a bust whole; 0.8 breaks it into fragments.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="none cuts the background away and builds no plate.",
                ),
                IO.DynamicCombo.Input(
                    "bend",
                    options=[
                        IO.DynamicCombo.Option("off", []),
                        IO.DynamicCombo.Option("convex", relief_bend_inputs()),
                        IO.DynamicCombo.Option("concave", relief_bend_inputs()),
                    ],
                    tooltip="convex bulges the front outward, concave curves it inward.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; results are non-deterministic regardless "
                    "of seed. Change it to request an output that was connected after the last run.",
                ),
                IO.Int.Input(
                    "mesh_resolution",
                    default=1024,
                    min=512,
                    max=2048,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Mesh points along the photo's longer side. At 1024 the GLB is about 50 MB and the STL "
                    "about 105 MB, at 2048 about 200 MB and 420 MB; a bend doubles them.",
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "color_mapping",
                    default=False,
                    tooltip="Color the GLB, OBJ, FBX and USDZ files with the photo. STL and 3MF stay white.",
                    advanced=True,
                ),
            ],
            outputs=[
                IO.File3DGLB.Output(display_name="GLB"),
                IO.File3DSTL.Output(display_name="STL"),
                IO.File3DOBJ.Output(display_name="OBJ"),
                IO.File3DFBX.Output(display_name="FBX"),
                IO.File3DUSDZ.Output(display_name="USDZ"),
                IO.File3DAny.Output(display_name="3MF"),
                IO.Image.Output(
                    display_name="depth_map",
                    tooltip="Depth map with the full 16-bit range; Save Image (Advanced) writes it as a 16-bit PNG.",
                ),
                IO.Image.Output(display_name="preview"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
                IO.Hidden.dynprompt,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(expr='{"type":"usd","usd":1.144}'),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        relief_type: dict,
        size_mm: float,
        carving_mode: str,
        base: dict,
        bend: dict,
        seed: int,
        mesh_resolution: int,
        color_mapping: bool,
    ) -> IO.NodeOutput:
        wired = [fmt for i, fmt in enumerate(RELIEF_FORMATS) if get_output_consumers(cls, i)]
        preview = bool(get_output_consumers(cls, 7))
        formats = wired or (["glb"] if preview else [])
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/meshy/openapi/v1/relief", method="POST"),
            response_model=MeshyTaskResponse,
            data=MeshyReliefRequest(
                image_url=await upload_image_to_comfyapi(cls, image, wait_label="Uploading image"),
                variant=relief_type["relief_type"],
                should_generate_model=bool(formats),
                size_mm=size_mm,
                carving_mode=carving_mode,
                carving_depth_mm=relief_type["carving_depth_mm"],
                remove_base=base["base"] == "none",
                removal_threshold=base.get("removal_threshold"),
                base_thickness_mm=base.get("base_thickness_mm"),
                bend_mode=bend["bend"],
                bend_axis=bend.get("bend_axis"),
                bend_angle=bend.get("bend_angle"),
                mesh_resolution=mesh_resolution,
                target_formats=formats,
                color_mapping=color_mapping,
            ),
        )
        task_id = response.result
        result = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/meshy/openapi/v1/relief/{task_id}"),
            response_model=MeshyReliefResult,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
        )
        urls = result.model_urls.model_dump(by_alias=True)
        files = []
        for fmt in RELIEF_FORMATS:
            files.append(await download_url_to_file_3d(urls[fmt], fmt) if fmt in wired else None)
        depth_map = None
        if get_output_consumers(cls, 6):
            data = BytesIO()
            await download_url_to_bytesio(result.depth_map_url, data)
            depth_map = depth_map_to_tensor(data)
        return IO.NodeOutput(
            *files,
            depth_map,
            await download_url_to_image_tensor(result.thumbnail_url) if preview else None,
        )


class MeshyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            MeshyTextToModelNode,
            MeshyRefineNode,
            MeshyImageToModelNode,
            MeshyMultiImageToModelNode,
            MeshyTextToModelSmartTopologyNode,
            MeshyImageToModelSmartTopologyNode,
            MeshyRigModelNode,
            MeshyAnimateModelNode,
            MeshyTextureNode,
            MeshyTextureMultiViewNode,
            MeshyTextureN1Node,
            MeshyImageToReliefNode,
        ]


async def comfy_entrypoint() -> MeshyExtension:
    return MeshyExtension()
