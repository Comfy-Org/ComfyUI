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
    MeshyRiggedResult,
    MeshyRiggingRequest,
    MeshyTaskResponse,
    MeshyTextToModelRequest,
    MeshyTextureRequest,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_file_3d,
    poll_op,
    sync_op,
    upload_images_to_comfyapi,
    validate_string,
)


class MeshyTextToModelNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MeshyTextToModelNode",
            display_name="Meshy: Text to Model",
            category="partner/3d/Meshy",
            inputs=[
                IO.Combo.Input("model", options=["latest"]),
                IO.String.Input("prompt", multiline=True, default=""),
                IO.Combo.Input("style", options=["realistic", "sculpture"]),
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
                expr="""{"type":"usd","usd":0.8}""",
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
    ) -> IO.NodeOutput:
        validate_string(prompt, field_name="prompt", min_length=1, max_length=600)
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
                IO.Combo.Input("model", options=["latest"]),
                IO.Custom("MESHY_TASK_ID").Input("meshy_task_id"),
                IO.Boolean.Input(
                    "enable_pbr",
                    default=False,
                    tooltip="Generate PBR Maps (metallic, roughness, normal) in addition to the base color. "
                    "Note: this should be set to false when using Sculpture style, "
                    "as Sculpture style generates its own set of PBR maps.",
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
                expr="""{"type":"usd","usd":0.4}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        meshy_task_id: str,
        enable_pbr: bool,
        texture_prompt: str,
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
                IO.Combo.Input("model", options=["latest"]),
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
                depends_on=IO.PriceBadgeDepends(widgets=["should_texture"]),
                expr="""
                (
                  $prices := {"true": 1.2, "false": 0.8};
                  {"type":"usd","usd": $lookup($prices, widgets.should_texture)}
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
    ) -> IO.NodeOutput:
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
                pose_mode=pose_mode.lower(),
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
                IO.Combo.Input("model", options=["latest"]),
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
                depends_on=IO.PriceBadgeDepends(widgets=["should_texture"]),
                expr="""
                (
                  $prices := {"true": 0.6, "false": 0.2};
                  {"type":"usd","usd": $lookup($prices, widgets.should_texture)}
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
    ) -> IO.NodeOutput:
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
                pose_mode=pose_mode.lower(),
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
                expr="""{"type":"usd","usd":0.2}""",
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
                expr="""{"type":"usd","usd":0.12}""",
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
                IO.Combo.Input("model", options=["latest"]),
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
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.Custom("MODEL_TASK_ID").Output(display_name="meshy_task_id"),
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
                expr="""{"type":"usd","usd":0.4}""",
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


class MeshyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            MeshyTextToModelNode,
            MeshyRefineNode,
            MeshyImageToModelNode,
            MeshyMultiImageToModelNode,
            MeshyRigModelNode,
            MeshyAnimateModelNode,
            MeshyTextureNode,
        ]


async def comfy_entrypoint() -> MeshyExtension:
    return MeshyExtension()
