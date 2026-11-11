import torch
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.luma import (
    LUMA_KEYFRAME_MODE_FRACTION,
    LUMA_KEYFRAME_MODE_SECONDS,
    Luma2Generation,
    Luma2GenerationRequest,
    Luma2ImageRef,
    Luma2VideoEdit,
    Luma2VideoOptions,
    LumaIO,
    LumaRay32KeyframeChain,
    LumaRay32KeyframeItem,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_image_tensor,
    download_url_to_video_output,
    poll_op,
    sync_op,
    upload_image_to_comfyapi,
    upload_video_to_comfyapi,
    validate_string,
)


def _luma2_uni1_common_inputs(max_image_refs: int) -> list:
    return [
        IO.Combo.Input(
            "style",
            options=["auto", "manga"],
            default="auto",
            tooltip="Style preset. 'auto' picks based on the prompt; "
            "'manga' applies a manga/anime aesthetic and requires a portrait "
            "aspect ratio (2:3, 9:16, 1:2, 1:3).",
        ),
        IO.Boolean.Input(
            "web_search",
            default=False,
            tooltip="Search the web for visual references before generating.",
        ),
        IO.Autogrow.Input(
            "image_ref",
            template=IO.Autogrow.TemplateNames(
                IO.Image.Input("image"),
                names=[f"image_{i}" for i in range(1, max_image_refs + 1)],
                min=0,
            ),
            optional=True,
            tooltip=f"Up to {max_image_refs} reference images for style/content guidance.",
        ),
    ]


async def _luma2_upload_image_refs(
    cls: type[IO.ComfyNode],
    refs: dict | None,
    max_count: int,
) -> list[Luma2ImageRef] | None:
    if not refs:
        return None
    out: list[Luma2ImageRef] = []
    for key in refs:
        url = await upload_image_to_comfyapi(cls, refs[key])
        out.append(Luma2ImageRef(url=url))
    if len(out) > max_count:
        raise ValueError(f"Maximum {max_count} reference images are allowed.")
    return out or None


async def _luma2_submit_and_poll(
    cls: type[IO.ComfyNode],
    request: Luma2GenerationRequest,
    *,
    estimated_duration: int | None = None,
) -> Luma2Generation:
    """Submit a Luma Agents generation and poll until done; returns the completed generation."""
    initial = await sync_op(
        cls,
        ApiEndpoint(path="/proxy/luma_2/generations", method="POST"),
        response_model=Luma2Generation,
        data=request,
    )
    if not initial.id:
        raise RuntimeError("Luma API did not return a generation id.")
    final = await poll_op(
        cls,
        ApiEndpoint(path=f"/proxy/luma_2/generations/{initial.id}", method="GET"),
        response_model=Luma2Generation,
        status_extractor=lambda r: r.state,
        progress_extractor=lambda r: None,
        estimated_duration=estimated_duration,
    )
    if not final.output or not final.output[0].url:
        msg = final.failure_reason or "no output returned"
        if final.failure_code:
            msg = f"{msg} [{final.failure_code}]"
        raise RuntimeError(f"Luma generation failed: {msg}")
    return final


class LumaImageNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="LumaImageNode2",
            display_name="Luma UNI-1 Image",
            category="partner/image/Luma",
            description="Generate images from text using the Luma UNI-1 model.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text description of the desired image. 1–6000 characters.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "uni-1",
                            [
                                IO.Combo.Input(
                                    "aspect_ratio",
                                    options=[
                                        "auto",
                                        "3:1",
                                        "2:1",
                                        "16:9",
                                        "3:2",
                                        "1:1",
                                        "2:3",
                                        "9:16",
                                        "1:2",
                                        "1:3",
                                    ],
                                    default="auto",
                                    tooltip="Output image aspect ratio. 'auto' lets "
                                    "the model pick based on the prompt.",
                                ),
                                *_luma2_uni1_common_inputs(max_image_refs=9),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "uni-1-max",
                            [
                                IO.Combo.Input(
                                    "aspect_ratio",
                                    options=[
                                        "auto",
                                        "3:1",
                                        "2:1",
                                        "16:9",
                                        "3:2",
                                        "1:1",
                                        "2:3",
                                        "9:16",
                                        "1:2",
                                        "1:3",
                                    ],
                                    default="auto",
                                    tooltip="Output image aspect ratio. 'auto' lets "
                                    "the model pick based on the prompt.",
                                ),
                                *_luma2_uni1_common_inputs(max_image_refs=9),
                            ],
                        ),
                    ],
                    tooltip="Model to use for generation.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[IO.Image.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model"], input_groups=["model.image_ref"]),
                expr="""
                (
                  $m := widgets.model;
                  $refs := $lookup(inputGroups, "model.image_ref");
                  $base := $m = "uni-1-max" ? 0.1 : 0.0404;
                  {"type":"usd","usd": $round($base + 0.003 * $refs, 4)}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: dict,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=6000)
        aspect_ratio = model["aspect_ratio"]
        style = model["style"]
        allowed_manga_ratios = {"2:3", "9:16", "1:2", "1:3"}
        if style == "manga" and aspect_ratio != "auto" and aspect_ratio not in allowed_manga_ratios:
            raise ValueError(
                f"'manga' style requires a portrait aspect ratio "
                f"({', '.join(sorted(allowed_manga_ratios))}) or 'auto'; got '{aspect_ratio}'."
            )
        request = Luma2GenerationRequest(
            prompt=prompt,
            model=model["model"],
            type="image",
            aspect_ratio=aspect_ratio if aspect_ratio != "auto" else None,
            style=style if style != "auto" else None,
            output_format="png",
            web_search=model["web_search"],
            image_ref=await _luma2_upload_image_refs(cls, model.get("image_ref"), max_count=9),
        )
        final = await _luma2_submit_and_poll(cls, request)
        return IO.NodeOutput(await download_url_to_image_tensor(final.output[0].url))


class LumaImageEditNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="LumaImageEditNode2",
            display_name="Luma UNI-1 Image Edit",
            category="partner/image/Luma",
            description="Edit an existing image with a text prompt using the Luma UNI-1 model.",
            inputs=[
                IO.Image.Input(
                    "source",
                    tooltip="Source image to edit.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Description of the desired edit. 1–6000 characters.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "uni-1",
                            _luma2_uni1_common_inputs(max_image_refs=8),
                        ),
                        IO.DynamicCombo.Option(
                            "uni-1-max",
                            _luma2_uni1_common_inputs(max_image_refs=8),
                        ),
                    ],
                    tooltip="Model to use for editing.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[IO.Image.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model"], input_groups=["model.image_ref"]),
                expr="""
                (
                  $m := widgets.model;
                  $refs := $lookup(inputGroups, "model.image_ref");
                  $base := $m = "uni-1-max" ? 0.103 : 0.0434;
                  {"type":"usd","usd": $round($base + 0.003 * $refs, 4)}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        source: Input.Image,
        prompt: str,
        model: dict,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=6000)
        request = Luma2GenerationRequest(
            prompt=prompt,
            model=model["model"],
            type="image_edit",
            source=Luma2ImageRef(url=await upload_image_to_comfyapi(cls, source)),
            style=model["style"] if model["style"] != "auto" else None,
            output_format="png",
            web_search=model["web_search"],
            image_ref=await _luma2_upload_image_refs(cls, model.get("image_ref"), max_count=8),
        )
        final = await _luma2_submit_and_poll(cls, request)
        return IO.NodeOutput(await download_url_to_image_tensor(final.output[0].url))


_BADGE_RAY32_VIDEO = IO.PriceBadge(
    depends_on=IO.PriceBadgeDepends(widgets=["resolution", "duration"]),
    expr="""
    (
      $p := {
        "360p": {"5s": 0.06, "10s": 0.18},
        "540p": {"5s": 0.15, "10s": 0.45},
        "720p": {"5s": 0.3, "10s": 0.9},
        "1080p": {"5s": 1.2, "10s": 3.6}
      };
      {"type": "usd", "usd": $lookup($lookup($p, widgets.resolution), widgets.duration)}
    )
    """,
)

_BADGE_RAY32_VIDEO_5S = IO.PriceBadge(
    depends_on=IO.PriceBadgeDepends(widgets=["resolution"]),
    expr="""
    (
      $p := {"360p": 0.06, "540p": 0.15, "720p": 0.3, "1080p": 1.2};
      {"type": "usd", "usd": $lookup($p, widgets.resolution)}
    )
    """,
)

_BADGE_RAY32_EDIT = IO.PriceBadge(
    depends_on=IO.PriceBadgeDepends(widgets=["resolution"]),
    expr="""
    (
      $p := {
        "360p": {"min": 0.54, "max": 1.08},
        "540p": {"min": 0.72, "max": 1.44},
        "720p": {"min": 1.08, "max": 2.16},
        "1080p": {"min": 2.16, "max": 4.32}
      };
      $r := $lookup($p, widgets.resolution);
      {"type": "range_usd", "min_usd": $r.min, "max_usd": $r.max, "format": {"note": "(by source length)"}}
    )
    """,
)

_BADGE_RAY32_REFRAME = IO.PriceBadge(
    depends_on=IO.PriceBadgeDepends(widgets=["resolution"]),
    expr="""
    (
      $p := {"360p": 0.03, "540p": 0.06, "720p": 0.12, "1080p": 0.36};
      {"type": "usd", "usd": $lookup($p, widgets.resolution), "format": {"suffix": "/second"}}
    )
    """,
)


def _ray32_seed_input() -> IO.Input:
    return IO.Int.Input(
        "seed",
        default=0,
        min=0,
        max=0xFFFFFFFFFFFFFFFF,
        control_after_generate=True,
        tooltip="Seed to determine if node should re-run; results are nondeterministic regardless of seed.",
    )


async def _ray32_generate(cls: type[IO.ComfyNode], request: Luma2GenerationRequest) -> IO.NodeOutput:
    """Run a ray-3.2 generation and return (video, generation_id)."""
    final = await _luma2_submit_and_poll(cls, request, estimated_duration=120)
    video = await download_url_to_video_output(final.output[0].url)
    return IO.NodeOutput(video, final.id or "")


class LumaRay32TextToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="LumaRay32TextToVideoNode",
            display_name="Luma Ray 3.2 Text to Video",
            category="partner/video/Luma",
            description="Generate a video from a text prompt using Luma's Ray 3.2 model.",
            inputs=[
                IO.String.Input("prompt", multiline=True, default="", tooltip="Text prompt for the video generation."),
                IO.Combo.Input("aspect_ratio", options=["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]),
                IO.Combo.Input("resolution", options=["360p", "540p", "720p", "1080p"], default="720p"),
                IO.Combo.Input("duration", options=["5s", "10s"]),
                IO.Boolean.Input(
                    "loop",
                    default=False,
                    tooltip="Make the video loop seamlessly. Only available with 5s duration.",
                ),
                _ray32_seed_input(),
            ],
            outputs=[IO.Video.Output(), IO.String.Output(display_name="generation_id")],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=_BADGE_RAY32_VIDEO,
        )

    @classmethod
    async def execute(
        cls, prompt: str, aspect_ratio: str, resolution: str, duration: str, loop: bool, seed: int
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1, max_length=6000)
        if loop and duration == "10s":
            raise ValueError("Looping is only available with 5s duration on Ray 3.2.")
        request = Luma2GenerationRequest(
            prompt=prompt,
            model="ray-3.2",
            type="video",
            aspect_ratio=aspect_ratio,
            video=Luma2VideoOptions(resolution=resolution, duration=duration, loop=loop or None),
        )
        return await _ray32_generate(cls, request)


class LumaRay32ImageToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="LumaRay32ImageToVideoNode",
            display_name="Luma Ray 3.2 Image to Video",
            category="partner/video/Luma",
            description="Generate a video from a start and/or end frame using Luma's Ray 3.2 model. "
            "Image-anchored generations are always 5 seconds.",
            inputs=[
                IO.String.Input("prompt", multiline=True, default="", tooltip="Text prompt for the video generation."),
                IO.Combo.Input("resolution", options=["360p", "540p", "720p", "1080p"], default="720p"),
                IO.Boolean.Input(
                    "loop",
                    default=False,
                    tooltip="Make the video loop seamlessly. Not available when an end_frame is set.",
                ),
                _ray32_seed_input(),
                IO.Image.Input("start_frame", optional=True, tooltip="First frame of the generated video."),
                IO.Image.Input("end_frame", optional=True, tooltip="Last frame of the generated video."),
            ],
            outputs=[IO.Video.Output(), IO.String.Output(display_name="generation_id")],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=_BADGE_RAY32_VIDEO_5S,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        resolution: str,
        loop: bool,
        seed: int,
        start_frame: torch.Tensor | None = None,
        end_frame: torch.Tensor | None = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1, max_length=6000)
        if start_frame is None and end_frame is None:
            raise ValueError("Provide at least one of start_frame / end_frame.")
        if loop and end_frame is not None:
            raise ValueError("Looping is not available when an end_frame is set.")
        video = Luma2VideoOptions(resolution=resolution, duration="5s", loop=loop or None)
        if start_frame is not None:
            url = await upload_image_to_comfyapi(cls, start_frame, mime_type="image/png")
            video.start_frame = Luma2ImageRef(url=url)
        if end_frame is not None:
            url = await upload_image_to_comfyapi(cls, end_frame, mime_type="image/png")
            video.end_frame = Luma2ImageRef(url=url)
        request = Luma2GenerationRequest(prompt=prompt, model="ray-3.2", type="video", video=video)
        return await _ray32_generate(cls, request)


class LumaRay32KeyframeNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="LumaRay32KeyframeNode",
            display_name="Luma Ray 3.2 Keyframe",
            category="partner/video/Luma",
            description="Anchor a guide image to a position on the Ray 3.2 output video timeline. Connect this to "
            "the 'keyframes' input of the Luma Ray 3.2 Keyframes to Video node; chain several together via the "
            "optional 'keyframes' input below.",
            inputs=[
                IO.Image.Input("image", tooltip="Guide image to place at the chosen moment of the output video."),
                IO.DynamicCombo.Input(
                    "position",
                    options=[
                        IO.DynamicCombo.Option(
                            "Fraction of duration (0.0-1.0)",
                            [
                                IO.Float.Input(
                                    "fraction",
                                    default=0.0,
                                    min=0.0,
                                    max=1.0,
                                    step=0.01,
                                    display_mode=IO.NumberDisplay.number,
                                    tooltip="Where in the output video this image applies " "(0.0 = start, 1.0 = end).",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "Absolute time (seconds)",
                            [
                                IO.Float.Input(
                                    "seconds",
                                    default=0.0,
                                    min=0.0,
                                    max=10.0,
                                    step=0.1,
                                    display_mode=IO.NumberDisplay.number,
                                    tooltip="Time in seconds from the start of the output video where this "
                                    "image applies.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="How to place this image on the output video's timeline.",
                ),
                IO.Custom(LumaIO.LUMA_RAY32_KEYFRAME).Input(
                    "keyframes",
                    optional=True,
                    tooltip="Optional earlier keyframes to chain with this one.",
                ),
            ],
            outputs=[IO.Custom(LumaIO.LUMA_RAY32_KEYFRAME).Output(display_name="keyframes")],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        position: dict,
        keyframes: LumaRay32KeyframeChain | None = None,
    ) -> IO.NodeOutput:
        chain = keyframes.clone() if keyframes is not None else LumaRay32KeyframeChain()
        if position["position"] == "Absolute time (seconds)":
            mode, value = LUMA_KEYFRAME_MODE_SECONDS, float(position["seconds"])
        else:
            mode, value = LUMA_KEYFRAME_MODE_FRACTION, float(position["fraction"])
        chain.add(LumaRay32KeyframeItem(image=image, mode=mode, value=value))
        return IO.NodeOutput(chain)


class LumaRay32KeyframesToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="LumaRay32KeyframesToVideoNode",
            display_name="Luma Ray 3.2 Keyframes to Video",
            category="partner/video/Luma",
            description="Generate a video that interpolates through a sequence of guide images, each anchored to a "
            "position on the timeline, using Luma Ray 3.2. Build the sequence with Luma Ray 3.2 Keyframe nodes "
            "(at least 2).",
            inputs=[
                IO.String.Input("prompt", multiline=True, default="", tooltip="Text prompt for the video generation."),
                IO.Combo.Input("resolution", options=["360p", "540p", "720p", "1080p"], default="720p"),
                IO.Combo.Input("duration", options=["5s", "10s"]),
                _ray32_seed_input(),
                IO.Custom(LumaIO.LUMA_RAY32_KEYFRAME).Input(
                    "keyframes",
                    tooltip="Keyframe sequence from Luma Ray 3.2 Keyframe nodes (at least 2).",
                ),
            ],
            outputs=[IO.Video.Output(), IO.String.Output(display_name="generation_id")],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=_BADGE_RAY32_VIDEO,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        resolution: str,
        duration: str,
        seed: int,
        keyframes: LumaRay32KeyframeChain | None = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1, max_length=6000)
        items = keyframes.items if keyframes is not None else []
        if len(items) < 2:
            raise ValueError(
                "Connect at least 2 Luma Ray 3.2 Keyframe nodes "
                "(use Luma Ray 3.2 Image to Video for a single start/end frame)."
            )
        if len(items) > 64:
            raise ValueError(f"Ray 3.2 supports at most 64 keyframes; got {len(items)}.")
        maxframe = 120 if duration == "5s" else 240
        duration_seconds = maxframe / 24  # 5.0 or 10.0
        # Resolve each keyframe to an output-frame index, then order by position
        # (so the user can chain keyframes in any order — the position is what places them)
        placed: list[tuple[int, torch.Tensor]] = []
        for item in items:
            if item.mode == LUMA_KEYFRAME_MODE_SECONDS:
                if item.value > duration_seconds:
                    raise ValueError(
                        f"Keyframe position {item.value:g}s is past the end of the {duration} video; "
                        f"use 0-{duration_seconds:g}s (or switch the keyframe to fraction mode)."
                    )
                idx = round(item.value * 24)
            else:
                idx = round(item.value * maxframe)
            placed.append((max(0, min(maxframe, idx)), item.image))
        placed.sort(key=lambda p: p[0])
        indexes = [idx for idx, _ in placed]
        for a, b in zip(indexes, indexes[1:]):
            if a == b:
                raise ValueError(
                    f"Two keyframes resolve to the same output frame ({a}) for a {duration} video "
                    f"(valid range 0-{maxframe}); give each keyframe a distinct position."
                )
        refs: list[Luma2ImageRef] = []
        for _, image in placed:
            url = await upload_image_to_comfyapi(cls, image, mime_type="image/png")
            refs.append(Luma2ImageRef(url=url))
        request = Luma2GenerationRequest(
            prompt=prompt,
            model="ray-3.2",
            type="video",
            video=Luma2VideoOptions(resolution=resolution, duration=duration, keyframes=refs, keyframe_indexes=indexes),
        )
        return await _ray32_generate(cls, request)


class LumaRay32VideoEditNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="LumaRay32VideoEditNode",
            display_name="Luma Ray 3.2 Video Edit",
            category="partner/video/Luma",
            description="Re-render an existing video under a new prompt using Luma Ray 3.2 (restyle, relight, add "
            "or remove elements) while keeping the original motion. Source video up to 18 seconds; the edited "
            "video keeps the source's length.",
            inputs=[
                IO.Video.Input("video", tooltip="Source video to edit. Up to 18 seconds."),
                IO.String.Input("prompt", multiline=True, default="", tooltip="Describes the desired edit."),
                IO.Combo.Input("resolution", options=["360p", "540p", "720p", "1080p"], default="720p"),
                IO.Combo.Input(
                    "strength",
                    options=[
                        "auto",
                        "adhere_1",
                        "adhere_2",
                        "adhere_3",
                        "flex_1",
                        "flex_2",
                        "flex_3",
                        "reimagine_1",
                        "reimagine_2",
                        "reimagine_3",
                    ],
                    default="auto",
                    tooltip="How strongly to preserve vs. reimagine the source. 'auto' lets Ray 3.2 choose; "
                    "adhere_* preserves the most, flex_* is balanced, reimagine_* changes the most.",
                ),
                _ray32_seed_input(),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="generation_id"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=_BADGE_RAY32_EDIT,
        )

    @classmethod
    async def execute(
        cls, video: Input.Video, prompt: str, resolution: str, strength: str, seed: int
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1, max_length=6000)
        try:
            duration = "5s" if video.get_duration() <= 5.0 else "10s"
        except Exception:
            duration = "10s"
        source_url = await upload_video_to_comfyapi(cls, video, max_duration=18)
        edit = Luma2VideoEdit(auto_controls=True) if strength == "auto" else Luma2VideoEdit(strength=strength)
        request = Luma2GenerationRequest(
            prompt=prompt,
            model="ray-3.2",
            type="video_edit",
            source=Luma2ImageRef(url=source_url, media_type="video/mp4"),
            video=Luma2VideoOptions(resolution=resolution, duration=duration, edit=edit),
        )
        return await _ray32_generate(cls, request)


class LumaRay32VideoReframeNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="LumaRay32VideoReframeNode",
            display_name="Luma Ray 3.2 Video Reframe",
            category="partner/video/Luma",
            description="Change the aspect ratio of an existing video, using Luma Ray 3.2 to fill the newly "
            "exposed canvas areas. Source video up to 30 seconds. Billed per second of output.",
            inputs=[
                IO.Video.Input("video", tooltip="Source video to reframe. Up to 30 seconds."),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Describes how the newly exposed canvas areas should be filled.",
                ),
                IO.Combo.Input("aspect_ratio", options=["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]),
                IO.Combo.Input("resolution", options=["360p", "540p", "720p", "1080p"], default="720p"),
                _ray32_seed_input(),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="generation_id"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=_BADGE_RAY32_REFRAME,
        )

    @classmethod
    async def execute(
        cls, video: Input.Video, prompt: str, aspect_ratio: str, resolution: str, seed: int
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False, min_length=1, max_length=6000)
        if resolution == "1080p" and aspect_ratio in {"9:16", "3:4"}:
            raise ValueError("1080p is not available for vertical aspect ratios (9:16, 3:4) when reframing.")
        source_url = await upload_video_to_comfyapi(cls, video, max_duration=30)
        request = Luma2GenerationRequest(
            prompt=prompt,
            model="ray-3.2",
            type="video_reframe",
            aspect_ratio=aspect_ratio,
            source=Luma2ImageRef(url=source_url, media_type="video/mp4"),
            video=Luma2VideoOptions(resolution=resolution),
        )
        return await _ray32_generate(cls, request)


class LumaRay32ExtendVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="LumaRay32ExtendVideoNode",
            display_name="Luma Ray 3.2 Extend Video",
            category="partner/video/Luma",
            description="Extend a previous Ray 3.2 generation forward (continue after it) or backward (lead-in "
            "before it). Connect the generation_id output of a prior Luma Ray 3.2 node."
            " Extensions are always 5 seconds.",
            inputs=[
                IO.String.Input(
                    "source_generation_id",
                    default="",
                    tooltip="generation_id of the prior Ray 3.2 video to extend."
                    " Connect the generation_id output of another Luma Ray 3.2 node.",
                ),
                IO.DynamicCombo.Input(
                    "direction",
                    options=[
                        IO.DynamicCombo.Option(
                            "Forward (continue after)",
                            [
                                IO.Boolean.Input(
                                    "loop",
                                    default=False,
                                    tooltip="Loop the extended video seamlessly (forward extend only).",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option("Backward (lead-in before)", []),
                    ],
                    tooltip="Forward continues after the prior clip; backward is prepended before it.",
                ),
                IO.String.Input("prompt", multiline=True, default="", tooltip="Text prompt for the new content."),
                IO.Combo.Input("resolution", options=["540p", "720p", "1080p"], default="720p"),
                _ray32_seed_input(),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="generation_id"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=_BADGE_RAY32_VIDEO_5S,
        )

    @classmethod
    async def execute(
        cls, source_generation_id: str, direction: dict, prompt: str, resolution: str, seed: int
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False, min_length=1, max_length=6000)
        gen_id = (source_generation_id or "").strip()
        if not gen_id:
            raise ValueError(
                "source_generation_id is required (connect the generation_id output of a prior Luma Ray 3.2 node)."
            )
        video = Luma2VideoOptions(resolution=resolution, duration="5s")
        ref = Luma2ImageRef(generation_id=gen_id)
        if direction["direction"] == "Forward (continue after)":
            video.start_frame = ref
            if direction.get("loop"):
                video.loop = True
        else:
            video.end_frame = ref
        request = Luma2GenerationRequest(prompt=prompt, model="ray-3.2", type="video", video=video)
        return await _ray32_generate(cls, request)


class LumaExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            LumaImageNode,
            LumaImageEditNode,
            LumaRay32TextToVideoNode,
            LumaRay32ImageToVideoNode,
            LumaRay32KeyframeNode,
            LumaRay32KeyframesToVideoNode,
            LumaRay32VideoEditNode,
            LumaRay32VideoReframeNode,
            LumaRay32ExtendVideoNode,
        ]


async def comfy_entrypoint() -> LumaExtension:
    return LumaExtension()
