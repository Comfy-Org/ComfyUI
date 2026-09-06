import os
import av
import torch
import folder_paths
import json
import weakref
from typing import Optional
from typing_extensions import override
from fractions import Fraction
from comfy_api.latest import ComfyExtension, io, ui, Input, InputImpl, Types
from comfy.cli_args import args

class SaveWEBM(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveWEBM",
            search_aliases=["export webm"],
            display_name="Save WEBM",
            category="video",
            is_experimental=True,
            inputs=[
                io.Image.Input("images", tooltip="RGBA images are saved with their alpha channel as transparency (vp9 codec only)."),
                io.String.Input("filename_prefix", default="ComfyUI"),
                io.Combo.Input("codec", options=["vp9", "av1"]),
                io.Float.Input("fps", default=24.0, min=0.01, max=1000.0, step=0.01),
                io.Float.Input("crf", default=32.0, min=0, max=63.0, step=1, tooltip="Higher crf means lower quality with a smaller file size, lower crf means higher quality higher filesize."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[io.Image.Output(display_name="images")]
        )

    @classmethod
    def execute(cls, images, codec, fps, filename_prefix, crf) -> io.NodeOutput:
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), images[0].shape[1], images[0].shape[0]
        )

        file = f"{filename}_{counter:05}_.webm"
        container = av.open(os.path.join(full_output_folder, file), mode="w")

        if cls.hidden.prompt is not None:
            container.metadata["prompt"] = json.dumps(cls.hidden.prompt)

        if cls.hidden.extra_pnginfo is not None:
            for x in cls.hidden.extra_pnginfo:
                container.metadata[x] = json.dumps(cls.hidden.extra_pnginfo[x])

        # Save transparency when the images carry an alpha channel (RGBA) and the codec supports it.
        # vp9 -> yuva420p; other codecs have no usable alpha path, so the alpha is ignored.
        save_alpha = images.shape[-1] == 4 and codec == "vp9"

        codec_map = {"vp9": "libvpx-vp9", "av1": "libsvtav1"}
        stream = container.add_stream(codec_map[codec], rate=Fraction(round(fps * 1000), 1000))
        stream.width = images.shape[-2]
        stream.height = images.shape[-3]
        stream.pix_fmt = "yuva420p" if save_alpha else ("yuv420p10le" if codec == "av1" else "yuv420p")
        stream.bit_rate = 0
        stream.options = {'crf': str(crf)}
        if codec == "av1":
            stream.options["preset"] = "6"

        for frame in images:
            if save_alpha:
                frame = av.VideoFrame.from_ndarray(torch.clamp(frame[..., :4] * 255, min=0, max=255).to(device=torch.device("cpu"), dtype=torch.uint8).numpy(), format="rgba")
            else:
                frame = av.VideoFrame.from_ndarray(torch.clamp(frame[..., :3] * 255, min=0, max=255).to(device=torch.device("cpu"), dtype=torch.uint8).numpy(), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        container.mux(stream.encode())
        container.close()

        return io.NodeOutput(images, ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))

def _save_video_codec_input(supported_codecs: list[str], *, optional=False, hidden=False):
    codec_options = []
    if "auto" in supported_codecs:
        codec_options.append(io.DynamicCombo.Option("auto", []))
    if "h264" in supported_codecs:
        codec_options.append(
            io.DynamicCombo.Option(
                "h264",
                [
                    io.DynamicCombo.Input(
                        "encoding",
                        display_name="encoding mode",
                        options=[
                            io.DynamicCombo.Option("auto", []),
                            io.DynamicCombo.Option(
                                "re-encode",
                                [
                                    io.Float.Input("crf", default=23.0, min=0.0, max=51.0, step=1.0, tooltip="Lower values produce higher quality and larger files."),
                                ],
                            ),
                        ],
                        optional=True,
                        tooltip="Automatic preserves compatible H.264 streams. Re-encode applies custom encoding options.",
                    ),
                ],
            )
        )
    if "av1" in supported_codecs:
        codec_options.append(
            io.DynamicCombo.Option(
                "av1",
                [
                    io.DynamicCombo.Input(
                        "encoding",
                        display_name="encoding mode",
                        options=[
                            io.DynamicCombo.Option("auto", []),
                            io.DynamicCombo.Option(
                                "re-encode",
                                [
                                    io.Float.Input("crf", default=30.0, min=0.0, max=63.0, step=1.0, tooltip="Lower values produce higher quality and larger files."),
                                ],
                            ),
                        ],
                        optional=True,
                        tooltip="Automatic preserves compatible AV1 streams. Re-encode applies custom encoding options.",
                    ),
                ],
            )
        )
    return io.DynamicCombo.Input(
        "codec",
        options=codec_options,
        optional=optional,
        tooltip="The output video codec. Auto preserves a compatible source stream. H.264 and AV1 re-encoding support SDR, HDR (HLG), and HDR PQ.",
        extra_dict={"hidden": True} if hidden else None,
    )


class SaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveVideo",
            search_aliases=["export video"],
            display_name="Save Video",
            category="video",
            essentials_category="Basics",
            description="Saves the input videos to your ComfyUI output directory.",
            inputs=[
                io.Video.Input("video", tooltip="The video to save."),
                io.String.Input("filename_prefix", default="video/ComfyUI", tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."),
                io.DynamicCombo.Input(
                    "format",
                    options=[
                        io.DynamicCombo.Option("auto", [_save_video_codec_input(["auto", "h264", "av1"])]),
                        io.DynamicCombo.Option("mp4", [_save_video_codec_input(["auto", "h264", "av1"])]),
                        io.DynamicCombo.Option("mkv", [_save_video_codec_input(["auto", "h264", "av1"])]),
                        io.DynamicCombo.Option("webm", [_save_video_codec_input(["auto", "av1"])]),
                    ],
                    tooltip="The output container. Auto uses MP4 for Auto/H.264 and WebM for AV1. MP4, MKV, and WebM select a specific container.",
                ),
                _save_video_codec_input(["auto", "h264", "av1"], optional=True, hidden=True),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[io.Video.Output("video", tooltip="The input video, unchanged.")],
        )

    @classmethod
    def execute(cls, video: Input.Video, filename_prefix, format: io.DynamicCombo.Type | str, codec: io.DynamicCombo.Type | None = None) -> io.NodeOutput:
        if isinstance(format, dict):
            format_name = format["format"]
            codec = format.get("codec") or codec
        else:
            format_name = format
        if codec is None:
            codec = {"codec": "auto"}
        codec_name = codec["codec"]
        if format_name == "auto":
            format_name = "webm" if codec_name == "av1" else "mp4"
        encoding = codec.get("encoding") or {}
        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height
        )
        saved_metadata = None
        if not args.disable_metadata:
            metadata = {}
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
            if len(metadata) > 0:
                saved_metadata = metadata
        file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension(format_name)}"
        video.save_to(
            os.path.join(full_output_folder, file),
            format=Types.VideoContainer(format_name),
            codec=Types.VideoCodec(codec_name),
            metadata=saved_metadata,
            crf=encoding.get("crf"),
        )

        return io.NodeOutput(video, ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))


class CreateVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CreateVideo",
            search_aliases=["images to video"],
            display_name="Create Video",
            category="video",
            essentials_category="Video Tools",
            description="Create a video from images.",
            inputs=[
                io.Image.Input("images", tooltip="The images to create a video from."),
                io.Float.Input("fps", default=30.0, min=1.0, max=120.0, step=1.0),
                io.Audio.Input("audio", optional=True, tooltip="The audio to add to the video."),
                io.Combo.Input(
                    "bit_depth",
                    options=["auto", 8, 10],
                    default="auto",
                    tooltip="Auto uses 8-bit for sRGB and 10-bit for HDR. Explicit 8-bit and 10-bit choices are independent of colorspace.",
                    optional=True,
                ),
                io.Combo.Input(
                    "color_space",
                    options=["sRGB", "HDR", "HDR PQ"],
                    default="sRGB",
                    optional=True,
                    tooltip="Colorspace of the input images. HDR selects BT.2020/HLG and HDR PQ selects BT.2020/PQ.",
                ),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(
        cls, images: Input.Image, fps: float, audio: Optional[Input.Audio] = None, bit_depth: int | str = "auto", color_space: str = "sRGB",
    ) -> io.NodeOutput:
        if bit_depth == "auto":
            bit_depth = 10 if color_space in ("HDR", "HDR PQ") else 8
        return io.NodeOutput(
            InputImpl.VideoFromComponents(
                Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps)),
                bit_depth=bit_depth,
                color_space=color_space,
            )
        )

class GetVideoComponents(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GetVideoComponents",
            search_aliases=["extract frames", "split video", "video to images", "demux"],
            display_name="Get Video Components",
            category="video",
            description="Extracts video frames, audio, frame rate, bit depth, and color space.",
            inputs=[
                io.Video.Input("video", tooltip="The video to extract components from."),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
                io.Float.Output(display_name="fps"),
                io.Combo.Output(display_name="bit_depth"),
                io.Combo.Output(display_name="color_space"),
            ],
        )

    @classmethod
    def execute(cls, video: Input.Video) -> io.NodeOutput:
        components = video.get_components()
        return io.NodeOutput(
            components.images,
            components.audio,
            float(components.frame_rate),
            video.get_bit_depth(),
            video.get_color_space(),
        )


class LoadVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return io.Schema(
            node_id="LoadVideo",
            search_aliases=["import video", "open video", "video file"],
            display_name="Load Video",
            category="video",
            essentials_category="Basics",
            has_intermediate_output=True,
            inputs=[
                io.Combo.Input("file", options=sorted(files), upload=io.UploadType.video),
                io.VideoEdit.Input(
                    "edit",
                    optional=True,
                    tooltip="Trim (seconds) and crop (pixels) applied on load. Zero values leave the video unchanged.",
                ),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(cls, file, edit=None) -> io.NodeOutput:
        video_path = folder_paths.get_annotated_filepath(file)
        source = InputImpl.VideoFromFile(video_path)
        video = apply_video_trim(source, (edit or {}).get("trim"))
        video = apply_video_crop(video, (edit or {}).get("crop"))
        if video is source:
            return io.NodeOutput(video, ui=preview_input_video(file, source))
        return io.NodeOutput(video, ui=save_video_preview(video))

    @classmethod
    def fingerprint_inputs(s, file, edit=None):
        video_path = folder_paths.get_annotated_filepath(file)
        mod_time = os.path.getmtime(video_path)
        # Instead of hashing the file, we can just use the modification time to avoid
        # rehashing large files.
        return mod_time

    @classmethod
    def validate_inputs(s, file, edit=None):
        if not folder_paths.exists_annotated_filepath(file):
            return "Invalid video file: {}".format(file)

        return True

_preview_results: "weakref.WeakKeyDictionary[Input.Video, tuple[str, ui.SavedResult]]" = weakref.WeakKeyDictionary()


def preview_input_video(file: str, video: Input.Video | None = None) -> ui.PreviewVideo:
    name, _ = folder_paths.annotated_filepath(file)
    subfolder, _, filename = name.replace("\\", "/").rpartition("/")
    result = ui.SavedResult(filename, subfolder, io.FolderType.input)
    if video is not None:
        _preview_results[video] = (folder_paths.get_annotated_filepath(file), result)
    return ui.PreviewVideo([result])


def save_video_preview(video: Input.Video) -> ui.PreviewVideo:
    cached = _preview_results.get(video)
    if cached is not None and os.path.isfile(cached[0]):
        return ui.PreviewVideo([cached[1]])

    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        "ComfyUI_temp_video", folder_paths.get_temp_directory(), 0, 0
    )
    preview_format = Types.VideoContainer.MP4
    file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension(preview_format)}"
    full_path = os.path.join(full_output_folder, file)
    video.save_to(
        full_path,
        format=preview_format,
        codec="auto",
        preset="ultrafast",
    )
    result = ui.SavedResult(file, subfolder, io.FolderType.temp)
    _preview_results[video] = (full_path, result)
    return ui.PreviewVideo([result])


def apply_video_trim(video: Input.Video, trim, strict_duration: bool = False) -> Input.Video:
    trim = trim or {}
    start_time = float(trim.get("start_time", 0.0))
    duration = float(trim.get("duration", 0.0))
    if duration < 0:
        raise ValueError(f"Trim duration must be >= 0, got {duration}")
    if start_time == 0.0 and duration == 0.0:
        return video

    trimmed = video.as_trimmed(start_time, duration, strict_duration=strict_duration)
    if trimmed is None:
        raise ValueError(
            f"Failed to trim video:\nSource duration: {video.get_duration()}\nStart time: {start_time}\nTarget duration: {duration}"
        )
    return trimmed


def apply_video_crop(video: Input.Video, crop) -> Input.Video:
    crop = crop or {}
    return video.as_cropped(
        int(crop.get("x", 0)),
        int(crop.get("y", 0)),
        int(crop.get("width", 0)),
        int(crop.get("height", 0)),
    )


class VideoSlice(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Video Slice",
            display_name="Trim Video",
            search_aliases=["trim video duration", "skip first frames", "frame load cap", "start time"],
            category="video",
            essentials_category="Video Tools",
            inputs=[
                io.Video.Input("video"),
                io.Float.Input(
                    "start_time",
                    default=0.0,
                    max=1e5,
                    min=-1e5,
                    step=0.001,
                    tooltip="Start time in seconds",
                ),
                io.Float.Input(
                    "duration",
                    default=0.0,
                    min=0.0,
                    step=0.001,
                    tooltip="Duration in seconds, or 0 for unlimited duration",
                ),
                io.Boolean.Input(
                    "strict_duration",
                    default=False,
                    tooltip="If True, when the specified duration is not possible, an error will be raised.",
                ),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(cls, video: io.Video.Type, start_time: float, duration: float, strict_duration: bool) -> io.NodeOutput:
        trimmed = video.as_trimmed(start_time, duration, strict_duration=strict_duration)
        if trimmed is not None:
            return io.NodeOutput(trimmed)
        raise ValueError(
            f"Failed to slice video:\nSource duration: {video.get_duration()}\nStart time: {start_time}\nTarget duration: {duration}"
        )


class VideoTrim(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VideoTrim",
            display_name="Trim Video (Advanced)",
            search_aliases=["trim video duration", "skip first frames", "cut video", "start time"],
            category="video",
            is_experimental=True,
            is_output_node=True,
            essentials_category="Video Tools",
            has_intermediate_output=True,
            inputs=[
                io.Video.Input("video"),
                io.VideoEdit.Input(
                    "trim",
                    features=["trim"],
                    tooltip="Trim window in seconds. Duration 0 keeps the video until the end.",
                ),
                io.Boolean.Input(
                    "strict_duration",
                    default=False,
                    advanced=True,
                    tooltip="If True, when the specified duration is not possible, an error will be raised.",
                ),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(cls, video: io.Video.Type, trim: io.VideoEdit.Type, strict_duration: bool) -> io.NodeOutput:
        trimmed = apply_video_trim(video, (trim or {}).get("trim"), strict_duration=strict_duration)
        return io.NodeOutput(trimmed, ui=save_video_preview(trimmed))


class VideoCrop(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VideoCrop",
            display_name="Crop Video",
            search_aliases=["crop video", "cut region", "spatial crop"],
            category="video",
            is_experimental=True,
            is_output_node=True,
            essentials_category="Video Tools",
            has_intermediate_output=True,
            inputs=[
                io.Video.Input("video"),
                io.VideoEdit.Input(
                    "crop",
                    features=["crop"],
                    tooltip="Crop region in pixels. Zero width/height keeps the full frame.",
                ),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(cls, video: io.Video.Type, crop: io.VideoEdit.Type) -> io.NodeOutput:
        cropped = apply_video_crop(video, (crop or {}).get("crop"))
        return io.NodeOutput(cropped, ui=save_video_preview(cropped))


class VideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SaveWEBM,
            SaveVideo,
            CreateVideo,
            GetVideoComponents,
            LoadVideo,
            VideoSlice,
            VideoTrim,
            VideoCrop,
        ]

async def comfy_entrypoint() -> VideoExtension:
    return VideoExtension()
