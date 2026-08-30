import os
import av
import torch
import folder_paths
import json
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
            inputs=[
                io.Combo.Input("file", options=sorted(files), upload=io.UploadType.video),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(cls, file) -> io.NodeOutput:
        video_path = folder_paths.get_annotated_filepath(file)
        return io.NodeOutput(InputImpl.VideoFromFile(video_path))

    @classmethod
    def fingerprint_inputs(s, file):
        video_path = folder_paths.get_annotated_filepath(file)
        mod_time = os.path.getmtime(video_path)
        # Instead of hashing the file, we can just use the modification time to avoid
        # rehashing large files.
        return mod_time

    @classmethod
    def validate_inputs(s, file):
        if not folder_paths.exists_annotated_filepath(file):
            return "Invalid video file: {}".format(file)

        return True

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


class ResampledVideo(Input.Video):
    """Lazy view of a source video retimed to a new frame rate by duplicating or dropping frames."""

    def __init__(self, video: Input.Video, frame_rate: Fraction):
        self.__video = video
        self.__frame_rate = frame_rate

    def __output_frame_count(self, source_count: int, source_rate: Fraction) -> int:
        if source_count == 0:
            return 0
        if source_rate <= 0:
            raise ValueError("Video frame rate must be positive")
        return max(1, round(source_count * self.__frame_rate / source_rate))

    def get_components(self) -> Types.VideoComponents:
        components = self.__video.get_components()
        source_count = components.images.shape[0]
        count = self.__output_frame_count(source_count, components.frame_rate)
        step = float(components.frame_rate / self.__frame_rate)
        # Nearest source frame per output tick, ties toward the earlier frame so
        # integer up/down ratios duplicate or drop frames uniformly.
        indices = (torch.arange(count, dtype=torch.float64) * step - 0.5).ceil().long().clamp_(min=0, max=max(source_count - 1, 0))
        return Types.VideoComponents(
            images=components.images[indices],
            frame_rate=self.__frame_rate,
            audio=components.audio,
            metadata=components.metadata,
            alpha=components.alpha[indices] if components.alpha is not None else None,
        )

    def get_frame_rate(self) -> Fraction:
        return self.__frame_rate

    def get_frame_count(self) -> int:
        return self.__output_frame_count(self.__video.get_frame_count(), self.__video.get_frame_rate())

    def get_duration(self) -> float:
        return float(Fraction(self.get_frame_count()) / self.__frame_rate)

    def get_dimensions(self) -> tuple[int, int]:
        return self.__video.get_dimensions()

    def get_bit_depth(self) -> int:
        return self.__video.get_bit_depth()

    def get_active_trim_window(self) -> tuple[float, float]:
        return self.__video.get_active_trim_window()

    def save_to(
        self,
        path,
        format: Types.VideoContainer = Types.VideoContainer.AUTO,
        codec: Types.VideoCodec = Types.VideoCodec.AUTO,
        metadata: Optional[dict] = None,
        bit_depth: int | None = None,
        crf: float | None = None,
    ):
        return InputImpl.VideoFromComponents(
            self.get_components(), bit_depth=self.__video.get_bit_depth()
        ).save_to(path, format=format, codec=codec, metadata=metadata, bit_depth=bit_depth, crf=crf)

    def as_trimmed(
        self,
        start_time: float | None = None,
        duration: float | None = None,
        strict_duration: bool = False,
    ) -> Optional[Input.Video]:
        trimmed = self.__video.as_trimmed(start_time, duration, strict_duration=strict_duration)
        if trimmed is None:
            return None
        return ResampledVideo(trimmed, self.__frame_rate)


class ResampleVideoFrameRate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ResampleVideoFrameRate",
            display_name="Resample Video Frame Rate",
            search_aliases=["force rate", "change fps", "framerate", "retime video"],
            category="video",
            essentials_category="Video Tools",
            description="Changes the frame rate of a video by duplicating or dropping frames, keeping the duration and audio unchanged.",
            inputs=[
                io.Video.Input("video", tooltip="The video to resample."),
                io.Float.Input(
                    "fps",
                    default=24.0,
                    min=0.01,
                    max=1000.0,
                    step=0.01,
                    tooltip="Frame rate to resample the video to. Frames are duplicated or dropped to keep the clip duration.",
                ),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(cls, video: Input.Video, fps: float) -> io.NodeOutput:
        frame_rate = Fraction(fps)
        if video.get_frame_rate() == frame_rate:
            return io.NodeOutput(video)
        return io.NodeOutput(ResampledVideo(video, frame_rate))


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
            ResampleVideoFrameRate,
        ]

async def comfy_entrypoint() -> VideoExtension:
    return VideoExtension()
