from av.bitstream import BitStreamFilterContext
from av.container import InputContainer
from av.subtitles.stream import SubtitleStream
from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc
from fractions import Fraction
from typing import Optional
from .._input import AudioInput, VideoInput
import av
import io
import itertools
import json
import numpy as np
import math
import os
import torch
from .._util import VideoContainer, VideoCodec, VideoComponents, normalize_crop_rect
import comfy.utils
import logging


VIDEO_ENCODERS = {
    VideoCodec.H264: "h264",
    VideoCodec.AV1: "libsvtav1",
}
VIDEO_CONTAINER_FORMATS = {
    VideoContainer.MP4: "mp4",
    VideoContainer.MKV: "matroska",
    VideoContainer.WEBM: "webm",
}
WEBM_STREAM_CODECS = {
    "video": {"av1", "vp8", "vp9"},
    "audio": {"opus", "vorbis"},
    "subtitle": {"webvtt"},
}
BT2020_NCL = 9
BT709_NCL = 1
HDR_COLOR_TRANSFERS = {
    "HDR": ColorTrc.ARIB_STD_B67,
    "HDR PQ": ColorTrc.SMPTE2084,
}
VIDEO_COLOR_TRANSFERS = {
    "sRGB": ColorTrc.IEC61966_2_1,
    **HDR_COLOR_TRANSFERS,
}
VIDEO_TRANSFER_COLOR_SPACES = {
    ColorTrc.BT709: "sRGB",
    ColorTrc.IEC61966_2_1: "sRGB",
    ColorTrc.ARIB_STD_B67: "HDR",
    ColorTrc.SMPTE2084: "HDR PQ",
}


def container_to_output_format(container_format: str | None) -> str | None:
    """
    A container's `format` may be a comma-separated list of formats.
    E.g., iso container's `format` may be `mov,mp4,m4a,3gp,3g2,mj2`.
    However, writing to a file/stream with `av.open` requires a single format,
    or `None` to auto-detect.
    """
    if not container_format:
        return None  # Auto-detect

    if "," not in container_format:
        return container_format

    formats = container_format.split(",")
    return formats[0]

def get_open_write_kwargs(
    dest: str | io.BytesIO, container_format: str, to_format: str | None
) -> dict:
    """Get kwargs for writing a `VideoFromFile` to a file/stream with `av.open`"""
    is_write_to_buffer = isinstance(dest, io.BytesIO)
    open_kwargs = {"mode": "w"}

    if is_write_to_buffer:
        # Set output format explicitly, since it cannot be inferred from file extension
        if to_format == VideoContainer.AUTO:
            to_format = container_format.lower()
        elif isinstance(to_format, VideoContainer):
            to_format = VIDEO_CONTAINER_FORMATS[to_format]
        elif isinstance(to_format, str):
            to_format = to_format.lower()
        open_kwargs["format"] = container_to_output_format(to_format)

    output_format = open_kwargs["format"] if is_write_to_buffer else os.path.splitext(dest)[1].lower().lstrip(".")
    if output_format in ("mov", "mp4"):
        # Preserve custom metadata tags (workflow, prompt, extra_pnginfo) in isobmff.
        movflags = "use_metadata_tags" if is_write_to_buffer else "use_metadata_tags+faststart"
        open_kwargs["options"] = {"movflags": movflags}

    return open_kwargs


def video_stream_bit_depth(stream) -> int:
    if stream is None or stream.format is None or not stream.format.components:
        return 8
    return max(component.bits for component in stream.format.components)


def isobmff_hevc_filter(output_container, stream, out_stream):
    """Apple players need the 'hvc1' sample entry, not FFmpeg's default 'hev1'. Annex B input without
    extradata makes the muxer build hvcC from the first packet and strip in-band parameter sets;
    'hvc1' sources already have a complete hvcC and only need the tag PyAV reset."""
    if output_container.format.name not in ("mp4", "mov") or stream.codec.canonical_name != "hevc":
        return None
    try:
        codec_tag = stream.codec_context.codec_tag
    except UnicodeDecodeError:
        codec_tag = ""
    if codec_tag == "hvc1":
        out_stream.codec_context.codec_tag = "hvc1"
        return None
    hevc_filter = BitStreamFilterContext("hevc_mp4toannexb", stream, out_stream)
    out_stream.codec_context.codec_tag = "hvc1"
    out_stream.codec_context.extradata = None
    return hevc_filter


def filter_hevc_packet(hevc_filter, packet):
    if packet.has_sidedata("new_extradata"):
        raise ValueError("HEVC with multiple sample descriptions cannot be remuxed as hvc1; re-encode it instead")
    return hevc_filter.filter(packet)


def last_decodable_audio_stream(container: InputContainer):
    """Streams FFmpeg has no decoder for have no codec context, and decoding their
    packets crashes the process (e.g. APAC spatial-audio track in iPhone)."""
    stream = next(
        (s for s in reversed(container.streams.audio) if s.codec_context is not None),
        None,
    )
    if stream is None and len(container.streams.audio):
        logging.warning("No decodable audio stream found in video; ignoring audio.")
    return stream


def probe_audio_params(container: InputContainer, audio_stream, max_packets: int = 200):
    """Containers probed only up to a window (mpegts) leave audio codec parameters unset when
    audio starts beyond it; learn them by decoding ahead. The caller must seek back afterwards.
    Returns (sample_rate, channels), zeros when the stream never yields a decodable frame."""
    for i, packet in enumerate(container.demux(audio_stream)):
        try:
            frames = packet.decode()
        except av.error.FFmpegError:
            frames = ()
        if frames:
            return frames[0].sample_rate, frames[0].layout.nb_channels
        if i >= max_packets:
            break
    return 0, 0


def write_output_metadata(container: InputContainer, output, metadata: dict | None):
    """Copy the source container's metadata, then overlay the caller's tags."""
    for key, value in container.metadata.items():
        if metadata is None or key not in metadata:
            output.metadata[key] = value
    if metadata is not None:
        for key, value in metadata.items():
            output.metadata[key] = value if isinstance(value, str) else json.dumps(value)


def video_output_config(path: str | io.BytesIO, format: VideoContainer, codec: VideoCodec) -> tuple[dict, VideoContainer, VideoCodec]:
    if isinstance(format, str):
        format = VideoContainer(format)
    if isinstance(codec, str):
        codec = VideoCodec(codec)

    if format == VideoContainer.AUTO:
        extension = os.path.splitext(os.fspath(path))[1].lower() if isinstance(path, (str, os.PathLike)) else ""
        format = {
            ".mkv": VideoContainer.MKV,
            ".webm": VideoContainer.WEBM,
        }.get(extension, VideoContainer.MP4)
    if codec == VideoCodec.AUTO:
        codec = VideoCodec.AV1 if format == VideoContainer.WEBM else VideoCodec.H264
    if format == VideoContainer.WEBM and codec != VideoCodec.AV1:
        raise ValueError("WebM output requires the AV1 codec")

    # FFmpeg's faststart pass reopens the output by filename, so it cannot be used with file-like objects.
    open_kwargs = {"mode": "w", "format": VIDEO_CONTAINER_FORMATS[format]}
    if format == VideoContainer.MP4:
        movflags = "use_metadata_tags+faststart" if isinstance(path, (str, os.PathLike)) else "use_metadata_tags"
        open_kwargs["options"] = {"movflags": movflags}
    return open_kwargs, format, codec


def set_video_color_properties(target, color_space):
    is_hdr = color_space in HDR_COLOR_TRANSFERS
    target.color_primaries = ColorPrimaries.BT2020 if is_hdr else ColorPrimaries.BT709
    target.color_trc = VIDEO_COLOR_TRANSFERS[color_space]
    target.colorspace = BT2020_NCL if is_hdr else BT709_NCL
    target.color_range = ColorRange.MPEG


def copy_color_properties(source, target):
    target.color_primaries = source.color_primaries
    target.color_trc = source.color_trc
    target.colorspace = source.colorspace
    target.color_range = source.color_range


def video_stream_color_space(stream) -> str | None:
    if stream is None:
        return None
    return VIDEO_TRANSFER_COLOR_SPACES.get(stream.color_trc)


def video_encoder_options(
    codec: VideoCodec, crf: float | None, preset: str | None = None
) -> dict[str, str]:
    options = {}
    if preset is not None and codec == VideoCodec.H264:
        options["preset"] = preset
    if crf is not None:
        if codec == VideoCodec.AV1 and crf == 0:
            options["svtav1-params"] = "lossless=1"
        else:
            options["crf"] = str(crf)
    return options


def webm_streams_compatible(streams) -> bool:
    for stream in streams:
        allowed_codecs = WEBM_STREAM_CODECS.get(stream.type)
        if allowed_codecs is not None and stream.codec_context is not None and stream.codec.canonical_name not in allowed_codecs:
            return False
    return True


def _rotation_quadrant(frame: av.VideoFrame) -> int:
    return int(round(frame.rotation // 90)) % 4 if frame.rotation else 0


class VideoFromFile(VideoInput):
    """
    Class representing video input from a file.
    """

    def __init__(self, file: str | io.BytesIO, *, start_time: float=0, duration: float=0,
                 crop: tuple[int, int, int, int] | None = None):
        """
        Initialize the VideoFromFile object based off of either a path on disk or a BytesIO object
        containing the file contents.
        """
        self.__file = file
        self.__start_time = start_time
        self.__duration = duration
        self.__crop = crop

    def get_stream_source(self) -> str | io.BytesIO:
        """
        Return the underlying file source for efficient streaming.
        This avoids unnecessary memory copies when the source is already a file path.
        """
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)
        return self.__file

    def get_active_trim_window(self) -> tuple[float, float]:
        start_time = self.__start_time
        if start_time < 0:
            start_time = max(self._get_raw_duration() + start_time, 0.0)
        return float(start_time), float(self.__duration)

    def get_dimensions(self) -> tuple[int, int]:
        """
        Returns the dimensions of the video input.

        Returns:
            Tuple of (width, height)
        """
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)  # Reset the BytesIO object to the beginning
        with av.open(self.__file, mode='r') as container:
            for stream in container.streams:
                if stream.type == 'video':
                    assert isinstance(stream, av.VideoStream)
                    if self.__crop is None:
                        return stream.width, stream.height

                    display_width, display_height = self._get_display_dimensions()
                    rect = normalize_crop_rect(*self.__crop, display_width, display_height)
                    if rect is not None:
                        return rect[2], rect[3]
                    return display_width, display_height
        raise ValueError(f"No video stream found in file '{self.__file}'")

    def _get_display_dimensions(self) -> tuple[int, int]:
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)
        with av.open(self.__file, mode='r') as container:
            for stream in container.streams:
                if stream.type == 'video':
                    assert isinstance(stream, av.VideoStream)
                    width, height = stream.width, stream.height
                    try:
                        frame = next(container.decode(stream), None)
                    except av.error.FFmpegError:
                        frame = None
                    if frame is not None and _rotation_quadrant(frame) % 2:
                        width, height = height, width
                    return width, height
        raise ValueError(f"No video stream found in file '{self.__file}'")

    def get_bit_depth(self) -> int:
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)  # Reset the BytesIO object to the beginning
        with av.open(self.__file, mode="r") as container:
            video_stream = container.streams.video[0] if len(container.streams.video) > 0 else None
            return video_stream_bit_depth(video_stream)

    def get_color_space(self) -> str:
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)
        with av.open(self.__file, mode="r") as container:
            video_stream = container.streams.video[0] if len(container.streams.video) > 0 else None
            return video_stream_color_space(video_stream) or "sRGB"

    def get_duration(self) -> float:
        """
        Returns the duration of the video in seconds.

        Returns:
            Duration in seconds
        """
        raw_duration = self._get_raw_duration()
        if self.__start_time < 0:
            duration_from_start = min(raw_duration, -self.__start_time)
        else:
            duration_from_start = raw_duration - self.__start_time
        if self.__duration:
            return min(self.__duration, duration_from_start)
        return duration_from_start

    def _get_raw_duration(self) -> float:
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)
        with av.open(self.__file, mode="r") as container:
            if container.duration is not None:
                return float(container.duration / av.time_base)

            # Fallback: calculate from frame count and frame rate
            video_stream = next(
                (s for s in container.streams if s.type == "video"), None
            )
            if video_stream and video_stream.frames and video_stream.average_rate:
                return float(video_stream.frames / video_stream.average_rate)

            # Last resort: decode frames to count them
            if video_stream and video_stream.average_rate:
                frame_count = 0
                container.seek(0)
                frame_iterator = (
                    container.decode(video_stream)
                    if video_stream.codec.capabilities & 0x100
                    else container.demux(video_stream)
                )
                for packet in frame_iterator:
                    frame_count += 1
                if frame_count > 0:
                    return float(frame_count / video_stream.average_rate)

        raise ValueError(f"Could not determine duration for file '{self.__file}'")

    def get_frame_count(self) -> int:
        """
        Returns the number of frames in the video without materializing them as
        torch tensors.
        """
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)

        with av.open(self.__file, mode="r") as container:
            video_stream = self._get_first_video_stream(container)
            # 1. Prefer the frames field if available and usable
            if (
                video_stream.frames
                and video_stream.frames > 0
                and not self.__start_time
                and not self.__duration
            ):
                return int(video_stream.frames)

            # 2. Try to estimate from duration and average_rate using only metadata
            if (
                getattr(video_stream, "duration", None) is not None
                and getattr(video_stream, "time_base", None) is not None
                and video_stream.average_rate
            ):
                raw_duration = float(video_stream.duration * video_stream.time_base)
                if self.__start_time < 0:
                    duration_from_start = min(raw_duration, -self.__start_time)
                else:
                    duration_from_start = raw_duration - self.__start_time
                duration_seconds = min(self.__duration, duration_from_start)
                estimated_frames = int(round(duration_seconds * float(video_stream.average_rate)))
                if estimated_frames > 0:
                    return estimated_frames

            # 3. Last resort: decode frames and count them (streaming)
            start_time, duration = self.get_active_trim_window()
            frame_count = 1
            start_pts = int(start_time / video_stream.time_base)
            end_pts = int((start_time + duration) / video_stream.time_base)
            container.seek(start_pts, stream=video_stream)
            frame_iterator = (
                container.decode(video_stream)
                if video_stream.codec.capabilities & 0x100
                else container.demux(video_stream)
            )
            for frame in frame_iterator:
                if frame.pts >= start_pts:
                    break
            else:
                raise ValueError(f"Could not determine frame count for file '{self.__file}'\nNo frames exist for start_time {self.__start_time}")
            for frame in frame_iterator:
                if frame.pts >= end_pts:
                    break
                frame_count += 1
            return frame_count

    def get_frame_rate(self) -> Fraction:
        """
        Returns the average frame rate of the video using container metadata
        without decoding all frames.
        """
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)

        with av.open(self.__file, mode="r") as container:
            video_stream = self._get_first_video_stream(container)
            # Preferred: use PyAV's average_rate (usually already a Fraction-like)
            if video_stream.average_rate:
                return Fraction(video_stream.average_rate)

            # Fallback: estimate from frames + duration if available
            if video_stream.frames and container.duration:
                duration_seconds = float(container.duration / av.time_base)
                if duration_seconds > 0:
                    return Fraction(video_stream.frames / duration_seconds).limit_denominator()

            # Last resort: match get_components_internal default
            return Fraction(1)

    def get_container_format(self) -> str:
        """
        Returns the container format of the video (e.g., 'mp4', 'mov', 'avi').

        Returns:
            Container format as string
        """
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)
        with av.open(self.__file, mode='r') as container:
            return container.format.name

    def get_components_internal(self, container: InputContainer) -> VideoComponents:
        video_stream = self._get_first_video_stream(container)
        video_stream.thread_type = "AUTO"
        start_time, duration = self.get_active_trim_window()

        # Get video frames
        frames = []
        audio_frames = []
        alphas = None
        start_pts = int(start_time / video_stream.time_base)
        end_pts = int((start_time + duration) / video_stream.time_base)

        if start_pts != 0:
            container.seek(start_pts, stream=video_stream)

        image_format = 'gbrpf32le'
        process_image_format = lambda a: a
        align_graph = None
        audio = None

        streams = [video_stream]
        has_first_audio_frame = False
        checked_alpha = False
        crop_rect = None
        crop_resolved = False

        # Default to False so we decode until EOF if duration is 0
        video_done = False
        audio_done = True

        audio_stream = last_decodable_audio_stream(container)
        if audio_stream is not None:
            streams += [audio_stream]
            resampler = av.audio.resampler.AudioResampler(format='fltp')
            audio_done = False

        for packet in container.demux(*streams):
            if video_done and audio_done:
                break

            if packet.stream.type == "video":
                if video_done:
                    continue
                try:
                    for frame in packet.decode():
                        if frame.pts < start_pts:
                            continue
                        if duration and frame.pts >= end_pts:
                            video_done = True
                            break

                        if not checked_alpha:
                            alpha_channel = False
                            for comp in frame.format.components:
                                if comp.is_alpha or frame.format.name == "pal8":
                                    alphas = []
                                    alpha_channel = True
                                    break
                            if frame.format.name in ("yuvj420p", "yuvj422p", "yuvj444p", "rgb24", "rgba", "pal8"):
                                process_image_format = lambda a: a.float() / 255.0
                                if alpha_channel:
                                    image_format = 'rgba'
                                else:
                                    image_format = 'rgb24'
                            else:
                                process_image_format = lambda a: a
                                if alpha_channel:
                                    image_format = 'gbrapf32le'
                                else:
                                    image_format = 'gbrpf32le'

                            checked_alpha = True

                        # Fix non-deterministic video decode when the video width is not a multiple of 32
                        # For non-yuvj pixel formats: most H.264/H.265 video and static images (e.g. lossy WebP via LoadImage)
                        # Pad both axes to a multiple of 32 and smear the border so the alignment padding never bleeds into the cropped edges
                        if image_format in ('gbrpf32le', 'gbrapf32le') and frame.width % 32 != 0:
                            if align_graph is None:
                                pad_w = ((frame.width + 31) // 32) * 32
                                pad_h = ((frame.height + 31) // 32) * 32
                                g = av.filter.Graph()
                                g_src = g.add_buffer(width=frame.width, height=frame.height,
                                                     format=frame.format.name, time_base=video_stream.time_base)
                                g_pad = g.add('pad', f'{pad_w}:{pad_h}:0:0')
                                g_fill = g.add('fillborders', f'left=0:right={pad_w - frame.width}:top=0:bottom={pad_h - frame.height}:mode=smear')
                                g_sink = g.add('buffersink')
                                g_src.link_to(g_pad)
                                g_pad.link_to(g_fill)
                                g_fill.link_to(g_sink)
                                g.configure()
                                align_graph = (g, g_src, g_sink)
                            align_graph[1].push(frame)
                            img = np.ascontiguousarray(align_graph[2].pull().to_ndarray(format=image_format)[:frame.height, :frame.width])
                        else:
                            img = frame.to_ndarray(format=image_format)
                        rotation_quadrant = _rotation_quadrant(frame)
                        if rotation_quadrant:
                            img = np.rot90(img, k=rotation_quadrant, axes=(0, 1)).copy()
                        if self.__crop is not None:
                            if not crop_resolved:
                                crop_rect = normalize_crop_rect(*self.__crop, img.shape[1], img.shape[0])
                                crop_resolved = True
                            if crop_rect is not None:
                                cx, cy, cw, ch = crop_rect
                                img = np.ascontiguousarray(img[cy:cy + ch, cx:cx + cw])
                        if alphas is None:
                            frames.append(torch.from_numpy(img))
                        else:
                            frames.append(torch.from_numpy(img[..., :-1]))
                            alphas.append(torch.from_numpy(img[..., -1:]))
                except av.error.InvalidDataError:
                    logging.info("pyav decode error")

            elif packet.stream.type == "audio":
                if audio_done:
                    continue

                aframes = itertools.chain.from_iterable(
                    map(resampler.resample, packet.decode())
                )
                for frame in aframes:
                    if duration and frame.time > start_time + duration:
                        audio_done = True
                        break

                    if not has_first_audio_frame:
                        offset_seconds = start_time - frame.pts * audio_stream.time_base
                        to_skip = max(0, int(offset_seconds * audio_stream.sample_rate))
                        if to_skip < frame.samples:
                            has_first_audio_frame = True
                            audio_frames.append(frame.to_ndarray()[..., to_skip:])
                    else:
                        audio_frames.append(frame.to_ndarray())

        images = process_image_format(torch.stack(frames)) if len(frames) > 0 else torch.zeros(0, 0, 0, 3)
        if alphas is not None:
            alphas = process_image_format(torch.stack(alphas)) if len(alphas) > 0 else torch.zeros(0, 0, 0, 1)

        # Get frame rate
        frame_rate = Fraction(video_stream.average_rate) if video_stream.average_rate else Fraction(1)

        if len(audio_frames) > 0:
            audio_data = np.concatenate(audio_frames, axis=1)  # shape: (channels, total_samples)
            if duration:
                audio_data = audio_data[..., :int(duration * audio_stream.sample_rate)]

            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)  # shape: (1, channels, total_samples)
            audio = AudioInput({
                "waveform": audio_tensor,
                "sample_rate": int(audio_stream.sample_rate) if audio_stream.sample_rate else 1,
            })

        metadata = container.metadata
        return VideoComponents(images=images, alpha=alphas, audio=audio, frame_rate=frame_rate, metadata=metadata)

    def get_components(self) -> VideoComponents:
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)  # Reset the BytesIO object to the beginning
        with av.open(self.__file, mode='r') as container:
            return self.get_components_internal(container)
        raise ValueError(f"No video stream found in file '{self.__file}'")

    def save_to(
        self,
        path: str | io.BytesIO,
        format: VideoContainer = VideoContainer.AUTO,
        codec: VideoCodec = VideoCodec.AUTO,
        metadata: Optional[dict] = None,
        bit_depth: int | None = None,
        crf: float | None = None,
        color_space: str | None = None,
        preset: str | None = None,
    ):
        if color_space is not None and color_space not in VIDEO_COLOR_TRANSFERS:
            raise ValueError(f"Unsupported video color space: {color_space}")
        _, output_format, _ = video_output_config(path, format, codec)
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)  # Reset the BytesIO object to the beginning
        with av.open(self.__file, mode='r') as container:
            container_format = container.format.name
            video_stream = container.streams.video[0] if len(container.streams.video) > 0 else None
            video_encoding = video_stream.codec.canonical_name if video_stream is not None else None
            source_bit_depth = video_stream_bit_depth(video_stream)
            source_color_space = video_stream_color_space(video_stream)
            if source_color_space is not None and color_space is not None and source_color_space != color_space:
                raise ValueError(
                    f"Cannot save {source_color_space} video as {color_space} without color conversion; "
                    f"use auto or {source_color_space}"
                )
            reuse_streams = True
            if format != VideoContainer.AUTO and VIDEO_CONTAINER_FORMATS[VideoContainer(format)] not in container_format.split(","):
                reuse_streams = False
            if output_format == VideoContainer.WEBM and not webm_streams_compatible(container.streams):
                reuse_streams = False
            if codec != VideoCodec.AUTO and codec != video_encoding and video_encoding is not None:
                reuse_streams = False
            if bit_depth is not None and video_encoding is not None and bit_depth != source_bit_depth:
                reuse_streams = False
            if crf is not None:
                reuse_streams = False
            if color_space is not None:
                reuse_streams = False
            if self.__start_time or self.__duration:
                reuse_streams = False
            if self.__crop is not None:
                reuse_streams = False

            if not reuse_streams:
                if bit_depth is None:
                    bit_depth = source_bit_depth
                return self._save_transcoded(container, path, format=format, codec=codec, metadata=metadata, bit_depth=bit_depth, crf=crf, color_space=color_space, preset=preset)

            streams = container.streams

            open_kwargs = get_open_write_kwargs(path, container_format, format)
            with av.open(path, **open_kwargs) as output_container:
                # Add metadata before writing any streams
                write_output_metadata(container, output_container, metadata)

                # Add streams to the new container. Streams with no codec context cannot be used as an output template.
                stream_map = {}
                hevc_filters = {}
                for stream in streams:
                    if isinstance(stream, (av.VideoStream, av.AudioStream, SubtitleStream)):
                        if stream.codec_context is None:
                            logging.warning("Skipping %s stream %d with unsupported codec", stream.type, stream.index)
                            continue
                        out_stream = output_container.add_stream_from_template(template=stream, opaque=True)
                        hevc_filter = isobmff_hevc_filter(output_container, stream, out_stream)
                        if hevc_filter is not None:
                            hevc_filters[stream] = hevc_filter
                        stream_map[stream] = out_stream

                # Write packets to the new container
                for packet in container.demux():
                    if packet.stream in stream_map and packet.dts is not None:
                        out_stream = stream_map[packet.stream]
                        hevc_filter = hevc_filters.get(packet.stream)
                        for out_packet in filter_hevc_packet(hevc_filter, packet) if hevc_filter else (packet,):
                            out_packet.stream = out_stream
                            output_container.mux(out_packet)

    def _save_transcoded(
        self,
        container: InputContainer,
        path: str | io.BytesIO,
        format: VideoContainer,
        codec: VideoCodec,
        metadata: dict | None,
        bit_depth: int,
        crf: float | None = None,
        color_space: str | None = None,
        preset: str | None = None,
    ):
        """Re-encode one frame at a time; peak memory does not scale with video length."""
        open_kwargs, output_format, output_codec = video_output_config(path, format, codec)
        video_stream = self._get_first_video_stream(container)
        video_stream.thread_type = "AUTO"
        start_time, duration = self.get_active_trim_window()
        start_pts = int(start_time / video_stream.time_base)
        end_pts = int((start_time + duration) / video_stream.time_base) if duration else None
        stream_end_pts = None
        if video_stream.duration is not None:
            stream_end_pts = (video_stream.start_time or 0) + video_stream.duration
        output_end_pts = end_pts
        if stream_end_pts is not None and (output_end_pts is None or stream_end_pts < output_end_pts):
            output_end_pts = stream_end_pts
        if start_pts != 0:
            container.seek(start_pts, stream=video_stream)

        audio_stream = last_decodable_audio_stream(container)
        source_color_space = video_stream_color_space(video_stream)
        preserve_source_color = source_color_space is not None
        pix_fmt = "yuv420p10le" if bit_depth >= 10 else "yuv420p"
        rate = Fraction(video_stream.average_rate) if video_stream.average_rate else Fraction(1)

        resampler = None
        sample_rate = 0
        audio_time_base = None
        duration_cap = None
        if audio_stream is not None:
            sample_rate = audio_stream.codec_context.sample_rate
            channels = audio_stream.codec_context.channels
            if not sample_rate:
                sample_rate, channels = probe_audio_params(container, audio_stream)
                container.seek(start_pts, stream=video_stream)
                if sample_rate:
                    audio_stream.codec_context.flush_buffers()
                else:
                    logging.warning("Audio stream parameters could not be determined; ignoring audio.")
                    audio_stream = None
        if audio_stream is not None:
            if output_format == VideoContainer.WEBM:
                sample_rate = 48000
            audio_time_base = Fraction(1, sample_rate)
            layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(channels, "stereo")
            resampler = av.audio.resampler.AudioResampler(format="fltp", layout=layout, rate=sample_rate)
            if duration:
                duration_cap = math.ceil(duration * sample_rate)

        if duration:
            window_seconds = duration
        else:
            try:
                window_seconds = max(self._get_raw_duration() - start_time, 0.0)
            except ValueError:
                window_seconds = 0.0
        progress_total = max(1, int(round(window_seconds * float(rate))))
        pbar = comfy.utils.ProgressBar(progress_total)

        streams = [video_stream] if audio_stream is None else [video_stream, audio_stream]
        pts_step = max(1, int(round((1 / rate) / video_stream.time_base)))
        video_done = False
        audio_done = audio_stream is None
        video_pts_offset = None
        last_video_pts = None
        last_video_end = None
        # rebased pts -> true display duration: the mp4 muxer pads the last sample with 1/rate otherwise
        video_frame_durations = {}
        source_size = None
        rotation_k = 0
        rotation_filter = None
        crop_rect = None
        crop_filter = None
        audio_started = False
        samples_written = 0
        pending_audio = []
        # The output opens lazily on the first kept frame: it decides the geometry (90/270 rotation swaps dims),
        # and never seeking back keeps webm/mkv leading audio intact.
        output = None
        out_video = None
        out_audio = None

        def audio_frame_from_ndarray(nd_planar):
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(nd_planar), format="fltp", layout=layout)
            frame.sample_rate = sample_rate
            return frame

        def drain_audio(final=False):
            # Audio may cover the pts span of the video written so far, capped by the requested duration
            nonlocal samples_written, audio_done
            if last_video_end is None:
                cap = 0
            else:
                cap = math.ceil(last_video_end * video_stream.time_base * sample_rate)
            if duration_cap is not None:
                cap = min(cap, duration_cap)
            while pending_audio and not audio_done:
                frame = pending_audio[0]
                if samples_written + frame.samples <= cap:
                    frame.pts = samples_written
                    frame.time_base = audio_time_base
                    output.mux(out_audio.encode(frame))
                    samples_written += frame.samples
                    pending_audio.pop(0)
                    continue
                if final:
                    keep = frame.to_ndarray()[..., :cap - samples_written]
                    if keep.shape[-1] > 0:
                        tail = audio_frame_from_ndarray(keep)
                        tail.pts = samples_written
                        tail.time_base = audio_time_base
                        output.mux(out_audio.encode(tail))
                        samples_written += keep.shape[-1]
                    pending_audio.clear()
                break
            if duration_cap is not None and samples_written >= duration_cap:
                audio_done = True
            return cap

        try:
            for packet in container.demux(*streams):
                if video_done and audio_done:
                    break

                if packet.stream == video_stream and not video_done:
                    try:
                        frames = packet.decode()
                    except av.error.InvalidDataError:
                        logging.info("pyav decode error")
                        continue
                    for frame in frames:
                        if frame.pts is not None and frame.pts < start_pts:
                            continue
                        if end_pts is not None and frame.pts is not None and frame.pts >= end_pts:
                            video_done = True
                            if last_video_pts is not None:
                                # the source continues past the window: hold the last kept frame to the window end
                                end_offset = video_pts_offset if video_pts_offset is not None else start_pts
                                last_video_end = max(last_video_end, end_pts - end_offset)
                            break
                        # the source's true display duration of this frame; average_rate is not a
                        # frame duration (sparse/VFR sources), so it is only the fallback
                        frame_duration = frame.duration if frame.duration else pts_step
                        if end_pts is not None and frame.pts is not None:
                            frame_duration = min(frame_duration, end_pts - frame.pts)
                        if output is None:
                            rotation_k = _rotation_quadrant(frame)
                            if rotation_k % 2:
                                out_width, out_height = frame.height, frame.width
                            else:
                                out_width, out_height = frame.width, frame.height
                            if self.__crop is not None:
                                crop_rect = normalize_crop_rect(*self.__crop, out_width, out_height)
                                if crop_rect is not None:
                                    out_width, out_height = crop_rect[2], crop_rect[3]
                            if (out_width % 2 or out_height % 2) and crop_rect is None:
                                even_width = out_width - out_width % 2
                                even_height = out_height - out_height % 2
                                if even_width > 0 and even_height > 0:
                                    crop_rect = (0, 0, even_width, even_height)
                                    out_width, out_height = even_width, even_height
                            if out_width % 2 or out_height % 2:
                                raise ValueError(f"{output_codec.value.upper()} output requires even dimensions, got {out_width}x{out_height}")
                            if any(component.is_alpha for component in frame.format.components):
                                logging.warning(
                                    "Transcoded video output does not support alpha; the alpha channel will be discarded."
                                )
                            source_size = (frame.width, frame.height)
                            output = av.open(path, **open_kwargs)
                            # Add metadata before writing any streams
                            write_output_metadata(container, output, metadata)
                            out_video = output.add_stream(VIDEO_ENCODERS[output_codec], rate=rate)
                            # no B-frames: reordering makes mp4 sample durations follow decode order,
                            # so irregular-VFR spans and trim windows land wrong
                            out_video.codec_context.max_b_frames = 0
                            out_video.width = out_width
                            out_video.height = out_height
                            out_video.pix_fmt = pix_fmt
                            out_video.options = video_encoder_options(output_codec, crf, preset)
                            if preserve_source_color:
                                copy_color_properties(video_stream, out_video.codec_context)
                            elif color_space is not None:
                                set_video_color_properties(out_video.codec_context, color_space)
                            # source pts pass through (rebased to 0), so variable frame rate survives
                            out_video.codec_context.time_base = video_stream.time_base
                            if audio_stream is not None:
                                audio_codec = "libopus" if output_format == VideoContainer.WEBM else "aac"
                                out_audio = output.add_stream(audio_codec, rate=sample_rate, layout=layout)
                        if (frame.width, frame.height) != source_size:
                            # encoding would silently rescale the new geometry into the old one
                            raise ValueError(
                                f"Video resolution changes mid-stream "
                                f"({source_size[0]}x{source_size[1]} -> {frame.width}x{frame.height}); cannot transcode"
                            )
                        if rotation_k:
                            if rotation_filter is None:
                                g = av.filter.Graph()
                                g_src = g.add_buffer(width=frame.width, height=frame.height,
                                                     format=frame.format.name, time_base=video_stream.time_base)
                                tail = g_src
                                for filter_name, filter_args in {1: [("transpose", "cclock")],
                                                                 2: [("hflip", None), ("vflip", None)],
                                                                 3: [("transpose", "clock")]}[rotation_k]:
                                    step = g.add(filter_name, filter_args)
                                    tail.link_to(step)
                                    tail = step
                                g_sink = g.add("buffersink")
                                tail.link_to(g_sink)
                                g.configure()
                                rotation_filter = (g_src, g_sink)
                            rotation_filter[0].push(frame)
                            frame = rotation_filter[1].pull()
                        if crop_rect is not None:
                            if crop_filter is None:
                                g = av.filter.Graph()
                                g_src = g.add_buffer(width=frame.width, height=frame.height,
                                                     format=frame.format.name, time_base=video_stream.time_base)
                                g_crop = g.add("crop", f"{crop_rect[2]}:{crop_rect[3]}:{crop_rect[0]}:{crop_rect[1]}")
                                g_sink = g.add("buffersink")
                                g_src.link_to(g_crop)
                                g_crop.link_to(g_sink)
                                g.configure()
                                crop_filter = (g_src, g_sink)
                            crop_filter[0].push(frame)
                            frame = crop_filter[1].pull()
                        if frame.color_range == ColorRange.JPEG and not preserve_source_color:
                            # compress full-range sources (yuvj/MJPEG) to limited range
                            frame = frame.reformat(format=pix_fmt, src_color_range="JPEG", dst_color_range="MPEG")
                        else:
                            frame = frame.reformat(format=pix_fmt)
                        if preserve_source_color:
                            copy_color_properties(video_stream, frame)
                        elif color_space is not None:
                            set_video_color_properties(frame, color_space)
                        frame_output_end = None
                        if frame.pts is not None:
                            if video_pts_offset is None:
                                video_pts_offset = frame.pts
                            frame.pts -= video_pts_offset
                            if output_end_pts is not None:
                                frame_output_end = output_end_pts - video_pts_offset
                                if frame.pts + frame_duration > frame_output_end:
                                    clamped_pts = frame_output_end - frame_duration
                                    if clamped_pts >= 0 and (last_video_pts is None or clamped_pts > last_video_pts):
                                        frame.pts = min(frame.pts, clamped_pts)
                                    elif frame.pts < frame_output_end:
                                        frame_duration = frame_output_end - frame.pts
                                    else:
                                        continue
                        if frame.pts is None or (last_video_pts is not None and frame.pts <= last_video_pts):
                            # broken sources emit missing/backward timestamps mid-stream, which the
                            # muxer rejects; nudge them forward by one nominal frame interval
                            frame.pts = 0 if last_video_pts is None else last_video_pts + pts_step
                            if frame_output_end is not None and frame.pts + frame_duration > frame_output_end:
                                if frame.pts >= frame_output_end:
                                    continue
                                frame_duration = frame_output_end - frame.pts
                        last_video_pts = frame.pts
                        last_video_end = frame.pts + frame_duration
                        video_frame_durations[frame.pts] = frame_duration
                        # the decoded pict_type would force x264's frame types (intra-only
                        # sources like MJPEG/ProRes would come out all-keyframe)
                        frame.pict_type = 0
                        for out_packet in out_video.encode(frame):
                            out_packet.duration = video_frame_durations.pop(out_packet.pts, 0)
                            output.mux(out_packet)
                        drain_audio()
                        pbar.update(1)

                elif packet.stream == audio_stream and not audio_done:
                    for resampled in itertools.chain.from_iterable(map(resampler.resample, packet.decode())):
                        frame_start = None
                        if resampled.pts is not None:
                            # passthrough frames keep the source stream's time base
                            tb = resampled.time_base if resampled.time_base else audio_time_base
                            frame_start = float(resampled.pts * tb)
                            if duration and not audio_started and frame_start >= start_time + duration:
                                audio_done = True
                                break
                        if not audio_started:
                            if frame_start is None:
                                frame_start = 0.0
                            to_skip = max(0, int((start_time - frame_start) * sample_rate))
                            if to_skip >= resampled.samples:
                                continue
                            audio_started = True
                            if duration and frame_start > start_time:
                                duration_cap = min(duration_cap, math.ceil((start_time + duration - frame_start) * sample_rate))
                            if to_skip:
                                pending_audio.append(audio_frame_from_ndarray(resampled.to_ndarray()[..., to_skip:]))
                                continue
                        pending_audio.append(resampled)
                        if video_done:
                            # the video window is complete so the cap is final, but containers
                            # that interleave audio behind video (fragmented mp4) still owe most
                            # of it: stop only once the demuxed audio covers the cap
                            cap = drain_audio()
                            if pending_audio or samples_written >= cap:
                                drain_audio(final=True)
                                audio_done = True
                                break

            if output is None:
                raise ValueError(f"No decodable video frames found in file '{self.__file}'")
            if out_audio is not None and not audio_done:
                drain_audio(final=True)
            window_fill = last_video_end - last_video_pts if video_done and last_video_pts is not None else 0
            for out_packet in out_video.encode(None):
                duration = video_frame_durations.pop(out_packet.pts, 0)
                if out_packet.pts == last_video_pts:
                    duration = max(duration, window_fill)
                out_packet.duration = duration
                output.mux(out_packet)
            if out_audio is not None:
                output.mux(out_audio.encode(None))
        except BaseException:
            if output is not None:
                output.close()
                if isinstance(path, (str, os.PathLike)) and os.path.exists(path):
                    os.remove(path)
            raise
        else:
            if output is not None:
                output.close()

    def _get_first_video_stream(self, container: InputContainer):
        if len(container.streams.video):
            return container.streams.video[0]
        raise ValueError(f"No video stream found in file '{self.__file}'")

    def as_trimmed(
        self, start_time: float = 0, duration: float = 0, strict_duration: bool = True
    ) -> VideoInput | None:
        trimmed = VideoFromFile(
            self.get_stream_source(),
            start_time=start_time + self.__start_time,
            duration=duration,
            crop=self.__crop,
        )
        if strict_duration and duration and trimmed.get_duration() < duration:
            return None
        return trimmed

    def as_cropped(
        self, x: int = 0, y: int = 0, width: int = 0, height: int = 0
    ) -> VideoInput:
        if int(width) <= 0 or int(height) <= 0:
            return self

        display_width, display_height = self._get_display_dimensions()
        outer = (
            normalize_crop_rect(*self.__crop, display_width, display_height)
            if self.__crop is not None
            else None
        )
        if outer is None:
            rect = normalize_crop_rect(x, y, width, height, display_width, display_height)
        else:
            inner = normalize_crop_rect(x, y, width, height, outer[2], outer[3])
            rect = (
                (outer[0] + inner[0], outer[1] + inner[1], inner[2], inner[3])
                if inner is not None
                else None
            )
        if rect is None:
            return self
        return VideoFromFile(
            self.get_stream_source(),
            start_time=self.__start_time,
            duration=self.__duration,
            crop=rect,
        )


class VideoFromComponents(VideoInput):
    """
    Class representing video input from tensors.
    """

    def __init__(self, components: VideoComponents, bit_depth: int = 8, color_space: str = "sRGB"):
        if color_space not in VIDEO_COLOR_TRANSFERS:
            raise ValueError(f"Unsupported video color space: {color_space}")
        self.__components = components
        # Tensor components have no inherent bit depth; this is the depth used when encoding.
        self.__bit_depth = bit_depth
        self.__color_space = color_space

    def get_components(self) -> VideoComponents:
        return VideoComponents(
            images=self.__components.images,
            audio=self.__components.audio,
            frame_rate=self.__components.frame_rate,
            metadata=self.__components.metadata,
            alpha=self.__components.alpha,
        )

    def get_bit_depth(self) -> int:
        return self.__bit_depth

    def get_color_space(self) -> str:
        return self.__color_space

    def save_to(
        self,
        path: str,
        format: VideoContainer = VideoContainer.AUTO,
        codec: VideoCodec = VideoCodec.AUTO,
        metadata: Optional[dict] = None,
        bit_depth: int | None = None,
        crf: float | None = None,
        color_space: str | None = None,
        preset: str | None = None,
    ):
        """Save the video to a file path or BytesIO buffer."""
        if color_space is None:
            color_space = self.__color_space
        if color_space is not None and color_space not in VIDEO_COLOR_TRANSFERS:
            raise ValueError(f"Unsupported video color space: {color_space}")
        open_kwargs, output_format, output_codec = video_output_config(path, format, codec)
        # None means "use the depth this video was created with" (CreateVideo's choice).
        if bit_depth is None:
            bit_depth = self.__bit_depth
        is_10bit = bit_depth >= 10
        with av.open(path, **open_kwargs) as output:
            # Add metadata before writing any streams
            if metadata is not None:
                for key, value in metadata.items():
                    output.metadata[key] = json.dumps(value)

            frame_rate = Fraction(round(self.__components.frame_rate * 1000), 1000)
            # Create a video stream
            pix_fmt = "yuv420p10le" if is_10bit else "yuv420p"
            video_stream = output.add_stream(VIDEO_ENCODERS[output_codec], rate=frame_rate)
            video_stream.width = self.__components.images.shape[2]
            video_stream.height = self.__components.images.shape[1]
            video_stream.pix_fmt = pix_fmt
            video_stream.options = video_encoder_options(output_codec, crf, preset)
            if color_space is not None:
                set_video_color_properties(video_stream.codec_context, color_space)

            # Create an audio stream
            audio_sample_rate = 1
            audio_resampler = None
            audio_stream: Optional[av.AudioStream] = None
            if self.__components.audio:
                source_audio_sample_rate = int(self.__components.audio['sample_rate'])
                audio_sample_rate = 48000 if output_format == VideoContainer.WEBM else source_audio_sample_rate
                waveform = self.__components.audio['waveform']
                waveform = waveform[0, :, :math.ceil((source_audio_sample_rate / frame_rate) * self.__components.images.shape[0])]
                waveform = torch.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
                layout = {1: 'mono', 2: 'stereo', 6: '5.1'}.get(waveform.shape[0], 'stereo')
                audio_codec = "libopus" if output_format == VideoContainer.WEBM else "aac"
                audio_stream = output.add_stream(audio_codec, rate=audio_sample_rate, layout=layout)
                if audio_sample_rate != source_audio_sample_rate:
                    audio_resampler = av.audio.resampler.AudioResampler(format="fltp", layout=layout, rate=audio_sample_rate)

            # Encode video
            for i, frame in enumerate(self.__components.images):
                if is_10bit:
                    # 16-bit RGB keeps float precision through the conversion to 10-bit YUV.
                    img = (frame.float() * 65535).clamp(0, 65535).cpu().numpy().astype(np.uint16)  # shape: (H, W, 3)
                    frame = av.VideoFrame.from_ndarray(img, format="rgb48le")
                else:
                    img = (frame * 255).clamp(0, 255).byte().cpu().numpy() # shape: (H, W, 3)
                    frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                dst_colorspace = None
                if color_space == "sRGB":
                    dst_colorspace = BT709_NCL
                elif color_space in HDR_COLOR_TRANSFERS:
                    dst_colorspace = BT2020_NCL
                frame = frame.reformat(format=pix_fmt, dst_colorspace=dst_colorspace)
                if color_space is not None:
                    set_video_color_properties(frame, color_space)
                for packet in video_stream.encode(frame):
                    output.mux(packet)

            # Flush video
            for packet in video_stream.encode(None):
                output.mux(packet)

            if audio_stream and self.__components.audio:
                try:
                    frame = av.AudioFrame.from_ndarray(waveform.float().cpu().contiguous().numpy(), format='fltp', layout=layout)
                    frame.sample_rate = source_audio_sample_rate
                    frame.pts = 0
                    frames = [frame] if audio_resampler is None else audio_resampler.resample(frame)
                    for frame in frames:
                        output.mux(audio_stream.encode(frame))
                    if audio_resampler is not None:
                        for frame in audio_resampler.resample(None):
                            output.mux(audio_stream.encode(frame))

                    # Flush encoder
                    output.mux(audio_stream.encode(None))
                except Exception as e:
                    logging.warning(f"[SaveVideo] Audio muxing exception suppressed: {e}")

    def as_trimmed(
        self,
        start_time: float | None = None,
        duration: float | None = None,
        strict_duration: bool = True,
    ) -> VideoInput | None:
        if self.get_duration() < start_time + duration:
            return None
        #TODO Consider tracking duration and trimming at time of save?
        return VideoFromFile(self.get_stream_source(), start_time=start_time, duration=duration)
