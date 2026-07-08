from av.container import InputContainer
from av.subtitles.stream import SubtitleStream
from fractions import Fraction
from typing import Optional
from .._input import AudioInput, VideoInput
import av
import io
import itertools
import json
import numpy as np
import math
import torch
from .._util import VideoContainer, VideoCodec, VideoComponents
import logging


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
    open_kwargs = {
        "mode": "w",
        # If isobmff, preserve custom metadata tags (workflow, prompt, extra_pnginfo)
        "options": {"movflags": "use_metadata_tags"},
    }

    is_write_to_buffer = isinstance(dest, io.BytesIO)
    if is_write_to_buffer:
        # Set output format explicitly, since it cannot be inferred from file extension
        if to_format == VideoContainer.AUTO:
            to_format = container_format.lower()
        elif isinstance(to_format, str):
            to_format = to_format.lower()
        open_kwargs["format"] = container_to_output_format(to_format)

    return open_kwargs


def video_stream_bit_depth(stream) -> int:
    if stream is None or stream.format is None or not stream.format.components:
        return 8
    return max(component.bits for component in stream.format.components)


class VideoFromFile(VideoInput):
    """
    Class representing video input from a file.
    """

    def __init__(self, file: str | io.BytesIO, *, start_time: float=0, duration: float=0):
        """
        Initialize the VideoFromFile object based off of either a path on disk or a BytesIO object
        containing the file contents.
        """
        self.__file = file
        self.__start_time = start_time
        self.__duration = duration

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
                    return stream.width, stream.height
        raise ValueError(f"No video stream found in file '{self.__file}'")

    def get_bit_depth(self) -> int:
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)  # Reset the BytesIO object to the beginning
        with av.open(self.__file, mode="r") as container:
            video_stream = container.streams.video[0] if len(container.streams.video) > 0 else None
            return video_stream_bit_depth(video_stream)

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
            if self.__start_time < 0:
                start_time = max(self._get_raw_duration() + self.__start_time, 0)
            else:
                start_time = self.__start_time
            frame_count = 1
            start_pts = int(start_time / video_stream.time_base)
            end_pts = int((start_time + self.__duration) / video_stream.time_base)
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
        if self.__start_time < 0:
            start_time = max(self._get_raw_duration() + self.__start_time, 0)
        else:
            start_time = self.__start_time

        # Get video frames
        frames = []
        audio_frames = []
        alphas = None
        start_pts = int(start_time / video_stream.time_base)
        end_pts = int((start_time + self.__duration) / video_stream.time_base)

        if start_pts != 0:
            container.seek(start_pts, stream=video_stream)

        image_format = 'gbrpf32le'
        process_image_format = lambda a: a
        align_graph = None
        audio = None

        streams = [video_stream]
        has_first_audio_frame = False
        checked_alpha = False

        # Default to False so we decode until EOF if duration is 0
        video_done = False
        audio_done = True

        if len(container.streams.audio):
            audio_stream = container.streams.audio[-1]
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
                        if self.__duration and frame.pts >= end_pts:
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
                        if frame.rotation != 0:
                            k = int(round(frame.rotation // 90))
                            img = np.rot90(img, k=k, axes=(0, 1)).copy()
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
                    if self.__duration and frame.time > start_time + self.__duration:
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
            if self.__duration:
                audio_data = audio_data[..., :int(self.__duration * audio_stream.sample_rate)]

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
    ):
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)  # Reset the BytesIO object to the beginning
        with av.open(self.__file, mode='r') as container:
            container_format = container.format.name
            video_stream = container.streams.video[0] if len(container.streams.video) > 0 else None
            video_encoding = video_stream.codec.name if video_stream is not None else None
            source_bit_depth = video_stream_bit_depth(video_stream)
            reuse_streams = True
            if format != VideoContainer.AUTO and format not in container_format.split(","):
                reuse_streams = False
            if codec != VideoCodec.AUTO and codec != video_encoding and video_encoding is not None:
                reuse_streams = False
            if bit_depth is not None and video_encoding is not None and bit_depth != source_bit_depth:
                reuse_streams = False
            if self.__start_time or self.__duration:
                reuse_streams = False

            if not reuse_streams:
                if bit_depth is None:
                    bit_depth = source_bit_depth
                components = self.get_components_internal(container)
                video = VideoFromComponents(components)
                return video.save_to(
                    path, format=format, codec=codec, metadata=metadata, bit_depth=bit_depth,
                )

            streams = container.streams

            open_kwargs = get_open_write_kwargs(path, container_format, format)
            with av.open(path, **open_kwargs) as output_container:
                # Copy over the original metadata
                for key, value in container.metadata.items():
                    if metadata is None or key not in metadata:
                        output_container.metadata[key] = value

                # Add our new metadata
                if metadata is not None:
                    for key, value in metadata.items():
                        if isinstance(value, str):
                            output_container.metadata[key] = value
                        else:
                            output_container.metadata[key] = json.dumps(value)

                # Add streams to the new container
                stream_map = {}
                for stream in streams:
                    if isinstance(stream, (av.VideoStream, av.AudioStream, SubtitleStream)):
                        out_stream = output_container.add_stream_from_template(template=stream, opaque=True)
                        stream_map[stream] = out_stream

                # Write packets to the new container
                for packet in container.demux():
                    if packet.stream in stream_map and packet.dts is not None:
                        packet.stream = stream_map[packet.stream]
                        output_container.mux(packet)

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
        )
        if trimmed.get_duration() < duration and strict_duration:
            return None
        return trimmed


class VideoFromComponents(VideoInput):
    """
    Class representing video input from tensors.
    """

    def __init__(self, components: VideoComponents, bit_depth: int = 8):
        self.__components = components
        # Tensor components have no inherent bit depth; this is the depth used when encoding.
        self.__bit_depth = bit_depth

    def get_components(self) -> VideoComponents:
        return VideoComponents(
            images=self.__components.images,
            audio=self.__components.audio,
            frame_rate=self.__components.frame_rate,
        )

    def get_bit_depth(self) -> int:
        return self.__bit_depth

    def save_to(
        self,
        path: str,
        format: VideoContainer = VideoContainer.AUTO,
        codec: VideoCodec = VideoCodec.AUTO,
        metadata: Optional[dict] = None,
        bit_depth: int | None = None,
    ):
        """Save the video to a file path or BytesIO buffer."""
        if format != VideoContainer.AUTO and format != VideoContainer.MP4:
            raise ValueError("Only MP4 format is supported for now")
        if codec != VideoCodec.AUTO and codec != VideoCodec.H264:
            raise ValueError("Only H264 codec is supported for now")
        # None means "use the depth this video was created with" (CreateVideo's choice).
        if bit_depth is None:
            bit_depth = self.__bit_depth
        is_10bit = bit_depth >= 10
        extra_kwargs = {}
        if isinstance(format, VideoContainer) and format != VideoContainer.AUTO:
            extra_kwargs["format"] = format.value
        elif isinstance(path, io.BytesIO):
            # BytesIO has no file extension, so av.open can't infer the format.
            # Default to mp4 since that's the only supported format anyway.
            extra_kwargs["format"] = "mp4"
        with av.open(path, mode='w', options={'movflags': 'use_metadata_tags'}, **extra_kwargs) as output:
            # Add metadata before writing any streams
            if metadata is not None:
                for key, value in metadata.items():
                    output.metadata[key] = json.dumps(value)

            frame_rate = Fraction(round(self.__components.frame_rate * 1000), 1000)
            # Create a video stream
            pix_fmt = "yuv420p10le" if is_10bit else "yuv420p"
            video_stream = output.add_stream('h264', rate=frame_rate)
            video_stream.width = self.__components.images.shape[2]
            video_stream.height = self.__components.images.shape[1]
            video_stream.pix_fmt = pix_fmt

            # Create an audio stream
            audio_sample_rate = 1
            audio_stream: Optional[av.AudioStream] = None
            if self.__components.audio:
                audio_sample_rate = int(self.__components.audio['sample_rate'])
                waveform = self.__components.audio['waveform']
                waveform = waveform[0, :, :math.ceil((audio_sample_rate / frame_rate) * self.__components.images.shape[0])]
                layout = {1: 'mono', 2: 'stereo', 6: '5.1'}.get(waveform.shape[0], 'stereo')
                audio_stream = output.add_stream('aac', rate=audio_sample_rate, layout=layout)

            # Encode video
            for i, frame in enumerate(self.__components.images):
                if is_10bit:
                    # 16-bit RGB keeps float precision through the conversion to 10-bit YUV.
                    img = (frame.float() * 65535).clamp(0, 65535).cpu().numpy().astype(np.uint16)  # shape: (H, W, 3)
                    frame = av.VideoFrame.from_ndarray(img, format="rgb48le")
                else:
                    img = (frame * 255).clamp(0, 255).byte().cpu().numpy() # shape: (H, W, 3)
                    frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                frame = frame.reformat(format=pix_fmt)
                packet = video_stream.encode(frame)
                output.mux(packet)

            # Flush video
            packet = video_stream.encode(None)
            output.mux(packet)

            if audio_stream and self.__components.audio:
                frame = av.AudioFrame.from_ndarray(waveform.float().cpu().contiguous().numpy(), format='fltp', layout=layout)
                frame.sample_rate = audio_sample_rate
                frame.pts = 0
                output.mux(audio_stream.encode(frame))

                # Flush encoder
                output.mux(audio_stream.encode(None))

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
