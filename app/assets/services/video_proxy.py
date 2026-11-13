"""Browser-playable H.264 proxies of videos browsers can't play, written to previews/ under their blake3."""

import os
import uuid

import av
from av.video.reformatter import ColorPrimaries, Colorspace, ColorTrc

import folder_paths
from app.assets.services.hashing import compute_blake3_hash

PROXY_LONG_SIDE = 1920
PROXY_SHORT_SIDE = 1080


def needs_video_proxy(system_metadata: dict | None) -> bool:
    return (system_metadata or {}).get("codec") == "prores"


def _proxy_size(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, PROXY_LONG_SIDE / max(width, height), PROXY_SHORT_SIDE / min(width, height))
    return max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2)


def _transcode(source: str, dest: str) -> tuple[int, int]:
    with av.open(source) as src, av.open(dest, "w", format="mp4", options={"movflags": "+faststart"}) as out:
        video = src.streams.video[0]
        video.thread_type = "AUTO"
        audio = src.streams.audio[0] if src.streams.audio else None
        width, height = _proxy_size(video.codec_context.width, video.codec_context.height)

        out_video = out.add_stream("libx264", rate=video.average_rate or video.guessed_rate)
        out_video.width, out_video.height, out_video.pix_fmt = width, height, "yuv420p"
        out_video.options = {"preset": "veryfast", "crf": "18"}
        out_video.codec_context.color_primaries = ColorPrimaries.BT709
        out_video.codec_context.color_trc = ColorTrc.IEC61966_2_1
        out_video.codec_context.colorspace = Colorspace.ITU709
        out_audio = out.add_stream("aac", rate=audio.rate, layout=audio.layout) if audio is not None else None

        graph = av.filter.Graph()
        graph.link_nodes(
            graph.add_buffer(template=video),
            graph.add(
                "scale",
                f"w={width}:h={height}:out_color_matrix=bt709:out_range=tv:out_primaries=bt709:out_transfer=iec61966-2-1:intent=perceptual",
            ),
            graph.add("format", "yuv420p"),
            graph.add("buffersink"),
        ).configure()

        def encode_filtered():
            while True:
                try:
                    frame = graph.pull()
                except (av.BlockingIOError, av.EOFError):
                    return
                out.mux(out_video.encode(frame))

        for packet in src.demux(*(s for s in (video, audio) if s is not None)):
            for frame in packet.decode():
                if packet.stream is video:
                    graph.push(frame)
                    encode_filtered()
                else:
                    out.mux(out_audio.encode(frame))
        graph.push(None)
        encode_filtered()
        out.mux(out_video.encode(None))
        if out_audio is not None:
            out.mux(out_audio.encode(None))
    return width, height


def write_video_proxy(source: str) -> dict:
    """Transcode ``source`` to previews/<blake3>.mp4 and return its preview ref."""
    directory = folder_paths.get_previews_directory()
    os.makedirs(directory, exist_ok=True)
    temp_path = os.path.join(directory, f"{uuid.uuid4().hex}.mp4.tmp")
    try:
        width, height = _transcode(source, temp_path)
        digest, _ = compute_blake3_hash(temp_path)
        filename = f"{digest}.mp4"
        os.replace(temp_path, os.path.join(directory, filename))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return {"filename": filename, "width": width, "height": height}
