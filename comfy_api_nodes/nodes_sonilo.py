import base64
import json
import logging
import time
from urllib.parse import urljoin

import aiohttp
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.util import (
    ApiEndpoint,
    audio_bytes_to_audio_input,
    upload_video_to_comfyapi,
    validate_string,
)
from comfy_api_nodes.util._helpers import (
    default_base_url,
    get_comfy_api_headers,
    get_node_id,
    is_processing_interrupted,
)
from comfy_api_nodes.util.common_exceptions import ProcessingInterrupted
from server import PromptServer

logger = logging.getLogger(__name__)


class SoniloVideoToMusic(IO.ComfyNode):
    """Generate music from video using Sonilo's AI model."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="SoniloVideoToMusic",
            display_name="Sonilo Video to Music",
            category="partner/audio/Sonilo",
            description="Generate music from video content using Sonilo's AI model. "
            "Analyzes the video and creates matching music.",
            inputs=[
                IO.Video.Input(
                    "video",
                    tooltip="Input video to generate music from. Maximum duration: 6 minutes.",
                ),
                IO.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Optional text prompt to guide music generation. "
                    "Leave empty for best quality - the model will fully analyze the video content.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed for reproducibility. Currently ignored by the Sonilo "
                    "service but kept for graph consistency.",
                ),
            ],
            outputs=[IO.Audio.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr='{"type":"usd","usd":0.009,"format":{"suffix":"/second"}}',
            ),
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        prompt: str = "",
        seed: int = 0,
    ) -> IO.NodeOutput:
        video_url = await upload_video_to_comfyapi(cls, video, max_duration=360)
        form = aiohttp.FormData()
        form.add_field("video_url", video_url)
        if prompt.strip():
            form.add_field("prompt", prompt.strip())
        audio_bytes = await _stream_sonilo_music(
            cls,
            ApiEndpoint(path="/proxy/sonilo/v2m/generate", method="POST"),
            form,
        )
        return IO.NodeOutput(audio_bytes_to_audio_input(audio_bytes))


class SoniloTextToMusic(IO.ComfyNode):
    """Generate music from a text prompt using Sonilo's AI model."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="SoniloTextToMusic",
            display_name="Sonilo Text to Music",
            category="partner/audio/Sonilo",
            description="Generate music from a text prompt using Sonilo's AI model.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Text prompt describing the music to generate.",
                ),
                IO.Int.Input(
                    "duration",
                    default=30,
                    min=1,
                    max=360,
                    tooltip="Target duration in seconds. Maximum: 6 minutes.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed for reproducibility. Currently ignored by the Sonilo "
                    "service but kept for graph consistency.",
                ),
            ],
            outputs=[IO.Audio.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["duration"]),
                expr='{"type":"usd","usd": 0.0025 * widgets.duration}',
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        duration: int = 1,
        seed: int = 0,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1, max_length=1000)
        form = aiohttp.FormData()
        form.add_field("prompt", prompt)
        form.add_field("duration", str(duration))
        audio_bytes = await _stream_sonilo_music(
            cls,
            ApiEndpoint(path="/proxy/sonilo/t2m/generate", method="POST"),
            form,
        )
        return IO.NodeOutput(audio_bytes_to_audio_input(audio_bytes))


async def _stream_sonilo_music(
    cls: type[IO.ComfyNode],
    endpoint: ApiEndpoint,
    form: aiohttp.FormData,
) -> bytes:
    """POST ``form`` to Sonilo, read the NDJSON stream, and return the first stream's audio bytes."""
    url = urljoin(default_base_url().rstrip("/") + "/", endpoint.path.lstrip("/"))

    headers = get_comfy_api_headers(cls)
    headers.update(endpoint.headers)

    node_id = get_node_id(cls)
    start_ts = time.monotonic()
    last_chunk_status_ts = 0.0
    audio_streams: dict[int, list[bytes]] = {}
    title: str | None = None

    timeout = aiohttp.ClientTimeout(total=1200.0, sock_read=300.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        PromptServer.instance.send_progress_text("Status: Queued", node_id)
        async with session.post(url, data=form, headers=headers) as resp:
            if resp.status >= 400:
                msg = await _extract_error_message(resp)
                raise Exception(f"Sonilo API error ({resp.status}): {msg}")

            while True:
                if is_processing_interrupted():
                    raise ProcessingInterrupted("Task cancelled")

                raw_line = await resp.content.readline()
                if not raw_line:
                    break

                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue

                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Sonilo: skipping malformed NDJSON line")
                    continue

                evt_type = evt.get("type")
                if evt_type == "error":
                    code = evt.get("code", "UNKNOWN")
                    message = evt.get("message", "Unknown error")
                    raise Exception(f"Sonilo generation error ({code}): {message}")
                if evt_type == "duration":
                    duration_sec = evt.get("duration_sec")
                    if duration_sec is not None:
                        PromptServer.instance.send_progress_text(
                            f"Status: Generating\nVideo duration: {duration_sec:.1f}s",
                            node_id,
                        )
                elif evt_type in ("titles", "title"):
                    # v2m sends a "titles" list, t2m sends a scalar "title"
                    if evt_type == "titles":
                        titles = evt.get("titles", [])
                        if titles:
                            title = titles[0]
                    else:
                        title = evt.get("title") or title
                    if title:
                        PromptServer.instance.send_progress_text(
                            f"Status: Generating\nTitle: {title}",
                            node_id,
                        )
                elif evt_type == "audio_chunk":
                    stream_idx = evt.get("stream_index", 0)
                    chunk_data = base64.b64decode(evt["data"])

                    if stream_idx not in audio_streams:
                        audio_streams[stream_idx] = []
                    audio_streams[stream_idx].append(chunk_data)

                    now = time.monotonic()
                    if now - last_chunk_status_ts >= 1.0:
                        total_chunks = sum(len(chunks) for chunks in audio_streams.values())
                        elapsed = int(now - start_ts)
                        status_lines = ["Status: Receiving audio"]
                        if title:
                            status_lines.append(f"Title: {title}")
                        status_lines.append(f"Chunks received: {total_chunks}")
                        status_lines.append(f"Time elapsed: {elapsed}s")
                        PromptServer.instance.send_progress_text("\n".join(status_lines), node_id)
                        last_chunk_status_ts = now
                elif evt_type == "complete":
                    break

    if not audio_streams:
        raise Exception("Sonilo API returned no audio data.")

    PromptServer.instance.send_progress_text("Status: Completed", node_id)
    selected_stream = 0 if 0 in audio_streams else min(audio_streams)
    return b"".join(audio_streams[selected_stream])


async def _extract_error_message(resp: aiohttp.ClientResponse) -> str:
    """Extract a human-readable error message from an HTTP error response."""
    try:
        error_body = await resp.json()
        detail = error_body.get("detail", {})
        if isinstance(detail, dict):
            return detail.get("message", str(detail))
        return str(detail)
    except Exception:
        return await resp.text()


class SoniloExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [SoniloVideoToMusic, SoniloTextToMusic]


async def comfy_entrypoint() -> SoniloExtension:
    return SoniloExtension()
