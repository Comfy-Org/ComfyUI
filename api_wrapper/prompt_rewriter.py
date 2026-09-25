"""MiMo-backed prompt rewriting for the MiniMax H3 endpoints.

MiniMax H3 does not want a one-line description; it wants a structured prompt
with an alignment instruction, ``integrated_multimodal_description`` broken
into timed shots, ``overall_soundscape`` and ``non_diegetic_music`` (or, in
reference mode, six sections keyed by ``<Picture i>`` / ``<Video k>`` /
``<Audio j>`` labels). Writing one by hand is most of the work of using the
model at all.

This module puts an LLM in front of that: it hands Xiaomi's MiMo v2.5 MiniMax's
own H3 prompt-writing guide (vendored under ``h3_prompt_skill/``) plus the
concrete parameters of the job about to run -- mode, duration, canvas, which
reference labels exist -- and asks for the finished H3 prompt. Callers who
already have one send ``raw_prompt`` instead and skip all of this.

The endpoint is OpenAI chat-completions compatible, so ``MIMO_BASE_URL`` can
point at a third-party host of the same model instead of Xiaomi's.
"""

import asyncio
import base64
import json
import logging
import mimetypes
import os
import re

import aiohttp

# Xiaomi's own OpenAI-compatible endpoint. Every model id comes from the same
# API. mimo-v2.6-flash (released 2026-09-22) is omnimodal
# (text/image/audio/video), costs the same as mimo-v2.5, and Xiaomi rates it
# above mimo-v2.5-pro, so it writes every prompt by default. mimo-v2.6-pro is the
# larger build at ~3x the price. The V2.5 ids stay selectable.
DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
MODEL_FLASH = "mimo-v2.6-flash"
MODEL_PRO = "mimo-v2.6-pro"
MODEL_V25 = "mimo-v2.5"
MODEL_V25_PRO = "mimo-v2.5-pro"
MODEL_AUTO = "auto"
MODEL_OPTIONS = (MODEL_AUTO, MODEL_FLASH, MODEL_PRO, MODEL_V25, MODEL_V25_PRO)
# Confirmed against the live API: sending an image to mimo-v2.5-pro fails with
# HTTP 404 {"message": "No endpoints found that support image input"} -- it is
# not ignored, the whole request is rejected. mimo-v2.5 and both V2.6 models
# read images (checked 2026-09-25).
TEXT_ONLY_MODELS = frozenset({MODEL_V25_PRO})

SKILL_DIR = os.environ.get("H3_PROMPT_SKILL_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "h3_prompt_skill")

# Which guide covers which task. Base modes share one document; full-reference
# rewrites have their own six-section format.
BASE_GUIDE = "base-en.txt"
REF_GUIDE = "ref-en.txt"

MODE_T2VA = "T2VA"
MODE_I2VA = "I2VA"
MODE_FL2VA = "FL2VA"
MODE_L2VA = "L2VA"
MODE_REF2VA = "Ref2VA"

MODE_BRIEFS = {
    MODE_T2VA: "Text to video with audio. There is no reference image; build the whole "
               "audiovisual timeline from the request. Do not emit an alignment instruction line.",
    MODE_I2VA: "Image to video with audio. <Picture 1> is the real first frame at 0.00 seconds "
               "and belongs to [Shot 1]. Start the alignment instruction line exactly as the guide "
               "specifies for I2VA, then develop forward from that frame.",
    MODE_FL2VA: "First-and-last-frame to video with audio. Picture 1 is the opening frame and "
                "Picture 2 is the closing frame. Emit the FL2VA alignment instruction line and "
                "describe the continuous path between them.",
    MODE_L2VA: "Last-frame to video with audio. <Picture 1> is the final frame. Emit the L2VA "
               "alignment instruction line, infer a plausible earlier state and converge onto it.",
    MODE_REF2VA: "Full-reference mode. Follow the six-section format (subject_definitions, summary, "
                 "retention_analysis, detailed_description, overall_soundscape, non_diegetic_music) "
                 "and keep every reference label consistent across all sections.",
}

_OUTPUT_RULES = """
You are a MiniMax H3 prompt writer. Rewrite the user's request into one finished
H3 generation prompt, following the guide above exactly: same field names, same
section order, same label and timing notation.

Hard rules:
- Output the prompt and nothing else. No preamble, no commentary, no markdown
  code fences, no headings that the guide does not define.
- Use only the reference labels listed as available. Never invent a label for
  content the caller did not supply, and never leave a label unresolved.
- Every shot timestamp must fall inside the target duration, and the described
  action must fill that duration -- do not write a 15-second scene for a 5-second
  video.
- Keep dialogue, lyrics and on-screen text in their original language and
  wording; write everything else in English.
- Prefer concrete visual and audio detail over adjectives like "cinematic" or
  "beautiful".
"""

_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

_guide_cache = {}


class RewriteError(Exception):
    """The prompt could not be rewritten. ``status`` is the HTTP status the
    wrapper should surface: 400 for a caller/config problem, 502 when the model
    endpoint itself failed."""

    def __init__(self, message, details="", status=400):
        super().__init__(message)
        self.message = message
        self.details = details
        self.status = status


def api_key():
    return (os.environ.get("MIMO_API_KEY") or "").strip()


def is_configured():
    """True when the rewriter has an API key and can actually run."""
    return bool(api_key())


def base_url():
    return (os.environ.get("MIMO_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def default_model():
    return (os.environ.get("MIMO_MODEL") or MODEL_AUTO).strip() or MODEL_AUTO


def resolve_model(requested):
    """Pick the concrete model id for a rewrite.

    ``auto`` (the default) sends every rewrite to mimo-v2.6-flash, which reads
    the context image when there is one. Picking a model explicitly always wins
    -- including the text-only mimo-v2.5-pro with an image attached, which the
    caller may do knowingly.
    """
    choice = (requested or "").strip() or default_model()
    if choice == MODEL_AUTO:
        return MODEL_FLASH
    if choice not in MODEL_OPTIONS:
        raise RewriteError("Invalid llm_model",
                           f"llm_model must be one of: {', '.join(MODEL_OPTIONS)}.")
    return choice


def supports_images(model):
    return model not in TEXT_ONLY_MODELS


def resolve_model_and_image(requested, image, image_is_explicit):
    """Pick the model, and decide whether the context image can travel with it.

    A text-only model plus an image is a request the API refuses outright, so
    it never reaches the wire. How that is reported depends on who asked for
    the image: an ``llm_image`` the caller uploaded is a contradiction worth a
    clear error, while the context image the wrapper adds by itself (the
    keyframe or first reference) is just dropped, since the caller only chose
    the model. Returns (model, image, note)."""
    model = resolve_model(requested)
    if not image or supports_images(model):
        return model, image, None
    if image_is_explicit:
        raise RewriteError(
            "The selected model cannot read images",
            f"llm_model={model} is text-only, and the API rejects any request carrying an image. "
            f"Send llm_image with llm_model={MODEL_FLASH} (omnimodal) or drop llm_model "
            "to let 'auto' pick it for you, or remove llm_image to write the prompt from text "
            f"alone with {model}.")
    return model, None, (f"{model} is text-only, so the prompt was written from text alone; "
                         f"use llm_model={MODEL_FLASH} or 'auto' to give it visual context")


def load_guide(name):
    """Read one vendored guide, cached. Missing files are a deployment error,
    not a caller error, so they say where the directory is expected."""
    if name not in _guide_cache:
        path = os.path.join(SKILL_DIR, "references", name)
        try:
            with open(path, encoding="utf-8") as f:
                _guide_cache[name] = f.read()
        except OSError as e:
            raise RewriteError(
                "H3 prompt-writing guide is missing",
                f"Could not read {path} ({e}). Set H3_PROMPT_SKILL_DIR to a directory "
                "containing SKILL.md and references/, or restore api_wrapper/h3_prompt_skill.",
                status=500) from e
    return _guide_cache[name]


def load_skill():
    """The skill's own workflow notes, prepended to the guide."""
    if "SKILL.md" not in _guide_cache:
        try:
            with open(os.path.join(SKILL_DIR, "SKILL.md"), encoding="utf-8") as f:
                # Drop the YAML front matter: it is agent-runtime metadata, not
                # guidance the model needs.
                text = f.read()
                if text.startswith("---"):
                    parts = text.split("---", 2)
                    text = parts[2] if len(parts) == 3 else text
                _guide_cache["SKILL.md"] = text.strip()
        except OSError:
            _guide_cache["SKILL.md"] = ""
    return _guide_cache["SKILL.md"]


def detect_mode(task_name, has_first_frame, has_last_frame):
    """Map the wrapper's task plus which keyframes were uploaded onto the H3
    input mode the guide is written around."""
    if task_name == "reference":
        return MODE_REF2VA
    if has_first_frame and has_last_frame:
        return MODE_FL2VA
    if has_last_frame:
        return MODE_L2VA
    if has_first_frame:
        return MODE_I2VA
    return MODE_T2VA


def _reference_lines(mode, reference_counts):
    """Human-readable list of the labels the model is allowed to use."""
    lines = []
    if mode == MODE_I2VA:
        lines.append("<Picture 1> = the first frame supplied by the caller (0.00 s).")
    elif mode == MODE_FL2VA:
        lines.append("Picture 1 = the first frame supplied by the caller (0.00 s).")
        lines.append("Picture 2 = the last frame supplied by the caller (end of the video).")
    elif mode == MODE_L2VA:
        lines.append("<Picture 1> = the last frame supplied by the caller (end of the video).")
    elif mode == MODE_REF2VA:
        for label, count in (("Picture", reference_counts.get("images", 0)),
                             ("Video", reference_counts.get("videos", 0)),
                             ("Audio", reference_counts.get("audios", 0))):
            for i in range(1, count + 1):
                lines.append(f"<{label} {i}> = reference {label.lower()} {i} supplied by the caller.")
    return lines or ["None -- no reference assets were supplied."]


def build_messages(*, intent, mode, duration, frames, width, height,
                   reference_counts, image_data_url=None):
    """The system + user messages for one rewrite."""
    guide = load_guide(REF_GUIDE if mode == MODE_REF2VA else BASE_GUIDE)
    system = "\n\n".join(part for part in (load_skill(), guide, _OUTPUT_RULES.strip()) if part)

    brief = [
        f"Input mode: {mode}. {MODE_BRIEFS[mode]}",
        f"Target duration: {duration:.2f} seconds ({frames} frames at 24 fps). "
        f"Every timestamp you write must be below {duration:.2f}.",
        f"Target canvas: {width}x{height}.",
        "Available reference labels:",
    ]
    brief.extend(f"  {line}" for line in _reference_lines(mode, reference_counts))
    if image_data_url:
        brief.append("An image is attached for visual context. Ground the description in what you "
                     "actually see in it -- subjects, clothing, setting, lighting, framing -- so the "
                     "generated video stays consistent with it.")
    brief.append("\nUser request:\n" + intent.strip())

    text = "\n".join(brief)
    if image_data_url:
        content = [{"type": "text", "text": text},
                   {"type": "image_url", "image_url": {"url": image_data_url}}]
    else:
        content = text
    return [{"role": "system", "content": system},
            {"role": "user", "content": content}]


MAX_IMAGE_BYTES = 8 * 1024 * 1024


def data_url_from_bytes(data, filename="image.png"):
    """Base64 data URI for the chat API's image_url content part."""
    if len(data) > MAX_IMAGE_BYTES:
        raise RewriteError("llm_image is too large",
                           f"The context image must be at most {MAX_IMAGE_BYTES // (1024 * 1024)} MB "
                           f"(got {len(data) // (1024 * 1024)} MB). Downscale it before uploading.")
    mime = mimetypes.guess_type(filename)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def data_url_from_path(path):
    """Read an uploaded image off disk into a data URI."""
    try:
        with open(path, "rb") as f:
            return data_url_from_bytes(f.read(), path)
    except OSError as e:
        raise RewriteError("Could not read the context image", str(e)) from e


def clean_output(text):
    """Strip the wrappers a chat model tends to add around the prompt."""
    text = _THINK.sub("", text or "").strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    return text


async def _post(session, url, headers, payload, timeout):
    async with session.post(url, headers=headers, json=payload,
                            timeout=aiohttp.ClientTimeout(total=timeout)) as response:
        body = await response.text()
        if response.status != 200:
            raise RewriteError(
                "The prompt-rewriting model rejected the request",
                f"{url} returned HTTP {response.status}: {body[:500]}",
                status=502)
        try:
            return json.loads(body)
        except ValueError as e:
            raise RewriteError("The prompt-rewriting model returned invalid JSON",
                               body[:500], status=502) from e


async def rewrite(*, intent, mode, duration, frames, width, height,
                  reference_counts=None, image=None, model=None):
    """Rewrite ``intent`` into an H3 prompt. ``image`` is a data URI for visual
    context. Returns (prompt, model_used)."""
    if not intent or not intent.strip():
        raise RewriteError("No prompt provided", "'prompt' must not be empty.")
    if not is_configured():
        raise RewriteError(
            "Prompt rewriting is not configured",
            "MIMO_API_KEY is not set on the server, so 'prompt' cannot be rewritten into an "
            "H3 prompt. Set MIMO_API_KEY (Xiaomi MiMo, https://mimo.mi.com), or send a "
            "ready-made H3 prompt as 'raw_prompt' instead to skip the rewrite.")

    data_url = image
    model_used = resolve_model(model)
    messages = build_messages(
        intent=intent, mode=mode, duration=duration, frames=frames,
        width=width, height=height, reference_counts=reference_counts or {},
        image_data_url=data_url)

    # max_completion_tokens, the thinking object and both auth headers below are
    # all confirmed accepted by the live API on both V2.5 models and on
    # mimo-v2.6-flash. That the endpoint
    # *honours* thinking:disabled is inferred from latency and output shape
    # rather than observed -- the response does not report it back.
    payload = {
        "model": model_used,
        "messages": messages,
        "max_completion_tokens": int(os.environ.get("MIMO_MAX_TOKENS", "4096")),
        "thinking": {"type": os.environ.get("MIMO_THINKING", "disabled")},
    }
    # In thinking mode MiMo pins temperature/top_p to its own defaults, so only
    # send a temperature when we are not asking it to think.
    if payload["thinking"]["type"] == "disabled":
        payload["temperature"] = float(os.environ.get("MIMO_TEMPERATURE", "0.7"))

    url = f"{base_url()}/chat/completions"
    # Xiaomi documents `api-key` as its own header and also accepts Bearer;
    # every OpenAI-compatible gateway wants Bearer. Send both so MIMO_BASE_URL
    # can point at either without a code change.
    key = api_key()
    headers = {"Authorization": f"Bearer {key}", "api-key": key,
               "Content-Type": "application/json"}
    timeout = float(os.environ.get("MIMO_TIMEOUT", "120"))

    try:
        async with aiohttp.ClientSession() as session:
            data = await _post(session, url, headers, payload, timeout)
    except asyncio.TimeoutError as e:
        raise RewriteError("The prompt-rewriting model timed out",
                           f"No response from {url} within {timeout:.0f}s. Raise MIMO_TIMEOUT, or "
                           "send 'raw_prompt' to skip the rewrite.", status=502) from e
    except aiohttp.ClientError as e:
        raise RewriteError("Could not reach the prompt-rewriting model",
                           f"{url}: {e}", status=502) from e

    try:
        prompt = clean_output(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as e:
        raise RewriteError("The prompt-rewriting model returned no completion",
                           json.dumps(data)[:500], status=502) from e
    if not prompt:
        raise RewriteError("The prompt-rewriting model returned an empty prompt",
                           f"model={model_used}; try again or send 'raw_prompt' instead.",
                           status=502)
    logging.info("H3 prompt written by %s for %s (%d chars) from: %s",
                 model_used, mode, len(prompt), intent.strip().replace("\n", " ")[:200])
    return prompt, model_used
