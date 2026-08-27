"""
Custom-node SDK: resources as abstractions (refs), a brokered ``ctx``, and a
provider registry that lets a separable overlay swap the in-process default
implementation for an isolated one — without the node author changing a line.

Design (see docs/secure_custom_nodes_WIP.md):

* A node's heavyweight values cross ``execute()`` as opaque **refs**
  (``ImageRef``, ``ModelRef``, ``AssetRef`` ...), never raw buffers/paths.
* Side effects go through **ctx** domains (``ctx.assets``, ``ctx.progress`` ...)
  rather than ``folder_paths`` / ``PromptServer`` / ambient globals.
* Everything here ships a **default in-process implementation** so open-source
  ComfyUI behaves exactly as today (a ref just wraps the real object; ctx is a
  thin passthrough). Zero behavior change, better authoring API.
* The **overlay** (proprietary, cloud-only, loaded by ``COMFY_OVERLAY_MODULE``)
  calls ``providers.register_*`` at import to replace those defaults with the
  isolated engine: out-of-process guests, shm/CUDA-IPC refs, an enforcing
  broker. Uninstall the overlay -> pure OSS.

Nothing isolation-specific lives in this file. This is the *seam*, not the
engine.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import sys
import threading
import uuid
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Optional,
    Protocol,
    runtime_checkable,
)

from ._profiling import InProcessProfiling
from ._preview_override import InProcessPreviewOverride

if TYPE_CHECKING:  # keep this module import-safe / torch-free at import time
    import torch

    from ._io import NodeOutput

logger = logging.getLogger(__name__)

# Env var an operator points at a directory/module implementing ``register``.
OVERLAY_ENV = "COMFY_OVERLAY_MODULE"


# --------------------------------------------------------------------------- #
# Refs — opaque, typed handles. In OSS a ref is resolved by an identity table
# (zero-copy, zero-overhead: it holds the real object). The overlay swaps the
# resolver for shm/CUDA-IPC across a process boundary. The ref token itself
# carries nothing exploitable; the host table is authoritative.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Ref:
    """Base opaque resource handle. ``kind`` is the io_type (IMAGE, LATENT...)."""

    kind: str
    id: str

    def __repr__(self) -> str:  # keep ids short in logs, never leak contents
        return f"<{type(self).__name__} {self.id[:8]}>"

    async def release(self) -> None:
        """Optional early free. **A node never has to call this.**

        Every ref a node receives or creates is released when `execute` returns,
        so the ordinary node — materialize, compute, return — manages nothing.
        That is deliberate: an API whose correctness depends on authors
        remembering to free is an API that leaks in the field.

        This exists for the uncommon node that builds many large intermediates
        in a loop and wants them gone before the end of the call::

            for step in steps:
                nxt = await img.op("scale", factor=0.99)
                await img.release()      # optional: drop the previous one now
                img = nxt

        Using a released ref raises rather than resolving to anything else, so
        an early free that turns out to be wrong fails loudly and immediately.
        """
        await current_runtime().refs.release(self)


class _TypedRef(Ref):
    KIND: str = "ANY"

    @classmethod
    def _wrap(cls, ref: Ref) -> "Ref":
        return cls(kind=cls.KIND, id=ref.id)


class ValueRef(_TypedRef):
    """A handle whose value is buffer-safe structured data."""

    KIND = "VALUE"

    async def value(self) -> Any:
        """Read a dict/list of tensors and JSON scalars."""
        return await current_runtime().refs.resolve(self)

    @classmethod
    async def from_value(cls, v: Any) -> "Ref":
        return cls._wrap(await current_runtime().refs.create(cls.KIND, v))


class TensorRef(_TypedRef):
    KIND = "TENSOR"

    # --- RAW ESCAPE HATCH (permissioned, discouraged) -------------------- #
    # `raw()` returns the underlying buffer object. It is NOT the preferred
    # interface: prefer operations on the asset (see ImageRef below). Raw
    # access requires the `raw`/`tensor.read` capability and forces a node to
    # the dedicated tier under the overlay. The SDK contract does not depend on
    # torch; the return is deliberately untyped (Any) here.
    async def raw(self) -> Any:
        return await current_runtime().refs.resolve(self)

    @classmethod
    async def _from_raw(cls, obj: Any) -> "TensorRef":
        return cls._wrap(await current_runtime().refs.create(cls.KIND, obj))  # type: ignore[return-value]


class ImageRef(TensorRef):
    KIND = "IMAGE"

    # --- PREFERRED INTERFACE: operations on the asset, by name. The heavy
    #     compute runs engine-side (trusted plane); the node never receives a
    #     buffer. `op` is generic dispatch — core ships a tiny built-in set and
    #     an overlay can extend the vocabulary without changing this contract.
    async def op(self, name: str, **params: Any) -> "ImageRef":
        return await current_runtime().ops.apply(name, self, params)

    # Convenience wrappers over the two built-in primitives.
    async def invert(self) -> "ImageRef":
        return await self.op("invert")

    async def scale(self, factor: float) -> "ImageRef":
        return await self.op("scale", factor=factor)


class MaskRef(TensorRef):
    KIND = "MASK"


class LatentRef(ValueRef):
    KIND = "LATENT"

    async def minimax_h3_token_count(
        self, conditioning: "CondRef",
    ) -> dict[str, Any]:
        return await current_runtime().ops.apply(
            "latent.minimax_h3_token_count", self,
            {"conditioning": conditioning})


# --------------------------------------------------------------------------- #
# Handles for live engine objects — MODEL, CLIP, VAE, CONDITIONING, GUIDER.
#
# These keep the OLD API'S NATURAL SHAPE deliberately: you still write
# ``vae.decode(latent)`` and ``clip.encode(text)``, because that is the mental
# model every node author already has. What changes is what the call MEANS. In
# the old API the node held the VAE and ran the decode itself, which is exactly
# why a node taking a VAE could not be sandboxed. Here the node holds a handle,
# the call is awaited, and the decode happens on the trusted plane against
# weights the node never sees.
#
# Compatible in shape, different in substance — which is the point. Familiar
# enough to convert to mechanically; strict enough that a converted node is
# sandboxable by construction rather than by inspection.
#
# Dispatch goes through the same named-op registry ``ImageRef.op`` uses, so an
# overlay can extend the vocabulary without touching this contract.
# --------------------------------------------------------------------------- #
class CondRef(ValueRef):
    KIND = "CONDITIONING"

    async def sequence_length(self) -> int:
        return int(await current_runtime().ops.apply(
            "cond.sequence_length", self, {}))

    async def combine(self, other: "CondRef") -> "CondRef":
        return await current_runtime().ops.apply("cond.combine", self,
                                                 {"other": other})

    async def concat(self, other: "CondRef") -> "CondRef":
        return await current_runtime().ops.apply("cond.concat", self,
                                                 {"other": other})


class GuiderRef(_TypedRef):
    """Opaque handle to a host-owned sampling guider."""

    KIND = "GUIDER"


class SamplerRef(_TypedRef):
    """Opaque handle to a host-owned sampler."""

    KIND = "SAMPLER"

    @classmethod
    async def self_refine_video(
        cls, stochastic_steps: list[dict[str, int]],
        certain_percentage: float, uncertainty_threshold: float,
        seed: int, verbose: bool = False,
        latent: Optional["LatentRef"] = None,
    ) -> "SamplerRef":
        return await current_runtime().ops.apply(
            "sampler.self_refine_video", latent, {
                "stochastic_steps": stochastic_steps,
                "certain_percentage": certain_percentage,
                "uncertainty_threshold": uncertainty_threshold,
                "seed": seed,
                "verbose": verbose,
            })


class WeightDiffCursorRef(_TypedRef):
    """Execution-scoped iterator over host-owned model weight differences.

    Each item contains one optional ``TensorRef`` plus its host-derived output
    key metadata. Advancing invalidates the previous tensor ref, so process or
    materialize that tensor before requesting the next item.
    """

    KIND = "WEIGHT_DIFF_CURSOR"

    async def next(self) -> Optional[dict[str, Any]]:
        return await current_runtime().ops.apply(
            "weight_diff.next", self, {})

    def __aiter__(self) -> "WeightDiffCursorRef":
        return self

    async def __anext__(self) -> dict[str, Any]:
        item = await self.next()
        if item is None:
            raise StopAsyncIteration
        return item


class ControlNetWeightsRef(_TypedRef):
    """Opaque Advanced-ControlNet weight policy."""

    KIND = "CONTROL_NET_WEIGHTS"

    @classmethod
    async def from_list(
        cls, weights: list[float], uncond_multiplier: float = 1.0,
        extras: Any = None,
    ) -> tuple["ControlNetWeightsRef", "TimestepKeyframeRef"]:
        result = await current_runtime().ops.apply(
            "advanced_control.weights_from_list", None, {
                "weights": weights,
                "uncond_multiplier": uncond_multiplier,
                "extras": {} if extras is None else extras,
            })
        return result[0], result[1]


class TimestepKeyframeRef(_TypedRef):
    """Opaque Advanced-ControlNet timestep-keyframe schedule."""

    KIND = "TIMESTEP_KEYFRAME"


class ModelRef(_TypedRef):
    KIND = "MODEL"

    # A MODEL crosses as a handle only. There is deliberately no `clone()`:
    # in the original API `model.clone()` is never the last line, it exists so
    # the next line can patch the copy, and a clone nothing can mutate has no
    # valid second line. `patch` below is that second line, expressed as data.

    async def patch(self, transform: str, **params: Any) -> "ModelRef":
        """Apply a named transform, returning a NEW model ref.

        The vocabulary is closed and core-owned (`_model_transforms.py`). You
        name a behaviour; the host decides what it means and does the clone.

            model = await model.patch("attention_impl", mode="sage")
            model = await model.patch("ffn_chunking", chunks=4)

        Transforms stack — each call returns a new ref, so the second line
        above does not undo the first. The return value is the whole point: a
        `patch` whose result is discarded has done nothing.

        This is not the original API's `set_model_attn1_patch(fn)`. That takes
        a FUNCTION, and a function is code, which is the one thing that cannot
        cross into the trusted process. Every transform in the table exists
        because the node asking for it wanted an engine setting, not a
        computation — a dropdown value, a chunk count, a bool.
        """
        for key, value in params.items():
            if callable(value):
                raise TypeError(
                    f"patch({transform!r}) was given a function for {key!r}. "
                    f"Transforms take DATA — a choice, a number, a flag — "
                    f"because guest code cannot run in the host process. If "
                    f"the behaviour you need is not in the host's transform "
                    f"table, it belongs in core, not in a callback")
        return await current_runtime().ops.apply(
            "model.patch", self, {"transform": transform, "params": params})

    async def transforms(self) -> list[dict]:
        """Return the transforms supported by the active host."""
        return await current_runtime().ops.apply("model.transforms", self, {})

    async def latent_scale_factor(self) -> float:
        return float(await current_runtime().ops.apply(
            "model.latent_scale_factor", self, {}))

    async def scheduled_cfg_guider(
        self, positive: "CondRef", negative: "CondRef", cfg: float,
        start_percent: float, end_percent: float,
    ) -> "GuiderRef":
        return await current_runtime().ops.apply(
            "guider.scheduled_cfg", self, {
                "positive": positive,
                "negative": negative,
                "cfg": cfg,
                "start_percent": start_percent,
                "end_percent": end_percent,
            })

    async def lora_weight_differences(
        self, original: "ModelRef", include_bias: bool = False,
    ) -> "WeightDiffCursorRef":
        return await current_runtime().ops.apply(
            "lora.weight_differences", self, {
                "original": original,
                "include_bias": bool(include_bias),
            })

    async def apply_dit_block_lora(
        self, asset: "AssetRef", strength_model: float,
        block_weights: list[dict[str, Any]],
    ) -> tuple["ModelRef", str]:
        result = await current_runtime().ops.apply(
            "model.apply_dit_block_lora", self, {
                "asset": asset,
                "strength_model": strength_model,
                "block_weights": block_weights,
            })
        return result[0], str(result[1])

    async def apply_ltx2_lora(
        self, asset: "AssetRef", strength_model: float,
        block_weights: list[dict[str, Any]], video: float,
        video_to_audio: float, audio: float, audio_to_video: float,
        other: float,
    ) -> tuple["ModelRef", str, str]:
        result = await current_runtime().ops.apply(
            "model.apply_ltx2_lora", self, {
                "asset": asset,
                "strength_model": strength_model,
                "block_weights": block_weights,
                "video": video,
                "video_to_audio": video_to_audio,
                "audio": audio,
                "audio_to_video": audio_to_video,
                "other": other,
            })
        return result[0], str(result[1]), str(result[2])


class ClipRef(_TypedRef):
    KIND = "CLIP"

    async def tokenize(self, text: str, **kwargs: Any) -> dict:
        """Mirrors ``clip.tokenize``, including its per-model kwargs.

        Tokenizing is a separate step from encoding because real nodes work
        between the two. ``CLIPTextEncodeSDXL`` builds one token dict from two
        prompts (``tokens["l"] = clip.tokenize(text_l)["l"]``) and pads the
        halves to equal length; the ACE nodes pass lyrics, bpm, duration and
        more as tokenizer kwargs. A combined call alone cannot express either.

        Tokens are plain data and cross the wire as themselves. Where a model's
        tokenizer returns tensors (image-conditioned encoders do), this raises
        at the wire rather than silently degrading.
        """
        return await current_runtime().ops.apply("clip.tokenize", self,
                                                 {"text": text, "kwargs": kwargs})

    async def encode_from_tokens_scheduled(self, tokens: dict,
                                           add_dict: dict | None = None) -> "CondRef":
        """Mirrors ``clip.encode_from_tokens_scheduled``, ``add_dict`` included.

        ``add_dict`` carries the conditioning extras every SDXL-family encoder
        sets — width, height, crop, target size — so it is part of the call, not
        an optional extra.
        """
        return await current_runtime().ops.apply(
            "clip.encode_from_tokens_scheduled", self,
            {"tokens": tokens, "add_dict": add_dict})

    async def encode(self, text: str) -> "CondRef":
        """The two steps above in one call, for the common case.

        Exactly what ``CLIPTextEncode`` does, and it saves a wire round trip.
        A convenience over the pair, not a replacement: anything that inspects
        or edits tokens uses ``tokenize`` + ``encode_from_tokens_scheduled``.
        """
        return await current_runtime().ops.apply("clip.encode", self,
                                                 {"text": text})

    async def lora_weight_differences(
        self, original: "ClipRef", include_bias: bool = False,
    ) -> "WeightDiffCursorRef":
        return await current_runtime().ops.apply(
            "lora.weight_differences", self, {
                "original": original,
                "include_bias": bool(include_bias),
            })


class GligenRef(_TypedRef):
    KIND = "GLIGEN"

    async def apply_batched(
        self, conditioning: "CondRef", clip: "ClipRef", text: str,
        boxes: list[tuple[int, int, int | float, int | float]],
    ) -> "CondRef":
        return await current_runtime().ops.apply(
            "gligen.apply_batched", self, {
                "conditioning": conditioning,
                "clip": clip,
                "text": text,
                "boxes": boxes,
            })


class VaeRef(_TypedRef):
    KIND = "VAE"

    async def decode(self, latent: "LatentRef") -> "ImageRef":
        return await current_runtime().ops.apply("vae.decode", self,
                                                 {"latent": latent})

    async def encode(self, image: "ImageRef") -> "LatentRef":
        """Mirrors ``vae.encode`` exactly. The caller owns any channel slicing.

        Pixels pass through untouched, matching core's ``VAEEncode``
        (``t = vae.encode(pixels)``). Slicing here would change results rather
        than shape — silently dropping alpha for a four-channel caller — and be
        invisible from the calling node's own source.
        """
        return await current_runtime().ops.apply("vae.encode", self,
                                                 {"image": image})

    async def encode_tiled(
        self, image: "ImageRef", tile_x: Optional[int] = None,
        tile_y: Optional[int] = None, overlap: Optional[int] = None,
        tile_t: Optional[int] = None, overlap_t: Optional[int] = None,
    ) -> "LatentRef":
        return await current_runtime().ops.apply("vae.encode_tiled", self, {
            "image": image,
            "tile_x": tile_x,
            "tile_y": tile_y,
            "overlap": overlap,
            "tile_t": tile_t,
            "overlap_t": overlap_t,
        })

    async def input_dtype(self) -> str:
        """The CPU dtype a node should use before VAE-specific preprocessing."""
        return str(await current_runtime().ops.apply(
            "vae.input_dtype", self, {}))

    async def encode_video(
        self, image: "ImageRef",
    ) -> tuple["LatentRef", int]:
        """Encode frames after applying this VAE's temporal frame constraint."""
        result = await current_runtime().ops.apply(
            "vae.encode_video", self, {"image": image})
        return result[0], int(result[1])

    async def decode_video(
        self, latent: "LatentRef", *, tiled: bool = False,
        tile_size: int = 512, overlap: int = 64,
        temporal_size: int = 4096, temporal_overlap: int = 16,
    ) -> "ImageRef":
        """Decode image/video latents and flatten video batches to frames."""
        return await current_runtime().ops.apply("vae.decode_video", self, {
            "latent": latent,
            "tiled": bool(tiled),
            "tile_size": int(tile_size),
            "overlap": int(overlap),
            "temporal_size": int(temporal_size),
            "temporal_overlap": int(temporal_overlap),
        })

    async def decode_audio(self, latent: "LatentRef") -> "AudioRef":
        """Decode an audio latent and attach the VAE's output sample rate."""
        return await current_runtime().ops.apply(
            "vae.decode_audio", self, {"latent": latent})

    async def downscale_index_formula(self) -> Optional[tuple[int, int, int]]:
        value = await current_runtime().ops.apply(
            "vae.downscale_index_formula", self, {})
        if value is None:
            return None
        return tuple(int(item) for item in value)

    async def merge(self, other: "VaeRef", ratio: float = 0.5) -> "VaeRef":
        return await current_runtime().ops.apply(
            "vae.merge", self, {"other": other, "ratio": float(ratio)})

    async def compile(
        self, *, backend: str = "inductor", mode: str = "default",
        fullgraph: bool = False, encoder: bool = True, decoder: bool = True,
    ) -> "VaeRef":
        return await current_runtime().ops.apply("vae.compile", self, {
            "backend": backend,
            "mode": mode,
            "fullgraph": fullgraph,
            "encoder": encoder,
            "decoder": decoder,
        })

    async def patch_triton(
        self, *, fuse_norm_silu: bool = True, channels_last: bool = True,
        int8_conv: bool = False, autotune: bool = False,
    ) -> "VaeRef":
        return await current_runtime().ops.apply("vae.patch_triton", self, {
            "fuse_norm_silu": fuse_norm_silu,
            "channels_last": channels_last,
            "int8_conv": int8_conv,
            "autotune": autotune,
        })


class ClipVisionOutputRef(_TypedRef):
    KIND = "CLIP_VISION_OUTPUT"

    async def image_embeds(self) -> TensorRef:
        return await current_runtime().ops.apply(
            "clip_vision_output.image_embeds", self, {})


class ClipVisionRef(_TypedRef):
    KIND = "CLIP_VISION"

    async def encode_image(
        self, image: ImageRef, crop: bool = True,
    ) -> ClipVisionOutputRef:
        return await current_runtime().ops.apply(
            "clip_vision.encode_image", self,
            {"image": image, "crop": bool(crop)})


class ControlNetRef(_TypedRef):
    KIND = "CONTROL_NET"

    async def with_union_type(self, type_number: Optional[int]) -> "ControlNetRef":
        return await current_runtime().ops.apply(
            "controlnet.with_union_type", self,
            {"type_number": type_number})

    async def compile(
        self, *, backend: str = "inductor", mode: str = "default",
        fullgraph: bool = False,
    ) -> "ControlNetRef":
        return await current_runtime().ops.apply("controlnet.compile", self, {
            "backend": backend,
            "mode": mode,
            "fullgraph": fullgraph,
        })


class StyleModelRef(_TypedRef):
    KIND = "STYLE_MODEL"

    async def apply(
        self, clip_vision_output: ClipVisionOutputRef,
        conditioning: CondRef, strength: float = 1.0,
    ) -> CondRef:
        return await current_runtime().ops.apply(
            "style_model.apply", self, {
                "clip_vision_output": clip_vision_output,
                "conditioning": conditioning,
                "strength": float(strength),
            })


class ClipSegRef(_TypedRef):
    KIND = "CLIPSEGMODEL"

    async def segment(
        self, images: ImageRef, text: str, threshold: float = 0.5,
        binary_mask: bool = True, combine_mask: bool = False,
        use_accelerator: bool = True, blur_sigma: float = 0.0,
        previous_mask: Optional[MaskRef] = None, invert: bool = False,
        image_background_level: float = 0.5,
    ) -> tuple[MaskRef, ImageRef]:
        result = await current_runtime().ops.apply(
            "clipseg.segment", self, {
                "images": images,
                "text": str(text),
                "threshold": float(threshold),
                "binary_mask": bool(binary_mask),
                "combine_mask": bool(combine_mask),
                "use_accelerator": bool(use_accelerator),
                "blur_sigma": float(blur_sigma),
                "previous_mask": previous_mask,
                "invert": bool(invert),
                "image_background_level": float(image_background_level),
            })
        return result[0], result[1]


class UpscaleModelRef(_TypedRef):
    KIND = "UPSCALE_MODEL"

    async def upscale(
        self, images: ImageRef, per_batch: int = 16,
        downscale_ratio: float = 1.0, downscale_method: str = "lanczos",
        precision: str = "float32",
    ) -> ImageRef:
        return await current_runtime().ops.apply(
            "upscale_model.upscale", self, {
                "images": images,
                "per_batch": int(per_batch),
                "downscale_ratio": float(downscale_ratio),
                "downscale_method": str(downscale_method),
                "precision": str(precision),
            })


class AudioRef(ValueRef):
    KIND = "AUDIO"


class VideoRef(_TypedRef):
    KIND = "VIDEO"

    async def encoded_source(self) -> ValueRef:
        """Return encoded bytes and trim metadata, never the source path."""
        return await current_runtime().ops.apply(
            "video.encoded_source", self, {})


class AssetRef(_TypedRef):
    """A file/model resolved by name+hash, tenant-scoped. Never a raw path."""

    KIND = "ASSET"


class OpaqueRef(_TypedRef):
    """A pass-through handle for a value with no sandbox materializer."""

    KIND = "OPAQUE"


# --------------------------------------------------------------------------- #
# Runtime — per-execution binding of (ref resolver, ctx), set by the active
# ExecutionBackend. Refs read it via a contextvar, mirroring how the engine
# exposes get_executing_context(). Authors never construct it.
# --------------------------------------------------------------------------- #
@dataclass
class Runtime:
    refs: "RefResolver"
    ctx: "Context"
    ops: "OpsProvider" = None  # engine-side operations (the preferred interface)


_active_runtime: "contextvars.ContextVar[Optional[Runtime]]" = contextvars.ContextVar(
    "comfy_sdk_runtime", default=None
)


def current_runtime() -> Runtime:
    rt = _active_runtime.get()
    if rt is None:
        raise RuntimeError(
            "No active Comfy SDK runtime. Ref/ctx access is only valid inside "
            "node execution."
        )
    return rt


def current_context() -> "Context":
    return current_runtime().ctx


class _RuntimeScope:
    """Context manager the ExecutionBackend uses to bind a runtime per node."""

    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime
        self._token: Any = None

    def __enter__(self) -> Runtime:
        self._token = _active_runtime.set(self._runtime)
        return self._runtime

    def __exit__(self, *exc: Any) -> None:
        _active_runtime.reset(self._token)

    @property
    def runtime(self) -> Runtime:
        return self._runtime


def bind_runtime(
    refs: "RefResolver", ctx: "Context", ops: "OpsProvider" = None
) -> _RuntimeScope:
    return _RuntimeScope(Runtime(refs=refs, ctx=ctx, ops=ops))


# --------------------------------------------------------------------------- #
# Provider interfaces. OSS ships the defaults below; the overlay overrides.
# --------------------------------------------------------------------------- #
@runtime_checkable
class RefResolver(Protocol):
    async def create(self, kind: str, obj: Any) -> Ref: ...
    async def resolve(self, ref: Ref) -> Any: ...
    async def release(self, ref: Ref) -> None: ...


_V2_NODE_METHODS = frozenset({
    "execute",
    "validate_inputs",
    "fingerprint_inputs",
    "check_lazy_status",
})
_V2_NODE_METHOD_ALIASES = {
    "EXECUTE_NORMALIZED": "execute",
    "EXECUTE_NORMALIZED_ASYNC": "execute",
}


def _normalize_v2_node_method(method: str) -> str:
    method = _V2_NODE_METHOD_ALIASES.get(method, method)
    if method not in _V2_NODE_METHODS:
        raise ValueError(
            f"V2 node method {method!r} is not allowed; "
            f"expected one of {sorted(_V2_NODE_METHODS)}")
    return method


@dataclass
class ExecutionPlan:
    """What the execution seam hands the backend to decide placement."""

    prompt_id: str
    node_id: str
    node_type: str
    tier: str = "default"  # overlay reads manifest tier; OSS is always "default"
    permissions: tuple[str, ...] = ()
    # Work-unit payload for out-of-process backends. ``refs`` means the node
    # explicitly consumes SDK handles; ``values`` means the backend may wrap
    # for transport and the guest must materialize those handles before
    # invoking the unchanged V2 body. In-process dispatch ignores the payload.
    node_module: str = ""
    inputs: Optional[dict] = None
    input_mode: str = "refs"
    prompt: Any = None
    extra_pnginfo: Any = None
    method: str = "execute"

    def __post_init__(self) -> None:
        self.method = _normalize_v2_node_method(self.method)
        if self.input_mode not in {"refs", "values"}:
            raise ValueError(
                f"V2 node input mode {self.input_mode!r} is not allowed; "
                f"expected 'refs' or 'values'")


@runtime_checkable
class ExecutionBackend(Protocol):
    async def dispatch(
        self,
        plan: ExecutionPlan,
        local_call: Callable[[], Awaitable[Any]],
        runtime: Optional[Runtime] = None,
    ) -> Any:
        """Run the node. Default just awaits ``local_call`` (in-process). The
        overlay routes ``tier == 'sandbox'`` nodes to a guest process instead,
        and calls ``local_call`` only for nodes that stay local. ``runtime`` is
        the host-side binding (refs/ctx/ops) for this node so an out-of-process
        backend can serve brokered guest calls against the same ref table."""
        ...


@runtime_checkable
class CtxProvider(Protocol):
    def build(self, plan: ExecutionPlan) -> "Context": ...


@runtime_checkable
class OpsProvider(Protocol):
    """Engine-side operations on assets — the preferred node interface. The
    node passes/receives refs; the buffer math happens here, on the trusted
    plane (in-process default) or in the engine (overlay). Dispatch is generic
    (``apply(op, image, params)``) so the op vocabulary is data, not API
    surface: an overlay adds ops without changing this contract, and a node
    can probe ``supports(op)`` to choose a fallback (e.g. the ``raw`` tier)."""

    # `subject` is the ref the op acts on — an ImageRef for the pixel ops, but
    # a VaeRef/ClipRef/ModelRef for the engine-object ops. The return is
    # deliberately Any: most ops yield a ref, `clip.tokenize` yields a plain
    # token dict, and annotating that as ImageRef was simply untrue.
    async def apply(self, op: str, subject: "Ref", params: dict) -> Any: ...
    def supports(self, op: str) -> bool: ...


# --------------------------------------------------------------------------- #
# ctx — the brokered side-effect surface. Interfaces first; in-process defaults
# implement the important ones over real core. Domains not needed by the POC
# are declared and stubbed so the shape is fixed.
# --------------------------------------------------------------------------- #
class AssetsDomain(Protocol):
    async def resolve(self, folder: str, name: str) -> AssetRef: ...
    async def path(self, ref: AssetRef) -> str: ...
    async def list(
        self, folder: str, prefix: str = "", recursive: bool = True,
    ) -> list[str]: ...
    async def read_bytes(self, ref: AssetRef) -> bytes: ...
    async def load_state_dict(
        self, ref: AssetRef, return_metadata: bool = False,
    ) -> Any: ...


class ProgressDomain(Protocol):
    async def update(self, value: float, total: float,
                     preview: Optional[ImageRef] = None) -> None: ...


class ScratchDomain(Protocol):
    async def dir(self) -> str: ...


class EventsDomain(Protocol):
    async def emit(self, event: str, data: dict) -> None: ...


class StorageDomain(Protocol):
    async def get(self, key: str) -> Optional[str]: ...
    async def set(self, key: str, value: str) -> None: ...


class CaptureDomain(Protocol):
    async def screen(self, region: Optional[tuple[int, int, int, int]] = None,
                     monitor: int = 1) -> ImageRef: ...
    async def camera(self, index: int = 0, width: Optional[int] = None,
                     height: Optional[int] = None) -> ImageRef: ...
    async def audio(self, duration: float, sample_rate: int = 44100,
                    channels: int = 1) -> AudioRef: ...


class UiDomain(Protocol):
    async def preview_images(self, images: ImageRef,
                             animated: bool = False) -> dict: ...
    async def preview_mask(self, mask: MaskRef,
                           animated: bool = False) -> dict: ...
    async def preview_audio(self, audio: AudioRef) -> dict: ...
    async def preview_animation(
        self, images: ImageRef, fps: float = 8.0,
    ) -> dict: ...
    async def preview_batch(
        self, value: TensorRef, max_thumb_size: int = 512,
        crf: int = 25, max_grid_frames: int = 1024,
    ) -> dict: ...


class OutputDomain(Protocol):
    async def save_images(
        self, images: ImageRef, filename_prefix: str = "ComfyUI",
        subfolder: str = "", compress_level: int = 4,
        caption: Optional[str] = None,
        caption_extension: str = ".txt",
    ) -> dict: ...
    async def save_images_with_alpha(
        self, images: ImageRef, mask: MaskRef,
        filename_prefix: str = "ComfyUI",
        subfolder: str = "", compress_level: int = 4,
    ) -> dict: ...
    async def save_text(
        self, text: str, filename_prefix: str = "text",
        subfolder: str = "", extension: str = ".txt",
    ) -> str: ...
    async def save_state_dict(
        self, state_dict: ValueRef, filename_prefix: str,
        metadata: Optional[dict[str, str]] = None,
    ) -> str: ...
    async def save_model(
        self, model: ModelRef, filename_prefix: str,
        model_key_prefix: str = "model.diffusion_model.",
    ) -> str: ...
    async def save_video(
        self, images: ImageRef, audio: Optional[AudioRef] = None,
        fps: float = 25.0, filename_prefix: str = "video/ComfyUI",
        format: str = "auto", codec: str = "auto",
    ) -> dict: ...


class GraphDomain(Protocol):
    async def widget_values(
        self, node_id: int | str = 0, node_title: str = "",
        linked_input: str = "any_input",
    ) -> dict[str, Any]: ...


class ModelsDomain(Protocol):
    async def list_diffusion_models(
        self, include_connectors: bool = False,
    ) -> list[str]: ...
    async def load_checkpoint(
        self, name: str, weight_dtype: str = "default",
        compute_dtype: str = "default", cublas_linear: bool = False,
    ) -> tuple[ModelRef, Optional[ClipRef], Optional[VaeRef]]: ...
    async def load_diffusion_model(
        self, name: str, extra_name: Optional[str] = None,
        weight_dtype: str = "default", compute_dtype: str = "default",
        cublas_linear: bool = False,
    ) -> ModelRef: ...
    async def load_gguf_model(
        self, name: str, extra_name: Optional[str] = None,
        dequant_dtype: str = "default", patch_dtype: str = "default",
        patch_on_device: bool = False,
    ) -> ModelRef: ...
    async def list_vae(self) -> list[str]: ...
    async def load_vae(
        self, name: str, device: str = "default",
        weight_dtype: str = "default",
    ) -> VaeRef: ...
    async def load_clipseg(self, model: str) -> ClipSegRef: ...
    async def generate_text(
        self, generator: str, input_text: str, max_new_tokens: int = 128,
    ) -> str: ...
    async def memory_cleanup(
        self, empty_cache: bool = True, collect_cycles: bool = True,
        unload_all_models: bool = False,
    ) -> tuple[int, int]: ...


class ProfilingDomain(Protocol):
    async def cuda_memory_start(
        self, *, enabled: str = "all", context: str = "all",
        stacks: str = "all", max_entries: int = 100000,
    ) -> None: ...
    async def cuda_memory_end(
        self, filename_prefix: str = "comfy_cuda_memory_history",
    ) -> str: ...
    async def cuda_memory_visualize(self, snapshot: str) -> str: ...


class PreviewOverrideDomain(Protocol):
    async def attach(
        self, model: ModelRef, *, max_resolution: int = 1024,
        jpeg_quality: int = 80, suppress_default_preview: bool = True,
        preview_frames: int = 1, preview_fps: int = 12,
        vae: Optional[VaeRef] = None, tiny_vae: str = "none",
    ) -> ModelRef: ...
    async def attach_ltx2(
        self, model: ModelRef, *, preview_rate: float = 8.0,
        latent_upscale_model: Optional[Ref] = None,
        vae: Optional[VaeRef] = None,
    ) -> ModelRef: ...
    async def frames(
        self, model: ModelRef, after_sample: Ref,
    ) -> ImageRef: ...


class Context(Protocol):
    assets: AssetsDomain
    progress: ProgressDomain
    scratch: ScratchDomain
    events: EventsDomain
    storage: StorageDomain
    capture: CaptureDomain
    ui: UiDomain
    output: OutputDomain
    graph: GraphDomain
    models: ModelsDomain
    profiling: ProfilingDomain
    preview_override: PreviewOverrideDomain
    # Declared for the contract; overlay/full-SDK implement. Stubbed in OSS
    # default until wired: models, sample, serve, secrets, net.


# --------------------------------------------------------------------------- #
# Default in-process implementations (OSS). All heavy imports are lazy so this
# module stays import-safe.
# --------------------------------------------------------------------------- #
class InProcessRefResolver:
    """Identity table. A ref wraps the real object; resolve returns it as-is.

    **The table is authoritative for a ref's kind.** Ref tokens cross the wire
    as ``{"kind": ..., "id": ..., "cls": ...}``, so the label on an arriving
    token is whatever the sender put there. `resolve` therefore checks it
    against the kind recorded at creation and refuses a mismatch.

    Unguessable ids (uuid4) stop a holder reaching a ref it was never handed.
    This check is the other half: possessing a handle is not the same as
    labelling it, and without the check an IMAGE id presented as
    ``{"cls": "VaeRef", "kind": "VAE"}`` would reach an op that dispatches on
    kind, handing a tensor to code expecting a VAE.
    """

    def __init__(self) -> None:
        self._table: dict[str, tuple[str, Any]] = {}
        self._closed = False

    async def create(self, kind: str, obj: Any) -> Ref:
        if self._closed:
            raise RuntimeError("this node execution has ended; its ref table is closed")
        rid = uuid.uuid4().hex
        self._table[rid] = (kind, obj)
        return Ref(kind=kind, id=rid)

    async def resolve(self, ref: Ref) -> Any:
        entry = self._table.get(ref.id)
        if entry is None:
            raise KeyError(f"unknown ref {ref!r}")
        kind, obj = entry
        if ref.kind != kind:
            raise TypeError(
                f"ref {ref.id[:8]} was created as {kind} but presented as "
                f"{ref.kind}; the holder of a handle does not get to relabel it")
        return obj

    async def release(self, ref: Ref) -> None:
        self._table.pop(ref.id, None)

    def clear(self) -> int:
        """Drop every entry. Returns how many.

        Called at the end of a node execution so the table's strong references
        go at a known point, rather than whenever the interpreter next collects
        the frame that owned it. A ref table can hold multi-gigabyte tensors;
        "freed eventually" is not a lifetime for those, and a reference cycle
        anywhere in the graph defers it indefinitely.

        Refcount timing also does not cross a process boundary, so it can never
        be the whole answer here — the out-of-band channel releases explicitly
        (`transport/shm.py`), and this is the in-process half of the same rule:
        nothing waits for the collector.
        """
        n = len(self._table)
        self._table.clear()
        self._closed = True
        return n


class _InProcessAssets:
    _LIST_MAX = 4096

    @staticmethod
    def _confined_path(base: str, name: str, folder: str) -> str:
        base = os.path.realpath(os.path.abspath(base))
        logical = str(name or "").replace("\\", "/")
        if "\x00" in logical or logical.startswith("/"):
            raise ValueError(f"asset name escapes the {folder} directory")
        parts = [part for part in logical.split("/") if part not in ("", ".")]
        if any(part == ".." for part in parts):
            raise ValueError(f"asset name escapes the {folder} directory")
        full = os.path.realpath(os.path.join(base, *parts))
        try:
            confined = os.path.commonpath((base, full)) == base
        except ValueError:
            confined = False
        if not confined:
            raise ValueError(f"asset name escapes the {folder} directory")
        return full

    @staticmethod
    def _confined_resolved_path(
        path: str, roots: list[str], folder: str,
    ) -> str:
        full = os.path.realpath(path)
        for candidate in roots:
            root = os.path.realpath(os.path.abspath(candidate))
            try:
                if os.path.commonpath((root, full)) == root:
                    return full
            except ValueError:
                continue
        raise ValueError(f"asset name escapes the {folder} directory")

    async def resolve(self, folder: str, name: str) -> AssetRef:
        import folder_paths  # lazy

        standard_folders = {
            "input": folder_paths.get_input_directory,
            "output": folder_paths.get_output_directory,
            "temp": folder_paths.get_temp_directory,
        }
        if folder in standard_folders:
            base = standard_folders[folder]()
            full = self._confined_path(base, name, folder)
            if not os.path.isfile(full):
                raise FileNotFoundError(f"no {folder} asset named {name!r}")
        else:
            roots = folder_paths.get_folder_paths(folder)
            full = None
            for root in roots:
                candidate = self._confined_path(root, name, folder)
                if os.path.isfile(candidate):
                    full = candidate
                    break
            if full is None:
                raise FileNotFoundError(
                    f"no {folder} asset named {name!r}")
        return AssetRef._wrap(await current_runtime().refs.create("ASSET", full))  # type: ignore[return-value]

    async def path(self, ref: AssetRef) -> str:
        return await current_runtime().refs.resolve(ref)

    async def list(
        self, folder: str, prefix: str = "", recursive: bool = True,
    ) -> list[str]:
        import folder_paths

        if folder != "input":
            names = sorted(
                str(name).replace("\\", "/")
                for name in folder_paths.get_filename_list(folder))
            logical_prefix = str(prefix or "").replace("\\", "/").strip("/")
            if logical_prefix:
                if "\x00" in logical_prefix or any(
                    part == ".." for part in logical_prefix.split("/")):
                    raise ValueError("asset prefix escapes the catalogue")
                marker = logical_prefix + "/"
                names = [
                    name for name in names
                    if name == logical_prefix or name.startswith(marker)]
                if not recursive:
                    names = [
                        name for name in names
                        if "/" not in name[len(marker):]]
            elif not recursive:
                names = [name for name in names if "/" not in name]
            if len(names) > self._LIST_MAX:
                raise ValueError(
                    f"asset catalogue exceeds {self._LIST_MAX} names")
            return names

        base = os.path.realpath(os.path.abspath(
            folder_paths.get_input_directory()))
        directory = self._confined_path(base, prefix, "input")
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"no input asset directory named {prefix!r}")

        names: list[str] = []
        for root, directories, files in os.walk(
                directory, followlinks=False):
            directories.sort()
            files.sort()
            for filename in files:
                full = os.path.join(root, filename)
                real = os.path.realpath(full)
                try:
                    confined = os.path.commonpath((base, real)) == base
                except ValueError:
                    confined = False
                if not confined or not os.path.isfile(real):
                    continue
                names.append(os.path.relpath(full, base).replace(os.sep, "/"))
                if len(names) > self._LIST_MAX:
                    raise ValueError(
                        f"input asset catalogue exceeds {self._LIST_MAX} names")
            if not recursive:
                break
        return sorted(names)

    async def read_bytes(self, ref: AssetRef) -> bytes:
        path = await self.path(ref)
        with open(path, "rb") as file:
            return file.read()

    async def load_state_dict(
        self, ref: AssetRef, return_metadata: bool = False,
    ) -> Any:
        import comfy.utils

        path = await self.path(ref)
        return comfy.utils.load_torch_file(
            path, safe_load=True, return_metadata=bool(return_metadata))


def _load_sdk_diffusion_model(
    model_path: str, model_options: Optional[dict] = None,
    extra_path: Optional[str] = None,
):
    import comfy.sd
    import comfy.utils

    options = {} if model_options is None else dict(model_options)
    state_dict, metadata = comfy.utils.load_torch_file(
        model_path, return_metadata=True)
    if extra_path is not None:
        state_dict.update(comfy.utils.load_torch_file(extra_path))
        prefix = comfy.sd.model_detection.unet_prefix_from_state_dict(
            state_dict)
        state_dict = comfy.utils.state_dict_prefix_replace(
            state_dict, {prefix: ""}, filter_keys=False)

    model = comfy.sd.load_diffusion_model_state_dict(
        state_dict, model_options=options, metadata=metadata)
    if model is None:
        raise RuntimeError("could not detect the selected diffusion model type")
    model.cached_patcher_init = (
        _load_sdk_diffusion_model,
        (model_path, options, extra_path),
    )
    return model


@dataclass
class _TextGeneratorEntry:
    tokenizer: Any
    model: Any
    device: Any
    lock: threading.Lock = field(default_factory=threading.Lock)


def _load_fixed_text_generator(generator: str) -> _TextGeneratorEntry:
    if generator != "superprompt-v1":
        raise ValueError(
            f"text generator {generator!r} is not in the trusted catalogue")
    try:
        import folder_paths
        import comfy.model_management
        from transformers import T5ForConditionalGeneration, T5Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "text generator 'superprompt-v1' requires the transformers "
            "and sentencepiece packages") from exc

    root = os.path.abspath(os.path.join(
        folder_paths.models_dir, "text_generation"))
    checkpoint = os.path.abspath(os.path.join(root, "superprompt-v1"))
    if os.path.commonpath((root, checkpoint)) != root:
        raise RuntimeError("fixed text-generator catalogue escaped its root")
    if not os.path.exists(checkpoint):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "text generator 'superprompt-v1' requires huggingface_hub "
                "to install its fixed checkpoint") from exc
        os.makedirs(root, exist_ok=True)
        snapshot_download(
            repo_id="roborovski/superprompt-v1",
            local_dir=checkpoint,
            local_dir_use_symlinks=False,
        )

    device = comfy.model_management.get_torch_device()
    try:
        tokenizer = T5Tokenizer.from_pretrained(
            "google/flan-t5-small", legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(
            checkpoint, device_map=device)
    except ImportError as exc:
        raise RuntimeError(
            "text generator 'superprompt-v1' requires the transformers "
            "and sentencepiece packages") from exc
    model.to(device)
    return _TextGeneratorEntry(tokenizer, model, device)


class _TextGeneratorCache:
    def __init__(self, max_entries: int = 1) -> None:
        if max_entries < 1:
            raise ValueError("text generator cache must hold at least one entry")
        self.max_entries = max_entries
        self._entries: dict[str, _TextGeneratorEntry] = {}
        self._lock = threading.Lock()
        self.loads = 0
        self.hits = 0
        self.evictions = 0

    def _entry(self, generator: str) -> _TextGeneratorEntry:
        with self._lock:
            entry = self._entries.get(generator)
            if entry is not None:
                self.hits += 1
                return entry
            entry = _load_fixed_text_generator(generator)
            self.loads += 1
            while len(self._entries) >= self.max_entries:
                _, evicted = self._entries.popitem()
                self._release(evicted)
                self.evictions += 1
            self._entries[generator] = entry
            return entry

    @staticmethod
    def _release(entry: _TextGeneratorEntry) -> None:
        with entry.lock:
            entry.model.to("cpu")

    def generate(
        self, generator: str, input_text: str, max_new_tokens: int,
    ) -> str:
        import torch

        entry = self._entry(generator)
        with entry.lock, torch.inference_mode():
            input_ids = entry.tokenizer(
                input_text, return_tensors="pt").input_ids.to(entry.device)
            outputs = entry.model.generate(
                input_ids, max_new_tokens=max_new_tokens)
            result = entry.tokenizer.decode(outputs[0])
        if not isinstance(result, str):
            raise TypeError("text generator decoder must return a string")
        return result

    def clear(self) -> int:
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            self._release(entry)
        return len(entries)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "loads": self.loads,
                "hits": self.hits,
                "evictions": self.evictions,
            }


_TEXT_GENERATOR_CACHE = _TextGeneratorCache()


def _fixed_gguf_node_module():
    import folder_paths
    import nodes

    node_class = nodes.NODE_CLASS_MAPPINGS.get("UnetLoaderGGUF")
    module = (
        None if node_class is None
        else sys.modules.get(getattr(node_class, "__module__", ""))
    )
    if module is None:
        raise RuntimeError(
            "GGUF model loading requires the fixed ComfyUI-GGUF extension; "
            "install or update https://github.com/city96/ComfyUI-GGUF")

    module_file = getattr(module, "__file__", None)
    custom_roots = folder_paths.folder_names_and_paths.get(
        "custom_nodes", ([], set()))[0]
    if isinstance(custom_roots, str):
        custom_roots = [custom_roots]
    allowed_roots = [
        os.path.realpath(os.path.join(root, folder))
        for root in custom_roots
        for folder in ("ComfyUI-GGUF", "comfyui-gguf")
    ]
    candidate = None if module_file is None else os.path.realpath(module_file)

    def allowed(root: str) -> bool:
        if candidate is None:
            return False
        try:
            return os.path.commonpath((root, candidate)) == root
        except ValueError:
            return False

    if not any(allowed(root) for root in allowed_roots):
        raise RuntimeError(
            "the registered GGUF loader is not the fixed ComfyUI-GGUF module")
    for name in ("GGMLOps", "gguf_sd_loader", "GGUFModelPatcher"):
        if not hasattr(module, name):
            raise RuntimeError(
                "the installed ComfyUI-GGUF extension is incompatible: "
                f"missing {name}")
    return module


class _InProcessModels:
    _CLIPSEG_MODELS = frozenset({
        "Kijai/clipseg-rd64-refined-fp16",
        "CIDAS/clipseg-rd64-refined",
    })
    _WEIGHT_DTYPES = frozenset({
        "default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2",
        "fp16", "bf16", "fp32",
    })
    _COMPUTE_DTYPES = frozenset({"default", "fp16", "bf16", "fp32"})

    @staticmethod
    def _model_name(name: str, field: str = "name") -> str:
        if not isinstance(name, str):
            raise TypeError(f"model {field} must be a string")
        logical = name.replace("\\", "/")
        if (not logical or "\x00" in logical or logical.startswith("/")
                or (len(logical) > 1 and logical[1] == ":")
                or any(part == ".." for part in logical.split("/"))):
            raise ValueError(f"model {field} must be a confined catalogue name")
        return logical

    @classmethod
    def _load_options(
        cls, weight_dtype: str, compute_dtype: str, cublas_linear: bool,
    ) -> tuple[dict, Any]:
        import torch

        if weight_dtype not in cls._WEIGHT_DTYPES:
            raise ValueError(
                f"unknown model weight dtype {weight_dtype!r}; choose "
                f"{sorted(cls._WEIGHT_DTYPES)}")
        if compute_dtype not in cls._COMPUTE_DTYPES:
            raise ValueError(
                f"unknown model compute dtype {compute_dtype!r}; choose "
                f"{sorted(cls._COMPUTE_DTYPES)}")
        if not isinstance(cublas_linear, bool):
            raise TypeError("cublas_linear must be a bool")

        dtypes = {
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        options = {}
        if weight_dtype == "fp8_e4m3fn_fast":
            options.update(dtype=torch.float8_e4m3fn, fp8_optimizations=True)
        elif weight_dtype != "default":
            options["dtype"] = dtypes[weight_dtype]
        if cublas_linear:
            options["cublas_ops"] = True
        compute = None if compute_dtype == "default" else dtypes[compute_dtype]
        return options, compute

    @staticmethod
    async def _ref(kind: str, ref_type: type[_TypedRef], value: Any):
        if value is None:
            return None
        return ref_type._wrap(await current_runtime().refs.create(kind, value))

    async def list_diffusion_models(
        self, include_connectors: bool = False,
    ) -> list[str]:
        import folder_paths

        if not isinstance(include_connectors, bool):
            raise TypeError("include_connectors must be a bool")
        names = list(folder_paths.get_filename_list("diffusion_models"))
        if include_connectors:
            names.extend(
                name for name in folder_paths.get_filename_list("text_encoders")
                if isinstance(name, str) and "connector" in name.lower())
        result = []
        seen = set()
        for name in names:
            try:
                logical = self._model_name(name)
            except (TypeError, ValueError):
                continue
            if logical not in seen:
                seen.add(logical)
                result.append(logical)
        return result

    async def load_checkpoint(
        self, name: str, weight_dtype: str = "default",
        compute_dtype: str = "default", cublas_linear: bool = False,
    ) -> tuple[ModelRef, Optional[ClipRef], Optional[VaeRef]]:
        import folder_paths
        import comfy.sd

        logical = self._model_name(name)
        options, compute = self._load_options(
            weight_dtype, compute_dtype, cublas_linear)
        path = folder_paths.get_full_path_or_raise("checkpoints", logical)
        model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
            path, output_vae=True, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            model_options=options)
        if compute is not None:
            model.set_model_compute_dtype(compute)
            model.force_cast_weights = False
        return (
            await self._ref("MODEL", ModelRef, model),
            await self._ref("CLIP", ClipRef, clip),
            await self._ref("VAE", VaeRef, vae),
        )

    async def load_diffusion_model(
        self, name: str, extra_name: Optional[str] = None,
        weight_dtype: str = "default", compute_dtype: str = "default",
        cublas_linear: bool = False,
    ) -> ModelRef:
        import folder_paths

        logical = self._model_name(name)
        options, compute = self._load_options(
            weight_dtype, compute_dtype, cublas_linear)
        path = folder_paths.get_full_path_or_raise(
            "diffusion_models", logical)
        extra_path = None
        if extra_name is not None:
            extra_logical = self._model_name(extra_name, "extra_name")
            extra_folder = (
                "text_encoders"
                if "connector" in extra_logical.lower()
                else "diffusion_models")
            extra_path = folder_paths.get_full_path_or_raise(
                extra_folder, extra_logical)

        model = _load_sdk_diffusion_model(path, options, extra_path)
        if compute is not None:
            model.set_model_compute_dtype(compute)
            model.force_cast_weights = False
        return await self._ref("MODEL", ModelRef, model)

    async def load_gguf_model(
        self, name: str, extra_name: Optional[str] = None,
        dequant_dtype: str = "default", patch_dtype: str = "default",
        patch_on_device: bool = False,
    ) -> ModelRef:
        import torch
        import folder_paths
        import comfy.model_detection
        import comfy.sd
        import comfy.utils

        logical = self._model_name(name)
        extra_logical = (
            None if extra_name in (None, "none")
            else self._model_name(extra_name, "extra_name"))
        dtypes = {
            "default": None,
            "target": "target",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dequant_dtype not in dtypes:
            raise ValueError(f"unknown GGUF dequant dtype {dequant_dtype!r}")
        if patch_dtype not in dtypes:
            raise ValueError(f"unknown GGUF patch dtype {patch_dtype!r}")
        if type(patch_on_device) is not bool:
            raise TypeError("patch_on_device must be a bool")

        gguf = _fixed_gguf_node_module()
        try:
            gguf_names = list(folder_paths.get_filename_list("unet_gguf"))
        except KeyError as exc:
            raise RuntimeError(
                "the installed ComfyUI-GGUF extension did not register its "
                "model catalogue") from exc
        if logical not in gguf_names:
            raise ValueError(f"unknown GGUF model catalogue name {logical!r}")

        ops = gguf.GGMLOps()
        ops.Linear = type("SDKGGUFLinear", (ops.Linear,), {})
        ops.Linear.dequant_dtype = dtypes[dequant_dtype]
        ops.Linear.patch_dtype = dtypes[patch_dtype]

        model_path = folder_paths.get_full_path_or_raise(
            "unet_gguf", logical)
        try:
            state_dict, extra = gguf.gguf_sd_loader(model_path)
        except TypeError:
            state_dict = gguf.gguf_sd_loader(model_path)
            extra = {}

        if extra_logical is not None:
            if extra_logical.endswith(".gguf"):
                if extra_logical not in gguf_names:
                    raise ValueError(
                        f"unknown extra GGUF catalogue name {extra_logical!r}")
                extra_path = folder_paths.get_full_path_or_raise(
                    "unet_gguf", extra_logical)
                try:
                    extra_state, _ = gguf.gguf_sd_loader(extra_path)
                except TypeError:
                    extra_state = gguf.gguf_sd_loader(extra_path)
            elif "connector" in extra_logical.lower():
                connectors = [
                    value for value in folder_paths.get_filename_list(
                        "text_encoders")
                    if isinstance(value, str)
                    and "connector" in value.lower()
                ]
                if extra_logical not in connectors:
                    raise ValueError(
                        f"unknown connector catalogue name {extra_logical!r}")
                extra_path = folder_paths.get_full_path_or_raise(
                    "text_encoders", extra_logical)
                extra_state = comfy.utils.load_torch_file(extra_path)
                prefix = comfy.model_detection.unet_prefix_from_state_dict(
                    extra_state)
                if prefix == "model.diffusion_model.":
                    stripped = comfy.utils.state_dict_prefix_replace(
                        extra_state, {prefix: ""}, filter_keys=True)
                    if stripped:
                        extra_state = stripped
            else:
                raise ValueError(
                    "extra GGUF model must be a catalogued .gguf file or "
                    "connector")
            state_dict.update(extra_state)

        model = comfy.sd.load_diffusion_model_state_dict(
            state_dict,
            model_options={"custom_operations": ops},
            metadata=extra.get("metadata", {}),
        )
        if model is None:
            raise RuntimeError(
                f"could not detect GGUF model type for {logical!r}")
        model = gguf.GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device
        return await self._ref("MODEL", ModelRef, model)

    async def list_vae(self) -> list[str]:
        import nodes

        return list(nodes.VAELoader.vae_list(nodes.VAELoader))

    async def load_vae(
        self, name: str, device: str = "default",
        weight_dtype: str = "default",
    ) -> VaeRef:
        import torch
        import comfy.model_management
        import comfy.sd
        import nodes

        devices = {
            "default": None,
            "main_device": comfy.model_management.get_torch_device(),
            "cpu": torch.device("cpu"),
        }
        dtypes = {
            "default": None,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        if device not in devices:
            raise ValueError(
                f"unknown VAE device {device!r}; choose default, main_device, or cpu")
        if weight_dtype not in dtypes:
            raise ValueError(
                f"unknown VAE dtype {weight_dtype!r}; choose default, bf16, fp16, or fp32")

        sd, metadata, vae_path = nodes.VAELoader.load_vae_state_dict(str(name))
        audio_keys = {
            "vocoder.conv_post.weight",
            "vocoder.vocoder.conv_post.weight",
            "vocoder.resblocks.0.convs1.0.weight",
            "vocoder.vocoder.resblocks.0.convs1.0.weight",
        }
        if audio_keys.intersection(sd):
            from comfy.utils import state_dict_prefix_replace

            audio_sd = state_dict_prefix_replace(
                dict(sd), {
                    "audio_vae.": "autoencoder.",
                    "vocoder.": "vocoder.",
                }, filter_keys=True)
            vae = comfy.sd.VAE(sd=audio_sd, metadata=metadata)
        else:
            vae = comfy.sd.VAE(
                sd=sd, metadata=metadata, device=devices[device],
                dtype=dtypes[weight_dtype])
        vae.throw_exception_if_invalid()
        if vae_path is not None and weight_dtype == "default":
            vae.patcher.cached_patcher_init = (
                comfy.sd.load_vae_patcher,
                (vae_path, metadata, devices[device]))
        return VaeRef._wrap(await current_runtime().refs.create("VAE", vae))  # type: ignore[return-value]

    async def memory_cleanup(
        self, empty_cache: bool = True, collect_cycles: bool = True,
        unload_all_models: bool = False,
    ) -> tuple[int, int]:
        import gc
        import comfy.model_management

        before = int(comfy.model_management.get_free_memory())
        if bool(empty_cache):
            comfy.model_management.soft_empty_cache()
        if bool(unload_all_models):
            comfy.model_management.unload_all_models()
            _TEXT_GENERATOR_CACHE.clear()
        if bool(collect_cycles):
            gc.collect()
        after = int(comfy.model_management.get_free_memory())
        return before, after

    async def load_clipseg(self, model: str) -> ClipSegRef:
        import folder_paths
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        model = str(model)
        if model not in self._CLIPSEG_MODELS:
            raise ValueError(
                f"CLIPSeg model {model!r} is not in the trusted model catalogue")
        root = os.path.abspath(os.path.join(folder_paths.models_dir, "clip_seg"))
        checkpoint = os.path.abspath(os.path.join(root, os.path.basename(model)))
        if os.path.commonpath((root, checkpoint)) != root:
            raise ValueError("CLIPSeg model name escapes its catalogue")
        if not os.path.exists(checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model, local_dir=checkpoint,
                local_dir_use_symlinks=False)
        value = {
            "model": CLIPSegForImageSegmentation.from_pretrained(checkpoint),
            "processor": CLIPSegProcessor.from_pretrained(checkpoint),
        }
        return ClipSegRef._wrap(await current_runtime().refs.create(
            "CLIPSEGMODEL", value))  # type: ignore[return-value]

    async def generate_text(
        self, generator: str, input_text: str, max_new_tokens: int = 128,
    ) -> str:
        if not isinstance(generator, str):
            raise TypeError("text generator must be a string")
        if generator != "superprompt-v1":
            raise ValueError(
                f"text generator {generator!r} is not in the trusted catalogue")
        if not isinstance(input_text, str):
            raise TypeError("text generator input must be a string")
        if len(input_text) > 32768:
            raise ValueError("text generator input exceeds 32768 characters")
        if type(max_new_tokens) is not int:
            raise TypeError("max_new_tokens must be an int")
        if not 1 <= max_new_tokens <= 4096:
            raise ValueError("max_new_tokens must be in [1, 4096]")
        return await asyncio.to_thread(
            _TEXT_GENERATOR_CACHE.generate,
            generator, input_text, max_new_tokens)


class _InProcessCapture:
    async def screen(self, region=None, monitor: int = 1) -> ImageRef:
        import asyncio
        import mss
        import numpy as np
        import torch

        def grab():
            with mss.mss() as capture:
                if region is None:
                    index = int(monitor)
                    if not 0 <= index < len(capture.monitors):
                        raise ValueError(
                            f"monitor {monitor} is outside "
                            f"0..{len(capture.monitors) - 1}")
                    target = capture.monitors[index]
                else:
                    left, top, right, bottom = map(int, region)
                    width = right - left
                    height = bottom - top
                    if width <= 0 or height <= 0:
                        raise ValueError(
                            "screen region must have positive width and height")
                    if width * height > 67_108_864:
                        raise ValueError("screen region exceeds 67108864 pixels")
                    target = {
                        "left": left,
                        "top": top,
                        "width": width,
                        "height": height,
                    }
                return np.asarray(
                    capture.grab(target))[..., :3][..., ::-1].copy()

        pixels = await asyncio.to_thread(grab)
        tensor = torch.from_numpy(pixels).to(torch.float32).div_(255.0).unsqueeze(0)
        return ImageRef._wrap(
            await current_runtime().refs.create("IMAGE", tensor))  # type: ignore[return-value]

    async def camera(self, index: int = 0, width=None,
                     height=None) -> ImageRef:
        import asyncio
        import cv2
        import torch

        camera_index = int(index)
        if camera_index < 0:
            raise ValueError("camera index must be non-negative")
        requested_width = None if width is None else int(width)
        requested_height = None if height is None else int(height)
        for name, value in (("width", requested_width),
                            ("height", requested_height)):
            if value is not None and not 1 <= value <= 16384:
                raise ValueError(f"camera {name} must be in [1, 16384]")
        if (requested_width is not None and requested_height is not None and
                requested_width * requested_height > 67_108_864):
            raise ValueError("camera frame exceeds 67108864 pixels")

        def read_frame():
            capture = cv2.VideoCapture(camera_index)
            try:
                if requested_width is not None:
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
                if requested_height is not None:
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(
                        f"camera {camera_index} did not return a frame")
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            finally:
                capture.release()

        frame = await asyncio.to_thread(read_frame)
        tensor = torch.from_numpy(frame.copy()).to(torch.float32).div_(255.0).unsqueeze(0)
        return ImageRef._wrap(
            await current_runtime().refs.create("IMAGE", tensor))  # type: ignore[return-value]

    async def audio(self, duration: float, sample_rate: int = 44100,
                    channels: int = 1) -> AudioRef:
        import asyncio
        import sounddevice
        import torch

        duration = float(duration)
        sample_rate = int(sample_rate)
        channels = int(channels)
        if not 0.0 < duration <= 60.0:
            raise ValueError("audio capture duration must be in (0, 60] seconds")
        if not 8000 <= sample_rate <= 192000:
            raise ValueError("audio sample rate must be in [8000, 192000]")
        if not 1 <= channels <= 8:
            raise ValueError("audio channels must be in [1, 8]")

        def record():
            data = sounddevice.rec(
                round(duration * sample_rate), samplerate=sample_rate,
                channels=channels, dtype="float32", blocking=True)
            return data.copy()

        data = await asyncio.to_thread(record)
        waveform = torch.from_numpy(data.T.copy()).unsqueeze(0)
        return AudioRef._wrap(await current_runtime().refs.create(
            "AUDIO", {"waveform": waveform, "sample_rate": sample_rate}))  # type: ignore[return-value]


class _InProcessUi:
    async def preview_images(self, images: ImageRef,
                             animated: bool = False) -> dict:
        from ._ui import PreviewImage

        value = await current_runtime().refs.resolve(images)
        return PreviewImage(value, animated=animated).as_dict()

    async def preview_mask(self, mask: MaskRef,
                           animated: bool = False) -> dict:
        from ._ui import PreviewMask

        value = await current_runtime().refs.resolve(mask)
        return PreviewMask(value, animated=animated).as_dict()

    async def preview_audio(self, audio: AudioRef) -> dict:
        from ._ui import PreviewAudio

        value = await current_runtime().refs.resolve(audio)
        return PreviewAudio(value).as_dict()

    async def preview_animation(
        self, images: ImageRef, fps: float = 8.0,
    ) -> dict:
        import math
        import random
        from ._io import FolderType
        from ._ui import ImageSaveHelper, SavedImages

        rate = float(fps)
        if not math.isfinite(rate) or not 0.01 <= rate <= 1000.0:
            raise ValueError("animation fps must be finite and in [0.01, 1000]")
        value = await current_runtime().refs.resolve(images)
        if len(value) == 0:
            raise ValueError("animation needs at least one frame")
        prefix = "AnimPreview_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        result = ImageSaveHelper.save_animated_webp(
            value, filename_prefix=prefix, folder_type=FolderType.temp,
            cls=None, fps=rate, lossless=False, quality=50, method=0)
        return SavedImages(
            [result], is_animated=len(value) != 1).as_dict() | {
                "text": [
                    f"{len(value)}x{int(value[0].shape[1])}x"
                    f"{int(value[0].shape[0])}"
                ]
            }

    async def preview_batch(
        self, value: TensorRef, max_thumb_size: int = 512,
        crf: int = 25, max_grid_frames: int = 1024,
    ) -> dict:
        import math
        import random
        from fractions import Fraction

        import av
        import numpy as np
        import torch
        import torch.nn.functional as functional
        from PIL import Image
        import comfy.model_management
        import folder_paths

        images = await current_runtime().refs.resolve(value)
        if not isinstance(images, torch.Tensor) or images.ndim not in (3, 4):
            raise TypeError("batch preview input must be an IMAGE or MASK tensor")
        thumb = int(max_thumb_size)
        quality = int(crf)
        limit = int(max_grid_frames)
        if not 512 <= thumb <= 1024:
            raise ValueError("batch preview max_thumb_size must be in [512, 1024]")
        if not 0 <= quality <= 51:
            raise ValueError("batch preview CRF must be in [0, 51]")
        if not 1 <= limit <= 4096:
            raise ValueError("batch preview max_grid_frames must be in [1, 4096]")
        if images.ndim == 3:
            images = images.reshape(
                (-1, 1, images.shape[-2], images.shape[-1])).movedim(
                    1, -1).expand(-1, -1, -1, 3)
        if images.shape[0] == 0 or images.shape[-1] < 3:
            raise ValueError("batch preview needs at least one RGB frame")

        batch, height, width, _ = images.shape
        if batch > limit:
            indices = torch.linspace(0, batch - 1, limit).round().long().tolist()
        else:
            indices = list(range(batch))
        total = len(indices)
        scale = min(1.0, thumb / max(height, width))
        new_width = max(2, int(round(width * scale)))
        new_height = max(2, int(round(height * scale)))
        new_width -= new_width & 1
        new_height -= new_height & 1
        strip_scale = min(1.0, 256 / max(new_height, new_width))
        strip_width = max(2, int(round(new_width * strip_scale)))
        strip_height = max(2, int(round(new_height * strip_scale)))
        strip_columns = max(1, int(math.ceil(math.sqrt(total))))
        strip_rows = int(math.ceil(total / strip_columns))
        strip = np.zeros(
            (strip_rows * strip_height, strip_columns * strip_width, 3),
            dtype=np.uint8)

        prefix = "kj_batch_preview_" + "".join(
            random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))
        full_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(
                prefix, folder_paths.get_temp_directory(),
                new_width, new_height))
        video_file = f"{filename}_{counter:05}_.mp4"
        strip_file = f"{filename}_{counter:05}_grid.jpg"
        video_path = os.path.join(full_folder, video_file)
        strip_path = os.path.join(full_folder, strip_file)

        container = None
        try:
            container = av.open(video_path, mode="w")
            stream = container.add_stream("libx264", rate=Fraction(30, 1))
            stream.width = new_width
            stream.height = new_height
            stream.pix_fmt = "yuv420p"
            stream.options = {
                "crf": str(quality), "preset": "ultrafast", "g": "1",
                "tune": "fastdecode",
            }
            work_device = comfy.model_management.get_torch_device()
            selected = images[indices, ..., :3].permute(
                0, 3, 1, 2).contiguous().to(work_device)
            if (new_height, new_width) != (height, width):
                mode = "area" if scale < 1.0 else "bilinear"
                selected = functional.interpolate(
                    selected, size=(new_height, new_width), mode=mode)
            video_frames = selected.mul(255).clamp(0, 255).to(
                dtype=torch.uint8, device="cpu").permute(
                    0, 2, 3, 1).contiguous().numpy()
            if (strip_height, strip_width) != (new_height, new_width):
                strip_tensor = functional.interpolate(
                    selected, size=(strip_height, strip_width), mode="area")
                strip_frames = strip_tensor.mul(255).clamp(0, 255).to(
                    dtype=torch.uint8, device="cpu").permute(
                        0, 2, 3, 1).contiguous().numpy()
            else:
                strip_frames = video_frames
            for index, frame_array in enumerate(video_frames):
                row = index // strip_columns
                column = index % strip_columns
                strip[
                    row * strip_height:(row + 1) * strip_height,
                    column * strip_width:(column + 1) * strip_width,
                ] = strip_frames[index]
                frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            container = None
            Image.fromarray(strip).save(strip_path, quality=85)
        except BaseException:
            if container is not None:
                container.close()
            for path in (video_path, strip_path):
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            raise

        return {"kj_batch_preview": [{
            "filename": video_file,
            "subfolder": subfolder,
            "type": "temp",
            "frame_count": total,
            "fps": 30,
            "thumb_w": new_width,
            "thumb_h": new_height,
            "strip_filename": strip_file,
            "strip_cols": strip_columns,
            "strip_cell_w": strip_width,
            "strip_cell_h": strip_height,
        }]}


class _InProcessOutput:
    _TEXT_EXTENSIONS = frozenset({
        ".txt", ".caption", ".json", ".yaml", ".yml", ".md", ".csv",
        ".tsv", ".xml", ".log", ".ini", ".toml",
    })

    def __init__(self, prompt: Any = None, extra_pnginfo: Any = None) -> None:
        from types import SimpleNamespace

        hidden = SimpleNamespace(prompt=prompt, extra_pnginfo=extra_pnginfo)
        self._metadata_owner = SimpleNamespace(hidden=hidden)

    @staticmethod
    def _prefix(filename_prefix: str, subfolder: str) -> str:
        filename_prefix = str(filename_prefix)
        subfolder = str(subfolder or "")
        if subfolder in (".", "output"):
            subfolder = ""
        if os.path.isabs(subfolder) or os.path.isabs(filename_prefix):
            raise ValueError("output names must be relative")
        prefix = os.path.normpath(
            os.path.join(subfolder, filename_prefix)
            if subfolder else filename_prefix)
        if prefix in ("", ".", os.pardir) or prefix.startswith(os.pardir + os.sep):
            raise ValueError("output name must stay inside the output directory")
        return prefix

    @classmethod
    def _extension(cls, extension: str) -> str:
        extension = os.path.basename(str(extension))
        if extension and not extension.startswith("."):
            extension = "." + extension
        extension = extension.lower()
        if extension not in cls._TEXT_EXTENSIONS:
            allowed = ", ".join(sorted(cls._TEXT_EXTENSIONS))
            raise ValueError(
                f"output text extension {extension!r} is not allowed; "
                f"choose one of {allowed}")
        return extension

    async def save_images(
        self, images: ImageRef, filename_prefix: str = "ComfyUI",
        subfolder: str = "", compress_level: int = 4,
        caption: Optional[str] = None,
        caption_extension: str = ".txt",
    ) -> dict:
        import folder_paths
        from ._io import FolderType
        from ._ui import ImageSaveHelper, SavedImages

        value = await current_runtime().refs.resolve(images)
        prefix = self._prefix(filename_prefix, subfolder)
        level = int(compress_level)
        if not 0 <= level <= 9:
            raise ValueError("PNG compression level must be in [0, 9]")
        results = ImageSaveHelper.save_images(
            value, prefix, FolderType.output, self._metadata_owner, level)
        if caption is not None:
            extension = self._extension(caption_extension)
            output_dir = os.path.abspath(folder_paths.get_output_directory())
            for result in results:
                stem = os.path.splitext(result.filename)[0]
                target = os.path.abspath(os.path.join(
                    output_dir, result.subfolder, stem + extension))
                if os.path.commonpath((output_dir, target)) != output_dir:
                    raise ValueError("caption target escapes the output directory")
                with open(target, "w", encoding="utf-8") as file:
                    file.write(str(caption))
        return SavedImages(results).as_dict()

    async def save_images_with_alpha(
        self, images: ImageRef, mask: MaskRef,
        filename_prefix: str = "ComfyUI", subfolder: str = "",
        compress_level: int = 4,
    ) -> dict:
        import numpy as np
        from PIL import Image as PILImage
        import folder_paths
        from ._io import FolderType
        from ._ui import ImageSaveHelper, SavedImages, SavedResult

        rt = current_runtime()
        pixels = await rt.refs.resolve(images)
        masks = await rt.refs.resolve(mask)
        if len(pixels) != len(masks):
            raise ValueError("image and alpha-mask batches must have equal length")
        level = int(compress_level)
        if not 0 <= level <= 9:
            raise ValueError("PNG compression level must be in [0, 9]")
        prefix = self._prefix(filename_prefix, subfolder)
        full_folder, filename, counter, saved_subfolder, _ = (
            folder_paths.get_save_image_path(
                prefix, folder_paths.get_output_directory(),
                pixels[0].shape[1], pixels[0].shape[0]))
        metadata = ImageSaveHelper._create_png_metadata(self._metadata_owner)
        results = []
        for batch_number, (image, alpha) in enumerate(zip(pixels, masks)):
            rgb = PILImage.fromarray(np.clip(
                255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8))
            alpha_image = PILImage.fromarray(
                np.clip(255.0 * (1.0 - alpha.cpu().float().numpy()), 0, 255)
                .astype(np.uint8), mode="L")
            if alpha_image.size != rgb.size:
                alpha_image = alpha_image.resize(rgb.size, PILImage.Resampling.LANCZOS)
            rgb.putalpha(alpha_image)
            batch_name = filename.replace("%batch_num%", str(batch_number))
            file = f"{batch_name}_{counter:05}_.png"
            rgb.save(os.path.join(full_folder, file), pnginfo=metadata,
                     compress_level=level)
            results.append(SavedResult(file, saved_subfolder, FolderType.output))
            counter += 1
        return SavedImages(results).as_dict()

    async def save_text(
        self, text: str, filename_prefix: str = "text",
        subfolder: str = "", extension: str = ".txt",
    ) -> str:
        import folder_paths

        extension = self._extension(extension)
        output_dir = os.path.abspath(folder_paths.get_output_directory())
        full_folder, filename, counter, saved_subfolder, _ = (
            folder_paths.get_save_image_path(
                self._prefix(filename_prefix, subfolder), output_dir))
        while True:
            file = f"{filename}_{counter:05}_{extension}"
            target = os.path.abspath(os.path.join(full_folder, file))
            if os.path.commonpath((output_dir, target)) != output_dir:
                raise ValueError("text target escapes the output directory")
            if not os.path.exists(target):
                break
            counter += 1
        with open(target, "w", encoding="utf-8") as stream:
            stream.write(str(text))
        return os.path.join(saved_subfolder, file) if saved_subfolder else file

    async def save_state_dict(
        self, state_dict: ValueRef, filename_prefix: str,
        metadata: Optional[dict[str, str]] = None,
    ) -> str:
        import torch
        import comfy.utils
        import folder_paths

        value = await current_runtime().refs.resolve(state_dict)
        if not isinstance(value, dict) or not value:
            raise TypeError("state_dict must be a non-empty mapping")
        if any(not isinstance(key, str) for key in value):
            raise TypeError("state_dict keys must be strings")
        if any(not isinstance(tensor, torch.Tensor) for tensor in value.values()):
            raise TypeError("state_dict values must be tensors")
        safe_metadata = None
        if metadata is not None:
            if not isinstance(metadata, dict) or any(
                not isinstance(key, str) or not isinstance(item, str)
                for key, item in metadata.items()
            ):
                raise TypeError("state_dict metadata must map strings to strings")
            safe_metadata = dict(metadata)

        output_dir = os.path.abspath(folder_paths.get_output_directory())
        full_folder, filename, counter, saved_subfolder, _ = (
            folder_paths.get_save_image_path(
                self._prefix(filename_prefix, ""), output_dir))
        tensors = {
            key: tensor.detach().contiguous().cpu()
            for key, tensor in value.items()
        }
        while True:
            file = f"{filename}_{counter:05}_.safetensors"
            target = os.path.abspath(os.path.join(full_folder, file))
            if os.path.commonpath((output_dir, target)) != output_dir:
                raise ValueError("state_dict target escapes the output directory")
            if not os.path.exists(target):
                break
            counter += 1
        comfy.utils.save_torch_file(tensors, target, metadata=safe_metadata)
        return os.path.join(saved_subfolder, file) if saved_subfolder else file

    async def save_model(
        self, model: ModelRef, filename_prefix: str,
        model_key_prefix: str = "model.diffusion_model.",
    ) -> str:
        import comfy.model_management

        if type(filename_prefix) is not str:
            raise TypeError("model filename_prefix must be a string")
        if type(model_key_prefix) is not str:
            raise TypeError("model_key_prefix must be a string")
        if len(filename_prefix) > 4096:
            raise ValueError("model filename_prefix is limited to 4096 characters")
        if len(model_key_prefix) > 4096:
            raise ValueError("model_key_prefix is limited to 4096 characters")
        if any(ord(character) < 32 for character in filename_prefix):
            raise ValueError("model filename_prefix contains control characters")
        if ("/" in model_key_prefix or "\\" in model_key_prefix
                or any(ord(character) < 32 for character in model_key_prefix)):
            raise ValueError("model_key_prefix must be a tensor-key prefix")
        prefix = self._prefix(filename_prefix, "")

        rt = current_runtime()
        if not isinstance(model, Ref) or model.kind != "MODEL":
            raise TypeError("save_model requires a MODEL ref")
        value = await rt.refs.resolve(model)
        comfy.model_management.load_models_gpu([value])
        source = value.state_dict_for_saving(None, None, None)
        default_prefix = "model.diffusion_model."
        state_dict = {}
        for key, tensor in source.items():
            output_key = (
                model_key_prefix + key[len(default_prefix):]
                if key.startswith(default_prefix) else key)
            state_dict[output_key] = (
                tensor if tensor.is_contiguous() else tensor.contiguous())

        state_ref = ValueRef._wrap(await rt.refs.create("VALUE", state_dict))
        try:
            return await self.save_state_dict(
                state_ref, prefix, metadata=None)  # type: ignore[arg-type]
        finally:
            await rt.refs.release(state_ref)

    async def save_video(
        self, images: ImageRef, audio: Optional[AudioRef] = None,
        fps: float = 25.0, filename_prefix: str = "video/ComfyUI",
        format: str = "auto", codec: str = "auto",
    ) -> dict:
        import math
        from fractions import Fraction

        import torch
        import folder_paths
        from comfy.cli_args import args
        from ._input_impl.video_types import VideoFromComponents
        from ._io import FolderType
        from ._ui import PreviewVideo, SavedResult
        from ._util.video_types import (
            VideoCodec, VideoComponents, VideoContainer,
        )

        rate = float(fps)
        if not math.isfinite(rate) or not 0.0 < rate <= 999.0:
            raise ValueError("video fps must be finite and in (0, 999]")
        try:
            container_type = VideoContainer(str(format))
        except ValueError as exc:
            raise ValueError(f"unsupported video format {format!r}") from exc
        try:
            codec_type = VideoCodec(str(codec))
        except ValueError as exc:
            raise ValueError(f"unsupported video codec {codec!r}") from exc

        rt = current_runtime()
        pixels = await rt.refs.resolve(images)
        if (not isinstance(pixels, torch.Tensor) or pixels.ndim != 4
                or len(pixels) == 0 or pixels.shape[-1] < 3):
            raise TypeError("save_video needs non-empty BHWC image frames")
        audio_value = None
        if audio is not None:
            audio_value = await rt.refs.resolve(audio)
            if (not isinstance(audio_value, dict)
                    or "waveform" not in audio_value
                    or "sample_rate" not in audio_value):
                raise TypeError("save_video audio must contain waveform and sample_rate")

        prefix = self._prefix(filename_prefix, "")
        output_dir = os.path.abspath(folder_paths.get_output_directory())
        full_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(
                prefix, output_dir, pixels.shape[2], pixels.shape[1]))
        file = (
            f"{filename}_{counter:05}_."
            f"{VideoContainer.get_extension(container_type)}")
        target = os.path.abspath(os.path.join(full_folder, file))
        if os.path.commonpath((output_dir, target)) != output_dir:
            raise ValueError("video target escapes the output directory")

        metadata = None
        if not args.disable_metadata:
            values = {}
            if self._metadata_owner.hidden.extra_pnginfo is not None:
                values.update(self._metadata_owner.hidden.extra_pnginfo)
            if self._metadata_owner.hidden.prompt is not None:
                values["prompt"] = self._metadata_owner.hidden.prompt
            if values:
                metadata = values
        video = VideoFromComponents(VideoComponents(
            images=pixels,
            audio=audio_value,
            frame_rate=Fraction(rate),
        ))
        try:
            video.save_to(
                target, format=container_type, codec=codec_type,
                metadata=metadata)
        except BaseException:
            try:
                os.unlink(target)
            except FileNotFoundError:
                pass
            raise
        return PreviewVideo([
            SavedResult(file, subfolder, FolderType.output),
        ]).as_dict()


class _InProcessGraph:
    def __init__(self, current_node_id: str, prompt: Any = None,
                 extra_pnginfo: Any = None) -> None:
        self._current_node_id = str(current_node_id)
        self._prompt = prompt if isinstance(prompt, dict) else {}
        self._workflow = (
            extra_pnginfo.get("workflow", {})
            if isinstance(extra_pnginfo, dict) else {})

    def _prompt_key(self, node_id: int | str) -> Optional[str]:
        wanted = str(node_id)
        if wanted in self._prompt:
            return wanted
        prefix = self._current_node_id.rsplit(":", 1)[0]
        if ":" in self._current_node_id:
            scoped = f"{prefix}:{wanted}"
            if scoped in self._prompt:
                return scoped
        matches = [key for key in self._prompt if key.rsplit(":", 1)[-1] == wanted]
        return matches[0] if len(matches) == 1 else None

    def _id_for_title(self, title: str) -> Optional[int | str]:
        candidates = list(self._workflow.get("nodes", []))
        definitions = self._workflow.get("definitions", {})
        for subgraph in definitions.get("subgraphs", []):
            candidates.extend(subgraph.get("nodes", []))
        matches = [node.get("id") for node in candidates
                   if node.get("title") == title]
        return matches[0] if len(matches) == 1 else None

    async def widget_values(
        self, node_id: int | str = 0, node_title: str = "",
        linked_input: str = "any_input",
    ) -> dict[str, Any]:
        target = None
        if node_title:
            target = self._id_for_title(str(node_title))
            if target is None:
                raise KeyError(f"no unique workflow node titled {node_title!r}")
        elif str(node_id) not in ("", "0"):
            target = node_id
        else:
            current = self._prompt.get(self._current_node_id, {})
            link = current.get("inputs", {}).get(linked_input)
            if not (isinstance(link, (list, tuple)) and len(link) >= 1):
                raise KeyError(
                    f"input {linked_input!r} on node {self._current_node_id} "
                    "is not linked")
            target = link[0]
        key = self._prompt_key(target)
        if key is None:
            raise KeyError(f"node {target!r} is not present in this prompt")
        values = self._prompt.get(key, {}).get("inputs")
        if not isinstance(values, dict):
            raise KeyError(f"node {target!r} has no prompt inputs")
        return dict(values)


class _InProcessProgress:
    def __init__(self, node_id: Optional[str]) -> None:
        self._node_id = node_id

    async def update(self, value: float, total: float,
                     preview: Optional[ImageRef] = None) -> None:
        from comfy.utils import ProgressBar  # lazy

        pb = ProgressBar(total, node_id=self._node_id)
        pb.update_absolute(value, total)


class _InProcessScratch:
    async def dir(self) -> str:
        import folder_paths  # lazy

        return folder_paths.get_temp_directory()


class _InProcessEvents:
    async def emit(self, event: str, data: dict) -> None:
        from server import PromptServer  # lazy

        inst = getattr(PromptServer, "instance", None)
        if inst is not None:
            inst.send_sync(event, data)


class _StubDomain:
    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, item: str) -> Any:
        raise NotImplementedError(
            f"ctx.{self._name}.{item} is defined in the SDK contract but not yet "
            f"implemented by the in-process default. Provided by the full SDK / "
            f"overlay."
        )


@dataclass
class InProcessContext:
    assets: Any
    progress: Any
    scratch: Any
    events: Any
    storage: Any
    capture: Any
    ui: Any
    output: Any
    graph: Any
    models: Any
    profiling: Any
    preview_override: Any
    sample: Any
    serve: Any
    secrets: Any
    net: Any


class InProcessCtxProvider:
    def build(self, plan: ExecutionPlan) -> Context:
        return InProcessContext(  # type: ignore[return-value]
            assets=_InProcessAssets(),
            progress=_InProcessProgress(plan.node_id),
            scratch=_InProcessScratch(),
            events=_InProcessEvents(),
            storage=_StubDomain("storage"),
            capture=_InProcessCapture(),
            ui=_InProcessUi(),
            output=_InProcessOutput(plan.prompt, plan.extra_pnginfo),
            graph=_InProcessGraph(
                plan.node_id, plan.prompt, plan.extra_pnginfo),
            models=_InProcessModels(),
            profiling=InProcessProfiling(
                f"in-process:{plan.node_module}", plan.node_id),
            preview_override=InProcessPreviewOverride(plan.node_id),
            sample=_StubDomain("sample"),
            serve=_StubDomain("serve"),
            secrets=_StubDomain("secrets"),
            net=_StubDomain("net"),
        )


class OpNotSupported(NotImplementedError):
    """Raised by ``apply`` for an op this provider does not implement. Carries
    the capability name so a node can decide to fall back (e.g. to ``raw``)."""

    def __init__(self, op: str) -> None:
        self.op = op
        self.capability = f"ops.{op}"
        super().__init__(
            f"op {op!r} is not supported by this ops provider "
            f"(capability: {self.capability})"
        )


@dataclass
class _WeightDiffCursorState:
    patcher: Any
    keys: list[tuple[str, str, str]]
    index: int = 0
    current_ref: Optional[TensorRef] = None
    closed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _sample_self_refine_video(
    model, x, sigmas, stochastic_step_map, certain_percentage=0.999,
    uncertainty_threshold=0.25, extra_args=None, callback=None, disable=None,
    verbose=False, video_shape=None, seed=None,
):
    import math

    import torch
    from comfy.k_diffusion.sampling import to_d
    from tqdm import tqdm

    extra_args = {} if extra_args is None else extra_args
    sigma_in = x.new_ones([x.shape[0]])

    if seed is not None:
        generator = torch.Generator(torch.device("cpu")).manual_seed(seed)

    pbar = tqdm(total=len(sigmas) - 1, disable=disable, desc="Sampling")

    for i in range(len(sigmas) - 1):
        current_num_anneal_steps = stochastic_step_map.get(i, 0)
        use_stochastic = current_num_anneal_steps > 0
        m = current_num_anneal_steps + 1 if use_stochastic else 1

        sigma, sigma_next = sigmas[i], sigmas[i + 1]

        prev_certain_mask = None
        prev_denoised = None
        prev_denoised_full = None
        prev_x_next = None
        prev_x_next_video = None
        is_certain = False

        for ii in range(m):
            if m > 1:
                pbar.set_description(
                    f"Step {i}/{len(sigmas)-1} (substep {ii+1}/{m})")
            if is_certain:
                x = prev_x_next
                break

            noise = torch.randn(
                x.shape, device=torch.device("cpu"), generator=generator).to(x)
            x_in = (
                x if ii == 0
                else (1.0 - sigma) * prev_denoised_full + sigma * noise)
            if ii > 0:
                x = x_in

            denoised = model(x_in, sigmas[i] * sigma_in, **extra_args)

            if callback is not None:
                callback({
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                })

            d = to_d(x, sigma, denoised)
            x_next = x + (sigma_next - sigma) * d

            if d.ndim == 3 and video_shape is not None:
                cut = math.prod(video_shape[1:])
                denoised_video = denoised[:, :, :cut].reshape(
                    [denoised.shape[0]] + list(video_shape)[1:])
                x_next_video = x_next[:, :, :cut].reshape(
                    [denoised.shape[0]] + list(video_shape)[1:])
                denoised_audio = denoised[:, :, cut:]
                x_next_audio = x_next[:, :, cut:]
                if verbose:
                    tqdm.write(
                        f"Video shape: {denoised_video.shape}, "
                        f"Audio shape: {denoised_audio.shape}")
            else:
                denoised_video = denoised
                x_next_video = x_next
                denoised_audio = None
                x_next_audio = None

            if use_stochastic and prev_denoised is not None:
                diff = denoised_video - prev_denoised
                uncertainty = (
                    torch.sqrt(torch.sum(diff ** 2, dim=1))
                    / denoised_video.shape[1])
                certain_mask = uncertainty < uncertainty_threshold

                if verbose:
                    tqdm.write(
                        f"Step {i}/{len(sigmas)-1} substep {ii+1}/{m}:")
                    tqdm.write(
                        f"Uncertainty: min {uncertainty.min():.4f}, "
                        f"max {uncertainty.max():.4f}, "
                        f"threshold {uncertainty_threshold}")
                    tqdm.write(
                        f"Certain pixels: {certain_mask.sum()}/"
                        f"{certain_mask.numel()} = "
                        f"{certain_mask.sum()/certain_mask.numel():.4f}")

                if prev_certain_mask is not None:
                    certain_mask = certain_mask | prev_certain_mask

                if certain_mask.sum() / certain_mask.numel() > (
                        certain_percentage):
                    is_certain = True
                    if verbose:
                        tqdm.write(
                            f"{ii}/{current_num_anneal_steps}: Certain region "
                            f"is more than {certain_percentage}, we are certain")

                certain_mask_float = certain_mask.float().unsqueeze(1)
                x_next_video = (
                    certain_mask_float * prev_x_next_video
                    + (1.0 - certain_mask_float) * x_next_video)
                denoised_video = (
                    certain_mask_float * prev_denoised
                    + (1.0 - certain_mask_float) * denoised_video)

                if x_next_audio is not None:
                    x_next = x_next.clone()
                    x_next[:, :, :cut] = x_next_video.reshape(
                        [x_next_video.shape[0], x_next.shape[1], -1])
                    denoised_full = denoised.clone()
                    denoised_full[:, :, :cut] = denoised_video.reshape(
                        [denoised_video.shape[0], denoised.shape[1], -1])
                else:
                    x_next = x_next_video
                    denoised_full = denoised_video

                prev_certain_mask = certain_mask
                prev_denoised = denoised_video
                prev_denoised_full = denoised_full
                prev_x_next_video = x_next_video
                prev_x_next = x_next
            elif use_stochastic:
                if x_next_audio is not None:
                    denoised_full = denoised.clone()
                    denoised_full[:, :, :cut] = denoised_video.reshape(
                        [denoised_video.shape[0], denoised.shape[1], -1])
                else:
                    denoised_full = denoised_video

                prev_certain_mask = None
                prev_denoised = denoised_video
                prev_denoised_full = denoised_full
                prev_x_next_video = x_next_video
                prev_x_next = x_next

            if use_stochastic and ii == m - 1:
                x = prev_x_next
            elif not use_stochastic:
                x = x_next

        pbar.update(1)
        if m == 1:
            pbar.set_description("Sampling")
    pbar.close()
    return x


class InProcessOps:
    """Default operations. Runs in the trusted process; uses the real buffers
    via the resolver but never hands them to the node. Kept torch-free (tensor
    arithmetic works on the resolved objects directly). The op set is a
    registry: core ships two primitives; ``register_op`` extends it (an overlay
    subclasses or registers richer ops)."""

    def __init__(self) -> None:
        self._ops: dict[str, Callable[..., Awaitable["ImageRef"]]] = {
            "invert": self._invert,
            "scale": self._scale,
            # Operations on live engine objects. These are what let a node
            # DECLARE a MODEL/CLIP/VAE input and still be sandboxable: the node
            # names the operation, the weights stay here.
            "vae.decode": self._vae_decode,
            "vae.encode": self._vae_encode,
            "vae.encode_tiled": self._vae_encode_tiled,
            "vae.input_dtype": self._vae_input_dtype,
            "vae.encode_video": self._vae_encode_video,
            "vae.decode_video": self._vae_decode_video,
            "vae.decode_audio": self._vae_decode_audio,
            "vae.downscale_index_formula": self._vae_downscale_index_formula,
            "vae.merge": self._vae_merge,
            "vae.compile": self._vae_compile,
            "vae.patch_triton": self._vae_patch_triton,
            "video.encoded_source": self._video_encoded_source,
            "clip.tokenize": self._clip_tokenize,
            "clip.encode_from_tokens_scheduled":
                self._clip_encode_from_tokens_scheduled,
            "clip.encode": self._clip_encode,
            "gligen.apply_batched": self._gligen_apply_batched,
            "latent.minimax_h3_token_count":
                self._latent_minimax_h3_token_count,
            "cond.sequence_length": self._cond_sequence_length,
            "cond.combine": self._cond_combine,
            "cond.concat": self._cond_concat,
            "advanced_control.weights_from_list":
                self._advanced_control_weights_from_list,
            "lora.weight_differences": self._lora_weight_differences,
            "weight_diff.next": self._weight_diff_next,
            "model.apply_dit_block_lora": self._model_apply_dit_block_lora,
            "model.apply_ltx2_lora": self._model_apply_ltx2_lora,
            "model.patch": self._model_patch,
            "model.transforms": self._model_transforms,
            "model.latent_scale_factor": self._model_latent_scale_factor,
            "guider.scheduled_cfg": self._guider_scheduled_cfg,
            "sampler.self_refine_video": self._sampler_self_refine_video,
            "clip_vision.encode_image": self._clip_vision_encode_image,
            "clip_vision_output.image_embeds":
                self._clip_vision_output_image_embeds,
            "controlnet.with_union_type": self._controlnet_with_union_type,
            "controlnet.compile": self._controlnet_compile,
            "style_model.apply": self._style_model_apply,
            "clipseg.segment": self._clipseg_segment,
            "upscale_model.upscale": self._upscale_model_upscale,
        }

    def register_op(self, name: str, fn: Callable[..., Awaitable["ImageRef"]]) -> None:
        self._ops[name] = fn

    def supports(self, op: str) -> bool:
        return op in self._ops

    async def apply(self, op: str, subject: "Ref", params: dict) -> Any:
        fn = self._ops.get(op)
        if fn is None:
            raise OpNotSupported(op)
        return await fn(subject, **params)

    async def _invert(self, image: "ImageRef") -> "ImageRef":
        t = await current_runtime().refs.resolve(image)
        return ImageRef._wrap(await current_runtime().refs.create("IMAGE", 1.0 - t))  # type: ignore[return-value]

    async def _scale(self, image: "ImageRef", factor: float) -> "ImageRef":
        t = await current_runtime().refs.resolve(image)
        return ImageRef._wrap(await current_runtime().refs.create("IMAGE", t * factor))  # type: ignore[return-value]

    # --- operations on live engine objects ------------------------------- #
    # Each resolves its handles to the real objects HERE, on the trusted plane,
    # runs core's own semantics, and returns a handle. A guest never holds the
    # model; it holds the name of what it wanted done.

    async def _vae_decode(self, vae: "VaeRef", latent: "LatentRef") -> "ImageRef":
        rt = current_runtime()
        v = await rt.refs.resolve(vae)
        samples = await rt.refs.resolve(latent)
        return ImageRef._wrap(await rt.refs.create("IMAGE", v.decode(samples["samples"])))  # type: ignore[return-value]

    async def _vae_encode(self, vae: "VaeRef", image: "ImageRef") -> "LatentRef":
        rt = current_runtime()
        v = await rt.refs.resolve(vae)
        pixels = await rt.refs.resolve(image)
        return LatentRef._wrap(await rt.refs.create(  # type: ignore[return-value]
            "LATENT", {"samples": v.encode(pixels)}))

    async def _vae_encode_tiled(
        self, vae: "VaeRef", image: "ImageRef", tile_x=None, tile_y=None,
        overlap=None, tile_t=None, overlap_t=None,
    ) -> "LatentRef":
        rt = current_runtime()
        value = await rt.refs.resolve(vae)
        pixels = await rt.refs.resolve(image)
        kwargs = {
            key: item for key, item in {
                "tile_x": tile_x,
                "tile_y": tile_y,
                "overlap": overlap,
                "tile_t": tile_t,
                "overlap_t": overlap_t,
            }.items() if item is not None
        }
        samples = value.encode_tiled(pixels, **kwargs)
        return LatentRef._wrap(await rt.refs.create(
            "LATENT", {"samples": samples}))  # type: ignore[return-value]

    async def _vae_input_dtype(self, vae: "VaeRef") -> str:
        value = await current_runtime().refs.resolve(vae)
        name = str(value.vae_dtype).removeprefix("torch.")
        if name not in {"float16", "bfloat16", "float32", "float64"}:
            raise TypeError(f"unsupported VAE input dtype {name!r}")
        return name

    async def _vae_encode_video(
        self, vae: "VaeRef", image: "ImageRef",
    ) -> tuple["LatentRef", int]:
        import torch

        rt = current_runtime()
        value = await rt.refs.resolve(vae)
        pixels = await rt.refs.resolve(image)
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4:
            raise TypeError("VAE video encode needs BHWC image frames")
        pixels = pixels.to(dtype=value.vae_dtype, device="cpu")
        try:
            temporal_compress = value.downscale_ratio[0]
            temporal_decompress = value.upscale_ratio[0]
            valid_frames = temporal_decompress(
                temporal_compress(pixels.shape[0]))
            if valid_frames < pixels.shape[0]:
                pixels = pixels[:valid_frames]
        except (TypeError, IndexError):
            pass
        samples = value.encode(pixels)
        latent = LatentRef._wrap(await rt.refs.create(
            "LATENT", {"samples": samples}))
        return latent, int(pixels.shape[0])

    async def _vae_decode_video(
        self, vae: "VaeRef", latent: "LatentRef", tiled: bool = False,
        tile_size: int = 512, overlap: int = 64,
        temporal_size: int = 4096, temporal_overlap: int = 16,
    ) -> "ImageRef":
        import torch

        rt = current_runtime()
        value = await rt.refs.resolve(vae)
        latent_value = await rt.refs.resolve(latent)
        if not isinstance(latent_value, dict) or "samples" not in latent_value:
            raise TypeError("VAE video decode needs a LATENT with samples")
        samples = latent_value["samples"]
        if getattr(samples, "is_nested", False):
            samples = samples.unbind()[0]

        if tiled:
            tile_size = int(tile_size)
            overlap = int(overlap)
            temporal_size = int(temporal_size)
            temporal_overlap = int(temporal_overlap)
            if not 64 <= tile_size <= 4096:
                raise ValueError("video VAE tile_size must be in [64, 4096]")
            if not 0 <= overlap <= 4096:
                raise ValueError("video VAE overlap must be in [0, 4096]")
            if not 8 <= temporal_size <= 4096:
                raise ValueError(
                    "video VAE temporal_size must be in [8, 4096]")
            if not 4 <= temporal_overlap <= 4096:
                raise ValueError(
                    "video VAE temporal_overlap must be in [4, 4096]")
            if tile_size < overlap * 4:
                overlap = tile_size // 4
            if temporal_size < temporal_overlap * 2:
                temporal_overlap //= 2
            temporal_compression = value.temporal_compression_decode()
            if temporal_compression is None:
                tile_t = None
                overlap_t = None
            else:
                tile_t = max(2, temporal_size // temporal_compression)
                overlap_t = max(
                    1, min(tile_t // 2,
                           temporal_overlap // temporal_compression))
            spatial_compression = value.spacial_compression_decode()
            tile_x = max(1, tile_size // spatial_compression)
            spatial_overlap = max(1, overlap // spatial_compression)
            images = value.decode_tiled(
                samples, tile_t=tile_t, tile_x=tile_x, tile_y=tile_x,
                overlap=spatial_overlap, overlap_t=overlap_t)
        else:
            images = value.decode(samples)

        if not isinstance(images, torch.Tensor):
            raise TypeError("VAE video decode did not return a tensor")
        if images.ndim == 5:
            images = images.reshape(
                -1, images.shape[-3], images.shape[-2], images.shape[-1])
        return ImageRef._wrap(await rt.refs.create("IMAGE", images))  # type: ignore[return-value]

    async def _vae_decode_audio(
        self, vae: "VaeRef", latent: "LatentRef",
    ) -> "AudioRef":
        rt = current_runtime()
        value = await rt.refs.resolve(vae)
        latent_value = await rt.refs.resolve(latent)
        if not isinstance(latent_value, dict) or "samples" not in latent_value:
            raise TypeError("VAE audio decode needs a LATENT with samples")
        samples = latent_value["samples"]
        if getattr(samples, "is_nested", False):
            samples = samples.unbind()[-1]
        audio = value.decode(samples)
        if hasattr(value, "first_stage_model"):
            audio = audio.movedim(-1, 1)
        audio = audio.to(samples.device)
        sample_rate = getattr(
            value, "audio_sample_rate_output",
            getattr(value, "output_sample_rate", None))
        if sample_rate is None:
            sample_rate = getattr(
                getattr(value, "first_stage_model", None),
                "output_sample_rate", 44100)
        return AudioRef._wrap(await rt.refs.create("AUDIO", {
            "waveform": audio,
            "sample_rate": int(sample_rate),
        }))  # type: ignore[return-value]

    async def _video_encoded_source(
        self, video: "VideoRef",
    ) -> "ValueRef":
        import io
        import torch

        rt = current_runtime()
        value = await rt.refs.resolve(video)
        source = value.get_stream_source()
        max_bytes = int(os.environ.get(
            "COMFY_SECURE_VIDEO_SOURCE_MAX", str(1024 * 1024 * 1024)))
        if isinstance(source, (str, os.PathLike)):
            size = os.path.getsize(source)
            if size > max_bytes:
                raise ValueError(
                    f"encoded video exceeds the {max_bytes}-byte limit")
            with open(source, "rb") as stream:
                content = stream.read(max_bytes + 1)
        elif isinstance(source, io.BytesIO):
            content = source.getvalue()
        elif hasattr(source, "read"):
            if hasattr(source, "seek"):
                source.seek(0)
            content = source.read(max_bytes + 1)
        else:
            raise TypeError("VIDEO stream source is not readable")
        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise TypeError("VIDEO stream source did not return bytes")
        if len(content) > max_bytes:
            raise ValueError(
                f"encoded video exceeds the {max_bytes}-byte limit")
        if hasattr(value, "get_active_trim_window"):
            start_time, duration = value.get_active_trim_window()
        else:
            start_time, duration = 0.0, 0.0
        data = torch.frombuffer(bytearray(content), dtype=torch.uint8)
        result = {
            "data": data,
            "start_time": float(start_time),
            "duration": float(duration),
        }
        return ValueRef._wrap(await rt.refs.create("VALUE", result))  # type: ignore[return-value]

    async def _vae_downscale_index_formula(
        self, vae: "VaeRef",
    ) -> Optional[tuple[int, int, int]]:
        value = await current_runtime().refs.resolve(vae)
        formula = value.downscale_index_formula
        if formula is None:
            return None
        if (not isinstance(formula, (list, tuple)) or len(formula) != 3
                or any(isinstance(item, bool) or not isinstance(item, int)
                       for item in formula)):
            raise TypeError(
                "VAE downscale_index_formula must be three integers or None")
        return tuple(formula)

    async def _vae_merge(
        self, vae: "VaeRef", other: "VaeRef", ratio: float = 0.5,
    ) -> "VaeRef":
        import math
        import torch
        from comfy.sd import VAE

        value = float(ratio)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("VAE merge ratio must be finite and in [0, 1]")
        rt = current_runtime()
        first = await rt.refs.resolve(vae)
        second = await rt.refs.resolve(other)
        first_sd = first.get_sd()
        second_sd = second.get_sd()
        mismatch = set(first_sd) ^ set(second_sd)
        if mismatch:
            raise ValueError(
                "Cannot merge: VAE architectures differ "
                f"({len(mismatch)} non-matching keys, e.g. {list(mismatch)[:3]}).")
        merged = {}
        for key, first_value in first_sd.items():
            second_value = second_sd[key]
            if first_value.shape != second_value.shape:
                raise ValueError(
                    f"Cannot merge: shape mismatch for {key!r} "
                    f"({tuple(first_value.shape)} vs {tuple(second_value.shape)})")
            if torch.is_floating_point(first_value):
                blended = torch.lerp(
                    first_value.float(),
                    second_value.to(device=first_value.device).float(), value)
                merged[key] = blended.to(dtype=first_value.dtype)
            else:
                merged[key] = first_value.clone()
        result = VAE(
            sd=merged, device=first.device, dtype=first.vae_dtype)
        result.throw_exception_if_invalid()
        return VaeRef._wrap(await rt.refs.create("VAE", result))  # type: ignore[return-value]

    @staticmethod
    def _checked_compile_options(
        backend: str, mode: str, fullgraph: bool,
    ) -> dict[str, Any]:
        if backend not in {"inductor", "cudagraphs"}:
            raise ValueError(
                "torch.compile backend must be inductor or cudagraphs")
        if mode not in {
            "default",
            "max-autotune",
            "max-autotune-no-cudagraphs",
            "reduce-overhead",
        }:
            raise ValueError(f"unsupported torch.compile mode {mode!r}")
        if not isinstance(fullgraph, bool):
            raise TypeError("torch.compile fullgraph must be a bool")
        return {
            "backend": backend,
            "mode": mode,
            "fullgraph": fullgraph,
        }

    async def _vae_compile(
        self, vae: "VaeRef", backend: str = "inductor",
        mode: str = "default", fullgraph: bool = False,
        encoder: bool = True, decoder: bool = True,
    ) -> "VaeRef":
        import copy
        import torch

        if not isinstance(encoder, bool) or not isinstance(decoder, bool):
            raise TypeError("VAE compile encoder and decoder flags must be bools")
        options = self._checked_compile_options(backend, mode, fullgraph)
        if not encoder and not decoder:
            return vae
        rt = current_runtime()
        source = await rt.refs.resolve(vae)
        stage_source = getattr(source, "first_stage_model", None)
        patcher_source = getattr(source, "patcher", None)
        if stage_source is None or patcher_source is None:
            raise TypeError("VAE compile needs a valid first-stage model")

        result = copy.copy(source)
        stage = copy.copy(stage_source)
        stage._modules = stage_source._modules.copy()
        result.first_stage_model = stage
        result.patcher = patcher_source.clone()

        targets = []
        if encoder:
            targets.append(
                "taesd_encoder" if hasattr(stage, "taesd_encoder")
                else "encoder")
        if decoder:
            targets.append(
                "taesd_decoder" if hasattr(stage, "taesd_decoder")
                else "decoder")
        for name in targets:
            module = getattr(stage, name, None)
            if module is None:
                raise TypeError(f"VAE has no compilable {name}")
            setattr(stage, name, torch.compile(module, **options))
        return VaeRef._wrap(await rt.refs.create(
            "VAE", result))  # type: ignore[return-value]

    async def _vae_patch_triton(
        self, vae: "VaeRef", fuse_norm_silu: bool = True,
        channels_last: bool = True, int8_conv: bool = False,
        autotune: bool = False,
    ) -> "VaeRef":
        import torch

        options = {
            "fuse_norm_silu": fuse_norm_silu,
            "channels_last": channels_last,
            "int8_conv": int8_conv,
            "autotune": autotune,
        }
        if any(type(value) is not bool for value in options.values()):
            raise TypeError("Patch Triton VAE options must be booleans")
        if not isinstance(vae, VaeRef) or vae.kind != "VAE":
            raise TypeError("Patch Triton VAE needs a VAE ref")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Patch Triton VAE requires an NVIDIA CUDA device")
        try:
            from . import _triton_vae
        except ImportError as exc:
            raise RuntimeError(
                "Patch Triton VAE requires the optional triton package") from exc

        rt = current_runtime()
        source = await rt.refs.resolve(vae)
        if (getattr(source, "first_stage_model", None) is None
                or getattr(source, "patcher", None) is None):
            raise TypeError("Patch Triton VAE needs a valid VAE")
        result = _triton_vae.patch_vae(source, **options)
        return VaeRef._wrap(await rt.refs.create(
            "VAE", result))  # type: ignore[return-value]

    async def _clip_tokenize(self, clip: "ClipRef", text: str, kwargs: dict) -> dict:
        c = await current_runtime().refs.resolve(clip)
        return c.tokenize(text, **(kwargs or {}))

    async def _clip_encode_from_tokens_scheduled(
            self, clip: "ClipRef", tokens: dict,
            add_dict: dict = None) -> "CondRef":
        rt = current_runtime()
        c = await rt.refs.resolve(clip)
        # `add_dict` defaults to {} in core rather than None; passing None
        # through would be a different call.
        cond = c.encode_from_tokens_scheduled(tokens, add_dict=add_dict or {})
        return CondRef._wrap(await rt.refs.create("CONDITIONING", cond))  # type: ignore[return-value]

    async def _clip_encode(self, clip: "ClipRef", text: str) -> "CondRef":
        rt = current_runtime()
        c = await rt.refs.resolve(clip)
        tokens = c.tokenize(text)
        return CondRef._wrap(await rt.refs.create(  # type: ignore[return-value]
            "CONDITIONING", c.encode_from_tokens_scheduled(tokens)))

    async def _gligen_apply_batched(
        self, gligen: "GligenRef", conditioning: "CondRef", clip: "ClipRef",
        text: str, boxes: list,
    ) -> "CondRef":
        import math

        if not isinstance(text, str):
            raise TypeError("GLIGEN text must be a string")
        if not isinstance(boxes, (list, tuple)):
            raise TypeError("GLIGEN boxes must be a list")
        if len(boxes) > 4096:
            raise ValueError("GLIGEN boxes are limited to 4096 batch items")

        checked = []
        for index, box in enumerate(boxes):
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise TypeError(
                    f"GLIGEN box {index} must contain height, width, y, x")
            height, width, y, x = box
            if any(isinstance(value, bool) for value in box):
                raise TypeError(f"GLIGEN box {index} cannot contain booleans")
            if not isinstance(height, int) or not isinstance(width, int):
                raise TypeError(
                    f"GLIGEN box {index} height and width must be integers")
            if not all(isinstance(value, (int, float)) for value in (y, x)):
                raise TypeError(
                    f"GLIGEN box {index} y and x must be numbers")
            if not all(math.isfinite(float(value)) for value in (y, x)):
                raise ValueError(
                    f"GLIGEN box {index} y and x must be finite")
            checked.append((height, width, y, x))

        rt = current_runtime()
        gligen_value = await rt.refs.resolve(gligen)
        source = await rt.refs.resolve(conditioning)
        clip_value = await rt.refs.resolve(clip)
        _, pooled = clip_value.encode_from_tokens(
            clip_value.tokenize(text), return_pooled=True)

        positions = [
            [(pooled, height, width, y, x)]
            for height, width, y, x in checked
        ]
        result = []
        for item in source:
            metadata = item[1].copy()
            previous = (
                metadata["gligen"][2]
                if "gligen" in metadata else [[] for _ in checked]
            )
            combined = [
                prior + current
                for prior, current in zip(previous, positions)
            ]
            metadata["gligen"] = (
                "position_batched", gligen_value, combined)
            result.append([item[0], metadata])
        return CondRef._wrap(await rt.refs.create(  # type: ignore[return-value]
            "CONDITIONING", result))

    async def _cond_combine(self, cond: "CondRef", other: "CondRef") -> "CondRef":
        rt = current_runtime()
        a = await rt.refs.resolve(cond)
        b = await rt.refs.resolve(other)
        return CondRef._wrap(await rt.refs.create("CONDITIONING", a + b))  # type: ignore[return-value]

    async def _latent_minimax_h3_token_count(
        self, latent: "LatentRef", conditioning: "CondRef",
    ) -> dict[str, Any]:
        import inspect

        try:
            from comfy.ldm.minimax.model import PackedLayout
        except ImportError as error:
            raise RuntimeError(
                "MiniMax H3 token counting requires core MiniMax H3 support; "
                "update ComfyUI") from error

        rt = current_runtime()
        latent_value = await rt.refs.resolve(latent)
        conditioning_value = await rt.refs.resolve(conditioning)
        if not isinstance(latent_value, dict) or "samples" not in latent_value:
            raise TypeError("MiniMax H3 token counting needs a LATENT with samples")
        samples = latent_value["samples"]
        if getattr(samples, "is_nested", False):
            video, audio = samples.unbind()[:2]
            audio_length = audio.shape[-1]
        else:
            video, audio_length = samples, 0
        if video.ndim != 5:
            raise ValueError(
                "MiniMax H3 token counting expected a video latent of shape "
                f"[B, C, T, H, W], got {tuple(video.shape)}")

        latent_length = video.shape[2]
        latent_height = (video.shape[3] + 1) // 2 * 2
        latent_width = (video.shape[4] + 1) // 2 * 2
        supported = set(inspect.signature(PackedLayout.__init__).parameters)

        def build_layout(condition, metadata):
            options = {
                "keyframes": metadata.get("minimax_keyframes"),
                "refs": metadata.get("minimax_refs"),
                "frame_count": metadata.get("minimax_frame_count"),
            }
            missing = [
                name for name, value in options.items()
                if value is not None and name not in supported
            ]
            if missing:
                raise RuntimeError(
                    "this ComfyUI version's MiniMax PackedLayout does not "
                    f"support {missing}; update ComfyUI")
            return PackedLayout(
                condition.shape[1], latent_length, latent_height,
                latent_width, audio_length,
                **{
                    name: value for name, value in options.items()
                    if value is not None and name in supported
                })

        layout = max(
            (build_layout(condition, metadata)
             for condition, metadata in conditioning_value),
            key=lambda item: item.seq_len)
        rows = {}
        segment_counts = {}
        for start, stop, kind in layout.segments:
            rows[kind] = rows.get(kind, 0) + stop - start
            segment_counts[kind] = segment_counts.get(kind, 0) + 1

        parts = [("total", str(layout.seq_len))]
        if layout.seq_len * 7168 >= 2**31:
            parts.append((
                "WARNING",
                "over the int32-safe attention range "
                f"({2**31 // 7168} tokens), sageattention kernels may overflow",
            ))
        parts.append(("text", str(rows.get("text", 0))))
        if "cond" in rows:
            count = segment_counts["cond"]
            parts.append((
                "keyframes",
                f"{rows['cond']} ({count} frame{'s' if count > 1 else ''})",
            ))
        if "ref_img" in rows:
            count = segment_counts["ref_img"]
            parts.append((
                "image/video refs",
                f"{rows['ref_img']} ({count} block{'s' if count > 1 else ''})",
            ))
        if "ref_audio" in rows:
            count = segment_counts["ref_audio"]
            parts.append((
                "audio refs",
                f"{rows['ref_audio']} ({count} block{'s' if count > 1 else ''})",
            ))
        parts.append(("audio", str(rows.get("audio", 0))))
        parts.append((
            "video",
            f"{rows.get('video', 0)} "
            f"({latent_length}x{latent_height // 2}x{latent_width // 2} patches)",
        ))
        return {
            "tokens": int(layout.seq_len),
            "breakdown": "\n".join(f"{key}: {value}" for key, value in parts),
        }

    async def _cond_sequence_length(self, cond: "CondRef") -> int:
        value = await current_runtime().refs.resolve(cond)
        if (
            not isinstance(value, (list, tuple))
            or not value
            or not isinstance(value[0], (list, tuple))
            or not value[0]
            or not hasattr(value[0][0], "shape")
            or len(value[0][0].shape) < 2
        ):
            raise TypeError(
                "conditioning must contain a sequence-shaped embedding")
        return int(value[0][0].shape[1])

    async def _cond_concat(self, cond: "CondRef", other: "CondRef") -> "CondRef":
        rt = current_runtime()
        to_concat = await rt.refs.resolve(cond)
        source = await rt.refs.resolve(other)
        import torch

        out = []
        cond_from = source[0][0]
        for t in to_concat:
            tw = torch.cat((t[0], cond_from), 1)
            out.append([tw, t[1].copy()])
        return CondRef._wrap(await rt.refs.create("CONDITIONING", out))  # type: ignore[return-value]

    async def _advanced_control_weights_from_list(
        self, _subject: Optional["Ref"], weights: list,
        uncond_multiplier: float = 1.0, extras: Any = None,
    ) -> tuple["ControlNetWeightsRef", "TimestepKeyframeRef"]:
        import importlib
        import math

        if not isinstance(weights, (list, tuple)):
            raise TypeError("ControlNet weights must be a list")
        if len(weights) > 4096:
            raise ValueError("ControlNet weights are limited to 4096 values")
        checked_weights = []
        for value in weights:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("ControlNet weights must contain only numbers")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("ControlNet weights must be finite")
            checked_weights.append(number)
        if (isinstance(uncond_multiplier, bool)
                or not isinstance(uncond_multiplier, (int, float))):
            raise TypeError("uncond_multiplier must be a number")
        multiplier = float(uncond_multiplier)
        if not math.isfinite(multiplier) or not 0.0 <= multiplier <= 1.0:
            raise ValueError("uncond_multiplier must be finite and in [0, 1]")

        rt = current_runtime()
        checked_extras = extras
        if isinstance(checked_extras, Ref):
            if checked_extras.kind != "VALUE":
                raise TypeError("ControlNet extras ref must contain VALUE data")
            checked_extras = await rt.refs.resolve(checked_extras)
        if checked_extras is None:
            checked_extras = {}
        if not isinstance(checked_extras, dict):
            raise TypeError("ControlNet extras must be a mapping")

        def validate_extra(value: Any, depth: int = 0) -> None:
            if depth > 32:
                raise ValueError("ControlNet extras nesting exceeds 32 levels")
            if _looks_like_tensor(value):
                return
            if value is None or isinstance(value, (str, bool, int)):
                return
            if isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError("ControlNet extras must contain finite numbers")
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    validate_extra(item, depth + 1)
                return
            if isinstance(value, dict):
                if not all(isinstance(key, str) for key in value):
                    raise TypeError("ControlNet extras keys must be strings")
                for item in value.values():
                    validate_extra(item, depth + 1)
                return
            raise TypeError(
                f"ControlNet extras cannot contain {type(value).__name__}")

        validate_extra(checked_extras)

        adv_control = importlib.import_module(
            "ComfyUI-Advanced-ControlNet.adv_control")
        control_weights = adv_control.utils.ControlWeights.controlnet(
            weights_input=checked_weights,
            uncond_multiplier=multiplier,
            extras=checked_extras,
        )
        keyframe = adv_control.utils.TimestepKeyframe(
            control_weights=control_weights)
        shortcut = adv_control.utils.TimestepKeyframeGroup.default(keyframe)
        weights_ref = ControlNetWeightsRef._wrap(await rt.refs.create(
            "CONTROL_NET_WEIGHTS", control_weights))
        shortcut_ref = TimestepKeyframeRef._wrap(await rt.refs.create(
            "TIMESTEP_KEYFRAME", shortcut))
        return weights_ref, shortcut_ref

    async def _model_apply_dit_block_lora(
        self, model: "ModelRef", asset: "AssetRef", strength_model: float,
        block_weights: list[dict[str, Any]],
    ) -> tuple["ModelRef", str]:
        import math

        import comfy.lora
        import folder_paths
        from comfy.utils import load_torch_file

        if (isinstance(strength_model, bool)
                or not isinstance(strength_model, (int, float))):
            raise TypeError("strength_model must be a number")
        strength = float(strength_model)
        if not math.isfinite(strength) or not -100.0 <= strength <= 100.0:
            raise ValueError("strength_model must be finite and in [-100, 100]")
        if not isinstance(block_weights, list) or len(block_weights) > 108:
            raise TypeError("block_weights must be a closed block-selection list")

        limits = {"double_blocks": 20, "single_blocks": 40, "blocks": 48}
        selected = []
        seen = set()
        for item in block_weights:
            if not isinstance(item, dict) or set(item) != {
                    "family", "index", "ratio"}:
                raise TypeError("each block selection needs family, index, ratio")
            family = item["family"]
            index = item["index"]
            ratio = item["ratio"]
            if family not in limits:
                raise ValueError(f"unsupported block family {family!r}")
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("block index must be an integer")
            if not 0 <= index < limits[family]:
                raise ValueError("block index is outside its closed family")
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
                raise TypeError("block ratio must be a number")
            ratio = float(ratio)
            if not math.isfinite(ratio) or not 0.0 <= ratio <= 10000.0:
                raise ValueError("block ratio must be finite and in [0, 10000]")
            identity = (family, index)
            if identity in seen:
                raise ValueError("duplicate block selection")
            seen.add(identity)
            selected.append((f"{family}.{index}.", ratio))

        rt = current_runtime()
        source_model = await rt.refs.resolve(model)
        if not isinstance(asset, AssetRef) or asset.kind != "ASSET":
            raise TypeError("DiT LoRA application needs an ASSET ref")
        path = await rt.refs.resolve(asset)
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError("LoRA ASSET ref does not contain a path")
        path = _InProcessAssets._confined_resolved_path(
            path, folder_paths.get_folder_paths("loras"), "loras")
        lora = load_torch_file(path, safe_load=True)
        if not isinstance(lora, dict):
            raise TypeError("LoRA asset must contain a state-dict mapping")

        weight_key = next(
            (key for key in lora if isinstance(key, str)
             and key.endswith("weight")), None)
        if weight_key is None:
            rank = "Couldn't find rank"
        else:
            weight = lora[weight_key]
            if not hasattr(weight, "shape") or len(weight.shape) < 1:
                raise TypeError("first LoRA weight has no rank dimension")
            rank = str(weight.shape[0])

        key_map = comfy.lora.model_lora_keys_unet(source_model.model, {})
        loaded = comfy.lora.load_lora(lora, key_map)
        for prefix, ratio in selected:
            for key in list(loaded):
                if isinstance(key, str):
                    matched = prefix in key
                elif isinstance(key, tuple):
                    matched = any(
                        isinstance(part, str) and prefix in part
                        for part in key)
                else:
                    matched = False
                if not matched:
                    continue
                if ratio == 0.0:
                    del loaded[key]
                    continue
                value = loaded[key]
                if hasattr(value, "weights"):
                    values = list(value.weights)
                    if len(values) < 3:
                        raise TypeError("LoRA adapter weights have no alpha slot")
                    values[2] = ratio
                    value.weights = tuple(values)

        patched = source_model.clone()
        patched.add_patches(loaded, strength)
        return ModelRef._wrap(await rt.refs.create("MODEL", patched)), rank

    async def _model_apply_ltx2_lora(
        self, model: "ModelRef", asset: "AssetRef", strength_model: float,
        block_weights: list[dict[str, Any]], video: float,
        video_to_audio: float, audio: float, audio_to_video: float,
        other: float,
    ) -> tuple["ModelRef", str, str]:
        import math

        import comfy.lora
        import folder_paths
        from comfy.utils import load_torch_file

        if (isinstance(strength_model, bool)
                or not isinstance(strength_model, (int, float))):
            raise TypeError("strength_model must be a number")
        strength = float(strength_model)
        if not math.isfinite(strength) or not -100.0 <= strength <= 100.0:
            raise ValueError("strength_model must be finite and in [-100, 100]")

        layer_strengths = {}
        for name, value in {
            "video": video,
            "video_to_audio": video_to_audio,
            "audio": audio,
            "audio_to_video": audio_to_video,
            "other": other,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
            layer_strengths[name] = value

        if not isinstance(block_weights, list) or len(block_weights) > 48:
            raise TypeError("block_weights must be a closed LTX2 block list")
        selected = []
        seen = set()
        for item in block_weights:
            if not isinstance(item, dict) or set(item) != {
                    "family", "index", "ratio"}:
                raise TypeError("each block selection needs family, index, ratio")
            family = item["family"]
            index = item["index"]
            ratio = item["ratio"]
            if family != "blocks":
                raise ValueError(f"unsupported LTX2 block family {family!r}")
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("block index must be an integer")
            if not 0 <= index < 48:
                raise ValueError("block index is outside the LTX2 block family")
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
                raise TypeError("block ratio must be a number")
            ratio = float(ratio)
            if not math.isfinite(ratio) or not 0.0 <= ratio <= 10000.0:
                raise ValueError("block ratio must be finite and in [0, 10000]")
            if index in seen:
                raise ValueError("duplicate LTX2 block selection")
            seen.add(index)
            selected.append((f"blocks.{index}.", ratio))

        rt = current_runtime()
        source_model = await rt.refs.resolve(model)
        if not isinstance(asset, AssetRef) or asset.kind != "ASSET":
            raise TypeError("LTX2 LoRA application needs an ASSET ref")
        path = await rt.refs.resolve(asset)
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError("LoRA ASSET ref does not contain a path")
        path = _InProcessAssets._confined_resolved_path(
            path, folder_paths.get_folder_paths("loras"), "loras")
        lora = load_torch_file(path, safe_load=True)
        if not isinstance(lora, dict):
            raise TypeError("LoRA asset must contain a state-dict mapping")

        weight_key = next(
            (key for key in lora if isinstance(key, str)
             and key.endswith("weight")), None)
        if weight_key is None:
            rank = "Couldn't find rank"
        else:
            weight = lora[weight_key]
            if not hasattr(weight, "shape") or len(weight.shape) < 1:
                raise TypeError("first LoRA weight has no rank dimension")
            rank = str(weight.shape[0])

        key_map = comfy.lora.model_lora_keys_unet(source_model.model, {})
        loaded = comfy.lora.load_lora(lora, key_map)
        if not isinstance(loaded, dict):
            raise TypeError("mapped LoRA patches must be a dictionary")

        keys_to_delete = []
        for prefix, ratio in selected:
            for key in list(loaded):
                if isinstance(key, str):
                    matched = prefix in key
                elif isinstance(key, tuple):
                    matched = any(
                        isinstance(part, str) and prefix in part
                        for part in key)
                else:
                    matched = False
                if not matched:
                    continue
                if ratio == 0.0:
                    keys_to_delete.append(key)
                    continue
                value = loaded[key]
                if hasattr(value, "weights"):
                    values = list(value.weights)
                    if len(values) < 3:
                        raise TypeError("LoRA adapter weights have no alpha slot")
                    values[2] = ratio
                    value.weights = tuple(values)

        for key in list(loaded):
            if key in keys_to_delete:
                continue
            key_str = (
                key if isinstance(key, str)
                else key[0] if isinstance(key, tuple)
                else str(key)
            )
            if "video_to_audio_attn" in key_str:
                multiplier = layer_strengths["video_to_audio"]
            elif "audio_to_video_attn" in key_str:
                multiplier = layer_strengths["audio_to_video"]
            elif "audio_attn" in key_str or "audio_ff.net" in key_str:
                multiplier = layer_strengths["audio"]
            elif "attn" in key_str or "ff.net" in key_str:
                multiplier = layer_strengths["video"]
            else:
                multiplier = layer_strengths["other"]

            if multiplier == 0.0:
                keys_to_delete.append(key)
            elif multiplier != 1.0:
                value = loaded[key]
                if hasattr(value, "weights"):
                    values = list(value.weights)
                    if len(values) < 3:
                        raise TypeError("LoRA adapter weights have no alpha slot")
                    alpha = values[2] if values[2] is not None else 1.0
                    values[2] = alpha * multiplier
                    value.weights = tuple(values)

        for key in keys_to_delete:
            loaded.pop(key, None)

        loaded_keys = []
        for key, value in loaded.items():
            key_str = key if isinstance(key, str) else str(key)
            if hasattr(value, "weights"):
                alpha = value.weights[2]
                alpha = alpha if alpha is not None else "None"
                loaded_keys.append(f"{key_str}: alpha={alpha}")
            else:
                loaded_keys.append(
                    f"{key_str}: type={type(value).__name__}")

        patched = source_model.clone()
        accepted = set(patched.add_patches(loaded, strength))
        for key in loaded:
            if key not in accepted:
                key_str = key if isinstance(key, str) else str(key)
                loaded_keys.append(f"NOT LOADED: {key_str}")
        info = "\n".join(loaded_keys)
        return (
            ModelRef._wrap(await rt.refs.create("MODEL", patched)), rank, info)

    async def _lora_weight_differences(
        self, finetuned: "Ref", original: "Ref", include_bias: bool = False,
    ) -> "WeightDiffCursorRef":
        if type(include_bias) is not bool:
            raise TypeError("include_bias must be a boolean")

        rt = current_runtime()
        finetuned_value = await rt.refs.resolve(finetuned)
        original_value = await rt.refs.resolve(original)
        if finetuned.kind != original.kind:
            raise TypeError("finetuned and original must both be MODEL or both be CLIP")

        if finetuned.kind == "MODEL":
            diff = finetuned_value.clone()
            diff.add_patches(
                original_value.get_key_patches("diffusion_model."),
                -1.0, 1.0)
            patcher = diff
            input_prefix = "diffusion_model."
            output_prefix = "diffusion_model."
        elif finetuned.kind == "CLIP":
            diff = finetuned_value.clone()
            patches = {
                key: value
                for key, value in original_value.get_key_patches().items()
                if not key.endswith(".position_ids")
                and not key.endswith(".logit_scale")
            }
            diff.add_patches(patches, -1.0, 1.0)
            patcher = diff.patcher
            input_prefix = ""
            output_prefix = "text_encoders."
        else:
            raise TypeError("weight differences require MODEL or CLIP refs")

        keys = []
        names = [name for name, _ in patcher.model.named_parameters()]
        names.extend(name for name, _ in patcher.model.named_buffers())
        for name in names:
            if not name.startswith(input_prefix):
                continue
            if name.endswith(".weight"):
                stem = name[len(input_prefix):-7]
                keys.append((name, f"{output_prefix}{stem}", "weight"))
            elif include_bias and name.endswith(".bias"):
                stem = name[len(input_prefix):-5]
                keys.append((name, f"{output_prefix}{stem}", "bias"))

        state = _WeightDiffCursorState(patcher=patcher, keys=keys)
        return WeightDiffCursorRef._wrap(await rt.refs.create(
            "WEIGHT_DIFF_CURSOR", state))  # type: ignore[return-value]

    async def _weight_diff_next(
        self, cursor: "WeightDiffCursorRef",
    ) -> Optional[dict[str, Any]]:
        rt = current_runtime()
        state = await rt.refs.resolve(cursor)
        if not isinstance(state, _WeightDiffCursorState):
            raise TypeError("weight-difference cursor has invalid host state")

        async with state.lock:
            if state.current_ref is not None:
                await rt.refs.release(state.current_ref)
                state.current_ref = None
            if state.closed:
                return None
            if state.index >= len(state.keys):
                state.patcher = None
                state.keys.clear()
                state.closed = True
                import comfy.model_management
                comfy.model_management.soft_empty_cache()
                return None

            source_key, output_key, item_kind = state.keys[state.index]
            state.index += 1
            weight = state.patcher.patch_weight_to_device(
                source_key, return_weight=True)
            item = {
                "output_key": output_key,
                "kind": item_kind,
                "position": state.index,
                "total": len(state.keys),
                "tensor": None,
                "ndim": None,
            }
            if weight is None:
                return item
            item["ndim"] = weight.ndim
            if weight.ndim == 5:
                return item
            state.current_ref = TensorRef._wrap(await rt.refs.create(
                "TENSOR", weight))  # type: ignore[assignment]
            item["tensor"] = state.current_ref
            return item

    async def _model_latent_scale_factor(self, model: "ModelRef") -> float:
        value = await current_runtime().refs.resolve(model)
        return float(value.model.latent_format.scale_factor)

    async def _clip_vision_encode_image(
        self, clip_vision: "ClipVisionRef", image: "ImageRef",
        crop: bool = True,
    ) -> "ClipVisionOutputRef":
        rt = current_runtime()
        encoder = await rt.refs.resolve(clip_vision)
        pixels = await rt.refs.resolve(image)
        output = encoder.encode_image(pixels, crop=bool(crop))
        return ClipVisionOutputRef._wrap(await rt.refs.create(
            "CLIP_VISION_OUTPUT", output))  # type: ignore[return-value]

    async def _clip_vision_output_image_embeds(
        self, output: "ClipVisionOutputRef",
    ) -> "TensorRef":
        rt = current_runtime()
        value = await rt.refs.resolve(output)
        return TensorRef._wrap(await rt.refs.create(
            "TENSOR", value.image_embeds))  # type: ignore[return-value]

    async def _controlnet_with_union_type(
        self, control_net: "ControlNetRef", type_number=None,
    ) -> "ControlNetRef":
        rt = current_runtime()
        value = await rt.refs.resolve(control_net)
        clone = value.copy()
        selected = [] if type_number is None else [int(type_number)]
        clone.set_extra_arg("control_type", selected)
        return ControlNetRef._wrap(await rt.refs.create(
            "CONTROL_NET", clone))  # type: ignore[return-value]

    async def _controlnet_compile(
        self, control_net: "ControlNetRef", backend: str = "inductor",
        mode: str = "default", fullgraph: bool = False,
    ) -> "ControlNetRef":
        import torch

        options = self._checked_compile_options(backend, mode, fullgraph)
        rt = current_runtime()
        source = await rt.refs.resolve(control_net)
        control_model = getattr(source, "control_model", None)
        if control_model is None or not callable(getattr(source, "copy", None)):
            raise TypeError("CONTROL_NET has no compilable control model")
        result = source.copy()
        result.control_model = torch.compile(control_model, **options)
        return ControlNetRef._wrap(await rt.refs.create(
            "CONTROL_NET", result))  # type: ignore[return-value]

    async def _style_model_apply(
        self, style_model: "StyleModelRef",
        clip_vision_output: "ClipVisionOutputRef",
        conditioning: "CondRef", strength: float = 1.0,
    ) -> "CondRef":
        import torch

        rt = current_runtime()
        model = await rt.refs.resolve(style_model)
        vision = await rt.refs.resolve(clip_vision_output)
        source = await rt.refs.resolve(conditioning)
        style = model.get_cond(vision).flatten(
            start_dim=0, end_dim=1).unsqueeze(dim=0)
        style = float(strength) * style
        result = [[torch.cat((item[0], style), dim=1), item[1].copy()]
                  for item in source]
        return CondRef._wrap(await rt.refs.create(
            "CONDITIONING", result))  # type: ignore[return-value]

    async def _clipseg_segment(
        self, clipseg: "ClipSegRef", images: "ImageRef", text: str,
        threshold: float = 0.5, binary_mask: bool = True,
        combine_mask: bool = False, use_accelerator: bool = True,
        blur_sigma: float = 0.0, previous_mask: Optional["MaskRef"] = None,
        invert: bool = False, image_background_level: float = 0.5,
    ) -> tuple["MaskRef", "ImageRef"]:
        from contextlib import nullcontext
        import numpy as np
        import torch
        import torch.nn.functional as functional
        import torchvision.transforms as transforms
        from PIL import Image
        import comfy.model_management

        threshold = float(threshold)
        blur_sigma = float(blur_sigma)
        background = float(image_background_level)
        if not 0.0 <= threshold <= 10.0:
            raise ValueError("CLIPSeg threshold must be in [0, 10]")
        if not 0.0 <= blur_sigma <= 100.0:
            raise ValueError("CLIPSeg blur_sigma must be in [0, 100]")
        if not 0.0 <= background <= 1.0:
            raise ValueError("CLIPSeg image background level must be in [0, 1]")

        rt = current_runtime()
        bundle = await rt.refs.resolve(clipseg)
        pixels = await rt.refs.resolve(images)
        previous = (None if previous_mask is None
                    else await rt.refs.resolve(previous_mask))
        model = bundle["model"]
        processor = bundle["processor"]
        offload_device = comfy.model_management.unet_offload_device()
        device = (comfy.model_management.get_torch_device()
                  if use_accelerator else torch.device("cpu"))
        dtype = comfy.model_management.unet_dtype()
        model.to(dtype).to(device)
        try:
            height, width = pixels.shape[1:3]
            source = pixels.to(device)
            autocast = (
                dtype != torch.float32
                and not comfy.model_management.is_device_mps(device))
            scope = (torch.autocast(
                comfy.model_management.get_autocast_device(device), dtype=dtype)
                if autocast else nullcontext())
            with scope:
                pil_images = [Image.fromarray(np.clip(
                    255.0 * image.cpu().numpy().squeeze(), 0, 255
                ).astype(np.uint8)) for image in source]
                inputs = processor(
                    text=[str(text)] * len(source), images=pil_images,
                    return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                outputs = model(**inputs)
            mask = torch.sigmoid(outputs.logits)
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask = torch.where(
                mask > threshold, mask,
                torch.tensor(0, dtype=torch.float, device=mask.device))
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = functional.interpolate(
                mask.unsqueeze(1), size=(height, width), mode="nearest"
            ).squeeze(1)
        finally:
            model.to(offload_device)

        if binary_mask:
            mask = (mask > 0).float()
        if blur_sigma > 0:
            kernel_size = 6 * int(blur_sigma) + 1
            mask = transforms.GaussianBlur(
                kernel_size=(kernel_size, kernel_size),
                sigma=(blur_sigma, blur_sigma))(mask)
        if combine_mask:
            mask = torch.max(mask, dim=0)[0].unsqueeze(0).repeat(
                len(source), 1, 1)
        comfy.model_management.soft_empty_cache()
        if previous is not None:
            if previous.shape != mask.shape:
                previous = functional.interpolate(
                    previous.unsqueeze(1), size=(height, width), mode="nearest")
            mask = mask + previous.to(device)
            torch.clamp(mask, min=0.0, max=1.0)
        if invert:
            mask = 1 - mask
        result_image = torch.clamp(
            source * mask.unsqueeze(-1)
            + (1 - mask.unsqueeze(-1)) * background,
            min=0.0, max=1.0).cpu().float()
        result_mask = mask.cpu().float()
        mask_ref = MaskRef._wrap(await rt.refs.create("MASK", result_mask))
        image_ref = ImageRef._wrap(await rt.refs.create("IMAGE", result_image))
        return mask_ref, image_ref  # type: ignore[return-value]

    async def _upscale_model_upscale(
        self, upscale_model: "UpscaleModelRef", images: "ImageRef",
        per_batch: int = 16, downscale_ratio: float = 1.0,
        downscale_method: str = "lanczos", precision: str = "float32",
    ) -> "ImageRef":
        import torch
        import comfy.model_management
        from comfy.utils import ProgressBar, common_upscale

        batch_size = int(per_batch)
        ratio = float(downscale_ratio)
        methods = {"nearest-exact", "bilinear", "area", "bicubic", "lanczos"}
        dtypes = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if not 1 <= batch_size <= 4096:
            raise ValueError("upscale per_batch must be in [1, 4096]")
        if not 0.01 <= ratio <= 1.0:
            raise ValueError("upscale downscale_ratio must be in [0.01, 1]")
        if downscale_method not in methods:
            raise ValueError(f"unknown upscale downscale method {downscale_method!r}")
        if precision not in dtypes:
            raise ValueError(f"unknown upscale precision {precision!r}")

        rt = current_runtime()
        model = await rt.refs.resolve(upscale_model)
        pixels = await rt.refs.resolve(images)
        parameter = next(model.model.parameters())
        previous_device = parameter.device
        previous_dtype = parameter.dtype
        dtype = dtypes[precision]
        device = comfy.model_management.get_torch_device()
        outputs = []
        try:
            model.to(device, dtype=dtype)
            source = pixels.movedim(-1, -3).to(dtype)
            progress = ProgressBar(source.shape[0])
            for start in range(0, source.shape[0], batch_size):
                batch = model(source[start:start + batch_size].to(device))
                outputs.append(batch.cpu())
                progress.update(batch.shape[0])
        finally:
            model.to(previous_device, dtype=previous_dtype)
        output = torch.cat(outputs, dim=0).permute(0, 2, 3, 1).cpu().float()
        if ratio < 1.0:
            height = int(output.shape[1] * ratio)
            width = int(output.shape[2] * ratio)
            output = common_upscale(
                output.movedim(-1, 1), width, height,
                downscale_method, "disabled").movedim(1, -1)
        return ImageRef._wrap(await rt.refs.create("IMAGE", output))  # type: ignore[return-value]

    async def _sampler_self_refine_video(
        self, latent: Optional["LatentRef"],
        stochastic_steps: list[dict[str, int]],
        certain_percentage: float, uncertainty_threshold: float,
        seed: int, verbose: bool = False,
    ) -> "SamplerRef":
        import math

        from comfy.samplers import KSAMPLER

        if not isinstance(stochastic_steps, list) or len(stochastic_steps) > 1000:
            raise TypeError("stochastic_steps must be a closed step list")
        step_map = {}
        for item in stochastic_steps:
            if not isinstance(item, dict) or set(item) != {
                    "step", "anneal_steps"}:
                raise TypeError("each stochastic step needs step and anneal_steps")
            step = item["step"]
            count = item["anneal_steps"]
            if isinstance(step, bool) or not isinstance(step, int):
                raise TypeError("stochastic step must be an integer")
            if not 0 <= step <= 999:
                raise ValueError("stochastic step must be in [0, 999]")
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError("anneal_steps must be an integer")
            if not 1 <= count <= 100:
                raise ValueError("anneal_steps must be in [1, 100]")
            if step in step_map:
                raise ValueError("duplicate stochastic step")
            step_map[step] = count

        scalars = {}
        for name, value in {
            "certain_percentage": certain_percentage,
            "uncertainty_threshold": uncertainty_threshold,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
            scalars[name] = value
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= seed <= 0xffffffffffffffff:
            raise ValueError("seed must be in [0, 2**64 - 1]")
        if type(verbose) is not bool:
            raise TypeError("verbose must be a boolean")

        rt = current_runtime()
        video_shape = None
        if latent is not None:
            if not isinstance(latent, LatentRef) or latent.kind != "LATENT":
                raise TypeError("self-refine sampler needs a LATENT ref")
            value = await rt.refs.resolve(latent)
            if not isinstance(value, dict) or "samples" not in value:
                raise TypeError("LATENT ref has no samples")
            samples = value["samples"]
            if not hasattr(samples, "shape"):
                raise TypeError("LATENT samples have no shape")
            video_shape = samples.shape

        sampler = KSAMPLER(_sample_self_refine_video, {
            "stochastic_step_map": step_map,
            "certain_percentage": scalars["certain_percentage"],
            "uncertainty_threshold": scalars["uncertainty_threshold"],
            "verbose": verbose,
            "video_shape": video_shape,
            "seed": seed,
        })
        return SamplerRef._wrap(await rt.refs.create("SAMPLER", sampler))  # type: ignore[return-value]

    async def _guider_scheduled_cfg(
        self, model: "ModelRef", positive: "CondRef", negative: "CondRef",
        cfg: float, start_percent: float, end_percent: float,
    ) -> "GuiderRef":
        import math

        import torch
        from comfy.samplers import CFGGuider, sampling_function

        def checked_scalar(name, value, lower, upper):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            result = float(value)
            if not math.isfinite(result) or not lower <= result <= upper:
                raise ValueError(
                    f"{name} must be finite and in [{lower}, {upper}]")
            return result

        cfg_value = checked_scalar("cfg", cfg, 0.0, 100.0)
        start_value = checked_scalar(
            "start_percent", start_percent, 0.0, 1.0)
        end_value = checked_scalar(
            "end_percent", end_percent, 0.0, 1.0)
        if model.kind != "MODEL":
            raise TypeError("scheduled CFG needs a MODEL ref")
        if positive.kind != "CONDITIONING" or negative.kind != "CONDITIONING":
            raise TypeError(
                "scheduled CFG needs positive and negative CONDITIONING refs")

        class ScheduledCFGGuider(CFGGuider):
            def set_cfg(self, value, start, end):
                self.cfg = value
                self.start_percent = start
                self.end_percent = end

            def predict_noise(
                self, x, timestep, model_options=None, seed=None,
            ):
                if model_options is None:
                    model_options = {}
                steps = model_options["transformer_options"]["sample_sigmas"]
                if isinstance(timestep, torch.Tensor):
                    timestep_value = timestep.reshape(-1)[0].to(steps)
                else:
                    timestep_value = torch.tensor(
                        timestep, device=steps.device, dtype=steps.dtype)
                matched_step_index = torch.isclose(
                    steps, timestep_value).nonzero()
                if len(matched_step_index) > 0:
                    current_step_index = matched_step_index.item()
                else:
                    for index in range(len(steps) - 1):
                        if ((steps[index] - timestep_value)
                                * (steps[index + 1] - timestep_value) <= 0):
                            current_step_index = index
                            break
                    else:
                        current_step_index = 0
                current_percent = current_step_index / (len(steps) - 1)
                if self.start_percent <= current_percent <= self.end_percent:
                    uncond = self.conds.get("negative", None)
                    scale = self.cfg
                else:
                    uncond = None
                    scale = 1.0
                return sampling_function(
                    self.inner_model, x, timestep, uncond,
                    self.conds.get("positive", None), scale,
                    model_options=model_options, seed=seed)

        rt = current_runtime()
        model_value = await rt.refs.resolve(model)
        positive_value = await rt.refs.resolve(positive)
        negative_value = await rt.refs.resolve(negative)
        guider = ScheduledCFGGuider(model_value)
        guider.set_conds(positive_value, negative_value)
        guider.set_cfg(cfg_value, start_value, end_value)
        return GuiderRef._wrap(
            await rt.refs.create("GUIDER", guider))  # type: ignore[return-value]

    async def _model_patch(self, model: "ModelRef", transform: str,
                           params: dict) -> "ModelRef":
        """Apply a named transform on the trusted plane.

        Validation happens HERE, not in the guest. A guest-side check is a
        convenience for the node author and nothing more — the value that
        arrives is whatever the guest chose to send.
        """
        from . import _model_transforms

        rt = current_runtime()
        checked = _model_transforms.validate(transform, params or {})

        # Ref-valued parameters resolve through the ref table, which enforces
        # the kind. The implementation receives a real object and never a token,
        # so it cannot be tricked into treating one kind as another.
        spec = _model_transforms.TRANSFORMS[transform].params
        for name, value in list(checked.items()):
            if isinstance(spec[name], _model_transforms.RefOf) and value is not None:
                checked[name] = await rt.refs.resolve(value)

        m = await rt.refs.resolve(model)
        return ModelRef._wrap(await rt.refs.create(  # type: ignore[return-value]
            "MODEL", _model_transforms.TRANSFORMS[transform].apply(m, **checked)))

    async def _model_transforms(self, model: "ModelRef") -> list[dict]:
        from . import _model_transforms

        await current_runtime().refs.resolve(model)
        return _model_transforms.describe_all()


class InProcessExecutionBackend:
    async def dispatch(
        self,
        plan: ExecutionPlan,
        local_call: Callable[[], Awaitable[Any]],
        runtime: Optional[Runtime] = None,
    ) -> Any:
        return await local_call()


# --------------------------------------------------------------------------- #
# Input/output ref marshaling for SDK nodes (those declaring ``SDK_REFS``).
# Heavy inputs become refs before execute(); output refs resolve back to real
# objects for downstream (legacy) nodes. This is what makes execute() see
# assets, not buffers. Under the overlay this happens at the process boundary.
# --------------------------------------------------------------------------- #
def _looks_like_tensor(v: Any) -> bool:
    return type(v).__name__ == "Tensor" and hasattr(v, "shape")


def _is_plain_data(v: Any) -> bool:
    """Whether a value can cross a process boundary as data.

    Deliberately a whitelist. Everything else is a live engine object — a
    ModelPatcher, a conditioning list holding tensors, a VAE — and handing one
    to an out-of-process node is either impossible (it will not serialize) or
    exactly what the boundary exists to prevent.
    """
    if v is None or isinstance(v, (str, bool, int, float, bytes)):
        return True
    if isinstance(v, (list, tuple)):
        return all(_is_plain_data(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _is_plain_data(x) for k, x in v.items())
    return False


#: Live engine objects a node may receive, by the ref type that stands in for
#: them. Detection is duck-typed because these classes live in `comfy.*`, which
#: this module must not import.
def _ref_type_for(v: Any) -> tuple[type, str]:
    """Choose the narrowest handle that preserves the value's authority."""
    if _looks_like_tensor(v):
        return ImageRef, "IMAGE"
    value_type = type(v)
    if value_type.__module__.endswith("adv_control.utils"):
        if value_type.__name__ == "ControlWeights":
            return ControlNetWeightsRef, "CONTROL_NET_WEIGHTS"
        if value_type.__name__ == "TimestepKeyframeGroup":
            return TimestepKeyframeRef, "TIMESTEP_KEYFRAME"
    if (hasattr(v, "encode_image") and hasattr(v, "patcher") and
            hasattr(v, "image_size")):
        return ClipVisionRef, "CLIP_VISION"
    if (type(v).__name__ == "Output" and hasattr(v, "image_embeds")):
        return ClipVisionOutputRef, "CLIP_VISION_OUTPUT"
    if (hasattr(v, "get_control") and hasattr(v, "set_extra_arg") and
            hasattr(v, "copy")):
        return ControlNetRef, "CONTROL_NET"
    if hasattr(v, "get_cond") and hasattr(v, "model"):
        return StyleModelRef, "STYLE_MODEL"
    if (type(v).__name__ == "ImageModelDescriptor"
            and hasattr(v, "model") and hasattr(v, "scale")
            and hasattr(v, "to")):
        return UpscaleModelRef, "UPSCALE_MODEL"
    if (hasattr(v, "get_stream_source") and hasattr(v, "get_components")
            and hasattr(v, "save_to")):
        return VideoRef, "VIDEO"
    if (hasattr(v, "model_patcher") and hasattr(v, "set_conds")
            and hasattr(v, "outer_sample") and hasattr(v, "predict_noise")):
        return GuiderRef, "GUIDER"
    if (type(v).__name__ == "KSAMPLER"
            and callable(getattr(v, "sample", None))
            and callable(getattr(v, "sampler_function", None))
            and isinstance(getattr(v, "extra_options", None), dict)):
        return SamplerRef, "SAMPLER"
    inner = getattr(v, "model", None)
    if (hasattr(v, "model_options") and hasattr(v, "load_device")
            and hasattr(inner, "set_position")
            and hasattr(inner, "set_empty")
            and hasattr(inner, "position_net")
            and hasattr(inner, "module_list")):
        return GligenRef, "GLIGEN"
    if hasattr(v, "model_options") and hasattr(v, "load_device"):
        return ModelRef, "MODEL"          # ModelPatcher
    if hasattr(v, "encode_from_tokens") or hasattr(v, "tokenize"):
        return ClipRef, "CLIP"
    if hasattr(v, "decode") and hasattr(v, "encode"):
        return VaeRef, "VAE"
    if isinstance(v, dict):
        if set(v) >= {"model", "processor"}:
            return ClipSegRef, "CLIPSEGMODEL"
        if "samples" in v:
            return LatentRef, "LATENT"
        if "waveform" in v and "sample_rate" in v:
            return AudioRef, "AUDIO"
        return ValueRef, "VALUE"
    if isinstance(v, (list, tuple)):
        if (v and isinstance(v[0], (list, tuple)) and len(v[0]) == 2 and
                _looks_like_tensor(v[0][0]) and isinstance(v[0][1], dict)):
            return CondRef, "CONDITIONING"
        return ValueRef, "VALUE"
    return OpaqueRef, "OPAQUE"


async def wrap_inputs(resolver: "RefResolver", inputs: dict) -> dict:
    """Replace live engine objects with refs before a node sees them.

    An SDK_REFS node is handed handles, never the objects themselves, so the
    same node body works in-process and out-of-process. The rule is by
    capability rather than by an enumerated type list: if a value cannot cross
    as data, it becomes a handle. That is what lets a node take a MODEL or a
    CONDITIONING — which are live engine objects — and still run in a guest.
    """
    out = {}
    for k, v in inputs.items():
        if _is_plain_data(v):
            out[k] = v
            continue
        ref_cls, kind = _ref_type_for(v)
        out[k] = ref_cls._wrap(await resolver.create(kind, v))
    return out


async def unwrap_outputs(resolver: "RefResolver", node_output: Any) -> Any:
    args = getattr(node_output, "result", None)
    if not args:
        return node_output
    async def resolve(value: Any) -> Any:
        if isinstance(value, Ref):
            return await resolver.resolve(value)
        if isinstance(value, list):
            return [await resolve(item) for item in value]
        if isinstance(value, tuple):
            return tuple([await resolve(item) for item in value])
        if isinstance(value, dict):
            return {key: await resolve(item) for key, item in value.items()}
        return value

    resolved = [await resolve(value) for value in args]
    from ._io import NodeOutput

    # Rebuilding the NodeOutput must preserve everything that is not a result.
    # Dropping `ui` here silently made every SDK_REFS node unable to be an
    # output node: ComfyUI only sends the `executed` event that carries results
    # to the frontend for nodes returning ui data, so a PreviewImage-style node
    # would run correctly and then display nothing. `expand` and
    # `block_execution` matter for the same reason — they are node output, not
    # node results, and resolving refs has no business discarding them.
    return NodeOutput(
        *resolved,
        ui=getattr(node_output, "ui", None),
        expand=getattr(node_output, "expand", None),
        block_execution=getattr(node_output, "block_execution", None),
    )


# --------------------------------------------------------------------------- #
# Provider registry — the seam the overlay attaches to.
# --------------------------------------------------------------------------- #
class Providers:
    def __init__(self) -> None:
        self.execution_backend: ExecutionBackend = InProcessExecutionBackend()
        self.ctx_provider: CtxProvider = InProcessCtxProvider()
        self.ops_provider: OpsProvider = InProcessOps()
        self.ref_resolver_factory: Callable[[], RefResolver] = InProcessRefResolver
        self._overlay_name: Optional[str] = None

    # Overlay entry points -------------------------------------------------- #
    def register_execution_backend(self, impl: ExecutionBackend) -> None:
        logger.info("SDK: execution backend -> %s", type(impl).__name__)
        self.execution_backend = impl

    def register_ctx_provider(self, impl: CtxProvider) -> None:
        logger.info("SDK: ctx provider -> %s", type(impl).__name__)
        self.ctx_provider = impl

    def register_ops_provider(self, impl: OpsProvider) -> None:
        logger.info("SDK: ops provider -> %s", type(impl).__name__)
        self.ops_provider = impl

    def register_ref_resolver_factory(self, factory: Callable[[], RefResolver]) -> None:
        logger.info("SDK: ref resolver -> %s", getattr(factory, "__name__", factory))
        self.ref_resolver_factory = factory

    @property
    def overlay_active(self) -> bool:
        return self._overlay_name is not None


providers = Providers()


# --------------------------------------------------------------------------- #
# Overlay loader — the "sidecar" attach point. OSS core calls load_overlay()
# once at startup; if COMFY_OVERLAY_MODULE is unset it is a no-op and behavior
# is pure in-process. The overlay module (proprietary, separate repo) exposes
# ``register(providers)`` and installs its implementations. Mirrors how
# load_custom_node imports by file path.
# --------------------------------------------------------------------------- #
def load_overlay(spec: Optional[str] = None) -> bool:
    spec = spec if spec is not None else os.environ.get(OVERLAY_ENV)
    if not spec:
        return False

    import importlib
    import importlib.util
    import sys

    module = None
    if os.path.exists(spec):
        norm = os.path.normpath(spec)
        if os.path.isdir(norm):
            # Package directory: put its parent on sys.path and import by name
            # so intra-package relative imports resolve normally.
            pkg_name = os.path.basename(norm)
            parent = os.path.dirname(norm)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            module = importlib.import_module(pkg_name)
        else:
            # Single .py file.
            modspec = importlib.util.spec_from_file_location("comfy_overlay", norm)
            if modspec and modspec.loader:
                module = importlib.util.module_from_spec(modspec)
                sys.modules["comfy_overlay"] = module
                modspec.loader.exec_module(module)
    else:
        module = importlib.import_module(spec)  # importable module name

    if module is None:
        logger.error("SDK overlay %r could not be loaded", spec)
        return False

    register = getattr(module, "register", None)
    if not callable(register):
        logger.error("SDK overlay %r has no register(providers) entrypoint", spec)
        return False

    register(providers)
    providers._overlay_name = getattr(module, "__name__", spec)
    logger.info("SDK overlay loaded: %s", providers._overlay_name)
    return True
