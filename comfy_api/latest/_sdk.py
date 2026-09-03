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

Layout
------
The file reads top-down as contract, then seam, then default implementation:

* **Refs** — opaque typed handles (``ImageRef``, ``LatentRef``, ``ModelRef``).
* **Engine-object handles** — MODEL/CLIP/VAE/CONDITIONING/GUIDER, which keep
  the original API's natural shape.
* **Runtime** — the per-execution binding a ref resolves through.
* **Provider interfaces** — ``RefResolver``, ``ExecutionBackend``,
  ``CtxProvider``: the three things an overlay may replace.
* **ctx domains** — the brokered side-effect surface (assets, output, graph…).
* **In-process defaults** — the OSS implementation of everything above,
  including ``InProcessOps``, the dispatch table behind ``Ref.op(name, ...)``.
* **Marshaling, provider registry, overlay loader** — how a node's heavy
  inputs become refs, and how an overlay attaches at startup.

Operations are addressed by string name (``"vae.decode"``) because that call
has to survive an out-of-process hop under the overlay; the typed ``Ref``
methods above are the interface node authors actually write against.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import re
import sys
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from ._weight_cache import WeightCache
from ._profiling import InProcessProfiling
from ._preview_override import InProcessPreviewOverride
from ._anima import InProcessAnima
from ._cloud_media import InProcessImgBB, InProcessLuma, InProcessSenseNova
from ._civitai import InProcessCivitai
from ._ollama import InProcessOllama
from ._llama_cpp import InProcessLlamaCpp

if TYPE_CHECKING:  # keep this module import-safe / torch-free at import time
    pass


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

    @classmethod
    def _wrap(cls, ref: "Ref") -> "Ref":
        """Re-type a handle. The base keeps the resolver's kind; a subclass that
        declares KIND asserts its own."""
        return cls(kind=getattr(cls, "KIND", None) or ref.kind, id=ref.id)

    async def op(self, name: str, **params: Any) -> Any:
        """Run a named operation on this handle.

        The operation vocabulary is data rather than API surface, so a pack can
        reach a capability this class knows nothing about — and wrap it in its
        own typed accessor — without core growing a method for it. Subclasses
        narrow the return type where it is always the same kind.
        """
        return await current_runtime().ops.apply(name, self, params)

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

    async def describe(self, max_value_chars: int = 32768) -> dict[str, Any]:
        """Return a bounded, inert description of this opaque value.

        The host projects only canonical kind/type, collection length, tensor
        shape, and a short redacted summary.  It never calls arbitrary object
        ``repr``, iteration, properties, or methods, and never exposes model
        weights, tensor values, paths, or the underlying host object.
        """
        return await current_runtime().ops.apply(
            "ref.describe", self, {"max_value_chars": max_value_chars})


class _TypedRef(Ref):
    KIND: str = "ANY"

    @classmethod
    def _wrap(cls, ref: Ref) -> "Ref":
        return cls(kind=cls.KIND, id=ref.id)


class ClosureRef(_TypedRef):
    """Handle to a retained node closure (D21).

    A pack function the host invokes at a declared sampling phase, for the
    rest of the prompt. The handle is what a node passes on — typically to
    ``ModelRef.patch("attach_closure", ...)`` — and carries no authority by
    itself: the host resolves it against the registry entry owning both the
    closure id and its validated captures, and releases both together at the
    prompt boundary.
    """

    KIND = "CLOSURE"

    async def attach_model(self, model: "ModelRef") -> "ModelRef":
        """Attach this closure to its declared canonical model phase.

        The host validates the closure kind, clones the model, and installs
        ComfyUI's matching pre- or post-CFG hook; the pack's function remains
        in its sandbox.
        """
        return await current_runtime().ctx.closures.attach_model(self, model)

    async def wrap_sampler(
        self, sampler: "SamplerRef", *,
        start_percent: Optional[float] = None,
        end_percent: Optional[float] = None,
    ) -> "SamplerRef":
        """Wrap a sampler's model calls with a ``model_sigma`` closure."""
        return await current_runtime().ctx.closures.attach_sampler(
            self,
            sampler,
            start_percent=start_percent,
            end_percent=end_percent,
        )

    async def as_latent_operation(self) -> "LatentOperationRef":
        """Expose a ``latent_operation`` closure as LATENT_OPERATION."""
        return await current_runtime().ctx.closures.create_latent_operation(
            self)

    async def as_sampler(self) -> "SamplerRef":
        """Expose a ``custom_sampler`` closure as a host-owned SAMPLER.

        The pack closure owns only the integration loop. During one sampling
        invocation the host supplies a narrow broker for denoise, noise,
        preview, and model-schedule projections; the broker is dead as soon as
        that invocation returns.
        """
        return await current_runtime().ctx.closures.create_sampler(self)

    async def attach_model_clip(
        self, model: "ModelRef", clip: "ClipRef",
    ) -> tuple["ModelRef", "ClipRef"]:
        """Attach one future-CLIP closure to a matching MODEL/CLIP pair.

        The operation is deliberately atomic: the host selects the canonical
        model family, clones both values, and installs matching typed markers
        so a doubled CLIP representation can never be paired with an ordinary
        model (or vice versa).
        """
        return await current_runtime().ctx.closures.attach_model_clip(
            self, model, clip)


class LatentOperationRef(_TypedRef):
    """Handle to a host-owned LATENT_OPERATION callable."""

    KIND = "LATENT_OPERATION"


class ValueRef(_TypedRef):
    """A handle whose value is buffer-safe structured data."""

    KIND = "VALUE"

    async def value(self) -> Any:
        """Read a dict/list of tensors and JSON scalars."""
        return await current_runtime().refs.resolve(self)

    @classmethod
    async def from_value(cls, v: Any) -> "Ref":
        return cls._wrap(await current_runtime().refs.create(cls.KIND, v))


class InterpolationStatesRef(_TypedRef):
    """A frame-interpolation skip policy projected from another pack."""

    KIND = "INTERPOLATION_STATES"

    async def skip_mask(self, pair_count: int) -> list[bool]:
        """Return which source-frame pairs must not be interpolated."""
        return await current_runtime().ops.apply(
            "interpolation_states.skip_mask", self,
            {"pair_count": pair_count},
        )


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

    # Convenience wrappers over the built-in image primitives.
    async def invert(self) -> "ImageRef":
        return await self.op("invert")

    async def scale(self, factor: float) -> "ImageRef":
        return await self.op("scale", factor=factor)

    async def rgb(self) -> "ImageRef":
        """Return the first three channels of an image as an opaque ref."""
        return await self.op("image.rgb")

    async def to_device(self, device: str = "auto") -> "ImageRef":
        """Clone this image onto a named ComfyUI-managed device.

        The device lookup and tensor move happen on the trusted plane.  Guests
        choose only ``auto``, ``cpu``, or ``gpu`` and never receive a device
        object or a raw tensor.
        """
        return await self.op("image.to_device", device=str(device))

    async def spatial_shape(self) -> tuple[int, int]:
        """Return image height and width without exposing its pixel buffer."""
        result = await self.op("image.spatial_shape")
        return int(result[0]), int(result[1])

    async def batch_size(self) -> int:
        """Return the number of images without exposing their pixel buffers."""
        return int(await self.op("image.batch_size"))

    async def select_batch(self, indices: list[int]) -> "ImageRef":
        """Select an ordered, bounded set of images from a BHWC batch."""
        return await self.op("image.select_batch", indices=list(indices))


class MaskRef(TensorRef):
    KIND = "MASK"

    async def grow(
        self, amount: int, tapered_corners: bool = False,
    ) -> "MaskRef":
        """Dilate or erode a mask through core's canonical morphology node."""
        return await current_runtime().ops.apply(
            "mask.grow", self, {
                "amount": int(amount),
                "tapered_corners": bool(tapered_corners),
            })


class LatentRef(ValueRef):
    KIND = "LATENT"

    @classmethod
    async def empty(
        cls, width: int, height: int, batch_size: int = 1,
        channels: int = 4,
        spatial_downscale_ratio: Optional[int] = None,
    ) -> "LatentRef":
        """Create a bounded zero latent without granting raw tensor access."""
        return await current_runtime().ops.apply(
            "latent.empty", None, {
                "width": int(width),
                "height": int(height),
                "batch_size": int(batch_size),
                "channels": int(channels),
                "spatial_downscale_ratio": spatial_downscale_ratio,
            })

    async def repeat_batch(self, amount: int) -> "LatentRef":
        """Repeat a latent through core's canonical batch operation."""
        return await current_runtime().ops.apply(
            "latent.repeat_batch", self, {"amount": int(amount)})

    async def noise_mask(self) -> Optional["MaskRef"]:
        """Return the latent's optional noise mask as an opaque mask ref."""
        return await current_runtime().ops.apply(
            "latent.noise_mask", self, {})

    async def spatial_shape(self) -> tuple[int, int]:
        """Return the latent sample height and width without exposing buffers."""
        result = await current_runtime().ops.apply(
            "latent.spatial_shape", self, {})
        return int(result[0]), int(result[1])

    async def resize(
        self, width: int, height: int, method: str = "bilinear",
    ) -> "LatentRef":
        """Resize latent spatial cells with ComfyUI's canonical interpolators."""
        return await current_runtime().ops.apply(
            "latent.resize", self, {
                "width": int(width),
                "height": int(height),
                "method": str(method),
            })

    async def random_noise(
        self, seed: int, source: str = "cpu", batch_size: Optional[int] = None,
    ) -> TensorRef:
        """Generate latent-shaped noise with a host-owned CPU/GPU RNG.

        RNG placement is an engine concern, especially for CUDA's distinct
        sequence. Higher-level variation mixing remains node-pack code.
        """
        return await current_runtime().ops.apply(
            "latent.random_noise", self, {
                "seed": int(seed),
                "source": str(source),
                "batch_size": batch_size,
            })

    async def composite(
        self, source: "LatentRef", *, x: int = 0, y: int = 0,
        resize_source: bool = False, mask: Optional["MaskRef"] = None,
    ) -> "LatentRef":
        """Composite a source latent using core's bounded latent operation."""
        return await current_runtime().ops.apply(
            "latent.composite", self, {
                "source": source,
                "x": int(x),
                "y": int(y),
                "resize_source": bool(resize_source),
                "mask": mask,
            })

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

    async def zero_out(self) -> "CondRef":
        """Zero embeddings while preserving the conditioning structure."""
        return await current_runtime().ops.apply("cond.zero_out", self, {})

    async def with_timestep_range(
        self, start: float, end: float,
    ) -> "CondRef":
        """Clone conditioning with a normalized sampling-percent range."""
        return await current_runtime().ops.apply(
            "cond.with_timestep_range", self, {
                "start": float(start),
                "end": float(end),
            })

    async def with_metadata(
        self, *, width: Optional[int] = None,
        height: Optional[int] = None,
        crop_w: Optional[int] = None,
        crop_h: Optional[int] = None,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> "CondRef":
        """Attach closed, scalar micro-conditioning metadata.

        This deliberately exposes only the six conventional SDXL-style size
        fields. Embeddings and arbitrary conditioning dictionaries remain
        host-owned.
        """
        return await current_runtime().ops.apply(
            "cond.with_metadata", self, {
                "width": width,
                "height": height,
                "crop_w": crop_w,
                "crop_h": crop_h,
                "target_width": target_width,
                "target_height": target_height,
            })

    async def has_spatial_metadata(self) -> bool:
        """Whether tile-relative conditioning must be cropped per image."""
        return bool(await current_runtime().ops.apply(
            "cond.has_spatial_metadata", self, {}))

    async def with_mask(
        self, mask: MaskRef, strength: float = 1.0,
        set_area_to_bounds: bool = False,
    ) -> "CondRef":
        """Attach core conditioning-mask metadata without exposing tensors."""
        return await current_runtime().ops.apply(
            "cond.with_mask", self, {
                "mask": mask,
                "strength": float(strength),
                "set_area_to_bounds": bool(set_area_to_bounds),
            })

    async def with_clip_vision_output(
        self, output: "ClipVisionOutputRef",
    ) -> "CondRef":
        """Attach one opaque CLIP-vision result to every conditioning row.

        The vision features remain host-owned.  This is the typed equivalent
        of setting ComfyUI's conventional ``clip_vision_output`` conditioning
        metadata key; prompt/layout policy stays with the calling node.
        """
        return await current_runtime().ops.apply(
            "cond.with_clip_vision_output", self, {"output": output})

    async def with_concat_latent(
        self, model: "ModelRef", latent: "LatentRef",
        extra_latent: Optional["LatentRef"] = None,
    ) -> "CondRef":
        """Attach model-formatted latent ``c_concat`` conditioning.

        This is the small reusable operation used by inpainting and layered
        diffusion models.  The guest chooses opaque latents; model-specific
        latent-format conversion stays in the trusted process.
        """
        return await current_runtime().ops.apply(
            "cond.with_concat_latent", self, {
                "model": model,
                "latent": latent,
                "extra_latent": extra_latent,
            })

    async def spatial_crop(
        self, *, x: int, y: int, width: int, height: int,
        source_width: int, source_height: int,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> "CondRef":
        """Crop 2D spatial conditioning to a latent-space window.

        Embeddings stay unchanged. Area prompts, masks, GLIGEN regions, and
        host-owned ControlNet/T2I hints are intersected with the requested
        window so tiled samplers can remain ordinary pack-side orchestration.
        Coordinates and dimensions use latent pixels.
        """
        return await current_runtime().ops.apply(
            "cond.spatial_crop", self, {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "source_width": source_width,
                "source_height": source_height,
                "target_width": target_width,
                "target_height": target_height,
            })


class GuiderRef(_TypedRef):
    """Opaque handle to a host-owned sampling guider."""

    KIND = "GUIDER"

    async def spatial_crop_inputs(
        self, *, regions: list[tuple[int, int, int, int]],
        source_width: int, source_height: int,
        target_width: int, target_height: int,
    ) -> "GuiderRef":
        """Clone this guider and crop model-owned spatial inputs for tiles.

        ``regions`` are pixel-space ``(left, top, right, bottom)`` rectangles
        on the source canvas; each result is resized to the target tile size.
        Only model patches that explicitly implement the core spatial-input
        protocol are changed; the guider policy stays opaque and otherwise
        unchanged.
        """
        return await current_runtime().ops.apply(
            "sampling.spatial_crop_inputs", self, {
                "regions": regions,
                "source_width": source_width,
                "source_height": source_height,
                "target_width": target_width,
                "target_height": target_height,
            })


class SamplerRef(_TypedRef):
    """Opaque handle to a host-owned sampler."""

    KIND = "SAMPLER"

    @classmethod
    async def named(
        cls, name: str, *, eta: Optional[float] = None,
        ge_gamma: Optional[float] = None,
    ) -> "SamplerRef":
        """Select a core sampler with its small, validated option set."""
        return await current_runtime().ops.apply(
            "sampler.named", None, {
                "name": str(name),
                "eta": eta,
                "ge_gamma": ge_gamma,
            })

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


class SigmasRef(_TypedRef):
    """Opaque one-dimensional host-owned sampling schedule."""

    KIND = "SIGMAS"

    async def steps(self) -> int:
        """Return the number of sampling intervals in this schedule.

        This exposes one bounded scalar (`len(sigmas) - 1`), not the schedule
        tensor. Custom-sampler nodes need it to call the generic sampling
        service without inventing a step count or materializing SIGMAS.
        """
        return int(await current_runtime().ops.apply(
            "sigmas.steps", self, {}))

    async def value_at(self, index: int) -> float:
        """Return one finite scalar from a bounded sampling schedule."""
        return float(await current_runtime().ops.apply(
            "sigmas.value_at", self, {"index": int(index)}))


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

    @classmethod
    async def scaled_soft(
        cls, base_multiplier: float = 0.825,
        uncond_multiplier: float = 1.0,
    ) -> tuple["ControlNetWeightsRef", "TimestepKeyframeRef"]:
        result = await current_runtime().ops.apply(
            "advanced_control.scaled_soft_weights", None, {
                "base_multiplier": base_multiplier,
                "uncond_multiplier": uncond_multiplier,
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

    async def ground_image(
        self, image: ImageRef, conditioning: CondRef, *,
        threshold: float = 0.5, refine_iterations: int = 2,
        individual_masks: bool = True, max_detections: int = 64,
    ) -> tuple[MaskRef, list[list[dict[str, float]]]]:
        """Text-ground objects in an image with a compatible vision MODEL.

        This is the generic intent exposed by SAM3/SAM3.1: conditioning and
        pixels remain host-owned, while the caller receives bounded masks and
        box data. Prompt construction and any layout policy stay pack-side.
        """
        result = await current_runtime().ops.apply(
            "model.ground_image", self, {
                "image": image,
                "conditioning": conditioning,
                "threshold": float(threshold),
                "refine_iterations": int(refine_iterations),
                "individual_masks": individual_masks,
                "max_detections": int(max_detections),
            })
        return result[0], result[1]

    async def spatial_crop_inputs(
        self, *, regions: list[tuple[int, int, int, int]],
        source_width: int, source_height: int,
        target_width: int, target_height: int,
    ) -> "ModelRef":
        """Clone the model with model-owned spatial inputs cropped to tiles.

        This is a small tiling primitive, not an upscaler implementation. The
        caller owns tile selection, target size, and orchestration; core only
        asks each model patch that declares spatial-input support to crop its
        own data.
        """
        return await current_runtime().ops.apply(
            "sampling.spatial_crop_inputs", self, {
                "regions": regions,
                "source_width": source_width,
                "source_height": source_height,
                "target_width": target_width,
                "target_height": target_height,
            })

    async def transforms(self) -> list[dict]:
        """Return the transforms supported by the active host."""
        return await current_runtime().ops.apply("model.transforms", self, {})

    async def latent_scale_factor(self) -> float:
        return float(await current_runtime().ops.apply(
            "model.latent_scale_factor", self, {}))

    async def is_flow(self) -> bool:
        """Whether the model uses ComfyUI's FLOW model family."""
        return bool(await current_runtime().ops.apply(
            "model.is_flow", self, {}))

    async def family(self) -> str:
        """Return the model's canonical base family, or ``unknown``."""
        return str(await current_runtime().ops.apply(
            "model.family", self, {}))

    async def unet_context_dim(self) -> Optional[int]:
        """Return a model's scalar UNet context dimension when published."""
        result = await current_runtime().ops.apply(
            "model.unet_context_dim", self, {})
        return None if result is None else int(result)

    async def is_zero_terminal_snr(self) -> bool:
        """Whether the model's sampling schedule uses zero terminal SNR."""
        return bool(await current_runtime().ops.apply(
            "model.is_zero_terminal_snr", self, {}))

    async def sigma_for_percent(
        self, percent: float, actual_endpoints: bool = False,
    ) -> float:
        """Project one sampling percentage through the model's schedule."""
        return float(await current_runtime().ops.apply(
            "model.sigma_for_percent", self, {
                "percent": float(percent),
                "actual_endpoints": bool(actual_endpoints),
            }))

    async def sampling_sigma_delta(
        self, *, steps: int, sampler_name: str, scheduler: str,
        start_step: int, end_step: int, denoise: float = 1.0,
        sigma_schedule: Optional[dict] = None,
    ) -> float:
        """Return one bounded scheduler delta in latent-value units."""
        return float(await current_runtime().ops.apply(
            "model.sampling_sigma_delta", self, {
                "steps": int(steps),
                "sampler_name": str(sampler_name),
                "scheduler": str(scheduler),
                "start_step": int(start_step),
                "end_step": int(end_step),
                "denoise": float(denoise),
                "sigma_schedule": sigma_schedule,
            }))

    async def scheduled_cfg_guider(
        self, positive: "CondRef", negative: "CondRef", cfg: float,
        start_percent: float = 0.0, end_percent: float = 1.0, *,
        bounds: Optional[dict] = None,
    ) -> "GuiderRef":
        return await current_runtime().ops.apply(
            "guider.scheduled_cfg", self, {
                "positive": positive,
                "negative": negative,
                "cfg": cfg,
                "start_percent": start_percent,
                "end_percent": end_percent,
                "bounds": bounds,
            })

    async def lora_weight_differences(
        self, original: "ModelRef", include_bias: bool = False,
    ) -> "WeightDiffCursorRef":
        return await current_runtime().ops.apply(
            "lora.weight_differences", self, {
                "original": original,
                "include_bias": bool(include_bias),
            })

    async def apply_lora(
        self, asset: "AssetRef", clip: Optional["ClipRef"],
        strength_model: float, strength_clip: float,
    ) -> tuple["ModelRef", Optional["ClipRef"]]:
        """Apply one resolved LoRA without exposing model weights or paths.

        ``asset`` must have been resolved from the host's ``loras`` catalogue.
        The trusted implementation confines its path again before loading it.
        """
        result = await current_runtime().ops.apply(
            "model.apply_lora", self, {
                "asset": asset,
                "clip": clip,
                "strength_model": strength_model,
                "strength_clip": strength_clip,
            })
        return result[0], result[1]

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

    async def set_last_layer(self, stop_at_clip_layer: int) -> "ClipRef":
        """Clone this CLIP and stop encoding at a bounded hidden layer."""
        return await current_runtime().ops.apply(
            "clip.set_last_layer", self,
            {"stop_at_clip_layer": int(stop_at_clip_layer)},
        )

    async def with_attention_impl(self, mode: str) -> "ClipRef":
        """Clone this encoder with one host-registered attention function."""
        return await current_runtime().ops.apply(
            "clip.with_attention_impl", self, {"mode": str(mode)})

    async def describe_tokens(self, tokens: dict) -> dict:
        """Describe bounded token IDs without exposing tokenizer objects.

        The result mirrors the token component/chunk structure. Each entry is
        ``{"id": int, "text": str, "special": bool}``; token weights stay
        in the caller's original token dictionary.
        """
        return await current_runtime().ops.apply(
            "clip.describe_tokens", self, {"tokens": tokens})

    async def scale_attention_weights(
        self, *, clip_l: Optional[list[float]] = None,
        clip_g: Optional[list[float]] = None,
        t5xxl: Optional[list[float]] = None,
        query: bool = True, key: bool = True,
        value: bool = True, output: bool = True,
    ) -> "ClipRef":
        """Scale selected CLIP/T5 attention projection weights.

        The guest supplies only bounded per-layer numbers and four projection
        switches. State-dict discovery and patching stay on the trusted plane;
        neither weights nor arbitrary key patterns cross the boundary.
        """
        return await current_runtime().ops.apply(
            "clip.scale_attention_weights", self, {
                "clip_l": clip_l,
                "clip_g": clip_g,
                "t5xxl": t5xxl,
                "query": bool(query),
                "key": bool(key),
                "value": bool(value),
                "output": bool(output),
            })

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

    async def encode_token_weights_component(
        self, component: str, tokens: list,
    ) -> tuple[TensorRef, Optional[TensorRef]]:
        """Encode token-weight pairs with one named CLIP component.

        This is the narrow seam needed by prompt-weight algorithms which do
        their own embedding arithmetic.  The guest receives embeddings, never
        text-encoder modules or weights.  ``component`` is limited to the
        conventional ``l`` and ``g`` encoders used by SD1/SDXL.
        """
        result = await current_runtime().ops.apply(
            "clip.encode_token_weights_component", self,
            {"component": str(component), "tokens": tokens},
        )
        return result[0], result[1]

    async def encode(self, text: str) -> "CondRef":
        """The two steps above in one call, for the common case.

        Exactly what ``CLIPTextEncode`` does, and it saves a wire round trip.
        A convenience over the pair, not a replacement: anything that inspects
        or edits tokens uses ``tokenize`` + ``encode_from_tokens_scheduled``.
        """
        return await current_runtime().ops.apply("clip.encode", self,
                                                 {"text": text})

    async def generate_text(
        self, prompt: str, image: Optional[ImageRef] = None,
        video: Optional[ImageRef] = None,
        max_length: int = 256, do_sample: bool = False,
        temperature: float = 1.0, top_k: Optional[int] = 50,
        top_p: float = 0.95, min_p: float = 0.0,
        repetition_penalty: float = 1.0, seed: Optional[int] = None,
        presence_penalty: float = 0.0, thinking: bool = False,
        use_default_template: bool = True, num_beams: int = 1,
    ) -> str:
        """Generate bounded text with a canonical Comfy text encoder.

        Image-conditioned token dictionaries may contain host tensors, so the
        tokenize/generate/decode sequence stays one opaque operation. Prompt
        construction and higher-level captioning policy remain pack-side.
        """
        return str(await current_runtime().ops.apply(
            "clip.generate_text", self, {
                "prompt": str(prompt),
                "image": image,
                "video": video,
                "max_length": int(max_length),
                "do_sample": bool(do_sample),
                "temperature": float(temperature),
                "top_k": None if top_k is None else int(top_k),
                "top_p": float(top_p),
                "min_p": float(min_p),
                "repetition_penalty": float(repetition_penalty),
                "seed": seed,
                "presence_penalty": float(presence_penalty),
                "thinking": bool(thinking),
                "use_default_template": bool(use_default_template),
                "num_beams": int(num_beams),
            }))

    async def lora_weight_differences(
        self, original: "ClipRef", include_bias: bool = False,
    ) -> "WeightDiffCursorRef":
        return await current_runtime().ops.apply(
            "lora.weight_differences", self, {
                "original": original,
                "include_bias": bool(include_bias),
            })


class LlamaCppModelRef(_TypedRef):
    """Opaque vendor-owned llama.cpp chat/VLM session."""

    KIND = "LLAMA_CPP_MODEL"

    async def generate(
        self, system: str, prompt: str,
        image: Optional[ImageRef] = None,
        video: Optional[ImageRef] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        seed: int = 1,
    ) -> str:
        return str(await current_context().integrations.llama_cpp.generate(
            self,
            system=system,
            prompt=prompt,
            image=image,
            video=video,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        ))


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

    async def latent_layout(self) -> dict[str, Optional[int]]:
        """Return the VAE's bounded latent channel/compression metadata."""
        return dict(await current_runtime().ops.apply(
            "vae.latent_layout", self, {}))

    async def decode(self, latent: "LatentRef") -> "ImageRef":
        return await current_runtime().ops.apply("vae.decode", self,
                                                 {"latent": latent})

    async def decode_tensor(self, latent: "LatentRef") -> "TensorRef":
        """Decode to an opaque BHWC tensor without assuming RGB channels.

        This is the raw-tier counterpart to :meth:`decode`.  It exists for
        pack-owned post-processing of VAEs whose decoded representation is not
        yet an IMAGE (for example, a channel-packed upscale VAE).  The model
        and decode stay host-owned; reading the returned tensor still requires
        the ordinary ``raw`` capability.
        """
        return await current_runtime().ops.apply(
            "vae.decode_tensor", self, {"latent": latent})

    async def decode_tiled(
        self, latent: "LatentRef", tile_size: int = 512,
        overlap: int = 64, temporal_size: int = 64,
        temporal_overlap: int = 8,
    ) -> "ImageRef":
        """Decode through ComfyUI's bounded, pixel-sized tiled operation.

        This mirrors the intent of core's ``VAEDecodeTiled`` node: spatial and
        temporal compression are derived from the host VAE, so a guest never
        needs to assume a latent compression ratio. Model state and execution
        remain on the trusted plane; the guest receives only an image handle.
        """
        return await current_runtime().ops.apply("vae.decode_tiled", self, {
            "latent": latent,
            "tile_size": int(tile_size),
            "overlap": int(overlap),
            "temporal_size": int(temporal_size),
            "temporal_overlap": int(temporal_overlap),
        })

    async def decode_tensor_tiled(
        self, latent: "LatentRef", tile_size: int = 512,
        overlap: int = 64, temporal_size: int = 64,
        temporal_overlap: int = 8,
    ) -> "TensorRef":
        """Tiled :meth:`decode_tensor` with the canonical VAE tile options."""
        return await current_runtime().ops.apply(
            "vae.decode_tensor_tiled", self, {
                "latent": latent,
                "tile_size": int(tile_size),
                "overlap": int(overlap),
                "temporal_size": int(temporal_size),
                "temporal_overlap": int(temporal_overlap),
            })

    async def encode(self, image: "ImageRef") -> "LatentRef":
        """Mirrors ``vae.encode`` exactly. The caller owns any channel slicing.

        Pixels pass through untouched, matching core's ``VAEEncode``
        (``t = vae.encode(pixels)``). Slicing here would change results rather
        than shape — silently dropping alpha for a four-channel caller — and be
        invisible from the calling node's own source.
        """
        return await current_runtime().ops.apply("vae.encode", self,
                                                 {"image": image})

    async def encode_for_inpaint(
        self, image: "ImageRef", mask: "MaskRef", grow_mask_by: int = 6,
    ) -> "LatentRef":
        """Run core VAEEncodeForInpaint with a bounded mask grow amount."""
        return await current_runtime().ops.apply(
            "vae.encode_for_inpaint", self, {
                "image": image,
                "mask": mask,
                "grow_mask_by": int(grow_mask_by),
            })

    async def encode_inpaint_conditioning(
        self, image: "ImageRef", mask: "MaskRef",
        positive: "CondRef", negative: "CondRef",
        noise_mask: bool = True,
    ) -> tuple["CondRef", "CondRef", "LatentRef"]:
        """Run core InpaintModelConditioning without exposing model tensors."""
        result = await current_runtime().ops.apply(
            "vae.encode_inpaint_conditioning", self, {
                "image": image,
                "mask": mask,
                "positive": positive,
                "negative": negative,
                "noise_mask": bool(noise_mask),
            })
        return result[0], result[1], result[2]

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

    async def concat(
        self, other: "ClipVisionOutputRef",
    ) -> "ClipVisionOutputRef":
        """Concatenate opaque penultimate vision tokens along their token axis."""
        return await current_runtime().ops.apply(
            "clip_vision_output.concat", self, {"other": other})

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

    async def apply(
        self, positive: CondRef, negative: CondRef, image: ImageRef,
        strength: float = 1.0, start_percent: float = 0.0,
        end_percent: float = 1.0, vae: Optional[VaeRef] = None,
    ) -> tuple[CondRef, CondRef]:
        """Apply this ControlNet while all referenced data stays host-owned."""
        return await current_runtime().ops.apply(
            "controlnet.apply", self, {
                "positive": positive,
                "negative": negative,
                "image": image,
                "strength": strength,
                "start_percent": start_percent,
                "end_percent": end_percent,
                "vae": vae,
            })

    async def apply_advanced(
        self, positive: CondRef, negative: CondRef, image: ImageRef,
        strength: float = 1.0, start_percent: float = 0.0,
        end_percent: float = 1.0, vae: Optional[VaeRef] = None,
        mask: Optional[MaskRef] = None,
        timestep_keyframe: Optional[TimestepKeyframeRef] = None,
        weights: Optional[ControlNetWeightsRef] = None,
    ) -> tuple[CondRef, CondRef]:
        """Apply Advanced-ControlNet scheduling, weights, and effect masks."""
        return await current_runtime().ops.apply(
            "controlnet.apply_advanced", self, {
                "positive": positive,
                "negative": negative,
                "image": image,
                "strength": strength,
                "start_percent": start_percent,
                "end_percent": end_percent,
                "vae": vae,
                "mask": mask,
                "timestep_keyframe": timestep_keyframe,
                "weights": weights,
            })

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

    async def predict_mask(
        self, images: ImageRef, text: str,
        use_accelerator: bool = True,
    ) -> MaskRef:
        """Return native-resolution sigmoid CLIPSeg predictions.

        This is the narrow inference primitive for nodes that own their own
        thresholding and morphology.  Model weights remain host-side.
        """
        return await current_runtime().ops.apply(
            "clipseg.predict_mask", self, {
                "images": images,
                "text": str(text),
                "use_accelerator": bool(use_accelerator),
            })

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


class ImageClassifierRef(_TypedRef):
    KIND = "IMAGE_CLASSIFIER"

    async def classify(
        self, images: ImageRef, use_accelerator: bool = True,
        top_k: int = 5,
    ) -> list[list[dict[str, Any]]]:
        """Classify a host-side image batch and return bounded label scores."""
        return await current_runtime().ops.apply(
            "image_classifier.classify", self, {
                "images": images,
                "use_accelerator": bool(use_accelerator),
                "top_k": int(top_k),
            })

    async def predict_scores(
        self, images: ImageRef,
    ) -> "ClassifierScoresRef":
        """Run a multi-label classifier and retain its score matrix host-side."""
        return await current_runtime().ops.apply(
            "image_classifier.predict_scores", self, {"images": images})


class ClassifierScoresRef(_TypedRef):
    """Opaque bounded batch-by-class scores from an image classifier."""

    KIND = "CLASSIFIER_SCORES"

    async def shape(self) -> tuple[int, int]:
        result = await current_runtime().ops.apply(
            "classifier_scores.shape", self, {})
        return int(result[0]), int(result[1])

    async def select_above(
        self, batch_index: int, start: int, end: int, threshold: float,
        offset: int = 0, limit: int = 512,
    ) -> dict[str, Any]:
        """Page score/index pairs above a threshold in one class range."""
        return await current_runtime().ops.apply(
            "classifier_scores.select_above", self, {
                "batch_index": int(batch_index),
                "start": int(start),
                "end": int(end),
                "threshold": float(threshold),
                "offset": int(offset),
                "limit": int(limit),
            })


class SemanticSegmentationRef(_TypedRef):
    """Opaque fixed-architecture semantic segmentation model."""

    KIND = "SEMANTIC_SEGMENTATION_MODEL"

    async def mask(
        self, image: ImageRef, classes: list[int],
    ) -> MaskRef:
        """Return the union of selected semantic class IDs as a mask."""
        return await current_runtime().ops.apply(
            "semantic_segmentation.mask", self, {
                "image": image,
                "classes": list(classes),
            })






class ObjectDetectorRef(_TypedRef):
    KIND = "OBJECT_DETECTOR"

    async def detect(
        self, image: ImageRef, threshold: float = 0.5,
        class_name: str = "all", max_detections: int = 100,
    ) -> list[list[dict[str, Any]]]:
        return await current_runtime().ops.apply(
            "object_detector.detect", self, {
                "image": image,
                "threshold": float(threshold),
                "class_name": str(class_name),
                "max_detections": int(max_detections),
            })


class ImagePreprocessorRef(_TypedRef):
    """Opaque host-created image preprocessor with one bounded operation."""

    KIND = "IMAGE_PREPROCESSOR"

    async def apply(
        self, image: ImageRef, mask: Optional[MaskRef] = None,
    ) -> ImageRef:
        return await current_runtime().ops.apply(
            "image_preprocessor.apply", self, {
                "image": image,
                "mask": mask,
            })


class InpaintModelRef(_TypedRef):
    """Opaque prompt-free image inpainting model."""

    KIND = "INPAINT_MODEL"

    async def inpaint(
        self, image: ImageRef, mask: MaskRef,
    ) -> ImageRef:
        """Fill the masked image region while keeping model weights host-side."""
        return await current_runtime().ops.apply(
            "inpaint_model.inpaint", self, {
                "image": image,
                "mask": mask,
            })


class BackgroundRemovalModelRef(_TypedRef):
    """Opaque ComfyUI background-removal model."""

    KIND = "BACKGROUND_REMOVAL_MODEL"

    async def mask(self, image: ImageRef) -> MaskRef:
        """Generate a foreground alpha mask through core's canonical model."""
        return await current_runtime().ops.apply(
            "background_removal.mask", self, {"image": image})


class BrushNetRef(_TypedRef):
    """Opaque BrushNet weights loaded by the canonical host extension.

    The pack keeps ownership of its pipeline dictionary and orchestration.  A
    guest can only ask the host to apply the already-loaded model to typed
    inputs; neither the live BrushNet object nor its tensors cross the wire.
    """

    KIND = "BRUSHNET_MODEL"

    async def apply(
        self, model: ModelRef, vae: VaeRef, image: ImageRef, mask: MaskRef,
        positive: CondRef, negative: CondRef, scale: float = 1.0,
        start_step: int = 0, end_step: int = 10000,
    ) -> tuple[ModelRef, CondRef, CondRef, LatentRef]:
        return await current_runtime().ops.apply(
            "brushnet.apply", self, {
                "model": model,
                "vae": vae,
                "image": image,
                "mask": mask,
                "positive": positive,
                "negative": negative,
                "scale": float(scale),
                "start_step": int(start_step),
                "end_step": int(end_step),
            })


class PowerPaintRef(_TypedRef):
    """Opaque PowerPaint model and token-extended CLIP pipeline."""

    KIND = "POWERPAINT_MODEL"

    async def apply(
        self, model: ModelRef, vae: VaeRef, image: ImageRef, mask: MaskRef,
        positive: CondRef, negative: CondRef, fitting: float = 1.0,
        function: str = "text guided", scale: float = 1.0,
        start_step: int = 0, end_step: int = 10000,
        save_memory: str = "none",
    ) -> tuple[ModelRef, CondRef, CondRef, LatentRef]:
        return await current_runtime().ops.apply(
            "powerpaint.apply", self, {
                "model": model,
                "vae": vae,
                "image": image,
                "mask": mask,
                "positive": positive,
                "negative": negative,
                "fitting": float(fitting),
                "function": str(function),
                "scale": float(scale),
                "start_step": int(start_step),
                "end_step": int(end_step),
                "save_memory": str(save_memory),
            })


class SamModelRef(_TypedRef):
    KIND = "SAM_MODEL"

    async def segment(
        self,
        image: ImageRef,
        boxes: list[Optional[list[float]]],
        point_coords: Optional[list[list[list[float]]]] = None,
        point_labels: Optional[list[list[int]]] = None,
        multimask_output: bool = True,
    ) -> tuple[MaskRef, list[list[float]]]:
        """Segment one host-side image from bounded boxes and point hints.

        The returned mask tensor is QxMxHxW, where Q is the query count and M
        is the number of masks per query. Model weights and predictor objects
        never enter the guest.
        """
        result = await current_runtime().ops.apply(
            "sam.segment", self, {
                "image": image,
                "boxes": boxes,
                "point_coords": point_coords,
                "point_labels": point_labels,
                "multimask_output": bool(multimask_output),
            })
        return result[0], result[1]

    async def segment_video(
        self, frames: ImageRef, boxes: list[list[float]],
    ) -> MaskRef:
        """Propagate frame-zero boxes through a host-side SAM2 video batch.

        The returned mask logits are QxFxHxW. Model state and predictor
        internals remain on the trusted plane; callers own thresholding and
        conversion into their pack-specific segment representation.
        """
        return await current_runtime().ops.apply(
            "sam.segment_video", self, {
                "frames": frames,
                "boxes": boxes,
            })


class UpscaleModelRef(_TypedRef):
    KIND = "UPSCALE_MODEL"

    async def upscale(
        self, images: ImageRef, per_batch: int = 16,
        downscale_ratio: float = 1.0, downscale_method: str = "lanczos",
        precision: str = "float32", tile_size: Optional[int] = None,
        channels_last: bool = False,
    ) -> ImageRef:
        return await current_runtime().ops.apply(
            "upscale_model.upscale", self, {
                "images": images,
                "per_batch": int(per_batch),
                "downscale_ratio": float(downscale_ratio),
                "downscale_method": str(downscale_method),
                "precision": str(precision),
                "tile_size": None if tile_size is None else int(tile_size),
                "channels_last": bool(channels_last),
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


@dataclass(frozen=True)
class HuggingFaceWeight:
    """A public Hugging Face weight file required by a node.

    Nodes declare these in ``SDK_REQUIRED_WEIGHTS``. The secure host reviews
    the declaration from the sealed pack manifest and installs the file before
    ``execute`` runs. An ``on_demand`` declaration is instead an allowlisted
    conditional dependency which the node explicitly requests when selected.
    ``catalogue_name`` is the stable name to pass to model loaders; it is never
    a filesystem path.
    """

    repo_id: str
    filename: str
    folder: str
    revision: str = "main"
    sha256: Optional[str] = None
    on_demand: bool = False

    def __post_init__(self) -> None:
        repo_id = _InProcessModels._hf_repo_id(self.repo_id)
        filename, extension = _InProcessModels._hf_weight_filename(
            self.filename)
        revision = _InProcessModels._hf_revision(self.revision)
        sha256 = _InProcessModels._hf_sha256(self.sha256)
        if extension == ".onnx" and sha256 is None:
            raise ValueError("Hugging Face ONNX weights require a sha256 pin")
        if type(self.on_demand) is not bool:
            raise TypeError("Hugging Face weight on_demand must be a bool")
        if not isinstance(self.folder, str):
            raise TypeError("Hugging Face weight folder must be a string")
        if self.folder not in _InProcessModels._HF_WEIGHT_FOLDERS:
            raise ValueError(
                "Hugging Face weights must target a known model catalogue")
        object.__setattr__(self, "repo_id", repo_id)
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "sha256", sha256)

    @property
    def catalogue_name(self) -> str:
        return (
            f"huggingface/{self.repo_id}/{self.revision}/{self.filename}"
        )


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
    required_weights: tuple[HuggingFaceWeight, ...] = ()
    # Work-unit payload for out-of-process backends. ``refs`` means the node
    # explicitly consumes SDK handles; ``values`` means the backend may wrap
    # for transport and the guest must materialize those handles before
    # invoking the unchanged V2 body. In-process dispatch ignores the payload.
    node_module: str = ""
    inputs: Optional[dict] = None
    input_mode: str = "refs"
    prompt: Any = None
    extra_pnginfo: Any = None
    dynamic_prompt: Any = None
    method: str = "execute"

    def __post_init__(self) -> None:
        self.method = _normalize_v2_node_method(self.method)
        self.required_weights = tuple(self.required_weights or ())
        if not all(isinstance(item, HuggingFaceWeight)
                   for item in self.required_weights):
            raise TypeError(
                "required_weights must contain HuggingFaceWeight declarations")
        self.permissions = tuple(self.permissions or ())
        if (self.required_weights
                and "models.download" not in self.permissions):
            self.permissions += ("models.download",)
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
    async def exists(self, folder: str, name: str) -> bool: ...
    async def delete_input(self, name: str) -> bool: ...
    async def path(self, ref: AssetRef) -> str: ...
    async def list(
        self, folder: str, prefix: str = "", recursive: bool = True,
    ) -> list[str]: ...
    async def latest(
        self, folder: str, prefix: str = "", suffix: str = "",
    ) -> Optional[str]: ...
    async def size(self, ref: AssetRef) -> int: ...
    async def digest(
        self, ref: AssetRef, algorithm: str = "sha256",
    ) -> str: ...
    async def read_range(
        self, ref: AssetRef, offset: int = 0,
        length: int = 8 * 1024 * 1024,
    ) -> bytes: ...
    async def read_bytes(self, ref: AssetRef) -> bytes: ...
    async def load_state_dict(
        self, ref: AssetRef, return_metadata: bool = False,
    ) -> Any: ...
    async def load_image(self, ref: AssetRef) -> ImageRef: ...
    async def load_latent(self, ref: AssetRef) -> LatentRef: ...


class ProgressDomain(Protocol):
    async def update(self, value: float, total: float,
                     preview: Optional[ImageRef] = None) -> None: ...


class ScratchDomain(Protocol):
    async def dir(self) -> str: ...


class EventsDomain(Protocol):
    async def emit(self, event: str, data: dict) -> None: ...


class InteractionDomain(Protocol):
    async def request(
        self, kind: str, payload: Any, *, reuse_last: bool = False,
        remember: bool = False, timeout: float = 540.0,
    ) -> Any: ...


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
        save_metadata: bool = True,
        extra_metadata: Optional[dict[str, Any]] = None,
        a1111_parameters: Optional[str] = None,
        image_format: str = "png", quality: int = 95,
        filenames: Optional[list[str]] = None,
        lossless: bool = False, optimize: bool = False,
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
    async def write_text(
        self, text: str, filename: str, folder: str = "output",
        mode: str = "overwrite", insert_newline: bool = False,
    ) -> str: ...
    async def save_workflow_json(
        self, filename: str, mode: str = "new_only",
    ) -> str: ...
    async def save_latent(
        self, latent: LatentRef,
        filename_prefix: str = "latents/LatentSender",
        preview_method: str = "Latent2RGB-SDXL",
    ) -> dict: ...
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
        encoder_options: Optional[dict[str, Any]] = None,
        loop_count: int = 0, bit_depth: int = 8,
        save_output: bool = True, save_metadata: bool = True,
    ) -> dict: ...
    async def save_animation(
        self, images: ImageRef, fps: float = 8.0,
        filename_prefix: str = "animation/ComfyUI",
        format: str = "webp", loop_count: int = 0,
        lossless: bool = True, quality: int = 90,
        save_output: bool = True,
    ) -> dict: ...
    async def save_image_sequence(
        self, images: ImageRef,
        filename_prefix: str = "sequence/ComfyUI",
        format: str = "png", bit_depth: int = 8,
        save_output: bool = True,
    ) -> dict: ...


class GraphDomain(Protocol):
    async def current_node_id(self) -> str: ...
    async def input_label(
        self, input_name: str, default: str = "",
    ) -> str: ...
    async def expand_nodes(
        self, nodes: list[dict[str, Any]], outputs: list[dict[str, Any]],
    ) -> dict[str, Any]: ...
    async def expand_loop(
        self, flow: Any, values: list[Any],
    ) -> dict[str, Any]: ...
    async def widget_values(
        self, node_id: int | str = 0, node_title: str = "",
        node_name: str = "", linked_input: str = "any_input",
    ) -> dict[str, Any]: ...
    async def block(self, reason: Optional[str] = None) -> Any: ...


class ExecutionDomain(Protocol):
    async def interrupt(self) -> bool: ...


class CivitaiDomain(Protocol):
    """Bounded read-only projection of the Civitai public model API."""

    async def search_models(
        self, username: str, query: Optional[str] = None,
        limit: int = 20, nsfw: bool = False,
    ) -> dict[str, Any]: ...
    async def model_version(
        self, model_version_id: int,
    ) -> dict[str, Any]: ...
    async def model_version_by_hash(
        self, hash_value: str, refresh: bool = False,
    ) -> dict[str, Any]: ...


class OllamaDomain(Protocol):
    """Bounded Ollama vendor API; endpoint is loopback or an admin profile."""

    async def list_models(self, endpoint: str) -> list[str]: ...
    async def generate(
        self, endpoint: str, model: str, system: str, prompt: str,
        images: Optional[ImageRef] = None,
        context: Optional[list[int]] = None, think: bool = False,
        options: Optional[dict[str, Any]] = None, keep_alive: int = 5,
        keep_alive_unit: str = "minutes",
        format: str | dict[str, Any] = "",
        timeout_seconds: float = 600.0,
    ) -> dict[str, Any]: ...
    async def chat(
        self, endpoint: str, model: str,
        messages: list[dict[str, Any]], images: Optional[ImageRef] = None,
        think: bool = False, options: Optional[dict[str, Any]] = None,
        keep_alive: int = 5, keep_alive_unit: str = "minutes",
        format: str | dict[str, Any] = "", timeout_seconds: float = 600.0,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]: ...


class LlmDomain(Protocol):
    """Provider-neutral bounded chat and function-tool contract."""

    async def chat(
        self, provider: str, profile: str, model: str,
        messages: list[dict[str, Any]], *,
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.8, max_tokens: int = 512,
        thinking: bool = False,
        response_format: str | dict[str, Any] = "",
        timeout_seconds: float = 600.0,
        vendor_options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]: ...


class WebSearchDomain(Protocol):
    """Fixed-profile web search with bounded normalized results."""

    async def search(
        self, query: str, *, provider_profile: str = "duckduckgo",
        limit: int = 5,
        vendor_options: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, str]]: ...


class LumaDomain(Protocol):
    """Fixed-origin Luma Dream Machine generation jobs (D32)."""

    async def create_video(
        self, api_key: str, prompt: str, model: str, *,
        loop: bool = False, aspect_ratio: Optional[str] = None,
        duration: Optional[str] = None, resolution: str = "720p",
        keyframes: Optional[dict[str, Any]] = None,
        save: bool = True, filename: str = "",
    ) -> dict[str, Any]: ...
    async def upscale_video(
        self, api_key: str, generation_id: str, resolution: str, *,
        save: bool = True, filename: str = "",
    ) -> dict[str, Any]: ...
    async def add_audio(
        self, api_key: str, generation_id: str, prompt: str,
        negative_prompt: str, *, save: bool = True, filename: str = "",
    ) -> dict[str, Any]: ...
    async def create_image(
        self, api_key: str, prompt: str, model: str, *,
        aspect_ratio: str = "1:1",
        image_ref: Optional[list[dict[str, Any]]] = None,
        style_ref: Optional[list[dict[str, Any]]] = None,
        character_ref: Optional[dict[str, Any]] = None,
        modify_image_ref: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]: ...


class ImgBBDomain(Protocol):
    """Upload one host-owned image to the fixed ImgBB endpoint (D32)."""

    async def upload(
        self, api_key: str, image: ImageRef, *,
        expiration_seconds: Optional[int] = None,
    ) -> str: ...


class SenseNovaDomain(Protocol):
    """Fixed-origin SenseNova chat, vision, and image generation (D33)."""

    async def chat(
        self, text: str, system_prompt: str, model: str, *,
        temperature: float = 0.7, top_p: float = 1.0,
        max_tokens: int = 2048, timeout_seconds: int = 120,
    ) -> dict[str, ValueRef]: ...
    async def vision_url(
        self, image_url: str, prompt: str, system_prompt: str, model: str, *,
        temperature: float = 0.2, top_p: float = 1.0,
        max_tokens: int = 2048, timeout_seconds: int = 120,
    ) -> dict[str, ValueRef]: ...
    async def vision_image(
        self, image: ImageRef, prompt: str, system_prompt: str, model: str, *,
        temperature: float = 0.2, top_p: float = 1.0,
        max_tokens: int = 2048, timeout_seconds: int = 120,
    ) -> dict[str, ValueRef]: ...
    async def generate_image(
        self, prompt: str, model: str, size: str, *,
        timeout_seconds: int = 300,
    ) -> dict[str, Ref]: ...


class LlamaCppDomain(Protocol):
    """Bounded llama.cpp vendor adapter over managed GGUF weights."""

    async def load_chat_model(
        self, model_weight: str, mmproj_weight: Optional[str] = None, *,
        family: str = "qwen3_vl", device: str = "auto",
        context_length: int = 8192, batch_size: int = 512,
        gpu_layers: int = -1, image_max_tokens: int = 4096,
        top_k: int = 0, pool_size: int = 4_194_304,
        cache: bool = True,
    ) -> LlamaCppModelRef: ...
    async def generate(
        self, model: LlamaCppModelRef, system: str, prompt: str,
        image: Optional[ImageRef] = None,
        video: Optional[ImageRef] = None,
        max_tokens: int = 512, temperature: float = 0.7,
        top_p: float = 0.9, repetition_penalty: float = 1.0,
        seed: int = 1,
    ) -> str: ...


class WanVideoDomain(Protocol):
    """Bounded metadata for WanVideo vendor-owned opaque model handles."""

    async def transformer_dim(self, model: Ref) -> int: ...


class AnimaDomain(Protocol):
    """Vendor-specific Anima model adapters."""

    async def apply_lllite(
        self, model: ModelRef, weights: AssetRef, image: ImageRef, *,
        strength: float = 1.0, start_percent: float = 0.0,
        end_percent: float = 1.0, preserve_wrapper: bool = True,
    ) -> ModelRef: ...


class IntegrationsDomain(Protocol):
    """Vendor pass-throughs with vendor-shaped, less-stable contracts."""

    anima: AnimaDomain
    civitai: CivitaiDomain
    imgbb: ImgBBDomain
    llm: LlmDomain
    llama_cpp: LlamaCppDomain
    luma: LumaDomain
    ollama: OllamaDomain
    sensenova: SenseNovaDomain
    wanvideo: WanVideoDomain
    web: WebSearchDomain


class ModelsDomain(Protocol):
    async def download_huggingface_weights(
        self, repo_id: str, filename: str, folder: str,
        revision: str = "main", sha256: Optional[str] = None,
    ) -> str: ...
    async def list_diffusion_models(
        self, include_connectors: bool = False,
    ) -> list[str]: ...
    async def load_checkpoint(
        self, name: str, weight_dtype: str = "default",
        compute_dtype: str = "default", cublas_linear: bool = False,
        config_name: Optional[str] = None,
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
    async def load_gguf_text_encoders(
        self, names: Sequence[str], clip_type: str,
    ) -> ClipRef: ...
    async def list_controlnet(self) -> list[str]: ...
    async def load_controlnet(
        self, name: str, model: Optional[ModelRef] = None,
    ) -> ControlNetRef: ...
    async def load_advanced_controlnet(
        self, name: str, model: Optional[ModelRef] = None,
        timestep_keyframe: Optional[TimestepKeyframeRef] = None,
    ) -> ControlNetRef: ...
    async def load_controlnet_plusplus(
        self, name: str, control_type: str = "none",
    ) -> ControlNetRef: ...
    async def list_vae(self) -> list[str]: ...
    async def load_vae(
        self, name: str, device: str = "default",
        weight_dtype: str = "default",
    ) -> VaeRef: ...
    async def load_upscale_model(self, name: str) -> UpscaleModelRef: ...
    async def load_clip_vision(self, model: str) -> ClipVisionRef: ...
    async def load_text_encoder(
        self, model: str, model_type: str,
        device: str = "default",
    ) -> ClipRef: ...
    async def load_language_model(
        self, weights: list[str], family: str,
        device: str = "default", cache: bool = True,
    ) -> ClipRef: ...
    async def load_ipadapter(
        self, model: str, clip_vision: ClipVisionRef,
    ) -> Ref: ...
    async def load_brushnet(
        self, model: str, dtype: str = "float16",
    ) -> BrushNetRef: ...
    async def load_powerpaint(
        self, model: str, base_clip: str, powerpaint_clip: str,
        dtype: str = "float16",
    ) -> PowerPaintRef: ...
    async def load_clipseg(self, model: str) -> ClipSegRef: ...
    async def load_image_classifier(
        self, model: str, architecture: str, labels: list[str],
    ) -> ImageClassifierRef: ...
    async def load_onnx_image_classifier(
        self, model: str, input_layout: str = "NHWC",
        channel_order: str = "BGR", resize_mode: str = "fit_pad",
        input_scale: float = 255.0,
        pad_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        activation: str = "identity", resize_filter: str = "lanczos",
    ) -> ImageClassifierRef: ...
    async def load_segformer(
        self, model: str, variant: str, num_labels: int,
    ) -> SemanticSegmentationRef: ...
    async def load_inpaint_model(
        self, model: str, architecture: str = "big-lama",
    ) -> InpaintModelRef: ...
    async def load_background_removal_model(
        self, model: str,
    ) -> BackgroundRemovalModelRef: ...
    async def load_object_detector(self, model: str) -> ObjectDetectorRef: ...
    async def load_sam(
        self, model: str, architecture: str = "vit_b",
        device_mode: str = "AUTO",
    ) -> SamModelRef: ...
    async def generate_text(
        self, generator: str, input_text: str, max_new_tokens: int = 128,
        weight: Optional[str] = None,
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


class SystemDomain(Protocol):
    async def stats(self) -> dict[str, Any]: ...
    async def monitor(self) -> dict[str, Any]: ...


class ClosuresDomain(Protocol):
    async def retain(
        self, kind: str, fn: Callable, *, captures: Optional[dict] = None,
    ) -> ClosureRef: ...
    async def attach_model(
        self, closure: ClosureRef, model: ModelRef,
    ) -> ModelRef: ...
    async def attach_sampler(
        self, closure: ClosureRef, sampler: SamplerRef, *,
        start_percent: Optional[float] = None,
        end_percent: Optional[float] = None,
    ) -> SamplerRef: ...
    async def create_latent_operation(
        self, closure: ClosureRef,
    ) -> LatentOperationRef: ...
    async def create_sampler(
        self, closure: ClosureRef,
    ) -> SamplerRef: ...
    async def attach_model_clip(
        self, closure: ClosureRef, model: ModelRef, clip: ClipRef,
    ) -> tuple[ModelRef, ClipRef]: ...


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
    execution: ExecutionDomain
    integrations: IntegrationsDomain
    models: ModelsDomain
    profiling: ProfilingDomain
    preview_override: PreviewOverrideDomain
    system: SystemDomain
    closures: ClosuresDomain
    interact: InteractionDomain
    sample: Any
    unsample: Any
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
    _SCAN_MAX = 100000
    _DIGEST_CACHE_MAX = 256
    _DIGEST_CACHE: "OrderedDict[tuple[Any, ...], str]" = OrderedDict()
    _DIGEST_LOCK = threading.Lock()
    _IMAGE_FILE_MAX = 64 * 1024 * 1024
    _IMAGE_PIXELS_MAX = 67_108_864

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

    async def exists(self, folder: str, name: str) -> bool:
        import folder_paths

        standard_folders = {
            "input": folder_paths.get_input_directory,
            "output": folder_paths.get_output_directory,
            "temp": folder_paths.get_temp_directory,
        }
        if folder in standard_folders:
            return os.path.isfile(self._confined_path(
                standard_folders[folder](), name, folder))
        return any(
            os.path.isfile(self._confined_path(root, name, folder))
            for root in folder_paths.get_folder_paths(folder)
        )

    async def delete_input(self, name: str) -> bool:
        import folder_paths

        path = self._confined_path(
            folder_paths.get_input_directory(), name, "input")
        if not os.path.isfile(path):
            return False
        os.remove(path)
        return True

    async def path(self, ref: AssetRef) -> str:
        return await current_runtime().refs.resolve(ref)

    async def list(
        self, folder: str, prefix: str = "", recursive: bool = True,
    ) -> list[str]:
        import folder_paths

        managed_folders = {
            "input": folder_paths.get_input_directory,
            "output": folder_paths.get_output_directory,
            "temp": folder_paths.get_temp_directory,
        }
        if folder not in managed_folders:
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

        base = os.path.realpath(os.path.abspath(managed_folders[folder]()))
        directory = self._confined_path(base, prefix, folder)
        if not os.path.exists(directory):
            return []
        if not os.path.isdir(directory):
            raise NotADirectoryError(
                f"{folder} asset prefix {prefix!r} is not a directory")

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
                        f"{folder} asset catalogue exceeds "
                        f"{self._LIST_MAX} names")
            if not recursive:
                break
        return sorted(names)

    async def latest(
        self, folder: str, prefix: str = "", suffix: str = "",
    ) -> Optional[str]:
        """Newest logical file in one managed user-media directory.

        This exposes neither paths nor a general stat primitive.  It exists
        for nodes whose behavior is specifically "reuse the latest output".
        """
        import folder_paths

        roots = {
            "input": folder_paths.get_input_directory,
            "output": folder_paths.get_output_directory,
            "temp": folder_paths.get_temp_directory,
        }
        if folder not in roots:
            raise ValueError("latest is limited to input, output, and temp assets")
        base = os.path.realpath(os.path.abspath(roots[folder]()))
        logical_prefix = str(prefix or "").replace("\\", "/").lstrip("/")
        if logical_prefix.startswith(folder + "/"):
            logical_prefix = logical_prefix[len(folder) + 1:]
        if ("\x00" in logical_prefix
                or any(part == ".." for part in logical_prefix.split("/"))):
            raise ValueError("asset prefix escapes the managed directory")
        suffix_value = str(suffix or "")
        if (len(suffix_value) > 256 or "\x00" in suffix_value
                or "/" in suffix_value or "\\" in suffix_value):
            raise ValueError("asset suffix must be a filename suffix")

        newest: Optional[tuple[int, str]] = None
        examined = 0
        for root, directories, files in os.walk(base, followlinks=False):
            directories.sort()
            files.sort()
            for filename in files:
                examined += 1
                if examined > self._SCAN_MAX:
                    raise ValueError(
                        f"managed directory exceeds {self._SCAN_MAX} files")
                full = os.path.realpath(os.path.join(root, filename))
                try:
                    confined = os.path.commonpath((base, full)) == base
                except ValueError:
                    confined = False
                if not confined or not os.path.isfile(full):
                    continue
                logical = os.path.relpath(full, base).replace(os.sep, "/")
                if (not logical.startswith(logical_prefix)
                        or not logical.endswith(suffix_value)):
                    continue
                candidate = (os.stat(full).st_mtime_ns, logical)
                if newest is None or candidate > newest:
                    newest = candidate
        return newest[1] if newest is not None else None

    async def read_bytes(self, ref: AssetRef) -> bytes:
        path = await self.path(ref)
        with open(path, "rb") as file:
            return file.read()

    async def size(self, ref: AssetRef) -> int:
        path = await self.path(ref)
        return int(os.path.getsize(path))

    async def digest(
        self, ref: AssetRef, algorithm: str = "sha256",
    ) -> str:
        """Hash a managed asset without returning its path or loading it whole.

        The cache key is the opened file's identity, so a replacement or edit
        invalidates the entry.  Only SHA-256 is exposed initially: the point is
        stable model identity, not a general cryptography surface.
        """
        if algorithm != "sha256":
            raise ValueError("asset digest algorithm must be sha256")
        path = await self.path(ref)
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError("ASSET ref does not contain a managed file")

        def compute() -> str:
            import hashlib

            with open(path, "rb") as stream:
                before = os.fstat(stream.fileno())
                key = (
                    os.path.realpath(os.fspath(path)), before.st_dev,
                    before.st_ino, before.st_size, before.st_mtime_ns,
                    before.st_ctime_ns, algorithm,
                )
                with self._DIGEST_LOCK:
                    cached = self._DIGEST_CACHE.get(key)
                    if cached is not None:
                        self._DIGEST_CACHE.move_to_end(key)
                        return cached
                hasher = hashlib.sha256()
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    hasher.update(chunk)
                after = os.fstat(stream.fileno())
                if (
                    before.st_dev, before.st_ino, before.st_size,
                    before.st_mtime_ns, before.st_ctime_ns,
                ) != (
                    after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns, after.st_ctime_ns,
                ):
                    raise RuntimeError("asset changed while its digest was computed")
                value = hasher.hexdigest()
            with self._DIGEST_LOCK:
                self._DIGEST_CACHE[key] = value
                self._DIGEST_CACHE.move_to_end(key)
                while len(self._DIGEST_CACHE) > self._DIGEST_CACHE_MAX:
                    self._DIGEST_CACHE.popitem(last=False)
            return value

        return await asyncio.to_thread(compute)

    async def read_range(
        self, ref: AssetRef, offset: int = 0,
        length: int = 8 * 1024 * 1024,
    ) -> bytes:
        start = int(offset)
        count = int(length)
        if start != offset or start < 0:
            raise ValueError("asset range offset must be a non-negative integer")
        if count != length or not 1 <= count <= 16 * 1024 * 1024:
            raise ValueError("asset range length must be in [1, 16 MiB]")
        path = await self.path(ref)
        with open(path, "rb") as file:
            file.seek(start)
            return file.read(count)

    async def load_state_dict(
        self, ref: AssetRef, return_metadata: bool = False,
    ) -> Any:
        import comfy.utils

        path = await self.path(ref)
        return comfy.utils.load_torch_file(
            path, safe_load=True, return_metadata=bool(return_metadata))

    async def load_image(self, ref: AssetRef) -> ImageRef:
        import numpy as np
        import torch
        from PIL import Image, ImageOps

        path = await self.path(ref)
        if os.path.getsize(path) > self._IMAGE_FILE_MAX:
            raise ValueError("image asset exceeds the encoded size limit")

        def decode():
            with Image.open(path) as source:
                source.seek(0)
                image = ImageOps.exif_transpose(source)
                width, height = image.size
                if (
                    width < 1 or height < 1
                    or width * height > self._IMAGE_PIXELS_MAX
                ):
                    raise ValueError("image asset dimensions exceed the limit")
                rgb = image.convert("RGB")
                rgb.load()
                return np.asarray(rgb, dtype=np.float32).copy()

        array = await asyncio.to_thread(decode)
        pixels = torch.from_numpy(array).div_(255.0).unsqueeze(0)
        return ImageRef._wrap(await current_runtime().refs.create(
            "IMAGE", pixels))  # type: ignore[return-value]

    async def load_latent(self, ref: AssetRef) -> LatentRef:
        """Load ComfyUI's safetensors-backed ``.latent`` format.

        This deliberately does not accept pickle or image containers.  A
        legacy ``.latent.png`` may still be decoded by a sandboxed node, but
        no image/EXIF/ZIP parser is moved into the trusted plane for it.
        """
        import torch
        from safetensors.torch import load_file

        path = await self.path(ref)
        if not str(path).lower().endswith(".latent"):
            raise ValueError("latent assets must use the safe .latent format")
        value = load_file(path, device="cpu")
        if not isinstance(value, dict) or "latent_tensor" not in value:
            raise ValueError("latent asset has no latent_tensor")
        tensor = value["latent_tensor"]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim not in (4, 5):
            raise ValueError("latent_tensor must be a 4D or 5D tensor")
        multiplier = 1.0 if "latent_format_version_0" in value else 1.0 / 0.18215
        result = {"samples": tensor.float() * multiplier}
        return LatentRef._wrap(await current_runtime().refs.create(
            "LATENT", result))  # type: ignore[return-value]


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


# --------------------------------------------------------------------------- #
# Weight caches for optional capabilities. Each capability contributes an entry
# type (one loaded model plus its metadata) and a loader; the bounded LRU
# behavior around them is generic and lives in ``WeightCache``. These back
# specific model families rather than the core diffusion contract, and are the
# part of this module most likely to move out to the packs that need them.
# --------------------------------------------------------------------------- #
def _release_model_to_cpu(entry: Any) -> None:
    """Move a cached model off the accelerator as it leaves the cache."""
    with entry.lock:
        entry.model.to("cpu")


def _loader(name: str) -> Callable[..., Any]:
    """Resolve a weight loader by name at call time.

    The caches below are module singletons built at import, but a loader may be
    replaced afterwards — tests substitute one to avoid loading real weights.
    Binding the function object here would capture the original and silently
    ignore the substitution.
    """
    return lambda *args: globals()[name](*args)


@dataclass
class _TextGeneratorEntry:
    tokenizer: Any
    model: Any
    device: Any
    lock: threading.Lock = field(default_factory=threading.Lock)


def _load_fixed_text_generator(
    generator: str, weight_path: str,
) -> _TextGeneratorEntry:
    if generator != "superprompt-v1":
        raise ValueError(
            f"text generator {generator!r} is not in the trusted catalogue")
    if (not isinstance(weight_path, str)
            or not weight_path.lower().endswith(".safetensors")
            or not os.path.isfile(weight_path)):
        raise ValueError(
            "text generator 'superprompt-v1' requires a SafeTensors weight")
    try:
        import comfy.model_management
        from safetensors.torch import load_file
        from transformers import (
            T5Config,
            T5ForConditionalGeneration,
            T5TokenizerFast,
        )
    except ImportError as exc:
        raise RuntimeError(
            "text generator 'superprompt-v1' requires the transformers "
            "and safetensors packages") from exc

    # SuperPrompt is a fine-tuned flan-t5-small model. Construct its fixed
    # architecture here so repository config can never become executable or
    # policy-bearing input to the trusted process.
    config = T5Config(
        vocab_size=32128,
        d_model=512,
        d_kv=64,
        d_ff=1024,
        num_layers=8,
        num_decoder_layers=8,
        num_heads=6,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="gated-gelu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        tie_word_embeddings=False,
    )
    # Some Transformers releases normalize this legacy field back to True in
    # T5Config.__init__. SuperPrompt stores a distinct learned lm_head, so set
    # it explicitly before model construction to prevent destructive tying.
    config.tie_word_embeddings = False
    model = T5ForConditionalGeneration(config)
    state = load_file(weight_path, device="cpu")
    shared = state.get("shared.weight")
    if shared is None:
        raise ValueError("SuperPrompt weights have no shared embedding")
    # SafeTensors deliberately stores the shared T5 embedding only once.
    # Restore the two state-dict aliases before performing a strict load.
    state["encoder.embed_tokens.weight"] = shared
    state["decoder.embed_tokens.weight"] = shared
    model.load_state_dict(state, strict=True)

    tokenizer_root = os.path.realpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "comfy",
        "text_encoders", "t5_tokenizer"))
    tokenizer_file = os.path.join(tokenizer_root, "tokenizer.json")
    if not os.path.isfile(tokenizer_file):
        raise RuntimeError("ComfyUI's bundled T5 tokenizer is unavailable")
    tokenizer = T5TokenizerFast(
        tokenizer_file=tokenizer_file,
        model_max_length=512,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        )

    device = comfy.model_management.get_torch_device()
    model.to(device)
    model.eval()
    return _TextGeneratorEntry(tokenizer, model, device)


class _TextGeneratorCache:
    def __init__(self, max_entries: int = 1) -> None:
        if max_entries < 1:
            raise ValueError("text generator cache must hold at least one entry")
        self.max_entries = max_entries
        self._entries: dict[tuple[Any, ...], _TextGeneratorEntry] = {}
        self._lock = threading.Lock()
        self.loads = 0
        self.hits = 0
        self.evictions = 0

    @staticmethod
    def _key(generator: str, weight_path: str) -> tuple[Any, ...]:
        status = os.stat(weight_path)
        return (
            generator,
            os.path.realpath(weight_path),
            status.st_dev,
            status.st_ino,
            status.st_size,
            status.st_mtime_ns,
            status.st_ctime_ns,
        )

    def _entry(
        self, generator: str, weight_path: str,
    ) -> _TextGeneratorEntry:
        key = self._key(generator, weight_path)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self.hits += 1
                return entry
            entry = _load_fixed_text_generator(generator, weight_path)
            self.loads += 1
            while len(self._entries) >= self.max_entries:
                _, evicted = self._entries.popitem()
                self._release(evicted)
                self.evictions += 1
            self._entries[key] = entry
            return entry

    @staticmethod
    def _release(entry: _TextGeneratorEntry) -> None:
        with entry.lock:
            entry.model.to("cpu")

    def generate(
        self, generator: str, weight_path: str, input_text: str,
        max_new_tokens: int,
    ) -> str:
        import torch

        entry = self._entry(generator, weight_path)
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


@dataclass
class _InpaintModelEntry:
    model: Any
    architecture: str
    lock: threading.Lock = field(default_factory=threading.Lock)

    def bundle(self) -> dict[str, Any]:
        return {
            "secure_kind": f"image_inpaint.{self.architecture}",
            "model": self.model,
            "architecture": self.architecture,
            "lock": self.lock,
        }


def _load_inpaint_model_weight(
    path: str, architecture: str,
) -> _InpaintModelEntry:
    import torch
    import comfy.ops
    from comfy.ldm.lama import BigLamaGenerator
    from safetensors.torch import load_file

    if architecture != "big-lama":
        raise ValueError(
            f"inpaint architecture {architecture!r} is not supported")
    state = load_file(path, device="cpu")
    if not state:
        raise ValueError("Big-LaMa SafeTensors file contains no weights")
    if any(not key.startswith("generator.") for key in state):
        raise ValueError("Big-LaMa weights contain an unexpected key prefix")
    floating_dtypes = {
        value.dtype for value in state.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    }
    if floating_dtypes != {torch.float32}:
        raise ValueError("Big-LaMa weights must use float32")
    state = {
        key.removeprefix("generator."): value
        for key, value in state.items()
    }
    with torch.device("meta"):
        model = BigLamaGenerator(comfy.ops.disable_weight_init)
    model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    return _InpaintModelEntry(model=model, architecture=architecture)




_INPAINT_MODEL_CACHE = WeightCache(
    load=_loader("_load_inpaint_model_weight"),
    max_entries=1,
    release=_release_model_to_cpu,
)


@dataclass
class _ClipSegEntry:
    model: Any
    processor: Any
    lock: threading.Lock = field(default_factory=threading.Lock)

    def bundle(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "processor": self.processor,
            "lock": self.lock,
        }


def _load_clipseg_weight(path: str) -> _ClipSegEntry:
    """Build the one supported CLIPSeg architecture from one SafeTensors file.

    The architecture and image-processing configuration are trusted code, and
    the CLIP tokenizer vocabulary is bundled with ComfyUI.  No model config,
    tokenizer, processor, or executable file is fetched from the model repo.
    """
    import torch
    from safetensors.torch import load_file
    from transformers import (
        CLIPSegConfig,
        CLIPSegForImageSegmentation,
        CLIPSegProcessor,
        CLIPSegTextConfig,
        CLIPSegVisionConfig,
        CLIPTokenizer,
        ViTImageProcessor,
    )

    state = load_file(path, device="cpu")
    if not state:
        raise ValueError("CLIPSeg SafeTensors file contains no weights")
    floating_dtypes = {
        value.dtype for value in state.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    }
    if len(floating_dtypes) != 1:
        raise ValueError("CLIPSeg weights must use one floating-point dtype")
    dtype = next(iter(floating_dtypes))
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("CLIPSeg weights use an unsupported dtype")

    text_config = CLIPSegTextConfig(
        hidden_act="quick_gelu",
        hidden_size=512,
        intermediate_size=2048,
        num_attention_heads=8,
        num_hidden_layers=12,
        max_position_embeddings=77,
        vocab_size=49408,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
    )
    vision_config = CLIPSegVisionConfig(
        hidden_act="quick_gelu",
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
        image_size=224,
        patch_size=16,
        num_channels=3,
    )
    config = CLIPSegConfig(
        text_config=text_config,
        vision_config=vision_config,
        projection_dim=512,
        reduce_dim=64,
        extract_layers=(3, 6, 9),
        conditional_layer=0,
        decoder_attention_dropout=0.0,
        decoder_hidden_act="quick_gelu",
        decoder_intermediate_size=2048,
        decoder_num_attention_heads=4,
        use_complex_transposed_convolution=True,
    )
    model = CLIPSegForImageSegmentation(config).to(dtype=dtype)

    # Transformers versions disagree on whether these deterministic buffers
    # are serialized.  They are arange-derived, not learned model weights.
    for key in (
        "clip.text_model.embeddings.position_ids",
        "clip.vision_model.embeddings.position_ids",
    ):
        state.pop(key, None)
    model.load_state_dict(state, strict=True)
    model.eval()

    tokenizer_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "comfy", "sd1_tokenizer"))
    vocab_file = os.path.join(tokenizer_root, "vocab.json")
    merges_file = os.path.join(tokenizer_root, "merges.txt")
    if not os.path.isfile(vocab_file) or not os.path.isfile(merges_file):
        raise RuntimeError("ComfyUI's bundled CLIP tokenizer is unavailable")
    tokenizer = CLIPTokenizer(
        vocab_file=vocab_file,
        merges_file=merges_file,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        model_max_length=77,
    )
    image_processor = ViTImageProcessor(
        do_resize=True,
        size={"height": 352, "width": 352},
        resample=2,
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=True,
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
    )
    processor = CLIPSegProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
    )
    return _ClipSegEntry(model=model, processor=processor)




_CLIPSEG_CACHE = WeightCache(
    load=_loader("_load_clipseg_weight"),
    max_entries=2,
    release=_release_model_to_cpu,
)


@dataclass
class _ImageClassifierEntry:
    model: Any
    processor: Any
    architecture: str
    num_labels: int
    lock: threading.Lock = field(default_factory=threading.Lock)


def _load_image_classifier_weight(
    path: str, architecture: str,
) -> _ImageClassifierEntry:
    """Build one closed image-classifier architecture from SafeTensors."""
    import torch
    from safetensors.torch import load_file
    from transformers import (
        BeitConfig,
        BeitForImageClassification,
        BeitImageProcessor,
        ConvNextImageProcessor,
        ResNetConfig,
        ResNetForImageClassification,
        ViTConfig,
        ViTForImageClassification,
        ViTImageProcessor,
    )

    state = load_file(path, device="cpu")
    if not state:
        raise ValueError("classifier SafeTensors file contains no weights")
    heads = {
        "vit-base-patch16-224": "classifier.weight",
        "beit-base-patch16-224": "classifier.weight",
        "resnet-50-224": "classifier.1.weight",
    }
    if architecture not in heads:
        raise ValueError("image classifier architecture is not supported")
    head = state.get(heads[architecture])
    if not isinstance(head, torch.Tensor) or head.ndim != 2:
        raise ValueError("classifier weights have no compatible output head")
    num_labels = int(head.shape[0])
    if not 1 <= num_labels <= 10_000:
        raise ValueError("classifier output count is outside the safe range")
    floating_dtypes = {
        value.dtype for value in state.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    }
    if len(floating_dtypes) != 1:
        raise ValueError("classifier weights must use one floating-point dtype")
    dtype = next(iter(floating_dtypes))
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("classifier weights use an unsupported dtype")

    if architecture == "vit-base-patch16-224":
        config = ViTConfig(
            num_labels=num_labels,
            attention_probs_dropout_prob=0.0,
            encoder_stride=16,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            hidden_size=768,
            image_size=224,
            initializer_range=0.02,
            intermediate_size=3072,
            layer_norm_eps=1e-12,
            num_attention_heads=12,
            num_channels=3,
            num_hidden_layers=12,
            patch_size=16,
            qkv_bias=True,
        )
        model = ViTForImageClassification(config)
        processor = ViTImageProcessor(
            do_resize=True,
            size={"height": 224, "width": 224},
            resample=2,
            do_rescale=True,
            rescale_factor=1.0 / 255.0,
            do_normalize=True,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        )
    elif architecture == "beit-base-patch16-224":
        config = BeitConfig(
            num_labels=num_labels,
            attention_probs_dropout_prob=0.0,
            drop_path_rate=0.1,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            hidden_size=768,
            image_size=224,
            initializer_range=0.02,
            intermediate_size=3072,
            layer_norm_eps=1e-12,
            layer_scale_init_value=0.1,
            num_attention_heads=12,
            num_channels=3,
            num_hidden_layers=12,
            patch_size=16,
            use_absolute_position_embeddings=False,
            use_mask_token=False,
            use_mean_pooling=True,
            use_relative_position_bias=True,
            use_shared_relative_position_bias=False,
        )
        model = BeitForImageClassification(config)
        processor = BeitImageProcessor(
            do_resize=True,
            size={"height": 224, "width": 224},
            resample=2,
            do_rescale=True,
            rescale_factor=1.0 / 255.0,
            do_normalize=True,
            do_center_crop=False,
            crop_size={"height": 224, "width": 224},
            do_reduce_labels=False,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        )
    else:
        config = ResNetConfig(
            num_labels=num_labels,
            depths=[3, 4, 6, 3],
            downsample_in_first_stage=False,
            embedding_size=64,
            hidden_act="relu",
            hidden_sizes=[256, 512, 1024, 2048],
            layer_type="bottleneck",
            num_channels=3,
            out_features=["stage4"],
            out_indices=[4],
        )
        model = ResNetForImageClassification(config)
        processor = ConvNextImageProcessor(
            do_resize=True,
            size={"shortest_edge": 224},
            resample=3,
            do_rescale=True,
            rescale_factor=1.0 / 255.0,
            do_normalize=True,
            image_mean=(0.485, 0.456, 0.406),
            image_std=(0.229, 0.224, 0.225),
        )

    model = model.to(dtype=dtype)
    model.load_state_dict(state, strict=True)
    model.eval()
    return _ImageClassifierEntry(
        model=model,
        processor=processor,
        architecture=architecture,
        num_labels=num_labels,
    )




_IMAGE_CLASSIFIER_CACHE = WeightCache(
    load=_loader("_load_image_classifier_weight"),
    max_entries=3,
    release=_release_model_to_cpu,
)


@dataclass
class _TextEncoderEntry:
    clip: Any
    model_type: str
    device: str
    lock: threading.Lock = field(default_factory=threading.Lock)


def _load_text_encoder_weight(
    path: str, model_type: str, device: str,
) -> _TextEncoderEntry:
    import torch
    import comfy.sd
    import folder_paths

    enum_name = model_type.replace("-", "_").upper()
    clip_type = comfy.sd.CLIPType.__members__.get(enum_name)
    if clip_type is None:
        raise ValueError(f"unknown Comfy text-encoder type {model_type!r}")
    model_options = {}
    if device == "cpu":
        model_options["load_device"] = torch.device("cpu")
        model_options["offload_device"] = torch.device("cpu")
    clip = comfy.sd.load_clip(
        ckpt_paths=[path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=clip_type,
        model_options=model_options,
    )
    if not callable(getattr(clip, "tokenize", None)):
        raise ValueError("the selected weight is not a Comfy text encoder")
    return _TextEncoderEntry(
        clip=clip, model_type=model_type, device=device)




_TEXT_ENCODER_CACHE = WeightCache(
    load=_loader("_load_text_encoder_weight"), max_entries=2)


_QWEN_LANGUAGE_FAMILIES = frozenset({
    "qwen3_vl_2b",
    "qwen3_vl_4b",
    "qwen3_vl_8b",
    "qwen3_vl_32b",
    "qwen2_5_vl_3b",
    "qwen2_5_vl_7b",
    "qwen3_0_6b",
    "qwen3_4b",
})


def _dequant_qwen_block_fp8(state_dict: dict[str, Any]) -> None:
    """Materialize official Qwen per-128 block FP8 weights safely.

    Native Comfy text encoders do not yet consume the official
    ``weight_scale_inv`` layout.  Dequantizing once during the trusted load
    preserves model semantics without exposing a low-level quantization API to
    node packs.
    """
    import torch

    scale_keys = [
        key for key in state_dict if key.endswith(".weight_scale_inv")
    ]
    for scale_key in scale_keys:
        weight_key = scale_key[:-len("_scale_inv")]
        weight = state_dict.get(weight_key)
        scale = state_dict.get(scale_key)
        if not isinstance(weight, torch.Tensor) or not isinstance(scale, torch.Tensor):
            raise ValueError("Qwen FP8 scale is missing its tensor weight")
        if weight.ndim != 2 or scale.ndim != 2:
            raise ValueError("Qwen FP8 block weights must be two-dimensional")
        expected = (
            (weight.shape[0] + 127) // 128,
            (weight.shape[1] + 127) // 128,
        )
        if tuple(scale.shape) != expected:
            raise ValueError(
                f"Qwen FP8 scale shape {tuple(scale.shape)} does not match "
                f"weight shape {tuple(weight.shape)}")
        # Materialize one 128-row block at a time.  Expanding every block
        # scale to a full FP32 matrix would transiently add several copies of
        # a 32B model's largest weights and can OOM before Comfy can offload.
        dequantized = torch.empty(
            weight.shape, device=weight.device, dtype=torch.bfloat16)
        for row_block in range(scale.shape[0]):
            start = row_block * 128
            end = min(start + 128, weight.shape[0])
            row_scale = scale[row_block].float().repeat_interleave(128)
            row_scale = row_scale[:weight.shape[1]].unsqueeze(0)
            dequantized[start:end].copy_(
                (weight[start:end].float() * row_scale).to(torch.bfloat16))
        state_dict[weight_key] = dequantized
        del state_dict[scale_key]


def _load_qwen_language_model(
    paths: tuple[str, ...], family: str, device: str,
) -> _TextEncoderEntry:
    import torch
    import comfy.sd
    import comfy.text_encoders.hunyuan_video
    import comfy.text_encoders.qwen3vl
    import comfy.text_encoders.qwen_image
    import comfy.text_encoders.qwen_generation
    import comfy.utils
    import folder_paths

    state_dict: dict[str, Any] = {}
    parameters = 0
    for path in paths:
        shard, metadata = comfy.utils.load_torch_file(
            path, safe_load=True, return_metadata=True)
        shard, _ = comfy.utils.convert_old_quants(
            shard, model_prefix="", metadata=metadata)
        duplicate = state_dict.keys() & shard.keys()
        if duplicate:
            raise ValueError(
                f"Qwen SafeTensor shards contain duplicate key "
                f"{next(iter(duplicate))!r}")
        state_dict.update(shard)
        parameters += comfy.utils.calculate_parameters(shard)

    _dequant_qwen_block_fp8(state_dict)
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("model.language_model."):
            key = "model." + key[len("model.language_model."):]
        elif key.startswith("model.visual."):
            key = "visual." + key[len("model.visual."):]
        elif key.startswith("lm_head."):
            key = "model.lm_head." + key[len("lm_head."):]
        normalized[key] = value
    state_dict = normalized

    class Target:
        params = {}

    detect_options = comfy.text_encoders.hunyuan_video.llama_detect(state_dict)
    model_options = {}
    if device == "cpu":
        model_options["load_device"] = torch.device("cpu")
        model_options["offload_device"] = torch.device("cpu")
    has_lm_head = "model.lm_head.weight" in state_dict
    internal_family = (
        family.replace("qwen3_vl_", "qwen3vl_", 1)
        if family.startswith("qwen3_vl_")
        else family
    )
    model_options[f"{internal_family}_model_config"] = {
        "lm_head": has_lm_head,
    }

    if family.startswith("qwen3_vl_"):
        Target.clip = comfy.text_encoders.qwen3vl.te(
            **detect_options, model_type=internal_family)
        Target.tokenizer = comfy.text_encoders.qwen3vl.generation_tokenizer(
            model_type=internal_family)
    elif family.startswith("qwen2_5_vl_"):
        Target.clip = comfy.text_encoders.qwen_image.vl_te(
            **detect_options, model_type=family)
        Target.tokenizer = comfy.text_encoders.qwen_image.vl_tokenizer(
            model_type=family)
    else:
        Target.clip = comfy.text_encoders.qwen_generation.te(
            **detect_options, model_type=family)
        Target.tokenizer = comfy.text_encoders.qwen_generation.tokenizer(
            model_type=family)

    clip = comfy.sd.CLIP(
        Target,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        parameters=parameters,
        state_dict=[state_dict],
        model_options=model_options,
    )
    if not callable(getattr(clip, "generate", None)):
        raise ValueError("the selected Qwen weights do not support generation")
    return _TextEncoderEntry(
        clip=clip, model_type=family, device=device)


class _LanguageModelCache:
    def __init__(self, max_entries: int = 1) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[Any, ...], _TextEncoderEntry] = (
            OrderedDict())
        self._lock = threading.Lock()
        self.loads = 0
        self.hits = 0

    @staticmethod
    def _key(
        paths: tuple[str, ...], family: str, device: str,
    ) -> tuple[Any, ...]:
        files = []
        for path in paths:
            status = os.stat(path)
            files.append((
                os.path.realpath(path), status.st_dev, status.st_ino,
                status.st_size, status.st_mtime_ns, status.st_ctime_ns,
            ))
        return family, device, tuple(files)

    def get(
        self, paths: tuple[str, ...], family: str, device: str,
        cache: bool,
    ) -> _TextEncoderEntry:
        if not cache:
            self.loads += 1
            return _load_qwen_language_model(paths, family, device)
        key = self._key(paths, family, device)
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self.hits += 1
                self._entries[key] = entry
                return entry
            entry = _load_qwen_language_model(paths, family, device)
            self.loads += 1
            while len(self._entries) >= self.max_entries:
                self._entries.popitem(last=False)
            self._entries[key] = entry
            return entry

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
        return count


_LANGUAGE_MODEL_CACHE = _LanguageModelCache()


@dataclass
class _SegformerEntry:
    model: Any
    variant: str
    num_labels: int
    lock: threading.Lock = field(default_factory=threading.Lock)


def _load_segformer_weight(
    path: str, variant: str, num_labels: int,
) -> _SegformerEntry:
    try:
        from safetensors.torch import load_file
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError as exc:
        raise RuntimeError(
            "SegFormer semantic segmentation requires transformers and "
            "safetensors") from exc

    depths = {
        "b2": [3, 4, 6, 3],
        "b3": [3, 4, 18, 3],
        "b5": [3, 6, 40, 3],
    }.get(variant)
    if depths is None:
        raise ValueError("SegFormer variant must be b2, b3, or b5")
    config = SegformerConfig(
        num_labels=num_labels,
        num_channels=3,
        depths=depths,
        hidden_sizes=[64, 128, 320, 512],
        decoder_hidden_size=768,
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        sr_ratios=[8, 4, 2, 1],
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        drop_path_rate=0.1,
        reshape_last_stage=True,
        semantic_loss_ignore_index=255,
    )
    model = SegformerForSemanticSegmentation(config)
    state = load_file(path, device="cpu")
    model_state = model.state_dict()
    if set(state) != set(model_state):
        try:
            from transformers.conversion_mapping import (
                get_model_conversion_mapping,
            )
            from transformers.core_model_loading import (
                WeightRenaming,
                rename_source_key,
            )
        except ImportError as exc:
            raise ValueError(
                "SegFormer weights do not match the installed Transformers "
                "version") from exc
        conversions = get_model_conversion_mapping(
            model, add_legacy=False)
        if not conversions or any(
            not isinstance(item, WeightRenaming) for item in conversions
        ):
            raise ValueError(
                "SegFormer checkpoint conversion is not a pure key rename")
        converted = {}
        for key, value in state.items():
            renamed, _pattern = rename_source_key(
                key, conversions, [], model.base_model_prefix, model_state)
            if renamed in converted:
                raise ValueError(
                    "SegFormer checkpoint conversion produced duplicate keys")
            converted[renamed] = value
        state = converted
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to("cpu")
    return _SegformerEntry(
        model=model, variant=variant, num_labels=num_labels)




_SEGFORMER_CACHE = WeightCache(
    load=_loader("_load_segformer_weight"),
    max_entries=2,
    release=_release_model_to_cpu,
)










def _validate_onnx_weight_file(path: str) -> None:
    """Admit only one self-contained graph made from standard ONNX domains."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "ONNX model validation requires the onnx package") from exc

    maximum = int(os.environ.get(
        "COMFY_SECURE_ONNX_WEIGHT_MAX", str(4 * 1024 * 1024 * 1024)))
    size = os.path.getsize(path)
    if maximum <= 0 or not 1 <= size <= maximum:
        raise ValueError("ONNX model file is outside the configured size limit")
    try:
        model = onnx.load(path, load_external_data=False)
    except Exception as exc:
        raise ValueError("download is not a valid ONNX model") from exc

    if model.functions or model.training_info:
        raise ValueError("ONNX model functions and training graphs are not allowed")
    allowed_domains = {"", "ai.onnx", "ai.onnx.ml"}
    # Exporters commonly leave unused provider-specific opset declarations in
    # otherwise standard graphs. They carry no executable behavior; enforce
    # the domain boundary on every actual node below.
    if any(
        not isinstance(item.domain, str) or len(item.domain) > 256
        or not 1 <= item.version <= 2**31 - 1
        for item in model.opset_import
    ):
        raise ValueError("ONNX model has an invalid operator-set declaration")

    node_count = 0
    tensor_count = 0
    graph_count = 0

    def check_tensor(tensor: Any) -> None:
        nonlocal tensor_count
        tensor_count += 1
        if tensor_count > 200_000:
            raise ValueError("ONNX model has too many tensors")
        if (tensor.data_location == onnx.TensorProto.EXTERNAL
                or len(tensor.external_data)):
            raise ValueError("external ONNX tensor data is not allowed")
        if len(tensor.dims) > 16 or any(
                dimension < 0 or dimension > 2**31 - 1
                for dimension in tensor.dims):
            raise ValueError("ONNX tensor dimensions are invalid")

    def check_graph(graph: Any) -> None:
        nonlocal graph_count, node_count
        graph_count += 1
        if graph_count > 1024:
            raise ValueError("ONNX model has too many nested graphs")
        for tensor in graph.initializer:
            check_tensor(tensor)
        for sparse in graph.sparse_initializer:
            check_tensor(sparse.values)
            check_tensor(sparse.indices)
        for node in graph.node:
            node_count += 1
            if node_count > 200_000:
                raise ValueError("ONNX model has too many operators")
            if node.domain not in allowed_domains:
                raise ValueError(
                    "ONNX model uses a non-standard operator domain")
            if not node.op_type or len(node.op_type) > 128:
                raise ValueError("ONNX model contains an invalid operator")
            for attribute in node.attribute:
                if attribute.HasField("t"):
                    check_tensor(attribute.t)
                for tensor in attribute.tensors:
                    check_tensor(tensor)
                if attribute.HasField("g"):
                    check_graph(attribute.g)
                for nested in attribute.graphs:
                    check_graph(nested)

    check_graph(model.graph)
    if node_count == 0 or tensor_count == 0:
        raise ValueError("ONNX model contains no executable weighted graph")
    try:
        onnx.checker.check_model(model, full_check=False)
    except Exception as exc:
        raise ValueError("ONNX model failed structural validation") from exc


@dataclass
class _OnnxImageClassifierEntry:
    session: Any
    input_name: str
    output_name: str
    input_height: int
    input_width: int
    class_count: int
    input_layouts: frozenset[str]
    lock: threading.Lock = field(default_factory=threading.Lock)


def _load_onnx_image_classifier(path: str) -> _OnnxImageClassifierEntry:
    _validate_onnx_weight_file(path)
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX image classification requires onnxruntime") from exc

    options = ort.SessionOptions()
    options.log_severity_level = 3
    available = set(ort.get_available_providers())
    providers = [
        provider for provider in (
            "CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available
    ]
    if not providers:
        raise RuntimeError("ONNX Runtime has no supported execution provider")
    try:
        session = ort.InferenceSession(
            path, sess_options=options, providers=providers)
    except Exception as exc:
        raise ValueError("ONNX image classifier could not be loaded") from exc
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("ONNX image classifier must have one input and output")
    model_input = inputs[0]
    model_output = outputs[0]
    if model_input.type != "tensor(float)" or model_output.type not in {
        "tensor(float)", "tensor(float16)", "tensor(double)",
    }:
        raise ValueError("ONNX image classifier must use floating-point tensors")
    input_shape = model_input.shape
    output_shape = model_output.shape
    if len(input_shape) != 4 or len(output_shape) != 2:
        raise ValueError("ONNX image classifier has an invalid tensor rank")

    # WD-style NHWC and common NCHW models are both admitted.  The selected
    # layout is checked again when the loader binds preprocessing options.
    nhwc = input_shape[3] == 3
    nchw = input_shape[1] == 3
    if not nhwc and not nchw:
        raise ValueError("ONNX image classifier must consume three channels")
    if nhwc and nchw:
        raise ValueError("ONNX classifier channel layout is ambiguous")
    height = input_shape[1] if nhwc else input_shape[2]
    width = input_shape[2] if nhwc else input_shape[3]
    class_count = output_shape[1]
    if (type(height) is not int or type(width) is not int
            or not 1 <= height <= 4096 or not 1 <= width <= 4096):
        raise ValueError("ONNX classifier spatial dimensions must be fixed")
    if type(class_count) is not int or not 1 <= class_count <= 16_384:
        raise ValueError("ONNX classifier output count is outside the safe range")
    return _OnnxImageClassifierEntry(
        session=session,
        input_name=model_input.name,
        output_name=model_output.name,
        input_height=height,
        input_width=width,
        class_count=class_count,
        input_layouts=frozenset(
            layout for layout, valid in (("NHWC", nhwc), ("NCHW", nchw))
            if valid),
    )




_ONNX_IMAGE_CLASSIFIER_CACHE = WeightCache(
    load=_loader("_load_onnx_image_classifier"), max_entries=3)










@dataclass
class _SamEntry:
    model: Any
    architecture: str
    lock: threading.Lock = field(default_factory=threading.Lock)


def _load_sam_weight(path: str, architecture: str) -> _SamEntry:
    try:
        from safetensors.torch import load_file
        from segment_anything import sam_model_registry
    except ImportError as exc:
        raise RuntimeError(
            "SAM models require segment-anything and safetensors") from exc
    if architecture not in {"vit_b", "vit_l", "vit_h"}:
        raise ValueError("SAM architecture must be vit_b, vit_l, or vit_h")
    constructor = sam_model_registry.get(architecture)
    if constructor is None:
        raise RuntimeError(
            f"segment-anything does not provide {architecture!r}")
    model = constructor(checkpoint=None)
    state = load_file(path, device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to("cpu")
    return _SamEntry(model=model, architecture=architecture)




_SAM_CACHE = WeightCache(
    load=_loader("_load_sam_weight"),
    max_entries=2,
    release=_release_model_to_cpu,
)


_SAM2_CONFIGS = {
    "sam2_hiera_tiny": "configs/sam2/sam2_hiera_t.yaml",
    "sam2_hiera_small": "configs/sam2/sam2_hiera_s.yaml",
    "sam2_hiera_base_plus": "configs/sam2/sam2_hiera_b+.yaml",
    "sam2_hiera_large": "configs/sam2/sam2_hiera_l.yaml",
    "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def _load_sam2_weight(path: str, architecture: str) -> _SamEntry:
    try:
        from safetensors.torch import load_file
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise RuntimeError(
            "SAM2 models require sam2 and safetensors") from exc
    config = _SAM2_CONFIGS.get(architecture)
    if config is None:
        raise ValueError("unknown SAM2 architecture")
    model = build_sam2_video_predictor(
        config, ckpt_path=None, device="cpu")
    model.load_state_dict(load_file(path, device="cpu"), strict=True)
    model.eval()
    return _SamEntry(model=model, architecture=architecture)




_SAM2_CACHE = WeightCache(
    load=_loader("_load_sam2_weight"),
    max_entries=1,
    release=_release_model_to_cpu,
)


def _advanced_control_module(relative: str):
    """Resolve a fixed module from the installed Advanced-ControlNet pack.

    Resolution is anchored to one of that pack's registered node classes; no
    guest-controlled module name or path participates in the import.
    """
    import importlib
    import nodes

    mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", {})
    for node_id in (
        "ACN_AdvancedControlNetApply",
        "ACN_ControlNet++LoaderSingle",
        "ACN_ScaledSoftControlNetWeights",
        "ControlNetLoaderAdvanced",
        "ScaledSoftControlNetWeights",
    ):
        node_class = mappings.get(node_id)
        module_name = getattr(node_class, "__module__", "")
        parts = module_name.split(".")
        if "adv_control" not in parts:
            continue
        base = ".".join(parts[:parts.index("adv_control") + 1])
        return importlib.import_module(f"{base}.{relative}")
    for base in ("ComfyUI-Advanced-ControlNet.adv_control", "adv_control"):
        try:
            package = importlib.import_module(base)
        except ModuleNotFoundError:
            continue
        candidate = getattr(package, relative, None)
        if candidate is not None:
            return candidate
        try:
            return importlib.import_module(f"{base}.{relative}")
        except ModuleNotFoundError:
            continue
    raise RuntimeError(
        "this operation requires the host-installed "
        "ComfyUI-Advanced-ControlNet extension")


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
    _HF_ENDPOINT = "https://huggingface.co"
    _HF_PYTORCH_WEIGHT_EXTENSIONS = frozenset({
        ".bin", ".ckpt", ".patch", ".pt", ".pth",
    })
    _HF_WEIGHT_EXTENSIONS = frozenset({
        ".safetensors", ".sft", ".gguf", ".onnx",
        *_HF_PYTORCH_WEIGHT_EXTENSIONS,
    })
    _HF_WEIGHT_FOLDERS = frozenset({
        "audio_encoders",
        "background_removal",
        "checkpoints",
        "clip_vision",
        "controlnet",
        "detection",
        "diffusion_models",
        "embeddings",
        "frame_interpolation",
        "geometry_estimation",
        "gligen",
        "hypernetworks",
        "ipadapter",
        "inpaint",
        "latent_upscale_models",
        "loras",
        "model_patches",
        "optical_flow",
        "onnx",
        "photomaker",
        "sams",
        "semantic_segmentation",
        "style_models",
        "text_encoders",
        "unet_gguf",
        "upscale_models",
        "vae",
        "vae_approx",
    })
    _HF_DOWNLOAD_LOCK = threading.Lock()
    _HF_VERIFIED_WEIGHTS: dict[
        str, tuple[int, int, int, int, int, Optional[str]]
    ] = {}

    _WEIGHT_DTYPES = frozenset({
        "default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2",
        "fp16", "bf16", "fp32",
    })
    _COMPUTE_DTYPES = frozenset({"default", "fp16", "bf16", "fp32"})

    @staticmethod
    def _hf_repo_id(repo_id: str) -> str:
        import re

        if not isinstance(repo_id, str):
            raise TypeError("Hugging Face repo_id must be a string")
        if len(repo_id) > 96 or repo_id.startswith(("http:", "https:")):
            raise ValueError("Hugging Face repo_id must name a model repository")
        parts = repo_id.split("/")
        component = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
        if (len(parts) not in (1, 2)
                or any(not component.fullmatch(part) for part in parts)
                or any(part.endswith(("-", ".")) for part in parts)
                or "--" in repo_id or ".." in repo_id
                or repo_id.endswith(".git")):
            raise ValueError("Hugging Face repo_id must name a model repository")
        return repo_id

    @classmethod
    def _hf_weight_filename(cls, filename: str) -> tuple[str, str]:
        import re

        if not isinstance(filename, str):
            raise TypeError("Hugging Face weight filename must be a string")
        logical = filename.replace("\\", "/")
        parts = logical.split("/")
        component = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
        if (not logical or len(logical) > 1024 or logical.startswith("/")
                or any(len(part) > 255 or not component.fullmatch(part)
                       for part in parts)):
            raise ValueError(
                "Hugging Face weight filename must be a confined repository path")
        extension = os.path.splitext(parts[-1])[1].lower()
        if extension not in cls._HF_WEIGHT_EXTENSIONS:
            raise ValueError(
                "Hugging Face downloads are limited to .safetensors, .sft, "
                ".gguf, validated .onnx, and restricted PyTorch "
                "tensor-archive weights")
        return logical, extension

    @staticmethod
    def _hf_revision(revision: str) -> str:
        import re

        if not isinstance(revision, str):
            raise TypeError("Hugging Face revision must be a string")
        if (not revision or len(revision) > 200
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", revision)
                or revision.endswith("/")
                or any(part in ("", ".", "..")
                       for part in revision.split("/"))):
            raise ValueError("Hugging Face revision is invalid")
        return revision

    @staticmethod
    def _hf_sha256(sha256: Optional[str]) -> Optional[str]:
        import re

        if sha256 is None:
            return None
        if not isinstance(sha256, str) or not re.fullmatch(
                r"[0-9a-fA-F]{64}", sha256):
            raise ValueError("Hugging Face weight sha256 must be 64 hex digits")
        return sha256.lower()

    @classmethod
    def _hf_weight_destination(
        cls, folder: str, repo_id: str, revision: str,
        filename: str, extension: str,
    ) -> tuple[str, str]:
        import folder_paths

        if not isinstance(folder, str) or folder not in cls._HF_WEIGHT_FOLDERS:
            raise ValueError(
                "Hugging Face weights must target a known model catalogue")
        registered = folder_paths.folder_names_and_paths.get(folder)
        if registered is None or not registered[0]:
            raise ValueError(f"model catalogue {folder!r} is not registered")
        extensions = {str(value).lower() for value in registered[1]}
        if extension not in extensions:
            raise ValueError(
                f"model catalogue {folder!r} does not accept {extension} weights")

        root = os.path.realpath(os.path.abspath(registered[0][0]))
        os.makedirs(root, exist_ok=True)
        logical = f"huggingface/{repo_id}/{revision}/{filename}"
        destination = _InProcessAssets._confined_path(root, logical, folder)
        return logical, destination

    @staticmethod
    def _verify_weight_file(path: str, extension: str) -> None:
        if extension in (".safetensors", ".sft"):
            from safetensors import safe_open

            try:
                with safe_open(path, framework="pt", device="cpu") as weights:
                    keys = list(weights.keys())
            except Exception as exc:
                raise ValueError("download is not a valid SafeTensors weight file") from exc
            if not keys:
                raise ValueError("SafeTensors file contains no weights")
            return

        if extension == ".onnx":
            _validate_onnx_weight_file(path)
            return

        if extension in _InProcessModels._HF_PYTORCH_WEIGHT_EXTENSIONS:
            import collections.abc
            import torch

            try:
                value = torch.load(
                    path, map_location="cpu", mmap=True, weights_only=True)
            except Exception as exc:
                raise ValueError(
                    "download is not a restricted PyTorch weight archive"
                ) from exc

            tensors = 0
            entries = 0
            stack = [value]
            seen: set[int] = set()
            while stack:
                item = stack.pop()
                entries += 1
                if entries > 10_000_000:
                    raise ValueError("PyTorch weight archive is too complex")
                if isinstance(item, torch.Tensor):
                    tensors += 1
                    continue
                if item is None or isinstance(
                    item, (bool, int, float, str, bytes)
                ):
                    continue
                identity = id(item)
                if identity in seen:
                    continue
                seen.add(identity)
                if isinstance(item, collections.abc.Mapping):
                    stack.extend(item.keys())
                    stack.extend(item.values())
                    continue
                if isinstance(item, (list, tuple)):
                    stack.extend(item)
                    continue
                raise ValueError(
                    "PyTorch weight archive contains non-weight objects")
            if tensors == 0:
                raise ValueError("PyTorch weight archive contains no tensors")
            return

        import struct

        with open(path, "rb") as file:
            header = file.read(24)
        if len(header) != 24 or header[:4] != b"GGUF":
            raise ValueError("download is not a supported GGUF weight file")
        version = struct.unpack("<I", header[4:8])[0]
        tensor_count, metadata_count = struct.unpack("<QQ", header[8:24])
        if (version not in (2, 3) or not 1 <= tensor_count <= 10_000_000
                or metadata_count > 10_000_000):
            raise ValueError("download is not a supported GGUF weight file")

    @staticmethod
    def _file_sha256(path: str) -> str:
        import hashlib

        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _weight_identity(path: str) -> tuple[int, int, int, int, int]:
        status = os.stat(path)
        return (
            status.st_dev,
            status.st_ino,
            status.st_size,
            status.st_mtime_ns,
            status.st_ctime_ns,
        )

    @classmethod
    def _download_huggingface_weights(
        cls, repo_id: str, filename: str, folder: str,
        revision: str, sha256: Optional[str],
    ) -> str:
        import inspect
        import shutil
        import tempfile

        repo_id = cls._hf_repo_id(repo_id)
        filename, extension = cls._hf_weight_filename(filename)
        revision = cls._hf_revision(revision)
        sha256 = cls._hf_sha256(sha256)
        if extension == ".onnx" and sha256 is None:
            raise ValueError("Hugging Face ONNX weights require a sha256 pin")
        logical, destination = cls._hf_weight_destination(
            folder, repo_id, revision, filename, extension)

        request = {
            "repo_id": repo_id,
            "filename": filename,
            "repo_type": "model",
            "revision": revision,
            "endpoint": cls._HF_ENDPOINT,
            "token": False,
        }

        with cls._HF_DOWNLOAD_LOCK:
            if os.path.isfile(destination):
                try:
                    identity = cls._weight_identity(destination)
                except OSError:
                    identity = None
                if identity is not None:
                    cached = cls._HF_VERIFIED_WEIGHTS.get(destination)
                    if (cached is not None and cached[:5] == identity
                            and (sha256 is None or cached[5] == sha256)):
                        return logical
                    try:
                        cls._verify_weight_file(destination, extension)
                        digest_matches = (
                            sha256 is None
                            or cls._file_sha256(destination) == sha256)
                    except (OSError, ValueError):
                        digest_matches = False
                    if digest_matches:
                        cls._HF_VERIFIED_WEIGHTS[destination] = (
                            *identity, sha256)
                        return logical

            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError(
                    "Hugging Face weight downloads require "
                    "huggingface_hub") from exc
            if "dry_run" not in inspect.signature(
                    hf_hub_download).parameters:
                raise RuntimeError(
                    "huggingface_hub is too old for bounded weight downloads")

            max_bytes = int(os.environ.get(
                "COMFY_SECURE_HF_WEIGHT_MAX",
                str(64 * 1024 * 1024 * 1024)))
            if max_bytes <= 0:
                raise RuntimeError(
                    "COMFY_SECURE_HF_WEIGHT_MAX must be positive")
            info = hf_hub_download(**request, dry_run=True)
            size = getattr(info, "file_size", None)
            if type(size) is not int or size <= 0:
                raise RuntimeError("Hugging Face did not report a valid weight size")
            if size > max_bytes:
                raise ValueError(
                    f"Hugging Face weight is {size} bytes, over the "
                    f"{max_bytes} byte limit")
            source = os.path.realpath(str(hf_hub_download(**request)))
            if not os.path.isfile(source) or os.path.getsize(source) != size:
                raise RuntimeError("Hugging Face weight download is incomplete")

            parent = os.path.dirname(destination)
            os.makedirs(parent, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=".hf-weight-", suffix=extension, dir=parent)
            os.close(descriptor)
            try:
                shutil.copyfile(source, temporary)
                cls._verify_weight_file(temporary, extension)
                if sha256 is not None and cls._file_sha256(temporary) != sha256:
                    raise ValueError("Hugging Face weight sha256 does not match")
                os.chmod(temporary, 0o644)
                os.replace(temporary, destination)
                cls._HF_VERIFIED_WEIGHTS[destination] = (
                    *cls._weight_identity(destination), sha256)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        return logical

    async def download_huggingface_weights(
        self, repo_id: str, filename: str, folder: str,
        revision: str = "main", sha256: Optional[str] = None,
    ) -> str:
        """Install one public Hugging Face weight file.

        SafeTensors/SFT, GGUF, and standard self-contained ONNX graphs are
        parsed structurally. PyTorch archives are
        admitted only when the restricted ``weights_only`` unpickler proves
        that the object graph is a tensor container. No URL, endpoint, token,
        destination path, or custom/external ONNX graph is accepted. ONNX
        downloads additionally require a SHA-256 pin. A valid installed file
        is reused without a network request. The returned value is a logical
        catalogue name, never a path.
        """
        return await asyncio.to_thread(
            self._download_huggingface_weights,
            repo_id, filename, folder, revision, sha256)

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
        config_name: Optional[str] = None,
    ) -> tuple[ModelRef, Optional[ClipRef], Optional[VaeRef]]:
        import folder_paths
        import comfy.sd

        logical = self._model_name(name)
        options, compute = self._load_options(
            weight_dtype, compute_dtype, cublas_linear)
        path = folder_paths.get_full_path_or_raise("checkpoints", logical)
        if config_name is None:
            model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
                path, output_vae=True, output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                model_options=options)
        else:
            config = self._model_name(config_name, "config_name")
            if (weight_dtype != "default" or compute_dtype != "default"
                    or cublas_linear):
                raise ValueError(
                    "an explicit checkpoint config cannot be combined with "
                    "weight_dtype, compute_dtype, or cublas_linear overrides")
            config_path = folder_paths.get_full_path_or_raise("configs", config)
            model, clip, vae = comfy.sd.load_checkpoint(
                config_path=config_path, ckpt_path=path,
                output_vae=True, output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"))
        if compute is not None:
            model.set_model_compute_dtype(compute)
            model.force_cast_weights = False
        return (
            await self._ref("MODEL", ModelRef, model),
            await self._ref("CLIP", ClipRef, clip),
            await self._ref("VAE", VaeRef, vae),
        )

    async def load_upscale_model(self, name: str) -> UpscaleModelRef:
        import folder_paths
        import comfy.utils
        from spandrel import ImageModelDescriptor, ModelLoader

        logical = self._model_name(name)
        path = folder_paths.get_full_path_or_raise("upscale_models", logical)
        state = comfy.utils.load_torch_file(path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in state:
            state = comfy.utils.state_dict_prefix_replace(
                state, {"module.": ""})
        model = ModelLoader().load_from_state_dict(state).eval()
        if not isinstance(model, ImageModelDescriptor):
            raise ValueError("upscale model must be a single-image model")
        return await self._ref("UPSCALE_MODEL", UpscaleModelRef, model)

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

    async def load_gguf_text_encoders(
        self, names: Sequence[str], clip_type: str,
    ) -> ClipRef:
        """Load one to four GGUF-quantized text encoders as a single CLIP.

        The text-encoder counterpart of ``load_gguf_model``. GGUF is a
        quantization container, not a vendor format, and loading a quantized
        text encoder is the same basic host operation as loading a quantized
        diffusion model — so this is a primitive, not pack support. The pack
        keeps its own catalogue presentation and node shapes; the host owns
        file resolution, the custom operations, and the patcher.

        ``names`` may mix ``.gguf`` files and ordinary state dicts, which is
        what the two-, three- and four-encoder loaders exist for.
        """
        import comfy.model_management
        import comfy.sd
        import comfy.utils
        import folder_paths

        if isinstance(names, str) or not isinstance(names, (list, tuple)):
            raise TypeError("names must be a sequence of catalogue names")
        # One per loader variant: CLIP, Dual, Triple, Quadruple. An unbounded
        # list would let one request pin arbitrarily many encoders.
        if not 1 <= len(names) <= 4:
            raise ValueError(
                f"expected 1 to 4 text encoders, got {len(names)}")

        # Strict, unlike core's in-process CLIPLoader, which falls back to
        # STABLE_DIFFUSION for an unknown type string. In-process that is a
        # convenience; for a guest-facing primitive it would silently build a
        # different model than the caller asked for.
        if not isinstance(clip_type, str) or not clip_type:
            raise TypeError("clip_type must be a string")
        resolved_type = getattr(
            comfy.sd.CLIPType, clip_type.upper().replace("-", "_"), None)
        if resolved_type is None:
            raise ValueError(f"unknown CLIP type {clip_type!r}")

        catalogue = set(folder_paths.get_filename_list("text_encoders"))
        for folder in ("clip", "clip_gguf"):
            try:
                catalogue.update(folder_paths.get_filename_list(folder))
            except KeyError:
                continue

        state_dicts = []
        for index, name in enumerate(names):
            logical = self._model_name(name, f"names[{index}]")
            if logical not in catalogue:
                raise ValueError(
                    f"unknown text encoder catalogue name {logical!r}")
            path = folder_paths.get_full_path_or_raise(
                "clip_gguf" if logical.endswith(".gguf") else "text_encoders",
                logical)
            if logical.endswith(".gguf"):
                gguf = _fixed_gguf_node_module()
                if not hasattr(gguf, "gguf_clip_loader"):
                    raise RuntimeError(
                        "the installed ComfyUI-GGUF extension is "
                        "incompatible: missing gguf_clip_loader")
                state_dicts.append(gguf.gguf_clip_loader(path))
            else:
                state = comfy.utils.load_torch_file(path, safe_load=True)
                if "scaled_fp8" in state:
                    # Upstream's own guard: scaled FP8 needs different custom
                    # operations and only one set can be active.
                    raise ValueError(
                        f"{logical!r} is scaled FP8, which cannot be mixed "
                        "with GGUF text encoders")
                state_dicts.append(state)

        gguf = _fixed_gguf_node_module()
        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type=resolved_type,
            state_dicts=state_dicts,
            model_options={
                "custom_operations": gguf.GGMLOps,
                "initial_device":
                    comfy.model_management.text_encoder_offload_device(),
            },
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        clip.patcher = gguf.GGUFModelPatcher.clone(clip.patcher)
        return await self._ref("CLIP", ClipRef, clip)

    async def list_controlnet(self) -> list[str]:
        import folder_paths

        result = []
        seen = set()
        for name in folder_paths.get_filename_list("controlnet"):
            try:
                logical = self._model_name(name)
            except (TypeError, ValueError):
                continue
            if logical not in seen:
                seen.add(logical)
                result.append(logical)
        return result

    async def load_controlnet(
        self, name: str, model: Optional[ModelRef] = None,
    ) -> ControlNetRef:
        import comfy.controlnet
        import folder_paths

        logical = self._model_name(name)
        path = folder_paths.get_full_path_or_raise("controlnet", logical)
        model_value = (
            None if model is None
            else await current_runtime().refs.resolve(model))
        control_net = comfy.controlnet.load_controlnet(path, model_value)
        if control_net is None:
            raise RuntimeError(
                "the selected file does not contain a valid ControlNet model")
        return ControlNetRef._wrap(await current_runtime().refs.create(
            "CONTROL_NET", control_net))  # type: ignore[return-value]

    async def load_advanced_controlnet(
        self, name: str, model: Optional[ModelRef] = None,
        timestep_keyframe: Optional[TimestepKeyframeRef] = None,
    ) -> ControlNetRef:
        import folder_paths

        logical = self._model_name(name)
        path = folder_paths.get_full_path_or_raise("controlnet", logical)
        rt = current_runtime()
        model_value = None if model is None else await rt.refs.resolve(model)
        keyframe_value = (
            None if timestep_keyframe is None
            else await rt.refs.resolve(timestep_keyframe))
        control = _advanced_control_module("control")
        control_net = control.load_controlnet(
            path, keyframe_value, model_value)
        if control_net is None:
            raise RuntimeError(
                "the selected file does not contain a valid ControlNet model")
        if control.is_advanced_controlnet(control_net):
            control_net.verify_all_weights()
        return ControlNetRef._wrap(await rt.refs.create(
            "CONTROL_NET", control_net))  # type: ignore[return-value]

    async def load_controlnet_plusplus(
        self, name: str, control_type: str = "none",
    ) -> ControlNetRef:
        import folder_paths

        choices = {
            "openpose", "depth", "hed/pidi/scribble/ted",
            "canny/lineart/mlsd", "normal", "segment", "tile",
            "inpaint/outpaint", "none",
        }
        if control_type not in choices:
            raise ValueError(
                f"unknown ControlNet++ control type {control_type!r}")
        logical = self._model_name(name)
        path = folder_paths.get_full_path_or_raise("controlnet", logical)
        plusplus = _advanced_control_module("control_plusplus")
        control_net = plusplus.load_controlnetplusplus(path)
        control_net.single_control_type = control_type
        control_net.verify_control_type(logical)
        return ControlNetRef._wrap(await current_runtime().refs.create(
            "CONTROL_NET", control_net))  # type: ignore[return-value]

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

    async def load_clip_vision(self, model: str) -> ClipVisionRef:
        import comfy.clip_vision
        import folder_paths

        model = self._model_name(model, "CLIP-Vision weight")
        if not model.lower().endswith((".safetensors", ".sft")):
            raise ValueError("CLIP-Vision weights must use SafeTensors")
        path = folder_paths.get_full_path_or_raise("clip_vision", model)
        value = await asyncio.to_thread(comfy.clip_vision.load, path)
        if value is None:
            raise ValueError("the selected weight is not a CLIP-Vision model")
        return ClipVisionRef._wrap(await current_runtime().refs.create(
            "CLIP_VISION", value))  # type: ignore[return-value]

    async def load_text_encoder(
        self, model: str, model_type: str,
        device: str = "default",
    ) -> ClipRef:
        import comfy.sd
        import folder_paths

        model = self._model_name(model, "text-encoder weight")
        if not model.lower().endswith((".safetensors", ".sft")):
            raise ValueError("text-encoder weights must use SafeTensors")
        model_type = str(model_type).replace("-", "_").lower()
        if model_type.upper() not in comfy.sd.CLIPType.__members__:
            raise ValueError(
                f"unknown Comfy text-encoder type {model_type!r}")
        device = str(device)
        if device not in {"default", "cpu"}:
            raise ValueError("text-encoder device must be default or cpu")
        path = folder_paths.get_full_path_or_raise("text_encoders", model)
        entry = await asyncio.to_thread(
            _TEXT_ENCODER_CACHE.get, path, model_type, device)
        setattr(entry.clip, "_secure_text_generation_lock", entry.lock)
        return ClipRef._wrap(await current_runtime().refs.create(
            "CLIP", entry.clip))  # type: ignore[return-value]

    async def load_language_model(
        self, weights: list[str], family: str,
        device: str = "default", cache: bool = True,
    ) -> ClipRef:
        import folder_paths

        if not isinstance(weights, (list, tuple)):
            raise TypeError("language-model weights must be a list")
        if not 1 <= len(weights) <= 16:
            raise ValueError("language models require 1 to 16 weight shards")
        logical_weights = []
        seen = set()
        for item in weights:
            logical = self._model_name(item, "language-model weight")
            if not logical.lower().endswith((".safetensors", ".sft")):
                raise ValueError(
                    "language-model weights must use SafeTensors")
            if logical in seen:
                raise ValueError("language-model weight shards must be unique")
            seen.add(logical)
            logical_weights.append(logical)
        family = str(family).lower()
        if family not in _QWEN_LANGUAGE_FAMILIES:
            raise ValueError(f"unsupported language-model family {family!r}")
        device = str(device)
        if device not in {"default", "cpu"}:
            raise ValueError("language-model device must be default or cpu")
        if type(cache) is not bool:
            raise TypeError("language-model cache must be a boolean")
        paths = tuple(
            folder_paths.get_full_path_or_raise("text_encoders", logical)
            for logical in logical_weights
        )
        entry = await asyncio.to_thread(
            _LANGUAGE_MODEL_CACHE.get, paths, family, device, cache)
        setattr(entry.clip, "_secure_text_generation_lock", entry.lock)
        setattr(entry.clip, "_secure_language_family", family)
        return ClipRef._wrap(await current_runtime().refs.create(
            "CLIP", entry.clip))  # type: ignore[return-value]

    async def load_ipadapter(
        self, model: str, clip_vision: ClipVisionRef,
    ) -> Ref:
        import folder_paths
        import nodes

        model = self._model_name(model, "IP-Adapter weight")
        if not model.lower().endswith((".safetensors", ".sft")):
            raise ValueError("IP-Adapter weights must use SafeTensors")
        folder_paths.get_full_path_or_raise("ipadapter", model)
        node_class = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get(
            "IPAdapterModelLoader")
        if node_class is None:
            raise RuntimeError(
                "IP-Adapter loading requires the host-installed "
                "ComfyUI IPAdapter Plus extension")
        result = await asyncio.to_thread(
            node_class().load_ipadapter_model, model)
        if (not isinstance(result, (tuple, list)) or not result
                or not isinstance(result[0], dict)
                or not isinstance(result[0].get("ip_adapter"), dict)
                or not result[0]["ip_adapter"]):
            raise ValueError("the selected weight is not an IP-Adapter model")
        vision = await current_runtime().refs.resolve(clip_vision)
        value = {
            "secure_kind": "ipadapter.pipeline",
            "ipadapter": result[0],
            "clip_vision": vision,
        }
        return await current_runtime().refs.create("IPADAPTER_PIPE", value)

    async def load_brushnet(
        self, model: str, dtype: str = "float16",
    ) -> BrushNetRef:
        import folder_paths
        import nodes

        model = self._model_name(model, "BrushNet weight")
        if not model.lower().endswith((".safetensors", ".sft")):
            raise ValueError("BrushNet weights must use SafeTensors")
        if dtype not in {"float16", "bfloat16", "float32", "float64"}:
            raise ValueError(
                "BrushNet dtype must be float16, bfloat16, float32, or float64")
        folder_paths.get_full_path_or_raise("inpaint", model)
        node_class = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get(
            "BrushNetLoader")
        if node_class is None:
            raise RuntimeError(
                "BrushNet loading requires the host-installed canonical "
                "ComfyUI-BrushNet extension")
        result = await asyncio.to_thread(
            node_class().brushnet_loading, model, dtype)
        if (not isinstance(result, (tuple, list)) or len(result) != 1
                or not isinstance(result[0], dict)
                or result[0].get("brushnet") is None
                or not isinstance(result[0].get("SDXL"), bool)
                or not isinstance(result[0].get("PP"), bool)
                or result[0].get("dtype") is None):
            raise ValueError(
                "the canonical BrushNet loader returned an invalid model")
        if result[0]["PP"]:
            raise ValueError(
                "the selected weight is PowerPaint, not a BrushNet model")
        return BrushNetRef._wrap(await current_runtime().refs.create(
            "BRUSHNET_MODEL", result[0]))  # type: ignore[return-value]

    async def load_powerpaint(
        self, model: str, base_clip: str, powerpaint_clip: str,
        dtype: str = "float16",
    ) -> PowerPaintRef:
        import folder_paths
        import nodes

        model = self._model_name(model, "PowerPaint weight")
        base_clip = self._model_name(base_clip, "base CLIP weight")
        powerpaint_clip = self._model_name(
            powerpaint_clip, "PowerPaint CLIP weight")
        for label, value in (
            ("PowerPaint", model),
            ("base CLIP", base_clip),
            ("PowerPaint CLIP", powerpaint_clip),
        ):
            if not value.lower().endswith((".safetensors", ".sft")):
                raise ValueError(f"{label} weights must use SafeTensors")
        if dtype not in {"float16", "bfloat16", "float32", "float64"}:
            raise ValueError(
                "PowerPaint dtype must be float16, bfloat16, float32, or "
                "float64")
        model_path = folder_paths.get_full_path_or_raise("inpaint", model)
        base_path = folder_paths.get_full_path_or_raise(
            "text_encoders", base_clip)
        clip_path = folder_paths.get_full_path_or_raise(
            "inpaint", powerpaint_clip)
        mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", {})
        brushnet_class = mappings.get("BrushNetLoader")
        clip_class = mappings.get("PowerPaintCLIPLoader")
        if brushnet_class is None or clip_class is None:
            raise RuntimeError(
                "PowerPaint loading requires the host-installed canonical "
                "ComfyUI-BrushNet extension")

        def load():
            brushnet_loader = brushnet_class()
            model_key = os.path.basename(model_path)
            brushnet_loader.inpaint_files = {
                model_key: os.path.dirname(model_path)}
            model_result = brushnet_loader.brushnet_loading(model_key, dtype)

            clip_loader = clip_class()
            base_key = os.path.basename(base_path)
            clip_key = os.path.basename(clip_path)
            clip_loader.clip_files = {base_key: os.path.dirname(base_path)}
            clip_loader.inpaint_files = {clip_key: os.path.dirname(clip_path)}
            clip_result = clip_loader.ppclip_loading(base_key, clip_key)
            return model_result, clip_result

        model_result, clip_result = await asyncio.to_thread(load)
        if (not isinstance(model_result, (tuple, list))
                or len(model_result) != 1
                or not isinstance(model_result[0], dict)
                or model_result[0].get("brushnet") is None
                or model_result[0].get("PP") is not True
                or model_result[0].get("dtype") is None):
            raise ValueError(
                "the canonical loader did not return a PowerPaint model")
        if (not isinstance(clip_result, (tuple, list))
                or len(clip_result) != 1 or clip_result[0] is None):
            raise ValueError(
                "the canonical loader did not return a PowerPaint CLIP")
        value = {
            "secure_kind": "powerpaint.pipeline",
            "powerpaint": model_result[0],
            "clip": clip_result[0],
        }
        return PowerPaintRef._wrap(await current_runtime().refs.create(
            "POWERPAINT_MODEL", value))  # type: ignore[return-value]

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
            _INPAINT_MODEL_CACHE.clear()
            _CLIPSEG_CACHE.clear()
            _IMAGE_CLASSIFIER_CACHE.clear()
            _ONNX_IMAGE_CLASSIFIER_CACHE.clear()
            _TEXT_ENCODER_CACHE.clear()
            _LANGUAGE_MODEL_CACHE.clear()
            InProcessLlamaCpp().clear()
            _SEGFORMER_CACHE.clear()
            _SAM_CACHE.clear()
        if bool(collect_cycles):
            gc.collect()
        after = int(comfy.model_management.get_free_memory())
        return before, after

    async def load_clipseg(self, model: str) -> ClipSegRef:
        import folder_paths

        model = self._model_name(model, "CLIPSeg weight")
        if not model.lower().endswith(".safetensors"):
            raise ValueError("CLIPSeg weights must use SafeTensors")
        path = folder_paths.get_full_path_or_raise("detection", model)
        value = (await asyncio.to_thread(_CLIPSEG_CACHE.get, path)).bundle()
        return ClipSegRef._wrap(await current_runtime().refs.create(
            "CLIPSEGMODEL", value))  # type: ignore[return-value]

    async def load_image_classifier(
        self, model: str, architecture: str, labels: list[str],
    ) -> ImageClassifierRef:
        import folder_paths

        model = self._model_name(model, "image classifier weight")
        if not model.lower().endswith(".safetensors"):
            raise ValueError("image classifier weights must use SafeTensors")
        architecture = str(architecture)
        if architecture not in {
            "vit-base-patch16-224",
            "beit-base-patch16-224",
            "resnet-50-224",
        }:
            raise ValueError("image classifier architecture is not supported")
        if not isinstance(labels, (list, tuple)):
            raise TypeError("image classifier labels must be a list")
        labels = tuple(str(label) for label in labels)
        if (not labels or len(labels) > 10_000
                or any(not label or len(label) > 256 for label in labels)):
            raise ValueError("image classifier labels are invalid")
        path = folder_paths.get_full_path_or_raise("detection", model)
        entry = await asyncio.to_thread(
            _IMAGE_CLASSIFIER_CACHE.get, path, architecture)
        if len(labels) != entry.num_labels:
            raise ValueError(
                "image classifier labels do not match the weight output count")
        value = {
            "model": entry.model,
            "processor": entry.processor,
            "architecture": entry.architecture,
            "labels": labels,
            "lock": entry.lock,
        }
        return ImageClassifierRef._wrap(await current_runtime().refs.create(
            "IMAGE_CLASSIFIER", value))  # type: ignore[return-value]

    async def load_onnx_image_classifier(
        self, model: str, input_layout: str = "NHWC",
        channel_order: str = "BGR", resize_mode: str = "fit_pad",
        input_scale: float = 255.0,
        pad_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        activation: str = "identity", resize_filter: str = "lanczos",
    ) -> ImageClassifierRef:
        """Bind a self-contained standard ONNX image classifier.

        Preprocessing is a closed, reusable transform.  Labels, category
        ranges, thresholds, exclusions, and output formatting remain node
        code; the host only retains and pages the numeric score matrix.
        """
        import math
        import folder_paths

        model = self._model_name(model, "ONNX image classifier")
        if not model.lower().endswith(".onnx"):
            raise ValueError("ONNX image classifiers must use .onnx files")
        input_layout = str(input_layout).upper()
        channel_order = str(channel_order).upper()
        resize_mode = str(resize_mode).lower()
        activation = str(activation).lower()
        resize_filter = str(resize_filter).lower()
        if input_layout not in {"NHWC", "NCHW"}:
            raise ValueError("ONNX classifier layout must be NHWC or NCHW")
        if channel_order not in {"RGB", "BGR"}:
            raise ValueError("ONNX classifier channel order must be RGB or BGR")
        if resize_mode not in {"fit_pad", "stretch"}:
            raise ValueError("ONNX classifier resize mode is not supported")
        if activation not in {"identity", "sigmoid", "softmax"}:
            raise ValueError("ONNX classifier activation is not supported")
        if resize_filter not in {"nearest", "bilinear", "bicubic", "lanczos"}:
            raise ValueError("ONNX classifier resize filter is not supported")
        input_scale = float(input_scale)
        if not math.isfinite(input_scale) or not 0 < input_scale <= 65_535:
            raise ValueError("ONNX classifier input scale is invalid")

        def triple(
            value: Any, field_name: str, *, nonzero: bool = False,
            unit: bool = False,
        ) -> tuple[float, float, float]:
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise ValueError(
                    f"ONNX classifier {field_name} must have three values")
            result = tuple(float(item) for item in value)
            if (any(not math.isfinite(item) or abs(item) > 1_000_000
                    for item in result)
                    or (nonzero and any(item == 0 for item in result))
                    or (unit and any(not 0 <= item <= 1 for item in result))):
                raise ValueError(f"ONNX classifier {field_name} is invalid")
            return result  # type: ignore[return-value]

        pad_color = triple(pad_color, "pad color", unit=True)
        mean = triple(mean, "mean")
        std = triple(std, "standard deviation", nonzero=True)
        path = folder_paths.get_full_path_or_raise("onnx", model)
        entry = await asyncio.to_thread(_ONNX_IMAGE_CLASSIFIER_CACHE.get, path)
        if input_layout not in entry.input_layouts:
            raise ValueError(
                f"ONNX classifier tensor is not laid out as {input_layout}")
        value = {
            "secure_kind": "image_classifier.onnx",
            "session": entry.session,
            "input_name": entry.input_name,
            "output_name": entry.output_name,
            "input_height": entry.input_height,
            "input_width": entry.input_width,
            "class_count": entry.class_count,
            "input_layout": input_layout,
            "channel_order": channel_order,
            "resize_mode": resize_mode,
            "input_scale": input_scale,
            "pad_color": pad_color,
            "mean": mean,
            "std": std,
            "activation": activation,
            "resize_filter": resize_filter,
            "lock": entry.lock,
        }
        return ImageClassifierRef._wrap(await current_runtime().refs.create(
            "IMAGE_CLASSIFIER", value))  # type: ignore[return-value]

    async def load_segformer(
        self, model: str, variant: str, num_labels: int,
    ) -> SemanticSegmentationRef:
        import folder_paths

        model = self._model_name(model, "SegFormer weight")
        if not model.lower().endswith((".safetensors", ".sft")):
            raise ValueError("SegFormer weights must use SafeTensors")
        variant = str(variant)
        if variant not in {"b2", "b3", "b5"}:
            raise ValueError("SegFormer variant must be b2, b3, or b5")
        if (isinstance(num_labels, bool) or not isinstance(num_labels, int)
                or not 1 <= num_labels <= 1024):
            raise ValueError("SegFormer num_labels must be in [1, 1024]")
        path = folder_paths.get_full_path_or_raise(
            "semantic_segmentation", model)
        entry = await asyncio.to_thread(
            _SEGFORMER_CACHE.get, path, variant, num_labels)
        return SemanticSegmentationRef._wrap(
            await current_runtime().refs.create(
                "SEMANTIC_SEGMENTATION_MODEL", entry)
        )  # type: ignore[return-value]


    async def load_inpaint_model(
        self, model: str, architecture: str = "big-lama",
    ) -> InpaintModelRef:
        import folder_paths

        model = self._model_name(model, "image inpaint weight")
        if not model.lower().endswith((".safetensors", ".sft")):
            raise ValueError("image inpaint weights must use SafeTensors")
        architecture = str(architecture)
        if architecture != "big-lama":
            raise ValueError("unknown image inpaint architecture")
        path = folder_paths.get_full_path_or_raise("detection", model)
        entry = await asyncio.to_thread(
            _INPAINT_MODEL_CACHE.get, path, architecture)
        return InpaintModelRef._wrap(await current_runtime().refs.create(
            "INPAINT_MODEL", entry.bundle()))  # type: ignore[return-value]

    async def load_background_removal_model(
        self, model: str,
    ) -> BackgroundRemovalModelRef:
        import folder_paths
        from comfy.bg_removal_model import load

        model = self._model_name(model, "background-removal weight")
        if not model.lower().endswith((".safetensors", ".sft")):
            raise ValueError(
                "background-removal weights must use SafeTensors")
        path = folder_paths.get_full_path_or_raise(
            "background_removal", model)
        value = await asyncio.to_thread(load, path)
        if value is None:
            raise ValueError(
                "the selected weight is not a supported ComfyUI "
                "background-removal model")
        bundle = {
            "secure_kind": "background_removal.comfy",
            "model": value,
            "lock": threading.Lock(),
        }
        return BackgroundRemovalModelRef._wrap(
            await current_runtime().refs.create(
                "BACKGROUND_REMOVAL_MODEL", bundle)
        )  # type: ignore[return-value]


    async def load_object_detector(self, model: str) -> ObjectDetectorRef:
        import comfy.model_base

        model = self._model_name(model, "object detector")
        if not model.lower().endswith((".safetensors", ".sft")):
            raise ValueError("object-detector weights must use SafeTensors")
        model_ref = await self.load_diffusion_model(model)
        patcher = await current_runtime().refs.resolve(model_ref)
        if not isinstance(patcher.model, comfy.model_base.RT_DETR_v4):
            raise ValueError(
                "the selected weight is not a supported RT-DETR model")
        value = {
            "secure_kind": "object_detector.rt_detr",
            "model": patcher,
        }
        return ObjectDetectorRef._wrap(await current_runtime().refs.create(
            "OBJECT_DETECTOR", value))  # type: ignore[return-value]

    async def load_sam(
        self, model: str, architecture: str = "vit_b",
        device_mode: str = "AUTO",
    ) -> SamModelRef:
        import folder_paths

        model = self._model_name(model, "SAM weight")
        if not model.lower().endswith((".safetensors", ".sft")):
            raise ValueError("SAM weights must use SafeTensors")
        architecture = str(architecture)
        if architecture not in {"vit_b", "vit_l", "vit_h", *_SAM2_CONFIGS}:
            raise ValueError("unknown SAM architecture")
        device_mode = str(device_mode)
        if device_mode not in {"AUTO", "Prefer GPU", "CPU"}:
            raise ValueError("SAM device_mode must be AUTO, Prefer GPU, or CPU")
        path = folder_paths.get_full_path_or_raise("sams", model)
        is_sam2 = architecture in _SAM2_CONFIGS
        cache = _SAM2_CACHE if is_sam2 else _SAM_CACHE
        entry = await asyncio.to_thread(cache.get, path, architecture)
        value = {
            "secure_kind": "sam.v2" if is_sam2 else "sam.v1",
            "model": entry.model,
            "architecture": entry.architecture,
            "device_mode": device_mode,
            "lock": entry.lock,
        }
        return SamModelRef._wrap(await current_runtime().refs.create(
            "SAM_MODEL", value))  # type: ignore[return-value]

    async def generate_text(
        self, generator: str, input_text: str, max_new_tokens: int = 128,
        weight: Optional[str] = None,
    ) -> str:
        import folder_paths

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
        if weight is None:
            raise ValueError(
                "text generator 'superprompt-v1' requires a declared weight")
        weight = self._model_name(weight, "weight")
        if not weight.lower().endswith(".safetensors"):
            raise ValueError("text generator weights must use SafeTensors")
        weight_path = folder_paths.get_full_path_or_raise(
            "text_encoders", weight)
        return await asyncio.to_thread(
            _TEXT_GENERATOR_CACHE.generate,
            generator, weight_path, input_text, max_new_tokens)


# --------------------------------------------------------------------------- #
# ctx domain implementations — one class per domain declared above, each doing
# in-process what the overlay does across a process boundary. Output and graph
# carry the most logic because they translate a node's declarative result into
# the engine's expected shapes.
# --------------------------------------------------------------------------- #
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


def _image_metadata_owner(
    prompt: Any = None,
    extra_pnginfo: Any = None,
    *,
    include_execution_metadata: bool = True,
    extra_metadata: Optional[dict[str, Any]] = None,
):
    """Build the small object expected by ComfyUI's image save helpers.

    The broker owns prompt/workflow injection. Nodes may add ordinary JSON
    fields, but never receive the hidden execution metadata merely to save it
    again.
    """
    import json
    from types import SimpleNamespace

    if extra_metadata is not None and not isinstance(extra_metadata, dict):
        raise TypeError("extra image metadata must be a mapping")
    try:
        encoded = json.dumps(extra_metadata or {}, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("extra image metadata must be JSON-compatible") from error
    if len(encoded.encode("utf-8")) > 1024 * 1024:
        raise ValueError("extra image metadata exceeds 1 MiB")
    normalized_extra = json.loads(encoded)

    metadata = {}
    if include_execution_metadata and isinstance(extra_pnginfo, dict):
        metadata.update(extra_pnginfo)
    metadata.update(normalized_extra)
    hidden = SimpleNamespace(
        prompt=prompt if include_execution_metadata else None,
        extra_pnginfo=metadata or None,
    )
    return SimpleNamespace(hidden=hidden)


def _a1111_parameters_payload(value: Optional[str]) -> tuple[Optional[str], bytes]:
    """Validate the closed Automatic1111 metadata profile.

    EXIF UserComment uses an eight-byte character-code prefix followed by
    UTF-16BE.  Keeping the payload below 60 KiB leaves deterministic room for
    TIFF/EXIF structure inside JPEG's single APP1 segment.
    """
    if value is None:
        return None, b""
    if not isinstance(value, str):
        raise TypeError("a1111_parameters must be a string or None")
    try:
        encoded = value.encode("utf-16-be")
    except UnicodeEncodeError as error:
        raise ValueError("a1111_parameters must contain valid Unicode") from error
    if len(encoded) > 60 * 1024:
        raise ValueError("a1111_parameters exceeds the 60 KiB EXIF payload limit")
    return value, b"UNICODE\x00" + encoded


def _add_a1111_exif(exif: Any, payload: bytes) -> Any:
    if not payload:
        return exif
    exif_ifd = exif.get_ifd(0x8769)
    exif_ifd[0x9286] = payload
    exif[0x8769] = exif_ifd
    return exif


class _InProcessUi:
    def __init__(self, prompt: Any = None, extra_pnginfo: Any = None) -> None:
        self._metadata_owner = _image_metadata_owner(prompt, extra_pnginfo)

    async def preview_images(self, images: ImageRef,
                             animated: bool = False) -> dict:
        from ._ui import PreviewImage

        value = await current_runtime().refs.resolve(images)
        return PreviewImage(
            value, animated=animated, cls=self._metadata_owner).as_dict()

    async def preview_mask(self, mask: MaskRef,
                           animated: bool = False) -> dict:
        from ._ui import PreviewMask

        value = await current_runtime().refs.resolve(mask)
        return PreviewMask(
            value, animated=animated, cls=self._metadata_owner).as_dict()

    async def preview_audio(self, audio: AudioRef) -> dict:
        from ._ui import PreviewAudio

        value = await current_runtime().refs.resolve(audio)
        return PreviewAudio(value, cls=self._metadata_owner).as_dict()

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
            cls=self._metadata_owner, fps=rate, lossless=False, quality=50,
            method=0)
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
    _STILL_FORMATS = {
        "png": ("PNG", frozenset({".png"}), ".png"),
        "jpg": ("JPEG", frozenset({".jpg", ".jpeg"}), ".jpg"),
        "jpeg": ("JPEG", frozenset({".jpg", ".jpeg"}), ".jpg"),
        "webp": ("WEBP", frozenset({".webp"}), ".webp"),
        "j2k": ("JPEG2000", frozenset({".j2k", ".jp2"}), ".j2k"),
        "jp2": ("JPEG2000", frozenset({".j2k", ".jp2"}), ".jp2"),
        "gif": ("GIF", frozenset({".gif"}), ".gif"),
        "tiff": ("TIFF", frozenset({".tiff"}), ".tiff"),
        "bmp": ("BMP", frozenset({".bmp"}), ".bmp"),
        "avif": ("AVIF", frozenset({".avif"}), ".avif"),
    }
    _IMAGE_BATCH_MAX = 4096

    def __init__(self, prompt: Any = None, extra_pnginfo: Any = None) -> None:
        self._prompt = prompt
        self._extra_pnginfo = extra_pnginfo
        self._metadata_owner = _image_metadata_owner(prompt, extra_pnginfo)

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

    @classmethod
    def _still_format(cls, value: str) -> tuple[str, frozenset[str], str]:
        key = str(value).lower()
        result = cls._STILL_FORMATS.get(key)
        if result is None:
            allowed = ", ".join(cls._STILL_FORMATS)
            raise ValueError(
                f"still-image format {value!r} is not supported; "
                f"choose one of {allowed}")
        return result

    @staticmethod
    def _logical_output_target(
        output_dir: str, filename: str, suffixes: frozenset[str],
    ) -> tuple[str, str, str, str]:
        """Validate one exact logical filename and resolve its confined target."""
        logical = str(filename).replace("\\", "/")
        if (
            not logical or "\x00" in logical or logical.startswith("/")
            or re.match(r"^[A-Za-z]:($|/)", logical)
            or len(logical.encode("utf-8")) > 1024
        ):
            raise ValueError("image filename must be a bounded relative path")
        parts = logical.split("/")
        if (
            len(parts) > 32
            or any(part in {"", ".", ".."} for part in parts)
            or any(len(part.encode("utf-8")) > 255 for part in parts)
        ):
            raise ValueError("image filename contains an unsafe path component")
        suffix = os.path.splitext(parts[-1])[1].lower()
        if suffix not in suffixes:
            expected = ", ".join(sorted(suffixes))
            raise ValueError(
                f"image filename suffix {suffix!r} does not match "
                f"the selected format ({expected})")

        root = os.path.realpath(os.path.abspath(output_dir))
        parent = os.path.realpath(os.path.join(root, *parts[:-1]))
        try:
            confined_parent = os.path.commonpath((root, parent)) == root
        except ValueError:
            confined_parent = False
        if not confined_parent:
            raise ValueError("image filename escapes the output directory")
        os.makedirs(parent, exist_ok=True)
        # Resolve once more after creation so an existing symlinked component
        # cannot turn a relative name into ambient filesystem authority.
        parent = os.path.realpath(os.path.join(root, *parts[:-1]))
        target = os.path.realpath(os.path.join(parent, parts[-1]))
        try:
            confined_target = os.path.commonpath((root, target)) == root
        except ValueError:
            confined_target = False
        if not confined_target:
            raise ValueError("image filename escapes the output directory")
        subfolder = "/".join(parts[:-1])
        return target, logical, parts[-1], subfolder

    @staticmethod
    def _save_pil_exclusive(rendered: Any, target: str, format: str,
                            options: dict[str, Any]) -> None:
        """Encode beside the destination, then publish without overwriting."""
        import tempfile

        parent = os.path.dirname(target)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".comfy-image-", suffix=".tmp", dir=parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                rendered.save(stream, format=format, **options)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError as error:
                raise FileExistsError(
                    f"output image already exists: "
                    f"{os.path.basename(target)!r}") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    async def save_images(
        self, images: ImageRef, filename_prefix: str = "ComfyUI",
        subfolder: str = "", compress_level: int = 4,
        caption: Optional[str] = None,
        caption_extension: str = ".txt",
        save_metadata: bool = True,
        extra_metadata: Optional[dict[str, Any]] = None,
        a1111_parameters: Optional[str] = None,
        image_format: str = "png", quality: int = 95,
        filenames: Optional[list[str]] = None,
        lossless: bool = False, optimize: bool = False,
    ) -> dict:
        import numpy as np
        from PIL import Image as PILImage
        import folder_paths
        from ._io import FolderType
        from ._ui import ImageSaveHelper, SavedImages, SavedResult

        value = await current_runtime().refs.resolve(images)
        if not hasattr(value, "ndim") or int(value.ndim) != 4:
            raise TypeError("saved IMAGE must contain a BHWC tensor")
        batch_size = int(value.shape[0])
        if not 1 <= batch_size <= self._IMAGE_BATCH_MAX:
            raise ValueError(
                f"saved IMAGE batch must be in [1, {self._IMAGE_BATCH_MAX}]")
        if int(value.shape[-1]) not in (1, 3, 4):
            raise ValueError("saved IMAGE must have 1, 3, or 4 channels")
        level = int(compress_level)
        if not 0 <= level <= 9:
            raise ValueError("PNG compression level must be in [0, 9]")
        pil_format, allowed_suffixes, default_suffix = self._still_format(
            image_format)
        a1111_text, a1111_exif = _a1111_parameters_payload(
            a1111_parameters)
        if a1111_text is not None and pil_format not in {"PNG", "JPEG", "WEBP"}:
            raise ValueError(
                "a1111_parameters is supported only for PNG, JPEG, and WebP")
        quality = int(quality)
        if not 1 <= quality <= 100:
            raise ValueError("image quality must be in [1, 100]")
        if type(lossless) is not bool or type(optimize) is not bool:
            raise TypeError("lossless and optimize must be booleans")
        metadata_owner = _image_metadata_owner(
            self._prompt,
            self._extra_pnginfo,
            include_execution_metadata=bool(save_metadata),
            extra_metadata=extra_metadata,
        )
        output_dir = folder_paths.get_output_directory()
        requested: list[tuple[str, str, str, str]] = []
        if filenames is not None:
            if not isinstance(filenames, (list, tuple)):
                raise TypeError("image filenames must be a sequence of strings")
            if len(filenames) != batch_size:
                raise ValueError(
                    "image filenames length must equal the IMAGE batch size")
            if any(not isinstance(name, str) for name in filenames):
                raise TypeError("every image filename must be a string")
            requested = [
                self._logical_output_target(output_dir, name, allowed_suffixes)
                for name in filenames
            ]
            targets = [entry[0] for entry in requested]
            if len(set(targets)) != len(targets):
                raise ValueError("image filenames must be unique within a batch")
            if any(os.path.lexists(target) for target in targets):
                raise FileExistsError("an exact output image already exists")
        else:
            prefix = self._prefix(filename_prefix, subfolder)
            full_folder, filename, counter, saved_subfolder, _ = (
                folder_paths.get_save_image_path(
                    prefix, folder_paths.get_output_directory(),
                    value[0].shape[1], value[0].shape[0]))
            for batch_number in range(batch_size):
                batch_name = filename.replace(
                    "%batch_num%", str(batch_number))
                file = f"{batch_name}_{counter:05}_{default_suffix}"
                logical = (
                    f"{str(saved_subfolder).replace(os.sep, '/')}/{file}"
                    if saved_subfolder else file)
                requested.append(self._logical_output_target(
                    output_dir, logical, allowed_suffixes))
                counter += 1

        results = []
        png_metadata = ImageSaveHelper._create_png_metadata(metadata_owner)
        if a1111_text is not None and pil_format == "PNG":
            if png_metadata is None:
                from PIL.PngImagePlugin import PngInfo
                png_metadata = PngInfo()
            png_metadata.add_text("parameters", a1111_text)
        for image, (target, _logical, file, saved_subfolder) in zip(
                value, requested):
            array = np.clip(
                255.0 * image.detach().cpu().numpy(), 0, 255
            ).astype(np.uint8)
            if array.shape[-1] == 1:
                array = array[..., 0]
            rendered = PILImage.fromarray(array)
            options: dict[str, Any] = {}
            if pil_format in {"JPEG", "JPEG2000"} and rendered.mode != "RGB":
                rendered = rendered.convert("RGB")
            if pil_format in {"PNG", "GIF"}:
                if png_metadata is not None:
                    options["pnginfo"] = png_metadata
                options["optimize"] = optimize
                if pil_format == "PNG":
                    options["compress_level"] = level
            elif pil_format == "JPEG":
                options.update(
                    quality=quality, optimize=optimize, subsampling=0)
            elif pil_format in {"WEBP", "AVIF"}:
                options.update(
                    quality=quality, lossless=lossless, optimize=optimize)
            elif pil_format == "JPEG2000":
                options["irreversible"] = not lossless
            elif pil_format == "TIFF":
                options["optimize"] = optimize

            if pil_format in {"WEBP", "AVIF", "JPEG2000", "TIFF"}:
                exif = ImageSaveHelper._create_webp_metadata(
                    rendered, metadata_owner)
                if pil_format == "WEBP":
                    exif = _add_a1111_exif(exif, a1111_exif)
                if len(exif):
                    options["exif"] = exif.tobytes()
            if pil_format == "JPEG":
                # JPEG has a hard one-segment EXIF ceiling. Trusted workflow
                # data can legitimately exceed it, so degrade metadata without
                # failing the user's image save: prompt first, then broker
                # workflow/extra data, then pack metadata. PNG/WebP keep their
                # full metadata behavior.
                owners = [metadata_owner]
                if bool(save_metadata):
                    owners.extend((
                        _image_metadata_owner(
                            None,
                            self._extra_pnginfo,
                            include_execution_metadata=True,
                            extra_metadata=extra_metadata,
                        ),
                        _image_metadata_owner(
                            None,
                            None,
                            include_execution_metadata=False,
                            extra_metadata=extra_metadata,
                        ),
                    ))
                owners.append(None)
                for owner in owners:
                    try:
                        jpeg_options = dict(options)
                        if owner is not None:
                            exif = ImageSaveHelper._create_webp_metadata(
                                rendered.copy(), owner)
                        else:
                            exif = rendered.copy().getexif()
                        exif = _add_a1111_exif(exif, a1111_exif)
                        if len(exif):
                            jpeg_options["exif"] = exif.tobytes()
                        self._save_pil_exclusive(
                            rendered, target, pil_format, jpeg_options)
                        break
                    except ValueError as error:
                        if "exif data is too long" not in str(error).lower():
                            raise
                else:  # the final no-EXIF attempt should make this unreachable
                    raise RuntimeError("JPEG image could not be encoded")
            else:
                self._save_pil_exclusive(
                    rendered, target, pil_format, options)
            results.append(SavedResult(
                file, saved_subfolder, FolderType.output))
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

    async def write_text(
        self, text: str, filename: str, folder: str = "output",
        mode: str = "overwrite", insert_newline: bool = False,
    ) -> str:
        """Write a specifically named text artifact inside output or temp."""
        import folder_paths

        roots = {
            "output": folder_paths.get_output_directory,
            "temp": folder_paths.get_temp_directory,
        }
        if folder not in roots:
            raise ValueError("text writes are limited to output or temp")
        if mode not in {"append", "overwrite", "new_only"}:
            raise ValueError("text write mode must be append, overwrite, or new_only")
        if not isinstance(insert_newline, bool):
            raise TypeError("insert_newline must be a boolean")
        value = str(text)
        if len(value.encode("utf-8")) > 16 * 1024 * 1024:
            raise ValueError("text output exceeds the 16 MiB limit")

        relative = os.path.normpath(str(filename))
        if (os.path.isabs(relative) or relative in ("", ".", os.pardir)
                or relative.startswith(os.pardir + os.sep)):
            raise ValueError("text filename must stay inside its output folder")
        self._extension(os.path.splitext(relative)[1])
        root = os.path.realpath(os.path.abspath(roots[folder]()))
        target = os.path.realpath(os.path.abspath(os.path.join(root, relative)))
        if os.path.commonpath((root, target)) != root:
            raise ValueError("text filename escapes its output folder")
        os.makedirs(os.path.dirname(target), exist_ok=True)

        if mode == "new_only":
            with open(target, "x", encoding="utf-8") as stream:
                stream.write(value)
        elif mode == "append":
            has_content = os.path.isfile(target) and os.path.getsize(target) > 0
            with open(target, "a", encoding="utf-8") as stream:
                if has_content and insert_newline:
                    stream.write("\n")
                stream.write(value)
        else:
            with open(target, "w", encoding="utf-8") as stream:
                stream.write(value)
        return relative.replace(os.sep, "/")

    async def save_workflow_json(
        self, filename: str, mode: str = "new_only",
    ) -> str:
        """Write the broker-owned active workflow without revealing it.

        The guest chooses only the confined logical output name and collision
        mode.  Prompt/workflow metadata remains on the trusted side.
        """
        import json

        if not isinstance(self._extra_pnginfo, dict):
            raise ValueError("this execution has no workflow metadata")
        workflow = self._extra_pnginfo.get("workflow")
        if not isinstance(workflow, dict):
            raise ValueError("this execution has no workflow object")
        try:
            encoded = json.dumps(
                workflow, ensure_ascii=False, allow_nan=False, indent=2)
        except (TypeError, ValueError) as error:
            raise ValueError("workflow metadata is not JSON-compatible") from error
        if not str(filename).lower().endswith(".json"):
            raise ValueError("workflow sidecars must use the .json extension")
        return await self.write_text(
            encoded, filename=str(filename), folder="output", mode=str(mode))

    @staticmethod
    def _latent_preview(samples: Any, preview_method: str):
        """Render the sender's identifying preview without loading a model."""
        import numpy as np
        import torch
        from PIL import Image as PILImage
        import comfy.latent_formats as latent_formats
        from latent_preview import Latent2RGBPreviewer

        formats = {
            "Latent2RGB-FLUX.1": latent_formats.Flux,
            "Latent2RGB-SDXL": latent_formats.SDXL,
            "Latent2RGB-SD15": latent_formats.SD15,
            "Latent2RGB-SD3": latent_formats.SD3,
            "Latent2RGB-SD-X4": latent_formats.SD_X4,
            "Latent2RGB-Playground-2.5": latent_formats.SDXL_Playground_2_5,
            "Latent2RGB-SC-Prior": latent_formats.SC_Prior,
            "Latent2RGB-SC-B": latent_formats.SC_B,
            "Latent2RGB-LTXV": latent_formats.LTXV,
            # Impact's upstream implementation falls back to a linear preview
            # for these labels when no approximate decoder is wired.
            "TAEF1": latent_formats.Flux,
            "TAESDXL": latent_formats.SDXL,
            "TAESD15": latent_formats.SD15,
            "TAESD3": latent_formats.SD3,
        }
        constructor = formats.get(str(preview_method))
        if constructor is None:
            allowed = ", ".join(sorted(formats))
            raise ValueError(
                f"unknown latent preview method {preview_method!r}; "
                f"choose one of {allowed}")
        latent_format = constructor()
        try:
            previewer = Latent2RGBPreviewer(
                latent_format.latent_rgb_factors,
                getattr(latent_format, "latent_rgb_factors_bias", None),
                getattr(latent_format, "latent_rgb_factors_reshape", None),
            )
            image = previewer.decode_latent_to_preview(samples)
        except Exception:
            # Some third-party latent shapes do not have a matching published
            # matrix.  The preview is identification only; keep sharing the
            # latent and render a bounded normalized RGB projection.
            value = samples
            if value.ndim == 5:
                value = value[:, :, 0]
            value = value[0].detach().float().cpu()
            if value.shape[0] < 3:
                value = value.expand(3, *value.shape[1:])
            value = value[:3].movedim(0, -1)
            low, high = value.amin(), value.amax()
            value = (value - low) / (high - low) if high > low else value * 0
            image = PILImage.fromarray(
                value.mul(255).clamp(0, 255).to(torch.uint8).numpy())
        minimum, maximum = min(image.size), max(image.size)
        scale = min(1.0, 256.0 / max(1, maximum))
        if minimum * scale < 128:
            scale = 128.0 / max(1, minimum)
        size = tuple(max(1, int(round(axis * scale))) for axis in image.size)
        if size != image.size:
            image = image.resize(size, resample=PILImage.Resampling.NEAREST)
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    async def save_latent(
        self, latent: LatentRef,
        filename_prefix: str = "latents/LatentSender",
        preview_method: str = "Latent2RGB-SDXL",
    ) -> dict:
        """Save a temporary, safely loadable latent plus a visual preview."""
        import os
        import torch
        import comfy.utils
        import folder_paths
        from ._io import FolderType
        from ._ui import ImageSaveHelper, SavedImages

        if not isinstance(latent, Ref) or latent.kind != "LATENT":
            raise TypeError("save_latent requires a LATENT ref")
        value = await current_runtime().refs.resolve(latent)
        if not isinstance(value, dict) or "samples" not in value:
            raise ValueError("latent value has no samples tensor")
        samples = value["samples"]
        if not isinstance(samples, torch.Tensor) or samples.ndim not in (4, 5):
            raise ValueError("latent samples must be a 4D or 5D tensor")
        prefix = self._prefix(filename_prefix, "")
        temp_dir = folder_paths.get_temp_directory()
        full_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(prefix, temp_dir))
        file = f"{filename}_{counter:05}_.latent"
        target = os.path.abspath(os.path.join(full_folder, file))
        temp_root = os.path.abspath(temp_dir)
        if os.path.commonpath((temp_root, target)) != temp_root:
            raise ValueError("latent target escapes the temp directory")
        comfy.utils.save_torch_file({
            "latent_tensor": samples.detach().contiguous().cpu(),
            "latent_format_version_0": torch.tensor([]),
        }, target)
        artifact = {
            "filename": file,
            "subfolder": subfolder,
            "type": "temp",
        }
        preview = self._latent_preview(samples, str(preview_method))
        previews = ImageSaveHelper.save_images(
            preview,
            f"{prefix}_preview",
            FolderType.temp,
            self._metadata_owner,
            4,
        )
        return {
            "latents": [artifact],
            "images": SavedImages(previews).as_dict()["images"],
            "artifact": artifact,
        }

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

    @staticmethod
    def _media_pixels(value: Any, operation: str):
        import torch

        if (not isinstance(value, torch.Tensor) or value.ndim != 4
                or len(value) == 0 or value.shape[-1] < 3):
            raise TypeError(f"{operation} needs non-empty BHWC image frames")
        return value

    def _media_target(
        self, pixels: Any, filename_prefix: str, extension: str,
        save_output: bool,
    ) -> tuple[str, str, str, Any]:
        import folder_paths
        from ._io import FolderType

        extension = str(extension).lower().lstrip(".")
        if not extension or not extension.isalnum():
            raise ValueError("media extension must be alphanumeric")
        folder_type = FolderType.output if bool(save_output) else FolderType.temp
        output_dir = os.path.abspath(
            folder_paths.get_output_directory()
            if folder_type == FolderType.output
            else folder_paths.get_temp_directory())
        prefix = self._prefix(filename_prefix, "")
        full_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(
                prefix, output_dir, pixels.shape[2], pixels.shape[1]))
        while True:
            file = f"{filename}_{counter:05}_.{extension}"
            target = os.path.abspath(os.path.join(full_folder, file))
            if os.path.commonpath((output_dir, target)) != output_dir:
                raise ValueError("media target escapes its managed directory")
            if not os.path.exists(target):
                return target, file, subfolder, folder_type
            counter += 1

    def _media_metadata(self, enabled: bool = True) -> Optional[dict[str, Any]]:
        from comfy.cli_args import args

        if not enabled or args.disable_metadata:
            return None
        values = {}
        if self._metadata_owner.hidden.extra_pnginfo is not None:
            values.update(self._metadata_owner.hidden.extra_pnginfo)
        if self._metadata_owner.hidden.prompt is not None:
            values["prompt"] = self._metadata_owner.hidden.prompt
        return values or None

    async def save_animation(
        self, images: ImageRef, fps: float = 8.0,
        filename_prefix: str = "animation/ComfyUI",
        format: str = "webp", loop_count: int = 0,
        lossless: bool = True, quality: int = 90,
        save_output: bool = True,
    ) -> dict:
        import json
        import math

        import numpy as np
        from PIL import Image as PILImage
        from ._ui import SavedImages, SavedResult

        rate = float(fps)
        if not math.isfinite(rate) or not 0.01 <= rate <= 1000.0:
            raise ValueError("animation fps must be finite and in [0.01, 1000]")
        loops = int(loop_count)
        if loops != loop_count or not 0 <= loops <= 100:
            raise ValueError("animation loop_count must be an integer in [0, 100]")
        quality_value = int(quality)
        if quality_value != quality or not 0 <= quality_value <= 100:
            raise ValueError("animation quality must be an integer in [0, 100]")
        image_format = str(format).lower()
        if image_format not in ("gif", "webp"):
            raise ValueError("animation format must be 'gif' or 'webp'")

        pixels = self._media_pixels(
            await current_runtime().refs.resolve(images), "save_animation")
        target, file, subfolder, folder_type = self._media_target(
            pixels, filename_prefix, image_format, bool(save_output))
        frames = []
        for value in pixels:
            array = np.clip(
                value.detach().cpu().float().numpy() * 255.0,
                0, 255).astype(np.uint8)
            frames.append(PILImage.fromarray(
                array[..., :4] if array.shape[-1] >= 4 else array[..., :3],
                mode="RGBA" if array.shape[-1] >= 4 else "RGB"))
        kwargs: dict[str, Any] = {
            "save_all": True,
            "append_images": frames[1:],
            "duration": max(1, round(1000.0 / rate)),
            "loop": loops,
        }
        metadata = self._media_metadata()
        if image_format == "gif":
            kwargs["disposal"] = 2
            if metadata is not None:
                kwargs["comment"] = json.dumps(
                    metadata, separators=(",", ":"), default=str
                ).encode("utf-8")[:65500]
        else:
            kwargs.update({
                "lossless": bool(lossless),
                "quality": quality_value,
                "method": 4,
            })
            if metadata is not None:
                exif = frames[0].getexif()
                exif[0x0110] = "prompt:" + json.dumps(
                    metadata, separators=(",", ":"), default=str)
                kwargs["exif"] = exif
        try:
            frames[0].save(target, format=image_format.upper(), **kwargs)
        except BaseException:
            try:
                os.unlink(target)
            except FileNotFoundError:
                pass
            raise
        return SavedImages([
            SavedResult(file, subfolder, folder_type),
        ], is_animated=len(frames) > 1).as_dict()

    async def save_image_sequence(
        self, images: ImageRef,
        filename_prefix: str = "sequence/ComfyUI",
        format: str = "png", bit_depth: int = 8,
        save_output: bool = True,
    ) -> dict:
        import av
        import folder_paths
        import numpy as np
        from PIL import Image as PILImage
        from ._io import FolderType
        from ._ui import ImageSaveHelper, SavedImages, SavedResult

        if str(format).lower() != "png":
            raise ValueError("image-sequence format must be 'png'")
        depth = int(bit_depth)
        if depth != bit_depth or depth not in (8, 16):
            raise ValueError("PNG sequence bit_depth must be 8 or 16")
        pixels = self._media_pixels(
            await current_runtime().refs.resolve(images),
            "save_image_sequence")
        folder_type = FolderType.output if bool(save_output) else FolderType.temp
        output_dir = os.path.abspath(
            folder_paths.get_output_directory()
            if folder_type == FolderType.output
            else folder_paths.get_temp_directory())
        prefix = self._prefix(filename_prefix, "")
        full_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(
                prefix, output_dir, pixels.shape[2], pixels.shape[1]))
        while True:
            stem = f"{filename}_{counter:05}_"
            first_target = os.path.abspath(os.path.join(
                full_folder, f"{stem}001.png"))
            if os.path.commonpath((output_dir, first_target)) != output_dir:
                raise ValueError("image sequence escapes its managed directory")
            if not os.path.exists(first_target):
                break
            counter += 1
        results = []
        created = []
        metadata = ImageSaveHelper._create_png_metadata(self._metadata_owner)
        try:
            for index, value in enumerate(pixels, 1):
                file = f"{stem}{index:03d}.png"
                target = os.path.abspath(os.path.join(full_folder, file))
                if os.path.commonpath((output_dir, target)) != output_dir:
                    raise ValueError("image sequence escapes its managed directory")
                array = np.clip(
                    value.detach().cpu().float().numpy()
                    * (65535.0 if depth == 16 else 255.0),
                    0, 65535 if depth == 16 else 255)
                array = array[..., :3].astype(
                    np.uint16 if depth == 16 else np.uint8)
                if depth == 8:
                    PILImage.fromarray(array, mode="RGB").save(
                        target, pnginfo=metadata, compress_level=4)
                else:
                    with av.open(target, mode="w", format="image2") as output:
                        stream = output.add_stream("png", rate=1)
                        stream.width = int(array.shape[1])
                        stream.height = int(array.shape[0])
                        stream.pix_fmt = "rgb48be"
                        frame = av.VideoFrame.from_ndarray(
                            array, format="rgb48le")
                        for packet in stream.encode(frame):
                            output.mux(packet)
                        for packet in stream.encode(None):
                            output.mux(packet)
                created.append(target)
                results.append(SavedResult(file, subfolder, folder_type))
        except BaseException:
            for target in created:
                try:
                    os.unlink(target)
                except FileNotFoundError:
                    pass
            raise
        pattern = f"{stem}%03d.png"
        return SavedImages(results).as_dict() | {"pattern": pattern}

    async def save_video(
        self, images: ImageRef, audio: Optional[AudioRef] = None,
        fps: float = 25.0, filename_prefix: str = "video/ComfyUI",
        format: str = "auto", codec: str = "auto",
        encoder_options: Optional[dict[str, Any]] = None,
        loop_count: int = 0, bit_depth: int = 8,
        save_output: bool = True, save_metadata: bool = True,
    ) -> dict:
        import json
        import math
        from fractions import Fraction

        import av
        import numpy as np
        import torch
        from ._ui import PreviewVideo, SavedResult

        rate = float(fps)
        if not math.isfinite(rate) or not 0.0 < rate <= 999.0:
            raise ValueError("video fps must be finite and in (0, 999]")
        loops = int(loop_count)
        if loops != loop_count or not 0 <= loops <= 100:
            raise ValueError("video loop_count must be an integer in [0, 100]")
        depth = int(bit_depth)
        if depth != bit_depth or depth not in (8, 16):
            raise ValueError("video bit_depth must be 8 or 16")
        if encoder_options is None:
            options: dict[str, Any] = {}
        elif type(encoder_options) is dict:
            options = dict(encoder_options)
        else:
            raise TypeError("video encoder_options must be a dictionary")
        allowed_option_names = frozenset({
            "pixel_format", "crf", "bitrate_kbps", "profile", "level",
            "coder", "context", "gop_size", "slices", "slice_crc",
        })
        unknown = set(options) - allowed_option_names
        if unknown:
            raise ValueError(
                "unsupported video encoder option(s): "
                + ", ".join(sorted(map(str, unknown))))

        containers = {
            "mp4": ("mp4", "mp4"),
            "webm": ("webm", "webm"),
            "mkv": ("matroska", "mkv"),
            "matroska": ("matroska", "mkv"),
            "mov": ("mov", "mov"),
        }
        codecs = {
            "h264": ("libx264", {"mp4"}, "yuv420p", {"yuv420p", "yuv420p10le"}),
            "hevc": ("libx265", {"mp4"}, "yuv420p10le", {"yuv420p", "yuv420p10le"}),
            "av1": ("libsvtav1", {"webm"}, "yuv420p", {"yuv420p", "yuv420p10le"}),
            "vp9": ("libvpx-vp9", {"webm"}, "yuv420p", {"yuv420p", "yuva420p"}),
            "prores": ("prores_ks", {"mov"}, "yuv422p10le", {"yuv422p10le", "yuv444p10le", "yuva444p10le"}),
            "ffv1": ("ffv1", {"mkv"}, "rgba64le", {
                "rgba64le", "bgra", "yuv420p", "yuv422p", "yuv444p",
                "yuva420p", "yuva422p", "yuva444p", "yuv420p10le",
                "yuv422p10le", "yuv444p10le", "yuv420p12le",
                "yuv422p12le", "yuv444p12le", "yuv420p14le",
                "yuv422p14le", "yuv444p14le", "yuv420p16le",
                "yuv422p16le", "yuv444p16le", "gray", "gray10le",
                "gray12le", "gray16le",
            }),
            "h264_nvenc": ("h264_nvenc", {"mp4"}, "yuv420p", {"yuv420p", "p010le"}),
            "hevc_nvenc": ("hevc_nvenc", {"mp4"}, "yuv420p", {"yuv420p", "p010le"}),
            "av1_nvenc": ("av1_nvenc", {"mp4"}, "yuv420p", {"yuv420p", "p010le"}),
        }
        codec_name = str(codec).lower()
        container_name = str(format).lower()
        if codec_name == "auto":
            codec_name = {
                "webm": "vp9", "mkv": "ffv1", "matroska": "ffv1",
                "mov": "prores",
            }.get(container_name, "h264")
        if codec_name not in codecs:
            raise ValueError(f"unsupported video codec {codec!r}")
        if container_name == "auto":
            container_name = {
                "av1": "webm", "vp9": "webm", "ffv1": "mkv",
                "prores": "mov",
            }.get(codec_name, "mp4")
        if container_name not in containers:
            raise ValueError(f"unsupported video format {format!r}")
        av_codec, compatible, default_pixel_format, allowed_pixel_formats = (
            codecs[codec_name])
        normalized_container = "mkv" if container_name == "matroska" else container_name
        if normalized_container not in compatible:
            raise ValueError(
                f"video codec {codec_name!r} is not valid in "
                f"{normalized_container!r}")
        pixel_format = str(options.get(
            "pixel_format", default_pixel_format)).lower()
        if pixel_format not in allowed_pixel_formats:
            raise ValueError(
                f"pixel format {pixel_format!r} is not permitted for "
                f"codec {codec_name!r}")

        def integer_option(name: str, minimum: int, maximum: int) -> Optional[int]:
            if name not in options:
                return None
            value = int(options[name])
            if value != options[name] or not minimum <= value <= maximum:
                raise ValueError(
                    f"video encoder option {name} must be an integer in "
                    f"[{minimum}, {maximum}]")
            return value

        crf = integer_option("crf", 0, 100)
        bitrate_kbps = integer_option("bitrate_kbps", 1, 999000)
        profile = str(options.get("profile", ""))
        if profile and codec_name != "prores":
            raise ValueError("video profile is currently supported only for ProRes")
        profile_values = {"lt": 1, "standard": 2, "hq": 3, "4444": 4, "4444xq": 5}
        if profile and profile not in profile_values:
            raise ValueError("unknown ProRes profile")
        ffv1_names = {"level", "coder", "context", "gop_size", "slices", "slice_crc"}
        if set(options) & ffv1_names and codec_name != "ffv1":
            raise ValueError("FFV1 tuning options require the ffv1 codec")
        level = integer_option("level", 0, 3)
        coder = integer_option("coder", 0, 2)
        context_model = integer_option("context", 0, 1)
        gop_size = integer_option("gop_size", 1, 300)
        slices = integer_option("slices", 1, 30)
        if slices is not None and slices not in {4, 6, 9, 12, 16, 20, 24, 30}:
            raise ValueError("FFV1 slices must be one of 4, 6, 9, 12, 16, 20, 24, 30")
        slice_crc = None
        if "slice_crc" in options:
            if type(options["slice_crc"]) is not bool:
                raise TypeError("FFV1 slice_crc must be a boolean")
            slice_crc = options["slice_crc"]

        rt = current_runtime()
        pixels = self._media_pixels(
            await rt.refs.resolve(images), "save_video")
        total_frames = len(pixels) * (loops + 1)
        if total_frames > 1_000_000:
            raise ValueError("video output is limited to 1,000,000 encoded frames")
        audio_value = None
        if audio is not None:
            audio_value = await rt.refs.resolve(audio)
            if (not isinstance(audio_value, dict)
                    or "waveform" not in audio_value
                    or "sample_rate" not in audio_value):
                raise TypeError("save_video audio must contain waveform and sample_rate")

        av_format, extension = containers[container_name]
        target, file, subfolder, folder_type = self._media_target(
            pixels, filename_prefix, extension, bool(save_output))
        metadata = self._media_metadata(bool(save_metadata))
        try:
            try:
                av.codec.Codec(av_codec, "w")
            except Exception as exc:
                raise RuntimeError(
                    f"video encoder {av_codec!r} is unavailable on this host"
                ) from exc
            open_options = (
                {"movflags": "use_metadata_tags"}
                if normalized_container == "mp4" else None)
            with av.open(
                target, mode="w", format=av_format,
                options=open_options,
            ) as output:
                if metadata is not None:
                    for key, value in metadata.items():
                        # Match ComfyUI's VideoFromComponents contract: every
                        # metadata value is a JSON value, including strings.
                        output.metadata[str(key)] = json.dumps(
                            value, default=str)
                frame_rate = Fraction(round(rate * 1000), 1000)
                video_stream = output.add_stream(av_codec, rate=frame_rate)
                width = int(pixels.shape[2])
                height = int(pixels.shape[1])
                alignment = 2 if any(
                    marker in pixel_format
                    for marker in ("420", "422", "p010")) else 1
                encoded_width = width + (-width % alignment)
                encoded_height = height + (-height % alignment)
                video_stream.width = encoded_width
                video_stream.height = encoded_height
                video_stream.pix_fmt = pixel_format
                stream_options = {}
                if crf is not None:
                    stream_options["crf"] = str(crf)
                if profile:
                    stream_options["profile"] = str(profile_values[profile])
                for name, value in (
                    ("level", level), ("coder", coder),
                    ("context", context_model), ("g", gop_size),
                    ("slices", slices),
                ):
                    if value is not None:
                        stream_options[name] = str(value)
                if slice_crc is not None:
                    stream_options["slicecrc"] = "1" if slice_crc else "0"
                if stream_options:
                    video_stream.options = stream_options
                if bitrate_kbps is not None:
                    video_stream.bit_rate = bitrate_kbps * 1000

                audio_stream = None
                waveform = None
                if audio_value is not None:
                    sample_rate = int(audio_value["sample_rate"])
                    if not 8000 <= sample_rate <= 192000:
                        raise ValueError("audio sample_rate must be in [8000, 192000]")
                    waveform = audio_value["waveform"]
                    if (not isinstance(waveform, torch.Tensor)
                            or waveform.ndim != 3 or len(waveform) == 0):
                        raise TypeError("audio waveform must have shape [batch, channels, samples]")
                    waveform = waveform[0].detach().cpu().float()
                    channels = int(waveform.shape[0])
                    layouts = {
                        1: "mono", 2: "stereo", 3: "3.0", 4: "quad",
                        5: "5.0", 6: "5.1", 7: "6.1", 8: "7.1",
                    }
                    if channels not in layouts:
                        raise ValueError("audio must contain between 1 and 8 channels")
                    audio_codec = {
                        "webm": "libopus", "mkv": "flac", "mov": "pcm_s16le",
                    }.get(normalized_container, "aac")
                    audio_stream = output.add_stream(
                        audio_codec, rate=sample_rate, layout=layouts[channels])

                wants_alpha = pixel_format.startswith(("rgba", "bgra", "yuva"))
                for _cycle in range(loops + 1):
                    for value in pixels:
                        array = value.detach().cpu().float().numpy()
                        if wants_alpha:
                            if array.shape[-1] < 4:
                                alpha = np.ones((*array.shape[:2], 1), dtype=array.dtype)
                                array = np.concatenate((array[..., :3], alpha), axis=-1)
                            else:
                                array = array[..., :4]
                        else:
                            array = array[..., :3]
                        if encoded_width != width or encoded_height != height:
                            array = np.pad(
                                array,
                                ((0, encoded_height - height),
                                 (0, encoded_width - width), (0, 0)),
                                mode="edge")
                        maximum = 65535.0 if depth == 16 else 255.0
                        array = np.clip(array * maximum, 0, maximum).astype(
                            np.uint16 if depth == 16 else np.uint8)
                        source_format = (
                            "rgba64le" if wants_alpha else "rgb48le"
                        ) if depth == 16 else (
                            "rgba" if wants_alpha else "rgb24")
                        frame = av.VideoFrame.from_ndarray(
                            np.ascontiguousarray(array), format=source_format)
                        frame = frame.reformat(
                            width=encoded_width, height=encoded_height,
                            format=pixel_format)
                        for packet in video_stream.encode(frame):
                            output.mux(packet)
                for packet in video_stream.encode(None):
                    output.mux(packet)

                if audio_stream is not None and waveform is not None:
                    sample_rate = int(audio_value["sample_rate"])
                    required = math.ceil(total_frames * sample_rate / rate)
                    if waveform.shape[1] < required:
                        waveform = torch.nn.functional.pad(
                            waveform, (0, required - waveform.shape[1]))
                    else:
                        waveform = waveform[:, :required]
                    audio_frame = av.AudioFrame.from_ndarray(
                        waveform.contiguous().numpy(), format="fltp",
                        layout=audio_stream.layout.name)
                    audio_frame.sample_rate = sample_rate
                    audio_frame.pts = 0
                    for packet in audio_stream.encode(audio_frame):
                        output.mux(packet)
                    for packet in audio_stream.encode(None):
                        output.mux(packet)
        except BaseException:
            try:
                os.unlink(target)
            except FileNotFoundError:
                pass
            raise
        return PreviewVideo([
            SavedResult(file, subfolder, folder_type),
        ]).as_dict()


class _InProcessGraph:
    def __init__(self, current_node_id: str, prompt: Any = None,
                 extra_pnginfo: Any = None, dynamic_prompt: Any = None) -> None:
        self._current_node_id = str(current_node_id)
        self._prompt = prompt if isinstance(prompt, dict) else {}
        self._workflow = (
            extra_pnginfo.get("workflow", {})
            if isinstance(extra_pnginfo, dict) else {})
        self._dynamic_prompt = dynamic_prompt

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

    async def current_node_id(self) -> str:
        """Return only this execution's node id, never the surrounding prompt."""
        return self._current_node_id

    async def input_label(
        self, input_name: str, default: str = "",
    ) -> str:
        """Return one display label from this node's workflow metadata."""
        input_name = str(input_name)
        default = str(default)
        if len(input_name) > 256 or len(default) > 256:
            raise ValueError("graph input names and labels are limited to 256 characters")
        candidates = list(self._workflow.get("nodes", []))
        definitions = self._workflow.get("definitions", {})
        for subgraph in definitions.get("subgraphs", []):
            candidates.extend(subgraph.get("nodes", []))
        wanted = self._current_node_id.rsplit(":", 1)[-1]
        matches = [
            node for node in candidates
            if str(node.get("id")) in {self._current_node_id, wanted}
        ]
        if len(matches) != 1:
            return default
        for slot in matches[0].get("inputs", []):
            if slot.get("name") == input_name:
                label = slot.get("label")
                if isinstance(label, str) and len(label) <= 256:
                    return label
                return default
        return default

    def _require_dynamic_prompt(self):
        if self._dynamic_prompt is None:
            raise RuntimeError(
                "graph expansion requires ComfyUI's active dynamic prompt")
        return self._dynamic_prompt

    @staticmethod
    def _expansion_link(value: Any, created: dict[str, Any]) -> Any:
        if isinstance(value, dict) and set(value) == {"node", "output"}:
            node_id = str(value["node"])
            if node_id not in created:
                raise KeyError(f"expansion references unknown node {node_id!r}")
            index = int(value["output"])
            if not 0 <= index <= 1024:
                raise ValueError("expansion output index is out of range")
            return created[node_id].out(index)
        if isinstance(value, dict):
            return {
                key: _InProcessGraph._expansion_link(item, created)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(
                _InProcessGraph._expansion_link(item, created)
                for item in value)
        return value

    async def expand_nodes(
        self, nodes: list[dict[str, Any]], outputs: list[dict[str, Any]], *,
        _external_node_types: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        """Build a bounded declarative graph expansion.

        Normal specs remain confined to the caller's pack namespace. A spec
        with ``clone_input`` may clone only the producer already wired directly
        to that named input on the current node. ``CreateList`` is the sole
        core node allowed in an expansion; it is a side-effect-free collection
        primitive and lets pack-side orchestration collect cloned outputs.

        ``_external_node_types`` is a trusted-host policy input, not part of the
        guest API.  The Secure Nodes transport fills it only from exact
        ``graph.expand.external:<node type>`` declarations after verifying that
        each used target is another converted-pack proxy.  Keeping the policy
        here lets pack code retain orchestration while preventing it from
        selecting arbitrary host or legacy nodes.
        """
        dynprompt = self._require_dynamic_prompt()
        if not isinstance(nodes, list) or not 1 <= len(nodes) <= 128:
            raise ValueError("graph expansion must contain 1..128 nodes")
        if not isinstance(outputs, list) or len(outputs) > 128:
            raise ValueError("graph expansion has too many outputs")
        if (not isinstance(_external_node_types, frozenset)
                or len(_external_node_types) > 64
                or not all(
                    isinstance(item, str)
                    and 1 <= len(item) <= 256
                    and "\x00" not in item
                    for item in _external_node_types
                )):
            raise ValueError("external expansion policy is invalid")
        current = dynprompt.get_node(self._current_node_id)
        current_type = str(current.get("class_type", ""))
        namespace = current_type.split(" ", 1)[0]
        if not namespace:
            raise ValueError("current node has no expansion namespace")

        from comfy_execution.graph_utils import GraphBuilder, is_link

        graph = GraphBuilder()
        created: dict[str, Any] = {}
        clones: dict[str, tuple[dict[str, Any], str]] = {}
        for spec in nodes:
            if not isinstance(spec, dict):
                raise TypeError("expansion node specs must be dictionaries")
            local_id = str(spec.get("id", ""))
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", local_id):
                raise ValueError(
                    "expansion node ids must be bounded local identifiers")
            if local_id in created:
                raise ValueError(f"duplicate expansion node id {local_id!r}")

            clone_input = spec.get("clone_input")
            if clone_input is not None:
                if set(spec) != {"id", "clone_input"}:
                    raise ValueError(
                        "clone specs accept only id and clone_input")
                clone_input = str(clone_input)
                if not re.fullmatch(r"[A-Za-z0-9_.* -]{1,256}", clone_input):
                    raise ValueError("invalid cloned input name")
                source = current.get("inputs", {}).get(clone_input)
                if not is_link(source):
                    raise ValueError(
                        f"current input {clone_input!r} is not a direct link")
                source_id = str(source[0])
                if source_id == self._current_node_id:
                    raise ValueError("an expansion cannot clone its current node")
                source_node = dynprompt.get_node(source_id)
                class_type = str(source_node.get("class_type", ""))
                if not class_type or len(class_type) > 256:
                    raise ValueError("linked producer has an invalid node type")
                clones[local_id] = (source_node, source_id)
            else:
                class_type = str(spec.get("class_type", ""))
                if (
                    len(class_type) > 256
                    or (
                        not class_type.startswith(namespace + " ")
                        and class_type != "CreateList"
                        and class_type not in _external_node_types
                    )
                ):
                    raise ValueError(
                        "expansions may create only their own pack nodes or "
                        "host-approved converted-pack nodes")
            created[local_id] = graph.node(class_type, local_id)

        for spec in nodes:
            local_id = str(spec["id"])
            target = created[local_id]
            if local_id in clones:
                source_node, source_id = clones[local_id]
                inputs = source_node.get("inputs", {})
                if not isinstance(inputs, dict) or len(inputs) > 256:
                    raise ValueError(
                        "linked producer inputs must be a bounded mapping")
                for name, value in inputs.items():
                    target.set_input(str(name), value)
                target.set_override_display_id(
                    str(dynprompt.get_display_node_id(source_id)))
                continue
            inputs = spec.get("inputs", {})
            if not isinstance(inputs, dict) or len(inputs) > 256:
                raise ValueError("expansion node inputs must be a bounded mapping")
            for name, value in inputs.items():
                name = str(name)
                if not re.fullmatch(r"[A-Za-z0-9_.* -]{1,256}", name):
                    raise ValueError("invalid expansion input name")
                target.set_input(name, self._expansion_link(value, created))

        result = [self._expansion_link(item, created) for item in outputs]
        return {"result": result, "expand": graph.finalize()}

    async def expand_loop(
        self, flow: Any, values: list[Any],
    ) -> dict[str, Any]:
        """Clone the bounded body between a loop opener and this closer."""
        from comfy_execution.graph_utils import GraphBuilder, is_link

        dynprompt = self._require_dynamic_prompt()
        if not is_link(flow) or int(flow[1]) != 0:
            raise ValueError("loop flow must be the opener's raw output-0 link")
        if not isinstance(values, list) or not 1 <= len(values) <= 100:
            raise ValueError("loop expansion requires 1..100 carried values")
        open_node = str(flow[0])
        close_node = self._current_node_id
        upstream: dict[str, list[str]] = {}
        parent_ids: list[str] = []

        def explore_dependencies(node_id: str) -> None:
            node_info = dynprompt.get_node(node_id)
            for value in node_info.get("inputs", {}).values():
                if not is_link(value):
                    continue
                parent_id = str(value[0])
                display_id = str(dynprompt.get_display_node_id(parent_id))
                display_node = dynprompt.get_node(display_id)
                if display_node.get("class_type") not in {
                    "easy forLoopEnd", "easy whileLoopEnd",
                }:
                    parent_ids.append(display_id)
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    explore_dependencies(parent_id)
                upstream[parent_id].append(node_id)

        explore_dependencies(close_node)
        parent_set = set(parent_ids)
        try:
            import nodes as comfy_nodes
            original = dynprompt.get_original_prompt()
            for output_id, node_info in original.items():
                node_class = comfy_nodes.NODE_CLASS_MAPPINGS.get(
                    node_info.get("class_type"))
                if not bool(getattr(node_class, "OUTPUT_NODE", False)):
                    continue
                for value in node_info.get("inputs", {}).values():
                    if not is_link(value) or value[0] not in parent_set:
                        continue
                    for parent_id in tuple(upstream):
                        display_id = str(dynprompt.get_display_node_id(parent_id))
                        if display_id == value[0] and output_id not in upstream[parent_id]:
                            child = str(output_id)
                            if "." in parent_id:
                                parts = parent_id.split(".")
                                parts[-1] = child
                                child = ".".join(parts)
                            upstream[parent_id].append(child)
        except (ImportError, KeyError, TypeError):
            pass

        contained: dict[str, bool] = {}

        def collect(node_id: str) -> None:
            for child_id in upstream.get(node_id, ()):
                if child_id not in contained:
                    contained[child_id] = True
                    collect(child_id)

        collect(open_node)
        contained[open_node] = True
        contained[close_node] = True
        if len(contained) > 512:
            raise ValueError("loop body exceeds the 512-node expansion limit")

        graph = GraphBuilder()
        clones: dict[str, Any] = {}
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            clone_id = "Recurse" if node_id == close_node else node_id
            clone = graph.node(original_node["class_type"], clone_id)
            clone.set_override_display_id(node_id)
            clones[node_id] = clone
        for node_id, clone in clones.items():
            original_node = dynprompt.get_node(node_id)
            for name, value in original_node.get("inputs", {}).items():
                if is_link(value) and str(value[0]) in clones:
                    clone.set_input(name, clones[str(value[0])].out(int(value[1])))
                else:
                    clone.set_input(name, value)
        opener = clones[open_node]
        for index, value in enumerate(values):
            opener.set_input(f"initial_value{index}", value)
        recurse = clones[close_node]
        return {
            "result": [recurse.out(index) for index in range(len(values))],
            "expand": graph.finalize(),
        }

    def _id_for_title(self, title: str) -> Optional[int | str]:
        candidates = list(self._workflow.get("nodes", []))
        definitions = self._workflow.get("definitions", {})
        for subgraph in definitions.get("subgraphs", []):
            candidates.extend(subgraph.get("nodes", []))
        matches = [node.get("id") for node in candidates
                   if node.get("title") == title]
        return matches[0] if len(matches) == 1 else None

    def _id_for_name(self, name: str) -> Optional[int | str]:
        """Resolve the visible type, S&R name, or title of one workflow node."""
        candidates = list(self._workflow.get("nodes", []))
        definitions = self._workflow.get("definitions", {})
        for subgraph in definitions.get("subgraphs", []):
            candidates.extend(subgraph.get("nodes", []))
        matches = []
        for node in candidates:
            visible_name = node.get("type")
            properties = node.get("properties")
            if isinstance(properties, dict):
                search_name = properties.get("Node name for S&R")
                if isinstance(search_name, str) and search_name:
                    visible_name = search_name
            if visible_name == name or node.get("title") == name:
                matches.append(node.get("id"))
        unique = list(dict.fromkeys(matches))
        return unique[0] if len(unique) == 1 else None

    async def widget_values(
        self, node_id: int | str = 0, node_title: str = "",
        node_name: str = "", linked_input: str = "any_input",
    ) -> dict[str, Any]:
        target = None
        if node_title and node_name:
            raise ValueError("choose node_title or node_name, not both")
        if node_title:
            target = self._id_for_title(str(node_title))
            if target is None:
                raise KeyError(f"no unique workflow node titled {node_title!r}")
        elif node_name:
            target = self._id_for_name(str(node_name))
            if target is None:
                raise KeyError(f"no unique workflow node named {node_name!r}")
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
        if self._dynamic_prompt is not None:
            target_id = str(target)
            if self._dynamic_prompt.has_node(target_id):
                values = self._dynamic_prompt.get_node(target_id).get("inputs")
                if isinstance(values, dict):
                    return dict(values)
        key = self._prompt_key(target)
        if key is None:
            raise KeyError(f"node {target!r} is not present in this prompt")
        values = self._prompt.get(key, {}).get("inputs")
        if not isinstance(values, dict):
            raise KeyError(f"node {target!r} has no prompt inputs")
        return dict(values)

    async def block(self, reason: Optional[str] = None) -> Any:
        """Return ComfyUI's branch-local execution blocker."""
        from comfy_execution.graph_utils import ExecutionBlocker

        if reason is not None and not isinstance(reason, str):
            raise TypeError("execution blocker reason must be a string or None")
        if isinstance(reason, str) and len(reason) > 4096:
            raise ValueError("execution blocker reason exceeds 4096 characters")
        return ExecutionBlocker(reason)


class _InProcessProgress:
    def __init__(self, node_id: Optional[str]) -> None:
        self._node_id = node_id

    async def update(self, value: float, total: float,
                     preview: Optional[ImageRef] = None) -> None:
        from comfy.utils import ProgressBar  # lazy

        preview_value = None
        if preview is not None:
            if not isinstance(preview, Ref) or preview.kind != "IMAGE":
                raise TypeError("progress preview must be an IMAGE ref")
            import numpy as np
            import torch
            from PIL import Image
            from comfy.cli_args import args

            image = torch.as_tensor(
                await current_runtime().refs.resolve(preview)).detach()
            if image.ndim == 4:
                if image.shape[0] < 1:
                    raise ValueError("progress preview image batch is empty")
                image = image[0]
            if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
                raise ValueError(
                    "progress preview must have HWC or BHWC image layout")
            if image.shape[0] < 1 or image.shape[1] < 1:
                raise ValueError("progress preview image is empty")
            if image.numel() > 128 * 1024 * 1024:
                raise ValueError("progress preview image is too large")
            image = image.to(device="cpu", dtype=torch.float32)
            image = torch.nan_to_num(image).clamp(0.0, 1.0)
            array = (image * 255.0).round().to(torch.uint8).numpy()
            if array.shape[-1] == 1:
                array = np.squeeze(array, axis=-1)
            preview_value = (
                "PNG",
                Image.fromarray(array),
                args.preview_size,
            )
        pb = ProgressBar(total, node_id=self._node_id)
        pb.update_absolute(value, total, preview=preview_value)


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


class _InProcessExecution:
    """Prompt-scoped execution control for nodes whose purpose is to stop."""

    def __init__(self, prompt_id: str) -> None:
        self._prompt_id = str(prompt_id)

    async def interrupt(self) -> bool:
        from server import PromptServer

        instance = getattr(PromptServer, "instance", None)
        queue = getattr(instance, "prompt_queue", None)
        targeted = getattr(queue, "interrupt_if_running", None)
        if callable(targeted):
            return bool(targeted(self._prompt_id))

        # Embedded/in-process hosts may not own a PromptQueue. In that case
        # this context itself is the only active execution owner.
        import nodes

        nodes.interrupt_processing()
        return True


class _InProcessSystem:
    async def stats(self) -> dict[str, Any]:
        """Return the bounded resource totals already exposed by ComfyUI."""
        import comfy.model_management as model_management

        primary = model_management.get_torch_device()
        cpu = model_management.torch.device("cpu")
        devices = list(model_management.get_all_torch_devices())
        if primary in devices:
            devices = [primary, *(device for device in devices if device != primary)]
        else:
            devices.insert(0, primary)

        entries = []
        for device in devices:
            total, torch_total = model_management.get_total_memory(
                device, torch_total_too=True)
            free, torch_free = model_management.get_free_memory(
                device, torch_free_too=True)
            entries.append({
                "name": model_management.get_torch_device_name(device),
                "type": device.type,
                "index": device.index,
                "vram_total": int(total),
                "vram_free": int(free),
                "torch_vram_total": int(torch_total),
                "torch_vram_free": int(torch_free),
            })
        return {
            "system": {
                "ram_total": int(model_management.get_total_memory(cpu)),
                "ram_free": int(model_management.get_free_memory(cpu)),
            },
            "devices": entries,
        }

    async def monitor(self) -> dict[str, Any]:
        from comfy.system_monitor import get_system_monitor_snapshot

        return get_system_monitor_snapshot()


class _StubDomain:
    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, item: str) -> Any:
        raise NotImplementedError(
            f"ctx.{self._name}.{item} is defined in the SDK contract but not yet "
            f"implemented by the in-process default. Provided by the full SDK / "
            f"overlay."
        )


class _InProcessClosures:
    """Trusted/default execution of the same closed node-closure contract.

    The isolated overlay supplies the prompt-scoped process boundary.  The
    ordinary in-process SDK needs the author surface to remain functional too;
    here the resolver owns the entry and the cloned model owns the callback,
    matching normal ComfyUI custom-node lifetime.
    """

    async def retain(self, kind: str, fn: Callable, *, captures=None):
        from . import _node_closures

        if not callable(fn):
            raise TypeError("a node closure needs a callable")
        spec = _node_closures.get_kind(kind)
        declared = spec.validate_captures(captures)
        resolved = {}
        for name, value in declared.items():
            if isinstance(value, list):
                resolved[name] = [
                    await current_runtime().refs.resolve(item) for item in value
                ]
            elif value is not None:
                resolved[name] = await current_runtime().refs.resolve(value)
        return ClosureRef._wrap(await current_runtime().refs.create(
            "CLOSURE", {
                "kind": kind,
                "fn": fn,
                "captures": resolved,
            },
        ))

    async def attach_model(self, closure: ClosureRef, model: ModelRef):
        if not isinstance(closure, ClosureRef):
            raise TypeError("closure must be a typed CLOSURE ref")
        if not isinstance(model, ModelRef):
            raise TypeError("model must be a typed MODEL ref")
        entry = await current_runtime().refs.resolve(closure)
        kind = entry.get("kind")
        if kind not in {
            "post_cfg", "pre_cfg", "conditioning_selection",
            "conditioning_preprocess",
            "model_input_block", "model_middle_block", "model_output_block",
            "regional_attention",
        }:
            raise ValueError(
                "this node closure kind cannot attach to a model")
        model_obj = await current_runtime().refs.resolve(model)
        fn = entry["fn"]
        if kind == "regional_attention":
            import math
            import torch
            from comfy.model_base import (
                Anima, BaseModel, CosmosPredict2, SDXL, SDXLRefiner,
            )

            if not hasattr(model_obj, "clone") or not hasattr(
                model_obj, "get_model_object"
            ):
                raise TypeError(
                    "regional attention requires a canonical MODEL patcher")
            base_model = getattr(model_obj, "model", None)
            model_type = type(base_model)
            canonical_family = (
                model_type is BaseModel
                or issubclass(model_type, (SDXL, SDXLRefiner))
            )
            cosmos_family = issubclass(model_type, (Anima, CosmosPredict2))
            if not (canonical_family or cosmos_family):
                raise TypeError(
                    "regional attention supports canonical SD/SDXL UNet and "
                    "Anima/Cosmos Predict2 models")
            has_negpip = bool(
                getattr(model_obj, "model_options", {}).get("ppm_negpip"))
            diffusion = model_obj.get_model_object("diffusion_model")
            if canonical_family:
                attention_sites = sum(
                    1 for _name, module in diffusion.named_modules()
                    if getattr(module, "attn2", None) is not None
                )
            else:
                from comfy.ldm.cosmos.predict2 import Attention as CosmosAttention

                attention_sites = sum(
                    1 for name, module in diffusion.named_modules()
                    if isinstance(module, CosmosAttention)
                    and "cross_attn" in name
                )
            if not 1 <= attention_sites <= 256:
                raise TypeError(
                    "canonical UNet regional attention requires 1..256 "
                    "cross-attention sites")
            captures = entry.get("captures", {})
            base_conditioning = captures.get("base_conditioning")
            conditionings = list(captures.get("conditionings", ()))
            masks = list(captures.get("masks", ()))
            if len(conditionings) > 31 or len(masks) != len(conditionings) + 1:
                raise ValueError(
                    "regional attention needs one base mask and one mask per "
                    "extra conditioning, with at most 32 regions")

            def conditioning_row(value, label):
                if not isinstance(value, list) or not value:
                    raise TypeError(
                        f"{label} must contain at least one conditioning row")
                row = value[0]
                if not (
                    isinstance(row, (list, tuple))
                    and len(row) == 2
                    and isinstance(row[0], torch.Tensor)
                    and row[0].ndim == 3
                    and row[0].shape[0] == 1
                    and isinstance(row[1], dict)
                ):
                    raise TypeError(
                        f"{label} does not have a canonical conditioning row")
                strength = row[1].get("strength", 1.0)
                if (
                    isinstance(strength, bool)
                    or not isinstance(strength, (int, float))
                    or not math.isfinite(float(strength))
                ):
                    raise TypeError(f"{label} strength must be finite")
                return row[0], float(strength)

            _base_tensor, base_strength = conditioning_row(
                base_conditioning, "base_conditioning")
            rows = [
                conditioning_row(value, f"conditionings[{index}]")
                for index, value in enumerate(conditionings)
            ]
            conditioning_tensors = [row[0] for row in rows]
            strengths = [row[1] for row in rows]
            for index, mask in enumerate(masks):
                if not (
                    isinstance(mask, torch.Tensor)
                    and mask.ndim == 3
                    and torch.is_floating_point(mask)
                ):
                    raise TypeError(
                        f"masks[{index}] must be a floating [B,H,W] tensor")
            if any(tuple(mask.shape) != tuple(masks[0].shape)
                   for mask in masks[1:]):
                raise ValueError(
                    "regional attention masks must have identical shapes")

            prepared = False
            prepared_device = None
            prepared_dtype = None
            state_key = f"secure_regional_attention_{id(fn)}"

            def call(*args):
                result = fn(*args)
                if isinstance(result, Awaitable):
                    raise TypeError(
                        "regional-attention closures must be synchronous")
                return result

            def prepare(query):
                nonlocal prepared, prepared_device, prepared_dtype
                if prepared:
                    if (
                        str(query.device) != prepared_device
                        or query.dtype != prepared_dtype
                    ):
                        raise TypeError(
                            "regional attention device/dtype changed after "
                            "preparation")
                    return
                result = call(
                    "prepare",
                    [value.to(query) for value in conditioning_tensors],
                    strengths,
                    base_strength,
                    [value.to(query) for value in masks],
                )
                if result is not None:
                    raise TypeError(
                        "regional attention prepare must return None")
                prepared_device = str(query.device)
                prepared_dtype = query.dtype
                prepared = True

            def validate(actual, shape, original, label):
                if not (
                    isinstance(actual, torch.Tensor)
                    and tuple(actual.shape) == tuple(shape)
                    and actual.dtype == original.dtype
                    and str(actual.device) == str(original.device)
                ):
                    raise TypeError(
                        f"regional attention {label} must preserve its "
                        "contracted shape, dtype, and device")

            if cosmos_family:
                import comfy.patcher_extension
                from comfy.ldm.cosmos.predict2 import Attention as CosmosAttention
                from comfy.sampler_helpers import convert_cond
                from comfy.samplers import process_conds

                if not all(hasattr(model_obj, name) for name in (
                    "add_wrapper_with_key", "add_object_patch",
                )):
                    raise TypeError(
                        "Anima/Cosmos MODEL does not expose its canonical "
                        "wrapper and object-patch hooks")
                converted = [convert_cond(value)[0] for value in conditionings]
                key_prefix = f"secure_regional_attention_{id(fn)}"
                conds_key = f"{key_prefix}_conditionings"
                negpip_masks_key = f"{key_prefix}_negpip_masks"
                shape_key = f"{key_prefix}_activation_shape"
                wrapper_key = f"{key_prefix}_wrapper"

                def prepare_cosmos(values, latent):
                    if len(values) != len(conditionings):
                        raise TypeError(
                            "Anima/Cosmos conditioning preparation changed the "
                            "declared region count")
                    reference = values[0] if values else latent
                    if not (
                        isinstance(reference, torch.Tensor)
                        and torch.is_floating_point(reference)
                    ):
                        raise TypeError(
                            "Anima/Cosmos regional attention preparation needs "
                            "floating tensors")
                    projected = [value.to(reference) for value in values]
                    result = call(
                        "prepare_cosmos", projected,
                        [value.to(reference) for value in masks],
                    )
                    if result is not None:
                        raise TypeError(
                            "regional attention Cosmos prepare must return None")
                    return projected

                def sample_wrapper(executor, *args, **kwargs):
                    if len(args) < 7:
                        raise TypeError(
                            "Anima/Cosmos sampler wrapper received an invalid "
                            "call shape")
                    guider, extra_options = args[0], args[2]
                    noise, latent_image, denoise_mask = args[4:7]
                    if not (
                        isinstance(extra_options, dict)
                        and isinstance(latent_image, torch.Tensor)
                    ):
                        raise TypeError(
                            "Anima/Cosmos sampler wrapper received invalid state")
                    prepared_conds = []
                    if converted:
                        seed = extra_options.get("seed")
                        if isinstance(seed, bool) or not isinstance(seed, int):
                            raise TypeError(
                                "Anima/Cosmos sampling seed must be an integer")
                        processed = process_conds(
                            guider.inner_model,
                            noise,
                            {"positive": converted},
                            latent_image.device,
                            latent_image,
                            denoise_mask,
                            seed,
                            latent_shapes=[latent_image.shape],
                        )["positive"]
                        prepared_negpip_masks = []
                        for item in processed:
                            model_conds = item.get("model_conds", {})
                            wrapper = model_conds.get("c_crossattn")
                            tensor = getattr(wrapper, "cond", None)
                            if not (
                                isinstance(tensor, torch.Tensor)
                                and tensor.ndim == 3
                                and tensor.shape[0] == 1
                            ):
                                raise TypeError(
                                    "Anima/Cosmos processed conditioning is "
                                    "not a canonical cross-attention tensor")
                            prepared_conds.append(tensor)
                            mask_wrapper = model_conds.get(
                                "c_ppm_negpip_mask")
                            if mask_wrapper is not None:
                                mask = getattr(mask_wrapper, "cond", None)
                                if not (
                                    isinstance(mask, torch.Tensor)
                                    and mask.ndim == 3
                                    and mask.shape[0] == 1
                                    and mask.shape[2] == 1
                                ):
                                    raise TypeError(
                                        "Anima/Cosmos processed NegPiP mask is "
                                        "not a canonical token sidecar")
                                prepared_negpip_masks.append(mask)
                        if prepared_negpip_masks and len(
                            prepared_negpip_masks
                        ) != len(prepared_conds):
                            raise TypeError(
                                "Anima/Cosmos regional NegPiP needs one sign "
                                "mask per extra conditioning")
                    prepared_conds = prepare_cosmos(
                        prepared_conds, latent_image)
                    model_options = extra_options.get("model_options")
                    if not isinstance(model_options, dict):
                        raise TypeError(
                            "Anima/Cosmos sampling has no model options")
                    transformer_options = dict(
                        model_options.get("transformer_options", {}))
                    transformer_options[conds_key] = prepared_conds
                    if prepared_negpip_masks:
                        transformer_options[negpip_masks_key] = [
                            value.to(latent_image)
                            for value in prepared_negpip_masks
                        ]
                    model_options["transformer_options"] = transformer_options
                    return executor(*args, **kwargs)

                def diffusion_wrapper(executor, *args, **kwargs):
                    if not args or not isinstance(args[0], torch.Tensor):
                        raise TypeError(
                            "Anima/Cosmos diffusion wrapper needs a latent tensor")
                    patch_spatial = getattr(
                        executor.class_obj, "patch_spatial", None)
                    if (
                        isinstance(patch_spatial, bool)
                        or not isinstance(patch_spatial, int)
                        or not 1 <= patch_spatial <= 64
                    ):
                        raise TypeError(
                            "Anima/Cosmos patch_spatial must be in 1..64")
                    latent = args[0]
                    if any(int(value) % patch_spatial
                           for value in latent.shape[-2:]):
                        raise ValueError(
                            "Anima/Cosmos latent dimensions must be divisible "
                            "by patch_spatial")
                    options = dict(kwargs.get("transformer_options", {}))
                    activation_shape = list(latent.shape)
                    activation_shape[-2] //= patch_spatial
                    activation_shape[-1] //= patch_spatial
                    options[shape_key] = tuple(activation_shape)
                    kwargs["transformer_options"] = options
                    return executor(*args, **kwargs)

                def forward_wrapper(previous_forward):
                    def wrapped(
                        query, context=None, rope_emb=None,
                        transformer_options=None,
                    ):
                        options = dict(transformer_options or {})
                        groups = options.get("cond_or_uncond")
                        if not (
                            isinstance(groups, (list, tuple))
                            and 1 <= len(groups) <= 16
                            and all(type(item) is int and item in {0, 1}
                                    for item in groups)
                            and isinstance(query, torch.Tensor)
                            and query.ndim == 3
                            and query.shape[0] % len(groups) == 0
                        ):
                            raise TypeError(
                                "Anima/Cosmos regional attention received "
                                "invalid cond/uncond groups")
                        prepared_conds = options.get(conds_key, [])
                        if context is None:
                            expanded_query, expanded_context = query, None
                            merge_groups = list(groups)
                        else:
                            if not (
                                isinstance(context, torch.Tensor)
                                and context.ndim == 3
                                and context.dtype == query.dtype
                                and str(context.device) == str(query.device)
                            ):
                                raise TypeError(
                                    "Anima/Cosmos context must match query")
                            batch = query.shape[0] // len(groups)
                            region_count = len(prepared_conds) + 1
                            output_batch = batch * sum(
                                1 if group == 1 else region_count
                                for group in groups)
                            token_lengths = [
                                int(value.shape[1]) for value in prepared_conds
                            ]
                            shapes = (
                                (output_batch, int(query.shape[1]),
                                 int(query.shape[2])),
                                (output_batch,
                                 math.lcm(int(context.shape[1]), *token_lengths),
                                 int(context.shape[2])),
                            )
                            if any(shape[0] * shape[1] > 131072
                                   for shape in shapes):
                                raise ValueError(
                                    "Anima/Cosmos regional attention expansion "
                                    "exceeds 131072 token positions")
                            main_negpip_mask = options.get("ppm_negpip_mask")
                            extra_negpip_masks = options.get(negpip_masks_key)
                            composed_negpip = (
                                main_negpip_mask is not None
                                or extra_negpip_masks is not None
                            )
                            if composed_negpip:
                                if not (
                                    isinstance(main_negpip_mask, torch.Tensor)
                                    and main_negpip_mask.ndim == 3
                                    and main_negpip_mask.shape[2] == 1
                                    and isinstance(extra_negpip_masks, list)
                                    and len(extra_negpip_masks)
                                    == len(prepared_conds)
                                    and all(
                                        isinstance(value, torch.Tensor)
                                        and value.ndim == 3
                                        and value.shape[0] == 1
                                        and value.shape[2] == 1
                                        and value.dtype == main_negpip_mask.dtype
                                        and str(value.device)
                                        == str(main_negpip_mask.device)
                                        for value in extra_negpip_masks
                                    )
                                ):
                                    raise TypeError(
                                        "Anima/Cosmos regional NegPiP sidecars "
                                        "are incomplete or incompatible")
                                mask_shape = (
                                    output_batch,
                                    math.lcm(
                                        int(main_negpip_mask.shape[1]),
                                        *(int(value.shape[1])
                                          for value in extra_negpip_masks),
                                    ),
                                    1,
                                )
                                if mask_shape[0] * mask_shape[1] > 131072:
                                    raise ValueError(
                                        "Anima/Cosmos regional NegPiP mask "
                                        "expansion exceeds 131072 positions")
                                result = call(
                                    "pre_cosmos_negpip", query, context,
                                    list(groups), main_negpip_mask,
                                    extra_negpip_masks)
                            else:
                                result = call(
                                    "pre_cosmos", query, context, list(groups))
                            expected_items = 3 if composed_negpip else 2
                            if not (
                                isinstance(result, tuple)
                                and len(result) == expected_items
                            ):
                                raise TypeError(
                                    "Anima/Cosmos regional attention pre returned "
                                    "the wrong tensor tuple")
                            validate(result[0], shapes[0], query, "query")
                            validate(result[1], shapes[1], context, "context")
                            if composed_negpip:
                                validate(
                                    result[2], mask_shape, main_negpip_mask,
                                    "NegPiP sign mask")
                                options["ppm_negpip_mask"] = result[2]
                                expanded_query, expanded_context = result[:2]
                            else:
                                expanded_query, expanded_context = result
                            merge_groups = []
                            for group in groups:
                                merge_groups.extend(
                                    [1] if group == 1
                                    else [0] * region_count)
                        output = previous_forward(
                            expanded_query, expanded_context, rope_emb, options)
                        activation_shape = options.get(shape_key)
                        if not (
                            isinstance(output, torch.Tensor)
                            and output.ndim == 3
                            and isinstance(activation_shape, (list, tuple))
                            and len(activation_shape) >= 2
                            and math.prod(activation_shape[-2:])
                            == output.shape[1]
                        ):
                            raise TypeError(
                                "Anima/Cosmos attention output or geometry is "
                                "invalid")
                        expected_shape = (
                            int(query.shape[0]), int(output.shape[1]),
                            int(output.shape[2]),
                        )
                        merged = call(
                            "post", output, merge_groups,
                            list(activation_shape))
                        validate(
                            merged, expected_shape, output, "blended output")
                        return merged

                    return wrapped

                patched = model_obj.clone()
                patched.add_wrapper_with_key(
                    comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
                    wrapper_key,
                    sample_wrapper,
                )
                patched.add_wrapper_with_key(
                    comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                    wrapper_key,
                    diffusion_wrapper,
                )
                for block_name, module in diffusion.named_modules():
                    if not (
                        isinstance(module, CosmosAttention)
                        and "cross_attn" in block_name
                    ):
                        continue
                    path = f"diffusion_model.{block_name}.forward"
                    patched.add_object_patch(
                        path,
                        forward_wrapper(model_obj.get_model_object(path)),
                    )
                return ModelRef._wrap(await current_runtime().refs.create(
                    "MODEL", patched))

            def attn2_patch(query, key, value, extra_options):
                if not (
                    isinstance(query, torch.Tensor)
                    and isinstance(key, torch.Tensor)
                    and isinstance(value, torch.Tensor)
                    and query.ndim == key.ndim == value.ndim == 3
                    and query.dtype == key.dtype == value.dtype
                    and str(query.device) == str(key.device) == str(value.device)
                ):
                    raise TypeError(
                        "regional attention requires rank-3 tensors with a "
                        "shared dtype and device")
                cond_or_uncond = extra_options.get("cond_or_uncond")
                if not (
                    isinstance(cond_or_uncond, (list, tuple))
                    and 1 <= len(cond_or_uncond) <= 16
                    and all(type(item) is int and item in {0, 1}
                            for item in cond_or_uncond)
                    and query.shape[0] % len(cond_or_uncond) == 0
                ):
                    raise TypeError(
                        "regional attention received invalid cond/uncond groups")
                prepare(query)
                batch = query.shape[0] // len(cond_or_uncond)
                region_count = len(conditionings) + 1
                output_batch = batch * sum(
                    1 if group == 1 else region_count
                    for group in cond_or_uncond
                )
                token_lengths = []
                for tensor in conditioning_tensors:
                    length = int(tensor.shape[1])
                    if has_negpip:
                        if length % 2:
                            raise ValueError(
                                "regional NegPiP conditioning needs an even "
                                "interleaved token count")
                        length //= 2
                    token_lengths.append(length)
                shapes = (
                    (output_batch, int(query.shape[1]), int(query.shape[2])),
                    (output_batch,
                     math.lcm(int(key.shape[1]), *token_lengths),
                     int(key.shape[2])),
                    (output_batch,
                     math.lcm(int(value.shape[1]), *token_lengths),
                     int(value.shape[2])),
                )
                if any(shape[0] * shape[1] > 131072 for shape in shapes):
                    raise ValueError(
                        "regional attention expansion exceeds 131072 token "
                        "positions")
                result = call(
                    "pre", query, key, value,
                    list(cond_or_uncond), has_negpip)
                if not isinstance(result, tuple) or len(result) != 3:
                    raise TypeError(
                        "regional attention pre must return a tensor triple")
                for actual, shape, original, label in zip(
                    result, shapes, (query, key, value),
                    ("query", "key", "value"), strict=True,
                ):
                    validate(actual, shape, original, label)
                merge_groups = []
                for group in cond_or_uncond:
                    merge_groups.extend(
                        [1] if group == 1 else [0] * region_count)
                if state_key in extra_options:
                    raise RuntimeError(
                        "regional attention pre was invoked twice for one site")
                extra_options[state_key] = {
                    "merge_groups": merge_groups,
                    "original_batch": int(query.shape[0]),
                }
                return result

            def attn2_output_patch(output, extra_options):
                if not isinstance(output, torch.Tensor) or output.ndim != 3:
                    raise TypeError(
                        "regional attention output must be rank-3")
                state = extra_options.pop(state_key, None)
                if not isinstance(state, dict):
                    raise RuntimeError(
                        "regional attention post has no matching pre phase")
                activation_shape = extra_options.get("activations_shape")
                if not (
                    isinstance(activation_shape, (list, tuple))
                    and len(activation_shape) >= 2
                    and all(type(value) is int and value > 0
                            for value in activation_shape[-2:])
                    and math.prod(activation_shape[-2:]) == output.shape[1]
                ):
                    raise TypeError(
                        "regional attention has invalid activation geometry")
                expected_shape = (
                    state["original_batch"], int(output.shape[1]),
                    int(output.shape[2]),
                )
                result = call(
                    "post", output, state["merge_groups"],
                    list(activation_shape))
                validate(result, expected_shape, output, "blended output")
                return result

            patched = model_obj.clone()
            patched.set_model_attn2_patch(attn2_patch)
            patched.set_model_attn2_output_patch(attn2_output_patch)
            return ModelRef._wrap(await current_runtime().refs.create(
                "MODEL", patched))
        if kind in {
            "model_input_block", "model_middle_block", "model_output_block",
        }:
            import torch
            from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel

            if not hasattr(model_obj, "clone") or not hasattr(
                model_obj, "get_model_object"
            ):
                raise TypeError(
                    "MODEL does not expose a canonical diffusion model")
            diffusion = model_obj.get_model_object("diffusion_model")
            if not isinstance(diffusion, UNetModel):
                raise TypeError(
                    "model block closures support only canonical 2D UNetModel")
            input_count = len(diffusion.input_blocks)
            output_count = len(diffusion.output_blocks)
            if input_count + output_count + 1 > 256:
                raise ValueError(
                    "canonical UNet exposes more than 256 block hooks")
            hook_names = {
                "model_input_block": "set_model_input_block_patch",
                "model_middle_block": "set_model_middle_block_after_patch",
                "model_output_block": "set_model_output_block_patch",
            }
            hook_name = hook_names[kind]
            if not hasattr(model_obj, hook_name):
                raise TypeError(
                    f"MODEL does not expose the canonical {kind} hook")

            def block_metadata(options, phase, count):
                sigmas = options.get("sigmas")
                if not isinstance(sigmas, torch.Tensor) or sigmas.numel() < 1:
                    raise TypeError("model block hook sigmas must be nonempty")
                block = options.get("block")
                if not (
                    isinstance(block, tuple)
                    and len(block) == 2
                    and block[0] == phase
                    and isinstance(block[1], int)
                    and not isinstance(block[1], bool)
                    and 0 <= block[1] < count
                ):
                    raise TypeError(
                        f"model block hook has invalid {phase} block metadata")
                return sigmas, int(block[1])

            def validate(actual, expected):
                if expected is None:
                    if actual is not None:
                        raise TypeError(
                            f"{kind} node closure changed a None skip")
                    return
                if isinstance(expected, tuple):
                    if not isinstance(actual, tuple) or len(actual) != len(expected):
                        raise TypeError(
                            f"{kind} node closure must preserve its tensor pair")
                    for value, original in zip(
                        actual, expected, strict=True
                    ):
                        validate(value, original)
                    return
                if not (
                    isinstance(actual, torch.Tensor)
                    and tuple(actual.shape) == tuple(expected.shape)
                    and actual.dtype == expected.dtype
                    and str(actual.device) == str(expected.device)
                ):
                    raise TypeError(
                        f"{kind} node closure must preserve shape, dtype, "
                        "and device")

            def input_block(hidden, options):
                sigmas, index = block_metadata(
                    options, "input", input_count)
                result = fn(hidden, sigmas, index)
                if isinstance(result, Awaitable):
                    raise TypeError(
                        "model-block closures must be synchronous")
                validate(result, hidden)
                return result

            def middle_block(args):
                sigmas, index = block_metadata(
                    args["transformer_options"], "middle", 1)
                result = fn(args["h"], sigmas, index)
                if isinstance(result, Awaitable):
                    raise TypeError(
                        "model-block closures must be synchronous")
                validate(result, args["h"])
                return {"h": result}

            def output_block(hidden, skip, options):
                sigmas, index = block_metadata(
                    options, "output", output_count)
                result = fn(hidden, skip, sigmas, index)
                if isinstance(result, Awaitable):
                    raise TypeError(
                        "model-block closures must be synchronous")
                validate(result, (hidden, skip))
                return result

            patched = model_obj.clone()
            getattr(patched, hook_name)({
                "model_input_block": input_block,
                "model_middle_block": middle_block,
                "model_output_block": output_block,
            }[kind])
            return ModelRef._wrap(await current_runtime().refs.create(
                "MODEL", patched))

        if kind == "conditioning_selection":
            from comfy.samplers import calc_cond_batch

            if not hasattr(model_obj, "clone") or not hasattr(
                model_obj, "set_model_sampler_calc_cond_batch_function"
            ):
                raise TypeError(
                    "MODEL does not expose conditional-batch selection")
            patched = model_obj.clone()
            previous = patched.model_options.get(
                "sampler_calc_cond_batch_function")

            def select_conditioning(args):
                presence = [value is not None for value in args["conds"]]
                sigma = args["sigma"]
                if hasattr(sigma, "reshape"):
                    sigma = sigma.reshape(-1)[0]
                if hasattr(sigma, "item"):
                    sigma = sigma.item()
                selected = fn(presence, float(sigma))
                if isinstance(selected, Awaitable):
                    raise TypeError(
                        "conditioning-selection closures must be synchronous")
                if (
                    not isinstance(selected, list)
                    or len(selected) != len(presence)
                    or not all(type(value) is bool for value in selected)
                    or any(value and not original for value, original in zip(
                        selected, presence, strict=True))
                ):
                    raise TypeError(
                        "conditioning-selection closure returned invalid presence")
                next_args = dict(args)
                next_args["conds"] = [
                    value if keep else None
                    for value, keep in zip(args["conds"], selected, strict=True)
                ]
                if previous is not None:
                    return previous(next_args)
                return calc_cond_batch(
                    next_args["model"], next_args["conds"],
                    next_args["input"], next_args["sigma"],
                    next_args["model_options"],
                )

            patched.set_model_sampler_calc_cond_batch_function(
                select_conditioning)
            return ModelRef._wrap(await current_runtime().refs.create(
                "MODEL", patched))

        if kind == "conditioning_preprocess":
            import copy
            import torch
            from comfy.samplers import calc_cond_batch

            if not hasattr(model_obj, "clone") or not hasattr(
                model_obj, "set_model_sampler_calc_cond_batch_function"
            ):
                raise TypeError(
                    "MODEL does not expose conditional-batch preprocessing")
            patched = model_obj.clone()
            previous = patched.model_options.get(
                "sampler_calc_cond_batch_function")

            def preprocess_conditioning(args):
                conds = copy.deepcopy(args["conds"])
                selected = []
                for cond in conds:
                    if cond is None:
                        continue
                    for item in cond:
                        model_conds = item.get("model_conds", {})
                        for key, wrapper in model_conds.items():
                            if key not in {"c_concat", "c_crossattn"}:
                                continue
                            tensor = getattr(wrapper, "cond", None)
                            if not isinstance(tensor, torch.Tensor) or not callable(
                                getattr(wrapper, "_copy_with", None)
                            ):
                                raise TypeError(
                                    "conditioning preprocessing requires canonical "
                                    "tensor conditioning wrappers")
                            selected.append((model_conds, key, wrapper, tensor))
                if selected:
                    tensors = [entry[3] for entry in selected]
                    noises = [torch.randn_like(tensor) for tensor in tensors]
                    result = fn(tensors, noises, args["sigma"])
                    if isinstance(result, Awaitable):
                        raise TypeError(
                            "conditioning-preprocess closures must be synchronous")

                    def validate(actual, expected):
                        if not isinstance(actual, list) or len(actual) != len(expected):
                            raise TypeError(
                                "conditioning-preprocess closure must preserve the "
                                "tensor list")
                        for value, original in zip(
                            actual, expected, strict=True
                        ):
                            if not (
                                isinstance(value, torch.Tensor)
                                and tuple(value.shape) == tuple(original.shape)
                                and value.dtype == original.dtype
                                and str(value.device) == str(original.device)
                            ):
                                raise TypeError(
                                    "conditioning-preprocess closure must preserve "
                                    "shape, dtype, and device")

                    validate(result, tensors)
                    for entry, value in zip(selected, result, strict=True):
                        model_conds, key, wrapper, _tensor = entry
                        model_conds[key] = wrapper._copy_with(value)

                next_args = dict(args)
                next_args["conds"] = conds
                if previous is not None:
                    return previous(next_args)
                return calc_cond_batch(
                    next_args["model"], next_args["conds"],
                    next_args["input"], next_args["sigma"],
                    next_args["model_options"],
                )

            patched.set_model_sampler_calc_cond_batch_function(
                preprocess_conditioning)
            return ModelRef._wrap(await current_runtime().refs.create(
                "MODEL", patched))

        hook_name = (
            "set_model_sampler_post_cfg_function"
            if kind == "post_cfg"
            else "set_model_sampler_pre_cfg_function"
        )
        if not hasattr(model_obj, "clone") or not hasattr(model_obj, hook_name):
            raise TypeError(
                f"MODEL does not expose the canonical {kind.replace('_', '-')} hook")

        def validate(actual, expected):
            if isinstance(expected, (list, tuple)):
                if not isinstance(actual, type(expected)) or len(actual) != len(expected):
                    raise TypeError(
                        f"{kind} node closure must preserve the prediction list")
                for item, expected_item in zip(actual, expected, strict=True):
                    validate(item, expected_item)
                return
            if not (
                hasattr(actual, "shape")
                and tuple(actual.shape) == tuple(expected.shape)
                and getattr(actual, "dtype", None) == expected.dtype
                and str(getattr(actual, "device", "")) == str(expected.device)
            ):
                raise TypeError(
                    f"{kind} node closure must preserve shape, dtype, and device")

        def post_cfg(args):
            result = fn(
                args["denoised"],
                args.get("cond_denoised"),
                args.get("uncond_denoised"),
                args["input"],
                args["sigma"],
                float(args["cond_scale"]),
            )
            if isinstance(result, Awaitable):
                raise TypeError(
                    "tensor-phase node closures must be synchronous functions")
            validate(result, args["denoised"])
            return result

        def pre_cfg(args):
            expected = list(args["conds_out"])
            result = fn(
                args["input"],
                expected,
                [value is not None for value in args["conds"]],
                args["sigma"],
            )
            if isinstance(result, Awaitable):
                raise TypeError(
                    "tensor-phase node closures must be synchronous functions")
            validate(result, expected)
            return result

        patched = model_obj.clone()
        getattr(patched, hook_name)(
            post_cfg if kind == "post_cfg" else pre_cfg,
            disable_cfg1_optimization=True,
        )
        return ModelRef._wrap(await current_runtime().refs.create(
            "MODEL", patched))

    async def attach_model_clip(
        self, closure: ClosureRef, model: ModelRef, clip: ClipRef,
    ) -> tuple[ModelRef, ClipRef]:
        """Default in-process adapter for a future CLIP weight closure."""
        import math
        import torch
        from comfy import model_management
        import comfy.conds
        import comfy.patcher_extension
        from comfy.model_base import Anima, BaseModel, Flux, SDXL, SDXLRefiner
        from comfy.sd1_clip import gen_empty_tokens

        if not isinstance(closure, ClosureRef):
            raise TypeError("closure must be a typed CLOSURE ref")
        if not isinstance(model, ModelRef):
            raise TypeError("model must be a typed MODEL ref")
        if not isinstance(clip, ClipRef):
            raise TypeError("clip must be a typed CLIP ref")
        entry = await current_runtime().refs.resolve(closure)
        if entry.get("kind") != "clip_token_weight_encoder":
            raise ValueError(
                "only clip_token_weight_encoder closures can attach a "
                "MODEL/CLIP pair")
        model_obj = await current_runtime().refs.resolve(model)
        clip_obj = await current_runtime().refs.resolve(clip)
        if not callable(getattr(model_obj, "clone", None)):
            raise TypeError("NegPiP requires a cloneable MODEL")
        if not callable(getattr(clip_obj, "clone", None)):
            raise TypeError("NegPiP requires a cloneable CLIP")
        patched_model = model_obj.clone()
        patched_clip = clip_obj.clone()
        patcher = getattr(patched_clip, "patcher", None)
        clip_model = getattr(patcher, "model", None)
        if (
            patcher is None
            or clip_model is None
            or not callable(getattr(patcher, "add_object_patch", None))
        ):
            raise TypeError("NegPiP requires a canonical CLIP component patcher")
        model_options = getattr(patched_model, "model_options", None)
        clip_options = getattr(patcher, "model_options", None)
        if not isinstance(model_options, dict) or not isinstance(
            clip_options, dict
        ):
            raise TypeError("NegPiP needs canonical model option mappings")
        model_marked = bool(model_options.get("ppm_negpip"))
        clip_marked = bool(clip_options.get("ppm_negpip"))
        if model_marked != clip_marked:
            raise TypeError("NegPiP refuses a half-patched MODEL/CLIP pair")
        components = [
            name for name in (
                "clip_g", "clip_l", "t5xxl", "llama", "qwen3_06b",
            )
            if hasattr(clip_model, name)
        ]

        async def store_pair():
            refs = current_runtime().refs
            return (
                ModelRef._wrap(await refs.create("MODEL", patched_model)),
                ClipRef._wrap(await refs.create("CLIP", patched_clip)),
            )

        if not components or model_marked:
            return await store_pair()
        base_model = getattr(patched_model, "model", None)
        model_type = type(base_model)
        canonical_family = (
            model_type is BaseModel
            or issubclass(model_type, (SDXL, SDXLRefiner))
        )
        anima_family = issubclass(model_type, Anima)
        if issubclass(model_type, Flux):
            raise ValueError(
                "NegPiP's FLUX full-forward replacement is unmaintained "
                "upstream and is not admitted by the bounded V2 adapter")
        if not (canonical_family or anima_family):
            return await store_pair()
        fn = entry["fn"]

        def call(*args):
            result = fn(*args)
            if isinstance(result, Awaitable):
                raise TypeError("NegPiP closures must be synchronous")
            return result

        if canonical_family:
            def validate_pairs(token_weight_pairs):
                if not isinstance(token_weight_pairs, list) or len(
                    token_weight_pairs
                ) > 2048:
                    raise ValueError("NegPiP needs at most 2048 token sections")
                tokens, weights = [], []
                total = 0
                has_weights = False
                max_length = 0
                for section_index, section in enumerate(token_weight_pairs):
                    if not isinstance(section, list) or not 1 <= len(
                        section
                    ) <= 4096:
                        raise ValueError(
                            f"NegPiP token section {section_index} needs "
                            "1..4096 entries")
                    total += len(section)
                    if total > 131072:
                        raise ValueError(
                            "NegPiP token input exceeds 131072 positions")
                    section_tokens, section_weights = [], []
                    for entry_index, pair in enumerate(section):
                        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                            raise TypeError(
                                f"NegPiP token entry {section_index}:"
                                f"{entry_index} must contain token and weight")
                        weight = pair[1]
                        if (
                            isinstance(weight, bool)
                            or not isinstance(weight, (int, float))
                            or not math.isfinite(float(weight))
                        ):
                            raise ValueError("NegPiP token weights must be finite")
                        section_tokens.append(pair[0])
                        section_weights.append(float(weight))
                        has_weights = has_weights or float(weight) != 1.0
                    tokens.append(section_tokens)
                    weights.append(section_weights)
                    max_length = max(max_length, len(section_tokens))
                return tokens, weights, has_weights, max_length

            def make_encoder(target):
                def encode_token_weights(token_weight_pairs):
                    tokens, weights, has_weights, max_length = validate_pairs(
                        token_weight_pairs)
                    sections = len(tokens)
                    empty_index = None
                    if has_weights or sections == 0:
                        generator = getattr(target, "gen_empty_tokens", None)
                        tokens.append(
                            generator(target.special_tokens, max_length)
                            if callable(generator)
                            else gen_empty_tokens(target.special_tokens, max_length)
                        )
                        empty_index = len(tokens) - 1
                    output = target.encode(tokens)
                    if not isinstance(output, (list, tuple)) or len(output) < 2:
                        raise TypeError("NegPiP base encoder returned invalid data")
                    encoded, pooled = output[:2]
                    if not (
                        isinstance(encoded, torch.Tensor)
                        and encoded.ndim == 3
                        and encoded.shape[0] == len(tokens)
                    ):
                        raise TypeError("NegPiP base encoder returned invalid rows")
                    if any(len(row) != encoded.shape[1] for row in weights):
                        raise ValueError(
                            "NegPiP token rows do not match encoded sequence length")
                    expected_shape = (
                        (1, int(encoded.shape[1]), int(encoded.shape[2]))
                        if sections == 0
                        else (
                            1,
                            sections * int(encoded.shape[1]) * 2,
                            int(encoded.shape[2]),
                        )
                    )
                    result = call("encode", encoded, weights, empty_index)
                    if not (
                        isinstance(result, torch.Tensor)
                        and tuple(result.shape) == expected_shape
                        and result.dtype == encoded.dtype
                        and str(result.device) == str(encoded.device)
                    ):
                        raise TypeError(
                            "NegPiP encode changed shape, dtype, or device")
                    intermediate = model_management.intermediate_device()
                    first_pooled = (
                        pooled[0:1].to(device=intermediate)
                        if isinstance(pooled, torch.Tensor)
                        else pooled
                    )
                    returned = (result.to(device=intermediate), first_pooled)
                    if len(output) > 2:
                        if not isinstance(output[2], dict):
                            raise TypeError("NegPiP encoder extras must be a mapping")
                        extras = {}
                        for name, value in output[2].items():
                            if name == "attention_mask":
                                if not isinstance(value, torch.Tensor):
                                    raise TypeError(
                                        "NegPiP attention mask must be a tensor")
                                value = value[:sections].flatten().unsqueeze(0)
                                value = value.to(device=intermediate)
                            extras[name] = value
                        returned = returned + (extras,)
                    return returned

                return encode_token_weights

            for component in components:
                target = getattr(clip_model, component)
                patcher.add_object_patch(
                    f"{component}.encode_token_weights", make_encoder(target))

            def attn2_negpip(query, key, value, _extra_options):
                if not (
                    isinstance(query, torch.Tensor)
                    and isinstance(key, torch.Tensor)
                    and isinstance(value, torch.Tensor)
                    and query.ndim == key.ndim == value.ndim == 3
                    and key.shape[1] % 2 == 0
                    and value.shape[1] % 2 == 0
                ):
                    raise TypeError(
                        "NegPiP attention needs even interleaved key/value rows")
                return query, key[:, 0::2], value[:, 1::2]

            if not callable(
                getattr(patched_model, "set_model_attn2_patch", None)
            ):
                raise TypeError("NegPiP MODEL has no cross-attention hook")
            patched_model.set_model_attn2_patch(attn2_negpip)

        if anima_family:
            if not all(callable(getattr(patched_model, name, None)) for name in (
                "get_model_object", "add_object_patch", "add_wrapper_with_key",
                "set_model_attn2_patch",
            )):
                raise TypeError("Anima NegPiP MODEL lacks canonical patcher hooks")
            previous_extra_conds = patched_model.get_model_object("extra_conds")
            if not callable(previous_extra_conds):
                raise TypeError("Anima extra_conds is not callable")

            def extra_conds(**kwargs):
                weights = kwargs.get("t5xxl_weights")
                sign_mask = None
                if weights is not None:
                    if not (
                        isinstance(weights, torch.Tensor)
                        and weights.ndim == 1
                        and torch.is_floating_point(weights)
                        and 1 <= weights.numel() <= 131072
                        and bool(torch.isfinite(weights).all())
                    ):
                        raise TypeError(
                            "Anima NegPiP weights must be a finite 1-D tensor")
                    absolute, sign_mask = call(
                        "anima_weights", weights, 512, None)
                    mask_length = max(int(weights.numel()), 512)
                    if not (
                        isinstance(absolute, torch.Tensor)
                        and tuple(absolute.shape) == tuple(weights.shape)
                        and absolute.dtype == weights.dtype
                        and str(absolute.device) == str(weights.device)
                        and isinstance(sign_mask, torch.Tensor)
                        and tuple(sign_mask.shape) == (1, mask_length, 1)
                        and sign_mask.dtype == torch.int32
                        and str(sign_mask.device) == str(weights.device)
                        and bool(((sign_mask == 1) | (sign_mask == -1)).all())
                    ):
                        raise TypeError(
                            "Anima NegPiP weight projection violated its contract")
                    kwargs["t5xxl_weights"] = absolute
                output = previous_extra_conds(**kwargs)
                if not isinstance(output, dict):
                    raise TypeError("Anima extra_conds must return a mapping")
                if sign_mask is not None:
                    output["c_ppm_negpip_mask"] = comfy.conds.CONDRegular(
                        sign_mask)
                return output

            def diffusion_wrapper(executor, *args, **kwargs):
                if len(args) < 3 or not isinstance(args[2], torch.Tensor):
                    raise TypeError(
                        "Anima NegPiP diffusion wrapper needs context tensor")
                sign_mask = kwargs.get("c_ppm_negpip_mask")
                options = dict(kwargs.get("transformer_options", {}))
                if sign_mask is not None:
                    if not isinstance(sign_mask, torch.Tensor):
                        raise TypeError("Anima NegPiP sidecar must be a tensor")
                    options["ppm_negpip_mask"] = sign_mask.to(args[2])
                kwargs["transformer_options"] = options
                return executor(*args, **kwargs)

            def anima_attn2(query, key, value, pe=None, attn_mask=None,
                            extra_options=None):
                sign_mask = (extra_options or {}).get("ppm_negpip_mask")
                if sign_mask is not None:
                    if not (
                        isinstance(sign_mask, torch.Tensor)
                        and sign_mask.ndim == 3
                        and sign_mask.shape[0] in {1, value.shape[0]}
                        and sign_mask.shape[1] == value.shape[1]
                        and sign_mask.shape[2] == 1
                        and str(sign_mask.device) == str(value.device)
                    ):
                        raise TypeError(
                            "Anima NegPiP sign mask does not match values")
                    value = value * sign_mask
                return {"q": query, "k": key, "v": value, "pe": pe}

            patched_model.add_object_patch("extra_conds", extra_conds)
            patched_model.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                f"secure_negpip_{id(fn)}",
                diffusion_wrapper,
            )
            patched_model.set_model_attn2_patch(anima_attn2)

        model_options["ppm_negpip"] = True
        clip_options["ppm_negpip"] = True
        return await store_pair()

    async def attach_sampler(
        self, closure: ClosureRef, sampler: SamplerRef, *,
        start_percent=None, end_percent=None,
    ):
        import math
        from comfy.samplers import KSAMPLER

        if not isinstance(closure, ClosureRef):
            raise TypeError("closure must be a typed CLOSURE ref")
        if not isinstance(sampler, SamplerRef):
            raise TypeError("sampler must be a typed SAMPLER ref")
        entry = await current_runtime().refs.resolve(closure)
        if entry.get("kind") != "model_sigma":
            raise ValueError(
                "only model_sigma node closures can wrap samplers")
        sampler_obj = await current_runtime().refs.resolve(sampler)
        for name, value in {
            "start_percent": start_percent,
            "end_percent": end_percent,
        }.items():
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"{name} must be numeric")
                value = float(value)
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name} must be finite and in [0, 1]")
        if (start_percent is None) != (end_percent is None):
            raise ValueError(
                "start_percent and end_percent must be supplied together")
        fn = entry["fn"]

        def wrapped_sampler(model_fn, x, sigmas, **kwargs):
            cfg = getattr(getattr(model_fn, "inner_model", None), "cfg", 1.0)
            cfg = float(cfg) if isinstance(cfg, (int, float)) else 1.0
            start_sigma = end_sigma = None
            if start_percent is not None:
                sampling = getattr(
                    getattr(getattr(model_fn, "inner_model", None),
                            "inner_model", None),
                    "model_sampling", None,
                )
                if sampling is None or not hasattr(sampling, "percent_to_sigma"):
                    raise TypeError(
                        "sampler model does not expose percent-to-sigma projection")
                start_sigma = round(float(
                    sampling.percent_to_sigma(float(start_percent))), 4)
                end_sigma = round(float(
                    sampling.percent_to_sigma(float(end_percent))), 4)

            def model_wrapper(latent, sigma, **extra_args):
                adjusted = fn(
                    sigma, sigmas, cfg, start_sigma, end_sigma)
                if isinstance(adjusted, Awaitable):
                    raise TypeError(
                        "tensor-phase node closures must be synchronous functions")
                if not (
                    hasattr(adjusted, "shape")
                    and tuple(adjusted.shape) == tuple(sigma.shape)
                    and adjusted.dtype == sigma.dtype
                    and str(adjusted.device) == str(sigma.device)
                ):
                    raise TypeError(
                        "model_sigma node closure must preserve shape, dtype, "
                        "and device")
                return model_fn(latent, adjusted, **extra_args)

            for name in ("inner_model", "sigmas"):
                if hasattr(model_fn, name):
                    setattr(model_wrapper, name, getattr(model_fn, name))
            return sampler_obj.sampler_function(
                model_wrapper,
                x,
                sigmas,
                **kwargs,
                **sampler_obj.extra_options,
            )

        value = KSAMPLER(wrapped_sampler)
        return SamplerRef._wrap(await current_runtime().refs.create(
            "SAMPLER", value))

    async def create_latent_operation(self, closure: ClosureRef):
        if not isinstance(closure, ClosureRef):
            raise TypeError("closure must be a typed CLOSURE ref")
        entry = await current_runtime().refs.resolve(closure)
        if entry.get("kind") != "latent_operation":
            raise ValueError(
                "only latent_operation closures can create LATENT_OPERATION")
        fn = entry["fn"]

        def operation(latent, **_kwargs):
            result = fn(latent)
            if isinstance(result, Awaitable):
                raise TypeError(
                    "tensor-phase node closures must be synchronous functions")
            if not (
                hasattr(result, "shape")
                and tuple(result.shape) == tuple(latent.shape)
                and getattr(result, "dtype", None) == latent.dtype
                and str(getattr(result, "device", "")) == str(latent.device)
            ):
                raise TypeError(
                    "latent_operation closure must preserve shape, dtype, "
                    "and device")
            return result

        return LatentOperationRef._wrap(
            await current_runtime().refs.create("LATENT_OPERATION", operation))

    async def create_sampler(self, closure: ClosureRef):
        """Default in-process adapter for a pack-owned sampling loop."""
        import math
        import torch
        import torch.nn.functional as functional
        import comfy.model_patcher
        import comfy.model_sampling
        from comfy.k_diffusion.sampling import (
            BrownianTreeNoiseSampler, default_noise_sampler,
        )
        from comfy.samplers import KSAMPLER

        if not isinstance(closure, ClosureRef):
            raise TypeError("closure must be a typed CLOSURE ref")
        entry = await current_runtime().refs.resolve(closure)
        if entry.get("kind") != "custom_sampler":
            raise ValueError(
                "only custom_sampler closures can create a SAMPLER")
        fn = entry["fn"]

        def sampler_function(model_fn, latent, sigmas, **kwargs):
            if not (
                isinstance(sigmas, torch.Tensor)
                and torch.is_floating_point(sigmas)
                and sigmas.ndim == 1
                and 2 <= len(sigmas) <= 4097
            ):
                raise ValueError(
                    "custom-sampler sigmas must be a finite floating-point, "
                    "nonnegative, nonincreasing 2..4097 vector ending at zero")
            sigma_values = sigmas.detach().to(
                device="cpu", dtype=torch.float64)
            if (
                not torch.isfinite(sigma_values).all()
                or bool((sigma_values < 0).any())
                or bool((sigma_values[:-1] < sigma_values[1:]).any())
                or not math.isclose(
                    float(sigma_values[-1]), 0.0, abs_tol=1e-8)
            ):
                raise ValueError(
                    "custom-sampler sigmas must be a finite floating-point, "
                    "nonnegative, nonincreasing 2..4097 vector ending at zero")
            if not (
                isinstance(latent, torch.Tensor)
                and torch.is_floating_point(latent)
                and latent.ndim >= 3
            ):
                raise TypeError(
                    "custom sampler needs a floating-point tensor latent")
            extra_args = kwargs.get("extra_args")
            if extra_args is None:
                extra_args = {}
            if not isinstance(extra_args, dict):
                raise TypeError(
                    "custom sampler received invalid host extra_args")
            extra_args = dict(extra_args)
            callback = kwargs.get("callback")
            seed = extra_args.get("seed")
            if seed is not None and (
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or not 0 <= seed <= (1 << 64) - 1
            ):
                raise ValueError(
                    "custom-sampler seed must be an unsigned 64-bit integer")
            get_sampling = getattr(
                getattr(getattr(model_fn, "inner_model", None),
                        "model_patcher", None),
                "get_model_object", None,
            )
            if not callable(get_sampling):
                raise TypeError(
                    "custom sampler model does not expose model sampling")
            model_sampling = get_sampling("model_sampling")

            def validate_value(value, *, allow_resize=False):
                if not (
                    isinstance(value, torch.Tensor)
                    and torch.is_floating_point(value)
                    and value.dtype == latent.dtype
                    and str(value.device) == str(latent.device)
                ):
                    raise TypeError(
                        "custom-sampler latent changed type, dtype, or device")
                shape = tuple(value.shape)
                original = tuple(latent.shape)
                if shape == original:
                    return
                if not allow_resize or (
                    len(shape) != 4
                    or len(original) != 4
                    or shape[:2] != original[:2]
                    or any(item < 1 for item in shape[2:])
                    or shape[2] > original[2] * 2
                    or shape[3] > original[3] * 2
                    or shape[2] * shape[3] > original[2] * original[3] * 4
                ):
                    raise ValueError(
                        "custom-sampler temporary resize is outside its bounds")

            def validate_result(value, expected, name):
                if not (
                    isinstance(value, torch.Tensor)
                    and torch.is_floating_point(value)
                    and tuple(value.shape) == tuple(expected.shape)
                    and value.dtype == expected.dtype
                    and str(value.device) == str(expected.device)
                ):
                    raise TypeError(
                        f"custom-sampler {name} must preserve shape, dtype, "
                        "and device")

            def scalar(value, name):
                if isinstance(value, bool):
                    raise TypeError(f"{name} must be numeric")
                if hasattr(value, "numel"):
                    if int(value.numel()) != 1:
                        raise ValueError(
                            f"{name} must contain exactly one value")
                    value = value.detach().reshape(-1)[0].item()
                try:
                    result = float(value)
                except (TypeError, ValueError) as error:
                    raise TypeError(f"{name} must be numeric") from error
                if not math.isfinite(result):
                    raise ValueError(f"{name} must be finite")
                return result

            class Broker:
                def __init__(self):
                    self.denoise_count = 0
                    self.noise_count = 0
                    self.preview_count = 0
                    self.schedule_count = 0
                    self.last_denoise = None
                    self.noise_samplers = {}

                async def denoise(
                    self, value, sigma, *, capture_uncond=False,
                    resize_context=None,
                ):
                    if not isinstance(capture_uncond, bool):
                        raise TypeError(
                            "capture_uncond must be a boolean")
                    if self.denoise_count >= 3 * (len(sigmas) - 1):
                        raise RuntimeError(
                            "custom sampler exceeded its denoise budget")
                    sigma_value = scalar(sigma, "sigma")
                    if sigma_value < 0:
                        raise ValueError("custom-sampler sigma is invalid")
                    mode = "none" if resize_context is None else str(
                        resize_context)
                    if mode not in {"none", "nearest-exact"}:
                        raise ValueError("unsupported sampler resize context")
                    validate_value(
                        value, allow_resize=mode == "nearest-exact")
                    call_args = dict(extra_args)

                    def restore():
                        return None

                    if mode == "nearest-exact":
                        target = tuple(value.shape[-2:])
                        old_latent = getattr(model_fn, "latent_image", None)
                        old_noise = getattr(model_fn, "noise", None)
                        old_mask = call_args.get("denoise_mask")

                        def resized(item):
                            return (
                                None if item is None else
                                functional.interpolate(
                                    item, size=target, mode=mode)
                            )

                        new_latent = resized(old_latent)
                        new_noise = resized(old_noise)
                        new_mask = resized(old_mask)
                        try:
                            model_fn.latent_image = new_latent
                            model_fn.noise = new_noise
                            if old_mask is not None:
                                call_args["denoise_mask"] = new_mask
                        except Exception:
                            model_fn.latent_image = old_latent
                            model_fn.noise = old_noise
                            raise

                        def restore():
                            model_fn.latent_image = old_latent
                            model_fn.noise = old_noise

                    try:
                        captured = [None]
                        if capture_uncond:
                            def capture(args):
                                captured[0] = args.get("uncond_denoised")
                                return args["denoised"]

                            call_args["model_options"] = (
                                comfy.model_patcher.
                                set_model_options_post_cfg_function(
                                    dict(call_args.get("model_options") or {}),
                                    capture,
                                    disable_cfg1_optimization=True,
                                ))
                        sigma_batch = torch.full(
                            (int(value.shape[0]),),
                            sigma_value,
                            dtype=value.dtype,
                            device=value.device,
                        )
                        denoised = model_fn(
                            value, sigma_batch, **call_args)
                    finally:
                        restore()
                    validate_result(denoised, value, "denoise result")
                    if captured[0] is not None:
                        validate_result(
                            captured[0], value, "unconditional result")
                    self.denoise_count += 1
                    self.last_denoise = [value, denoised, False]
                    return denoised, captured[0]

                async def noise_like(
                    self, value, *, kind="independent", step=0,
                    sigma_from=0.0, sigma_to=0.0, purpose="sampler",
                    noise_device=None, seeded=False,
                ):
                    if self.noise_count >= 4 * (len(sigmas) - 1):
                        raise RuntimeError(
                            "custom sampler exceeded its noise budget")
                    validate_value(value)
                    if not isinstance(kind, str) or kind not in {
                        "independent", "ancestral", "brownian",
                    }:
                        raise ValueError(
                            "unsupported custom-sampler noise kind")
                    if isinstance(step, bool) or not isinstance(step, int):
                        raise TypeError("custom-sampler noise step is invalid")
                    if not 0 <= step < len(sigmas) - 1:
                        raise ValueError(
                            "custom-sampler noise step is out of range")
                    if not isinstance(purpose, str) or not 1 <= len(purpose) <= 128:
                        raise ValueError(
                            "custom-sampler noise purpose is invalid")
                    if noise_device not in {None, "cpu", "latent"}:
                        raise ValueError(
                            "custom-sampler noise device is invalid")
                    if not isinstance(seeded, bool):
                        raise TypeError(
                            "custom-sampler seeded flag must be a boolean")
                    sigma_from_value = scalar(sigma_from, "sigma_from")
                    sigma_to_value = scalar(sigma_to, "sigma_to")
                    if (
                        sigma_from_value < 0
                        or sigma_to_value < 0
                    ):
                        raise ValueError(
                            "custom-sampler noise sigmas are invalid")
                    if kind == "brownian":
                        key = ("brownian", noise_device or "cpu")
                        sampler = self.noise_samplers.get(key)
                        if sampler is None:
                            positive = sigmas[sigmas > 0]
                            sampler = BrownianTreeNoiseSampler(
                                latent,
                                positive.min(),
                                sigmas.max(),
                                seed=seed,
                                cpu=(noise_device or "cpu") == "cpu",
                            )
                            self.noise_samplers[key] = sampler
                    else:
                        key = ("default", bool(seeded))
                        sampler = self.noise_samplers.get(key)
                        if sampler is None:
                            sampler = default_noise_sampler(
                                latent, seed=seed if seeded else None)
                            self.noise_samplers[key] = sampler
                    result = sampler(
                        sigma_from_value, sigma_to_value)
                    validate_result(result, value, "noise result")
                    self.noise_count += 1
                    return result

                async def preview(
                    self, step, value, sigma, sigma_hat, denoised,
                ):
                    del value, denoised
                    if isinstance(step, bool) or not isinstance(step, int):
                        raise TypeError("custom-sampler preview step is invalid")
                    if not 0 <= step < len(sigmas) - 1:
                        raise ValueError(
                            "custom-sampler preview step is out of range")
                    sigma_value = scalar(sigma, "sigma")
                    sigma_hat_value = scalar(sigma_hat, "sigma_hat")
                    if sigma_value < 0 or sigma_hat_value < 0:
                        raise ValueError(
                            "custom-sampler preview sigmas are invalid")
                    if self.last_denoise is None or self.last_denoise[2]:
                        raise RuntimeError(
                            "preview must follow one unpreviewed denoise")
                    if self.preview_count >= 3 * (len(sigmas) - 1):
                        raise RuntimeError(
                            "custom sampler exceeded its preview budget")
                    if callback is not None:
                        callback({
                            "x": self.last_denoise[0],
                            "i": step,
                            "sigma": sigma_value,
                            "sigma_hat": sigma_hat_value,
                            "denoised": self.last_denoise[1],
                        })
                    self.last_denoise[2] = True
                    self.preview_count += 1

                async def schedule_parameters(self, *, percent_offset=1e-4):
                    if self.schedule_count >= 4:
                        raise RuntimeError(
                            "custom sampler exceeded its schedule budget")
                    offset = scalar(percent_offset, "percent_offset")
                    if not 1e-8 <= offset <= 0.1:
                        raise ValueError("percent_offset is outside its bounds")
                    is_const = isinstance(
                        model_sampling, comfy.model_sampling.CONST)
                    noise_scale = float(getattr(
                        model_sampling, "noise_scale", 1.0))
                    if not math.isfinite(noise_scale) or abs(noise_scale) > 1e6:
                        raise ValueError(
                            "model sampling noise_scale is outside its bounds")
                    first_sigma = (
                        float(model_sampling.percent_to_sigma(offset))
                        if is_const else None
                    )
                    if first_sigma is not None and (
                        not math.isfinite(first_sigma) or first_sigma < 0
                    ):
                        raise ValueError(
                            "model sampling returned an invalid first sigma")
                    self.schedule_count += 1
                    return {
                        "parameterization": "const" if is_const else "sigma",
                        "noise_scale": noise_scale,
                        "first_sigma": first_sigma,
                    }

            result = fn(Broker(), latent, sigmas)
            if isinstance(result, Awaitable):
                result = asyncio.run(result)
            validate_value(result)
            return result

        return SamplerRef._wrap(await current_runtime().refs.create(
            "SAMPLER", KSAMPLER(sampler_function)))


class _InProcessWanVideo:
    """Read one scheduler-relevant scalar from a WanVideo model patcher.

    Scheduler construction and refinement remain untrusted pack code.  This
    adapter only resolves the opaque ref and projects the fixed scalar that
    WanVideo's CausVid schedule selects on.
    """

    async def transformer_dim(self, model: Ref) -> int:
        if not isinstance(model, Ref) or model.kind not in {"MODEL", "OPAQUE"}:
            raise TypeError("WanVideo model must be an opaque model ref")
        value = await current_runtime().refs.resolve(model)
        try:
            dimension = value.model.diffusion_model.dim
        except AttributeError as error:
            raise ValueError(
                "WanVideo model does not publish transformer dimension"
            ) from error
        if (isinstance(dimension, bool) or not isinstance(dimension, int)
                or not 1 <= dimension <= 65_536):
            raise ValueError("WanVideo transformer dimension is invalid")
        return dimension


class _InProcessLlm:
    """Normalize common chat/tool semantics onto closed vendor adapters."""

    @staticmethod
    def _vendor_options(value: Any) -> tuple[int, str]:
        if value is None:
            return 5, "minutes"
        if not isinstance(value, dict) or set(value) != {"ollama"}:
            raise ValueError("LLM vendor_options must contain only ollama")
        options = value["ollama"]
        if not isinstance(options, dict) or not set(options) <= {
            "keep_alive", "keep_alive_unit",
        }:
            raise ValueError("LLM Ollama vendor options are invalid")
        return options.get("keep_alive", 5), options.get(
            "keep_alive_unit", "minutes")

    @staticmethod
    def _messages(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list) or not 1 <= len(value) <= 256:
            raise ValueError("LLM messages must contain 1 to 256 entries")
        result = []
        for message in value:
            if not isinstance(message, dict):
                raise ValueError("LLM message has an invalid shape")
            role = message.get("role")
            if role in {"system", "user"}:
                if set(message) != {"role", "content"}:
                    raise ValueError("LLM message has an invalid shape")
                result.append(dict(message))
                continue
            if role == "assistant":
                if (not {"role", "content"}.issubset(message)
                        or not set(message) <= {
                            "role", "content", "thinking", "tool_calls",
                        }):
                    raise ValueError("LLM assistant message has an invalid shape")
                result.append(dict(message))
                continue
            if role == "tool":
                if set(message) != {"role", "name", "content"}:
                    raise ValueError("LLM tool message has an invalid shape")
                result.append({
                    "role": "tool",
                    "tool_name": message["name"],
                    "content": message["content"],
                })
                continue
            raise ValueError("LLM message role is invalid")
        return result

    async def chat(
        self, provider: str, profile: str, model: str,
        messages: list[dict[str, Any]], *,
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.8, max_tokens: int = 512,
        thinking: bool = False,
        response_format: str | dict[str, Any] = "",
        timeout_seconds: float = 600.0,
        vendor_options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if provider != "ollama":
            raise ValueError("LLM provider must be ollama")
        if (isinstance(temperature, bool)
                or type(temperature) not in {int, float}
                or not 0.0 <= float(temperature) <= 10.0):
            raise ValueError("LLM temperature must be in [0, 10]")
        if (isinstance(max_tokens, bool) or not isinstance(max_tokens, int)
                or not 1 <= max_tokens <= 32_768):
            raise ValueError("LLM max_tokens must be in [1, 32768]")
        if type(thinking) is not bool:
            raise TypeError("LLM thinking must be a bool")
        keep_alive, keep_alive_unit = self._vendor_options(vendor_options)
        response = await InProcessOllama().chat(
            endpoint=profile,
            model=model,
            messages=self._messages(messages),
            think=thinking,
            options={
                "temperature": float(temperature),
                "num_predict": max_tokens,
            },
            keep_alive=keep_alive,
            keep_alive_unit=keep_alive_unit,
            format=response_format,
            timeout_seconds=timeout_seconds,
            tools=tools,
        )
        result: dict[str, Any] = {
            "content": response["response"],
            "tool_calls": response.get("tool_calls", []),
        }
        if "thinking" in response:
            result["thinking"] = response["thinking"]
        return result


@dataclass(frozen=True)
class _InProcessIntegrations:
    anima: Any = field(default_factory=InProcessAnima)
    civitai: Any = field(default_factory=InProcessCivitai)
    imgbb: Any = field(default_factory=InProcessImgBB)
    llm: Any = field(default_factory=_InProcessLlm)
    llama_cpp: Any = field(default_factory=InProcessLlamaCpp)
    luma: Any = field(default_factory=InProcessLuma)
    ollama: Any = field(default_factory=InProcessOllama)
    sensenova: Any = field(default_factory=InProcessSenseNova)
    wanvideo: Any = field(default_factory=_InProcessWanVideo)
    web: Any = field(default_factory=lambda: _StubDomain("integrations.web"))


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
    execution: Any
    integrations: Any
    models: Any
    profiling: Any
    preview_override: Any
    system: Any
    closures: Any
    interact: Any
    sample: Any
    unsample: Any
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
            ui=_InProcessUi(plan.prompt, plan.extra_pnginfo),
            output=_InProcessOutput(plan.prompt, plan.extra_pnginfo),
            graph=_InProcessGraph(
                plan.node_id, plan.prompt, plan.extra_pnginfo,
                plan.dynamic_prompt),
            execution=_InProcessExecution(plan.prompt_id),
            integrations=_InProcessIntegrations(),
            models=_InProcessModels(),
            profiling=InProcessProfiling(
                f"in-process:{plan.node_module}", plan.node_id),
            preview_override=InProcessPreviewOverride(plan.node_id),
            system=_InProcessSystem(),
            closures=_InProcessClosures(),
            interact=_StubDomain("interact"),
            sample=_StubDomain("sample"),
            unsample=_StubDomain("unsample"),
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
    registry: core ships a closed primitive set; ``register_op`` extends it
    (an overlay subclasses or registers richer ops)."""

    def __init__(self) -> None:
        self._ops: dict[str, Callable[..., Awaitable["ImageRef"]]] = {
            "ref.describe": self._ref_describe,
            "interpolation_states.skip_mask":
                _vendor_ops.interpolation_states_skip_mask,
            "invert": self._invert,
            "scale": self._scale,
            "image.rgb": self._image_rgb,
            "image.to_device": self._image_to_device,
            "image.spatial_shape": self._image_spatial_shape,
            "image.batch_size": self._image_batch_size,
            "image.select_batch": self._image_select_batch,
            "mask.grow": self._mask_grow,
            # Operations on live engine objects. These are what let a node
            # DECLARE a MODEL/CLIP/VAE input and still be sandboxable: the node
            # names the operation, the weights stay here.
            "vae.decode": self._vae_decode,
            "vae.latent_layout": self._vae_latent_layout,
            "vae.decode_tensor": self._vae_decode_tensor,
            "vae.decode_tiled": self._vae_decode_tiled,
            "vae.decode_tensor_tiled": self._vae_decode_tensor_tiled,
            "vae.encode": self._vae_encode,
            "vae.encode_for_inpaint": self._vae_encode_for_inpaint,
            "vae.encode_inpaint_conditioning":
                self._vae_encode_inpaint_conditioning,
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
            "clip.encode_token_weights_component":
                self._clip_encode_token_weights_component,
            "clip.encode": self._clip_encode,
            "clip.set_last_layer": self._clip_set_last_layer,
            "clip.with_attention_impl": self._clip_with_attention_impl,
            "clip.describe_tokens": self._clip_describe_tokens,
            "clip.generate_text": self._clip_generate_text,
            "gligen.apply_batched": self._gligen_apply_batched,
            "latent.noise_mask": self._latent_noise_mask,
            "latent.repeat_batch": self._latent_repeat_batch,
            "latent.minimax_h3_token_count":
                self._latent_minimax_h3_token_count,
            "latent.empty": self._latent_empty,
            "sigmas.steps": self._sigmas_steps,
            "sigmas.value_at": self._sigmas_value_at,
            "sampler.named": self._sampler_named,
            "cond.sequence_length": self._cond_sequence_length,
            "cond.combine": self._cond_combine,
            "cond.concat": self._cond_concat,
            "cond.zero_out": self._cond_zero_out,
            "cond.with_timestep_range": self._cond_with_timestep_range,
            "cond.with_metadata": self._cond_with_metadata,
            "cond.has_spatial_metadata": self._cond_has_spatial_metadata,
            "cond.with_mask": self._cond_with_mask,
            "cond.with_clip_vision_output":
                self._cond_with_clip_vision_output,
            "cond.with_concat_latent": self._cond_with_concat_latent,
            "cond.spatial_crop": self._cond_spatial_crop,
            "latent.spatial_shape": self._latent_spatial_shape,
            "latent.resize": self._latent_resize,
            "latent.random_noise": self._latent_random_noise,
            "latent.composite": self._latent_composite,
            "clip.scale_attention_weights": self._clip_scale_attention_weights,
            "advanced_control.weights_from_list":
                _vendor_ops.advanced_control_weights_from_list,
            "advanced_control.scaled_soft_weights":
                _vendor_ops.advanced_control_scaled_soft_weights,
            "lora.weight_differences": self._lora_weight_differences,
            "weight_diff.next": self._weight_diff_next,
            "model.apply_lora": self._model_apply_lora,
            "model.apply_dit_block_lora": self._model_apply_dit_block_lora,
            "model.apply_ltx2_lora": self._model_apply_ltx2_lora,
            "sampling.spatial_crop_inputs":
                self._sampling_spatial_crop_inputs,
            "model.patch": self._model_patch,
            "model.ground_image": self._model_ground_image,
            "model.transforms": self._model_transforms,
            "model.is_flow": self._model_is_flow,
            "model.family": self._model_family,
            "model.unet_context_dim": self._model_unet_context_dim,
            "model.is_zero_terminal_snr": self._model_is_zero_terminal_snr,
            "model.sigma_for_percent": self._model_sigma_for_percent,
            "model.sampling_sigma_delta": self._model_sampling_sigma_delta,
            "model.latent_scale_factor": self._model_latent_scale_factor,
            "guider.scheduled_cfg": self._guider_scheduled_cfg,
            "sampler.self_refine_video": self._sampler_self_refine_video,
            "clip_vision.encode_image": self._clip_vision_encode_image,
            "clip_vision_output.image_embeds":
                self._clip_vision_output_image_embeds,
            "clip_vision_output.concat": self._clip_vision_output_concat,
            "controlnet.with_union_type": self._controlnet_with_union_type,
            "controlnet.apply": self._controlnet_apply,
            "controlnet.apply_advanced": self._controlnet_apply_advanced,
            "controlnet.compile": self._controlnet_compile,
            "style_model.apply": self._style_model_apply,
            "clipseg.predict_mask": _vendor_ops.clipseg_predict_mask,
            "clipseg.segment": _vendor_ops.clipseg_segment,
            "image_classifier.classify": _vendor_ops.image_classifier_classify,
            "image_classifier.predict_scores":
                _vendor_ops.image_classifier_predict_scores,
            "classifier_scores.shape": _vendor_ops.classifier_scores_shape,
            "classifier_scores.select_above":
                _vendor_ops.classifier_scores_select_above,
            "semantic_segmentation.mask": _vendor_ops.semantic_segmentation_mask,
            "object_detector.detect": self._object_detector_detect,
            "inpaint_model.inpaint": _vendor_ops.inpaint_model_inpaint,
            "background_removal.mask": self._background_removal_mask,
            "brushnet.apply": self._brushnet_apply,
            "powerpaint.apply": self._powerpaint_apply,
            "image_preprocessor.apply": _vendor_ops.image_preprocessor_apply,
            "ipadapter.apply": _vendor_ops.ipadapter_apply,
            "ipadapter.apply_tiled": _vendor_ops.ipadapter_apply_tiled,
            "ipadapter.encode": _vendor_ops.ipadapter_encode,
            "ipadapter.apply_embeds": _vendor_ops.ipadapter_apply_embeds,
            "ipadapter_embeds.combine": _vendor_ops.ipadapter_embeds_combine,
            "sam.segment": _vendor_ops.sam_segment,
            "sam.segment_video": _vendor_ops.sam_segment_video,
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

    async def _ref_describe(
        self, ref: "Ref", max_value_chars: int = 32768,
    ) -> dict[str, Any]:
        """Project safe diagnostics without invoking behavior on the value."""
        if (isinstance(max_value_chars, bool)
                or not isinstance(max_value_chars, int)):
            raise TypeError("ref description max_value_chars must be an integer")
        if not 32 <= max_value_chars <= 32768:
            raise ValueError(
                "ref description max_value_chars must be in [32, 32768]")

        value = await current_runtime().refs.resolve(ref)
        kind = ref.kind[:128] if isinstance(ref.kind, str) else "UNKNOWN"
        shape: list[int] | None = None
        length: int | None = None
        first: str | None = None
        summary: str | None = None
        type_name = f"opaque {kind}"

        # Only exact, trusted structural cases are inspected.  In particular,
        # do not use hasattr(), len(), iter(), str(), or repr() on an arbitrary
        # host object: each can execute pack- or vendor-defined Python.
        import torch

        if isinstance(value, torch.Tensor):
            shape = [int(item) for item in value.shape]
            if shape:
                length = shape[0]
                first_shape = shape[1:]
                first = f"<redacted tensor slice shape={first_shape}>"
            type_name = "Tensor"
            summary = (
                f"<{kind} tensor shape={shape} dtype={value.dtype} "
                f"device={value.device.type}>"
            )
        elif ref.kind == "LATENT" and type(value) is dict:
            type_name = "Latent"
            length = len(value)
            first = "<latent field>" if value else None
            samples = value.get("samples")
            if isinstance(samples, torch.Tensor):
                shape = [int(item) for item in samples.shape]
            summary = f"<LATENT fields={length} shape={shape}>"
        elif ref.kind == "CONDITIONING" and type(value) is list:
            type_name = "Conditioning"
            length = len(value)
            first = "<conditioning row>" if value else None
            summary = f"<CONDITIONING rows={length}>"
        elif ref.kind == "VALUE":
            exact = type(value)
            if exact in {list, tuple, dict, str, bytes}:
                type_name = exact.__name__
                length = len(value)
                first = "<redacted item>" if length else None
                summary = f"<{type_name} length={length}>"
            elif value is None or exact in {bool, int, float}:
                type_name = "NoneType" if value is None else exact.__name__
                summary = repr(value)
            else:
                type_name = "opaque VALUE"
                summary = "<opaque VALUE>"
        else:
            summary = f"<opaque {kind}>"

        truncated = len(summary) > max_value_chars
        if truncated:
            summary = summary[:max_value_chars - 1] + "…"
        return {
            "kind": kind,
            "type": type_name,
            "length": length,
            "first": first,
            "shape": shape,
            "summary": summary,
            "truncated": truncated,
        }


    async def _invert(self, image: "ImageRef") -> "ImageRef":
        t = await current_runtime().refs.resolve(image)
        return ImageRef._wrap(await current_runtime().refs.create("IMAGE", 1.0 - t))  # type: ignore[return-value]

    async def _scale(self, image: "ImageRef", factor: float) -> "ImageRef":
        t = await current_runtime().refs.resolve(image)
        return ImageRef._wrap(await current_runtime().refs.create("IMAGE", t * factor))  # type: ignore[return-value]

    async def _image_rgb(self, image: "ImageRef") -> "ImageRef":
        import torch

        rt = current_runtime()
        value = await rt.refs.resolve(image)
        if not isinstance(value, torch.Tensor) or value.ndim < 3:
            raise TypeError("IMAGE must contain a channel-last tensor")
        if value.shape[-1] < 3:
            raise ValueError("IMAGE must contain at least three channels")
        return ImageRef._wrap(await rt.refs.create(
            "IMAGE", value[..., :3]))  # type: ignore[return-value]

    async def _image_to_device(
        self, image: "ImageRef", device: str = "auto",
    ) -> "ImageRef":
        import torch
        import comfy.model_management

        choices = {
            "auto": comfy.model_management.intermediate_device,
            "gpu": comfy.model_management.get_torch_device,
            "cpu": lambda: torch.device("cpu"),
        }
        if device not in choices:
            raise ValueError("image device must be auto, cpu, or gpu")
        rt = current_runtime()
        value = await rt.refs.resolve(image)
        result = value.clone().to(choices[device]())
        torch.cuda.empty_cache()
        return ImageRef._wrap(
            await rt.refs.create("IMAGE", result)
        )  # type: ignore[return-value]

    async def _image_spatial_shape(
        self, image: "ImageRef",
    ) -> tuple[int, int]:
        import torch

        value = await current_runtime().refs.resolve(image)
        if not isinstance(value, torch.Tensor) or value.ndim not in (3, 4):
            raise TypeError("IMAGE must contain an HWC or BHWC tensor")
        if value.shape[-1] not in (1, 3, 4):
            raise ValueError("IMAGE must have 1, 3, or 4 channels")
        return int(value.shape[-3]), int(value.shape[-2])

    async def _image_batch_size(self, image: "ImageRef") -> int:
        import torch

        value = await current_runtime().refs.resolve(image)
        if not isinstance(value, torch.Tensor) or value.ndim not in (3, 4):
            raise TypeError("IMAGE must contain an HWC or BHWC tensor")
        if value.shape[-1] not in (1, 3, 4):
            raise ValueError("IMAGE must have 1, 3, or 4 channels")
        batch = 1 if value.ndim == 3 else int(value.shape[0])
        if not 1 <= batch <= 4096:
            raise ValueError("IMAGE batch size must be in [1, 4096]")
        return batch

    async def _image_select_batch(
        self, image: "ImageRef", indices: list[int],
    ) -> "ImageRef":
        import torch

        if (
            not isinstance(indices, list)
            or not 1 <= len(indices) <= 4096
            or any(isinstance(index, bool) or not isinstance(index, int)
                   for index in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError(
                "image batch indices must be 1..4096 unique integers")
        rt = current_runtime()
        value = await rt.refs.resolve(image)
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 4
            or value.shape[-1] not in (1, 3, 4)
            or not 1 <= int(value.shape[0]) <= 4096
        ):
            raise TypeError("image batch selection requires a BHWC IMAGE")
        if min(indices) < 0 or max(indices) >= int(value.shape[0]):
            raise IndexError("image batch index is out of range")
        selected = value[indices]
        if selected.numel() > 268_435_456:
            raise ValueError("selected image batch is too large")
        return ImageRef._wrap(await rt.refs.create(
            "IMAGE", selected))  # type: ignore[return-value]

    async def _mask_grow(
        self, mask: "MaskRef", amount: int,
        tapered_corners: bool = False,
    ) -> "MaskRef":
        import torch
        from comfy_extras.nodes_mask import GrowMask

        if isinstance(amount, bool) or not isinstance(amount, int):
            raise TypeError("mask grow amount must be an integer")
        if not -512 <= amount <= 512:
            raise ValueError("mask grow amount must be in [-512, 512]")
        if type(tapered_corners) is not bool:
            raise TypeError("mask tapered_corners must be a bool")
        rt = current_runtime()
        value = await rt.refs.resolve(mask)
        if not isinstance(value, torch.Tensor) or value.ndim < 2:
            raise TypeError("MASK must contain a tensor with spatial axes")
        result = GrowMask.execute(
            value.detach().cpu(), amount, tapered_corners).result[0]
        return MaskRef._wrap(await rt.refs.create(
            "MASK", result))  # type: ignore[return-value]

    # --- operations on live engine objects ------------------------------- #
    # Each resolves its handles to the real objects HERE, on the trusted plane,
    # runs core's own semantics, and returns a handle. A guest never holds the
    # model; it holds the name of what it wanted done.

    @staticmethod
    def _ensure_vae_current_defaults(value: Any) -> None:
        """Normalize legacy VAE objects at the trusted API boundary.

        Some external VAE loaders copied an older ComfyUI initializer instead
        of calling the current base initializer.  Current encode/decode code
        legitimately expects these two inert attributes.  Supplying their
        canonical defaults only when absent keeps every VAE operation usable
        without exposing or replacing the external loader's implementation.
        """
        for name, default in (
            ("handles_tiling", False),
            ("format_encoded", None),
        ):
            if not hasattr(value, name):
                setattr(value, name, default)

    @staticmethod
    def _normalize_decoded_tensor(value: Any) -> Any:
        import torch

        if not isinstance(value, torch.Tensor):
            raise TypeError("VAE decode must return a tensor")
        if value.ndim == 5:
            value = value.reshape(
                -1, value.shape[-3], value.shape[-2], value.shape[-1])
        if value.ndim != 4:
            raise TypeError("VAE tensor decode must return BHWC or BTHWC")
        if any(int(size) < 1 for size in value.shape):
            raise ValueError("VAE tensor decode returned an empty dimension")
        if int(value.shape[-1]) > 4096:
            raise ValueError("VAE tensor decode has too many channels")
        return value

    @staticmethod
    def _validate_vae_decode_tiles(
        tile_size: int, overlap: int, temporal_size: int,
        temporal_overlap: int,
    ) -> None:
        if not 64 <= tile_size <= 4096:
            raise ValueError("VAE decode tile_size must be in [64, 4096]")
        if not 0 <= overlap <= 4096:
            raise ValueError("VAE decode overlap must be in [0, 4096]")
        if not 8 <= temporal_size <= 4096:
            raise ValueError(
                "VAE decode temporal_size must be in [8, 4096]")
        if not 4 <= temporal_overlap <= 4096:
            raise ValueError(
                "VAE decode temporal_overlap must be in [4, 4096]")

    async def _vae_decode(self, vae: "VaeRef", latent: "LatentRef") -> "ImageRef":
        rt = current_runtime()
        v = await rt.refs.resolve(vae)
        self._ensure_vae_current_defaults(v)
        samples = await rt.refs.resolve(latent)
        return ImageRef._wrap(await rt.refs.create("IMAGE", v.decode(samples["samples"])))  # type: ignore[return-value]

    async def _vae_latent_layout(
        self, vae: "VaeRef",
    ) -> dict[str, Optional[int]]:
        value = await current_runtime().refs.resolve(vae)

        channels = getattr(value, "latent_channels", None)
        spatial_fn = getattr(value, "spacial_compression_encode", None)
        if (
            isinstance(channels, bool)
            or not isinstance(channels, int)
            or not 1 <= channels <= 4096
            or not callable(spatial_fn)
        ):
            raise ValueError(
                "VAE does not publish a bounded latent layout")
        spatial = spatial_fn()
        if (
            isinstance(spatial, bool)
            or not isinstance(spatial, int)
            or not 1 <= spatial <= 256
        ):
            raise ValueError(
                "VAE spatial compression must be an integer in [1, 256]")

        temporal = None
        temporal_fn = getattr(value, "temporal_compression_encode", None)
        if callable(temporal_fn):
            temporal = temporal_fn()
            if (
                isinstance(temporal, bool)
                or not isinstance(temporal, int)
                or not 1 <= temporal <= 256
            ):
                raise ValueError(
                    "VAE temporal compression must be an integer in [1, 256]")
        return {
            "channels": channels,
            "spatial_compression": spatial,
            "temporal_compression": temporal,
        }

    async def _vae_decode_tensor(
        self, vae: "VaeRef", latent: "LatentRef",
    ) -> "TensorRef":
        rt = current_runtime()
        value = await rt.refs.resolve(vae)
        self._ensure_vae_current_defaults(value)
        latent_value = await rt.refs.resolve(latent)
        decoded = self._normalize_decoded_tensor(
            value.decode(latent_value["samples"]))
        return TensorRef._wrap(await rt.refs.create(
            "TENSOR", decoded))  # type: ignore[return-value]

    async def _vae_decode_tiled(
        self, vae: "VaeRef", latent: "LatentRef", tile_size: int = 512,
        overlap: int = 64, temporal_size: int = 64,
        temporal_overlap: int = 8,
    ) -> "ImageRef":
        from nodes import VAEDecodeTiled

        self._validate_vae_decode_tiles(
            tile_size, overlap, temporal_size, temporal_overlap)
        rt = current_runtime()
        value = await rt.refs.resolve(vae)
        self._ensure_vae_current_defaults(value)
        samples = await rt.refs.resolve(latent)
        pixels = VAEDecodeTiled().decode(
            value, samples, tile_size, overlap, temporal_size,
            temporal_overlap)[0]
        return ImageRef._wrap(await rt.refs.create(
            "IMAGE", pixels))  # type: ignore[return-value]

    async def _vae_decode_tensor_tiled(
        self, vae: "VaeRef", latent: "LatentRef", tile_size: int = 512,
        overlap: int = 64, temporal_size: int = 64,
        temporal_overlap: int = 8,
    ) -> "TensorRef":
        from nodes import VAEDecodeTiled

        self._validate_vae_decode_tiles(
            tile_size, overlap, temporal_size, temporal_overlap)
        rt = current_runtime()
        value = await rt.refs.resolve(vae)
        self._ensure_vae_current_defaults(value)
        samples = await rt.refs.resolve(latent)
        decoded = VAEDecodeTiled().decode(
            value, samples, tile_size, overlap, temporal_size,
            temporal_overlap)[0]
        decoded = self._normalize_decoded_tensor(decoded)
        return TensorRef._wrap(await rt.refs.create(
            "TENSOR", decoded))  # type: ignore[return-value]

    async def _vae_encode(self, vae: "VaeRef", image: "ImageRef") -> "LatentRef":
        rt = current_runtime()
        v = await rt.refs.resolve(vae)
        self._ensure_vae_current_defaults(v)
        pixels = await rt.refs.resolve(image)
        return LatentRef._wrap(await rt.refs.create(  # type: ignore[return-value]
            "LATENT", {"samples": v.encode(pixels)}))

    async def _vae_encode_for_inpaint(
        self, vae: "VaeRef", image: "ImageRef", mask: "MaskRef",
        grow_mask_by: int = 6,
    ) -> "LatentRef":
        from nodes import VAEEncodeForInpaint

        if (
            isinstance(grow_mask_by, bool)
            or not isinstance(grow_mask_by, int)
            or not 0 <= grow_mask_by <= 64
        ):
            raise ValueError("inpaint mask growth must be an integer in [0, 64]")
        rt = current_runtime()
        vae_value = await rt.refs.resolve(vae)
        self._ensure_vae_current_defaults(vae_value)
        pixels = await rt.refs.resolve(image)
        mask_value = await rt.refs.resolve(mask)
        result = VAEEncodeForInpaint().encode(
            vae_value, pixels, mask_value, grow_mask_by)[0]
        return LatentRef._wrap(await rt.refs.create(
            "LATENT", result))  # type: ignore[return-value]

    async def _vae_encode_inpaint_conditioning(
        self, vae: "VaeRef", image: "ImageRef", mask: "MaskRef",
        positive: "CondRef", negative: "CondRef",
        noise_mask: bool = True,
    ) -> tuple["CondRef", "CondRef", "LatentRef"]:
        from nodes import InpaintModelConditioning

        if type(noise_mask) is not bool:
            raise TypeError("inpaint noise_mask must be a bool")
        rt = current_runtime()
        values = await asyncio.gather(
            rt.refs.resolve(positive),
            rt.refs.resolve(negative),
            rt.refs.resolve(image),
            rt.refs.resolve(vae),
            rt.refs.resolve(mask),
        )
        self._ensure_vae_current_defaults(values[3])
        result = InpaintModelConditioning().encode(
            values[0], values[1], values[2], values[3], values[4],
            noise_mask)
        return (
            CondRef._wrap(await rt.refs.create("CONDITIONING", result[0])),
            CondRef._wrap(await rt.refs.create("CONDITIONING", result[1])),
            LatentRef._wrap(await rt.refs.create("LATENT", result[2])),
        )  # type: ignore[return-value]

    async def _vae_encode_tiled(
        self, vae: "VaeRef", image: "ImageRef", tile_x=None, tile_y=None,
        overlap=None, tile_t=None, overlap_t=None,
    ) -> "LatentRef":
        rt = current_runtime()
        value = await rt.refs.resolve(vae)
        self._ensure_vae_current_defaults(value)
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
        self._ensure_vae_current_defaults(value)
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
        self._ensure_vae_current_defaults(value)
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
        self._ensure_vae_current_defaults(value)
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

    async def _clip_encode_token_weights_component(
        self, clip: "ClipRef", component: str, tokens: list,
    ) -> tuple[TensorRef, Optional[TensorRef]]:
        import math

        from comfy import model_management

        if component not in {"l", "g"}:
            raise ValueError("CLIP component must be 'l' or 'g'")
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= 2048:
            raise ValueError("CLIP component tokens need 1..2048 chunks")
        total = 0
        for chunk_index, chunk in enumerate(tokens):
            if not isinstance(chunk, list) or not 1 <= len(chunk) <= 4096:
                raise ValueError(
                    f"CLIP token chunk {chunk_index} needs 1..4096 entries")
            total += len(chunk)
            if total > 131072:
                raise ValueError("CLIP component token input is too large")
            for entry_index, entry in enumerate(chunk):
                if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                    raise TypeError(
                        f"CLIP token entry {chunk_index}:{entry_index} must "
                        "contain token and weight")
                weight = entry[1]
                if (isinstance(weight, bool)
                        or not isinstance(weight, (int, float))
                        or not math.isfinite(float(weight))):
                    raise ValueError(
                        f"CLIP token weight {chunk_index}:{entry_index} "
                        "must be finite")

        rt = current_runtime()
        c = await rt.refs.resolve(clip)
        stage = getattr(c, "cond_stage_model", None)
        target = getattr(stage, f"clip_{component}", None)
        if target is None:
            raise ValueError(
                f"this text encoder has no CLIP-{component.upper()} component")

        stage.reset_clip_options()
        if getattr(c, "layer_idx", None) is not None:
            stage.set_clip_options({"layer": c.layer_idx})
        c.load_model()
        device = c.patcher.load_device
        stage.set_clip_options({"execution_device": device})
        with model_management.cuda_device_context(device):
            output = target.encode_token_weights(tokens)
        embedding, pooled = output[:2]
        embedding = embedding.detach().to(device="cpu")
        embedding_ref = TensorRef._wrap(
            await rt.refs.create("TENSOR", embedding)
        )
        pooled_ref = None
        if pooled is not None:
            pooled_ref = TensorRef._wrap(await rt.refs.create(
                "TENSOR", pooled.detach().to(device="cpu")
            ))
        return embedding_ref, pooled_ref

    async def _clip_encode(self, clip: "ClipRef", text: str) -> "CondRef":
        rt = current_runtime()
        c = await rt.refs.resolve(clip)
        tokens = c.tokenize(text)
        return CondRef._wrap(await rt.refs.create(  # type: ignore[return-value]
            "CONDITIONING", c.encode_from_tokens_scheduled(tokens)))

    async def _clip_generate_text(
        self, clip: "ClipRef", prompt: str,
        image: Optional["ImageRef"] = None,
        video: Optional["ImageRef"] = None,
        max_length: int = 256, do_sample: bool = False,
        temperature: float = 1.0, top_k: Optional[int] = 50,
        top_p: float = 0.95, min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None, presence_penalty: float = 0.0,
        thinking: bool = False, use_default_template: bool = True,
        num_beams: int = 1,
    ) -> str:
        from contextlib import nullcontext
        import math
        import torch

        prompt = str(prompt)
        if len(prompt) > 32768:
            raise ValueError("text-generation prompt exceeds 32768 characters")
        if (isinstance(max_length, bool) or not isinstance(max_length, int)
                or not 1 <= max_length <= 4096):
            raise ValueError("text-generation max_length must be in [1, 4096]")
        if (top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int)
            or not 0 <= top_k <= 1000
        )):
            raise ValueError("text-generation top_k must be null or in [0, 1000]")
        if (isinstance(num_beams, bool) or not isinstance(num_beams, int)
                or not 1 <= num_beams <= 8):
            raise ValueError("text-generation num_beams must be in [1, 8]")
        if num_beams > 1 and do_sample:
            raise ValueError("beam generation cannot also enable sampling")
        if (seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int)
            or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF
        )):
            raise ValueError("text-generation seed must be a uint64 or null")
        values = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "min_p": float(min_p),
            "repetition_penalty": float(repetition_penalty),
            "presence_penalty": float(presence_penalty),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("text-generation numeric options must be finite")
        if not 0.0 < values["temperature"] <= 2.0:
            raise ValueError("text-generation temperature must be in (0, 2]")
        if not 0.0 <= values["top_p"] <= 1.0:
            raise ValueError("text-generation top_p must be in [0, 1]")
        if not 0.0 <= values["min_p"] <= 1.0:
            raise ValueError("text-generation min_p must be in [0, 1]")
        if not 0.0 < values["repetition_penalty"] <= 5.0:
            raise ValueError(
                "text-generation repetition_penalty must be in (0, 5]")
        if not 0.0 <= values["presence_penalty"] <= 5.0:
            raise ValueError(
                "text-generation presence_penalty must be in [0, 5]")
        if type(do_sample) is not bool or type(thinking) is not bool:
            raise TypeError("text-generation switches must be booleans")
        if type(use_default_template) is not bool:
            raise TypeError("use_default_template must be a boolean")

        rt = current_runtime()
        c = await rt.refs.resolve(clip)
        if (not callable(getattr(c, "tokenize", None))
                or not callable(getattr(c, "generate", None))
                or not callable(getattr(c, "decode", None))):
            raise ValueError(
                "the selected text encoder does not support generation")
        pixels = None
        if image is not None:
            pixels = await rt.refs.resolve(image)
            if (not isinstance(pixels, torch.Tensor) or pixels.ndim != 4
                    or pixels.shape[0] != 1 or pixels.shape[-1] < 3):
                raise ValueError(
                    "image-conditioned text generation needs one BHWC image")
            height, width = map(int, pixels.shape[1:3])
            if (height <= 0 or width <= 0
                    or height * width > 268_435_456):
                raise ValueError("text-generation image dimensions are invalid")
        video_pixels = None
        if video is not None:
            video_pixels = await rt.refs.resolve(video)
            if (not isinstance(video_pixels, torch.Tensor)
                    or video_pixels.ndim != 4
                    or not 1 <= video_pixels.shape[0] <= 64
                    or video_pixels.shape[-1] < 3):
                raise ValueError(
                    "video-conditioned text generation needs 1 to 64 BHWC frames")
            frames, height, width = map(int, video_pixels.shape[:3])
            if (height <= 0 or width <= 0
                    or frames * height * width > 268_435_456):
                raise ValueError("text-generation video dimensions are invalid")

        family = getattr(c, "_secure_language_family", None)
        if top_k is None:
            top_k = 20 if isinstance(family, str) and family.startswith("qwen3") else 50

        model_lock = getattr(c, "_secure_text_generation_lock", None)
        with model_lock if model_lock is not None else nullcontext():
            tokens = c.tokenize(
                prompt,
                image=pixels,
                video=video_pixels,
                skip_template=not use_default_template,
                min_length=1,
                thinking=thinking,
            )
            generate_options = {
                "do_sample": do_sample,
                "max_length": max_length,
                "temperature": values["temperature"],
                "top_k": top_k,
                "top_p": values["top_p"],
                "min_p": values["min_p"],
                "repetition_penalty": values["repetition_penalty"],
                "seed": seed,
                "presence_penalty": values["presence_penalty"],
            }
            if num_beams != 1:
                generate_options["num_beams"] = num_beams
            generated = c.generate(tokens, **generate_options)
            result = c.decode(generated)
        if not isinstance(result, str) or len(result) > 1_048_576:
            raise RuntimeError("text encoder returned invalid generated text")
        return result.strip()

    async def _clip_scale_attention_weights(
        self, clip: "ClipRef", clip_l=None, clip_g=None, t5xxl=None,
        query: bool = True, key: bool = True,
        value: bool = True, output: bool = True,
    ) -> "ClipRef":
        import math
        import re

        def scales(name, values, expected):
            if values is None:
                return None
            if not isinstance(values, (list, tuple)) or len(values) != expected:
                raise ValueError(
                    f"{name} must contain exactly {expected} layer scales")
            checked = []
            for index, item in enumerate(values):
                if (
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    or not 0.0 <= float(item) <= 5.0
                ):
                    raise ValueError(
                        f"{name}[{index}] must be finite and in [0, 5]")
                checked.append(float(item))
            return checked

        clip_l = scales("clip_l", clip_l, 12)
        clip_g = scales("clip_g", clip_g, 32)
        t5xxl = scales("t5xxl", t5xxl, 24)
        switches = (query, key, value, output)
        if any(not isinstance(item, bool) for item in switches):
            raise TypeError("attention projection switches must be booleans")

        rt = current_runtime()
        source = await rt.refs.resolve(clip)
        patched = source.clone()
        state = patched.patcher.model_state_dict()

        def selected(key_name, *, t5=False):
            if t5:
                return (
                    (query and ".q." in key_name)
                    or (key and ".k." in key_name)
                    or (value and ".v." in key_name)
                    or (output and ".o." in key_name)
                )
            return (
                (query and "q_proj" in key_name)
                or (key and "k_proj" in key_name)
                or (value and "v_proj" in key_name)
                or (output and "out_proj" in key_name)
            )

        dual_clip = clip_l is not None and clip_g is not None
        for key_name in state:
            layer_scales = None
            layer = None
            if "self_attn" in key_name:
                match = re.search(r"\.layers\.(\d+)\.", key_name)
                if match is not None:
                    layer = int(match.group(1))
                    if dual_clip:
                        if "clip_l" in key_name:
                            layer_scales = clip_l
                        elif "clip_g" in key_name:
                            layer_scales = clip_g
                    else:
                        layer_scales = clip_l if clip_l is not None else clip_g
                is_selected = selected(key_name)
            elif "SelfAttention" in key_name:
                match = re.search(r"\.block\.(\d+)\.", key_name)
                if match is not None:
                    layer = int(match.group(1))
                    layer_scales = t5xxl
                is_selected = selected(key_name, t5=True)
            else:
                continue
            if (
                layer_scales is not None
                and layer is not None
                and layer < len(layer_scales)
                and layer_scales[layer] != 1.0
                and is_selected
            ):
                patched.add_patches(
                    {key_name: (None,)}, 0.0, layer_scales[layer])

        return ClipRef._wrap(
            await rt.refs.create("CLIP", patched)
        )  # type: ignore[return-value]

    async def _clip_set_last_layer(
        self, clip: "ClipRef", stop_at_clip_layer: int,
    ) -> "ClipRef":
        if isinstance(stop_at_clip_layer, bool):
            raise TypeError("CLIP layer must be an integer")
        layer = int(stop_at_clip_layer)
        if not -24 <= layer <= -1:
            raise ValueError("CLIP layer must be in [-24, -1]")
        rt = current_runtime()
        source = await rt.refs.resolve(clip)
        output = source.clone()
        output.clip_layer(layer)
        return ClipRef._wrap(
            await rt.refs.create("CLIP", output)
        )  # type: ignore[return-value]

    async def _clip_with_attention_impl(
        self, clip: "ClipRef", mode: str,
    ) -> "ClipRef":
        from comfy.ldm.modules import attention as attn

        if not isinstance(mode, str) or not 1 <= len(mode) <= 128:
            raise ValueError("CLIP attention mode must be a bounded string")
        try:
            attention_function = attn.get_attention_function(mode)
        except KeyError as error:
            raise ValueError(
                f"CLIP attention function {mode!r} is not registered"
            ) from error
        if not callable(attention_function):
            raise TypeError("registered CLIP attention implementation is invalid")

        rt = current_runtime()
        source = await rt.refs.resolve(clip)
        output = source.clone()
        patcher = getattr(output, "patcher", None)
        if patcher is None or not isinstance(
            getattr(patcher, "model_options", None), dict
        ):
            raise TypeError("CLIP attention selection needs a valid patcher")

        def override(_default, *args, **kwargs):
            return attention_function(*args, **kwargs)

        transformer_options = patcher.model_options.setdefault(
            "transformer_options", {})
        transformer_options = transformer_options.copy()
        transformer_options["optimized_attention_override"] = override
        patcher.model_options["transformer_options"] = transformer_options
        return ClipRef._wrap(
            await rt.refs.create("CLIP", output)
        )  # type: ignore[return-value]

    async def _clip_describe_tokens(
        self, clip: "ClipRef", tokens: dict,
    ) -> dict:
        from comfy.sd1_clip import SDTokenizer

        if not isinstance(tokens, dict) or not 1 <= len(tokens) <= 16:
            raise ValueError("CLIP tokens need 1 to 16 components")
        rt = current_runtime()
        source = await rt.refs.resolve(clip)
        tokenizer_root = getattr(source, "tokenizer", None)
        tokenizers = [
            value for value in vars(tokenizer_root).values()
            if isinstance(value, SDTokenizer)
        ] if tokenizer_root is not None else []
        descriptions = {}
        total = 0
        for tokenizer in tokenizers:
            key = str(tokenizer.embedding_key).replace("clip_", "")
            if key not in tokens:
                continue
            chunks = tokens[key]
            if not isinstance(chunks, list) or not 1 <= len(chunks) <= 2048:
                raise ValueError(f"CLIP token component {key!r} is invalid")
            special = {
                value for value in (
                    tokenizer.start_token,
                    tokenizer.end_token,
                    tokenizer.pad_token,
                ) if isinstance(value, int)
            }
            inv_vocab = getattr(tokenizer, "inv_vocab", None)
            if not isinstance(inv_vocab, dict):
                raise TypeError(
                    f"CLIP tokenizer {key!r} has no inverse vocabulary")
            described_chunks = []
            for chunk_index, chunk in enumerate(chunks):
                if not isinstance(chunk, list) or len(chunk) > 4096:
                    raise ValueError(
                        f"CLIP token chunk {key!r}:{chunk_index} is invalid")
                described = []
                for entry_index, entry in enumerate(chunk):
                    if (not isinstance(entry, (tuple, list)) or len(entry) < 1
                            or isinstance(entry[0], bool)
                            or not isinstance(entry[0], int)):
                        raise TypeError(
                            f"CLIP token {key!r}:{chunk_index}:"
                            f"{entry_index} has an invalid ID")
                    token_id = entry[0]
                    token_text = inv_vocab.get(token_id)
                    if not isinstance(token_text, str):
                        raise ValueError(
                            f"CLIP token {token_id} has no text description")
                    if len(token_text.encode("utf-8")) > 1024:
                        raise ValueError("CLIP token text exceeds 1024 bytes")
                    described.append({
                        "id": token_id,
                        "text": token_text,
                        "special": token_id in special,
                    })
                    total += 1
                    if total > 32768:
                        raise ValueError(
                            "CLIP token descriptions exceed 32768 entries")
                described_chunks.append(described)
            descriptions[key] = described_chunks
        missing = set(tokens) - set(descriptions)
        if missing:
            raise ValueError(
                f"CLIP token components are not describable: {sorted(missing)}")
        return descriptions

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

    async def _cond_with_mask(
        self, cond: "CondRef", mask: "MaskRef", strength: float = 1.0,
        set_area_to_bounds: bool = False,
    ) -> "CondRef":
        import math
        from nodes import ConditioningSetMask

        strength = float(strength)
        if not math.isfinite(strength) or not 0.0 <= strength <= 10.0:
            raise ValueError(
                "conditioning mask strength must be finite and in [0, 10]")
        if type(set_area_to_bounds) is not bool:
            raise TypeError("set_area_to_bounds must be a bool")
        rt = current_runtime()
        conditioning = await rt.refs.resolve(cond)
        mask_value = await rt.refs.resolve(mask)
        area = "mask bounds" if set_area_to_bounds else "default"
        result = ConditioningSetMask().append(
            conditioning, mask_value, area, strength)[0]
        return CondRef._wrap(await rt.refs.create(
            "CONDITIONING", result))  # type: ignore[return-value]

    async def _cond_with_clip_vision_output(
        self, cond: "CondRef", output: "ClipVisionOutputRef",
    ) -> "CondRef":
        import torch

        rt = current_runtime()
        conditioning = await rt.refs.resolve(cond)
        vision = await rt.refs.resolve(output)
        states = getattr(vision, "penultimate_hidden_states", None)
        if (
            not isinstance(conditioning, (list, tuple))
            or not conditioning
            or not isinstance(states, torch.Tensor)
            or states.ndim < 2
            or states.numel() < 1
            or states.numel() > 268_435_456
        ):
            raise ValueError(
                "clip-vision conditioning needs bounded conditioning and "
                "penultimate hidden states")
        result = []
        for index, item in enumerate(conditioning):
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
                or not isinstance(item[1], dict)
            ):
                raise TypeError(
                    f"conditioning row {index} has an invalid shape")
            metadata = item[1].copy()
            metadata["clip_vision_output"] = vision
            result.append([item[0], metadata])
        return CondRef._wrap(await rt.refs.create(
            "CONDITIONING", result))  # type: ignore[return-value]

    async def _cond_with_concat_latent(
        self, cond: "CondRef", model: "ModelRef", latent: "LatentRef",
        extra_latent: Optional["LatentRef"] = None,
    ) -> "CondRef":
        import copy
        import torch
        from comfy.conds import CONDRegular

        rt = current_runtime()
        conditioning = await rt.refs.resolve(cond)
        model_value = await rt.refs.resolve(model)
        latent_value = await rt.refs.resolve(latent)
        extra_value = (
            None if extra_latent is None
            else await rt.refs.resolve(extra_latent))
        if not isinstance(conditioning, (list, tuple)) or not conditioning:
            raise TypeError(
                "concat-latent conditioning must contain embedding rows")

        def samples(value: Any, name: str):
            result = value.get("samples") if isinstance(value, dict) else None
            if (
                not isinstance(result, torch.Tensor)
                or result.ndim != 4
                or not 1 <= result.shape[0] <= 64
                or not 1 <= result.shape[1] <= 64
                or result.shape[2] <= 0
                or result.shape[3] <= 0
                or result.numel() > 268_435_456
            ):
                raise ValueError(
                    f"{name} must contain a bounded BCHW latent tensor")
            return result

        tensors = [samples(latent_value, "latent")]
        if extra_value is not None:
            tensors.append(samples(extra_value, "extra_latent"))
            if any(
                tensor.shape[0] != tensors[0].shape[0]
                or tensor.shape[2:] != tensors[0].shape[2:]
                for tensor in tensors[1:]
            ):
                raise ValueError(
                    "concat latents must share batch and spatial dimensions")
        concat = torch.cat(tensors, dim=1)
        latent_format = getattr(
            getattr(model_value, "model", None), "latent_format", None)
        process = getattr(latent_format, "process_in", None)
        if not callable(process):
            raise ValueError(
                "the selected model has no latent-format converter")
        formatted = process(concat)
        if not isinstance(formatted, torch.Tensor):
            raise TypeError("model latent-format conversion returned no tensor")

        result = []
        for index, row in enumerate(conditioning):
            if (
                not isinstance(row, (list, tuple))
                or len(row) != 2
                or not isinstance(row[1], dict)
            ):
                raise TypeError(
                    f"conditioning row {index} has an invalid structure")
            metadata = copy.copy(row[1])
            model_conds = copy.copy(metadata.get("model_conds") or {})
            model_conds["c_concat"] = CONDRegular(formatted)
            metadata["model_conds"] = model_conds
            result.append([row[0], metadata])
        return CondRef._wrap(await rt.refs.create(
            "CONDITIONING", result))  # type: ignore[return-value]

    async def _cond_spatial_crop(
        self, cond: "CondRef", x: int, y: int, width: int, height: int,
        source_width: int, source_height: int,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> "CondRef":
        import torch
        import torch.nn.functional as F

        values = {
            "x": x, "y": y, "width": width, "height": height,
            "source_width": source_width, "source_height": source_height,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"conditioning crop {name} must be an integer")
        if not 1 <= source_width <= 16384 or not 1 <= source_height <= 16384:
            raise ValueError(
                "conditioning crop source dimensions must be in [1, 16384]")
        if width < 1 or height < 1 or x < 0 or y < 0:
            raise ValueError("conditioning crop window must be positive")
        if x + width > source_width or y + height > source_height:
            raise ValueError("conditioning crop window exceeds its source")
        if (target_width is None) != (target_height is None):
            raise ValueError(
                "conditioning crop target width and height must be paired")
        if target_width is not None and (
            isinstance(target_width, bool)
            or isinstance(target_height, bool)
            or not isinstance(target_width, int)
            or not isinstance(target_height, int)
            or not 1 <= target_width <= 16384
            or not 1 <= target_height <= 16384
        ):
            raise ValueError(
                "conditioning crop target dimensions must be integers "
                "in [1, 16384]")

        rt = current_runtime()
        source = await rt.refs.resolve(cond)
        if not isinstance(source, (list, tuple)):
            raise TypeError("conditioning must be a list of embedding rows")

        def crop_spatial_tensor(value: Any) -> Any:
            if not isinstance(value, torch.Tensor) or value.ndim < 2:
                return value
            full_height, full_width = value.shape[-2:]
            if full_height <= 1 and full_width <= 1:
                return value.clone()
            left = round(x * full_width / source_width)
            right = round((x + width) * full_width / source_width)
            top = round(y * full_height / source_height)
            bottom = round((y + height) * full_height / source_height)
            left = min(max(0, left), full_width - 1)
            top = min(max(0, top), full_height - 1)
            right = min(full_width, max(left + 1, right))
            bottom = min(full_height, max(top + 1, bottom))
            cropped = value[..., top:bottom, left:right].clone()
            if target_width is None:
                return cropped
            scaled_width = max(1, round(
                target_width * full_width / source_width))
            scaled_height = max(1, round(
                target_height * full_height / source_height))
            if tuple(cropped.shape[-2:]) == (scaled_height, scaled_width):
                return cropped
            original_dtype = cropped.dtype
            leading = tuple(cropped.shape[:-2])
            resized = F.interpolate(
                cropped.reshape(-1, 1, *cropped.shape[-2:]).float(),
                size=(scaled_height, scaled_width),
                mode="bilinear", align_corners=False,
            )
            return resized.reshape(
                *leading, scaled_height, scaled_width).to(
                    dtype=original_dtype)

        controls: dict[int, Any] = {}

        def crop_control(control: Any) -> Any:
            if control is None:
                return None
            identity = id(control)
            if identity in controls:
                return controls[identity]
            copy_method = getattr(control, "copy", None)
            if not callable(copy_method):
                # Unknown conditioning extensions remain opaque. Core-owned
                # ControlNet/T2I types all implement copy().
                return control
            clone = copy_method()
            controls[identity] = clone
            if hasattr(control, "cond_hint_original"):
                clone.cond_hint_original = crop_spatial_tensor(
                    control.cond_hint_original)
            if hasattr(clone, "cond_hint"):
                clone.cond_hint = None
            if hasattr(clone, "control_input"):
                clone.control_input = None
            if hasattr(control, "extra_concat_orig"):
                clone.extra_concat_orig = [
                    crop_spatial_tensor(item)
                    for item in control.extra_concat_orig
                ]
            previous = crop_control(
                getattr(control, "previous_controlnet", None))
            setter = getattr(clone, "set_previous_controlnet", None)
            if callable(setter):
                setter(previous)
            elif hasattr(clone, "previous_controlnet"):
                clone.previous_controlnet = previous
            return clone

        def resolve_area(area: Any) -> tuple[int, int, int, int] | None:
            if not isinstance(area, (tuple, list)):
                return None
            if len(area) == 5 and area[0] == "percentage":
                return (
                    max(1, round(float(area[1]) * source_height)),
                    max(1, round(float(area[2]) * source_width)),
                    round(float(area[3]) * source_height),
                    round(float(area[4]) * source_width),
                )
            if len(area) != 4:
                return None
            return tuple(int(item) for item in area)

        def crop_area(area: Any) -> tuple[int, int, int, int] | None:
            resolved = resolve_area(area)
            if resolved is None:
                return None
            area_height, area_width, area_y, area_x = resolved
            left = max(x, area_x)
            top = max(y, area_y)
            right = min(x + width, area_x + area_width)
            bottom = min(y + height, area_y + area_height)
            if right <= left or bottom <= top:
                return ()  # type: ignore[return-value]
            area = (bottom - top, right - left, top - y, left - x)
            if target_width is None:
                return area
            area_height, area_width, area_y, area_x = area
            return (
                max(1, round(area_height * target_height / height)),
                max(1, round(area_width * target_width / width)),
                round(area_y * target_height / height),
                round(area_x * target_width / width),
            )

        def crop_mask(mask: Any) -> Any:
            tensor = torch.as_tensor(mask)
            original_ndim = tensor.ndim
            if original_ndim == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(1)
            elif tensor.ndim != 4 or tensor.shape[1] != 1:
                raise ValueError(
                    "conditioning crop mask must be HW, BHW, or B1HW")
            tensor = tensor.float()
            if tuple(tensor.shape[-2:]) != (source_height, source_width):
                tensor = F.interpolate(
                    tensor, size=(source_height, source_width),
                    mode="bilinear", align_corners=False)
            tensor = tensor[..., y:y + height, x:x + width]
            if target_width is not None and tuple(tensor.shape[-2:]) != (
                target_height, target_width,
            ):
                tensor = F.interpolate(
                    tensor, size=(target_height, target_width),
                    mode="bilinear", align_corners=False)
            if original_ndim == 2:
                return tensor[0, 0]
            if original_ndim == 3:
                return tensor[:, 0]
            return tensor

        def crop_gligen_positions(positions: Any) -> Any:
            if not isinstance(positions, (list, tuple)):
                return positions
            output = []
            for position in positions:
                if (
                    isinstance(position, (list, tuple))
                    and len(position) == 5
                    and all(isinstance(item, (int, float))
                            for item in position[1:])
                ):
                    embedding, item_height, item_width, item_y, item_x = position
                    cropped = crop_area((
                        item_height, item_width, item_y, item_x))
                    if cropped:
                        output.append((embedding, *cropped))
                elif isinstance(position, (list, tuple)):
                    output.append(crop_gligen_positions(position))
                else:
                    output.append(position)
            return output

        def has_gligen_position(positions: Any) -> bool:
            if not isinstance(positions, (list, tuple)):
                return False
            if (
                len(positions) == 5
                and all(isinstance(item, (int, float))
                        for item in positions[1:])
            ):
                return True
            return any(has_gligen_position(item) for item in positions)

        result = []
        for row in source:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                raise TypeError(
                    "conditioning rows must contain embedding and metadata")
            metadata = dict(row[1])
            if "area" in metadata:
                cropped_area = crop_area(metadata["area"])
                if not cropped_area:
                    continue
                metadata["area"] = cropped_area
            if "mask" in metadata:
                metadata["mask"] = crop_mask(metadata["mask"])
                if not torch.any(metadata["mask"] != 0):
                    continue
            if "gligen" in metadata:
                gligen = metadata["gligen"]
                if isinstance(gligen, (list, tuple)) and len(gligen) == 3:
                    positions = crop_gligen_positions(gligen[2])
                    if has_gligen_position(positions):
                        metadata["gligen"] = (
                            gligen[0], gligen[1], positions)
                    else:
                        metadata.pop("gligen")
            if "control" in metadata:
                metadata["control"] = crop_control(metadata["control"])
            if isinstance(metadata.get("reference_latents"), (list, tuple)):
                metadata["reference_latents"] = [
                    crop_spatial_tensor(item)
                    for item in metadata["reference_latents"]
                ]
            result.append([row[0], metadata])
        return CondRef._wrap(await rt.refs.create(  # type: ignore[return-value]
            "CONDITIONING", result))

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

    async def _sigmas_steps(self, sigmas: "SigmasRef") -> int:
        import torch

        value = await current_runtime().refs.resolve(sigmas)
        if (not isinstance(value, torch.Tensor) or value.ndim != 1
                or not 2 <= int(value.numel()) <= 10001
                or not torch.isfinite(value).all()):
            raise ValueError(
                "SIGMAS must contain 2 to 10001 finite scalar values")
        return int(value.numel()) - 1

    async def _sigmas_value_at(
        self, sigmas: "SigmasRef", index: int,
    ) -> float:
        import math
        import torch

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("SIGMAS index must be an integer")
        value = await current_runtime().refs.resolve(sigmas)
        if (not isinstance(value, torch.Tensor) or value.ndim != 1
                or not 1 <= int(value.numel()) <= 10001
                or not torch.isfinite(value).all()):
            raise ValueError(
                "SIGMAS must contain 1 to 10001 finite scalar values")
        if not -int(value.numel()) <= index < int(value.numel()):
            raise IndexError("SIGMAS index is outside the schedule")
        result = float(value[index].item())
        if not math.isfinite(result):
            raise ValueError("SIGMAS value is not finite")
        return result

    async def _sampler_named(
        self, _subject: Optional["Ref"], name: str,
        eta: Optional[float] = None,
        ge_gamma: Optional[float] = None,
    ) -> "SamplerRef":
        import math
        import comfy.samplers

        if not isinstance(name, str) or name not in comfy.samplers.SAMPLER_NAMES:
            raise ValueError("unknown core sampler name")
        supplied = {
            key: value for key, value in {
                "eta": eta,
                "ge_gamma": ge_gamma,
            }.items() if value is not None
        }
        allowed = {
            "euler_ancestral_cfg_pp": {"eta"},
            "gradient_estimation": {"ge_gamma"},
            "gradient_estimation_cfg_pp": {"ge_gamma"},
        }.get(name, set())
        unknown = set(supplied) - allowed
        if unknown:
            raise ValueError(
                f"sampler {name!r} does not accept options {sorted(unknown)}")
        checked = {}
        if eta is not None:
            if isinstance(eta, bool) or not isinstance(eta, (int, float)):
                raise TypeError("sampler eta must be numeric")
            eta = float(eta)
            if not math.isfinite(eta) or not 0.0 <= eta <= 100.0:
                raise ValueError("sampler eta must be finite and in [0, 100]")
            checked["eta"] = eta
        if ge_gamma is not None:
            if (isinstance(ge_gamma, bool)
                    or not isinstance(ge_gamma, (int, float))):
                raise TypeError("sampler ge_gamma must be numeric")
            ge_gamma = float(ge_gamma)
            if not math.isfinite(ge_gamma) or not 2.0 <= ge_gamma <= 5.0:
                raise ValueError(
                    "sampler ge_gamma must be finite and in [2, 5]")
            checked["ge_gamma"] = ge_gamma
        sampler = (
            comfy.samplers.ksampler(name, checked)
            if checked else comfy.samplers.sampler_object(name)
        )
        return SamplerRef._wrap(await current_runtime().refs.create(
            "SAMPLER", sampler))  # type: ignore[return-value]

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

    async def _cond_zero_out(self, cond: "CondRef") -> "CondRef":
        from nodes import ConditioningZeroOut

        rt = current_runtime()
        source = await rt.refs.resolve(cond)
        result = ConditioningZeroOut().zero_out(source)[0]
        return CondRef._wrap(await rt.refs.create(
            "CONDITIONING", result))  # type: ignore[return-value]

    async def _cond_with_timestep_range(
        self, cond: "CondRef", start: float, end: float,
    ) -> "CondRef":
        import math
        from nodes import ConditioningSetTimestepRange

        start = float(start)
        end = float(end)
        if (not math.isfinite(start) or not math.isfinite(end)
                or not 0.0 <= start <= end <= 1.0):
            raise ValueError(
                "conditioning timestep range must satisfy "
                "0 <= start <= end <= 1")
        rt = current_runtime()
        source = await rt.refs.resolve(cond)
        result = ConditioningSetTimestepRange().set_range(
            source, start, end)[0]
        return CondRef._wrap(await rt.refs.create(
            "CONDITIONING", result))  # type: ignore[return-value]

    async def _cond_with_metadata(
        self, cond: "CondRef", width=None, height=None,
        crop_w=None, crop_h=None, target_width=None, target_height=None,
    ) -> "CondRef":
        import node_helpers
        from nodes import MAX_RESOLUTION

        values = {
            key: value for key, value in {
                "width": width,
                "height": height,
                "crop_w": crop_w,
                "crop_h": crop_h,
                "target_width": target_width,
                "target_height": target_height,
            }.items() if value is not None
        }
        if not values:
            raise ValueError("conditioning metadata needs at least one field")
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"conditioning metadata {key} must be an int")
            if not 0 <= value <= MAX_RESOLUTION:
                raise ValueError(
                    f"conditioning metadata {key} is outside the host limit")
        rt = current_runtime()
        source = await rt.refs.resolve(cond)
        result = node_helpers.conditioning_set_values(source, values)
        return CondRef._wrap(await rt.refs.create(
            "CONDITIONING", result))  # type: ignore[return-value]

    async def _cond_has_spatial_metadata(self, cond: "CondRef") -> bool:
        """Inspect metadata shape, never embeddings, for safe tile batching."""
        value = await current_runtime().refs.resolve(cond)
        if not isinstance(value, (list, tuple)):
            raise TypeError("conditioning must be a list of embedding rows")
        spatial_keys = {
            "area", "mask", "gligen", "control", "reference_latents",
        }
        for index, row in enumerate(value):
            if (
                not isinstance(row, (list, tuple))
                or len(row) < 2
                or not isinstance(row[1], dict)
            ):
                raise TypeError(
                    f"conditioning row {index} has an invalid structure")
            if spatial_keys.intersection(row[1]):
                return True
        return False

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



    async def _model_apply_lora(
        self, model: "ModelRef", asset: "AssetRef",
        clip: Optional["ClipRef"], strength_model: float,
        strength_clip: float,
    ) -> tuple["ModelRef", Optional["ClipRef"]]:
        import math

        import comfy.sd
        import comfy.utils
        import folder_paths

        strengths = {}
        for name, value in {
            "strength_model": strength_model,
            "strength_clip": strength_clip,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            checked = float(value)
            if not math.isfinite(checked) or not -100.0 <= checked <= 100.0:
                raise ValueError(f"{name} must be finite and in [-100, 100]")
            strengths[name] = checked

        if not isinstance(model, ModelRef) or model.kind != "MODEL":
            raise TypeError("LoRA application needs a MODEL ref")
        if not isinstance(asset, AssetRef) or asset.kind != "ASSET":
            raise TypeError("LoRA application needs an ASSET ref")
        if clip is not None and (
                not isinstance(clip, ClipRef) or clip.kind != "CLIP"):
            raise TypeError("clip must be a CLIP ref or None")
        if clip is None and strengths["strength_clip"] != 0.0:
            raise ValueError("strength_clip must be zero when clip is None")
        if (strengths["strength_model"] == 0.0
                and strengths["strength_clip"] == 0.0):
            return model, clip

        rt = current_runtime()
        source_model = await rt.refs.resolve(model)
        source_clip = None if clip is None else await rt.refs.resolve(clip)
        path = await rt.refs.resolve(asset)
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError("LoRA ASSET ref does not contain a path")
        path = _InProcessAssets._confined_resolved_path(
            path, folder_paths.get_folder_paths("loras"), "loras")
        state_dict, metadata = comfy.utils.load_torch_file(
            path, safe_load=True, return_metadata=True)
        if not isinstance(state_dict, dict):
            raise TypeError("LoRA asset must contain a state-dict mapping")

        patched_model, patched_clip = comfy.sd.load_lora_for_models(
            source_model,
            source_clip,
            state_dict,
            strengths["strength_model"],
            strengths["strength_clip"],
            lora_metadata=metadata,
        )
        model_ref = ModelRef._wrap(await rt.refs.create(
            "MODEL", patched_model))
        clip_ref = None
        if patched_clip is not None:
            clip_ref = ClipRef._wrap(await rt.refs.create(
                "CLIP", patched_clip))
        return model_ref, clip_ref

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

    async def _latent_empty(
        self, _source, width: int, height: int, batch_size: int = 1,
        channels: int = 4,
        spatial_downscale_ratio: Optional[int] = None,
    ) -> "LatentRef":
        import torch

        values = {
            "width": width,
            "height": height,
            "batch_size": batch_size,
            "channels": channels,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values.values()
        ):
            raise TypeError("empty latent dimensions must be integers")
        if (
            spatial_downscale_ratio is not None
            and (
                isinstance(spatial_downscale_ratio, bool)
                or not isinstance(spatial_downscale_ratio, int)
            )
        ):
            raise TypeError("latent spatial downscale ratio must be an integer")
        ratio = 8 if spatial_downscale_ratio is None else spatial_downscale_ratio
        if (
            not 64 <= width <= 16384
            or not 64 <= height <= 16384
            or not 1 <= ratio <= 128
            or width % ratio
            or height % ratio
            or not 1 <= batch_size <= 64
            or not 1 <= channels <= 128
            or batch_size * channels * (width // ratio) * (height // ratio)
            > 16_777_216
        ):
            raise ValueError("empty latent dimensions exceed the bounded range")
        value = {
            "samples": torch.zeros(
                (batch_size, channels, height // ratio, width // ratio),
                dtype=torch.float32,
            )
        }
        if spatial_downscale_ratio is not None:
            # Canonical legacy spelling used by ComfyUI latent dictionaries.
            value["downscale_ratio_spacial"] = ratio
        return LatentRef._wrap(await current_runtime().refs.create(
            "LATENT", value))  # type: ignore[return-value]

    async def _latent_spatial_shape(
        self, latent: "LatentRef",
    ) -> tuple[int, int]:
        import torch

        value = await current_runtime().refs.resolve(latent)
        samples = value.get("samples") if isinstance(value, dict) else None
        if not isinstance(samples, torch.Tensor) or samples.ndim < 4:
            raise TypeError("LATENT must contain a sample tensor with spatial axes")
        height, width = map(int, samples.shape[-2:])
        if height <= 0 or width <= 0:
            raise ValueError("LATENT spatial dimensions must be positive")
        return height, width

    async def _latent_resize(
        self, latent: "LatentRef", width: int, height: int,
        method: str = "bilinear",
    ) -> "LatentRef":
        from comfy.utils import common_upscale
        import torch
        import torch.nn.functional as F

        if (
            isinstance(width, bool) or not isinstance(width, int)
            or isinstance(height, bool) or not isinstance(height, int)
        ):
            raise TypeError("latent resize dimensions must be integers")
        if not 1 <= width <= 16_384 or not 1 <= height <= 16_384:
            raise ValueError("latent resize dimensions must be in [1, 16384]")
        methods = {"nearest-exact", "bilinear", "area", "bicubic", "bislerp"}
        if method not in methods:
            raise ValueError(f"unknown latent resize method {method!r}")

        rt = current_runtime()
        source = await rt.refs.resolve(latent)
        if not isinstance(source, dict):
            raise TypeError("LATENT must be a mapping")
        samples = source.get("samples")
        if not isinstance(samples, torch.Tensor) or samples.ndim < 4:
            raise TypeError("LATENT must contain a sample tensor with spatial axes")
        if samples.numel() // max(1, samples.shape[-2] * samples.shape[-1]) * width * height > 67_108_864:
            raise ValueError("latent resize output exceeds the bounded tensor size")

        result = dict(source)
        result["samples"] = common_upscale(
            samples, width, height, method, "disabled")
        mask = source.get("noise_mask")
        if isinstance(mask, torch.Tensor):
            mask_value = mask
            while mask_value.ndim < 4:
                mask_value = mask_value.unsqueeze(1)
            result["noise_mask"] = F.interpolate(
                mask_value.float(), size=(height, width), mode="bilinear",
                align_corners=False,
            ).to(mask.dtype)
        return LatentRef._wrap(await rt.refs.create(
            "LATENT", result))  # type: ignore[return-value]

    async def _latent_random_noise(
        self, latent: "LatentRef", seed: int, source: str = "cpu",
        batch_size: Optional[int] = None,
    ) -> TensorRef:
        import math
        import numpy as np
        import torch

        from comfy import model_management

        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("noise seed must be an integer")
        if not 0 <= seed <= 0xffffffffffffffff:
            raise ValueError("noise seed must be in [0, 2**64 - 1]")
        if source not in {"cpu", "gpu"}:
            raise ValueError("noise source must be 'cpu' or 'gpu'")
        rt = current_runtime()
        value = await rt.refs.resolve(latent)
        if not isinstance(value, dict) or "samples" not in value:
            raise TypeError("LATENT ref has no samples")
        samples = value["samples"]
        if not isinstance(samples, torch.Tensor) or samples.ndim < 2:
            raise TypeError("LATENT samples must be a batched tensor")
        batch = int(samples.shape[0]) if batch_size is None else batch_size
        if isinstance(batch, bool) or not isinstance(batch, int):
            raise TypeError("noise batch_size must be an integer or None")
        if not 1 <= batch <= int(samples.shape[0]):
            raise ValueError("noise batch_size must fit the source latent")
        shape = (batch, *samples.shape[1:])
        if math.prod(shape) > 2_147_483_648:
            raise ValueError("requested noise tensor is too large")
        noise_indices = value.get("batch_index")
        if noise_indices is not None:
            if isinstance(noise_indices, torch.Tensor):
                noise_indices = noise_indices.detach().cpu().tolist()
            if not isinstance(noise_indices, (list, tuple)):
                raise TypeError("latent batch_index must be a sequence")
            noise_indices = list(noise_indices[:batch])
            if len(noise_indices) != batch or any(
                isinstance(index, bool) or not isinstance(index, (int, np.integer))
                or not 0 <= int(index) <= 65_535
                for index in noise_indices
            ):
                raise ValueError(
                    "latent batch_index must contain one bounded non-negative "
                    "integer per requested batch item"
                )
            noise_indices = [int(index) for index in noise_indices]
        device = (
            torch.device("cpu")
            if source == "cpu"
            else model_management.text_encoder_device()
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        source_samples = samples[:batch]
        if source == "cpu":
            # This is Comfy's canonical CPU stream (float32 generation followed
            # by a dtype cast), but with a private generator so the operation
            # does not mutate the trusted process's global RNG state.
            from comfy.sample import prepare_noise_inner

            noise = prepare_noise_inner(
                source_samples, generator, noise_indices
            )
        elif noise_indices is None:
            noise = torch.randn(
                shape,
                dtype=samples.dtype,
                layout=samples.layout,
                generator=generator,
                device=device,
            )
        else:
            unique, inverse = np.unique(noise_indices, return_inverse=True)
            selected = []
            for index in range(int(unique[-1]) + 1):
                item = torch.randn(
                    (1, *samples.shape[1:]),
                    dtype=samples.dtype,
                    layout=samples.layout,
                    generator=generator,
                    device=device,
                )
                if index in unique:
                    selected.append(item)
            noise = torch.cat([selected[index] for index in inverse], dim=0)
        noise = noise.to(device="cpu")
        return TensorRef._wrap(await rt.refs.create(
            "TENSOR", noise))  # type: ignore[return-value]

    async def _latent_noise_mask(
        self, latent: "LatentRef",
    ) -> Optional["MaskRef"]:
        import torch

        rt = current_runtime()
        value = await rt.refs.resolve(latent)
        mask = value.get("noise_mask") if isinstance(value, dict) else None
        if mask is None:
            return None
        if not isinstance(mask, torch.Tensor) or mask.ndim < 2:
            raise TypeError("LATENT noise_mask must be a tensor with spatial axes")
        return MaskRef._wrap(await rt.refs.create(
            "MASK", mask))  # type: ignore[return-value]

    async def _latent_repeat_batch(
        self, latent: "LatentRef", amount: int,
    ) -> "LatentRef":
        from nodes import RepeatLatentBatch

        if isinstance(amount, bool) or not isinstance(amount, int):
            raise TypeError("latent repeat amount must be an integer")
        if not 1 <= amount <= 64:
            raise ValueError("latent repeat amount must be in [1, 64]")
        rt = current_runtime()
        value = await rt.refs.resolve(latent)
        result = RepeatLatentBatch().repeat(value, amount)[0]
        return LatentRef._wrap(await rt.refs.create(
            "LATENT", result))  # type: ignore[return-value]

    async def _latent_composite(
        self, latent: "LatentRef", source: "LatentRef", x: int = 0,
        y: int = 0, resize_source: bool = False,
        mask: Optional["MaskRef"] = None,
    ) -> "LatentRef":
        from comfy_extras.nodes_mask import LatentCompositeMasked

        if (
            isinstance(x, bool) or not isinstance(x, int)
            or isinstance(y, bool) or not isinstance(y, int)
        ):
            raise TypeError("latent composite coordinates must be integers")
        if not -131_072 <= x <= 131_072 or not -131_072 <= y <= 131_072:
            raise ValueError("latent composite coordinates are out of range")
        if type(resize_source) is not bool:
            raise TypeError("latent composite resize_source must be a bool")
        rt = current_runtime()
        destination_value = await rt.refs.resolve(latent)
        source_value = await rt.refs.resolve(source)
        mask_value = None if mask is None else await rt.refs.resolve(mask)
        result = LatentCompositeMasked.execute(
            destination_value, source_value, x, y, resize_source,
            mask_value).result[0]
        return LatentRef._wrap(await rt.refs.create(
            "LATENT", result))  # type: ignore[return-value]

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

    async def _clip_vision_output_concat(
        self, output: "ClipVisionOutputRef",
        other: "ClipVisionOutputRef",
    ) -> "ClipVisionOutputRef":
        import torch
        import comfy.clip_vision

        rt = current_runtime()
        left = await rt.refs.resolve(output)
        right = await rt.refs.resolve(other)
        left_states = getattr(left, "penultimate_hidden_states", None)
        right_states = getattr(right, "penultimate_hidden_states", None)
        if (
            not isinstance(left_states, torch.Tensor)
            or not isinstance(right_states, torch.Tensor)
            or left_states.ndim < 2
            or right_states.ndim != left_states.ndim
            or left_states.dtype != right_states.dtype
            or left_states.device != right_states.device
            or any(
                left_states.shape[index] != right_states.shape[index]
                for index in range(left_states.ndim)
                if index != left_states.ndim - 2
            )
        ):
            raise ValueError(
                "CLIP-vision outputs must have compatible hidden states")
        combined = torch.cat((left_states, right_states), dim=-2)
        if combined.numel() < 1 or combined.numel() > 268_435_456:
            raise ValueError("combined CLIP-vision output is too large")
        result = comfy.clip_vision.Output()
        result.penultimate_hidden_states = combined
        return ClipVisionOutputRef._wrap(await rt.refs.create(
            "CLIP_VISION_OUTPUT", result))  # type: ignore[return-value]

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

    async def _controlnet_apply(
        self, control_net: "ControlNetRef", positive: "CondRef",
        negative: "CondRef", image: "ImageRef", strength: float = 1.0,
        start_percent: float = 0.0, end_percent: float = 1.0,
        vae: Optional["VaeRef"] = None,
    ) -> tuple["CondRef", "CondRef"]:
        import math

        def finite_number(value: Any, field: str) -> float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"ControlNet {field} must be a number")
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(f"ControlNet {field} must be finite")
            return result

        strength = finite_number(strength, "strength")
        start_percent = finite_number(start_percent, "start_percent")
        end_percent = finite_number(end_percent, "end_percent")
        if not -10.0 <= strength <= 10.0:
            raise ValueError("ControlNet strength must be in [-10, 10]")
        if not 0.0 <= start_percent <= end_percent <= 1.0:
            raise ValueError(
                "ControlNet percentages must satisfy "
                "0 <= start_percent <= end_percent <= 1")
        if strength == 0.0:
            return positive, negative

        rt = current_runtime()
        source = await rt.refs.resolve(control_net)
        positive_value = await rt.refs.resolve(positive)
        negative_value = await rt.refs.resolve(negative)
        pixels = await rt.refs.resolve(image)
        vae_value = None if vae is None else await rt.refs.resolve(vae)
        control_hint = pixels.movedim(-1, 1)
        control_nets: dict[Any, Any] = {}
        outputs = []
        for conditioning in (positive_value, negative_value):
            result = []
            for item in conditioning:
                metadata = item[1].copy()
                previous = metadata.get("control")
                if previous in control_nets:
                    applied = control_nets[previous]
                else:
                    applied = source.copy().set_cond_hint(
                        control_hint, strength,
                        (start_percent, end_percent), vae=vae_value,
                        extra_concat=[])
                    applied.set_previous_controlnet(previous)
                    control_nets[previous] = applied
                metadata["control"] = applied
                metadata["control_apply_to_uncond"] = False
                result.append([item[0], metadata])
            outputs.append(result)
        return (
            CondRef._wrap(await rt.refs.create(
                "CONDITIONING", outputs[0])),
            CondRef._wrap(await rt.refs.create(
                "CONDITIONING", outputs[1])),
        )  # type: ignore[return-value]

    async def _controlnet_apply_advanced(
        self, control_net: "ControlNetRef", positive: "CondRef",
        negative: "CondRef", image: "ImageRef", strength: float = 1.0,
        start_percent: float = 0.0, end_percent: float = 1.0,
        vae: Optional["VaeRef"] = None,
        mask: Optional["MaskRef"] = None,
        timestep_keyframe: Optional["TimestepKeyframeRef"] = None,
        weights: Optional["ControlNetWeightsRef"] = None,
    ) -> tuple["CondRef", "CondRef"]:
        import math

        def finite_number(value: Any, field: str) -> float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Advanced ControlNet {field} must be a number")
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(
                    f"Advanced ControlNet {field} must be finite")
            return result

        strength = finite_number(strength, "strength")
        start_percent = finite_number(start_percent, "start_percent")
        end_percent = finite_number(end_percent, "end_percent")
        if not -10.0 <= strength <= 10.0:
            raise ValueError(
                "Advanced ControlNet strength must be in [-10, 10]")
        if not 0.0 <= start_percent <= end_percent <= 1.0:
            raise ValueError(
                "Advanced ControlNet percentages must satisfy "
                "0 <= start_percent <= end_percent <= 1")
        if strength == 0.0:
            return positive, negative

        rt = current_runtime()
        source = await rt.refs.resolve(control_net)
        positive_value = await rt.refs.resolve(positive)
        negative_value = await rt.refs.resolve(negative)
        pixels = await rt.refs.resolve(image)
        vae_value = None if vae is None else await rt.refs.resolve(vae)
        mask_value = None if mask is None else await rt.refs.resolve(mask)
        keyframe_value = (
            None if timestep_keyframe is None
            else await rt.refs.resolve(timestep_keyframe))
        weights_value = (
            None if weights is None else await rt.refs.resolve(weights))
        advanced = _advanced_control_module("control")
        control_hint = pixels.movedim(-1, 1)
        control_nets: dict[Any, Any] = {}
        outputs = []
        for conditioning in (positive_value, negative_value):
            result = []
            for item in conditioning:
                metadata = item[1].copy()
                previous = metadata.get("control")
                if previous in control_nets:
                    applied = control_nets[previous]
                else:
                    applied = advanced.convert_to_advanced(
                        source.copy()).set_cond_hint(
                            control_hint, strength,
                            (start_percent, end_percent), vae_value)
                    if advanced.is_advanced_controlnet(applied):
                        applied.disarm()
                        wrapper_type = advanced.AbstractPreprocWrapper
                        is_wrapper = isinstance(control_hint, wrapper_type)
                        if (applied.allow_condhint_latents
                                and not applied.require_vae
                                and not is_wrapper):
                            raise TypeError(
                                f"{type(applied).__name__} requires a "
                                "preprocessed ControlNet image")
                        if (not applied.allow_condhint_latents and is_wrapper
                                and not applied.postpone_condhint_latents_check):
                            raise TypeError(
                                f"{type(applied).__name__} requires a normal image")
                        if (applied.require_vae
                                and not (applied.allow_condhint_latents
                                         and is_wrapper)
                                and vae_value is None):
                            raise ValueError(
                                f"{type(applied).__name__} requires a VAE")
                        if keyframe_value is not None:
                            applied.set_timestep_keyframes(keyframe_value)
                        if weights_value is not None:
                            applied.weights_override = weights_value
                        applied.verify_all_weights()
                    if mask_value is not None:
                        effect_mask = mask_value.clone()
                        if len(effect_mask.shape) < 3:
                            effect_mask = effect_mask.unsqueeze(0)
                        applied.set_cond_hint_mask(effect_mask)
                    applied.set_previous_controlnet(previous)
                    control_nets[previous] = applied
                metadata["control"] = applied
                metadata["control_apply_to_uncond"] = False
                result.append([item[0], metadata])
            outputs.append(result)
        return (
            CondRef._wrap(await rt.refs.create(
                "CONDITIONING", outputs[0])),
            CondRef._wrap(await rt.refs.create(
                "CONDITIONING", outputs[1])),
        )  # type: ignore[return-value]

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


    async def _background_removal_mask(
        self, background_model: "BackgroundRemovalModelRef",
        image: "ImageRef",
    ) -> "MaskRef":
        import torch

        rt = current_runtime()
        bundle = await rt.refs.resolve(background_model)
        pixels = await rt.refs.resolve(image)
        if (
            not isinstance(bundle, dict)
            or bundle.get("secure_kind") != "background_removal.comfy"
        ):
            raise ValueError("unknown background-removal model")
        if (
            not isinstance(pixels, torch.Tensor)
            or pixels.ndim != 4
            or pixels.shape[-1] < 3
            or not 1 <= len(pixels) <= 4096
        ):
            raise ValueError(
                "background removal requires a non-empty BHWC image batch")
        height, width = map(int, pixels.shape[1:3])
        if height <= 0 or width <= 0 or len(pixels) * height * width > 67_108_864:
            raise ValueError("background-removal image batch is too large")
        with bundle["lock"]:
            result = bundle["model"].encode_image(pixels[..., :3])
        if (
            not isinstance(result, torch.Tensor)
            or result.ndim != 3
            or tuple(result.shape) != (len(pixels), height, width)
        ):
            raise RuntimeError(
                "background-removal model returned an invalid mask")
        return MaskRef._wrap(await rt.refs.create(
            "MASK", result.clamp(0.0, 1.0)
        ))  # type: ignore[return-value]

    async def _brushnet_apply(
        self, brushnet: "BrushNetRef", model: "ModelRef", vae: "VaeRef",
        image: "ImageRef", mask: "MaskRef", positive: "CondRef",
        negative: "CondRef", scale: float = 1.0, start_step: int = 0,
        end_step: int = 10000,
    ) -> tuple["ModelRef", "CondRef", "CondRef", "LatentRef"]:
        import math
        import torch
        import nodes

        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise TypeError("BrushNet scale must be a number")
        scale = float(scale)
        if not math.isfinite(scale) or not 0.0 <= scale <= 10.0:
            raise ValueError("BrushNet scale must be finite and in [0, 10]")
        if (isinstance(start_step, bool) or not isinstance(start_step, int)
                or isinstance(end_step, bool) or not isinstance(end_step, int)):
            raise TypeError("BrushNet start_step and end_step must be integers")
        if not 0 <= start_step <= end_step <= 10000:
            raise ValueError(
                "BrushNet steps must satisfy 0 <= start_step <= end_step <= 10000")

        rt = current_runtime()
        brushnet_value = await rt.refs.resolve(brushnet)
        model_value = await rt.refs.resolve(model)
        vae_value = await rt.refs.resolve(vae)
        pixels = await rt.refs.resolve(image)
        mask_value = await rt.refs.resolve(mask)
        positive_value = await rt.refs.resolve(positive)
        negative_value = await rt.refs.resolve(negative)
        if (not isinstance(brushnet_value, dict)
                or brushnet_value.get("brushnet") is None
                or brushnet_value.get("PP") is not False):
            raise TypeError("BRUSHNET_MODEL is not a host-loaded BrushNet model")
        if (not isinstance(pixels, torch.Tensor)
                or pixels.ndim not in {3, 4} or pixels.shape[-1] < 3):
            raise ValueError("BrushNet image must be an HWC or BHWC tensor")
        if not isinstance(mask_value, torch.Tensor) or mask_value.ndim not in {2, 3}:
            raise ValueError("BrushNet mask must be an HW or BHW tensor")
        image_height, image_width = map(int, pixels.shape[-3:-1])
        if tuple(map(int, mask_value.shape[-2:])) != (image_height, image_width):
            raise ValueError("BrushNet image and mask dimensions must match")
        image_batch = 1 if pixels.ndim == 3 else int(pixels.shape[0])
        mask_batch = 1 if mask_value.ndim == 2 else int(mask_value.shape[0])
        if (image_batch < 1 or image_batch > 4096
                or mask_batch < 1 or mask_batch > 4096
                or image_batch * image_height * image_width > 67_108_864):
            raise ValueError("BrushNet image batch is too large")
        if not isinstance(positive_value, list) or not isinstance(negative_value, list):
            raise TypeError("BrushNet conditioning must be host conditioning lists")

        node_class = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get("BrushNet")
        if node_class is None:
            raise RuntimeError(
                "BrushNet application requires the host-installed canonical "
                "ComfyUI-BrushNet extension")
        result = await asyncio.to_thread(
            node_class().model_update,
            model_value, vae_value, pixels, mask_value, brushnet_value,
            positive_value, negative_value, scale, start_step, end_step,
        )
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise RuntimeError("the canonical BrushNet node returned invalid outputs")
        patched, positive_out, negative_out, latent = result
        if (patched is None or not isinstance(positive_out, list)
                or not isinstance(negative_out, list)
                or not isinstance(latent, dict)
                or not isinstance(latent.get("samples"), torch.Tensor)):
            raise RuntimeError("the canonical BrushNet node returned invalid outputs")
        return (
            ModelRef._wrap(await rt.refs.create("MODEL", patched)),
            CondRef._wrap(await rt.refs.create("CONDITIONING", positive_out)),
            CondRef._wrap(await rt.refs.create("CONDITIONING", negative_out)),
            LatentRef._wrap(await rt.refs.create("LATENT", latent)),
        )  # type: ignore[return-value]

    async def _powerpaint_apply(
        self, powerpaint: "PowerPaintRef", model: "ModelRef", vae: "VaeRef",
        image: "ImageRef", mask: "MaskRef", positive: "CondRef",
        negative: "CondRef", fitting: float = 1.0,
        function: str = "text guided", scale: float = 1.0,
        start_step: int = 0, end_step: int = 10000,
        save_memory: str = "none",
    ) -> tuple["ModelRef", "CondRef", "CondRef", "LatentRef"]:
        import math
        import torch
        import nodes

        for label, value, minimum, maximum in (
            ("fitting", fitting, 0.3, 1.0),
            ("scale", scale, 0.0, 10.0),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"PowerPaint {label} must be a number")
            if not math.isfinite(float(value)) or not minimum <= float(value) <= maximum:
                raise ValueError(
                    f"PowerPaint {label} must be finite and in "
                    f"[{minimum}, {maximum}]")
        fitting = float(fitting)
        scale = float(scale)
        functions = {
            "text guided", "shape guided", "object removal",
            "context aware", "image outpainting",
        }
        if function not in functions:
            raise ValueError(f"unknown PowerPaint function {function!r}")
        if save_memory not in {"none", "auto", "max"}:
            raise ValueError(f"unknown PowerPaint save_memory {save_memory!r}")
        if (isinstance(start_step, bool) or not isinstance(start_step, int)
                or isinstance(end_step, bool) or not isinstance(end_step, int)):
            raise TypeError("PowerPaint start_step and end_step must be integers")
        if not 0 <= start_step <= end_step <= 10000:
            raise ValueError(
                "PowerPaint steps must satisfy "
                "0 <= start_step <= end_step <= 10000")

        rt = current_runtime()
        bundle = await rt.refs.resolve(powerpaint)
        model_value = await rt.refs.resolve(model)
        vae_value = await rt.refs.resolve(vae)
        pixels = await rt.refs.resolve(image)
        mask_value = await rt.refs.resolve(mask)
        positive_value = await rt.refs.resolve(positive)
        negative_value = await rt.refs.resolve(negative)
        if (not isinstance(bundle, dict)
                or bundle.get("secure_kind") != "powerpaint.pipeline"
                or not isinstance(bundle.get("powerpaint"), dict)
                or bundle["powerpaint"].get("PP") is not True
                or bundle.get("clip") is None):
            raise TypeError(
                "POWERPAINT_MODEL is not a host-loaded PowerPaint pipeline")
        if (not isinstance(pixels, torch.Tensor)
                or pixels.ndim not in {3, 4} or pixels.shape[-1] < 3):
            raise ValueError("PowerPaint image must be an HWC or BHWC tensor")
        if not isinstance(mask_value, torch.Tensor) or mask_value.ndim not in {2, 3}:
            raise ValueError("PowerPaint mask must be an HW or BHW tensor")
        height, width = map(int, pixels.shape[-3:-1])
        if tuple(map(int, mask_value.shape[-2:])) != (height, width):
            raise ValueError("PowerPaint image and mask dimensions must match")
        image_batch = 1 if pixels.ndim == 3 else int(pixels.shape[0])
        mask_batch = 1 if mask_value.ndim == 2 else int(mask_value.shape[0])
        if (image_batch < 1 or image_batch > 4096
                or mask_batch < 1 or mask_batch > 4096
                or image_batch * height * width > 67_108_864):
            raise ValueError("PowerPaint image batch is too large")
        if not isinstance(positive_value, list) or not isinstance(negative_value, list):
            raise TypeError("PowerPaint conditioning must be host conditioning lists")

        node_class = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get("PowerPaint")
        if node_class is None:
            raise RuntimeError(
                "PowerPaint application requires the host-installed canonical "
                "ComfyUI-BrushNet extension")
        result = await asyncio.to_thread(
            node_class().model_update,
            model_value, vae_value, pixels, mask_value,
            bundle["powerpaint"], bundle["clip"], positive_value,
            negative_value, fitting, function, scale, start_step, end_step,
            save_memory,
        )
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise RuntimeError(
                "the canonical PowerPaint node returned invalid outputs")
        patched, positive_out, negative_out, latent = result
        if (patched is None or not isinstance(positive_out, list)
                or not isinstance(negative_out, list)
                or not isinstance(latent, dict)
                or not isinstance(latent.get("samples"), torch.Tensor)):
            raise RuntimeError(
                "the canonical PowerPaint node returned invalid outputs")
        return (
            ModelRef._wrap(await rt.refs.create("MODEL", patched)),
            CondRef._wrap(await rt.refs.create("CONDITIONING", positive_out)),
            CondRef._wrap(await rt.refs.create("CONDITIONING", negative_out)),
            LatentRef._wrap(await rt.refs.create("LATENT", latent)),
        )  # type: ignore[return-value]











    async def _object_detector_detect(
        self, detector: "ObjectDetectorRef", image: "ImageRef",
        threshold: float = 0.5, class_name: str = "all",
        max_detections: int = 100,
    ) -> list[list[dict[str, Any]]]:
        import torch
        from comfy.ldm.rt_detr.rtdetr_v4 import COCO_CLASSES
        from comfy_extras.nodes_rtdetr import detect

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("object-detector threshold must be in [0, 1]")
        if class_name != "all" and class_name not in COCO_CLASSES:
            raise ValueError("object-detector class_name is not a COCO class")
        if not 1 <= max_detections <= 4096:
            raise ValueError("object-detector max_detections must be in [1, 4096]")
        rt = current_runtime()
        bundle = await rt.refs.resolve(detector)
        if (not isinstance(bundle, dict)
                or bundle.get("secure_kind") != "object_detector.rt_detr"):
            raise TypeError(
                "OBJECT_DETECTOR is not a trusted RT-DETR bundle")
        pixels = await rt.refs.resolve(image)
        if (not isinstance(pixels, torch.Tensor) or pixels.ndim != 4
                or pixels.shape[-1] < 3 or not 1 <= pixels.shape[0] <= 64):
            raise ValueError(
                "object detection requires a non-empty BHWC RGB batch")
        return detect(
            bundle["model"], pixels[..., :3], threshold, class_name,
            max_detections)









    async def _upscale_model_upscale(
        self, upscale_model: "UpscaleModelRef", images: "ImageRef",
        per_batch: int = 16, downscale_ratio: float = 1.0,
        downscale_method: str = "lanczos", precision: str = "float32",
        tile_size: Optional[int] = None, channels_last: bool = False,
    ) -> "ImageRef":
        import torch
        import comfy.model_management
        import comfy.utils
        from comfy.utils import common_upscale

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
        initial_tile = None
        if tile_size is not None:
            initial_tile = int(tile_size)
            if initial_tile != tile_size or not 0 <= initial_tile <= 2048:
                raise ValueError("upscale tile_size must be in [0, 2048]")
            initial_tile = 512 if initial_tile == 0 else max(initial_tile, 128)
        if not isinstance(channels_last, bool):
            raise TypeError("upscale channels_last must be a boolean")

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
            if initial_tile is None:
                # Preserve the original operation for every existing caller.
                model.to(device, dtype=dtype)
                source = pixels.movedim(-1, -3).to(dtype)
                progress = comfy.utils.ProgressBar(source.shape[0])
                for start in range(0, source.shape[0], batch_size):
                    batch = model(source[start:start + batch_size].to(device))
                    outputs.append(batch.cpu())
                    progress.update(batch.shape[0])
            else:
                # WhiteRabbit's advanced node uses ComfyUI's tiled runner and
                # autocast.  The model itself stays in its admitted dtype.
                from contextlib import nullcontext

                model.to(device)
                for parameter in model.model.parameters():
                    if parameter.device != device:
                        parameter.data = parameter.data.to(device)
                        if parameter.grad is not None:
                            parameter.grad.data = parameter.grad.data.to(device)
                model.model.eval()
                memory_required = int(
                    comfy.model_management.module_size(model.model))
                scale = float(model.scale)
                memory_required += int(
                    (512 * 512 * 3) * pixels.element_size()
                    * max(scale, 1.0) * 384.0)
                memory_required += pixels.nelement() * pixels.element_size()
                comfy.model_management.free_memory(memory_required, device)
                for start in range(0, pixels.shape[0], batch_size):
                    current = pixels[start:start + batch_size].movedim(
                        -1, -3).to(device, non_blocking=True)
                    if channels_last and device.type == "cuda":
                        current = current.to(memory_format=torch.channels_last)
                    tile = initial_tile
                    while tile >= 128:
                        try:
                            steps = (
                                current.shape[0]
                                * comfy.utils.get_tiled_scale_steps(
                                    current.shape[3], current.shape[2],
                                    tile_x=tile, tile_y=tile, overlap=32)
                            )
                            progress = comfy.utils.ProgressBar(steps)
                            precision_context = nullcontext()
                            if device.type == "cuda" and precision != "float32":
                                precision_context = torch.autocast(
                                    device_type="cuda", dtype=dtype)
                            with precision_context:
                                batch = comfy.utils.tiled_scale(
                                    current,
                                    model,
                                    tile_x=tile,
                                    tile_y=tile,
                                    overlap=32,
                                    upscale_amount=scale,
                                    pbar=progress,
                                )
                            outputs.append(batch.cpu())
                            break
                        except Exception as error:
                            comfy.model_management.raise_non_oom(error)
                            tile //= 2
                    else:
                        raise RuntimeError(
                            "upscale model exhausted safe tile sizes after GPU OOM")
        finally:
            model.to(previous_device, dtype=previous_dtype)
        output = torch.cat(outputs, dim=0).permute(0, 2, 3, 1).cpu().float()
        if initial_tile is not None:
            output = output.clamp(0.0, 1.0)
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
        cfg: float, start_percent: float = 0.0, end_percent: float = 1.0,
        bounds: Optional[dict] = None,
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
        sigma_bounds = None
        if bounds is not None:
            if (
                not isinstance(bounds, dict)
                or set(bounds) != {"unit", "start", "end"}
                or bounds.get("unit") != "sigma"
            ):
                raise ValueError(
                    "scheduled CFG bounds must be an exact sigma range")
            sigma_start = (
                None if bounds["start"] is None else checked_scalar(
                    "bounds.start", bounds["start"], 0.0, 1_000_000.0)
            )
            sigma_end = (
                None if bounds["end"] is None else checked_scalar(
                    "bounds.end", bounds["end"], 0.0, 1_000_000.0)
            )
            if (
                sigma_start is not None
                and sigma_end is not None
                and sigma_start < sigma_end
            ):
                raise ValueError(
                    "scheduled CFG sigma start must be at least its end")
            sigma_bounds = (sigma_start, sigma_end)
        if model.kind != "MODEL":
            raise TypeError("scheduled CFG needs a MODEL ref")
        if positive.kind != "CONDITIONING" or negative.kind != "CONDITIONING":
            raise TypeError(
                "scheduled CFG needs positive and negative CONDITIONING refs")

        class ScheduledCFGGuider(CFGGuider):
            def set_cfg(self, value, start, end, sigma_range):
                self.cfg = value
                self.start_percent = start
                self.end_percent = end
                self.sigma_bounds = sigma_range

            def predict_noise(
                self, x, timestep, model_options=None, seed=None,
            ):
                if model_options is None:
                    model_options = {}
                if self.sigma_bounds is not None:
                    if isinstance(timestep, torch.Tensor):
                        current_sigma = timestep.reshape(-1)[0]
                    else:
                        current_sigma = float(timestep)
                    sigma_start, sigma_end = self.sigma_bounds
                    if isinstance(current_sigma, torch.Tensor):
                        active = (
                            sigma_start is None
                            or bool(current_sigma <= current_sigma.new_tensor(sigma_start))
                        ) and (
                            sigma_end is None
                            or bool(current_sigma > current_sigma.new_tensor(sigma_end))
                        )
                    else:
                        active = (
                            (sigma_start is None or current_sigma <= sigma_start)
                            and (sigma_end is None or current_sigma > sigma_end)
                        )
                else:
                    steps = model_options[
                        "transformer_options"]["sample_sigmas"]
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
                    active = (
                        self.start_percent <= current_percent
                        <= self.end_percent
                    )
                if active:
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
        guider.set_cfg(cfg_value, start_value, end_value, sigma_bounds)
        return GuiderRef._wrap(
            await rt.refs.create("GUIDER", guider))  # type: ignore[return-value]

    async def _sampling_spatial_crop_inputs(
        self, owner: Ref, regions: list, source_width: int,
        source_height: int, target_width: int, target_height: int,
    ) -> Ref:
        """Clone a model/guider and crop only self-declared spatial patches.

        Tile planning deliberately stays in the calling node. Core supplies a
        narrow ownership boundary: a model patch may implement
        ``spatial_crop_inputs`` to return an independent patch containing the
        requested windows. Unknown patches are left untouched.
        """
        import copy

        import comfy.model_patcher

        if owner.kind not in {"MODEL", "GUIDER"}:
            raise TypeError(
                "spatial_crop_inputs requires a MODEL or GUIDER ref")
        if isinstance(source_width, bool) or not isinstance(source_width, int):
            raise TypeError("source_width must be an integer")
        if isinstance(source_height, bool) or not isinstance(source_height, int):
            raise TypeError("source_height must be an integer")
        if not 1 <= source_width <= 1_000_000:
            raise ValueError("source_width must be in [1, 1000000]")
        if not 1 <= source_height <= 1_000_000:
            raise ValueError("source_height must be in [1, 1000000]")
        if isinstance(target_width, bool) or not isinstance(target_width, int):
            raise TypeError("target_width must be an integer")
        if isinstance(target_height, bool) or not isinstance(target_height, int):
            raise TypeError("target_height must be an integer")
        if not 1 <= target_width <= 16384:
            raise ValueError("target_width must be in [1, 16384]")
        if not 1 <= target_height <= 16384:
            raise ValueError("target_height must be in [1, 16384]")
        if not isinstance(regions, list) or not 1 <= len(regions) <= 4096:
            raise ValueError("regions must contain between 1 and 4096 tiles")
        if len(regions) * target_width * target_height > 268_435_456:
            raise ValueError("spatial crop tile batch exceeds the size limit")

        checked_regions: list[tuple[int, int, int, int]] = []
        for region in regions:
            if not isinstance(region, (list, tuple)) or len(region) != 4:
                raise TypeError("each spatial crop region must have four integers")
            if any(isinstance(value, bool) or not isinstance(value, int)
                   for value in region):
                raise TypeError("each spatial crop region must have four integers")
            left, top, right, bottom = region
            if not (0 <= left < right <= source_width):
                raise ValueError("spatial crop x coordinates are outside the source")
            if not (0 <= top < bottom <= source_height):
                raise ValueError("spatial crop y coordinates are outside the source")
            checked_regions.append((left, top, right, bottom))

        def crop_model(value):
            if not isinstance(value, comfy.model_patcher.ModelPatcher):
                raise TypeError(
                    "spatial_crop_inputs requires a ComfyUI ModelPatcher")
            cloned = value.clone()
            patches = cloned.model_options.get(
                "transformer_options", {}).get("patches", {})
            replacements: dict[int, Any] = {}
            for module_patches in patches.values():
                for index, patch in enumerate(module_patches):
                    crop = getattr(patch, "spatial_crop_inputs", None)
                    if not callable(crop):
                        continue
                    identity = id(patch)
                    if identity not in replacements:
                        replacements[identity] = crop(
                            regions=checked_regions,
                            source_width=source_width,
                            source_height=source_height,
                            target_width=target_width,
                            target_height=target_height,
                        )
                    module_patches[index] = replacements[identity]
            return cloned

        runtime = current_runtime()
        value = await runtime.refs.resolve(owner)
        if owner.kind == "MODEL":
            cropped = crop_model(value)
            return ModelRef._wrap(await runtime.refs.create("MODEL", cropped))

        # Guiders own one or more ModelPatchers. A shallow object copy retains
        # the guider algorithm/configuration while each model attribute gets an
        # independent cropped clone. This also covers dual-model guiders.
        guider = copy.copy(value)
        model_replacements: dict[int, Any] = {}
        found_model = False
        for name, candidate in vars(value).items():
            if not isinstance(candidate, comfy.model_patcher.ModelPatcher):
                continue
            found_model = True
            identity = id(candidate)
            if identity not in model_replacements:
                model_replacements[identity] = crop_model(candidate)
            setattr(guider, name, model_replacements[identity])
        if not found_model:
            raise TypeError(
                "spatial_crop_inputs requires a guider that owns a ModelPatcher")
        primary = getattr(guider, "model_patcher", None)
        if primary is not None and hasattr(value, "model_options"):
            guider.model_options = primary.model_options
        return GuiderRef._wrap(await runtime.refs.create("GUIDER", guider))

    async def _model_ground_image(
        self, model: "ModelRef", image: "ImageRef",
        conditioning: "CondRef", threshold: float = 0.5,
        refine_iterations: int = 2, individual_masks: bool = True,
        max_detections: int = 64,
    ) -> tuple["MaskRef", list[list[dict[str, float]]]]:
        """Run the canonical core text-grounding implementation on SAM3.

        The reusable boundary is deliberately smaller than SAM3_Detect's full
        point/box prompting node. It exposes the common text-grounding intent
        needed by layout, crop, and interrogation packs while leaving those
        packs' layout and selection algorithms outside core.
        """
        import math
        import torch

        if not math.isfinite(float(threshold)) or not 0.0 <= threshold <= 1.0:
            raise ValueError("grounding threshold must be finite and in [0, 1]")
        if (isinstance(refine_iterations, bool)
                or not isinstance(refine_iterations, int)
                or not 0 <= refine_iterations <= 5):
            raise ValueError("grounding refine_iterations must be in [0, 5]")
        if not isinstance(individual_masks, bool):
            raise TypeError("grounding individual_masks must be a bool")
        if (isinstance(max_detections, bool)
                or not isinstance(max_detections, int)
                or not 1 <= max_detections <= 256):
            raise ValueError("grounding max_detections must be in [1, 256]")

        rt = current_runtime()
        model_value = await rt.refs.resolve(model)
        pixels = await rt.refs.resolve(image)
        cond_value = await rt.refs.resolve(conditioning)
        base_model = getattr(model_value, "model", None)
        diffusion_model = getattr(base_model, "diffusion_model", None)
        config = getattr(getattr(base_model, "model_config", None),
                         "unet_config", None)
        image_family = config.get("image_model") if isinstance(config, dict) else None
        if (image_family not in {"SAM3", "SAM31"}
                or type(diffusion_model).__module__
                != "comfy.ldm.sam3.detector"
                or type(diffusion_model).__name__ != "SAM3Model"):
            raise TypeError(
                "model.ground_image requires an official SAM3/SAM3.1 MODEL")
        if (not isinstance(pixels, torch.Tensor) or pixels.ndim != 4
                or pixels.shape[0] < 1 or pixels.shape[0] > 64
                or pixels.shape[-1] < 3):
            raise ValueError("image grounding requires a bounded BHWC RGB batch")
        batch, height, width = map(int, pixels.shape[:3])
        if (height < 1 or width < 1
                or batch * height * width > 268_435_456):
            raise ValueError("image grounding input dimensions are invalid")
        mask_planes = max_detections if individual_masks else batch
        if mask_planes * height * width > 268_435_456:
            raise ValueError("image grounding mask result would be too large")
        if (not isinstance(cond_value, (list, tuple)) or not cond_value
                or not isinstance(cond_value[0], (list, tuple))):
            raise TypeError("image grounding requires text CONDITIONING")

        from comfy_extras.nodes_sam3 import SAM3_Detect

        output = SAM3_Detect.execute(
            model_value,
            pixels,
            conditioning=cond_value,
            threshold=float(threshold),
            refine_iterations=refine_iterations,
            individual_masks=individual_masks,
        )
        values = getattr(output, "result", output)
        if (not isinstance(values, (list, tuple)) or len(values) < 2
                or not isinstance(values[0], torch.Tensor)
                or values[0].ndim != 3
                or not isinstance(values[1], list)
                or len(values[1]) != batch):
            raise RuntimeError("SAM3 grounding returned an invalid result")

        masks = values[0]
        projected: list[list[dict[str, float]]] = []
        remaining = max_detections
        for raw_frame in values[1]:
            if not isinstance(raw_frame, list):
                raise RuntimeError("SAM3 grounding returned invalid frame boxes")
            frame: list[dict[str, float]] = []
            for raw_box in raw_frame[:remaining]:
                if not isinstance(raw_box, dict):
                    raise RuntimeError("SAM3 grounding returned an invalid box")
                box: dict[str, float] = {}
                for key in ("x", "y", "width", "height", "score"):
                    value = raw_box.get(key)
                    if (type(value) not in (int, float)
                            or not math.isfinite(float(value))):
                        raise RuntimeError(
                            f"SAM3 grounding box has invalid {key}")
                    box[key] = float(value)
                if (box["width"] < 0.0 or box["height"] < 0.0
                        or not 0.0 <= box["score"] <= 1.0
                        or abs(box["x"]) > width * 4
                        or abs(box["y"]) > height * 4
                        or box["width"] > width * 4
                        or box["height"] > height * 4):
                    raise RuntimeError("SAM3 grounding box is outside its bounds")
                frame.append(box)
            remaining -= len(frame)
            projected.append(frame)

        detection_count = sum(map(len, projected))
        if individual_masks:
            if masks.shape[0] < detection_count:
                raise RuntimeError("SAM3 grounding returned fewer masks than boxes")
            masks = masks[:detection_count]
        elif masks.shape[0] != batch:
            raise RuntimeError("SAM3 union masks do not match the image batch")
        if tuple(map(int, masks.shape[-2:])) != (height, width):
            raise RuntimeError("SAM3 grounding masks have the wrong dimensions")

        mask_ref = MaskRef._wrap(await rt.refs.create("MASK", masks))
        return mask_ref, projected  # type: ignore[return-value]

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

    async def _model_is_flow(self, model: "ModelRef") -> bool:
        import comfy.model_base

        value = await current_runtime().refs.resolve(model)
        return value.model.model_type == comfy.model_base.ModelType.FLOW

    async def _model_family(self, model: "ModelRef") -> str:
        import comfy.supported_models

        value = await current_runtime().refs.resolve(model)
        config = getattr(getattr(value, "model", None), "model_config", None)
        families = (
            ("sdxl_refiner", (comfy.supported_models.SDXLRefiner,)),
            ("sdxl", (comfy.supported_models.SDXL,)),
            ("sd1", (
                comfy.supported_models.SD15,
                comfy.supported_models.SD20,
            )),
            ("svd", (comfy.supported_models.SVD_img2vid,)),
            ("sd3", (comfy.supported_models.SD3,)),
            ("hunyuan_dit", (comfy.supported_models.HunyuanDiT,)),
            ("flux", (comfy.supported_models.Flux,)),
            ("mochi", (comfy.supported_models.GenmoMochi,)),
        )
        for family, classes in families:
            if isinstance(config, classes):
                return family
        return "unknown"

    async def _model_unet_context_dim(
        self, model: "ModelRef",
    ) -> Optional[int]:
        value = await current_runtime().refs.resolve(model)
        model_config = getattr(getattr(value, "model", None), "model_config", None)
        unet_config = getattr(model_config, "unet_config", None)
        context_dim = (
            unet_config.get("context_dim")
            if isinstance(unet_config, dict) else None
        )
        if (
            isinstance(context_dim, bool)
            or not isinstance(context_dim, (int, float))
            or not 1 <= float(context_dim) <= 1_000_000
        ):
            return None
        return int(context_dim)

    async def _model_is_zero_terminal_snr(
        self, model: "ModelRef",
    ) -> bool:
        value = await current_runtime().refs.resolve(model)
        try:
            model_sampling = value.get_model_object("model_sampling")
        except (AttributeError, KeyError) as error:
            raise ValueError("MODEL has no sampling schedule") from error
        return bool(getattr(model_sampling, "zsnr", False))

    async def _model_sigma_for_percent(
        self, model: "ModelRef", percent: float,
        actual_endpoints: bool = False,
    ) -> float:
        import math

        if isinstance(percent, bool) or not isinstance(percent, (int, float)):
            raise TypeError("sampling percent must be numeric")
        percent = float(percent)
        if not math.isfinite(percent) or not 0.0 <= percent <= 1.0:
            raise ValueError("sampling percent must be finite and in [0, 1]")
        if type(actual_endpoints) is not bool:
            raise TypeError("actual_endpoints must be a bool")
        value = await current_runtime().refs.resolve(model)
        try:
            model_sampling = value.get_model_object("model_sampling")
        except (AttributeError, KeyError) as error:
            raise ValueError("MODEL has no sampling schedule") from error
        result = model_sampling.percent_to_sigma(percent)
        if actual_endpoints and percent == 0.0:
            result = model_sampling.sigma_max
        elif actual_endpoints and percent == 1.0:
            result = model_sampling.sigma_min
        if hasattr(result, "item"):
            result = result.item()
        result = float(result)
        if not math.isfinite(result) or abs(result) > 1_000_000_000_000.0:
            raise ValueError("MODEL returned an invalid sigma")
        return result

    async def _model_sampling_sigma_delta(
        self, model: "ModelRef", steps: int, sampler_name: str,
        scheduler: str, start_step: int, end_step: int,
        denoise: float = 1.0,
        sigma_schedule: Optional[dict] = None,
    ) -> float:
        import math
        import comfy.model_management
        import comfy.samplers

        steps = int(steps)
        start_step = int(start_step)
        end_step = int(end_step)
        denoise = float(denoise)
        if not 1 <= steps <= 10000:
            raise ValueError("steps must be in [1, 10000]")
        if not 0 <= start_step <= end_step <= steps:
            raise ValueError("sigma step range is outside the schedule")
        if not math.isfinite(denoise) or not 0.0 <= denoise <= 1.0:
            raise ValueError("denoise must be finite and in [0, 1]")
        if sampler_name not in comfy.samplers.KSampler.SAMPLERS:
            raise ValueError("unknown sampler name")
        if scheduler not in comfy.samplers.KSampler.SCHEDULERS:
            raise ValueError("unknown scheduler name")
        value = await current_runtime().refs.resolve(model)
        comfy.model_management.load_model_gpu(value)
        if sigma_schedule is None:
            sampler = comfy.samplers.KSampler(
                value,
                steps=steps,
                device=comfy.model_management.get_torch_device(),
                sampler=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
                model_options=value.model_options,
            )
            sigmas = sampler.sigmas
        else:
            if not isinstance(sigma_schedule, dict):
                raise TypeError("sigma_schedule must be a mapping or None")
            total_steps = steps if denoise > 0.9999 else int(steps / denoise)
            kind = sigma_schedule.get("kind")
            if kind == "gits":
                if set(sigma_schedule) != {"kind", "coeff", "denoise"}:
                    raise ValueError("GITS sigma schedule has unknown fields")
                coeff = float(sigma_schedule["coeff"])
                schedule_denoise = float(sigma_schedule["denoise"])
                if not 0.8 <= coeff <= 1.5:
                    raise ValueError("GITS coefficient must be in [0.8, 1.5]")
                if not 0.0 <= schedule_denoise <= 1.0:
                    raise ValueError("GITS denoise must be in [0, 1]")
                from comfy_extras.nodes_gits import GITSScheduler
                sigmas = GITSScheduler.execute(
                    coeff, total_steps, schedule_denoise,
                )[0]
            elif kind == "ays":
                if set(sigma_schedule) != {"kind", "model_type", "denoise"}:
                    raise ValueError("AYS sigma schedule has unknown fields")
                model_type = str(sigma_schedule["model_type"])
                schedule_denoise = float(sigma_schedule["denoise"])
                if model_type not in {"SD1", "SDXL", "SVD"}:
                    raise ValueError("AYS model type must be SD1, SDXL, or SVD")
                if not 0.0 <= schedule_denoise <= 1.0:
                    raise ValueError("AYS denoise must be in [0, 1]")
                from comfy_extras.nodes_align_your_steps import (
                    AlignYourStepsScheduler,
                )
                sigmas = AlignYourStepsScheduler().get_sigmas(
                    model_type, total_steps, schedule_denoise,
                )[0]
            else:
                raise ValueError("sigma_schedule kind is not supported")
            if denoise <= 0.9999:
                sigmas = sigmas[-(steps + 1):]
        scale = float(value.model.latent_format.scale_factor)
        return float((sigmas[start_step] - sigmas[end_step]).detach().cpu()) / scale


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


def _is_ipadapter_pipe(v: Any) -> bool:
    """Recognize the fixed host pipeline shape without inspecting its models."""
    return (
        isinstance(v, dict)
        and v.get("secure_kind") == "ipadapter.pipeline"
        and set(v) == {"secure_kind", "ipadapter", "clip_vision"}
        and isinstance(v.get("ipadapter"), dict)
        and v.get("clip_vision") is not None
    ) or (
        isinstance(v, (list, tuple))
        and len(v) == 5
        and callable(v[4])
    )


def _is_image_preprocessor(v: Any) -> bool:
    """Recognize Inspire's host-only SEGS provider protocol narrowly."""
    value_type = type(v)
    return (
        value_type.__module__.endswith("inspire.segs_support")
        and value_type.__name__.endswith("_wrapper")
        and callable(getattr(v, "apply", None))
    )


def _is_interpolation_states(v: Any) -> bool:
    """Recognize the fixed Frame-Interpolation policy without behavior."""
    value_type = type(v)
    if value_type.__name__ != "InterpolationStateList":
        return False
    try:
        fields = object.__getattribute__(v, "__dict__")
    except (AttributeError, TypeError):
        return False
    return (
        type(fields) is dict
        and set(fields) == {"frame_indices", "is_skip_list"}
    )


#: Live engine objects a node may receive, by the ref type that stands in for
#: them. Detection is duck-typed because these classes live in `comfy.*`, which
#: this module must not import.
def _ref_type_for(v: Any) -> tuple[type, str]:
    """Choose the narrowest handle that preserves the value's authority."""
    if _looks_like_tensor(v) and getattr(v, "ndim", None) == 1:
        return SigmasRef, "SIGMAS"
    if _looks_like_tensor(v):
        return ImageRef, "IMAGE"
    if _is_ipadapter_pipe(v):
        return Ref, "IPADAPTER_PIPE"
    if _is_image_preprocessor(v):
        return ImagePreprocessorRef, "IMAGE_PREPROCESSOR"
    if _is_interpolation_states(v):
        return InterpolationStatesRef, "INTERPOLATION_STATES"
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
        if v.get("secure_kind") == "image_inpaint.big-lama" and set(v) >= {
            "model", "architecture", "lock",
        }:
            return InpaintModelRef, "INPAINT_MODEL"
        if v.get("secure_kind") in {"sam.v1", "sam.v2"} and set(v) >= {
            "model", "architecture", "device_mode", "lock",
        }:
            return SamModelRef, "SAM_MODEL"
        if v.get("secure_kind") == "object_detector.rt_detr" and "model" in v:
            return ObjectDetectorRef, "OBJECT_DETECTOR"
        if v.get("secure_kind") == "classifier_scores.v1" and "scores" in v:
            return ClassifierScoresRef, "CLASSIFIER_SCORES"
        if v.get("secure_kind") == "image_classifier.onnx" and set(v) >= {
            "session", "input_name", "output_name", "class_count", "lock",
        }:
            return ImageClassifierRef, "IMAGE_CLASSIFIER"
        if v.get("secure_kind") == "powerpaint.pipeline" and set(v) >= {
            "powerpaint", "clip",
        }:
            return PowerPaintRef, "POWERPAINT_MODEL"
        if set(v) >= {"model", "processor", "architecture", "labels"}:
            return ImageClassifierRef, "IMAGE_CLASSIFIER"
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
    async def wrap(value: Any) -> Any:
        if _is_plain_data(value):
            return value

        # Custom structured sockets (for example rgthree's CONTEXT) may carry
        # several live engine objects. Preserve mappings/sequences and replace
        # each leaf with its narrow handle; wrapping the whole structure as
        # VALUE would try to export MODEL/CLIP objects as data.
        if isinstance(value, dict) and not (
            "samples" in value
            or ("waveform" in value and "sample_rate" in value)
            or set(value) >= {"model", "processor"}
            or _is_ipadapter_pipe(value)
        ):
            return {key: await wrap(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)) and not (
            _is_ipadapter_pipe(value)
            or (
                value
                and isinstance(value[0], (list, tuple))
                and len(value[0]) == 2
                and _looks_like_tensor(value[0][0])
                and isinstance(value[0][1], dict)
            )
        ):
            wrapped = [await wrap(item) for item in value]
            return tuple(wrapped) if isinstance(value, tuple) else wrapped

        ref_cls, kind = _ref_type_for(value)
        return ref_cls._wrap(await resolver.create(kind, value))

    return {key: await wrap(value) for key, value in inputs.items()}


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
from . import _vendor_ops  # noqa: E402


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

    @property
    def engaged(self) -> bool:
        """Whether anything has replaced a default in-process provider.

        False on a stock install, which is what lets core skip the SDK seam
        and keep its original node-invocation path. The ops provider is
        excluded: ops are reachable only through the seam, so extending the op
        vocabulary alone cannot change how an ordinary node runs.
        """
        return not (
            type(self.execution_backend) is InProcessExecutionBackend
            and type(self.ctx_provider) is InProcessCtxProvider
            and self.ref_resolver_factory is InProcessRefResolver
        )


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
