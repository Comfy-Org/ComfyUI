"""Core-owned node-closure kinds — the closed table behind ``ctx.closures``.

A node closure is a pack-side function the host RETAINS past the dispatch that
registered it and invokes at a declared sampling phase. It exists for the case
`ModelRef.patch` cannot serve: nodes whose value IS pack-authored math running
during sampling (post-CFG rescaling, attention coupling, custom samplers), where
moving the algorithm into core is exactly what we refuse to do.

The design rule this table enforces (see `docs/design-node-closure-captures.md`
and D21 in `docs/v2-api-decisions.md`):

    A closure closes over PACK-PLANE DATA ONLY. Its host-plane environment is
    not captured, it is DECLARED — named here, validated and resolved at
    registration while the registering dispatch is still live, and retained
    inside the closure's own registry entry so one release frees both.

So this module owns, per kind: the phase, the argument signature the host
supplies per invocation, the capture schema, the capability operations (usually
none), and the bounds. A guest supplies a function and declared captures; it
never supplies a phase implementation, an op name, or a module path.

Capture params reuse the `_model_transforms` `Param` family deliberately —
`RefOf` there is already "declare a tensor the host callback retains, resolve at
patch time, kind-check before any implementation sees an object", which is the
same move at a different scope.
"""
from __future__ import annotations

from typing import Any

from ._model_transforms import ListOfRefs, Param, RefOf


class ClosureError(Exception):
    """A closure request the host refuses. Always says what was wrong."""


# --------------------------------------------------------------------------- #
# Bounds
#
# A closure pins its captures into PROMPT scope, which is longer than any other
# guest-reachable lifetime in the system. These caps are what stop a pack from
# turning that into unbounded host memory. They are per closure; the registry
# count cap (MAX_RETAINED_CLOSURES, transport/wire.py) bounds the other axis.
# --------------------------------------------------------------------------- #
MAX_CAPTURE_ENTRIES = 32
MAX_CAPTURE_BYTES = 512 * 1024 * 1024

class ClosureKind:
    """One closed contract: when we call it, with what, and what it may hold."""

    def __init__(
        self,
        *,
        phase: str,
        doc: str,
        arguments: tuple[str, ...],
        returns: str,
        captures: dict[str, Param] | None = None,
        capabilities: tuple[str, ...] = (),
        stateful: bool = False,
    ) -> None:
        self.phase = phase
        self.doc = doc
        self.arguments = arguments
        self.returns = returns
        self.captures = captures or {}
        self.capabilities = capabilities
        self.stateful = stateful

    def validate_captures(self, supplied: dict | None) -> dict:
        """Check declared captures against this kind's schema.

        Runs during the REGISTERING dispatch, while any ref token supplied is
        still resolvable. Refusal here fails the registering node with a named
        error, before anything is retained.
        """
        supplied = dict(supplied or {})
        if len(supplied) > MAX_CAPTURE_ENTRIES:
            raise ClosureError(
                f"{len(supplied)} captures exceeds the limit of "
                f"{MAX_CAPTURE_ENTRIES}")
        unknown = set(supplied) - set(self.captures)
        if unknown:
            raise ClosureError(
                f"closure kind {self.phase!r} has no capture(s) "
                f"{sorted(unknown)}; it declares {sorted(self.captures)}")
        checked = {}
        for name, spec in self.captures.items():
            if name not in supplied:
                if spec.required:
                    raise ClosureError(
                        f"closure kind {self.phase!r} requires capture {name!r}")
                continue
            checked[name] = spec.check(name, supplied[name])
        return checked

    def describe(self) -> dict:
        return {
            "phase": self.phase,
            "doc": self.doc,
            "arguments": list(self.arguments),
            "returns": self.returns,
            "captures": {n: s.describe() for n, s in self.captures.items()},
            "capabilities": list(self.capabilities),
            "stateful": self.stateful,
        }


# --------------------------------------------------------------------------- #
# The table
#
# Only kinds whose delivery path is implemented and tested belong here. A kind
# listed but unimplemented would be an API promise the host cannot keep, so the
# any future closure contract is deliberately absent until its delivery and,
# where applicable, capability plumbing lands. Scheduler providers are a
# separate declarative manifest surface, not a closure kind.
# --------------------------------------------------------------------------- #
KINDS: dict[str, ClosureKind] = {
    "post_cfg": ClosureKind(
        phase="post_cfg",
        doc=(
            "Called after each guided denoise prediction, at most once per "
            "model evaluation. Returns the adjusted guided prediction."
        ),
        arguments=("guided", "cond", "uncond", "latent", "sigma", "cfg"),
        returns="guided",
    ),
    "pre_cfg": ClosureKind(
        phase="pre_cfg",
        doc=(
            "Called after conditional model predictions and before CFG "
            "combines them, at most once per model evaluation. Returns the "
            "same prediction list with pack-authored adjustments."
        ),
        arguments=("latent", "predictions", "presence", "sigma"),
        returns="predictions",
    ),
    "conditioning_selection": ClosureKind(
        phase="conditioning_selection",
        doc=(
            "Called before the host evaluates a conditional batch. Receives "
            "only branch-presence booleans and scalar sigma, and may disable "
            "branches without seeing conditioning objects."
        ),
        arguments=("presence", "sigma"),
        returns="presence",
    ),
    "conditioning_preprocess": ClosureKind(
        phase="conditioning_preprocess",
        doc=(
            "Called before a host conditional batch is evaluated. Receives "
            "only c_concat/c_crossattn tensor leaves, matching host-generated "
            "noise tensors, and sigma; conditioning wrappers stay host-owned."
        ),
        arguments=("conditioning_tensors", "noise_tensors", "sigma"),
        returns="conditioning_tensors",
    ),
    "latent_operation": ClosureKind(
        phase="latent_operation",
        doc=(
            "Called when a downstream host node applies a LATENT_OPERATION. "
            "Receives one latent tensor and returns one tensor with identical "
            "shape, dtype, and device."
        ),
        arguments=("latent",),
        returns="latent",
    ),
    "model_input_block": ClosureKind(
        phase="model_input_block",
        doc=(
            "Called after one canonical 2D UNet input block and control "
            "application, before the host saves its skip activation."
        ),
        arguments=("hidden", "sigmas", "block_index"),
        returns="hidden",
    ),
    "model_middle_block": ClosureKind(
        phase="model_middle_block",
        doc=(
            "Called after the canonical 2D UNet middle block and control "
            "application."
        ),
        arguments=("hidden", "sigmas", "block_index"),
        returns="hidden",
    ),
    "model_output_block": ClosureKind(
        phase="model_output_block",
        doc=(
            "Called after the host retrieves and applies control to one "
            "canonical 2D UNet skip, before hidden/skip concatenation."
        ),
        arguments=("hidden", "skip", "sigmas", "block_index"),
        returns="hidden_skip",
    ),
    "model_sigma": ClosureKind(
        phase="model_sigma",
        doc=(
            "Called by a wrapped sampler immediately before each model "
            "evaluation. Returns the sigma tensor passed to that evaluation; "
            "the underlying sampler and model call remain host-owned."
        ),
        arguments=(
            "sigma", "sigmas", "cfg", "start_sigma", "end_sigma",
        ),
        returns="sigma",
    ),
    "custom_sampler": ClosureKind(
        phase="custom_sampler",
        doc=(
            "Called once for a complete sampling run. The pack owns the "
            "integration loop and invocation-local history; an invocation-only "
            "broker exposes bounded denoise, noise, preview, and schedule "
            "operations while models and conditioning stay host-owned."
        ),
        arguments=("broker", "latent", "sigmas"),
        returns="latent",
        capabilities=(
            "denoise", "noise_like", "preview", "schedule_parameters",
        ),
        stateful=True,
    ),
    "regional_attention": ClosureKind(
        phase="regional_attention",
        doc=(
            "A paired regional cross-attention phase for canonical UNet and "
            "Anima/Cosmos models. The closure is prepared from declared "
            "CONDITIONING/MASK refs, expands attention inputs at each site, "
            "then mask-blends the matching output to the original batch."
        ),
        arguments=(
            "phase", "conditionings_or_primary", "strengths_or_secondary",
            "base_strength_or_tertiary", "masks_or_metadata",
        ),
        returns="phase-dependent attention tensors",
        captures={
            "base_conditioning": RefOf("CONDITIONING"),
            "conditionings": ListOfRefs(
                "CONDITIONING", min_items=0, max_items=31),
            "masks": ListOfRefs("MASK", min_items=1, max_items=32),
        },
        stateful=True,
    ),
    "clip_token_weight_encoder": ClosureKind(
        phase="clip_token_weight_encoder",
        doc=(
            "Transforms future host-owned CLIP component encodes into a "
            "paired key/value representation, or projects Anima signed "
            "T5 weights into absolute weights plus a typed sign sidecar. "
            "MODEL, CLIP, token objects, encoders, and weights stay host-owned."
        ),
        arguments=(
            "phase", "encoded_rows_or_signed_weights",
            "weight_rows_or_minimum_length", "empty_row_or_none",
        ),
        returns="phase-dependent encoded tensor or weight/mask tensor pair",
    ),
}


def get_kind(name: str) -> ClosureKind:
    """Resolve a kind name, refusing anything not in the closed table."""
    if not isinstance(name, str):
        raise ClosureError("closure kind must be a string")
    kind = KINDS.get(name)
    if kind is None:
        raise ClosureError(
            f"unknown closure kind {name!r}; supported kinds are "
            f"{sorted(KINDS)}")
    return kind


def describe_kinds() -> dict:
    """The generated-documentation view of the whole table."""
    return {name: kind.describe() for name, kind in sorted(KINDS.items())}
