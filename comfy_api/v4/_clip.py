from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict, TypeVar

import torch

from comfy_api.internal.singleton import ProxiedSingleton
from comfy_api.latest import io


WeightedToken = tuple[object, float] | tuple[object, float, int]
ClipTokens = dict[str, list[list[WeightedToken]]]
_PatchKey = TypeVar("_PatchKey", bound=str | tuple[object, ...])


class ClipEncoding(TypedDict):
    """Any other keys are the encoder's extra conditioning options, passed through as they are:
    only passing them on in a conditioning entry is part of the API."""

    cond: torch.Tensor
    pooled_output: torch.Tensor | None


class ClipInfo(TypedDict):
    layer_idx: int | None


ClipEncodingResult = (
    torch.Tensor | tuple[torch.Tensor, torch.Tensor | None] | ClipEncoding
)


class Clip(ProxiedSingleton):
    async def clone(self, clip: io.Clip.Type) -> io.Clip.Type:
        """Copy the encoder's settings and patches while sharing its model weights."""
        return clip.clone()

    async def tokenize(
        self,
        clip: io.Clip.Type,
        text: str,
        *,
        return_word_ids: bool = False,
        images: list[torch.Tensor] | None = None,
        llama_template: str | None = None,
    ) -> ClipTokens:
        """Return editable weighted tokens per encoder name, as a list of chunks of the prompt.

        A token is `(value, weight)`, or `(value, weight, word_index)` with `return_word_ids=True`;
        the word index counts the prompt's weighted pieces of text, not its words, and is 0 for
        start, end and padding tokens. Values may include embeddings or
        multimodal payloads. A prompt longer than one chunk gives several, which encoding joins
        end to end. `images` and `llama_template` are for the text encoders that take them.
        """
        options = {"images": images, "llama_template": llama_template}
        return clip.tokenize(
            text,
            return_word_ids=return_word_ids,
            **{name: value for name, value in options.items() if value is not None},
        )

    async def encode_from_tokens(
        self,
        clip: io.Clip.Type,
        tokens: ClipTokens,
        *,
        return_pooled: bool | Literal["unprojected"] = False,
        return_dict: bool = False,
    ) -> ClipEncodingResult:
        """Encode once, without applying a conditioning schedule.

        `return_pooled=True` gives a (conditioning, pooled) pair; "unprojected" gives the pooled
        output before projection. `return_dict=True` gives a `ClipEncoding` instead of a pair, and
        "unprojected" still selects the unprojected pooled output in it.
        """
        return clip.encode_from_tokens(
            tokens, return_pooled=return_pooled, return_dict=return_dict
        )

    async def encode_tokens(
        self,
        clip: io.Clip.Type,
        tokens: ClipTokens,
        *,
        unprojected: bool = False,
        add_dict: dict[str, object] | None = None,
        show_pbar: bool = True,
    ) -> io.Conditioning.Type:
        """Encode CONDITIONING entries, honoring the encoder's hook schedule.

        `add_dict` is merged into each entry's options, replacing keys it shares. With a hook
        schedule (from SetClipHooks) there is an entry per scheduled range, and ranges outside
        `add_dict`'s `start_percent`/`end_percent` are left out. `unprojected` keeps the pooled
        output before projection; `show_pbar` reports progress over the scheduled ranges.
        """
        return clip.encode_from_tokens_scheduled(
            tokens,
            unprojected=unprojected,
            add_dict=add_dict or {},
            show_pbar=show_pbar,
        )

    async def add_patches(
        self,
        clip: io.Clip.Type,
        patches: Mapping[_PatchKey, Sequence[object]],
        strength_patch: float = 1.0,
        strength_model: float = 1.0,
    ) -> io.Clip.Type:
        """Return a patched clone; leave the input encoder unchanged.

        Keys are names from `weight_keys`, or `(name, (dim, start, length))` to patch one slice.
        A one-item sequence holding a tensor of the weight's shape makes the weight
        `weight * strength_model + diff * strength_patch`, after any patches the encoder already
        has. Keys the encoder does not have are ignored; a difference of another shape is skipped
        with a logged warning.
        """
        patched = clip.clone()
        patched.add_patches(
            {key: tuple(value) for key, value in patches.items()},
            strength_patch,
            strength_model,
        )
        return patched

    async def read_weight(self, clip: io.Clip.Type, key: str) -> torch.Tensor:
        """Read a weight with this encoder's patches applied, as ComfyUI computes it.

        Without patches the result shares the model's storage: clone it before changing it.
        While this encoder or a clone of it is loaded, the stored weight can already hold the
        loaded clone's patches, and this encoder's are applied on top of it. Raises
        `AttributeError` for a name not in `weight_keys`.
        """
        return clip.patcher.patch_weight_to_device(key, return_weight=True).detach()

    async def info(self, clip: io.Clip.Type) -> ClipInfo:
        """Describe the selected output layer; None selects the encoder's default.

        `layer_idx` counts as CLIPSetLastLayer's `stop_at_clip_layer` does: -1 is the last layer.
        """
        return {"layer_idx": clip.layer_idx}

    async def weight_keys(self, clip: io.Clip.Type) -> list[str]:
        """List model weight names in sorted order, excluding tokenizer data."""
        return sorted(clip.patcher.model_state_dict())
