"""Conditioning: setting values in the options of its entries."""

from typing import Any

import node_helpers as _node_helpers
from comfy_api.latest import io

# Arguments go to ComfyUI by name, so a parameter renamed there fails here instead of shifting.


def set_values(
    conditioning: io.Conditioning.Type, values: dict[str, Any], append: bool = False
) -> io.Conditioning.Type:
    """Return a copy of `conditioning` with `values` set in every entry's options.

    With `append`, a value is added (`+`) to the one already there. Only the list, the entries and
    the options dicts are new: tensors and every other option value, such as a list or dict, are
    shared with `conditioning`.
    """
    return _node_helpers.conditioning_set_values(
        conditioning=conditioning, values=values, append=append
    )


def set_values_with_timestep_range(
    conditioning: io.Conditioning.Type,
    values: dict[str, Any],
    start_percent: float = 0.0,
    end_percent: float = 1.0,
) -> io.Conditioning.Type:
    """Set `values` as `set_values` does, from `start_percent` to `end_percent` only.

    An entry's own `"start_percent"`/`"end_percent"` is respected: the part of its range outside
    the requested one keeps its options, as extra entries, and an entry whose range does not
    overlap is put in the result as it is, not copied. If `start_percent` is above `end_percent`,
    `conditioning` itself is returned and a warning is logged.
    """
    return _node_helpers.conditioning_set_values_with_timestep_range(
        conditioning=conditioning,
        values=values,
        start_percent=start_percent,
        end_percent=end_percent,
    )


__all__ = ["set_values", "set_values_with_timestep_range"]
