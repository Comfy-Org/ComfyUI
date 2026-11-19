"""Regenerate ``compositor_blend_golden.json``.

The golden file is the *shared contract* for layer blending. Every
implementation of these 26 modes must reproduce it within ``tolerance``:

* ``comfy_extras/compositor_blend.py`` - numpy, server-side compositing
* ``layerBlend.frag`` - GLSL, the live preview in the layer editor
* any future CPU reference in the frontend

Run from the repository root::

    python tests-unit/comfy_extras_test/compositor_blend_fixture_gen.py

and review the diff. A change to this file is a change to user-visible
blending behaviour in every implementation, so it should never be
regenerated just to make a test pass.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from comfy_extras.compositor_blend import CHANNEL_BLEND, HSL_BLEND, blend_pixel  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "compositor_blend_golden.json")

# Scalar grid for the per-channel modes: both endpoints, the midpoint, values
# just inside each endpoint, and values inside the 1e-6 epsilon guards.
SCALARS = [0.0, 1e-7, 0.001, 0.25, 0.5, 0.75, 0.999, 1.0 - 1e-7, 1.0]

# Colour pairs for the HSL modes, which read all three channels at once.
COLORS = [
    [0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
    [0.5, 0.5, 0.5],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.2, 0.4, 0.6],
    [0.9, 0.1, 0.35],
    [1e-7, 1e-7, 1e-7],
    [1e-7, 0.0, 0.0],
    [0.05, 0.05, 0.05],
]


def _round(value) -> float:
    return round(float(value), 7)


def build() -> dict:
    channel = {}
    for mode in CHANNEL_BLEND:
        rows = []
        for i in SCALARS:
            for l in SCALARS:
                out = blend_pixel(mode, np.float32([i] * 3), np.float32([l] * 3))
                rows.append([_round(i), _round(l), _round(np.asarray(out).reshape(3)[0])])
        channel[mode] = rows
    hsl = {}
    for mode in HSL_BLEND:
        rows = []
        for i in COLORS:
            for l in COLORS:
                out = blend_pixel(mode, np.float32(i), np.float32(l))
                rows.append([
                    [_round(v) for v in i],
                    [_round(v) for v in l],
                    [_round(v) for v in np.asarray(out).reshape(3)],
                ])
        hsl[mode] = rows
    return {
        "_comment": (
            "Golden blend values shared by comfy_extras/compositor_blend.py and "
            "layerBlend.frag. Inputs are unpremultiplied colours already in the "
            "blend space; outputs are unclamped (the compositor clamps once, at "
            "the end). 'channel' rows are [i, l, out] applied per channel; 'hsl' "
            "rows are [rgb_backdrop, rgb_layer, rgb_out]. Regenerate with "
            "tests-unit/comfy_extras_test/compositor_blend_fixture_gen.py."
        ),
        "tolerance": 1e-4,
        "channel": channel,
        "hsl": hsl,
    }


def dumps(data: dict) -> str:
    """One row per line, so a behaviour change shows up as a readable diff."""
    lines = ["{", f'  "_comment": {json.dumps(data["_comment"])},', f'  "tolerance": {data["tolerance"]},']
    for section in ("channel", "hsl"):
        lines.append(f'  "{section}": {{')
        modes = sorted(data[section])
        for m_index, mode in enumerate(modes):
            lines.append(f'    "{mode}": [')
            rows = data[section][mode]
            for r_index, row in enumerate(rows):
                comma = "" if r_index == len(rows) - 1 else ","
                lines.append(f"      {json.dumps(row)}{comma}")
            lines.append("    ]" + ("" if m_index == len(modes) - 1 else ","))
        lines.append("  }" + ("," if section == "channel" else ""))
    lines.append("}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    with open(GOLDEN_PATH, "w") as handle:
        handle.write(dumps(build()))
    sys.stdout.write(f"wrote {GOLDEN_PATH}\n")
