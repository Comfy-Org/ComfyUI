"""API versions, newest first, each numbered major.minor.patch except `latest`, the V3 API.

A pack imports one package and stays bound to it:

    from comfy_api.v4 import eng, io, ui

- Major: substantial differences from the major version before it. A new
  package, `v<major>`. The `_0` is dropped: `v4`, not `v4_0`.
- Minor: modest differences from the version before it, some of them breaking.
  A new package, `v<major>_<minor>`.
- Patch: additions to the API only. No feature of an earlier version is
  modified, and there is no new package: the import stays the same. A call that
  needs different inputs, results or behavior may be added beside the old one
  under a versioned name, `<name>_v2()`. A bug fix, or a change to what ComfyUI
  itself can do, is generally not a patch unless it adds to the API.

A pack checks for a patch with `eng.is_api_version_at_least("v4.0.0")`, or reads
`(major, minor, patch)` from `eng.get_api_version()`.

A new package is a different API from the one before it. It does not force a
pack to change: a pack keeps working on the package it imports until that package
is removed, and is converted only to move to a newer one. Breaking releases are
meant to be rare, and a major version long-lived, because moving packs to a new
one is hard even with tools.

`latest` remains the existing V3 API and `v0_0_2` its earlier numbered name;
`v0_0_1` is a template for adapters. Adding V4 does not change those packages.
"""

from comfy_api.v4 import ComfyAPI_v4
from comfy_api.latest import ComfyAPI_latest
from comfy_api.v0_0_2 import ComfyAPIAdapter_v0_0_2
from comfy_api.v0_0_1 import ComfyAPIAdapter_v0_0_1
from comfy_api.internal import ComfyAPIBase

supported_versions: list[type[ComfyAPIBase]] = [
    # The V4 API: V3's node definitions plus engine calls. Import `comfy_api.v4`.
    #   v4_0_0 - initial v4 version
    ComfyAPI_v4,
    # The existing V3 API. Import `comfy_api.latest`.
    ComfyAPI_latest,
    # The V3 API under an earlier numbered name. Import `comfy_api.v0_0_2`.
    ComfyAPIAdapter_v0_0_2,
    # A template for version adapters, not for packs.
    ComfyAPIAdapter_v0_0_1,
]

