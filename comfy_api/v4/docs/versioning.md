# Versioning

A pack imports one API package and stays bound to it. This page is about `comfy_api.v4`, whose
version is 4.0.0.

```python
from comfy_api.v4 import eng
```

## Versions and packages

An API version is numbered major.minor.patch.

- **Major**: substantial differences from the major version before it. It is a new package,
  named `v<major>`, without `_0`: `comfy_api.v4`, not `comfy_api.v4_0`.
- **Minor**: modest differences from the version before it, some of them breaking. It is a new
  package, named `v<major>_<minor>`.
- **Patch**: additions to the API only. No feature of an earlier version is modified, and there is
  no new package: the import stays the same. A bug fix, or a change to what ComfyUI itself can do,
  is generally not a patch unless it adds to the API.

So `comfy_api.v4` is always version 4.0.something, and only its patch number grows. An `eng` or
`util` call you use keeps its name, its parameters and the behaviour its documentation states for
as long as `comfy_api.v4` exists. Two things are outside that promise:

- A call that runs one of ComfyUI's own nodes, samplers or loaders, such as `eng.node.run` or
  `eng.sampler.sample`, does what ComfyUI's current one does.
- `io` and `ui` are V3's modules, passed through, and they gain what V3 gains. ComfyUI's tests
  fail if a change to V3 removes one of their names or changes how it is called.

A fix that makes a call do what its documentation says does not change the version.

A new package is a different API from the one before it. It does not force your pack to change:
the pack keeps working on the package it imports until that package is removed, and you move it
to a newer package only if you choose to.

## Versioned names

A patch cannot change a call. When a call needs different inputs, results or behaviour, the new
form is added beside the old one under a versioned name, `<name>_v2`, and the old call stays as it
was.

## Checking the version

On a ComfyUI without `comfy_api.v4`, the import raises `ModuleNotFoundError` and the pack does not
load. A patch can add calls, so a pack that uses a call added after 4.0.0 needs a ComfyUI that has
it. `eng.is_api_version_at_least` takes a version written `v4.0.0` or `4.0.0`:

```python
from comfy_api.v4 import eng

if not eng.is_api_version_at_least("v4.0.0"):
    raise RuntimeError("This pack needs a newer ComfyUI.")
```

`eng.get_api_version()` returns the version as `(major, minor, patch)`. Compare the whole tuple,
not each number on its own:

```python
from comfy_api.v4 import eng

major, minor, patch = eng.get_api_version()
supported = eng.get_api_version() >= (4, 0, 0)
```

`ComfyAPISync()` has the same two calls, for nodes that are not async.
