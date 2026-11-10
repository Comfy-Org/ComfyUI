# Engine extensions

`eng` covers what packs commonly need from the engine. A few packs need more: a new model family, a kernel
that replaces a VAE's layers, a patch to a model's modules. For those, a pack can ship an **engine
extension**, a small file of code that runs next to the engine with full access to its objects, and that
nodes call through `eng.ext`.

Engine extensions are for major packs, and each one is reviewed. Most packs never need one: check
[The engine](engine.md) and [Sampling](sampling.md) first.

## What one looks like

An extension is one Python file in the pack's `v2/engine/` folder, with a class named `EngineOps`:

```
my-pack/
    v2/
        __init__.py
        nodes.py
        engine/
            unet_tools.py
```

```py
# v2/engine/unet_tools.py
import torch

import comfy.lora


class _Scale(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class EngineOps:
    def lora_key_map(self, model):
        """How LoRA weight names map to this model's weights, as ComfyUI's LoRA loader maps them."""
        return comfy.lora.model_lora_keys_unet(model.model, {})

    def scale_output(self, model, scale):
        """A copy of a UNet model whose final layer's output is scaled."""
        patched = model.clone()
        out = patched.get_model_object("diffusion_model.out")
        patched.add_object_patch("diffusion_model.out", torch.nn.Sequential(out, _Scale(scale)))
        return patched
```

A node calls its methods by file name and method name:

```py
key_map = await eng.ext.unet_tools.lora_key_map(model)
model = await eng.ext.unet_tools.scale_output(model, 0.95)
```

- `eng.ext.<file>.<method>(...)` calls `EngineOps.<method>` in `v2/engine/<file>.py`. Calls are awaited like
  every `eng` call; the methods themselves may be plain or `async`.
- `EngineOps` is created once, when the pack loads. Methods whose names start with `_` can't be called.
- A pack reaches only its own extensions (and the common ones below), never another pack's.
- The file can't share a name with an `eng` group: no `model.py` or `vae.py`.

## Where it runs

An extension runs in ComfyUI's process, next to the engine, in every mode. Its methods get the engine's real
objects: a MODEL is ComfyUI's `ModelPatcher`, a VAE is ComfyUI's `VAE`. That's the point of an extension, and
why it's held to stricter rules than node code.

Under pack isolation the pack's nodes run in its own process, but its extensions still load in ComfyUI's.
A node's call to `eng.ext` is sent to ComfyUI like any other `eng` call: the engine objects it passes arrive
as the real objects, and the engine objects the method returns go back to the node as handles. Pass
extensions values and engine objects; the pack's own objects stay in its process and can't be passed.

Follow the same rules as `eng`'s own calls:

- **Return a copy.** Clone a model before patching it, as `scale_output` does, so the caller's model, and
  every other node's, is unchanged.
- **Return plain data or engine objects,** never the engine's internals (a module, a `model_options` dict).
- **Don't patch ComfyUI's modules or classes,** and don't use the network, subprocesses or `pickle`.
- **Keep it small.** An extension is the part that has to touch the engine; everything else belongs in the
  pack's nodes.

## What an extension may import

The standard library, ComfyUI's own modules (`comfy`, `nodes`, `folder_paths`...), and the packages ComfyUI
itself requires (`torch`, `numpy`, `safetensors`...). Nothing else: an extension runs in ComfyUI's process,
so it can't bring a dependency ComfyUI doesn't have.

This is checked when the pack loads, by reading the file: every `import`, and every
`importlib.import_module` or `__import__` with a literal name. A relative import, or an import whose name
isn't a literal, is refused. An extension that needs another package can be proposed as a common extension.

## The allowlist

ComfyUI loads only extensions it has reviewed. `ALLOWED_EXTENSIONS` in `comfy_api/latest/_engine.py` lists
each by file name and the SHA-256 of its contents:

```py
ALLOWED_EXTENSIONS: list[tuple[str, str]] = [
    ("video_attention.py", "sha256:20d6f85ece9c1824aba90b3870acf5814b0f9d96c099f17c3a1f28d2a978fffd"),
]
```

A pack that ships an extension not on the list doesn't load, in every mode. The hash is over the exact
bytes that run, so any change to the file, even a comment, needs a new review and a new hash. The folder
holds only extension files: any other file or subfolder in `v2/engine/` also fails the load.

**While you're developing one,** set `COMFY_DEV_ENGINE_EXTENSIONS=1` to load extensions that aren't on the
list. The import check still applies. Get the file's row for the list with:

```
python -c "import hashlib, sys; print('sha256:' + hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())" v2/engine/unet_tools.py
```

**To get one allowed,** its row has to be added to that list. The list is part of ComfyUI, so that's a
change to ComfyUI itself, reviewed like any other.

## Common extensions

Some extensions are useful to many packs. These ship with ComfyUI, in `comfy_api/latest/engine/common/`, and
any pack can call them by name without shipping a file:

```py
vae = await eng.ext.vae_triton.patch(vae, fuse_norm_silu=True, int8_conv=False)
```

| Extension | What it does |
|---|---|
| `vae_triton` | `patch(vae, fuse_norm_silu=True, channels_last=True, int8_conv=False, autotune=False)`: a copy of the VAE with fused Triton kernels for its norm and SiLU layers, and optional int8 convolutions. Wan, KL-family (SD1.5, SDXL, Flux) and LTX VAEs |

A common extension may also import the packages it lists in `REQUIRES`, such as `triton`. Those must be
installed in ComfyUI's environment; calling one that's missing raises an error that names the package.

If a pack ships a file with the same name, the pack's own file is used.
