# Files and state

Packs read and write files, keep presets, and cache expensive results. V4 gives each of those a call, so a
pack doesn't need to know where ComfyUI's folders are, doesn't write next to its own code, and keeps working
when its nodes run in another process.

| You want to | Use |
|---|---|
| Save an image, audio or a preview | The `ui` savers: [UI](ui.md) |
| Save bytes you encoded (a PNG, a `.safetensors`, text) | `eng.output.write` |
| Read a file the user uploaded or a node saved | `eng.files.read`, `list`, `exists` |
| Read a model file | `eng.models.read`, or a loader: [The engine](engine.md#loading-model-files) |
| Offer a dropdown of model files | `io.Combo.Input(..., options=io.ModelFolder("loras"))` |
| Keep a small value across runs and restarts | `eng.store` |
| Keep a large value in memory between runs | `eng.cache` |
| Give a library a path | `util.folders` |

## Saving output

`eng.output.write` saves bytes your node encoded, with ComfyUI's file naming:

```py
import io as bytes_io
from PIL import Image

buffer = bytes_io.BytesIO()
Image.fromarray(array).save(buffer, format="PNG")
saved = await eng.output.write(buffer.getvalue(), "my_pack/result", "png")
# {"filename": "result_00001_.png", "subfolder": "my_pack", "type": "output"}
```

- `filename_prefix` works as `SaveImage`'s does: subfolders, and `%date%`-style variables. It can't leave the
  output folder.
- A list of byte strings saves a batch with one counter run; `%batch_num%` in the prefix becomes each file's
  index.
- `folder="temp"` saves into the temp folder, for previews. ComfyUI empties it when it starts.

For images and audio, the `ui` savers do the encoding and metadata for you and return what the node shows;
they write through this same call.

## Reading files

Files in ComfyUI's input, output and temp folders are named by a ref, as ComfyUI's own nodes name them:

```py
data = await eng.files.read("photo.png")              # input folder
data = await eng.files.read("render.png [output]")
data = await eng.files.read("preview.png [temp]")

for entry in await eng.files.list("input", extensions=["png", "jpg"]):
    print(entry["ref"], entry["size"])

if await eng.files.exists(ref):
    ...
```

A ref can't leave those folders. For a dropdown of input images, list the input folder in `define_schema`
with `util.folders`, as ComfyUI's `LoadImage` does, and read the chosen file with `eng.files.read`.

## Model files

Model files are named by folder and file name, never by path:

```py
io.Combo.Input("model_name", options=io.ModelFolder("upscale_models"))   # the dropdown

weights = await eng.models.read("upscale_models", model_name)            # in execute
state_dict, metadata = weights["sd"], weights["metadata"]
```

A pack with its own kind of model file declares its folder on its `ComfyExtension`, so it appears in
ComfyUI's model folders (and in the user's `extra_model_paths.yaml` setup) like any other:

```py
class MyPack(ComfyExtension):
    async def get_model_folders(self):
        return {"my_pack_models": ["safetensors", "onnx"]}
```

Weights the pack needs, with their source and checksum, go in its declarations so they can be fetched before
it runs: see [Packs](packs.md#weights).

## Keeping values between runs

**`eng.store`** keeps small plain data (presets, settings, a counter) across runs and restarts, in ComfyUI's
user folder. Each value has a scope:

```py
presets = await eng.store.get("presets", default={})
presets[name] = values
await eng.store.set("presets", presets)                       # scope="pack": every workflow

await eng.store.set("last_seed", seed, scope="workflow")      # this workflow only
await eng.store.set("collapsed", True, scope="node")          # this node in this workflow
names = await eng.store.keys()
await eng.store.delete("presets")
```

Keys are up to 128 bytes and values up to 1 MB, with 256 entries and 8 MB per scope. The values must be plain
data: numbers, strings, lists and dicts.

**`eng.cache`** keeps large values (tensors, a model the pack built, an index) in ComfyUI's memory, least
recently used first out:

```py
detector = await eng.cache.get("detector")
if detector is None:
    detector = build_detector(await eng.models.read("my_pack_models", name))
    await eng.cache.set("detector", detector, ttl=900)        # seconds; default: until pushed out
```

Nothing in the cache survives a restart. Packs share `COMFY_PACK_CACHE_MB` (2,048 MB by default).

**Why not a module-level dict?** A module global is shared by every prompt, never freed, and, under pack
isolation, lives in a process that can be restarted. `eng.cache` is bounded, expires, and works the same in
every mode.

## Paths for libraries: `util.folders`

Some libraries insist on a path: a model hub client downloading into a cache folder, a tool that writes
temporary files. `util.folders` gives ComfyUI's folders by path, with the same names as `folder_paths`:

```py
out_dir = util.folders.get_output_directory()
lora_path = util.folders.get_full_path_or_raise("loras", lora_name)
names = util.folders.get_filename_list("loras")
```

Prefer the `eng` calls when they fit: they work wherever the node runs and keep the pack inside ComfyUI's
folders. A file written by path outside the output, temp and user folders may not be there after the node
returns.

## What not to do

- **Don't write next to the pack's code.** The pack's folder may be read-only, shared between users, or
  replaced on update. Use `eng.store`, or `eng.output.write` for real output.
- **Don't build paths from `__file__` to ComfyUI's folders.** Users move them. Use `util.folders`, `eng.files`
  or `eng.models`.
- **Don't keep caches in module globals.** Use `eng.cache`.
