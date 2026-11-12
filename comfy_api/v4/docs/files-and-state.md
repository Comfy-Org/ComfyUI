# Files, folders and state

**What a node produces is decided by its inputs.** Anything a pack gets another way, such as
model weights, must be the same every time. That is how ComfyUI is designed, and V4 packs are
expected to follow it.

A pack does not own a place on the disk. Opening paths the pack built itself, and keeping state
from one run to the next, are **not supported**: a pack that does either relies on the machine
it was written on. Python cannot stop a pack from doing either, so such a pack will seem to work
for its author and can fail for someone else.

## What a pack can count on

| What | How the pack gets it |
|---|---|
| The values a node works on, including the user's images, video and audio | The node's inputs. Everything a node needs arrives through them |
| A way to hand back its results | The node's outputs. ComfyUI's own nodes save them |
| A place for temporary files | `util.folders.get_temp_directory()` |
| A model ComfyUI loads | By name: `engine.models.list(folder)` for the choices and an `engine.models` call, such as `engine.models.load_checkpoint`, to load it |
| The pack's own files | The pack's folder, as it was installed |

```python
import os

from comfy_api.v4.util import folders

frames = os.path.join(folders.get_temp_directory(), "frames")
```

A pack may start a program, such as `ffmpeg`, to convert a file, as long as the files it reads and
writes are in the temp folder.

## Folder paths are provided for compatibility

A pack reads and writes files with ordinary Python, on paths `util.folders` gives it; V4 adds
only `util.files.load_torch_file` and `util.files.save_torch_file` for tensor files. `util.folders` has the
calls of ComfyUI's `folder_paths` that return paths in the user's folders, so that packs written
against it keep working:

- `util.folders.get_input_directory()`, `util.folders.get_annotated_filepath(name)`
- `util.folders.get_output_directory()`, `util.folders.get_save_image_path(...)`
- `util.folders.get_full_path(folder, name)`, `util.folders.get_folder_paths(folder)`,
  `util.folders.folder_names_and_paths`
- `util.folders.base_path`, `util.folders.models_dir`, `util.folders.get_user_directory()`

**ComfyUI decides what each of these calls answers**, and it is not always the folder the pack's
author had: ComfyUI's launch options move the folders, and a pack that runs in its own process
gets the folders that process was started with. So a pack must not assume that a path it is given
is one particular folder, or that a file it wrote stays. A pack that takes the user's files as
node inputs and returns its results as node outputs leaves reading and saving to ComfyUI's own
nodes, and depends on none of this.

## What is not supported

- **A path the pack built itself.** The home folder, `/tmp`, the current folder, ComfyUI's install
  folder, another pack's folder, a drive letter, a path kept in a settings file. None of these has
  to exist where the pack runs.
- **Needing a file from an earlier run.** A cache, a downloaded model, a settings file, a counter.
- **Values that did not come through the node's inputs deciding the result.** A route may serve
  the pack's page (a list, a preview, a check), but a node must not depend on what a route received
  earlier. Put the value in a widget, so that it is part of the workflow. The same goes for
  environment variables, for anything the pack keeps in memory from an earlier run, and for
  anything fetched from the network that can differ from one run to the next.

## Data that does not come through the inputs

Some packs need data that is not in the workflow, most often model weights the pack fetches
itself. That is allowed when the data is static: the same bytes whenever and wherever the pack
runs, so that it cannot change what a node produces for the same inputs.

- Fetch from an address whose content does not change, not one that follows a "latest" version,
  and check a hash where you can.
- Treat what you fetched as a cache: use it if it is there, and fetch it again if it is not.
- A pack that needs the network should say so where it is published, and fail with a clear
  message when it cannot reach what it needs.

## Every run is the first

A pack must work on its first run: nothing it wrote or kept in memory before may be needed.
ComfyUI empties the temp folder when it starts, and starts a pack's own process again after a
crash or a cancelled prompt, which loses what that process held in memory.

- **Within one run**, a file a node wrote in the temp folder can be read by another node of the
  same pack. Between packs, pass values through node outputs.
- **On the next run**, it may be gone. Code that uses something from an earlier run must work
  when it is missing: use it if it is there, otherwise build it again.

Models are not the pack's to keep either. One loaded through `engine.models` stays loaded for as
long as ComfyUI decides; one a pack loads with its own code lives only as long as the process it
was loaded in.
