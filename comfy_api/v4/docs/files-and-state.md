# Files, folders and state

**What a node produces is decided by its inputs.** Anything a pack gets another way, such as
model weights, must be the same every time. That is how ComfyUI is designed, and V4 packs are
expected to follow it.

A pack does not own a place on the disk. ComfyUI can run a pack on a machine that has none of the
folders the pack's author had: no home folder, no ComfyUI install folder, no `input`, `output` or
`models` folder, no files left by an earlier run. Opening paths in the user's folders, and keeping
state from one run to the next, are **not supported** there. Python cannot stop a pack from doing
either, so a pack that does will seem to work on its author's machine and fail elsewhere.

## What a pack can count on

| What | How the pack gets it |
|---|---|
| The values a node works on, including the user's images, video and audio | The node's inputs. Everything a node needs arrives through them |
| A way to hand back its results | The node's outputs. ComfyUI's own nodes save them |
| A place for temporary files | `util.folders.get_temp_directory()` |
| A model ComfyUI loads | By name: `util.folders.get_filename_list(folder)` for the choices and an `eng.models` call, such as `eng.models.load_checkpoint`, to load it |
| The pack's own files | The pack's folder, as it was installed |

```python
import os

from comfy_api.v4.util import folders

frames = os.path.join(folders.get_temp_directory(), "frames")
```

A pack may start a program, such as `ffmpeg`, to convert a file, as long as the files it reads and
writes are in the temp folder.

## Folder paths are provided for compatibility

V4 has no file functions of its own. A pack reads and writes files with ordinary Python, on
paths `util.folders` gives it. `util.folders` has the calls of ComfyUI's `folder_paths` that
return paths in the user's folders, so that packs written against it keep working:

- `util.folders.get_input_directory()`, `util.folders.get_annotated_filepath(name)`
- `util.folders.get_output_directory()`, `util.folders.get_save_image_path(...)`
- `util.folders.get_full_path(folder, name)`, `util.folders.get_folder_paths(folder)`,
  `util.folders.folder_names_and_paths`
- `util.folders.base_path`, `util.folders.models_dir`, `util.folders.get_user_directory()`

**ComfyUI decides what each of these calls answers.** On the user's own machine it is the user's
folder. A host may instead give a pack folders of its own for one run, or a path the pack cannot
open. So a pack must not assume that a path it is given is the user's real folder, that it can
read what it did not put there, or that a file it wrote stays. A pack that has to run everywhere
takes the user's files as node inputs and returns its results as node outputs, and leaves reading
and saving to ComfyUI's own nodes.

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
- Whether a pack may use the network is the host's decision, made pack by pack. A pack that needs
  the network must say so where it is published, and must fail with a clear message when the
  network is refused.

## Every run is the first

A pack must work on its first run, because a host may start every run from the pack as it was
installed: its folder unchanged, the temp folder empty, and nothing the pack wrote or kept in
memory before.

- **Within one run**, a file a node wrote in the temp folder can be read by another node of the
  same pack. Between packs, pass values through node outputs: two packs need not see the same
  folder.
- **On the next run**, it is probably gone. Code that uses something from an earlier run must work
  when it is missing: use it if it is there, otherwise build it again.

The one thing ComfyUI keeps for a pack is models: those it loads through `eng.models` stay loaded
for as long as ComfyUI decides. A model a pack loads with its own code lives only as long as the
pack's process.
