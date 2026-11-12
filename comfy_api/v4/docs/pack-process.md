# Running a pack in its own process

ComfyUI can run a V4 pack in a process of its own. The pack's nodes appear and connect as before,
and a workflow gives the same results. If the pack crashes, ComfyUI keeps running.

A pack does not need to be written for this, but it must keep to the rules under
[What a pack must not rely on](#what-a-pack-must-not-rely-on).

## Turning it on

Name the packs when you start ComfyUI, by their folder names in `custom_nodes`; a pack that is one
file is named by its file name, with or without its `.py`:

```sh
python main.py --pack-process my_pack another_pack
```

Only V4 packs can be named. A name that is not a V4 pack is refused with a message and ComfyUI
starts; a name that matches no pack is reported in the log. Packs that are not named run inside
ComfyUI as usual, and both kinds work in one workflow.

On Windows the option is not supported in this release: ComfyUI says so in the log and does not
load the named packs. `--pack-launcher` has no effect there.

## What happens

- ComfyUI starts one process per named pack, and that process imports the pack. ComfyUI itself
  imports none of the pack's code except its `prestartup_script.py`.
- The pack's `engine` calls, and its `util.folders` calls about model folders, are made in ComfyUI,
  on the thread and in the node where they would run otherwise. The other `util` functions run
  in the pack's process.
- The output, input, temp and user folders `util.folders` gives are those the process was
  started with: ComfyUI's own, unless the launcher chose others.
- Tensors and plain data are copied. A tensor's bytes go straight from the sender's memory into
  the receiver's, so while both sides hold it, it exists twice.
- The pack's routes are answered through ComfyUI at the same addresses.
- What the pack prints and logs appears where ComfyUI's own output does.
- A pack's `prestartup_script.py` still runs inside ComfyUI.

## What crosses between the pack and ComfyUI

These arrive as a copy of the same type and value: `None`, `bool`, `int`, `float`, `str`, `bytes`,
`bytearray`; a `list`, `tuple`, `dict`, `set`, `frozenset` or `OrderedDict` of them; tensors; numpy
arrays and numpy scalars; `torch.Size`, `torch.device`, `torch.dtype`; `pathlib` paths; and the
values `io` and `ui` define. Numpy arrays cross when they hold numbers; a member of an enum the
pack defines over `str` or `int` arrives as its plain value. A video or 3D value that holds a
file arrives holding the file's bytes, also when it held a path: the whole file is read and
copied.

Anything else stays in the process that made it: an object of the pack's own class, a subclass of
a type above, a `datetime`, a `complex`, a sparse or quantized tensor. The other side gets a
handle that names the type. A handle can be passed on, returned and printed with `print` or
`str`, and a function can be called through it; any other use, formatting it in an f-string
included, raises an error. An object of the pack's own class can be used only by that pack's nodes: a
node of another pack that runs in its own process fails with an error that names the object's
pack. A tensor that is a view or was transposed arrives contiguous, with the same values. A tensor
that arrives is the receiver's own: a change made in place on one side is not seen on the other.

## What a pack must not rely on

- **A `MODEL`, `CLIP`, `VAE` or other ComfyUI object is a handle, valid while the node that
  received it runs.** Pass it to `engine` calls or return it. Do not read its attributes, and do not
  keep it for later: not in a global, not inside an object of your own that another node receives,
  not in a hook for use after the node has returned. A hook is handed what it needs each time it
  is called.
- **Call `engine` only from the node's own code**, the functions it calls and the hooks ComfyUI calls
  back. A thread the pack starts, also with `asyncio.to_thread`, cannot call `engine`.
- **Reach ComfyUI only through `comfy_api.v4`.** A pack that imports ComfyUI's own modules, such
  as `folder_paths`, gets its own process's copy: what it changes there, ComfyUI does not see.
- **A model folder the pack adds must be inside ComfyUI's models folder or the pack's own
  folder**, or be one ComfyUI lists already. `util.folders.add_model_folder_path` raises
  `ValueError` for any other directory, and for a directory in the pack's folder if a link in
  that folder leads out of it. ComfyUI looks when the folder is added: it follows a link made
  in the folder afterwards when it lists the folder's files.
- **`engine.caching` is not available.** A cache provider has to be registered inside ComfyUI:
  `engine.caching.register_provider` raises `TypeError`.
- **A node's `ui` value must be one JSON can encode**: text, numbers, booleans, `None`, and
  lists and dicts of them. A node that returns any other fails with `TypeError`.
- **While a node waits for an `engine` call, another node of the same pack can run**, for example in
  the middle of a sampling step that calls the pack's hook. Do not rely on one node of a pack
  finishing before another starts.

## Cost

Rough figures from one laptop, on the CPU, to show the scale. On a GPU each crossing also copies
between the device and memory; that was not measured.

- Starting a pack's process: 2 to 3 seconds per pack when ComfyUI starts, and a few hundred
  megabytes of memory.
- One `engine` call: about a tenth of a millisecond.
- A tensor: about 0.1 seconds per gibibyte, each way.
- A hook ComfyUI calls once for each evaluation of the model, which some samplers do more than
  once a step (`set_cfg`, `set_pre_cfg`, `set_post_cfg`, `set_denoise_mask`,
  `set_unet_wrapper`, a sampler function): under a millisecond per call with a 1 MiB tensor
  each way.
- A hook ComfyUI calls in every block (the attention and block patches,
  `set_attention_override`) is called hundreds of times for one image, and each call copies its
  tensors both ways. Run such a pack inside ComfyUI.

## When something goes wrong

- **The pack's process crashes.** The node that was running fails with an error that names the
  pack, its exit code and the node. ComfyUI starts the process again the next time one of the
  pack's nodes or routes is used. After three crashes in ten minutes ComfyUI disables the pack
  until ComfyUI is restarted: its process is not started again, and the model folders it added
  are taken out. Its nodes stay listed, so a workflow that uses one still opens and fails on
  that node with an error that says the pack is disabled and why.
- **A node of the pack raises.** ComfyUI reports the error as it would in process, except that
  the report's `current_inputs` is empty: in ComfyUI's process every node of the pack runs as an
  async node, and ComfyUI reports no inputs for an async node that fails after it awaited.
- **The user cancels.** The pack is told at once. A node that checks
  (`engine.execution.check_interrupted()`, `engine.progress.update`) stops there. If the pack has not
  stopped after 5 seconds, ComfyUI ends its process; the next prompt starts it again.
- **The pack's process stops taking what ComfyUI sends it**, because it is stuck or has been
  suspended. After 10 seconds in which it takes nothing, ComfyUI ends it, and what was waiting
  for it fails as after a crash. Until then the node or route that is being sent waits;
  ComfyUI's server goes on answering other requests, except that a request for the pack's node
  definitions waits up to a second for the pack before it answers with the last ones it had.
- **A route is requested while the process is down.** ComfyUI starts the process and then
  answers, so the request takes as long as a start. If the process stops while it answers a
  request, that request is answered 503, and so is every request once ComfyUI has disabled the
  pack.
- **A node calls `sys.exit()`.** That node fails; the process keeps running.
- **The pack started a program**, such as `ffmpeg`. Whenever ComfyUI ends the pack's process,
  after a cancel, a crash or at exit, it ends that program too. There are two exceptions. A
  program that put itself in a session of its own is left running if the pack's process then
  stops by itself, or if ComfyUI is killed while the pack's process cannot run. A program that
  detached itself from the pack's process altogether, as a daemon does, is not ended at all:
  ComfyUI cannot tell that the pack started it.

## Starting the process yourself

ComfyUI can be given a function that starts pack processes in place of its own launcher:

```sh
python main.py --pack-process my_pack --pack-launcher my_module:launch
```

ComfyUI calls `launch(start)` each time it starts a pack's process. `start` says which pack to
start and which folders it gets; `launch` returns the process and ComfyUI's end of the socket
that connects them. ComfyUI ends the process, and asks whether it has ended, only through what
`launch` returned. The module `comfy_api/v4/_pack_launch.py` states the whole agreement.

## What it is not

A pack's process is not a security boundary: with ComfyUI's own launcher the pack runs with the
same user, files and network as ComfyUI. And whatever starts the process, part of a pack still
runs outside it:

- its `prestartup_script.py` runs inside ComfyUI when ComfyUI starts;
- the JavaScript in its web folder runs in the user's browser, as every pack's does;
- a node that declares the hidden inputs for the user's Comfy.org account receives them, as it
  does inside ComfyUI.
