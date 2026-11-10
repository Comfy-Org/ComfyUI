# UI: previews, progress, messages and questions

A node talks to the person using it in four ways: what it shows on the node when it finishes, progress while
it runs, events for the pack's own frontend code, and questions it waits on.

## Showing results on the node

Return a `ui` object with the node's outputs:

```py
from comfy_api.v4 import io, ui

return io.NodeOutput(image, ui=ui.PreviewImage(image, cls=cls))
```

| Class | Shows |
|---|---|
| `ui.PreviewImage(images, cls=cls)` | Images, saved to the temp folder |
| `ui.PreviewMask(mask, cls=cls)` | A mask, as an image |
| `ui.PreviewAudio(audio, cls=cls)` | Audio |
| `ui.PreviewVideo(...)`, `ui.PreviewUI3D(...)` | Video and 3D |
| `ui.PreviewText(text)` | Text |

Pass `cls=cls` so saved files carry the prompt and workflow metadata, as ComfyUI's own previews do.

**Saving rather than previewing** uses the savers, which write to the output folder with ComfyUI's naming and
metadata and return what to show:

```py
saved = ui.ImageSaveHelper.get_save_images_ui(images, filename_prefix="my_pack/result", cls=cls)
return io.NodeOutput(images, ui=saved)
```

`ui.AudioSaveHelper` does the same for audio. A node that saves is an output node: set
`is_output_node=True` on its schema.

All of these write through `eng.output.write`, so they work the same in ComfyUI's process and in a pack's own
process. Outside ComfyUI's process, the files are written when `execute` returns.

## Progress

```py
for i, frame in enumerate(frames):
    ...
    await eng.progress.update(i + 1, len(frames))
```

With a preview, as samplers show theirs:

```py
await eng.progress.update(step, total, preview=current_image)     # [H, W, C] or [B, H, W, C], 0..1
```

Text under the node's title while it runs:

```py
await eng.ui.progress_text(f"Tagging {i} of {n}")
```

## Events for your frontend

A pack with its own frontend code (a `web/` folder) can send it events:

```py
await eng.ui.send("my_pack.status", {"state": "loaded", "count": n})
```

The event goes to the client that queued the prompt, as `PromptServer.send_sync` did. Your frontend
extension listens for it with `api.addEventListener("my_pack.status", ...)`. The data must be plain JSON.

For requests the frontend makes on its own, such as fetching a list to show, declare a route: see
[Packs](packs.md#routes).

## Asking the user mid-run

A node can stop and wait for the user: to pick from a set of images, confirm or edit a prompt, or draw a
mask.

```py
answer = await eng.ui.interact("image-choice", payload)     # payload: what your frontend shows
```

| Kind | For |
|---|---|
| `"image-choice"` | Choosing one or more images |
| `"prompt-await"` | Confirming or editing text |
| `"mask-edit"` | Drawing or editing a mask |

- `payload` is what your frontend shows: plain JSON, up to 64 KB.
- `timeout` is in seconds, 1 to 540.
- `reuse_last=True` answers with this node's last answer of this kind without asking; `remember=True` keeps
  this answer for that.

The pack's frontend extension shows the request and posts the answer. ComfyUI sends the client a
`secure-node-interaction` event with `request_id`, `kind`, `node_id` and `payload`; the extension posts
`{"request_id", "response"}` to `/secure-nodes/interactions/respond`. The answer is checked against the kind
before the node gets it.
