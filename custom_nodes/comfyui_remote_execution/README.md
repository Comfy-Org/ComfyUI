# Remote execution

Runs the local ComfyUI editor against a remote ComfyUI server: Run, queue, progress, previews, node outputs, model dropdowns, the Issues panel, uploads, system info and logs all use the remote, while workflows, settings and extensions stay local.

- `__init__.py` / `web/`: frontend extension served by local ComfyUI. It starts the gateway; it has no nodes and makes no outbound requests.
- `gateway/`: separate loopback process (127.0.0.1:8190) that owns the remote connection and proxies the editor's remote routes.

## Use

Start ComfyUI as usual; the gateway starts with it (log: `data/gateway.log`). Open the **Remote** sidebar tab, enter the remote URL, **Connect**, and set **Execution** to Remote.

- Remote mode is remembered across page reloads (setting: *Remote execution → Stay in remote mode after reloading the page*). The gateway remembers the connected server and remote mode across restarts, and retries while the server is unreachable.
- While remote mode is on and the server is unavailable, remote requests fail; nothing runs locally instead.

## Model downloads on the remote

In remote mode the Issues panel's **Download** button for a missing model saves the file into the remote's model folder instead of this PC. This needs `remote_companion.py` installed on the remote: copy it into the remote ComfyUI's `custom_nodes` and restart it. On Colab, paste the file into a cell whose first line is `%%writefile /content/ComfyUI/custom_nodes/remote_companion.py` and run it. It downloads only on request, only `https://huggingface.co/`, `https://civitai.com/` or `https://civitai.red/` URLs, only model file types, and only into model folders. The Remote tab's **Model files** view shows download progress with Cancel, lists the remote's model files by folder and size with its free disk space, and deletes files (with a confirm step) to free space. Only files the server lists as models can be deleted. After a download or delete, node definitions are reloaded so dropdowns match.

Gated or login-only models need keys: enter a Hugging Face token and/or Civitai API key under **Access keys** in that tab. The gateway keeps them in `data/keys.json` on this PC (the page only sees whether one is set) and adds the matching key to a download request only for that site; the remote uses it as a Bearer header for that request and does not store it. Alternatively set `HF_TOKEN` / `CIVITAI_API_TOKEN` in the remote's environment (e.g. from Colab secrets).

For missing models, the tab's **Missing in this workflow** list offers a Download when ComfyUI's bundled templates name a link for the file (the link is also written into the node so the Issues panel shows its Download button), **Use instead** for another precision/format variant the remote already has, or a URL field.

## Behavior

- Routed to the remote: `/prompt`, `/queue`, `/history`, `/interrupt`, `/jobs`, `/free`, `/view`, `/view_metadata`, `/upload`, `/object_info`, `/models`, `/experiment/models`, `/embeddings`, `/system_stats`, the internal `/logs`, `/folder_paths` and `/files` routes, and the event socket. Node definitions are reloaded on every mode switch, so dropdowns and the missing model/media checks reflect the active server.
- `/prompt`: local comfy.org credentials are removed from `extra_data`, and a `prompt_id` is added. Input file names refer to remote files; when the remote lacks one that exists locally, the local file is uploaded as `remote_execution/<sha256 prefix>-<name>` (cached per server and content). Paths outside the input/output/temp folders are rejected.
- A `/prompt` without a ComfyUI answer (connection loss, tunnel error page) is never resent; the editor shows the prompt id to look for in the queue.
- Access: loopback only, local UI origin only, and a per-start token (header or query) or the SameSite=Strict session cookie the gateway sets for the native socket and image loads. The extension reads the token from the local-only `/remote_execution/session` route.

## Verified pair

| | Local | Remote |
|---|---|---|
| ComfyUI | 0.42.0 | 0.42.0 |
| Frontend | 1.57.1 | 1.57.1 |
| Platform | Windows, CPU | Linux, NVIDIA L4, CUDA 13.0 |

Frontend hooks: `api.apiURL` and `api.internalURL` (route remote paths to the gateway), `api.api_host` (the native socket; reconnects on close), `Comfy.RefreshNodeDefinitions`, `app.extensionManager.registerSidebarTab`.

## Tests

```
python -m unittest tests.test_gateway
```

Not yet implemented: authentication for the remote itself, and environment manifests/synchronization (installing custom nodes or models on the remote).
