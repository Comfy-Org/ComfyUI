# Pack isolation: each custom-node pack in its own process

```sh
COMFY_PACK_ISOLATION=true python main.py
```

Pack isolation runs each custom-node pack in its own **pack process**, as Chrome and Firefox run each
site in its own page process. It solves two problems:

1. **No dependency conflicts.** Each pack gets its own Python environment for its requirements. One
   pack that needs `transformers` 4.40 and another that needs 4.51 both work, and neither can break
   ComfyUI's own packages. A pack's requirements go into its environment, never into ComfyUI's.
2. **A pack can't take ComfyUI down.** A pack that crashes fails only its own nodes: its process exits,
   the prompt shows the pack's error, and ComfyUI stays up. The pack starts again the next time it's
   needed.

It also means:

- **An installed pack costs nothing until it's used.** ComfyUI imports none of them, so it starts as
  fast with hundreds of packs installed as with none.
- **Memory comes back.** A pack that's been idle for a while exits, and its memory is freed.
- **Packs reach the engine only through ComfyUI**, as page processes reach the screen and the network
  only through the browser.

It's isolation for stability and resources, not for trust. A pack process isn't a security sandbox:
ComfyUI and its pack processes trust each other.

## Installed is not running

Installing a pack puts it on disk. It doesn't start a process. A pack process starts only when:

1. **You render a workflow that uses one of the pack's nodes.** ComfyUI starts the pack before it
   validates the prompt (and registers a V1 pack's nodes then).
2. **The frontend calls one of the pack's endpoints,** such as a button in one of its nodes. ComfyUI
   starts the pack and forwards the request to it.

Until then, the pack has no process. How ComfyUI knows what a pack has:

- **A V2 pack is never imported at startup.** ComfyUI reads its manifest,
  [`comfy-pack.json`](../../docs/pack-manifest.md), and registers every node from the schemas in it.
  The node library lists them, and the frontend loads the pack's web files from disk, before the pack
  has ever run.
- **A V1 pack has no manifest,** so ComfyUI imports it once per version, in a short-lived process on
  the pack's own environment, and caches what it found (`user/pack_isolation_index/`). Its nodes are
  registered when the pack first runs.

## Before a pack runs

For a V2 pack, its manifest says what has to be in place, and ComfyUI makes sure it is before the
pack's process starts:

- **Weights:** every model file the manifest declares is in its model folder and matches its sha256.
  A missing one is downloaded from where the manifest says.
- **Requirements:** its environment exists (below).
- **Dependencies:** the packs it depends on are installed, in the versions it names. If one isn't,
  the pack doesn't run, and its nodes fail with what's missing.

`scripts/pack.sh install <pack>` does all of this ahead of time, so the first render doesn't wait.

## Each pack's environment

A pack with requirements (a V2 pack's manifest `requirements`, or a `requirements.txt`) gets its own
Python environment in `user/pack_envs/`, made once and again only when its requirements change: for
a V2 pack when it's installed or first runs, for a V1 pack when ComfyUI first reads it.

- **The pack's packages come first,** so its versions win for that pack alone.
- **ComfyUI's packages are visible underneath,** so torch and the other large packages aren't
  installed again for every pack.
- **`torch`, `torchvision` and `torchaudio` stay at ComfyUI's versions,** because tensors pass between
  ComfyUI and the pack. A pack that needs a different torch fails to install, with the reason, and
  only that pack is unavailable.

A pack with no requirements runs on ComfyUI's own Python; it has nothing to conflict with. The
environment's install log is next to it, `user/pack_envs/<pack>-<key>.log`.

## A pack process's life

| Browser page process | Pack process |
|---|---|
| Starts when a tab opens a site | Starts when a render uses the pack, or the frontend calls its endpoint |
| Discarded after the tab is in the background a while | Exits after `COMFY_PACK_IDLE_MINUTES` (default 15) without a render or endpoint call |
| Never discarded while it's working | Never stopped while it's running a node or answering an endpoint |
| A crash shows "Aw, snap" in that tab; the browser stays up | A crash fails that pack's nodes; ComfyUI stays up |
| Exits when the browser exits, or crashes | Exits when ComfyUI exits, or crashes: within seconds of its connection to ComfyUI closing |

Each render or endpoint call resets the pack's idle timer. ComfyUI's own nodes run in ComfyUI's
process as usual.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `COMFY_PACK_ISOLATION` | off | `true` turns pack isolation on. |
| `COMFY_PACKS_DIR` | ComfyUI's `custom_nodes` folders | The folders to load packs from instead, separated by `:` (`;` on Windows). |
| `COMFY_PACK_IDLE_MINUTES` | `15` | Stop a pack process after this many idle minutes. `0`: never. |
| `COMFY_PACK_ENVS` | on | `off`: every pack runs on ComfyUI's Python and shares its packages. |
| `COMFY_PACK_ENV_DIR` | `user/pack_envs` | Where pack environments are kept. |
| `COMFY_PACK_PIP_ARGS` | none | Extra `pip install` arguments for pack environments, such as an index URL. |
| `COMFY_PACK_LAUNCHER` | built-in | An executable that starts and stops pack processes (see below). |
| `COMFY_PACK_RUN_DIR` | `/tmp/comfy-packs-$UID` | Logs and run files of pack processes. |

The earlier names (`OOP_CUSTOM_NODES`, `OOP_PACKS_DIR`, `OOP_PACK_TTL_MINUTES`, `OOP_PACK_LAUNCHER`)
still work.

## Pack folders

A folder in `COMFY_PACKS_DIR` (or `custom_nodes`) is either:

- **flat:** `<folder>/<pack>/`, as `custom_nodes` is; or
- **a pack corpus:** `packs.json` and `<slug>/<key>/<Pack>-HEAD/`, loaded where it is.

A pack that has a `v2/` package (with an `__init__.py`) is loaded from `v2/`, its conversion.

## Which nodes work

- **Engine-API nodes** (`ENGINE_API = True`, written against `comfy_api.index`) work fully, the same as
  in ComfyUI's process.
- **Older nodes** work when they use only tensors and plain data. Engine objects (MODEL, CLIP, VAE)
  reach a pack process as handles, not as the objects themselves, so a node that reaches into them
  fails.

## Endpoints


- **V1 endpoints,** registered on `PromptServer.instance`, are found by the pack's description and
  forwarded as they are, including ComfyUI's `/api` copies of them.
- **V2 endpoints,** declared as `[[tool.comfy.routes]]` in the pack's `pyproject.toml` (so listed in
  its `comfy-pack.json`), live under `/ext/<namespace>/`. The namespace comes from the manifest's
  `name`: a readable slug plus 8 hex characters of a hash. It's the same wherever the pack is installed
  and across versions, and a pack can't choose another pack's. Requests are capped at 1 MB and 64
  query parameters, and responses are JSON or plain text. The V2 converter turns routes a V1 pack
  registered on `PromptServer.instance` into these, and rewrites the URLs in the pack's frontend.

## Running a pack by hand

```sh
scripts/pack.sh start <pack dir> <port>     # start a pack process on 127.0.0.1:<port>
scripts/pack.sh status                      # what's running
scripts/pack.sh describe <pack dir>         # its nodes, endpoints and web directory, as JSON
scripts/pack.sh stop <port>
```

A pack started by hand has no owner, so it runs until you stop it. `PYTHON` selects the interpreter.

## Running pack processes somewhere else

ComfyUI starts and stops pack processes through a launcher. The built-in one runs each pack as a local Python process
on its own environment. To run packs in containers, a sandbox or on another machine, set
`COMFY_PACK_LAUNCHER` to any executable that keeps this contract. ComfyUI needs no changes.

```
<launcher> start <pack dir> <port> <token file>
    When it returns, a pack process for <pack dir> accepts connections on 127.0.0.1:<port> and
    admits only the token in <token file>. Inside the container or sandbox, run:
    python -m comfy_api._impl.isolation.server _serve --pack <pack dir> --port <port> \
        --token-file <token file> --orphan-grace "$COMFY_PACK_ORPHAN_GRACE"

<launcher> stop <port>
    The pack process on <port> is gone.
```

ComfyUI also sets `COMFY_PACK_FOLDERS`, its folder layout as JSON (output, input, temp, user and model
folders), which the pack process applies so `util.folders` gives ComfyUI's paths; pass it on, and mount
those folders at the same paths if the pack runs in a container.

ComfyUI sets `COMFY_PACK_ORPHAN_GRACE` (5 seconds) when it calls the launcher. Passing it to `_serve`
is what makes the pack process exit when ComfyUI goes away, wherever it runs. ComfyUI gives `start` up
to five minutes to return, then up to five more for the port to accept connections.

## Limits

- **Engine extensions** (`v2/engine/*.py`) run next to the engine, so they must be on the same machine
  as ComfyUI.
- **One version of ComfyUI's torch** for every pack (see above).
- **Installers must not put a pack's requirements into ComfyUI's environment.** A V2 pack's requirements
  belong to its own environment only. ComfyUI-Manager today still installs a pack's requirements into
  ComfyUI's environment, which is where conflicts come from; under pack isolation, install with
  `scripts/pack.sh install` instead.
- **Permissions** in a manifest are enforced by a launcher that runs packs in a sandbox. The built-in
  launcher records them but runs packs as ordinary processes.
- **The channel uses pickle.** Keep pack processes on `127.0.0.1` with the per-start token. Across
  machines, the launcher must carry the port over a channel you trust.
- **An "accept any input" V1 node** loses that behaviour across the process boundary.
