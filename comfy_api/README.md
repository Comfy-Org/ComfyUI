# Custom-node resource SDK

Import the public facade with `from comfy_api.latest import sdk` and the node
declarations with `from comfy_api.latest import io`. The `latest` API is an
evolving contract. Private modules and the provider registry's implementation
classes are not an extension compatibility contract.

Nodes opt into resource handles with `SDK_REFS = True`. Their image, model and
other resource inputs become typed refs during execution; output refs resolve
back to native values for downstream nodes. Ordinary ComfyUI executes these
nodes in its own process. Installing this SDK alone does not isolate custom
Python code or grant it fewer operating-system permissions.

Use `sdk.ctx()` during node execution for supported host services. Retain
plain data between runs; refs belong to their execution runtime. Await SDK
operations and handle exceptions. A declared method is not proof that the
active provider implements it.

## Availability in ordinary ComfyUI

Resource operations, progress, output, graph access, model services and several
other domains have native implementations. Individual operations can still
require a configured provider, dependency, device, model or user interaction.
`OpNotSupported` reports an unavailable resource operation.

The default context currently raises `NotImplementedError` for every operation
in these domains:

- `storage`
- `interact`
- `sample`
- `unsample`
- `serve`
- `secrets`
- `net`

The generic `integrations.web` domain is also unavailable by default. An
optional host may supply these services; check that host's documented support.
Python `ctx.storage` does not currently provide persistent storage in ordinary
ComfyUI. Frontend `comfy.storage` is a separate API, and shared JavaScript/Python
storage is not established by either declaration.

When an optional service is essential, report its absence clearly to the user.
Do not silently claim success or fall back to private implementation imports.
