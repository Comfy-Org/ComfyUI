# Nodes and User-Facing Behavior

Detailed rules referenced from [AGENTS.md](../../AGENTS.md).

## Node Conventions

- Follow existing node conventions: `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`,
  `CATEGORY`, and registration through the local mapping used by that file.
- Keep node changes backward compatible by default. Add inputs with sensible
  defaults and avoid changing output types unless the request requires it.
- Model implementations should add the minimal number of ComfyUI nodes required
  to run the model. Reuse existing nodes as much as possible; adapting the model
  to work with existing nodes is strongly preferred over creating new nodes.
- Use `io.Autogrow` for a variable number of repeated inputs instead of a fixed
  series of numbered optional sockets. Set its minimum to zero when the model
  has a valid no-item path, and cap it only when the model has a real limit.
- Mark inputs optional when execution has a valid path that does not read them.
  If one optional input is needed only to process another optional input, do not
  force users on the path that supplies neither to connect it.

## Inputs and Outputs

- Conditioning nodes should normally output conditioning only. Do not expose
  input or intermediate images as convenience outputs for downstream sizing or
  routing; use the existing image path or a dedicated image operation instead.
- Nodes should output only values they own. Do not add pass-through outputs for
  workflow convenience unless the node is explicitly an output node. Existing
  models, latents, conditioning, or other inputs should flow directly to the
  next consumer instead of being re-emitted unchanged.
- Nodes should expose only inputs they actually read to produce current
  behavior. Do not add placeholder, pass-through, compatibility, or
  workflow-shaping inputs that are ignored or could flow directly to another
  node.
- Node-level code must not patch model code directly. Any node behavior that
  modifies, wraps, hooks, or changes model behavior must go through the model
  patcher class instead of reaching into model internals.

## Messages and Docs

- Warning and info messages should be short and actionable. Remove noisy or
  misleading messages rather than adding more logging.
- Documentation and README edits should be concise, factual, and tied to the
  changed behavior.
