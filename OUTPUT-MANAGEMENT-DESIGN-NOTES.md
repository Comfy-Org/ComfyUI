# ComfyUI Output Management: Design Notes

## Why a general facility

Date-based folders are a useful policy, but should not become a one-off CLI feature
inside the common save-path function. Output routing should be a first-class facility
that applies consistently to images, audio, video, 3D assets, and third-party saver
nodes.

The existing `folder_paths.get_save_image_path()` function is the natural integration
point: it is already shared by the built-in save paths. Its current legacy behavior
must remain the default to avoid moving existing users' output files.

## Core model

Introduce an output router/resolver which receives:

```python
request = OutputRequest(
    kind="output",              # output or temp
    filename_prefix="portraits/ComfyUI",
    extension=".png",
    width=1024,
    height=1024,
)

context = OutputContext.current()
route = output_router.resolve(request, context)
```

It returns a safe output directory, a filename stem, and the public relative
subfolder to return to the UI/history API. Existing `get_save_image_path()` can be
kept as a compatibility wrapper while saver implementations migrate.

Directory routing and filename formatting should be separate. A router must also own
or coordinate filename allocation so concurrent workers cannot select the same
counter by listing a folder simultaneously.

## Execution context

ComfyUI tracks prompt and node identifiers internally for queueing and execution.
They are deliberately not output-routing fields: they identify implementation
details, not a human-operational category. The router does not consume internal
execution counters or editable workflow inputs. Save time is taken when a path is
resolved. A future prompt-level context may expose explicitly provided, validated,
operator-meaningful metadata and a queued timestamp.

Neither a project nor a workflow label exists as a ComfyUI execution value. Do not
derive either from raw workflow metadata. If a concrete operator need warrants
them, they must come from a dedicated Output Context control or validated API
metadata, and remain optional.

## Current implementation and field semantics

The current implementation is an opt-in policy-file foundation, not the complete
router described in these notes. It resolves a relative destination subfolder and
an optional filename stem through the existing `folder_paths.get_save_image_path()`
compatibility path. It does not yet create `OutputRequest`, `OutputContext`, or
resolved-route objects, select a profile per workflow/API request, or reserve names
atomically.

The policy may set an absolute host-owned `output_directory` or a relative
directory below ComfyUI's configured output root. An explicit `--output-directory`
has higher precedence. The selected profile contributes a relative subfolder below
the resolved root and may contribute a filename stem. A policy is loaded at startup
from `user/output-policy.json` or from the explicit `--output-policy PATH` option.

### Available routing fields

| Field | Current source | Availability and trust | Operator value | Recommendation |
| --- | --- | --- | --- | --- |
| `date` | Save-time clock | Always available; evaluated when each save resolves | Useful for retention, browsing, and daily handoff | Good default grouping field |
| `width`, `height` | Save request | Available when the saver supplies dimensions; some generic saves may pass `0` | Helps distinguish variants and investigate sizing mistakes | Use only for archives where resolution is operationally meaningful |
| `prefix_dir` | Directory part of the existing saver `filename_prefix` | Workflow/saver input, containment-checked after rendering | Preserves the user's existing high-level categorization | Good default field |
| `prefix_stem` | Final component of `filename_prefix` | Workflow/saver input, containment-checked after rendering | Identifies the saver prefix but normally duplicates the filename stem | Usually keep in the filename; use as a directory only for a specific archival reason |

For example, a save prefix of `portraits/ComfyUI` produces
`prefix_dir = portraits` and `prefix_stem = ComfyUI`.

The practical default is therefore a small, human-oriented hierarchy such as
`{date:%Y-%m-%d}/{prefix_dir}`. Keep `prefix_stem` out of the default profile.
When dimensions are operationally useful, place them in the filename stem, for
example `{prefix_stem}_{width}x{height}`, rather than adding a resolution folder.
The all-fields sample policy exists to exercise and test field resolution; it is
not the recommended production layout.

### Filename templates and counter preservation

`folder_template` is required and renders a relative directory. An optional
`filename_template` renders only a filename stem; it cannot contain a drive,
path separator, or counter placeholder. Both templates support the same current
fields. Do not add an extension: saver nodes retain extension ownership.

Filename templates should use separator-free values, for example
`{prefix_stem}_{width}x{height}` or `{date:%Y-%m-%d}_{prefix_stem}`.
`prefix_dir` is normally a folder value; if it renders a separator in a filename
template, the save is rejected.

The router does not allocate or format counters. It returns the rendered stem to
the saver, which keeps its existing counter and extension behavior. For the
standard PNG saver, a policy stem of `ComfyUI_1024x1024` becomes
`ComfyUI_1024x1024_00001_.png`. A policy omitting `filename_template` keeps the
existing filename stem unchanged. The current counter scan uses the rendered stem,
so a dimension-bearing stem continues its own counter sequence.

### Current safety and failure behavior

The policy parser rejects unknown placeholders, unsupported date directives,
absolute rendered template paths, parent-directory segments, null bytes, and
invalid profile references. A filename template is also rejected if it is empty,
contains a path separator, drive, Windows-invalid filename character, or trailing
period/space. `{counter}` is not a supported placeholder. A placeholder required
by a selected template but not available for a particular save raises a clear save
error.

The rendered result is containment-checked under the selected output or temporary
root. This is not yet the complete metadata-segment sanitization model proposed
below: current metadata may contribute nested relative segments as long as the
final result remains contained. Project labels and workflow labels do not yet have
a dedicated validated metadata interface.

### Current non-goals and known limits

- Policies select defaults for the output and temporary roots only. Workflows and
  API callers cannot yet select an allowed profile.
- There is no `Output Context` node and no validated `project` or
  `workflow_label` field.
- The existing counter allocator still lists a folder before writing. Concurrent
  workers can therefore choose the same next counter; atomic reservation remains
  future work.
- Date routing is evaluated at save time. Queue-time routing is not implemented.

## Policy and templates

The following is the current v1 policy model. It accepts `output_directory`,
`defaults`, and profiles containing `folder_template` plus an optional
`filename_template`. It does not accept `project` or `workflow_label`.

A startup policy file is a good host-level configuration mechanism, but not the
whole feature. It should define allowed profiles and defaults. Workflows or API calls
may choose an allowed profile; they must not gain arbitrary filesystem-path access.

Suggested precedence:

1. Built-in default: legacy output behavior.
2. Host policy file, e.g. `user/output-policy.json`, selected with
   `--output-policy <path>`.
3. Optional workflow/API selection of an allowed profile.

Example policy:

```json
{
  "version": 1,
  "defaults": {
    "output": "daily",
    "temp": "daily"
  },
  "profiles": {
    "daily": {
      "folder_template": "{date:%Y-%m-%d}/{prefix_dir}",
      "filename_template": "{prefix_stem}_{width}x{height}"
    }
  }
}
```

Date-based output is then just a profile:

```text
{date:%Y-%m-%d}/{prefix_dir}
```

The configured output root remains controlled by ComfyUI. A host policy may set
its `output_directory`, while an explicit `--output-directory` takes precedence;
profiles choose a relative path below the resolved root.

## Supported fields

Use a deliberately restricted template grammar rather than arbitrary Python
formatting. Useful fields include:

| Field | Source |
| --- | --- |
| `date` | Save time, explicitly selected by policy |
| `width`, `height` | Save request |
| `prefix_dir`, `prefix_stem` | Existing filename prefix |

All rendered paths must remain relative to an allowed root; absolute paths and path
traversal (`..`) must be rejected. Metadata fields are sanitized path segments, not
user-supplied paths.

`project` and `workflow_label` do **not** currently come from the workflow JSON,
prompt UUID, node ID, or any other ComfyUI execution value. They
would require a deliberately designed source: either an operator-facing Output
Context control or an API caller supplying validated metadata. Until a concrete
operator need, ownership model, and validation rules are defined, they should not
appear in a default routing policy.

## UI and API integration

- A Save node may expose an optional `output_profile` selector populated only from
  allowed profiles. This makes routing selection portable with a workflow.
- A dedicated `Output Context` node could provide an explicit, operator-entered
  project/workflow label only if those fields prove operationally necessary.
- API callers can supply a validated `output_context` (for example, project or
  archive label) and choose an allowed profile.
- The existing UI/history response contract remains intact: save calls continue to
  return `filename`, `subfolder`, and `type`.

## Filename allocation

The current next-counter approach lists the target folder then writes the file, which
can race under concurrent saves. A robust allocator should:

1. Resolve and validate the folder and stem.
2. Create the folder.
3. Reserve a filename with exclusive creation, retrying the counter on collision.
4. Write the asset to that reservation.
5. Return the resulting relative filename/subfolder to the caller.

## PR test strategy

The automated routing test belongs at `folder_paths.get_save_image_path()`, the
shared boundary used by built-in saver nodes to obtain a directory, filename
stem, and counter. This is a save-path integration test rather than a full
prompt-graph execution test: it tests the policy contract without requiring a
model, sampler, or GPU. It covers the shipped sample policy, output and temporary
profile selection, folder resolution, filename-stem resolution, counter
continuity, legacy-stem fallback, output-root precedence, and unsafe-template
rejection.

An additional manual Save Image execution check confirms that the standard PNG
saver appends its unchanged counter and extension to the routed stem and returns
the expected UI/history subfolder. The runnable commands and expected paths are
maintained in `OUTPUT-ROUTING-POLICY.md` under **PR test procedure**. The targeted
test command also includes the existing save-path regression tests and
`/system_stats` contract test.

## Implementation options

| Option | Scope | Trade-off |
| --- | --- | --- |
| Generic CLI template | One global template such as `--output-folder-template` | Smallest change; no profiles |
| Startup policy file | Named profiles and restricted templates | Best general foundation |
| Workflow route input/node | Workflow selects an allowed profile/label | Portable and explicit |
| API output context | Automation supplies validated project/run metadata | Best for servers/automated jobs |
| Custom node only | External advanced saver node | Fast to ship, but not universal |

## Recommended incremental rollout

1. Add an internal `OutputPathResolver` and retain the current function as a
   compatibility wrapper.
2. Preserve the legacy profile as the default; add one generic global template or
   profile as an opt-in.
3. Add startup policy-file loading and named profiles.
4. Add optional workflow profile selection and validated API metadata.
5. Centralize atomic filename reservation.

This makes date-based organization a simple policy example while providing a
reusable, backward-compatible output-management facility.
