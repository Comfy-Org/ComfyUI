# Output Routing Policy

ComfyUI loads `user/output-policy.json` on startup after applying `--user-directory`.
Use `--output-policy PATH` to load an explicit policy file instead. Missing user
policies preserve the legacy prefix-based folder layout.

Policies can set `output_directory` and choose relative subfolders beneath it.
An absolute policy path is used as written. A relative policy path is resolved
under ComfyUI's already configured output directory; with the normal standalone
defaults, `"output_directory": "_Output"` resolves to
`<ComfyUI base>\\output\\_Output`. Relative policy paths cannot contain `..` or a
drive qualifier. An explicit `--output-directory` takes precedence over the
policy entirely. The policy does not change ComfyUI's temporary root; previews
use the existing `--temp-directory` setting or ComfyUI default.
Each profile has a required `folder_template` and an optional `filename_template`.
Folder templates cannot use an absolute path or `..`.

The supported placeholders are `{date:<strftime-format>}`, `{width}`, `{height}`,
`{prefix_dir}`, and `{prefix_stem}`. A placeholder without a value
for a save fails that save clearly.

`filename_template` uses the same placeholders but produces a filename **stem**,
not a path or complete filename. It must not contain a drive, slash, backslash,
extension, or counter placeholder. Leave it out to retain the original filename
stem. ComfyUI's saver nodes retain ownership of their counter and extension.
For example, a stem of `ComfyUI_1024x1024` is saved by the standard PNG saver as
`ComfyUI_1024x1024_00001_.png`.

For a filename template, use a separator-free date format such as
`{date:%Y-%m-%d}`. `{prefix_dir}` is normally a folder value, not a filename
value: if it contains a separator, the save is rejected. The usual filename
template is `{prefix_stem}_{width}x{height}`.

## Field reference

| Placeholder | Meaning | Source and limitations | Operator recommendation |
| --- | --- | --- | --- |
| `{date:%Y-%m-%d}` | The save date, rendered as a folder name such as `2026-09-06` | Taken when the save path is resolved. The format after `date:` is a restricted `strftime` format. | Good default grouping field for browsing, retention, and daily handoff. |
| `{width}` / `{height}` | The image dimensions provided to the save operation | Some generic saves may provide `0` when dimensions are not known. | Use only when resolution is operationally meaningful to the archive. |
| `{prefix_dir}` | The directory portion of the Save node's existing `filename_prefix` | With `portraits/ComfyUI`, this is `portraits`. It preserves the user’s existing high-level categorization. | Good default grouping field. |
| `{prefix_stem}` | The final portion of the Save node's existing `filename_prefix` | With `portraits/ComfyUI`, this is `ComfyUI`. It normally already appears in the generated filename. | Usually keep it in the filename; use it as a directory only for a specific archival reason. |

The same field definitions and design rationale also appear in
[`OUTPUT-MANAGEMENT-DESIGN-NOTES.md`](OUTPUT-MANAGEMENT-DESIGN-NOTES.md#available-routing-fields).

The sample policy at `output-policy.metadata-example.json` is the full-fields
coverage example. Both its output and temporary profiles use every supported
placeholder: date and prefix directory for routing; prefix stem and dimensions
for the filename. Given a date of 2026-09-06, a 1024x1024 image, and filename
prefix `campaign/final`, the standard PNG saver writes:

```text
F:\ComfyUI\_Output\2026-09-06\campaign\final_1024x1024_00001_.png
```

It is a field-coverage example, not a recommended default layout: resolution in
the filename is optional and should be retained only when it helps operators.
Start it with:

```bat
Start ComfyUI.bat --output-policy "%CD%\output-policy.metadata-example.json"
```

The policy itself supplies the date hierarchy; no date-specific command-line
option is needed.

## PR test procedure

From a normal ComfyUI checkout, run this automated test set from the repository
root before submitting the PR:

```bat
python -m pytest tests-unit/folder_paths_test/output_routing_test.py tests-unit/comfy_test/folder_path_test.py tests-unit/server_test/system_stats_test.py -q
```

From the ComfyUI-Easy-Install directory, use the bundled Python executable. This
is the command for zsh, Git Bash, or another POSIX-style shell on Windows:

```sh
./python_embeded/python.exe -m pytest \
  ComfyUI/tests-unit/folder_paths_test/output_routing_test.py \
  ComfyUI/tests-unit/comfy_test/folder_path_test.py \
  ComfyUI/tests-unit/server_test/system_stats_test.py -q
```

### What the automated command tests

- `output_routing_test.py` is the feature test. It uses temporary directories and
  no model or GPU. It loads the shipped policy, resolves output and temporary
  routes, checks the date/prefix folders, verifies that dimensions are inserted
  into the filename stem, confirms counter continuity, and rejects unsafe or
  unsupported templates.
- `folder_path_test.py` is the regression test for ComfyUI's existing save-path
  behavior. It ensures the routing work has not changed legacy save behavior.
- `system_stats_test.py` is the `/system_stats` response contract test. It checks
  that the server continues to expose the configured output, temporary, input,
  user, base, and current-working directories.

The command does not start the ComfyUI web server, run a workflow, download a
model, or require a GPU. A passing result confirms the policy/save-path contract;
the manual Save Image check below confirms a real saver node writes the expected
file name.

The routing test is the automated integration test at ComfyUI's shared saver
boundary: every built-in saver obtains its directory, filename stem, and counter
from `folder_paths.get_save_image_path()`. It verifies:

- the shipped policy loads and chooses distinct output and temporary profiles;
- date and prefix directory form the expected folders;
- dimensions move into the filename stem;
- an existing rendered filename increments its existing counter sequence;
- omitting `filename_template` preserves the legacy filename stem;
- policy roots respect explicit `--output-directory` precedence; and
- unsafe paths, unsupported placeholders, extensions, and counter placeholders
  are rejected.

The save-path and `/system_stats` tests run alongside it to protect the existing
save behavior and the output-directory server contract.

### Manual Save Image execution check

This short check exercises an actual Save Image node without requiring a
model-specific graph test.

1. Start ComfyUI with the sample policy. From the ComfyUI-Easy-Install directory
   in PowerShell, use the direct command-line form:

   ```powershell
   .\python_embeded\python.exe .\ComfyUI\main.py --windows-standalone-build --output-policy .\ComfyUI\output-policy.metadata-example.json
   ```

   From zsh or Git Bash, use the same command with forward slashes:

   ```sh
   ./python_embeded/python.exe ComfyUI/main.py --windows-standalone-build --output-policy ComfyUI/output-policy.metadata-example.json
   ```

2. Run any workflow containing Save Image. Set its `filename_prefix` to
   `campaign/final` and save a 1024x1024 image.
3. Confirm the first file is created under:

   ```text
   F:\ComfyUI\_Output\YYYY-MM-DD\campaign\final_1024x1024_00001_.png
   ```

4. Run it again with the same prefix and dimensions. Confirm the filename ends
   in `_00002_.png`; the folder does not gain a `1024x1024` level.
5. Trigger a preview and confirm it uses the temporary profile beneath ComfyUI's
   configured temporary root (not the policy's output root):

   ```text
   <temp-directory>\temp\YYYY-MM-DD\campaign
   ```

The test passes when the actual files, the UI/history relative `subfolder`, and
the incremented counter all agree with these paths. Do not pass
`--output-directory` during this check, because it intentionally overrides the
policy's sample root.
