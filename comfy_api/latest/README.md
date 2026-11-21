# `comfy_api.latest` is deprecated

`comfy_api.latest` is deprecated and will be removed at a future date.

Import `comfy_api.v3` instead. It is the same API, so only the import line changes:

```python
from comfy_api.v3 import ComfyExtension, io, ui
```

New node packs that import `comfy_api.latest` will be rejected when submitted to the
Node Registry in the near future. Packs that already import it keep working until it
is removed.

`latest` is the V3 API and no longer changes: packs import it directly, so a change
here would break them. How versions are numbered is explained in
`comfy_api/version_list.py`.
