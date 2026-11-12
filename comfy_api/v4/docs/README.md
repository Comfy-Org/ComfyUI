# V4 API documentation

- [Writing a V4 pack](writing-a-pack.md)
- [Files, folders and state](files-and-state.md): what a pack can count on, and what is not supported.
- [Moving a pack from V3 to V4](moving-from-v3.md)
- [Versioning](versioning.md)
- [Reference](reference.md): every call, generated from the package.
- [Running a pack in its own process](pack-process.md)

`reference.md` and `../generated/ComfyAPISyncStub.pyi` are generated from the docstrings and
signatures in `comfy_api/v4`. After changing a call's signature or docstring, run:

```sh
python -m comfy_api.v4._generate
```

`tests-unit/comfy_api_test/v4/generated_test.py` fails while the committed files differ from what
it writes, and `tests-unit/comfy_api_test/v4/docs_test.py` runs every Python example in these
pages.
