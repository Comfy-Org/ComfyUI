"""Core engine extension: ComfyUI settings a node may read."""

#: Readable settings: name -> how to read it. Nothing else is exposed.
_SETTINGS = {
    "disable_metadata": lambda args: bool(args.disable_metadata),
    "preview_method": lambda args: str(getattr(args.preview_method, "value", args.preview_method)),
    "preview_size": lambda args: int(args.preview_size),
    # --fast: the enabled optimizations' names, sorted (read-only: ComfyUI's own set isn't handed out)
    "fast": lambda args: sorted(str(getattr(f, "value", f)) for f in (args.fast or ())),
}


class EngineOps:
    def get(self, name):
        from comfy.cli_args import args

        reader = _SETTINGS.get(name)
        if reader is None:
            raise ValueError(f"unknown setting {name!r}; readable: {', '.join(sorted(_SETTINGS))}")
        return reader(args)
