"""Centralized MIME type initialization.

Call init_mime_types() once at startup to initialize the MIME type database
and register all custom types used across ComfyUI.
"""

import mimetypes

_initialized = False


def init_mime_types():
    """Initialize the MIME type database and register all custom types.

    Safe to call multiple times; only runs once.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    mimetypes.init()

    # Web types (used by server.py for static file serving)
    mimetypes.add_type('application/javascript; charset=utf-8', '.js')
    mimetypes.add_type('image/webp', '.webp')
    mimetypes.add_type('image/svg+xml', '.svg')

    # Model and data file types (used by asset scanning / metadata extraction)
    mimetypes.add_type("application/safetensors", ".safetensors")
    mimetypes.add_type("application/safetensors", ".sft")
    mimetypes.add_type("application/pytorch", ".pt")
    mimetypes.add_type("application/pytorch", ".pth")
    mimetypes.add_type("application/pickle", ".ckpt")
    mimetypes.add_type("application/pickle", ".pkl")
    mimetypes.add_type("application/gguf", ".gguf")
    mimetypes.add_type("application/yaml", ".yaml")
    mimetypes.add_type("application/yaml", ".yml")
