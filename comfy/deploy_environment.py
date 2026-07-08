import functools
import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_DEPLOY_ENV = "local-git"
_ENV_FILENAME = ".comfy_environment"

# Resolve the ComfyUI install directory (the parent of this `comfy/` package).
# We deliberately avoid `folder_paths.base_path` here because that is overridden
# by the `--base-directory` CLI arg to a user-supplied path, whereas the
# `.comfy_environment` marker is written by launchers/installers next to the
# ComfyUI install itself.
_COMFY_INSTALL_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


@functools.cache
def get_deploy_environment() -> str:
    env_file = os.path.join(_COMFY_INSTALL_DIR, _ENV_FILENAME)
    try:
        with open(env_file, encoding="utf-8") as f:
            # Cap the read so a malformed or maliciously crafted file (e.g.
            # a single huge line with no newline) can't blow up memory.
            first_line = f.readline(128).strip()
            value = "".join(c for c in first_line if 32 <= ord(c) < 127)
            if value:
                return value
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.error("Failed to read %s: %s", env_file, e)

    return _DEFAULT_DEPLOY_ENV
