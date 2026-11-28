from typing import Literal

from comfy.cli_args import args
from comfy_api.internal.singleton import ProxiedSingleton


class Settings(ProxiedSingleton):
    async def get(self, name: Literal["disable_metadata", "multi_user"]) -> bool:
        """Report how ComfyUI was started. Raises `ValueError` for any other name.

        `disable_metadata`: files a node saves must not carry the prompt or workflow.
        `multi_user`: each user has their own user data (settings, saved workflows, user files);
        input, output and model folders are shared.
        """
        if name == "disable_metadata":
            return args.disable_metadata
        if name == "multi_user":
            return args.multi_user
        raise ValueError(f"Unknown setting {name!r}.")
