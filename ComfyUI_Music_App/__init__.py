from comfy_api.latest import ComfyExtension, IO


WEB_DIRECTORY = "./web"


class MusicAppExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return []


async def comfy_entrypoint() -> MusicAppExtension:
    return MusicAppExtension()
