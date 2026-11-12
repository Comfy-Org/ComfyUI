from comfy_api.internal.singleton import ProxiedSingleton
from comfy_api.latest import io


class Guider(ProxiedSingleton):
    async def cfg(
        self,
        model: io.Model.Type,
        positive: io.Conditioning.Type,
        negative: io.Conditioning.Type,
        cfg: float,
    ) -> io.Guider.Type:
        """Return a classifier-free guidance guider, as the CFGGuider node gives."""
        from comfy_extras.nodes_custom_sampler import CFGGuider

        return CFGGuider.execute(
            model=model, positive=positive, negative=negative, cfg=cfg
        ).result[0]

    async def basic(
        self, model: io.Model.Type, conditioning: io.Conditioning.Type
    ) -> io.Guider.Type:
        """Return a guider for one conditioning without classifier-free guidance, as BasicGuider gives.

        A guidance value in the conditioning, such as FluxGuidance sets, still applies.
        """
        from comfy_extras.nodes_custom_sampler import BasicGuider

        return BasicGuider.execute(model=model, conditioning=conditioning).result[0]
