"""Core engine extension: the ``eng.vae`` rows (a VAE is ComfyUI's comfy.sd.VAE)."""


class EngineOps:
    def decode(self, vae, samples):
        return vae.decode(samples)

    def decode_tiled(self, vae, samples, **kwargs):
        return vae.decode_tiled(samples, **kwargs)

    def encode(self, vae, pixels):
        return vae.encode(pixels)

    def encode_tiled(self, vae, pixels, **kwargs):
        return vae.encode_tiled(pixels, **kwargs)

    def from_state_dict(self, sd):
        import comfy.sd

        vae = comfy.sd.VAE(sd=sd)
        vae.throw_exception_if_invalid()
        return vae
