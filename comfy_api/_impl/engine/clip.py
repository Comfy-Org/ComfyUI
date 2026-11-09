"""Core engine extension: the ``eng.clip`` rows (a CLIP is ComfyUI's text encoder)."""


class EngineOps:
    def clone(self, clip):
        return clip.clone()

    def tokenize(self, clip, text, **kwargs):
        return clip.tokenize(text, **kwargs)

    def encode_tokens(self, clip, tokens, **kwargs):
        return clip.encode_from_tokens_scheduled(tokens, **kwargs)

    def encode_from_tokens(self, clip, tokens, return_pooled=False, return_dict=False):
        return clip.encode_from_tokens(tokens, return_pooled=return_pooled, return_dict=return_dict)

    def add_patches(self, clip, patches, strength_patch=1.0, strength_model=1.0):
        patched = clip.clone()
        patched.add_patches({key: tuple(value) for key, value in patches.items()}, strength_patch, strength_model)
        return patched

    def read_weight(self, clip, key):
        # a plain tensor, as documented: an unpatched weight is an nn.Parameter, which would cross as a handle
        return clip.patcher.patch_weight_to_device(key, return_weight=True).detach()
