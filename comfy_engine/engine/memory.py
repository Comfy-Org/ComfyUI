"""Core engine extension: model memory."""


class EngineOps:
    def empty_cache(self):
        import comfy.model_management as mm

        mm.soft_empty_cache()
        return None

    def load_models(self, models):
        """Load models onto the GPU now, as samplers do before they run."""
        import comfy.model_management as mm

        mm.load_models_gpu([getattr(model, "patcher", model) for model in models])
        return None

    def unload_all(self):
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
        return None

    def free_bytes(self, device=None):
        import torch
        import comfy.model_management as mm

        return int(mm.get_free_memory(torch.device(device) if device else mm.get_torch_device()))
