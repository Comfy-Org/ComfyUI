import comfy.model_management as model_management


class FakeModel:
    def __init__(self):
        self.detach_count = 0
        self.parent = None
        self.load_device = "cpu"

    def detach(self, unpatch_weights):
        self.detach_count += 1


class FakeFinalizer:
    def __init__(self):
        self.detach_count = 0

    def detach(self):
        self.detach_count += 1


def test_model_unload_is_idempotent():
    model = FakeModel()
    finalizer = FakeFinalizer()
    loaded_model = model_management.LoadedModel(model)
    loaded_model.model_finalizer = finalizer

    assert loaded_model.model_unload() is True
    assert loaded_model.model_unload() is True
    assert model.detach_count == 1
    assert finalizer.detach_count == 1
