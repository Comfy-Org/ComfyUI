from contextlib import contextmanager, ExitStack
from functools import wraps
import threading
import weakref

import torch


# Shared wrappers retain only synchronization metadata here, never model storage.
_locks = weakref.WeakKeyDictionary()
_locks_guard = threading.Lock()


def _model_lock(model):
    with _locks_guard:
        lock = _locks.get(model)
        if lock is None:
            lock = threading.RLock()
            _locks[model] = lock
        return lock


@contextmanager
def _model_locks(model):
    # Shallow roots and compiled wrappers may retain the same child modules.
    # All operations acquire shared descendants in one order, without owning
    # or copying their weights in the synchronization registry.
    modules = sorted(model.modules(), key=id)
    with ExitStack() as scope:
        for module in modules:
            scope.enter_context(_model_lock(module))
        yield


def vae_operation(operation):
    @wraps(operation)
    def run(self, *args, **kwargs):
        if self.first_stage_model is None:
            return operation(self, *args, **kwargs)
        with _model_locks(self.first_stage_model):
            return operation(self, *args, **kwargs)
    return run


@contextmanager
def decode_policy(model, padding_mode):
    if padding_mode not in ("default", "circular"):
        raise ValueError("VAE padding_mode must be default or circular")
    with _model_locks(model):
        previous = []
        try:
            if padding_mode == "circular":
                for layer in model.modules():
                    if isinstance(layer, torch.nn.Conv2d):
                        previous.append((layer, layer.padding_mode))
                        layer.padding_mode = "circular"
            yield
        finally:
            for layer, padding in reversed(previous):
                layer.padding_mode = padding
