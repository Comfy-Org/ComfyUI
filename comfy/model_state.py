import copy
import itertools
import json
import os
import tempfile
import weakref

import safetensors
import torch

import comfy.memory_management
import comfy.model_management
import comfy.model_patcher
import comfy.storage
import comfy.utils

METADATA_KEY = "comfy_model_state"
SAVED_STATES = weakref.WeakValueDictionary()
_STATE_VERSIONS = itertools.count()


def _save_tensors(tensors, path, metadata):
    fd, temporary = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".model-state-")
    os.close(fd)
    try:
        comfy.utils.save_torch_file(tensors, temporary, {METADATA_KEY: metadata})
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_tensors(patcher):
    comfy.model_management.synchronize()
    state_dict = patcher.model.state_dict()
    if patcher.is_dynamic():
        patcher.flush_writes()
        tensors = {name: torch.empty(value.shape, dtype=value.dtype) for name, value in state_dict.items()}
        for name, value in state_dict.items():
            if not comfy.memory_management.read_tensor_file_slice_into(value, tensors[name]):
                raise RuntimeError(f"Could not read model state tensor {name}")
    else:
        tensors = {name: value.to(device="cpu", copy=True) for name, value in state_dict.items()}
    return tensors


class _StateValues:
    # Patcher-owned save records must not own their patcher: a GC cycle can destroy
    # the VBAR before the patcher's final writeback, particularly on Windows.
    def __init__(self, tensors, metadata):
        self.tensors = tensors
        self.metadata = metadata
        self.version = next(_STATE_VERSIONS)
        self.patcher = lambda: None
        self.saved_versions = {}

    def check_valid(self, patcher):
        if self.tensors is None and (patcher is None or patcher.attachments.get("model_state") is not self):
            raise RuntimeError("The model state was released without being written back; rerun the node that produced it")

    def read(self, patcher):
        self.check_valid(patcher)
        return self.tensors if self.tensors is not None else _read_tensors(patcher)

    def flush(self, patcher):
        self.tensors = self.read(patcher)

    def remove_pending(self, path):
        patcher = self.patcher()
        if patcher is not None:
            saves = patcher.attachments.get("model_state_saves", {})
            if saves.get(path) is self:
                saves.pop(path)

    def write_pending(self, patcher):
        saves = patcher.attachments.get("model_state_saves", {})
        paths = {path for path, values in saves.items() if values is self}
        if not paths:
            return
        for path in paths:
            saves.pop(path)  # readback can itself reclaim RAM pins
        try:
            tensors = None
            for path in tuple(paths):
                if ModelState._saved(path)[1] is self:
                    if tensors is None:
                        tensors = self.read(patcher)
                    metadata = self.saved_versions[path][1]
                    _save_tensors(tensors, path, metadata)
                    self.saved_versions[path] = ModelState.file_version(path), metadata
                paths.remove(path)
        except Exception:
            saves.update(dict.fromkeys(paths, self))
            raise


def _save_before_ram_unload(patcher):
    for values in set(patcher.attachments.get("model_state_saves", {}).values()):
        values.write_pending(patcher)


def _save_on_detach(patcher, unpatch_all):
    if unpatch_all:
        _save_before_ram_unload(patcher)


class ModelState:
    """A snapshot of a writable model. Compatible snapshots share one managed model."""

    def __init__(self, tensors=None, metadata=None, _values=None):
        self._values = _values or _StateValues({} if tensors is None else tensors, {} if metadata is None else metadata)
        self.patcher = self._values.patcher()
        self._owner = lambda: None

    @property
    def tensors(self):
        return self._values.tensors

    @property
    def metadata(self):
        return self._values.metadata

    @property
    def owner(self):
        return self._owner()

    @owner.setter
    def owner(self, patcher):
        self._owner = weakref.ref(patcher) if patcher is not None else lambda: None

    @classmethod
    def open(cls, path, create="if_missing"):
        """Open the latest saved snapshot, including an unwritten resident save."""
        path = os.path.abspath(path)
        _, saved = cls._saved(path)
        if create != "always" and saved is not None:
            metadata = saved.saved_versions[path][1]
            state = cls(_values=saved).copy()
            state._values.metadata = json.loads(metadata)
            return state
        if create != "always" and os.path.exists(path):
            with safetensors.safe_open(path, "pt") as file:
                metadata = file.metadata() or {}
                if METADATA_KEY not in metadata:
                    raise ValueError(f"Not a model state file: {path}")
                return cls({name: file.get_tensor(name).clone() for name in file.keys()}, json.loads(metadata[METADATA_KEY]))
        if create == "never":
            raise FileNotFoundError(f"Model state file does not exist: {path}")
        return cls()

    @staticmethod
    def file_version(path):
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            return None
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    @classmethod
    def _saved(cls, path):
        version = cls.file_version(path)
        saved = SAVED_STATES.get(path)
        if saved is not None and version != saved.saved_versions[path][0]:
            saved.remove_pending(path)
            saved.saved_versions.pop(path)
            SAVED_STATES.pop(path)
            saved = None
        return version, saved

    @classmethod
    def fingerprint(cls, path):
        path = os.path.abspath(path)
        version, saved = cls._saved(path)
        return (saved.version, saved.saved_versions[path][1]) if saved is not None else version

    def save(self, path, writethrough=True):
        self._values.check_valid(self.patcher)
        path = os.path.abspath(path)
        metadata = json.dumps(self.metadata)
        if writethrough or self.patcher is None:
            _save_tensors(self._values.read(self.patcher), path, metadata)
        previous = SAVED_STATES.get(path)
        if previous is not None and previous is not self._values:
            previous.remove_pending(path)
            previous.saved_versions.pop(path)
        SAVED_STATES[path] = self._values
        self._values.saved_versions[path] = self.file_version(path), metadata
        if not writethrough and self.patcher is not None:
            self.patcher.attachments.setdefault("model_state_saves", {})[path] = self._values
        else:
            self._values.remove_pending(path)

    def _comfy_cache_tensors(self):
        if self.tensors is None and self.patcher is not None and not self.patcher.is_dynamic():
            return self.patcher.model.state_dict()
        return self.tensors

    def copy(self):
        self.flush()
        state = ModelState(dict(self.tensors), copy.deepcopy(self.metadata))
        state.patcher, state.owner = self.patcher, self.owner
        state._values.version = self._values.version
        state._values.patcher = self._values.patcher
        return state

    def load(self, model, device):
        """Select this snapshot in the owner's managed model and return it ready to run."""
        self._values.check_valid(self.patcher)
        if self._values.saved_versions:
            values = self.copy()._values
            # Generation already holds a reference to this metadata.
            values.metadata = self.metadata
            self._values = values
        patcher = self.patcher
        owner = self.owner
        if owner is not None:
            while owner.parent is not None and owner.model is owner.parent.model:
                owner = owner.parent
            attached = owner.get_attachment("model_state_model")
            if attached is None or attached[0]() is not owner.model:
                attached = [weakref.ref(owner.model), None]
                owner.set_attachments("model_state_model", attached)
            patcher = attached[1]
        if patcher is not self.patcher:
            self.flush()
            self.release()
        previous = None
        if patcher is not None:
            previous = patcher.attachments.get("model_state")
            if previous is not None and previous is not self._values:
                previous.flush(patcher)
            geometry = {k: (v.shape, v.dtype) for k, v in model.state_dict().items()}
            if geometry != {k: (v.shape, v.dtype) for k, v in patcher.model.state_dict().items()} or device != patcher.load_device:
                self.flush()
                self.release()
                comfy.model_management.unload_model_and_clones(patcher, all_devices=True)
                patcher = None
        resident = patcher is not None and previous is not None and previous.version == self._values.version
        if patcher is not None:
            if not resident:
                patcher.unpatch_model(patcher.offload_device)
            model = patcher.model
        self._values.version = next(_STATE_VERSIONS)
        if not resident:
            dynamic = comfy.memory_management.aimdo_enabled and not comfy.model_management.is_device_cpu(device)
            if dynamic:
                tensors, _ = comfy.utils.create_safetensors(model.state_dict())
                for name, value in self.tensors.items():
                    comfy.memory_management.write_tensor_file_slice(tensors[name], value)
                model.load_state_dict(tensors, assign=True)
            else:
                model.to_empty(device="cpu")
                for name, value in model.state_dict().items():
                    if self.tensors:
                        value.copy_(self.tensors[name])
                    else:
                        value.zero_()
            if patcher is None:
                if dynamic:
                    patcher = comfy.model_patcher.ModelPatcherDynamic(model, device, torch.device("cpu"), fast_disk=comfy.storage.anonymous_fast_disk())
                else:
                    patcher = comfy.model_patcher.ModelPatcher(model, device, torch.device("cpu"))
                patcher.add_callback_with_key(comfy.model_patcher.CallbacksMP.ON_UNLOAD_RAM, "model_state", _save_before_ram_unload)
                patcher.add_callback_with_key(comfy.model_patcher.CallbacksMP.ON_DETACH, "model_state", _save_on_detach)
        self.patcher = patcher
        self._values.patcher = weakref.ref(patcher)
        patcher.attachments["model_state"] = self._values
        if owner is not None:
            attached[1] = patcher
        comfy.model_management.load_models_gpu([patcher], force_full_load=not patcher.is_dynamic())
        if patcher.is_dynamic():
            patcher.set_in_use_by_current_prompt(True)
        self._values.tensors = None
        return model

    def flush(self):
        """Materialize an immutable CPU snapshot without unloading the model."""
        self._values.flush(self.patcher)

    def release(self):
        """Discard an unwired result, leaving its shared model available for reuse."""
        if self.patcher is None:
            return
        self._values.write_pending(self.patcher)
        if self.patcher.attachments.get("model_state") is self._values:
            self.patcher.attachments.pop("model_state")
            if self.patcher.is_dynamic():
                self.patcher.discard_writes()
        if self.tensors is None:
            for path in self._values.saved_versions:
                if SAVED_STATES.get(path) is self._values:
                    SAVED_STATES.pop(path)
        self.patcher = None
