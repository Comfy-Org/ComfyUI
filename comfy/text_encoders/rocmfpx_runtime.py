"""Windows C API for charlie12345/ROCmFPX 3fca7f4bb.

This revision preserves the C ABI and includes Qwen3VL hidden-state output and
the softmax reduction race fix. Older builds are unsupported. Use DLLs from
one build: GGML's commit is not an ABI version for llama.
"""

import ctypes as ct
import importlib.util
import os
from pathlib import Path

import numpy as np


class ModelParams(ct.Structure):
    _fields_ = [
        ("devices", ct.c_void_p), ("tensor_buft_overrides", ct.c_void_p),
        ("n_gpu_layers", ct.c_int32), ("split_mode", ct.c_int),
        ("main_gpu", ct.c_int32), ("tensor_split", ct.c_void_p),
        ("progress_callback", ct.c_void_p), ("progress_callback_user_data", ct.c_void_p),
        ("kv_overrides", ct.c_void_p),
        ("vocab_only", ct.c_bool), ("use_mmap", ct.c_bool),
        ("use_direct_io", ct.c_bool), ("use_mlock", ct.c_bool),
        ("check_tensors", ct.c_bool), ("use_extra_bufts", ct.c_bool),
        ("no_host", ct.c_bool), ("no_alloc", ct.c_bool),
    ]


class ContextParams(ct.Structure):
    _fields_ = [
        ("n_ctx", ct.c_uint32), ("n_batch", ct.c_uint32),
        ("n_ubatch", ct.c_uint32), ("n_seq_max", ct.c_uint32),
        ("n_rs_seq", ct.c_uint32), ("n_threads", ct.c_int32),
        ("n_threads_batch", ct.c_int32),
        ("ctx_type", ct.c_int), ("rope_scaling_type", ct.c_int),
        ("pooling_type", ct.c_int), ("attention_type", ct.c_int),
        ("flash_attn_type", ct.c_int),
        ("rope_freq_base", ct.c_float), ("rope_freq_scale", ct.c_float),
        ("yarn_ext_factor", ct.c_float), ("yarn_attn_factor", ct.c_float),
        ("yarn_beta_fast", ct.c_float), ("yarn_beta_slow", ct.c_float),
        ("yarn_orig_ctx", ct.c_uint32), ("defrag_thold", ct.c_float),
        ("cb_eval", ct.c_void_p), ("cb_eval_user_data", ct.c_void_p),
        ("type_k", ct.c_int), ("type_v", ct.c_int),
        ("abort_callback", ct.c_void_p), ("abort_callback_data", ct.c_void_p),
        ("embeddings", ct.c_bool), ("offload_kqv", ct.c_bool),
        ("no_perf", ct.c_bool), ("op_offload", ct.c_bool),
        ("swa_full", ct.c_bool), ("kv_unified", ct.c_bool),
        ("samplers", ct.c_void_p), ("n_samplers", ct.c_size_t),
        ("ctx_other", ct.c_void_p),
    ]


class Batch(ct.Structure):
    _fields_ = [
        ("n_tokens", ct.c_int32), ("token", ct.POINTER(ct.c_int32)),
        ("embd", ct.POINTER(ct.c_float)), ("pos", ct.POINTER(ct.c_int32)),
        ("n_seq_id", ct.POINTER(ct.c_int32)),
        ("seq_id", ct.POINTER(ct.POINTER(ct.c_int32))),
        ("logits", ct.POINTER(ct.c_int8)),
    ]


class Tensor(ct.Structure):
    _fields_ = [
        ("type", ct.c_int), ("buffer", ct.c_void_p),
        ("ne", ct.c_int64 * 4), ("nb", ct.c_size_t * 4),
        ("op", ct.c_int), ("op_params", ct.c_int32 * 16),
        ("flags", ct.c_int32), ("src", ct.c_void_p * 10),
        ("view_src", ct.c_void_p), ("view_offs", ct.c_size_t),
        ("data", ct.c_void_p), ("name", ct.c_char * 64),
        ("extra", ct.c_void_p), ("padding", ct.c_char * 8),
    ]


class OverrideValue(ct.Union):
    _fields_ = [("val_i64", ct.c_int64), ("val_f64", ct.c_double),
                ("val_bool", ct.c_bool), ("val_str", ct.c_char * 128)]


class ModelOverride(ct.Structure):
    _anonymous_ = ("value",)
    _fields_ = [("tag", ct.c_int), ("key", ct.c_char * 128), ("value", OverrideValue)]


EvalCallback = ct.CFUNCTYPE(ct.c_bool, ct.POINTER(Tensor), ct.c_bool, ct.c_void_p)


def bind(library, name, result, *arguments):
    function = getattr(library, name)
    function.restype = result
    function.argtypes = list(arguments)
    return function


class FpxSession:
    def __init__(self, model_path, dll_directory, model_kind, gpu_layers=99,
                 kv_type="f32", flash_attn=False, device_index=0):
        if os.name != "nt" or ct.sizeof(ct.c_void_p) != 8:
            raise RuntimeError("This ROCmFPX binding requires Windows x64")
        self.hidden_size, self.layers = {"qwen_image_21": (4096, 36), "minimax_h3": (5120, 50)}[model_kind]
        self.headless = model_kind == "minimax_h3"
        self.kv_type = {"f32": 0, "f16": 1}[kv_type]
        self.flash_attn = int(flash_attn)
        self.model = self.context = None
        self.backend_initialized = False
        self.dll_directories = []
        dll_directory = Path(dll_directory).resolve()
        # gfx1151's default F16 BLAS accumulation changes Qwen conditioning.
        os.environ.pop("GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F", None)
        os.environ["GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F"] = "1"
        try:
            self.dll_directories.append(os.add_dll_directory(str(dll_directory)))
            for package in ("_rocm_sdk_core", "_rocm_sdk_libraries"):
                spec = importlib.util.find_spec(package)
                if spec is not None:
                    directory = Path(next(iter(spec.submodule_search_locations))) / "bin"
                    self.dll_directories.append(os.add_dll_directory(str(directory)))
            self.ggml = ct.CDLL(str(dll_directory / "ggml-base.dll"))
            bind(self.ggml, "ggml_version", ct.c_char_p)
            bind(self.ggml, "ggml_commit", ct.c_char_p)
            if self.ggml.ggml_version() != b"0.11.1" or self.ggml.ggml_commit() != b"3fca7f4bb":
                raise RuntimeError("Use ROCmFPX 3fca7f4bb DLLs with the softmax reduction fix")
            self.llama = ct.CDLL(str(dll_directory / "llama.dll"))
            self._bind_functions()
            self.llama.llama_backend_init()
            self.backend_initialized = True
            params = self.llama.llama_model_default_params()
            params.n_gpu_layers = gpu_layers
            params.main_gpu = device_index
            params.split_mode = 0
            overrides = (ModelOverride * 2)()
            if self.headless:
                overrides[0].tag = 2
                overrides[0].key = b"qwen3vl.hidden_states_only"
                overrides[0].val_bool = True
                params.kv_overrides = ct.cast(overrides, ct.c_void_p)
            self.model = self.llama.llama_model_load_from_file(os.fsencode(model_path), params)
            if not self.model:
                raise RuntimeError("ROCmFPX could not load the text encoder")
            metadata = {b"general.architecture": b"qwen3vl"}
            if self.headless:
                metadata[b"qwen3vl.n_deepstack_layers"] = b"3"
            for key, expected in metadata.items():
                value = ct.create_string_buffer(32)
                length = self.llama.llama_model_meta_val_str(self.model, key, value, len(value))
                if length != len(expected) or value.value != expected:
                    raise ValueError(f"ROCmFPX requires {key.decode()}={expected.decode()}")
            dimensions = (self.llama.llama_model_n_embd(self.model), self.llama.llama_model_n_layer(self.model))
            if dimensions != (self.hidden_size, self.layers):
                raise ValueError(f"Expected {model_kind} dimensions {(self.hidden_size, self.layers)}, got {dimensions}")
            input_size = self.llama.llama_model_n_embd_inp(self.model)
            if self.headless and input_size != 20480:
                raise ValueError("MiniMax H3 requires three DeepStack inputs and 5120 hidden channels")
            vocab = self.llama.llama_model_get_vocab(self.model)
            self.vocab_size = self.llama.llama_vocab_n_tokens(vocab)
        except Exception:
            self.close()
            raise

    def _bind_functions(self):
        llama, ggml = self.llama, self.ggml
        bind(llama, "llama_model_default_params", ModelParams)
        bind(llama, "llama_context_default_params", ContextParams)
        bind(llama, "llama_backend_init", None)
        bind(llama, "llama_backend_free", None)
        bind(llama, "llama_model_load_from_file", ct.c_void_p, ct.c_char_p, ModelParams)
        bind(llama, "llama_model_free", None, ct.c_void_p)
        bind(llama, "llama_init_from_model", ct.c_void_p, ct.c_void_p, ContextParams)
        bind(llama, "llama_free", None, ct.c_void_p)
        for name in ("llama_model_n_embd", "llama_model_n_embd_inp", "llama_model_n_layer"):
            bind(llama, name, ct.c_int32, ct.c_void_p)
        bind(llama, "llama_model_get_vocab", ct.c_void_p, ct.c_void_p)
        bind(llama, "llama_vocab_n_tokens", ct.c_int32, ct.c_void_p)
        bind(llama, "llama_model_meta_val_str", ct.c_int32, ct.c_void_p, ct.c_char_p, ct.POINTER(ct.c_char), ct.c_size_t)
        bind(llama, "llama_decode", ct.c_int32, ct.c_void_p, Batch)
        bind(llama, "llama_synchronize", None, ct.c_void_p)
        bind(llama, "llama_get_embeddings", ct.POINTER(ct.c_float), ct.c_void_p)
        bind(ggml, "ggml_get_name", ct.c_char_p, ct.POINTER(Tensor))
        bind(ggml, "ggml_nbytes", ct.c_size_t, ct.POINTER(Tensor))
        bind(ggml, "ggml_is_contiguous", ct.c_bool, ct.POINTER(Tensor))
        bind(ggml, "ggml_backend_tensor_get", None, ct.POINTER(Tensor), ct.c_void_p, ct.c_size_t, ct.c_size_t)

    def encode_tokens(self, ids):
        ids = np.asarray(ids)
        if ids.ndim == 2 and ids.shape[0] == 1:
            ids = ids[0]
        if self.headless or ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or ids.size == 0:
            raise ValueError("Qwen Image requires one nonempty token sequence")
        if ids.min() < 0 or ids.max() >= self.vocab_size:
            raise ValueError("Token ID outside the ROCmFPX vocabulary")
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        return self._encode(ids, None, np.arange(ids.size, dtype=np.int32))

    def encode_embeddings(self, packed, pos):
        packed, pos = np.asarray(packed), np.asarray(pos)
        if not self.headless or packed.ndim != 2 or packed.shape[1] != 20480 or packed.shape[0] == 0:
            raise ValueError("MiniMax H3 requires packed embeddings [sequence,20480]")
        if pos.shape != (4, packed.shape[0]) or not np.issubdtype(pos.dtype, np.integer):
            raise ValueError("MiniMax H3 requires integer mRoPE positions [4,sequence]")
        if np.any(pos < 0) or np.any(pos > 2147483647) or np.any(pos[3] != 0):
            raise ValueError("Invalid mRoPE positions")
        order = list(zip(*(plane.tolist() for plane in pos[:3])))
        if any(previous >= current for previous, current in zip(order, order[1:])):
            raise ValueError("These image positions require an index-causal mask unavailable in this ROCmFPX build")
        packed = np.ascontiguousarray(packed, dtype=np.float32)
        pos = np.ascontiguousarray(pos, dtype=np.int32)
        return self._encode(None, packed, pos)

    def _encode(self, ids, packed, positions):
        count = ids.size if ids is not None else packed.shape[0]
        hidden = np.empty((1, count, self.hidden_size), dtype=np.float32)
        capture_state = {"active": False, "count": 0, "error": None}

        @EvalCallback
        def capture(tensor, ask, user_data):
            if self.ggml.ggml_get_name(tensor) != b"l_out-35":
                return not ask
            if ask:
                return capture_state["active"]
            try:
                value = tensor.contents
                if value.type != 0 or tuple(value.ne) != (self.hidden_size, count, 1, 1):
                    raise RuntimeError("ROCmFPX returned an unexpected raw hidden tensor")
                if not self.ggml.ggml_is_contiguous(tensor) or self.ggml.ggml_nbytes(tensor) != hidden.nbytes:
                    raise RuntimeError("ROCmFPX raw hidden tensor is not contiguous FP32")
                self.ggml.ggml_backend_tensor_get(tensor, hidden.ctypes.data, 0, hidden.nbytes)
                capture_state["count"] += 1
            except Exception as error:
                # Python exceptions cannot propagate through a ctypes callback.
                capture_state["error"] = error
            return True

        params = self.llama.llama_context_default_params()
        size = max(count, int(positions.max()) + 1)
        params.n_ctx = max(256, ((size + 255) // 256) * 256)
        params.n_batch = params.n_ubatch = count
        params.n_seq_max = 1
        params.n_threads = params.n_threads_batch = 32
        params.pooling_type = params.attention_type = 0
        params.flash_attn_type = self.flash_attn
        params.type_k = params.type_v = self.kv_type
        params.embeddings = self.headless
        if not self.headless:
            params.cb_eval = ct.cast(capture, ct.c_void_p)
        self.context = self.llama.llama_init_from_model(self.model, params)
        if not self.context:
            raise RuntimeError("ROCmFPX context initialization failed")
        try:
            sequence_counts = np.ones(count, dtype=np.int32)
            sequence_zero = (ct.c_int32 * 1)(0)
            sequences = (ct.POINTER(ct.c_int32) * count)(*[ct.cast(sequence_zero, ct.POINTER(ct.c_int32))] * count)
            outputs = np.ones(count, dtype=np.int8)
            batch = Batch(count, ids.ctypes.data_as(ct.POINTER(ct.c_int32)) if ids is not None else None,
                          packed.ctypes.data_as(ct.POINTER(ct.c_float)) if packed is not None else None,
                          positions.ctypes.data_as(ct.POINTER(ct.c_int32)),
                          sequence_counts.ctypes.data_as(ct.POINTER(ct.c_int32)), sequences,
                          outputs.ctypes.data_as(ct.POINTER(ct.c_int8)))
            capture_state["active"] = True
            status = self.llama.llama_decode(self.context, batch)
            self.llama.llama_synchronize(self.context)
            capture_state["active"] = False
            if status != 0:
                raise RuntimeError(f"ROCmFPX prefill failed: {status}")
            if self.headless:
                pointer = self.llama.llama_get_embeddings(self.context)
                if not pointer:
                    raise RuntimeError("ROCmFPX returned no raw hidden states")
                np.copyto(hidden, np.ctypeslib.as_array(pointer, shape=(hidden.size,)).reshape(hidden.shape))
            elif capture_state["error"] is not None:
                raise capture_state["error"]
            elif capture_state["count"] != 1:
                raise RuntimeError("ROCmFPX did not return Qwen's pre-normalization hidden states")
            if not np.isfinite(hidden).all():
                raise RuntimeError("ROCmFPX returned non-finite hidden states")
            return hidden
        finally:
            self.llama.llama_free(self.context)
            self.context = None

    def close(self):
        if self.context:
            self.llama.llama_free(self.context)
            self.context = None
        if self.model:
            self.llama.llama_model_free(self.model)
            self.model = None
        if self.backend_initialized:
            self.llama.llama_backend_free()
            self.backend_initialized = False
        for handle in self.dll_directories:
            handle.close()
        self.dll_directories.clear()
