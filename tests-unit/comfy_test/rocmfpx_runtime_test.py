import ctypes as ct

import numpy as np
import pytest

from comfy.text_encoders.rocmfpx_runtime import ContextParams, FpxSession


@pytest.fixture
def session():
    # Bypass DLL/model loading; exercise the real validation and ctypes batch path.
    result = FpxSession.__new__(FpxSession)
    result.hidden_size = 5120
    result.headless = True
    result.kv_type = 0
    result.flash_attn = 0
    result.model = 1
    result.context = None
    return result


def h3_inputs():
    packed = np.arange(3 * 40960, dtype=np.float64).reshape(3, 40960)[:, ::2]
    positions = np.array([[1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=np.int64)
    return packed, positions


class BatchProbe:
    def __init__(self, packed, positions, status):
        self.packed = packed
        self.positions = positions
        self.status = status
        self.output = np.arange(3 * 5120, dtype=np.float32).reshape(1, 3, 5120)
        self.events = []

    def llama_context_default_params(self):
        return ContextParams()

    def llama_init_from_model(self, model, params):
        assert model == 1
        assert params.embeddings and params.n_seq_max == 1
        assert params.n_batch == params.n_ubatch == 3
        self.events.append("init")
        return 123

    def llama_decode(self, context, batch):
        assert context == 123 and batch.n_tokens == 3 and not batch.token
        np.testing.assert_array_equal(np.ctypeslib.as_array(batch.embd, shape=(3 * 20480,)), self.packed.astype(np.float32).ravel())
        np.testing.assert_array_equal(np.ctypeslib.as_array(batch.pos, shape=(12,)), self.positions.ravel())
        np.testing.assert_array_equal(np.ctypeslib.as_array(batch.n_seq_id, shape=(3,)), [1, 1, 1])
        np.testing.assert_array_equal(np.ctypeslib.as_array(batch.logits, shape=(3,)), [1, 1, 1])
        assert [batch.seq_id[i][0] for i in range(3)] == [0, 0, 0]
        self.events.append("decode")
        return self.status

    def llama_synchronize(self, context):
        assert context == 123
        self.events.append("sync")

    def llama_get_embeddings(self, context):
        assert context == 123
        self.events.append("output")
        return self.output.ctypes.data_as(ct.POINTER(ct.c_float))

    def llama_free(self, context):
        assert context == 123
        self.events.append("free")
        self.output.fill(np.nan)


@pytest.mark.parametrize("status", [0, 1])
def test_h3_batch_and_context_lifetime(session, status):
    packed, positions = h3_inputs()
    original_packed, original_positions = packed.copy(), positions.copy()
    session.llama = BatchProbe(packed, positions, status)
    expected = session.llama.output.copy()

    if status:
        with pytest.raises(RuntimeError, match="prefill failed: 1"):
            session.encode_embeddings(packed, positions)
        assert session.llama.events == ["init", "decode", "sync", "free"]
    else:
        hidden = session.encode_embeddings(packed, positions)
        np.testing.assert_array_equal(hidden, expected)
        assert not np.shares_memory(hidden, session.llama.output)
        assert session.llama.events == ["init", "decode", "sync", "output", "free"]

    assert session.context is None
    np.testing.assert_array_equal(packed, original_packed)
    np.testing.assert_array_equal(positions, original_positions)


@pytest.mark.parametrize("case, message", [
    ("width", "packed embeddings"),
    ("empty", "packed embeddings"),
    ("axes", "integer mRoPE positions"),
    ("float_positions", "integer mRoPE positions"),
    ("negative", "Invalid mRoPE positions"),
    ("overflow", "Invalid mRoPE positions"),
    ("fourth_axis", "Invalid mRoPE positions"),
    ("noncausal", "index-causal mask"),
])
def test_h3_invalid_inputs_fail_before_loading_a_context(session, case, message):
    packed, positions = h3_inputs()
    if case == "width":
        packed = packed[:, :-1]
    elif case == "empty":
        packed, positions = packed[:0], positions[:, :0]
    elif case == "axes":
        positions = positions[:3]
    elif case == "float_positions":
        positions = positions.astype(np.float32)
    elif case == "negative":
        positions[0, 0] = -1
    elif case == "overflow":
        positions[0, 0] = 2 ** 31
    elif case == "fourth_axis":
        positions[3, 0] = 1
    elif case == "noncausal":
        positions[:, 1] = positions[:, 0]

    with pytest.raises(ValueError, match=message):
        session.encode_embeddings(packed, positions)
    assert session.context is None
