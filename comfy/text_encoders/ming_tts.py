# Adapted from https://github.com/inclusionAI/Ming-3.0-tts (MIT, Copyright 2025 inclusionAI).
import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

import comfy.ops
from comfy.ldm.ming_tts.model import MingTextModel
from comfy.ldm.ming_tts.runtime import prompt_ids, validate_tokenizer
from comfy.ldm.ming_tts.speaker import CAMPPlus


class MingTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.data = tokenizer_data["tokenizer_json"]
        backend = Tokenizer.from_str(self.data.numpy().tobytes().decode("utf-8"))
        self.tiny = backend.token_to_id("<spk>") == 157175
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|role_end|>" if self.tiny else "<|im_end|>")
        validate_tokenizer(self.tokenizer, self.tiny)

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        ids, positions = prompt_ids(self.tokenizer, text, tiny=self.tiny)
        return {"ming_tts": [[(token, 1.0) for token in ids]], "speaker_positions": positions}

    def state_dict(self):
        return {"tokenizer_json": self.data}


class MingTEModel(MingTextModel):
    def __init__(self, config, device="cpu", dtype=None, model_options={}):
        dtype = model_options.get("dtype", torch.bfloat16)
        operations = model_options.get("custom_operations") or comfy.ops.mixed_precision_ops({}, dtype)
        super().__init__(config, device, dtype, operations)
        self.speaker = CAMPPlus(device, comfy.ops.manual_cast)
        self.dtypes = {dtype, torch.float32}
        self.compute_dtype = dtype
        self.execution_device = device

    def set_clip_options(self, options):
        self.execution_device = options.get("execution_device", self.execution_device)

    def reset_clip_options(self):
        pass

    def encode_token_weights(self, tokens):
        ids = torch.tensor([[token for token, _ in tokens["ming_tts"][0]]], device=self.execution_device)
        with torch.autocast(device_type=ids.device.type, dtype=self.compute_dtype, enabled=self.compute_dtype != torch.float32):
            embeds = self.model.model.embed_tokens(ids, out_dtype=self.compute_dtype)
            for position in tokens["speaker_positions"]:
                embeds[:, position] = self.spk_head(torch.zeros((1, 192), device=ids.device, dtype=self.compute_dtype))
            cache = self.model.model.init_kv_cache(1, ids.shape[1], ids.device, self.compute_dtype)
            return self.model.model(None, embeds=embeds, past_key_values=cache)[0], None

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict = {key: value for key, value in state_dict.items() if key != "tokenizer_json"}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def load_sd(self, state_dict):
        return self.load_state_dict(state_dict, strict=False, assign=getattr(self, "can_assign_sd", False))
