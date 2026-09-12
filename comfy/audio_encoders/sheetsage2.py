"""SheetSage2 audio-to-score generation with the released event vocabulary."""

import copy
import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.utils
from comfy.audio_encoders.mert2 import MERT2
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.llama import FixedKV


class DecoderAttention(nn.Module):
    def __init__(self, dim, heads, device=None, dtype=None, operations=None):
        super().__init__()
        self.heads = heads
        self.q_proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.k_proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.v_proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.out_proj = operations.Linear(dim, dim, device=device, dtype=dtype)

    def project(self, x, projection):
        return projection(x).reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)

    def forward(self, x, attention, mask=None, cache=None, memory=None):
        q = self.project(x, self.q_proj)
        if memory is not None:
            k, v = memory
        else:
            k, v = self.project(x, self.k_proj), self.project(x, self.v_proj)
            length = x.shape[1]
            if isinstance(cache, FixedKV):
                key, value = k.transpose(1, 2), v.transpose(1, 2)
                if length == 1 and cache.index > 0:
                    position = cache.position.view(-1, 1, 1, 1).expand_as(key)
                    cache.key.scatter_(1, position, key)
                    cache.value.scatter_(1, position, value)
                    valid = torch.arange(cache.key.shape[1], device=x.device)[None] < cache.seqlen[:, None]
                    mask = torch.zeros(valid.shape, device=x.device, dtype=x.dtype).masked_fill_(~valid, -torch.inf)[:, None, None]
                    out = attention(q, cache.key.transpose(1, 2), cache.value.transpose(1, 2), self.heads, mask=mask, skip_reshape=True)
                    return self.out_proj(out), cache
                cache.key[:, :length].copy_(key)
                cache.value[:, :length].copy_(value)
            elif cache is not None:
                key, value, index = cache
                key[:, :, index:index + length].copy_(k)
                value[:, :, index:index + length].copy_(v)
                k, v = key[:, :, :index + length], value[:, :, :index + length]
                cache = key, value, index + length
        out = attention(q, k, v, self.heads, mask=mask, skip_reshape=True)
        return self.out_proj(out), cache


class DecoderLayer(nn.Module):
    def __init__(self, dim, intermediate, heads, device=None, dtype=None, operations=None):
        super().__init__()
        self.self_attn = DecoderAttention(dim, heads, device=device, dtype=dtype, operations=operations)
        self.self_attn_layer_norm = operations.LayerNorm(dim, eps=1e-5, device=device, dtype=dtype)
        self.encoder_attn = DecoderAttention(dim, heads, device=device, dtype=dtype, operations=operations)
        self.encoder_attn_layer_norm = operations.LayerNorm(dim, eps=1e-5, device=device, dtype=dtype)
        self.fc1 = operations.Linear(dim, intermediate, device=device, dtype=dtype)
        self.fc2 = operations.Linear(intermediate, dim, device=device, dtype=dtype)
        self.final_layer_norm = operations.LayerNorm(dim, eps=1e-5, device=device, dtype=dtype)

    def forward(self, x, attention, mask, cache):
        self_cache, memory = cache
        out, self_cache = self.self_attn(x, attention, mask=mask, cache=self_cache)
        x = self.self_attn_layer_norm(x + out)
        out, _ = self.encoder_attn(x, attention, memory=memory)
        x = self.encoder_attn_layer_norm(x + out)
        return self.final_layer_norm(x + self.fc2(F.gelu(self.fc1(x)))), (self_cache, memory)


class Decoder(nn.Module):
    def __init__(self, dim, intermediate, heads, layers, max_tokens, device=None, dtype=None, operations=None):
        super().__init__()
        self.embed_positions = operations.Embedding(max_tokens + 2, dim, device=device, dtype=dtype)
        self.layernorm_embedding = operations.LayerNorm(dim, eps=1e-5, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            DecoderLayer(dim, intermediate, heads, device=device, dtype=dtype, operations=operations) for _ in range(layers)
        ])

    def forward(self, x, positions, cache):
        length = x.shape[1]
        fixed = isinstance(cache[0][0], FixedKV)
        index = cache[0][0].index if fixed else cache[0][0][2]
        mask = None
        if length > 1:
            mask = torch.full((length, index + length), -torch.inf, device=x.device, dtype=x.dtype).triu_(index + 1)
        x = self.layernorm_embedding(x + self.embed_positions(positions, out_dtype=x.dtype))
        graph = fixed and length == 1 and index > 0
        if graph:
            x = x.clone()
        attention = optimized_attention_for_device(x.device, mask=mask is not None or graph, small_input=True)
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, {"prefetch_dynamic_vbars": True})
        for i, layer in enumerate(self.layers):
            if fixed:
                cache[i][0].prepare(length)

            def core():
                nonlocal x
                out, cache[i] = layer(x, attention, mask, cache[i])
                if graph:
                    x.copy_(out)
                else:
                    x = out

            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, layer, x.dtype, core=core,
                                                  enable_graph=graph, malloc_scope="block")
            if fixed:
                cache[i][0].advance(length)
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="block")
        return x


class SheetSage2(nn.Module):
    def __init__(self, dim=512, intermediate=2048, heads=8, layers=6, max_tokens=5120,
                 mert_config=None, device=None, dtype=None, operations=None):
        super().__init__()
        mert_config = {} if mert_config is None else mert_config
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.encoder = MERT2(**mert_config, device=device, dtype=dtype, operations=operations)
        self.layer_weight = nn.Parameter(torch.empty(len(self.encoder.layers) + 1, device=device, dtype=dtype))
        self.encoder_projection = operations.Linear(mert_config.get("dim", 1024), dim, device=device, dtype=dtype)
        self.tokenizer = ScoreTokenizer()
        self.token_embedding = operations.Embedding(self.tokenizer.n_tokens, dim, device=device, dtype=dtype)
        self.decoder = Decoder(dim, intermediate, heads, layers, max_tokens, device=device, dtype=dtype, operations=operations)
        self.output_projection = operations.Linear(dim, self.tokenizer.n_tokens, bias=False, device=device, dtype=dtype)

    def get_dynamic_vram__units(self):
        return list(self.decoder.layers), []

    def encode(self, waveform, output_hidden_states=False):
        # The released encoder attends to the entire 300-second window, including its padding.
        waveform = F.pad(waveform, (0, max(0, 300 * 24000 - waveform.shape[-1])))
        mel = self.encoder.feature_extractor(waveform.float()).to(self.dtype)
        mixed, states = self.encoder(mel, self.layer_weight, output_hidden_states=output_hidden_states)
        return self.encoder_projection(mixed), states

    def forward(self, audio):
        return self.encode(audio.mean(dim=1), output_hidden_states=True)

    def init_cache(self, memory):
        batch, _, dim = memory.shape
        fixed = comfy.model_prefetch.malloc_graph_enabled(memory.device)
        cache = []
        for layer in self.decoder.layers:
            heads = layer.self_attn.heads
            shape = (batch, self.max_tokens, heads, dim // heads) if fixed else (batch, heads, self.max_tokens, dim // heads)
            # Fixed attention includes masked future slots, whose values must remain finite.
            key = torch.zeros(shape, device=memory.device, dtype=memory.dtype) if fixed else torch.empty(shape, device=memory.device, dtype=memory.dtype)
            value = torch.zeros_like(key) if fixed else torch.empty_like(key)
            if fixed:
                self_cache = FixedKV(key, value, 0, torch.empty(batch, device=memory.device, dtype=torch.long),
                                     torch.zeros(batch, device=memory.device, dtype=torch.int32))
            else:
                self_cache = key, value, 0
            cross = layer.encoder_attn
            cache.append((self_cache, (cross.project(memory, cross.k_proj), cross.project(memory, cross.v_proj))))
        return cache

    def decode(self, ids, positions, cache):
        x = self.token_embedding(ids, out_dtype=self.dtype)
        return self.output_projection(self.decoder(x, positions, cache)[:, -1:])

    def generate_tokens(self, memory, stop_seconds, prefix=None):
        tokenizer = self.tokenizer
        tokens = tokenizer.prompt_prefix() if prefix is None else list(prefix)
        state = PromptGrammarState(tokenizer)
        for token in tokens[tokens.index(tokenizer.out_token) + 1:]:
            state.update(token)
        cache = self.init_cache(memory)
        device = memory.device
        ids = torch.tensor([tokens], device=device, dtype=torch.long)
        positions = torch.arange(2, len(tokens) + 2, device=device)[None]
        logits = self.decode(ids, positions, cache)
        ids = torch.empty((1, 1), device=device, dtype=torch.long)
        positions = torch.full((1, 1), len(tokens) + 2, device=device, dtype=torch.long)
        fixed = isinstance(cache[0][0], FixedKV)
        progress = comfy.utils.ProgressBar(self.max_tokens - len(tokens))
        try:
            for step in comfy.utils.model_trange(self.max_tokens - len(tokens), desc="SheetSage2 transcription", unit="token"):
                comfy.model_management.throw_exception_if_processing_interrupted()
                scores = logits[0, -1].float().masked_fill(~state.allowed(device), -torch.inf)
                next_id = scores.argmax()
                token = next_id.item()
                tokens.append(token)
                progress.update_absolute(step + 1)
                if state.update(token):
                    break
                if tokenizer.time_token_start <= token < tokenizer.time_token_end and tokenizer.token_to_time_id(token) / tokenizer.time_hz >= stop_seconds:
                    tokens.append(tokenizer.eos_token)
                    break
                if len(tokens) == self.max_tokens:
                    logging.warning("SheetSage2 reached its token limit; the transcription may be incomplete.")
                    tokens.append(tokenizer.eos_token)
                    break
                ids.copy_(next_id)
                if fixed:
                    comfy.model_prefetch.malloc_graph_begin(device)
                logits.copy_(self.decode(ids, positions, cache))
                if fixed:
                    comfy.model_prefetch.malloc_graph_end()
                positions.add_(1)
        finally:
            comfy.model_prefetch.cleanup_prefetch_queues()
        return tokens

    def transcribe(self, waveform):
        duration = waveform.shape[-1] / 24000
        stitched = []
        for window in sliding_window_plan(duration):
            comfy.model_management.throw_exception_if_processing_interrupted()
            start = window["start"]
            prefix, base = overlap_prefix(stitched, self.tokenizer, start, window["prefix_end"])
            if prefix is not None and len(prefix) >= self.max_tokens - 128:
                raise ValueError("SheetSage2 overlap fills the token context; transcribe shorter audio sections.")
            segment = waveform[:, round(start * 24000):round(window["end"] * 24000)]
            memory, _ = self.encode(segment)
            stop = window["generation_stop"] if window["generation_stop"] is not None else min(duration - start, 300.0)
            tokens = self.generate_tokens(memory, stop, prefix)
            decoded = self.tokenizer.decode_sequence(tokens)
            lookup = event_time_map(decoded, 300.0)
            stitched.extend(stitched_window_events(decoded, lookup, start, window["accept_start"],
                                                  window["accept_end"], duration, global_subbeat_base=base))
        stitched.sort(key=lambda event: (event["time"], event["global_subbeat"]))
        return stitched


CHROMATIC_SHARPS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
STRUCTURE_LABELS = (
    "silence", "intro", "outro", "verse", "chorus", "bridge", "pre-chorus", "post-chorus", "interlude",
    "fade-out", "loop", "rap", "preshot", "irregular", "instrumental", "intro and verse", "pre-chorus and chorus",
    "verse and pre-chorus", "solo", "theme", "development", "variation", "pre-outro",
)
DURATION_TEMPLATES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
EVENT_FIELDS = ("timestamp", "rhythm", "structure", "key", "chord", "melody")
FIELD_TO_INDEX = {name: index for index, name in enumerate(EVENT_FIELDS)}


class ScoreTokenizer:
    pad_token, sos_token, eos_token, out_token = 0, 1, 2, 3
    time_hz = 100

    def __init__(self):
        self.meter_pairs = tuple((numerator, denominator) for numerator in range(1, 33) for denominator in (1, 2, 4, 8, 16, 32))
        self.full_chord_labels = ["N"]
        inversions = {"maj": ("/2", "/3", "/5"), "min": ("/2", "/b3", "/5"),
                      "maj7": ("/3", "/5", "/7"), "min7": ("/b3", "/5", "/b7"), "7": ("/3", "/5", "/b7")}
        for quality in ("maj", "min", "dim", "aug", "maj7", "min7", "7", "hdim7", "dim7", "minmaj7", "sus2", "sus4", "sus4(b7)", "maj6", "min6"):
            for root in CHROMATIC_SHARPS:
                self.full_chord_labels.extend(f"{root}:{quality}{inversion}" for inversion in (*inversions.get(quality, ()), ""))
        offset = 260
        self.ranges = []
        for name, count in (("subbeat_shift", 257), ("time", 30000), ("meter", 192), ("eighth_position", 256),
                            ("structure", len(STRUCTURE_LABELS)), ("key", 24), ("majmin_chord", 25),
                            ("full_chord", len(self.full_chord_labels)), ("pitch", 256), ("duration", len(DURATION_TEMPLATES))):
            setattr(self, f"{name}_token_start", offset)
            setattr(self, f"{name}_token_end", offset + count)
            self.ranges.append((name.replace("full_chord", "chord_full").replace("majmin_chord", "chord_majmin"), offset, offset + count))
            offset += count
        self.n_tokens = offset

    def prompt_prefix(self):
        return [self.sos_token, 4, 5, 6, 7, 9, 11, self.out_token]

    def token_type(self, token):
        for name, start, end in self.ranges:
            if start <= token < end:
                return name
        return {0: "pad", 1: "sos", 2: "eos", 3: "out"}.get(token, "prompt")

    def token_to_time_id(self, token):
        return token - self.time_token_start

    def decode_field(self, field, tokens):
        if field == "timestamp":
            return self.token_to_time_id(tokens[0]) / self.time_hz
        if field == "rhythm":
            rhythm = {}
            for token in tokens:
                if self.token_type(token) == "meter":
                    rhythm["meter"] = self.meter_pairs[token - self.meter_token_start]
                else:
                    rhythm["eighth_position"] = token - self.eighth_position_token_start
            return rhythm
        if field == "structure":
            return STRUCTURE_LABELS[tokens[0] - self.structure_token_start]
        if field == "key":
            index = tokens[0] - self.key_token_start
            return f"{CHROMATIC_SHARPS[index % 12]}:{'minor' if index >= 12 else 'major'}"
        if field == "chord":
            return self.full_chord_labels[tokens[0] - self.full_chord_token_start]
        notes, index = [], 0
        while index < len(tokens):
            pitch = tokens[index] - self.pitch_token_start
            index += 1
            duration = 0
            if index < len(tokens) and self.token_type(tokens[index]) == "duration":
                duration = tokens[index] - self.duration_token_start
                index += 1
            notes.append({"pitch": pitch % 128, "track": int(pitch >= 128),
                          "duration_bin": duration, "duration_steps": DURATION_TEMPLATES[duration]})
        return notes

    def decode_sequence(self, tokens):
        fields = {"time": "timestamp", "meter": "rhythm", "eighth_position": "rhythm", "structure": "structure",
                  "key": "key", "chord_full": "chord", "pitch": "melody", "duration": "melody"}
        position, subbeat, events = tokens.index(self.out_token) + 1, 0, []
        while position < len(tokens) and tokens[position] != self.eos_token:
            if self.token_type(tokens[position]) != "subbeat_shift":
                raise ValueError("SheetSage2 produced an event without a beat position.")
            while position < len(tokens) and self.token_type(tokens[position]) == "subbeat_shift":
                subbeat += tokens[position] - self.subbeat_shift_token_start
                position += 1
            payload = {}
            while position < len(tokens) and self.token_type(tokens[position]) not in ("subbeat_shift", "eos"):
                token = tokens[position]
                payload.setdefault(fields[self.token_type(token)], []).append(token)
                position += 1
            if payload:
                events.append({"subbeat": subbeat, "tokens_by_field": payload,
                               "values": {field: self.decode_field(field, values) for field, values in payload.items()}})
        return {"events": events}

    def encode_events(self, events):
        tokens, previous = self.prompt_prefix(), 0
        for event in events:
            shift = event["subbeat"] - previous
            while shift > 256:
                tokens.append(self.subbeat_shift_token_end - 1)
                shift -= 256
            tokens.append(self.subbeat_shift_token_start + shift)
            previous = event["subbeat"]
            for field in EVENT_FIELDS:
                tokens.extend(event["tokens_by_field"].get(field, ()))
        return tokens


def sliding_window_plan(duration, window_seconds=300.0, overlap_seconds=200.0, lookahead_seconds=100.0):
    hop = window_seconds - overlap_seconds
    start, accepted = 0.0, 0.0
    result = []
    while True:
        last = start + window_seconds >= duration - 1e-6
        accept_end = duration if last else start + window_seconds - lookahead_seconds
        result.append(dict(start=start, end=min(duration, start + window_seconds),
                           accept_start=accepted, accept_end=accept_end, prefix_end=accepted,
                           generation_stop=None if last else window_seconds - lookahead_seconds))
        if last:
            return result
        accepted = accept_end
        start = min(start + hop, duration - window_seconds)


def overlap_prefix(stitched, tokenizer, start, prefix_end):
    events = [event for event in stitched if start - 1e-4 <= event["time"] < prefix_end - 1e-4]
    events.sort(key=lambda event: (event["global_subbeat"], event["time"]))
    first = next((i for i, event in enumerate(events) if "timestamp" in event["values"] or "rhythm" in event["values"]), None)
    if first is None:
        return None, 0
    events = copy.deepcopy(events[first:])
    base = events[0]["global_subbeat"]
    context = {}
    for event in stitched:
        if event["time"] > events[0]["time"] + 1e-6:
            continue
        for field in ("structure", "key", "chord"):
            if event["tokens_by_field"].get(field):
                context[field] = event["tokens_by_field"][field]
        for token in event["tokens_by_field"].get("rhythm", ()):
            if tokenizer.token_type(token) == "meter":
                context["meter"] = token
    for event in events:
        event["subbeat"] = max(0, event["global_subbeat"] - base)
        if "timestamp" in event["tokens_by_field"]:
            time_id = min(29999, max(0, round((event["time"] - start) * tokenizer.time_hz)))
            event["tokens_by_field"]["timestamp"] = [tokenizer.time_token_start + time_id]
    first_fields = events[0]["tokens_by_field"]
    for field in ("structure", "key", "chord"):
        if field not in first_fields and field in context:
            first_fields[field] = list(context[field])
    rhythm = first_fields.get("rhythm", [])
    if any(tokenizer.token_type(token) == "eighth_position" for token in rhythm) and not any(tokenizer.token_type(token) == "meter" for token in rhythm) and "meter" in context:
        first_fields["rhythm"] = [context["meter"], *rhythm]
    return tokenizer.encode_events(events), base


def event_time_map(decoded, target_seconds):
    anchors = sorted({event["subbeat"]: event["values"]["timestamp"] for event in decoded["events"] if "timestamp" in event["values"]}.items())
    if not anchors:
        return lambda step: min(target_seconds, max(0.0, step * 0.125))
    steps, times = np.asarray(anchors, dtype=np.float64).T
    period = float(np.median(np.diff(times) / np.maximum(np.diff(steps), 1))) if len(anchors) > 1 else 0.125
    if not np.isfinite(period) or period <= 0:
        period = 0.125

    def lookup(step):
        if step <= steps[0]:
            return float(np.clip(times[0] + (step - steps[0]) * period, 0, target_seconds))
        if step >= steps[-1]:
            return float(np.clip(times[-1] + (step - steps[-1]) * period, 0, target_seconds))
        return float(np.interp(step, steps, times))

    return lookup


def stitched_window_events(decoded, lookup, start, accept_start, accept_end, duration, global_subbeat_base=0):
    accepted = []
    for source in decoded["events"]:
        time = start + lookup(source["subbeat"])
        if time < accept_start - 1e-4 or time >= accept_end - 1e-4 or time >= duration - 1e-4:
            continue
        event = copy.deepcopy(source)
        event["time"] = float(np.clip(time, 0, duration))
        event["global_subbeat"] = global_subbeat_base + event["subbeat"]
        if "timestamp" in event["values"]:
            event["values"]["timestamp"] = event["time"]
        for note in event["values"].get("melody", ()):
            note["end_time"] = min(duration, max(event["time"] + 0.04, start + lookup(event["subbeat"] + note["duration_steps"])))
        accepted.append(event)
    return accepted


class PromptGrammarState:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.generated_events = 0
        self.in_shift = True
        self.shift_run = 0
        self.payload_count = 0
        self.last_field_index = -1
        self.incomplete = None

    def _allow_field_starts(self, allowed):
        tokenizer = self.tokenizer
        if self.last_field_index < FIELD_TO_INDEX["timestamp"]:
            allowed[tokenizer.time_token_start : tokenizer.time_token_end] = True
        if self.last_field_index < FIELD_TO_INDEX["rhythm"]:
            allowed[tokenizer.meter_token_start : tokenizer.meter_token_end] = True
            allowed[
                tokenizer.eighth_position_token_start : tokenizer.eighth_position_token_end
            ] = True
        if self.last_field_index < FIELD_TO_INDEX["structure"]:
            allowed[
                tokenizer.structure_token_start : tokenizer.structure_token_end
            ] = True
        if self.last_field_index < FIELD_TO_INDEX["key"]:
            allowed[tokenizer.key_token_start : tokenizer.key_token_end] = True
        if self.last_field_index < FIELD_TO_INDEX["chord"]:
            allowed[
                tokenizer.full_chord_token_start : tokenizer.full_chord_token_end
            ] = True
        if self.last_field_index <= FIELD_TO_INDEX["melody"]:
            allowed[tokenizer.pitch_token_start : tokenizer.pitch_token_end] = True

    def allowed(self, device):
        tokenizer = self.tokenizer
        allowed = torch.zeros(tokenizer.n_tokens, dtype=torch.bool, device=device)
        can_end = self.payload_count > 0

        if can_end:
            allowed[tokenizer.eos_token] = True
        if self.payload_count > 0 or self.in_shift:
            if self.shift_run < 4:
                allowed[
                    tokenizer.subbeat_shift_token_start : tokenizer.subbeat_shift_token_end
                ] = True

        if self.incomplete == "rhythm_after_meter":
            allowed[
                tokenizer.eighth_position_token_start : tokenizer.eighth_position_token_end
            ] = True
            return allowed

        if self.incomplete == "melody_after_pitch":
            allowed[tokenizer.duration_token_start : tokenizer.duration_token_end] = True
            allowed[tokenizer.pitch_token_start : tokenizer.pitch_token_end] = True
            return allowed

        self._allow_field_starts(allowed)
        return allowed

    def update(self, token):
        tokenizer = self.tokenizer
        token = int(token)
        token_type = tokenizer.token_type(token)
        if token == tokenizer.eos_token:
            return True
        if token_type == "subbeat_shift":
            if not self.in_shift and self.payload_count > 0:
                self.generated_events += 1
                self.payload_count = 0
                self.last_field_index = -1
                self.incomplete = None
            self.in_shift = True
            self.shift_run += 1
            return False

        self.in_shift = False
        self.shift_run = 0
        self.payload_count += 1
        if token_type == "time":
            self.last_field_index = FIELD_TO_INDEX["timestamp"]
            self.incomplete = None
        elif token_type == "meter":
            self.last_field_index = FIELD_TO_INDEX["rhythm"]
            self.incomplete = "rhythm_after_meter"
        elif token_type == "eighth_position":
            self.last_field_index = FIELD_TO_INDEX["rhythm"]
            self.incomplete = None
        elif token_type == "structure":
            self.last_field_index = FIELD_TO_INDEX["structure"]
            self.incomplete = None
        elif token_type == "key":
            self.last_field_index = FIELD_TO_INDEX["key"]
            self.incomplete = None
        elif token_type == "chord_full":
            self.last_field_index = FIELD_TO_INDEX["chord"]
            self.incomplete = None
        elif token_type == "pitch":
            self.last_field_index = FIELD_TO_INDEX["melody"]
            self.incomplete = "melody_after_pitch"
        elif token_type == "duration":
            self.last_field_index = FIELD_TO_INDEX["melody"]
            self.incomplete = None
        else:
            raise RuntimeError(f"Unexpected prompt token type {token_type!r}")
        return False
