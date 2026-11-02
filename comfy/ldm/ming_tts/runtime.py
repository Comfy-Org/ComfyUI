# Adapted from https://github.com/inclusionAI/Ming-3.0-tts (MIT, Copyright 2025 inclusionAI).
import logging
import re
from contextlib import nullcontext

import torch
from torch.nn import functional as F

import comfy.audio
import comfy.model_management
import comfy.model_prefetch
import comfy.utils
from .model import sample_flow
from .speaker import filterbank


DEFAULT_PROMPT = "Please generate speech based on the following description.\n"
DENSE_TOKENS = {
    "<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645,
    "<|vision_start|>": 151652, "<|vision_end|>": 151653, "<|vision_pad|>": 151654,
    "<audioPatch>": 151705, "<audio>": 151706,
}
TINY_TOKENS = {"<audioPatch>": 157164, "<audio>": 157165, "</audio>": 157166, "<spk>": 157175, "</spk>": 157176}


def validate_tokenizer(tokenizer, tiny):
    for token, expected in (TINY_TOKENS if tiny else DENSE_TOKENS).items():
        if tokenizer.encode(token, add_special_tokens=False) != [expected]:
            raise ValueError(f"Ming TTS tokenizer mismatch: {token!r} must encode as {expected}. Use the checkpoint's tokenizer.")
    if tiny and tokenizer.eos_token_id != 156895:
        raise ValueError("Ming TTS tiny requires the released tokenizer with EOS 156895.")


def prompt_ids(tokenizer, text, tiny=False, instruction="", language="", prompt_text="", latent_steps=0,
               num_speakers=0):
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    instruction = instruction.strip()
    positions = []
    if tiny:
        if instruction and not instruction.startswith(("Speaking instruction:", "{")):
            instruction = "Speaking instruction: Speak in this style: " + instruction
        if num_speakers > 1:
            text = "".join(
                re.sub(r"(?i)^speaker[_ ]?(\d+)\s*[:：]\s*", r" speaker_\1:", line.strip()) + "\n"
                for line in text.splitlines() if line.strip()
            )
        if not language:
            lexical = re.sub(r"(?im)^\s*speaker[_ ]?\d+\s*[:：]\s*|\[[^\]\n]{1,128}\]|\([A-Za-z][A-Za-z _-]{0,127}\)", "", text)
            language = "Chinese" if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", lexical) else "English"
        prefix = "<role>HUMAN</role>" + DEFAULT_PROMPT
        if not instruction:
            prefix += f"Target language: {language}.\n"
        ids = encode(prefix)
        for i in range(max(1, num_speakers)):
            ids += encode(("\n" if i else "") + f"  speaker_{i + 1}:") + [157175]
            positions.append(len(ids))
            ids += [157164, 157176]
        ids += encode("\n Text input:\n") + encode(prompt_text) + encode(text) + encode("<role>ASSISTANT</role>")
        if instruction:
            ids += encode(instruction) + [156895]
        ids += [157165]
        ids += [157164] * latent_steps
    else:
        prompt = DEFAULT_PROMPT + (f"Target language: {language}.\n" if language else "")
        ids = encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
        ids += encode("<|im_start|>user\n") + encode(prompt)
        for i in range(num_speakers):
            ids += encode(f"  speaker_{i + 1}:") + encode("<|vision_start|>")
            positions.append(len(ids))
            ids += encode("<|vision_pad|>") + encode("<|vision_end|>\n")
        if not all(s in text for s in ("Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: ")):
            ids += encode(" Text input:\n")
        if latent_steps:
            ids += encode(prompt_text)
        ids += encode(text) + encode("<|im_end|>\n") + encode("<|im_start|>assistant\n")
        if instruction:
            ids += encode(instruction) + encode("<|im_end|>")
        ids += encode("<audio>") + encode("<audioPatch>") * latent_steps
    return ids, positions


def speaker_embedding(bundle, audio):
    comfy.model_management.load_models_gpu([bundle.patcher])
    device = bundle.patcher.load_device
    mono = audio["waveform"].to(device=device, dtype=torch.float32).mean(dim=1)
    wave = comfy.audio.resample(mono, audio["sample_rate"], 16000)
    wave = F.pad(wave, (0, max(0, 560 - wave.shape[-1])))
    with torch.autocast(device_type=device.type, enabled=False):
        return bundle.cond_stage_model.speaker(filterbank(wave)).cpu()


def split_sentences(text, max_length):
    # Prefer sentences, then clauses and word boundaries as in the reference frontend.
    units = r"[\u3400-\u9fff\u3040-\u30ff]|[^\W_\u3400-\u9fff\u3040-\u30ff]+(?:['’.-][^\W_\u3400-\u9fff\u3040-\u30ff]+)*"
    boundaries = r'''[.!?]+["'’”»)\]]*(?=\s|$)|[。！？]+["'’”»」』）)\]]*|\Z'''
    abbreviation = r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|[A-Z](?:\.[A-Z])*)\.$"
    chunks = []
    chunk_start = sentence_start = length = 0
    for match in re.finditer(boundaries, text):
        if match.group() == "." and re.search(abbreviation, text[:match.end()]):
            continue
        sentence_length = len(re.findall(units, text[sentence_start:match.end()]))
        if length and length + sentence_length > max_length:
            chunks.append(text[chunk_start:sentence_start].strip())
            chunk_start, length = sentence_start, 0
        length += sentence_length
        sentence_start = match.end()
    if text[chunk_start:].strip():
        chunks.append(text[chunk_start:].strip())
    fragments = []
    for chunk in chunks:
        words = list(re.finditer(units, chunk))
        while len(words) > max_length:
            end = words[max_length].start()
            clauses = list(re.finditer(r"[,;:](?=\s)|[，、；：]", chunk[:end]))
            if clauses:
                end = clauses[-1].end()
            fragments.append(chunk[:end].strip())
            chunk = chunk[end:].strip()
            words = list(re.finditer(units, chunk))
        fragments.append(chunk)
    return fragments


def generate(bundle, diffusion, text, seed=0, max_steps=250, cfg=2.0, sigma=0.25, temperature=0.0, instruction="", language="",
             speakers=(), separate_turns=True,
             min_steps=None, stop_threshold=0.5, chunk_length=90, _voice_context=None):
    model = bundle.cond_stage_model
    device, dtype = bundle.patcher.load_device, model.compute_dtype
    if not text.strip():
        raise ValueError("Enter text to synthesize.")
    if max_steps < 1:
        raise ValueError("max_steps must be positive.")
    if separate_turns and chunk_length < 1:
        raise ValueError("chunk_length must be positive.")
    num_speakers = len(speakers)
    if num_speakers > 1 and not model.tiny:
        raise ValueError("Multiple speakers require the Ming TTS tiny checkpoint.")
    turns = []
    if num_speakers > 1 and (separate_turns or not all("reference_latents" in s for s in speakers)):
        dialogue = re.split(r"(?im)^[ \t]*speaker[_ ]?(\d+)[ \t]*[:：][ \t]*", text)
        labeled = [(int(index) - 1, words.strip()) for index, words in zip(dialogue[1::2], dialogue[2::2]) if words.strip()]
        if labeled and not dialogue[0].strip():
            if any(index < 0 or index >= num_speakers for index, _ in labeled):
                raise ValueError("Each speaker label must have a connected Ming Speaker Encode input.")
            turns = [(words, [speakers[index]]) for index, words in labeled]
    elif separate_turns:
        turns = [(text, speakers)]
    if separate_turns:
        turns = [
            (chunk, voice) for words, voice in turns
            for paragraph in re.split(r"\n[ \t]*\n", words)
            for chunk in split_sentences(paragraph, chunk_length)
        ]
        if len(turns) == 1 and num_speakers <= 1:
            turns = []
    if turns:
        samples, stops = [], []
        voice_context = None
        for index, (words, voice) in enumerate(turns):
            if max_steps < 1:
                break
            latent, probabilities = generate(
                bundle, diffusion, words, seed=seed, max_steps=max_steps, cfg=cfg, sigma=sigma,
                temperature=temperature, instruction=instruction, language=language,
                speakers=voice, separate_turns=False,
                min_steps=min_steps, stop_threshold=stop_threshold,
                chunk_length=chunk_length, _voice_context=voice_context,
            )
            if index == 0 and model.tiny and instruction.strip() and not speakers:
                # Keep the first generated voice fixed across this execution's chunks.
                voice_context = (latent, words + "\n")
            samples.append(latent)
            stops.extend(probabilities)
            max_steps -= latent.shape[-1] // model.patch_size
            if max_steps == 0 and (index + 1 < len(turns) or probabilities[-1] <= stop_threshold):
                logging.warning("Ming TTS reached max_steps before finishing the text. Increase max_steps.")
        return torch.cat(samples, dim=-1), stops
    speaker = torch.cat([s["embedding"] for s in speakers], dim=0) if speakers else None
    references = [(i, s) for i, s in enumerate(speakers, 1) if "reference_latents" in s]
    reference_latents = torch.cat([s["reference_latents"] for _, s in references], dim=-1) if references else None
    reference_text = "".join(
        f" speaker_{i}:{s['reference_text']}\n" if num_speakers > 1 else s["reference_text"]
        for i, s in references
    )
    if _voice_context is not None and not speakers:
        reference_latents, reference_text = _voice_context
    generator = torch.Generator(device=device).manual_seed(seed)
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    comfy.model_management.load_models_gpu([bundle.patcher, diffusion])
    amp = torch.autocast(device_type=device.type, dtype=dtype) if dtype != torch.float32 else nullcontext()
    try:
        with amp:
            latents = reference_latents.transpose(1, 2).to(device=device, dtype=dtype) if reference_latents is not None and reference_text else None
            speaker = speaker.to(device=device, dtype=dtype) if speaker is not None else None
            steps = 0 if latents is None else latents.shape[1] // model.patch_size
            ids, speaker_positions = prompt_ids(
                bundle.tokenizer.tokenizer, text, model.tiny, instruction, language,
                reference_text if latents is not None else "", steps,
                num_speakers=num_speakers,
            )
            backbone = model.model.model
            ids = torch.tensor([ids], device=device, dtype=torch.long)
            embeds = backbone.embed_tokens(ids, out_dtype=dtype)
            if speaker_positions:
                if model.tiny and instruction.strip() and not speakers:
                    embeds[:, speaker_positions] = 0
                else:
                    embedding = speaker if speaker is not None else torch.zeros((1, 192), device=device, dtype=dtype)
                    embeds[:, speaker_positions] = model.spk_head(embedding).unsqueeze(0)
            if steps:
                projected = model.linear_proj_audio(latents.reshape(-1, model.patch_size, model.latent_dim))
                embeds[:, -steps:] = projected.reshape(1, steps, -1)
            if not model.tiny:
                max_steps = min(max_steps, (512 - ids.shape[1]) // model.patch_size)
            if max_steps < 1:
                raise ValueError("The Ming 0.5B prompt fills its 512-token reference cache. Shorten the text or reference transcript.")
            cache = backbone.init_kv_cache(1, ids.shape[1] + max_steps, device, dtype)
            buffers = backbone.init_decode_buffers(1, device, dtype)
            aggregator_buffers = model.linear_proj_audio.init_decode_buffers(1, model.patch_size + 1, device, dtype)
            flow_model = diffusion.model.diffusion_model
            flow_buffers = flow_model.init_decode_buffers(2, 1 + model.history_size + model.patch_size, device, dtype)
            hidden, _, cache = backbone(None, embeds=embeds, past_key_values=cache)
            hidden = hidden[:, -1:].clone()
            del embeds
            history = torch.zeros((1, model.history_size, model.latent_dim), device=device, dtype=torch.float32 if model.tiny else dtype)
            if latents is not None:
                count = min(model.history_size, latents.shape[1])
                history[:, -count:] = latents[:, -count:]
            output = torch.empty((1, max_steps * model.patch_size, model.latent_dim), device=device, dtype=history.dtype)
            last_sample = torch.empty((1, model.patch_size, model.latent_dim), device=device, dtype=history.dtype)
            stop = torch.empty((), device=device, dtype=torch.float32)
            stop_probs = []
            minimum = (model.config["tts_config"]["minimum_decode_steps"] if model.tiny else 5) if min_steps is None else min_steps
            progress = comfy.utils.ProgressBar(max_steps)
            with comfy.utils.model_trange(max_steps, desc="Ming TTS", unit="patch", disable=not comfy.utils.PROGRESS_BAR_ENABLED) as terminal_progress:
                for step in range(max_steps):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    # RNG stays outside captured kernels, preserving both reference random streams.
                    shape = (1, model.patch_size, model.latent_dim)
                    if model.tiny:
                        noise = torch.randn((1, model.latent_dim, model.patch_size), generator=cpu_generator).transpose(1, 2).to(device)
                        sde = torch.stack([torch.randn(shape, device=device, dtype=torch.float32, generator=generator) for _ in range(10)])
                    else:
                        noise = torch.randn(shape, device=device, dtype=dtype, generator=generator)
                        sde = torch.randn((10, *shape), device=device, dtype=dtype, generator=generator)
                    scope = comfy.model_prefetch.malloc_graph_scope(device) if step else nullcontext()
                    with scope:
                        if step:
                            embeds = model.linear_proj_audio(last_sample, decode_buffers=aggregator_buffers)
                            history.copy_(torch.cat((history, last_sample), dim=1)[:, -model.history_size:])
                            hidden_out, _, cache = backbone(None, embeds=embeds, past_key_values=cache, decode_buffers=buffers)
                            hidden.copy_(hidden_out[:, -1:])
                            del embeds, hidden_out
                        sampled = sample_flow(flow_model, hidden, history, noise, sde, cfg, sigma, temperature, model.tiny, decode_buffers=flow_buffers)
                        output[:, step * model.patch_size:(step + 1) * model.patch_size].copy_(sampled)
                        last_sample.copy_(sampled)
                        stop.copy_(model.stop_head(hidden)[0, 0].softmax(-1)[1])
                        del sampled
                    probability = stop.item()
                    stop_probs.append(probability)
                    progress.update(1)
                    terminal_progress.update(1)
                    if step + 1 >= minimum and probability > stop_threshold:
                        break
            latents = output[:, :(step + 1) * model.patch_size].clone()
        return latents.transpose(1, 2).contiguous().cpu(), stop_probs
    finally:
        comfy.model_prefetch.cleanup_prefetch_queues()
