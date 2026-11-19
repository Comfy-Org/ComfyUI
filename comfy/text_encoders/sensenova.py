import os
import re

import torch
from tqdm import tqdm
from transformers import Qwen2Tokenizer

from comfy import sd1_clip
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.utils
from comfy.ldm.sensenova import model as sensenova_model
from comfy.ldm.sensenova.conditioning import (
    IMAGE_START_ID, IM_END_ID, MERGED_PATCH_SIZE,
    prepare_prefix_inputs, split_reference_batches,
)
from comfy.text_encoders.llama import generate_dtype


SYSTEM_MESSAGE = (
    "You are an image generation and editing assistant that accurately understands and executes "
    "user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you "
    "MUST start with a <think></think> block. Put all reasoning inside the block using plain text. "
    "DO NOT include any image tags. Keep it reasonable and directly useful for producing the final "
    "image.\n\n2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n"
    "Task Types:\n\nA. Text-to-Image Generation:\n"
    "- Generate a high-quality image based on the user's description.\n"
    "- Ensure visual clarity, semantic consistency, and completeness.\n"
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n"
    "B. Image Editing:\n"
    "- Use the provided image(s) as input or reference for modification or transformation.\n"
    "- The result can be an edited image or a new image based on the reference(s).\n"
    "- Preserve all unspecified attributes unless explicitly changed.\n\n"
    "General Rules:\n"
    "- For any visible text in the image, follow the language specified for the rendered text in "
    "the user's description, not the language of the prompt. If no language is specified, use the "
    "user's input language."
)

INTERLEAVE_SYSTEM_MESSAGE = (
    "You are a multimodal assistant capable of reasoning with both text and images. "
    "You support two modes:\n\n"
    "Think Mode: When reasoning is needed, you MUST start with a <think></think> block "
    "and place all reasoning inside it. You MUST interleave text with generated images "
    "using tags like <image1>, <image2>. Images can ONLY be generated between <think> and "
    "</think>, and may be referenced in the final answer.\n\n"
    "Non-Think Mode: When no reasoning is needed, directly provide the answer without reasoning. "
    "Do not use tags like <image1>, <image2>; present any images naturally alongside the text.\n\n"
    "After the think block, always provide a concise, user-facing final answer. "
    "The answer may include text, images, or both. Match the user's language in both reasoning "
    "and the final answer."
)


def _generation_prompt_parts(text, thinking, mode, system_prompt=None):
    if mode == "interleave":
        system = INTERLEAVE_SYSTEM_MESSAGE
        assistant = "" if thinking else "<think>\n\n</think>\n\n"
    else:
        system = SYSTEM_MESSAGE
        assistant = "<think>\n" if thinking else "<think>\n\n</think>\n\n<img>"
    system = system_prompt or system
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user",
        "\n" + text,
        f"<|im_end|>\n<|im_start|>assistant\n{assistant}",
    )


def build_generation_prompt(text, thinking=False):
    return "".join(_generation_prompt_parts(text, thinking, "image"))


def build_unconditional_prompt():
    return "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<img>"


def build_interleave_unconditional_prompt():
    return "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"


class SenseNovaQwen2Tokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        tokenizer = Qwen2Tokenizer.from_pretrained(*args, **kwargs)
        existing_special_tokens = [
            token
            for _, token in sorted(tokenizer.added_tokens_decoder.items())
            if token.special
        ]
        extra_tokens = [
            "<IMG_CONTEXT>",
            "<img>",
            "</img>",
            "<quad>",
            "</quad>",
            "<ref>",
            "</ref>",
            "<box>",
            "</box>",
            "<|action_start|>",
            "<|action_end|>",
            "<|plugin|>",
            "<|interpreter|>",
        ]
        extra_tokens.extend(f"<FAKE_PAD_{index}>" for index in range(254))
        tokenizer.add_special_tokens(
            {"additional_special_tokens": existing_special_tokens + extra_tokens}
        )
        return tokenizer


class SenseNovaQwenTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "qwen25_tokenizer"
        )
        super().__init__(
            tokenizer_path,
            pad_with_end=False,
            embedding_size=4096,
            embedding_key="sensenova_u15",
            tokenizer_class=SenseNovaQwen2Tokenizer,
            has_start_token=False,
            has_end_token=False,
            pad_to_max_length=False,
            max_length=99999999,
            min_length=1,
            pad_token=151643,
            tokenizer_data=tokenizer_data,
        )


class SenseNovaTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(
            embedding_directory=embedding_directory,
            tokenizer_data=tokenizer_data,
            name="sensenova_u15",
            tokenizer=SenseNovaQwenTokenizer,
        )

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        thinking = kwargs.pop("thinking", False)
        # Generate Text supplies skip_template; CLIP Text Encode uses the image prompt.
        mode = kwargs.pop("mode", "interleave" if "skip_template" in kwargs else "image")
        skip_template = kwargs.get("skip_template", False)
        if skip_template:
            parts = (text,)
        elif text or thinking or kwargs.get("system_prompt"):
            parts = _generation_prompt_parts(text, thinking, mode, kwargs.get("system_prompt"))
        else:
            prompt = (
                build_interleave_unconditional_prompt()
                if mode == "interleave"
                else build_unconditional_prompt()
            )
            parts = (prompt,)
        values = []
        word_id = 1
        for index, part in enumerate(parts):
            part = sd1_clip.unescape_important(sd1_clip.escape_important(part))
            # Preserve SDTokenizer's segmentation even without textual-inversion embeddings.
            segments = re.split(r"(?<=\s)(?={})".format(re.escape(self.sensenova_u15.embedding_identifier)), part)
            for segment_index, segment in enumerate(segments):
                word_id += int(segment_index > 0)
                # Only the template may introduce protocol IDs; user markers remain text.
                ids = self.sensenova_u15.tokenizer.encode(
                    segment, add_special_tokens=False, split_special_tokens=index == 1 or skip_template,
                )
                values.extend((token, 1.0, word_id) if return_word_ids else (token, 1.0) for token in ids)
        tokens = {"sensenova_u15": [values], "sensenova_prompt": text}
        images = kwargs.get("images") or []
        if kwargs.get("image") is not None:
            images = [kwargs["image"], *images]
        if images:
            tokens["reference_latents"] = images
        return tokens


def clip_parameters(state_dict, prefix=""):
    return {
        "has_lm_head": prefix + "language_model.lm_head.weight" in state_dict,
        "dtype_model": state_dict[prefix + "language_model.model.norm.weight"].dtype,
        "quantization_metadata": comfy.utils.detect_layer_quantization(state_dict, prefix),
    }


class SenseNovaTextEncoder(sensenova_model.SenseNovaTextModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, has_lm_head=False, dtype_model=None, quantization_metadata=None):
        dtype = dtype_model or dtype
        operations = model_options.get("custom_operations")
        if operations is None:
            operations = comfy.ops.manual_cast if quantization_metadata is None else comfy.ops.mixed_precision_ops(quantization_metadata, dtype, full_precision_mm=True)
        super().__init__(has_lm_head=has_lm_head, device=device, dtype=dtype, operations=operations)
        self.dtypes = {dtype}
        self.execution_device = device

    def get_dynamic_vram__units(self):
        backbone = self.language_model.model
        units = [*backbone.layers, backbone.norm]
        if self.has_lm_head:
            units.append(self.language_model.lm_head)
        return units, [backbone.embed_tokens]

    def reset_clip_options(self):
        self.execution_device = None

    def set_clip_options(self, options):
        self.execution_device = options.get("execution_device", self.execution_device)

    def load_sd(self, sd):
        return self.load_state_dict(sd, strict=False, assign=getattr(self, "can_assign_sd", False))

    def _input_ids(self, tokens):
        if "text_input_ids" in tokens:
            return tokens["text_input_ids"]
        return torch.tensor([[int(value[0]) for value in batch] for batch in tokens["sensenova_u15"]], dtype=torch.long)

    def _prefix_arguments(self, tokens, dtype):
        input_ids = self._input_ids(tokens)
        thinking_start = tokens.get("sensenova_thinking_start")
        text_suffix = None
        if thinking_start is not None:
            text_suffix = input_ids[:, thinking_start:].to(self.execution_device)
            input_ids = input_ids[:, :thinking_start]
        input_ids, references, indexes, mask = prepare_prefix_inputs(
            input_ids, tokens.get("reference_latents"), MERGED_PATCH_SIZE, dtype,
            image_only=tokens.get("image_only", False),
            expand_images=not tokens.get("sensenova_interleave_expanded", False),
            generated_start=tokens.get("sensenova_interleave_start"),
        )
        return dict(
            text_input_ids=input_ids.to(self.execution_device),
            reference_images=[image.to(device=self.execution_device, dtype=dtype) for image in references] if references else None,
            prefix_indexes=indexes.to(self.execution_device) if indexes is not None else None,
            prefix_mask=mask.to(self.execution_device) if mask is not None else None,
            text_suffix=text_suffix,
        )

    def encode_token_weights(self, tokens):
        dtype = generate_dtype(self.execution_device)
        try:
            keys, values, time = self.preprocess_prefix(
                **self._prefix_arguments(tokens, dtype), compute_dtype=dtype,
                transformer_options={"prefetch_dynamic_vbars": True},
            )
            device = comfy.model_management.intermediate_device()
            input_ids = self._input_ids(tokens).to(device)
            extra = {
                "text_input_ids": input_ids,
                "prefix_keys": [key.to(device) for key in keys],
                "prefix_values": [value.to(device) for value in values],
                "prefix_time": time.to(device),
                "sensenova_reference_count": len(split_reference_batches(tokens.get("reference_latents") or [])),
            }
            if "sensenova_prompt" in tokens:
                extra["sensenova_prompt"] = tokens["sensenova_prompt"]
            if "reference_latents" in tokens:
                extra["reference_latents"] = tokens["reference_latents"]
            return input_ids.unsqueeze(-1).float(), None, extra
        finally:
            comfy.model_prefetch.cleanup_prefetch_queues()

    def generate(self, tokens, do_sample=True, max_length=256, temperature=1.0, top_k=50, top_p=0.95, min_p=0.0, repetition_penalty=1.0, seed=None, presence_penalty=0.0, mtp=True):
        dtype = generate_dtype(self.execution_device)
        prefix = self._prefix_arguments(tokens, dtype)
        prefix.pop("text_suffix")
        progress = comfy.utils.ProgressBar(max_length)
        try:
            with tqdm(total=max_length, desc="Generating tokens", unit="token") as console:
                def update_progress(value):
                    console.update(value - console.n)
                    progress.update_absolute(value)

                progress.update_absolute(0)
                token_ids, stop = self.generate_text_tokens(
                    **prefix, max_tokens=max_length, stop_token_ids=tokens.get("stop_token_ids", (IMAGE_START_ID,)),
                    do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p,
                    repetition_penalty=repetition_penalty, presence_penalty=presence_penalty, seed=seed,
                    compute_dtype=dtype, transformer_options={"prefetch_dynamic_vbars": True},
                    progress=update_progress,
                    interrupt=comfy.model_management.throw_exception_if_processing_interrupted,
                )
                console.total = console.n
                progress.update_absolute(console.n, console.n)
            if stop == IM_END_ID:
                token_ids.append(stop)
            return token_ids
        finally:
            comfy.model_prefetch.cleanup_prefetch_queues()
