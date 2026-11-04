from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch
from typing_extensions import override

import comfy.model_management
import comfy.model_prefetch
import comfy.utils
from comfy.ldm.sensenova.conditioning import (
    IMAGE_CONTEXT_ID,
    IMAGE_END_ID,
    IMAGE_START_ID,
    IM_END_ID,
    THINK_END_ID,
    THINK_START_ID,
    THINK_SUFFIX_IDS,
)
from comfy.ldm.sensenova.interleave import (
    append_image_to_prefix,
    expand_interleave_inputs,
    prefix_arguments,
)
from comfy.ldm.sensenova.model import HEAD_DIM, NUM_KV_HEADS, NUM_LAYERS
from comfy.ldm.sensenova.sampling import SenseNovaModelSampling
from comfy_api.latest import ComfyExtension, io, ui
from comfy_execution.graph_utils import ExecutionBlocker


class SenseNovaSamplingOptions(io.ComfyNode):
    """Configure SenseNova flow sampling parameters on a model patcher."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaSamplingOptions",
            display_name="SenseNova Sampling Options",
            category="model/patch/sensenova",
            description="Set the SenseNova flow shift.",
            inputs=[
                io.Model.Input(id="model"),
                io.Float.Input(id="shift", default=3.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def validate_inputs(cls, shift):
        if shift is not None and (not math.isfinite(shift) or shift <= 0):
            return "shift must be finite and greater than zero."
        return True

    @classmethod
    def execute(cls, *, model, shift: float) -> io.NodeOutput:
        validation = cls.validate_inputs(shift)
        if validation is not True:
            raise ValueError(validation)
        patched = model.clone()
        model_sampling = SenseNovaModelSampling(patched.model.model_config)
        model_sampling.set_parameters(shift=shift)
        patched.add_object_patch("model_sampling", model_sampling)
        return io.NodeOutput(patched)


class SenseNovaTextEncode(io.ComfyNode):
    """Encode a SenseNova image or interleave prompt."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaTextEncode",
            display_name="SenseNova Text Encode",
            category="model/conditioning/sensenova",
            description=(
                "Encode a SenseNova prompt. Use SenseNova Generate before the "
                "KSampler when image-generation thinking is required."
            ),
            inputs=[
                io.Clip.Input(id="clip"),
                io.String.Input(id="text", multiline=True, dynamic_prompts=True),
                io.Combo.Input(
                    id="mode",
                    options=["image", "interleave"],
                    default="image",
                    tooltip="For interleave generation, select interleave on both positive and negative encoders.",
                ),
                io.Boolean.Input(
                    id="thinking",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "Enable thinking on the positive prompt and pass it through "
                        "SenseNova Generate before KSampler. Keep false on the "
                        "negative prompt. Also controls interleave thinking."
                    ),
                ),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(
        cls,
        *,
        clip,
        text: str,
        thinking: bool,
        mode: str = "image",
    ) -> io.NodeOutput:
        tokenize_options = {"thinking": thinking}
        if mode == "interleave":
            tokenize_options["mode"] = mode
        tokens = clip.tokenize(text, **tokenize_options)
        metadata = {"sensenova_thinking": thinking}
        if mode == "interleave":
            metadata["sensenova_interleave"] = True
        conditioning = clip.encode_from_tokens_scheduled(
            tokens,
            add_dict=metadata,
        )
        return io.NodeOutput(conditioning)


class SenseNovaGenerate(io.ComfyNode):
    """Generate SenseNova image reasoning before standard image sampling."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaGenerate",
            display_name="SenseNova Generate",
            category="model/sampling/sensenova",
            description=(
                "Run SenseNova thinking when enabled; otherwise pass conditioning "
                "through unchanged. In interleave mode, stop at the next image "
                "request or EOS so an external KSampler can produce the image."
            ),
            inputs=[
                io.Model.Input("model", optional=True),
                io.Clip.Input("clip", optional=True),
                io.Conditioning.Input("conditioning"),
                io.Conditioning.Input(
                    "negative",
                    optional=True,
                    tooltip=(
                        "For interleave, connect a negative encoded in interleave mode. "
                        "In image mode, this input is passed through without text generation; "
                        "keep thinking disabled on its encoder."
                    ),
                ),
                io.Latent.Input(
                    "samples",
                    optional=True,
                    tooltip="Previous KSampler pixel latent, required after an image request.",
                ),
                io.Int.Input(
                    "max_think_tokens",
                    default=1024,
                    min=1,
                    max=8192,
                    advanced=True,
                    tooltip="Maximum generated tokens for thinking or one interleave text stage.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.String.Output(
                    display_name="generated_text",
                    tooltip="Text generated by this call, without protocol think wrappers. Use SenseNova Decode for the complete interleave result.",
                ),
                io.Conditioning.Output(display_name="negative"),
                io.Boolean.Output(display_name="image_requested"),
            ],
        )

    @classmethod
    def execute(
        cls,
        *,
        model=None,
        clip=None,
        conditioning,
        max_think_tokens,
        negative=None,
        samples=None,
    ) -> io.NodeOutput:
        if not conditioning:
            return io.NodeOutput(
                conditioning, "", negative or [], False, ui=ui.PreviewText("")
            )
        metadata = conditioning[0][1]
        if metadata.get("sensenova_interleave"):
            return cls._execute_interleave(
                model=model,
                clip=clip,
                positive=conditioning,
                negative=negative,
                samples=samples,
                max_text_tokens=max_think_tokens,
            )
        if not metadata.get("sensenova_thinking", False):
            return io.NodeOutput(
                conditioning, "", negative or [], True, ui=ui.PreviewText("")
            )
        if len(conditioning) != 1:
            raise ValueError(
                "SenseNova Generate requires one positive conditioning entry "
                "when thinking is enabled."
            )
        if metadata.get("text_input_ids") is None:
            raise ValueError("SenseNova Generate requires SenseNova text conditioning.")
        if model is None or clip is None:
            raise ValueError("Connect MODEL and CLIP to SenseNova Generate for text generation.")

        with _generation_context(model, metadata, max_think_tokens) as (diffusion_model, prefix_args, options):
            token_ids = diffusion_model.generate_thinking_tokens(
                *prefix_args,
                max_think_tokens=max_think_tokens,
                **options,
            )

        text = _decode_generated_text(clip, token_ids[:-1])

        input_ids = metadata["text_input_ids"]
        suffix = input_ids.new_tensor([*token_ids, *THINK_SUFFIX_IDS]).unsqueeze(0)
        updated_metadata = {
            **metadata,
            "text_input_ids": torch.cat((input_ids, suffix), dim=1),
            "sensenova_thinking_start": input_ids.shape[1],
            "sensenova_thinking": False,
        }
        updated_conditioning = [[conditioning[0][0], updated_metadata]]

        return io.NodeOutput(
            updated_conditioning,
            text,
            negative or [],
            True,
            ui=ui.PreviewText(text),
        )

    @classmethod
    def _execute_interleave(
        cls,
        *,
        model,
        clip,
        positive,
        negative,
        samples,
        max_text_tokens,
    ) -> io.NodeOutput:
        """Generate one interleave text stage for external KSampler orchestration."""

        if negative is None or len(positive) != 1 or len(negative) != 1:
            raise ValueError(
                "SenseNova interleave requires one positive and one negative conditioning entry."
            )
        if positive[0][1].get("sensenova_interleave_finished"):
            return io.NodeOutput(
                positive, "", negative, False, ui=ui.PreviewText("")
            )
        if model is None or clip is None:
            raise ValueError("Connect MODEL and CLIP to SenseNova Generate for text generation.")
        positive_metadata = expand_interleave_metadata(positive[0][1], image_only=False)
        negative_metadata = expand_interleave_metadata(negative[0][1], image_only=True)
        pending_image = bool(
            positive_metadata.get("sensenova_interleave_pending_image")
        )
        if pending_image != bool(
            negative_metadata.get("sensenova_interleave_pending_image")
        ):
            raise ValueError("SenseNova interleave conditioning histories are out of sync.")
        latent_samples = samples.get("samples") if isinstance(samples, dict) else None
        if pending_image:
            if latent_samples is None:
                raise ValueError(
                    "SenseNova interleave is waiting for samples from the preceding KSampler."
                )
            positive_metadata = append_interleave_image(
                positive_metadata, latent_samples
            )
            negative_metadata = append_interleave_image(
                negative_metadata, latent_samples
            )
        elif latent_samples is not None:
            raise ValueError(
                "SenseNova interleave received samples before requesting an image."
            )

        with _generation_context(model, positive_metadata, max_text_tokens) as (diffusion_model, prefix_args, options):
            event = generate_text_event(
                diffusion_model,
                prefix_args,
                max_text_tokens=max_text_tokens,
                **options,
            )

        positive_metadata = append_interleave_tokens(
            positive_metadata, event.token_ids, event.image_requested
        )
        positive_metadata["sensenova_interleave_finished"] = event.stop_reason == "eos"
        if event.image_requested:
            negative_ids = negative_metadata["text_input_ids"]
            image_start = [] if negative_ids.shape[1] and negative_ids[0, -1] == IMAGE_START_ID else [IMAGE_START_ID]
            negative_metadata = append_interleave_tokens(
                negative_metadata, image_start, True
            )
        history = positive_metadata["text_input_ids"][0, positive_metadata["sensenova_interleave_start"]:].tolist()
        think_start, think_end = _interleave_thinking_span(clip, history)
        stage_start = len(history) - len(event.token_ids)
        text = _decode_generated_text(clip, [
            token for index, token in enumerate(event.token_ids, start=stage_start)
            if index not in (think_start, think_end)
        ])
        updated_positive = [[positive[0][0], positive_metadata]]
        updated_negative = [[negative[0][0], negative_metadata]]
        return io.NodeOutput(
            updated_positive,
            text,
            updated_negative,
            event.image_requested,
            ui=ui.PreviewText(text),
        )


@dataclass
class InterleaveTextEvent:
    """One text-generation stage ending at an image request or stop condition."""

    token_ids: list[int]
    stop_reason: str

    @property
    def image_requested(self):
        return self.stop_reason == "image"


def generate_text_event(
    model,
    prefix_args,
    max_text_tokens,
    transformer_options=None,
    progress=None,
    interrupt=None,
    compute_dtype=None,
):
    """Decode one interleave text stage without carrying KV state across nodes."""

    token_ids, stop_token_id = model.generate_text_tokens(
        *prefix_args,
        max_tokens=max_text_tokens,
        stop_token_ids=(IMAGE_START_ID,),
        transformer_options=transformer_options,
        progress=progress,
        interrupt=interrupt,
        compute_dtype=compute_dtype,
    )
    stop_reason = {
        IM_END_ID: "eos",
        IMAGE_START_ID: "image",
    }.get(stop_token_id, "max_text_tokens")
    return InterleaveTextEvent(token_ids, stop_reason)


def expand_interleave_metadata(metadata, image_only):
    """Create a reconstructable token/image history for interleave stages."""

    expanded = dict(metadata)
    if expanded.get("sensenova_interleave_expanded"):
        return expanded

    input_ids, references = expand_interleave_inputs(
        expanded["text_input_ids"], expanded.get("reference_latents"), image_only
    )
    expanded.update(
        {
            "text_input_ids": input_ids,
            "reference_latents": references,
            "sensenova_interleave_expanded": True,
            "sensenova_interleave_pending_image": False,
            "sensenova_interleave_start": input_ids.shape[1],
            "sensenova_interleave_input_images": len(references),
        }
    )
    return expanded


def append_interleave_tokens(metadata, token_ids, image_requested):
    """Append generated text tokens to an expanded interleave history."""

    updated = dict(metadata)
    if token_ids:
        suffix = updated["text_input_ids"].new_tensor(token_ids).unsqueeze(0)
        updated["text_input_ids"] = torch.cat(
            (updated["text_input_ids"], suffix), dim=1
        )
    updated["sensenova_interleave_pending_image"] = image_requested
    return updated


def append_interleave_image(metadata, latent_samples):
    """Append one sampled image to an expanded interleave history."""

    if not metadata.get("sensenova_interleave_pending_image"):
        raise ValueError("SenseNova interleave received an image without an image request.")
    input_ids, image = append_image_to_prefix(metadata["text_input_ids"], latent_samples)
    updated = dict(metadata)
    updated["text_input_ids"] = input_ids
    updated["reference_latents"] = [
        *(metadata.get("reference_latents") or []),
        image,
    ]
    updated["sensenova_interleave_pending_image"] = False
    return updated


@contextmanager
def _generation_context(model, metadata, max_tokens):
    prefix_shape = model.model.extra_conds_shapes(**metadata)["prefix_mask"]
    prefix_length = prefix_shape[-1]
    dtype_size = comfy.model_management.dtype_size(
        model.get_model_object("manual_cast_dtype") or model.model.get_dtype()
    )
    kv_bytes = 2 * NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM * (prefix_length + max_tokens) * dtype_size
    mask_bytes = math.prod(prefix_shape) * dtype_size
    comfy.model_management.load_models_gpu([model], memory_required=kv_bytes + mask_bytes)
    with comfy.model_management.cuda_device_context(model.load_device):
        try:
            model.pre_run()
            compute_dtype = model.model.get_dtype_inference()
            prefix_args = prefix_arguments(
                metadata, model.load_device, compute_dtype
            )
            transformer_options = model.model_options.get("transformer_options", {}).copy()
            transformer_options["prefetch_dynamic_vbars"] = model.is_dynamic()
            options = {
                "compute_dtype": compute_dtype,
                "transformer_options": transformer_options,
                "progress": comfy.utils.ProgressBar(max_tokens).update_absolute,
                "interrupt": comfy.model_management.throw_exception_if_processing_interrupted,
            }
            yield model.model.diffusion_model, prefix_args, options
        finally:
            try:
                comfy.model_prefetch.cleanup_prefetch_queues()
            finally:
                model.cleanup()


def _decode_generated_text(clip, token_ids):
    text_tokens = [token for token in token_ids if token != IMAGE_START_ID]
    return clip.decode(text_tokens, skip_special_tokens=True) if text_tokens else ""


def _interleave_thinking_span(clip, token_ids):
    """Only a leading think block is protocol; later think tags are answer text."""

    start = next((index for index, token in enumerate(token_ids) if token == THINK_START_ID), len(token_ids))
    if start == len(token_ids) or clip.decode(token_ids[:start], skip_special_tokens=False).strip():
        return len(token_ids), len(token_ids)
    end = next((index for index in range(start + 1, len(token_ids)) if token_ids[index] == THINK_END_ID), len(token_ids))
    return start, end


def _decode_interleave_text(clip, token_ids):
    start, end = _interleave_thinking_span(clip, token_ids)
    has_thinking = start < len(token_ids)
    thinking_ids = []
    if has_thinking:
        thinking_ids = token_ids[start + 1:end]
        token_ids = token_ids[end + 1:]

    outputs = []
    for ids, is_thinking in ((token_ids, False), (thinking_ids, True)):
        parts = []
        text_tokens = []
        for token_id in ids:
            if token_id == IMAGE_START_ID:
                parts.append(clip.decode(text_tokens, skip_special_tokens=True))
                text_tokens = []
                if not is_thinking:
                    parts.append("<image>")
            elif token_id not in (IMAGE_CONTEXT_ID, IMAGE_END_ID):
                text_tokens.append(token_id)
        parts.append(clip.decode(text_tokens, skip_special_tokens=True))
        outputs.append("\n\n".join(part.strip() for part in parts if part.strip()) if is_thinking else "".join(parts))
    return outputs[0], outputs[1]


class SenseNovaDecode(io.ComfyNode):
    """Decode the generated portion of a SenseNova interleave history."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaDecode",
            display_name="SenseNova Decode",
            category="model/sampling/sensenova",
            description=(
                "Decode the final conditioning from SenseNova Generate into text, "
                "an image list, and thinking. With loops, use End Loop with "
                "accumulate disabled. Input reference images are excluded. "
                "Image consumers are skipped when no images were generated."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Conditioning.Input("conditioning"),
                io.Latent.Input(
                    "samples",
                    optional=True,
                    tooltip="Last KSampler pixel latent, if the loop ended on an image request.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="text"),
                io.Image.Output(display_name="images", is_output_list=True),
                io.String.Output(display_name="thinking"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        conditioning,
        samples=None,
    ) -> io.NodeOutput:
        if len(conditioning) != 1 or "sensenova_interleave_start" not in conditioning[0][1]:
            raise ValueError(
                "SenseNova Decode requires interleave conditioning from SenseNova Generate."
            )
        metadata = conditioning[0][1]
        if metadata["sensenova_interleave_pending_image"]:
            if samples is None:
                raise ValueError(
                    "SenseNova Decode is waiting for samples from the last KSampler. "
                    "Connect its LATENT output to samples."
                )
            metadata = append_interleave_image(metadata, samples["samples"])
        token_ids = metadata["text_input_ids"][0, metadata["sensenova_interleave_start"]:].tolist()
        text, thinking = _decode_interleave_text(clip, token_ids)
        images = [
            reference.clamp(0.0, 1.0)
            for reference in metadata["reference_latents"][metadata["sensenova_interleave_input_images"]:]
        ]
        return io.NodeOutput(text, images or [ExecutionBlocker(None)], thinking)


class SenseNovaExtension(ComfyExtension):
    """Register the native SenseNova node collection."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SenseNovaTextEncode,
            SenseNovaGenerate,
            SenseNovaSamplingOptions,
            SenseNovaDecode,
        ]


async def comfy_entrypoint() -> SenseNovaExtension:
    """Create the native SenseNova node extension."""

    return SenseNovaExtension()
