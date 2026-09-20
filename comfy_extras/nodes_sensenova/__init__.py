import re

import torch
from typing_extensions import override

import comfy.model_management
import comfy.sample
import comfy.utils
import latent_preview
from comfy.ldm.sensenova.interleave import (
    SenseNovaInterleaveSession,
    build_interleave_result,
    interleave_result_to_markdown,
    live_conditioning,
    prefix_arguments,
)
from comfy.ldm.sensenova.sampling import SenseNovaModelSampling
from comfy_api.latest import ComfyExtension, io, ui


InterleaveResultIO = io.Custom("SENSENOVA_INTERLEAVE_RESULT")
WEB_DIRECTORY = "./web"


def interleave_output_samples(result, latent_samples):
    """Return generated interleave images as a ComfyUI latent batch."""

    if not result.images:
        return latent_samples
    return torch.cat(result.images).to(
        device=comfy.model_management.intermediate_device(),
        dtype=comfy.model_management.intermediate_dtype(),
    )


def run_interleave(
    model,
    clip,
    positive,
    negative,
    noise_seed,
    cfg,
    sampler,
    sigmas,
    latent,
    max_text_tokens,
    max_images=1,
):
    """Run one SenseNova interleave session using standard ComfyUI sampling."""

    latent = latent.copy()
    latent_samples = comfy.sample.fix_empty_latent_channels(
        model,
        latent["samples"],
        latent.get("downscale_ratio_spacial"),
        latent.get("downscale_ratio_temporal"),
    )
    if latent_samples.shape[0] != 1:
        raise ValueError("SenseNova interleave requires a single latent image.")
    latent["samples"] = latent_samples

    positive_data = positive[0][1]
    negative_data = negative[0][1]
    if not positive_data.get("sensenova_interleave") or not negative_data.get(
        "sensenova_interleave"
    ):
        raise ValueError(
            "SenseNova interleave requires positive and negative conditioning "
            "encoded with mode=interleave."
        )

    comfy.model_management.load_models_gpu([model])
    device = model.load_device
    transformer_options = model.model_options.get("transformer_options", {}).copy()
    model.pre_run()
    try:
        diffusion_model = model.model.diffusion_model
        session = SenseNovaInterleaveSession(
            diffusion_model,
            positive_prefix=prefix_arguments(
                positive_data, device, diffusion_model.dtype, image_only=False
            ),
            negative_prefix=prefix_arguments(
                negative_data, device, diffusion_model.dtype, image_only=True
            ),
            decode_tokens=lambda values: clip.tokenizer.sensenova_u15.tokenizer.decode(
                values, skip_special_tokens=True
            ),
            transformer_options=transformer_options,
        )
        image_index = 0

        def sample_image(positive_prefix, negative_prefix):
            nonlocal image_index
            noise = comfy.sample.prepare_noise(
                latent_samples, noise_seed, [image_index]
            )
            callback = latent_preview.prepare_callback(
                model, sigmas.shape[-1] - 1, {}
            )
            samples = comfy.sample.sample_custom(
                model,
                noise,
                cfg,
                sampler,
                sigmas,
                live_conditioning(positive_prefix),
                live_conditioning(negative_prefix),
                latent_samples,
                noise_mask=latent.get("noise_mask"),
                callback=callback,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                seed=noise_seed,
            )
            image_index += 1
            comfy.model_management.load_models_gpu([model])
            model.pre_run()
            return samples.to(device=device, dtype=diffusion_model.dtype)

        progress = comfy.utils.ProgressBar(max_text_tokens)
        result = session.generate(
            sample_image,
            max_text_tokens=max_text_tokens,
            max_images=max_images,
            progress=progress.update_absolute,
            interrupt=comfy.model_management.throw_exception_if_processing_interrupted,
        )
    finally:
        model.cleanup()

    latent.pop("downscale_ratio_spacial", None)
    latent.pop("downscale_ratio_temporal", None)
    latent["samples"] = interleave_output_samples(result, latent_samples)
    return latent, result.text, build_interleave_result(result)


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
                io.Float.Input(id="shift", default=3.0, min=0.01, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, *, model, shift: float) -> io.NodeOutput:
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
                ),
                io.Boolean.Input(
                    id="thinking",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "Set true for T2I/Edit before SenseNova Generate; it also "
                        "controls interleave thinking."
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
                "through unchanged. Expose reasoning text for Preview as Text and "
                "connect the conditioning output to the KSampler positive input."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Conditioning.Input("conditioning"),
                io.Int.Input(
                    "max_think_tokens", default=1024, min=1, max=8192, advanced=True
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.String.Output(display_name="thinking"),
            ],
        )

    @classmethod
    def execute(
        cls, *, model, clip, conditioning, max_think_tokens
    ) -> io.NodeOutput:
        if not conditioning:
            return io.NodeOutput(conditioning, "", ui=ui.PreviewText(""))
        metadata = conditioning[0][1]
        if metadata.get("sensenova_interleave"):
            raise ValueError(
                "SenseNova Generate is for T2I/Edit thinking; use SenseNova Interleave "
                "for interleaved text and image generation."
            )
        if not metadata.get("sensenova_thinking", False):
            return io.NodeOutput(conditioning, "", ui=ui.PreviewText(""))
        if len(conditioning) != 1:
            raise ValueError(
                "SenseNova Generate requires one positive conditioning entry "
                "when thinking is enabled."
            )
        if metadata.get("text_input_ids") is None:
            raise ValueError("SenseNova Generate requires SenseNova text conditioning.")

        diffusion_model = model.model.diffusion_model
        device = model.load_device
        dtype = diffusion_model.dtype
        transformer_options = model.model_options.get(
            "transformer_options", {}
        ).copy()
        comfy.model_management.load_models_gpu([model])
        model.pre_run()
        try:
            prefix_args = prefix_arguments(
                metadata, device, dtype, image_only=False
            )
            progress = comfy.utils.ProgressBar(max_think_tokens)
            prefix_keys, prefix_values, prefix_time, token_ids = (
                diffusion_model.preprocess_thinking_prefix_with_tokens(
                    *prefix_args,
                    max_think_tokens=max_think_tokens,
                    transformer_options=transformer_options,
                    progress=progress.update_absolute,
                    interrupt=comfy.model_management.throw_exception_if_processing_interrupted,
                )
            )
        finally:
            model.cleanup()

        thinking = _format_thinking_text(
            clip.decode(token_ids, skip_special_tokens=True)
        )
        if not thinking:
            thinking = "SenseNova thinking completed without visible text."

        updated_conditioning = []
        for value, original_metadata in conditioning:
            updated_metadata = dict(original_metadata)
            # The KV prefix is now complete. Do not let KSampler fall back to
            # rebuilding the text/reference prefix or run thinking again.
            for key in (
                "text_input_ids",
                "reference_latents",
                "prefix_indexes",
                "prefix_mask",
                "sensenova_thinking",
                "sensenova_max_think_tokens",
                "sensenova_thinking_result",
            ):
                updated_metadata.pop(key, None)
            updated_metadata.update(
                {
                    "prefix_keys": prefix_keys,
                    "prefix_values": prefix_values,
                    "prefix_time": prefix_time,
                }
            )
            updated_conditioning.append([value, updated_metadata])

        return io.NodeOutput(
            updated_conditioning,
            thinking,
            ui=ui.PreviewText(thinking),
        )


def _format_thinking_text(text: str) -> str:
    """Return a complete SenseNova thinking block for text previews.

    The opening marker is part of the input prompt, while the decoder returns
    the generated body and the closing marker. Normalize both boundaries here
    so downstream text previews do not need to infer token-level framing.
    """

    text = text.strip()
    if not text:
        return ""
    if not text.startswith("<think>"):
        text = f"<think>\n{text}"
    if not text.endswith("</think>"):
        text = f"{text}\n</think>"
    return text


class SenseNovaInterleave(io.ComfyNode):
    """Generate interleaved SenseNova text and image output."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaInterleave",
            display_name="SenseNova Interleave",
            category="model/sampling/sensenova",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Int.Input(
                    "noise_seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Float.Input(
                    "cfg", default=4.0, min=0.0, max=100.0, step=0.1, round=0.01
                ),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent_image"),
                io.Int.Input(
                    "max_text_tokens", default=1024, min=1, max=8192, advanced=True
                ),
                io.Int.Input("max_images", default=4, min=1, max=10, advanced=True),
            ],
            outputs=[
                io.Latent.Output(display_name="samples"),
                io.String.Output(display_name="text"),
                InterleaveResultIO.Output(display_name="interleave_result"),
            ],
        )

    @classmethod
    def execute(
        cls,
        *,
        model,
        clip,
        positive,
        negative,
        noise_seed,
        cfg,
        sampler,
        sigmas,
        latent_image,
        max_text_tokens,
        max_images,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            *run_interleave(
                model,
                clip,
                positive,
                negative,
                noise_seed,
                cfg,
                sampler,
                sigmas,
                latent_image,
                max_text_tokens,
                max_images,
            )
        )


def _save_preview_images(images):
    if images is None or images.shape[0] == 0:
        return []
    return [dict(value) for value in ui.PreviewImage(images).as_dict()["images"]]


_IMAGE_REFERENCE_PATTERN = re.compile(r"<image(\d+)>")


def _interleave_preview_parts(interleave_result, include_think):
    """Resolve or hide final-answer image references for preview rendering."""

    parts = interleave_result.get("parts", [])
    image_parts = {
        int(part.get("index", 0)): part
        for part in parts
        if part.get("type") == "image"
    }
    referenced_images = set()
    resolved_text_parts = {}
    for part_index, part in enumerate(parts):
        if part.get("type") != "text":
            continue
        text = str(part.get("text", ""))
        cursor = 0
        resolved_parts = []
        for match in _IMAGE_REFERENCE_PATTERN.finditer(text):
            text_before_reference = text[cursor : match.start()].strip()
            if text_before_reference:
                resolved_parts.append(
                    {"type": "text", "text": text_before_reference}
                )
            image_index = int(match.group(1)) - 1
            if not include_think and image_index in image_parts:
                resolved_parts.append(image_parts[image_index])
                referenced_images.add(image_index)
            cursor = match.end()
        remaining_text = text[cursor:].strip()
        if remaining_text:
            resolved_parts.append({"type": "text", "text": remaining_text})
        resolved_text_parts[part_index] = resolved_parts

    display_parts = []
    for part_index, part in enumerate(parts):
        part_type = part.get("type")
        if part_type == "think":
            if include_think:
                display_parts.append(part)
            continue
        if part_type == "image":
            if int(part.get("index", 0)) not in referenced_images:
                display_parts.append(part)
            continue
        if part_type != "text":
            continue
        display_parts.extend(resolved_text_parts[part_index])
    return display_parts


class SenseNovaInterleavePreview(io.ComfyNode):
    """Display interleaved text and images with optional thinking details."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaInterleavePreview",
            display_name="SenseNova Interleave Preview",
            category="model/sampling/sensenova",
            is_output_node=True,
            inputs=[
                InterleaveResultIO.Input("interleave_result"),
                io.Boolean.Input("include_think", default=False),
                io.Image.Input("images", optional=True),
            ],
            outputs=[io.String.Output(display_name="markdown")],
        )

    @classmethod
    def execute(cls, *, interleave_result, include_think, images=None) -> io.NodeOutput:
        display_parts = _interleave_preview_parts(interleave_result, include_think)
        markdown = interleave_result_to_markdown(
            {"parts": display_parts}, include_think=include_think
        )
        saved_images = _save_preview_images(images)
        parts_payload = []
        for part in display_parts:
            part_type = part.get("type")
            if part_type in ("text", "think"):
                text = str(part.get("text", "")).strip()
                if text:
                    parts_payload.append({"type": part_type, "text": text})
            elif part_type == "image":
                index = int(part.get("index", 0))
                image = saved_images[index] if index < len(saved_images) else None
                if image is None:
                    parts_payload.append(
                        {"type": "image", "index": index, "missing": True}
                    )
                else:
                    parts_payload.append(
                        {
                            "type": "image",
                            "index": index,
                            "filename": image.get("filename", ""),
                            "subfolder": image.get("subfolder", ""),
                            "image_type": image.get("type", "temp"),
                        }
                    )
        return io.NodeOutput(
            markdown,
            ui={"text": [markdown], "parts": parts_payload},
        )


class SenseNovaExtension(ComfyExtension):
    """Register the native SenseNova node collection."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SenseNovaTextEncode,
            SenseNovaGenerate,
            SenseNovaSamplingOptions,
            SenseNovaInterleave,
            SenseNovaInterleavePreview,
        ]


async def comfy_entrypoint() -> SenseNovaExtension:
    """Create the native SenseNova node extension."""

    return SenseNovaExtension()
