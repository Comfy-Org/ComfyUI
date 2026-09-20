import torch
from typing_extensions import override

import comfy.model_management
import comfy.sample
import comfy.utils
import latent_preview
from comfy.ldm.sensenova.interleave import (
    SenseNovaInterleaveSession,
    append_interleave_image,
    append_interleave_tokens,
    build_interleave_result,
    expand_interleave_metadata,
    generate_text_event,
    live_conditioning,
    prefix_arguments,
    split_interleave_text,
)
from comfy.ldm.sensenova.sampling import SenseNovaModelSampling
from comfy_api.latest import ComfyExtension, io, ui


InterleaveResultIO = io.Custom("SENSENOVA_INTERLEAVE_RESULT")


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
                "through unchanged. In interleave mode, stop at the next image "
                "request or EOS so an external KSampler can produce the image."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Conditioning.Input("conditioning"),
                io.Conditioning.Input("negative", optional=True),
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
                    display_name="thinking",
                    tooltip="Thinking text for image mode, or the current text stage for interleave mode.",
                ),
                io.Conditioning.Output(display_name="negative"),
                io.Boolean.Output(display_name="image_requested"),
            ],
        )

    @classmethod
    def execute(
        cls,
        *,
        model,
        clip,
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
            negative or [],
            True,
            ui=ui.PreviewText(thinking),
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

        diffusion_model = model.model.diffusion_model
        device = model.load_device
        transformer_options = model.model_options.get(
            "transformer_options", {}
        ).copy()
        comfy.model_management.load_models_gpu([model])
        model.pre_run()
        try:
            progress = comfy.utils.ProgressBar(max_text_tokens)
            event = generate_text_event(
                diffusion_model,
                prefix_arguments(
                    positive_metadata,
                    device,
                    diffusion_model.dtype,
                    image_only=False,
                ),
                max_text_tokens=max_text_tokens,
                transformer_options=transformer_options,
                progress=progress.update_absolute,
                interrupt=comfy.model_management.throw_exception_if_processing_interrupted,
            )
        finally:
            model.cleanup()

        positive_metadata = append_interleave_tokens(
            positive_metadata, event.token_ids, event.image_requested
        )
        if event.image_requested:
            negative_metadata = append_interleave_tokens(
                negative_metadata, [151670], True
            )
        text_token_ids = [value for value in event.token_ids if value != 151670]
        text = (
            clip.decode(text_token_ids, skip_special_tokens=True).strip()
            if text_token_ids
            else ""
        )
        updated_positive = [[positive[0][0], positive_metadata]]
        updated_negative = [[negative[0][0], negative_metadata]]
        return io.NodeOutput(
            updated_positive,
            text,
            updated_negative,
            event.image_requested,
            ui=ui.PreviewText(text),
        )


def _format_thinking_text(text: str) -> str:
    """Return only the visible body of a SenseNova thinking result."""

    text = text.strip()
    if text.startswith("<think>"):
        text = text[len("<think>") :].lstrip()
    if "</think>" in text:
        text = text.split("</think>", 1)[0]
    return text.strip()


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


class SenseNovaInterleaveStage(io.ComfyNode):
    """Capture one externally sampled interleave stage for later collection."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaInterleaveStage",
            display_name="SenseNova Interleave Stage",
            category="model/sampling/sensenova",
            description=(
                "Bundle one SenseNova text stage, its image-request flag, and "
                "the image sampled for that stage. Feed accumulated stages to "
                "SenseNova Interleave Collector."
            ),
            inputs=[
                io.String.Input("text", optional=True),
                io.Boolean.Input("image_requested", default=False),
                io.Image.Input("image", optional=True),
                io.Conditioning.Input("conditioning", optional=True),
                io.Conditioning.Input("negative", optional=True),
                io.Latent.Input("samples", optional=True),
            ],
            outputs=[io.AnyType.Output(display_name="stage")],
        )

    @classmethod
    def execute(
        cls,
        text="",
        image_requested=False,
        image=None,
        conditioning=None,
        negative=None,
        samples=None,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            {
                "text": str(text or ""),
                "image_requested": bool(image_requested),
                "image": image,
                "conditioning": conditioning,
                "negative": negative,
                "samples": samples,
            }
        )


def _interleave_stage_images(stages):
    images = []
    for stage in stages:
        if not stage.get("image_requested") or stage.get("image") is None:
            continue
        image = stage["image"]
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError(
                "SenseNova Interleave Collector expects stage images in BHWC format."
            )
        images.extend(image[index : index + 1] for index in range(image.shape[0]))
    if not images:
        return None
    return torch.cat(images, dim=0)


class SenseNovaInterleaveCollector(io.ComfyNode):
    """Collect externally sampled stages into ordered text and images."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaInterleaveCollector",
            display_name="SenseNova Interleave Collector",
            category="model/sampling/sensenova",
            is_input_list=True,
            description=(
                "Collect accumulated SenseNova interleave stages into ordered text "
                "with image placeholders, an image batch, and separate thinking text."
            ),
            inputs=[
                io.AnyType.Input(
                    "stages",
                    tooltip="A list of SenseNova Interleave Stage records.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="text"),
                io.Image.Output(display_name="images"),
                io.String.Output(display_name="thinking"),
            ],
        )

    @classmethod
    def execute(
        cls,
        stages,
    ) -> io.NodeOutput:
        if isinstance(stages, dict):
            stages = [stages]
        if not isinstance(stages, (list, tuple)):
            raise ValueError(
                "SenseNova Interleave Collector expects a list of stage records."
            )
        if any(not isinstance(stage, dict) for stage in stages):
            raise ValueError(
                "SenseNova Interleave Collector received an invalid stage record."
            )
        if not stages:
            raise ValueError("SenseNova Interleave Collector received no stages.")

        text_parts = []
        for stage in stages:
            text_parts.append(str(stage.get("text") or ""))
            if stage.get("image_requested"):
                text_parts.append("<image>")

        text, thinking = split_interleave_text("".join(text_parts))
        images = _interleave_stage_images(stages)
        return io.NodeOutput(text, images, thinking)


class SenseNovaExtension(ComfyExtension):
    """Register the native SenseNova node collection."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SenseNovaTextEncode,
            SenseNovaGenerate,
            SenseNovaSamplingOptions,
            SenseNovaInterleave,
            SenseNovaInterleaveStage,
            SenseNovaInterleaveCollector,
        ]


async def comfy_entrypoint() -> SenseNovaExtension:
    """Create the native SenseNova node extension."""

    return SenseNovaExtension()
