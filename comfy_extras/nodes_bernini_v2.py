"""Native ComfyUI support for ByteDance Bernini v2."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from typing_extensions import override

import comfy.model_management
import comfy.model_sampling
import comfy.ops
import comfy.sampler_helpers
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
from comfy.ldm.bernini_v2.guidance import (
    compose_denoised_guidance,
    guidance_chunks,
    unipc_flow_sigmas,
)
from comfy.ldm.bernini_v2.media import fit_media_size, ordered_renderer_sources
from comfy.ldm.bernini_v2.planner import (
    BerniniV2Plan,
    create_plan,
    split_reference_images,
)
from comfy.ldm.bernini_v2.presets import task_preset
from comfy.ldm.bernini_v2.runtime import load_planner_runtime
from comfy.ldm.bernini_v2.unipc import sample_flow_unipc_bh2
from comfy_api.latest import ComfyExtension, io

BerniniV2PlannerType = io.Custom("BERNINI_V2_PLANNER")


class BerniniV2ModelSampling(
    comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST
):
    pass


def _portable_name(name: str) -> str:
    """Keep serialized workflow model names stable across host platforms."""
    return name.replace("\\", "/")


def _model_options(folder: str) -> list[str]:
    return [
        _portable_name(name) for name in folder_paths.get_filename_list(folder)
    ]


class BerniniV2WanLoader(io.ComfyNode):
    """Load one high- or low-noise renderer with Comfy's Wan model."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniV2WanLoader",
            display_name="Load Bernini v2 Wan Renderer",
            category="advanced/loaders/bernini_v2",
            description=(
                "Loads one Wan safetensors renderer from models/diffusion_models. "
                "The output is a standard MODEL and uses ComfyUI device/offload management."
            ),
            inputs=[
                io.Combo.Input(
                    "unet_name", options=_model_options("diffusion_models")
                ),
                io.Float.Input(
                    "flow_shift",
                    default=5.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Combo.Input(
                    "weight_dtype",
                    options=[
                        "bfloat16",
                        "float16",
                        "default",
                        "fp8_e4m3fn",
                        "fp8_e4m3fn_fast",
                        "fp8_e5m2",
                    ],
                    default="bfloat16",
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        unet_name: str,
        flow_shift: float = 5.0,
        weight_dtype: str = "bfloat16",
    ) -> io.NodeOutput:
        model_path = folder_paths.get_full_path_or_raise(
            "diffusion_models", unet_name
        )
        if not model_path.endswith(".safetensors"):
            raise ValueError(
                "Bernini v2 renderers must each be one .safetensors file"
            )
        state_dict = comfy.utils.load_torch_file(model_path, safe_load=True)
        if "scaled_fp8" in state_dict:
            state_dict, _ = comfy.utils.convert_old_quants(state_dict)
        native_quant = any(key.endswith(".comfy_quant") for key in state_dict)
        if native_quant and weight_dtype.startswith("fp8"):
            raise ValueError(
                "pre-quantized Comfy weights cannot be recast to FP8; choose bfloat16 or default"
            )
        model_options = {}
        if weight_dtype == "bfloat16":
            model_options["dtype"] = torch.bfloat16
        elif weight_dtype == "float16":
            model_options["dtype"] = torch.float16
        elif weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        if native_quant:
            compute_dtype = (
                torch.float16 if weight_dtype == "float16" else torch.bfloat16
            )
            model_options["dtype"] = compute_dtype
            model_options["custom_operations"] = comfy.ops.mixed_precision_ops(
                {}, compute_dtype
            )
            logging.info(
                "Bernini v2 quantized renderer compute dtype: %s", compute_dtype
            )
        model = comfy.sd.load_diffusion_model_state_dict(
            state_dict, model_options=model_options
        )
        if model is None:
            raise RuntimeError(
                f"ComfyUI could not detect the Wan model in {model_path}"
            )
        model = model.clone()

        model_sampling = BerniniV2ModelSampling(model.model.model_config)
        model_sampling.set_parameters(shift=flow_shift, multiplier=1000)
        model.add_object_patch("model_sampling", model_sampling)
        return io.NodeOutput(model)


class BerniniV2PlannerLoader(io.ComfyNode):
    """Load Qwen2.5-VL and Bernini planning heads under Comfy model management."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniV2PlannerLoader",
            display_name="Load Bernini v2 Planner",
            category="advanced/loaders/bernini_v2",
            description=(
                "Loads one standalone planner from models/text_encoders. It contains "
                "Qwen language/vision, Bernini heads, config, and tokenizer."
            ),
            inputs=[
                io.Combo.Input(
                    "planner_name", options=_model_options("text_encoders")
                ),
                io.Combo.Input(
                    "dtype",
                    options=["bfloat16", "float16"],
                    default="bfloat16",
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[BerniniV2PlannerType.Output(display_name="planner")],
        )

    @classmethod
    def execute(cls, planner_name: str, dtype: str = "bfloat16") -> io.NodeOutput:
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        planner_path = folder_paths.get_full_path_or_raise(
            "text_encoders", planner_name
        )
        planner = load_planner_runtime(planner_path, dtype=torch_dtype)
        return io.NodeOutput(planner)


BerniniV2PlanType = io.Custom("BERNINI_V2_PLAN")


class BerniniV2PlanNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniV2Plan",
            display_name="Bernini v2 Plan",
            category="conditioning/bernini_v2",
            description="Runs the native Qwen/VIT semantic planner and produces four renderer condition arms.",
            inputs=[
                BerniniV2PlannerType.Input("planner"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input(
                    "negative_prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    advanced=True,
                ),
                io.Combo.Input(
                    "task",
                    options=["t2i", "i2i", "t2v", "v2v", "r2v", "rv2v"],
                    default="t2v",
                ),
                io.Int.Input("width", default=848, min=16, max=8192, step=16),
                io.Int.Input("height", default=480, min=16, max=8192, step=16),
                io.Int.Input("length", default=33, min=1, max=8192, step=4),
                io.Image.Input("source_video", optional=True),
                io.Video.Input("video", optional=True, advanced=True),
                io.Autogrow.Input(
                    "reference_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("reference_image"),
                        prefix="reference_image_",
                        min=0,
                        max=8,
                    ),
                ),
                io.Float.Input(
                    "source_fps",
                    default=16.0,
                    min=0.01,
                    max=240.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Boolean.Input("use_task_defaults", default=True, advanced=True),
                io.Boolean.Input("match_source_size", default=True, advanced=True),
                io.Int.Input(
                    "max_media_size",
                    default=848,
                    min=240,
                    max=8192,
                    step=16,
                    advanced=True,
                ),
                io.Int.Input(
                    "planning_steps", default=25, min=1, max=100, advanced=True
                ),
                io.Int.Input(
                    "vit_denoising_steps", default=5, min=1, max=100, advanced=True
                ),
                io.Float.Input(
                    "vit_text_cfg",
                    default=1.2,
                    min=0.0,
                    max=20.0,
                    step=0.05,
                    advanced=True,
                ),
                io.Float.Input(
                    "vit_image_cfg",
                    default=1.0,
                    min=0.0,
                    max=20.0,
                    step=0.05,
                    advanced=True,
                ),
                io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF),
            ],
            outputs=[BerniniV2PlanType.Output(display_name="plan")],
        )

    @classmethod
    def execute(
        cls,
        planner,
        positive,
        negative,
        prompt,
        negative_prompt,
        task,
        width,
        height,
        length,
        source_video=None,
        video=None,
        reference_images=None,
        source_fps=16.0,
        use_task_defaults=True,
        match_source_size=True,
        max_media_size=848,
        planning_steps=25,
        vit_denoising_steps=5,
        vit_text_cfg=1.2,
        vit_image_cfg=1.0,
        seed=42,
    ) -> io.NodeOutput:
        if video is not None:
            if source_video is not None:
                raise ValueError(
                    "connect either source_video IMAGE batch or VIDEO, not both"
                )
            components = video.get_components()
            source_video = components.images
            source_fps = float(components.frame_rate)
        references = split_reference_images(reference_images)
        if use_task_defaults:
            preset = task_preset(task)
            planning_steps = preset["planning_steps"]
            vit_denoising_steps = preset["vit_denoising_steps"]
            max_media_size = preset["max_media_size"]
        size_source = source_video
        if task == "i2i" and references:
            size_source = references[0]
        if match_source_size and size_source is not None:
            height, width = fit_media_size(
                size_source.shape[1],
                size_source.shape[2],
                max_size=max_media_size,
            )
        plan = create_plan(
            planner,
            positive=positive,
            negative=negative,
            prompt=prompt,
            negative_prompt=negative_prompt,
            task=task,
            width=width,
            height=height,
            length=length,
            max_media_size=max_media_size,
            source_video=source_video,
            reference_images=references,
            source_fps=source_fps,
            planning_steps=planning_steps,
            vit_denoising_steps=vit_denoising_steps,
            vit_text_cfg=vit_text_cfg,
            vit_image_cfg=vit_image_cfg,
            seed=seed,
        )
        return io.NodeOutput(plan)


def _resize_source_media(image: torch.Tensor, max_size: int) -> torch.Tensor:
    resized_height, resized_width = fit_media_size(
        image.shape[1],
        image.shape[2],
        max_size=max_size,
    )
    channels_first = image[..., :3].movedim(-1, 1).float()
    if channels_first.shape[-2:] != (resized_height, resized_width):
        channels_first = F.interpolate(
            channels_first,
            size=(resized_height, resized_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    return channels_first.movedim(1, -1)


def encode_renderer_sources(
    plan: BerniniV2Plan,
    vae,
    *,
    encode_mode: str = "auto",
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, torch.Tensor]]:
    """VAE encode source video and reference images as separate Wan streams."""
    if encode_mode not in {"auto", "tiled"}:
        raise ValueError(f"unsupported VAE encode mode: {encode_mode!r}")
    encode = vae.encode_tiled if encode_mode == "tiled" else vae.encode
    video_latents = []
    if plan.source_video is not None:
        video = _resize_source_media(
            plan.source_video[: plan.length],
            plan.max_media_size,
        )
        video_latents.append(encode(video))

    image_latents = []
    for image in plan.reference_images:
        image_latents.append(
            encode(_resize_source_media(image[:1], plan.max_media_size))
        )

    latent = torch.zeros(
        [1, 16, ((plan.length - 1) // 4) + 1, plan.height // 8, plan.width // 8],
        device=comfy.model_management.intermediate_device(),
    )
    return video_latents, image_latents, {"samples": latent}


def _conditioning(context: torch.Tensor, context_latents: list[torch.Tensor]):
    metadata = {}
    if context_latents:
        metadata["context_latents"] = context_latents
    return [[context, metadata]]


class BerniniV2Guider(comfy.samplers.CFGGuider):
    """Four/five-arm guider matching the released Bernini v2 renderer."""

    def __init__(
        self,
        model_patcher,
        plan: BerniniV2Plan,
        video_latents: list[torch.Tensor],
        image_latents: list[torch.Tensor],
        *,
        omega_video: float,
        omega_image: float,
        omega_text: float,
        omega_target: float,
        scale: float,
        guidance_batch_size: str | int = "auto",
    ):
        super().__init__(model_patcher)
        self.omega_video = omega_video
        self.omega_image = omega_image
        self.omega_text = omega_text
        self.omega_target = omega_target
        self.scale = scale
        self.guidance_batch_size = guidance_batch_size
        self.rv2v = plan.task == "rv2v"

        # The released renderer appends reference-image VAE tokens before
        # source-video tokens. Wan assigns source ids/RoPE by list position, so
        # reversing these streams corrupts only the combined RV2V condition.
        all_sources = ordered_renderer_sources(
            image_sources=image_latents,
            video_sources=video_latents,
        )
        conditions = {
            "base": _conditioning(plan.contexts["wotxt_wovit"], []),
            "text": _conditioning(plan.contexts["wtxt_wovit"], all_sources),
            "target": _conditioning(plan.contexts["wtxt_wvit"], all_sources),
        }
        if self.rv2v:
            conditions["video"] = _conditioning(
                plan.contexts["wotxt_wovit"], video_latents
            )
            conditions["image"] = _conditioning(
                plan.contexts["wotxt_wovit"], all_sources
            )
        elif all_sources:
            conditions["source"] = _conditioning(
                plan.contexts["wotxt_wovit"], all_sources
            )
        self.arm_names = list(conditions)
        self.inner_set_conds(conditions)

    def _predict_arms(
        self, inner_model, conditions, x, timestep, model_options, *, scale
    ):
        predictions = {}
        for chunk in guidance_chunks(self.arm_names, self.guidance_batch_size):
            outputs = comfy.samplers.calc_cond_batch(
                inner_model,
                [conditions[name] for name in chunk],
                x,
                timestep,
                model_options,
            )
            predictions.update(zip(chunk, outputs, strict=True))
        return compose_denoised_guidance(
            predictions,
            x,
            timestep,
            omega_video=self.omega_video * scale,
            omega_image=self.omega_image * scale,
            omega_text=self.omega_text * scale,
            omega_target=self.omega_target * scale,
            rv2v=self.rv2v,
        )

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        del seed
        model_options = model_options or {}
        return self._predict_arms(
            self.inner_model,
            self.conds,
            x,
            timestep,
            model_options,
            scale=self.scale,
        )


class BerniniV2DualExpertGuider(BerniniV2Guider):
    """One guider that switches Wan experts without resetting sampler history."""

    def __init__(
        self,
        high_noise_model,
        low_noise_model,
        *args,
        boundary: float,
        omega_scale: float,
        **kwargs,
    ):
        super().__init__(high_noise_model, *args, scale=1.0, **kwargs)
        self.low_noise_model = low_noise_model
        self.boundary = boundary
        self.omega_scale = omega_scale
        self.low_inner = None
        self.low_conds = None
        self.low_loaded_models = []
        self.switched = False

    def outer_sample(
        self,
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=None,
        latent_shapes=None,
    ):
        low_conditions = {
            name: [condition.copy() for condition in self.conds[name]]
            for name in self.arm_names
        }
        self.low_inner, self.low_conds, self.low_loaded_models = (
            comfy.sampler_helpers.prepare_sampling(
                self.low_noise_model,
                noise.shape,
                low_conditions,
                self.low_noise_model.model_options,
            )
        )
        self.low_noise_model.pre_run()
        self.switched = False
        try:
            return super().outer_sample(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
        finally:
            self.low_noise_model.cleanup()
            comfy.sampler_helpers.cleanup_models(self.low_conds, self.low_loaded_models)
            self.low_inner = None
            self.low_conds = None
            self.low_loaded_models = []

    def inner_sample(
        self,
        noise,
        latent_image,
        device,
        sampler,
        sigmas,
        denoise_mask,
        callback,
        disable_pbar,
        seed,
        latent_shapes=None,
    ):
        self.low_inner.latent_shapes = latent_shapes
        low_latent = latent_image
        if low_latent is not None and torch.count_nonzero(low_latent) > 0:
            low_latent = self.low_inner.process_latent_in(low_latent)
        self.low_conds = comfy.samplers.process_conds(
            self.low_inner,
            noise,
            self.low_conds,
            device,
            low_latent,
            denoise_mask,
            seed,
            latent_shapes=latent_shapes,
        )
        return super().inner_sample(
            noise,
            latent_image,
            device,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        del seed
        model_options = model_options or {}
        use_low_noise = bool((timestep[0] < self.boundary).item())
        if use_low_noise:
            if not self.switched:
                memory_required, minimum_memory_required = (
                    comfy.sampler_helpers.estimate_memory(
                        self.low_noise_model,
                        x.shape,
                        self.low_conds,
                    )
                )
                comfy.model_management.load_models_gpu(
                    [self.low_noise_model, *self.low_loaded_models],
                    memory_required=memory_required,
                    minimum_memory_required=minimum_memory_required,
                )
                self.switched = True
            return self._predict_arms(
                self.low_inner,
                self.low_conds,
                x,
                timestep,
                model_options,
                scale=self.omega_scale,
            )
        return self._predict_arms(
            self.inner_model,
            self.conds,
            x,
            timestep,
            model_options,
            scale=1.0,
        )


class BerniniV2RendererGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniV2RendererGuider",
            display_name="Bernini v2 Renderer Guider",
            category="sampling/bernini_v2",
            description=(
                "VAE-encodes source media and creates one native Wan guider that switches "
                "experts at the official boundary without resetting UniPC sampler history."
            ),
            inputs=[
                BerniniV2PlanType.Input("plan"),
                io.Model.Input("high_noise_model"),
                io.Model.Input("low_noise_model"),
                io.Vae.Input("vae"),
                io.Float.Input(
                    "omega_video",
                    default=1.25,
                    min=0.0,
                    max=20.0,
                    step=0.05,
                    advanced=True,
                ),
                io.Float.Input(
                    "omega_image",
                    default=3.0,
                    min=0.0,
                    max=20.0,
                    step=0.05,
                    advanced=True,
                ),
                io.Float.Input(
                    "omega_text",
                    default=4.0,
                    min=0.0,
                    max=20.0,
                    step=0.05,
                    advanced=True,
                ),
                io.Float.Input(
                    "omega_target",
                    default=1.2,
                    min=0.0,
                    max=20.0,
                    step=0.05,
                    advanced=True,
                ),
                io.Float.Input(
                    "omega_scale",
                    default=0.75,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    advanced=True,
                ),
                io.Boolean.Input("use_task_defaults", default=True, advanced=True),
                io.Float.Input(
                    "boundary",
                    default=0.875,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    advanced=True,
                ),
                io.Combo.Input(
                    "guidance_batch_size",
                    options=["auto", "1", "2", "all"],
                    default="auto",
                    advanced=True,
                ),
                io.Combo.Input(
                    "vae_encode_mode",
                    options=["auto", "tiled"],
                    default="auto",
                    advanced=True,
                ),
            ],
            outputs=[
                io.Guider.Output(display_name="guider"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        plan,
        high_noise_model,
        low_noise_model,
        vae,
        omega_video,
        omega_image,
        omega_text,
        omega_target,
        omega_scale,
        use_task_defaults=True,
        boundary=0.875,
        guidance_batch_size="auto",
        vae_encode_mode="auto",
    ) -> io.NodeOutput:
        video_latents, image_latents, latent = encode_renderer_sources(
            plan, vae, encode_mode=vae_encode_mode
        )
        if use_task_defaults:
            preset = task_preset(plan.task)
            omega_video = preset["omega_video"]
            omega_image = preset["omega_image"]
            omega_text = preset["omega_text"]
            omega_target = preset["omega_target"]
            omega_scale = preset["omega_scale"]
        common = {
            "plan": plan,
            "video_latents": video_latents,
            "image_latents": image_latents,
            "omega_video": omega_video,
            "omega_image": omega_image,
            "omega_text": omega_text,
            "omega_target": omega_target,
            "guidance_batch_size": guidance_batch_size,
        }
        guider = BerniniV2DualExpertGuider(
            high_noise_model,
            low_noise_model,
            boundary=boundary,
            omega_scale=omega_scale,
            **common,
        )
        return io.NodeOutput(guider, latent)


class BerniniV2Scheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniV2Scheduler",
            display_name="Bernini v2 UniPC Sigmas",
            category="sampling/bernini_v2",
            description=(
                "Matches Diffusers UniPC flow-sigma spacing: linearly spaced training timesteps "
                "from 999 to 0, followed by the terminal zero sigma."
            ),
            inputs=[
                BerniniV2PlanType.Input("plan"),
                io.Int.Input("steps", default=40, min=1, max=10000, advanced=True),
                io.Float.Input(
                    "flow_shift", default=5.0, min=0.01, max=100.0, step=0.01
                ),
                io.Boolean.Input("use_task_defaults", default=True, advanced=True),
            ],
            outputs=[io.Sigmas.Output()],
        )

    @classmethod
    def execute(
        cls, plan, steps, flow_shift=5.0, use_task_defaults=True
    ) -> io.NodeOutput:
        if use_task_defaults:
            steps = task_preset(plan.task)["steps"]
        return io.NodeOutput(unipc_flow_sigmas(steps, flow_shift))


class BerniniV2UniPCSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniV2UniPCSampler",
            display_name="Bernini v2 Flow UniPC (BH2)",
            category="sampling/bernini_v2",
            description=(
                "The released order-2 UniPC BH2 solver for flow prediction. "
                "ComfyUI's generic UniPC uses a VP noise schedule and is not equivalent."
            ),
            inputs=[],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        return io.NodeOutput(comfy.samplers.KSAMPLER(sample_flow_unipc_bh2))


class BerniniV2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            BerniniV2WanLoader,
            BerniniV2PlannerLoader,
            BerniniV2PlanNode,
            BerniniV2RendererGuider,
            BerniniV2Scheduler,
            BerniniV2UniPCSampler,
        ]


async def comfy_entrypoint() -> BerniniV2Extension:
    return BerniniV2Extension()
