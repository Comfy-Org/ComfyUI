"""
Authoritative type contract for the secure custom-node SDK.

This is the backend counterpart to the frontend published API's generated
``comfy-api.d.ts``: the stable, versioned surface a node author compiles
against when importing ``from comfy_api.v0_0_3 import sdk`` (or a pinned
version). Type checkers prefer this stub over ``_sdk_public.py``, so it defines
the *contract*; the ``.py`` defines the in-process default *implementation*.

Two audiences, separated below:
  1. NODE AUTHORS — refs, ctx and its domains, and the ``ctx()`` accessor.
  2. HOST / OVERLAY — the provider seam (ExecutionBackend / CtxProvider /
     RefResolver / ExecutionPlan / providers). Node authors do not use these;
     they exist so a separable overlay can replace the defaults with an
     isolated engine. (Analogous to the frontend "host-plumbing declarations".)

If prose and this stub disagree, treat the stub + its implementation tests as
authoritative.
"""
from typing import Any, Awaitable, Callable, Optional, Protocol, TypeVar, runtime_checkable

# NOTE: the contract deliberately does NOT import torch or any backend module.
# Node code has no direct access to torch/CUDA/filesystem; it works through
# refs, operations, and ctx — all brokered.

# =========================================================================== #
# 1. NODE-AUTHOR SURFACE
# =========================================================================== #

# --- Refs: opaque, typed asset handles. A ref never exposes the buffer. The
#     PREFERRED interface is operations on the asset (below); raw buffer access
#     is a permissioned escape hatch. --- #
class Ref:
    kind: str
    id: str
    async def describe(self, max_value_chars: int = ...) -> dict[str, Any]: ...
    async def op(self, name: str, **params: Any) -> Any: ...

class ClosureRef(Ref):
    """Prompt-scoped handle to pack math retained for a sampling phase."""
    KIND: str
    async def attach_model(self, model: "ModelRef") -> "ModelRef": ...
    async def wrap_sampler(
        self,
        sampler: "SamplerRef",
        *,
        start_percent: Optional[float] = ...,
        end_percent: Optional[float] = ...,
    ) -> "SamplerRef": ...
    async def as_latent_operation(self) -> "LatentOperationRef": ...
    async def as_sampler(self) -> "SamplerRef": ...

class LatentOperationRef(Ref):
    """Host-owned delayed one-tensor operation backed by a node closure."""
    KIND: str

_T = TypeVar("_T", bound="TensorRef")

class ValueRef(Ref):
    KIND: str

class TensorRef(Ref):
    KIND: str
    # RAW ESCAPE HATCH — permissioned (`raw`/`tensor.read`), discouraged; forces
    # the dedicated tier under the overlay. Return is untyped by design (the
    # contract does not depend on torch).
    async def raw(self) -> Any: ...

class ImageRef(TensorRef):
    """IMAGE asset. Preferred interface = operations; the buffer stays engine-side.

    ``op`` is generic, UNTYPED dispatch — the transport seam. The overlay adds
    ops to the vocabulary as data. Node authors call statically-typed wrappers
    from the secure lib (e.g. ``comfy_secure_sdk.ops.desaturate(...)``) rather
    than passing op-name strings here directly."""
    KIND: str
    async def op(self, name: str, **params: Any) -> "ImageRef": ...
    async def invert(self) -> "ImageRef": ...     # built-in primitive
    async def scale(self, factor: float) -> "ImageRef": ...  # built-in primitive
    async def rgb(self) -> "ImageRef": ...
    async def to_device(self, device: str = ...) -> "ImageRef": ...
    async def spatial_shape(self) -> tuple[int, int]: ...
    async def batch_size(self) -> int: ...
    async def select_batch(self, indices: list[int]) -> "ImageRef": ...

class MaskRef(TensorRef):
    """MASK asset."""
    KIND: str
    async def grow(
        self, amount: int, tapered_corners: bool = ...,
    ) -> "MaskRef": ...

_L = TypeVar("_L", bound="LatentRef")

class LatentRef(Ref):
    """LATENT — dict {samples, ...}."""
    KIND: str
    @classmethod
    async def empty(
        cls,
        width: int,
        height: int,
        batch_size: int = ...,
        channels: int = ...,
        spatial_downscale_ratio: Optional[int] = ...,
    ) -> "LatentRef": ...
    async def repeat_batch(self, amount: int) -> "LatentRef": ...
    async def noise_mask(self) -> Optional[MaskRef]: ...
    async def spatial_shape(self) -> tuple[int, int]: ...
    async def resize(
        self, width: int, height: int, method: str = ...,
    ) -> "LatentRef": ...
    async def random_noise(
        self, seed: int, source: str = ...,
        batch_size: Optional[int] = ...,
    ) -> TensorRef: ...
    async def composite(
        self,
        source: "LatentRef",
        *,
        x: int = ...,
        y: int = ...,
        resize_source: bool = ...,
        mask: Optional[MaskRef] = ...,
    ) -> "LatentRef": ...
    async def value(self) -> dict: ...
    @classmethod
    async def from_value(cls: type[_L], v: dict) -> _L: ...

class CondRef(Ref):
    """CONDITIONING."""
    KIND: str
    async def sequence_length(self) -> int: ...
    async def combine(self, other: "CondRef") -> "CondRef": ...
    async def concat(self, other: "CondRef") -> "CondRef": ...
    async def zero_out(self) -> "CondRef": ...
    async def with_timestep_range(
        self, start: float, end: float,
    ) -> "CondRef": ...
    async def with_metadata(
        self, *, width: Optional[int] = ...,
        height: Optional[int] = ..., crop_w: Optional[int] = ...,
        crop_h: Optional[int] = ..., target_width: Optional[int] = ...,
        target_height: Optional[int] = ...,
    ) -> "CondRef": ...
    async def has_spatial_metadata(self) -> bool: ...
    async def with_mask(
        self,
        mask: MaskRef,
        strength: float = ...,
        set_area_to_bounds: bool = ...,
    ) -> "CondRef": ...
    async def with_clip_vision_output(
        self, output: "ClipVisionOutputRef",
    ) -> "CondRef": ...
    async def with_concat_latent(
        self,
        model: "ModelRef",
        latent: LatentRef,
        extra_latent: Optional[LatentRef] = ...,
    ) -> "CondRef": ...
    async def spatial_crop(
        self, *, x: int, y: int, width: int, height: int,
        source_width: int, source_height: int,
        target_width: Optional[int] = ...,
        target_height: Optional[int] = ...,
    ) -> "CondRef": ...

class GuiderRef(Ref):
    KIND: str
    async def spatial_crop_inputs(
        self, *, regions: list[tuple[int, int, int, int]],
        source_width: int, source_height: int,
        target_width: int, target_height: int,
    ) -> "GuiderRef": ...

class SamplerRef(Ref):
    KIND: str
    @classmethod
    async def named(
        cls, name: str, *, eta: Optional[float] = ...,
        ge_gamma: Optional[float] = ...,
    ) -> "SamplerRef": ...

class SigmasRef(Ref):
    KIND: str
    async def steps(self) -> int: ...
    async def value_at(self, index: int) -> float: ...

class InterpolationStatesRef(Ref):
    KIND: str
    async def skip_mask(self, pair_count: int) -> list[bool]: ...

class UpscaleModelRef(Ref):
    KIND: str
    async def upscale(
        self,
        images: ImageRef,
        per_batch: int = ...,
        downscale_ratio: float = ...,
        downscale_method: str = ...,
        precision: str = ...,
        tile_size: Optional[int] = ...,
        channels_last: bool = ...,
    ) -> ImageRef: ...

class ModelRef(Ref):
    """MODEL — patch/hook via ctx.models; weights never materialize in-node."""
    KIND: str
    async def patch(self, transform: str, **params: Any) -> "ModelRef": ...
    async def spatial_crop_inputs(
        self, *, regions: list[tuple[int, int, int, int]],
        source_width: int, source_height: int,
        target_width: int, target_height: int,
    ) -> "ModelRef": ...
    async def family(self) -> str: ...
    async def unet_context_dim(self) -> Optional[int]: ...
    async def sigma_for_percent(
        self, percent: float, actual_endpoints: bool = ...,
    ) -> float: ...
    async def is_zero_terminal_snr(self) -> bool: ...
    async def sampling_sigma_delta(
        self, *, steps: int, sampler_name: str, scheduler: str,
        start_step: int, end_step: int, denoise: float = ...,
        sigma_schedule: Optional[dict] = ...,
    ) -> float: ...

class ClipRef(Ref):
    KIND: str
    async def set_last_layer(self, stop_at_clip_layer: int) -> "ClipRef": ...
    async def with_attention_impl(self, mode: str) -> "ClipRef": ...
    async def describe_tokens(self, tokens: dict) -> dict: ...
    async def tokenize(self, text: str, **kwargs: Any) -> dict: ...
    async def encode_from_tokens_scheduled(
        self, tokens: dict, add_dict: Optional[dict] = ...,
    ) -> CondRef: ...
    async def encode_token_weights_component(
        self, component: str, tokens: list,
    ) -> tuple[TensorRef, Optional[TensorRef]]: ...
    async def encode(self, text: str) -> CondRef: ...
    async def generate_text(
        self,
        prompt: str,
        image: Optional[ImageRef] = ...,
        video: Optional[ImageRef] = ...,
        max_length: int = ...,
        do_sample: bool = ...,
        temperature: float = ...,
        top_k: Optional[int] = ...,
        top_p: float = ...,
        min_p: float = ...,
        repetition_penalty: float = ...,
        seed: Optional[int] = ...,
        presence_penalty: float = ...,
        thinking: bool = ...,
        use_default_template: bool = ...,
        num_beams: int = ...,
    ) -> str: ...

class LlamaCppModelRef(Ref):
    KIND: str
    async def generate(
        self,
        system: str,
        prompt: str,
        image: Optional[ImageRef] = ...,
        video: Optional[ImageRef] = ...,
        max_tokens: int = ...,
        temperature: float = ...,
        top_p: float = ...,
        repetition_penalty: float = ...,
        seed: int = ...,
    ) -> str: ...

class ClipVisionOutputRef(Ref):
    KIND: str
    async def concat(
        self, other: "ClipVisionOutputRef",
    ) -> "ClipVisionOutputRef": ...
    async def image_embeds(self) -> TensorRef: ...

class ClipVisionRef(Ref):
    KIND: str
    async def encode_image(
        self, image: ImageRef, crop: bool = ...,
    ) -> ClipVisionOutputRef: ...

class VaeRef(Ref):
    KIND: str
    async def latent_layout(self) -> dict[str, Optional[int]]: ...
    async def decode(self, latent: LatentRef) -> ImageRef: ...
    async def decode_tensor(self, latent: LatentRef) -> TensorRef: ...
    async def decode_tiled(
        self,
        latent: LatentRef,
        tile_size: int = ...,
        overlap: int = ...,
        temporal_size: int = ...,
        temporal_overlap: int = ...,
    ) -> ImageRef: ...
    async def decode_tensor_tiled(
        self,
        latent: LatentRef,
        tile_size: int = ...,
        overlap: int = ...,
        temporal_size: int = ...,
        temporal_overlap: int = ...,
    ) -> TensorRef: ...
    async def encode(self, image: ImageRef) -> LatentRef: ...
    async def encode_for_inpaint(
        self,
        image: ImageRef,
        mask: MaskRef,
        grow_mask_by: int = ...,
    ) -> LatentRef: ...
    async def encode_inpaint_conditioning(
        self,
        image: ImageRef,
        mask: MaskRef,
        positive: CondRef,
        negative: CondRef,
        noise_mask: bool = ...,
    ) -> tuple[CondRef, CondRef, LatentRef]: ...

class TimestepKeyframeRef(Ref):
    KIND: str

class ControlNetWeightsRef(Ref):
    KIND: str
    @classmethod
    async def from_list(
        cls,
        weights: list[float],
        uncond_multiplier: float = ...,
        extras: Any = ...,
    ) -> tuple["ControlNetWeightsRef", TimestepKeyframeRef]: ...
    @classmethod
    async def scaled_soft(
        cls,
        base_multiplier: float = ...,
        uncond_multiplier: float = ...,
    ) -> tuple["ControlNetWeightsRef", TimestepKeyframeRef]: ...

class ControlNetRef(Ref):
    KIND: str
    async def apply(
        self,
        positive: CondRef,
        negative: CondRef,
        image: ImageRef,
        strength: float = ...,
        start_percent: float = ...,
        end_percent: float = ...,
        vae: Optional[VaeRef] = ...,
    ) -> tuple[CondRef, CondRef]: ...
    async def apply_advanced(
        self,
        positive: CondRef,
        negative: CondRef,
        image: ImageRef,
        strength: float = ...,
        start_percent: float = ...,
        end_percent: float = ...,
        vae: Optional[VaeRef] = ...,
        mask: Optional[MaskRef] = ...,
        timestep_keyframe: Optional[TimestepKeyframeRef] = ...,
        weights: Optional[ControlNetWeightsRef] = ...,
    ) -> tuple[CondRef, CondRef]: ...
    async def with_union_type(
        self, type_number: Optional[int],
    ) -> "ControlNetRef": ...
    async def compile(
        self,
        *,
        backend: str = ...,
        mode: str = ...,
        fullgraph: bool = ...,
    ) -> "ControlNetRef": ...

class AudioRef(Ref):
    KIND: str

class VideoRef(Ref):
    KIND: str

class AssetRef(Ref):
    """A file/model resolved by name+hash, tenant-scoped. Never a raw path."""
    KIND: str

class ClipSegRef(Ref):
    KIND: str

class ImageClassifierRef(Ref):
    KIND: str
    async def classify(
        self,
        images: ImageRef,
        use_accelerator: bool = ...,
        top_k: int = ...,
    ) -> list[list[dict[str, Any]]]: ...
    async def predict_scores(
        self, images: ImageRef,
    ) -> ClassifierScoresRef: ...

class ClassifierScoresRef(Ref):
    async def shape(self) -> tuple[int, int]: ...
    async def select_above(
        self,
        batch_index: int,
        start: int,
        end: int,
        threshold: float,
        offset: int = ...,
        limit: int = ...,
    ) -> dict[str, Any]: ...

class SemanticSegmentationRef(Ref):
    KIND: str
    async def mask(
        self,
        image: ImageRef,
        classes: list[int],
    ) -> MaskRef: ...

class InpaintModelRef(Ref):
    KIND: str
    async def inpaint(
        self,
        image: ImageRef,
        mask: MaskRef,
    ) -> ImageRef: ...

class BackgroundRemovalModelRef(Ref):
    KIND: str
    async def mask(self, image: ImageRef) -> MaskRef: ...

class BrushNetRef(Ref):
    KIND: str
    async def apply(
        self,
        model: ModelRef,
        vae: VaeRef,
        image: ImageRef,
        mask: MaskRef,
        positive: CondRef,
        negative: CondRef,
        scale: float = ...,
        start_step: int = ...,
        end_step: int = ...,
    ) -> tuple[ModelRef, CondRef, CondRef, LatentRef]: ...

class PowerPaintRef(Ref):
    KIND: str
    async def apply(
        self,
        model: ModelRef,
        vae: VaeRef,
        image: ImageRef,
        mask: MaskRef,
        positive: CondRef,
        negative: CondRef,
        fitting: float = ...,
        function: str = ...,
        scale: float = ...,
        start_step: int = ...,
        end_step: int = ...,
        save_memory: str = ...,
    ) -> tuple[ModelRef, CondRef, CondRef, LatentRef]: ...

class ObjectDetectorRef(Ref):
    KIND: str
    async def detect(
        self,
        image: ImageRef,
        threshold: float = ...,
        class_name: str = ...,
        max_detections: int = ...,
    ) -> list[list[dict[str, Any]]]: ...

class ImagePreprocessorRef(Ref):
    KIND: str
    async def apply(
        self,
        image: ImageRef,
        mask: Optional[MaskRef] = ...,
    ) -> ImageRef: ...

class SamModelRef(Ref):
    KIND: str
    async def segment(
        self,
        image: ImageRef,
        boxes: list[Optional[list[float]]],
        point_coords: Optional[list[list[list[float]]]] = ...,
        point_labels: Optional[list[list[int]]] = ...,
        multimask_output: bool = ...,
    ) -> tuple[MaskRef, list[list[float]]]: ...
    async def segment_video(
        self,
        frames: ImageRef,
        boxes: list[list[float]],
    ) -> MaskRef: ...

class HuggingFaceWeight:
    """A public Hugging Face weight file declared in SDK_REQUIRED_WEIGHTS."""
    repo_id: str
    filename: str
    folder: str
    revision: str
    sha256: Optional[str]
    on_demand: bool
    def __init__(
        self,
        repo_id: str,
        filename: str,
        folder: str,
        revision: str = ...,
        sha256: Optional[str] = ...,
        on_demand: bool = ...,
    ) -> None: ...
    @property
    def catalogue_name(self) -> str: ...

# --- ctx domains: the brokered side-effect surface. In-process these call core
#     directly (allow-all); under the overlay they are policy-checked and
#     tenant-scoped. Domains marked (overlay) raise NotImplementedError in the
#     OSS in-process default until wired. --- #
class AssetsDomain(Protocol):
    async def resolve(self, folder: str, name: str) -> AssetRef: ...
    async def exists(self, folder: str, name: str) -> bool: ...
    async def delete_input(self, name: str) -> bool: ...
    async def path(self, ref: AssetRef) -> str: ...
    async def list(
        self, folder: str, prefix: str = ..., recursive: bool = ...,
    ) -> list[str]: ...
    async def latest(
        self, folder: str, prefix: str = ..., suffix: str = ...,
    ) -> Optional[str]: ...
    async def size(self, ref: AssetRef) -> int: ...
    async def digest(
        self, ref: AssetRef, algorithm: str = ...,
    ) -> str: ...
    async def read_range(
        self, ref: AssetRef, offset: int = ..., length: int = ...,
    ) -> bytes: ...
    async def read_bytes(self, ref: AssetRef) -> bytes: ...
    async def load_state_dict(
        self, ref: AssetRef, return_metadata: bool = ...
    ) -> Any: ...
    async def load_latent(self, ref: AssetRef) -> LatentRef: ...

class OutputDomain(Protocol):
    async def save_images(
        self,
        images: ImageRef,
        filename_prefix: str = ...,
        subfolder: str = ...,
        compress_level: int = ...,
        caption: Optional[str] = ...,
        caption_extension: str = ...,
        save_metadata: bool = ...,
        extra_metadata: Optional[dict[str, Any]] = ...,
        image_format: str = ...,
        quality: int = ...,
        filenames: Optional[list[str]] = ...,
        lossless: bool = ...,
        optimize: bool = ...,
    ) -> dict: ...
    async def save_images_with_alpha(
        self,
        images: ImageRef,
        mask: MaskRef,
        filename_prefix: str = ...,
        subfolder: str = ...,
        compress_level: int = ...,
    ) -> dict: ...
    async def save_text(
        self,
        text: str,
        filename_prefix: str = ...,
        subfolder: str = ...,
        extension: str = ...,
    ) -> str: ...
    async def write_text(
        self,
        text: str,
        filename: str,
        folder: str = ...,
        mode: str = ...,
        insert_newline: bool = ...,
    ) -> str: ...
    async def save_workflow_json(
        self, filename: str, mode: str = ...,
    ) -> str: ...
    async def save_latent(
        self,
        latent: LatentRef,
        filename_prefix: str = ...,
        preview_method: str = ...,
    ) -> dict: ...
    async def save_state_dict(
        self,
        state_dict: ValueRef,
        filename_prefix: str,
        metadata: Optional[dict[str, str]] = ...,
    ) -> str: ...
    async def save_model(
        self,
        model: ModelRef,
        filename_prefix: str,
        model_key_prefix: str = ...,
    ) -> str: ...
    async def save_video(
        self,
        images: ImageRef,
        audio: Optional[AudioRef] = ...,
        fps: float = ...,
        filename_prefix: str = ...,
        format: str | dict[str, Any] = ...,
        timeout_seconds: float = ...,
        codec: str = ...,
        encoder_options: Optional[dict[str, Any]] = ...,
        loop_count: int = ...,
        bit_depth: int = ...,
        save_output: bool = ...,
        save_metadata: bool = ...,
    ) -> dict: ...
    async def save_animation(
        self,
        images: ImageRef,
        fps: float = ...,
        filename_prefix: str = ...,
        format: str = ...,
        loop_count: int = ...,
        lossless: bool = ...,
        quality: int = ...,
        save_output: bool = ...,
    ) -> dict: ...
    async def save_image_sequence(
        self,
        images: ImageRef,
        filename_prefix: str = ...,
        format: str = ...,
        bit_depth: int = ...,
        save_output: bool = ...,
    ) -> dict: ...

class GraphDomain(Protocol):
    async def current_node_id(self) -> str: ...
    async def input_label(
        self, input_name: str, default: str = ...,
    ) -> str: ...
    async def expand_nodes(
        self,
        nodes: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
    ) -> dict[str, Any]: ...
    async def expand_loop(
        self, flow: Any, values: list[Any],
    ) -> dict[str, Any]: ...
    async def widget_values(
        self,
        node_id: int | str = ...,
        node_title: str = ...,
        node_name: str = ...,
        linked_input: str = ...,
    ) -> dict[str, Any]: ...
    async def block(self, reason: Optional[str] = ...) -> Any: ...

class ExecutionDomain(Protocol):
    async def interrupt(self) -> bool: ...

class ProgressDomain(Protocol):
    async def update(
        self, value: float, total: float, preview: Optional[ImageRef] = ...
    ) -> None: ...

class ScratchDomain(Protocol):
    async def dir(self) -> str: ...

class EventsDomain(Protocol):
    async def emit(self, event: str, data: dict) -> None: ...

class StorageDomain(Protocol):
    async def get(self, key: str) -> Optional[str]: ...
    async def set(self, key: str, value: str) -> None: ...

class InteractionDomain(Protocol):
    async def request(
        self,
        kind: str,
        payload: Any,
        *,
        reuse_last: bool = ...,
        remember: bool = ...,
        timeout: float = ...,
    ) -> Any: ...

class ModelsDomain(Protocol):
    async def download_huggingface_weights(
        self,
        repo_id: str,
        filename: str,
        folder: str,
        revision: str = ...,
        sha256: Optional[str] = ...,
    ) -> str: ...
    async def load_onnx_image_classifier(
        self,
        model: str,
        input_layout: str = ...,
        channel_order: str = ...,
        resize_mode: str = ...,
        input_scale: float = ...,
        pad_color: tuple[float, float, float] = ...,
        mean: tuple[float, float, float] = ...,
        std: tuple[float, float, float] = ...,
        activation: str = ...,
        resize_filter: str = ...,
    ) -> ImageClassifierRef: ...
    async def list_diffusion_models(
        self, include_connectors: bool = ...
    ) -> list[str]: ...
    async def load_checkpoint(
        self,
        name: str,
        weight_dtype: str = ...,
        compute_dtype: str = ...,
        cublas_linear: bool = ...,
    ) -> tuple[ModelRef, Optional[ClipRef], Optional[VaeRef]]: ...
    async def load_diffusion_model(
        self,
        name: str,
        extra_name: Optional[str] = ...,
        weight_dtype: str = ...,
        compute_dtype: str = ...,
        cublas_linear: bool = ...,
    ) -> ModelRef: ...
    async def load_gguf_model(
        self,
        name: str,
        extra_name: Optional[str] = ...,
        dequant_dtype: str = ...,
        patch_dtype: str = ...,
        patch_on_device: bool = ...,
    ) -> ModelRef: ...
    async def list_controlnet(self) -> list[str]: ...
    async def load_controlnet(
        self, name: str, model: Optional[ModelRef] = ...,
    ) -> ControlNetRef: ...
    async def load_advanced_controlnet(
        self,
        name: str,
        model: Optional[ModelRef] = ...,
        timestep_keyframe: Optional[TimestepKeyframeRef] = ...,
    ) -> ControlNetRef: ...
    async def load_controlnet_plusplus(
        self, name: str, control_type: str = ...,
    ) -> ControlNetRef: ...
    async def list_vae(self) -> list[str]: ...
    async def load_vae(
        self,
        name: str,
        device: str = ...,
        weight_dtype: str = ...,
    ) -> VaeRef: ...
    async def load_upscale_model(self, name: str) -> UpscaleModelRef: ...
    async def load_clip_vision(self, model: str) -> ClipVisionRef: ...
    async def load_text_encoder(
        self,
        model: str,
        model_type: str,
        device: str = ...,
    ) -> ClipRef: ...
    async def load_language_model(
        self,
        weights: list[str],
        family: str,
        device: str = ...,
        cache: bool = ...,
    ) -> ClipRef: ...
    async def load_ipadapter(
        self,
        model: str,
        clip_vision: ClipVisionRef,
    ) -> Ref: ...
    async def load_brushnet(
        self,
        model: str,
        dtype: str = ...,
    ) -> BrushNetRef: ...
    async def load_powerpaint(
        self,
        model: str,
        base_clip: str,
        powerpaint_clip: str,
        dtype: str = ...,
    ) -> PowerPaintRef: ...
    async def load_clipseg(self, model: str) -> ClipSegRef: ...
    async def load_image_classifier(
        self,
        model: str,
        architecture: str,
        labels: list[str],
    ) -> ImageClassifierRef: ...
    async def load_segformer(
        self,
        model: str,
        variant: str,
        num_labels: int,
    ) -> SemanticSegmentationRef: ...
    async def load_inpaint_model(
        self,
        model: str,
        architecture: str = ...,
    ) -> InpaintModelRef: ...
    async def load_background_removal_model(
        self,
        model: str,
    ) -> BackgroundRemovalModelRef: ...
    async def load_object_detector(self, model: str) -> ObjectDetectorRef: ...
    async def load_sam(
        self,
        model: str,
        architecture: str = ...,
        device_mode: str = ...,
    ) -> SamModelRef: ...
    async def generate_text(
        self,
        generator: str,
        input_text: str,
        max_new_tokens: int = ...,
        weight: Optional[str] = ...,
    ) -> str: ...
    async def memory_cleanup(
        self,
        empty_cache: bool = ...,
        collect_cycles: bool = ...,
        unload_all_models: bool = ...,
    ) -> tuple[int, int]: ...

class CivitaiDomain(Protocol):
    async def search_models(
        self,
        username: str,
        query: Optional[str] = ...,
        limit: int = ...,
        nsfw: bool = ...,
    ) -> dict[str, Any]: ...
    async def model_version(
        self, model_version_id: int,
    ) -> dict[str, Any]: ...
    async def model_version_by_hash(
        self, hash_value: str, refresh: bool = ...,
    ) -> dict[str, Any]: ...

class OllamaDomain(Protocol):
    async def list_models(self, endpoint: str) -> list[str]: ...
    async def generate(
        self,
        endpoint: str,
        model: str,
        system: str,
        prompt: str,
        images: Optional[ImageRef] = ...,
        context: Optional[list[int]] = ...,
        think: bool = ...,
        options: Optional[dict[str, Any]] = ...,
        keep_alive: int = ...,
        keep_alive_unit: str = ...,
        format: str = ...,
    ) -> dict[str, Any]: ...
    async def chat(
        self,
        endpoint: str,
        model: str,
        messages: list[dict[str, Any]],
        images: Optional[ImageRef] = ...,
        think: bool = ...,
        options: Optional[dict[str, Any]] = ...,
        keep_alive: int = ...,
        keep_alive_unit: str = ...,
        format: str | dict[str, Any] = ...,
        timeout_seconds: float = ...,
        tools: Optional[list[dict[str, Any]]] = ...,
    ) -> dict[str, Any]: ...

class LlmDomain(Protocol):
    async def chat(
        self,
        provider: str,
        profile: str,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict[str, Any]]] = ...,
        temperature: float = ...,
        max_tokens: int = ...,
        thinking: bool = ...,
        response_format: str | dict[str, Any] = ...,
        timeout_seconds: float = ...,
        vendor_options: Optional[dict[str, Any]] = ...,
    ) -> dict[str, Any]: ...

class WebSearchDomain(Protocol):
    async def search(
        self,
        query: str,
        *,
        provider_profile: str = ...,
        limit: int = ...,
        vendor_options: Optional[dict[str, Any]] = ...,
    ) -> list[dict[str, str]]: ...

class LlamaCppDomain(Protocol):
    async def load_chat_model(
        self,
        model_weight: str,
        mmproj_weight: Optional[str] = ...,
        *,
        family: str = ...,
        device: str = ...,
        context_length: int = ...,
        batch_size: int = ...,
        gpu_layers: int = ...,
        image_max_tokens: int = ...,
        top_k: int = ...,
        pool_size: int = ...,
        cache: bool = ...,
    ) -> LlamaCppModelRef: ...
    async def generate(
        self,
        model: LlamaCppModelRef,
        system: str,
        prompt: str,
        image: Optional[ImageRef] = ...,
        video: Optional[ImageRef] = ...,
        max_tokens: int = ...,
        temperature: float = ...,
        top_p: float = ...,
        repetition_penalty: float = ...,
        seed: int = ...,
    ) -> str: ...

class WanVideoDomain(Protocol):
    async def transformer_dim(self, model: Ref) -> int: ...

class AnimaDomain(Protocol):
    async def apply_lllite(
        self,
        model: ModelRef,
        weights: AssetRef,
        image: ImageRef,
        *,
        strength: float = ...,
        start_percent: float = ...,
        end_percent: float = ...,
        preserve_wrapper: bool = ...,
    ) -> ModelRef: ...

class IntegrationsDomain(Protocol):
    anima: AnimaDomain
    civitai: CivitaiDomain
    llm: LlmDomain
    llama_cpp: LlamaCppDomain
    ollama: OllamaDomain
    wanvideo: WanVideoDomain
    web: WebSearchDomain

class SystemDomain(Protocol):
    async def stats(self) -> dict[str, Any]: ...

class ClosuresDomain(Protocol):
    async def retain(
        self,
        kind: str,
        fn: Callable[..., Any],
        *,
        captures: Optional[dict[str, Any]] = ...,
    ) -> ClosureRef: ...
    async def attach_model(
        self, closure: ClosureRef, model: ModelRef,
    ) -> ModelRef: ...
    async def attach_sampler(
        self,
        closure: ClosureRef,
        sampler: SamplerRef,
        *,
        start_percent: Optional[float] = ...,
        end_percent: Optional[float] = ...,
    ) -> SamplerRef: ...
    async def create_latent_operation(
        self, closure: ClosureRef,
    ) -> LatentOperationRef: ...
    async def create_sampler(self, closure: ClosureRef) -> SamplerRef: ...

class Context(Protocol):
    assets: AssetsDomain
    progress: ProgressDomain
    scratch: ScratchDomain
    events: EventsDomain
    storage: StorageDomain
    interact: InteractionDomain
    output: OutputDomain
    graph: GraphDomain
    execution: ExecutionDomain
    integrations: IntegrationsDomain
    system: SystemDomain
    closures: ClosuresDomain
    # Declared in the contract; provided by the full SDK / overlay:
    models: ModelsDomain
    sample: Any   # engine-side sampling with per-step callbacks
    serve: Any    # namespaced, authenticated routes
    secrets: Any  # brokered from the BYOK store
    net: Any      # policy-checked fetch

def ctx() -> Context:
    """The active Context. Valid only inside node execution."""
    ...

def current_context() -> Context: ...

# =========================================================================== #
# 2. HOST / OVERLAY SEAM  (not for node authors)
# =========================================================================== #
class ExecutionPlan:
    prompt_id: str
    node_id: str
    node_type: str
    tier: str
    permissions: tuple[str, ...]
    required_weights: tuple[HuggingFaceWeight, ...]
    # Work-unit payload for out-of-process backends (set for SDK_REFS nodes).
    node_module: str
    inputs: Optional[dict]
    dynamic_prompt: Any
    def __init__(
        self,
        prompt_id: str,
        node_id: str,
        node_type: str,
        tier: str = ...,
        permissions: tuple[str, ...] = ...,
        required_weights: tuple[HuggingFaceWeight, ...] = ...,
        node_module: str = ...,
        inputs: Optional[dict] = ...,
        dynamic_prompt: Any = ...,
    ) -> None: ...

@runtime_checkable
class RefResolver(Protocol):
    async def create(self, kind: str, obj: Any) -> Ref: ...
    async def resolve(self, ref: Ref) -> Any: ...
    async def release(self, ref: Ref) -> None: ...

@runtime_checkable
class OpsProvider(Protocol):
    """Engine-side operations on assets. Generic dispatch: the op vocabulary is
    data, extensible by an overlay without changing this contract."""

    async def apply(self, op: str, image: ImageRef, params: dict) -> ImageRef: ...
    def supports(self, op: str) -> bool: ...

class OpNotSupported(NotImplementedError):
    op: str
    capability: str

class Runtime:
    """Per-node host binding: the ref table, brokered ctx, and ops the node
    executes against. Handed to ``ExecutionBackend.dispatch`` so out-of-process
    backends can broker guest calls against the same table."""

    refs: RefResolver
    ctx: Context
    ops: OpsProvider

@runtime_checkable
class ExecutionBackend(Protocol):
    async def dispatch(
        self,
        plan: ExecutionPlan,
        local_call: Callable[[], Awaitable[Any]],
        runtime: Optional[Runtime] = ...,
    ) -> Any: ...

@runtime_checkable
class CtxProvider(Protocol):
    def build(self, plan: ExecutionPlan) -> Context: ...

class _Providers:
    execution_backend: ExecutionBackend
    ctx_provider: CtxProvider
    ops_provider: OpsProvider
    ref_resolver_factory: Callable[[], RefResolver]
    @property
    def overlay_active(self) -> bool: ...
    def register_execution_backend(self, impl: ExecutionBackend) -> None: ...
    def register_ctx_provider(self, impl: CtxProvider) -> None: ...
    def register_ops_provider(self, impl: OpsProvider) -> None: ...
    def register_ref_resolver_factory(
        self, factory: Callable[[], RefResolver]
    ) -> None: ...

providers: _Providers
