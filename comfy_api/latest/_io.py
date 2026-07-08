from __future__ import annotations

import copy
import inspect
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Literal, TypedDict, TypeVar, TYPE_CHECKING
from typing_extensions import NotRequired, final

# used for type hinting
import torch

if TYPE_CHECKING:
    from spandrel import ImageModelDescriptor
    from comfy.clip_vision import ClipVisionModel
    from comfy.clip_vision import Output as ClipVisionOutput_
    from comfy.bg_removal_model import BackgroundRemovalModel
    from comfy.controlnet import ControlNet
    from comfy.hooks import HookGroup, HookKeyframeGroup
    from comfy.model_patcher import ModelPatcher
    from comfy.samplers import CFGGuider, Sampler
    from comfy.sd import CLIP, VAE
    from comfy.sd import StyleModel as StyleModel_
    from comfy_api.input import VideoInput, CurveInput as CurveInput_
from comfy_api.internal import (_ComfyNodeInternal, _NodeOutputInternal, classproperty, copy_class, first_real_override, is_class,
    prune_dict, shallow_clone_class)
from comfy_execution.graph_utils import ExecutionBlocker
from ._util import MESH, VOXEL, SPLAT, SVG as _SVG, File3D


class FolderType(str, Enum):
    input = "input"
    output = "output"
    temp = "temp"


class UploadType(str, Enum):
    image = "image_upload"
    audio = "audio_upload"
    video = "video_upload"
    model = "file_upload"


class RemoteOptions:
    def __init__(self, route: str, refresh_button: bool, control_after_refresh: Literal["first", "last"]="first",
                 timeout: int=None, max_retries: int=None, refresh: int=None):
        self.route = route
        """The route to the remote source."""
        self.refresh_button = refresh_button
        """Specifies whether to show a refresh button in the UI below the widget."""
        self.control_after_refresh = control_after_refresh
        """Specifies the control after the refresh button is clicked. If "first", the first item will be automatically selected, and so on."""
        self.timeout = timeout
        """The maximum amount of time to wait for a response from the remote source in milliseconds."""
        self.max_retries = max_retries
        """The maximum number of retries before aborting the request."""
        self.refresh = refresh
        """The TTL of the remote input's value in milliseconds. Specifies the interval at which the remote input's value is refreshed."""

    def as_dict(self):
        return prune_dict({
            "route": self.route,
            "refresh_button": self.refresh_button,
            "control_after_refresh": self.control_after_refresh,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "refresh": self.refresh,
        })


class NumberDisplay(str, Enum):
    number = "number"
    slider = "slider"
    gradient_slider = "gradientslider"


class ControlAfterGenerate(str, Enum):
    fixed = "fixed"
    increment = "increment"
    decrement = "decrement"
    randomize = "randomize"

class _ComfyType(ABC):
    Type = Any
    io_type: str = None

# NOTE: this is a workaround to make the decorator return the correct type
T = TypeVar("T", bound=type)
def comfytype(io_type: str, **kwargs):
    '''
    Decorator to mark nested classes as ComfyType; io_type will be bound to the class.

    A ComfyType may have the following attributes:
    - Type = <type hint here>
    - class Input(Input): ...
    - class Output(Output): ...
    '''
    def decorator(cls: T) -> T:
        if isinstance(cls, _ComfyType) or issubclass(cls, _ComfyType):
            # clone Input and Output classes to avoid modifying the original class
            new_cls = cls
            if hasattr(new_cls, "Input"):
                new_cls.Input = copy_class(new_cls.Input)
            if hasattr(new_cls, "Output"):
                new_cls.Output = copy_class(new_cls.Output)
        else:
            # copy class attributes except for special ones that shouldn't be in type()
            cls_dict = {
                k: v for k, v in cls.__dict__.items()
                if k not in ('__dict__', '__weakref__', '__module__', '__doc__')
            }
            # new class
            new_cls: ComfyTypeIO = type(
                cls.__name__,
                (cls, ComfyTypeIO),
                cls_dict
            )
            # metadata preservation
            new_cls.__module__ = cls.__module__
            new_cls.__doc__ = cls.__doc__
            # assign ComfyType attributes, if needed
        new_cls.io_type = io_type
        if hasattr(new_cls, "Input") and new_cls.Input is not None:
            new_cls.Input.Parent = new_cls
        if hasattr(new_cls, "Output") and new_cls.Output is not None:
            new_cls.Output.Parent = new_cls
        return new_cls
    return decorator

def Custom(io_type: str) -> type[ComfyTypeIO]:
    '''Create a ComfyType for a custom io_type.'''
    @comfytype(io_type=io_type)
    class CustomComfyType(ComfyTypeIO):
        ...
    return CustomComfyType

class _IO_V3:
    '''
    Base class for V3 Inputs and Outputs.
    '''
    Parent: _ComfyType = None

    def __init__(self):
        pass

    def validate(self):
        pass

    @property
    def io_type(self):
        return self.Parent.io_type

    @property
    def Type(self):
        return self.Parent.Type

class Input(_IO_V3):
    '''
    Base class for a V3 Input.
    '''
    def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None, extra_dict=None, raw_link: bool=None, advanced: bool=None):
        super().__init__()
        self.id = id
        self.display_name = display_name
        self.optional = optional
        self.tooltip = tooltip
        self.lazy = lazy
        self.extra_dict = extra_dict if extra_dict is not None else {}
        self.rawLink = raw_link
        self.advanced = advanced

    def as_dict(self):
        return prune_dict({
            "display_name": self.display_name,
            "optional": self.optional,
            "tooltip": self.tooltip,
            "lazy": self.lazy,
            "rawLink": self.rawLink,
            "advanced": self.advanced,
        }) | prune_dict(self.extra_dict)

    def get_io_type(self):
        return self.io_type

    def get_all(self) -> list[Input]:
        return [self]

class WidgetInput(Input):
    '''
    Base class for a V3 Input with widget.
    '''
    def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                 default: Any=None,
                 socketless: bool=None, widget_type: str=None, force_input: bool=None, extra_dict=None, raw_link: bool=None, advanced: bool=None):
        super().__init__(id, display_name, optional, tooltip, lazy, extra_dict, raw_link, advanced)
        self.default = default
        self.socketless = socketless
        self.widget_type = widget_type
        self.force_input = force_input

    def as_dict(self):
        return super().as_dict() | prune_dict({
            "default": self.default,
            "socketless": self.socketless,
            "widgetType": self.widget_type,
            "forceInput": self.force_input,
        })

    def get_io_type(self):
        return self.widget_type if self.widget_type is not None else super().get_io_type()


class Output(_IO_V3):
    def __init__(self, id: str=None, display_name: str=None, tooltip: str=None,
                 is_output_list=False):
        self.id = id
        self.display_name = display_name if display_name else id
        self.tooltip = tooltip
        self.is_output_list = is_output_list

    def as_dict(self):
        display_name = self.display_name if self.display_name else self.id
        return prune_dict({
            "display_name": display_name,
            "tooltip": self.tooltip,
            "is_output_list": self.is_output_list,
        })

    def get_io_type(self):
        return self.io_type


class ComfyTypeI(_ComfyType):
    '''ComfyType subclass that only has a default Input class - intended for types that only have Inputs.'''
    class Input(Input):
        ...

class ComfyTypeIO(ComfyTypeI):
    '''ComfyType subclass that has default Input and Output classes; useful for types with both Inputs and Outputs.'''
    class Output(Output):
        ...


@comfytype(io_type="BOOLEAN")
class Boolean(ComfyTypeIO):
    Type = bool

    class Input(WidgetInput):
        '''Boolean input.'''
        def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                    default: bool=None, label_on: str=None, label_off: str=None,
                    socketless: bool=None, force_input: bool=None, extra_dict=None, raw_link: bool=None, advanced: bool=None):
            super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, None, force_input, extra_dict, raw_link, advanced)
            self.label_on = label_on
            self.label_off = label_off
            self.default: bool

        def as_dict(self):
            return super().as_dict() | prune_dict({
                "label_on": self.label_on,
                "label_off": self.label_off,
            })

@comfytype(io_type="INT")
class Int(ComfyTypeIO):
    Type = int

    class Input(WidgetInput):
        '''Integer input.'''
        def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                    default: int=None, min: int=None, max: int=None, step: int=None, control_after_generate: bool | ControlAfterGenerate=None,
                    display_mode: NumberDisplay=None, socketless: bool=None, force_input: bool=None, extra_dict=None, raw_link: bool=None, advanced: bool=None):
            super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, None, force_input, extra_dict, raw_link, advanced)
            self.min = min
            self.max = max
            self.step = step
            self.control_after_generate = control_after_generate
            self.display_mode = display_mode
            self.default: int

        def as_dict(self):
            return super().as_dict() | prune_dict({
                "min": self.min,
                "max": self.max,
                "step": self.step,
                "control_after_generate": self.control_after_generate,
                "display": self.display_mode.value if self.display_mode else None,
            })

@comfytype(io_type="FLOAT")
class Float(ComfyTypeIO):
    Type = float

    class Input(WidgetInput):
        '''Float input.'''
        def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                    default: float=None, min: float=None, max: float=None, step: float=None, round: float=None,
                    display_mode: NumberDisplay=None, gradient_stops: list[dict]=None,
                    socketless: bool=None, force_input: bool=None, extra_dict=None, raw_link: bool=None, advanced: bool=None):
            super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, None, force_input, extra_dict, raw_link, advanced)
            self.min = min
            self.max = max
            self.step = step
            self.round = round
            self.display_mode = display_mode
            self.gradient_stops = gradient_stops
            self.default: float

        def as_dict(self):
            return super().as_dict() | prune_dict({
                "min": self.min,
                "max": self.max,
                "step": self.step,
                "round": self.round,
                "display": self.display_mode,
                "gradient_stops": self.gradient_stops,
            })

@comfytype(io_type="STRING")
class String(ComfyTypeIO):
    Type = str

    class Input(WidgetInput):
        '''String input.'''
        def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                    multiline=False, placeholder: str=None, default: str=None, dynamic_prompts: bool=None,
                    socketless: bool=None, force_input: bool=None, extra_dict=None, raw_link: bool=None, advanced: bool=None):
            super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, None, force_input, extra_dict, raw_link, advanced)
            self.multiline = multiline
            self.placeholder = placeholder
            self.dynamic_prompts = dynamic_prompts
            self.default: str

        def as_dict(self):
            return super().as_dict() | prune_dict({
                "multiline": self.multiline,
                "placeholder": self.placeholder,
                "dynamicPrompts": self.dynamic_prompts,
            })

@comfytype(io_type="COMBO")
class Combo(ComfyTypeIO):
    Type = str
    class Input(WidgetInput):
        """Combo input (dropdown)."""
        Type = str
        def __init__(
            self,
            id: str,
            options: list[str] | list[int] | type[Enum] = None,
            display_name: str=None,
            optional=False,
            tooltip: str=None,
            lazy: bool=None,
            default: str | int | Enum = None,
            control_after_generate: bool | ControlAfterGenerate=None,
            upload: UploadType=None,
            image_folder: FolderType=None,
            remote: RemoteOptions=None,
            socketless: bool=None,
            extra_dict=None,
            raw_link: bool=None,
            advanced: bool=None,
        ):
            if isinstance(options, type) and issubclass(options, Enum):
                options = [v.value for v in options]
            if isinstance(default, Enum):
                default = default.value
            super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, None, None, extra_dict, raw_link, advanced)
            self.multiselect = False
            self.options = options
            self.control_after_generate = control_after_generate
            self.upload = upload
            self.image_folder = image_folder
            self.remote = remote
            self.default: str

        def as_dict(self):
            return super().as_dict() | prune_dict({
                "multiselect": self.multiselect,
                "options": self.options,
                "control_after_generate": self.control_after_generate,
                **({self.upload.value: True} if self.upload is not None else {}),
                "image_folder": self.image_folder.value if self.image_folder else None,
                "remote": self.remote.as_dict() if self.remote else None,
            })

    class Output(Output):
        def __init__(self, id: str=None, display_name: str=None, options: list[str]=None, tooltip: str=None, is_output_list=False):
            super().__init__(id, display_name, tooltip, is_output_list)
            self.options = options if options is not None else []

@comfytype(io_type="COMBO")
class MultiCombo(ComfyTypeI):
    '''Multiselect Combo input (dropdown for selecting potentially more than one value).'''
    Type = list[str]
    class Input(Combo.Input):
        def __init__(self, id: str, options: list[str], display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                    default: list[str]=None, placeholder: str=None, chip: bool=None, control_after_generate: bool | ControlAfterGenerate=None,
                    socketless: bool=None, extra_dict=None, raw_link: bool=None, advanced: bool=None):
            super().__init__(id, options, display_name, optional, tooltip, lazy, default, control_after_generate, socketless=socketless, extra_dict=extra_dict, raw_link=raw_link, advanced=advanced)
            self.multiselect = True
            self.placeholder = placeholder
            self.chip = chip
            self.default: list[str]

        def as_dict(self):
            # Frontend expects `multi_select` to be an object config (not a boolean).
            # Keep top-level `multiselect` from Combo.Input for backwards compatibility.
            return super().as_dict() | prune_dict({
                "multi_select": prune_dict({
                    "placeholder": self.placeholder,
                    "chip": self.chip,
                }),
            })

@comfytype(io_type="IMAGE")
class Image(ComfyTypeIO):
    Type = torch.Tensor


@comfytype(io_type="WAN_CAMERA_EMBEDDING")
class WanCameraEmbedding(ComfyTypeIO):
    Type = torch.Tensor


@comfytype(io_type="WEBCAM")
class Webcam(ComfyTypeIO):
    Type = str

    class Input(WidgetInput):
        """Webcam input."""
        Type = str
        def __init__(
                self, id: str, display_name: str=None, optional=False,
                tooltip: str=None, lazy: bool=None, default: str=None, socketless: bool=None, extra_dict=None, raw_link: bool=None, advanced: bool=None
        ):
            super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, None, None, extra_dict, raw_link, advanced)


@comfytype(io_type="MASK")
class Mask(ComfyTypeIO):
    Type = torch.Tensor

@comfytype(io_type="LATENT")
class Latent(ComfyTypeIO):
    '''Latents are stored as a dictionary.'''
    class LatentDict(TypedDict):
        samples: torch.Tensor
        '''Latent tensors.'''
        noise_mask: NotRequired[torch.Tensor]
        batch_index: NotRequired[list[int]]
        type: NotRequired[str]
        '''Only needed if dealing with these types: audio, hunyuan3dv2'''
    Type = LatentDict

@comfytype(io_type="CONDITIONING")
class Conditioning(ComfyTypeIO):
    class PooledDict(TypedDict):
        pooled_output: torch.Tensor
        '''Pooled output from CLIP.'''
        control: NotRequired[ControlNet]
        '''ControlNet to apply to conditioning.'''
        control_apply_to_uncond: NotRequired[bool]
        '''Whether to apply ControlNet to matching negative conditioning at sample time, if applicable.'''
        cross_attn_controlnet: NotRequired[torch.Tensor]
        '''CrossAttn from CLIP to use for controlnet only.'''
        pooled_output_controlnet: NotRequired[torch.Tensor]
        '''Pooled output from CLIP to use for controlnet only.'''
        gligen: NotRequired[tuple[str, Gligen, list[tuple[torch.Tensor, int, ...]]]]
        '''GLIGEN to apply to conditioning.'''
        area: NotRequired[tuple[int, ...] | tuple[str, float, ...]]
        '''Set area of conditioning. First half of values apply to dimensions, the second half apply to coordinates.
        By default, the dimensions are based on total pixel amount, but the first value can be set to "percentage" to use a percentage of the image size instead.

        (1024, 1024, 0, 0) would apply conditioning to the top-left 1024x1024 pixels.

        ("percentage", 0.5, 0.5, 0, 0) would apply conditioning to the top-left 50% of the image.''' # TODO: verify its actually top-left
        strength: NotRequired[float]
        '''Strength of conditioning. Default strength is 1.0.'''
        mask: NotRequired[torch.Tensor]
        '''Mask to apply conditioning to.'''
        mask_strength: NotRequired[float]
        '''Strength of conditioning mask. Default strength is 1.0.'''
        set_area_to_bounds: NotRequired[bool]
        '''Whether conditioning mask should determine bounds of area - if set to false, latents are sampled at full resolution and result is applied in mask.'''
        concat_latent_image: NotRequired[torch.Tensor]
        '''Used for inpainting and specific models.'''
        concat_mask: NotRequired[torch.Tensor]
        '''Used for inpainting and specific models.'''
        concat_image: NotRequired[torch.Tensor]
        '''Used by SD_4XUpscale_Conditioning.'''
        noise_augmentation: NotRequired[float]
        '''Used by SD_4XUpscale_Conditioning.'''
        hooks: NotRequired[HookGroup]
        '''Applies hooks to conditioning.'''
        default: NotRequired[bool]
        '''Whether to this conditioning is 'default'; default conditioning gets applied to any areas of the image that have no masks/areas applied, assuming at least one area/mask is present during sampling.'''
        start_percent: NotRequired[float]
        '''Determines relative step to begin applying conditioning, expressed as a float between 0.0 and 1.0.'''
        end_percent: NotRequired[float]
        '''Determines relative step to end applying conditioning, expressed as a float between 0.0 and 1.0.'''
        clip_start_percent: NotRequired[float]
        '''Internal variable for conditioning scheduling - start of application, expressed as a float between 0.0 and 1.0.'''
        clip_end_percent: NotRequired[float]
        '''Internal variable for conditioning scheduling - end of application, expressed as a float between 0.0 and 1.0.'''
        attention_mask: NotRequired[torch.Tensor]
        '''Masks text conditioning; used by StyleModel among others.'''
        attention_mask_img_shape: NotRequired[tuple[int, ...]]
        '''Masks text conditioning; used by StyleModel among others.'''
        unclip_conditioning: NotRequired[list[dict]]
        '''Used by unCLIP.'''
        conditioning_lyrics: NotRequired[torch.Tensor]
        '''Used by AceT5Model.'''
        seconds_start: NotRequired[float]
        '''Used by StableAudio.'''
        seconds_total: NotRequired[float]
        '''Used by StableAudio.'''
        lyrics_strength: NotRequired[float]
        '''Used by AceStepAudio.'''
        width: NotRequired[int]
        '''Used by certain models (e.g. CLIPTextEncodeSDXL/Refiner, PixArtAlpha).'''
        height: NotRequired[int]
        '''Used by certain models (e.g. CLIPTextEncodeSDXL/Refiner, PixArtAlpha).'''
        aesthetic_score: NotRequired[float]
        '''Used by CLIPTextEncodeSDXL/Refiner.'''
        crop_w: NotRequired[int]
        '''Used by CLIPTextEncodeSDXL.'''
        crop_h: NotRequired[int]
        '''Used by CLIPTextEncodeSDXL.'''
        target_width: NotRequired[int]
        '''Used by CLIPTextEncodeSDXL.'''
        target_height: NotRequired[int]
        '''Used by CLIPTextEncodeSDXL.'''
        reference_latents: NotRequired[list[torch.Tensor]]
        '''Used by ReferenceLatent.'''
        guidance: NotRequired[float]
        '''Used by Flux-like models with guidance embed.'''
        guiding_frame_index: NotRequired[int]
        '''Used by Hunyuan ImageToVideo.'''
        ref_latent: NotRequired[torch.Tensor]
        '''Used by Hunyuan ImageToVideo.'''
        keyframe_idxs: NotRequired[list[int]]
        '''Used by LTXV.'''
        frame_rate: NotRequired[float]
        '''Used by LTXV.'''
        stable_cascade_prior: NotRequired[torch.Tensor]
        '''Used by StableCascade.'''
        elevation: NotRequired[list[float]]
        '''Used by SV3D.'''
        azimuth: NotRequired[list[float]]
        '''Used by SV3D.'''
        motion_bucket_id: NotRequired[int]
        '''Used by SVD-like models.'''
        fps: NotRequired[int]
        '''Used by SVD-like models.'''
        augmentation_level: NotRequired[float]
        '''Used by SVD-like models.'''
        clip_vision_output: NotRequired[ClipVisionOutput_]
        '''Used by WAN-like models.'''
        vace_frames: NotRequired[torch.Tensor]
        '''Used by WAN VACE.'''
        vace_mask: NotRequired[torch.Tensor]
        '''Used by WAN VACE.'''
        vace_strength: NotRequired[float]
        '''Used by WAN VACE.'''
        camera_conditions: NotRequired[Any] # TODO: assign proper type once defined
        '''Used by WAN Camera.'''
        time_dim_concat: NotRequired[torch.Tensor]
        '''Used by WAN Phantom Subject.'''
        time_dim_replace: NotRequired[torch.Tensor]
        '''Used by Kandinsky5 I2V.'''

    CondList = list[tuple[torch.Tensor, PooledDict]]
    Type = CondList

@comfytype(io_type="SAMPLER")
class Sampler(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = Sampler

@comfytype(io_type="SIGMAS")
class Sigmas(ComfyTypeIO):
    Type = torch.Tensor

@comfytype(io_type="NOISE")
class Noise(ComfyTypeIO):
    Type = torch.Tensor

@comfytype(io_type="GUIDER")
class Guider(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = CFGGuider

@comfytype(io_type="CLIP")
class Clip(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = CLIP

@comfytype(io_type="CONTROL_NET")
class ControlNet(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = ControlNet

@comfytype(io_type="VAE")
class Vae(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = VAE

@comfytype(io_type="MODEL")
class Model(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = ModelPatcher

@comfytype(io_type="BACKGROUND_REMOVAL")
class BackgroundRemoval(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = BackgroundRemovalModel

@comfytype(io_type="CLIP_VISION")
class ClipVision(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = ClipVisionModel

@comfytype(io_type="CLIP_VISION_OUTPUT")
class ClipVisionOutput(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = ClipVisionOutput_

@comfytype(io_type="STYLE_MODEL")
class StyleModel(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = StyleModel_

@comfytype(io_type="GLIGEN")
class Gligen(ComfyTypeIO):
    '''ModelPatcher that wraps around a 'Gligen' model.'''
    if TYPE_CHECKING:
        Type = ModelPatcher

@comfytype(io_type="UPSCALE_MODEL")
class UpscaleModel(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = ImageModelDescriptor

@comfytype(io_type="LATENT_UPSCALE_MODEL")
class LatentUpscaleModel(ComfyTypeIO):
    Type = Any

@comfytype(io_type="AUDIO")
class Audio(ComfyTypeIO):
    class AudioDict(TypedDict):
        waveform: torch.Tensor
        sampler_rate: int
    Type = AudioDict

@comfytype(io_type="VIDEO")
class Video(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = VideoInput

@comfytype(io_type="SVG")
class SVG(ComfyTypeIO):
    Type = _SVG

@comfytype(io_type="LORA_MODEL")
class LoraModel(ComfyTypeIO):
    Type = dict[str, torch.Tensor]

@comfytype(io_type="LOSS_MAP")
class LossMap(ComfyTypeIO):
    class LossMapDict(TypedDict):
        loss: list[torch.Tensor]
    Type = LossMapDict

@comfytype(io_type="VOXEL")
class Voxel(ComfyTypeIO):
    Type = VOXEL

@comfytype(io_type="MESH")
class Mesh(ComfyTypeIO):
    Type = MESH

@comfytype(io_type="SPLAT")
class Splat(ComfyTypeIO):
    Type = SPLAT


@comfytype(io_type="FILE_3D")
class File3DAny(ComfyTypeIO):
    """General 3D file type - accepts any supported 3D format."""
    Type = File3D


@comfytype(io_type="FILE_3D_GLB")
class File3DGLB(ComfyTypeIO):
    """GLB format 3D file - binary glTF, best for web and cross-platform."""
    Type = File3D


@comfytype(io_type="FILE_3D_GLTF")
class File3DGLTF(ComfyTypeIO):
    """GLTF format 3D file - JSON-based glTF with external resources."""
    Type = File3D


@comfytype(io_type="FILE_3D_FBX")
class File3DFBX(ComfyTypeIO):
    """FBX format 3D file - best for game engines and animation."""
    Type = File3D


@comfytype(io_type="FILE_3D_OBJ")
class File3DOBJ(ComfyTypeIO):
    """OBJ format 3D file - simple geometry format."""
    Type = File3D


@comfytype(io_type="FILE_3D_STL")
class File3DSTL(ComfyTypeIO):
    """STL format 3D file - best for 3D printing."""
    Type = File3D


@comfytype(io_type="FILE_3D_USDZ")
class File3DUSDZ(ComfyTypeIO):
    """USDZ format 3D file - Apple AR format."""
    Type = File3D


@comfytype(io_type="FILE_3D_PLY")
class File3DPLY(ComfyTypeIO):
    """PLY format 3D file - point cloud or Gaussian splat."""
    Type = File3D


@comfytype(io_type="FILE_3D_SPLAT")
class File3DSPLAT(ComfyTypeIO):
    """SPLAT format 3D file - 3D Gaussian splat."""
    Type = File3D


@comfytype(io_type="FILE_3D_SPZ")
class File3DSPZ(ComfyTypeIO):
    """SPZ format 3D file - compressed 3D Gaussian splat."""
    Type = File3D


@comfytype(io_type="FILE_3D_KSPLAT")
class File3DKSPLAT(ComfyTypeIO):
    """KSPLAT format 3D file - 3D Gaussian splat."""
    Type = File3D


@comfytype(io_type="FILE_3D_SPLAT_ANY")
class File3DSplatAny(ComfyTypeIO):
    """General 3D Gaussian splat file type - accepts any supported splat container (.ply / .spz / .splat / .ksplat)."""
    Type = File3D


@comfytype(io_type="FILE_3D_POINT_CLOUD_ANY")
class File3DPointCloudAny(ComfyTypeIO):
    """General point cloud file type - accepts any supported point cloud container (currently .ply)."""
    Type = File3D


@comfytype(io_type="HOOKS")
class Hooks(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = HookGroup

@comfytype(io_type="HOOK_KEYFRAMES")
class HookKeyframes(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = HookKeyframeGroup

@comfytype(io_type="TIMESTEPS_RANGE")
class TimestepsRange(ComfyTypeIO):
    '''Range defined by start and endpoint, between 0.0 and 1.0.'''
    Type = tuple[int, int]

@comfytype(io_type="LATENT_OPERATION")
class LatentOperation(ComfyTypeIO):
    Type = Callable[[torch.Tensor], torch.Tensor]

@comfytype(io_type="FLOW_CONTROL")
class FlowControl(ComfyTypeIO):
    # NOTE: only used in testing_nodes right now
    Type = tuple[str, Any]

@comfytype(io_type="ACCUMULATION")
class Accumulation(ComfyTypeIO):
    # NOTE: only used in testing_nodes right now
    class AccumulationDict(TypedDict):
        accum: list[Any]
    Type = AccumulationDict


@comfytype(io_type="LOAD3D_CAMERA")
class Load3DCamera(ComfyTypeIO):
    class CameraInfo(TypedDict):
        # Coordinate system: right-handed, Y-up, camera looks down -Z
        position: dict[str, float | int]  # scene units
        target: dict[str, float | int]  # scene units; OrbitControls focus point
        zoom: float | int  # dimensionless, 1 = 100%
        cameraType: str  # 'perspective' | 'orthographic'
        quaternion: NotRequired[dict[str, float | int]]  # normalized, dimensionless; camera world rotation
        fov: NotRequired[float | int]  # degrees, vertical FOV (perspective only)
        aspect: NotRequired[float | int]  # width / height (perspective only)
        near: NotRequired[float | int]  # scene units
        far: NotRequired[float | int]  # scene units
        frustum: NotRequired[dict[str, float | int]]  # orthographic only: {left, right, top, bottom} in scene units

    Type = CameraInfo


@comfytype(io_type="LOAD3D_MODEL_INFO")
class Load3DModelInfo(ComfyTypeIO):
    class Model3DTransform(TypedDict):
        # Coordinate system: right-handed, Y-up, world space
        position: dict[str, float | int]  # scene units
        quaternion: dict[str, float | int]  # normalized, dimensionless; world rotation
        scale: dict[str, float | int]  # dimensionless multiplier

    Type = list[Model3DTransform]


@comfytype(io_type="LOAD_3D")
class Load3D(ComfyTypeIO):
    """3D models are stored as a dictionary."""
    class Model3DDict(TypedDict):
        image: str
        mask: str
        normal: str
        camera_info: Load3DCamera.CameraInfo
        recording: NotRequired[str]
        model_3d_info: NotRequired[list[Load3DModelInfo.Model3DTransform]]

    Type = Model3DDict


@comfytype(io_type="LOAD_3D_ANIMATION")
class Load3DAnimation(Load3D):
    ...


@comfytype(io_type="PHOTOMAKER")
class Photomaker(ComfyTypeIO):
    Type = Any


@comfytype(io_type="POINT")
class Point(ComfyTypeIO):
    Type = Any # NOTE: I couldn't find any references in core code to POINT io_type. Does this exist?

@comfytype(io_type="FACE_ANALYSIS")
class FaceAnalysis(ComfyTypeIO):
    Type = Any # NOTE: I couldn't find any references in core code to POINT io_type. Does this exist?

@comfytype(io_type="BBOX")
class BBOX(ComfyTypeIO):
    Type = Any # NOTE: I couldn't find any references in core code to POINT io_type. Does this exist?

@comfytype(io_type="SEGS")
class SEGS(ComfyTypeIO):
    Type = Any # NOTE: I couldn't find any references in core code to POINT io_type. Does this exist?

@comfytype(io_type="*")
class AnyType(ComfyTypeIO):
    Type = Any

@comfytype(io_type="MODEL_PATCH")
class ModelPatch(ComfyTypeIO):
    Type = Any

@comfytype(io_type="AUDIO_ENCODER")
class AudioEncoder(ComfyTypeIO):
    Type = Any

@comfytype(io_type="AUDIO_ENCODER_OUTPUT")
class AudioEncoderOutput(ComfyTypeIO):
    Type = Any

@comfytype(io_type="TRACKS")
class Tracks(ComfyTypeIO):
    class TrackDict(TypedDict):
        track_path: torch.Tensor
        track_visibility: torch.Tensor
    Type = TrackDict

@comfytype(io_type="DICT")
class Dict(ComfyTypeIO):
    Type = dict

@comfytype(io_type="ARRAY")
class Array(ComfyTypeIO):
    Type = list

@comfytype(io_type="COMFY_MULTITYPED_V3")
class MultiType:
    Type = Any
    class Input(Input):
        '''
        Input that permits more than one input type; if `id` is an instance of `ComfyType.Input`, then that input will be used to create a widget (if applicable) with overridden values.
        '''
        def __init__(self, id: str | Input, types: list[type[_ComfyType] | _ComfyType], display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None, extra_dict=None, raw_link: bool=None, advanced: bool=None):
            # if id is an Input, then use that Input with overridden values
            self.input_override = None
            if isinstance(id, Input):
                self.input_override = copy.copy(id)
                optional = id.optional if id.optional is True else optional
                tooltip = id.tooltip if id.tooltip is not None else tooltip
                display_name = id.display_name if id.display_name is not None else display_name
                lazy = id.lazy if id.lazy is not None else lazy
                id = id.id
                # if is a widget input, make sure widget_type is set appropriately
                if isinstance(self.input_override, WidgetInput):
                    self.input_override.widget_type = self.input_override.get_io_type()
            super().__init__(id, display_name, optional, tooltip, lazy, extra_dict, raw_link, advanced)
            self._io_types = types

        @property
        def io_types(self) -> list[type[Input]]:
            '''
            Returns list of Input class types permitted.
            '''
            io_types = []
            for x in self._io_types:
                if not is_class(x):
                    io_types.append(type(x))
                else:
                    io_types.append(x)
            return io_types

        def get_io_type(self):
            # ensure types are unique and order is preserved
            str_types = [x.io_type for x in self.io_types]
            if self.input_override is not None:
                str_types.insert(0, self.input_override.get_io_type())
            return ",".join(list(dict.fromkeys(str_types)))

        def as_dict(self):
            if self.input_override is not None:
                return self.input_override.as_dict() | super().as_dict()
            else:
                return super().as_dict()

@comfytype(io_type="COMFY_MATCHTYPE_V3")
class MatchType(ComfyTypeIO):
    class Template:
        def __init__(self, template_id: str, allowed_types: _ComfyType | list[_ComfyType] = AnyType):
            self.template_id = template_id
            # account for syntactic sugar
            if not isinstance(allowed_types, Iterable):
                allowed_types = [allowed_types]
            for t in allowed_types:
                if not isinstance(t, type):
                    if not isinstance(t, _ComfyType):
                        raise ValueError(f"Allowed types must be a ComfyType or a list of ComfyTypes, got {t.__class__.__name__}")
                else:
                    if not issubclass(t, _ComfyType):
                        raise ValueError(f"Allowed types must be a ComfyType or a list of ComfyTypes, got {t.__name__}")
            self.allowed_types = allowed_types

        def as_dict(self):
            return {
                "template_id": self.template_id,
                "allowed_types": ",".join([t.io_type for t in self.allowed_types]),
            }

    class Input(Input):
        def __init__(self, id: str, template: MatchType.Template,
                    display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None, extra_dict=None, raw_link: bool=None, advanced: bool=None):
            super().__init__(id, display_name, optional, tooltip, lazy, extra_dict, raw_link, advanced)
            self.template = template

        def as_dict(self):
            return super().as_dict() | prune_dict({
                "template": self.template.as_dict(),
            })

    class Output(Output):
        def __init__(self, template: MatchType.Template, id: str=None, display_name: str=None, tooltip: str=None,
                     is_output_list=False):
            if not id and not display_name:
                display_name = "MATCHTYPE"
            super().__init__(id, display_name, tooltip, is_output_list)
            self.template = template

        def as_dict(self):
            return super().as_dict() | prune_dict({
                "template": self.template.as_dict(),
            })

class DynamicInput(Input, ABC):
    '''
    Abstract class for dynamic input registration.
    '''
    pass


class DynamicOutput(Output, ABC):
    '''
    Abstract class for dynamic output registration.
    '''
    pass


def handle_prefix(prefix_list: list[str] | None, id: str | None = None) -> list[str]:
    if prefix_list is None:
        prefix_list = []
    if id is not None:
        prefix_list = prefix_list + [id]
    return prefix_list

def finalize_prefix(prefix_list: list[str] | None, id: str | None = None) -> str:
    assert not (prefix_list is None and id is None)
    if prefix_list is None:
        return id
    elif id is not None:
        prefix_list = prefix_list + [id]
    return ".".join(prefix_list)

@comfytype(io_type="COMFY_AUTOGROW_V3")
class Autogrow(ComfyTypeI):
    Type = dict[str, Any]
    _MaxNames = 100  # NOTE: max 100 names for sanity

    class _AutogrowTemplate:
        def __init__(self, input: Input):
            # dynamic inputs are not allowed as the template input
            assert(not isinstance(input, DynamicInput))
            self.input = copy.copy(input)
            if isinstance(self.input, WidgetInput):
                self.input.force_input = True
            self.names: list[str] = []
            self.cached_inputs = {}

        def _create_input(self, input: Input, name: str):
            new_input = copy.copy(self.input)
            new_input.id = name
            return new_input

        def _create_cached_inputs(self):
            for name in self.names:
                self.cached_inputs[name] = self._create_input(self.input, name)

        def get_all(self) -> list[Input]:
            return list(self.cached_inputs.values())

        def as_dict(self):
            return prune_dict({
                "input": create_input_dict_v1([self.input]),
            })

        def validate(self):
            self.input.validate()

    class TemplatePrefix(_AutogrowTemplate):
        def __init__(self, input: Input, prefix: str, min: int=1, max: int=10):
            super().__init__(input)
            self.prefix = prefix
            assert(min >= 0)
            assert(max >= 1)
            assert(max <= Autogrow._MaxNames)
            self.min = min
            self.max = max
            self.names = [f"{self.prefix}{i}" for i in range(self.max)]
            self._create_cached_inputs()

        def as_dict(self):
            return super().as_dict() | prune_dict({
                "prefix": self.prefix,
                "min": self.min,
                "max": self.max,
            })

    class TemplateNames(_AutogrowTemplate):
        def __init__(self, input: Input, names: list[str], min: int=1):
            super().__init__(input)
            self.names = names[:Autogrow._MaxNames]
            assert(min >= 0)
            self.min = min
            self._create_cached_inputs()

        def as_dict(self):
            return super().as_dict() | prune_dict({
                "names": self.names,
                "min": self.min,
            })

    class Input(DynamicInput):
        def __init__(self, id: str, template: Autogrow.TemplatePrefix | Autogrow.TemplateNames,
                     display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None, extra_dict=None):
            super().__init__(id, display_name, optional, tooltip, lazy, extra_dict)
            self.template = template

        def as_dict(self):
            return super().as_dict() | prune_dict({
                "template": self.template.as_dict(),
            })

        def get_all(self) -> list[Input]:
            return [self] + self.template.get_all()

        def validate(self):
            self.template.validate()

    @staticmethod
    def _expand_schema_for_dynamic(out_dict: dict[str, Any], live_inputs: dict[str, Any], value: tuple[str, dict[str, Any]], input_type: str, curr_prefix: list[str] | None):
        # NOTE: purposely do not include self in out_dict; instead use only the template inputs
        # need to figure out names based on template type
        is_names = ("names" in value[1]["template"])
        is_prefix = ("prefix" in value[1]["template"])
        input = value[1]["template"]["input"]
        if is_names:
            min = value[1]["template"]["min"]
            names = value[1]["template"]["names"]
            max = len(names)
        elif is_prefix:
            prefix = value[1]["template"]["prefix"]
            min = value[1]["template"]["min"]
            max = value[1]["template"]["max"]
            names = [f"{prefix}{i}" for i in range(max)]
        # need to create a new input based on the contents of input
        template_input = None
        template_required = True
        for _input_type, dict_input in input.items():
            # for now, get just the first value from dict_input; if not required, min can be ignored
            if len(dict_input) == 0:
                continue
            template_input = list(dict_input.values())[0]
            template_required = _input_type == "required"
            break
        if template_input is None:
            raise Exception("template_input could not be determined from required or optional; this should never happen.")
        new_dict = {}
        new_dict_added_to = False
        # first, add possible inputs into out_dict
        for i, name in enumerate(names):
            expected_id = finalize_prefix(curr_prefix, name)
            # required
            if i < min and template_required:
                out_dict["required"][expected_id] = template_input
                type_dict = new_dict.setdefault("required", {})
            # optional
            else:
                out_dict["optional"][expected_id] = template_input
                type_dict = new_dict.setdefault("optional", {})
            if expected_id in live_inputs:
                # NOTE: prefix gets added in parse_class_inputs
                type_dict[name] = template_input
                new_dict_added_to = True
        # account for the edge case that all inputs are optional and no values are received
        if not new_dict_added_to:
            finalized_prefix = finalize_prefix(curr_prefix)
            out_dict["dynamic_paths"][finalized_prefix] = finalized_prefix
            out_dict["dynamic_paths_default_value"][finalized_prefix] = DynamicPathsDefaultValue.EMPTY_DICT
        parse_class_inputs(out_dict, live_inputs, new_dict, curr_prefix)

@comfytype(io_type="COMFY_DYNAMICCOMBO_V3")
class DynamicCombo(ComfyTypeI):
    Type = dict[str, Any]

    class Option:
        def __init__(self, key: str, inputs: list[Input]):
            self.key = key
            self.inputs = inputs

        def as_dict(self):
            return {
                "key": self.key,
                "inputs": create_input_dict_v1(self.inputs),
            }

    class Input(DynamicInput):
        def __init__(self, id: str, options: list[DynamicCombo.Option],
                    display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None, extra_dict=None):
            super().__init__(id, display_name, optional, tooltip, lazy, extra_dict)
            self.options = options

        def get_all(self) -> list[Input]:
            return [self] + [input for option in self.options for input in option.inputs]

        def as_dict(self):
            return super().as_dict() | prune_dict({
                "options": [o.as_dict() for o in self.options],
            })

        def validate(self):
            # make sure all nested inputs are validated
            for option in self.options:
                for input in option.inputs:
                    input.validate()

    @staticmethod
    def _expand_schema_for_dynamic(out_dict: dict[str, Any], live_inputs: dict[str, Any], value: tuple[str, dict[str, Any]], input_type: str, curr_prefix: list[str] | None):
        finalized_id = finalize_prefix(curr_prefix)
        if finalized_id in live_inputs:
            key = live_inputs[finalized_id]
            selected_option = None
            # get options from dict
            options: list[dict[str, str | dict[str, Any]]] = value[1]["options"]
            for option in options:
                if option["key"] == key:
                    selected_option = option
                    break
            if selected_option is not None:
                parse_class_inputs(out_dict, live_inputs, selected_option["inputs"], curr_prefix)
                # add self to inputs
                out_dict[input_type][finalized_id] = value
                out_dict["dynamic_paths"][finalized_id] = finalize_prefix(curr_prefix, curr_prefix[-1])

@comfytype(io_type="COMFY_DYNAMICSLOT_V3")
class DynamicSlot(ComfyTypeI):
    Type = dict[str, Any]

    class Input(DynamicInput):
        def __init__(self, slot: Input, inputs: list[Input],
                    display_name: str=None, tooltip: str=None, lazy: bool=None, extra_dict=None):
            assert(not isinstance(slot, DynamicInput))
            self.slot = copy.copy(slot)
            self.slot.display_name = slot.display_name if slot.display_name is not None else display_name
            optional = True
            self.slot.tooltip = slot.tooltip if slot.tooltip is not None else tooltip
            self.slot.lazy = slot.lazy if slot.lazy is not None else lazy
            self.slot.extra_dict = slot.extra_dict if slot.extra_dict is not None else extra_dict
            super().__init__(slot.id, self.slot.display_name, optional, self.slot.tooltip, self.slot.lazy, self.slot.extra_dict)
            self.inputs = inputs
            self.force_input = None
            # force widget inputs to have no widgets, otherwise this would be awkward
            if isinstance(self.slot, WidgetInput):
                self.force_input = True
                self.slot.force_input = True

        def get_all(self) -> list[Input]:
            return [self.slot] + self.inputs

        def as_dict(self):
            return super().as_dict() | prune_dict({
                "slotType": str(self.slot.get_io_type()),
                "inputs": create_input_dict_v1(self.inputs),
                "forceInput": self.force_input,
            })

        def validate(self):
            self.slot.validate()
            for input in self.inputs:
                input.validate()

    @staticmethod
    def _expand_schema_for_dynamic(out_dict: dict[str, Any], live_inputs: dict[str, Any], value: tuple[str, dict[str, Any]], input_type: str, curr_prefix: list[str] | None):
        finalized_id = finalize_prefix(curr_prefix)
        if finalized_id in live_inputs:
            inputs = value[1]["inputs"]
            parse_class_inputs(out_dict, live_inputs, inputs, curr_prefix)
            # add self to inputs
            out_dict[input_type][finalized_id] = value
            out_dict["dynamic_paths"][finalized_id] = finalize_prefix(curr_prefix, curr_prefix[-1])

@comfytype(io_type="IMAGECOMPARE")
class ImageCompare(ComfyTypeI):
  Type = dict

  class Input(WidgetInput):
      def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None,
                   socketless: bool=True, advanced: bool=None):
          super().__init__(id, display_name, optional, tooltip, None, None, socketless, None, None, None, None, advanced)

      def as_dict(self):
          return super().as_dict()


@comfytype(io_type="COLOR")
class Color(ComfyTypeIO):
  Type = str

  class Input(WidgetInput):
      def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None,
                   socketless: bool=True, advanced: bool=None, default: str="#ffffff"):
          super().__init__(id, display_name, optional, tooltip, None, default, socketless, None, None, None, None, advanced)
          self.default: str

      def as_dict(self):
          return super().as_dict()


@comfytype(io_type="COLORS")
class Colors(ComfyTypeIO):
    Type = list[Color.Type]

    class Input(WidgetInput):
        def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None,
                     socketless: bool=True, default: list[str]=None, advanced: bool=None):
            super().__init__(id, display_name, optional, tooltip, None, default, socketless, None, None, None, None, advanced)
            if default is None:
                self.default = []


@comfytype(io_type="BOUNDING_BOX")
class BoundingBox(ComfyTypeIO):
    class BoundingBoxDict(TypedDict):
        x: int
        y: int
        width: int
        height: int
    Type = BoundingBoxDict

    class Input(WidgetInput):
        def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None,
                     socketless: bool=True, default: dict=None, component: str=None, force_input: bool=None):
            super().__init__(id, display_name, optional, tooltip, None, default, socketless)
            self.component = component
            self.force_input = force_input
            if default is None:
                self.default = {"x": 0, "y": 0, "width": 512, "height": 512}

        def as_dict(self):
            d = super().as_dict()
            if self.component:
                d["component"] = self.component
            if self.force_input is not None:
                d["forceInput"] = self.force_input
            return d


@comfytype(io_type="CURVE")
class Curve(ComfyTypeIO):
    from comfy_api.input import CurvePoint
    if TYPE_CHECKING:
        Type = CurveInput_

    class Input(WidgetInput):
        def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None,
                     socketless: bool=True, default: list[tuple[float, float]]=None, advanced: bool=None):
            super().__init__(id, display_name, optional, tooltip, None, default, socketless, None, None, None, None, advanced)
            if default is None:
                self.default = [(0.0, 0.0), (1.0, 1.0)]

        def as_dict(self):
            d = super().as_dict()
            if self.default is not None:
                d["default"] = {"points": [list(p) for p in self.default], "interpolation": "monotone_cubic"}
            return d


@comfytype(io_type="BOUNDING_BOXES")
class BoundingBoxes(ComfyTypeIO):
    class BoundingBoxWithMetadata(BoundingBox.BoundingBoxDict):
        metadata: dict
    Type = list[BoundingBoxWithMetadata]

    class Input(WidgetInput):
        def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None,
                     socketless: bool=True, default: list[dict]=None, advanced: bool=None):
            super().__init__(id, display_name, optional, tooltip, None, default, socketless, None, None, None, None, advanced)
            if default is None:
                self.default = []


@comfytype(io_type="HISTOGRAM")
class Histogram(ComfyTypeIO):
    """A histogram represented as a list of bin counts."""
    Type = list[int]


@comfytype(io_type="RANGE")
class Range(ComfyTypeIO):
    from comfy_api.input import RangeInput
    if TYPE_CHECKING:
        Type = RangeInput

    class Input(WidgetInput):
        def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None,
                     socketless: bool=True, default: dict=None,
                     display: str=None,
                     gradient_stops: list=None,
                     show_midpoint: bool=None,
                     midpoint_scale: str=None,
                     value_min: float=None,
                     value_max: float=None,
                     advanced: bool=None):
            super().__init__(id, display_name, optional, tooltip, None, default, socketless, None, None, None, None, advanced)
            if default is None:
                self.default = {"min": 0.0, "max": 1.0}
            self.display = display
            self.gradient_stops = gradient_stops
            self.show_midpoint = show_midpoint
            self.midpoint_scale = midpoint_scale
            self.value_min = value_min
            self.value_max = value_max

        def as_dict(self):
            return super().as_dict() | prune_dict({
                "display": self.display,
                "gradient_stops": self.gradient_stops,
                "show_midpoint": self.show_midpoint,
                "midpoint_scale": self.midpoint_scale,
                "value_min": self.value_min,
                "value_max": self.value_max,
            })


DYNAMIC_INPUT_LOOKUP: dict[str, Callable[[dict[str, Any], dict[str, Any], tuple[str, dict[str, Any]], str, list[str] | None], None]] = {}
def register_dynamic_input_func(io_type: str, func: Callable[[dict[str, Any], dict[str, Any], tuple[str, dict[str, Any]], str, list[str] | None], None]):
    DYNAMIC_INPUT_LOOKUP[io_type] = func

def get_dynamic_input_func(io_type: str) -> Callable[[dict[str, Any], dict[str, Any], tuple[str, dict[str, Any]], str, list[str] | None], None]:
    return DYNAMIC_INPUT_LOOKUP[io_type]

def setup_dynamic_input_funcs():
    # Autogrow.Input
    register_dynamic_input_func(Autogrow.io_type, Autogrow._expand_schema_for_dynamic)
    # DynamicCombo.Input
    register_dynamic_input_func(DynamicCombo.io_type, DynamicCombo._expand_schema_for_dynamic)
    # DynamicSlot.Input
    register_dynamic_input_func(DynamicSlot.io_type, DynamicSlot._expand_schema_for_dynamic)

if len(DYNAMIC_INPUT_LOOKUP) == 0:
    setup_dynamic_input_funcs()

class V3Data(TypedDict):
    hidden_inputs: dict[str, Any]
    'Dictionary where the keys are the hidden input ids and the values are the values of the hidden inputs.'
    dynamic_paths: dict[str, Any]
    'Dictionary where the keys are the input ids and the values dictate how to turn the inputs into a nested dictionary.'
    dynamic_paths_default_value: dict[str, Any]
    'Dictionary where the keys are the input ids and the values are a string from DynamicPathsDefaultValue for the inputs if value is None.'
    create_dynamic_tuple: bool
    'When True, the value of the dynamic input will be in the format (value, path_key).'

class HiddenHolder:
    def __init__(self, unique_id: str, prompt: Any,
                 extra_pnginfo: Any, dynprompt: Any,
                 auth_token_comfy_org: str, api_key_comfy_org: str,
                 comfy_usage_source: str = None, **kwargs):
        self.unique_id = unique_id
        """UNIQUE_ID is the unique identifier of the node, and matches the id property of the node on the client side. It is commonly used in client-server communications (see messages)."""
        self.prompt = prompt
        """PROMPT is the complete prompt sent by the client to the server. See the prompt object for a full description."""
        self.extra_pnginfo = extra_pnginfo
        """EXTRA_PNGINFO is a dictionary that will be copied into the metadata of any .png files saved. Custom nodes can store additional information in this dictionary for saving (or as a way to communicate with a downstream node)."""
        self.dynprompt = dynprompt
        """DYNPROMPT is an instance of comfy_execution.graph.DynamicPrompt. It differs from PROMPT in that it may mutate during the course of execution in response to Node Expansion."""
        self.auth_token_comfy_org = auth_token_comfy_org
        """AUTH_TOKEN_COMFY_ORG is a token acquired from signing into a ComfyOrg account on frontend."""
        self.api_key_comfy_org = api_key_comfy_org
        """API_KEY_COMFY_ORG is an API Key generated by ComfyOrg that allows skipping signing into a ComfyOrg account on frontend."""
        self.comfy_usage_source = comfy_usage_source
        """COMFY_USAGE_SOURCE identifies the client that submitted the prompt (e.g. comfyui-frontend, comfy-cli, comfyui-mcp); forwarded to API nodes' upstream requests via the Comfy-Usage-Source header."""

    def __getattr__(self, key: str):
        '''If hidden variable not found, return None.'''
        return None

    @classmethod
    def from_dict(cls, d: dict | None):
        if d is None:
            d = {}
        return cls(
            unique_id=d.get(Hidden.unique_id, None),
            prompt=d.get(Hidden.prompt, None),
            extra_pnginfo=d.get(Hidden.extra_pnginfo, None),
            dynprompt=d.get(Hidden.dynprompt, None),
            auth_token_comfy_org=d.get(Hidden.auth_token_comfy_org, None),
            api_key_comfy_org=d.get(Hidden.api_key_comfy_org, None),
            comfy_usage_source=d.get(Hidden.comfy_usage_source, None),
        )

    @classmethod
    def from_v3_data(cls, v3_data: V3Data | None) -> HiddenHolder:
        return cls.from_dict(v3_data["hidden_inputs"] if v3_data else None)

class Hidden(str, Enum):
    '''
    Enumerator for requesting hidden variables in nodes.
    '''
    unique_id = "UNIQUE_ID"
    """UNIQUE_ID is the unique identifier of the node, and matches the id property of the node on the client side. It is commonly used in client-server communications (see messages)."""
    prompt = "PROMPT"
    """PROMPT is the complete prompt sent by the client to the server. See the prompt object for a full description."""
    extra_pnginfo = "EXTRA_PNGINFO"
    """EXTRA_PNGINFO is a dictionary that will be copied into the metadata of any .png files saved. Custom nodes can store additional information in this dictionary for saving (or as a way to communicate with a downstream node)."""
    dynprompt = "DYNPROMPT"
    """DYNPROMPT is an instance of comfy_execution.graph.DynamicPrompt. It differs from PROMPT in that it may mutate during the course of execution in response to Node Expansion."""
    auth_token_comfy_org = "AUTH_TOKEN_COMFY_ORG"
    """AUTH_TOKEN_COMFY_ORG is a token acquired from signing into a ComfyOrg account on frontend."""
    api_key_comfy_org = "API_KEY_COMFY_ORG"
    """API_KEY_COMFY_ORG is an API Key generated by ComfyOrg that allows skipping signing into a ComfyOrg account on frontend."""
    comfy_usage_source = "COMFY_USAGE_SOURCE"
    """COMFY_USAGE_SOURCE identifies the client that submitted the prompt (e.g. comfyui-frontend, comfy-cli, comfyui-mcp); forwarded to API nodes' upstream requests via the Comfy-Usage-Source header."""


@dataclass
class NodeInfoV1:
    input: dict=None
    input_order: dict[str, list[str]]=None
    is_input_list: bool=None
    output: list[str]=None
    output_is_list: list[bool]=None
    output_name: list[str]=None
    output_tooltips: list[str]=None
    output_matchtypes: list[str]=None
    name: str=None
    display_name: str=None
    description: str=None
    python_module: Any=None
    category: str=None
    output_node: bool=None
    deprecated: bool=None
    experimental: bool=None
    dev_only: bool=None
    api_node: bool=None
    price_badge: dict | None = None
    search_aliases: list[str]=None
    essentials_category: str=None
    has_intermediate_output: bool=None


@dataclass
class PriceBadgeDepends:
    widgets: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    input_groups: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not isinstance(self.widgets, list) or any(not isinstance(x, str) for x in self.widgets):
            raise ValueError("PriceBadgeDepends.widgets must be a list[str].")
        if not isinstance(self.inputs, list) or any(not isinstance(x, str) for x in self.inputs):
            raise ValueError("PriceBadgeDepends.inputs must be a list[str].")
        if not isinstance(self.input_groups, list) or any(not isinstance(x, str) for x in self.input_groups):
            raise ValueError("PriceBadgeDepends.input_groups must be a list[str].")

    def as_dict(self, schema_inputs: list["Input"]) -> dict[str, Any]:
        # Build lookup: widget_id -> io_type
        input_types: dict[str, str] = {}
        for inp in schema_inputs:
            all_inputs = inp.get_all()
            input_types[inp.id] = inp.get_io_type()  # First input is always the parent itself
            for nested_inp in all_inputs[1:]:
                # For DynamicCombo/DynamicSlot, nested inputs are prefixed with parent ID
                # to match frontend naming convention (e.g., "should_texture.enable_pbr")
                prefixed_id = f"{inp.id}.{nested_inp.id}"
                input_types[prefixed_id] = nested_inp.get_io_type()

        # Enrich widgets with type information, raising error for unknown widgets
        widgets_data: list[dict[str, str]] = []
        for w in self.widgets:
            if w not in input_types:
                raise ValueError(
                    f"PriceBadge depends_on.widgets references unknown widget '{w}'. "
                    f"Available widgets: {list(input_types.keys())}"
                )
            widgets_data.append({"name": w, "type": input_types[w]})

        return {
            "widgets": widgets_data,
            "inputs": self.inputs,
            "input_groups": self.input_groups,
        }


@dataclass
class PriceBadge:
    expr: str
    depends_on: PriceBadgeDepends = field(default_factory=PriceBadgeDepends)
    engine: str = field(default="jsonata")

    def validate(self) -> None:
        if self.engine != "jsonata":
            raise ValueError(f"Unsupported PriceBadge.engine '{self.engine}'. Only 'jsonata' is supported.")
        if not isinstance(self.expr, str) or not self.expr.strip():
            raise ValueError("PriceBadge.expr must be a non-empty string.")
        self.depends_on.validate()

    def as_dict(self, schema_inputs: list["Input"]) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "depends_on": self.depends_on.as_dict(schema_inputs),
            "expr": self.expr,
        }


@dataclass
class Schema:
    """Definition of V3 node properties."""

    node_id: str
    """ID of node - should be globally unique. If this is a custom node, add a prefix or postfix to avoid name clashes."""
    display_name: str = None
    """Display name of node."""
    category: str = "sd"
    """The category of the node, as per the "Add Node" menu."""
    inputs: list[Input] = field(default_factory=list)
    outputs: list[Output] = field(default_factory=list)
    hidden: list[Hidden] = field(default_factory=list)
    description: str=""
    """Node description, shown as a tooltip when hovering over the node."""
    search_aliases: list[str] = field(default_factory=list)
    """Alternative names for search. Useful for synonyms, abbreviations, or old names after renaming."""
    is_input_list: bool = False
    """A flag indicating if this node implements the additional code necessary to deal with OUTPUT_IS_LIST nodes.

    All inputs of ``type`` will become ``list[type]``, regardless of how many items are passed in.  This also affects ``check_lazy_status``.

    From the docs:

    A node can also override the default input behaviour and receive the whole list in a single call. This is done by setting a class attribute `INPUT_IS_LIST` to ``True``.

    Comfy Docs: https://docs.comfy.org/custom-nodes/backend/lists#list-processing
    """
    is_output_node: bool=False
    """Flags this node as an output node, causing any inputs it requires to be executed.

    If a node is not connected to any output nodes, that node will not be executed.  Usage::

    From the docs:

    By default, a node is not considered an output. Set ``OUTPUT_NODE = True`` to specify that it is.

    Comfy Docs: https://docs.comfy.org/custom-nodes/backend/server_overview#output-node
    """
    is_deprecated: bool=False
    """Flags a node as deprecated, indicating to users that they should find alternatives to this node."""
    is_experimental: bool=False
    """Flags a node as experimental, informing users that it may change or not work as expected."""
    is_dev_only: bool=False
    """Flags a node as dev-only, hiding it from search/menus unless dev mode is enabled."""
    is_api_node: bool=False
    """Flags a node as an API node. See: https://docs.comfy.org/tutorials/api-nodes/overview."""
    price_badge: PriceBadge | None = None
    """Optional client-evaluated pricing badge declaration for this node."""
    not_idempotent: bool=False
    """Flags a node as not idempotent; when True, the node will run and not reuse the cached outputs when identical inputs are provided on a different node in the graph."""
    enable_expand: bool=False
    """Flags a node as expandable, allowing NodeOutput to include 'expand' property."""
    accept_all_inputs: bool=False
    """When True, all inputs from the prompt will be passed to the node as kwargs, even if not defined in the schema."""
    essentials_category: str | None = None
    """Optional category for the Essentials tab. Path-based like category field (e.g., 'Basic', 'Image Tools/Editing')."""
    has_intermediate_output: bool=False
    """Flags this node as having intermediate output that should persist across page refreshes.

    Nodes with this flag behave like output nodes (their UI results are cached and resent
    to the frontend) but do NOT automatically get added to the execution list. This means
    they will only execute if they are on the dependency path of a real output node.

    Use this for nodes with interactive/operable UI regions that produce intermediate outputs
    (e.g., Image Crop, Painter) rather than final outputs (e.g., Save Image).
    """

    def validate(self):
        '''Validate the schema:
        - verify ids on inputs and outputs are unique - both internally and in relation to each other
        '''
        nested_inputs: list[Input] = []
        for input in self.inputs:
            if not isinstance(input, DynamicInput):
                nested_inputs.extend(input.get_all())
        input_ids = [i.id for i in nested_inputs]
        output_ids = [o.id for o in self.outputs]
        input_set = set(input_ids)
        output_set = set(output_ids)
        issues: list[str] = []
        # verify ids are unique per list
        if len(input_set) != len(input_ids):
            issues.append(f"Input ids must be unique, but {[item for item, count in Counter(input_ids).items() if count > 1]} are not.")
        if len(output_set) != len(output_ids):
            issues.append(f"Output ids must be unique, but {[item for item, count in Counter(output_ids).items() if count > 1]} are not.")
        if len(issues) > 0:
            raise ValueError("\n".join(issues))
        # validate inputs and outputs
        for input in self.inputs:
            input.validate()
        for output in self.outputs:
            output.validate()
        if self.price_badge is not None:
            self.price_badge.validate()

    def finalize(self):
        """Add hidden based on selected schema options, and give outputs without ids default ids."""
        # ensure inputs, outputs, and hidden are lists
        if self.inputs is None:
            self.inputs = []
        if self.outputs is None:
            self.outputs = []
        if self.hidden is None:
            self.hidden = []
        # if is an api_node, will need key-related hidden
        if self.is_api_node:
            if Hidden.auth_token_comfy_org not in self.hidden:
                self.hidden.append(Hidden.auth_token_comfy_org)
            if Hidden.api_key_comfy_org not in self.hidden:
                self.hidden.append(Hidden.api_key_comfy_org)
            if Hidden.comfy_usage_source not in self.hidden:
                self.hidden.append(Hidden.comfy_usage_source)
        # if is an output_node, will need prompt and extra_pnginfo
        if self.is_output_node:
            if Hidden.prompt not in self.hidden:
                self.hidden.append(Hidden.prompt)
            if Hidden.extra_pnginfo not in self.hidden:
                self.hidden.append(Hidden.extra_pnginfo)
        # give outputs without ids default ids
        for i, output in enumerate(self.outputs):
            if output.id is None:
                output.id = f"_{i}_{output.io_type}_"

    def get_v1_info(self, cls) -> NodeInfoV1:
        # get V1 inputs
        input = create_input_dict_v1(self.inputs)
        if self.hidden:
            for hidden in self.hidden:
                input.setdefault("hidden", {})[hidden.name] = (hidden.value,)
        # create separate lists from output fields
        output = []
        output_is_list = []
        output_name = []
        output_tooltips = []
        output_matchtypes = []
        any_matchtypes = False
        if self.outputs:
            for o in self.outputs:
                output.append(o.io_type)
                output_is_list.append(o.is_output_list)
                output_name.append(o.display_name if o.display_name else o.io_type)
                output_tooltips.append(o.tooltip if o.tooltip else None)
                # special handling for MatchType
                if isinstance(o, MatchType.Output):
                    output_matchtypes.append(o.template.template_id)
                    any_matchtypes = True
                else:
                    output_matchtypes.append(None)

        # clear out lists that are all None
        if not any_matchtypes:
            output_matchtypes = None

        info = NodeInfoV1(
            input=input,
            input_order={key: list(value.keys()) for (key, value) in input.items()},
            is_input_list=self.is_input_list,
            output=output,
            output_is_list=output_is_list,
            output_name=output_name,
            output_tooltips=output_tooltips,
            output_matchtypes=output_matchtypes,
            name=self.node_id,
            display_name=self.display_name,
            category=self.category,
            description=self.description,
            output_node=self.is_output_node,
            has_intermediate_output=self.has_intermediate_output,
            deprecated=self.is_deprecated,
            experimental=self.is_experimental,
            dev_only=self.is_dev_only,
            api_node=self.is_api_node,
            python_module=getattr(cls, "RELATIVE_PYTHON_MODULE", "nodes"),
            price_badge=self.price_badge.as_dict(self.inputs) if self.price_badge is not None else None,
            search_aliases=self.search_aliases if self.search_aliases else None,
            essentials_category=self.essentials_category,
        )
        return info

def get_finalized_class_inputs(d: dict[str, Any], live_inputs: dict[str, Any], include_hidden=False) -> tuple[dict[str, Any], V3Data]:
    out_dict = {
        "required": {},
        "optional": {},
        "dynamic_paths": {},
        "dynamic_paths_default_value": {},
    }
    d = d.copy()
    # ignore hidden for parsing
    hidden = d.pop("hidden", None)
    parse_class_inputs(out_dict, live_inputs, d)
    if hidden is not None and include_hidden:
        out_dict["hidden"] = hidden
    v3_data = {}
    dynamic_paths = out_dict.pop("dynamic_paths", None)
    if dynamic_paths is not None and len(dynamic_paths) > 0:
        v3_data["dynamic_paths"] = dynamic_paths
    # this list is used for autogrow, in the case all inputs are optional and no values are passed
    dynamic_paths_default_value = out_dict.pop("dynamic_paths_default_value", None)
    if dynamic_paths_default_value is not None and len(dynamic_paths_default_value) > 0:
        v3_data["dynamic_paths_default_value"] = dynamic_paths_default_value
    return out_dict, hidden, v3_data

def parse_class_inputs(out_dict: dict[str, Any], live_inputs: dict[str, Any], curr_dict: dict[str, Any], curr_prefix: list[str] | None=None) -> None:
    for input_type, inner_d in curr_dict.items():
        for id, value in inner_d.items():
            io_type = value[0]
            if io_type in DYNAMIC_INPUT_LOOKUP:
                # dynamic inputs need to be handled with lookup functions
                dynamic_input_func = get_dynamic_input_func(io_type)
                new_prefix = handle_prefix(curr_prefix, id)
                dynamic_input_func(out_dict, live_inputs, value, input_type, new_prefix)
            else:
                # non-dynamic inputs get directly transferred
                finalized_id = finalize_prefix(curr_prefix, id)
                out_dict[input_type][finalized_id] = value
                if curr_prefix:
                    out_dict["dynamic_paths"][finalized_id] = finalized_id

def create_input_dict_v1(inputs: list[Input]) -> dict:
    input = {
        "required": {}
    }
    for i in inputs:
        add_to_dict_v1(i, input)
    return input

def add_to_dict_v1(i: Input, d: dict):
    key = "optional" if i.optional else "required"
    as_dict = i.as_dict()
    # for v1, we don't want to include the optional key
    as_dict.pop("optional", None)
    d.setdefault(key, {})[i.id] = (i.get_io_type(), as_dict)

class DynamicPathsDefaultValue:
    EMPTY_DICT = "empty_dict"

def build_nested_inputs(values: dict[str, Any], v3_data: V3Data):
    paths = v3_data.get("dynamic_paths", None)
    default_value_dict = v3_data.get("dynamic_paths_default_value", {})
    if paths is None:
        return values
    values = values.copy()

    result = {}

    create_tuple = v3_data.get("create_dynamic_tuple", False)

    for key, path in paths.items():
        parts = path.split(".")
        current = result

        for i, p in enumerate(parts):
            is_last = (i == len(parts) - 1)

            if is_last:
                value = values.pop(key, None)
                if value is None:
                    # see if a default value was provided for this key
                    default_option = default_value_dict.get(key, None)
                    if default_option == DynamicPathsDefaultValue.EMPTY_DICT:
                        value = {}
                if create_tuple:
                    value = (value, key)
                current[p] = value
            else:
                current = current.setdefault(p, {})

    values.update(result)
    return values


class _ComfyNodeBaseInternal(_ComfyNodeInternal):
    """Common base class for storing internal methods and properties; DO NOT USE for defining nodes."""

    RELATIVE_PYTHON_MODULE = None
    SCHEMA = None

    # filled in during execution
    hidden: HiddenHolder = None

    @classmethod
    @abstractmethod
    def define_schema(cls) -> Schema:
        """Override this function with one that returns a Schema instance."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def execute(cls, **kwargs) -> NodeOutput:
        """Override this function with one that performs node's actions."""
        raise NotImplementedError

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        """Optionally, define this function to validate inputs; equivalent to V1's VALIDATE_INPUTS.

        If the function returns a string, it will be used as the validation error message for the node.
        """
        raise NotImplementedError

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> Any:
        """Optionally, define this function to fingerprint inputs; equivalent to V1's IS_CHANGED.

        If this function returns the same value as last run, the node will not be executed."""
        raise NotImplementedError

    @classmethod
    def check_lazy_status(cls, **kwargs) -> list[str]:
        """Optionally, define this function to return a list of input names that should be evaluated.

        This basic mixin impl. requires all inputs.

        :kwargs: All node inputs will be included here.  If the input is ``None``, it should be assumed that it has not yet been evaluated.  \
            When using ``INPUT_IS_LIST = True``, unevaluated will instead be ``(None,)``.

        Params should match the nodes execution ``FUNCTION`` (self, and all inputs by name).
        Will be executed repeatedly until it returns an empty list, or all requested items were already evaluated (and sent as params).

        Comfy Docs: https://docs.comfy.org/custom-nodes/backend/lazy_evaluation#defining-check-lazy-status
        """
        return [name for name in kwargs if kwargs[name] is None]

    def __init__(self):
        self.__class__.VALIDATE_CLASS()

    @classmethod
    def GET_BASE_CLASS(cls):
        return _ComfyNodeBaseInternal

    @final
    @classmethod
    def VALIDATE_CLASS(cls):
        if first_real_override(cls, "define_schema") is None:
            raise Exception(f"No define_schema function was defined for node class {cls.__name__}.")
        if first_real_override(cls, "execute") is None:
            raise Exception(f"No execute function was defined for node class {cls.__name__}.")

    @classproperty
    def FUNCTION(cls):  # noqa
        if inspect.iscoroutinefunction(cls.execute):
            return "EXECUTE_NORMALIZED_ASYNC"
        return "EXECUTE_NORMALIZED"

    @final
    @classmethod
    def EXECUTE_NORMALIZED(cls, *args, **kwargs) -> NodeOutput:
        to_return = cls.execute(*args, **kwargs)
        if to_return is None:
            to_return = NodeOutput()
        elif isinstance(to_return, NodeOutput):
            pass
        elif isinstance(to_return, tuple):
            to_return = NodeOutput(*to_return)
        elif isinstance(to_return, dict):
            to_return = NodeOutput.from_dict(to_return)
        elif isinstance(to_return, ExecutionBlocker):
            to_return = NodeOutput(block_execution=to_return.message)
        else:
            raise Exception(f"Invalid return type from node: {type(to_return)}")
        if to_return.expand is not None and not cls.SCHEMA.enable_expand:
            raise Exception(f"Node {cls.__name__} is not expandable, but expand included in NodeOutput; developer should set enable_expand=True on node's Schema to allow this.")
        return to_return

    @final
    @classmethod
    async def EXECUTE_NORMALIZED_ASYNC(cls, *args, **kwargs) -> NodeOutput:
        to_return = await cls.execute(*args, **kwargs)
        if to_return is None:
            to_return = NodeOutput()
        elif isinstance(to_return, NodeOutput):
            pass
        elif isinstance(to_return, tuple):
            to_return = NodeOutput(*to_return)
        elif isinstance(to_return, dict):
            to_return = NodeOutput.from_dict(to_return)
        elif isinstance(to_return, ExecutionBlocker):
            to_return = NodeOutput(block_execution=to_return.message)
        else:
            raise Exception(f"Invalid return type from node: {type(to_return)}")
        if to_return.expand is not None and not cls.SCHEMA.enable_expand:
            raise Exception(f"Node {cls.__name__} is not expandable, but expand included in NodeOutput; developer should set enable_expand=True on node's Schema to allow this.")
        return to_return

    @final
    @classmethod
    def PREPARE_CLASS_CLONE(cls, v3_data: V3Data | None) -> type[ComfyNode]:
        """Creates clone of real node class to prevent monkey-patching."""
        c_type: type[ComfyNode] = cls if is_class(cls) else type(cls)
        type_clone: type[ComfyNode] = shallow_clone_class(c_type)
        # set hidden
        type_clone.hidden = HiddenHolder.from_v3_data(v3_data)
        return type_clone
    #############################################
    # V1 Backwards Compatibility code
    #--------------------------------------------
    @final
    @classmethod
    def GET_NODE_INFO_V1(cls) -> dict[str, Any]:
        schema = cls.GET_SCHEMA()
        info = schema.get_v1_info(cls)
        return asdict(info)

    _DESCRIPTION = None
    @final
    @classproperty
    def DESCRIPTION(cls):  # noqa
        if cls._DESCRIPTION is None:
            cls.GET_SCHEMA()
        return cls._DESCRIPTION

    _CATEGORY = None
    @final
    @classproperty
    def CATEGORY(cls):  # noqa
        if cls._CATEGORY is None:
            cls.GET_SCHEMA()
        return cls._CATEGORY

    _EXPERIMENTAL = None
    @final
    @classproperty
    def EXPERIMENTAL(cls):  # noqa
        if cls._EXPERIMENTAL is None:
            cls.GET_SCHEMA()
        return cls._EXPERIMENTAL

    _DEPRECATED = None
    @final
    @classproperty
    def DEPRECATED(cls):  # noqa
        if cls._DEPRECATED is None:
            cls.GET_SCHEMA()
        return cls._DEPRECATED

    _DEV_ONLY = None
    @final
    @classproperty
    def DEV_ONLY(cls):  # noqa
        if cls._DEV_ONLY is None:
            cls.GET_SCHEMA()
        return cls._DEV_ONLY

    _API_NODE = None
    @final
    @classproperty
    def API_NODE(cls):  # noqa
        if cls._API_NODE is None:
            cls.GET_SCHEMA()
        return cls._API_NODE

    _OUTPUT_NODE = None
    @final
    @classproperty
    def OUTPUT_NODE(cls):  # noqa
        if cls._OUTPUT_NODE is None:
            cls.GET_SCHEMA()
        return cls._OUTPUT_NODE

    _HAS_INTERMEDIATE_OUTPUT = None
    @final
    @classproperty
    def HAS_INTERMEDIATE_OUTPUT(cls):  # noqa
        if cls._HAS_INTERMEDIATE_OUTPUT is None:
            cls.GET_SCHEMA()
        return cls._HAS_INTERMEDIATE_OUTPUT

    _INPUT_IS_LIST = None
    @final
    @classproperty
    def INPUT_IS_LIST(cls):  # noqa
        if cls._INPUT_IS_LIST is None:
            cls.GET_SCHEMA()
        return cls._INPUT_IS_LIST
    _OUTPUT_IS_LIST = None

    @final
    @classproperty
    def OUTPUT_IS_LIST(cls):  # noqa
        if cls._OUTPUT_IS_LIST is None:
            cls.GET_SCHEMA()
        return cls._OUTPUT_IS_LIST

    _RETURN_TYPES = None
    @final
    @classproperty
    def RETURN_TYPES(cls):  # noqa
        if cls._RETURN_TYPES is None:
            cls.GET_SCHEMA()
        return cls._RETURN_TYPES

    _RETURN_NAMES = None
    @final
    @classproperty
    def RETURN_NAMES(cls):  # noqa
        if cls._RETURN_NAMES is None:
            cls.GET_SCHEMA()
        return cls._RETURN_NAMES

    _OUTPUT_TOOLTIPS = None
    @final
    @classproperty
    def OUTPUT_TOOLTIPS(cls):  # noqa
        if cls._OUTPUT_TOOLTIPS is None:
            cls.GET_SCHEMA()
        return cls._OUTPUT_TOOLTIPS

    _NOT_IDEMPOTENT = None
    @final
    @classproperty
    def NOT_IDEMPOTENT(cls):  # noqa
        if cls._NOT_IDEMPOTENT is None:
            cls.GET_SCHEMA()
        return cls._NOT_IDEMPOTENT

    _ACCEPT_ALL_INPUTS = None
    @final
    @classproperty
    def ACCEPT_ALL_INPUTS(cls):  # noqa
        if cls._ACCEPT_ALL_INPUTS is None:
            cls.GET_SCHEMA()
        return cls._ACCEPT_ALL_INPUTS

    @final
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict]:
        schema = cls.FINALIZE_SCHEMA()
        info = schema.get_v1_info(cls)
        return info.input

    @final
    @classmethod
    def FINALIZE_SCHEMA(cls):
        """Call define_schema and finalize it."""
        schema = cls.define_schema()
        schema.finalize()
        return schema

    @final
    @classmethod
    def GET_SCHEMA(cls) -> Schema:
        """Validate node class, finalize schema, validate schema, and set expected class properties."""
        cls.VALIDATE_CLASS()
        schema = cls.FINALIZE_SCHEMA()
        schema.validate()
        if cls._DESCRIPTION is None:
            cls._DESCRIPTION = schema.description
        if cls._CATEGORY is None:
            cls._CATEGORY = schema.category
        if cls._EXPERIMENTAL is None:
            cls._EXPERIMENTAL = schema.is_experimental
        if cls._DEPRECATED is None:
            cls._DEPRECATED = schema.is_deprecated
        if cls._DEV_ONLY is None:
            cls._DEV_ONLY = schema.is_dev_only
        if cls._API_NODE is None:
            cls._API_NODE = schema.is_api_node
        if cls._OUTPUT_NODE is None:
            cls._OUTPUT_NODE = schema.is_output_node
        if cls._HAS_INTERMEDIATE_OUTPUT is None:
            cls._HAS_INTERMEDIATE_OUTPUT = schema.has_intermediate_output
        if cls._INPUT_IS_LIST is None:
            cls._INPUT_IS_LIST = schema.is_input_list
        if cls._NOT_IDEMPOTENT is None:
            cls._NOT_IDEMPOTENT = schema.not_idempotent
        if cls._ACCEPT_ALL_INPUTS is None:
            cls._ACCEPT_ALL_INPUTS = schema.accept_all_inputs

        if cls._RETURN_TYPES is None:
            output = []
            output_name = []
            output_is_list = []
            output_tooltips = []
            if schema.outputs:
                for o in schema.outputs:
                    output.append(o.io_type)
                    output_name.append(o.display_name if o.display_name else o.io_type)
                    output_is_list.append(o.is_output_list)
                    output_tooltips.append(o.tooltip if o.tooltip else None)

            cls._RETURN_TYPES = output
            cls._RETURN_NAMES = output_name
            cls._OUTPUT_IS_LIST = output_is_list
            cls._OUTPUT_TOOLTIPS = output_tooltips
        cls.SCHEMA = schema
        return schema
    #--------------------------------------------
    #############################################


class ComfyNode(_ComfyNodeBaseInternal):
    """Common base class for all V3 nodes."""

    @classmethod
    @abstractmethod
    def define_schema(cls) -> Schema:
        """Override this function with one that returns a Schema instance."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def execute(cls, **kwargs) -> NodeOutput:
        """Override this function with one that performs node's actions."""
        raise NotImplementedError

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        """Optionally, define this function to validate inputs; equivalent to V1's VALIDATE_INPUTS."""
        raise NotImplementedError

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> Any:
        """Optionally, define this function to fingerprint inputs; equivalent to V1's IS_CHANGED."""
        raise NotImplementedError

    @classmethod
    def check_lazy_status(cls, **kwargs) -> list[str]:
        """Optionally, define this function to return a list of input names that should be evaluated.

        This basic mixin impl. requires all inputs.

        :kwargs: All node inputs will be included here.  If the input is ``None``, it should be assumed that it has not yet been evaluated.  \
            When using ``INPUT_IS_LIST = True``, unevaluated will instead be ``(None,)``.

        Params should match the nodes execution ``FUNCTION`` (self, and all inputs by name).
        Will be executed repeatedly until it returns an empty list, or all requested items were already evaluated (and sent as params).

        Comfy Docs: https://docs.comfy.org/custom-nodes/backend/lazy_evaluation#defining-check-lazy-status
        """
        return [name for name in kwargs if kwargs[name] is None]

    @final
    @classmethod
    def GET_BASE_CLASS(cls):
        """DO NOT override this class. Will break things in execution.py."""
        return ComfyNode


class NodeOutput(_NodeOutputInternal):
    '''
    Standardized output of a node; can pass in any number of args and/or a UIOutput into 'ui' kwarg.
    '''
    def __init__(self, *args: Any, ui: _UIOutput | dict=None, expand: dict=None, block_execution: str=None):
        self.args = args
        self.ui = ui
        self.expand = expand
        self.block_execution = block_execution

    @property
    def result(self):
        return self.args if len(self.args) > 0 else None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeOutput:
        args = ()
        ui = None
        expand = None
        if "result" in data:
            result = data["result"]
            if isinstance(result, ExecutionBlocker):
                return cls(block_execution=result.message)
            args = result
        if "ui" in data:
            ui = data["ui"]
        if "expand" in data:
            expand = data["expand"]
        return cls(*args, ui=ui, expand=expand)

    def __getitem__(self, index) -> Any:
        return self.args[index]

class _UIOutput(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def as_dict(self) -> dict:
        ...


class InputMapOldId(TypedDict):
    """Map an old node input to a new node input by ID."""
    new_id: str
    old_id: str

class InputMapSetValue(TypedDict):
    """Set a specific value for a new node input."""
    new_id: str
    set_value: Any

InputMap = InputMapOldId | InputMapSetValue
"""
Input mapping for node replacement. Type is inferred by dictionary keys:
- {"new_id": str, "old_id": str} - maps old input to new input
- {"new_id": str, "set_value": Any} - sets a specific value for new input
"""

class OutputMap(TypedDict):
    """Map outputs of node replacement via indexes."""
    new_idx: int
    old_idx: int

class NodeReplace:
    """
    Defines a possible node replacement, mapping inputs and outputs of the old node to the new node.

    Also supports assigning specific values to the input widgets of the new node.

    Args:
        new_node_id: The class name of the new replacement node.
        old_node_id: The class name of the deprecated node.
        old_widget_ids: Ordered list of input IDs for widgets that may not have an input slot
            connected. The workflow JSON stores widget values by their relative position index,
            not by ID. This list maps those positional indexes to input IDs, enabling the
            replacement system to correctly identify widget values during node migration.
        input_mapping: List of input mappings from old node to new node.
        output_mapping: List of output mappings from old node to new node.
    """
    def __init__(self,
        new_node_id: str,
        old_node_id: str,
        old_widget_ids: list[str] | None=None,
        input_mapping: list[InputMap] | None=None,
        output_mapping: list[OutputMap] | None=None,
    ):
        self.new_node_id = new_node_id
        self.old_node_id = old_node_id
        self.old_widget_ids = old_widget_ids
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping

    def as_dict(self):
        """Create serializable representation of the node replacement."""
        return {
            "new_node_id": self.new_node_id,
            "old_node_id": self.old_node_id,
            "old_widget_ids": self.old_widget_ids,
            "input_mapping": list(self.input_mapping) if self.input_mapping else None,
            "output_mapping": list(self.output_mapping) if self.output_mapping else None,
        }


__all__ = [
    "FolderType",
    "UploadType",
    "RemoteOptions",
    "NumberDisplay",
    "ControlAfterGenerate",

    "comfytype",
    "Custom",
    "Input",
    "WidgetInput",
    "Output",
    "ComfyTypeI",
    "ComfyTypeIO",
    # Supported Types
    "Boolean",
    "Int",
    "Float",
    "String",
    "Combo",
    "MultiCombo",
    "Image",
    "WanCameraEmbedding",
    "Webcam",
    "Mask",
    "Latent",
    "Conditioning",
    "Sampler",
    "Sigmas",
    "Noise",
    "Guider",
    "Clip",
    "ControlNet",
    "Vae",
    "Model",
    "ModelPatch",
    "ClipVision",
    "ClipVisionOutput",
    "BackgroundRemoval",
    "AudioEncoder",
    "AudioEncoderOutput",
    "StyleModel",
    "Gligen",
    "UpscaleModel",
    "LatentUpscaleModel",
    "Audio",
    "Video",
    "SVG",
    "LoraModel",
    "LossMap",
    "Voxel",
    "Mesh",
    "Splat",
    "File3DAny",
    "File3DGLB",
    "File3DGLTF",
    "File3DFBX",
    "File3DOBJ",
    "File3DSTL",
    "File3DUSDZ",
    "File3DPLY",
    "File3DSPLAT",
    "File3DSPZ",
    "File3DKSPLAT",
    "File3DSplatAny",
    "File3DPointCloudAny",
    "Hooks",
    "HookKeyframes",
    "TimestepsRange",
    "LatentOperation",
    "FlowControl",
    "Accumulation",
    "Load3DCamera",
    "Load3DModelInfo",
    "Load3D",
    "Load3DAnimation",
    "Photomaker",
    "Point",
    "FaceAnalysis",
    "BBOX",
    "SEGS",
    "AnyType",
    "MultiType",
    "Tracks",
    "Dict",
    "Array",
    "Color",
    # Dynamic Types
    "MatchType",
    "DynamicCombo",
    "Autogrow",
    # Other classes
    "HiddenHolder",
    "Hidden",
    "NodeInfoV1",
    "Schema",
    "ComfyNode",
    "NodeOutput",
    "add_to_dict_v1",
    "V3Data",
    "ImageCompare",
    "PriceBadgeDepends",
    "PriceBadge",
    "BoundingBox",
    "BoundingBoxes",
    "Colors",
    "Curve",
    "Histogram",
    "Range",
    "NodeReplace",
]
