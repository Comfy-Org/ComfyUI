"""Comfy-specific type hinting"""

from typing import Literal, TypedDict, Optional
from typing_extensions import NotRequired
from abc import ABC, abstractmethod
from enum import Enum


class StrEnum(str, Enum):
    """Base class for string enums. Python's StrEnum is not available until 3.11."""

    def __str__(self) -> str:
        return self.value


class IO(StrEnum):
    """Node input/output data types.

    Includes functionality for ``"*"`` (`ANY`) and ``"MULTI,TYPES"``.
    """

    STRING = "STRING"
    IMAGE = "IMAGE"
    MASK = "MASK"
    LATENT = "LATENT"
    BOOLEAN = "BOOLEAN"
    INT = "INT"
    FLOAT = "FLOAT"
    COMBO = "COMBO"
    CONDITIONING = "CONDITIONING"
    SAMPLER = "SAMPLER"
    SIGMAS = "SIGMAS"
    GUIDER = "GUIDER"
    NOISE = "NOISE"
    CLIP = "CLIP"
    CONTROL_NET = "CONTROL_NET"
    VAE = "VAE"
    MODEL = "MODEL"
    LORA_MODEL = "LORA_MODEL"
    LOSS_MAP = "LOSS_MAP"
    CLIP_VISION = "CLIP_VISION"
    CLIP_VISION_OUTPUT = "CLIP_VISION_OUTPUT"
    STYLE_MODEL = "STYLE_MODEL"
    GLIGEN = "GLIGEN"
    UPSCALE_MODEL = "UPSCALE_MODEL"
    AUDIO = "AUDIO"
    WEBCAM = "WEBCAM"
    POINT = "POINT"
    FACE_ANALYSIS = "FACE_ANALYSIS"
    BBOX = "BBOX"
    SEGS = "SEGS"
    VIDEO = "VIDEO"

    ANY = "*"
    """Always matches any type, but at a price.

    Causes some functionality issues (e.g. reroutes, link types), and should be avoided whenever possible.
    """
    NUMBER = "FLOAT,INT"
    """A float or an int - could be either"""
    PRIMITIVE = "STRING,FLOAT,INT,BOOLEAN"
    """Could be any of: string, float, int, or bool"""

    def __ne__(self, value: object) -> bool:
        if self == "*" or value == "*":
            return False
        if not isinstance(value, str):
            return True
        a = frozenset(self.split(","))
        b = frozenset(value.split(","))
        return not (b.issubset(a) or a.issubset(b))


class RemoteInputOptions(TypedDict):
    route: str
    """The route to the remote source."""
    refresh_button: bool
    """Specifies whether to show a refresh button in the UI below the widget."""
    control_after_refresh: Literal["first", "last"]
    """Specifies the control after the refresh button is clicked. If "first", the first item will be automatically selected, and so on."""
    timeout: int
    """The maximum amount of time to wait for a response from the remote source in milliseconds."""
    max_retries: int
    """The maximum number of retries before aborting the request."""
    refresh: int
    """The TTL of the remote input's value in milliseconds. Specifies the interval at which the remote input's value is refreshed."""


class MultiSelectOptions(TypedDict):
    placeholder: NotRequired[str]
    """The placeholder text to display in the multi-select widget when no items are selected."""
    chip: NotRequired[bool]
    """Specifies whether to use chips instead of comma separated values for the multi-select widget."""


class InputTypeOptions(TypedDict):
    """Provides type hinting for the return type of the INPUT_TYPES node function.

    Due to IDE limitations with unions, for now all options are available for all types (e.g. `label_on` is hinted even when the type is not `IO.BOOLEAN`).

    Comfy Docs: https://docs.comfy.org/custom-nodes/backend/datatypes
    """

    default: NotRequired[bool | str | float | int | list | tuple]
    """The default value of the widget"""
    defaultInput: NotRequired[bool]
    """@deprecated in v1.16 frontend. v1.16 frontend allows input socket and widget to co-exist.
    - defaultInput on required inputs should be dropped.
    - defaultInput on optional inputs should be replaced with forceInput.
    Ref: https://github.com/Comfy-Org/ComfyUI_frontend/pull/3364
    """
    forceInput: NotRequired[bool]
    """Forces the input to be an input slot rather than a widget even a widget is available for the input type."""
    lazy: NotRequired[bool]
    """Declares that this input uses lazy evaluation"""
    rawLink: NotRequired[bool]
    """When a link exists, rather than receiving the evaluated value, you will receive the link (i.e. `["nodeId", <outputIndex>]`). Designed for node expansion."""
    tooltip: NotRequired[str]
    """Tooltip for the input (or widget), shown on pointer hover"""
    socketless: NotRequired[bool]
    """All inputs (including widgets) have an input socket to connect links. When ``true``, if there is a widget for this input, no socket will be created.
    Available from frontend v1.17.5
    Ref: https://github.com/Comfy-Org/ComfyUI_frontend/pull/3548
    """
    widgetType: NotRequired[str]
    """Specifies a type to be used for widget initialization if different from the input type.
    Available from frontend v1.18.0
    https://github.com/Comfy-Org/ComfyUI_frontend/pull/3550"""
    # class InputTypeNumber(InputTypeOptions):
    # default: float | int
    min: NotRequired[float]
    """The minimum value of a number (``FLOAT`` | ``INT``)"""
    max: NotRequired[float]
    """The maximum value of a number (``FLOAT`` | ``INT``)"""
    step: NotRequired[float]
    """The amount to increment or decrement a widget by when stepping up/down (``FLOAT`` | ``INT``)"""
    round: NotRequired[float]
    """Floats are rounded by this value (``FLOAT``)"""
    # class InputTypeBoolean(InputTypeOptions):
    # default: bool
    label_on: NotRequired[str]
    """The label to use in the UI when the bool is True (``BOOLEAN``)"""
    label_off: NotRequired[str]
    """The label to use in the UI when the bool is False (``BOOLEAN``)"""
    # class InputTypeString(InputTypeOptions):
    # default: str
    multiline: NotRequired[bool]
    """Use a multiline text box (``STRING``)"""
    placeholder: NotRequired[str]
    """Placeholder text to display in the UI when empty (``STRING``)"""
    # Deprecated:
    # defaultVal: str
    dynamicPrompts: NotRequired[bool]
    """Causes the front-end to evaluate dynamic prompts (``STRING``)"""
    # class InputTypeCombo(InputTypeOptions):
    image_upload: NotRequired[bool]
    """Specifies whether the input should have an image upload button and image preview attached to it. Requires that the input's name is `image`."""
    image_folder: NotRequired[Literal["input", "output", "temp"]]
    """Specifies which folder to get preview images from if the input has the ``image_upload`` flag.
    """
    remote: NotRequired[RemoteInputOptions]
    """Specifies the configuration for a remote input.
    Available after ComfyUI frontend v1.9.7
    https://github.com/Comfy-Org/ComfyUI_frontend/pull/2422"""
    control_after_generate: NotRequired[bool]
    """Specifies whether a control widget should be added to the input, adding options to automatically change the value after each prompt is queued. Currently only used for INT and COMBO types."""
    options: NotRequired[list[str | int | float]]
    """COMBO type only. Specifies the selectable options for the combo widget.
    Prefer:
    ["COMBO", {"options": ["Option 1", "Option 2", "Option 3"]}]
    Over:
    [["Option 1", "Option 2", "Option 3"]]
    """
    multi_select: NotRequired[MultiSelectOptions]
    """COMBO type only. Specifies the configuration for a multi-select widget.
    Available after ComfyUI frontend v1.13.4
    https://github.com/Comfy-Org/ComfyUI_frontend/pull/2987"""
    gradient_stops: NotRequired[list[dict]]
    """Gradient color stops for gradientslider display mode. Each stop is {"offset": float, "color": [r, g, b]}."""


class HiddenInputTypeDict(TypedDict):
    """Provides type hinting for the hidden entry of node INPUT_TYPES."""

    node_id: NotRequired[Literal["UNIQUE_ID"]]
    """UNIQUE_ID is the unique identifier of the node, and matches the id property of the node on the client side. It is commonly used in client-server communications (see messages)."""
    unique_id: NotRequired[Literal["UNIQUE_ID"]]
    """UNIQUE_ID is the unique identifier of the node, and matches the id property of the node on the client side. It is commonly used in client-server communications (see messages)."""
    prompt: NotRequired[Literal["PROMPT"]]
    """PROMPT is the complete prompt sent by the client to the server. See the prompt object for a full description."""
    extra_pnginfo: NotRequired[Literal["EXTRA_PNGINFO"]]
    """EXTRA_PNGINFO is a dictionary that will be copied into the metadata of any .png files saved. Custom nodes can store additional information in this dictionary for saving (or as a way to communicate with a downstream node)."""
    dynprompt: NotRequired[Literal["DYNPROMPT"]]
    """DYNPROMPT is an instance of comfy_execution.graph.DynamicPrompt. It differs from PROMPT in that it may mutate during the course of execution in response to Node Expansion."""


class InputTypeDict(TypedDict):
    """Provides type hinting for node INPUT_TYPES.

    Comfy Docs: https://docs.comfy.org/custom-nodes/backend/more_on_inputs
    """

    required: NotRequired[dict[str, tuple[IO, InputTypeOptions]]]
    """Describes all inputs that must be connected for the node to execute."""
    optional: NotRequired[dict[str, tuple[IO, InputTypeOptions]]]
    """Describes inputs which do not need to be connected."""
    hidden: NotRequired[HiddenInputTypeDict]
    """Offers advanced functionality and server-client communication.

    Comfy Docs: https://docs.comfy.org/custom-nodes/backend/more_on_inputs#hidden-inputs
    """


class ComfyNodeABC(ABC):
    """Abstract base class for Comfy nodes.  Includes the names and expected types of attributes.

    Comfy Docs: https://docs.comfy.org/custom-nodes/backend/server_overview
    """

    DESCRIPTION: str
    """Node description, shown as a tooltip when hovering over the node.

    Usage::

        # Explicitly define the description
        DESCRIPTION = "Example description here."

        # Use the docstring of the node class.
        DESCRIPTION = cleandoc(__doc__)
    """
    CATEGORY: str
    """The category of the node, as per the "Add Node" menu.

    Comfy Docs: https://docs.comfy.org/custom-nodes/backend/server_overview#category
    """
    EXPERIMENTAL: bool
    """Flags a node as experimental, informing users that it may change or not work as expected."""
    DEPRECATED: bool
    """Flags a node as deprecated, indicating to users that they should find alternatives to this node."""
    DEV_ONLY: bool
    """Flags a node as dev-only, hiding it from search/menus unless dev mode is enabled."""
    API_NODE: Optional[bool]
    """Flags a node as an API node. See: https://docs.comfy.org/tutorials/api-nodes/overview."""

    @classmethod
    @abstractmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        """Defines node inputs.

        * Must include the ``required`` key, which describes all inputs that must be connected for the node to execute.
        * The ``optional`` key can be added to describe inputs which do not need to be connected.
        * The ``hidden`` key offers some advanced functionality.  More info at: https://docs.comfy.org/custom-nodes/backend/more_on_inputs#hidden-inputs

        Comfy Docs: https://docs.comfy.org/custom-nodes/backend/server_overview#input-types
        """
        return {"required": {}}

    OUTPUT_NODE: bool
    """Flags this node as an output node, causing any inputs it requires to be executed.

    If a node is not connected to any output nodes, that node will not be executed.  Usage::

        OUTPUT_NODE = True

    From the docs:

    By default, a node is not considered an output. Set ``OUTPUT_NODE = True`` to specify that it is.

    Comfy Docs: https://docs.comfy.org/custom-nodes/backend/server_overview#output-node
    """
    INPUT_IS_LIST: bool
    """A flag indicating if this node implements the additional code necessary to deal with OUTPUT_IS_LIST nodes.

    All inputs of ``type`` will become ``list[type]``, regardless of how many items are passed in.  This also affects ``check_lazy_status``.

    From the docs:

    A node can also override the default input behaviour and receive the whole list in a single call. This is done by setting a class attribute `INPUT_IS_LIST` to ``True``.

    Comfy Docs: https://docs.comfy.org/custom-nodes/backend/lists#list-processing
    """
    OUTPUT_IS_LIST: tuple[bool, ...]
    """A tuple indicating which node outputs are lists, but will be connected to nodes that expect individual items.

    Connected nodes that do not implement `INPUT_IS_LIST` will be executed once for every item in the list.

    A ``tuple[bool]``, where the items match those in `RETURN_TYPES`::

        RETURN_TYPES = (IO.INT, IO.INT, IO.STRING)
        OUTPUT_IS_LIST = (True, True, False) # The string output will be handled normally

    From the docs:

    In order to tell Comfy that the list being returned should not be wrapped, but treated as a series of data for sequential processing,
    the node should provide a class attribute `OUTPUT_IS_LIST`, which is a ``tuple[bool]``, of the same length as `RETURN_TYPES`,
    specifying which outputs which should be so treated.

    Comfy Docs: https://docs.comfy.org/custom-nodes/backend/lists#list-processing
    """

    RETURN_TYPES: tuple[IO, ...]
    """A tuple representing the outputs of this node.

    Usage::

        RETURN_TYPES = (IO.INT, "INT", "CUSTOM_TYPE")

    Comfy Docs: https://docs.comfy.org/custom-nodes/backend/server_overview#return-types
    """
    RETURN_NAMES: tuple[str, ...]
    """The output slot names for each item in `RETURN_TYPES`, e.g. ``RETURN_NAMES = ("count", "filter_string")``

    Comfy Docs: https://docs.comfy.org/custom-nodes/backend/server_overview#return-names
    """
    OUTPUT_TOOLTIPS: tuple[str, ...]
    """A tuple of strings to use as tooltips for node outputs, one for each item in `RETURN_TYPES`."""
    FUNCTION: str
    """The name of the function to execute as a literal string, e.g. `FUNCTION = "execute"`

    Comfy Docs: https://docs.comfy.org/custom-nodes/backend/server_overview#function
    """


class CheckLazyMixin:
    """Provides a basic check_lazy_status implementation and type hinting for nodes that use lazy inputs."""

    def check_lazy_status(self, **kwargs) -> list[str]:
        """Returns a list of input names that should be evaluated.

        This basic mixin impl. requires all inputs.

        :kwargs: All node inputs will be included here.  If the input is ``None``, it should be assumed that it has not yet been evaluated.  \
            When using ``INPUT_IS_LIST = True``, unevaluated will instead be ``(None,)``.

        Params should match the nodes execution ``FUNCTION`` (self, and all inputs by name).
        Will be executed repeatedly until it returns an empty list, or all requested items were already evaluated (and sent as params).

        Comfy Docs: https://docs.comfy.org/custom-nodes/backend/lazy_evaluation#defining-check-lazy-status
        """

        need = [name for name in kwargs if kwargs[name] is None]
        return need


class FileLocator(TypedDict):
    """Provides type hinting for the file location"""

    filename: str
    """The filename of the file."""
    subfolder: str
    """The subfolder of the file."""
    type: Literal["input", "output", "temp"]
    """The root folder of the file."""
