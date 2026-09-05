import torch

class StubImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content": (['WHITE', 'BLACK', 'NOISE'],),
                "height": ("INT", {"default": 512, "min": 1, "max": 1024 ** 3, "step": 1}),
                "width": ("INT", {"default": 512, "min": 1, "max": 4096 ** 3, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 1024 ** 3, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stub_image"

    CATEGORY = "Testing/Stub Nodes"

    def stub_image(self, content, height, width, batch_size):
        if content == "WHITE":
            return (torch.ones(batch_size, height, width, 3),)
        elif content == "BLACK":
            return (torch.zeros(batch_size, height, width, 3),)
        elif content == "NOISE":
            return (torch.rand(batch_size, height, width, 3),)

class StubConstantImage:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "height": ("INT", {"default": 512, "min": 1, "max": 1024 ** 3, "step": 1}),
                "width": ("INT", {"default": 512, "min": 1, "max": 4096 ** 3, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 1024 ** 3, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stub_constant_image"

    CATEGORY = "Testing/Stub Nodes"

    def stub_constant_image(self, value, height, width, batch_size):
        return (torch.ones(batch_size, height, width, 3) * value,)

class StubMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "height": ("INT", {"default": 512, "min": 1, "max": 1024 ** 3, "step": 1}),
                "width": ("INT", {"default": 512, "min": 1, "max": 4096 ** 3, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 1024 ** 3, "step": 1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "stub_mask"

    CATEGORY = "Testing/Stub Nodes"

    def stub_mask(self, value, height, width, batch_size):
        return (torch.ones(batch_size, height, width) * value,)

class StubInt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -0xffffffff, "max": 0xffffffff, "step": 1}),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "stub_int"

    CATEGORY = "Testing/Stub Nodes"

    def stub_int(self, value):
        return (value,)

class StubFloat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "min": -1.0e38, "max": 1.0e38, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "stub_float"

    CATEGORY = "Testing/Stub Nodes"

    def stub_float(self, value):
        return (value,)

class StubStringOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "stub_string"

    CATEGORY = "Testing/Stub Nodes"

    def stub_string(self, value):
        return (value,)

class StubStringWithLength:
    """STRING input with declared bounds AND opted in to runtime validation (RUNTIME_INPUT_VALIDATION = True)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "hello", "minLength": 3, "maxLength": 10}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stub_string_with_length"
    RUNTIME_INPUT_VALIDATION = True

    CATEGORY = "Testing/Stub Nodes"

    def stub_string_with_length(self, text):
        return (torch.zeros(1, 64, 64, 3),)

class StubStringWithLengthNoFlag:
    """Same bounds as StubStringWithLength but NOT opted in - linked values must flow through unchecked."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "hello", "minLength": 3, "maxLength": 10}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stub_string_with_length_no_flag"

    CATEGORY = "Testing/Stub Nodes"

    def stub_string_with_length_no_flag(self, text):
        return (torch.zeros(1, 64, 64, 3),)

class StubIntWithBounds:
    """INT input with min/max bounds AND opted in to runtime validation."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 5, "min": 1, "max": 10}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stub_int_with_bounds"
    RUNTIME_INPUT_VALIDATION = True

    CATEGORY = "Testing/Stub Nodes"

    def stub_int_with_bounds(self, value):
        return (torch.zeros(1, 64, 64, 3),)

class StubComboWithOptions:
    """COMBO input opted in to runtime validation.

    Declares ``input_types`` in VALIDATE_INPUTS to bypass the engine's link-type compatibility
    check, allowing tests to link a STRING into a COMBO and exercise the runtime membership check.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "choice": (["RED", "GREEN", "BLUE"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stub_combo"
    RUNTIME_INPUT_VALIDATION = True

    CATEGORY = "Testing/Stub Nodes"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def stub_combo(self, choice):
        return (torch.zeros(1, 64, 64, 3),)

TEST_STUB_NODE_CLASS_MAPPINGS = {
    "StubImage": StubImage,
    "StubConstantImage": StubConstantImage,
    "StubMask": StubMask,
    "StubInt": StubInt,
    "StubFloat": StubFloat,
    "StubStringOutput": StubStringOutput,
    "StubStringWithLength": StubStringWithLength,
    "StubStringWithLengthNoFlag": StubStringWithLengthNoFlag,
    "StubIntWithBounds": StubIntWithBounds,
    "StubComboWithOptions": StubComboWithOptions,
}
TEST_STUB_NODE_DISPLAY_NAME_MAPPINGS = {
    "StubImage": "Stub Image",
    "StubConstantImage": "Stub Constant Image",
    "StubMask": "Stub Mask",
    "StubInt": "Stub Int",
    "StubFloat": "Stub Float",
    "StubStringOutput": "Stub String Output",
    "StubStringWithLength": "Stub String With Length",
    "StubStringWithLengthNoFlag": "Stub String With Length (No Flag)",
    "StubIntWithBounds": "Stub Int With Bounds",
    "StubComboWithOptions": "Stub Combo With Options",
}
