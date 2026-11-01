from comfy_execution.graph_utils import GraphBuilder
from .tools import VariantSupport

@VariantSupport()
class TestAccumulateNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "to_add": ("*",),
            },
            "optional": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION",)
    FUNCTION = "accumulate"

    CATEGORY = "Testing/Lists"

    def accumulate(self, to_add, accumulation = None):
        if accumulation is None:
            value = [to_add]
        else:
            value = accumulation["accum"] + [to_add]
        return ({"accum": value},)

@VariantSupport()
class TestAccumulationHeadNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION", "*",)
    FUNCTION = "accumulation_head"

    CATEGORY = "Testing/Lists"

    def accumulation_head(self, accumulation):
        accum = accumulation["accum"]
        if len(accum) == 0:
            return (accumulation, None)
        else:
            return ({"accum": accum[1:]}, accum[0])

class TestAccumulationTailNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION", "*",)
    FUNCTION = "accumulation_tail"

    CATEGORY = "Testing/Lists"

    def accumulation_tail(self, accumulation):
        accum = accumulation["accum"]
        if len(accum) == 0:
            return (None, accumulation)
        else:
            return ({"accum": accum[:-1]}, accum[-1])

@VariantSupport()
class TestAccumulationToListNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("*",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "accumulation_to_list"

    CATEGORY = "Testing/Lists"

    def accumulation_to_list(self, accumulation):
        return (accumulation["accum"],)

@VariantSupport()
class TestListToAccumulationNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list": ("*",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION",)
    INPUT_IS_LIST = (True,)

    FUNCTION = "list_to_accumulation"

    CATEGORY = "Testing/Lists"

    def list_to_accumulation(self, list):
        return ({"accum": list},)

@VariantSupport()
class TestAccumulationGetLengthNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("INT",)

    FUNCTION = "accumlength"

    CATEGORY = "Testing/Lists"

    def accumlength(self, accumulation):
        return (len(accumulation['accum']),)

@VariantSupport()
class TestAccumulationGetItemNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
                "index": ("INT", {"default":0, "step":1})
            },
        }

    RETURN_TYPES = ("*",)

    FUNCTION = "get_item"

    CATEGORY = "Testing/Lists"

    def get_item(self, accumulation, index):
        return (accumulation['accum'][index],)

@VariantSupport()
class TestAccumulationSetItemNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
                "index": ("INT", {"default":0, "step":1}),
                "value": ("*",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION",)

    FUNCTION = "set_item"

    CATEGORY = "Testing/Lists"

    def set_item(self, accumulation, index, value):
        new_accum = accumulation['accum'][:]
        new_accum[index] = value
        return ({"accum": new_accum},)

class TestIntMathOperation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 1}),
                "b": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 1}),
                "operation": (["add", "subtract", "multiply", "divide", "modulo", "power"],),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "int_math_operation"

    CATEGORY = "Testing/Logic"

    def int_math_operation(self, a, b, operation):
        if operation == "add":
            return (a + b,)
        elif operation == "subtract":
            return (a - b,)
        elif operation == "multiply":
            return (a * b,)
        elif operation == "divide":
            return (a // b,)
        elif operation == "modulo":
            return (a % b,)
        elif operation == "power":
            return (a ** b,)


from .flow_control import NUM_FLOW_SOCKETS
@VariantSupport()
class TestForLoopOpen:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remaining": ("INT", {"default": 1, "min": 0, "max": 100000, "step": 1}),
            },
            "optional": {
                f"initial_value{i}": ("*",) for i in range(1, NUM_FLOW_SOCKETS)
            },
            "hidden": {
                "initial_value0": ("*",)
            }
        }

    RETURN_TYPES = tuple(["FLOW_CONTROL", "INT",] + ["*"] * (NUM_FLOW_SOCKETS-1))
    RETURN_NAMES = tuple(["flow_control", "remaining"] + [f"value{i}" for i in range(1, NUM_FLOW_SOCKETS)])
    FUNCTION = "for_loop_open"

    CATEGORY = "Testing/Flow"

    def for_loop_open(self, remaining, **kwargs):
        graph = GraphBuilder()
        if "initial_value0" in kwargs:
            remaining = kwargs["initial_value0"]
        graph.node("TestWhileLoopOpen", condition=remaining, initial_value0=remaining, **{(f"initial_value{i}"): kwargs.get(f"initial_value{i}", None) for i in range(1, NUM_FLOW_SOCKETS)})
        outputs = [kwargs.get(f"initial_value{i}", None) for i in range(1, NUM_FLOW_SOCKETS)]
        return {
            "result": tuple(["stub", remaining] + outputs),
            "expand": graph.finalize(),
        }

@VariantSupport()
class TestForLoopClose:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow_control": ("FLOW_CONTROL", {"rawLink": True}),
            },
            "optional": {
                f"initial_value{i}": ("*",{"rawLink": True}) for i in range(1, NUM_FLOW_SOCKETS)
            },
        }

    RETURN_TYPES = tuple(["*"] * (NUM_FLOW_SOCKETS-1))
    RETURN_NAMES = tuple([f"value{i}" for i in range(1, NUM_FLOW_SOCKETS)])
    FUNCTION = "for_loop_close"

    CATEGORY = "Testing/Flow"

    def for_loop_close(self, flow_control, **kwargs):
        graph = GraphBuilder()
        while_open = flow_control[0]
        sub = graph.node("TestIntMathOperation", operation="subtract", a=[while_open,1], b=1)
        cond = graph.node("TestToBoolNode", value=sub.out(0))
        input_values = {f"initial_value{i}": kwargs.get(f"initial_value{i}", None) for i in range(1, NUM_FLOW_SOCKETS)}
        while_close = graph.node("TestWhileLoopClose",
                flow_control=flow_control,
                condition=cond.out(0),
                initial_value0=sub.out(0),
                **input_values)
        return {
            "result": tuple([while_close.out(i) for i in range(1, NUM_FLOW_SOCKETS)]),
            "expand": graph.finalize(),
        }

NUM_LIST_SOCKETS = 10
@VariantSupport()
class TestMakeListNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value1": ("*",),
            },
            "optional": {
                f"value{i}": ("*",) for i in range(1, NUM_LIST_SOCKETS)
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "make_list"
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "Testing/Lists"

    def make_list(self, **kwargs):
        result = []
        for i in range(NUM_LIST_SOCKETS):
            if f"value{i}" in kwargs:
                result.append(kwargs[f"value{i}"])
        return (result,)

@VariantSupport()
class TestArrayValueEcho:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "values": ("ARRAY",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "emit"
    CATEGORY = "Testing/Lists"
    OUTPUT_NODE = True

    def emit(self, values):
        return {"ui": {"arrays": [values]}}

UTILITY_NODE_CLASS_MAPPINGS = {
    "TestAccumulateNode": TestAccumulateNode,
    "TestAccumulationHeadNode": TestAccumulationHeadNode,
    "TestAccumulationTailNode": TestAccumulationTailNode,
    "TestAccumulationToListNode": TestAccumulationToListNode,
    "TestListToAccumulationNode": TestListToAccumulationNode,
    "TestAccumulationGetLengthNode": TestAccumulationGetLengthNode,
    "TestAccumulationGetItemNode": TestAccumulationGetItemNode,
    "TestAccumulationSetItemNode": TestAccumulationSetItemNode,
    "TestForLoopOpen": TestForLoopOpen,
    "TestForLoopClose": TestForLoopClose,
    "TestIntMathOperation": TestIntMathOperation,
    "TestMakeListNode": TestMakeListNode,
    "TestArrayValueEcho": TestArrayValueEcho,
}
UTILITY_NODE_DISPLAY_NAME_MAPPINGS = {
    "TestAccumulateNode": "Accumulate",
    "TestAccumulationHeadNode": "Accumulation Head",
    "TestAccumulationTailNode": "Accumulation Tail",
    "TestAccumulationToListNode": "Accumulation to List",
    "TestListToAccumulationNode": "List to Accumulation",
    "TestAccumulationGetLengthNode": "Accumulation Get Length",
    "TestAccumulationGetItemNode": "Accumulation Get Item",
    "TestAccumulationSetItemNode": "Accumulation Set Item",
    "TestForLoopOpen": "For Loop Open",
    "TestForLoopClose": "For Loop Close",
    "TestIntMathOperation": "Int Math Operation",
    "TestMakeListNode": "Make List",
    "TestArrayValueEcho": "Array Value Echo",
}
