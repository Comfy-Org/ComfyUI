class StartLoop:
    LOOP_BOUNDARY = "start"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("INT", "BOOLEAN", "BOOLEAN", "*", "*")
    RETURN_NAMES = ("iteration_index", "is_first", "is_last", "list_item", "current_iteration_value")
    CATEGORY = "utilities/looping"


class EndLoop:
    LOOP_BOUNDARY = "end"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"output_value": ("*",)}}

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("outputs",)
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "utilities/looping"


NODE_CLASS_MAPPINGS = {
    "StartLoop": StartLoop,
    "EndLoop": EndLoop,
}

NODE_DISPLAY_NAME_MAPPINGS = {"StartLoop": "Start Loop", "EndLoop": "End Loop"}
