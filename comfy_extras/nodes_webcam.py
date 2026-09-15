import nodes
import folder_paths

MAX_RESOLUTION = nodes.MAX_RESOLUTION


class WebcamCapture(nodes.LoadImage):
    SEARCH_ALIASES = ["camera input", "live capture", "camera feed", "snapshot"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("WEBCAM", {}),
                "width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "capture_on_queue": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_capture"

    CATEGORY = "image"

    def load_capture(self, image, **kwargs):
        return super().load_image(folder_paths.get_annotated_filepath(image))

    @classmethod
    def IS_CHANGED(cls, image, width, height, capture_on_queue):
        if capture_on_queue:
            return float("NaN")
        return super().IS_CHANGED(image)


NODE_CLASS_MAPPINGS = {
    "WebcamCapture": WebcamCapture,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WebcamCapture": "Webcam Capture",
}
