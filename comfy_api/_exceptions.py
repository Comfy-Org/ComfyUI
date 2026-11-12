class InterruptProcessingException(BaseException):
    """Raised when the user cancels the prompt; ComfyUI reports it as an interruption, not an error.

    It is a `BaseException`, so `except Exception` lets it through to ComfyUI.
    """
