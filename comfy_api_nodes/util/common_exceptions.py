class NetworkError(Exception):
    """Base exception for network-related errors with diagnostic information."""


class LocalNetworkError(NetworkError):
    """Exception raised when local network connectivity issues are detected."""


class ApiServerError(NetworkError):
    """Exception raised when the API server is unreachable but internet is working."""


class ApiResponseError(Exception):
    def __init__(self, message: str, sensitive_values: tuple[str, ...] = ()):
        super().__init__(message)
        self.sensitive_values = sensitive_values


class ProcessingInterrupted(Exception):
    """Operation was interrupted by user/runtime via processing_interrupted()."""
