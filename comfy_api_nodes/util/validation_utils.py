import logging

import torch

from comfy_api.latest import Input


def get_image_dimensions(image: torch.Tensor) -> tuple[int, int]:
    if len(image.shape) == 4:
        return image.shape[1], image.shape[2]
    elif len(image.shape) == 3:
        return image.shape[0], image.shape[1]
    else:
        raise ValueError("Invalid image tensor shape.")


def validate_image_dimensions(
    image: torch.Tensor,
    min_width: int | None = None,
    max_width: int | None = None,
    min_height: int | None = None,
    max_height: int | None = None,
):
    height, width = get_image_dimensions(image)

    if min_width is not None and width < min_width:
        raise ValueError(f"Image width must be at least {min_width}px, got {width}px")
    if max_width is not None and width > max_width:
        raise ValueError(f"Image width must be at most {max_width}px, got {width}px")
    if min_height is not None and height < min_height:
        raise ValueError(f"Image height must be at least {min_height}px, got {height}px")
    if max_height is not None and height > max_height:
        raise ValueError(f"Image height must be at most {max_height}px, got {height}px")


def validate_image_aspect_ratio(
    image: torch.Tensor,
    min_ratio: tuple[float, float] | None = None,  # e.g. (1, 4)
    max_ratio: tuple[float, float] | None = None,  # e.g. (4, 1)
    *,
    strict: bool = True,  # True -> (min, max); False -> [min, max]
) -> float:
    """Validates that image aspect ratio is within min and max. If a bound is None, that side is not checked."""
    w, h = get_image_dimensions(image)
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image dimensions: {w}x{h}")
    ar = w / h
    _assert_ratio_bounds(ar, min_ratio=min_ratio, max_ratio=max_ratio, strict=strict)
    return ar


def validate_images_aspect_ratio_closeness(
    first_image: torch.Tensor,
    second_image: torch.Tensor,
    min_rel: float,  # e.g. 0.8
    max_rel: float,  # e.g. 1.25
    *,
    strict: bool = False,  # True -> (min, max); False -> [min, max]
) -> float:
    """
    Validates that the two images' aspect ratios are 'close'.
    The closeness factor is C = max(ar1, ar2) / min(ar1, ar2)  (C >= 1).
    We require C <= limit, where limit = max(max_rel, 1.0 / min_rel).

    Returns the computed closeness factor C.
    """
    w1, h1 = get_image_dimensions(first_image)
    w2, h2 = get_image_dimensions(second_image)
    if min(w1, h1, w2, h2) <= 0:
        raise ValueError("Invalid image dimensions")
    ar1 = w1 / h1
    ar2 = w2 / h2
    closeness = max(ar1, ar2) / min(ar1, ar2)
    limit = max(max_rel, 1.0 / min_rel)
    if (closeness >= limit) if strict else (closeness > limit):
        raise ValueError(
            f"Aspect ratios must be close: ar1/ar2={ar1/ar2:.2g}, "
            f"allowed range {min_rel}–{max_rel} (limit {limit:.2g})."
        )
    return closeness


def validate_aspect_ratio_string(
    aspect_ratio: str,
    min_ratio: tuple[float, float] | None = None,  # e.g. (1, 4)
    max_ratio: tuple[float, float] | None = None,  # e.g. (4, 1)
    *,
    strict: bool = False,  # True -> (min, max); False -> [min, max]
) -> float:
    """Parses 'X:Y' and validates it against optional bounds. Returns the numeric ratio."""
    ar = _parse_aspect_ratio_string(aspect_ratio)
    _assert_ratio_bounds(ar, min_ratio=min_ratio, max_ratio=max_ratio, strict=strict)
    return ar


def validate_video_dimensions(
    video: Input.Video,
    min_width: int | None = None,
    max_width: int | None = None,
    min_height: int | None = None,
    max_height: int | None = None,
):
    try:
        width, height = video.get_dimensions()
    except Exception as e:
        logging.error("Error getting dimensions of video: %s", e)
        return

    if min_width is not None and width < min_width:
        raise ValueError(f"Video width must be at least {min_width}px, got {width}px")
    if max_width is not None and width > max_width:
        raise ValueError(f"Video width must be at most {max_width}px, got {width}px")
    if min_height is not None and height < min_height:
        raise ValueError(f"Video height must be at least {min_height}px, got {height}px")
    if max_height is not None and height > max_height:
        raise ValueError(f"Video height must be at most {max_height}px, got {height}px")


def validate_video_duration(
    video: Input.Video,
    min_duration: float | None = None,
    max_duration: float | None = None,
):
    try:
        duration = video.get_duration()
    except Exception as e:
        logging.error("Error getting duration of video: %s", e)
        return

    epsilon = 0.0001
    if min_duration is not None and min_duration - epsilon > duration:
        raise ValueError(f"Video duration must be at least {min_duration}s, got {duration}s")
    if max_duration is not None and duration > max_duration + epsilon:
        raise ValueError(f"Video duration must be at most {max_duration}s, got {duration}s")


def validate_video_frame_count(
    video: Input.Video,
    min_frame_count: int | None = None,
    max_frame_count: int | None = None,
):
    try:
        frame_count = video.get_frame_count()
    except Exception as e:
        logging.error("Error getting frame count of video: %s", e)
        return

    if min_frame_count is not None and min_frame_count > frame_count:
        raise ValueError(f"Video frame count must be at least {min_frame_count}, got {frame_count}")
    if max_frame_count is not None and frame_count > max_frame_count:
        raise ValueError(f"Video frame count must be at most {max_frame_count}, got {frame_count}")


def get_number_of_images(images):
    if isinstance(images, torch.Tensor):
        return images.shape[0] if images.ndim >= 4 else 1
    return len(images)


def validate_audio_duration(
    audio: Input.Audio,
    min_duration: float | None = None,
    max_duration: float | None = None,
) -> None:
    sr = int(audio["sample_rate"])
    dur = int(audio["waveform"].shape[-1]) / sr
    eps = 1.0 / sr
    if min_duration is not None and dur + eps < min_duration:
        raise ValueError(f"Audio duration must be at least {min_duration}s, got {dur + eps:.2f}s")
    if max_duration is not None and dur - eps > max_duration:
        raise ValueError(f"Audio duration must be at most {max_duration}s, got {dur - eps:.2f}s")


def validate_string(
    string: str,
    strip_whitespace=True,
    field_name="prompt",
    min_length=None,
    max_length=None,
):
    if string is None:
        raise Exception(f"Field '{field_name}' cannot be empty.")
    if strip_whitespace:
        string = string.strip()
    if min_length and len(string) < min_length:
        raise Exception(
            f"Field '{field_name}' cannot be shorter than {min_length} characters; was {len(string)} characters long."
        )
    if max_length and len(string) > max_length:
        raise Exception(
            f"Field '{field_name}' cannot be longer than {max_length} characters; was {len(string)} characters long."
        )


def validate_container_format_is_mp4(video: Input.Video) -> None:
    """Validates video container format is MP4."""
    container_format = video.get_container_format()
    if container_format not in ["mp4", "mov,mp4,m4a,3gp,3g2,mj2"]:
        raise ValueError(f"Only MP4 container format supported. Got: {container_format}")


def _ratio_from_tuple(r: tuple[float, float]) -> float:
    a, b = r
    if a <= 0 or b <= 0:
        raise ValueError(f"Ratios must be positive, got {a}:{b}.")
    return a / b


def _assert_ratio_bounds(
    ar: float,
    *,
    min_ratio: tuple[float, float] | None = None,
    max_ratio: tuple[float, float] | None = None,
    strict: bool = True,
) -> None:
    """Validate a numeric aspect ratio against optional min/max ratio bounds."""
    lo = _ratio_from_tuple(min_ratio) if min_ratio is not None else None
    hi = _ratio_from_tuple(max_ratio) if max_ratio is not None else None

    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo  # normalize order if caller swapped them

    if lo is not None:
        if (ar <= lo) if strict else (ar < lo):
            op = "<" if strict else "≤"
            raise ValueError(f"Aspect ratio `{ar:.2g}` must be {op} {lo:.2g}.")
    if hi is not None:
        if (ar >= hi) if strict else (ar > hi):
            op = "<" if strict else "≤"
            raise ValueError(f"Aspect ratio `{ar:.2g}` must be {op} {hi:.2g}.")


def _parse_aspect_ratio_string(ar_str: str) -> float:
    """Parse 'X:Y' with integer parts into a positive float ratio X/Y."""
    parts = ar_str.split(":")
    if len(parts) != 2:
        raise ValueError(f"Aspect ratio must be 'X:Y' (e.g., 16:9), got '{ar_str}'.")
    try:
        a = int(parts[0].strip())
        b = int(parts[1].strip())
    except ValueError as exc:
        raise ValueError(f"Aspect ratio must contain integers separated by ':', got '{ar_str}'.") from exc
    if a <= 0 or b <= 0:
        raise ValueError(f"Aspect ratio parts must be positive integers, got {a}:{b}.")
    return a / b
