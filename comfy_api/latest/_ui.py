from __future__ import annotations

import json
import os
import random
import uuid
from io import BytesIO

import av
import numpy as np
import torch
try:
    import torchaudio
    TORCH_AUDIO_AVAILABLE = True
except:
    TORCH_AUDIO_AVAILABLE = False
from PIL import Image as PILImage
from PIL.PngImagePlugin import PngInfo

import folder_paths

# used for image preview
from comfy.cli_args import args
from ._io import ComfyNode, FolderType, Image, _UIOutput


class SavedResult(dict):
    def __init__(self, filename: str, subfolder: str, type: FolderType):
        super().__init__(filename=filename, subfolder=subfolder,type=type.value)

    @property
    def filename(self) -> str:
        return self["filename"]

    @property
    def subfolder(self) -> str:
        return self["subfolder"]

    @property
    def type(self) -> FolderType:
        return FolderType(self["type"])


class SavedImages(_UIOutput):
    """A UI output class to represent one or more saved images, potentially animated."""
    def __init__(self, results: list[SavedResult], is_animated: bool = False):
        super().__init__()
        self.results = results
        self.is_animated = is_animated

    def as_dict(self) -> dict:
        data = {"images": self.results}
        if self.is_animated:
            data["animated"] = (True,)
        return data


class SavedAudios(_UIOutput):
    """UI wrapper around one or more audio files on disk (FLAC / MP3 / Opus)."""
    def __init__(self, results: list[SavedResult]):
        super().__init__()
        self.results = results

    def as_dict(self) -> dict:
        return {"audio": self.results}


def _get_directory_by_folder_type(folder_type: FolderType) -> str:
    if folder_type == FolderType.input:
        return folder_paths.get_input_directory()
    if folder_type == FolderType.output:
        return folder_paths.get_output_directory()
    return folder_paths.get_temp_directory()


class ImageSaveHelper:
    """A helper class with static methods to handle image saving and metadata."""

    @staticmethod
    def _convert_tensor_to_pil(image_tensor: torch.Tensor) -> PILImage.Image:
        """Converts a single torch tensor to a PIL Image."""
        return PILImage.fromarray(np.clip(255.0 * image_tensor.cpu().numpy(), 0, 255).astype(np.uint8))

    @staticmethod
    def _create_png_metadata(cls: type[ComfyNode] | None) -> PngInfo | None:
        """Creates a PngInfo object with prompt and extra_pnginfo."""
        if args.disable_metadata or cls is None or not cls.hidden:
            return None
        metadata = PngInfo()
        if cls.hidden.prompt:
            metadata.add_text("prompt", json.dumps(cls.hidden.prompt))
        if cls.hidden.extra_pnginfo:
            for x in cls.hidden.extra_pnginfo:
                metadata.add_text(x, json.dumps(cls.hidden.extra_pnginfo[x]))
        return metadata

    @staticmethod
    def _create_animated_png_metadata(cls: type[ComfyNode] | None) -> PngInfo | None:
        """Creates a PngInfo object with prompt and extra_pnginfo for animated PNGs (APNG)."""
        if args.disable_metadata or cls is None or not cls.hidden:
            return None
        metadata = PngInfo()
        if cls.hidden.prompt:
            metadata.add(
                b"comf",
                "prompt".encode("latin-1", "strict")
                + b"\0"
                + json.dumps(cls.hidden.prompt).encode("latin-1", "strict"),
                after_idat=True,
            )
        if cls.hidden.extra_pnginfo:
            for x in cls.hidden.extra_pnginfo:
                metadata.add(
                    b"comf",
                    x.encode("latin-1", "strict")
                    + b"\0"
                    + json.dumps(cls.hidden.extra_pnginfo[x]).encode("latin-1", "strict"),
                    after_idat=True,
                )
        return metadata

    @staticmethod
    def _create_webp_metadata(pil_image: PILImage.Image, cls: type[ComfyNode] | None) -> PILImage.Exif:
        """Creates EXIF metadata bytes for WebP images."""
        exif_data = pil_image.getexif()
        if args.disable_metadata or cls is None or cls.hidden is None:
            return exif_data
        if cls.hidden.prompt is not None:
            exif_data[0x0110] = "prompt:{}".format(json.dumps(cls.hidden.prompt))  # EXIF 0x0110 = Model
        if cls.hidden.extra_pnginfo is not None:
            inital_exif_tag = 0x010F  # EXIF 0x010f = Make
            for key, value in cls.hidden.extra_pnginfo.items():
                exif_data[inital_exif_tag] = "{}:{}".format(key, json.dumps(value))
                inital_exif_tag -= 1
        return exif_data

    @staticmethod
    def save_images(
        images, filename_prefix: str, folder_type: FolderType, cls: type[ComfyNode] | None, compress_level = 4,
    ) -> list[SavedResult]:
        """Saves a batch of images as individual PNG files."""
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, _get_directory_by_folder_type(folder_type), images[0].shape[1], images[0].shape[0]
        )
        results = []
        metadata = ImageSaveHelper._create_png_metadata(cls)
        for batch_number, image_tensor in enumerate(images):
            img = ImageSaveHelper._convert_tensor_to_pil(image_tensor)
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=compress_level)
            results.append(SavedResult(file, subfolder, folder_type))
            counter += 1
        return results

    @staticmethod
    def get_save_images_ui(images, filename_prefix: str, cls: type[ComfyNode] | None, compress_level=4) -> SavedImages:
        """Saves a batch of images and returns a UI object for the node output."""
        return SavedImages(
                ImageSaveHelper.save_images(
                images,
                filename_prefix=filename_prefix,
                folder_type=FolderType.output,
                cls=cls,
                compress_level=compress_level,
            )
        )

    @staticmethod
    def save_animated_png(
        images, filename_prefix: str, folder_type: FolderType, cls: type[ComfyNode] | None, fps: float, compress_level: int
    ) -> SavedResult:
        """Saves a batch of images as a single animated PNG."""
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, _get_directory_by_folder_type(folder_type), images[0].shape[1], images[0].shape[0]
        )
        pil_images = [ImageSaveHelper._convert_tensor_to_pil(img) for img in images]
        metadata = ImageSaveHelper._create_animated_png_metadata(cls)
        file = f"{filename}_{counter:05}_.png"
        save_path = os.path.join(full_output_folder, file)
        pil_images[0].save(
            save_path,
            pnginfo=metadata,
            compress_level=compress_level,
            save_all=True,
            duration=int(1000.0 / fps),
            append_images=pil_images[1:],
        )
        return SavedResult(file, subfolder, folder_type)

    @staticmethod
    def get_save_animated_png_ui(
        images, filename_prefix: str, cls: type[ComfyNode] | None, fps: float, compress_level: int
    ) -> SavedImages:
        """Saves an animated PNG and returns a UI object for the node output."""
        result = ImageSaveHelper.save_animated_png(
            images,
            filename_prefix=filename_prefix,
            folder_type=FolderType.output,
            cls=cls,
            fps=fps,
            compress_level=compress_level,
        )
        return SavedImages([result], is_animated=len(images) > 1)

    @staticmethod
    def save_animated_webp(
        images,
        filename_prefix: str,
        folder_type: FolderType,
        cls: type[ComfyNode] | None,
        fps: float,
        lossless: bool,
        quality: int,
        method: int,
    ) -> SavedResult:
        """Saves a batch of images as a single animated WebP."""
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, _get_directory_by_folder_type(folder_type), images[0].shape[1], images[0].shape[0]
        )
        pil_images = [ImageSaveHelper._convert_tensor_to_pil(img) for img in images]
        pil_exif = ImageSaveHelper._create_webp_metadata(pil_images[0], cls)
        file = f"{filename}_{counter:05}_.webp"
        pil_images[0].save(
            os.path.join(full_output_folder, file),
            save_all=True,
            duration=int(1000.0 / fps),
            append_images=pil_images[1:],
            exif=pil_exif,
            lossless=lossless,
            quality=quality,
            method=method,
        )
        return SavedResult(file, subfolder, folder_type)

    @staticmethod
    def get_save_animated_webp_ui(
        images,
        filename_prefix: str,
        cls: type[ComfyNode] | None,
        fps: float,
        lossless: bool,
        quality: int,
        method: int,
    ) -> SavedImages:
        """Saves an animated WebP and returns a UI object for the node output."""
        result = ImageSaveHelper.save_animated_webp(
            images,
            filename_prefix=filename_prefix,
            folder_type=FolderType.output,
            cls=cls,
            fps=fps,
            lossless=lossless,
            quality=quality,
            method=method,
        )
        return SavedImages([result], is_animated=len(images) > 1)


class AudioSaveHelper:
    """A helper class with static methods to handle audio saving and metadata."""
    _OPUS_RATES = [8000, 12000, 16000, 24000, 48000]

    @staticmethod
    def save_audio(
        audio: dict,
        filename_prefix: str,
        folder_type: FolderType,
        cls: type[ComfyNode] | None,
        format: str = "flac",
        quality: str = "128k",
    ) -> list[SavedResult]:
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, _get_directory_by_folder_type(folder_type)
        )

        metadata = {}
        if not args.disable_metadata and cls is not None:
            if cls.hidden.prompt is not None:
                metadata["prompt"] = json.dumps(cls.hidden.prompt)
            if cls.hidden.extra_pnginfo is not None:
                for x in cls.hidden.extra_pnginfo:
                    metadata[x] = json.dumps(cls.hidden.extra_pnginfo[x])

        results = []
        for batch_number, waveform in enumerate(audio["waveform"].cpu()):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}.{format}"
            output_path = os.path.join(full_output_folder, file)

            # Use original sample rate initially
            sample_rate = audio["sample_rate"]

            # Handle Opus sample rate requirements
            if format == "opus":
                if sample_rate > 48000:
                    sample_rate = 48000
                elif sample_rate not in AudioSaveHelper._OPUS_RATES:
                    # Find the next highest supported rate
                    for rate in sorted(AudioSaveHelper._OPUS_RATES):
                        if rate > sample_rate:
                            sample_rate = rate
                            break
                    if sample_rate not in AudioSaveHelper._OPUS_RATES:  # Fallback if still not supported
                        sample_rate = 48000

                # Resample if necessary
                if sample_rate != audio["sample_rate"]:
                    if not TORCH_AUDIO_AVAILABLE:
                        raise Exception("torchaudio is not available; cannot resample audio.")
                    waveform = torchaudio.functional.resample(waveform, audio["sample_rate"], sample_rate)

            # Create output with specified format
            output_buffer = BytesIO()
            output_container = av.open(output_buffer, mode="w", format=format)

            # Set metadata on the container
            for key, value in metadata.items():
                output_container.metadata[key] = value

            layout = "mono" if waveform.shape[0] == 1 else "stereo"
            # Set up the output stream with appropriate properties
            if format == "opus":
                out_stream = output_container.add_stream("libopus", rate=sample_rate, layout=layout)
                if quality == "64k":
                    out_stream.bit_rate = 64000
                elif quality == "96k":
                    out_stream.bit_rate = 96000
                elif quality == "128k":
                    out_stream.bit_rate = 128000
                elif quality == "192k":
                    out_stream.bit_rate = 192000
                elif quality == "320k":
                    out_stream.bit_rate = 320000
            elif format == "mp3":
                out_stream = output_container.add_stream("libmp3lame", rate=sample_rate, layout=layout)
                if quality == "V0":
                    # TODO i would really love to support V3 and V5 but there doesn't seem to be a way to set the qscale level, the property below is a bool
                    out_stream.codec_context.qscale = 1
                elif quality == "128k":
                    out_stream.bit_rate = 128000
                elif quality == "320k":
                    out_stream.bit_rate = 320000
            else:  # format == "flac":
                out_stream = output_container.add_stream("flac", rate=sample_rate, layout=layout)

            frame = av.AudioFrame.from_ndarray(
                waveform.movedim(0, 1).reshape(1, -1).float().numpy(),
                format="flt",
                layout=layout,
            )
            frame.sample_rate = sample_rate
            frame.pts = 0
            output_container.mux(out_stream.encode(frame))

            # Flush encoder
            output_container.mux(out_stream.encode(None))

            # Close containers
            output_container.close()

            # Write the output to file
            output_buffer.seek(0)
            with open(output_path, "wb") as f:
                f.write(output_buffer.getbuffer())

            results.append(SavedResult(file, subfolder, folder_type))
            counter += 1

        return results

    @staticmethod
    def get_save_audio_ui(
        audio, filename_prefix: str, cls: type[ComfyNode] | None, format: str = "flac", quality: str = "128k",
    ) -> SavedAudios:
        """Save and instantly wrap for UI."""
        return SavedAudios(
            AudioSaveHelper.save_audio(
                audio,
                filename_prefix=filename_prefix,
                folder_type=FolderType.output,
                cls=cls,
                format=format,
                quality=quality,
            )
        )


class PreviewImage(_UIOutput):
    def __init__(self, image: Image.Type, animated: bool = False, cls: type[ComfyNode] = None, **kwargs):
        self.values = ImageSaveHelper.save_images(
            image,
            filename_prefix="ComfyUI_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5)),
            folder_type=FolderType.temp,
            cls=cls,
            compress_level=1,
        )
        self.animated = animated

    def as_dict(self):
        return {
            "images": self.values,
            "animated": (self.animated,)
        }


class PreviewMask(PreviewImage):
    def __init__(self, mask: PreviewMask.Type, animated: bool=False, cls: ComfyNode=None, **kwargs):
        preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        super().__init__(preview, animated, cls, **kwargs)


class PreviewAudio(_UIOutput):
    def __init__(self, audio: dict, cls: type[ComfyNode] = None, **kwargs):
        self.values = AudioSaveHelper.save_audio(
            audio,
            filename_prefix="ComfyUI_temp_" + "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5)),
            folder_type=FolderType.temp,
            cls=cls,
            format="flac",
            quality="128k",
        )

    def as_dict(self) -> dict:
        return {"audio": self.values}


class PreviewVideo(_UIOutput):
    def __init__(self, values: list[SavedResult | dict], **kwargs):
        self.values = values

    def as_dict(self):
        return {"images": self.values, "animated": (True,)}


class PreviewUI3D(_UIOutput):
    def __init__(self, model_file, camera_info, **kwargs):
        self.model_file = model_file
        self.camera_info = camera_info
        self.bg_image_path = None
        bg_image = kwargs.get("bg_image", None)
        if bg_image is not None:
            img_array = (bg_image[0].cpu().numpy() * 255).astype(np.uint8)
            img = PILImage.fromarray(img_array)
            temp_dir = folder_paths.get_temp_directory()
            filename = f"bg_{uuid.uuid4().hex}.png"
            bg_image_path = os.path.join(temp_dir, filename)
            img.save(bg_image_path, compress_level=1)
            self.bg_image_path = f"temp/{filename}"

    def as_dict(self):
        return {"result": [self.model_file, self.camera_info, self.bg_image_path]}


class PreviewUI3DAdvanced(_UIOutput):
    def __init__(self, model_file, camera_info, model_3d_info):
        self.model_file = model_file
        self.camera_info = camera_info
        self.model_3d_info = model_3d_info

    def as_dict(self):
        return {"result": [self.model_file, self.camera_info, self.model_3d_info]}


class PreviewText(_UIOutput):
    def __init__(self, value: str, **kwargs):
        self.value = value

    def as_dict(self):
        return {"text": (self.value,)}


__all__ = [
    "SavedResult",
    "SavedImages",
    "SavedAudios",
    "ImageSaveHelper",
    "AudioSaveHelper",
    "PreviewImage",
    "PreviewMask",
    "PreviewAudio",
    "PreviewVideo",
    "PreviewUI3D",
    "PreviewUI3DAdvanced",
    "PreviewText",
]
