import logging
import os
import json

import numpy as np
import torch
from PIL import Image
from typing_extensions import override

import folder_paths
import node_helpers
from comfy_api.latest import ComfyExtension, io


def load_and_process_images(image_files, input_dir):
    """Utility function to load and process a list of images.

    Args:
        image_files: List of image filenames
        input_dir: Base directory containing the images
        resize_method: How to handle images of different sizes ("None", "Stretch", "Crop", "Pad")

    Returns:
        torch.Tensor: Batch of processed images
    """
    if not image_files:
        raise ValueError("No valid images found in input")

    output_images = []

    for file in image_files:
        image_path = os.path.join(input_dir, file)
        img = node_helpers.pillow(Image.open, image_path)

        if img.mode == "I":
            img = img.point(lambda i: i * (1 / 255))
        img = img.convert("RGB")
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        output_images.append(img_tensor)

    return output_images


class LoadImageDataSetFromFolderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadImageDataSetFromFolder",
            search_aliases=["load folder", "load from folder", "load dataset", "load images", "import dataset"],
            display_name="Load Image (from Folder)",
            category="image",
            description="Load a dataset of images from a specified folder and return a list of images. Supported formats: PNG, JPG, JPEG, WEBP.",
            is_experimental=True,
            inputs=[
                io.Combo.Input(
                    "folder",
                    options=folder_paths.get_input_subfolders(),
                    tooltip="The folder to load images from.",
                )
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    is_output_list=True,
                    tooltip="List of loaded images",
                )
            ],
        )

    @classmethod
    def execute(cls, folder):
        sub_input_dir = os.path.join(folder_paths.get_input_directory(), folder)
        valid_extensions = [".png", ".jpg", ".jpeg", ".webp"]
        image_files = [
            f
            for f in os.listdir(sub_input_dir)
            if any(f.lower().endswith(ext) for ext in valid_extensions)
        ]
        output_tensor = load_and_process_images(image_files, sub_input_dir)
        return io.NodeOutput(output_tensor)


class LoadImageTextDataSetFromFolderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadImageTextDataSetFromFolder",
            search_aliases=["load folder", "load from folder", "load dataset", "load images", "import dataset"],
            display_name="Load Image-Text (from Folder)",
            category="image",
            description="Load a dataset of pairs of images and text captions from a specified folder and return them as a list. Supported formats: PNG, JPG, JPEG, WEBP.",
            is_experimental=True,
            inputs=[
                io.Combo.Input(
                    "folder",
                    options=folder_paths.get_input_subfolders(),
                    tooltip="The folder to load images and text captions from.",
                )
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    is_output_list=True,
                    tooltip="List of loaded images",
                ),
                io.String.Output(
                    display_name="texts",
                    is_output_list=True,
                    tooltip="List of text captions",
                ),
            ],
        )

    @classmethod
    def execute(cls, folder):
        logging.info(f"Loading images from folder: {folder}")

        sub_input_dir = os.path.join(folder_paths.get_input_directory(), folder)
        valid_extensions = [".png", ".jpg", ".jpeg", ".webp"]

        image_files = []
        for item in os.listdir(sub_input_dir):
            path = os.path.join(sub_input_dir, item)
            if any(item.lower().endswith(ext) for ext in valid_extensions):
                image_files.append(path)
            elif os.path.isdir(path):
                # Support kohya-ss/sd-scripts folder structure
                repeat = 1
                if item.split("_")[0].isdigit():
                    repeat = int(item.split("_")[0])
                image_files.extend(
                    [
                        os.path.join(path, f)
                        for f in os.listdir(path)
                        if any(f.lower().endswith(ext) for ext in valid_extensions)
                    ]
                    * repeat
                )

        caption_file_path = [
            f.replace(os.path.splitext(f)[1], ".txt") for f in image_files
        ]
        captions = []
        for caption_file in caption_file_path:
            caption_path = os.path.join(sub_input_dir, caption_file)
            if os.path.exists(caption_path):
                with open(caption_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
                    captions.append(caption)
            else:
                captions.append("")

        output_tensor = load_and_process_images(image_files, sub_input_dir)

        logging.info(f"Loaded {len(output_tensor)} images from {sub_input_dir}.")
        return io.NodeOutput(output_tensor, captions)


def save_images_to_folder(image_list, output_dir, prefix="image", overwrite=True):
    """Utility function to save a list of image tensors to disk.

    Args:
        image_list: List of image tensors (each [1, H, W, C] or [H, W, C] or [C, H, W])
        output_dir: Directory to save images to
        prefix: Filename prefix

    Returns:
        List of saved filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    for idx, img_tensor in enumerate(image_list):
        # Handle different tensor shapes
        if isinstance(img_tensor, torch.Tensor):
            # Remove batch dimension if present [1, H, W, C] -> [H, W, C]
            if img_tensor.dim() == 4 and img_tensor.shape[0] == 1:
                img_tensor = img_tensor.squeeze(0)

            # If tensor is [C, H, W], permute to [H, W, C]
            if img_tensor.dim() == 3 and img_tensor.shape[0] in [1, 3, 4]:
                if (
                    img_tensor.shape[0] <= 4
                    and img_tensor.shape[1] > 4
                    and img_tensor.shape[2] > 4
                ):
                    img_tensor = img_tensor.permute(1, 2, 0)

            # Convert to numpy and scale to 0-255
            img_array = img_tensor.cpu().numpy()
            img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)

            # Convert to PIL Image
            img = Image.fromarray(img_array)
        else:
            raise ValueError(f"Expected torch.Tensor, got {type(img_tensor)}")

        # Save image
        if overwrite:
            filename = f"{prefix}_{idx:05d}.png"
        else:
            _, _, counter, _, resolved_prefix = folder_paths.get_save_image_path(prefix, output_dir)
            filename = f"{resolved_prefix}_{counter:05}_{idx:05d}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        saved_files.append(filename)

    return saved_files


class SaveImageDataSetToFolderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImageDataSetToFolder",
            search_aliases=["save folder", "save to folder", "save dataset", "save images", "export dataset"],
            display_name="Save Image (to Folder) (DEPRECATED)",
            category="image",
            description="Save a dataset of images to a specified folder. Supported formats: PNG.",
            is_experimental=True,
            is_output_node=True,
            is_input_list=True,  # Receive images as list
            inputs=[
                io.Image.Input("images", tooltip="List of images to save."),
                io.String.Input(
                    "folder_name",
                    default="dataset",
                    tooltip="Name of the folder to save images to (inside output directory).",
                ),
                io.String.Input(
                    "filename_prefix",
                    default="image",
                    tooltip="Prefix for saved image filenames.",
                    advanced=True,
                ),
                io.Combo.Input(
                    "mode",
                    default="overwrite",
                    options=["overwrite", "increment"],
                    tooltip="Whether to overwrite existing files or increment filenames to avoid overwriting."
                ),
            ],
            outputs=[],
            is_deprecated=True,  # This node is redundant and superseded by existing Save Image nodes where the target folder can be specified in the filename_prefix
        )

    @classmethod
    def execute(cls, images, folder_name, filename_prefix, mode):
        # Extract scalar values
        folder_name = folder_name[0]
        filename_prefix = filename_prefix[0]
        mode = mode[0]

        output_dir = os.path.join(folder_paths.get_output_directory(), folder_name)
        saved_files = save_images_to_folder(images, output_dir, filename_prefix, mode=='overwrite')

        logging.info(f"Saved {len(saved_files)} images to {output_dir}.")
        return io.NodeOutput()


class SaveImageTextDataSetToFolderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImageTextDataSetToFolder",
            search_aliases=["save folder", "save to folder", "save dataset", "save images", "save text", "export dataset"],
            display_name="Save Image-Text (to Folder)",
            category="image",
            description="Save a dataset of pairs of images and text captions to a specified folder. Images are saved as PNG files and captions are saved as TXT files with the same filename_prefix.",
            is_experimental=True,
            is_output_node=True,
            is_input_list=True,  # Receive both images and texts as lists
            inputs=[
                io.Image.Input("images", tooltip="List of images to save."),
                io.String.Input("texts",
                    optional=True,
                    force_input=True,
                    tooltip="List of text captions to save."
                ),
                io.String.Input(
                    "folder_name",
                    default="dataset",
                    tooltip="Name of the folder to save images to (inside output directory).",
                ),
                io.String.Input(
                    "filename_prefix",
                    default="image",
                    tooltip="Prefix for saved image filenames.",
                    advanced=True,
                ),
                io.Combo.Input(
                    "mode",
                    default="overwrite",
                    options=["overwrite", "increment"],
                    tooltip="Whether to overwrite existing files or increment filenames to avoid overwriting."
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, images, folder_name, filename_prefix, mode, texts=None):
        # Extract scalar values
        folder_name = folder_name[0]
        filename_prefix = filename_prefix[0]
        mode = mode[0]

        output_dir = os.path.join(folder_paths.get_output_directory(), folder_name)
        saved_files = save_images_to_folder(images, output_dir, filename_prefix, mode=='overwrite')

        # Save captions
        if texts:
            for idx, (filename, caption) in enumerate(zip(saved_files, texts)):
                caption_filename = filename.replace(".png", ".txt")
                caption_path = os.path.join(output_dir, caption_filename)
                with open(caption_path, "w", encoding="utf-8") as f:
                    f.write(caption)

        logging.info(f"Saved {len(saved_files)} images and captions to {output_dir}.")
        return io.NodeOutput()


# ========== Helper Functions for Transform Nodes ==========


def tensor_to_pil(img_tensor):
    """Convert tensor to PIL Image."""
    if img_tensor.dim() == 4 and img_tensor.shape[0] == 1:
        img_tensor = img_tensor.squeeze(0)
    img_array = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def pil_to_tensor(img):
    """Convert PIL Image to tensor."""
    img_array = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_array)[None,]


# ========== Base Classes for Transform Nodes ==========


class ImageProcessingNode(io.ComfyNode):
    """Base class for image processing nodes that operate on images.

    Child classes should set:
        node_id: Unique node identifier (required)
        search_aliases: List of search aliases (optional)
        display_name: Display name (optional, defaults to node_id)
        description: Node description (optional)
        extra_inputs: List of additional io.Input objects beyond "images" (optional)
        is_group_process: None (auto-detect), True (group), or False (individual) (optional)
        is_output_list: True (list output) or False (single output) (optional, default True)
        is_deprecated: True if the node is deprecated (optional, default False)

    Child classes must implement ONE of:
        _process(cls, image, **kwargs) -> tensor  (for single-item processing)
        _group_process(cls, images, **kwargs) -> list[tensor]  (for group processing)
    """

    node_id = None
    search_aliases = []
    display_name = None
    description = None
    extra_inputs = []
    is_group_process = None  # None = auto-detect, True/False = explicit
    is_output_list = None  # None = auto-detect based on processing mode
    is_deprecated = False
    @classmethod
    def _detect_processing_mode(cls):
        """Detect whether this node uses group or individual processing.

        Returns:
            bool: True if group processing, False if individual processing
        """
        # Explicit setting takes precedence
        if cls.is_group_process is not None:
            return cls.is_group_process

        # Check which method is overridden by looking at the defining class in MRO
        base_class = ImageProcessingNode

        # Find which class in MRO defines _process
        process_definer = None
        for klass in cls.__mro__:
            if "_process" in klass.__dict__:
                process_definer = klass
                break

        # Find which class in MRO defines _group_process
        group_definer = None
        for klass in cls.__mro__:
            if "_group_process" in klass.__dict__:
                group_definer = klass
                break

        # Check what was overridden (not defined in base class)
        has_process = process_definer is not None and process_definer is not base_class
        has_group = group_definer is not None and group_definer is not base_class

        if has_process and has_group:
            raise ValueError(
                f"{cls.__name__}: Cannot override both _process and _group_process. "
                "Override only one, or set is_group_process explicitly."
            )
        if not has_process and not has_group:
            raise ValueError(
                f"{cls.__name__}: Must override either _process or _group_process"
            )

        return has_group

    @classmethod
    def _ensure_image_list(cls, images):
        """Normalize to a flat list of [1, H, W, C] tensors."""
        if isinstance(images, torch.Tensor):
            if images.ndim != 4:
                raise ValueError(f"Expected 4D image tensor, got shape {tuple(images.shape)}")
            return [images[i:i+1] for i in range(images.shape[0])]

        flat = []
        for item in images:
            if not isinstance(item, torch.Tensor) or item.ndim != 4:
                raise ValueError(f"Expected 4D image tensor, got {type(item).__name__} shape {getattr(item, 'shape', None)}")
            flat.extend([item[i:i+1] for i in range(item.shape[0])])
        return flat

    @classmethod
    def define_schema(cls):
        if cls.node_id is None:
            raise NotImplementedError(f"{cls.__name__} must set node_id class variable")

        is_group = cls._detect_processing_mode()

        # Auto-detect is_output_list if not explicitly set
        # Single processing: False (backend collects results into list)
        # Group processing: True by default (can be False for single-output nodes)
        output_is_list = (
            cls.is_output_list if cls.is_output_list is not None else is_group
        )

        inputs = [
            io.Image.Input(
                "images",
                tooltip=(
                    "List of images to process." if is_group else "Image to process."
                ),
            )
        ]
        inputs.extend(cls.extra_inputs)

        return io.Schema(
            node_id=cls.node_id,
            search_aliases=cls.search_aliases,
            display_name=cls.display_name or cls.node_id,
            category=cls.category,
            description=cls.description,
            is_experimental=True,
            is_input_list=is_group,  # True for group, False for individual
            inputs=inputs,
            outputs=[
                io.Image.Output(
                    display_name="images",
                    is_output_list=output_is_list,
                    tooltip="Processed images",
                )
            ],
        )

    @classmethod
    def execute(cls, images, **kwargs):
        """Execute the node. Routes to _process or _group_process based on mode."""
        is_group = cls._detect_processing_mode()

        if is_group:
            images = cls._ensure_image_list(images)

        # Extract scalar values from lists for parameters
        params = {}
        for k, v in kwargs.items():
            if isinstance(v, list) and len(v) == 1:
                params[k] = v[0]
            else:
                params[k] = v

        if is_group:
            # Group processing: images is list, call _group_process
            result = cls._group_process(images, **params)
        else:
            # Individual processing: images is single item, call _process
            result = cls._process(images, **params)

        return io.NodeOutput(result)

    @classmethod
    def _process(cls, image, **kwargs):
        """Override this method for single-item processing.

        Args:
            image: tensor - Single image tensor
            **kwargs: Additional parameters (already extracted from lists)

        Returns:
            tensor - Processed image
        """
        raise NotImplementedError(f"{cls.__name__} must implement _process method")

    @classmethod
    def _group_process(cls, images, **kwargs):
        """Override this method for group processing.

        Args:
            images: list[tensor] - List of image tensors
            **kwargs: Additional parameters (already extracted from lists)

        Returns:
            list[tensor] - Processed images
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement _group_process method"
        )


class TextProcessingNode(io.ComfyNode):
    """Base class for text processing nodes that operate on texts.

    Child classes should set:
        node_id: Unique node identifier (required)
        search_aliases: List of search aliases (optional)
        display_name: Display name (optional, defaults to node_id)
        description: Node description (optional)
        extra_inputs: List of additional io.Input objects beyond "texts" (optional)
        is_group_process: None (auto-detect), True (group), or False (individual) (optional)
        is_output_list: True (list output) or False (single output) (optional, default True)
        is_deprecated: True if the node is deprecated (optional, default False)

    Child classes must implement ONE of:
        _process(cls, text, **kwargs) -> str  (for single-item processing)
        _group_process(cls, texts, **kwargs) -> list[str]  (for group processing)
    """

    node_id = None
    search_aliases = []
    display_name = None
    description = None
    extra_inputs = []
    is_group_process = None  # None = auto-detect, True/False = explicit
    is_output_list = None  # None = auto-detect based on processing mode
    is_deprecated = False
    @classmethod
    def _detect_processing_mode(cls):
        """Detect whether this node uses group or individual processing.

        Returns:
            bool: True if group processing, False if individual processing
        """
        # Explicit setting takes precedence
        if cls.is_group_process is not None:
            return cls.is_group_process

        # Check which method is overridden by looking at the defining class in MRO
        base_class = TextProcessingNode

        # Find which class in MRO defines _process
        process_definer = None
        for klass in cls.__mro__:
            if "_process" in klass.__dict__:
                process_definer = klass
                break

        # Find which class in MRO defines _group_process
        group_definer = None
        for klass in cls.__mro__:
            if "_group_process" in klass.__dict__:
                group_definer = klass
                break

        # Check what was overridden (not defined in base class)
        has_process = process_definer is not None and process_definer is not base_class
        has_group = group_definer is not None and group_definer is not base_class

        if has_process and has_group:
            raise ValueError(
                f"{cls.__name__}: Cannot override both _process and _group_process. "
                "Override only one, or set is_group_process explicitly."
            )
        if not has_process and not has_group:
            raise ValueError(
                f"{cls.__name__}: Must override either _process or _group_process"
            )

        return has_group

    @classmethod
    def define_schema(cls):
        if cls.node_id is None:
            raise NotImplementedError(f"{cls.__name__} must set node_id class variable")

        is_group = cls._detect_processing_mode()

        inputs = [
            io.String.Input(
                "texts",
                tooltip="List of texts to process." if is_group else "Text to process.",
            )
        ]
        inputs.extend(cls.extra_inputs)

        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name or cls.node_id,
            category="text",
            is_experimental=True,
            is_input_list=is_group,  # True for group, False for individual
            inputs=inputs,
            outputs=[
                io.String.Output(
                    display_name="texts",
                    is_output_list=cls.is_output_list,
                    tooltip="Processed texts",
                )
            ],
        )

    @classmethod
    def execute(cls, texts, **kwargs):
        """Execute the node. Routes to _process or _group_process based on mode."""
        is_group = cls._detect_processing_mode()

        # Extract scalar values from lists for parameters
        params = {}
        for k, v in kwargs.items():
            if isinstance(v, list) and len(v) == 1:
                params[k] = v[0]
            else:
                params[k] = v

        if is_group:
            # Group processing: texts is list, call _group_process
            result = cls._group_process(texts, **params)
        else:
            # Individual processing: texts is single item, call _process
            result = cls._process(texts, **params)

        # Wrap result based on is_output_list
        if cls.is_output_list:
            # Result should already be a list (or will be for individual)
            return io.NodeOutput(result if is_group else [result])
        else:
            # Single output - wrap in list for NodeOutput
            return io.NodeOutput([result])

    @classmethod
    def _process(cls, text, **kwargs):
        """Override this method for single-item processing.

        Args:
            text: str - Single text string
            **kwargs: Additional parameters (already extracted from lists)

        Returns:
            str - Processed text
        """
        raise NotImplementedError(f"{cls.__name__} must implement _process method")

    @classmethod
    def _group_process(cls, texts, **kwargs):
        """Override this method for group processing.

        Args:
            texts: list[str] - List of text strings
            **kwargs: Additional parameters (already extracted from lists)

        Returns:
            list[str] - Processed texts
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement _group_process method"
        )


# ========== Image Transform Nodes ==========


class ResizeImagesByShorterEdgeNode(ImageProcessingNode):
    node_id = "ResizeImagesByShorterEdge"
    display_name = "Resize Images by Shorter Edge (DEPRECATED)"
    category = "image/transform"
    description = "Resize images so that the shorter edge matches the specified dimension while preserving aspect ratio."
    is_deprecated = True  # This node is superseded by Resize Image/Mask with resize_type = scale shorter dimension
    extra_inputs = [
        io.Int.Input(
            "shorter_edge",
            default=512,
            min=1,
            max=8192,
            tooltip="Target dimension for the shorter edge.",
        ),
    ]

    @classmethod
    def _process(cls, image, shorter_edge):
        img = tensor_to_pil(image)
        w, h = img.size
        if w < h:
            new_w = shorter_edge
            new_h = int(h * (shorter_edge / w))
        else:
            new_h = shorter_edge
            new_w = int(w * (shorter_edge / h))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return pil_to_tensor(img)


class ResizeImagesByLongerEdgeNode(ImageProcessingNode):
    node_id = "ResizeImagesByLongerEdge"
    display_name = "Resize Images by Longer Edge (DEPRECATED)"
    category = "image/transform"
    description = "Resize images so that the longer edge matches the specified dimension while preserving aspect ratio."
    is_deprecated = True  # This node is superseded by Resize Image/Mask with resize_type = scale longer dimension
    extra_inputs = [
        io.Int.Input(
            "longer_edge",
            default=1024,
            min=1,
            max=8192,
            tooltip="Target dimension for the longer edge.",
        ),
    ]

    @classmethod
    def _process(cls, image, longer_edge):
        resized_images = []
        for image_i in image:
            img = tensor_to_pil(image_i)
            w, h = img.size
            if w > h:
                new_w = longer_edge
                new_h = int(h * (longer_edge / w))
            else:
                new_h = longer_edge
                new_w = int(w * (longer_edge / h))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            resized_images.append(pil_to_tensor(img))
        return torch.cat(resized_images, dim=0)


class CenterCropImagesNode(ImageProcessingNode):
    node_id = "CenterCropImages"
    search_aliases=["crop", "cut", "trim"]
    display_name="Crop Image (Center)"
    category="image/transform"
    description = "Center crop an image to the specified dimensions."
    extra_inputs = [
        io.Int.Input("width", default=512, min=1, max=8192, tooltip="Crop width."),
        io.Int.Input("height", default=512, min=1, max=8192, tooltip="Crop height."),
    ]

    @classmethod
    def _process(cls, image, width, height):
        img = tensor_to_pil(image)
        left = max(0, (img.width - width) // 2)
        top = max(0, (img.height - height) // 2)
        right = min(img.width, left + width)
        bottom = min(img.height, top + height)
        img = img.crop((left, top, right, bottom))
        return pil_to_tensor(img)


class RandomCropImagesNode(ImageProcessingNode):
    node_id = "RandomCropImages"
    search_aliases=["crop", "cut", "trim"]
    display_name = "Crop Image (Random)"
    category="image/transform"
    description = "Randomly crop an image to the specified dimensions."

    extra_inputs = [
        io.Int.Input("width", default=512, min=1, max=8192, tooltip="Crop width."),
        io.Int.Input("height", default=512, min=1, max=8192, tooltip="Crop height."),
        io.Int.Input(
            "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, tooltip="Random seed."
        ),
    ]

    @classmethod
    def _process(cls, image, width, height, seed):
        np.random.seed(seed % (2**32 - 1))
        img = tensor_to_pil(image)
        max_left = max(0, img.width - width)
        max_top = max(0, img.height - height)
        left = np.random.randint(0, max_left + 1) if max_left > 0 else 0
        top = np.random.randint(0, max_top + 1) if max_top > 0 else 0
        right = min(img.width, left + width)
        bottom = min(img.height, top + height)
        img = img.crop((left, top, right, bottom))
        return pil_to_tensor(img)


class NormalizeImagesNode(ImageProcessingNode):
    node_id = "NormalizeImages"
    search_aliases=["normalize", "normalize colors"]
    display_name = "Normalize Image Colors"
    category = "image/color"
    description = "Normalize images using mean and standard deviation."
    extra_inputs = [
        io.Float.Input(
            "mean",
            default=0.5,
            min=0.0,
            max=1.0,
            tooltip="Mean value for normalization.",
            advanced=True,
        ),
        io.Float.Input(
            "std",
            default=0.5,
            min=0.001,
            max=1.0,
            tooltip="Standard deviation for normalization.",
            advanced=True,
        ),
    ]

    @classmethod
    def _process(cls, image, mean, std):
        return (image - mean) / std


class AdjustBrightnessNode(ImageProcessingNode):
    node_id = "AdjustBrightness"
    search_aliases=["brightness"]
    display_name = "Adjust Brightness"
    category="image/adjustments"
    description = "Adjust the brightness of an image."
    extra_inputs = [
        io.Float.Input(
            "factor",
            default=1.0,
            min=0.0,
            max=2.0,
            tooltip="Brightness factor. 1.0 = no change, <1.0 = darker, >1.0 = brighter.",
        ),
    ]

    @classmethod
    def _process(cls, image, factor):
        return (image * factor).clamp(0.0, 1.0)


class AdjustContrastNode(ImageProcessingNode):
    node_id = "AdjustContrast"
    search_aliases=["contrast"]
    display_name = "Adjust Contrast"
    category="image/adjustments"
    description = "Adjust the contrast of an image."
    extra_inputs = [
        io.Float.Input(
            "factor",
            default=1.0,
            min=0.0,
            max=2.0,
            tooltip="Contrast factor. 1.0 = no change, <1.0 = less contrast, >1.0 = more contrast.",
        ),
    ]

    @classmethod
    def _process(cls, image, factor):
        return ((image - 0.5) * factor + 0.5).clamp(0.0, 1.0)


class ShuffleDatasetNode(ImageProcessingNode):
    node_id = "ShuffleDataset"
    search_aliases=["shuffle", "randomize", "mix"]
    display_name = "Shuffle Images List"
    category = "image/batch"
    description = "Randomly shuffle the order of images in a list."
    is_group_process = True  # Requires full list to shuffle
    extra_inputs = [
        io.Int.Input(
            "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, tooltip="Random seed."
        ),
    ]

    @classmethod
    def _group_process(cls, images, seed):
        np.random.seed(seed % (2**32 - 1))
        indices = np.random.permutation(len(images))
        return [images[i] for i in indices]


class ShuffleImageTextDatasetNode(io.ComfyNode):
    """Special node that shuffles both images and texts together."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ShuffleImageTextDataset",
            search_aliases=["shuffle", "randomize", "mix"],
            display_name = "Shuffle Pairs of Image-Text",
            category = "image/batch",
            description = "Randomly shuffle the order of pairs of image-text in a list.",
            is_experimental=True,
            is_input_list=True,
            inputs=[
                io.Image.Input("images", tooltip="List of images to shuffle."),
                io.String.Input("texts", tooltip="List of texts to shuffle.", force_input=True),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip="Random seed.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    is_output_list=True,
                    tooltip="Shuffled images",
                ),
                io.String.Output(
                    display_name="texts", is_output_list=True, tooltip="Shuffled texts"
                ),
            ],
        )

    @classmethod
    def execute(cls, images, texts, seed):
        seed = seed[0]  # Extract scalar
        np.random.seed(seed % (2**32 - 1))
        indices = np.random.permutation(len(images))
        shuffled_images = [images[i] for i in indices]
        shuffled_texts = [texts[i] for i in indices]
        return io.NodeOutput(shuffled_images, shuffled_texts)


# ========== Text Transform Nodes ==========


class TextToLowercaseNode(TextProcessingNode):
    node_id = "TextToLowercase"
    search_aliases=["lowercase"]
    display_name = "Convert Text to Lowercase (DEPRECATED)"
    category = "text"
    description = "Convert text to lowercase."
    is_deprecated = True  # This node is superseded by the Convert Text Case node

    @classmethod
    def _process(cls, text):
        return text.lower()


class TextToUppercaseNode(TextProcessingNode):
    node_id = "TextToUppercase"
    search_aliases=["uppercase"]
    display_name = "Convert Text to Uppercase (DEPRECATED)"
    category = "text"
    description = "Convert text to uppercase."
    is_deprecated = True  # This node is superseded by the Convert Text Case node

    @classmethod
    def _process(cls, text):
        return text.upper()


class TruncateTextNode(TextProcessingNode):
    node_id = "TruncateText"
    search_aliases=["truncate", "cut", "shorten"]
    display_name = "Truncate Text"
    category = "text"
    description = "Truncate text to a maximum length."
    extra_inputs = [
        io.Int.Input(
            "max_length", default=77, min=1, max=10000, tooltip="Maximum text length."
        ),
    ]

    @classmethod
    def _process(cls, text, max_length):
        return text[:max_length]


class AddTextPrefixNode(TextProcessingNode):
    node_id = "AddTextPrefix"
    display_name = "Add Text Prefix (DEPRECATED)"
    category = "text"
    description = "Add a prefix to all texts."
    is_deprecated = True  # This node is superseded by the Concatenate Text node
    extra_inputs = [
        io.String.Input("prefix", default="", tooltip="Prefix to add."),
    ]

    @classmethod
    def _process(cls, text, prefix):
        return prefix + text


class AddTextSuffixNode(TextProcessingNode):
    node_id = "AddTextSuffix"
    display_name = "Add Text Suffix (DEPRECATED)"
    category = "text"
    description = "Add a suffix to all texts."
    is_deprecated = True  # This node is superseded by the Concatenate Text node
    extra_inputs = [
        io.String.Input("suffix", default="", tooltip="Suffix to add."),
    ]

    @classmethod
    def _process(cls, text, suffix):
        return text + suffix


class ReplaceTextNode(TextProcessingNode):
    node_id = "ReplaceText"
    display_name = "Replace Text (DEPRECATED)"
    category = "text"
    description = "Replace text in all texts."
    is_deprecated = True  # This node is superseded by the other Replace Text node
    extra_inputs = [
        io.String.Input("find", default="", tooltip="Text to find."),
        io.String.Input("replace", default="", tooltip="Text to replace with."),
    ]

    @classmethod
    def _process(cls, text, find, replace):
        return text.replace(find, replace)


class StripWhitespaceNode(TextProcessingNode):
    node_id = "StripWhitespace"
    display_name = "Strip Whitespace (DEPRECATED)"
    category = "text"
    description = "Strip leading and trailing whitespace from all texts."
    is_deprecated = True  # This node is superseded by the Trim Text node

    @classmethod
    def _process(cls, text):
        return text.strip()


# ========== Group Processing Example Nodes ==========


class ImageDeduplicationNode(ImageProcessingNode):
    """Remove duplicate or very similar images from a list using perceptual hashing."""

    node_id = "ImageDeduplication"
    search_aliases=["deduplicate", "remove duplicates", "similarity filter"]
    display_name = "Deduplicate Images"
    category = "image/batch"
    description = "Remove duplicate or very similar images from a list."
    is_group_process = True  # Requires full list to compare images
    extra_inputs = [
        io.Float.Input(
            "similarity_threshold",
            default=0.95,
            min=0.0,
            max=1.0,
            tooltip="Similarity threshold (0-1). Higher means more similar. Images above this threshold are considered duplicates.",
            advanced=True,
        ),
    ]

    @classmethod
    def _group_process(cls, images, similarity_threshold):
        """Remove duplicate images using perceptual hashing."""
        if len(images) == 0:
            return []

        # Compute simple perceptual hash for each image
        def compute_hash(img_tensor):
            """Compute a simple perceptual hash by resizing to 8x8 and comparing to average."""
            img = tensor_to_pil(img_tensor)
            # Resize to 8x8
            img_small = img.resize((8, 8), Image.Resampling.LANCZOS).convert("L")
            # Get pixels
            pixels = list(img_small.getdata())
            # Compute average
            avg = sum(pixels) / len(pixels)
            # Create hash (1 if above average, 0 otherwise)
            hash_bits = "".join("1" if p > avg else "0" for p in pixels)
            return hash_bits

        def hamming_distance(hash1, hash2):
            """Compute Hamming distance between two hash strings."""
            return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

        # Compute hashes for all images
        hashes = [compute_hash(img) for img in images]

        # Find duplicates
        keep_indices = []
        for i in range(len(images)):
            is_duplicate = False
            for j in keep_indices:
                # Compare hashes
                distance = hamming_distance(hashes[i], hashes[j])
                similarity = 1.0 - (distance / 64.0)  # 64 bits total
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    logging.info(
                        f"Image {i} is similar to image {j} (similarity: {similarity:.3f}), skipping"
                    )
                    break

            if not is_duplicate:
                keep_indices.append(i)

        # Return only unique images
        unique_images = [images[i] for i in keep_indices]
        logging.info(
            f"Deduplication: kept {len(unique_images)} out of {len(images)} images"
        )
        return unique_images


class ImageGridNode(ImageProcessingNode):
    """Combine multiple images into a single grid/collage."""

    node_id = "ImageGrid"
    search_aliases=["grid", "collage", "combine"]
    display_name = "Make Image Grid"
    category="image/batch"
    description = "Arrange multiple images into a grid layout."
    is_group_process = True  # Requires full list to create grid
    is_output_list = False  # Outputs single grid image
    extra_inputs = [
        io.Int.Input(
            "columns",
            default=4,
            min=1,
            max=20,
            tooltip="Number of columns in the grid.",
        ),
        io.Int.Input(
            "cell_width",
            default=256,
            min=32,
            max=2048,
            tooltip="Width of each cell in the grid.",
            advanced=True,
        ),
        io.Int.Input(
            "cell_height",
            default=256,
            min=32,
            max=2048,
            tooltip="Height of each cell in the grid.",
            advanced=True,
        ),
        io.Int.Input(
            "padding", default=4, min=0, max=50, tooltip="Padding between images.", advanced=True
        ),
    ]

    @classmethod
    def _group_process(cls, images, columns, cell_width, cell_height, padding):
        """Arrange images into a grid."""
        if len(images) == 0:
            raise ValueError("Cannot create grid from empty image list")

        # Calculate grid dimensions
        num_images = len(images)
        rows = (num_images + columns - 1) // columns  # Ceiling division

        # Calculate total grid size
        grid_width = columns * cell_width + (columns - 1) * padding
        grid_height = rows * cell_height + (rows - 1) * padding

        # Create blank grid
        grid = Image.new("RGB", (grid_width, grid_height), (0, 0, 0))

        # Place images
        for idx, img_tensor in enumerate(images):
            row = idx // columns
            col = idx % columns

            # Convert to PIL and resize to cell size
            img = tensor_to_pil(img_tensor)
            img = img.resize((cell_width, cell_height), Image.Resampling.LANCZOS)

            # Calculate position
            x = col * (cell_width + padding)
            y = row * (cell_height + padding)

            # Paste into grid
            grid.paste(img, (x, y))

        logging.info(
            f"Created {columns}x{rows} grid with {num_images} images ({grid_width}x{grid_height})"
        )
        return pil_to_tensor(grid)


class MergeImageListsNode(ImageProcessingNode):
    """Merge multiple image lists into a single list."""

    node_id = "MergeImageLists"
    search_aliases=["list", "merge list", "make list"]
    display_name = "Merge Image Lists (DEPRECATED)"
    category = "image/batch"
    description = "Concatenate multiple image lists into one."
    is_group_process = True  # Receives images as list
    is_deprecated = True  # This node is superseded by the Create List node

    @classmethod
    def _group_process(cls, images):
        """Simply return the images list (already merged by input handling)."""
        # When multiple list inputs are connected, they're concatenated
        # For now, this is a simple pass-through
        logging.info(f"Merged image list contains {len(images)} images")
        return images


class MergeTextListsNode(TextProcessingNode):
    """Merge multiple text lists into a single list."""

    node_id = "MergeTextLists"
    display_name = "Merge Text Lists (DEPRECATED)"
    category = "text"
    description = "Concatenate multiple text lists into one."
    is_group_process = True  # Receives texts as list
    is_deprecated = True  # This node is superseded by the Create List node

    @classmethod
    def _group_process(cls, texts):
        """Simply return the texts list (already merged by input handling)."""
        # When multiple list inputs are connected, they're concatenated
        # For now, this is a simple pass-through
        logging.info(f"Merged text list contains {len(texts)} texts")
        return texts


# ========== Training Dataset Nodes ==========


class ResolutionBucket(io.ComfyNode):
    """Bucket latents and conditions by resolution for efficient batch training."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ResolutionBucket",
            search_aliases=["bucket by resolution", "group by resolution", "batch by resolution"],
            display_name="Resolution Bucket",
            category="model/training",
            description="Group latents and conditionings into buckets",
            is_experimental=True,
            is_input_list=True,
            inputs=[
                io.Latent.Input(
                    "latents",
                    tooltip="List of latent dicts to bucket by resolution.",
                ),
                io.Conditioning.Input(
                    "conditioning",
                    tooltip="List of conditioning lists (must match latents length).",
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latents",
                    is_output_list=True,
                    tooltip="List of batched latent dicts, one per resolution bucket.",
                ),
                io.Conditioning.Output(
                    display_name="conditioning",
                    is_output_list=True,
                    tooltip="List of condition lists, one per resolution bucket.",
                ),
            ],
        )

    @classmethod
    def execute(cls, latents, conditioning):
        # latents: list[{"samples": tensor}] where tensor is (B, C, H, W), typically B=1
        # conditioning: list[list[cond]]

        # Validate lengths match
        if len(latents) != len(conditioning):
            raise ValueError(
                f"Number of latents ({len(latents)}) does not match number of conditions ({len(conditioning)})."
            )

        # Flatten latents and conditions to individual samples
        flat_latents = []  # list of (C, H, W) tensors
        flat_conditions = []  # list of condition lists

        for latent_dict, cond in zip(latents, conditioning):
            samples = latent_dict["samples"]  # (B, C, H, W)
            batch_size = samples.shape[0]

            # cond is a list of conditions with length == batch_size
            for i in range(batch_size):
                flat_latents.append(samples[i])  # (C, H, W)
                flat_conditions.append(cond[i])  # single condition

        # Group by resolution (H, W)
        buckets = {}  # (H, W) -> {"latents": list, "conditions": list}

        for latent, cond in zip(flat_latents, flat_conditions):
            # latent shape is (..., H, W) (B, C, H, W) or (B, T, C, H ,W)
            h, w = latent.shape[-2], latent.shape[-1]
            key = (h, w)

            if key not in buckets:
                buckets[key] = {"latents": [], "conditions": []}

            buckets[key]["latents"].append(latent)
            buckets[key]["conditions"].append(cond)

        # Convert buckets to output format
        output_latents = []  # list[{"samples": tensor}] where tensor is (Bi, ..., H, W)
        output_conditions = []  # list[list[cond]] where each inner list has Bi conditions

        for (h, w), bucket_data in buckets.items():
            # Stack latents into batch: list of (..., H, W) -> (Bi, ..., H, W)
            stacked_latents = torch.stack(bucket_data["latents"], dim=0)
            output_latents.append({"samples": stacked_latents})

            # Conditions stay as list of condition lists
            output_conditions.append(bucket_data["conditions"])

            logging.info(
                f"Resolution bucket ({h}x{w}): {len(bucket_data['latents'])} samples"
            )

        logging.info(f"Created {len(buckets)} resolution buckets from {len(flat_latents)} samples")
        return io.NodeOutput(output_latents, output_conditions)


class MakeTrainingDataset(io.ComfyNode):
    """Encode images with VAE and texts with CLIP to create a training dataset."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MakeTrainingDataset",
            search_aliases=["encode dataset"],
            display_name="Make Training Dataset",
            category="model/training",
            description="Encode images with VAE and texts with CLIP to create a training dataset of latents and conditionings.",
            is_experimental=True,
            is_input_list=True,  # images and texts as lists
            inputs=[
                io.Image.Input("images", tooltip="List of images to encode."),
                io.Vae.Input(
                    "vae", tooltip="VAE model for encoding images to latents."
                ),
                io.Clip.Input(
                    "clip", tooltip="CLIP model for encoding text to conditioning."
                ),
                io.String.Input(
                    "texts",
                    optional=True,
                    tooltip="List of text captions. Can be length n (matching images), 1 (repeated for all), or omitted (uses empty string).",
                    force_input=True
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latents",
                    is_output_list=True,
                    tooltip="List of latent dicts",
                ),
                io.Conditioning.Output(
                    display_name="conditioning",
                    is_output_list=True,
                    tooltip="List of conditioning lists",
                ),
            ],
        )

    @classmethod
    def execute(cls, images, vae, clip, texts=None):
        # Extract scalars (vae and clip are single values wrapped in lists)
        vae = vae[0]
        clip = clip[0]

        # Handle text list
        num_images = len(images)

        if texts is None or len(texts) == 0:
            # Treat as [""] for unconditional training
            texts = [""]

        if len(texts) == 1 and num_images > 1:
            # Repeat single text for all images
            texts = texts * num_images
        elif len(texts) != num_images:
            raise ValueError(
                f"Number of texts ({len(texts)}) does not match number of images ({num_images}). "
                f"Text list should have length {num_images}, 1, or 0."
            )

        # Encode images with VAE
        logging.info(f"Encoding {num_images} images with VAE...")
        latents_list = []  # list[{"samples": tensor}]
        for img_tensor in images:
            # img_tensor is [1, H, W, 3]
            latent_tensor = vae.encode(img_tensor[:, :, :, :3])
            latents_list.append({"samples": latent_tensor})

        # Encode texts with CLIP
        logging.info(f"Encoding {len(texts)} texts with CLIP...")
        conditioning_list = []  # list[list[cond]]
        for text in texts:
            if text == "":
                cond = clip.encode_from_tokens_scheduled(clip.tokenize(""))
            else:
                tokens = clip.tokenize(text)
                cond = clip.encode_from_tokens_scheduled(tokens)
            conditioning_list.append(cond)

        logging.info(
            f"Created dataset with {len(latents_list)} latents and {len(conditioning_list)} conditioning."
        )
        return io.NodeOutput(latents_list, conditioning_list)


class SaveTrainingDataset(io.ComfyNode):
    """Save encoded training dataset (latents + conditioning) to disk."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveTrainingDataset",
            search_aliases=["export dataset", "save dataset"],
            display_name="Save Training Dataset",
            category="model/training",
            description="Save encoded training dataset (latents + conditioning) to disk for efficient loading during training.",
            is_experimental=True,
            is_output_node=True,
            is_input_list=True,  # Receive lists
            inputs=[
                io.Latent.Input(
                    "latents",
                    tooltip="List of latent dicts from MakeTrainingDataset.",
                ),
                io.Conditioning.Input(
                    "conditioning",
                    tooltip="List of conditioning lists from MakeTrainingDataset.",
                ),
                io.String.Input(
                    "folder_name",
                    default="training_dataset",
                    tooltip="Name of folder to save dataset (inside output directory).",
                ),
                io.Int.Input(
                    "shard_size",
                    default=1000,
                    min=1,
                    max=100000,
                    tooltip="Number of samples per shard file.",
                    advanced=True,
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, latents, conditioning, folder_name, shard_size):
        # Extract scalars
        folder_name = folder_name[0]
        shard_size = shard_size[0]

        # latents: list[{"samples": tensor}]
        # conditioning: list[list[cond]]

        # Validate lengths match
        if len(latents) != len(conditioning):
            raise ValueError(
                f"Number of latents ({len(latents)}) does not match number of conditions ({len(conditioning)}). "
                f"Something went wrong in dataset preparation."
            )

        # Create output directory
        output_dir = os.path.join(folder_paths.get_output_directory(), folder_name)
        os.makedirs(output_dir, exist_ok=True)

        # Prepare data pairs
        num_samples = len(latents)
        num_shards = (num_samples + shard_size - 1) // shard_size  # Ceiling division

        logging.info(
            f"Saving {num_samples} samples to {num_shards} shards in {output_dir}..."
        )

        # Save data in shards
        for shard_idx in range(num_shards):
            start_idx = shard_idx * shard_size
            end_idx = min(start_idx + shard_size, num_samples)

            # Get shard data (list of latent dicts and conditioning lists)
            shard_data = {
                "latents": latents[start_idx:end_idx],
                "conditioning": conditioning[start_idx:end_idx],
            }

            # Save shard
            shard_filename = f"shard_{shard_idx:04d}.pkl"
            shard_path = os.path.join(output_dir, shard_filename)

            with open(shard_path, "wb") as f:
                torch.save(shard_data, f)

            logging.info(
                f"Saved shard {shard_idx + 1}/{num_shards}: {shard_filename} ({end_idx - start_idx} samples)"
            )

        # Save metadata
        metadata = {
            "num_samples": num_samples,
            "num_shards": num_shards,
            "shard_size": shard_size,
        }
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logging.info(f"Successfully saved {num_samples} samples to {output_dir}.")
        return io.NodeOutput()


class LoadTrainingDataset(io.ComfyNode):
    """Load encoded training dataset from disk."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadTrainingDataset",
            search_aliases=["import dataset", "training data"],
            display_name="Load Training Dataset",
            category="model/training",
            description="Load encoded training dataset (latents + conditioning) from disk for use in training.",
            is_experimental=True,
            inputs=[
                io.String.Input(
                    "folder_name",
                    default="training_dataset",
                    tooltip="Name of folder containing the saved dataset (inside output directory).",
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latents",
                    is_output_list=True,
                    tooltip="List of latent dicts",
                ),
                io.Conditioning.Output(
                    display_name="conditioning",
                    is_output_list=True,
                    tooltip="List of conditioning lists",
                ),
            ],
        )

    @classmethod
    def execute(cls, folder_name):
        # Get dataset directory
        dataset_dir = os.path.join(folder_paths.get_output_directory(), folder_name)

        if not os.path.exists(dataset_dir):
            raise ValueError(f"Dataset directory not found: {dataset_dir}")

        # Find all shard files
        shard_files = sorted(
            [
                f
                for f in os.listdir(dataset_dir)
                if f.startswith("shard_") and f.endswith(".pkl")
            ]
        )

        if not shard_files:
            raise ValueError(f"No shard files found in {dataset_dir}")

        logging.info(f"Loading {len(shard_files)} shards from {dataset_dir}...")

        # Load all shards
        all_latents = []  # list[{"samples": tensor}]
        all_conditioning = []  # list[list[cond]]

        for shard_file in shard_files:
            shard_path = os.path.join(dataset_dir, shard_file)

            with open(shard_path, "rb") as f:
                shard_data = torch.load(f, weights_only=True)

            all_latents.extend(shard_data["latents"])
            all_conditioning.extend(shard_data["conditioning"])

            logging.info(f"Loaded {shard_file}: {len(shard_data['latents'])} samples")

        logging.info(
            f"Successfully loaded {len(all_latents)} samples from {dataset_dir}."
        )
        return io.NodeOutput(all_latents, all_conditioning)


# ========== Extension Setup ==========


class DatasetExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            # Data loading/saving nodes
            LoadImageDataSetFromFolderNode,
            LoadImageTextDataSetFromFolderNode,
            SaveImageDataSetToFolderNode,
            SaveImageTextDataSetToFolderNode,
            # Image transform nodes
            ResizeImagesByShorterEdgeNode,
            ResizeImagesByLongerEdgeNode,
            CenterCropImagesNode,
            RandomCropImagesNode,
            NormalizeImagesNode,
            AdjustBrightnessNode,
            AdjustContrastNode,
            ShuffleDatasetNode,
            ShuffleImageTextDatasetNode,
            # Text transform nodes
            TextToLowercaseNode,
            TextToUppercaseNode,
            TruncateTextNode,
            AddTextPrefixNode,
            AddTextSuffixNode,
            ReplaceTextNode,
            StripWhitespaceNode,
            # Group processing examples
            ImageDeduplicationNode,
            ImageGridNode,
            MergeImageListsNode,
            MergeTextListsNode,
            # Training dataset nodes
            MakeTrainingDataset,
            SaveTrainingDataset,
            LoadTrainingDataset,
            ResolutionBucket,
        ]


async def comfy_entrypoint() -> DatasetExtension:
    return DatasetExtension()
