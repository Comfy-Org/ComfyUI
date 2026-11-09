"""Hugging Face model-weight downloads for pack manifests' ``weights``.

Lifted unchanged from the secure overlay's old SDK (compat/comfy_api_latest/_sdk.py:
``_InProcessModels._download_huggingface_weights`` and its helpers, ``_InProcessAssets._confined_path``,
``_validate_onnx_weight_file``), its tested downloader: repo, file, revision and sha256 checks, a folder
allowlist, a size cap (``COMFY_SECURE_HF_WEIGHT_MAX``), structural verification (SafeTensors, GGUF,
ONNX, weights-only PyTorch), and a temp-file-then-replace install at
``<model folder>/huggingface/<repo>/<revision>/<file>``, the layout converted packs reference.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from typing import Any, Optional


def _validate_onnx_weight_file(path: str) -> None:
    """Admit only one self-contained graph made from standard ONNX domains."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "ONNX model validation requires the onnx package") from exc

    maximum = int(os.environ.get(
        "COMFY_SECURE_ONNX_WEIGHT_MAX", str(4 * 1024 * 1024 * 1024)))
    size = os.path.getsize(path)
    if maximum <= 0 or not 1 <= size <= maximum:
        raise ValueError("ONNX model file is outside the configured size limit")
    try:
        model = onnx.load(path, load_external_data=False)
    except Exception as exc:
        raise ValueError("download is not a valid ONNX model") from exc

    if model.functions or model.training_info:
        raise ValueError("ONNX model functions and training graphs are not allowed")
    allowed_domains = {"", "ai.onnx", "ai.onnx.ml"}
    # Exporters commonly leave unused provider-specific opset declarations in
    # otherwise standard graphs. They carry no executable behavior; enforce
    # the domain boundary on every actual node below.
    if any(
        not isinstance(item.domain, str) or len(item.domain) > 256
        or not 1 <= item.version <= 2**31 - 1
        for item in model.opset_import
    ):
        raise ValueError("ONNX model has an invalid operator-set declaration")

    node_count = 0
    tensor_count = 0
    graph_count = 0

    def check_tensor(tensor: Any) -> None:
        nonlocal tensor_count
        tensor_count += 1
        if tensor_count > 200_000:
            raise ValueError("ONNX model has too many tensors")
        if (tensor.data_location == onnx.TensorProto.EXTERNAL
                or len(tensor.external_data)):
            raise ValueError("external ONNX tensor data is not allowed")
        if len(tensor.dims) > 16 or any(
                dimension < 0 or dimension > 2**31 - 1
                for dimension in tensor.dims):
            raise ValueError("ONNX tensor dimensions are invalid")

    def check_graph(graph: Any) -> None:
        nonlocal graph_count, node_count
        graph_count += 1
        if graph_count > 1024:
            raise ValueError("ONNX model has too many nested graphs")
        for tensor in graph.initializer:
            check_tensor(tensor)
        for sparse in graph.sparse_initializer:
            check_tensor(sparse.values)
            check_tensor(sparse.indices)
        for node in graph.node:
            node_count += 1
            if node_count > 200_000:
                raise ValueError("ONNX model has too many operators")
            if node.domain not in allowed_domains:
                raise ValueError(
                    "ONNX model uses a non-standard operator domain")
            if not node.op_type or len(node.op_type) > 128:
                raise ValueError("ONNX model contains an invalid operator")
            for attribute in node.attribute:
                if attribute.HasField("t"):
                    check_tensor(attribute.t)
                for tensor in attribute.tensors:
                    check_tensor(tensor)
                if attribute.HasField("g"):
                    check_graph(attribute.g)
                for nested in attribute.graphs:
                    check_graph(nested)

    check_graph(model.graph)
    if node_count == 0 or tensor_count == 0:
        raise ValueError("ONNX model contains no executable weighted graph")
    try:
        onnx.checker.check_model(model, full_check=False)
    except Exception as exc:
        raise ValueError("ONNX model failed structural validation") from exc


class _InProcessAssets:
    @staticmethod
    def _confined_path(base: str, name: str, folder: str) -> str:
        base = os.path.realpath(os.path.abspath(base))
        logical = str(name or "").replace("\\", "/")
        if "\x00" in logical or logical.startswith("/"):
            raise ValueError(f"asset name escapes the {folder} directory")
        parts = [part for part in logical.split("/") if part not in ("", ".")]
        if any(part == ".." for part in parts):
            raise ValueError(f"asset name escapes the {folder} directory")
        full = os.path.realpath(os.path.join(base, *parts))
        try:
            confined = os.path.commonpath((base, full)) == base
        except ValueError:
            confined = False
        if not confined:
            raise ValueError(f"asset name escapes the {folder} directory")
        return full


class _InProcessModels:
    _HF_ENDPOINT = "https://huggingface.co"

    _HF_PYTORCH_WEIGHT_EXTENSIONS = frozenset({
        ".bin", ".ckpt", ".patch", ".pt", ".pth",
    })

    _HF_WEIGHT_EXTENSIONS = frozenset({
        ".safetensors", ".sft", ".gguf", ".onnx",
        *_HF_PYTORCH_WEIGHT_EXTENSIONS,
    })

    _HF_WEIGHT_FOLDERS = frozenset({
        "audio_encoders",
        "background_removal",
        "checkpoints",
        "clip_vision",
        "controlnet",
        "detection",
        "diffusion_models",
        "embeddings",
        "frame_interpolation",
        "geometry_estimation",
        "gligen",
        "hypernetworks",
        "ipadapter",
        "inpaint",
        "latent_upscale_models",
        "loras",
        "model_patches",
        "optical_flow",
        "onnx",
        "photomaker",
        "sams",
        "semantic_segmentation",
        "style_models",
        "text_encoders",
        "unet_gguf",
        "upscale_models",
        "vae",
        "vae_approx",
    })

    _HF_DOWNLOAD_LOCK = threading.Lock()

    _HF_VERIFIED_WEIGHTS: dict[
        str, tuple[int, int, int, int, int, Optional[str]]
    ] = {}

    @staticmethod
    def _hf_repo_id(repo_id: str) -> str:
        import re

        if not isinstance(repo_id, str):
            raise TypeError("Hugging Face repo_id must be a string")
        if len(repo_id) > 96 or repo_id.startswith(("http:", "https:")):
            raise ValueError("Hugging Face repo_id must name a model repository")
        parts = repo_id.split("/")
        component = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
        if (len(parts) not in (1, 2)
                or any(not component.fullmatch(part) for part in parts)
                or any(part.endswith(("-", ".")) for part in parts)
                or "--" in repo_id or ".." in repo_id
                or repo_id.endswith(".git")):
            raise ValueError("Hugging Face repo_id must name a model repository")
        return repo_id

    @classmethod
    def _hf_weight_filename(cls, filename: str) -> tuple[str, str]:
        import re

        if not isinstance(filename, str):
            raise TypeError("Hugging Face weight filename must be a string")
        logical = filename.replace("\\", "/")
        parts = logical.split("/")
        component = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
        if (not logical or len(logical) > 1024 or logical.startswith("/")
                or any(len(part) > 255 or not component.fullmatch(part)
                       for part in parts)):
            raise ValueError(
                "Hugging Face weight filename must be a confined repository path")
        extension = os.path.splitext(parts[-1])[1].lower()
        if extension not in cls._HF_WEIGHT_EXTENSIONS:
            raise ValueError(
                "Hugging Face downloads are limited to .safetensors, .sft, "
                ".gguf, validated .onnx, and restricted PyTorch "
                "tensor-archive weights")
        return logical, extension

    @staticmethod
    def _hf_revision(revision: str) -> str:
        import re

        if not isinstance(revision, str):
            raise TypeError("Hugging Face revision must be a string")
        if (not revision or len(revision) > 200
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", revision)
                or revision.endswith("/")
                or any(part in ("", ".", "..")
                       for part in revision.split("/"))):
            raise ValueError("Hugging Face revision is invalid")
        return revision

    @staticmethod
    def _hf_sha256(sha256: Optional[str]) -> Optional[str]:
        import re

        if sha256 is None:
            return None
        if not isinstance(sha256, str) or not re.fullmatch(
                r"[0-9a-fA-F]{64}", sha256):
            raise ValueError("Hugging Face weight sha256 must be 64 hex digits")
        return sha256.lower()

    @classmethod
    def _hf_weight_destination(
        cls, folder: str, repo_id: str, revision: str,
        filename: str, extension: str,
    ) -> tuple[str, str]:
        import folder_paths

        if not isinstance(folder, str) or folder not in cls._HF_WEIGHT_FOLDERS:
            raise ValueError(
                "Hugging Face weights must target a known model catalogue")
        registered = folder_paths.folder_names_and_paths.get(folder)
        if registered is None or not registered[0]:
            raise ValueError(f"model catalogue {folder!r} is not registered")
        extensions = {str(value).lower() for value in registered[1]}
        if extension not in extensions:
            raise ValueError(
                f"model catalogue {folder!r} does not accept {extension} weights")

        root = os.path.realpath(os.path.abspath(registered[0][0]))
        os.makedirs(root, exist_ok=True)
        logical = f"huggingface/{repo_id}/{revision}/{filename}"
        destination = _InProcessAssets._confined_path(root, logical, folder)
        return logical, destination

    @staticmethod
    def _verify_weight_file(path: str, extension: str) -> None:
        if extension in (".safetensors", ".sft"):
            from safetensors import safe_open

            try:
                with safe_open(path, framework="pt", device="cpu") as weights:
                    keys = list(weights.keys())
            except Exception as exc:
                raise ValueError("download is not a valid SafeTensors weight file") from exc
            if not keys:
                raise ValueError("SafeTensors file contains no weights")
            return

        if extension == ".onnx":
            _validate_onnx_weight_file(path)
            return

        if extension in _InProcessModels._HF_PYTORCH_WEIGHT_EXTENSIONS:
            import collections.abc
            import torch

            try:
                # Torch's legacy format cannot be memory mapped. Keep restricted
                # loading for both formats and mmap for the modern ZIP format.
                with open(path, "rb") as file:
                    use_mmap = file.read(4) == b"PK\x03\x04"
                value = torch.load(
                    path, map_location="cpu", mmap=use_mmap, weights_only=True)
            except Exception as exc:
                raise ValueError(
                    "download is not a restricted PyTorch weight archive"
                ) from exc

            tensors = 0
            entries = 0
            stack = [value]
            seen: set[int] = set()
            while stack:
                item = stack.pop()
                entries += 1
                if entries > 10_000_000:
                    raise ValueError("PyTorch weight archive is too complex")
                if isinstance(item, torch.Tensor):
                    tensors += 1
                    continue
                if item is None or isinstance(
                    item, (bool, int, float, str, bytes)
                ):
                    continue
                identity = id(item)
                if identity in seen:
                    continue
                seen.add(identity)
                if isinstance(item, collections.abc.Mapping):
                    stack.extend(item.keys())
                    stack.extend(item.values())
                    continue
                if isinstance(item, (list, tuple)):
                    stack.extend(item)
                    continue
                raise ValueError(
                    "PyTorch weight archive contains non-weight objects")
            if tensors == 0:
                raise ValueError("PyTorch weight archive contains no tensors")
            return

        import struct

        with open(path, "rb") as file:
            header = file.read(24)
        if len(header) != 24 or header[:4] != b"GGUF":
            raise ValueError("download is not a supported GGUF weight file")
        version = struct.unpack("<I", header[4:8])[0]
        tensor_count, metadata_count = struct.unpack("<QQ", header[8:24])
        if (version not in (2, 3) or not 1 <= tensor_count <= 10_000_000
                or metadata_count > 10_000_000):
            raise ValueError("download is not a supported GGUF weight file")

    @staticmethod
    def _file_sha256(path: str) -> str:
        import hashlib

        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _weight_identity(path: str) -> tuple[int, int, int, int, int]:
        status = os.stat(path)
        return (
            status.st_dev,
            status.st_ino,
            status.st_size,
            status.st_mtime_ns,
            status.st_ctime_ns,
        )

    @classmethod
    def _download_huggingface_weights(
        cls, repo_id: str, filename: str, folder: str,
        revision: str, sha256: Optional[str],
    ) -> str:
        import inspect
        import shutil
        import tempfile

        repo_id = cls._hf_repo_id(repo_id)
        filename, extension = cls._hf_weight_filename(filename)
        revision = cls._hf_revision(revision)
        sha256 = cls._hf_sha256(sha256)
        if extension == ".onnx" and sha256 is None:
            raise ValueError("Hugging Face ONNX weights require a sha256 pin")
        logical, destination = cls._hf_weight_destination(
            folder, repo_id, revision, filename, extension)

        request = {
            "repo_id": repo_id,
            "filename": filename,
            "repo_type": "model",
            "revision": revision,
            "endpoint": cls._HF_ENDPOINT,
            "token": False,
        }

        with cls._HF_DOWNLOAD_LOCK:
            if os.path.isfile(destination):
                try:
                    identity = cls._weight_identity(destination)
                except OSError:
                    identity = None
                if identity is not None:
                    cached = cls._HF_VERIFIED_WEIGHTS.get(destination)
                    if (cached is not None and cached[:5] == identity
                            and (sha256 is None or cached[5] == sha256)):
                        return logical
                    try:
                        cls._verify_weight_file(destination, extension)
                        digest_matches = (
                            sha256 is None
                            or cls._file_sha256(destination) == sha256)
                    except (OSError, ValueError):
                        digest_matches = False
                    if digest_matches:
                        cls._HF_VERIFIED_WEIGHTS[destination] = (
                            *identity, sha256)
                        return logical

            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError(
                    "Hugging Face weight downloads require "
                    "huggingface_hub") from exc
            if "dry_run" not in inspect.signature(
                    hf_hub_download).parameters:
                raise RuntimeError(
                    "huggingface_hub is too old for bounded weight downloads")

            max_bytes = int(os.environ.get(
                "COMFY_SECURE_HF_WEIGHT_MAX",
                str(64 * 1024 * 1024 * 1024)))
            if max_bytes <= 0:
                raise RuntimeError(
                    "COMFY_SECURE_HF_WEIGHT_MAX must be positive")
            info = hf_hub_download(**request, dry_run=True)
            size = getattr(info, "file_size", None)
            if type(size) is not int or size <= 0:
                raise RuntimeError("Hugging Face did not report a valid weight size")
            if size > max_bytes:
                raise ValueError(
                    f"Hugging Face weight is {size} bytes, over the "
                    f"{max_bytes} byte limit")
            source = os.path.realpath(str(hf_hub_download(**request)))
            if not os.path.isfile(source) or os.path.getsize(source) != size:
                raise RuntimeError("Hugging Face weight download is incomplete")

            parent = os.path.dirname(destination)
            os.makedirs(parent, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=".hf-weight-", suffix=extension, dir=parent)
            os.close(descriptor)
            try:
                shutil.copyfile(source, temporary)
                cls._verify_weight_file(temporary, extension)
                if sha256 is not None and cls._file_sha256(temporary) != sha256:
                    raise ValueError("Hugging Face weight sha256 does not match")
                os.chmod(temporary, 0o644)
                os.replace(temporary, destination)
                cls._HF_VERIFIED_WEIGHTS[destination] = (
                    *cls._weight_identity(destination), sha256)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        return logical


def download(weight: dict) -> str:
    """Install one declared weight from Hugging Face; its catalogue name in its model folder."""
    return _InProcessModels._download_huggingface_weights(
        weight["repo_id"], weight["filename"], weight["folder"], weight.get("revision") or "main", weight.get("sha256"))
