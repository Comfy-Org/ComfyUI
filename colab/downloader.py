# Downloads models with aria2. On Colab, aria2 and aria2p are installed by colab/install.py.
#
#   import sys; sys.path.append("/content/ComfyUI/colab")
#   from downloader import Downloader
#
#   with Downloader("/content/ComfyUI/models") as downloader:
#       downloader.diffusion_model("https://civitai.com/api/download/models/869391", rename="model.safetensors")
#       downloader.download("model_patches", "https://huggingface.co/.../resolve/main/model.safetensors")
#
# API keys default to the CIVITAI_API_KEY and HUG_TOKEN environment variables, then Colab secrets.

import os
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal, Optional, IO, List
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

import aria2p

try:
    from google.colab import userdata
except ImportError:  # Not on Colab, API keys only come from the environment.
    userdata = None


def print_line(*args) -> None:
    print("{:<17} {:<9} {:>8} {:>12} {:>12} {:>8}  {}".format(*args))


def rpc_running() -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("localhost", 6800)) == 0


def parse_url(url: str):
    parsed = urlparse(url)

    # Base URL (scheme + netloc + path, without query/fragment)
    base_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))

    # Parameters as dict[str, list[str]] (handles repeated keys)
    params_dict = parse_qs(parsed.query, keep_blank_values=True)

    return base_url, params_dict


# Civitai serves the same API from several domains, e.g. civitai.com and civitai.red.
def is_civitai(url: str) -> bool:
    return (urlparse(url).hostname or "").split(".")[-2:-1] == ["civitai"]


# Civitai's Cloudflare blocks aria2 on the redirect, so resolve the final link with curl first.
def resolve_url(url: str) -> str:
    return subprocess.check_output(["curl", "-LIs", "-o", os.devnull, "-w", "%{url_effective}", url], text=True)


@dataclass
class DownloadConfig:
    url: str
    filename: Optional[str] = None
    headers: Optional[List[str]] = None
    http_auth_challenge: bool = False


class Downloader:
    LOG_LEVEL = Literal["debug", "info", "notice", "warn", "error"]

    def __init__(
        self,
        models_base_dir: str,
        civitai_api_key: str = None,
        huggingface_api_key: str = None,
        delete_input_file_after_download: bool = True,
        console_log_level: LOG_LEVEL = "error",
        max_concurrent_downloads: int = 50,
        max_connections_per_server: int = 16,
        split: int = 16,
        min_split_size: str = "1M",
        diffusion_models_dir: str = "diffusion_models",
        loras_dir: str = "loras",
        vae_dir: str = "vae",
        clip_dir: str = "clip",
        clip_vision_dir: str = "clip_vision",
        text_encoders_dir: str = "text_encoders",
        upscale_models_dir: str = "upscale_models",
        debug: bool = False,
    ):
        self._input_file_path: Path = Path("tmp_downloader_inputs.txt").resolve()
        self._input_file: Optional[IO] = None
        self._delete_input_file_after_download: bool = delete_input_file_after_download

        # API Keys
        self.civitai_api_key: str = civitai_api_key or self._get_secret("CIVITAI_API_KEY")
        self.huggingface_api_key: str = huggingface_api_key or self._get_secret("HUG_TOKEN")

        # Aria2c Config
        self.console_log_level: Downloader.LOG_LEVEL = console_log_level
        self.max_concurrent_downloads: int = max_concurrent_downloads
        self.max_connections_per_server: int = max_connections_per_server
        self.split: int = split
        self.min_split_size: str = min_split_size

        # Model Dirs
        self.models_base_dir: str = models_base_dir
        self.diffusion_models_dir: str = diffusion_models_dir
        self.loras_dir: str = loras_dir
        self.vae_dir: str = vae_dir
        self.clip_dir: str = clip_dir
        self.clip_vision_dir: str = clip_vision_dir
        self.text_encoders_dir: str = text_encoders_dir
        self.upscale_models_dir: str = upscale_models_dir

        self.debug: bool = debug

        # Reuse the daemon started by an earlier Downloader (e.g. the model_manager node) instead of failing to bind 6800.
        # aria2c --daemon doesn't work on Windows, so run it as a background process instead.
        if not rpc_running():
            log_path = Path(models_base_dir).parent / "aria2c.log"
            aria2c = subprocess.Popen([
                "aria2c", "--enable-rpc", "--auto-file-renaming=false",
                "-x", str(self.max_connections_per_server), "-s", str(self.split), "-k", self.min_split_size,
                f"--max-concurrent-downloads={self.max_concurrent_downloads}", "--disable-ipv6=true",
                f"--log={log_path}", f"--log-level={self.console_log_level}",
                "--optimize-concurrent-downloads=true",
            ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            while not rpc_running():
                if aria2c.poll() is not None:
                    raise RuntimeError(f"aria2c exited with code {aria2c.returncode}, see {log_path}")
                time.sleep(0.1)

        self._aria2: aria2p.API = aria2p.API(
            aria2p.Client(
                host="http://localhost",
                port=6800,
                secret=""
            )
        )

    def __enter__(self):
        self._input_file = open(str(self._input_file_path), "w")
        if self.debug:
            print(f"Opened Input File: {self._input_file_path=}, {self._input_file=}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.debug:
            print(f"__exit__({exc_type=}, {exc_val=}, {exc_tb})")

        if exc_type is None:
            self._input_file_download()

        self._close_input_file()
        return False

    @property
    def _should_download_input_file(self) -> bool:
        return bool(self._input_file and self._input_file_path.exists() and self._input_file_path.is_file())

    def download(self, model_sub_dir: str, url: str, rename: str = None) -> List[aria2p.Download]:
        return self._download(
            url,
            directory=self._build_directory(model_sub_dir),
            rename=rename,
        )

    def diffusion_model(self, *urls, rename: str = None):
        self._download(
            *urls,
            directory=self._build_directory(self.diffusion_models_dir),
            rename=rename,
        )

    def lora(self, *urls, rename: str = None):
        self._download(
            *urls,
            directory=self._build_directory(self.loras_dir),
            rename=rename,
        )

    def vae(self, *urls, rename: str = None):
        self._download(
            *urls,
            directory=self._build_directory(self.vae_dir),
            rename=rename,
        )

    def clip(self, *urls, rename: str = None):
        self._download(
            *urls,
            directory=self._build_directory(self.clip_dir),
            rename=rename,
        )

    def clip_vision(self, *urls, rename: str = None):
        self._download(
            *urls,
            directory=self._build_directory(self.clip_vision_dir),
            rename=rename,
        )

    def text_encoder(self, *urls, rename: str = None):
        self._download(
            *urls,
            directory=self._build_directory(self.text_encoders_dir),
            rename=rename,
        )

    def upscale_model(self, *urls, rename: str = None):
        self._download(
            *urls,
            directory=self._build_directory(self.upscale_models_dir),
            rename=rename,
        )

    def _build_directory(self, model_dir: str) -> str:
        return f"{self.models_base_dir}/{model_dir}"

    @classmethod
    def _get_secret(cls, secret_name: str) -> str:
        if secret_name in os.environ or userdata is None:
            return os.environ.get(secret_name, "")
        try:
            return userdata.get(secret_name)
        except Exception:
            return ""

    def _input_file_download(self):
        if self._should_download_input_file:
            print(f"Starting Concurrent Downloads")
            if self.debug:
                print(
                    f"\nInput File: {self._input_file_path}\n"
                    f"{self.max_concurrent_downloads=}\n"
                )

            if self._input_file and not self._input_file.closed:
                self._input_file.flush()

            download_start_time = time.perf_counter()

            downloads = self._aria2.add(str(self._input_file_path))
            print_line("GID", "STATUS", "PROGRESS", "DOWN_SPEED", "UP_SPEED", "ETA", "NAME")
            for download in downloads:
                print_line(
                    download.gid,
                    download.status,
                    download.progress_string(),
                    download.download_speed_string(),
                    download.upload_speed_string(),
                    download.eta_string(),
                    download.name,
                )

            download_end_time = time.perf_counter()
            download_time = timedelta(seconds=(download_end_time - download_start_time))
            print(f"Total Download Time: {download_time}\n")

            self._close_input_file()
        else:
            print("Nothing to download")

    def _close_input_file(self):
        if self._input_file:
            if self.debug:
                print(f"Closing Input File")

            self._input_file.close()
            self._input_file = None

        if self._delete_input_file_after_download:
            if self._input_file_path.exists() and self._input_file_path.is_file():
                if self.debug:
                    print(f"Deleting File: {self._input_file_path}")

                self._input_file_path.unlink(missing_ok=True)

    def _download(
        self,
        *urls,
        directory: str,
        rename: str = None,
        max_connections_per_server: int = None,
        split: int = None,
        min_split_size: str = None,
    ):
        _max_connections_per_server = max_connections_per_server or self.max_connections_per_server
        _split = split or self.split
        _min_split_size = min_split_size or self.min_split_size

        download_configs = []
        for url in urls:
            url = url.strip()
            url = url.rstrip("/")
            if "huggingface.co" in url:
                download_configs.append(self._build_hf_download_config(url, rename))
            elif is_civitai(url):
                download_configs.append(self._build_civit_ai_download_config(url, rename))
            else:
                print(f"Unrecognized URL format, skipping: {url}")
                continue

        downloads = []
        download_config: DownloadConfig
        for download_config in download_configs:
            if self._input_file:
                # Add download to file to be downloaded concurrently
                out_config = f"  out={download_config.filename}\n" if download_config.filename else ""
                http_auth_challenge = "true" if download_config.http_auth_challenge else "false"

                headers = ""
                if download_config.headers:
                    for header in download_config.headers:
                        headers += f"  header={header}\n"

                file_config_entry: str = (
                    f"{download_config.url}\n"
                    f"  dir={directory}\n"
                    f"{out_config}"
                    f"  min-split-size={_min_split_size}\n"
                    f"  split={_split}\n"
                    f"  max-connection-per-server={_max_connections_per_server}\n"
                    f"  http-auth-challenge={http_auth_challenge}\n"
                    f"{headers}"
                )
                if self.debug:
                    print(f"Adding download to input file:\n{file_config_entry}")

                self._input_file.write(file_config_entry)
            else:
                aria2_options = {"dir": directory}
                if download_config.filename:
                    aria2_options["out"] = download_config.filename

                if download_config.http_auth_challenge:
                    aria2_options["http-auth-challenge"] = "true"

                if download_config.headers:
                    aria2_options["header"] = ",".join(download_config.headers)

                print(f"Starting Download >> {download_config.url}")
                download_start_time = time.perf_counter()
                download: aria2p.Download = self._aria2.add_uris([download_config.url], options=aria2_options)
                downloads.append(download)
                print_line("GID", "STATUS", "PROGRESS", "DOWN_SPEED", "UP_SPEED", "ETA", "NAME")
                print_line(
                    download.gid,
                    download.status,
                    download.progress_string(),
                    download.download_speed_string(),
                    download.upload_speed_string(),
                    download.eta_string(),
                    download.name,
                )

                download_end_time = time.perf_counter()

                download_time = timedelta(seconds=(download_end_time - download_start_time))
                print(f"Download Time: {download_time}\n")
        return downloads

    def _build_hf_download_config(self, url: str, rename: Optional[str]) -> DownloadConfig:
        if rename:
            filename = rename
        else:
            filename = url.split("/")[-1]
            filename = filename.removesuffix("?download=true")

        return DownloadConfig(
            url=url,
            filename=filename,
            headers=[f"Authorization: Bearer {self.huggingface_api_key}"],
            http_auth_challenge=True,
        )

    def _build_civit_ai_download_config(self, url: str, rename: Optional[str]) -> DownloadConfig:
        base_url, params = parse_url(url)
        params.setdefault("token", [self.civitai_api_key])

        # Keep the other query parameters, fileId picks a specific file of the model version.
        resolved_url = resolve_url(f"{base_url}?{urlencode(params, doseq=True)}")

        return DownloadConfig(
            url=resolved_url,
            filename=rename,
            http_auth_challenge=False,
        )
