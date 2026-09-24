"""Tests for the CUDA docker setup: compose + Dockerfile + .dockerignore.

These are structural checks (docker itself is not required): the compose YAML
must parse, request an NVIDIA GPU, build the staged Dockerfile, persist models
as volumes, and the Dockerfile must order dependency installation before the
source copy so Docker's cache keeps the base stage until requirements change.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class TestDockerCompose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "docker-compose.yml")) as f:
            cls.compose = yaml.safe_load(f)

    def test_services_present(self):
        self.assertIn("comfyui", self.compose["services"])
        self.assertIn("developer-api", self.compose["services"])

    def test_gpu_reservation(self):
        devices = self.compose["services"]["comfyui"]["deploy"]["resources"]["reservations"]["devices"]
        self.assertEqual(devices[0]["driver"], "nvidia")
        self.assertIn("gpu", devices[0]["capabilities"][0])

    def test_build_target_and_entrypoint_env(self):
        # The server flags are built by docker/entrypoint.sh (the same script a
        # RunPod pod runs), so compose sets env and has no `command` of its own.
        cf = self.compose["services"]["comfyui"]
        self.assertEqual(cf["build"]["target"], "comfyui")
        self.assertNotIn("command", cf)
        self.assertTrue(cf["image"].startswith("${COMFYUI_IMAGE:-shivanshtalwar0/comfyui"))
        env = cf["environment"]
        self.assertTrue(any(e.startswith("HF_TOKEN") for e in env))
        for entry in ("VRAM_HEADROOM_GB=${VRAM_HEADROOM_GB:-2}",
                      "ASYNC_OFFLOAD_STREAMS=${ASYNC_OFFLOAD_STREAMS:-0}",
                      "AUTO_DOWNLOAD_MODELS=${AUTO_DOWNLOAD_MODELS:-1}",
                      "COMFYUI_ARGS=${COMFYUI_ARGS:-}",
                      "WRAPPER_AUTH_TOKEN=${WRAPPER_AUTH_TOKEN:-}"):
            self.assertIn(entry, env)
        self.assertEqual(cf["ports"], ["${COMFYUI_HOST_PORT:-8188}:8188"])

    def test_entrypoint_builds_the_server_flags(self):
        with open(os.path.join(ROOT, "docker", "entrypoint.sh")) as f:
            script = f.read()
        self.assertIn("--auto-download-models", script)
        self.assertIn('--vram-headroom "${VRAM_HEADROOM_GB:-2}"', script)
        self.assertIn("--disable-async-offload", script)  # async offload default-off
        self.assertIn("--fast-disk", script)
        self.assertIn("COMFYUI_ARGS", script)
        self.assertIn("/workspace", script)  # RunPod network volume
        self.assertIn('"prefetch"', script)
        # One copy per checkpoint on a per-GB billed volume (not HF cache + copy).
        self.assertIn('COMFY_HF_DOWNLOAD_MODE="${COMFY_HF_DOWNLOAD_MODE:-direct}"', script)
        self.assertTrue(os.access(os.path.join(ROOT, "docker", "entrypoint.sh"), os.X_OK),
                        "docker/entrypoint.sh must be executable")

    def test_model_and_state_volumes(self):
        volumes = self.compose["services"]["comfyui"]["volumes"]
        for host in ("./input", "./output", "./temp", "./user", "./api_server/workflows", "./.triton"):
            self.assertTrue(any(v.startswith(host) for v in volumes), f"volume {host} missing")
        self.assertTrue(any("/opt/ComfyUI/models" in v for v in volumes), "models mount missing")

    def test_models_dir_is_configurable(self):
        volumes = self.compose["services"]["comfyui"]["volumes"]
        models_mount = next(v for v in volumes if "/opt/ComfyUI/models" in v)
        self.assertTrue(models_mount.startswith("${MODELS_DIR:-./models}"),
                        "models mount must default to ./models and be overridable via MODELS_DIR")

    def test_healthcheck_and_ordering(self):
        cf = self.compose["services"]["comfyui"]
        self.assertIn("system_stats", cf["healthcheck"]["test"][-1])
        dep = self.compose["services"]["developer-api"]["depends_on"]["comfyui"]
        self.assertEqual(dep["condition"], "service_healthy")


class TestDockerfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "Dockerfile")) as f:
            cls.dockerfile = f.read()

    def test_stages_and_cache_order(self):
        self.assertIn("AS base", self.dockerfile)
        self.assertIn("AS comfyui", self.dockerfile)
        requirements_copy = self.dockerfile.index("COPY requirements.txt manager_requirements.txt ./")
        source_copy = self.dockerfile.index("COPY . .")
        self.assertLess(requirements_copy, source_copy,
                        "dependency install must come before the source copy to keep the base cached")

    def test_c_compiler_installed(self):
        # Triton JIT-compiles kernels at runtime (e.g. the flux2 text encoder's
        # RoPE path) and needs a C compiler inside the container.
        apt = self.dockerfile[self.dockerfile.index("apt-get install"):self.dockerfile.index("rm -rf /var/lib/apt/lists")]
        for tool in ("gcc", "g++", "make"):
            self.assertIn(tool, apt, f"{tool} must be installed for Triton JIT")

    def test_cuda_only_and_entrypoint(self):
        self.assertIn("ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128", self.dockerfile)
        self.assertIn("assert torch.version.cuda", self.dockerfile)
        torch_install = self.dockerfile.index('--index-url "${TORCH_INDEX_URL}"')
        requirements_install = self.dockerfile.index("pip install -r requirements.txt")
        self.assertLess(torch_install, requirements_install,
                        "CUDA torch must be installed before requirements.txt")
        self.assertIn('ENTRYPOINT ["tini", "--", "/opt/ComfyUI/docker/entrypoint.sh"]', self.dockerfile)
        self.assertIn("HEALTHCHECK", self.dockerfile)

    def test_copy_sources_exist(self):
        for path in ("requirements.txt", "manager_requirements.txt"):
            self.assertTrue(os.path.isfile(os.path.join(ROOT, path)),
                            f"{path} referenced by Dockerfile but missing")


class TestDockerignore(unittest.TestCase):
    def test_excludes_weights_and_state(self):
        with open(os.path.join(ROOT, ".dockerignore")) as f:
            ignore = f.read()
        for excluded in (".git", "models/*", "output/*", "*.safetensors"):
            self.assertIn(excluded, ignore)

    def test_excludes_local_environments_and_secrets(self):
        # Every top-level dot-entry: .venv (~1.5 GB), tool caches, .claude
        # worktrees and .env secrets must never reach the image.
        with open(os.path.join(ROOT, ".dockerignore")) as f:
            lines = [line.strip() for line in f]
        self.assertIn("/.*", lines)


if __name__ == "__main__":
    unittest.main()
