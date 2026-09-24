"""Tests for the CUDA docker setup: compose + Dockerfile + .dockerignore + CI.

These are structural checks (docker itself is not required): the compose YAML
must parse, request an NVIDIA GPU, build the staged Dockerfile, persist models
as volumes, and the Dockerfile must order dependency installation before the
source copy so Docker's cache keeps the base stage until requirements change.

The `runpod` target is the slim RunPod variant: no torch or requirements in
the image; the entrypoint builds them once into a virtualenv on the network
volume under a lock and every later pod reuses it.
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def read(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return f.read()


def dockerfile_stages(text):
    """Stage name -> (FROM image, instruction text without comments), in file order."""
    stages = {}
    current = None
    for line in text.splitlines():
        match = re.match(r"FROM\s+(\S+)(?:\s+AS\s+(\S+))?\s*$", line, re.IGNORECASE)
        if match:
            current = match.group(2) or match.group(1)
            stages[current] = [match.group(1), []]
        elif current and not line.lstrip().startswith("#"):
            stages[current][1].append(line)
    return {name: (image, "\n".join(lines)) for name, (image, lines) in stages.items()}


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

    def test_full_target_stays_the_default(self):
        # `docker build .` (no --target) builds the last stage: keep that the
        # full image the local rig and compose have always used.
        stages = dockerfile_stages(self.dockerfile)
        self.assertEqual(list(stages)[-1], "comfyui")
        self.assertEqual(stages["comfyui"][0], "base")
        self.assertEqual(stages["base"][0], "system")
        self.assertIn("COPY . .", stages["comfyui"][1])


class TestRunpodTarget(unittest.TestCase):
    """The slim RunPod image: system packages + uv + sources, no Python deps."""

    @classmethod
    def setUpClass(cls):
        cls.dockerfile = read("Dockerfile")
        cls.stages = dockerfile_stages(cls.dockerfile)
        cls.runpod = cls.stages["runpod"][1]

    def test_builds_from_the_shared_system_stage(self):
        # Not from `base`: that would drag torch and requirements along.
        self.assertEqual(self.stages["runpod"][0], "system")
        system = self.stages["system"][1]
        for tool in ("ffmpeg", "gcc", "g++", "make", "git", "libglib2.0-0", "libgl1", "tini"):
            self.assertIn(tool, system, f"{tool} must be in the shared apt layer")

    def test_no_torch_or_requirements_install(self):
        for forbidden in ("pip install", "--index-url", "COPY --from=base", "site-packages"):
            self.assertNotIn(forbidden, self.runpod, f"runpod target must not contain {forbidden!r}")

    def test_uv_binary_from_a_pinned_image(self):
        self.assertIn("COPY --from=uv /uv /usr/local/bin/uv", self.runpod)
        self.assertEqual(self.stages["uv"][0], "${UV_IMAGE}")
        match = re.search(r"^ARG UV_IMAGE=ghcr\.io/astral-sh/uv:(\S+)$", self.dockerfile, re.MULTILINE)
        self.assertIsNotNone(match, "UV_IMAGE must default to the official uv image")
        self.assertRegex(match.group(1), r"^\d+\.\d+\.\d+$", "pin a uv version, not a floating tag")

    def test_env_key_is_baked(self):
        self.assertIn("ARG TORCH_INDEX_URL", self.runpod)
        # Everything the env's contents depend on goes into the hashed inputs.
        for part in ("recipe=${ENV_RECIPE}", "python=", "sys.version_info[:2]", "platform=$(uname -m)",
                     "torch_index=${TORCH_INDEX_URL}", "sha256sum requirements.txt"):
            self.assertIn(part, self.runpod)
        self.assertIn("> docker/env-inputs", self.runpod)
        self.assertIn("sha256sum docker/env-inputs | cut -c1-16 > docker/env-key", self.runpod)
        self.assertIn("COMFY_IMAGE_VARIANT=runpod", self.runpod)

    def test_entrypoint_and_healthcheck(self):
        self.assertIn('ENTRYPOINT ["tini", "--", "/opt/ComfyUI/docker/entrypoint.sh"]', self.runpod)
        self.assertIn("HEALTHCHECK", self.runpod)
        self.assertIn("chmod +x docker/entrypoint.sh", self.runpod)

    def test_cpu_torch_escape_hatch_is_never_baked(self):
        # COMFY_ALLOW_CPU_TORCH exists for local tests only; the published
        # images must stay CUDA only, so nothing may turn it on by default.
        self.assertNotIn("COMFY_ALLOW_CPU_TORCH", self.dockerfile)
        self.assertNotIn("COMFY_ALLOW_CPU_TORCH", read(".github", "workflows", "docker-publish.yml"))


class TestRunpodEntrypoint(unittest.TestCase):
    """docker/entrypoint.sh builds and reuses the Python env on the volume."""

    @classmethod
    def setUpClass(cls):
        cls.script = read("docker", "entrypoint.sh")

    def test_variant_and_env_location(self):
        self.assertIn('"${COMFY_IMAGE_VARIANT:-}" == "runpod"', self.script)
        self.assertIn('ENV_KEY=$(cat "$COMFY_ROOT/docker/env-key")', self.script)
        self.assertIn('ENVS_DIR="${COMFY_ENVS_DIR:-$DATA_DIR/envs}"', self.script)
        self.assertIn('ENV_DIR="$ENVS_DIR/$ENV_KEY"', self.script)
        # The torch index comes from the baked inputs, so it always matches the key.
        self.assertIn("s/^torch_index=//p", self.script)

    def test_fails_fast_without_a_volume(self):
        block = self.script[self.script.index('if [[ "${COMFY_IMAGE_VARIANT:-}" == "runpod" ]]; then'):]
        no_volume = block[:block.index("fi")]
        self.assertIn('-z "$DATA_DIR"', no_volume)
        self.assertIn("no volume is mounted", no_volume)
        self.assertIn("exit 64", no_volume)

    def test_network_filesystem_lock(self):
        # mkdir is atomic on a network filesystem where flock may not be.
        self.assertIn('ENV_LOCK="$ENVS_DIR/.lock-$ENV_KEY"', self.script)
        self.assertIn('mkdir "$ENV_LOCK" 2>/dev/null', self.script)
        code = "\n".join(line for line in self.script.splitlines() if not line.lstrip().startswith("#"))
        self.assertNotIn("flock", code)
        self.assertIn('>"$ENV_LOCK/owner"', self.script)
        self.assertIn('>"$ENV_LOCK/heartbeat"', self.script)
        self.assertIn('ENV_LOCK_STALE_SECONDS="${COMFY_ENV_LOCK_STALE_SECONDS:-120}"', self.script)
        self.assertIn('ENV_WAIT_SECONDS="${COMFY_ENV_WAIT_SECONDS:-1800}"', self.script)
        # Taking over a stale lock is itself exclusive, and re-checks the state.
        self.assertIn('mkdir "$ENV_LOCK.break"', self.script)
        self.assertIn('"$(lock_state)" == "$stale"', self.script)
        # The lock is released on failure too.
        self.assertIn("trap release_env_lock EXIT", self.script)

    def test_env_build_recipe(self):
        build = self.script[self.script.index("build_env() {"):self.script.index("ensure_env() {")]
        self.assertIn('uv venv --python "$base_python" "$ENV_DIR"', build)
        torch_install = build.index('uv pip install --python "$ENV_DIR/bin/python" --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio')
        requirements_install = build.index('uv pip install --python "$ENV_DIR/bin/python" -r "$COMFY_ROOT/requirements.txt"')
        cuda_assert = build.index("cuda = torch.version.cuda")
        marker = build.index('mv -f "$ENV_DIR/$ENV_MARKER_NAME.tmp" "$ENV_DIR/$ENV_MARKER_NAME"')
        self.assertLess(torch_install, requirements_install, "CUDA torch must be installed before requirements.txt")
        self.assertLess(requirements_install, cuda_assert)
        self.assertLess(cuda_assert, marker, "the completion marker is written last")
        # The CPU escape hatch is explicit and off by default.
        self.assertIn('os.environ.get("COMFY_ALLOW_CPU_TORCH") != "1"', build)
        # A partial env (no marker) is removed before rebuilding.
        self.assertIn('rm -rf "$ENV_DIR"', build)
        # Only a builder that still owns the lock may mark the env complete.
        self.assertLess(build.index('"$(lock_owner)" != "$ENV_TOKEN"'), marker)
        # Nothing cached twice on the per-GB billed volume.
        self.assertIn("default_cache=/tmp/uv-cache", build)
        self.assertIn("UV_LINK_MODE=copy", build)
        self.assertIn("UV_PYTHON_DOWNLOADS=never", build)

    def test_completion_marker(self):
        self.assertIn("ENV_MARKER_NAME=.comfy-env-complete", self.script)
        self.assertIn('env_ready() { [[ -f "$ENV_DIR/$ENV_MARKER_NAME" && -x "$ENV_DIR/bin/python" ]]; }', self.script)

    def test_everything_runs_inside_the_env(self):
        activate = self.script.index('export PATH="$ENV_DIR/bin:$PATH"')
        self.assertLess(self.script.index("\tensure_env\n"), activate)
        self.assertLess(activate, self.script.index('if [[ "${1:-}" == "prefetch" ]]; then'))
        self.assertLess(activate, self.script.index('exec "$@"'))
        self.assertLess(activate, self.script.index("exec python main.py"))

    def test_prefetch_env_only(self):
        self.assertIn('"$arg" == "--env-only"', self.script)
        self.assertIn('exec python "$COMFY_ROOT/docker/prefetch_models.py" "${prefetch_args[@]}"', self.script)

    def test_logs_found_vs_built_timings(self):
        self.assertIn("reusing it (checked in $((SECONDS - t0))s)", self.script)
        self.assertIn("built in $((SECONDS - t0))s", self.script)


class TestPublishWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = yaml.safe_load(read(".github", "workflows", "docker-publish.yml"))
        cls.jobs = cls.workflow["jobs"]

    @staticmethod
    def steps_using(job, action):
        return [s for s in job["steps"] if s.get("uses", "").startswith(action)]

    def test_both_targets_built_and_pushed(self):
        for job, target in (("build", "comfyui"), ("runpod", "runpod")):
            builds = self.steps_using(self.jobs[job], "docker/build-push-action")
            self.assertEqual({b["with"]["target"] for b in builds}, {target})
            self.assertTrue(any(b["with"].get("push") is True for b in builds), f"{job} never pushes")
            self.assertTrue(any(b["with"].get("load") is True for b in builds), f"{job} is not smoke-tested")

    def test_full_tags_unchanged(self):
        tags = self.steps_using(self.jobs["build"], "docker/metadata-action")[0]["with"]["tags"]
        self.assertIn("type=raw,value=latest,enable={{is_default_branch}}", tags)
        self.assertIn("type=sha,prefix=sha-,format=short", tags)

    def test_runpod_tags(self):
        meta = self.steps_using(self.jobs["runpod"], "docker/metadata-action")[0]["with"]
        self.assertIn("type=raw,value=runpod,enable={{is_default_branch}}", meta["tags"])
        self.assertIn("type=sha,prefix=runpod-sha-,format=short", meta["tags"])
        self.assertIn("type=semver,pattern={{version}},prefix=runpod-", meta["tags"])
        # :latest is the full image; the runpod job must never move it.
        self.assertIn("latest=false", meta["flavor"])
        self.assertEqual(meta["images"], "docker.io/${{ env.IMAGE }}")

    def test_separate_build_caches(self):
        def cache_refs(job):
            refs = set()
            for build in self.steps_using(self.jobs[job], "docker/build-push-action"):
                for key in ("cache-from", "cache-to"):
                    refs.update(re.findall(r":(buildcache[\w-]*)", build["with"].get(key, "")))
            return refs
        self.assertEqual(cache_refs("build"), {"buildcache"})
        self.assertEqual(cache_refs("runpod"), {"buildcache-runpod"})

    def test_runpod_smoke_test_builds_then_reuses_the_env(self):
        runs = "\n".join(s.get("run", "") for s in self.jobs["runpod"]["steps"])
        self.assertIn("-v runpod-smoke-ws:/workspace", runs)
        self.assertIn("COMFYUI_ARGS=--cpu", runs)
        self.assertIn("WRAPPER_AUTH_TOKEN=smoke-token", runs)
        self.assertIn('"401"', runs)
        self.assertIn("building it", runs)
        self.assertIn("reusing it", runs)
        self.assertIn("boot second", runs)
        self.assertIn("prefetch --dry-run", runs)

    def test_full_smoke_tests_kept(self):
        runs = "\n".join(s.get("run", "") for s in self.jobs["build"]["steps"])
        self.assertIn("assert torch.version.cuda", runs)
        self.assertIn("prefetch --dry-run", runs)
        self.assertIn('"401"', runs)


class TestReadme(unittest.TestCase):
    def test_documents_the_runpod_variant(self):
        readme = read("README.md")
        for text in ("docker.io/shivanshtalwar0/comfyui:runpod", "prefetch --env-only", "/workspace/envs/<key>"):
            self.assertTrue(text in readme, f"README must document {text!r}")


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
