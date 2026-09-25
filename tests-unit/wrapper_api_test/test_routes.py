"""Tests for the wrapper route helpers (output file collection, setup errors)."""

import asyncio
import os
import time
import unittest

from api_wrapper import routes as wrapper_routes
from api_wrapper import workflows as wrapper_workflows


class TestCollectOutputFiles(unittest.TestCase):
    def _touch(self, *parts):
        path = os.path.join(*parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"test")
        return path

    def test_collects_images_videos_audio_3d(self):
        import folder_paths

        output_dir = folder_paths.get_output_directory()
        img = self._touch(output_dir, "wrapper", "a.png")
        vid = self._touch(output_dir, "wrapper", "b.mp4")
        aud = self._touch(output_dir, "wrapper", "c.wav")
        obj = self._touch(output_dir, "wrapper", "d.obj")
        entry = {
            "outputs": {
                "10": {
                    "images": [{"filename": "a.png", "subfolder": "wrapper", "type": "output"}],
                    "videos": [{"filename": "b.mp4", "subfolder": "wrapper", "type": "output"}],
                },
                "11": {
                    "audio": [{"filename": "c.wav", "subfolder": "wrapper", "type": "output"}],
                    "3d": [{"filename": "d.obj", "subfolder": "wrapper", "type": "output"}],
                },
            }
        }
        files = wrapper_routes._collect_output_files(entry)
        self.assertEqual([os.path.basename(f) for f in files], ["a.png", "b.mp4", "c.wav", "d.obj"])
        for path in (img, vid, aud, obj):
            os.remove(path)

    def test_skips_missing_and_foreign_keys(self):
        entry = {
            "outputs": {
                "10": {
                    "images": [{"filename": "missing.png", "subfolder": "", "type": "output"}],
                    "latents": [{"whatever": True}],
                }
            }
        }
        self.assertEqual(wrapper_routes._collect_output_files(entry), [])


class TestSetupErrors(unittest.TestCase):
    def test_setup_error_carries_missing_models(self):
        err = wrapper_routes._SetupError(
            "Required models could not be downloaded",
            "hint",
            missing=[{"folder": "diffusion_models", "filename": "x.safetensors", "error": "401"}],
        )
        self.assertEqual(err.message, "Required models could not be downloaded")
        self.assertEqual(err.missing[0]["filename"], "x.safetensors")
        self.assertEqual(err.details, "hint")

    def test_setup_error_defaults(self):
        err = wrapper_routes._SetupError("boom")
        self.assertEqual(err.missing, [])
        self.assertEqual(err.details, "")


class TestTimeoutParsing(unittest.TestCase):
    def test_missing_or_zero_means_no_limit(self):
        self.assertIsNone(wrapper_routes._parse_timeout(None))
        self.assertIsNone(wrapper_routes._parse_timeout(""))
        self.assertIsNone(wrapper_routes._parse_timeout("0"))

    def test_explicit_cap(self):
        self.assertEqual(wrapper_routes._parse_timeout("120"), 120)

    def test_out_of_range_rejected(self):
        for bad in ("1", "100000", "abc"):
            with self.assertRaises(ValueError):
                wrapper_routes._parse_timeout(bad)


class _FakePromptQueue:
    """Minimal stand-in: history dict + queue lists, mutable from the test."""

    def __init__(self):
        self.history = {}
        self.currently_running = {}
        self.queue = []

    def get_history(self, prompt_id=None):
        if prompt_id is None:
            return self.history
        return {prompt_id: self.history[prompt_id]} if prompt_id in self.history else {}


class TestWaitForPrompt(unittest.IsolatedAsyncioTestCase):
    async def test_returns_when_history_appears(self):
        queue = _FakePromptQueue()
        queue.currently_running = {0: (1, "job-1", {}, {}, {}, {})}

        async def finish_later():
            await asyncio.sleep(0.05)
            queue.history["job-1"] = {"status": {"status_str": "success", "completed": True, "messages": []}}

        await asyncio.gather(wrapper_routes._wait_for_prompt(queue, "job-1", sleep_interval=0.01), finish_later())
        self.assertEqual(queue.history["job-1"]["status"]["status_str"], "success")

    async def test_no_timeout_waits_past_any_deadline(self):
        queue = _FakePromptQueue()
        queue.currently_running = {0: (1, "job-1", {}, {}, {}, {})}

        async def finish_later():
            await asyncio.sleep(0.2)  # longer than a naive 0.1s cap
            queue.history["job-1"] = {"status": {"status_str": "success", "completed": True, "messages": []}}

        entry = await asyncio.gather(wrapper_routes._wait_for_prompt(queue, "job-1", sleep_interval=0.01), finish_later())
        self.assertIsNotNone(entry[0])

    async def test_timeout_returns_none(self):
        queue = _FakePromptQueue()
        queue.currently_running = {0: (1, "job-1", {}, {}, {}, {})}
        entry = await wrapper_routes._wait_for_prompt(queue, "job-1", timeout=0.05, sleep_interval=0.01)
        self.assertIsNone(entry)

    async def test_vanished_job_returns_none(self):
        queue = _FakePromptQueue()  # job never queued: not running, not queued
        entry = await wrapper_routes._wait_for_prompt(queue, "job-1", sleep_interval=0.01)
        self.assertIsNone(entry)

    async def test_running_job_without_timeout_keeps_waiting(self):
        queue = _FakePromptQueue()
        queue.currently_running = {0: (1, "job-1", {}, {}, {}, {})}
        start = time.time()
        entry = await wrapper_routes._wait_for_prompt(queue, "job-1", timeout=0.2, sleep_interval=0.01)
        # job still running after the cap -> None (timed out)
        self.assertIsNone(entry)
        self.assertGreaterEqual(time.time() - start, 0.2)


class TestMiniMaxSetupContract(unittest.TestCase):
    """The per-task setups must return exactly the kwargs their builder
    accepts (a mismatch crashed the text task with an unexpected
    'ref_image_size' argument), and own their own defaults (H3's canvas/length
    defaults, not the shared handler's)."""

    def setUp(self):
        self._download_models = wrapper_routes._download_models
        self._raise_if_missing = wrapper_routes._raise_if_missing
        self._get_torch_device = wrapper_routes.model_management.get_torch_device
        import types

        async def no_download(*args, **kwargs):
            return []

        wrapper_routes._download_models = no_download
        wrapper_routes._raise_if_missing = lambda missing: None
        wrapper_routes.model_management.get_torch_device = lambda: types.SimpleNamespace(type="cpu")

    def tearDown(self):
        wrapper_routes._download_models = self._download_models
        wrapper_routes._raise_if_missing = self._raise_if_missing
        wrapper_routes.model_management.get_torch_device = self._get_torch_device

    def test_text_setup_kwargs_build(self):
        kwargs, _ = asyncio.run(wrapper_routes._setup_minimax_h3_text({}, []))
        self.assertNotIn("ref_image_size", kwargs)
        self.assertEqual(kwargs["steps"], wrapper_workflows.MINIMAX_H3_DEFAULT_STEPS)
        self.assertEqual(kwargs["width"], wrapper_workflows.MINIMAX_H3_DEFAULT_WIDTH)
        self.assertEqual(kwargs["height"], wrapper_workflows.MINIMAX_H3_DEFAULT_HEIGHT)
        self.assertEqual(kwargs["unet_name"], wrapper_workflows.MINIMAX_H3_UNET_INT8)
        self.assertEqual(kwargs["clip_name"], wrapper_workflows.MINIMAX_H3_CLIP_NVFP4)
        graph = wrapper_workflows.build_minimax_h3_text_to_video(prompt="x", **kwargs)
        self.assertIn("1", graph)

    def test_image_setup_kwargs_build(self):
        kwargs, _ = asyncio.run(wrapper_routes._setup_minimax_h3_image({}, []))
        self.assertNotIn("ref_image_size", kwargs)
        kwargs["first_frame"] = "wrapper/a.png"
        graph = wrapper_workflows.build_minimax_h3_image_to_video(prompt="x", **kwargs)
        self.assertIn("1", graph)

    def test_reference_setup_kwargs_build(self):
        kwargs, _ = asyncio.run(wrapper_routes._setup_minimax_h3_reference({}, []))
        self.assertEqual(kwargs["ref_image_size"], "match")
        graph = wrapper_workflows.build_minimax_h3_reference_to_video(prompt="x", **kwargs)
        self.assertIn("1", graph)

    def test_steps_field_forwarded(self):
        kwargs, _ = asyncio.run(wrapper_routes._setup_minimax_h3_text({"steps": "20"}, []))
        self.assertEqual(kwargs["steps"], 20)


class TestPromptRewriteHook(unittest.TestCase):
    """The glue between an incoming request and the rewriter: which mode it
    reports, which labels it declares, and which upload it uses for visual
    context."""

    def setUp(self):
        from api_wrapper import prompt_rewriter

        self.calls = []
        self._rewrite = prompt_rewriter.rewrite
        self._data_url = prompt_rewriter.data_url_from_path

        async def fake_rewrite(**kwargs):
            self.calls.append(kwargs)
            return "REWRITTEN", "mimo-v2.5-pro"

        prompt_rewriter.rewrite = fake_rewrite
        prompt_rewriter.data_url_from_path = lambda path: f"data:image/png;base64,#{path}"

    def tearDown(self):
        from api_wrapper import prompt_rewriter

        prompt_rewriter.rewrite = self._rewrite
        prompt_rewriter.data_url_from_path = self._data_url

    def _run(self, task_name, build_kwargs, upload_refs, fields=None):
        base = {"prompt": "a wrestler chokeslams another", "duration": 5.0,
                "width": 864, "height": 480}
        base.update(build_kwargs)
        return asyncio.run(wrapper_routes._rewrite_prompt(
            task_name, fields or {}, base, upload_refs))

    def test_image_task_reports_i2va_and_grounds_on_the_first_frame(self):
        prompt, note, record = self._run("image", {"first_frame": "wrapper/a.png"},
                                         {"image": ["wrapper/a.png"]})
        self.assertEqual(record["prompt"], "REWRITTEN")
        self.assertEqual(record["source"], "mimo-v2.5-pro")
        self.assertEqual(record["mode"], "I2VA")
        self.assertEqual(record["input_prompt"], "a wrestler chokeslams another")
        self.assertEqual(prompt, "REWRITTEN")
        call = self.calls[0]
        self.assertEqual(call["mode"], "I2VA")
        self.assertEqual(call["frames"], 124)  # 5.0s snapped to the H3 grid
        self.assertAlmostEqual(call["duration"], 124 / 24, places=2)
        # no llm_image, so the keyframe itself is the visual context
        self.assertIn("wrapper/a.png", call["image"])
        self.assertIn("visual context", note)

    def test_llm_image_wins_over_the_keyframe(self):
        self._run("image", {"first_frame": "wrapper/a.png", "last_frame": "wrapper/b.png"},
                  {"image": ["wrapper/a.png"], "last_frame": ["wrapper/b.png"],
                   "llm_image": ["wrapper/ctx.png"]})
        call = self.calls[0]
        self.assertEqual(call["mode"], "FL2VA")  # both keyframes present
        self.assertIn("wrapper/ctx.png", call["image"])

    def test_text_task_without_uploads_sends_no_image(self):
        _, note, record = self._run("text", {}, {})
        self.assertIsNone(record["context_image"])
        call = self.calls[0]
        self.assertEqual(call["mode"], "T2VA")
        self.assertIsNone(call["image"])
        self.assertNotIn("visual context", note)

    def test_reference_counts_include_the_video_soundtrack_label(self):
        self._run("reference", {}, {
            "ref_images": ["wrapper/1.png", "wrapper/2.png"],
            "ref_videos": ["wrapper/v.mp4"],
            "ref_video_audios": ["wrapper/va.wav"],
            "ref_audios": ["wrapper/a1.wav", "wrapper/a2.wav"],
        })
        call = self.calls[0]
        self.assertEqual(call["mode"], "Ref2VA")
        # a reference video's soundtrack gets its own <Audio j>, so 1 + 2 = 3
        self.assertEqual(call["reference_counts"], {"images": 2, "videos": 1, "audios": 3})
        self.assertIn("wrapper/1.png", call["image"])

    def test_llm_model_field_is_forwarded(self):
        self._run("text", {}, {}, fields={"llm_model": "mimo-v2.5"})
        self.assertEqual(self.calls[0]["model"], "mimo-v2.5")


class TestPromptRecord(unittest.TestCase):
    """The H3 prompt is generated rather than supplied, so the only record of
    what a video was actually made from is the one the wrapper writes."""

    def setUp(self):
        import folder_paths

        self.video = os.path.join(folder_paths.get_output_directory(), "wrapper", "record_test.mp4")
        os.makedirs(os.path.dirname(self.video), exist_ok=True)
        with open(self.video, "wb") as f:
            f.write(b"MP4")
        self.sidecar = os.path.splitext(self.video)[0] + ".prompt.json"

    def tearDown(self):
        for path in (self.video, self.sidecar):
            if os.path.exists(path):
                os.remove(path)

    def test_sidecar_lands_next_to_the_video_and_round_trips(self):
        import json

        record = {"source": "mimo-v2.5", "mode": "I2VA", "job_id": "abc",
                  "input_prompt": "two wrestlers", "prompt": "integrated_multimodal_description: ...",
                  "settings": {"width": 864, "duration": 5.17}}
        path = wrapper_routes._save_prompt_record(self.video, record)
        self.assertEqual(path, self.sidecar)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), record)

    def test_non_ascii_prompt_survives(self):
        """H3 keeps dialogue in its original language, so the record must not
        be ascii-escaped into something unreadable."""
        import json

        record = {"prompt": '一个男人说: <d>[Chinese] 我们走吧。</d>', "source": "mimo-v2.5"}
        with open(wrapper_routes._save_prompt_record(self.video, record), encoding="utf-8") as f:
            self.assertIn("我们走吧", f.read())
        with open(self.sidecar, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["prompt"], record["prompt"])

    def test_unwritable_target_does_not_lose_the_video(self):
        missing = os.path.join(self.video, "no", "such", "dir", "x.mp4")
        self.assertIsNone(wrapper_routes._save_prompt_record(missing, {"prompt": "x"}))

    def test_logging_includes_the_generated_prompt(self):
        record = {"source": "mimo-v2.5", "mode": "Ref2VA",
                  "input_prompt": "a cat", "prompt": "THE GENERATED H3 PROMPT"}
        with self.assertLogs(level="INFO") as logs:
            wrapper_routes._log_prompt_record(record, "job-1", "minimaxh3", "reference")
        joined = "\n".join(logs.output)
        self.assertIn("THE GENERATED H3 PROMPT", joined)
        self.assertIn("mimo-v2.5", joined)
        self.assertIn("Ref2VA", joined)
        self.assertIn("job-1", joined)
        self.assertIn("a cat", joined)

    def test_verbatim_raw_prompt_is_not_logged_as_a_rewrite(self):
        record = {"source": "raw_prompt", "mode": "I2VA",
                  "input_prompt": None, "prompt": "MY OWN PROMPT"}
        with self.assertLogs(level="INFO") as logs:
            wrapper_routes._log_prompt_record(record, "job-2", "minimaxh3", "image")
        joined = "\n".join(logs.output)
        self.assertIn("MY OWN PROMPT", joined)
        self.assertNotIn("rewritten from", joined)


if __name__ == "__main__":
    unittest.main()


class TestJobProvenance(unittest.TestCase):
    """What a job ran with, for a caller that polls and never sees the headers."""

    def test_raw_prompt_jobs_report_seed_models_and_settings(self):
        provenance = wrapper_routes._job_provenance(
            {"seed": 42, "unet_name": "h3_ref2va_int8.safetensors", "clip_name": "enc.safetensors",
             "width": 864, "height": 480, "steps": 20, "prompt": "subject_definitions: ..."},
            "raw_prompt used verbatim; no LLM rewrite", {"source": "raw_prompt", "mode": "Ref2VA"})
        self.assertEqual(provenance["seed"], 42)
        self.assertEqual(provenance["models"], ["h3_ref2va_int8.safetensors", "enc.safetensors"])
        self.assertEqual(provenance["settings"], {"width": 864, "height": 480, "steps": 20})
        self.assertEqual((provenance["prompt_source"], provenance["mode"]), ("raw_prompt", "Ref2VA"))
        self.assertNotIn("prompt", provenance)  # the caller sent it; no need to echo it

    def test_a_rewritten_prompt_is_kept(self):
        provenance = wrapper_routes._job_provenance(
            {"seed": 1, "prompt": "integrated_multimodal_description: [Shot 1] ..."}, None,
            {"source": "mimo", "mode": "T2VA"})
        self.assertEqual(provenance["prompt"], "integrated_multimodal_description: [Shot 1] ...")

    def test_the_record_is_bounded_newest_first(self):
        saved = dict(wrapper_routes._PROVENANCE)
        try:
            wrapper_routes._PROVENANCE.clear()
            for i in range(wrapper_routes._PROVENANCE_MAX + 5):
                wrapper_routes._remember_provenance(f"job-{i}", {"seed": i})
            self.assertEqual(len(wrapper_routes._PROVENANCE), wrapper_routes._PROVENANCE_MAX)
            self.assertNotIn("job-0", wrapper_routes._PROVENANCE)
            self.assertIn(f"job-{wrapper_routes._PROVENANCE_MAX + 4}", wrapper_routes._PROVENANCE)
        finally:
            wrapper_routes._PROVENANCE.clear()
            wrapper_routes._PROVENANCE.update(saved)
