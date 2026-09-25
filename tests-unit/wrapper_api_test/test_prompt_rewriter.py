"""Unit tests for the MiMo-backed H3 prompt rewriter.

The model endpoint is OpenAI chat-completions compatible, so these run a real
aiohttp server on localhost, point MIMO_BASE_URL at it, and assert on the
request the rewriter actually sends -- that is the part that has to be right
and cannot be checked against the live API without a key.
"""

import asyncio
import base64
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from aiohttp import web

from api_wrapper import prompt_rewriter as pr

PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


class _MockAPI:
    """Stands in for the MiMo endpoint; records the last request it saw."""

    def __init__(self, reply="rewritten prompt", status=200, body=None):
        self.reply, self.status, self.body = reply, status, body
        self.request = None
        self.headers = None

    async def handler(self, request):
        self.request = await request.json()
        self.headers = dict(request.headers)
        if self.body is not None:
            return web.Response(text=self.body, status=self.status)
        return web.json_response(
            {"choices": [{"message": {"content": self.reply}}]}, status=self.status)


class RewriterTestCase(unittest.TestCase):
    """Boots the mock API and points the rewriter at it for each test."""

    def run_rewrite(self, api, **kwargs):
        async def go():
            app = web.Application()
            app.router.add_post("/chat/completions", api.handler)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = runner.addresses[0][1]
            previous = dict(os.environ)
            os.environ["MIMO_BASE_URL"] = f"http://127.0.0.1:{port}"
            os.environ["MIMO_API_KEY"] = "test-key"
            try:
                params = dict(intent="two wrestlers, one chokeslams the other",
                              mode=pr.MODE_I2VA, duration=5.17, frames=124,
                              width=864, height=480)
                params.update(kwargs)
                return await pr.rewrite(**params)
            finally:
                os.environ.clear()
                os.environ.update(previous)
                await runner.cleanup()

        return asyncio.run(go())


class TestRequestShape(RewriterTestCase):
    def test_sends_guide_job_parameters_and_bearer_auth(self):
        api = _MockAPI()
        prompt, model = self.run_rewrite(api)
        self.assertEqual(prompt, "rewritten prompt")
        self.assertEqual(model, pr.MODEL_FLASH)  # auto -> flash, image or not

        # both auth headers, so MIMO_BASE_URL can point at Xiaomi (api-key) or
        # any OpenAI-compatible gateway (Bearer)
        self.assertEqual(api.headers["Authorization"], "Bearer test-key")
        self.assertEqual(api.headers["api-key"], "test-key")
        self.assertEqual(api.request["model"], pr.MODEL_FLASH)
        # thinking off by default keeps the rewrite off the critical path;
        # the object shape is what the MiMo API documents
        self.assertEqual(api.request["thinking"], {"type": "disabled"})
        self.assertIn("max_completion_tokens", api.request)

        system, user = api.request["messages"]
        self.assertEqual(system["role"], "system")
        # the vendored guide, not a paraphrase of it, is what the model gets
        self.assertIn("integrated_multimodal_description", system["content"])
        self.assertIn("overall_soundscape", system["content"])
        self.assertIn("Output the prompt and nothing else", system["content"])

        # the rewrite is written against the job that will actually run
        self.assertIn("Input mode: I2VA", user["content"])
        self.assertIn("124 frames", user["content"])
        self.assertIn("864x480", user["content"])
        self.assertIn("<Picture 1>", user["content"])
        self.assertIn("two wrestlers", user["content"])

    def test_reference_mode_gets_the_reference_guide_and_every_label(self):
        api = _MockAPI()
        self.run_rewrite(api, mode=pr.MODE_REF2VA,
                         reference_counts={"images": 2, "videos": 1, "audios": 3})
        system, user = api.request["messages"]
        # ref-en.txt, not base-en.txt
        self.assertIn("subject_definitions", system["content"])
        self.assertIn("retention_analysis", system["content"])
        for label in ("<Picture 1>", "<Picture 2>", "<Video 1>",
                      "<Audio 1>", "<Audio 2>", "<Audio 3>"):
            self.assertIn(label, user["content"])
        self.assertNotIn("<Picture 3>", user["content"])

    def test_image_becomes_an_openai_image_url_part_for_the_flash_model(self):
        api = _MockAPI()
        _, model = self.run_rewrite(api, image=pr.data_url_from_bytes(PNG, "frame.png"))
        self.assertEqual(model, pr.MODEL_FLASH)  # omnimodal, so it keeps the image
        self.assertEqual(api.request["model"], pr.MODEL_FLASH)

        content = api.request["messages"][1]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertIn("An image is attached", content[0]["text"])

    def test_explicit_model_overrides_auto_routing(self):
        api = _MockAPI()
        _, model = self.run_rewrite(api, image=pr.data_url_from_bytes(PNG, "f.png"),
                                    model=pr.MODEL_PRO)
        self.assertEqual(model, pr.MODEL_PRO)


class TestTextOnlyModelGuard(unittest.TestCase):
    """mimo-v2.5-pro rejects any request carrying an image (live API: HTTP 404
    'No endpoints found that support image input'), so the combination has to
    be caught here rather than surfaced as an upstream error."""

    def setUp(self):
        self.image = pr.data_url_from_bytes(PNG, "ctx.png")

    def test_uploaded_image_with_a_text_only_model_is_refused(self):
        with self.assertRaises(pr.RewriteError) as ctx:
            pr.resolve_model_and_image(pr.MODEL_V25_PRO, self.image, image_is_explicit=True)
        self.assertEqual(ctx.exception.status, 400)
        # the message has to name both ways out
        self.assertIn(pr.MODEL_FLASH, ctx.exception.details)
        self.assertIn("auto", ctx.exception.details)

    def test_wrapper_supplied_image_is_dropped_not_refused(self):
        """The caller only picked the model; the keyframe became a context
        image on the wrapper's initiative, so it yields instead of erroring."""
        model, image, note = pr.resolve_model_and_image(
            pr.MODEL_V25_PRO, self.image, image_is_explicit=False)
        self.assertEqual(model, pr.MODEL_V25_PRO)
        self.assertIsNone(image)
        self.assertIn("text-only", note)

    def test_omnimodal_model_keeps_the_image(self):
        expected = {pr.MODEL_FLASH: pr.MODEL_FLASH, pr.MODEL_AUTO: pr.MODEL_FLASH, None: pr.MODEL_FLASH,
                    pr.MODEL_PRO: pr.MODEL_PRO, pr.MODEL_V25: pr.MODEL_V25}
        for requested, resolved in expected.items():
            model, image, note = pr.resolve_model_and_image(
                requested, self.image, image_is_explicit=True)
            self.assertEqual(model, resolved, requested)
            self.assertEqual(image, self.image)
            self.assertIsNone(note)

    def test_text_only_model_without_an_image_is_untouched(self):
        model, image, note = pr.resolve_model_and_image(pr.MODEL_V25_PRO, None, False)
        self.assertEqual((model, image, note), (pr.MODEL_V25_PRO, None, None))

    def test_capability_table(self):
        self.assertFalse(pr.supports_images(pr.MODEL_V25_PRO))
        for model in (pr.MODEL_FLASH, pr.MODEL_PRO, pr.MODEL_V25):
            self.assertTrue(pr.supports_images(model), model)

    def test_every_listed_model_resolves_and_anything_else_is_refused(self):
        for model in pr.MODEL_OPTIONS[1:]:
            self.assertEqual(pr.resolve_model(model), model)
        with self.assertRaises(pr.RewriteError) as ctx:
            pr.resolve_model("mimo-v2.4")
        self.assertIn(pr.MODEL_FLASH, ctx.exception.details)


class TestOutputHandling(RewriterTestCase):
    def test_strips_code_fences_and_reasoning(self):
        api = _MockAPI(reply="<think>planning</think>```text\nintegrated_multimodal_description: x\n```")
        prompt, _ = self.run_rewrite(api)
        self.assertEqual(prompt, "integrated_multimodal_description: x")

    def test_upstream_failure_is_a_502_not_a_silent_passthrough(self):
        """A failed rewrite must not fall through to generating from the raw
        intent -- that would quietly produce a bad video."""
        api = _MockAPI(status=500, body="upstream exploded")
        with self.assertRaises(pr.RewriteError) as ctx:
            self.run_rewrite(api)
        self.assertEqual(ctx.exception.status, 502)

        api = _MockAPI(reply="   ")
        with self.assertRaises(pr.RewriteError) as ctx:
            self.run_rewrite(api)
        self.assertEqual(ctx.exception.status, 502)


class TestConfiguration(unittest.TestCase):
    def test_missing_api_key_explains_both_ways_out(self):
        previous = os.environ.pop("MIMO_API_KEY", None)
        try:
            self.assertFalse(pr.is_configured())
            with self.assertRaises(pr.RewriteError) as ctx:
                asyncio.run(pr.rewrite(intent="a cat", mode=pr.MODE_T2VA, duration=5.17,
                                       frames=124, width=864, height=480))
            self.assertEqual(ctx.exception.status, 400)
            self.assertIn("MIMO_API_KEY", ctx.exception.details)
            self.assertIn("raw_prompt", ctx.exception.details)
        finally:
            if previous is not None:
                os.environ["MIMO_API_KEY"] = previous

    def test_mode_detection_follows_the_uploaded_keyframes(self):
        self.assertEqual(pr.detect_mode("text", False, False), pr.MODE_T2VA)
        self.assertEqual(pr.detect_mode("image", True, False), pr.MODE_I2VA)
        self.assertEqual(pr.detect_mode("image", True, True), pr.MODE_FL2VA)
        self.assertEqual(pr.detect_mode("image", False, True), pr.MODE_L2VA)
        self.assertEqual(pr.detect_mode("reference", False, False), pr.MODE_REF2VA)

    def test_oversized_context_image_is_rejected(self):
        with self.assertRaises(pr.RewriteError):
            pr.data_url_from_bytes(b"x" * (pr.MAX_IMAGE_BYTES + 1), "big.png")

    def test_vendored_guides_are_present(self):
        """The rewrite is only as good as the guide; a missing file must be a
        loud deployment error, not a degraded prompt."""
        self.assertIn("integrated_multimodal_description", pr.load_guide(pr.BASE_GUIDE))
        self.assertIn("subject_definitions", pr.load_guide(pr.REF_GUIDE))
        self.assertIn("T2VA", pr.load_skill())


if __name__ == "__main__":
    unittest.main()
