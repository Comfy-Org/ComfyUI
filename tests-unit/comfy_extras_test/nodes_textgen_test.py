from comfy_extras.nodes_textgen import parse_ltx2_prompt


USER_PROMPT = "a fennec girl waving at the camera"


class TestParseLTX2Prompt:
    @staticmethod
    def _parse(generated_text: str) -> str:
        return parse_ltx2_prompt(generated_text, USER_PROMPT)

    def test_plain_text(self):
        assert self._parse("an enhanced prompt") == "an enhanced prompt"

    def test_dangling_close_keeps_text_before_it(self):
        # Non-thinking mode primes a "final" channel, whose close decodes to a lone </think>.
        assert self._parse("an enhanced prompt</think>") == "an enhanced prompt"

    def test_text_after_reasoning_block(self):
        assert self._parse("<think>reasoning</think>an enhanced prompt") == "an enhanced prompt"

    def test_reasoning_only_falls_back_to_prompt(self):
        assert self._parse("<think>reasoning</think>") == USER_PROMPT

    def test_empty_generation_falls_back_to_prompt(self):
        assert self._parse("") == USER_PROMPT

    def test_multiline_reasoning_block(self):
        assert self._parse("<think>\nline one\nline two\n</think>\nan enhanced prompt") == "an enhanced prompt"

    def test_reasoning_block_before_dangling_close(self):
        assert self._parse("<think>reasoning</think>an enhanced prompt</think>") == "an enhanced prompt"

    def test_channel_markers_are_stripped(self):
        assert self._parse("<|channel>final\nan enhanced prompt<channel|>") == "an enhanced prompt"

    def test_turn_markers_are_stripped(self):
        assert self._parse("<|turn>model\nan enhanced prompt") == "an enhanced prompt"

    def test_qwen35_orphan_close_keeps_final_answer(self):
        # Regression case from upstream PR #15611: Qwen3.5 with thinking=False
        # still emits a bare </think> with no opening <think> tag. The previous
        # regex required <think> to start the match, so the reasoning chain
        # leaked to downstream KSampler. parse_ltx2_prompt keeps the text
        # after the last </think>.
        qwen35_output = (
            "Let me analyze the request and draft a caption.\n"
            "Final answer is just the paragraph.\n"
            "</think>\n"
            "A woman in a pink dress walks through a sunny garden."
        )
        assert self._parse(qwen35_output) == "A woman in a pink dress walks through a sunny garden."

    def test_never_returns_empty(self):
        for generated_text in [
            "an enhanced prompt</think>",
            "<think>x</think>",
            "<think>x</think>an enhanced prompt",
            "an enhanced prompt",
            "</think>",
            "<channel|>",
            "   ",
        ]:
            assert self._parse(generated_text) != ""
