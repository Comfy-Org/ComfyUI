import re


SPECIAL_TOKEN_IDS = {
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|audio_cfg|>": 151654,
    "<|audio_start|>": 151669,
    "<|audio_end|>": 151670,
    "<|caption_start|>": 151671,
    "<|caption_end|>": 151672,
    "<|lyrics_start|>": 151673,
    "<|lyrics_end|>": 151674,
}
AUDIO_CODE_OFFSET = 151675

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LYRIC_TAG_RE = re.compile(r"\s*(\[[^\]]+\])\s*")


def _remove_markdown_format(text):
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", raw_line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines.append(line.rstrip())
    text = "\n".join(lines)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text.replace("• ", "").replace("    ", "")


def clean_caption(caption):
    def replace_special(match):
        inner = match.group(1).strip()
        parts = inner.split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

    text = _SPECIAL_TAG_RE.sub(replace_special, caption)
    text = _remove_markdown_format(text)
    return re.sub(r"\n{2,}", "\n", text)


def normalize_lyrics(lyrics):
    parts = _LYRIC_TAG_RE.split(lyrics)
    text = "\n".join(part.lower() if part.startswith("[") else part for part in parts if part)
    text = text.replace(" ^ ", "\n")
    return f"[start]\n{text}"


def build_prompt(caption, lyrics):
    return (
        "<|im_start|><|caption_start|>"
        f"{clean_caption(caption)}"
        "<|caption_end|><|lyrics_start|>"
        f"{normalize_lyrics(lyrics)}"
        "<|lyrics_end|><|im_end|><|audio_start|>"
    )


def validate_tokenizer(tokenizer):
    for token, expected in SPECIAL_TOKEN_IDS.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != expected:
            raise ValueError(f"MiniMax Music3 tokenizer mismatch for {token}: expected {expected}, got {token_id}")
