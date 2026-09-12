"""
Pure-Python byte-level BPE tokenizer.
Supports loading from HuggingFace tokenizer.json (LLaMA-style)
and from Mistral tekken JSON blobs.
No dependency on the `transformers`, `tokenizers`, or `regex` packages.
"""
import base64
import json
import os
import re
import unicodedata


# This is also the default pattern used by the previous MistralConverter path.
_LLAMA_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
_CONTRACTIONS = ("'re", "'ve", "'ll", "'s", "'t", "'m", "'d")


def _is_letter(c):
    return unicodedata.category(c)[0] == "L"


def _is_number(c):
    return unicodedata.category(c)[0] == "N"


def _is_whitespace(c):
    return c in " \t\n\r\v\f\x85\u2028\u2029" or unicodedata.category(c) == "Zs"


def _split_llama(text):
    pieces = []
    i = 0
    while i < len(text):
        contraction = None
        if text[i] == "'":
            for suffix in _CONTRACTIONS:
                if text[i:i + len(suffix)].casefold() == suffix:
                    contraction = text[i:i + len(suffix)]
                    break
        if contraction is not None:
            pieces.append(contraction)
            i += len(contraction)
            continue

        j = i
        if text[j] not in "\r\n" and not _is_letter(text[j]) and not _is_number(text[j]):
            j += 1
        if j < len(text) and _is_letter(text[j]):
            j += 1
            while j < len(text) and _is_letter(text[j]):
                j += 1
            pieces.append(text[i:j])
            i = j
            continue

        if _is_number(text[i]):
            j = i + 1
            while j < len(text) and j - i < 3 and _is_number(text[j]):
                j += 1
            pieces.append(text[i:j])
            i = j
            continue

        j = i
        if text[j] == " ":
            j += 1
        punct_start = j
        while j < len(text) and not _is_whitespace(text[j]) and not _is_letter(text[j]) and not _is_number(text[j]):
            j += 1
        if j > punct_start:
            while j < len(text) and text[j] in "\r\n":
                j += 1
            pieces.append(text[i:j])
            i = j
            continue

        if _is_whitespace(text[i]):
            j = i + 1
            while j < len(text) and _is_whitespace(text[j]):
                j += 1
            last_newline = max(text.rfind("\r", i, j), text.rfind("\n", i, j))
            if last_newline >= i:
                j = last_newline + 1
            elif j < len(text) and j - i > 1:
                j -= 1
            pieces.append(text[i:j])
            i = j
            continue

        pieces.append(text[i])
        i += 1
    return pieces


def _make_split_pattern(pattern_str):
    if pattern_str != _LLAMA_PATTERN:
        raise ValueError(f"Unsupported tokenizer split pattern: {pattern_str}")
    return _split_llama


def _bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class BPETokenizer:
    """Byte-level BPE tokenizer with optional BOS prepending."""

    def __init__(self, vocab, merges_by_pair, special_token_ids, pattern_str,
                 byte_encoder, byte_decoder, bos_id=None):
        self._vocab = vocab                          # str -> int
        self._inv_vocab = {v: k for k, v in vocab.items()}
        self._merges = merges_by_pair                # (str, str) -> priority int
        self._special_token_ids = special_token_ids  # str -> int
        self._special_ids = set(special_token_ids.values())
        self._byte_encoder = byte_encoder
        self._byte_decoder = byte_decoder
        self._bos_id = bos_id

        self._split = _make_split_pattern(pattern_str)
        sorted_specials = sorted(special_token_ids.keys(), key=len, reverse=True)
        if sorted_specials:
            self._special_split = re.compile(
                '(' + '|'.join(re.escape(s) for s in sorted_specials) + ')'
            )
        else:
            self._special_split = None

    def _bpe_encode_piece(self, chars):
        if len(chars) <= 1:
            return chars
        while True:
            min_rank = float('inf')
            best_pair = None
            for i in range(len(chars) - 1):
                r = self._merges.get((chars[i], chars[i + 1]), float('inf'))
                if r < min_rank:
                    min_rank = r
                    best_pair = (chars[i], chars[i + 1])
            if best_pair is None:
                break
            merged = best_pair[0] + best_pair[1]
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == best_pair[0] and chars[i + 1] == best_pair[1]:
                    new_chars.append(merged)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
            if len(chars) == 1:
                break
        return chars

    def _encode_raw(self, text):
        ids = []
        parts = self._special_split.split(text) if self._special_split else [text]
        for part in parts:
            if not part:
                continue
            if part in self._special_token_ids:
                ids.append(self._special_token_ids[part])
            else:
                for piece in self._split(part):
                    byte_chars = [self._byte_encoder[b] for b in piece.encode('utf-8')]
                    for tok in self._bpe_encode_piece(byte_chars):
                        ids.append(self._vocab[tok])
        return ids

    def __call__(self, text):
        ids = self._encode_raw(text)
        if self._bos_id is not None:
            ids = [self._bos_id] + ids
        return {"input_ids": ids}

    def get_vocab(self):
        return dict(self._vocab)

    def decode(self, token_ids, skip_special_tokens=True):
        buf = bytearray()
        for tid in token_ids:
            s = self._inv_vocab.get(tid, '')
            if tid in self._special_ids:
                if not skip_special_tokens:
                    buf.extend(s.encode('utf-8'))
            else:
                for c in s:
                    buf.append(self._byte_decoder[c])
        return buf.decode('utf-8', errors='replace')


def _extract_pattern(pretok):
    if pretok.get('type') == 'Sequence':
        for sub in pretok.get('pretokenizers', []):
            if sub.get('type') == 'Split':
                pat = sub.get('pattern', {})
                if 'Regex' in pat:
                    return pat['Regex']
    elif pretok.get('type') == 'Split':
        pat = pretok.get('pattern', {})
        if 'Regex' in pat:
            return pat['Regex']
    return None


def _extract_bos_id(post_processor, special_token_ids):
    if post_processor.get('type') == 'TemplateProcessing':
        single = post_processor.get('single', [])
        if single and 'SpecialToken' in single[0]:
            bos_str = single[0]['SpecialToken']['id']
            return special_token_ids.get(bos_str)
    return None


def from_tokenizer_json(path):
    """Load a BPETokenizer from a directory containing tokenizer.json."""
    tok_file = os.path.join(path, 'tokenizer.json')
    with open(tok_file, encoding='utf-8') as f:
        data = json.load(f)

    vocab = dict(data['model']['vocab'])  # str -> int

    merges_by_pair = {}
    for i, merge_str in enumerate(data['model'].get('merges', [])):
        a, b = merge_str.split(' ', 1)
        if (a, b) not in merges_by_pair:
            merges_by_pair[(a, b)] = i

    special_token_ids = {}
    for tok in data.get('added_tokens', []):
        special_token_ids[tok['content']] = tok['id']
        vocab[tok['content']] = tok['id']  # include in vocab for inv_vocab decode

    pattern = _extract_pattern(data.get('pre_tokenizer', {}))
    if pattern is None:
        raise ValueError(f"Could not extract regex pattern from {tok_file}")

    bos_id = _extract_bos_id(data.get('post_processor', {}), special_token_ids)

    byte_encoder = _bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    return BPETokenizer(vocab, merges_by_pair, special_token_ids, pattern,
                        byte_encoder, byte_decoder, bos_id=bos_id)


def from_tekken_json(data):
    """Build a BPETokenizer from a Mistral tekken JSON blob (bytes or str)."""
    mistral_vocab = json.loads(data)
    config = mistral_vocab["config"]

    byte_encoder = _bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    def tbts(b):
        return "".join(byte_encoder[ord(c)] for c in b.decode("latin-1"))

    special_token_offset = config["default_num_special_tokens"]
    max_vocab = config["default_vocab_size"] - special_token_offset

    raw_vocab = {}
    for w in mistral_vocab["vocab"]:
        r = w["rank"]
        if r >= max_vocab:
            continue
        raw_vocab[base64.b64decode(w["token_bytes"])] = r + special_token_offset

    special_tokens_dict = {}
    for w in mistral_vocab["special_tokens"]:
        if "token_bytes" in w:
            special_tokens_dict[base64.b64decode(w["token_bytes"])] = w["rank"]
        else:
            special_tokens_dict[w["token_str"]] = w["rank"]

    all_special = list(special_tokens_dict.keys())
    combined = dict(special_tokens_dict)
    combined.update(raw_vocab)

    bpe_vocab = {}
    merge_triples = []
    for token, rank in combined.items():
        if token not in all_special:
            bpe_vocab[tbts(token)] = rank
            if len(token) == 1:
                continue
            local = []
            for i in range(1, len(token)):
                pl, pr = token[:i], token[i:]
                if pl in combined and pr in combined and (pl + pr) in combined:
                    local.append((pl, pr, rank))
            local.sort(key=lambda x: (combined[x[0]], combined[x[1]]))
            merge_triples.extend(local)
        else:
            tok_str = token.decode("utf-8", errors="replace") if isinstance(token, bytes) else token
            bpe_vocab[tok_str] = rank

    merge_triples.sort(key=lambda v: v[2])

    merges_by_pair = {}
    for i, (pl, pr, _) in enumerate(merge_triples):
        pair = (tbts(pl), tbts(pr))
        if pair not in merges_by_pair:
            merges_by_pair[pair] = i

    special_str_ids = {}
    for tok in all_special:
        tok_str = tok.decode("utf-8", errors="replace") if isinstance(tok, bytes) else tok
        if tok_str in bpe_vocab:
            special_str_ids[tok_str] = bpe_vocab[tok_str]

    return BPETokenizer(bpe_vocab, merges_by_pair, special_str_ids, _LLAMA_PATTERN,
                        byte_encoder, byte_decoder, bos_id=None)


class LlamaTokenizerFast:
    """Drop-in replacement for transformers.LlamaTokenizerFast (read-only use)."""

    @staticmethod
    def from_pretrained(path, **kwargs):
        return from_tokenizer_json(path)
