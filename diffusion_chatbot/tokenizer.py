import json
import re
from collections import Counter


SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[BOS]", "[SEP]", "[EOS]", "[UNK]"]
TOKEN_RE = re.compile(r"\[[A-Z]+\]|[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\s]")


class WordTokenizer:
    def __init__(self, token_to_id=None):
        if token_to_id is None:
            token_to_id = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def pad_id(self):
        return self.token_to_id["[PAD]"]

    @property
    def mask_id(self):
        return self.token_to_id["[MASK]"]

    @property
    def bos_id(self):
        return self.token_to_id["[BOS]"]

    @property
    def sep_id(self):
        return self.token_to_id["[SEP]"]

    @property
    def eos_id(self):
        return self.token_to_id["[EOS]"]

    @property
    def unk_id(self):
        return self.token_to_id["[UNK]"]

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    @classmethod
    def train(cls, texts, vocab_size=1024):
        counter = Counter()
        for text in texts:
            counter.update(basic_tokenize(text))

        token_to_id = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        for token, _count in counter.most_common(max(0, vocab_size - len(token_to_id))):
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

    def encode_text(self, text):
        return [self.token_to_id.get(token, self.unk_id) for token in basic_tokenize(text)]

    def encode_prompt(self, text, max_len):
        ids = [self.bos_id] + self.encode_text(text) + [self.sep_id]
        return pad_or_trim(ids, max_len, self.pad_id)

    def encode_response(self, text, max_len):
        ids = self.encode_text(text) + [self.eos_id]
        return pad_or_trim(ids, max_len, self.pad_id)

    def decode_ids(self, ids, stop_at_eos=True, skip_special=True):
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), "[UNK]")
            if stop_at_eos and token == "[EOS]":
                break
            if skip_special and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return detokenize(tokens)

    def tokens_from_ids(self, ids):
        return [self.id_to_token.get(int(idx), "[UNK]") for idx in ids]

    def to_json(self):
        return json.dumps({"token_to_id": self.token_to_id}, sort_keys=True)

    @classmethod
    def from_json(cls, payload):
        data = json.loads(payload)
        return cls(data["token_to_id"])


def basic_tokenize(text):
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def pad_or_trim(ids, max_len, pad_id):
    ids = list(ids[:max_len])
    if len(ids) < max_len:
        ids.extend([pad_id] * (max_len - len(ids)))
    return ids


def detokenize(tokens):
    out = ""
    no_space_before = {".", ",", "!", "?", ":", ";", "%", "'", ")"}
    no_space_after = {"(", "$"}
    for token in tokens:
        if not out:
            out = token
        elif token in no_space_before:
            out += token
        elif out[-1:] in no_space_after:
            out += token
        else:
            out += " " + token
    return out
