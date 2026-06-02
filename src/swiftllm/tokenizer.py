from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import Regex, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|pad|>",
    "<|system_start|>",
    "<|system_end|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]

# This pattern is intentionally close to GPT-4 style tokenization.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class SwiftTokenizer:
    def __init__(self, tokenizer: HFTokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def train_from_iterator(
        cls,
        text_iterator: Iterable[str],
        vocab_size: int,
        min_frequency: int = 0,
    ) -> "SwiftTokenizer":
        tokenizer = HFTokenizer(BPE(byte_fallback=True, unk_token=None, fuse_unk=False))
        tokenizer.normalizer = None
        tokenizer.pre_tokenizer = Sequence([
            Split(pattern=Regex(SPLIT_PATTERN), behavior="isolated", invert=False),
            ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = None
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=min_frequency,
            initial_alphabet=ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    @classmethod
    def from_file(cls, tokenizer_path: str | Path) -> "SwiftTokenizer":
        p = Path(tokenizer_path)
        if p.is_dir():
            p = p / "tokenizer.json"
        tokenizer = HFTokenizer.from_file(str(p))
        return cls(tokenizer)

    def save(self, tokenizer_dir: str | Path) -> None:
        p = Path(tokenizer_dir)
        p.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(p / "tokenizer.json"))

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def id_to_token(self, token_id: int) -> str | None:
        return self.tokenizer.id_to_token(token_id)

    def token_to_id(self, token: str) -> int | None:
        return self.tokenizer.token_to_id(token)

    def get_bos_token_id(self) -> int:
        token_id = self.token_to_id("<|bos|>")
        if token_id is None:
            raise ValueError("Missing BOS token in tokenizer")
        return token_id

    def get_pad_token_id(self) -> int:
        token_id = self.token_to_id("<|pad|>")
        if token_id is None:
            raise ValueError("Missing PAD token in tokenizer")
        return token_id

    def get_special_tokens(self) -> list[str]:
        return [t for t in SPECIAL_TOKENS if self.token_to_id(t) is not None]

    def encode_ordinary(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def encode(self, text, prepend=None, append=None, num_threads=None):
        if isinstance(text, str):
            ids = self.encode_ordinary(text)
            if prepend is not None:
                prepend_id = prepend if isinstance(prepend, int) else self.token_to_id(prepend)
                if prepend_id is None:
                    raise ValueError(f"Unknown special token: {prepend}")
                ids.insert(0, prepend_id)
            if append is not None:
                append_id = append if isinstance(append, int) else self.token_to_id(append)
                if append_id is None:
                    raise ValueError(f"Unknown special token: {append}")
                ids.append(append_id)
            return ids
        if isinstance(text, list):
            return [self.encode(t, prepend=prepend, append=append, num_threads=num_threads) for t in text]
        raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def build_token_bytes(self) -> torch.Tensor:
        vocab_size = self.get_vocab_size()
        token_bytes = torch.zeros(vocab_size, dtype=torch.int64)
        special = set(self.get_special_tokens())
        for token_id in range(vocab_size):
            token = self.id_to_token(token_id)
            if token is None or token in special:
                token_bytes[token_id] = 0
            else:
                token_bytes[token_id] = len(self.decode([token_id], skip_special_tokens=False).encode("utf-8"))
        return token_bytes

    def encode_messages(self, messages: list[dict], add_generation_prompt: bool = False) -> tuple[list[int], list[int]]:
        ids: list[int] = [self.get_bos_token_id()]
        labels: list[int] = [-1]

        role_tokens = {
            "system": ("<|system_start|>", "<|system_end|>", False),
            "user": ("<|user_start|>", "<|user_end|>", False),
            "assistant": ("<|assistant_start|>", "<|assistant_end|>", True),
        }

        for message in messages:
            role = message["role"]
            content = message["content"]
            if role not in role_tokens:
                raise ValueError(f"Unsupported role: {role}")

            start_token, end_token, train_on_content = role_tokens[role]
            start_id = self.token_to_id(start_token)
            end_id = self.token_to_id(end_token)
            if start_id is None or end_id is None:
                raise ValueError(f"Missing chat token for role: {role}")

            content_ids = self.encode_ordinary(content)
            ids.append(start_id)
            labels.append(-1)
            ids.extend(content_ids)
            if train_on_content:
                labels.extend(content_ids)
            else:
                labels.extend([-1] * len(content_ids))
            ids.append(end_id)
            labels.append(-1)

        if add_generation_prompt:
            assistant_start = self.token_to_id("<|assistant_start|>")
            if assistant_start is None:
                raise ValueError("Missing assistant start token")
            ids.append(assistant_start)
            labels.append(-1)

        return ids, labels

    def encode_prompt(self, messages: list[dict]) -> list[int]:
        ids, _ = self.encode_messages(messages, add_generation_prompt=True)
        return ids
