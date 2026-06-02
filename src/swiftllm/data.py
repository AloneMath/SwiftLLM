from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch

from swiftllm.config import DataConfig
from swiftllm.tokenizer import SwiftTokenizer


class ClimbMixShardManager:
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self.data_dir = Path(cfg.data_dir)

    def train_paths(self) -> list[Path]:
        paths = [self.data_dir / f"{self.cfg.shard_prefix}{idx:05d}.parquet" for idx in range(self.cfg.train_shards)]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing train shards: {missing}")
        return paths

    def val_path(self) -> Path:
        p = self.data_dir / f"{self.cfg.shard_prefix}{self.cfg.val_shard_index:05d}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing validation shard: {p}")
        return p


class TextIterator:
    def __init__(
        self,
        paths: list[Path],
        text_column: str,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.paths = paths
        self.text_column = text_column
        self.rng = random.Random(seed)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))

    def iter_texts(self) -> Iterator[str]:
        paths = self.paths[:]
        self.rng.shuffle(paths)
        sample_idx = 0
        for path in paths:
            pf = pq.ParquetFile(path)
            row_group_ids = list(range(pf.num_row_groups))
            self.rng.shuffle(row_group_ids)
            for rg_id in row_group_ids:
                rg = pf.read_row_group(rg_id, columns=[self.text_column])
                texts = rg.column(self.text_column).to_pylist()
                self.rng.shuffle(texts)
                for text in texts:
                    if text:
                        if (sample_idx % self.world_size) == self.rank:
                            yield text
                        sample_idx += 1


class TokenStreamDataset:
    def __init__(self, text_iter: TextIterator, tokenizer: SwiftTokenizer) -> None:
        self.text_iter = text_iter
        self.tokenizer = tokenizer

    def iter_tokens(self) -> Iterator[int]:
        bos = self.tokenizer.get_bos_token_id()
        for text in self.text_iter.iter_texts():
            ids = self.tokenizer.encode_ordinary(text)
            if not ids:
                continue
            yield bos
            for token in ids:
                yield token


class PackedBatchIterator:
    def __init__(
        self,
        token_dataset: TokenStreamDataset,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> None:
        self.token_dataset = token_dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.token_iter = self.token_dataset.iter_tokens()

    def __iter__(self) -> "PackedBatchIterator":
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        tokens_needed = self.batch_size * (self.seq_len + 1)
        flat: list[int] = []
        for _ in range(tokens_needed):
            try:
                flat.append(next(self.token_iter))
            except StopIteration:
                self.token_iter = self.token_dataset.iter_tokens()
                flat.append(next(self.token_iter))

        t = torch.tensor(flat, dtype=torch.long)
        t = t.view(self.batch_size, self.seq_len + 1)
        x = t[:, :-1].to(self.device, non_blocking=True)
        y = t[:, 1:].to(self.device, non_blocking=True)
        return x, y


class TokenCacheReader:
    def __init__(self, cache_path: str | Path) -> None:
        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Token cache not found: {self.cache_path}")
        if self.cache_path.suffix.lower() != ".npy":
            raise ValueError(f"Token cache must be .npy: {self.cache_path}")

        arr = np.load(self.cache_path, mmap_mode="r")
        if arr.ndim != 1:
            raise ValueError(f"Token cache must be 1D: {self.cache_path}")
        if arr.dtype not in (np.uint16, np.uint32):
            raise ValueError(f"Token cache must be uint16/uint32: {self.cache_path}")
        if arr.size < 2:
            raise ValueError(f"Token cache is too small: {self.cache_path}")
        self.arr = arr
        self.size = int(arr.size)

    def get(self, start: int, length: int) -> np.ndarray:
        end = start + length
        if end <= self.size:
            return self.arr[start:end]

        first = self.arr[start:self.size]
        rem = end - self.size
        second = self.arr[0:rem]
        return np.concatenate((first, second), axis=0)


class PackedTokenCacheIterator:
    def __init__(
        self,
        cache_path: str | Path,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.reader = TokenCacheReader(cache_path)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.rng = random.Random(seed)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        tokens_per_row = self.seq_len + 1
        self.tokens_per_step = self.batch_size * tokens_per_row
        base = self.rng.randrange(0, max(1, self.reader.size - 1))
        self.pos = (base + self.rank * self.tokens_per_step) % self.reader.size

    def __iter__(self) -> "PackedTokenCacheIterator":
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        tokens_per_row = self.seq_len + 1
        needed = self.tokens_per_step
        flat = self.reader.get(self.pos, needed)
        stride = needed * self.world_size
        self.pos = (self.pos + stride) % self.reader.size

        t = torch.from_numpy(flat.astype(np.int64, copy=False)).view(self.batch_size, tokens_per_row)
        x = t[:, :-1].to(self.device, non_blocking=True)
        y = t[:, 1:].to(self.device, non_blocking=True)
        return x, y


class SFTExample:
    def __init__(self, input_ids: list[int], labels: list[int]) -> None:
        self.input_ids = input_ids
        self.labels = labels


class SFTDataset:
    def __init__(self, jsonl_path: str | Path, tokenizer: SwiftTokenizer, max_seq_len: int) -> None:
        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        if not self.path.exists():
            raise FileNotFoundError(f"SFT dataset not found: {self.path}")

        self.examples: list[SFTExample] = []
        self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                messages = row.get("messages")
                if not messages:
                    continue
                ids, labels = self.tokenizer.encode_messages(messages)
                if len(ids) < 2:
                    continue
                if len(ids) > self.max_seq_len:
                    ids = ids[-self.max_seq_len :]
                    labels = labels[-self.max_seq_len :]
                self.examples.append(SFTExample(ids, labels))

        if not self.examples:
            raise ValueError(f"No usable SFT examples in {self.path}")

    def __len__(self) -> int:
        return len(self.examples)


class SFTBatchIterator:
    def __init__(
        self,
        dataset: SFTDataset,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.rng = random.Random(seed)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.indices = list(range(len(dataset.examples)))
        self.rank_indices: list[int] = []
        self.pos = 0
        self._shuffle()

    def _shuffle(self) -> None:
        self.rng.shuffle(self.indices)
        self.rank_indices = self.indices[self.rank :: self.world_size]
        if not self.rank_indices:
            raise ValueError(
                f"Rank {self.rank} has no SFT samples. "
                f"dataset_size={len(self.indices)} world_size={self.world_size}"
            )
        self.pos = 0

    def __iter__(self) -> "SFTBatchIterator":
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pos + self.batch_size > len(self.rank_indices):
            self._shuffle()

        batch_idx = self.rank_indices[self.pos : self.pos + self.batch_size]
        self.pos += self.batch_size

        pad_id = self.dataset.tokenizer.get_pad_token_id()
        x = torch.full((self.batch_size, self.seq_len), pad_id, dtype=torch.long)
        y = torch.full((self.batch_size, self.seq_len), -1, dtype=torch.long)

        for i, ex_idx in enumerate(batch_idx):
            ex = self.dataset.examples[ex_idx]
            ids = ex.input_ids[: self.seq_len]
            labels = ex.labels[: self.seq_len]
            n = len(ids)
            x[i, :n] = torch.tensor(ids, dtype=torch.long)
            y[i, :n] = torch.tensor(labels, dtype=torch.long)

        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def estimate_tokens_per_step(batch_size: int, seq_len: int, grad_accum_steps: int) -> int:
    return batch_size * seq_len * grad_accum_steps


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def format_params(num_params: int) -> str:
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    return str(num_params)


def approx_flops_per_token(n_layers: int, d_model: int, d_ff: int, seq_len: int) -> float:
    attn = 4.0 * n_layers * d_model * d_model
    mlp = 3.0 * n_layers * d_model * d_ff
    context = 2.0 * n_layers * d_model * seq_len
    return attn + mlp + context


def chinchilla_target_tokens(num_params: int, ratio: float = 20.0) -> int:
    return int(math.ceil(num_params * ratio))
