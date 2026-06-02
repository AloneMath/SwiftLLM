from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swiftllm.config import load_config
from swiftllm.data import ClimbMixShardManager, TextIterator
from swiftllm.tokenizer import SwiftTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build local token cache (.npy) from parquet shards")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--out-dir", type=str, default="./artifacts/token_cache", help="Output directory")
    p.add_argument("--dtype", type=str, default="uint16", choices=["uint16", "uint32"])
    p.add_argument("--max-train-tokens", type=int, default=0, help="0 means no limit")
    p.add_argument("--max-val-tokens", type=int, default=0, help="0 means no limit")
    p.add_argument("--progress-every", type=int, default=1_000_000)
    return p.parse_args()


def select_dtype(dtype_name: str, vocab_size: int):
    if dtype_name == "uint16":
        if vocab_size > 65535:
            raise ValueError(f"vocab_size={vocab_size} exceeds uint16 range, use --dtype uint32")
        return np.uint16
    return np.uint32


def encode_texts_to_cache(
    text_iter: TextIterator,
    tokenizer: SwiftTokenizer,
    out_path: Path,
    np_dtype,
    max_tokens: int,
    progress_every: int,
) -> int:
    bos = tokenizer.get_bos_token_id()
    tokens: list[int] = []
    count = 0
    next_report = progress_every

    for text in text_iter.iter_texts():
        ids = tokenizer.encode_ordinary(text)
        if not ids:
            continue

        tokens.append(bos)
        tokens.extend(ids)
        count += 1 + len(ids)

        if progress_every > 0 and count >= next_report:
            print(f"{out_path.name}: {count:,} tokens")
            next_report += progress_every

        if max_tokens > 0 and count >= max_tokens:
            break

    if not tokens:
        raise ValueError(f"No tokens produced for {out_path}")

    arr = np.asarray(tokens, dtype=np_dtype)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr, allow_pickle=False)
    return int(arr.size)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    tokenizer = SwiftTokenizer.from_file(cfg.data.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    np_dtype = select_dtype(args.dtype, vocab_size)

    mgr = ClimbMixShardManager(cfg.data)
    out_dir = Path(args.out_dir)
    train_out = out_dir / "train_tokens.npy"
    val_out = out_dir / "val_tokens.npy"

    train_text = TextIterator(
        paths=mgr.train_paths(),
        text_column=cfg.data.text_column,
        seed=cfg.train.seed,
    )
    val_text = TextIterator(
        paths=[mgr.val_path()],
        text_column=cfg.data.text_column,
        seed=cfg.train.seed + 1,
    )

    train_count = encode_texts_to_cache(
        text_iter=train_text,
        tokenizer=tokenizer,
        out_path=train_out,
        np_dtype=np_dtype,
        max_tokens=args.max_train_tokens,
        progress_every=args.progress_every,
    )
    val_count = encode_texts_to_cache(
        text_iter=val_text,
        tokenizer=tokenizer,
        out_path=val_out,
        np_dtype=np_dtype,
        max_tokens=args.max_val_tokens,
        progress_every=args.progress_every,
    )

    print(f"saved train cache: {train_out} ({train_count:,} tokens)")
    print(f"saved val cache: {val_out} ({val_count:,} tokens)")
    print(f"dtype: {np_dtype.__name__}, vocab_size: {vocab_size}")


if __name__ == "__main__":
    main()
