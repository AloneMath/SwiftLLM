from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swiftllm.config import load_config
from swiftllm.data import ClimbMixShardManager, TextIterator
from swiftllm.tokenizer import SwiftTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train tokenizer on local pretraining shards")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--min-frequency", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    mgr = ClimbMixShardManager(cfg.data)
    text_iter = TextIterator(
        paths=mgr.train_paths(),
        text_column=cfg.data.text_column,
        seed=cfg.train.seed,
    )

    tokenizer = SwiftTokenizer.train_from_iterator(
        text_iterator=text_iter.iter_texts(),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    out_dir = Path(cfg.data.tokenizer_path)
    tokenizer.save(out_dir)
    print(f"saved tokenizer to: {out_dir}")
    print(f"vocab_size: {tokenizer.get_vocab_size()}")
    print(f"bos_id: {tokenizer.get_bos_token_id()} pad_id: {tokenizer.get_pad_token_id()}")


if __name__ == "__main__":
    main()
