from __future__ import annotations

import argparse
import math
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
    p = argparse.ArgumentParser(description="Evaluate tokenizer compression as bytes/token")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--max-texts", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    tokenizer = SwiftTokenizer.from_file(cfg.data.tokenizer_path)
    mgr = ClimbMixShardManager(cfg.data)
    text_iter = TextIterator(
        paths=[mgr.val_path()],
        text_column=cfg.data.text_column,
        seed=cfg.train.seed,
    )

    total_bytes = 0
    total_tokens = 0
    n = 0
    for text in text_iter.iter_texts():
        ids = tokenizer.encode_ordinary(text)
        if not ids:
            continue
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(ids)
        n += 1
        if n >= args.max_texts:
            break

    if total_tokens == 0:
        raise RuntimeError("No tokens produced during tokenizer eval")

    bpt = total_bytes / total_tokens
    tpb = total_tokens / total_bytes if total_bytes > 0 else math.inf

    print(f"texts: {n}")
    print(f"bytes_per_token: {bpt:.4f}")
    print(f"tokens_per_byte: {tpb:.4f}")


if __name__ == "__main__":
    main()
