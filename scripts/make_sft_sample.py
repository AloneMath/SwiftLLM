from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swiftllm.tokenizer import SwiftTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a tiny sample SFT dataset")
    p.add_argument("--out", type=str, default="./data/sft_sample.jsonl")
    p.add_argument("--num", type=int, default=200)
    return p.parse_args()


def build_example(i: int) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"What is {i} plus {i}?"},
            {"role": "assistant", "content": f"The answer is {2*i}."},
        ]
    }


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for i in range(args.num):
            f.write(json.dumps(build_example(i), ensure_ascii=True) + "\n")

    print(f"saved sample SFT data: {out}")


if __name__ == "__main__":
    main()
