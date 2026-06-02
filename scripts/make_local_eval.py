from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_lines(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def make_gsm8k_samples(n: int) -> list[dict]:
    rows = []
    for i in range(n):
        a = i + i
        rows.append({
            "question": f"What is {i} plus {i}?",
            "answer": f"#### {a}",
        })
    return rows


def make_humaneval_samples(n: int) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append({
            "prompt": f"def add_{i}(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n",
        })
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create local eval jsonl files")
    p.add_argument("--out-dir", type=str, default="./data/local_eval")
    p.add_argument("--gsm8k-num", type=int, default=50)
    p.add_argument("--humaneval-num", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    write_lines(out_dir / "gsm8k.jsonl", make_gsm8k_samples(args.gsm8k_num))
    write_lines(out_dir / "humaneval.jsonl", make_humaneval_samples(args.humaneval_num))
    print(f"saved local eval files under: {out_dir}")


if __name__ == "__main__":
    main()
