from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `python -m scripts.train` work without requiring editable install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swiftllm.config import load_config
from swiftllm.train_loop import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SwiftLLM")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
