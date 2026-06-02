from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import tiktoken

# Make `python -m scripts.sample` work without requiring editable install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swiftllm.config import load_config
from swiftllm.model import SwiftLLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample from SwiftLLM")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--ckpt", type=str, default="", help="Checkpoint path; if empty, use latest")
    p.add_argument("--prompt", type=str, required=True, help="Prompt text")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    return p.parse_args()


def find_latest_checkpoint(ckpt_dir: Path) -> Path:
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return ckpts[-1]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(cfg.train.device)
    enc = tiktoken.get_encoding("gpt2")

    model = SwiftLLM(cfg.model).to(device)
    ckpt_path = Path(args.ckpt) if args.ckpt else find_latest_checkpoint(Path(cfg.train.checkpoint_dir))

    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    tokens = enc.encode_ordinary(args.prompt)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    y = model.generate(
        x,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    out = enc.decode(y[0].tolist())
    print(f"checkpoint: {ckpt_path}")
    print("---")
    print(out)


if __name__ == "__main__":
    main()
