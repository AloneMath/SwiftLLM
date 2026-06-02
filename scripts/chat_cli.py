from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swiftllm.config import load_config
from swiftllm.model import SwiftLLM
from swiftllm.tokenizer import SwiftTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chat CLI for SwiftLLM")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def load_chat_model(cfg_path: str, ckpt_path: str):
    cfg = load_config(cfg_path)
    tokenizer = SwiftTokenizer.from_file(cfg.data.tokenizer_path)
    cfg.model.vocab_size = tokenizer.get_vocab_size()

    device = torch.device(cfg.train.device)
    model = SwiftLLM(cfg.model).to(device)

    payload = torch.load(ckpt_path, map_location=device)
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=True)
    model.eval()
    return cfg, tokenizer, model, device


@torch.no_grad()
def infer_reply(model, tokenizer, device, messages, max_new_tokens: int, temperature: float) -> str:
    prompt_ids = tokenizer.encode_prompt(messages)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=50)
    out = tokenizer.decode(y[0].tolist(), skip_special_tokens=False)

    marker = "<|assistant_start|>"
    idx = out.rfind(marker)
    if idx >= 0:
        reply = out[idx + len(marker):]
    else:
        reply = out

    end_marker = "<|assistant_end|>"
    eidx = reply.find(end_marker)
    if eidx >= 0:
        reply = reply[:eidx]

    return reply.strip()


def main() -> None:
    args = parse_args()
    cfg, tokenizer, model, device = load_chat_model(args.config, args.ckpt)

    messages = [{"role": "system", "content": "You are a concise and helpful assistant."}]
    print("SwiftLLM CLI ready. Type /exit to quit.")

    while True:
        user = input("you> ").strip()
        if not user:
            continue
        if user.lower() in {"/exit", "exit", "quit"}:
            break

        messages.append({"role": "user", "content": user})
        reply = infer_reply(
            model,
            tokenizer,
            device,
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"bot> {reply}")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
