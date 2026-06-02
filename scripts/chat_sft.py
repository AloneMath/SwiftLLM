from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swiftllm.checkpoint import save_checkpoint
from swiftllm.config import load_config
from swiftllm.data import SFTBatchIterator, SFTDataset
from swiftllm.metrics import RunLogger, ThroughputStats, estimate_elapsed_hours
from swiftllm.model import SwiftLLM
from swiftllm.tokenizer import SwiftTokenizer


def resolve_dtype(name: str) -> torch.dtype:
    lut = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in lut:
        raise ValueError(f"Unsupported compute dtype: {name}")
    return lut[name]


def cosine_lr(step: int, total: int, base: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return base * float(step + 1) / float(max(1, warmup))
    if step >= total:
        return min_lr
    ratio = (step - warmup) / float(max(1, total - warmup))
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (base - min_lr)


@torch.no_grad()
def eval_sft_bpb(model, val_it, eval_batches: int, token_bytes: torch.Tensor, dtype: torch.dtype, use_amp: bool) -> float:
    model.eval()
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=token_bytes.device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=token_bytes.device)

    for _ in range(eval_batches):
        x, y = next(val_it)
        with torch.autocast(device_type=x.device.type, dtype=dtype, enabled=use_amp):
            logits = model(x)
        logits = logits.view(-1, logits.size(-1))
        y_flat = y.view(-1)

        ce = torch.nn.functional.cross_entropy(logits, y_flat, reduction="none", ignore_index=-1)
        valid = y_flat >= 0
        y_safe = torch.where(valid, y_flat, torch.zeros_like(y_flat))
        bytes_flat = torch.where(valid, token_bytes[y_safe], torch.zeros_like(y_safe, dtype=token_bytes.dtype))
        total_nats += (ce * (bytes_flat > 0)).sum()
        total_bytes += bytes_flat.sum()

    model.train()
    if int(total_bytes.item()) == 0:
        return float("inf")
    return float(total_nats.item() / (math.log(2) * int(total_bytes.item())))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervised finetuning (SFT) for chat")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--train-jsonl", type=str, required=True)
    p.add_argument("--val-jsonl", type=str, default="")
    p.add_argument("--resume-ckpt", type=str, default="", help="Base/pretrained checkpoint path")
    p.add_argument("--sft-steps", type=int, default=3000)
    p.add_argument("--sft-lr", type=float, default=1e-4)
    p.add_argument("--sft-min-lr", type=float, default=1e-5)
    p.add_argument("--sft-warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(cfg.train.device)
    dtype = resolve_dtype(cfg.train.compute_dtype)
    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and dtype == torch.float16))

    tokenizer = SwiftTokenizer.from_file(cfg.data.tokenizer_path)
    cfg.model.vocab_size = tokenizer.get_vocab_size()

    model = SwiftLLM(cfg.model).to(device)
    if args.resume_ckpt:
        payload = torch.load(args.resume_ckpt, map_location=device)
        state = payload.get("model", payload)
        model.load_state_dict(state, strict=True)
        print(f"loaded base checkpoint: {args.resume_ckpt}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.sft_lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.train.weight_decay,
        fused=(device.type == "cuda"),
    )

    train_ds = SFTDataset(args.train_jsonl, tokenizer, cfg.train.max_seq_len)
    val_path = args.val_jsonl if args.val_jsonl else args.train_jsonl
    val_ds = SFTDataset(val_path, tokenizer, cfg.train.max_seq_len)

    train_it = SFTBatchIterator(train_ds, cfg.train.micro_batch_size, cfg.train.max_seq_len, device, cfg.train.seed)
    val_it = SFTBatchIterator(val_ds, cfg.train.micro_batch_size, cfg.train.max_seq_len, device, cfg.train.seed + 1)

    token_bytes = tokenizer.build_token_bytes().to(device)

    sft_run_name = f"{cfg.train.run_name}_sft"
    sft_ckpt_dir = Path(cfg.train.checkpoint_dir) / "sft"
    sft_ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(sft_run_name, str(sft_ckpt_dir))

    run_start = time.time()
    t_log_window = time.time()
    running_loss = 0.0

    for step in range(1, args.sft_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        lr = cosine_lr(step - 1, args.sft_steps, args.sft_lr, args.sft_min_lr, args.sft_warmup)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = next(train_it)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            loss = model(x, y)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

        running_loss += loss.item()

        if step % cfg.train.log_every == 0:
            elapsed = time.time() - t_log_window
            avg_loss = running_loss / cfg.train.log_every
            tokens = cfg.train.micro_batch_size * cfg.train.max_seq_len * cfg.train.log_every
            tok_per_sec = tokens / max(elapsed, 1e-6)
            elapsed_hours = estimate_elapsed_hours(run_start)
            print(f"sft_step={step:5d} lr={lr:.3e} train_loss={avg_loss:.4f} tok_per_sec={tok_per_sec:,.0f}")

            logger.log(
                ThroughputStats(
                    step=step,
                    lr=lr,
                    train_loss=avg_loss,
                    val_bpb=None,
                    core_metric=None,
                    tok_per_sec=tok_per_sec,
                    step_ms=(elapsed / cfg.train.log_every) * 1000.0,
                    elapsed_hours=elapsed_hours,
                    peak_vram_gb=(torch.cuda.max_memory_allocated() / (1024 ** 3)) if device.type == "cuda" else None,
                )
            )

            running_loss = 0.0
            t_log_window = time.time()

        if step % args.eval_every == 0:
            val_bpb = eval_sft_bpb(model, val_it, args.eval_batches, token_bytes, dtype, use_amp)
            print(f"sft_step={step:5d} val_bpb={val_bpb:.5f}")

        if step % args.save_every == 0:
            out_path = sft_ckpt_dir / f"step_{step:06d}.pt"
            save_checkpoint(out_path, model, optimizer, type("S", (), {"step": step, "best_val_loss": 0.0})())
            print(f"saved sft checkpoint: {out_path}")

    final_ckpt = sft_ckpt_dir / f"step_{args.sft_steps:06d}.pt"
    torch.save({"model": model.state_dict(), "meta": {"stage": "sft", "step": args.sft_steps}}, final_ckpt)
    print(f"sft completed: {final_ckpt}")


if __name__ == "__main__":
    main()
