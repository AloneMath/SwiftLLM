from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from swiftllm.config import Config


@dataclass
class CheckpointState:
    step: int
    best_val_loss: float


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    # Keep unwrapping until we reach the real SwiftLLM module.
    # Handles wrappers such as DDP (.module) and torch.compile (._orig_mod).
    inner = model
    while True:
        ddp_inner = getattr(inner, "module", None)
        if ddp_inner is not None:
            inner = ddp_inner
            continue

        compiled_inner = getattr(inner, "_orig_mod", None)
        if compiled_inner is not None:
            inner = compiled_inner
            continue

        return inner


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: CheckpointState,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = _unwrap_model(model)
    payload = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": {
            "step": state.step,
            "best_val_loss": state.best_val_loss,
        },
    }
    torch.save(payload, p)


def load_latest_checkpoint(ckpt_dir: str | Path) -> tuple[Path | None, dict | None]:
    p = Path(ckpt_dir)
    if not p.exists():
        return None, None
    checkpoints = sorted(p.glob("step_*.pt"))
    if not checkpoints:
        return None, None
    latest = checkpoints[-1]
    payload = torch.load(latest, map_location="cpu")
    return latest, payload


def restore_training_state(
    cfg: Config,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float]:
    latest_path, payload = load_latest_checkpoint(cfg.train.checkpoint_dir)
    if payload is None:
        return 0, float("inf")

    model_to_load = _unwrap_model(model)
    model_to_load.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    step = int(payload["state"]["step"])
    best_val_loss = float(payload["state"]["best_val_loss"])
    print(f"Resumed from {latest_path} at step={step} best_val_loss={best_val_loss:.4f}")
    return step, best_val_loss
