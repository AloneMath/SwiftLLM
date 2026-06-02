from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from swiftllm.config import Config
from swiftllm.model import SwiftLLM


@dataclass
class CheckpointState:
    step: int
    best_val_loss: float


def _unwrap_model(model: SwiftLLM) -> SwiftLLM:
    # torch.compile may wrap model in OptimizedModule and expose original model as _orig_mod
    inner = getattr(model, "_orig_mod", None)
    if inner is not None:
        return inner
    return model


def save_checkpoint(path: str | Path, model: SwiftLLM, optimizer: torch.optim.Optimizer, state: CheckpointState) -> None:
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
    model: SwiftLLM,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float]:
    latest_path, payload = load_latest_checkpoint(cfg.train.checkpoint_dir)
    if payload is None:
        return 0, float("inf")

    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    step = int(payload["state"]["step"])
    best_val_loss = float(payload["state"]["best_val_loss"])
    print(f"Resumed from {latest_path} at step={step} best_val_loss={best_val_loss:.4f}")
    return step, best_val_loss
