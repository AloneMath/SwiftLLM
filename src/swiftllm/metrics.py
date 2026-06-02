from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ThroughputStats:
    step: int
    lr: float
    train_loss: float
    val_bpb: float | None
    core_metric: float | None
    tok_per_sec: float
    step_ms: float
    elapsed_hours: float
    peak_vram_gb: float | None


class RunLogger:
    def __init__(self, run_name: str, checkpoint_dir: str) -> None:
        base = Path(checkpoint_dir)
        self.log_dir = base / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / f"{run_name}_metrics.csv"
        self.jsonl_path = self.log_dir / f"{run_name}_metrics.jsonl"

        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step",
                    "lr",
                    "train_loss",
                    "val_bpb",
                    "core_metric",
                    "tok_per_sec",
                    "step_ms",
                    "elapsed_hours",
                    "peak_vram_gb",
                ])

    def log(self, item: ThroughputStats) -> None:
        row = [
            item.step,
            item.lr,
            item.train_loss,
            item.val_bpb,
            item.core_metric,
            item.tok_per_sec,
            item.step_ms,
            item.elapsed_hours,
            item.peak_vram_gb,
        ]

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row_to_dict(item), ensure_ascii=True) + "\n")


def row_to_dict(item: ThroughputStats) -> dict[str, Any]:
    return {
        "step": item.step,
        "lr": item.lr,
        "train_loss": item.train_loss,
        "val_bpb": item.val_bpb,
        "core_metric": item.core_metric,
        "tok_per_sec": item.tok_per_sec,
        "step_ms": item.step_ms,
        "elapsed_hours": item.elapsed_hours,
        "peak_vram_gb": item.peak_vram_gb,
    }


def estimate_elapsed_hours(start_time: float) -> float:
    return (time.time() - start_time) / 3600.0


def target_progress_text(
    val_bpb: float | None,
    core: float | None,
    elapsed_hours: float,
    target_val_bpb: float,
    target_core: float,
    target_hours: float,
) -> str:
    vtxt = "n/a"
    ctxt = "n/a"
    ttxt = f"{elapsed_hours:.2f}h/{target_hours:.2f}h"

    if val_bpb is not None:
        vtxt = f"{val_bpb:.5f}/{target_val_bpb:.5f}"
    if core is not None:
        ctxt = f"{core:.4f}/{target_core:.4f}"

    return f"target_progress val_bpb={vtxt} core={ctxt} time={ttxt}"
