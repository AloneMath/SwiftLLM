from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunSummary:
    run_name: str
    best_train_loss: float
    best_val_bpb: float | None
    best_core: float | None
    max_tok_per_sec: float
    avg_tok_per_sec: float
    min_step_ms: float
    avg_step_ms: float
    peak_vram_gb: float | None
    hours_at_best_val: float | None


def parse_float(x: str) -> float | None:
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() == "none":
        return None
    return float(x)


def summarize_csv(path: Path) -> RunSummary:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows in {path}")

    run_name = path.stem.replace("_metrics", "")
    train_losses = [float(r["train_loss"]) for r in rows]
    tok_vals = [float(r["tok_per_sec"]) for r in rows]
    step_ms_vals = [float(r["step_ms"]) for r in rows]

    val_pairs = [(parse_float(r["val_bpb"]), float(r["elapsed_hours"])) for r in rows]
    val_pairs = [(v, h) for v, h in val_pairs if v is not None]

    core_vals = [parse_float(r["core_metric"]) for r in rows]
    core_vals = [x for x in core_vals if x is not None]

    vram_vals = [parse_float(r["peak_vram_gb"]) for r in rows]
    vram_vals = [x for x in vram_vals if x is not None]

    best_val = None
    best_val_h = None
    if val_pairs:
        best_val, best_val_h = min(val_pairs, key=lambda x: x[0])

    return RunSummary(
        run_name=run_name,
        best_train_loss=min(train_losses),
        best_val_bpb=best_val,
        best_core=max(core_vals) if core_vals else None,
        max_tok_per_sec=max(tok_vals),
        avg_tok_per_sec=sum(tok_vals) / len(tok_vals),
        min_step_ms=min(step_ms_vals),
        avg_step_ms=sum(step_ms_vals) / len(step_ms_vals),
        peak_vram_gb=max(vram_vals) if vram_vals else None,
        hours_at_best_val=best_val_h,
    )


def fmt(x: float | None, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def compare_runs(
    baseline_csv: Path,
    memory_csv: Path,
    target_val_bpb: float,
    target_core: float,
    target_hours: float,
) -> str:
    b = summarize_csv(baseline_csv)
    m = summarize_csv(memory_csv)

    lines = []
    lines.append("=== SwiftLLM Experiment Comparison ===")
    lines.append(f"baseline={b.run_name}")
    lines.append(f"memory={m.run_name}")
    lines.append("")
    lines.append("Metric, Baseline, Memory, Better")

    def better_lower(xb, xm):
        if xb is None and xm is None:
            return "n/a"
        if xb is None:
            return "memory"
        if xm is None:
            return "baseline"
        return "baseline" if xb < xm else "memory"

    def better_higher(xb, xm):
        if xb is None and xm is None:
            return "n/a"
        if xb is None:
            return "memory"
        if xm is None:
            return "baseline"
        return "baseline" if xb > xm else "memory"

    lines.append(
        f"best_val_bpb, {fmt(b.best_val_bpb, 5)}, {fmt(m.best_val_bpb, 5)}, "
        f"{better_lower(b.best_val_bpb, m.best_val_bpb)}"
    )
    lines.append(
        f"best_core, {fmt(b.best_core, 4)}, {fmt(m.best_core, 4)}, "
        f"{better_higher(b.best_core, m.best_core)}"
    )
    lines.append(
        f"max_tok_per_sec, {fmt(b.max_tok_per_sec, 0)}, {fmt(m.max_tok_per_sec, 0)}, "
        f"{better_higher(b.max_tok_per_sec, m.max_tok_per_sec)}"
    )
    lines.append(
        f"avg_tok_per_sec, {fmt(b.avg_tok_per_sec, 0)}, {fmt(m.avg_tok_per_sec, 0)}, "
        f"{better_higher(b.avg_tok_per_sec, m.avg_tok_per_sec)}"
    )
    lines.append(
        f"min_step_ms, {fmt(b.min_step_ms, 1)}, {fmt(m.min_step_ms, 1)}, "
        f"{better_lower(b.min_step_ms, m.min_step_ms)}"
    )
    lines.append(
        f"peak_vram_gb, {fmt(b.peak_vram_gb, 2)}, {fmt(m.peak_vram_gb, 2)}, "
        f"{better_lower(b.peak_vram_gb, m.peak_vram_gb)}"
    )

    lines.append("")
    lines.append("Targets")
    lines.append(f"target_val_bpb <= {target_val_bpb:.5f}")
    lines.append(f"target_core >= {target_core:.4f}")
    lines.append(f"target_hours <= {target_hours:.2f}")

    for run in [b, m]:
        val_ok = run.best_val_bpb is not None and run.best_val_bpb <= target_val_bpb
        core_ok = run.best_core is not None and run.best_core >= target_core
        time_ok = run.hours_at_best_val is not None and run.hours_at_best_val <= target_hours
        lines.append(
            f"{run.run_name}: val_ok={val_ok} core_ok={core_ok} time_ok={time_ok} "
            f"(best_val={fmt(run.best_val_bpb,5)}, best_core={fmt(run.best_core,4)}, "
            f"time_at_best_val={fmt(run.hours_at_best_val,2)}h)"
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare baseline and memory experiment logs")
    p.add_argument("--baseline-csv", type=str, required=True)
    p.add_argument("--memory-csv", type=str, required=True)
    p.add_argument("--target-val-bpb", type=float, default=0.7180)
    p.add_argument("--target-core", type=float, default=0.2565)
    p.add_argument("--target-hours", type=float, default=1.65)
    p.add_argument("--out", type=str, default="", help="Optional output text file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    baseline_csv = Path(args.baseline_csv)
    memory_csv = Path(args.memory_csv)

    report = compare_runs(
        baseline_csv=baseline_csv,
        memory_csv=memory_csv,
        target_val_bpb=args.target_val_bpb,
        target_core=args.target_core,
        target_hours=args.target_hours,
    )

    print(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"saved report: {out_path}")


if __name__ == "__main__":
    main()
