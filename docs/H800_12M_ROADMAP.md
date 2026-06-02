# H800 12-Month Roadmap

This roadmap starts on June 2, 2026 and runs for 12 months.
It is designed for teams that can access H800 GPU clusters and want to build a strong practical assistant model without depending on H100/A100.

## Scope and assumptions

- Primary objective: maximize end-to-end task completion, not raw benchmark rank.
- Hardware profile:
  - Minimum viable: 8 x H800 80GB
  - Recommended: 32 x H800 80GB
  - Stretch: 64+ x H800 80GB
- Current repository status:
  - Single-process training path is implemented.
  - Multi-GPU scale-out (DDP/FSDP, data sharding, fault-tolerant restarts) should be treated as an early infrastructure milestone.

## Success metrics

Track these metrics every week:

1. Task completion rate on your internal agent benchmark.
2. Tool-call success rate and recovery rate after tool errors.
3. Code benchmark pass rate and regression count.
4. Training throughput tokens/sec and GPU utilization.
5. Training stability: NaN events, divergence events, restart frequency.
6. Cost efficiency: tokens per dollar and eval quality per dollar.

## Data and model budget guidance

- Pretraining tokens: 500B to 1T high-quality tokens.
- SFT examples: 1M to 2M examples (with tool-use formats).
- Preference pairs: 200K to 500K pairs (DPO/KTO style).
- Tool trajectories: 1M to 3M traces with verification labels.
- Main model family:
  - Stage A: ~1.3B dense model for fast iteration.
  - Stage B: ~3B dense or sparse variant for capability lift.
  - Stage C: distilled deployable model for latency/cost targets.

## Quarter-by-quarter execution

## Q1 (Month 1-3): Foundation and Stage-A bring-up

1. Build reliable training operations:
   - Standard run metadata, metrics schema, and checkpoint hygiene.
   - Resume-safe runs with automatic health checks.
2. Build data pipeline:
   - Deduplication, quality filtering, language filtering, source weighting.
   - Version every dataset release.
3. Run Stage-A pretraining:
   - Train a 1.3B model to validate stability and throughput.
   - Target early milestones on loss trend and training efficiency.

Deliverables:

1. Reproducible training runbook.
2. Data release `v0`.
3. First stable 1.3B checkpoint series.

## Q2 (Month 4-6): Core capability expansion

1. Scale data and context:
   - Increase pretraining coverage and long-context exposure.
2. Train mainline checkpoints:
   - Continue 1.3B and start 3B branch.
3. Establish strict eval gates:
   - Weekly pass/fail thresholds on code and tool tasks.

Deliverables:

1. Data release `v1` and `v2`.
2. Mainline 1.3B and 3B checkpoints.
3. Automated regression dashboard.

## Q3 (Month 7-9): Post-training and agent behavior

1. SFT at scale:
   - Add structured instructions, multi-step tool plans, and failure recovery examples.
2. Preference optimization:
   - DPO/KTO rounds on hard task subsets.
3. Agent loop hardening:
   - Plan -> Tool -> Verify -> Retry with explicit stopping criteria.

Deliverables:

1. SFT dataset release and training recipe.
2. Preference dataset release and tuning recipe.
3. Stable agent behavior profile on internal benchmark.

## Q4 (Month 10-12): Distillation and production readiness

1. Distill from strongest checkpoint into lower-latency model.
2. Quantize and optimize inference stack (8-bit / 4-bit paths).
3. Publish model cards, eval reports, and reproducibility docs.

Deliverables:

1. Deployable distilled checkpoint.
2. Quantized inference profiles with quality/latency tradeoffs.
3. Public technical report and release package.

## Immediate execution plan for this repository

## Week 1

1. Run token cache generation:

```powershell
python -m scripts.build_token_cache --config .\configs\train_1p3b_h800_stage0.yaml --out-dir .\artifacts\token_cache_8shards --dtype uint16
```

2. Launch Stage-A run:

```powershell
python -m scripts.train --config .\configs\train_1p3b_h800_stage0.yaml
```

3. Track:
   - tokens/sec
   - memory usage
   - validation bpb trend
   - checkpoint size and save duration

## Week 2

1. Tune `grad_accum_steps`, `eval_every`, and `attention_backend`.
2. Compare eager vs `torch.compile`.
3. Fix bottlenecks before increasing model or token budget.

## Risks and mitigations

1. Data quality bottleneck:
   - Mitigation: establish strict data quality gates before scale.
2. Training instability at larger scale:
   - Mitigation: add automatic run abort/restart thresholds.
3. Cost overrun:
   - Mitigation: keep weekly go/no-go gates tied to target metrics.
4. Evaluation drift:
   - Mitigation: lock benchmark versioning and keep a frozen test split.
