# Design Notes

This document explains how the current code translates key ideas from:

- nanochat
- modded-nanogpt
- LLaMA
- Conditional Memory via Scalable Lookup

## Scope and philosophy

The target is rapid local iteration on a single consumer GPU.
This repository intentionally prioritizes:

- readability
- stable defaults
- easy ablations

## Borrowed and adapted ideas

### From nanochat

- Keep a compact end-to-end training path.
- Avoid giant config frameworks.
- Focus on one clear experiment loop.

### From modded-nanogpt

- Strong training defaults and pragmatic optimizer setup.
- Keep benchmarks and speed diagnostics central.
- Treat implementation details as first-class for throughput.

### From LLaMA

- RMSNorm
- RoPE positional encoding
- SwiGLU-style feed-forward block
- Decoder-only transformer objective

### From scalable lookup memory

- Optional conditional memory module with top-k lookup.
- Learned key-value memory table.
- Query-conditioned retrieval merged by a learned gate.

This is a compact approximation for experimentation, not a full reproduction.

## Why 300M for this repo

300M is a practical compromise for an RTX 3070:

- enough capacity to make architecture changes observable
- still trainable with long context and accumulation
- significantly faster iteration than 1B+ models

## Data policy in this project

- Base dataset: `karpathy/climbmix-400b-shuffle`
- Default local run: first 8 training shards
- Validation: fixed last shard (`shard_06542.parquet`)

This policy enables quick iteration while preserving comparability across runs.

## Immediate experiments to run

1. Baseline: `train_300m_3070.yaml`
2. Memory ablation: `train_300m_mem.yaml`
3. Context stress test: increase `max_seq_len`, reduce `grad_accum_steps`
4. Data scaling test: increase `train_shards` from 8 to 16 and compare loss slope
