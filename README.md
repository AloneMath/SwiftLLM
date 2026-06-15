# SwiftLLM

SwiftLLM is a compact, English-only LLM codebase designed for fast local iteration on a single GPU.
This project now supports a full no-web, no-RL pipeline:

1. tokenizer train/eval
2. base pretrain + bpb/core tracking
3. SFT train
4. chat eval (local GSM8K / ARC / MMLU / SmolTalk / HumanEval)
5. chat CLI

## Install (pip)

### Linux/macOS (bash)

```bash
cd <repo-root>
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
pip install -e .[dev]
```

### Windows (PowerShell)

```powershell
cd <repo-root>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install -e .[dev]
```

## Cross-platform command notes

- Commands below use forward-slash paths (for example `./configs/...`), which work on Linux/macOS and on Windows PowerShell.
- For rented cloud GPUs, connect to the remote machine (usually Linux) and run the same commands from the repository root.
- Commands are intentionally shown as single-line invocations for shell compatibility.

## Local data setup

This repository tracks source code and configs only.
Training/evaluation data, checkpoints, and generated artifacts are intentionally not versioned.

Put your own pretraining shards in `./base_data` (or update config paths to your storage layout).

The config uses:

- train shards: `00000-00007`
- val shard: `00008`

## Full pipeline commands

## Distributed launch (torchrun)

Use this when `train.distributed: true` in your config (for example `./configs/train_1p3b_h800_stage0.yaml`).

### Multi-GPU pretrain (8 GPUs, single node)

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.train --config ./configs/train_1p3b_h800_stage0.yaml
```

### Multi-GPU SFT (8 GPUs, single node)

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft --config ./configs/train_1p3b_h800_stage0.yaml --train-jsonl ./data/sft_train.jsonl --val-jsonl ./data/sft_val.jsonl --resume-ckpt ./checkpoints/swiftllm_1p3b_h800_stage0/step_040000.pt --sft-steps 3000
```

### 1) Train tokenizer

```bash
python -m scripts.tok_train --config ./configs/train_300m_3070.yaml --vocab-size 50257
```

### 2) Evaluate tokenizer compression

```bash
python -m scripts.tok_eval --config ./configs/train_300m_3070.yaml --max-texts 2000
```

### 3) Base pretrain

```bash
python -m scripts.train --config ./configs/train_300m_3070.yaml
```

### 3b) Base pretrain (fast mode: compile + gradient checkpointing)

```bash
python -m scripts.train --config ./configs/train_300m_3070_fast.yaml
```

### 3b-1) Base pretrain (A800 fast preset: flash attention + compile + token cache)

```bash
python -m scripts.train --config ./configs/train_300m_a800_fast.yaml
```

### 3c) Build token cache (optional, speeds up training input pipeline)

```bash
python -m scripts.build_token_cache --config ./configs/train_300m_3070_fast.yaml --out-dir ./artifacts/token_cache_8shards --dtype uint16
```

### 3d) 5-minute smoke with token cache

```bash
python -m scripts.train --config ./configs/train_300m_3070_5min_fast.yaml
```

### 3e) 5-minute speed test (token cache, no gradient checkpointing)

```bash
python -m scripts.train --config ./configs/train_300m_3070_5min_speed.yaml
```

### 3f) 5-minute optimizer A/B (AdamW vs Muon)

```bash
python -m scripts.train --config ./configs/train_300m_3070_5min_adamw_speed.yaml
python -m scripts.train --config ./configs/train_300m_3070_5min_muon.yaml
```

For a closer-to-5-minute AdamW smoke run on RTX 3070:

```bash
python -m scripts.train --config ./configs/train_300m_3070_adamw_5min_true.yaml
```

Attention backend A/B (same config except SDPA backend):

```bash
python -m scripts.train --config ./configs/train_300m_3070_adamw_5min_flash.yaml
python -m scripts.train --config ./configs/train_300m_3070_adamw_5min_math.yaml
```

Then compare logs:

```bash
python -m scripts.compare_runs --baseline-csv ./checkpoints/swiftllm_300m_3070_5min_adamw_speed/logs/swiftllm_300m_3070_5min_adamw_speed_metrics.csv --memory-csv ./checkpoints/swiftllm_300m_3070_5min_muon/logs/swiftllm_300m_3070_5min_muon_metrics.csv --out ./reports/ab_adamw_vs_muon.txt
```

### 4) Prepare SFT data (quick sample)

```bash
python -m scripts.make_sft_sample --out ./data/sft_train.jsonl --num 200
python -m scripts.make_sft_sample --out ./data/sft_val.jsonl --num 50
```

### 5) SFT train

```bash
python -m scripts.chat_sft --config ./configs/train_300m_3070.yaml --train-jsonl ./data/sft_train.jsonl --val-jsonl ./data/sft_val.jsonl --resume-ckpt ./checkpoints/swiftllm_300m_3070/step_030000.pt --sft-steps 3000
```

### 6) Chat evaluation (pure local eval set)

```bash
python -m scripts.chat_eval --config ./configs/quick_5min_run.yaml --ckpt ./checkpoints/swiftllm_quick_5min_run/sft/step_000800.pt --eval-root ./data_eval --tasks all --gsm8k-samples 10 --arc-samples 10 --mmlu-samples 10 --smoltalk-samples 10 --humaneval-samples 5 --gsm8k-max-new-tokens 64 --humaneval-max-new-tokens 128 --out ./reports/chat_eval.json
```

### 7) Run chat CLI

```bash
python -m scripts.chat_cli --config ./configs/train_300m_3070.yaml --ckpt ./checkpoints/swiftllm_300m_3070/sft/step_003000.pt
```

### Quick eval smoke (few minutes)

```bash
python -m scripts.chat_eval --config ./configs/train_300m_3070_5min_fast.yaml --ckpt ./checkpoints/swiftllm_300m_3070_5min/sft/step_000120.pt --eval-root ./data_eval --tasks gsm8k,humaneval --gsm8k-samples 20 --humaneval-samples 5 --gsm8k-max-new-tokens 48 --humaneval-max-new-tokens 96 --out ./reports/chat_eval_fast.json
```

## About evaluation data

- Keep GSM8K, ARC, MMLU, SmolTalk, and HumanEval as eval-only data.
- Do not mix them into base pretraining.
- All evals can run fully offline from `./data_eval`.
- HumanEval is executed locally with a pass/fail harness.

## Notes

- If you want real pass@k later, extend the HumanEval harness with multiple samples.

## Acknowledgements

- Reference repository: [karpathy/nanochat](https://github.com/karpathy/nanochat)
- Thanks to Andrej Karpathy for his education efforts and open-source work: [https://github.com/karpathy](https://github.com/karpathy)
