from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    data_dir: str
    tokenizer_path: str = "./artifacts/tokenizer"
    shard_prefix: str = "shard_"
    train_shards: int = 8
    val_shard_index: int = 6542
    text_column: str = "text"
    token_cache_dir: str = ""


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 16
    d_model: int = 1024
    d_ff: int = 2816
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    dropout: float = 0.0
    use_conditional_memory: bool = False
    memory_slots: int = 64
    memory_k: int = 4


@dataclass
class TargetConfig:
    target_val_bpb: float = 0.7180
    target_core: float = 0.2565
    target_hours: float = 1.65


@dataclass
class BenchmarkConfig:
    enable_core_eval: bool = False
    core_every: int = 5000
    core_max_per_task: int = 100
    core_bundle_path: str = ""


@dataclass
class TrainConfig:
    run_name: str = "swiftllm_300m"
    device: str = "cuda"
    seed: int = 42
    compute_dtype: str = "float16"
    num_steps: int = 60000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_adjust_lr_fn: str = "match_rms_adamw"
    muon_ns_steps: int = 5
    muon_eps: float = 1e-7
    attention_backend: str = "auto"
    grad_clip: float = 1.0
    micro_batch_size: int = 1
    grad_accum_steps: int = 32
    max_seq_len: int = 1024
    eval_every: int = 1000
    eval_batches: int = 32
    log_every: int = 20
    save_every: int = 1000
    compile_model: bool = False
    compile_mode: str = "default"
    gradient_checkpointing: bool = False
    distributed: bool = False
    dist_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    dist_timeout_sec: int = 1800
    checkpoint_dir: str = "./checkpoints/swiftllm_300m"


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    targets: TargetConfig
    benchmark: BenchmarkConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | Path) -> Config:
    p = Path(path)
    raw = _load_yaml(p)
    data = DataConfig(**raw["data"])
    model = ModelConfig(**raw["model"])
    train = TrainConfig(**raw["train"])
    targets = TargetConfig(**raw.get("targets", {}))
    benchmark = BenchmarkConfig(**raw.get("benchmark", {}))

    if train.max_seq_len != model.max_seq_len:
        raise ValueError("train.max_seq_len must equal model.max_seq_len")

    return Config(data=data, model=model, train=train, targets=targets, benchmark=benchmark)
