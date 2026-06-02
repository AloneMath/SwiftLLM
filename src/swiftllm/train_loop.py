from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import tiktoken
from torch.nn.parallel import DistributedDataParallel as DDP

from swiftllm.checkpoint import CheckpointState, restore_training_state, save_checkpoint
from swiftllm.config import Config
from swiftllm.core_eval import GPT2TokenizerCompat, evaluate_core
from swiftllm.data import (
    ClimbMixShardManager,
    PackedBatchIterator,
    PackedTokenCacheIterator,
    TextIterator,
    TokenStreamDataset,
    count_parameters,
    format_params,
)
from swiftllm.dist import barrier, cleanup_distributed, init_distributed, is_main_process
from swiftllm.eval import evaluate_bpb
from swiftllm.metrics import RunLogger, ThroughputStats, estimate_elapsed_hours, target_progress_text
from swiftllm.model import SwiftLLM
from swiftllm.tokenizer import SwiftTokenizer


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    return torch.device(device_name)


def resolve_dtype(name: str) -> torch.dtype:
    lut = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in lut:
        raise ValueError(f"Unsupported compute dtype: {name}")
    return lut[name]


def cosine_lr(step: int, cfg: Config) -> float:
    base = cfg.train.lr
    min_lr = cfg.train.min_lr
    warmup = cfg.train.warmup_steps
    total = cfg.train.num_steps

    if step < warmup:
        return base * float(step + 1) / float(max(1, warmup))

    if step >= total:
        return min_lr

    ratio = (step - warmup) / float(max(1, total - warmup))
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (base - min_lr)


def build_adamw_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay_params = []
    no_decay_params = []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def build_muon_hybrid_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> tuple[list[torch.nn.Parameter], list[dict], dict[str, int]]:
    muon_params: list[torch.nn.Parameter] = []
    adam_decay_params: list[torch.nn.Parameter] = []
    adam_no_decay_params: list[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_embedding_or_head = ("tok_emb" in name) or ("lm_head" in name)
        if p.dim() == 2 and not is_embedding_or_head:
            muon_params.append(p)
        elif p.dim() >= 2:
            adam_decay_params.append(p)
        else:
            adam_no_decay_params.append(p)

    adamw_groups = [
        {"params": adam_decay_params, "weight_decay": weight_decay},
        {"params": adam_no_decay_params, "weight_decay": 0.0},
    ]
    counts = {
        "muon": sum(p.numel() for p in muon_params),
        "adam_decay": sum(p.numel() for p in adam_decay_params),
        "adam_no_decay": sum(p.numel() for p in adam_no_decay_params),
    }
    return muon_params, adamw_groups, counts


class CombinedOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        named_optimizers: list[tuple[str, torch.optim.Optimizer]],
        params_for_scaler: list[torch.nn.Parameter],
        mode: str,
    ) -> None:
        super().__init__([{"params": params_for_scaler}], defaults={})
        self.named_optimizers = named_optimizers
        self.mode = mode

    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("CombinedOptimizer does not support closure")
        for _, optimizer in self.named_optimizers:
            optimizer.step()
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        for _, optimizer in self.named_optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def set_lrs(self, lr_map: dict[str, float]) -> None:
        for name, optimizer in self.named_optimizers:
            if name not in lr_map:
                continue
            lr = float(lr_map[name])
            for group in optimizer.param_groups:
                group["lr"] = lr

    def state_dict(self) -> dict:
        return {
            "_combined_optimizer": True,
            "mode": self.mode,
            "optimizers": {name: opt.state_dict() for name, opt in self.named_optimizers},
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if not state_dict.get("_combined_optimizer", False):
            raise ValueError("Checkpoint optimizer payload is not a CombinedOptimizer state")
        saved_mode = state_dict.get("mode")
        if saved_mode != self.mode:
            raise ValueError(f"Optimizer mode mismatch: checkpoint={saved_mode} current={self.mode}")

        saved_opts = state_dict.get("optimizers", {})
        for name, optimizer in self.named_optimizers:
            if name not in saved_opts:
                raise ValueError(f"Missing optimizer state for '{name}' in checkpoint")
            optimizer.load_state_dict(saved_opts[name])


def build_optimizer(
    cfg: Config,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, dict[str, int], str]:
    optimizer_name = cfg.train.optimizer.lower()

    if optimizer_name == "adamw":
        optim_groups = build_adamw_param_groups(model, cfg.train.weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=cfg.train.lr,
            betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
            fused=(device.type == "cuda"),
        )
        counts = {
            "adam_decay": sum(p.numel() for p in optim_groups[0]["params"]),
            "adam_no_decay": sum(p.numel() for p in optim_groups[1]["params"]),
        }
        info = (
            f"AdamW groups: decay={counts['adam_decay']:,} params, "
            f"no_decay={counts['adam_no_decay']:,} params, "
            f"betas=({cfg.train.adam_beta1:.2f}, {cfg.train.adam_beta2:.2f})"
        )
        return optimizer, counts, info

    if optimizer_name == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("optimizer=muon requested but torch.optim.Muon is not available")

        muon_params, adamw_groups, counts = build_muon_hybrid_param_groups(model, cfg.train.weight_decay)
        if len(muon_params) == 0:
            raise RuntimeError("optimizer=muon requested but no eligible 2D hidden-layer parameters were found")

        adamw = torch.optim.AdamW(
            adamw_groups,
            lr=cfg.train.lr,
            betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
            fused=(device.type == "cuda"),
        )
        muon = torch.optim.Muon(
            muon_params,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            momentum=cfg.train.muon_momentum,
            nesterov=cfg.train.muon_nesterov,
            eps=cfg.train.muon_eps,
            ns_steps=cfg.train.muon_ns_steps,
            adjust_lr_fn=cfg.train.muon_adjust_lr_fn,
        )

        optimizer = CombinedOptimizer(
            named_optimizers=[("adamw", adamw), ("muon", muon)],
            params_for_scaler=[p for p in model.parameters() if p.requires_grad],
            mode="muon",
        )
        info = (
            f"Muon hybrid groups: muon={counts['muon']:,} params, "
            f"adam_decay={counts['adam_decay']:,}, "
            f"adam_no_decay={counts['adam_no_decay']:,}, "
            f"muon_adjust_lr_fn={cfg.train.muon_adjust_lr_fn}"
        )
        return optimizer, counts, info

    raise ValueError(f"Unsupported optimizer: {cfg.train.optimizer}. Use 'adamw' or 'muon'.")


def build_iterators(
    cfg: Config,
    device: torch.device,
    tokenizer: SwiftTokenizer,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[PackedBatchIterator | PackedTokenCacheIterator, PackedBatchIterator | PackedTokenCacheIterator]:
    if cfg.data.token_cache_dir:
        cache_dir = Path(cfg.data.token_cache_dir)
        train_cache = cache_dir / "train_tokens.npy"
        val_cache = cache_dir / "val_tokens.npy"
        if train_cache.exists() and val_cache.exists():
            train_it = PackedTokenCacheIterator(
                cache_path=train_cache,
                batch_size=cfg.train.micro_batch_size,
                seq_len=cfg.train.max_seq_len,
                device=device,
                seed=cfg.train.seed,
                rank=rank,
                world_size=world_size,
            )
            val_it = PackedTokenCacheIterator(
                cache_path=val_cache,
                batch_size=cfg.train.micro_batch_size,
                seq_len=cfg.train.max_seq_len,
                device=device,
                seed=cfg.train.seed + 1,
                rank=rank,
                world_size=world_size,
            )
            print(f"Using token cache: {cache_dir}")
            return train_it, val_it
        else:
            print(f"Token cache not complete at {cache_dir}, fallback to parquet streaming")

    mgr = ClimbMixShardManager(cfg.data)

    train_text = TextIterator(
        paths=mgr.train_paths(),
        text_column=cfg.data.text_column,
        seed=cfg.train.seed,
        rank=rank,
        world_size=world_size,
    )
    val_text = TextIterator(
        paths=[mgr.val_path()],
        text_column=cfg.data.text_column,
        seed=cfg.train.seed + 1,
        rank=rank,
        world_size=world_size,
    )

    train_ds = TokenStreamDataset(train_text, tokenizer)
    val_ds = TokenStreamDataset(val_text, tokenizer)

    train_it = PackedBatchIterator(train_ds, cfg.train.micro_batch_size, cfg.train.max_seq_len, device)
    val_it = PackedBatchIterator(val_ds, cfg.train.micro_batch_size, cfg.train.max_seq_len, device)
    return train_it, val_it


def train(cfg: Config) -> None:
    dist_enabled = False
    rank = 0
    world_size = 1
    local_rank = 0

    try:
        dist_enabled, rank, world_size, local_rank = init_distributed(
            enabled=cfg.train.distributed,
            backend=cfg.train.dist_backend,
            timeout_sec=cfg.train.dist_timeout_sec,
        )

        is_main = is_main_process()

        def main_print(msg: str) -> None:
            if is_main:
                print(msg)

        set_seed(cfg.train.seed + rank)
        device = resolve_device(cfg.train.device)
        if dist_enabled and device.type == "cuda":
            device = torch.device("cuda", local_rank)
        dtype = resolve_dtype(cfg.train.compute_dtype)

        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            torch.cuda.reset_peak_memory_stats()

        tokenizer = SwiftTokenizer.from_file(cfg.data.tokenizer_path)
        cfg.model.vocab_size = tokenizer.get_vocab_size()

        model: torch.nn.Module = SwiftLLM(cfg.model).to(device)
        model.set_gradient_checkpointing(bool(cfg.train.gradient_checkpointing))
        model.set_attention_backend(cfg.train.attention_backend)
        n_params = count_parameters(model)
        optimizer, _, optimizer_info = build_optimizer(cfg, model, device)

        step, best_val_bpb = restore_training_state(cfg, model, optimizer)

        if cfg.train.compile_model:
            if not hasattr(torch, "compile"):
                main_print("compile requested but torch.compile is not available in this PyTorch build")
            else:
                try:
                    model = torch.compile(model, mode=cfg.train.compile_mode)
                    main_print(f"torch.compile enabled (mode={cfg.train.compile_mode})")
                except Exception as exc:
                    main_print(f"torch.compile failed, falling back to eager mode: {exc}")

        if dist_enabled:
            if device.type == "cuda":
                model = DDP(
                    model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=cfg.train.ddp_find_unused_parameters,
                )
            else:
                model = DDP(model, find_unused_parameters=cfg.train.ddp_find_unused_parameters)

        train_it, val_it = build_iterators(cfg, device, tokenizer, rank=rank, world_size=world_size)

        scaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and dtype == torch.float16))
        use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)

        logger: RunLogger | None = RunLogger(cfg.train.run_name, cfg.train.checkpoint_dir) if is_main else None
        run_start = time.time()

        core_tokenizer = GPT2TokenizerCompat(tiktoken.get_encoding("gpt2"))
        token_bytes = tokenizer.build_token_bytes().to(device)

        last_core = None
        last_val_bpb = None

        main_print("=" * 88)
        main_print(f"Run: {cfg.train.run_name}")
        main_print(f"Device: {device}")
        main_print(f"Distributed: {dist_enabled} (rank={rank}, world_size={world_size}, local_rank={local_rank})")
        main_print(f"Compute dtype: {cfg.train.compute_dtype}")
        main_print(f"Parameters: {n_params:,} ({format_params(n_params)})")
        main_print(f"Tokenizer: {cfg.data.tokenizer_path} (vocab={cfg.model.vocab_size})")
        main_print(f"Token cache dir: {cfg.data.token_cache_dir or '(disabled)'}")
        main_print(f"Conditional memory: {cfg.model.use_conditional_memory}")
        main_print(f"Gradient checkpointing: {cfg.train.gradient_checkpointing}")
        main_print(f"Attention backend: {cfg.train.attention_backend}")
        main_print(f"Compile model: {cfg.train.compile_model} ({cfg.train.compile_mode})")
        main_print(f"Optimizer: {cfg.train.optimizer}")
        main_print(optimizer_info)
        main_print(f"Checkpoint dir: {cfg.train.checkpoint_dir}")
        main_print(
            f"Targets: val_bpb<={cfg.targets.target_val_bpb}, CORE>={cfg.targets.target_core}, "
            f"time<={cfg.targets.target_hours}h"
        )
        main_print("=" * 88)

        t_log_window = time.time()
        running_loss = 0.0
        step_time_ms_accum = 0.0

        while step < cfg.train.num_steps:
            t_step0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0

            lr = cosine_lr(step, cfg)
            if isinstance(optimizer, CombinedOptimizer):
                optimizer.set_lrs({"adamw": lr, "muon": lr})
            else:
                for group in optimizer.param_groups:
                    group["lr"] = lr

            for _ in range(cfg.train.grad_accum_steps):
                x, y = next(train_it)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                    loss = model(x, y)
                    loss = loss / cfg.train.grad_accum_steps

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss_accum += loss.item()

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()

            running_loss += loss_accum
            step += 1
            step_ms = (time.time() - t_step0) * 1000.0
            step_time_ms_accum += step_ms

            if step % cfg.train.log_every == 0:
                elapsed = time.time() - t_log_window
                avg_loss = running_loss / cfg.train.log_every
                tokens = (
                    cfg.train.micro_batch_size
                    * cfg.train.max_seq_len
                    * cfg.train.grad_accum_steps
                    * cfg.train.log_every
                )
                tok_per_sec = tokens / max(elapsed, 1e-6)
                elapsed_hours = estimate_elapsed_hours(run_start)
                avg_step_ms = step_time_ms_accum / cfg.train.log_every

                peak_vram_gb = None
                if device.type == "cuda":
                    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

                progress = target_progress_text(
                    val_bpb=last_val_bpb,
                    core=last_core,
                    elapsed_hours=elapsed_hours,
                    target_val_bpb=cfg.targets.target_val_bpb,
                    target_core=cfg.targets.target_core,
                    target_hours=cfg.targets.target_hours,
                )

                if is_main:
                    print(
                        f"step={step:6d} lr={lr:.3e} train_loss={avg_loss:.4f} tok_per_sec={tok_per_sec:,.0f} "
                        f"step_ms={avg_step_ms:.1f} elapsed={elapsed_hours:.2f}h"
                    )
                    print(progress)

                    if logger is not None:
                        logger.log(
                            ThroughputStats(
                                step=step,
                                lr=lr,
                                train_loss=avg_loss,
                                val_bpb=last_val_bpb,
                                core_metric=last_core,
                                tok_per_sec=tok_per_sec,
                                step_ms=avg_step_ms,
                                elapsed_hours=elapsed_hours,
                                peak_vram_gb=peak_vram_gb,
                            )
                        )

                running_loss = 0.0
                step_time_ms_accum = 0.0
                t_log_window = time.time()

            if cfg.train.eval_every > 0 and step % cfg.train.eval_every == 0:
                val_bpb = evaluate_bpb(
                    model=model,
                    batch_iter=val_it,
                    eval_batches=cfg.train.eval_batches,
                    token_bytes=token_bytes,
                    autocast_dtype=dtype,
                    use_amp=use_amp,
                )
                last_val_bpb = val_bpb
                if is_main:
                    print(f"step={step:6d} val_bpb={val_bpb:.5f}")
                best_val_bpb = min(best_val_bpb, val_bpb)

            if (
                cfg.benchmark.enable_core_eval
                and cfg.benchmark.core_every > 0
                and step % cfg.benchmark.core_every == 0
            ):
                if dist_enabled:
                    barrier()
                if is_main:
                    core_model = model.module if isinstance(model, DDP) else model
                    core_model.eval()
                    core_out = evaluate_core(
                        model=core_model,
                        tokenizer=core_tokenizer,
                        work_dir=Path(cfg.train.checkpoint_dir),
                        device=device,
                        max_per_task=cfg.benchmark.core_max_per_task,
                        bundle_path=cfg.benchmark.core_bundle_path,
                    )
                    core_model.train()
                    last_core = float(core_out["core_metric"])
                    print(f"step={step:6d} core_metric={last_core:.4f}")
                if dist_enabled:
                    barrier()

            if cfg.train.save_every > 0 and step % cfg.train.save_every == 0:
                if dist_enabled:
                    barrier()
                if is_main:
                    ckpt_path = Path(cfg.train.checkpoint_dir) / f"step_{step:06d}.pt"
                    save_checkpoint(
                        path=ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        state=CheckpointState(step=step, best_val_loss=best_val_bpb),
                    )
                    print(f"saved checkpoint: {ckpt_path}")
                if dist_enabled:
                    barrier()

        if dist_enabled:
            barrier()
        if is_main:
            final_ckpt = Path(cfg.train.checkpoint_dir) / f"step_{step:06d}.pt"
            save_checkpoint(
                path=final_ckpt,
                model=model,
                optimizer=optimizer,
                state=CheckpointState(step=step, best_val_loss=best_val_bpb),
            )
            print(f"training completed. final checkpoint: {final_ckpt}")
        if dist_enabled:
            barrier()
    finally:
        cleanup_distributed()
