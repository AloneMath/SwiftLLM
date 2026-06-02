from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_distributed_env() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def get_rank() -> int:
    if _dist_ready():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if _dist_ready():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if _dist_ready():
        dist.barrier()


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if _dist_ready():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if _dist_ready():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= float(get_world_size())
    return t


def init_distributed(
    enabled: bool,
    backend: str = "nccl",
    timeout_sec: int = 1800,
) -> tuple[bool, int, int, int]:
    if not enabled:
        return False, 0, 1, 0

    if not is_distributed_env():
        raise RuntimeError(
            "distributed=true but torchrun environment is missing. "
            "Launch with torchrun --standalone --nproc_per_node=<N> ..."
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(seconds=int(timeout_sec)),
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return True, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if _dist_ready():
        dist.destroy_process_group()
