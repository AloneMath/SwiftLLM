from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from swiftllm.dist import all_reduce_sum


@torch.no_grad()
def evaluate_bpb(
    model,
    batch_iter,
    eval_batches: int,
    token_bytes: torch.Tensor,
    autocast_dtype: torch.dtype,
    use_amp: bool,
) -> float:
    model.eval()

    total_nats = torch.tensor(0.0, dtype=torch.float32, device=token_bytes.device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=token_bytes.device)

    for _ in range(eval_batches):
        x, y = next(batch_iter)
        with torch.autocast(device_type=x.device.type, dtype=autocast_dtype, enabled=use_amp):
            logits = model(x)

        logits = logits.view(-1, logits.size(-1))
        y_flat = y.view(-1)

        ce = F.cross_entropy(logits, y_flat, reduction="none", ignore_index=-1)
        valid = y_flat >= 0
        y_safe = torch.where(valid, y_flat, torch.zeros_like(y_flat))
        bytes_flat = torch.where(valid, token_bytes[y_safe], torch.zeros_like(y_safe, dtype=token_bytes.dtype))

        total_nats += (ce * (bytes_flat > 0)).sum()
        total_bytes += bytes_flat.sum()

    model.train()

    total_nats = all_reduce_sum(total_nats)
    total_bytes = all_reduce_sum(total_bytes)

    tb = int(total_bytes.item())
    if tb == 0:
        return float("inf")

    return float(total_nats.item() / (math.log(2) * tb))
