from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from swiftllm.config import ModelConfig

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:
    SDPBackend = None  # type: ignore[assignment]
    sdpa_kernel = None  # type: ignore[assignment]


_BACKEND_WARNED: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg in _BACKEND_WARNED:
        return
    _BACKEND_WARNED.add(msg)
    print(msg)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def precompute_rope_cache(seq_len: int, dim: int, theta: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    half = dim // 2
    idx = torch.arange(half, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (theta ** (idx / half))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    bsz, t, heads, d = x.shape
    x = x.view(bsz, t, heads, d // 2, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    cos_t = cos[:t].unsqueeze(0).unsqueeze(2)
    sin_t = sin[:t].unsqueeze(0).unsqueeze(2)
    y1 = x1 * cos_t - x2 * sin_t
    y2 = x1 * sin_t + x2 * cos_t
    y = torch.stack((y1, y2), dim=-1).flatten(-2)
    return y


def apply_rope_with_offset(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_start: int,
) -> torch.Tensor:
    bsz, t, heads, d = x.shape
    x = x.view(bsz, t, heads, d // 2, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    cos_t = cos[pos_start : pos_start + t].unsqueeze(0).unsqueeze(2)
    sin_t = sin[pos_start : pos_start + t].unsqueeze(0).unsqueeze(2)
    y1 = x1 * cos_t - x2 * sin_t
    y2 = x1 * sin_t + x2 * cos_t
    y = torch.stack((y1, y2), dim=-1).flatten(-2)
    return y


class ConditionalMemory(nn.Module):
    def __init__(self, d_model: int, slots: int, top_k: int) -> None:
        super().__init__()
        self.slots = slots
        self.top_k = top_k
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.val_proj = nn.Linear(d_model, d_model, bias=False)
        self.mem_keys = nn.Parameter(torch.randn(slots, d_model) * 0.02)
        self.mem_vals = nn.Parameter(torch.randn(slots, d_model) * 0.02)
        self.gate = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.key_proj(x), dim=-1)
        k = F.normalize(self.mem_keys, dim=-1)
        scores = torch.einsum("btd,sd->bts", q, k)

        k_top = min(self.top_k, self.slots)
        values, idx = torch.topk(scores, k=k_top, dim=-1)
        weights = torch.softmax(values, dim=-1)

        selected_vals = self.mem_vals[idx]
        memory = torch.einsum("btk,btkd->btd", weights, selected_vals)
        g = torch.sigmoid(self.gate(x))
        return x + g * self.val_proj(memory)


class GatedMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gate(x)) * self.up(x)
        h = self.down(h)
        return self.dropout(h)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.repeat_kv = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attention_backend = "auto"

    def set_attention_backend(self, backend: str) -> None:
        self.attention_backend = backend.lower()

    def _resolve_backend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        dropout_p: float,
        is_causal: bool,
    ) -> str:
        requested = self.attention_backend
        if requested == "auto":
            return "auto"
        if requested not in {"flash", "efficient", "math", "cudnn"}:
            raise ValueError(
                f"Unsupported attention_backend: {requested}. "
                "Use auto|flash|efficient|math|cudnn."
            )

        if q.device.type != "cuda":
            _warn_once(f"[attention_backend] requested={requested} but device={q.device.type}; fallback=math")
            return "math"
        if SDPBackend is None or sdpa_kernel is None:
            _warn_once(f"[attention_backend] requested={requested} but SDP backend API missing; fallback=auto")
            return "auto"

        params_ctor = getattr(torch.backends.cuda, "SDPAParams", None)
        if params_ctor is None:
            _warn_once(f"[attention_backend] requested={requested} but SDPAParams missing; fallback=auto")
            return "auto"
        try:
            params = params_ctor(q, k, v, attn_mask, dropout_p, is_causal, False)
        except Exception:
            _warn_once(f"[attention_backend] requested={requested} but SDPAParams init failed; fallback=auto")
            return "auto"

        if requested == "flash":
            can_use = (
                hasattr(torch.backends.cuda, "can_use_flash_attention")
                and torch.backends.cuda.can_use_flash_attention(params, debug=False)
            )
            if not can_use:
                _warn_once("[attention_backend] requested=flash but unavailable on this build/hardware; fallback=auto")
                return "auto"
            return "flash"

        if requested == "efficient":
            can_use = (
                hasattr(torch.backends.cuda, "can_use_efficient_attention")
                and torch.backends.cuda.can_use_efficient_attention(params, debug=False)
            )
            if not can_use:
                _warn_once("[attention_backend] requested=efficient but unavailable for current shapes; fallback=auto")
                return "auto"
            return "efficient"

        if requested == "cudnn":
            can_use = (
                hasattr(torch.backends.cuda, "can_use_cudnn_attention")
                and torch.backends.cuda.can_use_cudnn_attention(params, debug=False)
            )
            if not can_use:
                _warn_once("[attention_backend] requested=cudnn but unavailable for current shapes; fallback=auto")
                return "auto"
            return "cudnn"

        return "math"

    def _sdpa_context(self, backend: str):
        if backend == "auto":
            return nullcontext()
        if sdpa_kernel is None or SDPBackend is None:
            return nullcontext()
        if backend == "flash":
            return sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
        if backend == "efficient":
            return sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION])
        if backend == "math":
            return sdpa_kernel(backends=[SDPBackend.MATH])
        if backend == "cudnn":
            return sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION])
        return nullcontext()

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        dropout_p: float,
        is_causal: bool,
    ) -> torch.Tensor:
        backend = self._resolve_backend(q, k, v, attn_mask, dropout_p, is_causal)
        with self._sdpa_context(backend):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, t, _ = x.shape

        q = self.q_proj(x).view(bsz, t, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, t, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, t, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.repeat_kv > 1:
            k = k.repeat_interleave(self.repeat_kv, dim=2)
            v = v.repeat_interleave(self.repeat_kv, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = self._sdpa(
            q=q,
            k=k,
            v=v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(bsz, t, self.d_model)
        return self.o_proj(y)

    def forward_with_cache(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_k: torch.Tensor | None,
        past_v: torch.Tensor | None,
        pos_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, t, _ = x.shape

        q = self.q_proj(x).view(bsz, t, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, t, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, t, self.n_kv_heads, self.head_dim)

        q = apply_rope_with_offset(q, cos, sin, pos_start)
        k = apply_rope_with_offset(k, cos, sin, pos_start)

        if past_k is not None and past_v is not None:
            k_full = torch.cat((past_k, k), dim=1)
            v_full = torch.cat((past_v, v), dim=1)
        else:
            k_full = k
            v_full = v

        if self.repeat_kv > 1:
            k_attn = k_full.repeat_interleave(self.repeat_kv, dim=2)
            v_attn = v_full.repeat_interleave(self.repeat_kv, dim=2)
        else:
            k_attn = k_full
            v_attn = v_full

        q = q.transpose(1, 2)
        k_attn = k_attn.transpose(1, 2)
        v_attn = v_attn.transpose(1, 2)

        q_len = q.size(-2)
        k_len = k_attn.size(-2)
        q_pos = torch.arange(pos_start, pos_start + q_len, device=q.device).unsqueeze(1)
        k_pos = torch.arange(k_len, device=q.device).unsqueeze(0)
        allow = k_pos <= q_pos
        attn_mask = torch.zeros((q_len, k_len), dtype=q.dtype, device=q.device)
        attn_mask = attn_mask.masked_fill(~allow, float("-inf"))

        y = self._sdpa(
            q=q,
            k=k_attn,
            v=v_attn,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )

        y = y.transpose(1, 2).contiguous().view(bsz, t, self.d_model)
        return self.o_proj(y), k_full, v_full


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.dropout)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = GatedMLP(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.memory = (
            ConditionalMemory(cfg.d_model, cfg.memory_slots, cfg.memory_k)
            if cfg.use_conditional_memory
            else None
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        y = self.mlp(self.norm2(x))
        x = x + y
        if self.memory is not None:
            x = self.memory(x)
        return x

    def forward_with_cache(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_k: torch.Tensor | None,
        past_v: torch.Tensor | None,
        pos_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, new_k, new_v = self.attn.forward_with_cache(self.norm1(x), cos, sin, past_k, past_v, pos_start)
        x = x + attn_out
        y = self.mlp(self.norm2(x))
        x = x + y
        if self.memory is not None:
            x = self.memory(x)
        return x, new_k, new_v


class SwiftLLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)
        self.gradient_checkpointing = False

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_rope_cache(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rope_cos.numel() == 0 or self.rope_cos.shape[0] < seq_len or self.rope_cos.device != device:
            cos, sin = precompute_rope_cache(seq_len, self.cfg.d_model // self.cfg.n_heads, self.cfg.rope_theta, device)
            self.rope_cos = cos
            self.rope_sin = sin
        return self.rope_cos, self.rope_sin

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    def set_attention_backend(self, backend: str) -> None:
        for block in self.blocks:
            block.attn.set_attention_backend(backend)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_reduction: str = "mean",
    ) -> torch.Tensor:
        _, t = idx.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {t} exceeds max_seq_len {self.cfg.max_seq_len}")

        x = self.tok_emb(idx)
        cos, sin = self._get_rope_cache(t, idx.device)

        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                x = activation_checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits

        if loss_reduction == "none":
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="none",
                ignore_index=-1,
            ).view(targets.shape)
            return loss

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction=loss_reduction,
            ignore_index=-1,
        )
        return loss

    def _prefill_with_cache(
        self,
        idx: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]:
        _, t = idx.shape
        if t == 0:
            raise ValueError("Input must contain at least one token")
        if t > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {t} exceeds max_seq_len {self.cfg.max_seq_len}")

        x = self.tok_emb(idx)
        cos, sin = self._get_rope_cache(self.cfg.max_seq_len, idx.device)

        past_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        pos_start = 0
        for block in self.blocks:
            x, k_full, v_full = block.forward_with_cache(x, cos, sin, None, None, pos_start)
            past_kv.append((k_full, v_full))

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, past_kv, cos, sin

    def _decode_one_with_cache(
        self,
        next_token: torch.Tensor,
        past_kv: list[tuple[torch.Tensor, torch.Tensor]],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        x = self.tok_emb(next_token)
        pos_start = past_kv[0][0].size(1)

        new_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block, (past_k, past_v) in zip(self.blocks, past_kv):
            x, k_full, v_full = block.forward_with_cache(x, cos, sin, past_k, past_v, pos_start)
            new_kv.append((k_full, v_full))

        x = self.norm(x)
        logits = self.lm_head(x[:, -1, :])
        return logits, new_kv

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
    ) -> torch.Tensor:
        self.eval()

        if idx.size(1) > self.cfg.max_seq_len:
            idx = idx[:, -self.cfg.max_seq_len :]

        logits, past_kv, cos, sin = self._prefill_with_cache(idx)
        logits = logits[:, -1, :]

        for _ in range(max_new_tokens):
            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_token], dim=1)

            if idx.size(1) > self.cfg.max_seq_len:
                idx = idx[:, -self.cfg.max_seq_len :]
                logits, past_kv, cos, sin = self._prefill_with_cache(idx)
                logits = logits[:, -1, :]
            else:
                logits, past_kv = self._decode_one_with_cache(next_token, past_kv, cos, sin)

        return idx


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_flops(cfg: ModelConfig) -> float:
    return 6.0 * cfg.n_layers * cfg.d_model * cfg.d_model + 12.0 * cfg.n_layers * cfg.d_model * cfg.d_ff

