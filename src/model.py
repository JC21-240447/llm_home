r"""Model A: a small decoder-only Transformer (GPT style), written from scratch with PyTorch.
Model A: 小さなデコーダーのみの Transformer(GPT 型)。PyTorch で1から書いたもの。

==========================  ENGLISH  ==========================

What the model does: it reads a sequence of token numbers and, at every position,
outputs a score (logit) for every token in the vocabulary: "how likely is each
token to come next?". Training (Day 9) teaches it to give the real next token a
high score. Generation (Day 10) turns the scores into text.

Building blocks (from input to output):
    1. Token embedding   : token number -> vector of size d_model
    2. N x Block         : each block is
                              x = x + Attention(RMSNorm(x))   (look at earlier tokens)
                              x = x + MLP(RMSNorm(x))         (process each position)
                           ("pre-norm" with "residual" connections)
    3. Final RMSNorm
    4. Output layer      : vector -> one score per vocabulary token.
                           It SHARES its weights with the token embedding
                           ("weight tying").

Design choices (defended in the Day 8 walkthrough):
    - RoPE (rotary position embedding): position is encoded by rotating the
      query/key vectors. No learned position table, so nothing ties the weights to
      one maximum length.
    - Causal attention: position t may only look at positions 0..t. Two versions:
      a readable "manual" one (matrix products + mask + softmax) and a fast one
      (torch.nn.functional.scaled_dot_product_attention). The tests prove they agree.
    - RMSNorm, no biases, GELU MLP with 4x width, weight tying, scaled init of the
      residual projections (std / sqrt(2 * n_layer)).

SERVER NOTE: pure PyTorch; identical on Windows and Linux, on CPU and CUDA.

==========================  日本語  ==========================

モデルの役割: トークン番号の列を読み、各位置で、語彙のすべてのトークンについて
「次に来る可能性の高さ」を表す点数(ロジット)を出す。学習(Day 9)で本当の次のトークンに
高い点数を付けるように教え、生成(Day 10)で点数を文章に変える。

部品(入力から出力まで):
    1. トークン埋め込み : トークン番号 → 大きさ d_model のベクトル
    2. ブロック × N     : x = x + Attention(RMSNorm(x))  (前のトークンを見る)
                          x = x + MLP(RMSNorm(x))        (各位置を処理する)
    3. 最後の RMSNorm
    4. 出力層           : ベクトル → 語彙の各トークンの点数。埋め込みと重みを共有する(重み共有)

サーバーで使う場合の注意: PyTorch だけで書いているので、Windows と Linux、CPU と CUDA で同じです。
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTENTION_MODES: tuple[str, ...] = ("fast", "manual")


@dataclass
class ModelConfig:
    """All shape settings of the model. Saved next to every checkpoint (Day 9)."""

    vocab_size: int = 1674          # from data/tokenizer/tokenizer.json (Day 6)
    block_size: int = 256           # longest sequence the model reads at once (tokens)
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    mlp_ratio: int = 4              # MLP hidden size = mlp_ratio * d_model
    dropout: float = 0.0
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    attention: str = "fast"         # "fast" or "manual" (same result, see tests)
    name: str = field(default="custom")

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if (self.d_model // self.n_head) % 2 != 0:
            raise ValueError("head size must be even for RoPE")
        if self.attention not in ATTENTION_MODES:
            raise ValueError(f"attention must be one of {ATTENTION_MODES}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**data)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: Path) -> "ModelConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# The three trial sizes for Day 11 (about 1M, 5M and 15M parameters with our 1,674-token vocabulary).
PRESETS: dict[str, dict] = {
    "tiny":   {"n_layer": 4, "n_head": 4, "d_model": 128},
    "small":  {"n_layer": 6, "n_head": 4, "d_model": 256},
    "medium": {"n_layer": 8, "n_head": 6, "d_model": 384},
}


def preset_config(name: str, vocab_size: int, block_size: int = 256, **overrides) -> ModelConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return ModelConfig(vocab_size=vocab_size, block_size=block_size, name=name, **{**PRESETS[name], **overrides})


# --------------------------------------------------------------------------- building blocks


class RMSNorm(nn.Module):
    """Scales each vector to a fixed average size (root mean square = 1), then by a learned gain.
    Computed in float32 for stability under bf16/fp16."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(x.dtype)


def rope_tables(head_dim: int, length: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin tables of shape (length, head_dim/2) for rotary position embedding."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(length, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (x[2i], x[2i+1]) of each vector by an angle that grows with the position.
    x: (batch, heads, T, head_dim); cos/sin: (T, head_dim/2).
    After rotation, the dot product of a query and a key depends only on how far apart they are."""
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    c, s = cos[None, None], sin[None, None]
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)
    return out.to(x.dtype)


class CausalSelfAttention(nn.Module):
    """Each position mixes information from itself and EARLIER positions only."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head, self.head_dim, self.mode = cfg.n_head, cfg.head_dim, cfg.attention
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.proj.is_residual_out = True  # type: ignore[attr-defined]  # scaled init
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        # (b, t, d) -> (b, heads, t, head_dim)
        q, k, v = (z.view(b, t, self.n_head, self.head_dim).transpose(1, 2) for z in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.mode == "manual":
            y = self._manual(q, k, v)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.resid_drop(self.proj(y))

    def _manual(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """The same computation written out step by step (for reading and for the tests)."""
        t = q.size(-2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)        # (b, h, t, t)
        future = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))                      # hide later positions
        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)                 # each row sums to 1
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return weights @ v


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.d_model
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)
        self.down.is_residual_out = True  # type: ignore[attr-defined]
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.gelu(self.up(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


# --------------------------------------------------------------------------- the model


class ModelA(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight          # weight tying: one matrix, two uses
        cos, sin = rope_tables(cfg.head_dim, cfg.block_size, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            std = self.cfg.init_std
            if getattr(m, "is_residual_out", False):
                std /= math.sqrt(2 * self.cfg.n_layer)     # keeps the residual stream from growing with depth
            nn.init.normal_(m.weight, mean=0.0, std=std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    def set_attention(self, mode: str) -> None:
        """Switch every layer between the fast and the manual attention (same weights)."""
        if mode not in ATTENTION_MODES:
            raise ValueError(f"attention must be one of {ATTENTION_MODES}")
        self.cfg.attention = mode
        for blk in self.blocks:
            blk.attn.mode = mode

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                ignore_index: int = -100) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx: (batch, T) token numbers. Returns logits (batch, T, vocab) and, if targets are
        given, the average cross-entropy loss of predicting targets[:, t] at position t."""
        b, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError(f"sequence length {t} is longer than block_size {self.cfg.block_size}")
        cos, sin = self.rope_cos[:t], self.rope_sin[:t]
        x = self.drop(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=ignore_index)
        return logits, loss


# --------------------------------------------------------------------------- parameter counting


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Unique parameters (the tied matrix is counted once)."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():          # parameters() already yields shared tensors once
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    emb = model.tok_emb.weight.numel() if isinstance(model, ModelA) else 0
    return {"total": total, "embedding": emb, "non_embedding": total - emb}


def expected_parameters(cfg: ModelConfig) -> int:
    """The same count from the formula, written out (checked by a test)."""
    d, h = cfg.d_model, cfg.mlp_ratio * cfg.d_model
    per_block = (3 * d * d + d * d) + (d * h + h * d) + 2 * d   # attention + MLP + two norm gains
    return cfg.vocab_size * d + cfg.n_layer * per_block + d       # embedding (tied) + blocks + final norm
