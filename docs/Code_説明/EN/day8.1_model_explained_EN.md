# Day 8.1 — model.py — Simple Line-by-Line Explanation (English)

> **How the three Day 8 files fit together**
> - **Day 8.1 (this file) `model.py`** — the *model itself*: Model A, a small GPT-style Transformer written from scratch with PyTorch.
> - **Day 8 `day8_model_check.py`** — the *driver*: builds the model at three sizes, times it, checks it on real data, saves the configs and writes a report.
> - **Day 8.2 `test_model.py`** — the *inspector*: automatic tests for the model.
>
> Suggested reading order: **8.1 → 8 → 8.2**. This is the most important file of the project so far; take your time with section 0.

---

## 0. The big picture (read this first)

Model A is a **text-continuation machine**. You give it a row of token numbers (from the Day 6/7 tokenizer) and, **at every position**, it answers one question:

> "Given everything I have read so far, **how likely is each token in the vocabulary to come next?**"

It answers with a list of **scores (called logits)**, one score per vocabulary token. Training (Day 9) teaches it to give the *real* next token a high score. Generation (Day 10) turns scores into text.

### The journey of the numbers

```
token IDs  (batch, T)                 e.g. [1, 412, 380, ...]
   │  1. Token embedding               each ID → a vector of d_model numbers (its "ID card")
   ▼
vectors    (batch, T, d_model)
   │  2. N × Block                     each Block =  look at earlier tokens (Attention)
   │                                                 + think about each position (MLP)
   ▼
vectors    (batch, T, d_model)
   │  3. Final RMSNorm                 tidy up the sizes
   │  4. Output layer                  vector → one score per vocabulary token
   ▼
logits     (batch, T, vocab_size)
```

### A simple analogy

- **Embedding** = every token has a **coordinate card** (a list of numbers) describing it.
- **Attention** = like a reader who, at each word, **glances back at earlier words** and decides which ones matter right now (it can **never peek ahead**).
- **MLP** = each position **thinks privately** about what it has gathered.
- **Residual connection** (`x = x + ...`) = a **conveyor belt**: every layer only *adds notes* to what is already on the belt, it does not throw the belt away.
- **Normalisation** = keep the numbers at a sensible size so nothing explodes.
- **RoPE** = a way to tell the model **how far apart** two tokens are, by *rotating* their vectors.
- **Weight tying** = the **same dictionary** is used for reading tokens in and writing scores out.

### The pieces in this file

| Piece | What it is |
|---|---|
| `ModelConfig` + `PRESETS` + `preset_config` | The settings (sizes) and the three trial sizes (tiny / small / medium) |
| `RMSNorm` | Keeps vector sizes under control |
| `rope_tables`, `apply_rope` | Position information by rotation |
| `CausalSelfAttention` | "Look at earlier tokens" (manual version + fast version) |
| `MLP` | "Think about each position" |
| `Block` | One attention + one MLP, wrapped with norms and residuals |
| `ModelA` | The whole model |
| `count_parameters`, `expected_parameters` | Count the learnable numbers (two ways, to cross-check) |

---

## 1. Words you need to know (mini dictionary)

| Word | Simple meaning |
|---|---|
| **Transformer** | The neural-network design used by GPT-style models. |
| **Decoder-only / GPT style** | The model only reads left-to-right and predicts the **next** token. |
| **Embedding** | A table that turns a token ID into a vector (list of numbers). |
| **Vector / `d_model`** | A list of numbers; `d_model` is how long it is (the model's "width"). |
| **Parameter** | A learnable number inside the model. "1M parameters" = one million such numbers. |
| **Logit** | A raw score. Higher = the model thinks that token is more likely next. |
| **Softmax** | Turns scores into probabilities that add up to 1. |
| **Cross-entropy loss** | How "surprised" the model is by the real next token: `−log(probability given to it)`. Lower = better. A model that knows nothing (uniform guess) has loss `ln(vocab size)`. |
| **Attention** | Each position mixes information from other positions, weighted by relevance. |
| **Query / Key / Value (q, k, v)** | q = "what am I looking for?", k = "what do I offer?", v = "what I hand over if chosen". The match score q·k decides how much of v is taken. |
| **Head / `n_head`** | Attention is done several times in parallel (several "heads"), each looking for different things. |
| **Causal mask** | A rule that hides **later** positions, so a position can only see itself and earlier ones. |
| **MLP** | A small two-layer network applied to each position separately. |
| **GELU** | A smooth "on/off" function used between the MLP's two layers. |
| **Residual connection** | `x = x + f(x)`: keep the input and add the layer's result. |
| **RMSNorm** | Rescales each vector so its average size is 1 (then a learned gain). |
| **RoPE** | Rotary position embedding (see section 5.2). |
| **Weight tying** | Embedding matrix and output matrix are the **same** matrix. |
| **Bias** | An extra constant added in a linear layer. This model uses none (`bias=False`). |
| **`block_size`** | The longest sequence the model reads at once (tokens). |
| **Buffer** | A tensor that belongs to the model (moves with it to the GPU) but is **not learned**. |

---

## 2. The docstring

- Same style as before (English + Japanese). It explains what the model does and the building blocks from input to output, then the **design choices**: RoPE (no learned position table, so no hard link to one maximum length), causal attention in two versions (readable "manual" and fast), RMSNorm, no biases, GELU MLP with 4× width, weight tying, and **scaled initialisation** of the residual projections (`std / sqrt(2 × n_layer)`).
- "SERVER NOTE": pure PyTorch → identical on Windows and Linux, CPU and GPU.

---

## 3. Imports and the attention modes

```python
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTENTION_MODES: tuple[str, ...] = ("fast", "manual")
```
- `from __future__`, `json`, `math`, `dataclass`/`asdict`/`field`, `Path` — as in earlier days.
- **New:** `torch` (PyTorch, the deep-learning library), `torch.nn as nn` (building blocks for networks: `Linear`, `Embedding`, `Module`, ...), `torch.nn.functional as F` (functions without learnable parts: `softmax`, `gelu`, `cross_entropy`, ...).
- `ATTENTION_MODES` — the two allowed attention versions.

---

## 4. `ModelConfig` — all the settings

```python
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
```

| Setting | Meaning |
|---|---|
| `vocab_size` | Number of tokens (from Day 6). Default 1674 is only a placeholder. |
| `block_size` | Longest input (256 tokens). |
| `n_layer` | Number of Blocks stacked (depth). |
| `n_head` | Number of attention heads. |
| `d_model` | Vector width. |
| `mlp_ratio` | The MLP's hidden layer is `mlp_ratio × d_model` wide (4×). |
| `dropout` | Randomly switching off numbers during training to reduce over-fitting; `0.0` = off. |
| `rope_base` | A constant that sets the rotation speeds in RoPE (10000, the usual value). |
| `norm_eps` | A tiny number added in RMSNorm to avoid dividing by zero. |
| `init_std` | Standard deviation of the random starting weights (0.02). |
| `attention` | `"fast"` or `"manual"`. |
| `name` | Label (`tiny`, `small`, ..., or `custom`). |

```python
    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if (self.d_model // self.n_head) % 2 != 0:
            raise ValueError("head size must be even for RoPE")
        if self.attention not in ATTENTION_MODES:
            raise ValueError(f"attention must be one of {ATTENTION_MODES}")
```
- Runs right after creation (as in Day 6.1) and **rejects impossible settings**:
  - `d_model` must split evenly into heads (each head gets `d_model / n_head` numbers);
  - the **head size must be even**, because RoPE rotates numbers **in pairs**;
  - `attention` must be a known mode.

```python
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head
```
- `head_dim` = how many numbers each head works with (`d_model ÷ n_head`). `//` = whole-number division.

```python
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
```
- `to_dict` / `from_dict` — convert to and from a plain dictionary (`**data` unpacks it into settings, as in Day 7.1's `read_meta`).
- `save` / `load` — write/read a JSON file (LF line endings, as before). These are the `configs/model_*.json` files that Day 9 and Day 11 will read.

### The three trial sizes

```python
PRESETS: dict[str, dict] = {
    "tiny":   {"n_layer": 4, "n_head": 4, "d_model": 128},
    "small":  {"n_layer": 6, "n_head": 4, "d_model": 256},
    "medium": {"n_layer": 8, "n_head": 6, "d_model": 384},
}
```
- Three candidate model sizes for **Day 11** (which one to use is decided there, by validation loss).
- Using the formula in section 7 with a 1,674-token vocabulary, the parameter counts are about:

| Preset | Layers | Heads | Width | Head size | Parameters (≈) |
|---|---|---|---|---|---|
| tiny | 4 | 4 | 128 | 32 | **1.00 million** |
| small | 6 | 4 | 256 | 64 | **5.15 million** |
| medium | 8 | 6 | 384 | 64 | **14.8 million** |

(My calculation from `expected_parameters`; Day 8's report prints the exact numbers for your real vocabulary.)

```python
def preset_config(name: str, vocab_size: int, block_size: int = 256, **overrides) -> ModelConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return ModelConfig(vocab_size=vocab_size, block_size=block_size, name=name, **{**PRESETS[name], **overrides})
```
- Builds a config from a preset name. Unknown name → clear error.
- `{**PRESETS[name], **overrides}` merges two dictionaries; the **overrides win**, so `preset_config("small", 1674, dropout=0.1)` changes only dropout. `**` then passes the merged settings as keyword arguments.

---

## 5. Building blocks

### 5.1 `RMSNorm`

```python
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
```
- Every layer in PyTorch is a class that **inherits** from `nn.Module`. `super().__init__()` must be called first. `forward` is the function that does the work when you call the layer.
- `nn.Parameter(torch.ones(dim))` — a **learnable** list of `dim` numbers, starting at 1 (the "gain").
- `forward`:
  1. `x32 = x.float()` — work in 32-bit for accuracy (the model may run in 16-bit).
  2. `x32.pow(2).mean(dim=-1, keepdim=True)` — the **mean of the squares** over the last dimension (each vector separately). `rsqrt` = 1/√. So `normed = x / sqrt(mean(x²) + eps)` — each vector now has an average size (root mean square) of **1**.
  3. Multiply by the learned gain, and convert back to the original number type (`.to(x.dtype)`).
- Why? Without it, numbers can grow or shrink layer after layer and training becomes unstable.

### 5.2 RoPE: `rope_tables` and `apply_rope`

The model must know **the order** of tokens ("A before B" differs from "B before A"). RoPE does it by **rotating** each token's query and key vectors by an angle that depends on the position.

```python
def rope_tables(head_dim: int, length: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin tables of shape (length, head_dim/2) for rotary position embedding."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(length, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()
```
- Prepares the rotation angles **once**.
- `torch.arange(0, head_dim, 2)` = 0, 2, 4, … (one entry per **pair** of numbers). `inv_freq` = a list of `head_dim/2` rotation speeds, from fast to slow (`1 / base^(2i/head_dim)`).
- `torch.arange(length)` = positions 0, 1, 2, …. `torch.outer(positions, inv_freq)` multiplies every position by every speed → a table `angles` of shape `(length, head_dim/2)`: **the angle for each position and each pair**.
- Return the cosines and sines of those angles.

```python
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    c, s = cos[None, None], sin[None, None]
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)
    return out.to(x.dtype)
```
- `x` has shape `(batch, heads, T, head_dim)`.
- `x[..., 0::2]` takes the **even-numbered** entries and `x[..., 1::2]` the **odd-numbered** ones — i.e. the pairs `(x[2i], x[2i+1])`.
- `cos[None, None]` adds two size-1 dimensions so the `(T, head_dim/2)` table lines up with `(batch, heads, T, head_dim/2)`.
- The formula `(x1·c − x2·s, x1·s + x2·c)` is the standard **2-D rotation** of each pair by its angle.
- `torch.stack((..., ...), dim=-1).flatten(-2)` puts the rotated pairs back into the original interleaved layout.
- **Key property (the docstring says it):** after rotation, the dot product of a query and a key depends **only on how far apart** the two positions are — the model learns "3 tokens back", not "absolute position 17". Day 8.2 tests this.

### 5.3 `CausalSelfAttention`

```python
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
```
- Remember the settings (heads, head size, mode).
- `self.qkv` — **one** linear layer that produces queries, keys and values **together** (`3 × d_model` outputs; split later). `nn.Linear(in, out, bias=False)` = multiply by a learnable matrix.
- `self.proj` — a final linear layer that mixes the heads' results back to `d_model`.
- `self.proj.is_residual_out = True` — a **sticky note** on this layer ("I write into the residual stream"). `ModelA._init_weights` reads it to shrink the starting weights. (`# type: ignore` just silences the type checker.)
- `self.resid_drop` — dropout (does nothing when `dropout = 0.0`).

```python
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q, k, v = (z.view(b, t, self.n_head, self.head_dim).transpose(1, 2) for z in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
```
- `b, t, d` = batch size, sequence length, width.
- `self.qkv(x)` → `(b, t, 3d)`; `.split(d, dim=2)` cuts the last dimension into **3 pieces of size d** = q, k, v.
- `view(b, t, n_head, head_dim).transpose(1, 2)` cuts each width-`d` vector into `n_head` pieces of `head_dim` and moves the heads dimension forward → shape `(b, heads, t, head_dim)`. (The same line is applied to q, k, v with a generator.)
- RoPE is applied to **q and k only** (positions matter for matching; the value v just carries content).

```python
        if self.mode == "manual":
            y = self._manual(q, k, v)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.resid_drop(self.proj(y))
```
- Two ways to compute the same thing:
  - `"manual"`: the readable step-by-step version `_manual` below;
  - otherwise PyTorch's **fast built-in** `scaled_dot_product_attention` with `is_causal=True` (the causal mask is built in). Dropout only while training (`self.training`).
- Then put the heads back together: move the heads dimension back, `.contiguous()` (make the memory layout tidy), `.view(b, t, d)`, apply `proj` and dropout.

```python
    def _manual(self, q, k, v):
        t = q.size(-2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)        # (b, h, t, t)
        future = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))                      # hide later positions
        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)                 # each row sums to 1
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return weights @ v
```
The attention recipe, step by step:
1. `scores = q @ kᵀ / √head_dim` — for every pair of positions (i, j): "how well does position i's **query** match position j's **key**?" `@` is matrix multiplication; the result is a `(t, t)` table per head. Dividing by `√head_dim` keeps the numbers from getting too large.
2. `future = torch.triu(ones(t, t, bool), diagonal=1)` — a table that is `True` **above the diagonal** (= j later than i = the future).
3. `masked_fill(future, -inf)` — set future scores to **minus infinity**, so after softmax they get **weight 0**. This is the **causal mask**: no peeking ahead.
4. `softmax(..., dim=-1)` — turn each row of scores into weights that **add up to 1** (computed in float32, then converted back).
5. Dropout on the weights (off if `dropout = 0`).
6. `weights @ v` — each position takes a **weighted average of the values** of the positions it attends to.

### 5.4 `MLP`

```python
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
```
- A small network applied **to each position separately**: widen (`up`: d → 4d), apply **GELU**, narrow back (`down`: 4d → d), dropout. (Same sticky note on `down`.)
- Attention moves information **between** positions; the MLP **processes** it inside each position.

### 5.5 `Block`

```python
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
```
- One Transformer layer: two norms, an attention, an MLP.
- `forward` = the two lines from the docstring:
  - `x = x + Attention(RMSNorm(x))` — normalise, attend, **add the result to x** (residual);
  - `x = x + MLP(RMSNorm(x))` — normalise, process, add.
- "**Pre-norm**" = the norm is applied *before* each sub-layer. The residual `x + ...` keeps the original information flowing straight through all layers (the conveyor belt), which makes deep models trainable.

---

## 6. `ModelA` — the whole model

```python
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
```
- `tok_emb` — the embedding table: `vocab_size` rows × `d_model` columns; row `i` is the vector for token `i`.
- `blocks` — `n_layer` Blocks in a `ModuleList` (a list PyTorch knows about, so their parameters are found).
- `norm_f` — the final RMSNorm.
- `lm_head` — the **output layer**: vector (`d_model`) → one score per vocabulary token (`vocab_size`). Its weight matrix has shape `(vocab_size, d_model)` — **exactly the same shape as the embedding table**.
- `self.lm_head.weight = self.tok_emb.weight` — **weight tying**: the *same* matrix is used to read tokens in and write scores out. Benefits: far fewer parameters (a big saving when the vocabulary is large relative to `d_model`) and a natural link between "what a token looks like" and "how likely it is to be output".
- `register_buffer(..., persistent=False)` — store the RoPE tables as **buffers**: they move with the model to the GPU but are not learned, and `persistent=False` means they are **not saved** in checkpoints (they can always be recomputed).
- `self.apply(self._init_weights)` — run the initialisation function on every sub-layer.

```python
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            std = self.cfg.init_std
            if getattr(m, "is_residual_out", False):
                std /= math.sqrt(2 * self.cfg.n_layer)     # keeps the residual stream from growing with depth
            nn.init.normal_(m.weight, mean=0.0, std=std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
```
- Give every weight **small random starting values** (normal distribution, mean 0, std 0.02).
- For layers carrying the "sticky note" (`is_residual_out`: attention `proj` and MLP `down`), make the std **smaller by `√(2 × n_layer)`**. Each Block adds two contributions to the conveyor belt, so a deeper model would otherwise have a belt whose values grow with depth. This keeps the start stable.
- `getattr(m, "is_residual_out", False)` reads the sticky note, or `False` if there is none.
- (RMSNorm gains stay at 1; the tied matrix is initialised with the same std either way.)
- Small starting weights are also why the **initial loss is close to `ln(vocab)`**: the starting scores are nearly all equal, i.e. a uniform guess (Day 8 checks this).

```python
    def set_attention(self, mode: str) -> None:
        if mode not in ATTENTION_MODES:
            raise ValueError(f"attention must be one of {ATTENTION_MODES}")
        self.cfg.attention = mode
        for blk in self.blocks:
            blk.attn.mode = mode
```
- Switch every layer between `fast` and `manual` attention **with the same weights**. Used by the tests and Day 8 to prove the two agree.

```python
    def forward(self, idx, targets=None, ignore_index=-100):
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
```
- Input `idx`: token IDs, shape `(batch, T)`.
- Reject sequences longer than `block_size`.
- Take the first `t` rows of the RoPE tables (`[:t]`).
- `self.tok_emb(idx)` — look up each ID's vector → `(batch, T, d_model)`; optional dropout.
- Pass through every Block in order (`for blk in self.blocks`).
- Final norm, then the output layer → `logits` of shape `(batch, T, vocab_size)`: for **every position** a score for **every token**.

```python
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=ignore_index)
        return logits, loss
```
- If `targets` (the correct next tokens) are given, compute the **loss**:
  - `logits.view(-1, vocab)` flattens `(batch, T, vocab)` into one long list of rows (one per position); `targets.reshape(-1)` flattens the targets the same way;
  - `F.cross_entropy` applies softmax and takes `−log(probability of the correct token)` for each row, then averages. (`.float()` = computed in 32-bit.)
  - `ignore_index=-100`: positions whose target is `-100` are **skipped** (for padding later).
- Return both. Without targets, `loss` is `None` (used for generation).

**How training uses it (Day 9):** the input is tokens `0..T-1` (`x`), the targets are tokens `1..T` (`y`, shifted by one), so at position `t` the model is asked to predict token `t+1`.

---

## 7. Counting parameters

```python
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
```
- Counts the learnable numbers. `p.numel()` = number of elements in the tensor. `id(p)` is the object's identity, so a shared (tied) tensor would not be counted twice (PyTorch's `parameters()` already lists it once; the check is a safety).
- `embedding` = the embedding matrix size (counted once even though used twice); `non_embedding` = everything else.

```python
def expected_parameters(cfg: ModelConfig) -> int:
    """The same count from the formula, written out (checked by a test)."""
    d, h = cfg.d_model, cfg.mlp_ratio * cfg.d_model
    per_block = (3 * d * d + d * d) + (d * h + h * d) + 2 * d   # attention + MLP + two norm gains
    return cfg.vocab_size * d + cfg.n_layer * per_block + d       # embedding (tied) + blocks + final norm
```
- The same number, **calculated by hand-written formula** (a cross-check):
  - attention: `qkv` = `d × 3d`, `proj` = `d × d` → `3d² + d²`;
  - MLP: `up` = `d × h`, `down` = `h × d`, with `h = 4d`;
  - two RMSNorm gains = `2d`;
  - plus the embedding `vocab × d` (once), `n_layer` blocks, and the final norm `d`.
- With `h = 4d` this is `12d² + 2d` per block. No biases and no learned position table, so nothing else is counted.
- Day 8 and Day 8.2 both check that `count_parameters(model) == expected_parameters(cfg)`: if the code and the hand formula agree, it is very likely that nothing is missing or doubled.

---

## 8. Shapes at a glance (batch 32, T 256, tiny, vocab 1,674)

| Step | Shape |
|---|---|
| input IDs | (32, 256) |
| after embedding | (32, 256, 128) |
| inside attention: q, k, v | (32, 4, 256, 32) each |
| attention score table | (32, 4, 256, 256) per layer |
| after 4 blocks and norm | (32, 256, 128) |
| logits | (32, 256, 1674) |
| loss | one number |

## 9. Small things worth knowing

- **Everything is a straight line of matrix multiplications** plus a few small functions; no recurrence.
- **Attention cost grows with `T²`** (the score table is `T × T`), which is why `block_size` is kept at 256.
- **Dropout is 0** by default, so training and evaluation behave the same; the `.train()` / `.eval()` switch matters only if you turn dropout on.
- **`decode`-style checks of IDs:** the model does not check that token IDs are `< vocab_size`; an ID that is too large would raise a PyTorch indexing error.
- **Tied weights and checkpoints:** the output matrix and the embedding are one tensor; this is why the parameter count includes it only once.
- **The default `vocab_size=1674` in `ModelConfig` is a placeholder**; real runs take the vocabulary from the tokenizer (Day 8 does this).

## 10. Quick summary in one paragraph

`model.py` defines Model A: token IDs become vectors; several Blocks each let every position look at earlier positions (causal attention with rotary position information) and then think about what it gathered (MLP), with normalisation and residual connections; a final layer, tied to the embedding, outputs a score for every vocabulary token at every position; the file also stores the settings and the three trial sizes (tiny/small/medium) and counts parameters two ways to cross-check.
