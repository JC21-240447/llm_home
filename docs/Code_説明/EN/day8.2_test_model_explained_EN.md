# Day 8.2 — test_model.py — Simple Line-by-Line Explanation (English)

> **How the three Day 8 files fit together**
> - **Day 8.1 `model.py`** — the model (**read it first**).
> - **Day 8 `day8_model_check.py`** — the driver script.
> - **Day 8.2 (this file) `test_model.py`** — the *inspector*: 12 automatic tests that check the model is correct. They all run on the **CPU in a few seconds**.
>
> pytest basics (test functions, `assert`, `parametrize`, `tmp_path`, `pytest.raises`) are explained in Days 6.2 and 7.2; this file only explains what is new.

---

## 0. The big picture (read this first)

A model can have bugs that **do not crash anything**: it still runs, still produces numbers, but the numbers are wrong (for example it secretly peeks at the future, or the manual and fast attentions differ). These tests catch those silent bugs by checking **mathematical promises** of the model:

| Promise | Test |
|---|---|
| Output has the right shape, and a loss only when targets are given | `test_output_shapes_and_loss` |
| Before training the loss ≈ `ln(vocabulary)` | `test_initial_loss_is_close_to_uniform_guess` |
| **The future never changes the past** (causality), for both attention versions | `test_causality_future_token_does_not_change_the_past` (×2) |
| Manual attention = fast attention | `test_manual_and_fast_attention_agree` |
| Embedding and output matrix are really the same memory | `test_weight_tying` |
| Parameter count = the hand formula | `test_parameter_count_matches_formula` |
| The three presets are in the planned size range | `test_presets_are_in_the_planned_size_range` |
| RoPE depends only on **relative** position | `test_rope_depends_only_on_relative_position` |
| Too-long input is rejected | `test_too_long_input_is_rejected` |
| Config can be saved/loaded; bad config is rejected | `test_config_round_trip` |
| The model can memorise one small batch | `test_overfit_one_batch` |

Run (same as before):

```
.venv\Scripts\python.exe -m pytest tests -q
```

---

## 1. New words

| Word | Simple meaning |
|---|---|
| **`torch.randint(a, b, shape)`** | A tensor of random whole numbers from `a` to `b−1`. Here: random fake token IDs. |
| **`torch.allclose(a, b, atol=...)`** | `True` if two tensors are equal within a tiny tolerance (`atol`). Needed because decimal arithmetic is never exactly identical. |
| **`.eval()`** | Evaluation mode (dropout off, deterministic). |
| **`torch.no_grad()`** | Don't record gradients (faster; not needed for checking). |
| **`pytest.approx(x, abs=...)`** | "approximately equal to x" for a single number. |
| **`data_ptr()`** | The memory address of a tensor's data. Same address = the same memory. |
| **`ndim`** | Number of dimensions of a tensor (0 = a single number). |

---

## 2. Docstring, imports and setup

```python
"""Unit tests for src/model.py (Day 8). All run on the CPU in a few seconds.
...
    .venv\\Scripts\\python.exe -m pytest tests -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model import (  # noqa: E402
    ATTENTION_MODES, ModelA, ModelConfig, apply_rope, count_parameters, expected_parameters, preset_config, rope_tables,
)
```
- Same style as Days 6.2/7.2 (non-raw docstring so backslashes are doubled; `REPO_ROOT`/`sys.path` trick). New: `math` and `torch`. Imports everything from the model file that the tests need.

---

## 3. Helpers

```python
VOCAB = 50


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=VOCAB, block_size=32, n_layer=2, n_head=4, d_model=64)
    return ModelConfig(**{**base, **kw})


def make_model(seed: int = 0, **kw) -> ModelA:
    torch.manual_seed(seed)
    return ModelA(tiny_cfg(**kw)).eval()
```
- `VOCAB = 50` — a **tiny pretend vocabulary** so tests are fast.
- `tiny_cfg(**kw)` — a **very small config** (2 layers, 4 heads, width 64, block size 32). Extra keyword arguments **override** the defaults (`{**base, **kw}`, the same merging trick as `preset_config`).
- `make_model(seed, **kw)` — fix the random seed, build the small model and put it in **eval mode**. Fixed seed = same weights every run → reproducible tests.

---

## 4. The tests

### 4.1 `test_output_shapes_and_loss`

```python
def test_output_shapes_and_loss() -> None:
    m = make_model()
    idx = torch.randint(0, VOCAB, (3, 10))
    logits, loss = m(idx, idx)
    assert logits.shape == (3, 10, VOCAB)
    assert loss is not None and loss.ndim == 0
    assert m(idx)[1] is None
```
- 3 random sequences of 10 token IDs. Run the model with targets (here simply the same tokens).
- `logits.shape` must be `(3, 10, VOCAB)`: **a score for every token at every position**.
- The loss must exist and be a **single number** (`ndim == 0`).
- Without targets (`m(idx)`), element `[1]` (the loss) must be `None`.

### 4.2 `test_initial_loss_is_close_to_uniform_guess`

```python
def test_initial_loss_is_close_to_uniform_guess() -> None:
    """Before training the model knows nothing: loss should be about ln(vocab size)."""
    m = make_model()
    idx = torch.randint(0, VOCAB, (8, 32))
    _, loss = m(idx, torch.randint(0, VOCAB, (8, 32)))
    assert abs(loss.item() - math.log(VOCAB)) < 0.3
```
- Random inputs **and random targets**. An untrained model should be a uniform guesser, so the loss must be within 0.3 of `ln(50) ≈ 3.91`. (This is the small-scale version of Day 8's check; it catches bad initialisation or wrong wiring.)

### 4.3 `test_causality_future_token_does_not_change_the_past` (runs twice: `fast` and `manual`)

```python
@pytest.mark.parametrize("mode", ATTENTION_MODES)
def test_causality_future_token_does_not_change_the_past(mode: str) -> None:
    m = make_model(attention=mode)
    idx = torch.randint(0, VOCAB, (2, 20))
    changed = idx.clone()
    k = 12
    changed[:, k] = (changed[:, k] + 1) % VOCAB
    with torch.no_grad():
        a, _ = m(idx)
        b, _ = m(changed)
    assert torch.allclose(a[:, :k], b[:, :k], atol=1e-6)        # earlier positions: identical
    assert not torch.allclose(a[:, k:], b[:, k:], atol=1e-4)    # position k and later: changed
```
- **The most important test.** It proves the model **never uses the future**.
- `parametrize("mode", ATTENTION_MODES)` — run the test once for each attention version (`fast`, `manual`).
- Take 2 random sequences of 20 tokens, **copy** them (`idx.clone()`), and change **one token at position 12** in the copy (`+1`, wrapping round with `% VOCAB`).
- Run both through the model (`a`, `b` = the logits).
- Check 1: for positions **before 12** (`[:, :k]`) the outputs must be **identical** (within `1e-6`) — changing a later token must not affect earlier outputs. If it did, the model would be peeking into the future.
- Check 2: from position 12 onward (`[:, k:]`) the outputs must **differ** — otherwise the test would be meaningless (the change must actually matter where it is allowed to).

### 4.4 `test_manual_and_fast_attention_agree`

```python
def test_manual_and_fast_attention_agree() -> None:
    m = make_model()
    idx = torch.randint(0, VOCAB, (2, 32))
    with torch.no_grad():
        m.set_attention("fast")
        fast, _ = m(idx)
        m.set_attention("manual")
        manual, _ = m(idx)
    assert torch.allclose(fast, manual, atol=1e-5)
```
- Same weights, same input, switch the attention version with `set_attention` and compare the outputs: they must agree within `1e-5`. This proves the **readable manual code** and PyTorch's **fast code** compute the same thing.

### 4.5 `test_weight_tying`

```python
def test_weight_tying() -> None:
    m = make_model()
    assert m.lm_head.weight is m.tok_emb.weight
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()
```
- The output layer's weight and the embedding table must be **the same object** (`is`) and live at the **same memory address** (`data_ptr()`). So the matrix is stored once, not copied.

### 4.6 `test_parameter_count_matches_formula`

```python
def test_parameter_count_matches_formula() -> None:
    for cfg in (tiny_cfg(), preset_config("tiny", 1674), preset_config("small", 1674)):
        m = ModelA(cfg)
        counts = count_parameters(m)
        assert counts["total"] == expected_parameters(cfg)
        assert counts["embedding"] == cfg.vocab_size * cfg.d_model
```
- For three configurations (the small test config, preset tiny, preset small), build the model and check that the **counted** parameters equal the **hand formula** (Day 8.1 §7), and that the embedding count equals `vocab_size × d_model`. (The medium preset is skipped to keep the test fast.)

### 4.7 `test_presets_are_in_the_planned_size_range`

```python
def test_presets_are_in_the_planned_size_range() -> None:
    sizes = {name: expected_parameters(preset_config(name, 1674)) for name in ("tiny", "small", "medium")}
    assert 0.5e6 < sizes["tiny"] < 2e6
    assert 3e6 < sizes["small"] < 8e6
    assert 10e6 < sizes["medium"] < 20e6
```
- Using only the **formula** (no model is built, so it is instant), check that each preset's size is in the planned range: tiny 0.5–2 million, small 3–8 million, medium 10–20 million. (`0.5e6` = 500,000.) With 1,674 tokens they are about 1.0M, 5.15M and 14.8M, inside the ranges. This protects against someone accidentally editing `PRESETS` and breaking the plan for Day 11.

### 4.8 `test_rope_depends_only_on_relative_position`

```python
def test_rope_depends_only_on_relative_position() -> None:
    """Rotary embedding: q·k is the same when both positions shift by the same amount."""
    torch.manual_seed(1)
    cos, sin = rope_tables(16, 64, 10000.0)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(pq: int, pk: int) -> float:
        rq = apply_rope(q, cos[pq:pq + 1], sin[pq:pq + 1])
        rk = apply_rope(k, cos[pk:pk + 1], sin[pk:pk + 1])
        return float((rq * rk).sum())

    assert score(10, 3) == pytest.approx(score(40, 33), abs=1e-4)
    assert score(10, 3) != pytest.approx(score(10, 9), abs=1e-3)
```
- Tests the key RoPE property from Day 8.1 §5.2.
- Build rotation tables for head size 16 and 64 positions. Make one random query `q` and one random key `k` (each a vector of 16 numbers; shape `(1,1,1,16)`).
- `score(pq, pk)` — rotate `q` as if it sits at position `pq` (taking just that one row of the tables, `cos[pq:pq+1]`), rotate `k` as if at position `pk`, and return their **dot product** (`(rq * rk).sum()`).
- Check 1: positions (10, 3) and (40, 33) have the **same distance (7)** → the scores must be **equal** (within `1e-4`).
- Check 2: positions (10, 3) (distance 7) and (10, 9) (distance **1**) → the scores must **differ**. So the score depends on the distance, and not on anything else.
- `pytest.approx(value, abs=...)` makes `==` / `!=` mean "approximately equal / not approximately equal".

### 4.9 `test_too_long_input_is_rejected`

```python
def test_too_long_input_is_rejected() -> None:
    m = make_model()
    with pytest.raises(ValueError):
        m(torch.zeros(1, 33, dtype=torch.long))
```
- The test model has `block_size = 32`. A sequence of **33** tokens (`torch.zeros(1, 33, dtype=torch.long)` = 33 zeros as integers) must raise `ValueError` (Day 8.1's check in `forward`).

### 4.10 `test_config_round_trip`

```python
def test_config_round_trip(tmp_path: Path) -> None:
    cfg = preset_config("small", 1674, dropout=0.1)
    path = tmp_path / "config.json"
    cfg.save(path)
    assert ModelConfig.load(path) == cfg
    with pytest.raises(ValueError):
        ModelConfig(d_model=100, n_head=3)
```
- Make a config (preset small with `dropout=0.1` overridden), save it to a temp file, load it back: it must **equal** the original (dataclass equality compares every field).
- And a **bad** config must be rejected: `d_model=100` is not divisible by `n_head=3` → `ValueError` (from `__post_init__`).

### 4.11 `test_overfit_one_batch`

```python
def test_overfit_one_batch() -> None:
    """A correct model and loss must be able to memorise one small batch almost perfectly."""
    torch.manual_seed(0)
    m = ModelA(tiny_cfg()).train()
    idx = torch.randint(0, VOCAB, (4, 17))
    x, y = idx[:, :-1], idx[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    first = None
    for _ in range(300):
        _, loss = m(x, y)
        first = first if first is not None else loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert first > 3.0
    assert loss.item() < 0.1
```
- The small-scale version of Day 8's memorisation check, on **random fake data**: 4 sequences of 17 random tokens, with `x` = first 16 and `y` = shifted by one.
- Train in **train mode** with AdamW for 300 steps (the same loop as Day 8).
- `first = first if first is not None else loss.item()` — remember the **first** loss only (a compact way to set it once).
- Pass only if the first loss is **above 3.0** (starts near `ln 50 ≈ 3.9`, i.e. it really knew nothing) and the last loss is **below 0.1** (memorised). Random data cannot be "understood", so the only way to get a low loss is to memorise — which proves gradients, loss and optimizer all work.

---

## 5. How many tests?

| Test | Runs |
|---|---|
| shapes and loss | 1 |
| initial loss | 1 |
| causality | 2 (fast, manual) |
| manual vs fast | 1 |
| weight tying | 1 |
| parameter count | 1 |
| preset size range | 1 |
| RoPE relative position | 1 |
| too-long input | 1 |
| config round trip | 1 |
| overfit one batch | 1 |
| **Total** | **12** |

Expected result: `12 passed` (this file has no "skip" tests because everything runs without real data).

## 6. What these tests do *not* cover

- **Training on real data** and generation (Days 9–10).
- **GPU / bf16 / fp16 behaviour** — all tests run on the CPU in fp32 (Day 8's script covers the GPU part).
- `dropout > 0` behaviour (train vs eval differences).
- The `ignore_index` option of the loss (for padding, used later).
- The medium preset (checked by Day 8 itself, and by the size-range test through the formula).

## 7. Quick summary in one paragraph

These 12 fast CPU tests check that Model A gives outputs of the right shape, starts at the uniform-guess loss, **never lets the future influence the past** (for both attention versions), that the manual and fast attention agree, that weights are really tied, that the parameter count matches the formula and the presets stay in their planned size range, that RoPE depends only on relative position, that over-long input and invalid configs are rejected, that configs save and load exactly, and that the model can memorise one small batch.
