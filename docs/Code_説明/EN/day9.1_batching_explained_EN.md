# Day 9.1 — batching.py — Simple Line-by-Line Explanation (English)

> **How the five Day 9 files fit together**
> - **Day 9.1 (this file) `batching.py`** — cuts **training and validation batches** out of the Day 7 token files.
> - **Day 9.2 `training.py`** — the **training loop** (the heart of Day 9). It uses this file.
> - **Day 9 `day9_train.py`** — the command-line script that starts/resumes a training run.
> - **Day 9 `day9_compare_runs.py`** — compares an uninterrupted run with a stopped-and-resumed run.
> - **Day 9.3 `test_training.py`** — automatic tests for 9.1 and 9.2.
>
> Suggested reading order: **9.1 → 9.2 → 9 (train) → 9 (compare) → 9.3**.

---

## 0. The big picture (read this first)

The model learns by playing one game millions of times: **"here is some text, predict the next token at every position."** This file prepares the **practice sheets** for that game.

Remember the Day 7 token stream: `<bos> question <eos> <bos> question <eos> …` (one very long row of numbers).

### One training example = one WINDOW

Cut `block_size + 1` consecutive tokens from the stream (say 257 tokens for block size 256):

```
window :  t0  t1  t2  t3  ...  t255  t256
x      :  t0  t1  t2  t3  ...  t255            (what the model reads)
y      :      t1  t2  t3  ...  t255  t256      (the next token at every position)
```
At position `t0` the model must predict `t1`; at `t1` it must predict `t2`; and so on. So **one window gives 256 "next-token" exercises at once.** (Same idea as Day 8's `real_batch`.)

### Two kinds of batches

| Kind | Where windows start | Why |
|---|---|---|
| **Training** | **Random** positions | Every step sees different windows → the model does not memorise an order |
| **Validation** | **Fixed**, side by side, covering the whole stream | Every evaluation measures exactly the same thing, so the numbers are comparable over time |

### Why the random generator's state must be saved

If a run is stopped and resumed, it must continue with **exactly the batches the uninterrupted run would have used**. So the random generator's internal state is saved in every checkpoint (`state_dict` / `load_state_dict`).

---

## 1. Words you need to know

| Word | Simple meaning |
|---|---|
| **Window** | A short piece cut from the token stream (`block_size + 1` tokens). |
| **Batch** | Several windows processed together (`batch_size`). |
| **Memory map** | The file is opened without loading it into RAM; only the windows used are read (Day 7.1). |
| **RNG (random number generator) state** | The generator's internal "position in its sequence". Saving and restoring it makes the next random numbers **repeat exactly**. |
| **Generator function / `yield`** | A function that hands out results **one at a time** instead of building a whole list. |

---

## 2. The docstring and imports

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from src.token_data import TOKEN_DTYPE, read_meta
```
- `Path`, `numpy`, `torch` — as before. `Iterator` — a type meaning "something you can loop over one item at a time" (used for the generator below).
- From **Day 7.1**: `TOKEN_DTYPE` (little-endian `uint16`) and `read_meta` (reads the `*_meta.json` label).

---

## 3. `open_tokens`

```python
def open_tokens(tokens_dir: Path, split: str) -> np.ndarray:
    """The token stream of one split as a read-only memory map (checked against its meta file)."""
    meta = read_meta(tokens_dir, split)
    tokens = np.memmap(tokens_dir / f"{split}.bin", dtype=TOKEN_DTYPE, mode="r")
    if len(tokens) != meta.tokens:
        raise ValueError(f"{split}.bin does not match {split}_meta.json")
    return tokens
```
- Read the meta label of the split, open `train.bin` (or `val.bin`) as a **read-only memory map** with the exact Day 7.1 type, and check that the file has as many tokens as the label says (otherwise → `ValueError`).
- Compare Day 8, which opened the memory map directly: this is the **library version** with the safety check built in.

---

## 4. `RandomWindows`

```python
class RandomWindows:
    """Random training windows with a saveable random state."""

    def __init__(self, tokens: np.ndarray, block_size: int, batch_size: int, seed: int) -> None:
        if len(tokens) <= block_size + 1:
            raise ValueError("token stream is shorter than one window")
        self.tokens = tokens
        self.block_size = block_size
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
```
- A small class that hands out random training batches.
- Reject a stream that is too short for even one window.
- Store the stream, the sizes, and a **seeded NumPy random generator** `self.rng` (the same `default_rng` as Day 8).

```python
    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.rng.integers(0, len(self.tokens) - self.block_size - 1, size=self.batch_size)
        win = np.stack([np.asarray(self.tokens[s:s + self.block_size + 1], dtype=np.int64) for s in starts])
        w = torch.from_numpy(win)
        return w[:, :-1], w[:, 1:]
```
- **Exactly Day 8's `real_batch` logic**, now inside the class:
  1. pick `batch_size` random start positions;
  2. cut `block_size + 1` tokens from each start (also copying them out of the memory map as 64-bit integers);
  3. stack into a table `(batch, block_size+1)`, turn into a PyTorch tensor;
  4. return `x = w[:, :-1]` and `y = w[:, 1:]`.
- Each call uses up some of the generator's randomness, so **the next call gives different windows**.
- (Windows may cross from one question into the next, which is fine.)

```python
    def state_dict(self) -> dict:
        return {"bit_generator": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state["bit_generator"]
```
- `state_dict` — returns the generator's **internal state** (a dictionary of plain numbers) so it can be saved in a checkpoint.
- `load_state_dict` — puts a saved state back. After this, the generator produces **exactly the same future numbers** as the original would have. The names `state_dict` / `load_state_dict` copy PyTorch's convention for saving things.

---

## 5. The validation windows

```python
def validation_windows(tokens: np.ndarray, block_size: int) -> int:
    """Number of fixed validation windows (the last incomplete piece is not used)."""
    return (len(tokens) - 1) // block_size
```
- How many **full** windows fit: `(length − 1) // block_size`. The `−1` is because each window needs one extra token for `y`. A left-over piece shorter than a window is not used. Example: 100 tokens, block 10 → `99 // 10 = 9` windows.

```python
def validation_batches(tokens, block_size, batch_size, max_windows=None) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Non-overlapping windows covering the stream from the start, in batches."""
    n = validation_windows(tokens, block_size)
    if max_windows is not None:
        n = min(n, max_windows)
    for first in range(0, n, batch_size):
        idx = range(first, min(first + batch_size, n))
        win = np.stack([np.asarray(tokens[i * block_size:i * block_size + block_size + 1], dtype=np.int64)
                        for i in idx])
        w = torch.from_numpy(win)
        yield w[:, :-1], w[:, 1:]
```
- `n` = number of windows to use; `max_windows` (optional) caps it for a faster, partial evaluation.
- `for first in range(0, n, batch_size)` steps through the windows `batch_size` at a time. `idx` = the window numbers in this batch (the last batch can be smaller).
- **Window number `i` starts at token `i × block_size`**, so the `x` parts are **side by side with no gaps and no overlap**: window 0 covers tokens 0–9, window 1 covers 10–19, … (the raw slice is one token longer only so `y` can be shifted by one).
- `yield` — hands out one batch `(x, y)` and **pauses**; the caller's loop asks for the next. That is what makes this a **generator**: batches are made on demand, never all at once.
- Because the windows are fixed, **every evaluation measures the same text** and is comparable with the previous one.

---

## 6. Small things worth knowing

- **Training windows are random with replacement.** There is no "epoch" boundary; some tokens are seen more often than others by chance. Day 9.2 reports `tokens_seen ÷ training tokens` as "passes over the data".
- **Validation uses every token as a target at most once**, but the last few tokens (less than one window) are ignored.
- **`next_batch` duplicates Day 8's `real_batch`** (same idea, same code), because Day 8 was written first.
- **Same type as the files on disk:** `TOKEN_DTYPE` ensures the memory map reads the little-endian `uint16` Day 7.1 wrote.

## 7. Quick summary in one paragraph

`batching.py` turns the Day 7 token stream into next-token exercises: random windows (`x` = tokens, `y` = the same tokens shifted by one) for training, with a random generator whose state can be saved and restored so a resumed run sees exactly the same batches, and fixed side-by-side windows covering the whole validation stream so every evaluation measures the same thing.
