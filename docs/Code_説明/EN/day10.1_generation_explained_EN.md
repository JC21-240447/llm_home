# Day 10.1 — generation.py — Simple Line-by-Line Explanation (English)

> **How the three Day 10 files fit together**
> - **Day 10.1 (this file) `generation.py`** — the *engine*: turns the model's scores into next-token candidates and into generated text.
> - **Day 10 `day10_generate.py`** — the *driver*: three commands (`candidates`, `generate`, `report`) that use this engine.
> - **Day 10.2 `test_generation.py`** — the *inspector*: 12 automatic tests (no trained model needed).
>
> Suggested reading order: **10.1 → 10 → 10.2**.

---

## 0. The big picture (read this first)

Model A (Day 8) does **one** thing: given the tokens so far, it gives a **score for every token** of the vocabulary ("how likely is this to come next?"). To get *text* we must repeat a small cycle, one token at a time:

```
tokens so far → model → scores → ① clean up the scores → ② choose ONE token → add it → repeat
                                                     stop at <eos> (end of question) or after 40 tokens
```

### Step 1 — From scores to probabilities (next-token candidates)

`softmax` turns the scores into **probabilities that add up to 1**. Showing the 10 most probable tokens is school plan **task 10**: "what does the model think comes next?"

### Step 2 — Clean up the scores (`process_logits`)

| Adjustment | What it does | Why |
|---|---|---|
| **Ban special tokens** | `<pad>`, `<bos>`, `<unk>`, `<unused_*>` can never be written (`<eos>` is allowed) | They are not real text |
| **Repetition penalty** (optional) | Tokens already used become a little less likely | Stops loops |
| **Repetition limits** | Same token at most **4 times in a row**; same 3-token sequence at most **2 times** | Stops loops like `5000000000…`, **without** blocking real questions |

The limits (4 and 2) were **measured from the training questions** on Day 10: they are the largest repetition that really occurs in real questions (for example `社員番号100003` has several zeros in a row; `1991年から1997年` repeats a 3-token pattern). A token that would break a limit is **removed** (score `−∞`).

### Step 3 — Choose one token

| Method | How it chooses |
|---|---|
| **Greedy** | Always the most likely token. Same input → same output. |
| **Sampling** | A random draw according to the probabilities, shaped by **temperature**, **top-k** and **top-p**, with a **fixed seed** (so it is reproducible). |

#### Temperature — how "bold" is the choice?

Divide the scores by the temperature before softmax. Example with two candidates, scores 2 and 1:

| Temperature | Probabilities | Effect |
|---|---|---|
| 0.5 (< 1) | 88% / 12% | **Sharper** — safer, more repetitive |
| 1.0 | 73% / 27% | Normal |
| 2.0 (> 1) | 62% / 38% | **Flatter** — more random, more mistakes |

#### Top-k and top-p — cut off the unlikely tail

- **top-k = 50:** keep only the 50 best tokens.
- **top-p = 0.95 ("nucleus"):** keep the smallest group of top tokens whose probabilities add up to 0.95. Example: probabilities `0.5, 0.3, 0.15, 0.05`, p = 0.9 → keep the first three (0.5 + 0.3 + 0.15 = 0.95 ≥ 0.9), drop 0.05, then renormalise.

### Step 4 — Repeat

Until `<eos>` or `max_new_tokens` (40).

---

## 1. Words you need to know (mini dictionary)

| Word | Simple meaning |
|---|---|
| **Logits** | The model's raw scores (one per token). |
| **Softmax** | Scores → probabilities adding up to 1. |
| **Greedy decoding** | Always take the top token. |
| **Sampling** | Draw a token at random with the given probabilities. |
| **Temperature / top-k / top-p** | The three "randomness controls" above. |
| **Seed / `torch.Generator`** | Makes random draws repeatable. |
| **Context** | The tokens so far (prompt + generated). |
| **`-inf` masking** | Setting a score to minus infinity so its probability becomes exactly 0. |
| **Bigram / trigram** | A sequence of 2 / 3 neighbouring tokens. |
| **distinct-2** | Variety measure: different 2-token sequences ÷ all 2-token sequences. |
| **CRC32** | A fixed checksum number computed from text (same every time, on every computer). |
| **`<eos>` / ⟨eos⟩** | The "end of question" token. |

---

## 2. The docstring, imports and constants

- The docstring (English + Japanese) explains steps 1–4 above.

```python
from __future__ import annotations

import math
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from src.bpe_tokenizer import BOS_ID, BYTE_OFFSET, EOS_ID, ByteBPETokenizer
from src.model import ModelA, ModelConfig
from src.training import load_checkpoint
```
- Most are known. New: `zlib` (for CRC32), `Callable` and `Sequence` (type hints: "a function" / "a list-like sequence").
- From earlier days: the ID constants and tokenizer (**Day 6.1**), the model (**Day 8.1**), and `load_checkpoint` (**Day 9.2**).

```python
TRAIN_MAX_SAME_TOKEN_RUN: int = 4
TRAIN_MAX_TRIGRAM_REPEATS: int = 2

LogitsFn = Callable[[list[int]], torch.Tensor]
```
- The two **limits measured from the training questions**.
- `LogitsFn` — a **type alias**: "a function that takes a list of token IDs and returns a tensor of scores". The generation code works with *any* such function (a real model, or a fake one in the tests).

---

## 3. `SamplingConfig` — all the generation settings

```python
@dataclass
class SamplingConfig:
    method: str = "greedy"              # "greedy" or "sample"
    temperature: float = 1.0
    top_k: int = 0                      # 0 = off
    top_p: float = 1.0                  # 1.0 = off
    repetition_penalty: float = 1.0     # 1.0 = off
    max_run: int | None = None          # None = off; TRAIN_MAX_SAME_TOKEN_RUN to switch on
    max_trigram_repeats: int | None = None  # None = off; TRAIN_MAX_TRIGRAM_REPEATS to switch on
    max_new_tokens: int = 40
    seed: int = 42
    name: str = "custom"
```
- Every setting has a "**does nothing**" default (`0`, `1.0`, `None`), so by default you get plain greedy generation.
- `max_run` / `max_trigram_repeats` switch the repetition limits **on** when set to the training values.
- `max_new_tokens` = length limit (40); `seed` = for sampling; `name` = label for tables.

```python
    def __post_init__(self) -> None:
        if self.method not in ("greedy", "sample"):
            raise ValueError("method must be 'greedy' or 'sample'")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0 (use method='greedy' for the most likely token)")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty < 1:
            raise ValueError("repetition_penalty must be >= 1")

    def to_dict(self) -> dict:
        return asdict(self)
```
- Reject nonsense settings (same pattern as Day 8.1's `ModelConfig`): unknown method; temperature ≤ 0 (division by zero; use greedy instead); `top_p` outside (0, 1]; a penalty below 1 (that would *encourage* repeats).
- `to_dict` — settings as a dictionary (printed in the generated reports).

### The four presets

```python
PRESET_SETTINGS: dict[str, SamplingConfig] = {
    "greedy": SamplingConfig(name="greedy"),
    "greedy+limits": SamplingConfig(name="greedy+limits", max_run=TRAIN_MAX_SAME_TOKEN_RUN,
                                    max_trigram_repeats=TRAIN_MAX_TRIGRAM_REPEATS),
    "sample": SamplingConfig(name="sample", method="sample", temperature=0.8, top_k=50, top_p=0.95,
                             max_run=TRAIN_MAX_SAME_TOKEN_RUN, max_trigram_repeats=TRAIN_MAX_TRIGRAM_REPEATS),
    "sample-hot": SamplingConfig(name="sample-hot", method="sample", temperature=1.5,
                                 max_run=TRAIN_MAX_SAME_TOKEN_RUN, max_trigram_repeats=TRAIN_MAX_TRIGRAM_REPEATS),
}
```

| Preset | What it is | Purpose |
|---|---|---|
| `greedy` | top token, no controls | What Day 9's `samples.md` showed |
| `greedy+limits` | greedy + repetition limits | See the limits at work |
| `sample` | temperature 0.8, top-k 50, top-p 0.95, limits | The "recommended" setting |
| `sample-hot` | temperature 1.5, limits | Shows what **too much randomness** does |

They are **fixed**, so the Day 10 report can be reproduced.

### The result box `Generation`

```python
@dataclass
class Generation:
    prompt: str
    ids: list[int]          # generated tokens only (without the prompt; <eos> included if reached)
    text: str               # decoded generated text (without <eos>)
    stop_reason: str        # "eos" or "max_new_tokens"

    @property
    def finished(self) -> bool:
        return self.stop_reason == "eos"
```
- The outcome of one generation: the prompt, the **new** token IDs (with `<eos>` if reached), the decoded **continuation text** (without the prompt and without `<eos>`), and why it stopped.
- `finished` — `True` if the model **ended the question by itself** (`<eos>`), `False` if it was cut at the length limit.

---

## 4. Loading and wrapping the model

### 4.1 `token_text`

```python
def token_text(tok: ByteBPETokenizer, token_id: int) -> str:
    """A readable form of one token: the text, <eos>, or <xx..> for a piece of a character."""
    if token_id == EOS_ID:
        return "<eos>"
    raw = tok.token_bytes[token_id]
    if token_id < BYTE_OFFSET:
        return raw.decode("utf-8")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return "<" + raw.hex() + ">"
```
- A readable label for one token (used in the candidate tables): `<eos>` for the end token; special/reserved tokens by their ASCII names; ordinary tokens as text; and a **piece of a character** (bytes that are not valid UTF-8 alone) as `<hex>` — the same convention as `ByteBPETokenizer.pieces` (Day 6.1).

### 4.2 `load_model`

```python
def load_model(checkpoint: Path, device: torch.device) -> tuple[ModelA, dict]:
    state = load_checkpoint(checkpoint, device)
    model = ModelA(ModelConfig.from_dict(state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    info = {"path": str(checkpoint), "step": state["step"], "best_val": state["best_val"],
            "tokenizer_sha256": state["tokenizer_sha256"], "model_name": state["model_config"].get("name", "")}
    return model, info
```
- Load a Day 9 checkpoint (`best.pt` or `last.pt`) with `load_checkpoint` (safe loading, Day 9.2).
- Rebuild the model **from the settings stored inside the checkpoint** (`state["model_config"]`), load its weights, switch to **eval mode**.
- Return the model plus an `info` dictionary (path, step, best validation loss, **the tokenizer fingerprint stored at training time**, model name) — so callers can check the tokenizer still matches.

### 4.3 `model_logits_fn`

```python
def model_logits_fn(model: ModelA, device: torch.device) -> LogitsFn:
    """Wrap the model as 'token IDs so far -> scores for the next token' (crops to the block size)."""
    block = model.cfg.block_size

    @torch.no_grad()
    def fn(ids: list[int]) -> torch.Tensor:
        x = torch.tensor([ids[-block:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        return logits[0, -1].float().cpu()

    return fn
```
- Turns the model into the simple function type `LogitsFn`. This is a **closure**: the inner `fn` remembers `model`, `device` and `block`.
- `fn(ids)`: keep the last `block_size` tokens, make a batch of one, run the model, take the scores at the **last position** (`logits[0, -1]`), as 32-bit floats on the **CPU**. (`@torch.no_grad()` — no gradients.)
- Why a function? So the generation logic (everything below) does **not need to know anything about models or GPUs**, and the tests can replace it with a tiny fake function.
- (No "cache": every new token re-runs the whole sequence. Simple, and fast enough for ≤ 40 tokens.)

---

## 5. The pieces

### 5.1 `next_token_candidates`

```python
def next_token_candidates(logits: torch.Tensor, k: int = 10) -> list[tuple[int, float]]:
    """The k most likely next tokens with their probabilities (the raw model, no adjustments)."""
    probs = F.softmax(logits.float(), dim=-1)
    top = torch.topk(probs, k)
    return [(int(i), float(p)) for p, i in zip(top.values, top.indices)]
```
- `softmax` → probabilities; `torch.topk(probs, k)` → the `k` biggest values and their positions (indices = token IDs), already sorted from biggest. Returns `(token ID, probability)` pairs. **No adjustments** — this shows what the raw model believes.

### 5.2 `banned_special_ids`

```python
def banned_special_ids(vocab_size: int) -> list[int]:
    """Every ID below the byte range except <eos>: never written in a question."""
    return [i for i in range(min(BYTE_OFFSET, vocab_size)) if i != EOS_ID]
```
- IDs 0–103 (`<pad>`, `<bos>`, `<unk>`, the 100 `<unused_*>`) **except `<eos>`** (ID 2). (`min(...)` protects against a tiny vocabulary in tests.)

### 5.3 `run_length_at_end`

```python
def run_length_at_end(ids: Sequence[int]) -> int:
    if not ids:
        return 0
    n = 1
    while n < len(ids) and ids[-1 - n] == ids[-1]:
        n += 1
    return n
```
- How many **identical tokens in a row** are at the end of `ids` (e.g. `[7, 5, 5, 5]` → 3). `ids[-1]` is the last; `ids[-1-n]` looks `n` places further back; stop at the first different one.

### 5.4 `process_logits` — the clean-up

```python
def process_logits(logits, context, generated, cfg) -> torch.Tensor:
    """Apply bans and repetition control. `context` = prompt + generated (without <bos>);
    `generated` = only the new tokens. Returns new logits (removed tokens are -inf)."""
    out = logits.clone().float()
    out[banned_special_ids(out.numel())] = float("-inf")
```
- Copy the scores (`clone`, so the original is untouched) and set the banned special tokens to `−∞`.

```python
    if cfg.repetition_penalty > 1.0 and generated:
        used = torch.tensor(sorted(set(generated)), dtype=torch.long)
        vals = out[used]
        out[used] = torch.where(vals > 0, vals / cfg.repetition_penalty, vals * cfg.repetition_penalty)
```
- **Repetition penalty** (the "CTRL-style" rule), only if enabled and some tokens were already generated: for each **already-used** token, **positive scores are divided** by the penalty and **negative scores multiplied** — both make the token **less likely**. (Dividing a negative score would wrongly make it *more* likely, hence the two cases; `torch.where(condition, a, b)` picks `a` where the condition holds, else `b`.) Tokens never used are unchanged.

```python
    if cfg.max_run is not None and run_length_at_end(context) >= cfg.max_run:
        out[context[-1]] = float("-inf")         # one more would make the run too long
```
- **Run limit:** if the context already ends with `max_run` (4) identical tokens, forbid that same token once more (a 5th would be too long).

```python
    if cfg.max_trigram_repeats is not None and len(context) >= 2:
        a, b = context[-2], context[-1]
        counts: dict[int, int] = {}
        for x, y, z in zip(context, context[1:], context[2:]):
            if x == a and y == b:
                counts[z] = counts.get(z, 0) + 1
        for z, n in counts.items():
            if n >= cfg.max_trigram_repeats:
                out[z] = float("-inf")
    return out
```
- **3-token-sequence limit:** `a, b` = the last two tokens. Look through the context for every earlier place where `a, b` appeared and note **which token `z` followed** (`counts[z]`). If the sequence `a b z` has already occurred `max_trigram_repeats` (2) times, forbid `z` — a third repeat of the same 3-token sequence is blocked. Return the cleaned scores.
- Example: context `A B C A B C A B` — `A B C` has occurred twice, so after `A B` the token `C` is banned.

### 5.3 `choose_token`

```python
def choose_token(logits, cfg, generator) -> int:
    if not torch.isfinite(logits).any():
        return EOS_ID                            # everything was removed: end the question
    if cfg.method == "greedy":
        return int(logits.argmax())
```
- If **every** token was removed (all `−∞`), end the question gracefully (`<eos>`). Greedy → the position of the biggest score.

```python
    scores = logits / cfg.temperature
    if cfg.top_k > 0:
        kth = torch.topk(scores, min(cfg.top_k, scores.numel())).values[-1]
        scores = scores.masked_fill(scores < kth, float("-inf"))
    probs = F.softmax(scores, dim=-1)
```
- **Temperature:** divide the scores. **Top-k:** find the k-th best score (`kth`, the smallest of the top-k) and set everything **below** it to `−∞`. Then `softmax` → probabilities.

```python
    if cfg.top_p < 1.0:
        sorted_p, order = torch.sort(probs, descending=True)
        before = torch.cumsum(sorted_p, dim=0) - sorted_p          # probability of the tokens ranked above
        keep = before < cfg.top_p                                   # always keeps the top token
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[order[keep]] = True
        probs = torch.where(mask, probs, torch.zeros_like(probs))
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=generator))
```
- **Top-p:** sort probabilities from biggest; `cumsum − sorted_p` = the total probability of the tokens ranked **above** each one (`before`). Keep a token if that total is still **below** `p` — so the top token (before = 0) is **always kept**. `mask[order[keep]] = True` marks the kept tokens in the original order; zero out the others and **renormalise** (divide by the new sum).
- `torch.multinomial(probs, 1, generator=generator)` — **draw one token at random** according to `probs`, using the seeded generator.

### 5.4 `prompt_seed`

```python
def prompt_seed(seed: int, prompt: str) -> int:
    return (seed * 1_000_003 + zlib.crc32(prompt.encode("utf-8"))) % (2**63)
```
- One fixed integer for each `(seed, prompt)` pair. `zlib.crc32` is a **deterministic checksum** of the prompt text (unlike Python's `hash`, which changes between runs).
- Why per prompt? With **one shared seed**, different prompts would get the **same random draws**; this gives **each prompt its own random sequence**, and the same prompt always gets the same one (reproducible). `% 2**63` keeps it a valid seed size.

### 5.5 `generate` — the whole loop

```python
def generate(logits_fn: LogitsFn, tok: ByteBPETokenizer, prompt: str, cfg: SamplingConfig) -> Generation:
    """Continue `prompt` until <eos> or cfg.max_new_tokens. Reproducible: same inputs + seed -> same output."""
    prompt_ids = tok.encode(prompt)
    ids = [BOS_ID] + prompt_ids
    generated: list[int] = []
    generator = torch.Generator().manual_seed(prompt_seed(cfg.seed, prompt)) if cfg.method == "sample" else None
    stop = "max_new_tokens"
    for _ in range(cfg.max_new_tokens):
        logits = process_logits(logits_fn(ids), prompt_ids + generated, generated, cfg)
        nxt = choose_token(logits, cfg, generator)
        ids.append(nxt)
        generated.append(nxt)
        if nxt == EOS_ID:
            stop = "eos"
            break
    text_ids = [i for i in generated if i != EOS_ID]
    return Generation(prompt, generated, tok.decode(text_ids), stop)
```
- `prompt_ids` — the prompt as tokens. `ids = [BOS_ID] + prompt_ids` — the model's input starts with `<bos>`, exactly like the training data (`<bos> question <eos>`).
- `generated` — the new tokens. For sampling, create a **seeded random generator** for this prompt; greedy needs none.
- Each round: ask the model for scores (`logits_fn(ids)`), **clean them** (context = prompt + generated, *without* `<bos>`), **choose** a token, add it to both lists; stop at `<eos>` (`stop = "eos"`) — otherwise the loop ends after `max_new_tokens` with `"max_new_tokens"`.
- Return a `Generation`: the decoded text **without** `<eos>`.

---

## 6. Quick automatic checks (school plan task 6 items)

```python
def max_same_token_run(ids: Sequence[int]) -> int:
    best = run = 0
    for i, t in enumerate(ids):
        run = run + 1 if i and ids[i - 1] == t else 1
        best = max(best, run)
    return best
```
- The **longest run of identical tokens** anywhere in `ids`: `run` counts the current streak (`i and …` = not the first item and equal to the previous one), `best` keeps the maximum.

```python
def max_trigram_count(ids: Sequence[int]) -> int:
    counts: dict[tuple[int, int, int], int] = {}
    for tri in zip(ids, ids[1:], ids[2:]):
        counts[tri] = counts.get(tri, 0) + 1
    return max(counts.values(), default=0)
```
- Count every 3-token sequence; return the **most frequent one's count** (`default=0` for very short input).

```python
def is_repetitive(ids: Sequence[int]) -> bool:
    return max_same_token_run(ids) > TRAIN_MAX_SAME_TOKEN_RUN or max_trigram_count(ids) > TRAIN_MAX_TRIGRAM_REPEATS
```
- **A3 check:** "more repetition than any real training question has" — 5+ identical tokens in a row, or one 3-token sequence 3+ times.

```python
def distinct_2(all_ids: Sequence[Sequence[int]]) -> float:
    pairs = [p for ids in all_ids for p in zip(ids, ids[1:])]
    return len(set(pairs)) / len(pairs) if pairs else 0.0
```
- **A4 check (variety):** collect every 2-token pair of every output; `distinct-2` = number of **different** pairs ÷ **all** pairs. Close to 1 = varied; low = repetitive/boring. (0.0 if there are no pairs.)

```python
def perplexity_of(loss: float) -> float:
    return math.exp(loss)
```
- Perplexity from a loss (Day 9.2); a small helper not used elsewhere in this file.

---

## 7. Small things worth knowing

- **Greedy and sampling are both reproducible.** Greedy has no randomness; sampling uses a seed per prompt. On the **same machine** the same inputs give the same text. (Between CPU and GPU the model's last digits can differ, which in a rare near-tie could change a token.)
- **The limits never block real questions** because 4 and 2 are the maxima measured in the training questions.
- **Repetition penalty is off in all presets** (`1.0`); it is available for experiments.
- **No KV cache:** the model re-reads the whole sequence at every token. Fine for 40 tokens.
- **The prompt starts after `<bos>`.** Training used `<bos> question <eos>`, so generation does the same.
- **`<eos>` is the only special token that may be produced.** If everything else is banned/removed, the question simply ends.

## 8. Quick summary in one paragraph

`generation.py` wraps a trained model as a function from tokens to scores, shows the top next-token probabilities, cleans the scores (bans special tokens, optionally penalises repeats, enforces the repetition limits measured from the training questions), then chooses one token — greedily, or by seeded sampling shaped by temperature, top-k and top-p — and repeats until `<eos>` or 40 tokens; it also provides quick checks for repetition and variety (distinct-2).
