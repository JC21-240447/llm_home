# Day 10.2 — test_generation.py — Simple Line-by-Line Explanation (English)

> **How the three Day 10 files fit together**
> - **Day 10.1 `generation.py`** — the code being tested (**read it first**).
> - **Day 10 `day10_generate.py`** — the driver script.
> - **Day 10.2 (this file) `test_generation.py`** — **12 automatic tests**, CPU only, a few seconds. **No trained checkpoint is needed** — the tests use hand-made scores and a fake model.
>
> pytest basics were explained in Days 6.2–9.3; only new things are explained here.

---

## 0. The big picture (read this first)

Generation is easy to get *almost* right and hard to get *exactly* right (an off-by-one in a limit, a wrong sign in a penalty...). These tests check each small piece **with tiny hand-made numbers where the correct answer is obvious**:

| Promise | Test |
|---|---|
| Top candidates are sorted probabilities | `test_candidates_are_sorted_probabilities` |
| Special tokens are never written, but `<eos>` is allowed | `test_special_tokens_are_never_written_but_eos_is_allowed` |
| The run limit is exactly "4 allowed, 5th blocked" | `test_same_token_run_limit_matches_training_data` |
| The 3-token-sequence limit is exactly "twice allowed, 3rd blocked" | `test_trigram_limit_matches_training_data` |
| Repetition penalty lowers only used tokens (right sign for negatives) | `test_repetition_penalty_lowers_only_used_tokens` |
| Greedy takes the top token | `test_greedy_takes_the_top_token` |
| top-k = 1 and tiny top-p behave like greedy | `test_top_k_one_and_tiny_top_p_behave_like_greedy` |
| Temperature really changes how often the 2nd choice is picked | `test_temperature_changes_how_often_the_second_token_is_chosen` |
| Seeds are fixed per prompt and differ between prompts | `test_prompt_seed_is_fixed_per_prompt_and_differs_between_prompts` |
| `generate` stops at `<eos>`, stops at the length limit, and is reproducible | `test_generate_stops_at_eos_and_is_reproducible` |
| Repetition check and distinct-2 give the right values | `test_repetition_check_and_distinct_2` |
| Invalid settings are refused | `test_invalid_settings_are_refused` |

Run (as before): `.venv\Scripts\python.exe -m pytest tests -q`

---

## 1. New words

| Word | Simple meaning |
|---|---|
| **Fake model** | A tiny function that returns hand-chosen scores instead of a real network — makes the test fast and predictable. |
| **`pytest.mark.skipif(cond, reason=…)`** | A decorator: **skip** the test if the condition is true (here: when the tokenizer file is missing). |
| **`torch.Generator().manual_seed(n)`** | A seeded random generator (Day 10.1). |

---

## 2. Setup and the helper `flat`

```python
from __future__ import annotations

import math, sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.bpe_tokenizer import BOS_ID, BYTE_OFFSET, EOS_ID, ByteBPETokenizer  # noqa: E402
from src.generation import (TRAIN_MAX_SAME_TOKEN_RUN, TRAIN_MAX_TRIGRAM_REPEATS, SamplingConfig, banned_special_ids,
                            choose_token, distinct_2, generate, is_repetitive, next_token_candidates, process_logits,
                            prompt_seed)  # noqa: E402

TOKENIZER = REPO_ROOT / "data/tokenizer/tokenizer.json"
V = 400
A, B, C = 300, 301, 302      # three ordinary byte-range token IDs used in the tests
```
- Usual setup. `TOKENIZER` — the Day 6 tokenizer file (only one test needs it). `V = 400` — a **pretend vocabulary size**. `A, B, C` — three **ordinary token IDs** (300, 301, 302, all above the special range 0–103) used as stand-ins for "token A/B/C".

```python
def flat(**scores: float) -> torch.Tensor:
    t = torch.zeros(V)
    for k, v in scores.items():
        t[int(k[1:])] = v
    return t
```
- A tiny helper to **build score tables**: start with all zeros, then set chosen tokens. The trick: **keyword names carry the token number** — `flat(t300=10.0)` sets index `300` to `10.0` (`k[1:]` drops the leading `t`, `int(...)` makes it a number). `**scores` collects the keyword arguments into a dictionary.

---

## 3. The tests

### 3.1 `test_candidates_are_sorted_probabilities`

```python
def test_candidates_are_sorted_probabilities() -> None:
    logits = torch.randn(V, generator=torch.Generator().manual_seed(0))
    cands = next_token_candidates(logits, 10)
    probs = [p for _, p in cands]
    assert len(cands) == 10 and probs == sorted(probs, reverse=True)
    assert probs[0] == pytest.approx(torch.softmax(logits, 0).max().item())
    assert 0 < sum(probs) <= 1
```
- Random (but seeded) scores for 400 tokens. Ask for the top 10.
- Exactly 10 results, **sorted from biggest to smallest**; the first equals the true maximum probability; and the total of the top 10 is between 0 and 1 (probabilities never exceed 1 in total).

### 3.2 `test_special_tokens_are_never_written_but_eos_is_allowed`

```python
def test_special_tokens_are_never_written_but_eos_is_allowed() -> None:
    logits = torch.zeros(V)
    logits[BOS_ID] = 100.0
    out = process_logits(logits, [A], [], SamplingConfig())
    assert all(out[i] == float("-inf") for i in banned_special_ids(V))
    assert EOS_ID not in banned_special_ids(V) and torch.isfinite(out[EOS_ID])
    assert torch.isfinite(out[BYTE_OFFSET:]).all()
```
- Give `<bos>` a huge score (100) — a trap. After cleaning: **every banned special ID is `−∞`** (so the trap fails), `<eos>` is **not** in the banned list and its score stays finite, and **all normal tokens (ID ≥ 104)** are still finite (nothing else was removed by accident).

### 3.3 `test_same_token_run_limit_matches_training_data`

```python
def test_same_token_run_limit_matches_training_data() -> None:
    cfg = SamplingConfig(max_run=TRAIN_MAX_SAME_TOKEN_RUN)
    logits = flat(t300=10.0)
    ok = process_logits(logits, [A] * (TRAIN_MAX_SAME_TOKEN_RUN - 1), [], cfg)   # 3 in a row: a 4th is allowed
    assert torch.isfinite(ok[A])
    stop = process_logits(logits, [A] * TRAIN_MAX_SAME_TOKEN_RUN, [], cfg)        # 4 in a row: a 5th is not
    assert stop[A] == float("-inf")
```
- The classic **boundary test**: with limit 4, the context `[A, A, A]` (3 in a row) must still **allow** a 4th `A`, and `[A, A, A, A]` (4 in a row) must **block** a 5th. (`[A] * 3` is a list of three `A`s.) This catches off-by-one mistakes.

### 3.4 `test_trigram_limit_matches_training_data`

```python
def test_trigram_limit_matches_training_data() -> None:
    cfg = SamplingConfig(max_trigram_repeats=TRAIN_MAX_TRIGRAM_REPEATS)
    once = [A, B, C, A, B]                    # A B C seen once: A B C again is allowed (like 1991年から1997年)
    assert torch.isfinite(process_logits(flat(t302=5.0), once, [], cfg)[C])
    twice = [A, B, C, A, B, C, A, B]          # seen twice: a third time is not
    assert process_logits(flat(t302=5.0), twice, [], cfg)[C] == float("-inf")
```
- Limit 2. Context `A B C A B` — the sequence `A B C` has appeared **once**, and we are at `A B`, so `C` (a second `A B C`) must be **allowed**. Context `A B C A B C A B` — `A B C` has appeared **twice**, so `C` would make a **third** → **blocked**.

### 3.5 `test_repetition_penalty_lowers_only_used_tokens`

```python
def test_repetition_penalty_lowers_only_used_tokens() -> None:
    logits = flat(t300=2.0, t301=-1.0, t302=2.0)
    out = process_logits(logits, [A, B], [A, B], SamplingConfig(repetition_penalty=2.0))
    assert out[A] == pytest.approx(1.0)       # positive score divided
    assert out[B] == pytest.approx(-2.0)      # negative score multiplied
    assert out[C] == pytest.approx(2.0)       # unused token unchanged
```
- Scores: A = 2, B = −1, C = 2; A and B were already used; penalty 2.
- A (positive) → `2 / 2 = 1`; B (negative) → `−1 × 2 = −2` (**more negative = less likely**, the correct direction); C (not used) → still `2`. This guards against the classic sign mistake.

### 3.6 `test_greedy_takes_the_top_token`

```python
def test_greedy_takes_the_top_token() -> None:
    assert choose_token(flat(t300=1.0, t301=3.0), SamplingConfig(), None) == B
```
- Scores: A = 1, B = 3 → greedy must pick **B** (301). (No random generator needed: `None`.)

### 3.7 `test_top_k_one_and_tiny_top_p_behave_like_greedy`

```python
def test_top_k_one_and_tiny_top_p_behave_like_greedy() -> None:
    logits = flat(t300=1.0, t301=3.0, t302=2.0)
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        assert choose_token(logits, SamplingConfig(method="sample", top_k=1), g) == B
        assert choose_token(logits, SamplingConfig(method="sample", top_p=0.01), g) == B
```
- Even in **sampling** mode, `top_k=1` (keep only the best) and a tiny `top_p=0.01` (keep only the top token, which is always kept) leave **one** candidate, so the draw must be **B every time** — checked 20 times for each. Proves both filters really cut the tail.

### 3.8 `test_temperature_changes_how_often_the_second_token_is_chosen`

```python
def test_temperature_changes_how_often_the_second_token_is_chosen() -> None:
    logits = torch.full((V,), float("-inf"))
    logits[A], logits[B] = 2.0, 1.0

    def share_b(t: float) -> float:
        g = torch.Generator().manual_seed(1)
        cfg = SamplingConfig(method="sample", temperature=t)
        return sum(choose_token(logits, cfg, g) == B for _ in range(2000)) / 2000

    expected = 1 / (1 + math.e)               # softmax probability of B at temperature 1
    assert share_b(1.0) == pytest.approx(expected, abs=0.04)
    assert share_b(0.5) < share_b(1.0) < share_b(2.0)
```
- Only two candidates (A = 2, B = 1; everything else `−∞`). `share_b(t)` = the **fraction of 2000 draws that pick B** at temperature `t`.
- At temperature 1, B's probability is `e¹ / (e² + e¹) = 1 / (1 + e) ≈ 27%`; the measured share must be within ±0.04 of it (random sampling is never exact).
- And the ordering: **colder (0.5) → B chosen less often; hotter (2.0) → more often**. That is what temperature is supposed to do.

### 3.9 `test_prompt_seed_is_fixed_per_prompt_and_differs_between_prompts`

```python
def test_prompt_seed_is_fixed_per_prompt_and_differs_between_prompts() -> None:
    assert prompt_seed(42, "営業部の") == prompt_seed(42, "営業部の")
    assert prompt_seed(42, "営業部の") != prompt_seed(42, "開発部の")
    assert prompt_seed(42, "営業部の") != prompt_seed(7, "営業部の")
```
- Same seed + same prompt → **same** number (reproducible); a different prompt → a different number; a different seed → a different number.

### 3.10 `test_generate_stops_at_eos_and_is_reproducible` (skipped if no tokenizer)

```python
@pytest.mark.skipif(not TOKENIZER.exists(), reason="needs data/tokenizer/tokenizer.json (Day 6)")
def test_generate_stops_at_eos_and_is_reproducible() -> None:
    tok = ByteBPETokenizer.load(TOKENIZER)
    target = tok.encode("社員を教えてください。")

    def fake_model(ids: list[int]) -> torch.Tensor:      # writes the target, then <eos>
        n = len(ids) - 1 - len(tok.encode("営業部の"))
        logits = torch.zeros(tok.vocab_size)
        logits[target[n] if n < len(target) else EOS_ID] = 20.0
        return logits
```
- `@pytest.mark.skipif(...)` — skipped when the Day 6 tokenizer is not available.
- A **fake model** — a function from "tokens so far" to scores: it works out `n` = how many tokens were already generated (`len(ids)` minus `<bos>` minus the prompt's tokens) and gives a **huge score (20) to the next token of a fixed sentence** `社員を教えてください。`, and after the sentence is complete, to `<eos>`. So the "model" always writes exactly that sentence and then stops.

```python
    g = generate(fake_model, tok, "営業部の", SamplingConfig())
    assert g.finished and g.text == "社員を教えてください。" and g.ids[-1] == EOS_ID
    s1 = generate(fake_model, tok, "営業部の", SamplingConfig(method="sample", temperature=1.0))
    s2 = generate(fake_model, tok, "営業部の", SamplingConfig(method="sample", temperature=1.0))
    assert s1.ids == s2.ids
```
- Greedy generation must **finish**, the text must be the sentence, and the last generated ID must be `<eos>`.
- Sampling twice with the same settings must give **identical IDs** (reproducibility with the fixed seed).

```python
    def never_ends(ids: list[int]) -> torch.Tensor:
        logits = torch.zeros(tok.vocab_size)
        logits[EOS_ID] = -50.0
        return logits

    cut = generate(never_ends, tok, "営業部の", SamplingConfig(max_new_tokens=7))
    assert cut.stop_reason == "max_new_tokens" and len(cut.ids) == 7
```
- A second fake model that **refuses to ever output `<eos>`** (score −50). With `max_new_tokens=7`, generation must **stop at the length limit**: reason `"max_new_tokens"` and exactly **7** tokens.
- This is possible only because `generate` accepts any scores **function**, not just a real model (Day 10.1 §4.3).

### 3.11 `test_repetition_check_and_distinct_2`

```python
def test_repetition_check_and_distinct_2() -> None:
    assert not is_repetitive([A, A, A, A])            # 4 in a row occurs in training questions
    assert is_repetitive([A] * 5)
    assert is_repetitive([A, B, C] * 3)
    assert distinct_2([[A, B, A, B]]) == pytest.approx(2 / 3)
    assert distinct_2([]) == 0.0
```
- `A A A A` (4 in a row) is **not** repetitive (it happens in real questions); `A` ×5 **is**; `A B C A B C A B C` (the same 3-token sequence 3 times) **is**.
- `distinct_2([[A, B, A, B]])`: the pairs are `(A,B)`, `(B,A)`, `(A,B)` → 2 different out of 3 → `2/3`. An empty input gives `0.0` (no division by zero).

### 3.12 `test_invalid_settings_are_refused`

```python
def test_invalid_settings_are_refused() -> None:
    for bad in ({"method": "beam"}, {"temperature": 0}, {"top_p": 0}, {"repetition_penalty": 0.5}):
        with pytest.raises(ValueError):
            SamplingConfig(**bad)
```
- Four invalid settings (unknown method, zero temperature, zero top-p, a penalty below 1); each must raise `ValueError` (`SamplingConfig(**bad)` unpacks the dictionary into keyword arguments).

---

## 4. How many tests?

**12 tests.** Expected: `12 passed`, or `11 passed, 1 skipped` if the tokenizer file is not there (test 3.10).

## 5. What these tests do *not* cover

- Generation with a **real trained model** (that is what the Day 10 report script shows).
- `load_model` and `model_logits_fn` (they need a checkpoint / a model).
- `token_text`, `max_same_token_run` and `max_trigram_count` directly (they are exercised through `is_repetitive`).
- The command-line script `day10_generate.py` and the Markdown report format.

## 6. Quick summary in one paragraph

The 12 CPU tests check the generation engine with tiny hand-made scores and a fake model: candidates are sorted probabilities, special tokens are banned but `<eos>` is allowed, the repetition limits block exactly at the boundary measured from the training questions, the penalty treats negative scores correctly, greedy / top-k / top-p / temperature behave as defined, seeds are stable per prompt, generation stops at `<eos>` or at the length limit reproducibly, the repetition and distinct-2 checks give the expected values, and invalid settings are refused.
