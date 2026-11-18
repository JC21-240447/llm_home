# Day 12.1 — generation.py (updated) — What Changed + Line-by-Line Explanation (English)

> **This is the UPDATED version of `generation.py`** that you first saw on Day 10.1.
> - **Everything from the top of the file down to `perplexity_of` is identical** to Day 10.1. Its full line-by-line explanation is in **`day10.1_generation_explained_EN.md`** (and the JA file) — read that for `SamplingConfig`, the presets, `process_logits`, `choose_token`, `generate`, `is_repetitive`, `distinct_2`, etc.
> - **One new function was added at the very end: `quick_checks`.** This file explains it in detail.
>
> **How the five Day 12 files fit together:** `day12_train_main.py` (main training; **calls `quick_checks`**), `day12_monitor.py` (watch a run), **`generation.py` (12.1, this file)**, `training.py` (12.2, early stopping), `test_training.py` (12.3, two new tests).

---

## 0. The big picture (read this first)

On Day 11, the script `day11_size_trials.py` had its **own** function `generation_check` that generated text for the prompts and computed the **task 6 automatic items**:

| Item | Meaning |
|---|---|
| **A2 finished** | fraction of outputs that ended with `<eos>` |
| **A3 repetitive** | fraction with more repetition than any real training question |
| **A4 distinct-2** | variety of the outputs |
| **A5 new** (reference only) | finished outputs that are **not** a training question |

On Day 12 the main-training script needs the **same** checks. Instead of copying the code, the logic was **moved into the library** as `quick_checks` so that any script can call it. (Day 12 uses it; the Day 11 script you pasted still has its own copy.)

This is a normal software habit: **when two scripts need the same function, put it in the shared library.**

---

## 1. Unchanged parts (pointer table)

| Part of the file | Explained in |
|---|---|
| `TRAIN_MAX_SAME_TOKEN_RUN`, `TRAIN_MAX_TRIGRAM_REPEATS`, `LogitsFn` | Day 10.1 §2 |
| `SamplingConfig`, `PRESET_SETTINGS`, `Generation` | Day 10.1 §3 |
| `token_text`, `load_model`, `model_logits_fn` | Day 10.1 §4 |
| `next_token_candidates`, `banned_special_ids`, `run_length_at_end`, `process_logits`, `choose_token`, `prompt_seed`, `generate` | Day 10.1 §5 |
| `max_same_token_run`, `max_trigram_count`, `is_repetitive`, `distinct_2`, `perplexity_of` | Day 10.1 §6 |

---

## 2. The new function: `quick_checks`

```python
def quick_checks(logits_fn: LogitsFn, tok: ByteBPETokenizer, prompts: Sequence[str], cfg: SamplingConfig,
                 train_set: set[str] | None = None) -> dict:
    """Task 6 automatic items on a list of prompts (Day 11/12): finished (A2), repetitive (A3), distinct-2 (A4),
    new sentences (A5, reference), and the generated texts."""
```
- **Inputs:**
  - `logits_fn` — the model wrapped as "tokens → scores" (`model_logits_fn`, Day 10.1);
  - `tok` — the tokenizer;
  - `prompts` — the list of beginnings;
  - `cfg` — **one** generation setting (e.g. `PRESET_SETTINGS["greedy+limits"]`);
  - `train_set` — optional: the set of training questions (for the "new" count). `set[str] | None = None` means "a set of strings, or nothing (default)".
- **Output:** a dictionary with the four numbers and the texts.

```python
    gens = [generate(logits_fn, tok, p, cfg) for p in prompts]
    n = max(1, len(gens))
    body = [[i for i in g.ids if i != EOS_ID] for g in gens]
```
- `gens` — generate a continuation for **every prompt** with the given setting (a list of `Generation` objects, Day 10.1).
- `n = max(1, len(gens))` — the number of outputs, but **never 0**, so the divisions below cannot fail with "division by zero" even if the prompt list is empty.
- `body` — for each output, the **generated token IDs without `<eos>`** (`[i for i in g.ids if i != EOS_ID]`). It is computed **once** and reused by two of the checks below (the Day 11 version rebuilt it twice inline).

```python
    return {
        "finished": sum(g.finished for g in gens) / n,
        "repetitive": sum(is_repetitive(tok.encode(g.prompt) + b) for g, b in zip(gens, body)) / n,
        "distinct2": distinct_2(body),
        "new": None if train_set is None else sum((g.prompt + g.text) not in train_set for g in gens if g.finished),
        "texts": {g.prompt: g.text + (" ⟨eos⟩" if g.finished else " …") for g in gens},
    }
```
- **`finished` (A2):** `sum(g.finished ...)` adds up the `True`s (each counts as 1); divided by `n` → a **fraction** (0.0–1.0).
- **`repetitive` (A3):** for each output, check `is_repetitive` on **prompt tokens + generated tokens** (the repetition is judged on the whole sentence). `zip(gens, body)` pairs each output with its token list. Divided by `n` → a fraction.
- **`distinct2` (A4):** variety over **all** outputs together (`distinct_2(body)`).
- **`new` (A5, reference):** if no `train_set` was given → `None`. Otherwise count the **finished** outputs whose full sentence (`prompt + text`) is **not** in the training set.
- **`texts`:** a dictionary **prompt → generated text**, with ` ⟨eos⟩` appended if the model ended the question by itself, or ` …` if it was cut at the length limit — ready to drop into a Markdown report.

---

## 3. How it is used (Day 12)

```python
checks[name] = {s: quick_checks(fn, tok, prompts, PRESET_SETTINGS[s], train_set)
                for s in ("greedy+limits", "sample")}
gl = checks[name]["greedy+limits"]
r.finished_rate, r.repetitive_rate, r.distinct2 = gl["finished"], gl["repetitive"], gl["distinct2"]
```
- A dictionary comprehension runs `quick_checks` for **two settings**; the `greedy+limits` numbers are copied into the run summary (Day 11.1) — they decide whether the run **passes the task 6 guidelines**; the `sample` results are only for the report.

## 4. Small things worth knowing

- **Same formulas as Day 11's `generation_check`**, so Day 11 and Day 12 numbers are comparable.
- **`finished`/`repetitive` are fractions**, `new` is a **count** (it is `None` without a training set).
- **`texts` is keyed by prompt**: duplicate prompts would overwrite each other.
- **No new test** covers `quick_checks` directly in this batch (it relies on `generate`, `is_repetitive` and `distinct_2`, which Day 10.2 tests).

## 5. Quick summary in one paragraph

`generation.py` is unchanged except for one new function at the end, `quick_checks`, which generates text for a list of prompts with one setting and returns the task 6 automatic items — the finished fraction, the repetitive fraction, distinct-2, the count of new sentences (optional) and the texts — so that both Day 11 and Day 12 can use the same code.
