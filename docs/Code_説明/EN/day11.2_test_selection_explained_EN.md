# Day 11.2 — test_selection.py — Simple Line-by-Line Explanation (English)

> **How the three Day 11 files fit together**
> - **Day 11.1 `selection.py`** — the code being tested (**read it first**).
> - **Day 11 `day11_size_trials.py`** — the driver.
> - **Day 11.2 (this file) `test_selection.py`** — **6 automatic tests**, CPU only, **instant** (no model, no training; just small numbers).
>
> pytest basics were explained in Days 6.2–10.2; only new things are explained here.

---

## 0. The big picture (read this first)

The decision rule is the most "argued-about" part of Day 11 — in the faculty review someone will ask *"why did you choose this size?"*. These tests prove the program applies the rule **exactly as written**, with tiny made-up runs where the right answer can be worked out by hand:

| Promise | Test |
|---|---|
| The summary finds the best point, the gap, the rise after best, the median speed, peak memory, elapsed time | `test_summary_reads_best_point_gap_and_rise` |
| A clearly better (lower-loss) bigger model wins | `test_lowest_loss_wins_when_clearly_better` |
| Within the tolerance, the **smaller** model wins | `test_smaller_model_wins_within_tolerance` |
| A run failing the task 6 checks is **never** chosen, even with the best loss | `test_runs_failing_the_task6_checks_are_never_chosen` |
| If nobody passes, there is **no** choice | `test_no_candidate_gives_no_choice` |
| Same size: the lower loss wins | `test_same_size_tie_prefers_lower_loss_then_no_dropout` |

Run (as before): `.venv\Scripts\python.exe -m pytest tests -q`

---

## 1. Setup and the helper `run`

```python
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.selection import RunSummary, choose, summarize_run  # noqa: E402
from src.training import LOG_COLUMNS  # noqa: E402
```
- Usual setup. It imports the code under test (**Day 11.1**) and `LOG_COLUMNS` from **Day 9.2** — the 12 column names of `log.csv`, so the fake log has exactly the real format.

```python
def run(name: str, params: int, best: float, finished: float = 1.0, repetitive: float = 0.0,
        dropout: float = 0.0) -> RunSummary:
    return RunSummary(name=name, size=name.split("_")[0], dropout=dropout, parameters=params, best_val=best,
                      best_step=100, train_at_best=1.0, final_val=best + 0.1, final_step=800, tokens_per_sec=None,
                      peak_vram_mb=None, elapsed_s=10.0, finished_rate=finished, repetitive_rate=repetitive,
                      distinct2=0.5)
```
- A **factory for fake runs**. You say only what matters (name, parameters, best validation loss, and optionally finished/repetitive rates and dropout); everything else gets a harmless default. `name.split("_")[0]` takes the size from a name like `tiny_d00`. By default a fake run **passes** the checks (finished 100 %, repetitive 0 %).

---

## 2. The tests

### 2.1 `test_summary_reads_best_point_gap_and_rise`

```python
def test_summary_reads_best_point_gap_and_rise(tmp_path: Path) -> None:
    log = tmp_path / "log.csv"
    rows = [
        {"step": 0, "event": "eval", "val_loss": "7.40"},
        {"step": 25, "event": "train", "train_loss": "3.00", "tokens_per_sec": "100"},
        {"step": 25, "event": "eval", "val_loss": "2.80"},
        {"step": 50, "event": "train", "train_loss": "1.50", "tokens_per_sec": "300", "peak_vram_mb": "400"},
        {"step": 50, "event": "eval", "val_loss": "2.60"},
        {"step": 75, "event": "train", "train_loss": "1.20", "tokens_per_sec": "200"},
        {"step": 75, "event": "eval", "val_loss": "2.90", "elapsed_s": "12.5"},
    ]
    with log.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in LOG_COLUMNS})
```
- Builds a **fake training log** in a temporary folder: seven rows telling a small story (validation loss 7.40 → 2.80 → **2.60** → 2.90: it gets best at step 50, then **rises**). Each row only mentions the columns that matter; `{k: r.get(k, "") for k in LOG_COLUMNS}` fills **every other column with an empty string**, so the file has all 12 columns, as the real one.

```python
    s = summarize_run(log, "tiny_d00", "tiny", 0.0, 1000)
    assert (s.best_step, s.best_val, s.train_at_best) == (50, 2.60, 1.50)
    assert s.gap == pytest.approx(1.10)
    assert (s.final_step, s.final_val) == (75, 2.90)
    assert s.rise_after_best == pytest.approx(0.30)
    assert s.tokens_per_sec == 200 and s.peak_vram_mb == 400 and s.elapsed_s == 12.5
```
- Run the summary and check each number **by hand-calculation**:
  - **best point:** step 50, validation loss 2.60, and the training loss at that step 1.50;
  - **gap** = 2.60 − 1.50 = **1.10**;
  - **final point:** step 75, loss 2.90; **rise after best** = 2.90 − 2.60 = **0.30**;
  - **median speed:** the speeds are 100, 300, 200 → sorted 100, 200, 300 → the middle is **200**;
  - **peak memory** 400 (the only value present), **elapsed** 12.5 (the largest elapsed time in the log).

### 2.2 `test_lowest_loss_wins_when_clearly_better`

```python
def test_lowest_loss_wins_when_clearly_better() -> None:
    d = choose([run("tiny_d00", 1_000_000, 2.70), run("small_d00", 5_000_000, 2.60)])
    assert d.chosen.name == "small_d00"
```
- tiny 2.70 vs small 2.60: the difference is **0.10**, far more than the 0.02 tolerance, so only small is "as good" → **small wins** even though it is bigger. (The rule does not always prefer small models — only when the difference is within noise.)

### 2.3 `test_smaller_model_wins_within_tolerance`

```python
def test_smaller_model_wins_within_tolerance() -> None:
    d = choose([run("tiny_d00", 1_000_000, 2.615), run("small_d00", 5_000_000, 2.600),
                run("medium_d00", 15_000_000, 2.640)])
    assert d.chosen.name == "tiny_d00"            # 0.015 worse than the best: within 0.02, and 5x smaller
    assert set(d.as_good) == {"tiny_d00", "small_d00"}
    assert "smaller" in d.reason
```
- The lowest loss is small's 2.600. tiny is 0.015 worse (**within 0.02**), medium is 0.040 worse (**outside**).
- **tiny is chosen** (5× fewer parameters than small); the "as good" set is exactly `{tiny, small}` (compared as **sets**, so the order does not matter); and the reason sentence mentions the **smaller** model — the explanation that will be printed in the report.

### 2.4 `test_runs_failing_the_task6_checks_are_never_chosen`

```python
def test_runs_failing_the_task6_checks_are_never_chosen() -> None:
    d = choose([run("tiny_d00", 1_000_000, 2.50, finished=0.80), run("small_d00", 5_000_000, 2.60),
                run("medium_d00", 15_000_000, 2.55, repetitive=0.10)])
    assert d.chosen.name == "small_d00"
    assert set(d.rejected) == {"tiny_d00", "medium_d00"}
```
- tiny has the **best loss (2.50)** but finishes only 80 % of sentences (< 90 %) → **rejected**; medium has a good loss (2.55) but 10 % repetitive (> 5 %) → **rejected**. Only small passes → **small is chosen**, although its loss is the worst. `d.rejected` lists exactly the two failing runs.
- The point: **generation quality is a hard filter before loss is even compared**.

### 2.5 `test_no_candidate_gives_no_choice`

```python
def test_no_candidate_gives_no_choice() -> None:
    d = choose([run("tiny_d00", 1_000_000, 2.50, finished=0.5)])
    assert d.chosen is None and "no run passed" in d.reason
```
- The only run fails (50 % finished) → **no candidates → `chosen` is `None`** and the reason says "no run passed…". The driver then reports "No run chosen" and writes no decision file, instead of silently picking something bad.

### 2.6 `test_same_size_tie_prefers_lower_loss_then_no_dropout`

```python
def test_same_size_tie_prefers_lower_loss_then_no_dropout() -> None:
    d = choose([run("tiny_d10", 1_000_000, 2.605, dropout=0.1), run("tiny_d00", 1_000_000, 2.600)])
    assert d.chosen.name == "tiny_d00"
```
- Two runs of the **same size** (same parameters): with dropout 0.1 the loss is 2.605, without dropout 2.600. Both are within the tolerance and equal in size, so the tie-break (Day 11.1 §5) goes to the **lower validation loss** → `tiny_d00`.
- (The test's *name* also mentions "then no dropout" — the further tie-break when the losses are exactly equal. This test only checks the first part, where the lower loss decides.)

---

## 3. How many tests?

**6 tests**, expected `6 passed` (instantly).

## 4. What these tests do *not* cover

- **The exact boundaries**: a loss exactly 0.02 above the best (the rule uses `<=`), finished exactly 90 %, repetitive exactly 5 %.
- A run with **`None` rates** (no generation check) — `passes_checks` returns `False`, but it is not tested directly here (only the "no candidate" case with low rates).
- The **exact-tie dropout rule** (equal parameters *and* equal loss → no dropout).
- `generation_check`, the figure, the report and the decision file (they need models and files; they are exercised by running the script).

## 5. Quick summary in one paragraph

The 6 instant tests build tiny fake runs and a fake log to prove that the summary computes the best point, gap, rise, median speed and memory correctly, and that the decision rule picks the lowest-loss run when it is clearly better, prefers the smaller model within the 0.02 tolerance, never chooses a run that fails the finished/repetitive checks, returns no choice when nobody passes, and breaks same-size ties by the lower loss.
