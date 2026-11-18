# Day 11.1 — selection.py — Simple Line-by-Line Explanation (English)

> **How the three Day 11 files fit together**
> - **Day 11.1 (this file) `selection.py`** — the **decision logic**: summarise a training run, then apply a **written rule** to choose the model.
> - **Day 11 `day11_size_trials.py`** — the *driver*: trains every (size, dropout) pair, checks generation, calls this file to decide, writes the report.
> - **Day 11.2 `test_selection.py`** — 6 automatic tests for the decision rule.
>
> Suggested reading order: **11.1 → 11 → 11.2**.

---

## 0. The big picture (read this first)

On Day 8 you made **three model sizes** (tiny / small / medium). On Day 11 you train all of them (and with dropout 0.0 and 0.1 = **6 runs**) with the **same budget**, and then you must **choose one** for the main training. How do you choose *fairly*?

The dangerous way: look at all the numbers, then pick the one you like and invent a reason. The honest way: **write the rule BEFORE you look at the results**, and let a program apply it. That is what this file is.

### The rule (from the docstring)

| Step | Rule | Why |
|---|---|---|
| 1 | A run is a **candidate** only if its best checkpoint passes the **task 6 automatic guidelines** on the fixed prompts: **finished ≥ 90 %**, **repetitive ≤ 5 %** | A model with a great loss that cannot finish a sentence or loops is no good |
| 2 | Among candidates, find the **lowest validation loss** | Validation = text the model did not train on |
| 3 | Every candidate within **0.02** of that loss is "**as good**" | A difference below 0.02 (about **2 %** in perplexity) is within run-to-run noise: no proof the bigger model is better |
| 4 | Among those, take the one with the **fewest parameters** (tie → lower validation loss) | Smaller = faster, cheaper, and memorises less |

### A tiny example

| Run | Parameters | Best val loss | Passes checks? |
|---|---|---|---|
| tiny | 1.0M | 2.615 | yes |
| small | 5.2M | 2.600 | yes |
| medium | 14.8M | 2.640 | yes |

Lowest = 2.600 (small). "As good" = within 0.02 → tiny (0.015 worse) and small. **Fewest parameters → tiny is chosen**, even though small has the lowest loss.

---

## 1. Words you need to know (mini dictionary)

| Word | Simple meaning |
|---|---|
| **Validation loss** | Loss on held-out text (Day 5). Lower = better generalisation. |
| **Best step** | The step where the validation loss was lowest. |
| **Overfitting** | The model starts memorising the training text: training loss keeps falling, validation loss stops improving or rises. |
| **Gap** | Validation loss − training loss at the best step. A big gap = it handles unseen wording much worse than what it practised. |
| **Rise after best** | Final validation loss − best validation loss: how much it got worse after its best point (0 = never rose). |
| **Dropout** | Randomly switching off some numbers during training to reduce memorising (Day 8.1). |
| **Perplexity** | `exp(loss)` (Day 9.2). A loss difference of 0.02 ≈ 2 % in perplexity. |
| **Tolerance** | How big a difference we still treat as "no real difference" (0.02). |
| **Median** | The middle value of a sorted list (more robust than the average). |

---

## 2. The docstring and constants

```python
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path

TOLERANCE: float = 0.02
MIN_FINISHED: float = 0.90
MAX_REPETITIVE: float = 0.05
```
- Standard imports. The three **numbers of the rule** are named constants at the top, so the rule is visible in one place: tolerance 0.02, at least 90 % finished, at most 5 % repetitive.

---

## 3. `RunSummary` — the facts about one run

```python
@dataclass
class RunSummary:
    name: str
    size: str
    dropout: float
    parameters: int
    best_val: float
    best_step: int
    train_at_best: float | None
    final_val: float
    final_step: int
    tokens_per_sec: float | None       # median over the run
    peak_vram_mb: float | None
    elapsed_s: float
    finished_rate: float | None = None  # filled in after the generation check
    repetitive_rate: float | None = None
    distinct2: float | None = None
    notes: list[str] = field(default_factory=list)
```

| Field | Meaning |
|---|---|
| `name`, `size`, `dropout` | Which run (`tiny_d00` = tiny, dropout 0.0) |
| `parameters` | Number of learnable numbers (from the formula, Day 8.1) |
| `best_val`, `best_step` | Lowest validation loss and the step where it happened |
| `train_at_best` | The training loss logged at that step (`None` if missing) |
| `final_val`, `final_step` | The last validation result |
| `tokens_per_sec` | Typical speed (median over the run) |
| `peak_vram_mb` | Peak GPU memory |
| `elapsed_s` | Total training time |
| `finished_rate`, `repetitive_rate`, `distinct2` | Generation-check results (task 6 items A2–A4). They start as `None` and are **filled in later** by Day 11's driver |
| `notes` | Free-text notes (`default_factory=list`: its own list per object, as in Day 5) |

```python
    @property
    def gap(self) -> float | None:
        """Validation minus training loss at the best step: how much worse unseen wording is."""
        return None if self.train_at_best is None else self.best_val - self.train_at_best

    @property
    def rise_after_best(self) -> float:
        """How much the validation loss rose from its best to the end (0 = never rose)."""
        return self.final_val - self.best_val
```
- Two **calculated values** (`@property`: used without parentheses, as in Day 8.1):
  - `gap` = validation − training loss at the best step (`None` if there is no training loss);
  - `rise_after_best` = final − best validation loss.
- Example: best 2.60, training 1.50 → gap 1.10; final 2.90 → rise 0.30 (it overfitted after the best point).

```python
    def passes_checks(self) -> bool:
        if self.finished_rate is None or self.repetitive_rate is None:
            return False
        return self.finished_rate >= MIN_FINISHED and self.repetitive_rate <= MAX_REPETITIVE

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gap"], d["rise_after_best"] = self.gap, self.rise_after_best
        return d
```
- `passes_checks` — **step 1 of the rule**: if the generation check has not been done (rates are `None`) → **fail** (no check = not a candidate). Otherwise: finished ≥ 90 % **and** repetitive ≤ 5 %.
- `to_dict` — the summary as a dictionary, **plus** the two calculated values (they are properties, so `asdict` would not include them).

---

## 4. `summarize_run`

```python
def _f(x: str) -> float | None:
    return float(x) if x not in ("", None) else None
```
- A tiny helper: text → number, or `None` for an empty cell (CSV cells are text; empty means "no value").

```python
def summarize_run(log_path: Path, name: str, size: str, dropout: float, parameters: int) -> RunSummary:
    with log_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    evals = [(int(r["step"]), float(r["val_loss"])) for r in rows if r["event"] == "eval" and r["val_loss"]]
    if not evals:
        raise ValueError(f"{log_path}: no validation results")
    trains = {int(r["step"]): float(r["train_loss"]) for r in rows if r["event"] == "train" and r["train_loss"]}
```
- Read the run's `log.csv` (the Day 9.2 format).
- `evals` — a list of `(step, validation loss)` for every **eval** row. No evaluations → `ValueError`.
- `trains` — a dictionary **step → training loss** from the **train** rows (same as Day 9's compare script).

```python
    best_step, best_val = min(evals, key=lambda e: (e[1], e[0]))
    final_step, final_val = evals[-1]
```
- **Best point:** the evaluation with the **lowest loss** (`e[1]`); if two are equal, the **earlier step** (`e[0]`). The key `(loss, step)` is compared as a pair.
- **Final point:** the last evaluation (`evals[-1]`).

```python
    tps = sorted(v for v in (_f(r["tokens_per_sec"]) for r in rows if r["event"] == "train") if v)
    vram = [v for v in (_f(r["peak_vram_mb"]) for r in rows) if v is not None]
    elapsed = max((float(r["elapsed_s"]) for r in rows if r["elapsed_s"]), default=0.0)
```
- `tps` — the speeds (tokens/s) of the train rows, **dropping empty/zero values** (`if v`), sorted — so the **median** can be taken.
- `vram` — all GPU-memory values that exist (CPU runs have none).
- `elapsed` — the largest elapsed time in the log (= total time), `0.0` if none.

```python
    return RunSummary(
        name=name, size=size, dropout=dropout, parameters=parameters, best_val=best_val, best_step=best_step,
        train_at_best=trains.get(best_step), final_val=final_val, final_step=final_step,
        tokens_per_sec=tps[len(tps) // 2] if tps else None, peak_vram_mb=max(vram) if vram else None,
        elapsed_s=elapsed,
    )
```
- Build the summary: the training loss **at the best step** (`trains.get(best_step)`, `None` if absent), the **median speed** (`tps[len(tps)//2]` = the middle item), the **peak** memory (`max`), and the elapsed time.

---

## 5. `Decision` and `choose` — the rule in code

```python
@dataclass
class Decision:
    chosen: RunSummary | None
    lowest_val: float | None
    as_good: list[str]
    rejected: dict[str, str]
    reason: str
```
- The outcome: the **chosen run** (or `None`), the lowest validation loss among candidates, the names of the runs that count as "as good", the **rejected** runs with the reason, and a **sentence explaining the decision**.

```python
def choose(runs: list[RunSummary], tolerance: float = TOLERANCE) -> Decision:
    rejected = {r.name: (f"finished {r.finished_rate:.0%}, repetitive {r.repetitive_rate:.0%}"
                         if r.finished_rate is not None else "no generation check")
                for r in runs if not r.passes_checks()}
    candidates = [r for r in runs if r.passes_checks()]
    if not candidates:
        return Decision(None, None, [], rejected, "no run passed the task 6 automatic guidelines")
```
- **Step 1 (candidates):** `rejected` — a dictionary built by a comprehension: for every run that fails the checks, its name and a short reason (`:.0%` formats a fraction as a percent, e.g. `finished 80%, repetitive 10%`; or "no generation check"). `candidates` — the runs that pass.
- If **nobody** passes → no choice (`chosen = None`) with an explanatory reason. (The driver then writes "No run chosen" and no decision file.)

```python
    lowest = min(r.best_val for r in candidates)
    as_good = [r for r in candidates if r.best_val - lowest <= tolerance]
    chosen = min(as_good, key=lambda r: (r.parameters, r.best_val, r.dropout))
```
- **Step 2:** the lowest validation loss among candidates.
- **Step 3:** `as_good` = candidates whose loss is **at most `tolerance` above** the lowest (`<=`, so exactly 0.02 counts).
- **Step 4:** `min(..., key=(parameters, best_val, dropout))` — compare by **parameters first** (fewest wins), then by **lower validation loss**, then by **lower dropout** (an exact tie prefers no dropout: simpler).

```python
    best_overall = min(candidates, key=lambda r: r.best_val)
    if chosen is best_overall:
        reason = f"{chosen.name} has the lowest validation loss ({chosen.best_val:.4f})"
        if len(as_good) > 1:
            reason += " and is also the smallest of the runs within the tolerance"
    else:
        reason = (f"{best_overall.name} has the lowest validation loss ({best_overall.best_val:.4f}), but "
                  f"{chosen.name} ({chosen.best_val:.4f}) is within {tolerance} of it with fewer parameters "
                  f"({chosen.parameters:,} vs {best_overall.parameters:,}), so the smaller model is chosen")
    return Decision(chosen, lowest, [r.name for r in as_good], rejected, reason)
```
- Write the **explanation sentence**: if the chosen run is also the one with the lowest loss, say so (and that it is also the smallest of the near-equal runs, if there were several); otherwise explain **"X has the lowest loss, but Y is within 0.02 with fewer parameters, so the smaller model is chosen"** (`chosen is best_overall` compares the *same object*).
- Return the `Decision`. This sentence goes straight into the report — a ready-made justification for the faculty review.

---

## 6. Small things worth knowing

- **The rule is the point.** It is written *before* the results, so the choice cannot be bent to taste. The report prints the rule next to the result.
- **A run with the best loss can be rejected** by step 1 (e.g. loss 2.50 but it finishes only 80 % of sentences).
- **The 0.02 tolerance is an assumption**, stated in the docstring ("within run-to-run noise"). The driver trains **each configuration once (one seed)**, so the noise is not measured here; 0.02 is a reasonable, conservative guess.
- **`gap` uses an *interval average* of the training loss** (Day 9.2 logs the mean over the last `log_every` steps, with dropout active if dropout > 0), so it is an approximate measure of memorising, good for comparing runs, not an exact number.
- **Dropout only matters through the loss**: the rule does not prefer or avoid dropout except as a last tie-break.
- **The test split is not used** anywhere here — selection is by validation loss only.
- **`passes_checks` returning `False` when rates are `None`** is deliberate: forgetting the generation check can never let a run through.

## 7. Quick summary in one paragraph

`selection.py` summarises a run's log (best validation loss and step, the training loss there, the gap, the rise after the best point, speed and memory) and applies a rule written in advance: keep only runs whose best checkpoint finishes ≥ 90 % of sentences with ≤ 5 % repetition, treat every run within 0.02 of the lowest validation loss as equally good, and choose the one with the fewest parameters — returning the choice together with a plain-language reason.
