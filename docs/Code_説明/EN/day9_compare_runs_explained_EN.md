# Day 9 — day9_compare_runs.py — Simple Line-by-Line Explanation (English)

> **How the five Day 9 files fit together**
> - **Day 9.1 `batching.py`** and **Day 9.2 `training.py`** — batching and the training loop.
> - **Day 9 `day9_train.py`** — starts/resumes a run.
> - **Day 9 (this file) `day9_compare_runs.py`** — the **proof** that stop-and-resume works: it compares two finished runs.
> - **Day 9.3 `test_training.py`** — automatic tests (it contains the same proof as a unit test).

---

## 0. The big picture (read this first)

You trained the same model twice:

| Run | What happened |
|---|---|
| **a** (`smoke_a`) | trained straight through, never stopped |
| **b** (`smoke_b`) | trained until `stop_after` (say step 300), **stopped**, then **resumed** to the end |

If resuming is implemented correctly, **b must look exactly like a**: the same loss at every step. If a checkpoint forgot something (optimizer memory, random state, ...), the two curves would drift apart. This script reads both `log.csv` files and **checks four things**:

1. both runs reached the same final step;
2. the **training loss** at every logged step is the same in both;
3. the **validation losses** are the same;
4. the curve of run b has **no jump** at the resume step.

It then draws a picture (`docs/figures/day9_smoke_loss.png`) and writes a report (`docs/DAY9_SMOKE_REPORT.md`) with the learning results and the sample texts of the first and last evaluation.

Run it:

```
.venv\Scripts\python.exe scripts\day9_compare_runs.py --a smoke_a --b smoke_b
```

---

## 1. Words you need to know

| Word | Simple meaning |
|---|---|
| **Uninterrupted run** | A run that was never stopped (the reference). |
| **Resume step** | The step at which run b was stopped (its `stop_after`). |
| **Tolerance** | How big a difference we still accept as "the same". |
| **Matplotlib** | The Python library used to draw the figure. |
| **Backend `Agg`** | Draws straight into an image file, without opening a window. |
| **Perplexity** | `exp(loss)` (see Day 9.2). |

---

## 2. The docstring, imports and constants

```python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

SAME_TOL: float = 1e-6        # "identical"
CONTINUOUS_TOL: float = 0.02  # "the same curve" (GPU arithmetic may differ in the last digits)
```
- Standard imports (plus `re` for regular expressions, as in Day 5). `matplotlib` is imported later, inside `main`.
- Two tolerances:
  - `SAME_TOL = 1e-6` — a difference below this = **"identical"** (typical on the CPU);
  - `CONTINUOUS_TOL = 0.02` — a difference below this = **"the same curve"** (on a GPU, arithmetic may differ in the last digits and the tiny differences can grow a little over many steps).

---

## 3. Helper functions

### 3.1 `read_log`

```python
def read_log(path: Path) -> tuple[dict[int, float], dict[int, float], list[dict]]:
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    train = {int(r["step"]): float(r["train_loss"]) for r in rows if r["event"] == "train" and r["train_loss"]}
    val = {int(r["step"]): float(r["val_loss"]) for r in rows if r["event"] == "eval" and r["val_loss"]}
    return train, val, rows
```
- Reads a run's `log.csv` (the 12-column file from Day 9.2) as a list of dictionaries.
- `train` — a dictionary **step → training loss**, from rows whose `event` is `"train"` **and** whose loss cell is not empty (`and r["train_loss"]`, an empty string counts as false). `val` — the same for `"eval"` rows and validation loss.
- Returns the two dictionaries and the raw rows.

### 3.2 `sample_sections`

```python
def sample_sections(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [s.strip() for s in re.split(r"\n(?=## step )", text) if s.strip().startswith("## step")]
```
- Splits `samples.md` into one section per evaluation. `re.split(r"\n(?=## step )", text)` cuts at each line break that is **followed by** `## step ` (the `(?=…)` "lookahead" looks ahead without removing those characters, so each section keeps its heading). Keep only the parts that really start with `## step`.

---

## 4. `main` — the comparison

### 4.1 Options and loading

```python
    ap = argparse.ArgumentParser(description="Day 9: compare an uninterrupted and a resumed run.")
    ap.add_argument("--a", required=True, help="run name of the uninterrupted run")
    ap.add_argument("--b", required=True, help="run name of the stopped-and-resumed run")
    ap.add_argument("--runs-root", default="runs/day9")
    ap.add_argument("--figure", type=Path, default=REPO_ROOT / "docs/figures/day9_smoke_loss.png")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY9_SMOKE_REPORT.md")
    a = ap.parse_args()
```
- `--a`, `--b` (**required**) — the two run names; `--runs-root`, `--figure`, `--report` — locations with defaults.

```python
    dir_a, dir_b = REPO_ROOT / a.runs_root / a.a, REPO_ROOT / a.runs_root / a.b
    train_a, val_a, rows_a = read_log(dir_a / "log.csv")
    train_b, val_b, _ = read_log(dir_b / "log.csv")
    cfg_a = json.loads((dir_a / "train_config.json").read_text(encoding="utf-8"))
    cfg_b = json.loads((dir_b / "train_config.json").read_text(encoding="utf-8"))
```
- Load both logs and both saved settings. (`a.a` is the run name from `--a`; `a` is the argparse namespace.)

```python
    resume_step = cfg_b.get("stop_after")
    if resume_step is None:
        raise SystemExit(f"{a.b} was not stopped early (no stop_after in its train_config.json)")
    keys = [k for k in cfg_a if k not in {"run_dir", "stop_after", "resume", "overwrite", "machine", "device"}]
    differing = [k for k in keys if cfg_a.get(k) != cfg_b.get(k)]
    if differing:
        raise SystemExit(f"the two runs use different settings: {differing}")
```
- `resume_step` — the step where run b was stopped, read from **b's saved settings** (a resume keeps the original `train_config.json`, so `stop_after` stays recorded). If it is missing, run b was not an interrupted run → stop.
- **Fair-test check:** every setting **except** the ones allowed to differ (`run_dir`, `stop_after`, `resume`, `overwrite`, `machine`, `device`) must be **identical** in both runs; otherwise the comparison is meaningless → stop and list the differing keys.

### 4.2 The numbers

```python
    common = sorted(set(train_a) & set(train_b))
    diffs = [abs(train_a[s] - train_b[s]) for s in common]
    vcommon = sorted(set(val_a) & set(val_b))
    vdiffs = [abs(val_a[s] - val_b[s]) for s in vcommon]
    max_diff = max(diffs + vdiffs) if diffs else float("nan")
    final_a, final_b = max(train_a), max(train_b)
    before = max(s for s in train_b if s <= resume_step)
    after = min(s for s in train_b if s > resume_step)
    jump_b = abs(train_b[after] - train_b[before])
    jump_a = abs(train_a.get(after, math.nan) - train_a.get(before, math.nan))
```
- `common` — logged steps present in **both** runs (`&` = set intersection). `diffs` — the **absolute difference** of the training loss at each of them. `vcommon` / `vdiffs` — the same for validation losses.
- `max_diff` — the **largest difference** over everything (training and validation); `NaN` if nothing is common.
- `final_a`, `final_b` — the last logged step of each run.
- **The jump check:** `before` = the last logged step **at or before** the resume step in run b; `after` = the first logged step **after** it. `jump_b` = how much the loss changed between those two steps in run b (which straddles the stop/resume). `jump_a` = the change over the **same two steps** in the uninterrupted run (`.get(..., math.nan)` avoids a crash if a step is missing). If resuming is clean, the two jumps are the same.

### 4.3 The four checks and the verdict

```python
    checks = {
        "same final step": final_a == final_b == cfg_a["max_steps"],
        "training losses match": max(diffs) < CONTINUOUS_TOL,
        "validation losses match": max(vdiffs) < CONTINUOUS_TOL,
        "no jump at the resume step": abs(jump_b - jump_a) < CONTINUOUS_TOL,
    }
    verdict = "identical" if max_diff < SAME_TOL else ("the same curve" if all(checks.values()) else "DIFFERENT")
```
- The four checks as a dictionary of True/False:
  1. both runs ended at the **configured `max_steps`** (a chained comparison `a == b == c`);
  2. biggest training-loss difference below `0.02`;
  3. biggest validation-loss difference below `0.02`;
  4. the loss jump at the resume step is the same in both runs (within `0.02`).
- **Verdict:** differences below `1e-6` → **"identical"**; otherwise, if all four checks pass → **"the same curve"** (typical on GPU); otherwise **"DIFFERENT"**.

---

## 5. The figure

```python
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a.figure.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
```
- Import matplotlib **here** (only needed for the figure; **it must be installed** — it was not in Day 1's package list). `matplotlib.use("Agg")` picks the **file-only backend**: no window opens (important on servers). Make sure the output folder exists; create a figure (`fig`) with one set of axes (`ax`), 8 × 4.5 inches at 150 dots per inch.

```python
    ax.plot(list(train_a), list(train_a.values()), lw=2.4, color="#9db7d5", label=f"training loss, {a.a} (no stop)")
    ax.plot(list(train_b), list(train_b.values()), lw=1.2, ls="--", color="#1f4e79",
            label=f"training loss, {a.b} (stopped and resumed)")
    ax.plot(list(val_a), list(val_a.values()), "o", ms=6, color="#e3a8a0", label=f"validation loss, {a.a}")
    ax.plot(list(val_b), list(val_b.values()), "x", ms=6, color="#a33a2c", label=f"validation loss, {a.b}")
```
- Four lines/markers: x = steps, y = losses.
  - run a's training loss: **thick light-blue line**;
  - run b's training loss: **thin dashed dark-blue line** drawn **on top** — if the two runs are identical, the dashed line sits exactly on the thick one;
  - validation losses: circles (a) and crosses (b).

```python
    ax.axvline(resume_step, color="gray", lw=1, ls=":")
    ax.text(resume_step, ax.get_ylim()[1] * 0.97, f"  stopped / resumed at step {resume_step}", va="top",
            fontsize=8, color="gray")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (cross-entropy)")
    ax.set_title("Day 9 smoke run: uninterrupted vs stopped and resumed")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.figure)
    plt.close(fig)
```
- A dotted vertical line at the resume step with a label near the top (`ax.get_ylim()[1] * 0.97` = 97% of the top of the y-axis), axis labels, title, light grid, legend; `tight_layout` avoids clipping; `savefig` writes the PNG; `plt.close` frees memory.

---

## 6. The report

```python
    train_meta = json.loads((REPO_ROOT / "data/tokens/train_meta.json").read_text(encoding="utf-8"))
    last_row = [r for r in rows_a if r["event"] == "train"][-1]
    tps = [float(r["tokens_per_sec"]) for r in rows_a if r["event"] == "train" and r["tokens_per_sec"]]
    epochs = int(last_row["tokens_seen"]) / int(train_meta["tokens"])
    secs = sample_sections(dir_a / "samples.md")
    fig_rel = a.figure.resolve().relative_to(a.report.resolve().parent).as_posix()
```
- Data for the report (from run a):
  - the Day 7 meta (number of training tokens);
  - `last_row` — the **last training row** (`[-1]` = last item): tokens seen so far, GPU memory, ...;
  - `tps` — all tokens-per-second values (non-empty);
  - `epochs` = tokens seen ÷ training tokens = **how many passes over the training text**;
  - `secs` — the sample sections;
  - `fig_rel` — the figure path **relative to the report's folder**, with `/` slashes (`as_posix()`), so the Markdown image link works.

The Markdown `lines` list contains:

| Section | Content |
|---|---|
| Header | date; which runs; model config, batch × accum, steps, peak learning rate, warm-up |
| Figure | `![loss curves](…)` |
| **Resume check** | A table: each of the 4 checks → PASS/FAIL, the **largest difference**, the **verdict** |
| **Learning** | A table "Step 0 vs Final": validation loss, **validation perplexity**, training loss. `val_a[min(val_a)]` = the loss at the **earliest** evaluation (step 0, before training) and `val_a[max(val_a)]` = the **latest** |
| Training facts | tokens seen and passes over the data; **median** speed (`sorted(tps)[len(tps)//2]`); peak GPU memory or "- (CPU run)" |
| **Greedy samples** | The first and last evaluation's sample sections of run a (`[secs[0], "", secs[-1]]`; or all of them if fewer than two) |

- `*(… for k, v in checks.items())` unpacks one table row per check (same trick as earlier days).

```python
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text("\n".join(lines), encoding="utf-8", newline="\n")
```
- Save the report (UTF-8, LF line endings).

---

## 7. Printing and the exit code

```python
    print(f"Final step: {a.a} {final_a}, {a.b} {final_b} (resumed at {resume_step})")
    for k, v in checks.items():
        print(f"  {k:<28} {'PASS' if v else 'FAIL'}")
    print(f"Largest difference between the runs: {max_diff:.2e} -> {verdict}")
    print(f"Validation loss: {val_a[min(val_a)]:.4f} (step 0) -> {val_a[max(val_a)]:.4f} (step {max(val_a)})")
    print(f"Figure: {a.figure}\nReport: {a.report}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
```
- Print the final steps, a PASS/FAIL line per check (`:<28` = left-aligned in 28 characters), the largest difference with the verdict, the validation loss from step 0 to the end, and the file locations.
- **Exit code 0 only if all four checks pass** (else 1) — usable as an automatic check.

---

## 8. How to read the results

- **Verdict "identical"** (largest difference < 1e-6): the resumed run reproduces the uninterrupted run exactly. Typical on the CPU.
- **"the same curve"**: the differences are tiny (< 0.02) and all four checks pass. Typical on a GPU.
- **"DIFFERENT"**: resuming is broken (or the runs had different settings/data). Look at which check failed.
- **The figure:** the dashed line should lie on top of the thick line; the crosses should sit inside the circles.
- **Learning table:** validation loss should fall from about `ln(vocab)` at step 0; perplexity falls steeply.
- **Samples:** the first evaluation's text is nonsense; the last should look like a question.

## 9. Small things worth knowing

- **It compares the logs, not the checkpoints.** The log is what both runs wrote, so equal logs mean equal training.
- **Stop at a multiple of `log_every`** (see Day 9.2 §7). Otherwise the first logged interval after resuming averages fewer steps, so a few `train_loss` values can differ from run a even though the training is identical.
- **If there are no common logged steps**, `max(diffs)` fails with a Python error; with the normal smoke setup this does not happen.
- **Needs matplotlib** (`pip install matplotlib` into the venv); it is not in Day 1's checked package list.
- **GPU runs** may differ in the last digits (non-deterministic GPU maths), which is why the tolerances exist.

## 10. Quick summary in one paragraph

The script reads the logs of an uninterrupted run and a stopped-and-resumed run, checks that they reached the same final step, that training and validation losses agree at every logged step and that there is no jump at the resume step, labels the result "identical" / "the same curve" / "DIFFERENT", draws both curves on one figure, and writes a Markdown report with the learning results and the first and last sample texts, exiting with 0 only if everything passes.
