# Day 12 — day12_monitor.py — Simple Line-by-Line Explanation (English)

> **How the five Day 12 files fit together**
> - **Day 12 `day12_train_main.py`** — the main training (3 seeds, early stopping, final model).
> - **Day 12 (this file) `day12_monitor.py`** — a small **read-only dashboard**: shows how a running (or finished) training is doing.
> - **Day 12.1 `generation.py`**, **Day 12.2 `training.py`**, **Day 12.3 `test_training.py`** — updated library files (see their own explanations).
>
> This script never imports the training code: it only reads the files that the training writes.

---

## 0. The big picture (read this first)

Main training takes a while. You want to know: *"How far is it? Is the validation loss still improving? How long is left?"* — **without** disturbing it.

This script is like a **car dashboard**: you glance at it; it does not drive.

- It **only reads** `log.csv` and `train_config.json` (and `status.json` if present). It **never touches the training itself**, so it is **safe to run in a second terminal while training is going**.

```
.venv\Scripts\python.exe scripts\day12_monitor.py --run runs/day12/main_s42
.venv\Scripts\python.exe scripts\day12_monitor.py --run runs/day12/main_s42 --watch 5
```
(In VS Code open a second terminal with the **+** button. `--watch 5` refreshes every 5 seconds; **Ctrl+C** quits the monitor — not the training.)

### What the screen looks like

```
Run runs/day12/main_s42  [running]
  progress   [###########...................] step 150 / 400
  train loss 1.9521 (last)
  val loss   2.1034 at step 150   |   best 2.0877 (perplexity 8.07) at step 130
  early stop 20 / 100 steps without improvement
  speed      38,420 tokens/s   |   time left at most 1,640 s (less if early stopping triggers)
```
(Numbers are made up to show the layout.)

---

## 1. Words you need to know

| Word | Simple meaning |
|---|---|
| **Monitor** | A program that only observes. |
| **ETA** | "Estimated time of arrival": the estimated time left. |
| **Patience / early-stop counter** | How many steps since the last improvement, out of the allowed number (Day 12.2). |
| **ANSI escape code** | Special characters that control the terminal. `\033[2J` = clear the screen; `\033[H` = move the cursor to the top-left. |

---

## 2. Imports and constants

```python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
```
- Only standard modules (no `torch`!), so the monitor starts instantly. `time` is for the pause between refreshes; `math` for perplexity. `REPO_ROOT` as in earlier days.

---

## 3. `summary` — build the text to show

### 3.1 Reading the files

```python
def summary(run_dir: Path) -> str:
    log, cfg_path = run_dir / "log.csv", run_dir / "train_config.json"
    if not log.exists() or not cfg_path.exists():
        return f"{run_dir}: no log yet"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    with log.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    evals = [(int(r["step"]), float(r["val_loss"])) for r in rows if r["event"] == "eval" and r["val_loss"]]
    trains = [r for r in rows if r["event"] == "train" and r["train_loss"]]
    if not evals:
        return f"{run_dir}: started, no validation yet"
```
- If the log or the settings file do not exist yet (training has not started) → return a short message instead of crashing.
- Read the saved settings (`cfg`) and the log rows.
- `evals` — `(step, validation loss)` pairs from the **eval** rows; `trains` — the **train** rows that have a loss (kept as dictionaries).
- If there is **no validation yet** → "started, no validation yet" (the first evaluation happens at step 0, so this is very brief).

### 3.2 The numbers

```python
    step = max(int(r["step"]) for r in rows)
    best_step, best_val = min(evals, key=lambda e: (e[1], e[0]))
    last_val_step, last_val = evals[-1]
    tps = [float(r["tokens_per_sec"]) for r in trains[-5:] if r["tokens_per_sec"]]
    tokens_per_step = cfg["batch_size"] * cfg.get("grad_accum", 1) * 256
    left = cfg["max_steps"] - step
    eta = left * tokens_per_step / (sum(tps) / len(tps)) if tps and left > 0 else 0.0
```
- `step` — the **current step** = the largest step number in the log.
- `best_step, best_val` — the lowest validation loss and where it was (ties → earlier step; same as Day 11.1).
- `last_val_step, last_val` — the most recent validation result.
- `tps` — the **tokens/s of the last 5 training rows** (`trains[-5:]` = the last five), ignoring empty cells. A short window gives a *recent* speed.
- `tokens_per_step` = batch × accumulation × **256** (the block size, written in as a constant); `cfg.get("grad_accum", 1)` uses 1 if the key is missing.
- `left` = steps remaining; `eta` (seconds) = **tokens still to process ÷ tokens per second** (`sum(tps)/len(tps)` = the average speed). 0 if there is no speed yet or nothing left.

### 3.3 Status and progress bar

```python
    patience = cfg.get("early_stop_patience")
    status_file = run_dir / "status.json"
    status = json.loads(status_file.read_text(encoding="utf-8"))["status"] if status_file.exists() else "running"
    bar_n = int(30 * step / max(1, cfg["max_steps"]))
```
- `patience` — the early-stopping setting (`None` if it is off).
- `status` — from **`status.json`** (Day 12.2) if it exists (`finished` / `early_stopped` / `stopped`), otherwise `"running"`.
- `bar_n` — how many of the **30 bar characters** are filled: `30 × step ÷ max_steps` (`max(1, …)` avoids dividing by zero).

### 3.4 The lines

```python
    lines = [
        f"Run {run_dir.relative_to(REPO_ROOT).as_posix() if run_dir.is_relative_to(REPO_ROOT) else run_dir}  [{status}]",
        f"  progress   [{'#' * bar_n}{'.' * (30 - bar_n)}] step {step} / {cfg['max_steps']}",
        f"  train loss {float(trains[-1]['train_loss']):.4f} (last)" if trains else "  train loss -",
        f"  val loss   {last_val:.4f} at step {last_val_step}   |   best {best_val:.4f} "
        f"(perplexity {math.exp(best_val):.2f}) at step {best_step}",
    ]
```
- **Line 1:** the run folder (relative to the repo if possible, so no personal folder names) and its **[status]**.
- **Line 2:** the **progress bar**: `'#' * bar_n` filled characters + `'.' * (30 - bar_n)` empty ones, then "step X / max".
- **Line 3:** the **last training loss** (`trains[-1]` = the newest train row) — a **conditional expression** shows `train loss -` if there is none yet.
- **Line 4:** the **latest validation loss** and its step, and the **best** validation loss with its **perplexity** (`exp(best_val)`) and step.

```python
    if patience:
        waited = last_val_step - best_step
        lines.append(f"  early stop {waited} / {patience} steps without improvement"
                     + ("  -> will stop at the next check" if waited >= patience else ""))
    if tps:
        lines.append(f"  speed      {sum(tps) / len(tps):,.0f} tokens/s   |   time left at most {eta:,.0f} s "
                     f"(less if early stopping triggers)")
    return "\n".join(lines)
```
- If early stopping is on: `waited` = steps between the best point and the **latest evaluation** — the **early-stop counter** "waited / patience". When it reaches the patience the extra text appears (in practice you will usually see the status `early_stopped` straight afterwards).
- If a speed is known: show the **average recent speed** and the **maximum time left** — "at most" because early stopping may end the run sooner.
- Join the lines with line breaks and return the whole text.

---

## 4. `main`

```python
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description="Day 12: monitor a training run.")
    ap.add_argument("--run", required=True, help="run folder, for example runs/day12/main_s42")
    ap.add_argument("--watch", type=float, default=0, help="refresh every N seconds (0 = print once)")
    a = ap.parse_args()
    run_dir = (REPO_ROOT / a.run).resolve()
```
- Same console safety as before. Two options: `--run` (**required**: the run folder) and `--watch N` (seconds between refreshes; `0` = print once).

```python
    if not a.watch:
        print(summary(run_dir))
        return 0
    try:
        while True:
            print("\033[2J\033[H" + summary(run_dir), flush=True)   # clear the screen, then print
            time.sleep(a.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
```
- **Once mode** (`--watch` is 0): print the summary and finish.
- **Watch mode:** an endless loop: print `"\033[2J\033[H"` (clear the screen and move the cursor to the top-left) followed by the fresh summary (`flush=True` sends it to the screen at once), then `time.sleep(N)`. **Ctrl+C** raises `KeyboardInterrupt`, which is caught → the monitor ends quietly with exit code 0. (Pressing Ctrl+C here stops only the monitor.)

---

## 5. How to read the screen

- **progress:** how far the schedule is (early stopping may end before 100 %).
- **val loss vs best:** if the latest is *above* the best, validation is not improving at the moment.
- **early stop X / patience:** X growing towards the patience means training may stop soon; it resets to 0 when a new best appears.
- **time left at most:** a ceiling, not a promise.
- **[status]:** `running`, or the final status `finished` / `early_stopped` / `stopped`.

## 6. Small things worth knowing

- **`[running]` can be wrong.** The status is `running` whenever there is **no `status.json`** — also if the training crashed or was killed. And in a **resumed** run the *old* `status.json` (e.g. `stopped`) stays until the run ends, so the monitor can show `[stopped]` while it is actually training. Check the step number / speed to be sure.
- **The 256 is hard-coded** (block size) in the time estimate.
- **It needs `Path.is_relative_to`** (Python 3.9+), fine for your venv.
- **ANSI screen clearing** works in the VS Code terminal and normal Linux terminals; in a terminal that does not understand it you will see odd characters.
- **Reading while training writes** is safe in practice because the training flushes complete rows.
- **Only the last 5 speed values** are averaged, so the number jumps a little between refreshes.

## 7. Quick summary in one paragraph

`day12_monitor.py` is a read-only dashboard: it reads a run's `log.csv`, `train_config.json` and `status.json` and prints a progress bar, the latest training and validation loss, the best validation loss (with perplexity and step), the early-stopping counter, the recent speed and an upper bound for the time left — once, or refreshed every N seconds with `--watch` — without ever touching the training.
