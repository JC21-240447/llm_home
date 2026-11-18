# Day 12 — day12_train_main.py — Simple Line-by-Line Explanation (English)

> **How the five Day 12 files fit together**
> - **Day 12 (this file) `day12_train_main.py`** — the **main training**: the Day 11 configuration, early stopping, **3 seeds**, **one final model**.
> - **Day 12 `day12_monitor.py`** — watch a run from a second terminal.
> - **Day 12.1 `generation.py`** (adds `quick_checks`), **Day 12.2 `training.py`** (adds early stopping + `status.json`), **Day 12.3 `test_training.py`** (tests of early stopping).
>
> Suggested reading order: **12.2 → 12.1 → 12 (this file) → 12 (monitor) → 12.3**.

---

## 0. The big picture (read this first)

Day 11 answered "which size and dropout?" and wrote `configs/day11_decision.json`. Day 12 is the **real training** with that choice, designed to give a result you can **trust and defend**.

| Design choice | Why |
|---|---|
| **Read the decision file** | Size and dropout come from Day 11's written rule, not from memory |
| **3 seeds (42, 43, 44)** | Day 11 showed **one seed can be off by up to 0.06** in validation loss. So report a **mean and a spread**, not one lucky number |
| **400-step schedule** | Day 11's best point was at steps 125–225; with a shorter schedule the **learning rate has already decayed** by then (a 800-step schedule is still at a high rate at step 200) |
| **Validation every 10 steps** | frequent checks so the best point is not missed |
| **Early stopping, patience 100** | stop when 10 evaluations in a row bring no improvement (Day 12.2) |
| **Final model = the seed with the lowest validation loss** (among those that pass the checks) | a written, reproducible choice |
| **The test set is NOT used** | it is kept for the final, honest evaluation (**Day 13**) |

### What is written

| Output | Content |
|---|---|
| `runs/day12/main_s42`, `main_s43`, `main_s44` | one run folder per seed (git-ignored) |
| `docs/DAY12_MAIN_TRAINING.md` | results, comparison with Day 11, sample sentences |
| `docs/figures/day12_main.png` | training and validation curves of the 3 seeds |
| `configs/day12_final.json` | **which checkpoint is the final model** (Day 13 reads it) |
| `run_log.md` | one entry |

### Resumable

Run the **same command** again after Ctrl+C or a crash: **done seeds are skipped**, an unfinished seed continues from its `last.pt`.

```
.venv\Scripts\python.exe scripts\day12_train_main.py --machine home
```
Watch from a second terminal: `.venv\Scripts\python.exe scripts\day12_monitor.py --run runs/day12/main_s42 --watch 5` (Linux server later: `.venv/bin/python scripts/day12_train_main.py --machine server`).

---

## 1. Words you need to know

| Word | Simple meaning |
|---|---|
| **Seed** | The random starting point (initial weights and batch order). Different seeds → slightly different results. |
| **Mean** | The average. |
| **Spread** | Here: **largest − smallest** of the three best validation losses. A rough measure of how much the result depends on luck. |
| **Early stopping / patience** | See Day 12.2. |
| **Final model** | The one checkpoint (`best.pt` of one seed) used from now on. |

---

## 2. Imports and constants

```python
from __future__ import annotations

import argparse, json, math, platform, statistics, sys
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.bpe_tokenizer import ByteBPETokenizer  # noqa: E402
from src.generation import PRESET_SETTINGS, load_model, model_logits_fn, quick_checks  # noqa: E402
from src.model import ModelConfig, expected_parameters  # noqa: E402
from src.selection import MAX_REPETITIVE, MIN_FINISHED, RunSummary, summarize_run  # noqa: E402
from src.training import TrainConfig, train  # noqa: E402

SCRIPT_VERSION: str = "day12-v1"
KEY_PROMPTS: tuple[str, ...] = ("営業部の", "ボーナスが", "勤続年数が", "大阪支社で", "課長の", "今日は")
```
- New standard module: **`statistics`** (`statistics.mean`). Everything else comes from the project: the tokenizer (**6.1**), generation with the new **`quick_checks`** (**12.1**), the model formula (**8.1**), the run summary and the check thresholds `MIN_FINISHED`/`MAX_REPETITIVE` (**11.1**), and the training loop with early stopping (**12.2**).
- `KEY_PROMPTS` — six beginnings shown in the report's sample table.

---

## 3. Helper functions

### 3.1 `read_prompts`
- Same as Days 10 and 11 (one prompt per line; blank lines and `#` comments skipped).

### 3.2 `run_status`

```python
def run_status(run_dir: Path) -> str:
    """'new', 'done' (finished or early-stopped) or 'unfinished' (stopped by Ctrl+C or killed)."""
    status = run_dir / "status.json"
    if status.exists() and json.loads(status.read_text(encoding="utf-8"))["status"] in ("finished", "early_stopped"):
        return "done"
    return "unfinished" if (run_dir / "last.pt").exists() else "new"
```
- Decides what to do with a seed, using **`status.json`** (Day 12.2) — simpler and more reliable than Day 11's log parsing:
  - `status.json` says **finished** or **early_stopped** → **`done`** (skip);
  - otherwise, if a `last.pt` exists → **`unfinished`** (stopped with Ctrl+C, or killed: **resume**);
  - otherwise → **`new`**.
- (A run with status `stopped` counts as unfinished, because it was interrupted on purpose and can be resumed.)

### 3.3 `draw_figure`

```python
def draw_figure(runs: list[RunSummary], runs_root: Path, path: Path) -> None:
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    for c, r in zip(colours, runs):
        with (runs_root / r.name / "log.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        ev = [(int(x["step"]), float(x["val_loss"])) for x in rows if x["event"] == "eval" and x["val_loss"]
              and int(x["step"]) > 0]
        tr = [(int(x["step"]), float(x["train_loss"])) for x in rows if x["event"] == "train" and x["train_loss"]]
        ax.plot([s for s, _ in tr], [v for _, v in tr], color=c, lw=1.0, alpha=0.45, ls="--")
        ax.plot([s for s, _ in ev], [v for _, v in ev], color=c, lw=1.8, label=f"{r.name} validation")
        ax.plot([r.best_step], [r.best_val], "o", color=c, ms=6)
    ax.set_ylim(0.8, 4.0)
    ...
```
- As in Day 11 (matplotlib with the file-only backend; **matplotlib must be installed**), but now **one colour per seed** (`zip(colours, runs)`) and **two lines per seed**: the **validation loss** (solid, thicker, with the legend label) and the **training loss** (dashed, thin, semi-transparent: `alpha=0.45`), plus a **dot at the best point**. The gap between the dashed and solid line is the memorising gap.
- `ax.set_ylim(0.8, 4.0)` — fixed y-axis range so the three seeds are easy to compare (values outside 0.8–4.0 would be cut off). Then labels, title, grid, legend, save, close (as before).

---

## 4. `parse_args`

```python
ap.add_argument("--machine", choices=["home", "server", "school"], default="home")
ap.add_argument("--decision", type=Path, default=REPO_ROOT / "configs/day11_decision.json")
ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
ap.add_argument("--max-steps", type=int, default=400)
ap.add_argument("--warmup-steps", type=int, default=60)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--eval-every", type=int, default=10)
ap.add_argument("--patience", type=int, default=100)
ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
ap.add_argument("--eval-max-windows", type=int, default=None, help="(testing only) fewer validation windows")
ap.add_argument("--runs-root", default="runs/day12")
ap.add_argument("--prompts", ...)  ap.add_argument("--report", ...)  ap.add_argument("--figure", ...)
ap.add_argument("--final", type=Path, default=REPO_ROOT / "configs/day12_final.json")
ap.add_argument("--no-log", action="store_true")
```
- The settings from section 0 as defaults. `--seeds` accepts several integers (`nargs="+", type=int`). `--patience` = the early-stopping patience. `--final` = where the final-model file is written. Others as in Day 11. **To resume, use the same options as the first time.**

---

## 5. `main`

### 5.1 Read the decision

```python
    a = parse_args()
    decision = json.loads(a.decision.read_text(encoding="utf-8"))
    size, dropout, model_config = decision["size"], float(decision["dropout"]), decision["model_config"]
    runs_root = REPO_ROOT / a.runs_root
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"=== Day 12 main training | {size}, dropout {dropout} (from {a.decision.name}) | seeds {a.seeds} | "
          f"{device_name} ===")
```
- Load **Day 11's decision file** and take the size, the dropout and the path of the model config. The device is chosen as in Day 11 (nested conditional). Print a header showing where the configuration came from.

### 5.2 Step 1 — train each seed

```python
    statuses: dict[str, dict] = {}
    for i, seed in enumerate(a.seeds, 1):
        name = f"main_s{seed}"
        run_dir = runs_root / name
        state = run_status(run_dir)
        print(f"\n[{i}/{len(a.seeds)}] {name}: {state}")
        if state != "done":
            cfg = TrainConfig(
                run_dir=f"{a.runs_root}/{name}", model_config=model_config, batch_size=a.batch_size,
                max_steps=a.max_steps, warmup_steps=a.warmup_steps, lr=a.lr, dropout=dropout,
                log_every=a.eval_every, eval_every=a.eval_every, eval_max_windows=a.eval_max_windows,
                early_stop_patience=a.patience, prompts=[], device=a.device, seed=seed, machine=a.machine,
                resume=(state == "unfinished"),
            )
            result = train(cfg, REPO_ROOT)
            if result["stopped_early"]:
                print("\nStopped. Run the same command again to continue from here.")
                return 0
        statuses[name] = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
```
- For each seed: name it `main_s42` etc.; find its state; skip if **done**.
- Otherwise build the `TrainConfig`: **identical settings for all seeds except `seed`**; `log_every = eval_every = 10`; **`early_stop_patience=a.patience`** (new in Day 12.2); `prompts=[]` (no sample texts during training); `resume` if unfinished.
- `train(cfg, REPO_ROOT)` runs it. If it was **interrupted** (`stopped_early`, i.e. Ctrl+C), print a hint and **exit**; the same command continues later. (An **early-stopped** run has `stopped_early = False`, so the loop moves on to the next seed.)
- Then read the seed's **`status.json`** (how it ended and at which step) into `statuses` — done also for seeds that were skipped.

### 5.3 Step 2 — summaries and generation checks (no test set)

```python
    tok = ByteBPETokenizer.load(REPO_ROOT / "data/tokenizer/tokenizer.json")
    prompts = read_prompts(a.prompts)
    train_set = set((REPO_ROOT / "data/corpus/train.txt").read_text(encoding="utf-8").split("\n"))
    params = expected_parameters(ModelConfig.load(REPO_ROOT / model_config))
    runs: list[RunSummary] = []
    checks: dict[str, dict] = {}
    print("\n--- generation check on each best.pt ---")
    for seed in a.seeds:
        name = f"main_s{seed}"
        r = summarize_run(runs_root / name / "log.csv", name, size, dropout, params)
        model, _ = load_model(runs_root / name / "best.pt", device)
        fn = model_logits_fn(model, device)
        checks[name] = {s: quick_checks(fn, tok, prompts, PRESET_SETTINGS[s], train_set)
                        for s in ("greedy+limits", "sample")}
        gl = checks[name]["greedy+limits"]
        r.finished_rate, r.repetitive_rate, r.distinct2 = gl["finished"], gl["repetitive"], gl["distinct2"]
        runs.append(r)
        print(f"  {name}: best val {r.best_val:.4f} at step {r.best_step} ({statuses[name]['status']} at step "
              f"{statuses[name]['step']}) | finished {gl['finished']:.0%} | repetitive {gl['repetitive']:.0%} | "
              f"distinct-2 {gl['distinct2']:.2f}")
        del model
```
- Load the tokenizer, prompts, training questions (for "new") and the **parameter count** from the formula.
- For each seed: the run summary (**Day 11.1**), load its **`best.pt`**, and run the **task 6 checks** with **Day 12.1's `quick_checks`** for two settings (`greedy+limits`, `sample`). The greedy numbers go into the summary. Print a line per seed: best loss and step, **how it ended**, and the three check numbers. Free the model.

### 5.4 Statistics and the final model

```python
    vals = [r.best_val for r in runs]
    mean_val = statistics.mean(vals)
    spread = max(vals) - min(vals)
    passing = [r for r in runs if r.passes_checks()]
    final = min(passing or runs, key=lambda r: r.best_val)
    d11 = [decision.get("best_val_seed42"), decision.get("best_val_seed43")]
    d11 = [v for v in d11 if v is not None]
    d11_mean = statistics.mean(d11) if d11 else None
```
- `vals` — each seed's best validation loss. `mean_val` — their **mean**. `spread` — **max − min**.
- `passing` — seeds whose best checkpoint passes the task 6 guidelines (`passes_checks`, Day 11.1: finished ≥ 90 %, repetitive ≤ 5 %).
- **`final`** — the seed with the **lowest validation loss among the passing ones**. `passing or runs` means: if **none** pass, fall back to **all** seeds (a warning is printed below).
- `d11` — the **Day 11 trial values** for the same configuration, read from two **optional keys** `best_val_seed42` and `best_val_seed43` of the decision file (`dict.get` returns `None` if absent); keep only the ones that exist; `d11_mean` = their mean, or `None`. *(The Day 11 script you pasted does not write these two keys; unless you added them to the file, e.g. after extra Day 11 seeds, the comparison with Day 11 shows `-`.)*

```python
    print(f"\nBest validation loss per seed: {', '.join(f'{v:.4f}' for v in vals)}")
    print(f"Mean {mean_val:.4f} (perplexity {math.exp(mean_val):.2f}), spread {spread:.4f}")
    if d11_mean is not None:
        print(f"Day 11 trials of the same configuration (800-step schedule): mean {d11_mean:.4f} -> "
              f"change {mean_val - d11_mean:+.4f}")
    print(f"Final model: {final.name} (best val {final.best_val:.4f} at step {final.best_step})"
          + ("" if final.passes_checks() else "  WARNING: does not pass the task 6 checks"))
```
- Print the per-seed values, the **mean (with perplexity) and spread**, the **change versus Day 11** (`:+.4f` always shows the sign) if available, and the **final model** — with a **WARNING** if even the chosen one fails the checks.

### 5.5 Step 3 — outputs

```python
    draw_figure(runs, runs_root, a.figure)
    final_ckpt = f"{a.runs_root}/{final.name}/best.pt"
    a.final.parent.mkdir(parents=True, exist_ok=True)
    a.final.write_text(json.dumps({
        "final_checkpoint": final_ckpt, "run": final.name, "size": size, "dropout": dropout,
        "model_config": model_config, "parameters": params, "best_val": round(final.best_val, 4),
        "best_step": final.best_step, "seeds": a.seeds, "best_val_per_seed": [round(v, 4) for v in vals],
        "mean_best_val": round(mean_val, 4), "spread": round(spread, 4), "test_set_used": False,
        "decided": f"{datetime.now():%Y-%m-%d %H:%M}", "script": SCRIPT_VERSION,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
```
- Draw the figure. Then write **`configs/day12_final.json`** — the **hand-over file for Day 13**: the **final checkpoint path**, the run, size, dropout, model config, parameters, best loss and step, **all seeds with their best losses**, **mean and spread**, **`"test_set_used": False`** (a recorded promise that the test set was untouched), a timestamp and the script version.

```python
    write_report(a, runs, statuses, checks, final, mean_val, spread, d11, d11_mean, device_name, size, dropout,
                 params, len(prompts))
    print(f"\nReport: {a.report}\nFigure: {a.figure}\nFinal model file: {a.final}  ->  {final_ckpt}")
    if not a.no_log:
        with (REPO_ROOT / "run_log.md").open("a", encoding="utf-8") as f:
            f.write(...)   # heading; script/device/PyTorch; configuration; one line per seed; mean/spread; final model
        print("Appended an entry to run_log.md")
    return 0
```
- Write the report (section 6), print the locations, append the run log (configuration, each seed's best loss and how it ended, mean/spread, the final checkpoint). `return 0`.

---

## 6. `_cell` and `write_report`

```python
def _cell(text: str) -> str:
    """Make a generated sentence safe inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ").replace("<", "&lt;")
```
- Makes generated text safe in a Markdown table: escape `|`, turn line breaks into spaces, and turn `<` into `&lt;` so pieces such as `<e5ae>` or `<eos>` are **not mistaken for HTML tags** and vanish.

`write_report` builds `docs/DAY12_MAIN_TRAINING.md` (the figure path is made relative to the report, as before). Sections:

| Section | Content |
|---|---|
| Header | version, date, machine, device, versions |
| Configuration | "Configuration from Day 11: **size, dropout** (N parameters)", batch × 256, the **400-step schedule** (warm-up, peak LR, cosine to 10 %), validation interval, **early-stopping patience**, seeds; "**The test set was not used.**" |
| **Final model** | the `best.pt` path, its best validation loss, perplexity and step; on the fixed prompts (greedy + limits) the **finished / repetitive / distinct-2** numbers next to the guidelines (`MIN_FINISHED`, `MAX_REPETITIVE`) |
| Figure | the curves |
| **All seeds** | One row per seed: **how it ended** ("early_stopped at step 230"), best validation loss, perplexity, best step, training loss there, **gap**, finished, repetitive, distinct-2, and **new sentences** as `new/finished-count` (`round(finished_rate × n_prompts)` gives the number of finished outputs); then the **mean and spread** |
| **Compared with Day 11** | a table: Day 11 (800 steps, number of seeds, mean) vs Day 12 (N steps + early stopping, 3 seeds, mean), and the sentence "**Change: ±x. Read it against the seed spread: a change smaller than the spread is not evidence of improvement.**" (empty if no Day 11 numbers) |
| Sample sentences | the final model's texts for the `KEY_PROMPTS` with greedy + limits and with sampling (`_cell` applied) |

- The "change smaller than the spread is not evidence" sentence is the honest-statistics point: with a seed spread of, say, 0.06, a 0.02 difference could just be luck.

---

## 7. The whole run in order

1. Read the Day 11 decision (size, dropout, model config).
2. For each seed: skip if done; resume if unfinished; otherwise train (same settings, early stopping on).
3. For each seed: summarise the log; check generation on `best.pt` (`quick_checks`).
4. Compute mean, spread; choose the **final seed** (lowest validation loss among those passing the checks).
5. Figure → `configs/day12_final.json` → report → run log.

## 8. How to read the results

- **Mean and spread:** the honest result is "mean ± spread", e.g. validation loss 2.05 with spread 0.04 (perplexity ≈ 7.8).
- **How it ended:** `early_stopped at step X` shows the model peaked early and then overfitted; `finished` means it kept improving to the end of the schedule.
- **Gap column:** large = memorising; compare with Day 11.
- **Final model:** the file `configs/day12_final.json` tells Day 13 which checkpoint to evaluate on the **test set**, once.
- **WARNING line:** no seed passed the generation checks — investigate before going on.

## 9. Small things worth knowing

- **Choosing the best of three seeds by validation loss is slightly optimistic**, which is exactly why the **test set** is kept for Day 13.
- **Same options to resume.** Day 12.2 refuses to resume with different settings (including a different `--patience`).
- **Comparison with Day 11 needs two optional keys** in the decision file (`best_val_seed42`, `best_val_seed43`); without them the comparison is skipped (`-`).
- **Some numbers are written in the code**: "800 steps" in the report text, the block size `256` inside the monitor/time estimates, and the figure's y-range 0.8–4.0.
- **The figure has 5 colours**, so with more than 5 seeds the extra runs would not be drawn.
- **Spread = max − min**, not a standard deviation; fine for 3 seeds.
- **`stopped_early`** (Ctrl+C) leaves the script; **early stopping** does not — it is a normal, complete end of that seed.
- **Needs matplotlib.**

## 10. Quick summary in one paragraph

`day12_train_main.py` takes the size and dropout chosen on Day 11 and trains them with three seeds on a 400-step schedule, validating every 10 steps with early stopping after 100 steps without improvement (resumable, `status.json` per seed), checks each seed's best checkpoint on the fixed prompts, reports the mean and spread of the best validation losses and the change against Day 11, picks the final model as the lowest-loss seed that passes the generation checks, and writes the curves figure, a Markdown report and `configs/day12_final.json` — all without touching the test set, which is saved for Day 13.
