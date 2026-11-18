# Day 11 — day11_size_trials.py — Simple Line-by-Line Explanation (English)

> **How the three Day 11 files fit together**
> - **Day 11.1 `selection.py`** — the decision rule (**read it first**).
> - **Day 11 (this file) `day11_size_trials.py`** — the **driver**: trains every (size, dropout) pair on the same budget, checks generation on each best checkpoint, calls `selection.py` to decide, and writes the report, the figure and the decision file.
> - **Day 11.2 `test_selection.py`** — tests for the decision rule.

---

## 0. The big picture (read this first)

Question of Day 11: **which size of Model A (tiny / small / medium) — and which dropout (0.0 / 0.1) — should the main training (Day 12) use?**

The script runs a fair experiment:

```
for each (size, dropout)  →  6 runs by default:
        train with EXACTLY the same settings (only size and dropout differ)
then for each run:
        take its best checkpoint (lowest validation loss) → check generation on the 12 fixed prompts
then:
        apply the written rule (selection.py) → choose one → save the decision
```

### "Same budget" = a fair race

| Setting | Value |
|---|---|
| Batch | 32 windows × 256 tokens (8,192 tokens per step) |
| Steps | 800 (≈ 6.5 million tokens per run, about 3 passes over the training text) |
| Warm-up | 60 steps |
| Peak learning rate | 0.001 |
| Seed | 42 (the same starting batch order for every run) |
| Validation | every **25 steps** (Day 9 showed the best point can come in the first few hundred steps, so checks must be frequent) |

### Outputs

| File | Content |
|---|---|
| `runs/day11/<size>_d<dropout>/` | one Day 9 run folder per trial (e.g. `tiny_d00`, `medium_d10`; git-ignored) |
| `docs/DAY11_SIZE_TRIALS.md` | results table, the decision and the reason, sample sentences |
| `docs/figures/day11_val_loss.png` | validation curves of all runs |
| `configs/day11_decision.json` | the chosen size and dropout (**Day 12 reads it**) |
| `run_log.md` | one entry |

### Resumable

Run the same command again after Ctrl+C or a crash: **finished runs are skipped**, an **unfinished run continues from its `last.pt`**.

```
.venv\Scripts\python.exe scripts\day11_size_trials.py --machine home
```
(Linux server later: `.venv/bin/python scripts/day11_size_trials.py --machine server`.)

---

## 1. Words you need to know

Most words are from Day 11.1 (validation loss, best step, gap, tolerance) and Day 9.2 (run, checkpoint, resume). New Python idea: **`nargs="+"`** — an `argparse` option that accepts **one or more values** (e.g. `--sizes tiny small`).

---

## 2. Imports and constants

```python
from __future__ import annotations

import argparse, json, math, platform, sys
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.bpe_tokenizer import EOS_ID, ByteBPETokenizer  # noqa: E402
from src.generation import (PRESET_SETTINGS, distinct_2, generate, is_repetitive, load_model, model_logits_fn)  # noqa: E402
from src.model import ModelConfig, expected_parameters  # noqa: E402
from src.selection import TOLERANCE, RunSummary, choose, summarize_run  # noqa: E402
from src.training import TrainConfig, train  # noqa: E402

SCRIPT_VERSION: str = "day11-v1"
KEY_PROMPTS: tuple[str, ...] = ("営業部の", "ボーナスが", "勤続年数が", "今日は")
```
- The usual setup. It reuses almost everything built so far: the tokenizer (**Day 6.1**), the generation engine (**Day 10.1**), the model's parameter formula (**Day 8.1**), the decision logic (**Day 11.1**) and the training loop (**Day 9.2**).
- `KEY_PROMPTS` — the four beginnings shown in the report's sample-sentence tables (they should be lines of your prompt file; otherwise the table shows `-`).

---

## 3. Helper functions

### 3.1 `run_name`

```python
def run_name(size: str, dropout: float) -> str:
    return f"{size}_d{int(round(dropout * 100)):02d}"
```
- Names a run folder: `tiny` + 0.0 → `tiny_d00`; `medium` + 0.1 → `medium_d10`. `round(dropout * 100)` avoids floating-point surprises (`0.1 × 100` is `10.000000000000002`); `:02d` = two digits with a leading zero.

### 3.2 `read_prompts`

```python
def read_prompts(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
```
- Same as Day 10: one prompt per line, skip blank lines and `#` comments.

### 3.3 `run_state`

```python
def run_state(run_dir: Path, max_steps: int) -> str:
    """'new', 'finished' or 'unfinished' (judged from the log: the last evaluation must be at max_steps)."""
    log = run_dir / "log.csv"
    if not (run_dir / "last.pt").exists() or not log.exists():
        return "new"
    steps = [int(line.split(",", 1)[0]) for line in log.read_text(encoding="utf-8").splitlines()[1:]
             if ",eval," in line]
    return "finished" if steps and steps[-1] >= max_steps else "unfinished"
```
- Decides what to do with a trial: **`new`** (no checkpoint or log yet), **`finished`**, or **`unfinished`**.
- It reads the log as plain text: skip the header (`[1:]`); for every **eval** row (a line containing `,eval,`) take the first field (the step) — `line.split(",", 1)[0]` cuts at the first comma. If the **last evaluation reached `max_steps`** → finished (Day 9.2 always evaluates at the last step); otherwise unfinished.

### 3.4 `generation_check`

```python
def generation_check(ckpt: Path, tok: ByteBPETokenizer, prompts: list[str], device: torch.device,
                     train_set: set[str]) -> dict:
    model, _ = load_model(ckpt, device)
    fn = model_logits_fn(model, device)
    out: dict = {}
    for setting in ("greedy+limits", "sample"):
        gens = [generate(fn, tok, p, PRESET_SETTINGS[setting]) for p in prompts]
        n = len(gens)
        out[setting] = {
            "finished": sum(g.finished for g in gens) / n,
            "repetitive": sum(is_repetitive(tok.encode(g.prompt) + [i for i in g.ids if i != EOS_ID])
                              for g in gens) / n,
            "distinct2": distinct_2([[i for i in g.ids if i != EOS_ID] for g in gens]),
            "new": sum((g.prompt + g.text) not in train_set for g in gens if g.finished),
            "texts": {g.prompt: (g.text + (" ⟨eos⟩" if g.finished else " …")) for g in gens},
        }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out
```
- Loads a run's **best checkpoint** (Day 10.1) and generates a continuation for every prompt with **two settings**: `greedy+limits` and `sample` (Day 10.1 presets).
- For each setting it records the **task 6 automatic items** (same formulas as Day 10's report):
  - `finished` — **fraction** of outputs ending with `<eos>`;
  - `repetitive` — fraction with more repetition than any real question (`EOS_ID` removed first);
  - `distinct2` — variety;
  - `new` — **count** of finished outputs that are not training questions (reference only);
  - `texts` — the generated text for each prompt, marked `⟨eos⟩` (finished) or `…` (cut), for the report.
- Finally free the model and GPU memory (`del model`, `empty_cache`) so the next run starts clean.
- (The decision will use the **greedy+limits** numbers; `sample` is shown for information.)

### 3.5 `draw_figure`

```python
def draw_figure(runs: list[RunSummary], root: Path, path: Path) -> None:
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"tiny": "#1f77b4", "small": "#2ca02c", "medium": "#d62728"}
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    for r in runs:
        with (root / r.name / "log.csv").open(encoding="utf-8", newline="") as f:
            ev = [(int(x["step"]), float(x["val_loss"])) for x in csv.DictReader(f)
                  if x["event"] == "eval" and x["val_loss"] and int(x["step"]) > 0]
        ax.plot([s for s, _ in ev], [v for _, v in ev], color=colours.get(r.size, "gray"),
                ls="-" if r.dropout == 0 else "--", lw=1.6, label=f"{r.size}, dropout {r.dropout}")
        ax.plot([r.best_step], [r.best_val], "o", color=colours.get(r.size, "gray"), ms=5)
    ax.set_xlabel("step")
    ax.set_ylabel("validation loss")
    ax.set_title("Day 11 size trials: validation loss (dots = lowest point of each run)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
```
- Draws **all validation curves in one picture** (matplotlib with the file-only `Agg` backend, as in Day 9's compare script — **matplotlib must be installed**).
- **Colour = size** (tiny blue, small green, medium red; unknown sizes grey); **line style = dropout** (solid for 0, dashed otherwise). Only evaluations after step 0 are drawn (step 0 is ≈ `ln(vocab)` and would squash the picture). A **dot** marks each run's best point.
- Then labels, title, grid, legend, save, close.

---

## 4. `parse_args`

```python
ap.add_argument("--machine", choices=["home", "server", "school"], default="home")
ap.add_argument("--sizes", nargs="+", default=["tiny", "small", "medium"])
ap.add_argument("--dropouts", nargs="+", type=float, default=[0.0, 0.1])
ap.add_argument("--max-steps", type=int, default=800)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--eval-every", type=int, default=25)
ap.add_argument("--warmup-steps", type=int, default=60)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
ap.add_argument("--eval-max-windows", type=int, default=None, help="(testing only) fewer validation windows")
ap.add_argument("--runs-root", default="runs/day11")
ap.add_argument("--prompts", type=Path, default=REPO_ROOT / "prompts/day10_prompts.txt")
ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY11_SIZE_TRIALS.md")
ap.add_argument("--figure", type=Path, default=REPO_ROOT / "docs/figures/day11_val_loss.png")
ap.add_argument("--decision", type=Path, default=REPO_ROOT / "configs/day11_decision.json")
ap.add_argument("--no-log", action="store_true")
```
- The experiment's settings, all with the defaults described in section 0. `--sizes` and `--dropouts` accept **several values** (`nargs="+"`), e.g. `--sizes tiny small --dropouts 0.0`. `--eval-max-windows` is for quick tests. The `Path` options and `--no-log` work as before.
- **Important for resuming:** to continue an unfinished run, use the **same options** as the first time (Day 9.2 refuses to resume with different settings).

---

## 5. `main`

### 5.1 Setup

```python
    a = parse_args()
    runs_root = REPO_ROOT / a.runs_root
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    trials = [(s, d) for s in a.sizes for d in a.dropouts]
    print(f"=== Day 11 size trials | machine={a.machine} | {device_name} | {len(trials)} runs x {a.max_steps} steps ===")
```
- `device` uses a **nested conditional**: `auto` + GPU available → `cuda`; `auto` otherwise → `cpu`; if you named a device explicitly, use it.
- `trials` — **every combination** of size × dropout (a list comprehension with two loops: 3 × 2 = 6 pairs, in the order tiny 0.0, tiny 0.1, small 0.0, …).

### 5.2 Step 1 — train every trial

```python
    for i, (size, dropout) in enumerate(trials, 1):
        name = run_name(size, dropout)
        run_dir = runs_root / name
        state = run_state(run_dir, a.max_steps)
        print(f"\n[{i}/{len(trials)}] {name}: {state}")
        if state == "finished":
            continue
        cfg = TrainConfig(
            run_dir=f"{a.runs_root}/{name}", model_config=f"configs/model_{size}.json", batch_size=a.batch_size,
            max_steps=a.max_steps, warmup_steps=a.warmup_steps, lr=a.lr, dropout=dropout,
            log_every=a.eval_every, eval_every=a.eval_every, eval_max_windows=a.eval_max_windows,
            prompts=[], device=a.device, seed=a.seed, machine=a.machine, resume=(state == "unfinished"),
        )
        summary = train(cfg, REPO_ROOT)
        if summary["stopped_early"]:
            print("\nStopped. Run the same command again to continue from here.")
            return 0
```
- For each trial: name it, look at its state, print `[3/6] small_d00: new`, and **skip it if finished**.
- Otherwise build a Day 9.2 `TrainConfig`:
  - the model config is `configs/model_<size>.json` (Day 8);
  - **identical settings** for every run (batch, steps, warm-up, learning rate, seed) except `dropout` (and the size);
  - `log_every = eval_every = 25` → a training-loss row and an evaluation row at every 25 steps (so Day 11.1 finds the training loss at the best step);
  - `prompts=[]` → **no sample texts during training** (faster; generation is checked later on the best checkpoint);
  - `resume=(state == "unfinished")` → continue an unfinished run, start a new one otherwise.
- `train(cfg, REPO_ROOT)` (Day 9.2) does the whole run. If it **stopped early** (you pressed Ctrl+C), print a hint and **exit**; running the same command again continues.
- Because every run uses the **same seed**, every run sees the **same sequence of training windows** — another way the comparison is fair.

### 5.3 Step 2 — summarise, check generation, decide

```python
    tok = ByteBPETokenizer.load(REPO_ROOT / "data/tokenizer/tokenizer.json")
    prompts = read_prompts(a.prompts)
    train_set = set((REPO_ROOT / "data/corpus/train.txt").read_text(encoding="utf-8").split("\n"))
    runs: list[RunSummary] = []
    gen: dict[str, dict] = {}
    print("\n--- generation check on each best.pt ---")
    for size, dropout in trials:
        name = run_name(size, dropout)
        params = expected_parameters(ModelConfig.load(REPO_ROOT / f"configs/model_{size}.json"))
        r = summarize_run(runs_root / name / "log.csv", name, size, dropout, params)
        g = generation_check(runs_root / name / "best.pt", tok, prompts, device, train_set)
        gl = g["greedy+limits"]
        r.finished_rate, r.repetitive_rate, r.distinct2 = gl["finished"], gl["repetitive"], gl["distinct2"]
        runs.append(r)
        gen[name] = g
        print(f"  {name:<10} best val {r.best_val:.4f} at step {r.best_step:>4} | finished {gl['finished']:.0%} | "
              f"repetitive {gl['repetitive']:.0%} | distinct-2 {gl['distinct2']:.2f}")
    decision = choose(runs)
```
- Load the tokenizer, the prompts, and the **set of training questions** (for the "new" count).
- For every trial: the parameter count from the **formula** (Day 8.1), the run's **summary** from its log (Day 11.1), and the **generation check** on its `best.pt`. The check's `greedy+limits` rates are **copied into the summary** (`r.finished_rate = …`) — the fields that started as `None`. Print one line per run.
- `choose(runs)` (Day 11.1) applies the written rule.

### 5.4 Step 3 — print, figure, report, decision file, log

```python
    print(f"\n{'run':<10} {'params':>11} {'best val':>9} {'ppl':>6} {'step':>5} {'train@best':>10} "
          f"{'gap':>6} {'final val':>9} {'tok/s':>9} {'peak MB':>8}")
    for r in runs:
        print(...)  # one aligned row per run; missing numbers shown as nan / 0
    print(f"\nDecision: {decision.chosen.name if decision.chosen else 'NONE'} - {decision.reason}")
```
- A neat results table on the screen (alignment codes as before; `math.exp(best_val)` = perplexity; `float('nan')` / `0` stand in for missing values) and the decision with its reason.

```python
    draw_figure(runs, runs_root, a.figure)
    write_report(a, runs, gen, decision, device_name, prompts)
    if decision.chosen:
        c = decision.chosen
        a.decision.parent.mkdir(parents=True, exist_ok=True)
        a.decision.write_text(json.dumps({
            "chosen_run": c.name, "size": c.size, "dropout": c.dropout, "model_config": f"configs/model_{c.size}.json",
            "parameters": c.parameters, "best_val": round(c.best_val, 4), "best_step": c.best_step,
            "tokens_per_step": a.batch_size * 256, "trial_steps": a.max_steps, "tolerance": TOLERANCE,
            "reason": decision.reason, "decided": f"{datetime.now():%Y-%m-%d %H:%M}", "script": SCRIPT_VERSION,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
```
- Draw the figure, write the report (section 6).
- **If a run was chosen**, write `configs/day11_decision.json` — the **hand-over file for Day 12**: the chosen run name, size, dropout, the model-config path, parameters, best validation loss and step, tokens per step (`batch × 256`; 256 = the block size), the number of trial steps, the tolerance, the **reason sentence**, a timestamp and the script version. If nobody was chosen, no file is written.

```python
    print(f"Report: {a.report}\nFigure: {a.figure}\nDecision file: {a.decision}")
    if not a.no_log:
        with (REPO_ROOT / "run_log.md").open("a", encoding="utf-8") as f:
            f.write(...)   # heading; script/device/PyTorch; the trials; one line per run (best val, step, final); the decision
        print("Appended an entry to run_log.md")
    return 0
```
- Print the locations and append the run-log entry (the list of trials, each run's best and final validation loss, and the decision with its reason). `return 0`.

---

## 6. `write_report` — the Markdown report

```python
def write_report(a, runs, gen, decision, device_name, prompts) -> None:
    fig_rel = a.figure.resolve().relative_to(a.report.resolve().parent).as_posix()
    c = decision.chosen
    md = [ ... ]
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text("\n".join(md), encoding="utf-8", newline="\n")
```
- `fig_rel` — the figure path relative to the report's folder, so the image link works (same trick as Day 9's compare script). The report sections:

| Section | Content |
|---|---|
| Header | version, date, machine, device, PyTorch/Python; "Every run: batch …, steps, warm-up, learning rate, seed, validation every N steps. **Only the size and the dropout differ.**" |
| **Decision** | `**Chosen: <run>**` with size, dropout, parameters, best validation loss, perplexity and step — or "No run chosen" (a conditional expression: the chosen-run text is only built when a run exists); the **reason**; the **rule written out** (finished ≥ 90 %, repetitive ≤ 5 % with greedy + limits on the N fixed prompts; runs within the tolerance are equally good; fewest parameters wins); **which runs were within the tolerance** and **which were rejected and why** |
| Figure | the validation-curve picture |
| **Results** | One row per run: parameters, **best validation loss**, perplexity, best step, training loss there, **gap**, final validation loss, **rise after best**, tokens/s, peak MB, finished, repetitive, distinct-2 (missing values shown as `-`) |
| Explanations | **Gap** = validation − training loss at the best step ("how much harder unseen sentence patterns are"); **Rise after best** = final − best ("how much it overfitted afterwards"); the three generation items |
| **Sample sentences** | Two tables (**greedy + limits**, and **sampling** with temperature 0.8 / top-k 50 / top-p 0.95): rows = runs, columns = the four `KEY_PROMPTS`, cells = the generated text with ⟨eos⟩ or … (`|` escaped; `-` if that prompt is not in your prompt file) |

- Unpacking with `*( … for r in runs)` produces one table row per run, as in earlier days.

---

## 7. `__main__`

```python
if __name__ == "__main__":
    sys.exit(main())
```
- Same ending as always. (`write_report` is defined *after* `main` in the file; that is fine because it is only called when `main` runs.)

---

## 8. The whole run in order

1. Read options → list the 6 trials.
2. For each trial: skip if finished, resume if unfinished, otherwise train from scratch (`train` from Day 9.2) — **all with the same settings**.
3. For each trial: summarise the log, check generation on `best.pt`.
4. `choose(...)` applies the written rule.
5. Print the table and decision → figure → report → `configs/day11_decision.json` → run log.

## 9. How to read the results

- **Best val loss / perplexity:** lower = better generalisation. Compare within about 0.02 as "the same".
- **At step:** a best step far before 800 means the run **overfitted** after it. **Rise after best** shows how much.
- **Gap:** a large gap (validation much higher than training) = memorising; dropout should shrink it.
- **Finished / Repetitive:** a run below 90 % / above 5 % is **rejected**, whatever its loss.
- **Tokens/s, Peak MB:** speed and memory cost of each size (an unchosen bigger model may be slower for Day 12).
- **Decision file:** open `configs/day11_decision.json` — it is what Day 12 uses.

## 10. Small things worth knowing

- **Same arguments to resume.** An unfinished run only resumes if the options match the first time; otherwise Day 9.2 refuses.
- **Half-created folders.** If a run was killed before its first checkpoint, the folder may exist without `last.pt`; the state is then `new`, but Day 9.2 refuses to start a "new" run in a folder that already has files — delete that run folder (or use a new `--runs-root`) and rerun.
- **One run per configuration (one seed).** The tolerance 0.02 is therefore an assumption about noise, not something this script measures.
- **No tokenizer fingerprint check in `generation_check`** (it uses `load_model` directly). The tokenizer is verified at training time by `train` (Day 9.2) and by `load_checked` in Day 10; here it is the same tokenizer file as the runs used.
- **`tokens_per_step` in the decision file uses `batch_size × 256`**, with 256 written in as the block size.
- **Selection uses only validation data.** The test split is untouched here, so it can still give an honest final score later (picking the best of several runs by validation loss is slightly optimistic).
- **`texts` are keyed by prompt**, so duplicate prompts in the prompt file would overwrite each other in the report.
- **Needs matplotlib** for the figure.

## 11. Quick summary in one paragraph

`day11_size_trials.py` trains Model A at three sizes with dropout 0.0 and 0.1 (6 resumable runs on identical settings and seed, validating every 25 steps), checks generation on each run's best checkpoint with the 12 fixed prompts, lets `selection.py` apply the rule written beforehand (pass the finished/repetitive checks, treat losses within 0.02 as equal, take the fewest parameters), and then saves the validation-curve figure, a Markdown report with the decision and reason, and `configs/day11_decision.json` for Day 12.
