# Day 9 — day9_train.py — Simple Line-by-Line Explanation (English)

> **How the five Day 9 files fit together**
> - **Day 9.1 `batching.py`** and **Day 9.2 `training.py`** — the batching and the training loop (read them first).
> - **Day 9 (this file) `day9_train.py`** — the **command-line script** that starts or resumes a run by calling `train(...)`.
> - **Day 9 `day9_compare_runs.py`** — proves that stop-and-resume gives the same result.
> - **Day 9.3 `test_training.py`** — automatic tests.

---

## 0. The big picture (read this first)

`training.py` contains all the logic. This script is only the **front door**: it reads your command-line options, turns them into a `TrainConfig`, calls `train(...)`, prints a summary and adds a note to `run_log.md`.

A run lives in its own folder `runs/day9/<run-name>/` (kept out of Git).

### The three things you can do

```
# 1. start a new run
.venv\Scripts\python.exe scripts\day9_train.py --run-name smoke_a --max-steps 600

# 2. stop early on purpose (simulates an interruption), then resume
.venv\Scripts\python.exe scripts\day9_train.py --run-name smoke_b --max-steps 600 --stop-after 300
.venv\Scripts\python.exe scripts\day9_train.py --run-name smoke_b --resume
```
- **Ctrl+C during training:** the current step finishes, `last.pt` is saved, and it stops. Continue later with `--resume`.
- A **resumed** run takes **all its settings from the run's own `train_config.json`**; only `--stop-after`, `--machine` and `--device` may be given again.
- On a GPU without bf16 (RTX 2060), `--precision auto` chooses fp16 with a gradient scaler.
- Linux server later: `.venv/bin/python scripts/day9_train.py --machine server ...`.

The two "smoke" runs (`smoke_a` uninterrupted, `smoke_b` stopped and resumed) are what `day9_compare_runs.py` compares.

---

## 1. Words you need to know

All words are from Day 9.2 (run, resume, checkpoint, log). One new argparse idea: an **argument group** (`ap.add_argument_group(...)`) — a heading in `--help` that groups related options.

---

## 2. The docstring

- English + Japanese. It explains the run folder contents (`train_config.json`, `model_config.json`, `log.csv`, `samples.md`, `last.pt`, `best.pt`), the three usage examples, the Ctrl+C behaviour and the resume rule.

---

## 3. Imports and setup

```python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.training import TrainConfig, train  # noqa: E402

SCRIPT_VERSION: str = "day9-v1"
```
- Standard imports and the `sys.path` trick from earlier days. `torch` is imported here only to print its version in the run log. From **Day 9.2**: the settings class `TrainConfig` and the `train` function.

---

## 4. `parse_args`

```python
def parse_args() -> argparse.Namespace:
    d = TrainConfig(run_dir="")
    ap = argparse.ArgumentParser(description="Day 9: train or resume Model A.")
    ap.add_argument("--run-name", required=True, help="folder name under --runs-root")
    ap.add_argument("--runs-root", default="runs/day9")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after", type=int, default=None)
    ap.add_argument("--machine", choices=["home", "server", "school"], default="home")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--overwrite", action="store_true", help="start again in an existing run folder")
```
- `d = TrainConfig(run_dir="")` — a settings object created **only to read the default values** (same trick as Day 5's `SplitConfig`): the defaults are written once in `TrainConfig` and reused here.
- The first group of options (always valid):
  - `--run-name` (**required**) — the folder name; `--runs-root` — its parent (default `runs/day9`);
  - `--resume` — a flag: continue an existing run;
  - `--stop-after N` — stop at step N on purpose;
  - `--machine`, `--device` — labels / device choice (as in Day 8);
  - `--overwrite` — allow starting again in a folder that already has files.

```python
    g = ap.add_argument_group("settings of a new run (ignored with --resume)")
    g.add_argument("--model-config", default=d.model_config)
    g.add_argument("--batch-size", type=int, default=d.batch_size)
    g.add_argument("--grad-accum", type=int, default=d.grad_accum)
    g.add_argument("--max-steps", type=int, default=d.max_steps)
    g.add_argument("--warmup-steps", type=int, default=d.warmup_steps)
    g.add_argument("--lr", type=float, default=d.lr)
    g.add_argument("--min-lr-ratio", type=float, default=d.min_lr_ratio)
    g.add_argument("--weight-decay", type=float, default=d.weight_decay)
    g.add_argument("--grad-clip", type=float, default=d.grad_clip)
    g.add_argument("--dropout", type=float, default=None)
    g.add_argument("--log-every", type=int, default=d.log_every)
    g.add_argument("--eval-every", type=int, default=d.eval_every)
    g.add_argument("--eval-max-windows", type=int, default=None)
    g.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default=d.precision)
    g.add_argument("--seed", type=int, default=d.seed)
    g.add_argument("--prompt", action="append", help="prompt for the samples (repeatable)")
    return ap.parse_args()
```
- The second group: the **settings of a new run** (ignored with `--resume`). All of them mirror `TrainConfig` fields (see the table in Day 9.2 §3); every line is the same pattern: an option name, a type, and the default taken from `d`. (`--dropout` and `--eval-max-windows` default to `None` = "use the model config's / use all windows".)
- `--prompt` with `action="append"` (as in Day 7's demo): give it several times to set your own sample prompts.

---

## 5. `build_config`

```python
def build_config(a: argparse.Namespace) -> TrainConfig:
    run_dir = f"{a.runs_root}/{a.run_name}"
    if a.resume:
        saved_path = REPO_ROOT / run_dir / "train_config.json"
        if not saved_path.exists():
            raise SystemExit(f"nothing to resume: {saved_path} does not exist")
        cfg = TrainConfig.from_dict(json.loads(saved_path.read_text(encoding="utf-8")))
        cfg.resume, cfg.stop_after, cfg.machine, cfg.device, cfg.overwrite = True, a.stop_after, a.machine, a.device, False
        return cfg
```
- `run_dir` — the run's folder as text.
- **Resume mode:** read the run's saved `train_config.json` and rebuild the settings from it (**the command-line settings are ignored**). Then override only five things (a tuple assignment): `resume = True`, `stop_after` = whatever you typed (so **`--resume` without `--stop-after` clears an old stop point and runs to the end**), the machine and device labels, and `overwrite = False`.

```python
    cfg = TrainConfig(
        run_dir=run_dir, model_config=a.model_config, batch_size=a.batch_size, grad_accum=a.grad_accum,
        max_steps=a.max_steps, warmup_steps=a.warmup_steps, lr=a.lr, min_lr_ratio=a.min_lr_ratio,
        weight_decay=a.weight_decay, grad_clip=a.grad_clip, dropout=a.dropout, log_every=a.log_every,
        eval_every=a.eval_every, eval_max_windows=a.eval_max_windows, precision=a.precision, device=a.device,
        seed=a.seed, stop_after=a.stop_after, machine=a.machine, overwrite=a.overwrite,
    )
    if a.prompt:
        cfg.prompts = a.prompt
    return cfg
```
- **New-run mode:** build the settings from the command-line values; if you gave `--prompt`s, replace the default prompts.

---

## 6. `append_run_log`

```python
def append_run_log(summary: dict, cfg: TrainConfig) -> None:
    status = "stopped early (resume later)" if summary["stopped_early"] else "finished"
    body = [
        f"- Machine label: {cfg.machine}; device: {summary['device']}; precision: {summary['precision']}",
        f"- Script: scripts/day9_train.py ({SCRIPT_VERSION}); PyTorch {torch.__version__}",
        f"- Run: {cfg.run_dir}; model config {cfg.model_config}; {'resumed' if cfg.resume else 'new run'}",
        f"- Step {summary['step']} of {summary['max_steps']} ({status}); best validation loss "
        f"{summary['best_val']:.4f}; total training time {summary['elapsed_s']:.0f} s",
    ]
    with (REPO_ROOT / "run_log.md").open("a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} | {cfg.machine} | Day 9 train {Path(cfg.run_dir).name}\n\n"
                + "\n".join(body) + "\n")
```
- Same style as earlier days: bullets (machine, device, precision, versions, which run and whether it was resumed, **step reached, status, best validation loss, total time**) appended to `run_log.md`. `Path(cfg.run_dir).name` = just the run name in the heading. There is no `--no-log` here.

---

## 7. `main`

```python
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    cfg = build_config(a)
    print(f"=== Day 9 train | run {cfg.run_dir} | machine={cfg.machine} | {'resume' if cfg.resume else 'new run'} ===")
    summary = train(cfg, REPO_ROOT)
    print(f"\n{'Stopped early' if summary['stopped_early'] else 'Finished'} at step {summary['step']} of "
          f"{summary['max_steps']}; best validation loss {summary['best_val']:.4f}")
    print(f"Run folder: {summary['run_dir']}")
    append_run_log(summary, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
- Console safety (as before), read options, build the settings, print a header, **run the training** (`train(cfg, REPO_ROOT)` — all the work happens there), print the final status and the run folder, append to the run log, return 0. Same ending as always.

---

## 8. The whole run in order

1. Read options → `build_config` (new run from the command line, or resume from the saved config).
2. `train(...)` (Day 9.2): checks → build → (resume) → loop → checkpoints, logs, samples.
3. Print summary → `run_log.md`.

## 9. How to read the results

- **During training:** lines like `step 100 | loss 4.12 | lr 9.8e-04 | 38,000 tokens/s`, and at each evaluation `validation loss … (perplexity …)` with `* best, saved best.pt` when it improves.
- **Afterwards:** open `runs/day9/<run>/log.csv` (curves), `samples.md` (generated text at each evaluation) and use `best.pt`.
- **Healthy sign:** the validation loss falls from about `ln(vocab)` and the samples go from nonsense to question-like text.

## 10. Small things worth knowing

- **Resume ignores your other options.** If you type `--lr 0.01 --resume`, the learning rate is **not** changed (the saved settings win). To change settings, start a new run.
- **A new run refuses an existing folder** that already has files — use a new `--run-name`, `--resume`, or `--overwrite`.
- **`--stop-after` is for tests/trials**; for real runs you normally just use Ctrl+C.
- **The run log gets an entry even for a stopped run** (status "stopped early").

## 11. Quick summary in one paragraph

`day9_train.py` is the command-line front door of training: it builds a `TrainConfig` either from your options (new run) or from the run's saved `train_config.json` (resume), calls `train(...)` from `training.py`, prints the outcome and appends a note to `run_log.md`.
